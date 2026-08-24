#!/usr/bin/env python3
"""使用 PXMeter 一键评估 Protenix 预测结果的全部默认指标。

该脚本读取 ``batch_infer_indices.py`` 生成的推理目录和汇总文件，自动：

1. 从推理目录的 ``batch_summary.json`` 读取实际 PDB、seed 和 sample 范围；
2. 将 ``.cif`` 或 ``.cif.gz`` 参考结构准备为 PXMeter 所需的 ``.cif`` 视图，
   并仅在临时副本中移除 ``_exptl`` 元数据，使参考侧保留 SO4/GOL/PEG
   等结晶辅助实体；
3. 调用 PXMeter 对全部 seed/sample 做逐结构评估；
4. 使用 PXMeter 默认逻辑汇总全部链和界面的 DockQ、LDDT、RMSD 等指标。

提供 ``--sabdab-summary-csv`` 时，还会以 ANARCII 识别 assembly 中的抗体链，
使用本轮 indices CSV 的 entity/interface 定位抗原并生成专项汇总，同时计算六条
IMGT CDR 的框架对齐主链 RMSD。SAbDab 链标注只用于范围、元数据和映射审计。

示例::

    python myscripts/step2-evaluate_abag.py \
      --pred-dir /data/my_runs/protenix_base_v1_val \
      --ref-dir /data/protenix_data_sabdab2/mmcif \
      --output-root /data/my_runs/protenix_base_v1_val_pxmeter \
      --sabdab-summary-csv /data/sabdab_summary_all.csv \
      --num-cpu 16
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence
from tqdm import tqdm

from pxmeter_utils import (
    REPO_ROOT,
    ConfigurationError,
    EvaluationError,
    add_pxmeter_to_pythonpath,
    publish_artifact,
    run_pxmeter_module,
    validate_pxmeter_modules,
    validate_summary_outputs,
    write_json_atomic,
)


LOGGER = logging.getLogger("evaluate_pxmeter")
REQUIRED_SABDAB_COLUMNS = {
    "PDB",
    "INSTANCE",
    "Hchain",
    "Lchain",
    "antigen_chain",
    "antigen_type",
}
REQUIRED_INDEX_COLUMNS = {
    "pdb_id",
    "type",
    "entity_1_id",
    "entity_2_id",
    "chain_1_id",
    "chain_2_id",
    "mol_1_type",
    "mol_2_type",
    "cluster_id",
    "eval_type",
}
NA_VALUES = {"", "NA", "N/A", "NULL", "NONE", ".", "?"}
PDB_CODE_RE = re.compile(
    r"^(?:pdb_0000)?([0-9][0-9a-z]{3})(?:$|[-_.])", re.IGNORECASE
)
CDR_BACKBONE_METRICS = (
    "cdr_h1_bb_rmsd",
    "cdr_h2_bb_rmsd",
    "cdr_h3_bb_rmsd",
    "cdr_l1_bb_rmsd",
    "cdr_l2_bb_rmsd",
    "cdr_l3_bb_rmsd",
)
THREAD_LIMIT_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def pxmeter_seed_dir(seed: int) -> str:
    """Return the seed directory name emitted by PXMeter's Protenix evaluator."""

    return str(seed)


@dataclass(frozen=True)
class BatchInfo:
    """批量推理产生的、评估完整性检查所需的元数据。"""

    seeds: tuple[int, ...]
    samples: int
    pdb_ids: tuple[str, ...]
    ref_assembly_id: str
    indices_csv: Path


@dataclass(frozen=True)
class WorkerPlan:
    """Worker counts assigned to the sequential evaluation stages."""

    total_workers: int
    eval_workers: int
    aggregate_workers: int


def build_worker_plan(
    num_cpu: int, detected_cpu_count: int | None = None
) -> WorkerPlan:
    """Use the requested worker budget for each sequential processing stage."""

    if num_cpu == -1:
        total_workers = detected_cpu_count
        if total_workers is None:
            try:
                from joblib import cpu_count

                # joblib/loky accounts for CPU affinity and container quotas.
                total_workers = cpu_count()
            except ImportError:
                get_affinity = getattr(os, "sched_getaffinity", None)
                total_workers = (
                    len(get_affinity(0)) if get_affinity is not None else os.cpu_count()
                )
    else:
        total_workers = num_cpu
    total_workers = total_workers or 1
    return WorkerPlan(
        total_workers,
        total_workers,
        total_workers,
    )


@dataclass(frozen=True)
class IndexRecord:
    """val_clean.csv 中一条 chain/interface 定位记录。"""

    pdb_id: str
    record_type: str
    entity_1_id: str
    entity_2_id: str
    chain_1_id: str
    chain_2_id: str
    mol_1_type: str
    mol_2_type: str
    cluster_id: str
    eval_type: str


@dataclass(frozen=True)
class SAbDabInstance:
    """SAbDab2 中一组抗体实例及其已整理抗原链。"""

    instance_id: str
    pdb_code: str
    heavy_chains: tuple[str, ...]
    light_chains: tuple[str, ...]
    antigen_chains: tuple[str, ...]
    antigen_types: tuple[str, ...]


@dataclass
class AntibodyTarget:
    """一个完整 biological assembly 的抗体专项元数据。"""

    alias: str
    pdb_id: str
    pdb_code: str
    reference_cif: Path
    antibody_chains: dict[str, str]
    antigen_chains: dict[str, dict[str, str]]
    candidate_interfaces: set[tuple[str, str]]
    ligand_label_asym_ids: set[str]
    sabdab_instances: tuple[str, ...]
    interface_metadata: dict[tuple[str, str], dict[str, str]]
    sabdab_metadata: dict[str, str]
    cdr_annotations: dict[str, tuple[tuple[str, ...], str]] = field(
        default_factory=dict
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行 PXMeter 并汇总 Protenix 全部默认链和界面评估结果。"
    )
    parser.add_argument(
        "--pred-dir",
        required=True,
        type=Path,
        help="batch_infer_indices.py 生成的 Protenix 推理目录。",
    )
    parser.add_argument(
        "--ref-dir",
        required=True,
        type=Path,
        help=(
            "参考 mmCIF 目录，文件名应为小写 PDB ID.cif "
            "或 PDB ID.cif.gz。"
        ),
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="PXMeter 逐样本结果、配置和汇总结果的根目录。",
    )
    parser.add_argument(
        "--num-cpu",
        type=int,
        default=-1,
        help=(
            "每个串行阶段的最大 worker 数；"
            "-1 表示使用全部可用 CPU（默认）。"
        ),
    )
    parser.add_argument(
        "--sabdab-summary-csv",
        type=Path,
        default=None,
        help=(
            "可选的 SAbDab2 summary CSV。提供后启用抗体链识别、"
            "SAbDab 抗原专项汇总和六条 CDR 主链 RMSD。"
        ),
    )
    return parser.parse_args(argv)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ConfigurationError(f"Missing {label}: {resolved}")
    return resolved


def require_dir(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise ConfigurationError(f"Missing {label}: {resolved}")
    return resolved


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    args.pred_dir = require_dir(args.pred_dir, "prediction directory")
    args.ref_dir = require_dir(args.ref_dir, "reference mmCIF directory")
    args.sabdab_summary_csv = getattr(args, "sabdab_summary_csv", None)
    if args.sabdab_summary_csv is not None:
        args.sabdab_summary_csv = require_file(
            args.sabdab_summary_csv, "SAbDab summary CSV"
        )
    args.output_root = args.output_root.expanduser().resolve()
    if args.num_cpu == 0 or args.num_cpu < -1:
        raise ConfigurationError("--num-cpu must be -1 or a positive integer")
    args.output_root.mkdir(parents=True, exist_ok=True)
    if not os.access(args.output_root, os.W_OK):
        raise ConfigurationError(
            f"Output directory is not writable: {args.output_root}"
        )
    return args


def load_batch_info(pred_dir: Path) -> BatchInfo:
    """读取 batch summary，并严格校验本轮推理和参考结构范围。"""

    summary_path = require_file(pred_dir / "batch_summary.json", "batch summary")
    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(
            f"Cannot read valid JSON from {summary_path}: {exc}"
        ) from exc

    seeds = summary.get("seeds") if isinstance(summary, dict) else None
    samples = summary.get("samples") if isinstance(summary, dict) else None
    pdb_ids = summary.get("pdb_ids") if isinstance(summary, dict) else None
    ref_assembly_id = (
        summary.get("ref_assembly_id") if isinstance(summary, dict) else None
    )
    indices_csv = summary.get("indices_csv") if isinstance(summary, dict) else None
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ConfigurationError(
            f"Invalid seeds in {summary_path}; expected a non-empty list of "
            "unique non-negative integers"
        )
    if type(samples) is not int or samples <= 0:
        raise ConfigurationError(
            f"Invalid samples in {summary_path}; expected a positive integer"
        )
    if pdb_ids is None or ref_assembly_id is None:
        raise ConfigurationError(
            f"Legacy batch summary lacks pdb_ids/ref_assembly_id: {summary_path}. "
            "Rerun step1-batch_infer_indices.py with the same arguments and "
            "without --overwrite; complete predictions will be skipped while "
            "the summary is upgraded."
        )
    if (
        not isinstance(pdb_ids, list)
        or not pdb_ids
        or any(
            not isinstance(pdb_id, str)
            or not pdb_id
            or pdb_id != pdb_id.strip().lower()
            for pdb_id in pdb_ids
        )
        or len(set(pdb_ids)) != len(pdb_ids)
    ):
        raise ConfigurationError(
            f"Invalid pdb_ids in {summary_path}; expected a non-empty list of "
            "unique, normalized lowercase strings"
        )
    if not isinstance(ref_assembly_id, str) or not ref_assembly_id.strip():
        raise ConfigurationError(
            f"Invalid ref_assembly_id in {summary_path}; expected a non-empty string"
        )
    if not isinstance(indices_csv, str) or not indices_csv.strip():
        raise ConfigurationError(
            f"Invalid indices_csv in {summary_path}; expected the non-empty path "
            "recorded by step1-batch_infer_indices.py"
        )
    return BatchInfo(
        tuple(seeds),
        samples,
        tuple(pdb_ids),
        ref_assembly_id.strip(),
        Path(indices_csv).expanduser().resolve(),
    )


def load_batch_seeds(pred_dir: Path) -> list[int]:
    """兼容旧调用方：只返回 batch summary 中的 seed。"""

    return list(load_batch_info(pred_dir).seeds)


def load_target_index_records(
    indices_csv: Path,
    pdb_ids: Sequence[str],
) -> dict[str, tuple[IndexRecord, ...]]:
    """加载本轮目标的 val chain/interface 行，并严格校验定位字段。"""

    path = require_file(indices_csv, "indices CSV recorded in batch summary")
    target_ids = set(pdb_ids)
    grouped: dict[str, list[IndexRecord]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_INDEX_COLUMNS - fields)
        if missing:
            raise ConfigurationError(
                f"{path} is missing required index columns: {', '.join(missing)}"
            )
        for row_number, row in enumerate(reader, start=2):
            pdb_id = (row.get("pdb_id") or "").strip().lower()
            if pdb_id not in target_ids:
                continue
            record_type = (row.get("type") or "").strip().lower()
            entity_1_id = (row.get("entity_1_id") or "").strip()
            entity_2_id = (row.get("entity_2_id") or "").strip()
            chain_1_id = (row.get("chain_1_id") or "").strip()
            chain_2_id = (row.get("chain_2_id") or "").strip()
            if record_type not in {"chain", "interface"}:
                raise ConfigurationError(
                    f"Invalid type={record_type!r} in {path}:{row_number}; "
                    "expected chain or interface"
                )
            if not entity_1_id or not chain_1_id:
                raise ConfigurationError(
                    f"Missing entity_1_id/chain_1_id in {path}:{row_number}"
                )
            if not (row.get("mol_1_type") or "").strip():
                raise ConfigurationError(
                    f"Missing mol_1_type in {path}:{row_number}"
                )
            if not (row.get("cluster_id") or "").strip() or not (
                row.get("eval_type") or ""
            ).strip():
                raise ConfigurationError(
                    f"Missing cluster_id/eval_type in {path}:{row_number}"
                )
            if record_type == "interface" and (not entity_2_id or not chain_2_id):
                raise ConfigurationError(
                    f"Missing entity_2_id/chain_2_id for interface in "
                    f"{path}:{row_number}"
                )
            if record_type == "interface" and not (
                row.get("mol_2_type") or ""
            ).strip():
                raise ConfigurationError(
                    f"Missing mol_2_type for interface in {path}:{row_number}"
                )
            grouped[pdb_id].append(
                IndexRecord(
                    pdb_id=pdb_id,
                    record_type=record_type,
                    entity_1_id=entity_1_id,
                    entity_2_id=entity_2_id,
                    chain_1_id=chain_1_id,
                    chain_2_id=chain_2_id,
                    mol_1_type=(row.get("mol_1_type") or "").strip(),
                    mol_2_type=(row.get("mol_2_type") or "").strip(),
                    cluster_id=(row.get("cluster_id") or "").strip(),
                    eval_type=(row.get("eval_type") or "").strip(),
                )
            )
    missing_targets = [pdb_id for pdb_id in pdb_ids if not grouped.get(pdb_id)]
    if missing_targets:
        raise ConfigurationError(
            f"Indices CSV {path} has no rows for batch target(s): "
            + ", ".join(missing_targets)
        )
    return {pdb_id: tuple(grouped[pdb_id]) for pdb_id in pdb_ids}


def is_na(value: object) -> bool:
    return str(value or "").strip().upper() in NA_VALUES


def split_sabdab_tokens(value: object) -> tuple[str, ...]:
    """拆分 SAbDab 的 ``|`` 字段，保持顺序并去重。"""

    if is_na(value):
        return ()
    return tuple(
        dict.fromkeys(
            token.strip()
            for token in str(value).split("|")
            if token.strip() and not is_na(token)
        )
    )


def canonical_pdb_code(value: object) -> str | None:
    match = PDB_CODE_RE.match(str(value or "").strip())
    return match.group(1).lower() if match else None


def load_sabdab_instances(
    path: Path,
) -> dict[str, tuple[SAbDabInstance, ...]]:
    """读取 SAbDab2 summary；抗原类型是集合，绝不与链按位置配对。"""

    grouped: dict[str, list[SAbDabInstance]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_SABDAB_COLUMNS - fields)
        if missing:
            raise ConfigurationError(
                f"{path} is missing required SAbDab columns: {', '.join(missing)}"
            )
        for row_number, row in enumerate(reader, start=2):
            pdb_code = canonical_pdb_code(row.get("PDB"))
            instance_id = (row.get("INSTANCE") or "").strip()
            if not pdb_code or not instance_id:
                LOGGER.warning(
                    "Skipping malformed SAbDab row %d (PDB=%r, INSTANCE=%r)",
                    row_number,
                    row.get("PDB"),
                    row.get("INSTANCE"),
                )
                continue
            grouped[pdb_code].append(
                SAbDabInstance(
                    instance_id=instance_id,
                    pdb_code=pdb_code,
                    heavy_chains=split_sabdab_tokens(row.get("Hchain")),
                    light_chains=split_sabdab_tokens(row.get("Lchain")),
                    antigen_chains=split_sabdab_tokens(row.get("antigen_chain")),
                    antigen_types=split_sabdab_tokens(row.get("antigen_type")),
                )
            )
    if not grouped:
        raise ConfigurationError(f"No valid SAbDab records found in {path}")
    return {key: tuple(value) for key, value in grouped.items()}


def add_pxmeter_to_current_process(pxmeter_root: Path | None) -> None:
    """让抗体预处理复用与子进程相同的 PXMeter 源码。"""

    if pxmeter_root is None:
        return
    root = str(pxmeter_root)
    if root not in sys.path:
        sys.path.insert(0, root)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_unresolved(
    rows: list[dict[str, str]],
    pdb_id: str,
    reason: str,
    detail: str,
    instance_id: str = "",
    chain_id: str = "",
) -> None:
    rows.append(
        {
            "pdb_id": pdb_id,
            "instance_id": instance_id,
            "chain_id": chain_id,
            "reason": reason,
            "detail": detail,
        }
    )


def publish_unresolved_diagnostics(
    rows: Sequence[dict[str, str]], staging_path: Path, output_root: Path
) -> str:
    """在零目标退出前发布诊断，并返回按原因计数的摘要。"""

    write_csv(
        staging_path,
        ("pdb_id", "instance_id", "chain_id", "reason", "detail"),
        rows,
    )
    publish_artifact(
        staging_path,
        output_root / "antibody" / "unresolved.csv",
    )
    counts = Counter(row.get("reason", "unknown") for row in rows)
    return ", ".join(
        f"{reason}={count}" for reason, count in sorted(counts.items())
    )


def validate_reference_cifs(
    ref_dir: Path, pdb_ids: Sequence[str]
) -> dict[str, Path]:
    """解析每个 PDB 的参考 CIF，普通 ``.cif`` 优先于 ``.cif.gz``。"""

    reference_cifs: dict[str, Path] = {}
    missing: list[str] = []
    for pdb_id in pdb_ids:
        plain_cif = ref_dir / f"{pdb_id}.cif"
        compressed_cif = ref_dir / f"{pdb_id}.cif.gz"
        if plain_cif.is_file():
            reference_cifs[pdb_id] = plain_cif
        elif compressed_cif.is_file():
            reference_cifs[pdb_id] = compressed_cif
        else:
            missing.append(pdb_id)

    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f" ... and {len(missing) - 10} more"
        raise ConfigurationError(
            f"Missing {len(missing)} reference CIF file(s) in {ref_dir}: "
            f"{preview}{suffix}"
        )
    return reference_cifs


def build_target_aliases(pdb_ids: Sequence[str]) -> dict[str, str]:
    """创建不含下划线的稳定 PXMeter 临时 ID 到原始 ID 映射。"""

    return {
        f"pxm{index:06d}": pdb_id
        for index, pdb_id in enumerate(sorted(pdb_ids))
    }


def write_pxmeter_reference_cif(
    source: Path,
    destination: Path,
    pdb_id: str,
) -> None:
    """写出 PXMeter 专用的临时参考 CIF。

    PXMeter 会在晶体学参考结构中删除 SO4、GOL、PEG 等结晶辅助
    实体，但不会在预测结构中删除它们。临时副本去掉 ``_exptl``
    类别后，PXMeter 将不再触发该清理逻辑，从而让参考与预测保持一致。
    原始参考文件始终不会被修改。
    """

    from biotite.structure.io import pdbx

    try:
        opener = gzip.open if source.suffix == ".gz" else open
        with opener(source, "rt", encoding="utf-8") as handle:
            cif_file = pdbx.CIFFile.read(handle)

        block = cif_file.block
        if "exptl" in block:
            del block["exptl"]

        cif_file.write(destination)

        # 在启动长时间评估前立即发现截断、不可读或写出无效的 CIF。
        prepared_cif = pdbx.CIFFile.read(destination)
        prepared_block = prepared_cif.block
        if "exptl" in prepared_block:
            raise ValueError("temporary CIF still contains the exptl category")
        if "atom_site" not in prepared_block:
            raise ValueError("temporary CIF does not contain an atom_site category")
        _ = prepared_block["atom_site"]
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise ConfigurationError(
            f"Cannot prepare PXMeter reference CIF for {pdb_id} from "
            f"{source}: {exc}"
        ) from exc


def create_reference_cif_view(
    reference_cifs: dict[str, Path],
    alias_to_pdb_id: dict[str, str],
    view_dir: Path,
) -> None:
    """创建保留结晶辅助实体、使用无下划线 ID 的参考视图。"""

    view_dir.mkdir(parents=True, exist_ok=False)
    for alias, pdb_id in tqdm(
        alias_to_pdb_id.items(),
        desc="准备参考 CIF",
        unit="PDB",
        disable=not sys.stderr.isatty(),
    ):
        source = reference_cifs[pdb_id]
        destination = view_dir / f"{alias}.cif"
        write_pxmeter_reference_cif(source, destination, pdb_id)


def antibody_role(chain_type: str) -> str | None:
    """将 ANARCII chain_type 规范化为评估角色。"""

    parts = {part for part in re.split(r"[_+|]", chain_type.upper()) if part}
    has_heavy = "H" in parts
    has_light = bool(parts & {"K", "L"})
    if has_heavy and has_light:
        return "antibody_scfv"
    if has_heavy:
        return "antibody_heavy"
    if has_light:
        return "antibody_light"
    return None


def ligand_has_pxmeter_pocket(structure, ligand_chain_id: str) -> bool:
    """镜像 PXMeter RMSD 的 10 Å/至少三个聚合物骨架原子前置条件。"""

    import numpy as np
    from scipy.spatial import KDTree

    from pxmeter.constants import POLYMER

    ligand_mask = structure.uni_chain_id == ligand_chain_id
    if not np.any(ligand_mask):
        return False
    polymer_mask = structure.get_mask_for_given_entity_types(POLYMER)
    backbone = polymer_mask & np.isin(
        structure.atom_array.atom_name, ["CA", "C1'"]
    )
    nearby = np.zeros(len(structure.atom_array), dtype=bool)
    tree = KDTree(structure.atom_array.coord)
    for indices in tree.query_ball_point(
        structure.atom_array.coord[ligand_mask], r=10.0
    ):
        nearby[np.asarray(indices, dtype=int)] = True
    candidate_chains = structure.uni_chain_id[nearby & backbone]
    return any(
        int(np.sum(candidate_chains == chain_id)) >= 3
        for chain_id in np.unique(candidate_chains)
    )


def join_unique(values: Sequence[object]) -> str:
    """将审计字段合并为稳定、去重的 ``|`` 分隔字符串。"""

    tokens: list[str] = []
    for value in values:
        for token in str(value or "").split("|"):
            token = token.strip()
            if token and token not in tokens:
                tokens.append(token)
    return "|".join(tokens)


def merge_audit_metadata(
    destination: dict[str, str], source: dict[str, str]
) -> None:
    for key, value in source.items():
        destination[key] = join_unique((destination.get(key, ""), value))


def unresolved_sabdab_author_chains(
    instances: Sequence[SAbDabInstance], available_auth_ids: set[str]
) -> list[tuple[str, str, str]]:
    """返回不能精确映射的 SAbDab author chain；不做任何后缀变换。"""

    expected = [
        (item.instance_id, "heavy", chain)
        for item in instances
        for chain in item.heavy_chains
    ] + [
        (item.instance_id, "light", chain)
        for item in instances
        for chain in item.light_chains
    ] + [
        (item.instance_id, "antigen", chain)
        for item in instances
        for chain in item.antigen_chains
    ]
    return [item for item in expected if item[2] not in available_auth_ids]


def prepare_antibody_targets(
    alias_to_pdb_id: dict[str, str],
    ref_view_dir: Path,
    ref_assembly_id: str,
    sabdab_by_pdb: dict[str, tuple[SAbDabInstance, ...]],
    index_records_by_pdb: dict[str, tuple[IndexRecord, ...]],
) -> tuple[
    dict[str, AntibodyTarget],
    list[dict[str, str]],
    list[dict[str, str]],
    dict[str, list[dict[str, str]]],
]:
    """以 val entity + ANARCII 定位界面，SAbDab 仅提供范围和审计元数据。"""

    try:
        import numpy as np

        from pxmeter.constants import LIGAND, PROTEIN
        from pxmeter.data.struct import Structure
        from pxmeter.metrics.antibody.annotation import AntibodyAnnotator
    except ImportError as exc:
        raise ConfigurationError(
            "Antibody mode requires PXMeter's numpy/biotite/ANARCII dependencies"
        ) from exc

    annotator = AntibodyAnnotator(
        scheme="imgt",
        seq_type="unknown",
        ncpu=1,
    )
    targets: dict[str, AntibodyTarget] = {}
    annotation_rows: list[dict[str, str]] = []
    unresolved: list[dict[str, str]] = []
    internal_ligand_rows: list[dict[str, str]] = []
    public_ligand_rows: list[dict[str, str]] = []

    for alias, pdb_id in tqdm(
        alias_to_pdb_id.items(),
        desc="识别抗体和抗原",
        unit="PDB",
        disable=not sys.stderr.isatty(),
    ):
        pdb_code = canonical_pdb_code(pdb_id)
        if pdb_code is None:
            append_unresolved(
                unresolved,
                pdb_id,
                "invalid_pdb_id",
                "pdb_id cannot be normalized to a four-character PDB code",
            )
            continue
        instances = sabdab_by_pdb.get(pdb_code, ())
        if not instances:
            append_unresolved(
                unresolved,
                pdb_id,
                "pdb_not_in_sabdab",
                f"no SAbDab row for canonical PDB {pdb_code}",
            )
            continue

        reference_cif = ref_view_dir / f"{alias}.cif"
        try:
            structure = Structure.from_mmcif(
                reference_cif,
                model=1,
                altloc="first",
                assembly_id=ref_assembly_id,
            )
        except Exception as exc:
            append_unresolved(
                unresolved,
                pdb_id,
                "assembly_parse_failed",
                str(exc),
            )
            continue

        protein_entities: list[str] = []
        protein_sequences: list[str] = []
        for entity_id, sequence in structure.entity_poly_seq.items():
            if structure.entity_poly_type.get(entity_id) == PROTEIN and sequence:
                protein_entities.append(str(entity_id))
                protein_sequences.append(sequence)
        try:
            annotations = annotator.annotate(protein_sequences)
        except Exception as exc:
            append_unresolved(
                unresolved,
                pdb_id,
                "anarcii_failed",
                str(exc),
            )
            annotations = [(["-"] * len(seq), "Unknown") for seq in protein_sequences]
        entity_to_chain_type = {
            entity: result[1]
            for entity, result in zip(protein_entities, annotations)
        }
        cdr_annotations = {
            entity: (tuple(map(str, result[0])), str(result[1]))
            for entity, result in zip(protein_entities, annotations)
        }

        chain_details: dict[str, dict[str, str]] = {}
        auth_to_uni: dict[str, set[str]] = defaultdict(set)
        label_to_uni: dict[str, set[str]] = defaultdict(set)
        entity_to_uni: dict[str, set[str]] = defaultdict(set)
        for uni_chain in np.unique(structure.uni_chain_id):
            mask = structure.uni_chain_id == uni_chain
            index = int(np.flatnonzero(mask)[0])
            entity_id = str(structure.atom_array.label_entity_id[index])
            label_id = str(structure.atom_array.label_asym_id[index])
            auth_id = str(structure.atom_array.auth_asym_id[index])
            entity_type = structure.entity_poly_type.get(entity_id, LIGAND)
            chain_type = entity_to_chain_type.get(entity_id, "Unknown")
            role = antibody_role(chain_type) or "non_antibody"
            details = {
                "uni_chain_id": str(uni_chain),
                "label_asym_id": label_id,
                "auth_asym_id": auth_id,
                "entity_id": entity_id,
                "entity_type": entity_type,
                "anarcii_type": chain_type,
                "role": role,
            }
            chain_details[str(uni_chain)] = details
            auth_to_uni[auth_id].add(str(uni_chain))
            label_to_uni[label_id].add(str(uni_chain))
            entity_to_uni[entity_id].add(str(uni_chain))

        antibody_chains = {
            chain_id: details["role"]
            for chain_id, details in chain_details.items()
            if details["role"].startswith("antibody_")
        }
        if not antibody_chains:
            append_unresolved(
                unresolved,
                pdb_id,
                "antibody_not_identified",
                "ANARCII did not identify a heavy, kappa, or lambda variable domain",
            )

        instance_ids = [instance.instance_id for instance in instances]
        sabdab_heavy = [chain for item in instances for chain in item.heavy_chains]
        sabdab_light = [chain for item in instances for chain in item.light_chains]
        sabdab_antigens = [chain for item in instances for chain in item.antigen_chains]
        sabdab_types = [kind for item in instances for kind in item.antigen_types]
        unresolved_sabdab_chains: list[str] = []
        for instance_id, chain_role, author_chain in unresolved_sabdab_author_chains(
            instances, set(auth_to_uni)
        ):
            # 只允许 author ID 的精确匹配；绝不删除 M1/E2 等数字后缀。
            unresolved_sabdab_chains.append(
                f"{instance_id}:{chain_role}:{author_chain}"
            )
            append_unresolved(
                unresolved,
                pdb_id,
                "sabdab_chain_mapping_unresolved",
                "exact SAbDab author chain is absent from assembly; "
                "val entity + ANARCII localization remains eligible",
                instance_id=instance_id,
                chain_id=author_chain,
            )
        sabdab_metadata = {
            "sabdab_instances": join_unique(instance_ids),
            "sabdab_heavy_chains": join_unique(sabdab_heavy),
            "sabdab_light_chains": join_unique(sabdab_light),
            "sabdab_antigen_chains": join_unique(sabdab_antigens),
            "sabdab_antigen_types": join_unique(sabdab_types),
            "sabdab_chain_mapping_status": (
                "unresolved" if unresolved_sabdab_chains else "exact"
            ),
            "sabdab_unresolved_chains": join_unique(unresolved_sabdab_chains),
        }

        entity_audit: dict[str, dict[str, str]] = defaultdict(dict)
        for record in index_records_by_pdb.get(pdb_id, ()):
            for side in (1, 2):
                entity_id = getattr(record, f"entity_{side}_id")
                if not entity_id:
                    continue
                merge_audit_metadata(
                    entity_audit[entity_id],
                    {
                        "val_chain_ids": getattr(record, f"chain_{side}_id"),
                        "val_mol_types": getattr(record, f"mol_{side}_type"),
                        "val_cluster_ids": record.cluster_id,
                        "val_eval_types": record.eval_type,
                        "val_record_types": record.record_type,
                    },
                )

        antigen_chains: dict[str, dict[str, str]] = {}
        candidate_interfaces: set[tuple[str, str]] = set()
        interface_metadata: dict[tuple[str, str], dict[str, str]] = {}
        ligand_label_ids: set[str] = set()
        for record in index_records_by_pdb.get(pdb_id, ()):
            if record.record_type != "interface":
                continue
            side_1_chains = entity_to_uni.get(record.entity_1_id, set())
            side_2_chains = entity_to_uni.get(record.entity_2_id, set())
            if not side_1_chains or not side_2_chains:
                missing_entities = [
                    entity
                    for entity, chains in (
                        (record.entity_1_id, side_1_chains),
                        (record.entity_2_id, side_2_chains),
                    )
                    if not chains
                ]
                append_unresolved(
                    unresolved,
                    pdb_id,
                    "val_entity_unmapped",
                    "reference CIF has no chain for val entity ID(s): "
                    + ",".join(missing_entities),
                    chain_id=f"{record.chain_1_id},{record.chain_2_id}",
                )
                continue
            side_1_antibodies = side_1_chains & antibody_chains.keys()
            side_2_antibodies = side_2_chains & antibody_chains.keys()
            if bool(side_1_antibodies) == bool(side_2_antibodies):
                continue
            if side_1_antibodies:
                ab_entity, ag_entity = record.entity_1_id, record.entity_2_id
                ab_val_chain, ag_val_chain = record.chain_1_id, record.chain_2_id
                ab_chains, ag_chains = side_1_antibodies, side_2_chains
            else:
                ab_entity, ag_entity = record.entity_2_id, record.entity_1_id
                ab_val_chain, ag_val_chain = record.chain_2_id, record.chain_1_id
                ab_chains, ag_chains = side_2_antibodies, side_1_chains
            audit = {
                "val_antibody_entity_id": ab_entity,
                "val_antigen_entity_id": ag_entity,
                "val_antibody_chain_ids": ab_val_chain,
                "val_antigen_chain_ids": ag_val_chain,
                "val_cluster_ids": record.cluster_id,
                "val_eval_types": record.eval_type,
                "positioning_source": "val_entity_id+anarcii",
                **sabdab_metadata,
            }
            for antigen_chain in ag_chains:
                details = antigen_chains.setdefault(
                    antigen_chain, dict(chain_details[antigen_chain])
                )
                merge_audit_metadata(details, audit)
                if details["entity_type"] == LIGAND:
                    label_id = details["label_asym_id"]
                    expanded = label_to_uni.get(label_id, set())
                    if len(expanded) != 1:
                        append_unresolved(
                            unresolved,
                            pdb_id,
                            "ligand_assembly_ambiguous",
                            f"label_asym_id {label_id} expands to {len(expanded)} chains",
                            chain_id=antigen_chain,
                        )
                        continue
                    try:
                        pocket_valid = ligand_has_pxmeter_pocket(
                            structure, antigen_chain
                        )
                    except Exception as exc:
                        append_unresolved(
                            unresolved,
                            pdb_id,
                            "ligand_pocket_check_failed",
                            str(exc),
                            chain_id=antigen_chain,
                        )
                        continue
                    if not pocket_valid:
                        append_unresolved(
                            unresolved,
                            pdb_id,
                            "ligand_pocket_invalid",
                            "fewer than three polymer backbone atoms form a 10 A pocket",
                            chain_id=antigen_chain,
                        )
                        continue
                    ligand_label_ids.add(label_id)
                    continue
                for antibody_chain in ab_chains:
                    if antibody_chain == antigen_chain:
                        continue
                    pair = tuple(sorted((antibody_chain, antigen_chain)))
                    candidate_interfaces.add(pair)
                    metadata = interface_metadata.setdefault(pair, {})
                    merge_audit_metadata(metadata, audit)

        # SAbDab 缺少抗原链元数据不会覆盖 val+ANARCII 的结构定位结果。
        for instance in instances:
            if not instance.antigen_chains:
                append_unresolved(
                    unresolved,
                    pdb_id,
                    "sabdab_antigen_missing",
                    "SAbDab instance has no curated antigen_chain",
                    instance_id=instance.instance_id,
                )
                continue

        for chain_id, details in chain_details.items():
            antigen = antigen_chains.get(chain_id)
            audit = entity_audit.get(details["entity_id"], {})
            exact_sabdab_antigen = any(
                chain_id in auth_to_uni.get(author_chain, set())
                for author_chain in sabdab_antigens
            )
            annotation_rows.append(
                {
                    "pdb_id": pdb_id,
                    **details,
                    **audit,
                    "is_val_antigen": str(antigen is not None),
                    "is_sabdab_antigen": str(exact_sabdab_antigen),
                    "val_antibody_entity_id": antigen.get(
                        "val_antibody_entity_id", ""
                    ) if antigen else "",
                    "val_antigen_entity_id": antigen.get(
                        "val_antigen_entity_id", ""
                    ) if antigen else "",
                    "positioning_source": antigen.get(
                        "positioning_source", ""
                    ) if antigen else "",
                    **sabdab_metadata,
                }
            )

        if antibody_chains and (candidate_interfaces or ligand_label_ids):
            targets[pdb_id] = AntibodyTarget(
                alias=alias,
                pdb_id=pdb_id,
                pdb_code=pdb_code,
                reference_cif=reference_cif,
                antibody_chains=antibody_chains,
                antigen_chains=antigen_chains,
                candidate_interfaces=candidate_interfaces,
                ligand_label_asym_ids=ligand_label_ids,
                sabdab_instances=tuple(dict.fromkeys(instance_ids)),
                interface_metadata=interface_metadata,
                sabdab_metadata=sabdab_metadata,
                cdr_annotations=cdr_annotations,
            )
            for label_id in sorted(ligand_label_ids):
                internal_ligand_rows.append(
                    {"entry_id": alias, "label_asym_id": label_id}
                )
                public_ligand_rows.append(
                    {"entry_id": pdb_id, "label_asym_id": label_id}
                )
        else:
            append_unresolved(
                unresolved,
                pdb_id,
                "no_resolved_antibody_antigen",
                "no reliable antibody-antigen interface or unambiguous ligand remained",
            )

    return targets, annotation_rows, unresolved, {
        "internal": internal_ligand_rows,
        "public": public_ligand_rows,
    }


def validate_target_predictions(
    pred_dir: Path,
    pdb_ids: Sequence[str],
    seeds: Sequence[int],
    samples: int,
) -> int:
    """逐 PDB/seed 校验主 CIF 和置信度 JSON，避免漏样本后静默评估。"""

    problems: list[str] = []
    for pdb_id in tqdm(
        pdb_ids,
        desc="检查预测文件",
        unit="PDB",
        disable=not sys.stderr.isatty(),
    ):
        pdb_dir = pred_dir / pdb_id
        if not pdb_dir.is_dir():
            problems.append(f"missing prediction directory: {pdb_dir}")
            continue

        for seed in seeds:
            prediction_dir = pdb_dir / f"seed_{seed}" / "predictions"
            expected_cifs = {
                prediction_dir / f"{pdb_id}_sample_{sample}.cif"
                for sample in range(samples)
            }
            expected_confidences = {
                prediction_dir
                / f"{pdb_id}_summary_confidence_sample_{sample}.json"
                for sample in range(samples)
            }
            actual_cifs = {
                path
                for path in prediction_dir.glob(f"{pdb_id}_sample_*.cif")
                if path.is_file() and not path.name.endswith("_wounresol.cif")
            }
            actual_confidences = {
                path
                for path in prediction_dir.glob(
                    f"{pdb_id}_summary_confidence_sample_*.json"
                )
                if path.is_file()
            }
            missing = sorted(
                str(path)
                for path in (expected_cifs | expected_confidences)
                - (actual_cifs | actual_confidences)
            )
            extra = sorted(
                str(path)
                for path in (actual_cifs | actual_confidences)
                - (expected_cifs | expected_confidences)
            )
            if missing:
                problems.append(f"missing {len(missing)} file(s); first: {missing[0]}")
            if extra:
                problems.append(f"unexpected {len(extra)} file(s); first: {extra[0]}")

    if problems:
        preview = "; ".join(problems[:10])
        suffix = "" if len(problems) <= 10 else f"; ... and {len(problems) - 10} more"
        raise ConfigurationError(f"Incomplete target predictions: {preview}{suffix}")
    return len(pdb_ids) * len(seeds) * samples


def create_target_prediction_view(
    pred_dir: Path,
    alias_to_pdb_id: dict[str, str],
    seeds: Sequence[int],
    samples: int,
    view_dir: Path,
) -> None:
    """仅链接主 CIF 和 summary confidence，隐藏其他辅助文件。"""

    view_dir.mkdir(parents=True, exist_ok=False)
    for alias, pdb_id in alias_to_pdb_id.items():
        for seed in seeds:
            source_dir = pred_dir / pdb_id / f"seed_{seed}" / "predictions"
            destination_dir = (
                view_dir / alias / f"seed_{seed}" / "predictions"
            )
            destination_dir.mkdir(parents=True)
            for sample in range(samples):
                filenames = (
                    f"{pdb_id}_sample_{sample}.cif",
                    f"{pdb_id}_summary_confidence_sample_{sample}.json",
                )
                for filename in filenames:
                    (destination_dir / filename).symlink_to(
                        (source_dir / filename).resolve()
                    )


def restore_pxmeter_target_ids(
    per_sample_dir: Path, alias_to_pdb_id: dict[str, str]
) -> None:
    """将 PXMeter 输出的临时 ID 恢复为索引中的原始 target ID。"""

    error_dir = per_sample_dir / "ERR"
    for alias, pdb_id in alias_to_pdb_id.items():
        alias_result_dir = per_sample_dir / alias
        pdb_result_dir = per_sample_dir / pdb_id
        if alias_result_dir.is_dir():
            if pdb_result_dir.exists():
                raise EvaluationError(
                    f"Cannot restore PXMeter target {alias} to {pdb_id}: "
                    f"destination already exists: {pdb_result_dir}"
                )
            alias_result_dir.rename(pdb_result_dir)
            for metrics_json in pdb_result_dir.rglob("sample_*_metrics.json"):
                try:
                    with metrics_json.open("r", encoding="utf-8") as handle:
                        metrics = json.load(handle)
                except (OSError, json.JSONDecodeError) as exc:
                    raise EvaluationError(
                        f"Cannot restore target ID in {metrics_json}: {exc}"
                    ) from exc
                if not isinstance(metrics, dict):
                    raise EvaluationError(
                        f"Cannot restore target ID in {metrics_json}: "
                        "expected a JSON object"
                    )
                metrics["entry_id"] = pdb_id
                write_json_atomic(metrics_json, metrics)

        alias_error_dir = error_dir / alias
        if alias_error_dir.is_dir():
            pdb_error_dir = error_dir / pdb_id
            if pdb_error_dir.exists():
                raise EvaluationError(
                    f"Cannot restore PXMeter error target {alias} to {pdb_id}: "
                    f"destination already exists: {pdb_error_dir}"
                )
            alias_error_dir.rename(pdb_error_dir)


def count_metric_jsons(per_sample_dir: Path) -> int:
    return sum(
        1
        for path in per_sample_dir.rglob("sample_*_metrics.json")
        if path.is_file()
    )


def find_error_logs(per_sample_dir: Path) -> list[Path]:
    error_dir = per_sample_dir / "ERR"
    if not error_dir.is_dir():
        return []
    return sorted(path for path in error_dir.rglob("*.log") if path.is_file())


def validate_evaluation_results(
    expected_count: int, per_sample_dir: Path
) -> tuple[int, list[Path]]:
    metric_count = count_metric_jsons(per_sample_dir)
    error_logs = find_error_logs(per_sample_dir)
    problems = []
    if error_logs:
        problems.append(f"PXMeter wrote {len(error_logs)} error log(s)")
    if metric_count != expected_count:
        problems.append(
            f"metric JSON count {metric_count} does not match prediction CIF "
            f"count {expected_count}"
        )
    if problems:
        details = "; ".join(problems)
        if error_logs:
            details += f"; first error: {error_logs[0]}"
        raise EvaluationError(details)
    return metric_count, error_logs


def calculate_six_cdr_rmsds(
    ref_struct,
    model_struct,
    cached_annotations: dict[str, tuple[tuple[str, ...], str]] | None = None,
) -> tuple[dict[str, dict[str, float]], list[dict[str, str]]]:
    """Calculate six IMGT CDR backbone RMSDs on PXMeter-mapped structures."""

    import numpy as np

    from pxmeter.constants import PROTEIN
    from pxmeter.metrics.antibody.annotation import AntibodyAnnotator
    from pxmeter.metrics.rmsd import partially_aligned_rmsd

    protein_mask = ref_struct.get_mask_for_given_entity_types(PROTEIN)
    entity_ids = np.unique(ref_struct.atom_array.label_entity_id[protein_mask])
    annotations_by_entity = {
        str(entity_id): (tuple(regions), chain_type)
        for entity_id, (regions, chain_type) in (cached_annotations or {}).items()
    }
    missing_entities = [
        entity_id
        for entity_id in entity_ids
        if str(entity_id) not in annotations_by_entity
        and ref_struct.entity_poly_seq.get(entity_id, "")
    ]
    if missing_entities:
        unique_sequences: list[str] = []
        entity_to_sequence_index: dict[str, int] = {}
        for entity_id in missing_entities:
            sequence = ref_struct.entity_poly_seq.get(entity_id, "")
            try:
                sequence_index = unique_sequences.index(sequence)
            except ValueError:
                unique_sequences.append(sequence)
                sequence_index = len(unique_sequences) - 1
            entity_to_sequence_index[str(entity_id)] = sequence_index
        annotations = AntibodyAnnotator(
            scheme="imgt",
            seq_type="unknown",
            ncpu=1,
        ).annotate(unique_sequences)
        for entity_id, sequence_index in entity_to_sequence_index.items():
            regions, chain_type = annotations[sequence_index]
            annotations_by_entity[entity_id] = (tuple(regions), chain_type)

    if not annotations_by_entity:
        return {}, [{"chain_id": "", "reason": "no_protein_sequence"}]

    results: dict[str, dict[str, float]] = {}
    problems: list[dict[str, str]] = []
    for chain_id in np.unique(ref_struct.uni_chain_id[protein_mask]):
        chain_mask = ref_struct.uni_chain_id == chain_id
        first = int(np.flatnonzero(chain_mask)[0])
        entity_id = str(ref_struct.atom_array.label_entity_id[first])
        annotation = annotations_by_entity.get(entity_id)
        if annotation is None:
            problems.append(
                {
                    "chain_id": str(chain_id),
                    "reason": "antibody_annotation_missing",
                }
            )
            continue
        regions, chain_type = annotation
        role = antibody_role(chain_type)
        if role is None:
            continue
        if role == "antibody_scfv":
            problems.append(
                {
                    "chain_id": str(chain_id),
                    "reason": "scfv_multi_domain_cdr_ambiguous",
                }
            )
            continue

        atom_indices = np.flatnonzero(chain_mask)
        res_ids = ref_struct.atom_array.res_id[atom_indices]
        atom_names = ref_struct.atom_array.atom_name[atom_indices]
        valid_res = (res_ids >= 1) & (res_ids <= len(regions))
        atom_indices = atom_indices[valid_res]
        res_ids = res_ids[valid_res]
        atom_names = atom_names[valid_res]
        if not len(atom_indices):
            problems.append(
                {"chain_id": str(chain_id), "reason": "no_numbered_atoms"}
            )
            continue
        atom_regions = np.array([regions[int(res_id) - 1] for res_id in res_ids])
        ref_coords = ref_struct.atom_array.coord[atom_indices]
        model_coords = model_struct.atom_array.coord[atom_indices]
        valid_coords = np.isfinite(ref_coords).all(axis=1) & np.isfinite(
            model_coords
        ).all(axis=1)
        if model_struct.valid_mask is not None:
            valid_coords &= model_struct.valid_mask[atom_indices]
        if ref_struct.valid_mask is not None:
            valid_coords &= ref_struct.valid_mask[atom_indices]
        atom_regions = atom_regions[valid_coords]
        atom_names = atom_names[valid_coords]
        ref_coords = ref_coords[valid_coords]
        model_coords = model_coords[valid_coords]
        backbone = np.isin(atom_names, ["N", "CA", "C", "O"])
        framework = np.isin(atom_regions, ["FR1", "FR2", "FR3", "FR4"])
        align_mask = backbone & framework
        if int(np.sum(align_mask)) < 3:
            problems.append(
                {
                    "chain_id": str(chain_id),
                    "reason": "insufficient_framework_backbone_atoms",
                }
            )
            continue

        prefix = "h" if role == "antibody_heavy" else "l"
        chain_results: dict[str, float] = {}
        for cdr_number in (1, 2, 3):
            cdr_mask = backbone & (atom_regions == f"CDR{cdr_number}")
            metric_name = f"cdr_{prefix}{cdr_number}_bb_rmsd"
            if not np.any(cdr_mask):
                problems.append(
                    {
                        "chain_id": str(chain_id),
                        "reason": f"{metric_name}_atoms_missing",
                    }
                )
                continue
            _, rmsd_value, _, _ = partially_aligned_rmsd(
                src_pose=model_coords,
                tar_pose=ref_coords,
                align_mask=align_mask,
                rmsd_mask=cdr_mask,
                eps=1e-12,
            )
            chain_results[metric_name] = float(rmsd_value)
        if chain_results:
            results[str(chain_id)] = chain_results
    return results, problems


@dataclass(frozen=True)
class CDRTask:
    pdb_id: str
    seed: int
    sample: int
    reference_cif: Path
    prediction_cif: Path
    metrics_path: Path
    ref_assembly_id: str
    cached_annotations: dict[str, tuple[tuple[str, ...], str]]


@dataclass(frozen=True)
class CDRTaskResult:
    pdb_id: str
    seed: int
    sample: int
    metrics_path: Path
    calculated: dict[str, dict[str, float]]
    problems: tuple[dict[str, str], ...]
    error: str = ""


def calculate_cdr_task(task: CDRTask) -> CDRTaskResult:
    """Map one prediction and calculate its CDR metrics without writing files."""

    try:
        from pxmeter.mapping import MappingResult

        mapping = MappingResult.from_cifs(
            ref_cif=task.reference_cif,
            model_cif=task.prediction_cif,
            ref_model=1,
            ref_assembly_id=task.ref_assembly_id,
            ref_altloc="first",
        )
        ref_struct, model_struct = mapping.get_mapped_structures()
        calculated, problems = calculate_six_cdr_rmsds(
            ref_struct,
            model_struct,
            task.cached_annotations,
        )
        return CDRTaskResult(
            task.pdb_id,
            task.seed,
            task.sample,
            task.metrics_path,
            calculated,
            tuple(problems),
        )
    except Exception as exc:
        return CDRTaskResult(
            task.pdb_id,
            task.seed,
            task.sample,
            task.metrics_path,
            {},
            (),
            str(exc),
        )


def postprocess_cdr_metrics(
    targets: dict[str, AntibodyTarget],
    pred_dir: Path,
    per_sample_dir: Path,
    batch_info: BatchInfo,
    num_workers: int,
    unresolved: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Calculate six CDR metrics in parallel and serialize writes in the parent."""

    from concurrent.futures import ProcessPoolExecutor, as_completed

    tasks = [
        CDRTask(
            pdb_id=pdb_id,
            seed=seed,
            sample=sample,
            reference_cif=target.reference_cif,
            prediction_cif=(
                pred_dir
                / pdb_id
                / f"seed_{seed}"
                / "predictions"
                / f"{pdb_id}_sample_{sample}.cif"
            ),
            metrics_path=(
                per_sample_dir
                / pdb_id
                / pxmeter_seed_dir(seed)
                / f"sample_{sample}_metrics.json"
            ),
            ref_assembly_id=batch_info.ref_assembly_id,
            cached_annotations=target.cdr_annotations,
        )
        for pdb_id, target in sorted(targets.items())
        for seed in batch_info.seeds
        for sample in range(batch_info.samples)
    ]
    if not tasks:
        return []
    num_workers = min(num_workers, len(tasks))
    if num_workers == 1:
        results = [
            calculate_cdr_task(task)
            for task in tqdm(
                tasks,
                total=len(tasks),
                desc="计算六条 CDR RMSD",
                unit="样本",
                disable=not sys.stderr.isatty(),
            )
        ]
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_task = {
                executor.submit(calculate_cdr_task, task): task for task in tasks
            }
            results = []
            for future in tqdm(
                as_completed(future_to_task),
                total=len(future_to_task),
                desc="计算六条 CDR RMSD",
                unit="样本",
                disable=not sys.stderr.isatty(),
            ):
                task = future_to_task[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append(
                        CDRTaskResult(
                            task.pdb_id,
                            task.seed,
                            task.sample,
                            task.metrics_path,
                            {},
                            (),
                            str(exc),
                        )
                    )
    results.sort(
        key=lambda result: (
            result.pdb_id,
            result.seed,
            result.sample,
        )
    )

    rows: list[dict[str, Any]] = []
    for result in results:
        if result.error:
            append_unresolved(
                unresolved,
                result.pdb_id,
                "cdr_postprocess_failed",
                f"seed={result.seed}, sample={result.sample}: {result.error}",
            )
            continue
        for problem in result.problems:
            append_unresolved(
                unresolved,
                result.pdb_id,
                "cdr_unavailable",
                f"seed={result.seed}, sample={result.sample}: {problem['reason']}",
                chain_id=problem.get("chain_id", ""),
            )
        try:
            with result.metrics_path.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            append_unresolved(
                unresolved,
                result.pdb_id,
                "cdr_metrics_read_failed",
                f"seed={result.seed}, sample={result.sample}: {exc}",
            )
            continue
        if not isinstance(metrics, dict):
            append_unresolved(
                unresolved,
                result.pdb_id,
                "cdr_metrics_read_failed",
                (
                    f"seed={result.seed}, sample={result.sample}: "
                    "metrics JSON is not an object"
                ),
            )
            continue

        chain_metrics = metrics.setdefault("chain", {})
        if not isinstance(chain_metrics, dict):
            append_unresolved(
                unresolved,
                result.pdb_id,
                "cdr_metrics_read_failed",
                (
                    f"seed={result.seed}, sample={result.sample}: "
                    "chain metrics is not an object"
                ),
            )
            continue
        changed = False
        target = targets[result.pdb_id]
        for chain_id, values in sorted(result.calculated.items()):
            published = chain_metrics.setdefault(chain_id, {})
            recomputed_h3 = values.get("cdr_h3_bb_rmsd")
            native_h3 = published.get("cdr_h3_bb_rmsd")
            if native_h3 is not None and recomputed_h3 is not None:
                if abs(float(native_h3) - recomputed_h3) > 1e-5:
                    append_unresolved(
                        unresolved,
                        result.pdb_id,
                        "cdr_h3_validation_mismatch",
                        (
                            f"seed={result.seed}, sample={result.sample}: "
                            f"native={native_h3}, recomputed={recomputed_h3}"
                        ),
                        chain_id=chain_id,
                    )
            for name, value in values.items():
                if name != "cdr_h3_bb_rmsd":
                    published[name] = value
                    changed = True
            row: dict[str, Any] = {
                "entry_id": result.pdb_id,
                "seed": result.seed,
                "sample": result.sample,
                "chain_id": chain_id,
                "antibody_role": target.antibody_chains.get(chain_id, ""),
                "cdr_h3_recomputed": recomputed_h3,
            }
            for metric_name in CDR_BACKBONE_METRICS:
                row[metric_name] = published.get(metric_name)
            rows.append(row)
        if changed:
            write_json_atomic(result.metrics_path, metrics)

    rows.sort(
        key=lambda row: (
            str(row["entry_id"]),
            int(row["seed"]),
            int(row["sample"]),
            str(row["chain_id"]),
        )
    )
    return rows


def write_cdr_summary(path: Path, cdr_rows: Sequence[dict[str, Any]]) -> None:
    import statistics

    rows = []
    for metric_name in CDR_BACKBONE_METRICS:
        values = [
            float(row[metric_name])
            for row in cdr_rows
            if row.get(metric_name) is not None
        ]
        rows.append(
            {
                "metric": metric_name,
                "count": len(values),
                "mean": statistics.fmean(values) if values else "",
                "median": statistics.median(values) if values else "",
                "std": statistics.pstdev(values) if values else "",
            }
        )
    write_csv(path, ("metric", "count", "mean", "median", "std"), rows)


def build_antibody_subset_and_details(
    targets: dict[str, AntibodyTarget],
    per_sample_dir: Path,
    batch_info: BatchInfo,
    unresolved: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """把 SAbDab候选与 PXMeter实际产出的链/界面求交集。"""

    subset: dict[tuple[str, str, str, str], dict[str, str]] = {}
    detail_rows: list[dict[str, Any]] = []
    for pdb_id, target in targets.items():
        actual_interfaces: set[tuple[str, str]] = set()
        actual_chains: set[str] = set()
        metric_documents: list[tuple[int, int, dict]] = []
        for seed in batch_info.seeds:
            for sample in range(batch_info.samples):
                path = (
                    per_sample_dir
                    / pdb_id
                    / pxmeter_seed_dir(seed)
                    / f"sample_{sample}_metrics.json"
                )
                with path.open("r", encoding="utf-8") as handle:
                    metrics = json.load(handle)
                metric_documents.append((seed, sample, metrics))
                actual_chains.update(metrics.get("chain", {}))
                for key in metrics.get("interface", {}):
                    parts = tuple(sorted(key.split(",")))
                    if len(parts) == 2:
                        actual_interfaces.add(parts)

        selected_interfaces = target.candidate_interfaces & actual_interfaces
        for pair in sorted(target.candidate_interfaces - actual_interfaces):
            append_unresolved(
                unresolved,
                pdb_id,
                "antigen_interface_not_emitted",
                "PXMeter did not emit metrics for the curated antibody-antigen pair",
                chain_id=",".join(pair),
            )
        for chain_1, chain_2 in sorted(selected_interfaces):
            subset_row = {
                "type": "interface",
                "entry_id": pdb_id,
                "chain_id_1": chain_1,
                "chain_id_2": chain_2,
            }
            subset[("interface", pdb_id, chain_1, chain_2)] = subset_row
        for chain_id in sorted(target.antibody_chains):
            if chain_id in actual_chains:
                subset[("chain", pdb_id, chain_id, "")] = {
                    "type": "chain",
                    "entry_id": pdb_id,
                    "chain_id_1": chain_id,
                    "chain_id_2": "",
                }
        for chain_id, details in sorted(target.antigen_chains.items()):
            if details["label_asym_id"] in target.ligand_label_asym_ids:
                if chain_id in actual_chains:
                    subset[("chain", pdb_id, chain_id, "")] = {
                        "type": "chain",
                        "entry_id": pdb_id,
                        "chain_id_1": chain_id,
                        "chain_id_2": "",
                    }

        for seed, sample, metrics in metric_documents:
            for chain_1, chain_2 in selected_interfaces:
                values = metrics.get("interface", {}).get(f"{chain_1},{chain_2}")
                if values is None:
                    values = metrics.get("interface", {}).get(f"{chain_2},{chain_1}")
                if values is None:
                    continue
                antigen_chain = (
                    chain_1 if chain_1 in target.antigen_chains else chain_2
                )
                antibody_chain = chain_2 if antigen_chain == chain_1 else chain_1
                antigen = target.antigen_chains.get(antigen_chain, {})
                audit = target.interface_metadata.get(
                    tuple(sorted((chain_1, chain_2))), {}
                )
                row: dict[str, Any] = {
                    "entry_id": pdb_id,
                    "seed": seed,
                    "sample": sample,
                    "type": "interface",
                    "antibody_chain": antibody_chain,
                    "antibody_role": target.antibody_chains.get(antibody_chain, ""),
                    "antigen_chain": antigen_chain,
                    "antigen_entity_type": antigen.get("entity_type", ""),
                    "sabdab_antigen_types": antigen.get(
                        "sabdab_antigen_types", ""
                    ),
                    **target.sabdab_metadata,
                    **audit,
                    "lddt": values.get("lddt"),
                    "bb_lddt": values.get("bb_lddt"),
                    "dockq": values.get("dockq"),
                }
                dockq_info = values.get("dockq_info") or {}
                for name in (
                    "F1",
                    "iRMSD",
                    "LRMSD",
                    "fnat",
                    "fnonnat",
                    "clashes",
                ):
                    row[name] = dockq_info.get(name)
                detail_rows.append(row)
            for antigen_chain, antigen in target.antigen_chains.items():
                if antigen["label_asym_id"] not in target.ligand_label_asym_ids:
                    continue
                values = metrics.get("chain", {}).get(antigen_chain)
                if not values:
                    continue
                detail_rows.append(
                    {
                        "entry_id": pdb_id,
                        "seed": seed,
                        "sample": sample,
                        "type": "ligand",
                        "antibody_chain": "",
                        "antibody_role": "",
                        "antigen_chain": antigen_chain,
                        "antigen_entity_type": antigen.get("entity_type", ""),
                        "sabdab_antigen_types": antigen.get(
                            "sabdab_antigen_types", ""
                        ),
                        **target.sabdab_metadata,
                        **{
                            key: antigen.get(key, "")
                            for key in (
                                "val_antibody_entity_id",
                                "val_antigen_entity_id",
                                "val_antibody_chain_ids",
                                "val_antigen_chain_ids",
                                "val_cluster_ids",
                                "val_eval_types",
                                "positioning_source",
                            )
                        },
                        "lig_rmsd": values.get("lig_rmsd"),
                        "pocket_rmsd": values.get("pocket_rmsd"),
                        "lddt_pli": values.get("lddt_pli"),
                        "ref_pocket_chain": values.get("ref_pocket_chain"),
                        "pb_valid_json": json.dumps(
                            metrics.get("pb_valid", {}).get(antigen_chain, {}),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    }
                )
    return list(subset.values()), detail_rows


def publish_run_results(run_root: Path, output_root: Path) -> None:
    """将已完整验证的本次结果发布到用户指定的固定路径。"""

    artifact_names = (
        "per_sample",
        "summary",
        "antibody",
        "pxmeter_paths.json",
        "per_sample_metrics.parquet",
        "per_sample_metrics.csv",
        "per_sample_pb_valid.parquet",
        "per_sample_pb_valid.csv",
    )
    for name in artifact_names:
        source = run_root / name
        if source.exists() or source.is_symlink():
            publish_artifact(source, output_root / name)
        elif name.startswith("per_sample_"):
            # 聚合格式可能在 CSV/Parquet 间变化；清除上次留下的另一种格式，
            # 避免 PXMeter 后续优先读取旧 CSV。
            stale = output_root / name
            if stale.is_dir() and not stale.is_symlink():
                shutil.rmtree(stale)
            elif stale.exists() or stale.is_symlink():
                stale.unlink()

    # 旧版脚本发布过抗体专用 subset；全界面评估成功后移除该遗留文件，
    # 避免用户误以为本次汇总仍经过抗体界面过滤。
    retired_subset = output_root / "antibody_antigen_subset.csv"
    if retired_subset.exists() or retired_subset.is_symlink():
        retired_subset.unlink()


def run(args: argparse.Namespace) -> int:
    args = validate_args(args)
    worker_plan = build_worker_plan(args.num_cpu)
    for variable in THREAD_LIMIT_ENV_VARS:
        os.environ[variable] = "1"
    LOGGER.info(
        "Sequential stage workers: evaluation=%d, CDR=%d, aggregation=%d; "
        "BLAS/OpenMP threads per worker=1",
        worker_plan.eval_workers,
        worker_plan.total_workers,
        worker_plan.aggregate_workers,
    )
    pxmeter_root = validate_pxmeter_modules(REPO_ROOT)
    batch_info = load_batch_info(args.pred_dir)
    antibody_mode = args.sabdab_summary_csv is not None
    sabdab_by_pdb = (
        load_sabdab_instances(args.sabdab_summary_csv) if antibody_mode else {}
    )
    index_records_by_pdb = (
        load_target_index_records(batch_info.indices_csv, batch_info.pdb_ids)
        if antibody_mode
        else {}
    )
    if antibody_mode:
        add_pxmeter_to_current_process(pxmeter_root)

    # 所有中间结果都写入临时目录。只有评估与聚合完整成功后才覆盖正式结果，
    # 因此旧 metrics/ERR/summary 不会污染重跑，失败也不会破坏上一份结果。
    with tempfile.TemporaryDirectory(
        prefix=".evaluate_pxmeter_", dir=args.output_root
    ) as temporary_dir:
        run_root = Path(temporary_dir)
        per_sample_dir = run_root / "per_sample"
        paths_json = run_root / "pxmeter_paths.json"
        summary_dir = run_root / "summary"
        pred_view_dir = run_root / "target_predictions"
        ref_view_dir = run_root / "reference_cifs"
        antibody_dir = run_root / "antibody"
        antibody_summary_dir = antibody_dir / "summary"
        per_sample_dir.mkdir(parents=True)
        summary_dir.mkdir(parents=True)

        pdb_ids = batch_info.pdb_ids
        reference_cifs = validate_reference_cifs(args.ref_dir, pdb_ids)
        prediction_count = validate_target_predictions(
            args.pred_dir,
            pdb_ids,
            batch_info.seeds,
            batch_info.samples,
        )
        alias_to_pdb_id = build_target_aliases(pdb_ids)
        create_reference_cif_view(
            reference_cifs, alias_to_pdb_id, ref_view_dir
        )
        LOGGER.info(
            "Prepared temporary reference CIFs without _exptl metadata; "
            "crystallization aids such as SO4/GOL/PEG will participate in "
            "PXMeter mapping and metrics"
        )
        create_target_prediction_view(
            args.pred_dir,
            alias_to_pdb_id,
            batch_info.seeds,
            batch_info.samples,
            pred_view_dir,
        )

        antibody_targets: dict[str, AntibodyTarget] = {}
        annotation_rows: list[dict[str, str]] = []
        unresolved_rows: list[dict[str, str]] = []
        ligand_rows: dict[str, list[dict[str, str]]] = {
            "internal": [],
            "public": [],
        }
        internal_ligand_csv = run_root / "antibody_lig_info_internal.csv"
        if antibody_mode:
            antibody_prepare_started = time.perf_counter()
            (
                antibody_targets,
                annotation_rows,
                unresolved_rows,
                ligand_rows,
            ) = prepare_antibody_targets(
                alias_to_pdb_id,
                ref_view_dir,
                batch_info.ref_assembly_id,
                sabdab_by_pdb,
                index_records_by_pdb,
            )
            antibody_dir.mkdir(parents=True)
            write_csv(
                antibody_dir / "annotations.csv",
                (
                    "pdb_id",
                    "uni_chain_id",
                    "label_asym_id",
                    "auth_asym_id",
                    "entity_id",
                    "entity_type",
                    "anarcii_type",
                    "role",
                    "is_val_antigen",
                    "is_sabdab_antigen",
                    "val_chain_ids",
                    "val_mol_types",
                    "val_cluster_ids",
                    "val_eval_types",
                    "val_record_types",
                    "val_antibody_entity_id",
                    "val_antigen_entity_id",
                    "positioning_source",
                    "sabdab_instances",
                    "sabdab_heavy_chains",
                    "sabdab_light_chains",
                    "sabdab_antigen_chains",
                    "sabdab_antigen_types",
                    "sabdab_chain_mapping_status",
                    "sabdab_unresolved_chains",
                ),
                annotation_rows,
            )
            if not antibody_targets:
                reason_summary = publish_unresolved_diagnostics(
                    unresolved_rows,
                    antibody_dir / "unresolved.csv",
                    args.output_root,
                )
                raise ConfigurationError(
                    "SAbDab antibody mode found no val entity + ANARCII target; "
                    f"unresolved.csv was published. Reasons: {reason_summary or 'none'}"
                )
            write_csv(
                antibody_dir / "ligand_info.csv",
                ("entry_id", "label_asym_id"),
                ligand_rows["public"],
            )
            if ligand_rows["internal"]:
                write_csv(
                    internal_ligand_csv,
                    ("entry_id", "label_asym_id"),
                    ligand_rows["internal"],
                )
            LOGGER.info(
                "Prepared antibody metadata for %d/%d target(s); "
                "ligand chains=%d; elapsed=%.2fs",
                len(antibody_targets),
                len(pdb_ids),
                len(ligand_rows["internal"]),
                time.perf_counter() - antibody_prepare_started,
            )

        LOGGER.info(
            "Prepared all default chain/interface metrics for %d PDB(s); "
            "seeds=%s, samples=%d",
            len(pdb_ids),
            ",".join(map(str, batch_info.seeds)),
            batch_info.samples,
        )

        env = os.environ.copy()
        env["PXM_MMCIF_DIR"] = str(ref_view_dir)
        for variable in THREAD_LIMIT_ENV_VARS:
            env[variable] = "1"
        add_pxmeter_to_pythonpath(env, pxmeter_root)
        workflow = tqdm(
            total=4 if antibody_mode else 2,
            desc="PXMeter 全界面评估",
            unit="阶段",
            disable=not sys.stderr.isatty(),
        )
        try:
            run_eval_args = [
                "-i",
                str(pred_view_dir),
                "-o",
                str(per_sample_dir),
                "-m",
                "protenix",
                "-r",
                batch_info.ref_assembly_id,
                "-n",
                str(worker_plan.eval_workers),
            ]
            if antibody_mode:
                run_eval_args.extend(
                    ["-C", "metric.calc_cdr_h3_bb_rmsd=true"]
                )
                if ligand_rows["internal"]:
                    run_eval_args.extend(["-l", str(internal_ligand_csv)])
            evaluation_started = time.perf_counter()
            run_pxmeter_module(
                "benchmark.run_eval",
                tuple(run_eval_args),
                env,
            )
            LOGGER.info(
                "PXMeter evaluation elapsed=%.2fs",
                time.perf_counter() - evaluation_started,
            )
            workflow.update(1)

            restore_pxmeter_target_ids(per_sample_dir, alias_to_pdb_id)
            metric_count, error_logs = validate_evaluation_results(
                prediction_count, per_sample_dir
            )
            antibody_subset_rows: list[dict[str, str]] = []
            cdr_rows: list[dict[str, Any]] = []
            if antibody_mode:
                cdr_started = time.perf_counter()
                cdr_rows = postprocess_cdr_metrics(
                    antibody_targets,
                    args.pred_dir,
                    per_sample_dir,
                    batch_info,
                    worker_plan.total_workers,
                    unresolved_rows,
                )
                LOGGER.info(
                    "Six-CDR postprocessing elapsed=%.2fs",
                    time.perf_counter() - cdr_started,
                )
                antibody_subset_rows, antibody_detail_rows = (
                    build_antibody_subset_and_details(
                        antibody_targets,
                        per_sample_dir,
                        batch_info,
                        unresolved_rows,
                    )
                )
                write_csv(
                    antibody_dir / "subset.csv",
                    ("type", "entry_id", "chain_id_1", "chain_id_2"),
                    antibody_subset_rows,
                )
                write_csv(
                    antibody_dir / "per_sample_metrics.csv",
                    (
                        "entry_id",
                        "seed",
                        "sample",
                        "type",
                        "antibody_chain",
                        "antibody_role",
                        "antigen_chain",
                        "antigen_entity_type",
                        "val_antibody_entity_id",
                        "val_antigen_entity_id",
                        "val_antibody_chain_ids",
                        "val_antigen_chain_ids",
                        "val_cluster_ids",
                        "val_eval_types",
                        "positioning_source",
                        "sabdab_instances",
                        "sabdab_heavy_chains",
                        "sabdab_light_chains",
                        "sabdab_antigen_chains",
                        "sabdab_antigen_types",
                        "sabdab_chain_mapping_status",
                        "sabdab_unresolved_chains",
                        "lddt",
                        "bb_lddt",
                        "dockq",
                        "F1",
                        "iRMSD",
                        "LRMSD",
                        "fnat",
                        "fnonnat",
                        "clashes",
                        "lig_rmsd",
                        "pocket_rmsd",
                        "lddt_pli",
                        "ref_pocket_chain",
                        "pb_valid_json",
                    ),
                    antibody_detail_rows,
                )
                write_csv(
                    antibody_dir / "cdr_metrics.csv",
                    (
                        "entry_id",
                        "seed",
                        "sample",
                        "chain_id",
                        "antibody_role",
                        *CDR_BACKBONE_METRICS,
                        "cdr_h3_recomputed",
                    ),
                    cdr_rows,
                )
                antibody_summary_dir.mkdir(parents=True)
                write_csv(
                    antibody_dir / "unresolved.csv",
                    ("pdb_id", "instance_id", "chain_id", "reason", "detail"),
                    sorted(
                        unresolved_rows,
                        key=lambda row: (
                            row.get("pdb_id", ""),
                            row.get("instance_id", ""),
                            row.get("chain_id", ""),
                            row.get("reason", ""),
                            row.get("detail", ""),
                        ),
                    ),
                )
                workflow.update(1)
            trial_name = args.pred_dir.name
            write_json_atomic(
                paths_json,
                {
                    trial_name: {
                        "model": "protenix",
                        "seeds": list(batch_info.seeds),
                        "dataset_path": {"Custom": str(per_sample_dir)},
                    }
                },
            )

            aggregation_started = time.perf_counter()
            run_pxmeter_module(
                "benchmark.show_intersection_results",
                (
                    "-i",
                    str(paths_json),
                    "-o",
                    str(summary_dir),
                    "-n",
                    str(worker_plan.aggregate_workers),
                    "--overwrite_agg",
                ),
                env,
            )
            LOGGER.info(
                "PXMeter full aggregation elapsed=%.2fs",
                time.perf_counter() - aggregation_started,
            )
            workflow.update(1)
            if antibody_mode and antibody_subset_rows:
                subset_aggregation_started = time.perf_counter()
                run_pxmeter_module(
                    "benchmark.show_intersection_results",
                    (
                        "-i",
                        str(paths_json),
                        "-o",
                        str(antibody_summary_dir),
                        "-n",
                        str(worker_plan.aggregate_workers),
                        "--subset_csv",
                        str(antibody_dir / "subset.csv"),
                    ),
                    env,
                )
                LOGGER.info(
                    "PXMeter antibody subset aggregation elapsed=%.2fs",
                    time.perf_counter() - subset_aggregation_started,
                )
                antibody_summary_table = (
                    antibody_summary_dir / "Summary_table.csv"
                )
                if not antibody_summary_table.is_file():
                    raise EvaluationError(
                        "PXMeter antibody aggregation completed without "
                        f"{antibody_summary_table}"
                    )
                workflow.update(1)
            elif antibody_mode:
                workflow.update(1)
            if antibody_mode:
                # PXMeter aggregation may create/replace files in this directory;
                # publish the custom six-CDR summary last.
                write_cdr_summary(
                    antibody_summary_dir / "CDR_results.csv", cdr_rows
                )
        finally:
            workflow.close()

        validate_summary_outputs(summary_dir)

        # 发布后重写配置中的路径，避免临时目录被清理后 pxmeter_paths.json 失效。
        write_json_atomic(
            paths_json,
            {
                args.pred_dir.name: {
                    "model": "protenix",
                    "seeds": list(batch_info.seeds),
                    "dataset_path": {
                        "Custom": str(args.output_root / "per_sample")
                    },
                }
            },
        )
        publish_run_results(run_root, args.output_root)

    LOGGER.info(
        "PXMeter finished: prediction_cifs=%d, metric_jsons=%d, errors=%d",
        prediction_count,
        metric_count,
        len(error_logs),
    )
    LOGGER.info("DockQ results: %s", args.output_root / "summary/DockQ_results.csv")
    LOGGER.info("LDDT results: %s", args.output_root / "summary/LDDT_results.csv")
    rmsd_results = args.output_root / "summary/RMSD_results.csv"
    if rmsd_results.is_file():
        LOGGER.info("RMSD results: %s", rmsd_results)
    else:
        LOGGER.info(
            "RMSD results were not generated because PXMeter found no "
            "ligand RMSD metrics to aggregate"
        )
    LOGGER.info("Summary table: %s", args.output_root / "summary/Summary_table.csv")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    try:
        return run(parse_args(argv))
    except ConfigurationError as exc:
        LOGGER.error("Configuration error: %s", exc)
        return 2
    except EvaluationError as exc:
        LOGGER.error("Evaluation failed: %s", exc)
        return 1
    except OSError as exc:
        LOGGER.error("I/O error: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
