#!/usr/bin/env python3
# ruff: noqa: N999
"""将 PXMeter 评估结果整理为精简的中文 DockQ/LDDT/RMSD 报表。

报表只比较两种选择方式：

* ``Oracle最佳``：使用已观测的真实指标选取最佳样本；
* ``预测排序最佳``：使用 Protenix ``ranking_score`` 选取样本。

界面级 Oracle 的各行可以来自不同样本；PDB 级 Oracle 始终选择一个
完整样本，排序依据为该 PDB 所有 Protein-X 界面的平均 DockQ。

执行流程：读取明细定义和逐样本指标，重新完成 Oracle/预测最佳选择，
生成实例表与汇总表，最后以临时目录和备份回滚机制原子发布全部 CSV。
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import re
import shutil
import sys
import tempfile
import uuid
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_DIR = REPO_ROOT / "mydata" / "protenix_base_default_v1_preds_eval"

ORACLE = "Oracle最佳"
PREDICTED_BEST = "预测排序最佳"
METHODS = (ORACLE, PREDICTED_BEST)
METHOD_ORDER = {ORACLE: 0, PREDICTED_BEST: 1}
THRESHOLDS = (0.23, 0.49, 0.80)

OUTPUT_FILENAMES = (
    "DockQ_prot_interface_sum.csv",
    "DockQ_prot_interface_inst.csv",
    "DockQ_pdb_sum.csv",
    "DockQ_pdb_inst.csv",
    "DockQ_abag_interface_sum.csv",
    "DockQ_abag_interface_inst.csv",
)
OPTIONAL_OUTPUT_FILENAMES = (
    "LDDT_structure_sum.csv",
    "LDDT_structure_inst.csv",
    "LDDT_pdb_sum.csv",
    "LDDT_pdb_inst.csv",
    "LDDT_abag_interface_sum.csv",
    "LDDT_abag_interface_inst.csv",
    "RMSD_ligand_sum.csv",
    "RMSD_ligand_inst.csv",
    "RMSD_ligand_pdb_sum.csv",
    "RMSD_ligand_pdb_inst.csv",
    "RMSD_cdr_sum.csv",
    "RMSD_cdr_inst.csv",
    "RMSD_cdr_pdb_sum.csv",
    "RMSD_cdr_pdb_inst.csv",
)
MANAGED_OUTPUT_FILENAMES = OUTPUT_FILENAMES + OPTIONAL_OUTPUT_FILENAMES
PDB_OUTPUT_DIRNAME = "pdbs"
PDB_EXPORT_COUNT_KEY = "PDB结构文件"

DETAIL_REQUIRED_COLUMNS = {
    "name",
    "eval_dataset",
    "eval_type",
    "entry_id",
    "entity_id_1",
    "entity_id_2",
    "chain_id_1",
    "chain_id_2",
    "cluster_id",
    "ranker",
    "subset",
}
SUBSET_REQUIRED_COLUMNS = {"type", "entry_id", "chain_id_1", "chain_id_2"}
ANNOTATION_REQUIRED_COLUMNS = {
    "pdb_id",
    "uni_chain_id",
    "role",
    "is_val_antigen",
}

PROT_INST_COLUMNS = (
    "模型名称",
    "评估数据集",
    "数据子集",
    "PDB编号",
    "界面类型",
    "链1",
    "链2",
    "实体1",
    "实体2",
    "cluster编号",
    "选择方式",
    "seed",
    "sample",
    "定位键",
    "选择依据分数",
    "DockQ",
    "DockQ质量等级",
    "是否DockQ≥0.23",
    "是否DockQ≥0.49",
    "是否DockQ≥0.80",
)

PROT_SUM_COLUMNS = (
    "界面类型",
    "界面实例数量",
    "独立cluster数量",
    "PDB数量",
    "选择方式",
    "聚合方式",
    "平均DockQ",
    "DockQ范围（最小值~最大值）",
    "成功率（%，DockQ≥0.23）",
    "成功率（%，DockQ≥0.49）",
    "成功率（%，DockQ≥0.80）",
)

PDB_INST_COLUMNS = (
    "PDB编号",
    "包含的界面类型",
    "界面数量",
    "选择方式",
    "seed",
    "sample",
    "定位键",
    "选择依据分数",
    "PDB界面平均DockQ",
    "DockQ范围（最小值~最大值）",
    "最低界面DockQ",
    "PDB整体质量等级",
    "DockQ≥0.23界面数",
    "DockQ≥0.23界面成功率（%）",
    "PDB是否DockQ≥0.23",
    "DockQ≥0.49界面数",
    "DockQ≥0.49界面成功率（%）",
    "PDB是否DockQ≥0.49",
    "DockQ≥0.80界面数",
    "DockQ≥0.80界面成功率（%）",
    "PDB是否DockQ≥0.80",
)

PDB_SUM_COLUMNS = (
    "PDB数量",
    "界面总数",
    "选择方式",
    "PDB等权平均DockQ",
    "PDB平均DockQ范围（最小值~最大值）",
    "DockQ≥0.23成功PDB数",
    "DockQ≥0.23成功PDB率（%）",
    "DockQ≥0.49成功PDB数",
    "DockQ≥0.49成功PDB率（%）",
    "DockQ≥0.80成功PDB数",
    "DockQ≥0.80成功PDB率（%）",
    "平均每PDB界面数量",
)

ABAG_INST_COLUMNS = PROT_INST_COLUMNS + (
    "抗体链",
    "抗原链",
    "抗体链类型",
    "抗原-抗体界面分组",
)

ABAG_SUM_COLUMNS = (
    "抗原-抗体界面分组",
    "界面实例数量",
    "独立cluster数量",
    "PDB数量",
    "选择方式",
    "聚合方式",
    "平均DockQ",
    "DockQ范围（最小值~最大值）",
    "成功率（%，DockQ≥0.23）",
    "成功率（%，DockQ≥0.49）",
    "成功率（%，DockQ≥0.80）",
)


class SimplifyError(RuntimeError):
    """Raised when the source data cannot produce trustworthy summaries."""


@dataclass(frozen=True)
class InterfaceDefinition:
    name: str
    eval_dataset: str
    eval_type: str
    entry_id: str
    chain_1: str
    chain_2: str
    entity_1: str
    entity_2: str
    cluster_id: str
    subset: str

    @property
    def pair(self) -> tuple[str, str]:
        return canonical_pair(self.chain_1, self.chain_2)


@dataclass(frozen=True)
class SampleCandidate:
    entry_id: str
    seed: int
    sample: int
    ranking_score: float
    dockq_by_pair: Mapping[tuple[str, str], float]

    @property
    def locator(self) -> str:
        return f"{self.entry_id}|seed={self.seed}|sample={self.sample}"


@dataclass(frozen=True)
class SelectedInterface:
    definition: InterfaceDefinition
    method: str
    candidate: SampleCandidate
    dockq: float
    selection_score: float


@dataclass(frozen=True)
class PdbStructureExport:
    source: Path
    filename: str
    decompress_gzip: bool = False


def canonical_pair(chain_1: str, chain_2: str) -> tuple[str, str]:
    if not chain_1 or not chain_2 or chain_1 == chain_2:
        raise SimplifyError(f"Invalid interface chain pair: {chain_1!r}, {chain_2!r}")
    return tuple(sorted((chain_1, chain_2)))  # type: ignore[return-value]


def parse_finite_float(value: Any, label: str, *, dockq: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SimplifyError(f"Invalid {label}: {value!r}") from exc
    if not math.isfinite(number):
        raise SimplifyError(f"Invalid {label}: {value!r}")
    if dockq and not 0.0 <= number <= 1.0:
        raise SimplifyError(f"{label} must be in [0, 1], got {number}")
    return number


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise SimplifyError(f"Missing required CSV: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = sorted(required - columns)
        if missing:
            raise SimplifyError(f"Missing columns in {path}: {', '.join(missing)}")
        return [dict(row) for row in reader]


def _normalized_definition(row: Mapping[str, str]) -> InterfaceDefinition:
    required_values = (
        "name",
        "eval_dataset",
        "eval_type",
        "entry_id",
        "entity_id_1",
        "entity_id_2",
        "chain_id_1",
        "chain_id_2",
        "cluster_id",
    )
    empty = [column for column in required_values if not row[column].strip()]
    if empty:
        raise SimplifyError("Empty required DockQ detail value(s): " + ", ".join(empty))
    chain_1 = row["chain_id_1"].strip()
    chain_2 = row["chain_id_2"].strip()
    entity_1 = row["entity_id_1"].strip()
    entity_2 = row["entity_id_2"].strip()
    if chain_2 < chain_1:
        chain_1, chain_2 = chain_2, chain_1
        entity_1, entity_2 = entity_2, entity_1
    return InterfaceDefinition(
        name=row["name"].strip(),
        eval_dataset=row["eval_dataset"].strip(),
        eval_type=row["eval_type"].strip(),
        entry_id=row["entry_id"].strip(),
        chain_1=chain_1,
        chain_2=chain_2,
        entity_1=entity_1,
        entity_2=entity_2,
        cluster_id=row["cluster_id"].strip(),
        subset=row["subset"].strip(),
    )


def load_interface_definitions(
    path: Path, *, protein_only: bool, ranker: str
) -> dict[tuple[str, tuple[str, str]], InterfaceDefinition]:
    rows = read_csv(path, DETAIL_REQUIRED_COLUMNS)
    definitions: dict[tuple[str, tuple[str, str]], InterfaceDefinition] = {}
    for row in rows:
        if row["ranker"].strip() != ranker:
            continue
        definition = _normalized_definition(row)
        if protein_only and "Protein" not in definition.eval_type.split("-"):
            continue
        key = (definition.entry_id, definition.pair)
        previous = definitions.get(key)
        if previous is not None and previous != definition:
            raise SimplifyError(
                "Conflicting metadata for interface "
                f"{definition.entry_id}:{definition.chain_1},{definition.chain_2}"
            )
        definitions[key] = definition
    if not definitions:
        label = "Protein-X " if protein_only else ""
        raise SimplifyError(
            f"No valid {label}interface definitions for ranker {ranker!r} in {path}"
        )
    return definitions


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SimplifyError(f"Missing required JSON: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SimplifyError(f"Cannot parse JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SimplifyError(f"Expected a JSON object in {path}")
    return value


SAMPLE_RE = re.compile(r"^sample_(\d+)_metrics\.json$")


def load_sample_candidates(
    per_sample_dir: Path,
    needed_pairs_by_entry: Mapping[str, set[tuple[str, str]]],
) -> dict[str, list[SampleCandidate]]:
    if not per_sample_dir.is_dir():
        raise SimplifyError(f"Missing per-sample directory: {per_sample_dir}")
    candidates: dict[str, list[SampleCandidate]] = defaultdict(list)
    seen: set[tuple[str, int, int]] = set()
    for entry_id in sorted(needed_pairs_by_entry):
        entry_dir = per_sample_dir / entry_id
        if not entry_dir.is_dir():
            raise SimplifyError(f"Missing per-sample PDB directory: {entry_dir}")
        for metrics_path in sorted(entry_dir.glob("*/sample_*_metrics.json")):
            match = SAMPLE_RE.match(metrics_path.name)
            if match is None:
                continue
            try:
                seed = int(metrics_path.parent.name)
                sample = int(match.group(1))
            except ValueError as exc:
                raise SimplifyError(
                    f"Invalid seed/sample path: {metrics_path}"
                ) from exc
            sample_key = (entry_id, seed, sample)
            if sample_key in seen:
                raise SimplifyError(f"Duplicate per-sample result: {sample_key}")
            seen.add(sample_key)

            metrics = load_json(metrics_path)
            interface_metrics = metrics.get("interface")
            if not isinstance(interface_metrics, dict):
                raise SimplifyError(f"Missing interface metrics in {metrics_path}")
            available: dict[tuple[str, str], float] = {}
            for raw_pair, metric_values in interface_metrics.items():
                if not isinstance(raw_pair, str) or not isinstance(metric_values, dict):
                    continue
                chains = [item.strip() for item in raw_pair.split(",")]
                if len(chains) != 2:
                    continue
                pair = canonical_pair(chains[0], chains[1])
                if "dockq" in metric_values:
                    available[pair] = parse_finite_float(
                        metric_values["dockq"],
                        f"DockQ in {metrics_path} for {raw_pair}",
                        dockq=True,
                    )
            missing_pairs = sorted(needed_pairs_by_entry[entry_id] - available.keys())
            if missing_pairs:
                preview = ", ".join(f"{a},{b}" for a, b in missing_pairs[:5])
                raise SimplifyError(
                    f"Missing required interface DockQ in {metrics_path}: {preview}"
                )

            confidence_path = metrics_path.with_name(
                f"sample_{sample}_confidences.json"
            )
            confidence = load_json(confidence_path)
            complex_confidence = confidence.get("complex")
            if not isinstance(complex_confidence, dict):
                raise SimplifyError(
                    f"Missing complex confidence metrics in {confidence_path}"
                )
            ranking_score = parse_finite_float(
                complex_confidence.get("ranking_score"),
                f"ranking_score in {confidence_path}",
            )
            candidates[entry_id].append(
                SampleCandidate(
                    entry_id=entry_id,
                    seed=seed,
                    sample=sample,
                    ranking_score=ranking_score,
                    dockq_by_pair={
                        pair: available[pair]
                        for pair in needed_pairs_by_entry[entry_id]
                    },
                )
            )
        if not candidates[entry_id]:
            raise SimplifyError(f"No per-sample metrics found for {entry_id}")
    return dict(candidates)


def choose_predicted(candidates: Sequence[SampleCandidate]) -> SampleCandidate:
    return max(
        candidates,
        key=lambda item: (item.ranking_score, -item.seed, -item.sample),
    )


def choose_interface_oracle(
    candidates: Sequence[SampleCandidate], pair: tuple[str, str]
) -> SampleCandidate:
    return max(
        candidates,
        key=lambda item: (
            item.dockq_by_pair[pair],
            item.ranking_score,
            -item.seed,
            -item.sample,
        ),
    )


def choose_pdb_oracle(
    candidates: Sequence[SampleCandidate], pairs: Sequence[tuple[str, str]]
) -> SampleCandidate:
    def key(candidate: SampleCandidate) -> tuple[float, float, float, int, int]:
        values = [candidate.dockq_by_pair[pair] for pair in pairs]
        return (
            mean(values),
            min(values),
            candidate.ranking_score,
            -candidate.seed,
            -candidate.sample,
        )

    return max(candidates, key=key)


def select_interfaces(
    oracle_definitions: Mapping[tuple[str, tuple[str, str]], InterfaceDefinition],
    predicted_definitions: Mapping[tuple[str, tuple[str, str]], InterfaceDefinition],
    candidates_by_entry: Mapping[str, Sequence[SampleCandidate]],
) -> list[SelectedInterface]:
    # 先确认两种 ranker 覆盖同一批界面，再从原始逐样本指标
    # 重新选择；不直接信任汇总 CSV，以保留可追溯的 seed/sample。
    if oracle_definitions.keys() != predicted_definitions.keys():
        oracle_only = sorted(oracle_definitions.keys() - predicted_definitions.keys())
        predicted_only = sorted(
            predicted_definitions.keys() - oracle_definitions.keys()
        )
        raise SimplifyError(
            "Oracle and predicted rankers contain different interfaces; "
            f"oracle-only={oracle_only[:3]}, predicted-only={predicted_only[:3]}"
        )
    predicted_by_entry = {
        entry: choose_predicted(candidates)
        for entry, candidates in candidates_by_entry.items()
    }
    selected: list[SelectedInterface] = []
    for entry_id, pair in sorted(oracle_definitions):
        oracle_definition = oracle_definitions[(entry_id, pair)]
        predicted_definition = predicted_definitions[(entry_id, pair)]
        comparable_oracle = oracle_definition.__dict__.copy()
        comparable_predicted = predicted_definition.__dict__.copy()
        comparable_oracle.pop("cluster_id")
        comparable_predicted.pop("cluster_id")
        if comparable_oracle != comparable_predicted:
            raise SimplifyError(
                "Oracle and predicted rankers have conflicting metadata for "
                f"{entry_id}:{pair[0]},{pair[1]}"
            )
        candidates = candidates_by_entry.get(entry_id)
        if not candidates:
            raise SimplifyError(f"No candidates available for {entry_id}")
        oracle = choose_interface_oracle(candidates, pair)
        predicted = predicted_by_entry[entry_id]
        selected.extend(
            (
                SelectedInterface(
                    definition=oracle_definition,
                    method=ORACLE,
                    candidate=oracle,
                    dockq=oracle.dockq_by_pair[pair],
                    selection_score=oracle.dockq_by_pair[pair],
                ),
                SelectedInterface(
                    definition=predicted_definition,
                    method=PREDICTED_BEST,
                    candidate=predicted,
                    dockq=predicted.dockq_by_pair[pair],
                    selection_score=predicted.ranking_score,
                ),
            )
        )
    validate_two_methods(selected)
    return selected


def validate_two_methods(rows: Sequence[SelectedInterface]) -> None:
    methods_by_interface: dict[tuple[str, tuple[str, str]], list[str]] = defaultdict(
        list
    )
    predicted_locator_by_entry: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        methods_by_interface[(row.definition.entry_id, row.definition.pair)].append(
            row.method
        )
        if row.method == PREDICTED_BEST:
            predicted_locator_by_entry[row.definition.entry_id].add(
                row.candidate.locator
            )
    expected = [ORACLE, PREDICTED_BEST]
    for key, methods in methods_by_interface.items():
        if sorted(methods, key=METHOD_ORDER.get) != expected:
            raise SimplifyError(f"Interface {key} does not have exactly two methods")
    inconsistent = {
        entry: locators
        for entry, locators in predicted_locator_by_entry.items()
        if len(locators) != 1
    }
    if inconsistent:
        raise SimplifyError(
            "Predicted-best interfaces do not share one sample per PDB: "
            + ", ".join(sorted(inconsistent))
        )


def quality_label(dockq: float) -> str:
    if dockq >= 0.80:
        return "高质量"
    if dockq >= 0.49:
        return "中等质量"
    if dockq >= 0.23:
        return "可接受质量"
    return "不正确"


def yes_no(value: bool) -> str:
    return "是" if value else "否"


def dockq_text(value: float) -> str:
    return f"{value:.4f}"


def percent(value: float) -> str:
    return f"{value * 100.0:.2f}"


def dockq_range(values: Sequence[float]) -> str:
    return f"{min(values):.4f}~{max(values):.4f}"


def selected_to_prot_row(item: SelectedInterface) -> dict[str, Any]:
    definition = item.definition
    return {
        "模型名称": definition.name,
        "评估数据集": definition.eval_dataset,
        "数据子集": definition.subset,
        "PDB编号": definition.entry_id,
        "界面类型": definition.eval_type,
        "链1": definition.chain_1,
        "链2": definition.chain_2,
        "实体1": definition.entity_1,
        "实体2": definition.entity_2,
        "cluster编号": definition.cluster_id,
        "选择方式": item.method,
        "seed": item.candidate.seed,
        "sample": item.candidate.sample,
        "定位键": item.candidate.locator,
        "选择依据分数": dockq_text(item.selection_score),
        "DockQ": dockq_text(item.dockq),
        "DockQ质量等级": quality_label(item.dockq),
        "是否DockQ≥0.23": yes_no(item.dockq >= 0.23),
        "是否DockQ≥0.49": yes_no(item.dockq >= 0.49),
        "是否DockQ≥0.80": yes_no(item.dockq >= 0.80),
    }


def cluster_equal_statistics(
    rows: Sequence[SelectedInterface],
) -> tuple[float, dict[float, float]]:
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_cluster[row.definition.cluster_id].append(row.dockq)
    if not by_cluster:
        raise SimplifyError("Cannot aggregate an empty cluster collection")
    average = mean(mean(values) for values in by_cluster.values())
    rates = {
        threshold: mean(
            mean(float(value >= threshold) for value in values)
            for values in by_cluster.values()
        )
        for threshold in THRESHOLDS
    }
    return average, rates


def build_prot_sum_rows(
    selected: Sequence[SelectedInterface],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[SelectedInterface]] = defaultdict(list)
    for row in selected:
        groups[(row.definition.eval_type, row.method)].append(row)
    output: list[dict[str, Any]] = []
    for (eval_type, method), rows in sorted(
        groups.items(), key=lambda item: (item[0][0], METHOD_ORDER[item[0][1]])
    ):
        average, rates = cluster_equal_statistics(rows)
        values = [row.dockq for row in rows]
        output.append(
            {
                "界面类型": eval_type,
                "界面实例数量": len(rows),
                "独立cluster数量": len({row.definition.cluster_id for row in rows}),
                "PDB数量": len({row.definition.entry_id for row in rows}),
                "选择方式": method,
                "聚合方式": "cluster等权",
                "平均DockQ": dockq_text(average),
                "DockQ范围（最小值~最大值）": dockq_range(values),
                "成功率（%，DockQ≥0.23）": percent(rates[0.23]),
                "成功率（%，DockQ≥0.49）": percent(rates[0.49]),
                "成功率（%，DockQ≥0.80）": percent(rates[0.80]),
            }
        )
    return output


def build_pdb_inst_rows(
    definitions: Mapping[tuple[str, tuple[str, str]], InterfaceDefinition],
    candidates_by_entry: Mapping[str, Sequence[SampleCandidate]],
) -> list[dict[str, Any]]:
    definitions_by_entry: dict[str, list[InterfaceDefinition]] = defaultdict(list)
    for definition in definitions.values():
        definitions_by_entry[definition.entry_id].append(definition)
    output: list[dict[str, Any]] = []
    for entry_id in sorted(definitions_by_entry):
        entry_definitions = sorted(
            definitions_by_entry[entry_id], key=lambda item: item.pair
        )
        pairs = [definition.pair for definition in entry_definitions]
        candidates = candidates_by_entry[entry_id]
        selections = (
            (ORACLE, choose_pdb_oracle(candidates, pairs)),
            (PREDICTED_BEST, choose_predicted(candidates)),
        )
        for method, candidate in selections:
            values = [candidate.dockq_by_pair[pair] for pair in pairs]
            minimum = min(values)
            selection_score = (
                mean(values) if method == ORACLE else candidate.ranking_score
            )
            row: dict[str, Any] = {
                "PDB编号": entry_id,
                "包含的界面类型": "|".join(
                    sorted({item.eval_type for item in entry_definitions})
                ),
                "界面数量": len(values),
                "选择方式": method,
                "seed": candidate.seed,
                "sample": candidate.sample,
                "定位键": candidate.locator,
                "选择依据分数": dockq_text(selection_score),
                "PDB界面平均DockQ": dockq_text(mean(values)),
                "DockQ范围（最小值~最大值）": dockq_range(values),
                "最低界面DockQ": dockq_text(minimum),
                "PDB整体质量等级": quality_label(minimum),
            }
            for threshold in THRESHOLDS:
                label = f"{threshold:.2f}"
                success_count = sum(value >= threshold for value in values)
                row[f"DockQ≥{label}界面数"] = success_count
                row[f"DockQ≥{label}界面成功率（%）"] = percent(
                    success_count / len(values)
                )
                row[f"PDB是否DockQ≥{label}"] = yes_no(minimum >= threshold)
            output.append(row)
    return output


def build_pdb_sum_rows(pdb_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in (ORACLE, PREDICTED_BEST):
        rows = [row for row in pdb_rows if row["选择方式"] == method]
        if not rows:
            raise SimplifyError(f"No PDB rows for {method}")
        averages = [float(row["PDB界面平均DockQ"]) for row in rows]
        summary: dict[str, Any] = {
            "PDB数量": len(rows),
            "界面总数": sum(int(row["界面数量"]) for row in rows),
            "选择方式": method,
            "PDB等权平均DockQ": dockq_text(mean(averages)),
            "PDB平均DockQ范围（最小值~最大值）": dockq_range(averages),
            "平均每PDB界面数量": (f"{mean(int(row['界面数量']) for row in rows):.2f}"),
        }
        for threshold in THRESHOLDS:
            label = f"{threshold:.2f}"
            count = sum(row[f"PDB是否DockQ≥{label}"] == "是" for row in rows)
            summary[f"DockQ≥{label}成功PDB数"] = count
            summary[f"DockQ≥{label}成功PDB率（%）"] = percent(count / len(rows))
        output.append(summary)
    return output


def parse_bool(value: str, label: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise SimplifyError(f"Invalid boolean {label}: {value!r}")


def load_abag_roles(
    annotations_path: Path,
) -> dict[tuple[str, str], tuple[str, bool]]:
    rows = read_csv(annotations_path, ANNOTATION_REQUIRED_COLUMNS)
    roles: dict[tuple[str, str], tuple[str, bool]] = {}
    for row in rows:
        entry_id = row["pdb_id"].strip()
        chain_id = row["uni_chain_id"].strip()
        role = row["role"].strip()
        if not entry_id or not chain_id or not role:
            raise SimplifyError(f"Empty antibody annotation in {annotations_path}")
        value = (
            role,
            parse_bool(
                row["is_val_antigen"],
                f"is_val_antigen for {entry_id}:{chain_id}",
            ),
        )
        key = (entry_id, chain_id)
        if key in roles and roles[key] != value:
            raise SimplifyError(f"Conflicting antibody annotation for {key}")
        roles[key] = value
    return roles


def load_abag_subset(path: Path) -> set[tuple[str, tuple[str, str]]]:
    rows = read_csv(path, SUBSET_REQUIRED_COLUMNS)
    result = {
        (
            row["entry_id"].strip(),
            canonical_pair(row["chain_id_1"].strip(), row["chain_id_2"].strip()),
        )
        for row in rows
        if row["type"].strip().lower() == "interface"
    }
    if not result:
        raise SimplifyError(f"No antibody interfaces in {path}")
    return result


AB_ROLE_LABELS = {
    "antibody_heavy": ("重链", "重链-抗原"),
    "antibody_light": ("轻链", "轻链-抗原"),
    "antibody_scfv": ("scFv", "scFv-抗原"),
}


def identify_abag_interface(
    definition: InterfaceDefinition,
    roles: Mapping[tuple[str, str], tuple[str, bool]],
) -> tuple[str, str, str, str]:
    chain_annotations = []
    for chain in (definition.chain_1, definition.chain_2):
        key = (definition.entry_id, chain)
        if key not in roles:
            raise SimplifyError(f"Missing antibody annotation for {key}")
        role, is_antigen = roles[key]
        chain_annotations.append((chain, role, is_antigen))
    antibody = [item for item in chain_annotations if item[1] in AB_ROLE_LABELS]
    antigen = [
        item for item in chain_annotations if item[1] == "non_antibody" and item[2]
    ]
    if len(antibody) != 1 or len(antigen) != 1:
        raise SimplifyError(
            "Expected exactly one antibody and one antigen chain for "
            f"{definition.entry_id}:{definition.chain_1},{definition.chain_2}"
        )
    antibody_chain, antibody_role, _ = antibody[0]
    antigen_chain = antigen[0][0]
    type_label, group_label = AB_ROLE_LABELS[antibody_role]
    return antibody_chain, antigen_chain, type_label, group_label


def build_abag_rows(
    selected: Sequence[SelectedInterface],
    subset: set[tuple[str, tuple[str, str]]],
    roles: Mapping[tuple[str, str], tuple[str, bool]],
) -> tuple[list[dict[str, Any]], dict[int, str]]:
    rows: list[dict[str, Any]] = []
    groups: dict[int, str] = {}
    for index, item in enumerate(selected):
        key = (item.definition.entry_id, item.definition.pair)
        if key not in subset:
            raise SimplifyError(f"AbAg interface is absent from subset.csv: {key}")
        antibody, antigen, type_label, group_label = identify_abag_interface(
            item.definition, roles
        )
        row = selected_to_prot_row(item)
        row.update(
            {
                "抗体链": antibody,
                "抗原链": antigen,
                "抗体链类型": type_label,
                "抗原-抗体界面分组": group_label,
            }
        )
        rows.append(row)
        groups[index] = group_label
    return rows, groups


def build_abag_sum_rows(
    selected: Sequence[SelectedInterface], groups: Mapping[int, str]
) -> list[dict[str, Any]]:
    group_order = {
        "全部抗原-抗体界面": 0,
        "重链-抗原": 1,
        "轻链-抗原": 2,
        "scFv-抗原": 3,
    }
    grouped: dict[tuple[str, str], list[SelectedInterface]] = defaultdict(list)
    for index, item in enumerate(selected):
        grouped[("全部抗原-抗体界面", item.method)].append(item)
        grouped[(groups[index], item.method)].append(item)
    output: list[dict[str, Any]] = []
    for (group, method), rows in sorted(
        grouped.items(),
        key=lambda item: (group_order[item[0][0]], METHOD_ORDER[item[0][1]]),
    ):
        average, rates = cluster_equal_statistics(rows)
        values = [row.dockq for row in rows]
        output.append(
            {
                "抗原-抗体界面分组": group,
                "界面实例数量": len(rows),
                "独立cluster数量": len({row.definition.cluster_id for row in rows}),
                "PDB数量": len({row.definition.entry_id for row in rows}),
                "选择方式": method,
                "聚合方式": "cluster等权",
                "平均DockQ": dockq_text(average),
                "DockQ范围（最小值~最大值）": dockq_range(values),
                "成功率（%，DockQ≥0.23）": percent(rates[0.23]),
                "成功率（%，DockQ≥0.49）": percent(rates[0.49]),
                "成功率（%，DockQ≥0.80）": percent(rates[0.80]),
            }
        )
    return output


def write_csv(
    path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def build_pdb_structure_exports(
    pred_dir: Path,
    ref_dir: Path,
    pdb_rows: Sequence[Mapping[str, Any]],
) -> list[PdbStructureExport]:
    pred_dir = pred_dir.expanduser().resolve()
    ref_dir = ref_dir.expanduser().resolve()
    if not pred_dir.is_dir():
        raise SimplifyError(f"Prediction directory does not exist: {pred_dir}")
    if not ref_dir.is_dir():
        raise SimplifyError(f"Reference directory does not exist: {ref_dir}")

    suffix_by_method = {
        ORACLE: "_oracle.cif",
        PREDICTED_BEST: "_pbest.cif",
    }
    exports: list[PdbStructureExport] = []
    seen_destinations: set[str] = set()
    for row in pdb_rows:
        entry_id = str(row["PDB编号"])
        if not entry_id or Path(entry_id).name != entry_id or entry_id in {".", ".."}:
            raise SimplifyError(
                f"Invalid PDB identifier for structure export: {entry_id!r}"
            )
        method = str(row["选择方式"])
        suffix = suffix_by_method.get(method)
        if suffix is None:
            raise SimplifyError(f"Unsupported PDB selection method: {method!r}")
        try:
            seed = int(row["seed"])
            sample = int(row["sample"])
        except (TypeError, ValueError) as exc:
            raise SimplifyError(
                f"Invalid seed/sample for PDB structure export: {entry_id}"
            ) from exc

        source = (
            pred_dir
            / entry_id
            / f"seed_{seed}"
            / "predictions"
            / f"{entry_id}_sample_{sample}.cif"
        )
        if not source.is_file():
            raise SimplifyError(
                f"Missing selected PDB structure for {entry_id} ({method}): {source}"
            )
        filename = f"{entry_id}{suffix}"
        if filename in seen_destinations:
            raise SimplifyError(f"Duplicate PDB structure destination: {filename}")
        seen_destinations.add(filename)
        exports.append(PdbStructureExport(source=source, filename=filename))

    for entry_id in sorted({str(row["PDB编号"]) for row in pdb_rows}):
        plain_source = ref_dir / f"{entry_id}.cif"
        compressed_source = ref_dir / f"{entry_id}.cif.gz"
        if plain_source.is_file():
            source = plain_source
            decompress_gzip = False
        elif compressed_source.is_file():
            source = compressed_source
            decompress_gzip = True
        else:
            raise SimplifyError(
                f"Missing ground-truth PDB structure for {entry_id}: "
                f"expected {plain_source} or {compressed_source}"
            )
        filename = f"{entry_id}_gt.cif"
        if filename in seen_destinations:
            raise SimplifyError(f"Duplicate PDB structure destination: {filename}")
        seen_destinations.add(filename)
        exports.append(
            PdbStructureExport(
                source=source,
                filename=filename,
                decompress_gzip=decompress_gzip,
            )
        )
    return exports


def remove_published_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def copy_pdb_structure(export: PdbStructureExport, destination: Path) -> None:
    if export.decompress_gzip:
        with gzip.open(export.source, "rb") as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target)
        shutil.copystat(export.source, destination)
    else:
        shutil.copy2(export.source, destination)


def publish_csv_bundle(
    output_dir: Path,
    tables: Mapping[str, tuple[Sequence[str], Sequence[Mapping[str, Any]]]],
    pdb_exports: Sequence[PdbStructureExport],
) -> None:
    # 先在同一文件系统的临时目录写完所有 CSV 和 CIF，再逐个
    # 原子替换；任何一步失败都恢复旧输出，避免产生新旧混合的结果集。
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".dockq_simplify_stage_", dir=output_dir.parent)
    )
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for filename in MANAGED_OUTPUT_FILENAMES:
            if filename not in tables:
                continue
            columns, rows = tables[filename]
            write_csv(staging / filename, columns, rows)
        staged_pdb_dir = staging / PDB_OUTPUT_DIRNAME
        staged_pdb_dir.mkdir()
        for export in pdb_exports:
            copy_pdb_structure(export, staged_pdb_dir / export.filename)
        # 先备份所有受管文件。``tables`` 未包含的是当前数据集不适用的
        # 可选指标；同步移除它们的旧版本，防止将过期结果误认为新结果。
        managed_destinations = [
            *(output_dir / filename for filename in MANAGED_OUTPUT_FILENAMES),
            output_dir / PDB_OUTPUT_DIRNAME,
        ]
        for destination in managed_destinations:
            if destination.exists():
                backup = output_dir / (f".{destination.name}.backup.{uuid.uuid4().hex}")
                os.replace(destination, backup)
                backups[destination] = backup
        for filename in MANAGED_OUTPUT_FILENAMES:
            if filename not in tables:
                continue
            destination = output_dir / filename
            os.replace(staging / filename, destination)
            published.append(destination)
        pdb_destination = output_dir / PDB_OUTPUT_DIRNAME
        os.replace(staged_pdb_dir, pdb_destination)
        published.append(pdb_destination)
    except Exception:
        for destination in reversed(published):
            remove_published_path(destination)
        for destination, backup in backups.items():
            if backup.exists():
                os.replace(backup, destination)
        raise
    else:
        for backup in backups.values():
            remove_published_path(backup)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


PB_VALID_KEYS = (
    "sanitization",
    "molecular_formula",
    "molecular_bonds",
    "tetrahedral_chirality",
    "double_bond_stereochemistry",
    "bond_lengths",
    "bond_angles",
    "aromatic_ring_flatness",
    "non_aromatic_ring_non_flatness",
    "double_bond_flatness",
    "sp2_center_flatness",
    "amide_flatness",
    "sp3_center_non_flatness",
    "internal_steric_clash",
    "internal_energy",
    "minimum_distance_to_protein",
    "minimum_distance_to_organic_cofactors",
    "minimum_distance_to_inorganic_cofactors",
    "volume_overlap_with_protein",
    "volume_overlap_with_organic_cofactors",
    "volume_overlap_with_inorganic_cofactors",
    "no_radicals",
)

CDR_METRICS = (
    "cdr_h1_bb_rmsd",
    "cdr_h2_bb_rmsd",
    "cdr_h3_bb_rmsd",
    "cdr_l1_bb_rmsd",
    "cdr_l2_bb_rmsd",
    "cdr_l3_bb_rmsd",
)
CDR_LABELS = {
    "cdr_h1_bb_rmsd": "CDR-H1",
    "cdr_h2_bb_rmsd": "CDR-H2",
    "cdr_h3_bb_rmsd": "CDR-H3",
    "cdr_l1_bb_rmsd": "CDR-L1",
    "cdr_l2_bb_rmsd": "CDR-L2",
    "cdr_l3_bb_rmsd": "CDR-L3",
}
@dataclass(frozen=True)
class RawSample:
    entry_id: str
    seed: int
    sample: int
    ranking_score: float
    complex_metrics: Mapping[str, float]
    chain_metrics: Mapping[str, Mapping[str, float]]
    interface_metrics: Mapping[tuple[str, str], Mapping[str, float]]

    @property
    def locator(self) -> str:
        return f"{self.entry_id}|seed={self.seed}|sample={self.sample}"


@dataclass(frozen=True)
class LddtDefinition:
    name: str
    eval_dataset: str
    subset: str
    eval_type: str
    entry_id: str
    entity_1: str
    entity_2: str
    chain_1: str
    chain_2: str
    cluster_id: str

    @property
    def level(self) -> str:
        return "界面" if self.chain_2 else "单链"

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.entry_id, self.eval_type, self.chain_1, self.chain_2)


@dataclass(frozen=True)
class SelectedLddt:
    definition: LddtDefinition
    method: str
    sample: RawSample
    lddt: float
    bb_lddt: float | None
    selection_score: float


@dataclass(frozen=True)
class LigandRecord:
    entry_id: str
    chain_id: str
    seed: int
    sample: int
    ranking_score: float
    lig_rmsd: float
    pocket_rmsd: float
    lddt_pli: float | None
    pb_valid: bool
    pb_failures: tuple[str, ...]
    name: str
    eval_dataset: str
    subset: str
    entity_id: str
    cluster_id: str

    @property
    def locator(self) -> str:
        return f"{self.entry_id}|seed={self.seed}|sample={self.sample}"


@dataclass(frozen=True)
class CdrRecord:
    entry_id: str
    chain_id: str
    antibody_role: str
    metric: str
    seed: int
    sample: int
    ranking_score: float
    rmsd: float

    @property
    def locator(self) -> str:
        return f"{self.entry_id}|seed={self.seed}|sample={self.sample}"


def finite(value: Any, label: str, *, unit_interval: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SimplifyError(f"Invalid {label}: {value!r}") from exc
    if not math.isfinite(result):
        raise SimplifyError(f"Invalid {label}: {value!r}")
    if unit_interval and not 0.0 <= result <= 1.0:
        raise SimplifyError(f"{label} must be in [0, 1], got {result}")
    return result


def nonnegative(value: Any, label: str) -> float:
    result = finite(value, label)
    if result < 0:
        raise SimplifyError(f"{label} must be non-negative, got {result}")
    return result


def optional_finite(
    value: Any, label: str, *, unit_interval: bool = False
) -> float | None:
    """Parse an optional numeric CSV field without turning missing data into zero."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return finite(value, label, unit_interval=unit_interval)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SimplifyError(f"Cannot parse {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SimplifyError(f"Expected a JSON object in {path}")
    return value


def load_raw_samples(per_sample_dir: Path) -> dict[str, list[RawSample]]:
    if not per_sample_dir.is_dir():
        raise SimplifyError(f"Missing per-sample directory: {per_sample_dir}")
    output: dict[str, list[RawSample]] = defaultdict(list)
    seen: set[tuple[str, int, int]] = set()
    for path in sorted(per_sample_dir.glob("*/*/sample_*_metrics.json")):
        match = SAMPLE_RE.match(path.name)
        if match is None:
            continue
        entry_id = path.parent.parent.name
        try:
            seed = int(path.parent.name)
            sample = int(match.group(1))
        except ValueError as exc:
            raise SimplifyError(f"Invalid seed/sample path: {path}") from exc
        key = (entry_id, seed, sample)
        if key in seen:
            raise SimplifyError(f"Duplicate raw sample: {key}")
        seen.add(key)
        metrics = read_json(path)
        confidence = read_json(path.with_name(f"sample_{sample}_confidences.json"))
        complex_conf = confidence.get("complex")
        if not isinstance(complex_conf, dict):
            raise SimplifyError(f"Missing complex confidence in {path}")
        ranking_score = finite(
            complex_conf.get("ranking_score"), f"ranking_score in {path}"
        )

        complex_metrics = _numeric_metric_map(
            metrics.get("complex"), f"complex metrics in {path}"
        )
        chain_root = metrics.get("chain")
        interface_root = metrics.get("interface")
        if not isinstance(chain_root, dict) or not isinstance(interface_root, dict):
            raise SimplifyError(f"Missing chain/interface metrics in {path}")
        chains = {
            str(chain): _numeric_metric_map(values, f"chain {chain} in {path}")
            for chain, values in chain_root.items()
        }
        interfaces: dict[tuple[str, str], Mapping[str, float]] = {}
        for raw_pair, values in interface_root.items():
            parts = [part.strip() for part in str(raw_pair).split(",")]
            if len(parts) != 2:
                raise SimplifyError(f"Invalid interface key {raw_pair!r} in {path}")
            pair = canonical_pair(*parts)
            if pair in interfaces:
                raise SimplifyError(f"Duplicate interface {pair} in {path}")
            interfaces[pair] = _numeric_metric_map(
                values, f"interface {raw_pair} in {path}"
            )
        output[entry_id].append(
            RawSample(
                entry_id,
                seed,
                sample,
                ranking_score,
                complex_metrics,
                chains,
                interfaces,
            )
        )
    if not output:
        raise SimplifyError(f"No raw samples in {per_sample_dir}")
    return dict(output)


def _numeric_metric_map(value: Any, label: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise SimplifyError(f"Expected metric object for {label}")
    output = {}
    for key, item in value.items():
        if isinstance(item, bool):
            continue
        if isinstance(item, (int, float)):
            number = finite(item, f"{key} in {label}")
            if key in {"lddt", "bb_lddt", "lddt_pli"} and not 0 <= number <= 1:
                raise SimplifyError(f"{key} in {label} must be in [0, 1]")
            if key.endswith("rmsd") and number < 0:
                raise SimplifyError(f"{key} in {label} must be non-negative")
            output[str(key)] = number
    return output


def predicted_sample(samples: Sequence[RawSample]) -> RawSample:
    return max(samples, key=lambda row: (row.ranking_score, -row.seed, -row.sample))


def fmt(value: float | None, digits: int = 4) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def pct(value: float) -> str:
    return f"{value * 100:.2f}"


def value_range(values: Sequence[float]) -> str:
    return f"{min(values):.4f}~{max(values):.4f}"


def cluster_mean(rows: Sequence[Any], value_getter: Any) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row.definition.cluster_id].append(float(value_getter(row)))
    if not grouped:
        raise SimplifyError("Cannot aggregate empty clusters")
    return mean(mean(values) for values in grouped.values())


def cluster_rate(rows: Sequence[Any], predicate: Any) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row.definition.cluster_id].append(float(predicate(row)))
    return mean(mean(values) for values in grouped.values())


def grouped_rate(groups: Iterable[Sequence[Any]], predicate: Any) -> float:
    """Return the equal-weight mean of per-group Boolean rates."""
    return mean(mean(float(predicate(row)) for row in group) for group in groups)


LDDT_DETAIL_COLUMNS = {
    "name",
    "eval_dataset",
    "subset",
    "eval_type",
    "entry_id",
    "entity_id_1",
    "entity_id_2",
    "chain_id_1",
    "chain_id_2",
    "cluster_id",
    "ranker",
}


def load_lddt_definitions(
    path: Path, ranker: str
) -> dict[tuple[str, str, str, str], LddtDefinition]:
    output = {}
    for row in read_csv(path, LDDT_DETAIL_COLUMNS):
        if row["ranker"].strip() != ranker:
            continue
        entry = row["entry_id"].strip()
        eval_type = row["eval_type"].strip()
        chain_1 = row["chain_id_1"].strip()
        chain_2 = row["chain_id_2"].strip()
        entity_1 = row["entity_id_1"].strip()
        entity_2 = row["entity_id_2"].strip()
        if not entry or not eval_type or not chain_1 or not row["cluster_id"].strip():
            raise SimplifyError(f"Empty LDDT identity field in {path}")
        if chain_2 and chain_2 < chain_1:
            chain_1, chain_2 = chain_2, chain_1
            entity_1, entity_2 = entity_2, entity_1
        definition = LddtDefinition(
            row["name"].strip(),
            row["eval_dataset"].strip(),
            row["subset"].strip(),
            eval_type,
            entry,
            entity_1,
            entity_2,
            chain_1,
            chain_2,
            row["cluster_id"].strip(),
        )
        if definition.key in output and output[definition.key] != definition:
            raise SimplifyError(f"Conflicting LDDT definition: {definition.key}")
        output[definition.key] = definition
    if not output:
        raise SimplifyError(f"No LDDT rows for ranker {ranker} in {path}")
    return output


def lddt_values(
    sample: RawSample, definition: LddtDefinition
) -> tuple[float, float | None]:
    if definition.eval_type == "LDDT-PLI":
        metrics = sample.chain_metrics.get(definition.chain_1)
        metric_name = "lddt_pli"
    elif definition.chain_2:
        metrics = sample.interface_metrics.get(
            canonical_pair(definition.chain_1, definition.chain_2)
        )
        metric_name = "lddt"
    else:
        metrics = sample.chain_metrics.get(definition.chain_1)
        metric_name = "lddt"
    if metrics is None or metric_name not in metrics:
        raise SimplifyError(
            f"Missing {metric_name} for {definition.key} in {sample.locator}"
        )
    return metrics[metric_name], metrics.get("bb_lddt")


def select_lddt(
    oracle_defs: Mapping[tuple[str, str, str, str], LddtDefinition],
    predicted_defs: Mapping[tuple[str, str, str, str], LddtDefinition],
    samples: Mapping[str, Sequence[RawSample]],
) -> list[SelectedLddt]:
    if oracle_defs.keys() != predicted_defs.keys():
        raise SimplifyError("LDDT best and best.ranking_score object sets differ")
    predicted = {entry: predicted_sample(values) for entry, values in samples.items()}
    output = []
    for key in sorted(oracle_defs):
        oracle_def = oracle_defs[key]
        pred_def = predicted_defs[key]
        entry_samples = samples.get(oracle_def.entry_id)
        if not entry_samples:
            raise SimplifyError(f"Missing samples for {oracle_def.entry_id}")
        ranked = []
        for sample in entry_samples:
            lddt, bb_lddt = lddt_values(sample, oracle_def)
            ranked.append(
                (
                    lddt,
                    sample.ranking_score,
                    -sample.seed,
                    -sample.sample,
                    sample,
                    bb_lddt,
                )
            )
        _, _, _, _, oracle_sample, oracle_bb = max(ranked, key=lambda row: row[:4])
        oracle_lddt, _ = lddt_values(oracle_sample, oracle_def)
        pred_sample = predicted[oracle_def.entry_id]
        pred_lddt, pred_bb = lddt_values(pred_sample, pred_def)
        output.extend(
            (
                SelectedLddt(
                    oracle_def,
                    ORACLE,
                    oracle_sample,
                    oracle_lddt,
                    oracle_bb,
                    oracle_lddt,
                ),
                SelectedLddt(
                    pred_def,
                    PREDICTED_BEST,
                    pred_sample,
                    pred_lddt,
                    pred_bb,
                    pred_sample.ranking_score,
                ),
            )
        )
    return output


LDDT_STRUCTURE_INST_COLUMNS = (
    "模型名称",
    "评估数据集",
    "数据子集",
    "PDB编号",
    "评估层级",
    "对象类型",
    "链1",
    "链2",
    "实体1",
    "实体2",
    "cluster编号",
    "选择方式",
    "seed",
    "sample",
    "定位键",
    "选择依据分数",
    "LDDT",
    "Backbone LDDT",
    "是否LDDT-PLI≥0.5（参考）",
)
LDDT_STRUCTURE_SUM_COLUMNS = (
    "评估层级",
    "对象类型",
    "实例数量",
    "独立cluster数量",
    "PDB数量",
    "选择方式",
    "聚合方式",
    "平均LDDT",
    "实例中位数LDDT",
    "LDDT范围（最小值~最大值）",
    "平均Backbone LDDT",
    "LDDT-PLI≥0.5参考率（%）",
)


def selected_lddt_inst_row(row: SelectedLddt) -> dict[str, Any]:
    definition = row.definition
    return {
        "模型名称": definition.name,
        "评估数据集": definition.eval_dataset,
        "数据子集": definition.subset,
        "PDB编号": definition.entry_id,
        "评估层级": definition.level,
        "对象类型": definition.eval_type,
        "链1": definition.chain_1,
        "链2": definition.chain_2,
        "实体1": definition.entity_1,
        "实体2": definition.entity_2,
        "cluster编号": definition.cluster_id,
        "选择方式": row.method,
        "seed": row.sample.seed,
        "sample": row.sample.sample,
        "定位键": row.sample.locator,
        "选择依据分数": fmt(row.selection_score),
        "LDDT": fmt(row.lddt),
        "Backbone LDDT": fmt(row.bb_lddt),
        "是否LDDT-PLI≥0.5（参考）": (
            yes_no(row.lddt >= 0.5) if definition.eval_type == "LDDT-PLI" else ""
        ),
    }


def lddt_structure_tables(
    selected: Sequence[SelectedLddt],
) -> dict[str, tuple[Sequence[str], Sequence[Mapping[str, Any]]]]:
    inst = [selected_lddt_inst_row(row) for row in selected]
    inst.sort(
        key=lambda row: (
            row["PDB编号"],
            row["对象类型"],
            row["链1"],
            row["链2"],
            METHOD_ORDER[row["选择方式"]],
        )
    )
    grouped: dict[tuple[str, str, str], list[SelectedLddt]] = defaultdict(list)
    for row in selected:
        grouped[(row.definition.level, row.definition.eval_type, row.method)].append(
            row
        )
    sums = []
    for (level, eval_type, method), rows in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][1], METHOD_ORDER[item[0][2]]),
    ):
        values = [row.lddt for row in rows]
        bb_rows = [row for row in rows if row.bb_lddt is not None]
        sums.append(
            {
                "评估层级": level,
                "对象类型": eval_type,
                "实例数量": len(rows),
                "独立cluster数量": len({row.definition.cluster_id for row in rows}),
                "PDB数量": len({row.definition.entry_id for row in rows}),
                "选择方式": method,
                "聚合方式": "cluster等权",
                "平均LDDT": fmt(cluster_mean(rows, lambda item: item.lddt)),
                "实例中位数LDDT": fmt(median(values)),
                "LDDT范围（最小值~最大值）": value_range(values),
                "平均Backbone LDDT": (
                    fmt(cluster_mean(bb_rows, lambda item: item.bb_lddt))
                    if bb_rows
                    else ""
                ),
                "LDDT-PLI≥0.5参考率（%）": (
                    pct(cluster_rate(rows, lambda item: item.lddt >= 0.5))
                    if eval_type == "LDDT-PLI"
                    else ""
                ),
            }
        )
    return {
        "LDDT_structure_sum.csv": (LDDT_STRUCTURE_SUM_COLUMNS, sums),
        "LDDT_structure_inst.csv": (LDDT_STRUCTURE_INST_COLUMNS, inst),
    }


LDDT_PDB_INST_COLUMNS = (
    "PDB编号",
    "选择方式",
    "seed",
    "sample",
    "定位键",
    "选择依据分数",
    "Complex LDDT",
    "Complex Backbone LDDT",
    "Oracle相对预测最佳LDDT提升",
)
LDDT_PDB_SUM_COLUMNS = (
    "PDB数量",
    "选择方式",
    "PDB等权平均LDDT",
    "PDB中位数LDDT",
    "PDB LDDT范围（最小值~最大值）",
    "PDB等权平均Backbone LDDT",
    "Oracle相对预测最佳平均提升",
)


def lddt_pdb_tables(
    samples: Mapping[str, Sequence[RawSample]],
) -> dict[str, tuple[Sequence[str], Sequence[Mapping[str, Any]]]]:
    chosen: dict[tuple[str, str], RawSample] = {}
    for entry, values in samples.items():
        valid = [row for row in values if "lddt" in row.complex_metrics]
        if not valid:
            raise SimplifyError(f"Missing complex LDDT for {entry}")
        chosen[(entry, ORACLE)] = max(
            valid,
            key=lambda row: (
                row.complex_metrics["lddt"],
                row.ranking_score,
                -row.seed,
                -row.sample,
            ),
        )
        chosen[(entry, PREDICTED_BEST)] = predicted_sample(valid)
    inst = []
    for entry in sorted(samples):
        oracle_value = chosen[(entry, ORACLE)].complex_metrics["lddt"]
        predicted_value = chosen[(entry, PREDICTED_BEST)].complex_metrics["lddt"]
        for method in METHODS:
            sample = chosen[(entry, method)]
            inst.append(
                {
                    "PDB编号": entry,
                    "选择方式": method,
                    "seed": sample.seed,
                    "sample": sample.sample,
                    "定位键": sample.locator,
                    "选择依据分数": fmt(
                        sample.complex_metrics["lddt"]
                        if method == ORACLE
                        else sample.ranking_score
                    ),
                    "Complex LDDT": fmt(sample.complex_metrics["lddt"]),
                    "Complex Backbone LDDT": fmt(sample.complex_metrics.get("bb_lddt")),
                    "Oracle相对预测最佳LDDT提升": fmt(oracle_value - predicted_value)
                    if method == ORACLE
                    else "",
                }
            )
    sums = []
    oracle_mean = mean(
        chosen[(entry, ORACLE)].complex_metrics["lddt"] for entry in samples
    )
    predicted_mean = mean(
        chosen[(entry, PREDICTED_BEST)].complex_metrics["lddt"] for entry in samples
    )
    for method in METHODS:
        selected = [chosen[(entry, method)] for entry in sorted(samples)]
        values = [row.complex_metrics["lddt"] for row in selected]
        bb_values = [
            row.complex_metrics["bb_lddt"]
            for row in selected
            if "bb_lddt" in row.complex_metrics
        ]
        sums.append(
            {
                "PDB数量": len(selected),
                "选择方式": method,
                "PDB等权平均LDDT": fmt(mean(values)),
                "PDB中位数LDDT": fmt(median(values)),
                "PDB LDDT范围（最小值~最大值）": value_range(values),
                "PDB等权平均Backbone LDDT": fmt(mean(bb_values)) if bb_values else "",
                "Oracle相对预测最佳平均提升": fmt(oracle_mean - predicted_mean)
                if method == ORACLE
                else "",
            }
        )
    return {
        "LDDT_pdb_sum.csv": (LDDT_PDB_SUM_COLUMNS, sums),
        "LDDT_pdb_inst.csv": (LDDT_PDB_INST_COLUMNS, inst),
    }


def abag_group(
    definition: LddtDefinition, roles: Mapping[tuple[str, str], tuple[str, bool]]
) -> tuple[str, str, str, str]:
    annotated = []
    for chain in (definition.chain_1, definition.chain_2):
        key = (definition.entry_id, chain)
        if key not in roles:
            raise SimplifyError(f"Missing AbAg annotation for {key}")
        annotated.append((chain, *roles[key]))
    antibody = [row for row in annotated if row[1] in AB_ROLE_LABELS]
    antigen = [row for row in annotated if row[1] == "non_antibody" and row[2]]
    if len(antibody) != 1 or len(antigen) != 1:
        raise SimplifyError(f"Invalid AbAg roles for {definition.key}")
    type_label, group = AB_ROLE_LABELS[antibody[0][1]]
    return antibody[0][0], antigen[0][0], type_label, group


LDDT_ABAG_INST_COLUMNS = LDDT_STRUCTURE_INST_COLUMNS + (
    "抗体链",
    "抗原链",
    "抗体链类型",
    "抗原-抗体界面分组",
)
LDDT_ABAG_SUM_COLUMNS = (
    "抗原-抗体界面分组",
    "实例数量",
    "独立cluster数量",
    "PDB数量",
    "选择方式",
    "聚合方式",
    "平均LDDT",
    "实例中位数LDDT",
    "LDDT范围（最小值~最大值）",
    "平均Backbone LDDT",
)


def lddt_abag_tables(
    selected: Sequence[SelectedLddt], roles: Mapping[tuple[str, str], tuple[str, bool]]
) -> dict[str, tuple[Sequence[str], Sequence[Mapping[str, Any]]]]:
    selected = [
        row for row in selected if row.definition.eval_type == "Protein-Protein"
    ]
    if not selected:
        return {}
    inst = []
    groups: dict[int, str] = {}
    for index, item in enumerate(selected):
        antibody, antigen, type_label, group = abag_group(item.definition, roles)
        row = selected_lddt_inst_row(item)
        row.update(
            {
                "抗体链": antibody,
                "抗原链": antigen,
                "抗体链类型": type_label,
                "抗原-抗体界面分组": group,
            }
        )
        inst.append(row)
        groups[index] = group
    inst.sort(
        key=lambda row: (
            row["PDB编号"],
            row["抗原-抗体界面分组"],
            row["抗体链"],
            METHOD_ORDER[row["选择方式"]],
        )
    )
    grouped: dict[tuple[str, str], list[SelectedLddt]] = defaultdict(list)
    for index, row in enumerate(selected):
        grouped[("全部抗原-抗体界面", row.method)].append(row)
        grouped[(groups[index], row.method)].append(row)
    group_order = {
        "全部抗原-抗体界面": 0,
        "重链-抗原": 1,
        "轻链-抗原": 2,
        "scFv-抗原": 3,
    }
    sums = []
    for (group, method), rows in sorted(
        grouped.items(),
        key=lambda item: (group_order[item[0][0]], METHOD_ORDER[item[0][1]]),
    ):
        values = [row.lddt for row in rows]
        bb_rows = [row for row in rows if row.bb_lddt is not None]
        sums.append(
            {
                "抗原-抗体界面分组": group,
                "实例数量": len(rows),
                "独立cluster数量": len({row.definition.cluster_id for row in rows}),
                "PDB数量": len({row.definition.entry_id for row in rows}),
                "选择方式": method,
                "聚合方式": "cluster等权",
                "平均LDDT": fmt(cluster_mean(rows, lambda item: item.lddt)),
                "实例中位数LDDT": fmt(median(values)),
                "LDDT范围（最小值~最大值）": value_range(values),
                "平均Backbone LDDT": fmt(
                    cluster_mean(bb_rows, lambda item: item.bb_lddt)
                )
                if bb_rows
                else "",
            }
        )
    return {
        "LDDT_abag_interface_sum.csv": (LDDT_ABAG_SUM_COLUMNS, sums),
        "LDDT_abag_interface_inst.csv": (LDDT_ABAG_INST_COLUMNS, inst),
    }


RMSD_DETAIL_COLUMNS = {
    "name",
    "eval_dataset",
    "subset",
    "entry_id",
    "entity_id_1",
    "chain_id_1",
    "cluster_id",
    "ranker",
    "eval_type",
}
LIGAND_SOURCE_COLUMNS = {
    "entry_id",
    "seed",
    "sample",
    "type",
    "antigen_chain",
    "lig_rmsd",
    "pocket_rmsd",
    "lddt_pli",
    "pb_valid_json",
}


def ligand_metadata(path: Path, ranker: str) -> dict[tuple[str, str], dict[str, str]]:
    output = {}
    for row in read_csv(path, RMSD_DETAIL_COLUMNS):
        if row["ranker"].strip() != ranker or row["eval_type"].strip() != "RMSD":
            continue
        key = (row["entry_id"].strip(), row["chain_id_1"].strip())
        output[key] = row
    return output


def load_ligands(
    input_dir: Path, raw_samples: Mapping[str, Sequence[RawSample]]
) -> list[LigandRecord]:
    source = input_dir / "antibody" / "per_sample_metrics.csv"
    details = input_dir / "summary" / "RMSD_details.csv"
    if not source.is_file() or not details.is_file():
        return []
    oracle_meta = ligand_metadata(details, "best")
    predicted_meta = ligand_metadata(details, "best.ranking_score")
    if not oracle_meta:
        return []
    if oracle_meta.keys() != predicted_meta.keys():
        raise SimplifyError("Ligand RMSD best and predicted object sets differ")
    ranking = {
        (row.entry_id, row.seed, row.sample): row.ranking_score
        for values in raw_samples.values()
        for row in values
    }
    samples_by_key = {
        (row.entry_id, row.seed, row.sample): row
        for values in raw_samples.values()
        for row in values
    }
    output = []
    seen = set()
    for row in read_csv(source, LIGAND_SOURCE_COLUMNS):
        if row["type"].strip() != "ligand" or not row["lig_rmsd"].strip():
            continue
        entry, chain = row["entry_id"].strip(), row["antigen_chain"].strip()
        key = (entry, chain)
        if key not in oracle_meta:
            continue
        seed, sample = int(row["seed"]), int(row["sample"])
        identity = (entry, chain, seed, sample)
        if identity in seen:
            raise SimplifyError(f"Duplicate ligand record: {identity}")
        seen.add(identity)
        rank_key = (entry, seed, sample)
        if rank_key not in ranking:
            raise SimplifyError(f"Missing ranking score for {rank_key}")
        raw_sample = samples_by_key[rank_key]
        raw_lddt_pli = raw_sample.chain_metrics.get(chain, {}).get("lddt_pli")
        lddt_pli = optional_finite(
            row["lddt_pli"], "LDDT-PLI", unit_interval=True
        )
        if lddt_pli is None:
            lddt_pli = raw_lddt_pli
        try:
            pb = json.loads(row["pb_valid_json"])
        except json.JSONDecodeError as exc:
            raise SimplifyError(f"Invalid PoseBusters JSON for {identity}") from exc
        if not isinstance(pb, dict):
            raise SimplifyError(f"Invalid PoseBusters object for {identity}")
        missing = [key for key in PB_VALID_KEYS if key not in pb]
        if missing:
            raise SimplifyError(
                f"Missing PoseBusters checks for {identity}: {missing}"
            )
        invalid_types = [key for key in PB_VALID_KEYS if not isinstance(pb[key], bool)]
        if invalid_types:
            raise SimplifyError(
                f"Non-boolean PoseBusters checks for {identity}: {invalid_types}"
            )
        failures = tuple(key for key in PB_VALID_KEYS if not pb[key])
        meta = oracle_meta[key]
        output.append(
            LigandRecord(
                entry,
                chain,
                seed,
                sample,
                ranking[rank_key],
                nonnegative(row["lig_rmsd"], "ligand RMSD"),
                nonnegative(row["pocket_rmsd"], "pocket RMSD"),
                lddt_pli,
                not failures,
                failures,
                meta["name"].strip(),
                meta["eval_dataset"].strip(),
                meta["subset"].strip(),
                meta["entity_id_1"].strip(),
                meta["cluster_id"].strip(),
            )
        )
    if oracle_meta and not output:
        raise SimplifyError("Ligand RMSD domain exists but has no usable records")
    return output


def group_ligands(
    records: Sequence[LigandRecord],
) -> dict[tuple[str, str], list[LigandRecord]]:
    output: dict[tuple[str, str], list[LigandRecord]] = defaultdict(list)
    for row in records:
        output[(row.entry_id, row.chain_id)].append(row)
    return output


def selected_ligands(records: Sequence[LigandRecord]) -> list[tuple[str, LigandRecord]]:
    by_ligand = group_ligands(records)
    predicted_key: dict[str, tuple[int, int]] = {}
    for entry in {row.entry_id for row in records}:
        candidates = {
            (row.seed, row.sample): row.ranking_score
            for row in records
            if row.entry_id == entry
        }
        predicted_key[entry] = max(
            candidates, key=lambda key: (candidates[key], -key[0], -key[1])
        )
    output = []
    for (entry, _), rows in sorted(by_ligand.items()):
        oracle = min(
            rows,
            key=lambda row: (
                row.lig_rmsd,
                not row.pb_valid,
                row.lddt_pli is None,
                -(row.lddt_pli or 0.0),
                -row.ranking_score,
                row.seed,
                row.sample,
            ),
        )
        seed_sample = predicted_key[entry]
        predicted = next(
            (row for row in rows if (row.seed, row.sample) == seed_sample),
            None,
        )
        if predicted is None:
            raise SimplifyError(
                "Predicted-best sample is missing ligand metrics for "
                f"{entry}:{rows[0].chain_id}; seed={seed_sample[0]}, "
                f"sample={seed_sample[1]}"
            )
        output.extend(((ORACLE, oracle), (PREDICTED_BEST, predicted)))
    return output


LIG_INST_COLUMNS = (
    "模型名称",
    "评估数据集",
    "数据子集",
    "PDB编号",
    "配体链",
    "配体实体",
    "cluster编号",
    "选择方式",
    "seed",
    "sample",
    "定位键",
    "选择依据分数",
    "Ligand RMSD（Å）",
    "Pocket RMSD（Å）",
    "LDDT-PLI",
    "是否RMSD≤2Å",
    "是否PB-valid",
    "是否RMSD≤2Å且PB-valid",
    "是否LDDT-PLI≥0.5（参考）",
    "PoseBusters未通过项",
)
LIG_SUM_COLUMNS = (
    "配体实例数量",
    "独立cluster数量",
    "PDB数量",
    "选择方式",
    "聚合方式",
    "平均Ligand RMSD（Å）",
    "中位数Ligand RMSD（Å）",
    "Ligand RMSD范围（Å）",
    "平均Pocket RMSD（Å）",
    "中位数Pocket RMSD（Å）",
    "Pocket RMSD范围（Å）",
    "平均LDDT-PLI",
    "RMSD≤2Å成功率（%）",
    "PB-valid率（%）",
    "RMSD≤2Å且PB-valid率（%）",
    "LDDT-PLI≥0.5参考率（%）",
)


def ligand_tables(
    records: Sequence[LigandRecord],
) -> dict[str, tuple[Sequence[str], Sequence[Mapping[str, Any]]]]:
    selected = selected_ligands(records)
    inst = []
    for method, row in selected:
        inst.append(
            {
                "模型名称": row.name,
                "评估数据集": row.eval_dataset,
                "数据子集": row.subset,
                "PDB编号": row.entry_id,
                "配体链": row.chain_id,
                "配体实体": row.entity_id,
                "cluster编号": row.cluster_id,
                "选择方式": method,
                "seed": row.seed,
                "sample": row.sample,
                "定位键": row.locator,
                "选择依据分数": fmt(
                    row.lig_rmsd if method == ORACLE else row.ranking_score
                ),
                "Ligand RMSD（Å）": fmt(row.lig_rmsd),
                "Pocket RMSD（Å）": fmt(row.pocket_rmsd),
                "LDDT-PLI": fmt(row.lddt_pli),
                "是否RMSD≤2Å": yes_no(row.lig_rmsd <= 2),
                "是否PB-valid": yes_no(row.pb_valid),
                "是否RMSD≤2Å且PB-valid": yes_no(row.lig_rmsd <= 2 and row.pb_valid),
                "是否LDDT-PLI≥0.5（参考）": (
                    yes_no(row.lddt_pli >= 0.5)
                    if row.lddt_pli is not None
                    else ""
                ),
                "PoseBusters未通过项": "|".join(row.pb_failures),
            }
        )
    inst.sort(
        key=lambda row: (row["PDB编号"], row["配体链"], METHOD_ORDER[row["选择方式"]])
    )
    sums = []
    for method in METHODS:
        rows = [row for selected_method, row in selected if selected_method == method]
        lig = [row.lig_rmsd for row in rows]
        pocket = [row.pocket_rmsd for row in rows]
        cluster_groups: dict[str, list[LigandRecord]] = defaultdict(list)
        for row in rows:
            cluster_groups[row.cluster_id].append(row)
        lddt_groups = [
            values
            for group in cluster_groups.values()
            if (
                values := [
                    item.lddt_pli for item in group if item.lddt_pli is not None
                ]
            )
        ]
        lddt_cluster_means = [mean(values) for values in lddt_groups]
        lddt_cluster_rates = [
            mean(value >= 0.5 for value in values) for values in lddt_groups
        ]

        sums.append(
            {
                "配体实例数量": len(rows),
                "独立cluster数量": len(cluster_groups),
                "PDB数量": len({row.entry_id for row in rows}),
                "选择方式": method,
                "聚合方式": "cluster等权",
                "平均Ligand RMSD（Å）": fmt(
                    mean(
                        mean(item.lig_rmsd for item in group)
                        for group in cluster_groups.values()
                    )
                ),
                "中位数Ligand RMSD（Å）": fmt(median(lig)),
                "Ligand RMSD范围（Å）": value_range(lig),
                "平均Pocket RMSD（Å）": fmt(
                    mean(
                        mean(item.pocket_rmsd for item in group)
                        for group in cluster_groups.values()
                    )
                ),
                "中位数Pocket RMSD（Å）": fmt(median(pocket)),
                "Pocket RMSD范围（Å）": value_range(pocket),
                "平均LDDT-PLI": fmt(
                    mean(lddt_cluster_means) if lddt_cluster_means else None
                ),
                "RMSD≤2Å成功率（%）": pct(
                    grouped_rate(
                        cluster_groups.values(), lambda item: item.lig_rmsd <= 2
                    )
                ),
                "PB-valid率（%）": pct(
                    grouped_rate(cluster_groups.values(), lambda item: item.pb_valid)
                ),
                "RMSD≤2Å且PB-valid率（%）": pct(
                    grouped_rate(
                        cluster_groups.values(),
                        lambda item: item.lig_rmsd <= 2 and item.pb_valid,
                    )
                ),
                "LDDT-PLI≥0.5参考率（%）": (
                    pct(mean(lddt_cluster_rates))
                    if lddt_cluster_rates
                    else ""
                ),
            }
        )
    return {
        "RMSD_ligand_sum.csv": (LIG_SUM_COLUMNS, sums),
        "RMSD_ligand_inst.csv": (LIG_INST_COLUMNS, inst),
    }


LIG_PDB_INST_COLUMNS = (
    "PDB编号",
    "配体数量",
    "选择方式",
    "seed",
    "sample",
    "定位键",
    "选择依据分数",
    "平均Ligand RMSD（Å）",
    "中位数Ligand RMSD（Å）",
    "最大Ligand RMSD（Å）",
    "RMSD≤2Å配体率（%）",
    "PB-valid配体率（%）",
    "RMSD≤2Å且PB-valid配体率（%）",
    "PDB是否所有配体RMSD≤2Å",
    "PDB是否所有配体PB-valid",
    "PDB是否所有配体RMSD≤2Å且PB-valid",
)
LIG_PDB_SUM_COLUMNS = (
    "PDB数量",
    "配体总数",
    "选择方式",
    "PDB等权平均Ligand RMSD（Å）",
    "PDB中位数Ligand RMSD（Å）",
    "PDB平均RMSD范围（Å）",
    "所有配体RMSD≤2Å的PDB率（%）",
    "所有配体PB-valid的PDB率（%）",
    "所有配体RMSD≤2Å且PB-valid的PDB率（%）",
)


def ligand_pdb_tables(
    records: Sequence[LigandRecord],
) -> dict[str, tuple[Sequence[str], Sequence[Mapping[str, Any]]]]:
    by_entry_sample: dict[tuple[str, int, int], list[LigandRecord]] = defaultdict(list)
    for row in records:
        by_entry_sample[(row.entry_id, row.seed, row.sample)].append(row)
    selected: dict[tuple[str, str], list[LigandRecord]] = {}
    for entry in sorted({row.entry_id for row in records}):
        candidates = [
            rows
            for (candidate_entry, _, _), rows in by_entry_sample.items()
            if candidate_entry == entry
        ]
        expected = {row.chain_id for row in candidates[0]}
        if any({row.chain_id for row in rows} != expected for rows in candidates):
            raise SimplifyError(f"Inconsistent ligand set across samples for {entry}")

        def oracle_key(rows: Sequence[LigandRecord]) -> tuple[Any, ...]:
            lddt_values = [
                row.lddt_pli for row in rows if row.lddt_pli is not None
            ]
            return (
                mean(row.lig_rmsd for row in rows),
                -mean(row.pb_valid for row in rows),
                not lddt_values,
                -mean(lddt_values) if lddt_values else 0.0,
                -rows[0].ranking_score,
                rows[0].seed,
                rows[0].sample,
            )

        oracle = min(
            candidates,
            key=oracle_key,
        )
        predicted = max(
            candidates,
            key=lambda rows: (rows[0].ranking_score, -rows[0].seed, -rows[0].sample),
        )
        selected[(entry, ORACLE)], selected[(entry, PREDICTED_BEST)] = oracle, predicted
    inst = []
    for entry in sorted({row.entry_id for row in records}):
        for method in METHODS:
            rows = selected[(entry, method)]
            first = rows[0]
            values = [row.lig_rmsd for row in rows]
            all_rmsd = all(value <= 2 for value in values)
            all_pb = all(row.pb_valid for row in rows)
            inst.append(
                {
                    "PDB编号": entry,
                    "配体数量": len(rows),
                    "选择方式": method,
                    "seed": first.seed,
                    "sample": first.sample,
                    "定位键": first.locator,
                    "选择依据分数": fmt(
                        mean(values) if method == ORACLE else first.ranking_score
                    ),
                    "平均Ligand RMSD（Å）": fmt(mean(values)),
                    "中位数Ligand RMSD（Å）": fmt(median(values)),
                    "最大Ligand RMSD（Å）": fmt(max(values)),
                    "RMSD≤2Å配体率（%）": pct(mean(value <= 2 for value in values)),
                    "PB-valid配体率（%）": pct(mean(row.pb_valid for row in rows)),
                    "RMSD≤2Å且PB-valid配体率（%）": pct(
                        mean(row.lig_rmsd <= 2 and row.pb_valid for row in rows)
                    ),
                    "PDB是否所有配体RMSD≤2Å": yes_no(all_rmsd),
                    "PDB是否所有配体PB-valid": yes_no(all_pb),
                    "PDB是否所有配体RMSD≤2Å且PB-valid": yes_no(all_rmsd and all_pb),
                }
            )
    sums = []
    for method in METHODS:
        rows = [row for row in inst if row["选择方式"] == method]
        averages = [float(row["平均Ligand RMSD（Å）"]) for row in rows]
        sums.append(
            {
                "PDB数量": len(rows),
                "配体总数": sum(int(row["配体数量"]) for row in rows),
                "选择方式": method,
                "PDB等权平均Ligand RMSD（Å）": fmt(mean(averages)),
                "PDB中位数Ligand RMSD（Å）": fmt(median(averages)),
                "PDB平均RMSD范围（Å）": value_range(averages),
                "所有配体RMSD≤2Å的PDB率（%）": pct(
                    mean(row["PDB是否所有配体RMSD≤2Å"] == "是" for row in rows)
                ),
                "所有配体PB-valid的PDB率（%）": pct(
                    mean(row["PDB是否所有配体PB-valid"] == "是" for row in rows)
                ),
                "所有配体RMSD≤2Å且PB-valid的PDB率（%）": pct(
                    mean(
                        row["PDB是否所有配体RMSD≤2Å且PB-valid"] == "是" for row in rows
                    )
                ),
            }
        )
    return {
        "RMSD_ligand_pdb_sum.csv": (LIG_PDB_SUM_COLUMNS, sums),
        "RMSD_ligand_pdb_inst.csv": (LIG_PDB_INST_COLUMNS, inst),
    }


def load_cdr(
    input_dir: Path, samples: Mapping[str, Sequence[RawSample]]
) -> list[CdrRecord]:
    path = input_dir / "antibody" / "cdr_metrics.csv"
    if not path.is_file():
        return []
    required = {"entry_id", "seed", "sample", "chain_id", "antibody_role", *CDR_METRICS}
    rows = read_csv(path, required)
    ranking = {
        (row.entry_id, row.seed, row.sample): row.ranking_score
        for values in samples.values()
        for row in values
    }
    output = []
    seen = set()
    for row in rows:
        entry, chain, seed, sample = (
            row["entry_id"].strip(),
            row["chain_id"].strip(),
            int(row["seed"]),
            int(row["sample"]),
        )
        for metric in CDR_METRICS:
            if not row[metric].strip():
                continue
            identity = (entry, chain, metric, seed, sample)
            if identity in seen:
                raise SimplifyError(f"Duplicate CDR record: {identity}")
            seen.add(identity)
            if (entry, seed, sample) not in ranking:
                raise SimplifyError(
                    f"Missing CDR ranking score: {(entry, seed, sample)}"
                )
            output.append(
                CdrRecord(
                    entry,
                    chain,
                    row["antibody_role"].strip(),
                    metric,
                    seed,
                    sample,
                    ranking[(entry, seed, sample)],
                    nonnegative(row[metric], metric),
                )
            )
    return output


def selected_cdr(records: Sequence[CdrRecord]) -> list[tuple[str, CdrRecord]]:
    groups: dict[tuple[str, str, str], list[CdrRecord]] = defaultdict(list)
    for row in records:
        groups[(row.entry_id, row.chain_id, row.metric)].append(row)
    predicted_keys = {}
    for entry in {row.entry_id for row in records}:
        candidates = {
            (row.seed, row.sample): row.ranking_score
            for row in records
            if row.entry_id == entry
        }
        predicted_keys[entry] = max(
            candidates, key=lambda key: (candidates[key], -key[0], -key[1])
        )
    output = []
    for (entry, chain, metric), rows in sorted(groups.items()):
        oracle = min(
            rows, key=lambda row: (row.rmsd, -row.ranking_score, row.seed, row.sample)
        )
        predicted = next(
            (
                row
                for row in rows
                if (row.seed, row.sample) == predicted_keys[entry]
            ),
            None,
        )
        if predicted is None:
            seed, sample = predicted_keys[entry]
            raise SimplifyError(
                "Predicted-best sample is missing CDR metrics for "
                f"{entry}:{chain}:{CDR_LABELS[metric]}; seed={seed}, sample={sample}"
            )
        output.extend(((ORACLE, oracle), (PREDICTED_BEST, predicted)))
    return output


CDR_INST_COLUMNS = (
    "PDB编号",
    "抗体链",
    "抗体链类型",
    "CDR类型",
    "选择方式",
    "seed",
    "sample",
    "定位键",
    "选择依据分数",
    "CDR主链RMSD（Å）",
    "是否RMSD≤1Å（参考）",
    "是否RMSD≤2Å（参考）",
)
CDR_SUM_COLUMNS = (
    "CDR类型",
    "实例数量",
    "PDB数量",
    "选择方式",
    "平均RMSD（Å）",
    "中位数RMSD（Å）",
    "RMSD标准差（Å）",
    "RMSD范围（Å）",
    "RMSD≤1Å参考率（%）",
    "RMSD≤2Å参考率（%）",
)


def cdr_tables(
    records: Sequence[CdrRecord],
) -> dict[str, tuple[Sequence[str], Sequence[Mapping[str, Any]]]]:
    selected = selected_cdr(records)
    inst = []
    for method, row in selected:
        role = AB_ROLE_LABELS.get(row.antibody_role, (row.antibody_role, ""))[0]
        inst.append(
            {
                "PDB编号": row.entry_id,
                "抗体链": row.chain_id,
                "抗体链类型": role,
                "CDR类型": CDR_LABELS[row.metric],
                "选择方式": method,
                "seed": row.seed,
                "sample": row.sample,
                "定位键": row.locator,
                "选择依据分数": fmt(
                    row.rmsd if method == ORACLE else row.ranking_score
                ),
                "CDR主链RMSD（Å）": fmt(row.rmsd),
                "是否RMSD≤1Å（参考）": yes_no(row.rmsd <= 1),
                "是否RMSD≤2Å（参考）": yes_no(row.rmsd <= 2),
            }
        )
    inst.sort(
        key=lambda row: (
            row["CDR类型"],
            row["PDB编号"],
            row["抗体链"],
            METHOD_ORDER[row["选择方式"]],
        )
    )
    sums = []
    for metric in CDR_METRICS:
        for method in METHODS:
            rows = [
                row
                for selected_method, row in selected
                if selected_method == method and row.metric == metric
            ]
            if not rows:
                continue
            values = [row.rmsd for row in rows]
            sums.append(
                {
                    "CDR类型": CDR_LABELS[metric],
                    "实例数量": len(rows),
                    "PDB数量": len({row.entry_id for row in rows}),
                    "选择方式": method,
                    "平均RMSD（Å）": fmt(mean(values)),
                    "中位数RMSD（Å）": fmt(median(values)),
                    "RMSD标准差（Å）": fmt(pstdev(values)),
                    "RMSD范围（Å）": value_range(values),
                    "RMSD≤1Å参考率（%）": pct(mean(value <= 1 for value in values)),
                    "RMSD≤2Å参考率（%）": pct(mean(value <= 2 for value in values)),
                }
            )
    return {
        "RMSD_cdr_sum.csv": (CDR_SUM_COLUMNS, sums),
        "RMSD_cdr_inst.csv": (CDR_INST_COLUMNS, inst),
    }


CDR_PDB_INST_COLUMNS = (
    "PDB编号",
    "抗体链数量",
    "CDR数量",
    "选择方式",
    "seed",
    "sample",
    "定位键",
    "选择依据分数",
    "平均CDR RMSD（Å）",
    "中位数CDR RMSD（Å）",
    "最大CDR RMSD（Å）",
    "CDR RMSD≤1Å参考率（%）",
    "CDR RMSD≤2Å参考率（%）",
    "PDB是否所有CDR RMSD≤1Å（参考）",
    "PDB是否所有CDR RMSD≤2Å（参考）",
)
CDR_PDB_SUM_COLUMNS = (
    "PDB数量",
    "CDR总数",
    "选择方式",
    "PDB等权平均CDR RMSD（Å）",
    "PDB中位数CDR RMSD（Å）",
    "PDB平均CDR RMSD范围（Å）",
    "所有CDR RMSD≤1Å的PDB参考率（%）",
    "所有CDR RMSD≤2Å的PDB参考率（%）",
)


def cdr_pdb_tables(
    records: Sequence[CdrRecord],
) -> dict[str, tuple[Sequence[str], Sequence[Mapping[str, Any]]]]:
    groups: dict[tuple[str, int, int], list[CdrRecord]] = defaultdict(list)
    for row in records:
        groups[(row.entry_id, row.seed, row.sample)].append(row)
    selected = {}
    for entry in sorted({row.entry_id for row in records}):
        candidates = [
            rows for (candidate, _, _), rows in groups.items() if candidate == entry
        ]
        expected = {(row.chain_id, row.metric) for row in candidates[0]}
        if any(
            {(row.chain_id, row.metric) for row in rows} != expected
            for rows in candidates
        ):
            raise SimplifyError(f"Inconsistent CDR set across samples for {entry}")
        selected[(entry, ORACLE)] = min(
            candidates,
            key=lambda rows: (
                mean(row.rmsd for row in rows),
                max(row.rmsd for row in rows),
                -rows[0].ranking_score,
                rows[0].seed,
                rows[0].sample,
            ),
        )
        selected[(entry, PREDICTED_BEST)] = max(
            candidates,
            key=lambda rows: (rows[0].ranking_score, -rows[0].seed, -rows[0].sample),
        )
    inst = []
    for entry in sorted({row.entry_id for row in records}):
        for method in METHODS:
            rows = selected[(entry, method)]
            first = rows[0]
            values = [row.rmsd for row in rows]
            inst.append(
                {
                    "PDB编号": entry,
                    "抗体链数量": len({row.chain_id for row in rows}),
                    "CDR数量": len(rows),
                    "选择方式": method,
                    "seed": first.seed,
                    "sample": first.sample,
                    "定位键": first.locator,
                    "选择依据分数": fmt(
                        mean(values) if method == ORACLE else first.ranking_score
                    ),
                    "平均CDR RMSD（Å）": fmt(mean(values)),
                    "中位数CDR RMSD（Å）": fmt(median(values)),
                    "最大CDR RMSD（Å）": fmt(max(values)),
                    "CDR RMSD≤1Å参考率（%）": pct(mean(value <= 1 for value in values)),
                    "CDR RMSD≤2Å参考率（%）": pct(mean(value <= 2 for value in values)),
                    "PDB是否所有CDR RMSD≤1Å（参考）": yes_no(max(values) <= 1),
                    "PDB是否所有CDR RMSD≤2Å（参考）": yes_no(max(values) <= 2),
                }
            )
    sums = []
    for method in METHODS:
        rows = [row for row in inst if row["选择方式"] == method]
        averages = [float(row["平均CDR RMSD（Å）"]) for row in rows]
        sums.append(
            {
                "PDB数量": len(rows),
                "CDR总数": sum(int(row["CDR数量"]) for row in rows),
                "选择方式": method,
                "PDB等权平均CDR RMSD（Å）": fmt(mean(averages)),
                "PDB中位数CDR RMSD（Å）": fmt(median(averages)),
                "PDB平均CDR RMSD范围（Å）": value_range(averages),
                "所有CDR RMSD≤1Å的PDB参考率（%）": pct(
                    mean(row["PDB是否所有CDR RMSD≤1Å（参考）"] == "是" for row in rows)
                ),
                "所有CDR RMSD≤2Å的PDB参考率（%）": pct(
                    mean(row["PDB是否所有CDR RMSD≤2Å（参考）"] == "是" for row in rows)
                ),
            }
        )
    return {
        "RMSD_cdr_pdb_sum.csv": (CDR_PDB_SUM_COLUMNS, sums),
        "RMSD_cdr_pdb_inst.csv": (CDR_PDB_INST_COLUMNS, inst),
    }


def build_optional_metric_tables(
    input_dir: Path,
) -> dict[str, tuple[Sequence[str], Sequence[Mapping[str, Any]]]]:
    tables: dict[str, tuple[Sequence[str], Sequence[Mapping[str, Any]]]] = {}
    lddt_details = input_dir / "summary" / "LDDT_details.csv"
    abag_lddt_details = input_dir / "antibody" / "summary" / "LDDT_details.csv"
    annotations = input_dir / "antibody" / "annotations.csv"
    ligand_source = input_dir / "antibody" / "per_sample_metrics.csv"
    rmsd_details = input_dir / "summary" / "RMSD_details.csv"
    cdr_source = input_dir / "antibody" / "cdr_metrics.csv"
    if not any(
        path.is_file()
        for path in (lddt_details, abag_lddt_details, ligand_source, cdr_source)
    ):
        print(
            "警告：未发现 LDDT/RMSD/CDR 可选评估数据，仅生成 DockQ。", file=sys.stderr
        )
        return tables

    raw_samples = load_raw_samples(input_dir / "per_sample")

    if lddt_details.is_file():
        oracle_defs = load_lddt_definitions(lddt_details, "best")
        predicted_defs = load_lddt_definitions(lddt_details, "best.ranking_score")
        selected = select_lddt(oracle_defs, predicted_defs, raw_samples)
        tables.update(lddt_structure_tables(selected))
        tables.update(lddt_pdb_tables(raw_samples))
    else:
        print(f"警告：跳过 LDDT，缺少 {lddt_details}", file=sys.stderr)

    if abag_lddt_details.is_file() and annotations.is_file():
        oracle_defs = load_lddt_definitions(abag_lddt_details, "best")
        predicted_defs = load_lddt_definitions(abag_lddt_details, "best.ranking_score")
        selected = select_lddt(oracle_defs, predicted_defs, raw_samples)
        tables.update(lddt_abag_tables(selected, load_abag_roles(annotations)))
    else:
        print("警告：跳过 AbAg LDDT，缺少抗体 LDDT 明细或注释。", file=sys.stderr)

    ligand_records = (
        load_ligands(input_dir, raw_samples)
        if ligand_source.is_file() and rmsd_details.is_file()
        else []
    )
    if ligand_records:
        tables.update(ligand_tables(ligand_records))
        tables.update(ligand_pdb_tables(ligand_records))
    else:
        print("警告：跳过配体 RMSD，数据集无可用配体评估。", file=sys.stderr)

    cdr_records = load_cdr(input_dir, raw_samples) if cdr_source.is_file() else []
    if cdr_records:
        tables.update(cdr_tables(cdr_records))
        tables.update(cdr_pdb_tables(cdr_records))
    else:
        print("警告：跳过 CDR RMSD，数据集无可用 CDR 评估。", file=sys.stderr)
    return tables


def build_reports(
    input_dir: Path, pred_dir: Path, ref_dir: Path, output_dir: Path
) -> dict[str, int]:
    input_dir = input_dir.expanduser().resolve()
    pred_dir = pred_dir.expanduser().resolve()
    ref_dir = ref_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise SimplifyError(f"Input directory does not exist: {input_dir}")
    if not pred_dir.is_dir():
        raise SimplifyError(f"Prediction directory does not exist: {pred_dir}")
    if not ref_dir.is_dir():
        raise SimplifyError(f"Reference directory does not exist: {ref_dir}")

    main_details = input_dir / "summary" / "DockQ_details.csv"
    antibody_details = input_dir / "antibody" / "summary" / "DockQ_details.csv"
    prot_oracle_definitions = load_interface_definitions(
        main_details, protein_only=True, ranker="best"
    )
    prot_predicted_definitions = load_interface_definitions(
        main_details, protein_only=True, ranker="best.ranking_score"
    )
    abag_oracle_definitions = load_interface_definitions(
        antibody_details,
        protein_only=False,
        ranker="best",
    )
    abag_predicted_definitions = load_interface_definitions(
        antibody_details,
        protein_only=False,
        ranker="best.ranking_score",
    )
    abag_subset = load_abag_subset(input_dir / "antibody" / "subset.csv")
    abag_roles = load_abag_roles(input_dir / "antibody" / "annotations.csv")

    needed_pairs: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for definition in (
        *prot_oracle_definitions.values(),
        *prot_predicted_definitions.values(),
        *abag_oracle_definitions.values(),
        *abag_predicted_definitions.values(),
    ):
        needed_pairs[definition.entry_id].add(definition.pair)
    candidates = load_sample_candidates(input_dir / "per_sample", needed_pairs)

    prot_selected = select_interfaces(
        prot_oracle_definitions, prot_predicted_definitions, candidates
    )
    abag_selected = select_interfaces(
        abag_oracle_definitions, abag_predicted_definitions, candidates
    )
    prot_inst = [selected_to_prot_row(row) for row in prot_selected]
    prot_inst.sort(
        key=lambda row: (
            row["PDB编号"],
            row["链1"],
            row["链2"],
            METHOD_ORDER[str(row["选择方式"])],
        )
    )
    prot_sum = build_prot_sum_rows(prot_selected)
    pdb_inst = build_pdb_inst_rows(prot_predicted_definitions, candidates)
    pdb_inst.sort(key=lambda row: (row["PDB编号"], METHOD_ORDER[str(row["选择方式"])]))
    pdb_sum = build_pdb_sum_rows(pdb_inst)
    pdb_exports = build_pdb_structure_exports(pred_dir, ref_dir, pdb_inst)
    abag_inst, abag_groups = build_abag_rows(abag_selected, abag_subset, abag_roles)
    abag_inst.sort(
        key=lambda row: (
            row["PDB编号"],
            row["抗原-抗体界面分组"],
            row["抗体链"],
            row["抗原链"],
            METHOD_ORDER[str(row["选择方式"])],
        )
    )
    abag_sum = build_abag_sum_rows(abag_selected, abag_groups)

    tables = {
        "DockQ_prot_interface_sum.csv": (PROT_SUM_COLUMNS, prot_sum),
        "DockQ_prot_interface_inst.csv": (PROT_INST_COLUMNS, prot_inst),
        "DockQ_pdb_sum.csv": (PDB_SUM_COLUMNS, pdb_sum),
        "DockQ_pdb_inst.csv": (PDB_INST_COLUMNS, pdb_inst),
        "DockQ_abag_interface_sum.csv": (ABAG_SUM_COLUMNS, abag_sum),
        "DockQ_abag_interface_inst.csv": (ABAG_INST_COLUMNS, abag_inst),
    }
    tables.update(build_optional_metric_tables(input_dir))
    publish_csv_bundle(output_dir, tables, pdb_exports)
    counts = {filename: len(rows) for filename, (_, rows) in tables.items()}
    counts[PDB_EXPORT_COUNT_KEY] = len(pdb_exports)
    return counts


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "生成只包含 Oracle 与预测排序最佳的 DockQ/LDDT/RMSD "
            "精简中文汇总和 PDB 结构。"
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"PXMeter 评估根目录（默认：{DEFAULT_INPUT_DIR}）。",
    )
    parser.add_argument(
        "--pred-dir",
        required=True,
        type=Path,
        help="batch_infer_indices.py 生成的 Protenix 原始预测根目录。",
    )
    parser.add_argument(
        "--ref-dir",
        required=True,
        type=Path,
        help="真实参考结构目录，支持 <PDB>.cif 和 <PDB>.cif.gz。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录（默认：<input-dir>/summary_simplify）。",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir or args.input_dir / "summary_simplify"
    try:
        counts = build_reports(args.input_dir, args.pred_dir, args.ref_dir, output_dir)
    except SimplifyError as exc:
        print(f"精简汇总失败：{exc}", file=sys.stderr)
        return 2
    print(f"精简汇总已生成：{output_dir.expanduser().resolve()}")
    for filename in MANAGED_OUTPUT_FILENAMES:
        if filename in counts:
            print(f"  {filename}: {counts[filename]} 行")
    print(f"  {PDB_OUTPUT_DIRNAME}/: {counts[PDB_EXPORT_COUNT_KEY]} 个 CIF")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
