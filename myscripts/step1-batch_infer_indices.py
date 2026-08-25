#!/usr/bin/env python3
# Copyright 2025 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""对一个索引 CSV 批量运行 Protenix v1/v2，并输出 PXMeter 兼容目录。

使用示例
--------
单卡::

    python myscripts/step1-batch_infer_indices.py \
      --indices-csv /data/protenix_data_sabdab2/indices/val.csv \
      --data-root /data/protenix_data_sabdab2 \
      --model-name protenix_base_default_v1.0.0 \
      --checkpoint /data/my_runs/protenix_finetune_sabdab2.pt \
      --output-dir /data/my_runs/base_v1_val

八卡::

    torchrun --standalone --nproc_per_node=8 \
      myscripts/step1-batch_infer_indices.py \
      --indices-csv /data/protenix_data_sabdab2/indices/test.csv \
      --data-root /data/protenix_data_sabdab2 \
      --model-name protenix-v2 \
      --checkpoint /data/protenix_models/protenix-v2.pt \
      --output-dir /data/my_runs/protenix_v2_test

``--output-dir`` 指向的目录本身就是 PXMeter 在 ``-m protenix`` 模式下的
``infer_dir``。默认使用 5 个 seed（1、2、3、4、5），每个 seed 生成 5 个
sample，因此每个 case 共生成 25 个候选结构。默认使用 4 次 Pairformer recycle
和 100 个扩散去噪步骤。可用 ``--limit 1 --cycles 1 --steps 20 --samples 1
--seeds 1`` 做快速冒烟测试。

``--checkpoint`` 可指向任意文件名的官方或微调权重；模型架构由
``--model-name`` 明确选择。Protenix-v2 会自动跳过超过 2560 tokens 的条目。

执行流程：先校验数据、权重与输出目录的运行身份，再按
``num_tokens²`` 将 PDB 均衡分配给各 GPU rank；每个 rank 按 seed
独立推理并原子更新进度，最后由 rank 0 汇总全局状态。
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import hashlib
import json
import logging
import os
import shutil
import sys
import threading
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence


# 直接运行 ``python myscripts/step1-batch_infer_indices.py`` 时，Python 默认只把
# myscripts/ 加入模块搜索路径。显式加入仓库根目录，确保 torchrun 的每个子进程
# 都能导入同级的 protenix、configs 和 runner 包，而不依赖外部 PYTHONPATH。
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL_NAME = "protenix_base_default_v1.0.0"
SUPPORTED_MODEL_NAMES = (DEFAULT_MODEL_NAME, "protenix-v2")
MODEL_TOKEN_LIMITS = {"protenix-v2": 2560}
REQUIRED_CONFIDENCE_KEYS = {"ranking_score", "plddt", "ptm", "iptm"}
LOGGER = logging.getLogger("batch_infer_indices")
PROGRESS_DIR_NAME = ".batch_progress"
PROGRESS_POLL_SECONDS = 0.5
PROGRESS_HEARTBEAT_SECONDS = 60.0
FINAL_STATUS_TIMEOUT_SECONDS = 600.0
STAT_KEYS = ("assigned", "succeeded", "skipped", "failed")
ASSIGNMENT_STRATEGY = "greedy_num_tokens_squared"
OUTPUT_LOCK_NAME = ".batch_infer.lock"
RUN_IDENTITY_NAME = ".batch_run_identity.json"


class SafeFeatureDataset:
    """封装单样本特征生成，使 DataLoader worker 的异常可按样本返回。"""

    def __init__(self, dataset: Any, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[int, Any, str | None]:
        dataset_index = self.indices[item]
        try:
            data = self.dataset.process_one(
                dataset_index, return_atom_token_array=True
            )
            return dataset_index, data, None
        except Exception:
            # 不让单个坏样本终止整个 DataLoader；主进程负责记录错误并继续。
            return dataset_index, None, traceback.format_exc()


def first_item(batch: list[Any]) -> Any:
    """batch size 固定为 1，保持 Protenix 嵌套数据结构不变。"""

    return batch[0]


def make_prediction_input(input_feature_dict: dict[str, Any]) -> dict[str, Any]:
    """为一次模型调用创建独立的顶层特征字典。

    Protenix 的 inference forward 会为降低显存峰值而原地删除 MSA、模板等特征，
    ``to_device()`` 也会原地替换字典里的 tensor。这里只复制字典容器、不复制
    tensor，因此既能让多个 seed 复用原始 CPU 特征，也不会复制大型特征张量。
    """

    return {"input_feature_dict": input_feature_dict.copy()}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "读取一个 indices CSV，运行 Protenix v1/v2 推理，并输出 "
            "PXMeter 兼容的预测结果。"
        )
    )
    parser.add_argument(
        "--indices-csv", required=True, type=Path, help="单个 val.csv 或 test.csv。"
    )
    parser.add_argument(
        "--data-root", required=True, type=Path, help="Protenix 预处理数据根目录。"
    )
    parser.add_argument(
        "--model-name",
        choices=SUPPORTED_MODEL_NAMES,
        default=DEFAULT_MODEL_NAME,
        help=f"模型架构（默认：{DEFAULT_MODEL_NAME}）。",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="与 --model-name 架构匹配的官方或微调权重（文件名不限）。",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="本次推理的 PXMeter 输入目录。"
    )
    parser.add_argument(
        "--seeds",
        default="1,2,3,4,5",
        help="逗号分隔的整数随机种子（默认：1,2,3,4,5）。",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=4,
        help="Pairformer 循环次数（默认：4）。",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="扩散去噪步数（默认：100）。",
    )
    parser.add_argument("--samples", type=int, default=5, help="每个 seed 的候选数。")
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp32", "fp16"),
        default="bf16",
        help="模型推理精度。",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="CPU 特征加载 worker 数。"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=-1,
        help=(
            "跳过超过该 token 数的条目；-1 表示使用模型上限"
            "（v2 为 2560，v1 不过滤）。"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="只处理按 PDB 聚合后的前 N 个条目；-1 表示全部处理。",
    )
    parser.add_argument(
        "--use-msa", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--use-rna-msa",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启用 RNA MSA（默认启用；可用 --no-use-rna-msa 关闭）。",
    )
    parser.add_argument(
        "--use-template", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--kalign-binary",
        type=Path,
        default=None,
        help="Kalign 可执行文件；默认从 PATH 中查找。",
    )
    parser.add_argument(
        "--triangle-multiplicative",
        choices=("cuequivariance", "torch"),
        default="cuequivariance",
    )
    parser.add_argument(
        "--triangle-attention",
        choices=("cuequivariance", "torch", "triattention"),
        default="cuequivariance",
    )
    parser.add_argument(
        "--enable-cache", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--enable-fusion", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--enable-tf32", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="即使结果文件完整也重新计算。",
    )
    parser.add_argument(
        "--log-level",
        choices=("verbose", "quiet"),
        default="verbose",
        help=(
            "日志详细程度：verbose 输出脚本进度和详细状态（默认）；"
            "quiet 仅输出错误。"
        ),
    )
    return parser.parse_args(argv)


def configure_logging(log_level: str) -> None:
    """配置日志；脚本输出 INFO，但抑制 Protenix 内部的大量 INFO。"""

    # verbose 只放开本脚本的 INFO；第三方 logger 保持 WARNING，避免多 rank
    # 的 Protenix 内部日志冲坏全局进度条。quiet 则只保留错误。
    root_level = logging.WARNING if log_level == "verbose" else logging.ERROR
    log_format = "%(asctime)s %(levelname)s [%(process)d] %(message)s"
    formatter = logging.Formatter(log_format)
    logging.basicConfig(
        level=root_level,
        format=log_format,
        force=True,
    )
    for handler in logging.getLogger().handlers:
        handler.setLevel(root_level)

    # 脚本自身使用独立 handler，不受根 logger 的 WARNING 级别影响。
    LOGGER.handlers.clear()
    script_handler = logging.StreamHandler()
    script_handler.setFormatter(formatter)
    script_handler.setLevel(
        logging.ERROR if log_level == "quiet" else logging.INFO
    )
    LOGGER.addHandler(script_handler)
    LOGGER.setLevel(logging.ERROR if log_level == "quiet" else logging.INFO)
    LOGGER.propagate = False


def parse_seeds(raw: str) -> list[int]:
    """解析并校验逗号分隔的随机种子。"""

    try:
        seeds = [int(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError(f"--seeds must be comma-separated integers: {raw!r}") from exc
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must not contain duplicates")
    if any(seed < 0 or seed >= 2**32 for seed in seeds):
        raise ValueError("--seeds values must be in the range [0, 2**32)")
    return seeds


def deep_update_dict(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """递归合并模型配置，避免浅层 update 丢失未覆盖的嵌套默认值。"""

    for key, value in overrides.items():
        if (
            isinstance(value, Mapping)
            and key in base
            and isinstance(base[key], Mapping)
        ):
            nested = dict(base[key])
            base[key] = deep_update_dict(nested, value)
        else:
            base[key] = deepcopy(value)
    return base


def effective_max_tokens(model_name: str, requested: int) -> int:
    """应用模型硬上限，同时允许用户请求更严格的过滤。"""

    model_limit = MODEL_TOKEN_LIMITS.get(model_name)
    if model_limit is None:
        return requested
    if requested == -1 or requested > model_limit:
        return model_limit
    return requested


def filtered_pdb_ids_from_csv(indices_csv: Path, max_tokens: int) -> tuple[list[str], bool]:
    """从标准 indices CSV 统计会被 token 上限过滤的 PDB。"""

    if max_tokens == -1:
        return [], True
    with indices_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        pdb_column = next(
            (name for name in ("pdb_id", "entry_id") if name in fieldnames),
            None,
        )
        if pdb_column is None or "num_tokens" not in fieldnames:
            return [], False
        filtered = set()
        for row in reader:
            try:
                num_tokens = int(float(row["num_tokens"]))
            except (KeyError, TypeError, ValueError):
                continue
            pdb_id = str(row.get(pdb_column, "")).strip()
            if pdb_id and num_tokens > max_tokens:
                filtered.add(pdb_id)
    return sorted(filtered), True


def checkpoint_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """计算 checkpoint 内容指纹；仅由 rank 0 在持有输出锁时调用。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_checkpoint_state_dict(checkpoint: Any) -> dict[str, Any]:
    """校验 Protenix checkpoint，并移除可选的 DDP ``module.`` 前缀。"""

    if not isinstance(checkpoint, Mapping) or "model" not in checkpoint:
        raise ValueError("Checkpoint must be a mapping containing a 'model' state dict")
    state_dict = checkpoint["model"]
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("Checkpoint 'model' must be a non-empty state dict")
    normalized = dict(state_dict)
    if all(str(key).startswith("module.") for key in normalized):
        normalized = {
            str(key)[len("module.") :]: value
            for key, value in normalized.items()
        }
    return normalized


def load_state_dict_strict(
    model: Any,
    state_dict: Mapping[str, Any],
    checkpoint_path: Path,
    model_name: str,
) -> None:
    """严格加载权重，并将尺寸不匹配转换成可操作的模型选择错误。"""

    try:
        model.load_state_dict(state_dict=dict(state_dict), strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} is incompatible with model "
            f"{model_name!r}; verify that --model-name matches the checkpoint "
            "architecture."
        ) from exc


def require_readable_file(path: Path, label: str) -> Path:
    """返回规范化绝对路径，并确认文件存在且可读。"""

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"{label} is not readable: {path}")
    return path


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    """在加载大型依赖和模型之前完成参数及数据目录检查。"""

    args.indices_csv = require_readable_file(args.indices_csv, "indices CSV")
    args.checkpoint = require_readable_file(args.checkpoint, "checkpoint")
    args.data_root = args.data_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    if not args.data_root.is_dir():
        raise NotADirectoryError(f"Missing data root: {args.data_root}")
    if args.indices_csv.suffix.lower() != ".csv":
        raise ValueError(f"--indices-csv must point to a .csv file: {args.indices_csv}")

    for name, value in (
        ("cycles", args.cycles),
        ("steps", args.steps),
        ("samples", args.samples),
    ):
        if value is not None and value <= 0:
            raise ValueError(f"--{name} must be a positive integer, got {value}")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.max_tokens == 0 or args.max_tokens < -1:
        raise ValueError("--max-tokens must be -1 or a positive integer")
    if args.limit == 0 or args.limit < -1:
        raise ValueError("--limit must be -1 or a positive integer")

    args.requested_max_tokens = args.max_tokens
    args.max_tokens = effective_max_tokens(args.model_name, args.max_tokens)
    (
        args.filtered_over_token_limit,
        args.token_filter_audit_available,
    ) = filtered_pdb_ids_from_csv(args.indices_csv, args.max_tokens)

    # CCD、聚类文件和预处理后的 bioassembly 是任何模式都需要的基础数据。
    required_files = [
        "common/components.cif",
        "common/components.cif.rdkit_mol.pkl",
        "common/clusters-by-entity-40.txt",
    ]
    required_dirs = [
        "mmcif",
        "mmcif_bioassembly",
    ]
    # 蛋白模板和蛋白 MSA 通过相同的序列映射定位预计算目录。
    if args.use_msa or args.use_template:
        required_files.append("pdb_seqs/seq_to_pdb_index.json")
        required_dirs.append("mmcif_msa_template")
    if args.use_rna_msa:
        required_files.append("rna_msa/rna_sequence_to_pdb_chains.json")
        required_dirs.append("rna_msa/msas")
    if args.use_template:
        required_files.extend(
            [
                "common/release_date_cache.json",
                "common/obsolete_to_successor.json",
            ]
        )
    for relative in required_files:
        path = args.data_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing required data file: {path}")
    for relative in required_dirs:
        path = args.data_root / relative
        if not path.is_dir():
            raise NotADirectoryError(f"Missing required data directory: {path}")

    if args.use_template:
        if args.kalign_binary is None:
            resolved = shutil.which("kalign")
            if resolved is None:
                raise FileNotFoundError(
                    "Template inference requires Kalign. Install it or pass "
                    "--kalign-binary /path/to/kalign."
                )
            args.kalign_binary = Path(resolved).resolve()
        else:
            args.kalign_binary = args.kalign_binary.expanduser().resolve()
        if not args.kalign_binary.is_file() or not os.access(
            args.kalign_binary, os.X_OK
        ):
            raise PermissionError(
                f"Kalign is missing or not executable: {args.kalign_binary}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not os.access(args.output_dir, os.W_OK):
        raise PermissionError(f"Output directory is not writable: {args.output_dir}")
    args.seeds = parse_seeds(args.seeds)
    return args


def configure_environment(args: argparse.Namespace) -> None:
    """设置会在 Protenix 配置模块导入时读取的数据根目录。"""

    # 必须先设置环境变量，再导入 configs.*；否则路径会被固化成错误的默认值。
    os.environ["PROTENIX_ROOT_DIR"] = str(args.data_root)
    os.environ["PROTENIX_TEMPLATE_MMCIF_DIR"] = str(args.data_root / "mmcif")


def build_configs(args: argparse.Namespace) -> Any:
    """合并基础、模型和命令行配置，生成推理 runner 所需配置。"""

    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from configs.configs_model_type import model_configs
    from protenix.config.config import parse_configs
    from runner.inference import update_gpu_compatible_configs

    if args.model_name not in model_configs:
        raise ValueError(
            f"Installed Protenix does not support {args.model_name!r}. "
            "Install Protenix >= 2.0.0 or use a matching source checkout."
        )

    # 必须在 parse_configs 前深度合并。v2 会覆盖 c_z 等深层架构参数，浅层
    # ConfigDict.update 会把整个 model 子树替换掉并丢失其余默认配置。
    raw_configs = deepcopy(
        {
            **configs_base,
            **{"data": data_configs},
            **inference_configs,
        }
    )
    deep_update_dict(raw_configs, model_configs[args.model_name])
    configs = parse_configs(raw_configs, fill_required_with_null=True)
    configs.model_name = args.model_name
    configs.load_checkpoint_path = str(args.checkpoint)
    configs.load_checkpoint_dir = str(args.checkpoint.parent)
    configs.dump_dir = str(args.output_dir)
    configs.seeds = args.seeds
    if args.cycles is not None:
        configs.model.N_cycle = args.cycles
    if args.steps is not None:
        configs.sample_diffusion.N_step = args.steps
    configs.sample_diffusion.N_sample = args.samples
    configs.dtype = args.dtype
    configs.use_msa = args.use_msa
    configs.use_template = args.use_template
    configs.use_rna_msa = args.use_rna_msa
    configs.need_atom_confidence = False
    configs.sorted_by_ranking_score = True
    configs.triangle_multiplicative = args.triangle_multiplicative
    configs.triangle_attention = args.triangle_attention
    configs.enable_diffusion_shared_vars_cache = args.enable_cache
    configs.enable_efficient_fusion = args.enable_fusion
    configs.enable_tf32 = args.enable_tf32
    configs.data.num_dl_workers = args.num_workers
    if args.use_template:
        configs.data.template.kalign_binary_path = str(args.kalign_binary)
    configs.load_strict = True
    configs = update_gpu_compatible_configs(configs)
    args.effective_cycles = int(configs.model.N_cycle)
    args.effective_steps = int(configs.sample_diffusion.N_step)
    return configs


def build_dataset(args: argparse.Namespace, configs: Any) -> Any:
    """从单个 CSV 构建按 PDB 聚合、无裁剪的推理数据集。"""

    from ml_collections.config_dict import ConfigDict

    from protenix.data.pipeline.dataset import (
        BaseSingleDataset,
        get_msa_featurizer,
        get_template_featurizer,
    )

    # 复用 sabdab2_val 的 MSA/模板日期过滤策略，但替换 CSV 和数据路径。
    dataset_name = "batch_indices"
    dataset_config = deepcopy(configs.data.sabdab2_val.to_dict())
    base_info = dataset_config["base_info"]
    base_info.update(
        {
            "mmcif_dir": str(args.data_root / "mmcif"),
            "bioassembly_dict_dir": str(args.data_root / "mmcif_bioassembly"),
            "indices_fpath": str(args.indices_csv),
            "pdb_list": "",
            "limits": args.limit,
            "max_n_token": args.max_tokens,
            "sort_by_n_token": False,
            "group_by_pdb_id": True,
            "find_eval_chain_interface": False,
        }
    )
    dataset_config["cropping_configs"]["crop_size"] = -1
    dataset_config["msa"]["enable_prot_msa"] = args.use_msa
    dataset_config["msa"]["enable_rna_msa"] = args.use_rna_msa
    if args.use_rna_msa:
        dataset_config["msa"]["rna_seq_or_filename_to_msadir_jsons"] = [
            str(args.data_root / "rna_msa/rna_sequence_to_pdb_chains.json")
        ]
        dataset_config["msa"]["rna_msadir_raw_paths"] = [
            str(args.data_root / "rna_msa/msas")
        ]
        dataset_config["msa"]["rna_indexing_methods"] = ["sequence"]
    dataset_config["template"]["enable_prot_template"] = args.use_template
    if args.use_template:
        dataset_config["template"]["kalign_binary_path"] = str(args.kalign_binary)

    configs.data[dataset_name] = ConfigDict(dataset_config)

    # 关闭某项特征时不要创建 featurizer。相关构造函数即使 enable=False 也会
    # 读取映射/日期文件，既浪费时间，也会让 --no-use-* 模式错误依赖这些文件。
    msa_featurizer = (
        get_msa_featurizer(configs, dataset_name, stage="test")
        if args.use_msa or args.use_rna_msa
        else None
    )
    template_featurizer = (
        get_template_featurizer(configs, dataset_name, stage="test")
        if args.use_template
        else None
    )
    return BaseSingleDataset(
        name=dataset_name,
        **base_info,
        cropping_configs=dataset_config["cropping_configs"],
        msa_featurizer=msa_featurizer,
        template_featurizer=template_featurizer,
        ref_pos_augment=False,
        lig_atom_rename=False,
        shuffle_mols=False,
        shuffle_sym_ids=False,
        constraint={"enable": False},
        error_dir=None,
    )


def sample_name(dataset: Any, index: int) -> str:
    """取得按 PDB 聚合后的样本名称。"""

    return str(dataset._get_sample_indice(index)["pdb_id"])


def selected_pdb_ids(dataset: Any) -> list[str]:
    """返回过滤、聚合和 ``--limit`` 后本轮实际覆盖的 PDB。"""

    pdb_ids = [sample_name(dataset, index) for index in range(len(dataset))]
    if len(set(pdb_ids)) != len(pdb_ids):
        raise ValueError("Grouped inference dataset contains duplicate PDB IDs")
    return pdb_ids


def sample_num_tokens(dataset: Any, index: int) -> int:
    """取得用于静态负载估算的 PDB token 数。"""

    return max(1, int(dataset._get_sample_indice(index)["num_tokens"]))


def get_balanced_assignments(
    index_num_tokens: Sequence[tuple[int, int]], world_size: int
) -> tuple[list[list[int]], list[int]]:
    """按 ``num_tokens²`` 将任务确定性地贪心分配到各 rank。"""

    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")

    assignments = [[] for _ in range(world_size)]
    estimated_loads = [0 for _ in range(world_size)]
    weighted_indices = sorted(
        (
            (index, max(1, int(num_tokens)) ** 2)
            for index, num_tokens in index_num_tokens
        ),
        key=lambda item: (-item[1], item[0]),
    )
    for index, weight in weighted_indices:
        target_rank = min(
            range(world_size),
            key=lambda candidate: (
                estimated_loads[candidate],
                len(assignments[candidate]),
                candidate,
            ),
        )
        assignments[target_rank].append(index)
        estimated_loads[target_rank] += weight
    return assignments, estimated_loads


def plan_rank_work(
    dataset: Any,
    output_dir: Path,
    seeds: Sequence[int],
    samples: int,
    world_size: int,
    overwrite: bool = False,
) -> tuple[list[list[int]], list[list[int]], list[int]]:
    """识别完整结果，并只对待处理 PDB 做 token 感知的负载均衡。"""

    pending = []
    complete = []
    for index in range(len(dataset)):
        pdb_id = sample_name(dataset, index)
        if not overwrite and outputs_complete(
            output_dir, pdb_id, seeds, samples
        ):
            complete.append(index)
        else:
            pending.append((index, sample_num_tokens(dataset, index)))

    pending_by_rank, estimated_loads = get_balanced_assignments(
        pending, world_size
    )
    skipped_by_rank = [[] for _ in range(world_size)]
    for ordinal, index in enumerate(complete):
        skipped_by_rank[ordinal % world_size].append(index)
    return pending_by_rank, skipped_by_rank, estimated_loads


def predictions_dir(output_dir: Path, pdb_id: str, seed: int) -> Path:
    """返回 PXMeter Protenix evaluator 约定的 predictions 目录。"""

    return output_dir / pdb_id / f"seed_{seed}" / "predictions"


def expected_paths(
    output_dir: Path, pdb_id: str, seed: int, samples: int
) -> list[tuple[Path, Path]]:
    """列出一个 PDB/seed 应生成的 CIF 与置信度 JSON 文件对。"""

    directory = predictions_dir(output_dir, pdb_id, seed)
    return [
        (
            directory / f"{pdb_id}_sample_{sample}.cif",
            directory / f"{pdb_id}_summary_confidence_sample_{sample}.json",
        )
        for sample in range(samples)
    ]


def cif_output_valid(path: Path) -> bool:
    """确认 CIF 非空、可解析且包含至少一个原子记录。"""

    if not path.is_file():
        return False
    try:
        if path.stat().st_size == 0:
            return False
        from biotite.structure.io import pdbx

        cif_file = pdbx.CIFFile.read(path)
        block = cif_file.block
        if "atom_site" not in block:
            return False
        atom_site = block["atom_site"]
        if "Cartn_x" not in atom_site:
            return False
        return len(atom_site["Cartn_x"].as_array()) > 0
    except Exception:
        return False


def confidence_output_valid(path: Path) -> bool:
    """确认置信度 JSON 可读，且关键字段存在并具有有限数值。"""

    if not path.is_file():
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            confidence = json.load(handle)
        if not isinstance(confidence, dict):
            return False
        for key in REQUIRED_CONFIDENCE_KEYS:
            value = confidence.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return False
            if not float("-inf") < float(value) < float("inf"):
                return False
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return True


def seed_outputs_complete(
    output_dir: Path, pdb_id: str, seed: int, samples: int
) -> bool:
    """深度检查一个 PDB/seed 的文件集合与文件内容。"""

    pairs = expected_paths(output_dir, pdb_id, seed, samples)
    directory = predictions_dir(output_dir, pdb_id, seed)
    expected_cifs = {cif_path for cif_path, _ in pairs}
    expected_confidences = {confidence_path for _, confidence_path in pairs}
    # save_structure_cif() 会额外生成 ``*_wounresol.cif``；它不是独立 sample。
    actual_cifs = {
        path
        for path in directory.glob(f"{pdb_id}_sample_*.cif")
        if not path.name.endswith("_wounresol.cif")
    }
    actual_confidences = set(
        directory.glob(f"{pdb_id}_summary_confidence_sample_*.json")
    )
    if actual_cifs != expected_cifs or actual_confidences != expected_confidences:
        return False
    return all(
        cif_output_valid(cif_path)
        and confidence_output_valid(confidence_path)
        for cif_path, confidence_path in pairs
    )


def outputs_complete(
    output_dir: Path, pdb_id: str, seeds: Sequence[int], samples: int
) -> bool:
    """检查所有 seed/sample 的文件和关键置信度字段是否完整。"""

    return all(
        seed_outputs_complete(output_dir, pdb_id, seed, samples)
        for seed in seeds
    )


def clean_seed_outputs(output_dir: Path, pdb_id: str, seed: int) -> None:
    """重跑某个 seed 前清理旧样本，避免残留样本被 PXMeter 扫描。"""

    directory = predictions_dir(output_dir, pdb_id, seed)
    if not directory.is_dir():
        return
    patterns = (
        f"{pdb_id}_sample_*.cif",
        f"{pdb_id}_summary_confidence_sample_*.json",
        f"{pdb_id}_full_data_sample_*.json",
    )
    for pattern in patterns:
        for path in directory.glob(pattern):
            path.unlink()


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """追加一条结构化错误记录。每个 rank 使用独立文件，无写入竞争。"""

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def progress_enabled(log_level: str) -> bool:
    """verbose 模式启用全局进度；quiet 完全关闭进度基础设施。"""

    return log_level == "verbose"


def reset_progress_dir(progress_dir: Path) -> None:
    """清理上次中断的状态并创建本次运行的进度目录。"""

    if progress_dir.is_symlink() or progress_dir.is_file():
        progress_dir.unlink()
    elif progress_dir.exists():
        shutil.rmtree(progress_dir)
    progress_dir.mkdir(parents=True)


def write_progress_state(
    progress_dir: Path, rank: int, state: dict[str, Any]
) -> None:
    """原子写入单个 rank 的最新状态，避免 rank 0 读到半个 JSON。"""

    path = progress_dir / f"rank_{rank}.json"
    temporary = progress_dir / f"rank_{rank}.json.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False)
    temporary.replace(path)


def append_progress_event(
    progress_dir: Path, rank: int, event: dict[str, Any]
) -> None:
    """追加进度事件；每个 rank 独占一个 JSONL，不存在写入竞争。"""

    record = {"rank": rank, "timestamp": time.time(), **event}
    append_jsonl(progress_dir / f"events_rank_{rank}.jsonl", record)


def read_progress_states(
    progress_dir: Path, world_size: int
) -> list[dict[str, Any]]:
    """读取当前可用的 rank 状态；短暂缺失或原子替换时留待下次轮询。"""

    states = []
    for rank in range(world_size):
        path = progress_dir / f"rank_{rank}.json"
        try:
            with path.open("r", encoding="utf-8") as handle:
                state = json.load(handle)
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            continue
        if isinstance(state, dict):
            states.append(state)
    return states


def aggregate_progress(states: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """将各 rank 状态聚合为全局 PDB 进度。"""

    totals = {
        key: sum(int(state.get(key, 0)) for state in states)
        for key in ("assigned", "succeeded", "skipped", "failed")
    }
    totals["processed"] = (
        totals["succeeded"] + totals["skipped"] + totals["failed"]
    )
    active = [state for state in states if state.get("current_pdb")]
    latest = max(active, key=lambda state: state.get("updated_at", 0), default=None)
    totals["active"] = len(active)
    totals["latest"] = latest
    return totals


def finished_rank_states(
    states: Sequence[dict[str, Any]], world_size: int
) -> list[dict[str, Any]] | None:
    """全部 rank 完成时返回按 rank 排序的状态，否则返回 ``None``。"""

    by_rank = {
        int(state["rank"]): state
        for state in states
        if type(state.get("rank")) is int
    }
    expected_ranks = set(range(world_size))
    if set(by_rank) != expected_ranks:
        return None
    ordered = [by_rank[rank] for rank in range(world_size)]
    if not all(state.get("finished") is True for state in ordered):
        return None
    return ordered


def wait_for_finished_rank_states(
    progress_dir: Path,
    world_size: int,
    poll_seconds: float = PROGRESS_POLL_SECONDS,
    timeout_seconds: float = FINAL_STATUS_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """通过共享状态文件等待各 rank，超时后报告未完成的 rank。"""

    deadline = time.monotonic() + timeout_seconds
    while True:
        current_states = read_progress_states(progress_dir, world_size)
        finished = finished_rank_states(current_states, world_size)
        if finished is not None:
            return finished
        if time.monotonic() >= deadline:
            by_rank = {
                int(state["rank"]): state
                for state in current_states
                if type(state.get("rank")) is int
            }
            incomplete = [
                rank
                for rank in range(world_size)
                if rank not in by_rank or by_rank[rank].get("finished") is not True
            ]
            activity = {
                rank: {
                    "current_pdb": by_rank.get(rank, {}).get("current_pdb"),
                    "current_seed": by_rank.get(rank, {}).get("current_seed"),
                    "updated_at": by_rank.get(rank, {}).get("updated_at"),
                }
                for rank in incomplete
            }
            raise TimeoutError(
                f"Timed out after {timeout_seconds:.0f}s waiting for rank(s) "
                f"{incomplete}; last activity: {activity}"
            )
        time.sleep(poll_seconds)


def write_initialization_state(
    progress_dir: Path,
    rank: int,
    error: str | None,
) -> None:
    """原子发布配置/数据集初始化结果，供所有 rank 在 collective 前检查。"""

    write_json_atomic(
        progress_dir / f"init_rank_{rank}.json",
        {
            "rank": rank,
            "ok": error is None,
            "error": error,
            "updated_at": time.time(),
        },
    )


def wait_for_initialization_states(
    progress_dir: Path,
    world_size: int,
    timeout_seconds: float = FINAL_STATUS_TIMEOUT_SECONDS,
    poll_seconds: float = PROGRESS_POLL_SECONDS,
) -> list[dict[str, Any]]:
    """等待所有 rank 发布初始化状态，避免某一 rank 异常时其余 rank 进 barrier。"""

    deadline = time.monotonic() + timeout_seconds
    while True:
        states = []
        for rank in range(world_size):
            path = progress_dir / f"init_rank_{rank}.json"
            try:
                with path.open("r", encoding="utf-8") as handle:
                    state = json.load(handle)
            except (FileNotFoundError, OSError, json.JSONDecodeError):
                continue
            if isinstance(state, dict) and state.get("rank") == rank:
                states.append(state)
        if len(states) == world_size:
            return states
        if time.monotonic() >= deadline:
            present = {state["rank"] for state in states}
            missing = sorted(set(range(world_size)) - present)
            raise TimeoutError(
                f"Timed out after {timeout_seconds:.0f}s waiting for "
                f"initialization state from rank(s) {missing}"
            )
        time.sleep(poll_seconds)


def finish_distributed_planning(dist_module: Any) -> None:
    """在推理开始前同步规划结果并统一关闭分布式进程组。"""

    if dist_module.is_available() and dist_module.is_initialized():
        dist_module.barrier()
        dist_module.destroy_process_group()


def aggregate_rank_stats(
    states: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, int]], dict[str, int]]:
    """从最终 rank 状态构造兼容原格式的 per-rank 和全局统计。"""

    per_rank = [
        {key: int(state.get(key, 0)) for key in STAT_KEYS}
        for state in states
    ]
    total = {
        key: sum(rank_stats[key] for rank_stats in per_rank)
        for key in STAT_KEYS
    }
    return per_rank, total


def batch_exit_code(total_stats: dict[str, int]) -> int:
    """将全局 case 统计转换为批任务退出码。"""

    return 1 if int(total_stats.get("failed", 0)) > 0 else 0


def write_json_atomic(path: Path, data: Any) -> None:
    """原子写 JSON，避免消费者读到尚未完成的最终汇总。"""

    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def make_run_identity(args: argparse.Namespace) -> dict[str, Any]:
    """构造会影响输出内容或断点续跑判定的稳定运行身份。"""

    checkpoint_stat = args.checkpoint.stat()
    return {
        "schema_version": 1,
        "model_name": args.model_name,
        "checkpoint": {
            "path": str(args.checkpoint),
            "size": checkpoint_stat.st_size,
            "sha256": checkpoint_sha256(args.checkpoint),
        },
        "indices_csv": str(args.indices_csv),
        "data_root": str(args.data_root),
        "seeds": args.seeds,
        "samples": args.samples,
        "cycles": args.effective_cycles,
        "steps": args.effective_steps,
        "dtype": args.dtype,
        "max_tokens": args.max_tokens,
        "limit": args.limit,
        "use_msa": args.use_msa,
        "use_rna_msa": args.use_rna_msa,
        "use_template": args.use_template,
        "triangle_multiplicative": args.triangle_multiplicative,
        "triangle_attention": args.triangle_attention,
        "enable_cache": args.enable_cache,
        "enable_fusion": args.enable_fusion,
        "enable_tf32": args.enable_tf32,
    }


def output_has_predictions(output_dir: Path) -> bool:
    """判断目录是否已有可被本脚本复用或覆盖的预测结构。"""

    return next(output_dir.glob("*/seed_*/predictions/*.cif"), None) is not None


def clear_prediction_outputs(output_dir: Path) -> None:
    """移除旧运行的 PDB 输出目录；仅在身份冲突且显式 overwrite 时调用。"""

    pdb_dirs = {
        cif_path.parents[2]
        for cif_path in output_dir.glob("*/seed_*/predictions/*.cif")
    }
    for pdb_dir in sorted(pdb_dirs):
        shutil.rmtree(pdb_dir)


def ensure_run_identity(
    output_dir: Path,
    identity: dict[str, Any],
    overwrite: bool,
) -> None:
    """阻止不同模型、权重或推理配置在同一目录中断点混用。"""

    identity_path = output_dir / RUN_IDENTITY_NAME
    existing = None
    if identity_path.exists():
        try:
            with identity_path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            if not overwrite:
                raise ValueError(
                    f"Invalid run identity file {identity_path}; use --overwrite "
                    "to replace it."
                ) from exc
    elif output_has_predictions(output_dir):
        if not overwrite:
            raise ValueError(
                f"Output directory {output_dir} contains predictions without a run "
                "identity; use a new directory or pass --overwrite."
            )
        clear_prediction_outputs(output_dir)

    if existing is not None and existing != identity and not overwrite:
        raise ValueError(
            f"Output directory {output_dir} belongs to a different model, "
            "checkpoint, dataset, or inference configuration; use a new directory "
            "or pass --overwrite."
        )
    if existing is not None and existing != identity and overwrite:
        clear_prediction_outputs(output_dir)
    if existing != identity:
        write_json_atomic(identity_path, identity)


@contextlib.contextmanager
def output_directory_lock(output_dir: Path, enabled: bool = True):
    """阻止两个批任务同时修改同一输出目录。

    torchrun 的所有 rank 属于同一个任务，因此仅 rank 0 持锁；文件锁会在进程
    异常退出时由操作系统自动释放，不会留下阻塞后续运行的僵尸锁。
    """

    if not enabled:
        yield
        return

    lock_path = output_dir / OUTPUT_LOCK_NAME
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            owner = handle.read().strip() or "unknown owner"
            raise RuntimeError(
                f"Another batch inference is already writing to {output_dir}; "
                f"lock owner: {owner}"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {"pid": os.getpid(), "started_at": time.time()},
                ensure_ascii=False,
            )
        )
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_progress_events(
    path: Path, offset: int
) -> tuple[list[dict[str, Any]], int]:
    """从 offset 增量读取完整 JSONL 事件，未写完的末行留到下次。"""

    events = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            handle.seek(offset)
            while True:
                line_offset = handle.tell()
                line = handle.readline()
                if not line:
                    return events, handle.tell()
                if not line.endswith("\n"):
                    return events, line_offset
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    events.append(event)
    except (FileNotFoundError, OSError):
        return events, offset


def format_progress_event(event: dict[str, Any]) -> str:
    """将结构化事件格式化为 rank 0 输出的一行详细日志。"""

    timestamp = time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(event.get("timestamp", time.time()))
    )
    level = str(event.get("level", "INFO")).upper()
    rank = event.get("rank", "?")
    return f"{timestamp} {level} [rank={rank}] {event.get('message', '')}"


class GlobalProgressMonitor:
    """由 rank 0 轮询共享目录，集中展示所有 GPU 的 PDB 进度和事件。"""

    def __init__(
        self,
        progress_dir: Path,
        world_size: int,
        total: int,
        stream: Any = None,
    ) -> None:
        self.progress_dir = progress_dir
        self.world_size = world_size
        self.total = total
        self.stream = stream if stream is not None else sys.stderr
        self.is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._bar: Any = None
        self._processed = 0
        self._event_offsets = {rank: 0 for rank in range(world_size)}
        self._last_heartbeat = 0.0

    def start(self) -> None:
        """启动后台轮询；TTY 使用 tqdm，非 TTY 使用普通心跳日志。"""

        if self.is_tty:
            from tqdm import tqdm

            self._bar = tqdm(
                total=self.total,
                desc="PDB",
                unit="pdb",
                dynamic_ncols=True,
                file=self.stream,
            )
        self._poll(force_heartbeat=True)
        self._thread = threading.Thread(
            target=self._run, name="global-pdb-progress", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """停止轮询并执行最后一次刷新。"""

        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=PROGRESS_POLL_SECONDS * 4)
        self._poll(force_heartbeat=not self.is_tty)
        if self._bar is not None:
            self._bar.close()

    def _run(self) -> None:
        while not self._stop.wait(PROGRESS_POLL_SECONDS):
            self._poll()

    def _poll(self, force_heartbeat: bool = False) -> None:
        states = read_progress_states(self.progress_dir, self.world_size)
        progress = aggregate_progress(states)
        processed = min(progress["processed"], self.total)

        if self._bar is not None:
            if processed > self._processed:
                self._bar.update(processed - self._processed)
            latest = progress["latest"]
            activity = ""
            if latest is not None:
                activity = f" r{latest.get('rank')}:{latest.get('current_pdb')}"
                if latest.get("current_seed") is not None:
                    activity += (
                        f" seed={latest.get('seed_index')}/"
                        f"{latest.get('seed_count')}"
                    )
            self._bar.set_postfix_str(
                f"ok={progress['succeeded']} fail={progress['failed']} "
                f"skip={progress['skipped']} active={progress['active']}"
                f"{activity}",
                refresh=True,
            )

        self._processed = max(self._processed, processed)
        self._emit_new_events()

        now = time.monotonic()
        heartbeat_due = now - self._last_heartbeat >= PROGRESS_HEARTBEAT_SECONDS
        if self._bar is None and (force_heartbeat or heartbeat_due):
            LOGGER.info(
                "Global progress: %d/%d PDB; succeeded=%d, failed=%d, "
                "skipped=%d, active=%d",
                processed,
                self.total,
                progress["succeeded"],
                progress["failed"],
                progress["skipped"],
                progress["active"],
            )
            self._last_heartbeat = now

    def _emit_new_events(self) -> None:
        for rank in range(self.world_size):
            path = self.progress_dir / f"events_rank_{rank}.jsonl"
            events, offset = read_progress_events(path, self._event_offsets[rank])
            self._event_offsets[rank] = offset
            for event in events:
                message = format_progress_event(event)
                if self._bar is not None:
                    from tqdm import tqdm

                    tqdm.write(message, file=self.stream)
                else:
                    level = str(event.get("level", "INFO")).upper()
                    if level == "ERROR":
                        LOGGER.error(
                            "[rank=%s] %s",
                            event.get("rank", "?"),
                            event.get("message", ""),
                        )
                    else:
                        LOGGER.info(
                            "[rank=%s] %s",
                            event.get("rank", "?"),
                            event.get("message", ""),
                        )


def _run_unlocked(args: argparse.Namespace) -> int:
    """执行单卡或 torchrun 多卡批量推理，返回进程退出码。"""

    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader

    from protenix.utils.distributed import DIST_WRAPPER
    from protenix.utils.file_io import load_gzip_pickle
    from protenix.utils.seed import seed_everything
    from runner.inference import (
        InferenceRunner,
        fix_cterminal_carboxyl_oxygens,
        update_inference_configs,
    )

    class ExplicitCheckpointInferenceRunner(InferenceRunner):
        """保持官方 Runner 行为，但严格使用命令行给出的 checkpoint 路径。"""

        def load_checkpoint(self) -> None:
            checkpoint_path = Path(self.configs.load_checkpoint_path)
            self.print(
                f"Loading {self.configs.model_name} from {checkpoint_path}, "
                "strict: True"
            )
            try:
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location=self.device,
                    weights_only=False,
                )
                state_dict = normalize_checkpoint_state_dict(checkpoint)
                load_state_dict_strict(
                    self.model,
                    state_dict,
                    checkpoint_path,
                    self.configs.model_name,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load checkpoint {checkpoint_path} for model "
                    f"{self.configs.model_name!r}: {exc}"
                ) from exc
            self.model.eval()
            self.print("Finish loading checkpoint.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Protenix batch inference script")

    rank = DIST_WRAPPER.rank
    world_size = DIST_WRAPPER.world_size
    if rank == 0 and args.log_level == "verbose":
        LOGGER.info("Initializing dataset and model on %d rank(s)...", world_size)

    use_progress = progress_enabled(args.log_level)
    progress_dir = args.output_dir / PROGRESS_DIR_NAME
    # 状态文件同时承担最终协调职责，因此 quiet 模式也必须启用。这里只在所有
    # rank 开始写状态前使用一次短 barrier；耗时推理结束后不再调用 collective。
    if rank == 0:
        reset_progress_dir(progress_dir)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # 配置或数据集初始化发生异常时先写共享状态；所有 rank 确认状态齐全后才会
    # 进入后续 collective，避免健康 rank 永久等待已经退出的 rank。
    configs = None
    dataset = None
    run_identity = None
    initialization_error = None
    try:
        configs = build_configs(args)
        dataset = build_dataset(args, configs)
        if rank == 0:
            run_identity = make_run_identity(args)
            ensure_run_identity(
                args.output_dir,
                run_identity,
                overwrite=args.overwrite,
            )
    except Exception:
        initialization_error = traceback.format_exc()
    write_initialization_state(progress_dir, rank, initialization_error)
    initialization_states = wait_for_initialization_states(
        progress_dir,
        world_size,
        timeout_seconds=FINAL_STATUS_TIMEOUT_SECONDS,
    )
    initialization_failures = [
        state for state in initialization_states if state.get("ok") is not True
    ]
    if initialization_failures:
        # 状态已齐全，因此各 rank 会从同一路径到达这里并同步关闭进程组。
        finish_distributed_planning(dist)
        failed_ranks = [state.get("rank") for state in initialization_failures]
        first_error = initialization_failures[0].get("error", "unknown error")
        raise RuntimeError(
            f"Configuration/dataset initialization failed on rank(s) "
            f"{failed_ranks}; first error:\n{first_error}"
        )

    # 所有 rank 在任何推理写盘前得到相同的完成快照。只对待处理 PDB 按
    # num_tokens² 做确定性均衡；完整 PDB 仅由一个 rank 计入 skipped。
    pending_by_rank, skipped_by_rank, estimated_loads = plan_rank_work(
        dataset=dataset,
        output_dir=args.output_dir,
        seeds=args.seeds,
        samples=args.samples,
        world_size=world_size,
        overwrite=args.overwrite,
    )
    # 防止较快 rank 在较慢 rank 尚未完成输出快照时开始写入，从而让各 rank
    # 得到不一致的 pending 集合。这是本次运行最后一个 collective；后续推理
    # 各 rank 完全独立，最终统计也只通过共享状态文件协调。
    # 所有 rank 在同一个同步点销毁 NCCL 进程组。不能让先完成推理的 rank
    # 提前销毁，否则长尾任务可能令各 rank 的 destroy_process_group() 调用
    # 间隔超过进程组 timeout，并在 ncclCommAbort 阶段挂起。
    finish_distributed_planning(dist)
    pending_indices = pending_by_rank[rank]
    skipped_indices = skipped_by_rank[rank]
    assigned_count = len(pending_indices) + len(skipped_indices)
    skipped = len(skipped_indices)

    data_view = SafeFeatureDataset(dataset, pending_indices)
    # num_workers=0 时特征在主进程生成，因此也要显式固定 Python/NumPy/Torch RNG。
    seed_everything(
        seed=(args.seeds[0] + rank) % (2**32),
        deterministic=configs.deterministic,
    )
    # 固定 DataLoader 的 worker 基础种子，使 MSA 子采样在相同参数下可复现。
    loader_generator = torch.Generator()
    loader_generator.manual_seed((args.seeds[0] + rank) % (2**32))
    dataloader = DataLoader(
        data_view,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=first_item,
        persistent_workers=args.num_workers > 0,
        generator=loader_generator,
    )
    error_log = args.output_dir / f"errors_rank_{rank}.jsonl"
    # 错误日志只描述本轮运行；避免断点续跑成功后仍残留历史失败记录。
    error_log.unlink(missing_ok=True)
    stats = {
        "assigned": assigned_count,
        "succeeded": 0,
        "skipped": skipped,
        "failed": 0,
    }
    progress_state = {
        "rank": rank,
        **stats,
        "current_pdb": None,
        "current_seed": None,
        "seed_index": None,
        "seed_count": len(args.seeds),
        "pending": len(pending_indices),
        "estimated_token_cost": estimated_loads[rank],
        "finished": False,
        "finished_at": None,
        "updated_at": time.time(),
    }

    def publish_progress() -> None:
        progress_state.update(stats)
        progress_state["updated_at"] = time.time()
        write_progress_state(progress_dir, rank, progress_state)

    def emit_event(message: str, level: str = "INFO") -> None:
        if use_progress:
            append_progress_event(
                progress_dir, rank, {"level": level, "message": message}
            )
        elif level == "ERROR":
            LOGGER.error("[rank=%d] %s", rank, message)

    publish_progress()
    emit_event(
        f"assigned {assigned_count} PDB(s): {len(pending_indices)} pending, "
        f"{skipped} skipped, estimated_token_cost={estimated_loads[rank]}"
    )
    if use_progress:
        for index in skipped_indices:
            emit_event(f"[{sample_name(dataset, index)}] complete; skipping")

    progress_monitor = None
    if use_progress and rank == 0:
        progress_monitor = GlobalProgressMonitor(
            progress_dir=progress_dir,
            world_size=world_size,
            total=len(dataset),
        )
        progress_monitor.start()

    runner = None
    if pending_indices:
        try:
            # 每个 torchrun rank 各加载一份模型，并绑定自己的 local_rank GPU。
            runner = ExplicitCheckpointInferenceRunner(configs)
        except Exception:
            stats["failed"] += len(pending_indices)
            append_jsonl(
                error_log,
                {
                    "stage": "model_initialization",
                    "pending_pdb_count": len(pending_indices),
                    "error": traceback.format_exc(),
                },
            )
            emit_event(
                f"model initialization failed; {len(pending_indices)} pending "
                f"PDB(s) marked failed; see {error_log}",
                level="ERROR",
            )
            # 保持本 rank 存活到状态文件协调结束，让 rank 0 能产出明确汇总。
            dataloader = ()
            publish_progress()

    for dataset_index, data, data_error in dataloader:
        pdb_id = sample_name(dataset, dataset_index)
        pdb_started = time.time()
        progress_state.update(
            current_pdb=pdb_id,
            current_seed=None,
            seed_index=None,
        )
        publish_progress()
        if data_error is not None:
            stats["failed"] += 1
            append_jsonl(
                error_log,
                {"pdb_id": pdb_id, "stage": "data", "error": data_error},
            )
            emit_event(
                f"[{pdb_id}] data preparation failed; see {error_log}",
                level="ERROR",
            )
            progress_state.update(current_pdb=None, current_seed=None, seed_index=None)
            publish_progress()
            continue

        input_feature_dict = None
        atom_array = None
        try:
            atom_array = data.pop("cropped_atom_array")
            data.pop("cropped_token_array", None)
            bioassembly_path = Path(data["basic"]["bioassembly_dict_fpath"])
            n_token = data["basic"]["N_token"].item()

            # 训练数据集还会返回真值标签。推理时只保留输入特征，避免把标签搬到
            # GPU，既防止不必要的显存占用，也确保模型前向没有真值信息泄漏。
            input_feature_dict = data["input_feature_dict"]
            del data

            # CIF writer 需要原始 bioassembly 中的 entity polymer 类型信息。
            bioassembly = load_gzip_pickle(bioassembly_path)
            entity_poly_type = {
                key: value
                for key, value in bioassembly["entity_poly_type"].items()
                if value != "non-polymer"
            }
            del bioassembly

            # 与 get_raw_atom_array() 保持一致。清零未使用的 charge 注释，可避免
            # DockQ/PXMeter 下游解析某些 CIF 时失败。
            atom_array.charge = np.zeros(len(atom_array), dtype=np.int32)
        except Exception:
            stats["failed"] += 1
            append_jsonl(
                error_log,
                {
                    "pdb_id": pdb_id,
                    "stage": "prepare_inference",
                    "error": traceback.format_exc(),
                },
            )
            emit_event(
                f"[{pdb_id}] inference input preparation failed; see {error_log}",
                level="ERROR",
            )
            del input_feature_dict, atom_array
            torch.cuda.empty_cache()
            progress_state.update(current_pdb=None, current_seed=None, seed_index=None)
            publish_progress()
            continue

        pdb_failed = False
        for seed_index, seed in enumerate(args.seeds, start=1):
            prediction = None
            progress_state.update(
                current_seed=seed,
                seed_index=seed_index,
            )
            publish_progress()
            if not args.overwrite and seed_outputs_complete(
                args.output_dir, pdb_id, seed, args.samples
            ):
                emit_event(f"[{pdb_id} seed={seed}] complete; reusing")
                continue
            try:
                # 不完整或强制重跑时，先移除该 seed 的旧样本，避免样本数变化后
                # 遗留文件被 PXMeter 当成新的候选结构。
                clean_seed_outputs(args.output_dir, pdb_id, seed)
                seed_everything(seed=seed, deterministic=configs.deterministic)
                runner.update_model_configs(
                    update_inference_configs(configs, n_token)
                )
                started = time.time()
                # inference forward 会原地删除 profile/msa/template 等特征；每个
                # seed 必须使用独立的字典容器，否则第二个 seed 起会缺少特征。
                prediction = runner.predict(
                    make_prediction_input(input_feature_dict)
                )
                prediction["coordinate"] = fix_cterminal_carboxyl_oxygens(
                    prediction["coordinate"], atom_array
                )
                runner.dumper.dump(
                    dataset_name="",
                    pdb_id=pdb_id,
                    seed=seed,
                    pred_dict=prediction,
                    atom_array=atom_array,
                    entity_poly_type=entity_poly_type,
                )
                emit_event(
                    f"[{pdb_id} seed={seed}] succeeded in "
                    f"{time.time() - started:.1f}s"
                )
            except Exception:
                pdb_failed = True
                append_jsonl(
                    error_log,
                    {
                        "pdb_id": pdb_id,
                        "seed": seed,
                        "stage": "inference",
                        "error": traceback.format_exc(),
                    },
                )
                emit_event(
                    f"[{pdb_id} seed={seed}] inference failed; see {error_log}",
                    level="ERROR",
                )
            finally:
                # DataDumper 写盘后立刻释放坐标和置信度张量，避免上一条预测在
                # 下一次 model forward 期间仍占用 GPU 显存。
                del prediction
                torch.cuda.empty_cache()

        if not pdb_failed and outputs_complete(
            args.output_dir, pdb_id, args.seeds, args.samples
        ):
            stats["succeeded"] += 1
            emit_event(
                f"[{pdb_id}] completed {len(args.seeds)} seed(s) x "
                f"{args.samples} sample(s) in {time.time() - pdb_started:.1f}s"
            )
        else:
            stats["failed"] += 1
            emit_event(
                f"[{pdb_id}] failed after {time.time() - pdb_started:.1f}s",
                level="ERROR",
            )

        progress_state.update(current_pdb=None, current_seed=None, seed_index=None)
        publish_progress()

        # 当前 PDB 的原始 CPU 特征不再使用，主动释放后再进入下一条数据。
        del input_feature_dict, atom_array
        torch.cuda.empty_cache()

    # 每个 rank 原子发布最终状态。非零 rank 无需在 NCCL collective 中等待；
    # rank 0 通过共享文件等待，因此慢 rank 耗时超过 NCCL timeout 也不会中断。
    progress_state.update(
        current_pdb=None,
        current_seed=None,
        seed_index=None,
        finished=True,
        finished_at=time.time(),
    )
    publish_progress()

    exit_code = 0
    if rank == 0:
        try:
            final_states = wait_for_finished_rank_states(
                progress_dir,
                world_size,
                timeout_seconds=FINAL_STATUS_TIMEOUT_SECONDS,
            )
        finally:
            if progress_monitor is not None:
                progress_monitor.stop()
        gathered, total_stats = aggregate_rank_stats(final_states)
        summary = {
            "model": args.model_name,
            "checkpoint": run_identity["checkpoint"],
            "indices_csv": str(args.indices_csv),
            "pdb_ids": selected_pdb_ids(dataset),
            "filtered_over_token_limit": {
                "count": len(args.filtered_over_token_limit),
                "pdb_ids": args.filtered_over_token_limit,
                "audit_available": args.token_filter_audit_available,
            },
            "ref_assembly_id": "1",
            "world_size": world_size,
            "seeds": args.seeds,
            "samples": args.samples,
            "cycles": args.effective_cycles,
            "steps": args.effective_steps,
            "requested_max_tokens": args.requested_max_tokens,
            "effective_max_tokens": args.max_tokens,
            "dtype": args.dtype,
            "use_msa": args.use_msa,
            "use_rna_msa": args.use_rna_msa,
            "use_template": args.use_template,
            "assignment_strategy": ASSIGNMENT_STRATEGY,
            "final_status_timeout_seconds": FINAL_STATUS_TIMEOUT_SECONDS,
            "per_rank": gathered,
            "total": total_stats,
        }
        summary_path = args.output_dir / "batch_summary.json"
        write_json_atomic(summary_path, summary)
        LOGGER.info(
            "Batch finished: succeeded=%d, skipped=%d, failed=%d; summary=%s",
            total_stats["succeeded"],
            total_stats["skipped"],
            total_stats["failed"],
            summary_path,
        )
        exit_code = batch_exit_code(total_stats)

    if rank == 0 and progress_dir.is_dir():
        shutil.rmtree(progress_dir)
    # 非零 rank 始终正常退出，避免 torchrun 在 rank 0 完成全局汇总前提前终止
    # 其他仍在工作的进程；全局失败最终由 rank 0 的退出码表达。
    return exit_code


def run(args: argparse.Namespace) -> int:
    """在 rank 0 持有输出目录锁的情况下执行批量推理。"""

    from protenix.utils.distributed import DIST_WRAPPER

    with output_directory_lock(
        args.output_dir,
        enabled=DIST_WRAPPER.rank == 0,
    ):
        return _run_unlocked(args)


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口：参数错误返回 2，样本失败由 run() 返回 1。"""

    try:
        args = validate_args(parse_args(argv))
        configure_logging(args.log_level)
        if args.requested_max_tokens != args.max_tokens:
            LOGGER.info(
                "%s enforces max_tokens=%d; requested value %d was adjusted",
                args.model_name,
                args.max_tokens,
                args.requested_max_tokens,
            )
        if not args.token_filter_audit_available:
            LOGGER.warning(
                "Could not audit token-filtered PDB IDs because %s lacks "
                "pdb_id/entry_id or num_tokens columns",
                args.indices_csv,
            )
        configure_environment(args)
        return run(args)
    except (FileNotFoundError, NotADirectoryError, PermissionError, ValueError) as exc:
        logging.error("%s", exc)
        return 2
    except (RuntimeError, TimeoutError) as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
