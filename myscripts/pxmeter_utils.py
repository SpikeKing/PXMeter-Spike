"""Dependency-light helpers for local PXMeter evaluation workflows."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PXMETER_MODULES = (
    Path("pxmeter/__init__.py"),
    Path("benchmark/run_eval.py"),
    Path("benchmark/show_intersection_results.py"),
)


class ConfigurationError(RuntimeError):
    """本地文件或命令行配置无效。"""


class EvaluationError(RuntimeError):
    """PXMeter 执行或结果校验失败。"""


def is_pxmeter_source_root(path: Path) -> bool:
    """确认目录包含本工作流需要的 PXMeter 源码模块。"""

    return path.is_dir() and all(
        (path / relative).is_file() for relative in PXMETER_MODULES
    )


def validate_pxmeter_modules(path: Path) -> Path:
    """解析并校验本地 PXMeter 源码根目录。"""

    root = path.expanduser().resolve()
    if not is_pxmeter_source_root(root):
        missing = [
            str(relative)
            for relative in PXMETER_MODULES
            if not (root / relative).is_file()
        ]
        raise ConfigurationError(
            f"Invalid PXMeter source root {root}; missing: {', '.join(missing)}"
        )
    return root


def add_pxmeter_to_pythonpath(env: dict[str, str], pxmeter_root: Path) -> None:
    """将本地 PXMeter 源码目录置于子进程 PYTHONPATH 首位。"""

    root = str(pxmeter_root)
    existing = [
        item for item in env.get("PYTHONPATH", "").split(os.pathsep) if item
    ]
    env["PYTHONPATH"] = os.pathsep.join(
        [root, *(item for item in existing if item != root)]
    )


def run_pxmeter_module(
    module: str,
    arguments: Sequence[str],
    env: Mapping[str, str],
) -> None:
    """使用当前 Python 运行 PXMeter 模块，并将输出直接转发到终端。"""

    command = [sys.executable, "-m", module, *arguments]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=dict(env),
        check=False,
    )
    if completed.returncode != 0:
        raise EvaluationError(
            f"PXMeter module {module} exited with status {completed.returncode}"
        )


def write_json_atomic(path: Path, value: object) -> None:
    """通过同目录临时文件原子写入 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _remove_artifact(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def publish_artifact(source: Path, destination: Path) -> None:
    """替换单个发布产物；替换失败时恢复旧产物。"""

    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = destination.with_name(
        f".{destination.name}.backup.{uuid.uuid4().hex}"
    )
    had_destination = destination.exists() or destination.is_symlink()
    if had_destination:
        os.replace(destination, backup)
    try:
        os.replace(source, destination)
    except Exception:
        if had_destination:
            os.replace(backup, destination)
        raise
    else:
        if had_destination:
            _remove_artifact(backup)


def validate_summary_outputs(summary_dir: Path) -> None:
    """确认 PXMeter 始终应生成的聚合结果存在。"""

    required = (
        summary_dir / "Summary_table.csv",
        summary_dir / "DockQ_results.csv",
        summary_dir / "LDDT_results.csv",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise EvaluationError(
            "PXMeter aggregation did not generate required output(s): "
            + ", ".join(missing)
        )
