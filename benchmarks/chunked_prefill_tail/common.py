"""Provenance, summary, and immutable-output helpers for release diagnostics."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATHS = (
    Path("nanovllm"),
    Path("benchmarks/chunked_prefill_tail"),
    Path("tests/run_varlen_graph_config.py"),
    Path("tests/run_varlen_511_contract.py"),
    Path("pyproject.toml"),
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def command_output(*command: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_files() -> list[Path]:
    files: set[Path] = set()
    for relative in SOURCE_PATHS:
        target = ROOT / relative
        if target.is_dir():
            files.update(path for path in target.rglob("*.py") if path.is_file())
        elif target.is_file():
            files.add(target)
        else:
            raise FileNotFoundError(f"release source path is missing: {relative}")
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def source_identity() -> dict[str, object]:
    aggregate = hashlib.sha256()
    entries = []
    for path in _source_files():
        relative = path.relative_to(ROOT).as_posix()
        file_hash = sha256_file(path)
        size = path.stat().st_size
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(bytes.fromhex(file_hash))
        entries.append({"path": relative, "bytes": size, "sha256": file_hash})
    return {
        "algorithm": "sha256(path\\0size\\0sha256(file))",
        "aggregate_sha256": aggregate.hexdigest(),
        "file_count": len(entries),
        "files": entries,
    }


def git_identity() -> dict[str, object]:
    status = command_output("git", "status", "--porcelain=v1", "--untracked-files=all")
    return {
        "commit": command_output("git", "rev-parse", "HEAD"),
        "tree": command_output("git", "rev-parse", "HEAD^{tree}"),
        "branch": command_output("git", "branch", "--show-current"),
        "clean": not status,
        "status": status.splitlines(),
    }


def release_pin() -> dict[str, object]:
    return {"git": git_identity(), "source": source_identity()}


def validate_release_pin(expected_commit: str, expected_source_sha256: str) -> dict:
    if not COMMIT_RE.fullmatch(expected_commit):
        raise ValueError("--expected-commit must be a full lowercase 40-hex commit")
    if not SHA256_RE.fullmatch(expected_source_sha256):
        raise ValueError("--expected-source-sha256 must be lowercase 64-hex SHA-256")
    actual = release_pin()
    if not actual["git"]["clean"]:
        raise RuntimeError(
            "release diagnostics require a clean worktree; status="
            + repr(actual["git"]["status"])
        )
    if actual["git"]["commit"] != expected_commit:
        raise RuntimeError(
            f"commit pin mismatch: expected {expected_commit}, "
            f"found {actual['git']['commit']}"
        )
    actual_source = actual["source"]["aggregate_sha256"]
    if actual_source != expected_source_sha256:
        raise RuntimeError(
            f"source pin mismatch: expected {expected_source_sha256}, "
            f"found {actual_source}"
        )
    return actual


def model_identity(model_path: Path) -> dict[str, object]:
    resolved = model_path.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"model path is not a directory: {resolved}")
    files = sorted(path for path in resolved.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"model directory contains no files: {resolved}")
    aggregate = hashlib.sha256()
    entries = []
    for path in files:
        relative = path.relative_to(resolved).as_posix()
        file_hash = sha256_file(path)
        size = path.stat().st_size
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(bytes.fromhex(file_hash))
        entries.append({"path": relative, "bytes": size, "sha256": file_hash})
    return {
        "argument": str(model_path),
        "resolved_path": str(resolved),
        "aggregate_sha256": aggregate.hexdigest(),
        "file_count": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "files": entries,
    }


def environment_identity(torch_module, transformers_module) -> dict[str, object]:
    import flash_attn

    cuda = torch_module.cuda
    retained_environment_names = (
        "CUDA_VISIBLE_DEVICES",
        "CUDA_LAUNCH_BLOCKING",
        "CUBLAS_WORKSPACE_CONFIG",
        "HF_HOME",
        "NCCL_DEBUG",
        "NCCL_P2P_DISABLE",
        "PYTHONHASHSEED",
        "PYTHONPATH",
        "TORCHINDUCTOR_CACHE_DIR",
    )
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch_module.__version__,
        "cuda_build": torch_module.version.cuda,
        "transformers": transformers_module.__version__,
        "flash_attn": flash_attn.__version__,
        "python_gc_enabled": gc.isenabled(),
        # A fixed non-secret allowlist makes execution-affecting state visible
        # without copying credentials or unrelated process environment.
        "environment_variables": {
            name: os.environ.get(name) for name in retained_environment_names
        },
    }
    if cuda.is_available():
        properties = cuda.get_device_properties(0)
        result.update({
            "gpu": cuda.get_device_name(0),
            "gpu_total_memory_bytes": properties.total_memory,
            "compute_capability": list(cuda.get_device_capability(0)),
            "driver": command_output(
                "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
            ).splitlines()[0],
        })
    return result


def nearest_rank_percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize an empty sample")
    index = min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def timing_summary(rows: list[dict]) -> dict[str, object] | None:
    if not rows:
        return None
    wall = [float(row["wall_ms"]) for row in rows]
    cuda = [float(row["cuda_ms"]) for row in rows]
    return {
        "count": len(rows),
        "wall_ms": {
            "median": sorted(wall)[len(wall) // 2]
            if len(wall) % 2 else (sorted(wall)[len(wall) // 2 - 1] + sorted(wall)[len(wall) // 2]) / 2,
            "p95": nearest_rank_percentile(wall, 0.95),
            "min": min(wall),
            "max": max(wall),
        },
        "cuda_ms": {
            "median": sorted(cuda)[len(cuda) // 2]
            if len(cuda) % 2 else (sorted(cuda)[len(cuda) // 2 - 1] + sorted(cuda)[len(cuda) // 2]) / 2,
            "p95": nearest_rank_percentile(cuda, 0.95),
            "min": min(cuda),
            "max": max(cuda),
        },
    }


def base_result(kind: str, argv: list[str], pin: dict, model: dict, environment: dict) -> dict:
    return {
        "schema_version": 1,
        "kind": kind,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "argv": argv,
        "cwd": str(Path.cwd()),
        "provenance": pin,
        "model": model,
        "environment": environment,
    }


def immutable_write_bytes(path: Path, data: bytes) -> None:
    target = path.expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o444,
    )
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written == 0:
                raise OSError("zero-byte write while retaining diagnostic JSON")
            view = view[written:]
        # Creation mode is filtered by umask; force the evidence contract's
        # exact read-only mode after the complete payload is present.
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def immutable_write_json(path: Path, payload: dict) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    immutable_write_bytes(path, data)


def handle_pin_query(print_source_sha256: bool, show_pin: bool) -> bool:
    if print_source_sha256:
        print(source_identity()["aggregate_sha256"])
        return True
    if show_pin:
        print(json.dumps(release_pin(), indent=2, sort_keys=True))
        return True
    return False
