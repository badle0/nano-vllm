"""Shared, dependency-light helpers for request-metrics certification."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = 2
BENCHMARK_NAME = "request_metrics_certification"


def canonical_json_bytes(value) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def exclusive_json_dump(path: Path, value) -> None:
    """Write JSON once, refusing to replace even an empty existing artifact."""
    if not path.parent.is_dir():
        raise ValueError(f"output parent does not exist: {path.parent}")
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise ValueError(f"refusing to overwrite existing output: {path}") from exc


def parse_positive_int_csv(value: str, *, name: str) -> list[int]:
    try:
        parsed = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated integer list") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise ValueError(f"{name} values must all be positive")
    if len(parsed) != len(set(parsed)):
        raise ValueError(f"{name} values must be unique")
    return parsed


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile needs at least one value")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_observations(observations: list[dict]) -> dict:
    if len(observations) < 2:
        raise ValueError("at least one cold and one steady observation are required")
    cold = observations[0]
    steady = observations[1:]
    elapsed = [item["elapsed_s"] for item in steady]
    throughput = [item["output_tokens_per_s"] for item in steady]
    return {
        "cold": {
            "elapsed_s": cold["elapsed_s"],
            "output_tokens_per_s": cold["output_tokens_per_s"],
        },
        "steady": {
            "count": len(steady),
            "median_elapsed_s": percentile(elapsed, 0.5),
            "p95_elapsed_s": percentile(elapsed, 0.95),
            "median_output_tokens_per_s": percentile(throughput, 0.5),
            "p05_output_tokens_per_s": percentile(throughput, 0.05),
            "max_host_peak_rss_kib": max(
                item["resources"]["host_peak_rss_kib"] for item in steady
            ),
            "max_cuda_peak_allocated_bytes": max(
                item["resources"]["cuda_peak_allocated_bytes"] for item in steady
            ),
            "max_cuda_peak_reserved_bytes": max(
                item["resources"]["cuda_peak_reserved_bytes"] for item in steady
            ),
        },
    }


def derive_observation_seed(
    seed: int,
    pair_id: int,
    batch: int,
    output_length: int,
    repetition: int,
) -> int:
    material = f"{seed}:{pair_id}:{batch}:{output_length}:{repetition}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & ((1 << 63) - 1)


def case_key(batch: int, output_length: int) -> str:
    return f"b{batch}_o{output_length}"


def _git(source_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(source_root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def source_identity(source_root: Path, expected_head: str) -> dict:
    root = source_root.resolve(strict=True)
    head = _git(root, "rev-parse", "HEAD")
    expected = _git(root, "rev-parse", expected_head)
    if head != expected:
        raise ValueError(f"source HEAD mismatch: {head} != {expected}")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise ValueError(f"source worktree is not clean:\n{status}")
    return {
        "root": str(root),
        "head": head,
        "expected_head": expected,
        "tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "branch": _git(root, "branch", "--show-current") or None,
        "clean": True,
        "status_porcelain": status,
    }


def model_identity(model_root: Path) -> dict:
    root = model_root.resolve(strict=True)
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(root).parts
    )
    if not files:
        raise ValueError(f"model directory contains no files: {root}")
    records = []
    for path in files:
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "root": str(root),
        "file_count": len(records),
        "files": records,
        "fingerprint_sha256": canonical_sha256(records),
    }


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def environment_identity(torch) -> dict:
    try:
        driver_query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        driver_versions = driver_query.stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        driver_versions = None
    properties = torch.cuda.get_device_properties(0)
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "transformers": package_version("transformers"),
        "flash_attn": package_version("flash-attn"),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_total_memory_bytes": properties.total_memory,
        "gpu_multiprocessor_count": properties.multi_processor_count,
        "visible_cuda_devices": torch.cuda.device_count(),
        "nvidia_driver_versions": driver_versions,
        "selected_environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "PYTHONPATH",
                "TORCHINDUCTOR_CACHE_DIR",
            )
        },
    }


def current_rss_kib() -> int | None:
    try:
        resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") // 1024
    except (OSError, ValueError, IndexError):
        return None


def peak_rss_kib() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(value // 1024)
    return int(value)
