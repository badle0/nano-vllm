#!/usr/bin/env python3
"""Go/no-go benchmark for CUB segmented BF16 keys-only radix sort."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = Path(__file__).resolve().parent


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def percentile(samples: list[float], quantile: float) -> float:
    ordered = sorted(samples)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def load_extension(build_directory: Path):
    build_directory.mkdir(parents=True, exist_ok=True)
    major, minor = torch.cuda.get_device_capability()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    return load(
        name="nano_vllm_topp_cub_bf16_sort_v1",
        sources=[
            str(SOURCE_DIR / "cub_bf16_sort_extension.cpp"),
            str(SOURCE_DIR / "cub_bf16_sort_extension.cu"),
        ],
        build_directory=str(build_directory),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=True,
    )


def time_cuda(operation, warmups: int, iterations: int) -> list[float]:
    for _ in range(warmups):
        operation()
    torch.cuda.synchronize()

    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--vocab", type=int, default=151_936)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument(
        "--build-directory",
        type=Path,
        default=Path("/workspace/.cache/nano-vllm/topp-cub-bf16-sort"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.batch <= 0 or args.vocab <= 0:
        raise ValueError("batch and vocab must be positive")

    extension = load_extension(args.build_directory)
    torch.manual_seed(args.seed)
    logits = torch.randn(
        args.batch,
        args.vocab,
        device="cuda",
        dtype=torch.bfloat16,
    )
    temperatures = torch.full(
        (args.batch,),
        args.temperature,
        device="cuda",
        dtype=torch.float32,
    )
    sorted_keys = torch.empty_like(logits)
    scratch_bytes = int(extension.workspace_size(args.batch, args.vocab))
    scratch = torch.empty(scratch_bytes, device="cuda", dtype=torch.uint8)

    operation = lambda: extension.sort_out(logits, sorted_keys, scratch)
    operation()
    torch.cuda.synchronize()

    candidate_values = sorted_keys.float().div(temperatures.unsqueeze(1))
    reference_values = torch.sort(
        logits.float().div(temperatures.unsqueeze(1)),
        dim=-1,
        descending=False,
    ).values
    mismatch_count = int((candidate_values != reference_values).sum().item())
    exact_match = mismatch_count == 0
    del candidate_values, reference_values
    torch.cuda.empty_cache()

    candidate_samples = time_cuda(operation, args.warmups, args.iterations)

    def reference_sort():
        return torch.sort(logits, dim=-1, descending=False).values

    reference_samples = time_cuda(
        reference_sort,
        args.warmups,
        args.iterations,
    )

    candidate_median = statistics.median(candidate_samples)
    gate_ms = 0.75
    result = {
        "schema_version": 1,
        "benchmark": "topp_cub_bf16_segmented_sort_keys_gate",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--short"),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "driver": torch.cuda.driver_version()
            if hasattr(torch.cuda, "driver_version")
            else None,
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
        },
        "configuration": {
            "batch": args.batch,
            "vocab": args.vocab,
            "dtype": "bfloat16",
            "temperature": args.temperature,
            "seed": args.seed,
            "warmups": args.warmups,
            "iterations": args.iterations,
            "build_directory": str(args.build_directory),
        },
        "correctness": {
            "comparison": (
                "cub_sort_keys(logits).float()/temperature == "
                "torch.sort(logits.float()/temperature).values"
            ),
            "exact_match": exact_match,
            "mismatch_count": mismatch_count,
        },
        "memory": {
            "scratch_bytes": scratch_bytes,
            "scratch_mib": scratch_bytes / 2**20,
            "output_bytes": sorted_keys.numel() * sorted_keys.element_size(),
            "output_mib": (
                sorted_keys.numel() * sorted_keys.element_size() / 2**20
            ),
        },
        "candidate_cub_sort_keys": {
            "median_cuda_ms": candidate_median,
            "p95_cuda_ms": percentile(candidate_samples, 0.95),
            "samples_cuda_ms": candidate_samples,
        },
        "reference_torch_bf16_sort_values": {
            "median_cuda_ms": statistics.median(reference_samples),
            "p95_cuda_ms": percentile(reference_samples, 0.95),
            "samples_cuda_ms": reference_samples,
        },
        "gate": {
            "maximum_median_cuda_ms": gate_ms,
            "passed": exact_match and candidate_median <= gate_ms,
            "decision": (
                "continue fused exact CUDA cutoff prototype"
                if exact_match and candidate_median <= gate_ms
                else "stop exact CUB path at keys-only gate"
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))

    if not exact_match:
        raise SystemExit("CUB keys-only values did not match torch.sort")


if __name__ == "__main__":
    main()
