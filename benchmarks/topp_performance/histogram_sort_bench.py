#!/usr/bin/env python3
"""Development benchmark for the BF16 top-p counting-sort primitive."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch
import transformers

from topp_histogram import sort_bf16_scaled_values


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)]


def command_output(*command: str) -> str:
    return subprocess.check_output(command, text=True).strip()


def measure(invoke, warmups: int, iterations: int) -> dict[str, object]:
    for _ in range(warmups):
        output = invoke()
        del output
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(iterations):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        output = invoke()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end))
        del output
    return {
        "median_cuda_ms": statistics.median(samples),
        "p95_cuda_ms": percentile(samples, 0.95),
        "samples_cuda_ms": samples,
        "peak_incremental_allocated_mib": (
            torch.cuda.max_memory_allocated() - baseline
        )
        / 2**20,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--vocab", type=int, default=151_936)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.batch, args.vocab, args.iterations) < 1 or args.warmups < 0:
        parser.error("batch, vocab, and iterations must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
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

    reference = lambda: torch.sort(
        logits.float() / temperatures.unsqueeze(1), dim=-1
    ).values
    candidate = lambda: sort_bf16_scaled_values(logits, temperatures)
    expected = reference()
    actual = candidate()
    if not torch.equal(actual, expected):
        mismatch = int((actual != expected).sum().item())
        raise RuntimeError(f"candidate differs from reference in {mismatch} values")
    del expected, actual

    result = {
        "schema_version": 1,
        "benchmark": "topp_bf16_histogram_sort_development",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "commit": command_output("git", "rev-parse", "HEAD"),
        "git_status": command_output("git", "status", "--short"),
        "argv": sys.argv,
        "execution": {
            "cwd": str(Path.cwd()),
            "pythonpath": os.environ.get("PYTHONPATH"),
            "torchinductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "driver": command_output(
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ),
        },
        "configuration": {
            "batch": args.batch,
            "vocab": args.vocab,
            "dtype": "bfloat16",
            "temperature": args.temperature,
            "warmups": args.warmups,
            "iterations": args.iterations,
            "seed": args.seed,
        },
        "reference_torch_sort": measure(reference, args.warmups, args.iterations),
        "candidate_histogram_sort": measure(candidate, args.warmups, args.iterations),
        "exact_sorted_value_match": True,
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")


if __name__ == "__main__":
    main()
