#!/usr/bin/env python3
"""Fresh-process B=256 microbenchmark for repaired greedy and top-k routes."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
import transformers

from nanovllm.layers.sampler import Sampler


EXPECTED_COMMITS = {
    "greedy": "ec988708bbe1e4e84e97c3fc0599378c01920e3a",
    "topk": "8759c877382f11ea16b40fdbd0dace7000b5e9ba",
}
SCENARIOS = {
    "greedy": ("greedy_all",),
    "topk": ("disabled", "one_active_top_k_50", "all_active_top_k_50"),
}


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)]


def command_output(*command: str) -> str:
    return subprocess.check_output(command, text=True).strip()


def incremental_peak_mib(baseline_bytes: int) -> float:
    return (torch.cuda.max_memory_allocated() - baseline_bytes) / 2**20


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("feature", choices=tuple(SCENARIOS))
    parser.add_argument("scenario")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--vocab", type=int, default=151_936)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.scenario not in SCENARIOS[args.feature]:
        parser.error(
            f"scenario {args.scenario!r} is not valid for {args.feature}; "
            f"choose one of {SCENARIOS[args.feature]}"
        )
    if args.batch < 1 or args.vocab < 2 or args.top_k < 1:
        parser.error("batch, vocab, and top-k must be positive")
    if args.top_k >= args.vocab:
        parser.error("top-k must be smaller than vocab for an active route")
    if args.warmups < 0 or args.iterations < 1:
        parser.error("warmups must be non-negative and iterations must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    commit = command_output("git", "rev-parse", "HEAD")
    expected_commit = EXPECTED_COMMITS[args.feature]
    if commit != expected_commit:
        raise SystemExit(
            f"wrong checkout for {args.feature}: {commit}; expected {expected_commit}"
        )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    sampler = Sampler().cuda().eval()
    source_logits = torch.randn(
        args.batch,
        args.vocab,
        dtype=torch.bfloat16,
        device="cuda",
    )
    temperatures = torch.ones(args.batch, dtype=torch.float32, device="cuda")

    work_logits = None
    row_indices = None
    if args.feature == "topk" and args.scenario != "disabled":
        work_logits = source_logits.clone()
        if args.scenario == "one_active_top_k_50":
            row_indices = torch.zeros(1, dtype=torch.int64, device="cuda")

    def prepare() -> None:
        if work_logits is not None:
            work_logits.copy_(source_logits)

    def invoke() -> torch.Tensor:
        if args.feature == "greedy":
            return sampler.greedy(source_logits)
        if args.scenario == "disabled":
            return sampler(source_logits, temperatures)
        filtered = sampler.filter_top_k(work_logits, row_indices, args.top_k)
        return sampler(filtered, temperatures)

    prepare()
    torch.cuda.synchronize()
    cold_baseline_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    cold_begin = torch.cuda.Event(enable_timing=True)
    cold_end = torch.cuda.Event(enable_timing=True)
    cold_begin.record()
    cold_wall_started = time.perf_counter()
    output = invoke()
    cold_end.record()
    cold_end.synchronize()
    cold_wall_ms = (time.perf_counter() - cold_wall_started) * 1e3
    cold_cuda_ms = cold_begin.elapsed_time(cold_end)
    cold_peak_incremental_mib = incremental_peak_mib(cold_baseline_bytes)
    del output

    for _ in range(args.warmups):
        prepare()
        output = invoke()
        del output
    torch.cuda.synchronize()

    steady_baseline_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    samples_ms = []
    for _ in range(args.iterations):
        prepare()
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        output = invoke()
        end.record()
        end.synchronize()
        samples_ms.append(begin.elapsed_time(end))
        del output
    steady_peak_incremental_mib = incremental_peak_mib(steady_baseline_bytes)

    result = {
        "schema_version": 1,
        "benchmark": "repaired_sampling_microbenchmark",
        "feature": args.feature,
        "scenario": args.scenario,
        "commit": commit,
        "seed": args.seed,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "argv": sys.argv,
        "execution": {
            "cwd": str(Path.cwd()),
            "pythonpath": os.environ.get("PYTHONPATH"),
            "torchinductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "nvidia_driver": command_output(
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ),
        },
        "configuration": {
            "batch": args.batch,
            "vocab": args.vocab,
            "dtype": "bfloat16",
            "top_k": args.top_k if args.feature == "topk" else None,
            "warmups_after_cold": args.warmups,
            "steady_iterations": args.iterations,
            "percentile_method": "nearest_rank",
            "input_restore_outside_timed_region": work_logits is not None,
        },
        "cold": {
            "wall_ms": cold_wall_ms,
            "cuda_ms": cold_cuda_ms,
            "peak_incremental_allocated_mib": cold_peak_incremental_mib,
        },
        "steady": {
            "median_cuda_ms": statistics.median(samples_ms),
            "p95_cuda_ms": percentile(samples_ms, 0.95),
            "min_cuda_ms": min(samples_ms),
            "max_cuda_ms": max(samples_ms),
            "peak_incremental_allocated_mib": steady_peak_incremental_mib,
            "samples_cuda_ms": samples_ms,
        },
    }

    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n")


if __name__ == "__main__":
    main()
