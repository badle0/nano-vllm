#!/usr/bin/env python3
"""Benchmark and semantic characterization for the Qrita top-p prototype."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

import torch
import transformers

from qrita_topp_proto import (
    allocate_qrita_workspace,
    qrita_top_p_bf16_raw_,
)


ROOT = Path(__file__).resolve().parents[2]
SAMPLER_PATH = ROOT / "nanovllm/layers/sampler.py"
SAMPLER_SPEC = importlib.util.spec_from_file_location("qrita_bench_sampler", SAMPLER_PATH)
assert SAMPLER_SPEC is not None and SAMPLER_SPEC.loader is not None
SAMPLER_MODULE = importlib.util.module_from_spec(SAMPLER_SPEC)
SAMPLER_SPEC.loader.exec_module(SAMPLER_MODULE)
Sampler = SAMPLER_MODULE.Sampler


def percentile(samples: list[float], quantile: float) -> float:
    ordered = sorted(samples)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def time_in_place(
    source: torch.Tensor,
    work: torch.Tensor,
    operation: Callable[[], torch.Tensor],
    warmups: int,
    iterations: int,
) -> dict[str, object]:
    """Time only filtering; restore the common work buffer before each event."""

    for _ in range(warmups):
        work.copy_(source)
        operation()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    samples = []
    for _ in range(iterations):
        work.copy_(source)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return {
        "median_cuda_ms": statistics.median(samples),
        "p95_cuda_ms": percentile(samples, 0.95),
        "samples_cuda_ms": samples,
        "peak_transient_allocated_mib": (
            torch.cuda.max_memory_allocated() - baseline
        )
        / 2**20,
    }


def exact_filter(
    sampler: Sampler,
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    probability_cutoffs: torch.Tensor,
) -> torch.Tensor:
    return sampler.filter_top_p(
        logits,
        temperatures,
        row_indices=None,
        probability_cutoffs=probability_cutoffs,
    )


def support_delta(
    original: torch.Tensor,
    exact: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, object]:
    exact_support = torch.isfinite(exact)
    candidate_support = torch.isfinite(candidate)
    differing = exact_support != candidate_support
    rows_differing = differing.any(dim=1)
    exact_counts = exact_support.sum(dim=1).float()
    candidate_counts = candidate_support.sum(dim=1).float()
    return {
        "support_differing_elements": int(differing.sum().item()),
        "support_differing_rows": int(rows_differing.sum().item()),
        "candidate_extra_elements": int(
            (candidate_support & ~exact_support).sum().item()
        ),
        "candidate_missing_elements": int(
            (exact_support & ~candidate_support).sum().item()
        ),
        "exact_support_count_median": float(exact_counts.median().item()),
        "candidate_support_count_median": float(candidate_counts.median().item()),
        "exact_support_count_range": [
            int(exact_counts.min().item()),
            int(exact_counts.max().item()),
        ],
        "candidate_support_count_range": [
            int(candidate_counts.min().item()),
            int(candidate_counts.max().item()),
        ],
        "candidate_retained_raw_value_mismatches": int(
            (
                candidate_support
                & (candidate.view(torch.int16) != original.view(torch.int16))
            )
            .sum()
            .item()
        ),
        "exact_retained_raw_value_mismatches": int(
            (
                exact_support
                & (exact.view(torch.int16) != original.view(torch.int16))
            )
            .sum()
            .item()
        ),
    }


def eager_sample(
    filtered_logits: torch.Tensor,
    temperatures: torch.Tensor,
) -> torch.Tensor:
    scaled = filtered_logits.float().div(temperatures.unsqueeze(1))
    probabilities = torch.softmax(scaled, dim=-1)
    noise = torch.empty_like(probabilities).exponential_(1).clamp_min_(1e-10)
    return probabilities.div_(noise).argmax(dim=-1)


def token_delta(
    exact: torch.Tensor,
    candidate: torch.Tensor,
    temperatures: torch.Tensor,
    seeds: int,
) -> dict[str, object]:
    differing = 0
    total = exact.size(0) * seeds
    for seed_offset in range(seeds):
        seed = 20260900 + seed_offset
        torch.cuda.manual_seed(seed)
        exact_tokens = eager_sample(exact, temperatures)
        torch.cuda.manual_seed(seed)
        candidate_tokens = eager_sample(candidate, temperatures)
        differing += int((exact_tokens != candidate_tokens).sum().item())
    return {
        "seed_count": seeds,
        "token_draws": total,
        "differing_token_draws": differing,
        "difference_fraction": differing / total,
    }


def make_distribution(
    name: str,
    rows: int,
    columns: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    if name == "gaussian":
        return torch.randn(
            rows,
            columns,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
    if name == "forced_ties":
        values = torch.randint(
            -32,
            33,
            (rows, columns),
            dtype=torch.int32,
            device=device,
            generator=generator,
        )
        return values.to(torch.bfloat16).div_(8)
    if name == "peaked":
        logits = torch.randn(
            rows,
            columns,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ).mul_(0.25)
        peak_count = min(16, columns)
        peaks = torch.linspace(
            10.0,
            5.0,
            peak_count,
            dtype=torch.bfloat16,
            device=device,
        )
        logits[:, :peak_count] = peaks
        return logits
    if name == "uniform":
        return torch.zeros(
            rows,
            columns,
            dtype=torch.bfloat16,
            device=device,
        )
    raise ValueError(f"unknown distribution: {name}")


def characterize(
    rows: int,
    columns: int,
    temperature: float,
    top_p: float,
    token_seeds: int,
    seed: int,
) -> dict[str, object]:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    sampler = Sampler().cuda()
    results = {}
    for name in ("gaussian", "forced_ties", "peaked", "uniform"):
        original = make_distribution(name, rows, columns, device, generator)
        temperatures = torch.full(
            (rows,), temperature, dtype=torch.float32, device=device
        )
        top_ps = torch.full((rows,), top_p, dtype=torch.float32, device=device)
        cutoffs = torch.full(
            (rows,),
            1.0 - float(top_p),
            dtype=torch.float32,
            device=device,
        )
        workspace = allocate_qrita_workspace(original)

        exact = original.clone()
        candidate = original.clone()
        exact_filter(sampler, exact, temperatures, cutoffs)
        before_rng = torch.cuda.get_rng_state().clone()
        qrita_top_p_bf16_raw_(candidate, temperatures, top_ps, workspace)
        torch.cuda.synchronize()
        after_rng = torch.cuda.get_rng_state().clone()

        results[name] = {
            **support_delta(original, exact, candidate),
            "tokens": token_delta(
                exact, candidate, temperatures, token_seeds
            ),
            "candidate_filter_rng_neutral": bool(
                torch.equal(before_rng, after_rng)
            ),
        }
        del original, temperatures, top_ps, cutoffs, workspace, exact, candidate
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--vocab", type=int, default=151_936)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--characterization-batch", type=int, default=16)
    parser.add_argument("--characterization-vocab", type=int, default=8192)
    parser.add_argument("--token-seeds", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.top_p != 0.9:
        parser.error("this focused prototype supports only --top-p 0.9")
    if min(args.batch, args.vocab, args.iterations) <= 0 or args.warmups < 0:
        parser.error("batch, vocab, iterations must be positive; warmups nonnegative")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    source = torch.randn(
        args.batch,
        args.vocab,
        dtype=torch.bfloat16,
        device="cuda",
    )
    temperatures = torch.full(
        (args.batch,), args.temperature, dtype=torch.float32, device="cuda"
    )
    top_ps = torch.full(
        (args.batch,), args.top_p, dtype=torch.float32, device="cuda"
    )
    cutoffs = torch.full(
        (args.batch,),
        1.0 - float(args.top_p),
        dtype=torch.float32,
        device="cuda",
    )
    workspace = allocate_qrita_workspace(source)
    candidate_work = torch.empty_like(source)
    exact_work = torch.empty_like(source)
    sampler = Sampler().cuda()

    # Compile, then compare one production-shape output before timing.
    candidate_work.copy_(source)
    qrita_top_p_bf16_raw_(candidate_work, temperatures, top_ps, workspace)
    exact_work.copy_(source)
    exact_filter(sampler, exact_work, temperatures, cutoffs)
    torch.cuda.synchronize()
    production_delta = support_delta(source, exact_work, candidate_work)

    torch.cuda.manual_seed(args.seed + 1)
    before_rng = torch.cuda.get_rng_state().clone()
    candidate_work.copy_(source)
    qrita_top_p_bf16_raw_(candidate_work, temperatures, top_ps, workspace)
    torch.cuda.synchronize()
    after_rng = torch.cuda.get_rng_state().clone()
    rng_neutral = bool(torch.equal(before_rng, after_rng))

    candidate_timing = time_in_place(
        source,
        candidate_work,
        lambda: qrita_top_p_bf16_raw_(
            candidate_work, temperatures, top_ps, workspace
        ),
        args.warmups,
        args.iterations,
    )
    exact_timing = time_in_place(
        source,
        exact_work,
        lambda: exact_filter(sampler, exact_work, temperatures, cutoffs),
        args.warmups,
        args.iterations,
    )

    characterization = characterize(
        args.characterization_batch,
        args.characterization_vocab,
        args.temperature,
        args.top_p,
        args.token_seeds,
        args.seed + 100,
    )

    result = {
        "schema_version": 1,
        "benchmark": "qrita_topp_p90_bf16_raw_development",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "commit": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--short"),
        "argv": sys.argv,
        "execution": {
            "cwd": str(Path.cwd()),
            "pythonpath": os.environ.get("PYTHONPATH"),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "triton": triton_version(),
            "transformers": transformers.__version__,
            "cuda_build": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "driver": subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ],
                text=True,
            ).strip(),
        },
        "configuration": {
            "batch": args.batch,
            "vocab": args.vocab,
            "dtype": "bfloat16",
            "temperature": args.temperature,
            "top_p": args.top_p,
            "warmups": args.warmups,
            "iterations": args.iterations,
            "seed": args.seed,
            "characterization_batch": args.characterization_batch,
            "characterization_vocab": args.characterization_vocab,
            "token_seeds": args.token_seeds,
        },
        "memory": {
            "common_raw_work_buffer_mib": source.numel() * source.element_size() / 2**20,
            "candidate_persistent_workspace_mib": (
                workspace.numel() * workspace.element_size() / 2**20
            ),
            "exact_persistent_workspace_mib": 0.0,
        },
        "production_shape_semantics": production_delta,
        "candidate_filter_rng_neutral": rng_neutral,
        "candidate_qrita": candidate_timing,
        "reference_exact_full_sort": exact_timing,
        "speedup_exact_over_candidate": (
            exact_timing["median_cuda_ms"] / candidate_timing["median_cuda_ms"]
        ),
        "characterization": characterization,
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n")
    print(payload)


def triton_version() -> str:
    import triton

    return triton.__version__


if __name__ == "__main__":
    main()
