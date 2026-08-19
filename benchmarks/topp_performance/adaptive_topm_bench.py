#!/usr/bin/env python3
"""Bounded characterization of an experimental adaptive top-M top-p route."""

from __future__ import annotations

import json
import math
import statistics
import sys

import torch

from nanovllm.layers.sampler import Sampler


BATCH = 256
VOCAB = 151_936
TEMPERATURE = 0.6
TOP_P = 0.9
CUTOFF = 1.0 - TOP_P
WARMUPS = 3
ITERATIONS = 10


@torch.inference_mode()
def exact_filter(logits: torch.Tensor, sampler: Sampler) -> torch.Tensor:
    temperatures = torch.full(
        (logits.size(0),), TEMPERATURE, device=logits.device
    )
    cutoffs = torch.full((logits.size(0),), CUTOFF, device=logits.device)
    return sampler.filter_top_p(logits.clone(), temperatures, None, cutoffs)


@torch.inference_mode()
def adaptive_filter(
    logits: torch.Tensor, sampler: Sampler, candidate_count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    # Rank in BF16 (positive row-wise temperature preserves order), but evaluate
    # full-vocabulary mass and candidate probabilities in FP32.
    values, indices = torch.topk(
        logits, candidate_count, dim=-1, largest=True, sorted=True
    )
    temperatures = torch.full(
        (logits.size(0),), TEMPERATURE, device=logits.device
    )
    log_norm = torch.logsumexp(
        logits.float() / temperatures.unsqueeze(1), dim=-1
    )
    candidate_probs = torch.exp(
        values.float() / temperatures.unsqueeze(1) - log_norm.unsqueeze(1)
    )
    candidate_mass = candidate_probs.sum(dim=-1)
    tail_mass = 1.0 - candidate_mass
    cumulative = tail_mass.unsqueeze(1) + torch.flip(
        torch.cumsum(torch.flip(candidate_probs, dims=(1,)), dim=1), dims=(1,)
    )
    remove = cumulative <= CUTOFF
    remove[:, 0] = False

    # A top-M boundary tie means topk may have omitted equal-valued candidates.
    complete_boundary = (
        (logits >= values[:, -1:].contiguous()).sum(dim=-1) == candidate_count
    )
    # If the cutoff splits an equal-score group, topk's unstable tie order need
    # not match torch.sort's unstable tie order. Fall back conservatively.
    kept = ~remove
    kept_count = kept.sum(dim=-1)
    lowest_kept_slot = (kept_count - 1).clamp_min(0).unsqueeze(1)
    cutoff_value = values.gather(1, lowest_kept_slot)
    equal_cutoff = values == cutoff_value
    equal_kept = (equal_cutoff & kept).sum(dim=-1)
    equal_total = equal_cutoff.sum(dim=-1)
    no_split_cutoff_tie = (equal_kept == 0) | (equal_kept == equal_total)
    # The 1e-6 guard deliberately rejects numerically marginal mass decisions.
    boundary_tie_is_irrelevant = remove[:, -1] | complete_boundary
    fast = (
        (candidate_mass >= TOP_P + 1e-6)
        & boundary_tie_is_irrelevant
        & no_split_cutoff_tie
    )

    output = torch.full_like(logits, float("-inf"))
    output.scatter_(1, indices, torch.where(kept, values, float("-inf")))
    failed_rows = (~fast).nonzero(as_tuple=False).flatten()
    failed_logits = logits.index_select(0, failed_rows)
    failed_temperatures = temperatures.index_select(0, failed_rows)
    failed_cutoffs = torch.full(
        (failed_rows.numel(),), CUTOFF, device=logits.device
    )
    exact_failed = sampler.filter_top_p(
        failed_logits.clone(), failed_temperatures, None, failed_cutoffs
    )
    output.index_copy_(0, failed_rows, exact_failed)
    return output, fast


def percentile(samples: list[float], q: float) -> float:
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)]


def measure(invoke) -> dict[str, object]:
    for _ in range(WARMUPS):
        output = invoke()
        del output
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(ITERATIONS):
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
        "peak_incremental_allocated_mib": (
            torch.cuda.max_memory_allocated() - baseline
        ) / 2**20,
        "samples_cuda_ms": samples,
    }


def make_logits(kind: str, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(
        BATCH, VOCAB, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    if kind == "peaked_model_like":
        # Preserve a broad model-like tail but give each row a sparse semantic head.
        head = torch.randn(
            BATCH, 512, device="cuda", dtype=torch.bfloat16, generator=generator
        ).mul_(1.5).add_(8.0)
        slots = torch.arange(512, device="cuda").expand(BATCH, -1)
        logits.scatter_(1, slots, head)
    return logits


def assess(logits: torch.Tensor, sampler: Sampler, candidate_count: int) -> dict:
    expected = exact_filter(logits, sampler)
    actual, fast = adaptive_filter(logits, sampler, candidate_count)
    expected_support = torch.isfinite(expected)
    actual_support = torch.isfinite(actual)
    row_mismatch = (expected_support != actual_support).any(dim=-1)
    fast_mismatch = row_mismatch & fast

    token_mismatches = 0
    rng_mismatches = 0
    temperatures = torch.full((BATCH,), TEMPERATURE, device="cuda")
    for seed in (101, 202, 303, 404, 505, 606, 707, 808):
        torch.cuda.manual_seed(seed)
        expected_tokens = sampler(expected, temperatures)
        expected_state = torch.cuda.get_rng_state().clone()
        torch.cuda.manual_seed(seed)
        actual_tokens = sampler(actual, temperatures)
        actual_state = torch.cuda.get_rng_state().clone()
        token_mismatches += int((expected_tokens != actual_tokens).sum().item())
        rng_mismatches += int(not torch.equal(expected_state, actual_state))

    result = {
        "fast_rows": int(fast.sum().item()),
        "fallback_rows": int((~fast).sum().item()),
        "support_mismatch_rows": int(row_mismatch.sum().item()),
        "support_mismatch_elements": int(
            (expected_support != actual_support).sum().item()
        ),
        "false_certified_rows": int(fast_mismatch.sum().item()),
        "retained_value_mismatch_elements": int(
            ((actual != expected) & expected_support & actual_support).sum().item()
        ),
        "sampled_token_mismatches_8_seeds": token_mismatches,
        "cuda_rng_state_mismatches_8_seeds": rng_mismatches,
    }
    del expected, actual, expected_support, actual_support
    torch.cuda.empty_cache()
    result["exact_full_route"] = measure(lambda: exact_filter(logits, sampler))
    result["adaptive_full_route"] = measure(
        lambda: adaptive_filter(logits, sampler, candidate_count)
    )
    return result


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(20260825)
    torch.cuda.manual_seed_all(20260825)
    sampler = Sampler().cuda().eval()
    # Pay torch.compile's sampling cost before correctness/timing.
    sampler(torch.zeros((1, 32), device="cuda"), torch.ones(1, device="cuda"))
    result = {
        "configuration": {
            "batch": BATCH,
            "vocab": VOCAB,
            "dtype": "bfloat16",
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "warmups": WARMUPS,
            "iterations": ITERATIONS,
            "candidate_counts": [4096, 8192],
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
        },
        "scenarios": {},
    }
    for scenario_index, kind in enumerate(("gaussian", "peaked_model_like")):
        logits = make_logits(kind, 20260825 + scenario_index)
        result["scenarios"][kind] = {}
        for candidate_count in (4096, 8192):
            result["scenarios"][kind][str(candidate_count)] = assess(
                logits, sampler, candidate_count
            )
        del logits
        torch.cuda.empty_cache()
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    print()


if __name__ == "__main__":
    main()
