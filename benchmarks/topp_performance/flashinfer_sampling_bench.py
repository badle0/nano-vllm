#!/usr/bin/env python3
"""Development-only end-to-end FlashInfer top-p sampling benchmark.

This script compares the current exact Transformers-compatible filter plus
``Sampler.forward`` against both FlashInfer's raw sorting-free primitive and
nano-vLLM's production ``Sampler.sample_top_p_flashinfer`` wrapper.  It records
the complete shipped-route cost as the acceptance result, alongside the raw
primitive lower bound and the semantic/RNG-contract differences.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
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


ROOT = Path(__file__).resolve().parents[2]
SAMPLER_PATH = ROOT / "nanovllm/layers/sampler.py"
SAMPLER_SPEC = importlib.util.spec_from_file_location(
    "flashinfer_bench_sampler", SAMPLER_PATH
)
assert SAMPLER_SPEC is not None and SAMPLER_SPEC.loader is not None
SAMPLER_MODULE = importlib.util.module_from_spec(SAMPLER_SPEC)
SAMPLER_SPEC.loader.exec_module(SAMPLER_MODULE)
Sampler = SAMPLER_MODULE.Sampler

DEFAULT_FLASHINFER_PATH = Path("/tmp/nv_flashinfer_proto_nodeps")
FLASHINFER_PATH = Path(
    os.environ.get("NANOVLLM_FLASHINFER_PATH", DEFAULT_FLASHINFER_PATH)
).resolve()
if not (FLASHINFER_PATH / "flashinfer" / "sampling.py").is_file():
    raise RuntimeError(
        "isolated FlashInfer package not found; set NANOVLLM_FLASHINFER_PATH"
    )
sys.path.insert(0, str(FLASHINFER_PATH))

import flashinfer  # noqa: E402
from flashinfer import sampling as flashinfer_sampling  # noqa: E402
from flashinfer.jit import env as flashinfer_jit_env  # noqa: E402
from flashinfer.version import __git_commit__ as flashinfer_git_commit  # noqa: E402


Operation = Callable[[], torch.Tensor]
Prepare = Callable[[], None]


def percentile(samples: list[float], quantile: float) -> float:
    ordered = sorted(samples)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def driver_version() -> str:
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def measure_route(
    operation: Operation,
    warmups: int,
    iterations: int,
    prepare: Prepare | None = None,
) -> dict[str, object]:
    """Measure a complete route, including allocations performed by the route."""

    for _ in range(warmups):
        if prepare is not None:
            prepare()
        output = operation()
        del output
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    samples = []
    for _ in range(iterations):
        if prepare is not None:
            prepare()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = operation()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
        del output
    peak = torch.cuda.max_memory_allocated()
    return {
        "median_cuda_ms": statistics.median(samples),
        "p95_cuda_ms": percentile(samples, 0.95),
        "samples_cuda_ms": samples,
        "baseline_allocated_mib": baseline / 2**20,
        "peak_transient_allocated_mib": (peak - baseline) / 2**20,
    }


def cuda_rng_state() -> torch.Tensor:
    return torch.cuda.get_rng_state().clone()


def describe_rng_state(state: torch.Tensor) -> dict[str, object]:
    cpu_state = state.cpu().contiguous()
    description: dict[str, object] = {
        "num_bytes": cpu_state.numel(),
        "hex": bytes(cpu_state.tolist()).hex(),
    }
    if cpu_state.numel() % 8 == 0:
        description["int64_words"] = cpu_state.view(torch.int64).tolist()
    return description


def seeded_call(
    operation: Operation,
    seed: int,
    prepare: Prepare | None = None,
) -> dict[str, object]:
    torch.cuda.manual_seed(seed)
    if prepare is not None:
        prepare()
    before = cuda_rng_state()
    tokens = operation()
    torch.cuda.synchronize()
    after = cuda_rng_state()
    return {
        "tokens": tokens.detach().clone(),
        "before": before,
        "after": after,
    }


def exact_filter(
    sampler: Sampler,
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    cutoffs: torch.Tensor,
) -> torch.Tensor:
    return sampler.filter_top_p(
        logits,
        temperatures,
        row_indices=None,
        probability_cutoffs=cutoffs,
    )


def repeated_flashinfer_draws(
    probabilities: torch.Tensor,
    top_p: float,
    draws: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if probabilities.shape[0] != 1:
        raise ValueError("repeated draw helper requires one probability row")
    indices = torch.zeros(draws, dtype=torch.int32, device=probabilities.device)
    torch.cuda.manual_seed(seed)
    samples, valid = flashinfer_sampling.top_p_sampling_from_probs(
        probabilities,
        top_p,
        indices=indices,
        deterministic=True,
        return_valid=True,
    )
    torch.cuda.synchronize()
    return samples, valid


def frequency_report(
    samples: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, object]:
    vocab = expected.numel()
    counts = torch.bincount(samples.to(torch.int64), minlength=vocab).cpu()
    frequencies = counts.double() / samples.numel()
    expected_cpu = expected.detach().double().cpu()
    positive = expected_cpu > 0
    variances = samples.numel() * expected_cpu * (1.0 - expected_cpu)
    residuals = torch.zeros_like(expected_cpu)
    residuals[positive] = (
        counts.double()[positive] - samples.numel() * expected_cpu[positive]
    ) / variances[positive].sqrt()
    return {
        "draws": samples.numel(),
        "counts": counts.tolist(),
        "frequencies": frequencies.tolist(),
        "expected_probabilities": expected_cpu.tolist(),
        "observed_token_ids": torch.nonzero(counts, as_tuple=False)
        .flatten()
        .tolist(),
        "draws_outside_expected_support": int(counts[~positive].sum().item()),
        "max_abs_frequency_error": float(
            (frequencies - expected_cpu).abs().max().item()
        ),
        "max_abs_standardized_residual_on_support": float(
            residuals[positive].abs().max().item()
        ),
    }


def characterize_distribution(
    name: str,
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    draws: int,
    seed: int,
    expected_candidate_support: torch.Tensor | None = None,
) -> dict[str, object]:
    sampler = Sampler().cuda()
    row = logits.to(device="cuda", dtype=torch.float32).unsqueeze(0)
    temperatures = torch.tensor([temperature], dtype=torch.float32, device="cuda")
    cutoffs = torch.tensor(
        [1.0 - float(top_p)], dtype=torch.float32, device="cuda"
    )
    exact = row.clone()
    exact_filter(sampler, exact, temperatures, cutoffs)
    exact_support = torch.isfinite(exact[0])
    probabilities = flashinfer_sampling.softmax(row, temperature=temperatures)

    samples, valid = repeated_flashinfer_draws(
        probabilities, top_p, draws, seed
    )
    if expected_candidate_support is None:
        expected_candidate_support = exact_support
    expected_candidate_support = expected_candidate_support.to(
        device="cuda", dtype=torch.bool
    )
    expected = probabilities[0].double() * expected_candidate_support
    expected.div_(expected.sum())
    report = frequency_report(samples, expected)
    report.update(
        {
            "name": name,
            "temperature": temperature,
            "top_p": top_p,
            "logits": row[0].cpu().tolist(),
            "softmax_probabilities": probabilities[0].cpu().tolist(),
            "softmax_sum": float(probabilities[0].sum().item()),
            "exact_support_token_ids": torch.nonzero(
                exact_support, as_tuple=False
            )
            .flatten()
            .cpu()
            .tolist(),
            "expected_candidate_support_token_ids": torch.nonzero(
                expected_candidate_support, as_tuple=False
            )
            .flatten()
            .cpu()
            .tolist(),
            "all_flashinfer_rows_valid": bool(valid.all().item()),
            "invalid_flashinfer_rows": int((~valid).sum().item()),
        }
    )
    return report


def characterize_semantics(draws: int, seed: int) -> dict[str, object]:
    known_probabilities = torch.tensor(
        [0.41, 0.24, 0.16, 0.09, 0.055, 0.025, 0.013, 0.007],
        dtype=torch.float64,
    )
    known_logits = known_probabilities.log().float()
    known_support = torch.tensor(
        [True, True, True, False, False, False, False, False]
    )
    known = characterize_distribution(
        "known_unique_logits",
        known_logits,
        temperature=1.0,
        top_p=0.8,
        draws=draws,
        seed=seed,
        expected_candidate_support=known_support,
    )
    known["designed_probabilities"] = known_probabilities.tolist()
    known["sanity_pass"] = bool(
        known["exact_support_token_ids"] == [0, 1, 2]
        and known["draws_outside_expected_support"] == 0
        and known["all_flashinfer_rows_valid"]
        and known["max_abs_standardized_residual_on_support"] <= 6.0
    )

    uniform_logits = torch.zeros(8, dtype=torch.float32)
    uniform = characterize_distribution(
        "uniform_tie",
        uniform_logits,
        temperature=1.0,
        top_p=0.6,
        draws=draws,
        seed=seed + 1,
        expected_candidate_support=torch.ones(8, dtype=torch.bool),
    )
    uniform["interpretation"] = (
        "The current torch.sort(stable=False) full-sort route selects token "
        "identities at the tie boundary according to its pinned CUDA ordering; "
        "the candidate is characterized against tie-symmetric support."
    )

    tied_logits = torch.tensor(
        [4.0, 4.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=torch.float32,
    )
    tied_support = torch.tensor(
        [True, True, True, False, False, False, False, False]
    )
    forced_tie = characterize_distribution(
        "forced_top_tie",
        tied_logits,
        temperature=1.0,
        top_p=0.5,
        draws=draws,
        seed=seed + 2,
        expected_candidate_support=tied_support,
    )
    forced_tie["interpretation"] = (
        "The candidate is characterized against all equal boundary maxima; "
        "the current torch.sort(stable=False) support may keep only a token-ID "
        "subset according to its pinned CUDA ordering."
    )
    return {
        "draws_per_distribution": draws,
        "known_unique_logits": known,
        "uniform_tie": uniform,
        "forced_top_tie": forced_tie,
    }


def validate_heterogeneous_temperature_api() -> dict[str, object]:
    logits = torch.tensor(
        [
            [1.5, 0.2, -0.3, -1.0],
            [0.7, 0.1, -0.4, -1.2],
            [2.0, 1.0, 0.0, -2.0],
            [0.8, 0.6, 0.4, 0.2],
        ],
        dtype=torch.bfloat16,
        device="cuda",
    )
    temperatures = torch.tensor(
        [0.5, 0.6, 0.8, 1.0], dtype=torch.float32, device="cuda"
    )
    actual = flashinfer_sampling.softmax(logits, temperature=temperatures)
    expected = torch.softmax(logits.float() / temperatures[:, None], dim=-1)
    torch.cuda.synchronize()
    return {
        "temperatures": temperatures.cpu().tolist(),
        "max_abs_error_vs_torch": float((actual - expected).abs().max().item()),
        "row_sum_max_abs_error": float(
            (actual.sum(dim=-1) - 1.0).abs().max().item()
        ),
        "all_finite": bool(torch.isfinite(actual).all().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--vocab", type=int, default=151_936)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--statistical-draws", type=int, default=131_072)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if min(args.batch, args.vocab, args.iterations, args.statistical_draws) <= 0:
        parser.error("batch, vocab, iterations, and draws must be positive")
    if args.warmups < 0:
        parser.error("warmups must be nonnegative")
    if not (0.0 < args.top_p <= 1.0) or args.temperature <= 0.0:
        parser.error("top-p must be in (0, 1] and temperature must be positive")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing evidence: {args.output}")
    if args.output.resolve().is_relative_to(ROOT):
        raise SystemExit("raw evidence output must be outside the git checkout")
    commit = git_output("rev-parse", "HEAD")
    if commit != args.expected_commit:
        raise SystemExit(
            f"wrong checkout: expected {args.expected_commit}, observed {commit}"
        )
    git_status = git_output("status", "--short")
    if git_status:
        raise SystemExit(f"benchmark requires a clean checkout:\n{git_status}")

    started_at = datetime.now(UTC)
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
    cutoffs = torch.full(
        (args.batch,),
        1.0 - float(args.top_p),
        dtype=torch.float32,
        device="cuda",
    )
    top_ps = torch.full(
        (args.batch,), args.top_p, dtype=torch.float32, device="cuda"
    )
    exact_work = torch.empty_like(source)
    sampler = Sampler().cuda()

    def exact_prepare() -> None:
        # Repeated measurements restore the work buffer outside the timed
        # region. Production receives fresh model logits and needs no copy.
        exact_work.copy_(source)

    def exact_complete() -> torch.Tensor:
        exact_filter(sampler, exact_work, temperatures, cutoffs)
        return sampler(exact_work, temperatures)

    def flashinfer_direct() -> torch.Tensor:
        probabilities = flashinfer_sampling.softmax(
            source, temperature=temperatures
        )
        return flashinfer_sampling.top_p_sampling_from_probs(
            probabilities,
            top_ps,
            deterministic=True,
        )

    def flashinfer_production() -> torch.Tensor:
        return sampler.sample_top_p_flashinfer(
            source,
            temperatures,
            top_ps,
        )

    # Force both torch.compile and FlashInfer JIT/cache initialization outside
    # all measurements and semantic RNG checks.
    exact_prepare()
    exact_warm = exact_complete()
    direct_warm = flashinfer_direct()
    production_warm = flashinfer_production()
    torch.cuda.synchronize()
    if not (
        exact_warm.shape
        == direct_warm.shape
        == production_warm.shape
        == (args.batch,)
    ):
        raise RuntimeError("sampling routes returned unexpected output shapes")
    del exact_warm, direct_warm, production_warm

    api_validation = validate_heterogeneous_temperature_api()
    if not api_validation["all_finite"] or api_validation["row_sum_max_abs_error"] > 1e-5:
        raise RuntimeError("FlashInfer heterogeneous-temperature softmax validation failed")

    deterministic_seed = args.seed + 1
    exact_first = seeded_call(
        exact_complete, deterministic_seed, prepare=exact_prepare
    )
    exact_second = seeded_call(
        exact_complete, deterministic_seed, prepare=exact_prepare
    )
    candidate_first = seeded_call(flashinfer_production, deterministic_seed)
    candidate_second = seeded_call(flashinfer_production, deterministic_seed)

    deterministic_report = {
        "seed": deterministic_seed,
        "exact_same_seed_tokens_repeat": bool(
            torch.equal(exact_first["tokens"], exact_second["tokens"])
        ),
        "exact_same_seed_rng_state_repeat": bool(
            torch.equal(exact_first["after"], exact_second["after"])
        ),
        "flashinfer_same_seed_tokens_repeat": bool(
            torch.equal(candidate_first["tokens"], candidate_second["tokens"])
        ),
        "flashinfer_same_seed_rng_state_repeat": bool(
            torch.equal(candidate_first["after"], candidate_second["after"])
        ),
        "fixed_seed_differing_tokens": int(
            (exact_first["tokens"] != candidate_first["tokens"]).sum().item()
        ),
        "fixed_seed_token_comparisons": args.batch,
        "fixed_seed_difference_fraction": float(
            (exact_first["tokens"] != candidate_first["tokens"])
            .float()
            .mean()
            .item()
        ),
        "common_rng_state_before": bool(
            torch.equal(exact_first["before"], candidate_first["before"])
        ),
        "same_rng_state_after_routes": bool(
            torch.equal(exact_first["after"], candidate_first["after"])
        ),
        "exact_rng_state_advanced": not torch.equal(
            exact_first["before"], exact_first["after"]
        ),
        "flashinfer_rng_state_advanced": not torch.equal(
            candidate_first["before"], candidate_first["after"]
        ),
        "rng_states": {
            "before": describe_rng_state(exact_first["before"]),
            "after_exact": describe_rng_state(exact_first["after"]),
            "after_flashinfer": describe_rng_state(candidate_first["after"]),
        },
    }

    exact_timing = measure_route(
        exact_complete,
        args.warmups,
        args.iterations,
        prepare=exact_prepare,
    )
    flashinfer_direct_timing = measure_route(
        flashinfer_direct, args.warmups, args.iterations
    )
    flashinfer_production_timing = measure_route(
        flashinfer_production, args.warmups, args.iterations
    )
    semantics = characterize_semantics(
        args.statistical_draws, args.seed + 100
    )

    finished_at = datetime.now(UTC)
    result = {
        "schema_version": 1,
        "benchmark": "flashinfer_complete_topp_sampling_development",
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "commit": commit,
        "branch": git_output("branch", "--show-current"),
        "git_status": git_status,
        "argv": sys.argv,
        "execution": {
            "cwd": str(Path.cwd()),
            "pythonpath": os.environ.get("PYTHONPATH"),
            "flashinfer_workspace_base": os.environ.get(
                "FLASHINFER_WORKSPACE_BASE"
            ),
            "flashinfer_resolved_cache_dir": str(
                flashinfer_jit_env.FLASHINFER_CACHE_DIR
            ),
            "flashinfer_resolved_workspace_dir": str(
                flashinfer_jit_env.FLASHINFER_WORKSPACE_DIR
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_build": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "driver": driver_version(),
        },
        "flashinfer_provenance": {
            "version": flashinfer.__version__,
            "git_commit": flashinfer_git_commit,
            "module_file": str(Path(flashinfer.__file__).resolve()),
            "isolated_package_root": str(FLASHINFER_PATH),
            "project": "https://github.com/flashinfer-ai/flashinfer",
            "license": "Apache-2.0",
            "api": [
                "flashinfer.sampling.softmax",
                "flashinfer.sampling.top_p_sampling_from_probs",
            ],
            "top_p_deterministic_argument": True,
        },
        "source": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": file_sha256(Path(__file__).resolve()),
            "origin_url": git_output("remote", "get-url", "origin"),
        },
        "configuration": {
            "batch": args.batch,
            "vocab": args.vocab,
            "dtype": "bfloat16",
            "temperature": args.temperature,
            "temperature_argument_kind": "per-row float32 tensor",
            "top_p": args.top_p,
            "top_p_argument_kind": "per-row float32 tensor",
            "warmups": args.warmups,
            "iterations": args.iterations,
            "statistical_draws": args.statistical_draws,
            "seed": args.seed,
        },
        "measurement_contract": {
            "exact_complete_route": (
                "exact filter_top_p; existing compiled Sampler.forward"
            ),
            "flashinfer_direct_route": (
                "FlashInfer FP32 softmax with per-row temperature; "
                "deterministic top_p_sampling_from_probs"
            ),
            "flashinfer_production_route": (
                "Sampler.sample_top_p_flashinfer, including greedy argmax, "
                "FlashInfer FP32 softmax and deterministic top-p sample, "
                "sample dtype normalization, and greedy torch.where"
            ),
            "exact_restore_in_timed_region": False,
            "exact_restore_before_each_measurement": True,
            "restore_rationale": (
                "restore is repeated-measurement setup only; production receives "
                "fresh model logits and does not clone them"
            ),
            "flashinfer_source_is_not_mutated": True,
            "jit_and_torch_compile_warmed_before_measurement": True,
            "cuda_events_on_current_stream": True,
            "transient_memory_definition": (
                "max_memory_allocated minus memory_allocated after warmup/GC"
            ),
        },
        "persistent_inputs": {
            "source_bf16_mib": source.numel() * source.element_size() / 2**20,
            "exact_restore_buffer_bf16_mib": (
                exact_work.numel() * exact_work.element_size() / 2**20
            ),
        },
        "heterogeneous_temperature_api_validation": api_validation,
        "determinism_and_rng_contract": deterministic_report,
        "current_exact_complete": exact_timing,
        "flashinfer_direct_primitive": flashinfer_direct_timing,
        "flashinfer_production_complete": flashinfer_production_timing,
        "speedup_exact_over_flashinfer_production": (
            exact_timing["median_cuda_ms"]
            / flashinfer_production_timing["median_cuda_ms"]
        ),
        "statistical_semantic_characterization": semantics,
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
