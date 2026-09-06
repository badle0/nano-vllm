#!/usr/bin/env python3
"""Self-pinned full-completion chunk-tail latency evidence.

This reproduces the retained 16-short/2-long request-metrics protocol while
exercising the engine-owned Python-GC lease.  It observes the public metrics
path and cheap engine-level counters, but does not wrap the timed model path.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import statistics
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from benchmarks.chunked_prefill_tail.common import (
    ROOT,
    base_result,
    environment_identity,
    handle_pin_query,
    immutable_write_json,
    model_identity,
    validate_release_pin,
)


SCHEMA_VERSION = 1
KIND = "chunked_prefill_full_completion_run"
PROTOCOL = "full_completion_request_metrics_engine_gc_v1"
SUPPORTED_TAUS = (256, 512)
DEFAULT_MODEL = "/workspace/models/Qwen3-0.6B"
MAX_ITL_SLO_MS = 10.0


def certification_workload(tau: int) -> dict[str, object]:
    if tau not in SUPPORTED_TAUS:
        raise ValueError(f"tau must be one of {SUPPORTED_TAUS}")
    return {
        "tau": tau,
        "warmup_requests": 16,
        "warmup_prompt_tokens": 64,
        "warmup_completion_tokens": 4,
        "interactive_requests": 16,
        "interactive_prompt_tokens": 64,
        "interactive_steps_before_long_admission": 40,
        "long_requests": 2,
        "long_prompt_tokens": 2048,
        "max_completion_tokens_per_request": 256,
        "temperature": 0.6,
        "ignore_eos": True,
        "top_k": -1,
        "top_p": 1.0,
        "prompt_token_min_inclusive": 100,
        "prompt_token_max_exclusive": 10_000,
        "max_model_len": 4096,
        # The accepted fixed historical harness pinned min(512, tau), which
        # also satisfies Config's max_num_batched_tokens >= max_num_seqs gate.
        "max_num_seqs": min(512, tau),
        "gpu_memory_utilization": 0.8,
        "enforce_eager": False,
        "tensor_parallel_size": 1,
        "disable_python_gc": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retain one full-completion chunk-tail certification run."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tau", type=int, choices=SUPPORTED_TAUS, default=256)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--expected-commit")
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--print-source-sha256", action="store_true")
    parser.add_argument("--show-pin", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    missing = [
        name
        for name in ("seed", "expected_commit", "expected_source_sha256", "output")
        if getattr(args, name) is None
    ]
    if missing:
        raise ValueError(
            "runtime certification requires "
            + ", ".join("--" + name.replace("_", "-") for name in missing)
        )
    if not 0 <= args.seed < 2**63:
        raise ValueError("--seed must be in [0, 2**63)")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite retained output: {args.output}")
    try:
        args.output.expanduser().resolve().relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise ValueError("--output must be outside the pinned source worktree")
    model_root = Path(args.model).expanduser().resolve(strict=True)
    try:
        args.output.expanduser().resolve().relative_to(model_root)
    except ValueError:
        pass
    else:
        raise ValueError("--output must be outside the pinned model directory")


def _token_vector_sha256(token_ids: list[int]) -> str:
    encoded = json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prompt_manifest(label: str, prompts: list[list[int]]) -> dict[str, object]:
    aggregate = hashlib.sha256()
    rows = []
    for index, prompt in enumerate(prompts):
        digest = _token_vector_sha256(prompt)
        aggregate.update(index.to_bytes(8, "big"))
        aggregate.update(len(prompt).to_bytes(8, "big"))
        aggregate.update(bytes.fromhex(digest))
        rows.append({"index": index, "tokens": len(prompt), "sha256": digest})
    return {
        "label": label,
        "count": len(rows),
        "aggregate_sha256": aggregate.hexdigest(),
        "prompts": rows,
    }


def summarize_requests(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("cannot summarize an empty request cohort")

    def values(name: str) -> list[float]:
        return [float(row["metrics"][name]) * 1000.0 for row in rows]

    ttft = values("engine_ttft")
    mean_itl = values("engine_mean_itl")
    max_itl = values("engine_max_itl")
    return {
        "count": len(rows),
        "engine_ttft_p50_ms": statistics.median(ttft),
        "engine_ttft_max_ms": max(ttft),
        "engine_mean_itl_p50_ms": statistics.median(mean_itl),
        "engine_mean_itl_max_ms": max(mean_itl),
        "engine_max_itl_p50_ms": statistics.median(max_itl),
        "engine_max_itl_max_ms": max(max_itl),
        "prompt_tokens": sum(int(row["metrics"]["num_prompt_tokens"]) for row in rows),
        "completion_tokens": sum(
            int(row["metrics"]["num_completion_tokens"]) for row in rows
        ),
    }


def _record_finished_requests(
    outputs,
    cohort_by_id: dict[int, str],
    collected: dict[int, dict[str, object]],
) -> None:
    required_metrics = {
        "engine_queue_time",
        "engine_ttft",
        "engine_e2e",
        "engine_mean_itl",
        "engine_max_itl",
        "engine_itls",
        "submission_to_engine",
        "submission_to_first_token",
        "submission_to_engine_finish",
        "num_prompt_tokens",
        "num_completion_tokens",
    }
    for seq_id, token_ids, metrics in outputs:
        if seq_id not in cohort_by_id:
            raise RuntimeError(f"completed unknown measured sequence {seq_id}")
        if seq_id in collected:
            raise RuntimeError(f"sequence {seq_id} completed more than once")
        missing = required_metrics - metrics.keys()
        if missing:
            raise RuntimeError(
                f"sequence {seq_id} metrics missing fields: {sorted(missing)}"
            )
        if len(token_ids) != metrics["num_completion_tokens"]:
            raise RuntimeError(f"sequence {seq_id} token/metric count mismatch")
        collected[seq_id] = {
            "seq_id": seq_id,
            "cohort": cohort_by_id[seq_id],
            "completion_token_ids": list(token_ids),
            "completion_token_ids_sha256": _token_vector_sha256(list(token_ids)),
            "metrics": metrics,
        }


def _run_workload(
    llm,
    sampling_params_type,
    torch_module,
    seed: int,
    *,
    clock: Callable[[], float] = perf_counter,
) -> dict[str, object]:
    """Run the exact protocol; dependencies are injectable for CPU fake tests."""
    spec = certification_workload(llm.model_runner.config.max_num_batched_tokens)
    runner = llm.model_runner
    scheduler = llm.scheduler
    block_manager = scheduler.block_manager
    vocab_size = int(runner.config.hf_config.vocab_size)
    if vocab_size < spec["prompt_token_max_exclusive"]:
        raise ValueError(
            "model vocabulary cannot represent the pinned prompt range: "
            f"{vocab_size} < {spec['prompt_token_max_exclusive']}"
        )

    rng = random.Random(seed)

    def prompts(count: int, length: int) -> list[list[int]]:
        return [
            [
                rng.randrange(
                    spec["prompt_token_min_inclusive"],
                    spec["prompt_token_max_exclusive"],
                )
                for _ in range(length)
            ]
            for _ in range(count)
        ]

    warmup_prompts = prompts(spec["warmup_requests"], spec["warmup_prompt_tokens"])
    warmup_outputs = llm.generate(
        warmup_prompts,
        sampling_params_type(
            ignore_eos=True,
            max_tokens=spec["warmup_completion_tokens"],
        ),
        use_tqdm=False,
    )
    if len(warmup_outputs) != spec["warmup_requests"]:
        raise RuntimeError("warmup did not complete the pinned request count")
    if any(
        len(output["token_ids"]) != spec["warmup_completion_tokens"]
        for output in warmup_outputs
    ):
        raise RuntimeError("warmup did not produce the pinned completion length")

    interactive_prompts = prompts(
        spec["interactive_requests"], spec["interactive_prompt_tokens"]
    )
    long_prompts = prompts(spec["long_requests"], spec["long_prompt_tokens"])
    params = sampling_params_type(
        temperature=spec["temperature"],
        max_tokens=spec["max_completion_tokens_per_request"],
        ignore_eos=spec["ignore_eos"],
        top_k=spec["top_k"],
        top_p=spec["top_p"],
    )

    torch_module.manual_seed(seed)
    torch_module.cuda.manual_seed_all(seed)
    torch_module.cuda.synchronize()
    torch_module.cuda.reset_peak_memory_stats()
    allocated_before = int(torch_module.cuda.memory_allocated())
    reserved_before = int(torch_module.cuda.memory_reserved())
    free_blocks_before = len(block_manager.free_block_ids)
    used_blocks_before = len(block_manager.used_block_ids)
    misses_before = int(getattr(runner, "varlen_miss", 0))

    completed_outputs = {}

    def drain(outputs) -> None:
        # Match the accepted fixed historical harness's per-step work: one
        # loop and one dict assignment only when a request completes.
        for output in outputs:
            completed_outputs[output[0]] = output

    started = clock()
    for prompt in interactive_prompts:
        llm.add_request(prompt, params)

    for _ in range(spec["interactive_steps_before_long_admission"]):
        outputs, _ = llm.step_with_metrics()
        drain(outputs)

    long_admitted = clock()
    for prompt in long_prompts:
        llm.add_request(prompt, params)
    while not llm.is_finished():
        outputs, _ = llm.step_with_metrics()
        drain(outputs)

    torch_module.cuda.synchronize()
    ended = clock()
    cohort_by_id = {
        seq_id: (
            "interactive" if output[2]["num_prompt_tokens"] == 64 else "long"
        )
        for seq_id, output in completed_outputs.items()
    }
    collected: dict[int, dict[str, object]] = {}
    _record_finished_requests(completed_outputs.values(), cohort_by_id, collected)
    if len(collected) != spec["interactive_requests"] + spec["long_requests"]:
        raise RuntimeError(
            f"expected 18 completed measured requests, found {len(collected)}"
        )
    requests = [collected[seq_id] for seq_id in sorted(collected)]
    interactive_rows = [row for row in requests if row["cohort"] == "interactive"]
    long_rows = [row for row in requests if row["cohort"] == "long"]
    interactive_ids = sorted(row["seq_id"] for row in interactive_rows)
    long_ids = sorted(row["seq_id"] for row in long_rows)
    if len(interactive_rows) != spec["interactive_requests"]:
        raise RuntimeError("interactive completion count mismatch")
    if len(long_rows) != spec["long_requests"]:
        raise RuntimeError("long completion count mismatch")
    expected_completion_tokens = (
        (spec["interactive_requests"] + spec["long_requests"])
        * spec["max_completion_tokens_per_request"]
    )
    actual_completion_tokens = sum(
        int(row["metrics"]["num_completion_tokens"]) for row in requests
    )
    if actual_completion_tokens != expected_completion_tokens:
        raise RuntimeError(
            "measured completion-token count mismatch: "
            f"{actual_completion_tokens} != {expected_completion_tokens}"
        )
    whole_run_s = ended - started
    if whole_run_s <= 0:
        raise RuntimeError("non-positive measured wall time")

    return {
        "workload": {
            "spec": spec,
            "prompt_manifests": {
                "warmup": prompt_manifest("warmup", warmup_prompts),
                "interactive": prompt_manifest("interactive", interactive_prompts),
                "long": prompt_manifest("long", long_prompts),
            },
            "warmup_completion_token_ids_sha256": [
                _token_vector_sha256(list(output["token_ids"]))
                for output in warmup_outputs
            ],
            "interactive_seq_ids": interactive_ids,
            "long_seq_ids": long_ids,
        },
        "requests": requests,
        "summary": {
            "whole_run_s": whole_run_s,
            "after_long_admission_s": ended - long_admitted,
            "total_completion_tokens": actual_completion_tokens,
            "completion_tokens_per_s": actual_completion_tokens / whole_run_s,
            "interactive": summarize_requests(interactive_rows),
            "long": summarize_requests(long_rows),
            "all": summarize_requests(requests),
            "single_run_latency_certified": False,
            "single_run_policy": (
                "exactly five validated tau-256 runs are required for certification"
            ),
        },
        "telemetry": {
            "routing": {
                "per_step_routes_observed": False,
                "per_step_routes_reason": (
                    "observing StepOutput adds work between scheduler token "
                    "timestamps and changes the next ITL"
                ),
                "selected_graph_keys_observed": False,
                "selected_graph_keys_reason": (
                    "model-path wrapping is excluded from the certification timing path"
                ),
                "configured_varlen_graph_keys": [
                    list(key) for key in sorted(getattr(runner, "varlen_graphs", {}))
                ],
                "configured_decode_graph_batch_sizes": list(
                    getattr(runner, "graph_bs", ())
                ),
                "varlen_miss_before": misses_before,
                "varlen_miss_after": int(getattr(runner, "varlen_miss", 0)),
                "varlen_miss_delta": (
                    int(getattr(runner, "varlen_miss", 0)) - misses_before
                ),
            },
            "memory": {
                "allocated_before_bytes": allocated_before,
                "reserved_before_bytes": reserved_before,
                "allocated_after_bytes": int(torch_module.cuda.memory_allocated()),
                "reserved_after_bytes": int(torch_module.cuda.memory_reserved()),
                "peak_allocated_bytes": int(torch_module.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch_module.cuda.max_memory_reserved()),
                "kv_blocks_free_before": free_blocks_before,
                "kv_blocks_used_before": used_blocks_before,
                "kv_blocks_free_after": len(block_manager.free_block_ids),
                "kv_blocks_used_after": len(block_manager.used_block_ids),
            },
        },
    }


def _main_impl(
    argv: list[str] | None,
    register_engine: Callable[[object], None],
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if handle_pin_query(args.print_source_sha256, args.show_pin):
        return 0
    _validate_args(args)

    # Keep release-pin inspection usable on CPU hosts.
    import torch
    import transformers

    from nanovllm import LLM, SamplingParams

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full-completion certification")
    if not gc.isenabled():
        raise RuntimeError(
            "start certification in a fresh process with Python GC enabled"
        )

    pin = validate_release_pin(args.expected_commit, args.expected_source_sha256)
    model = model_identity(Path(args.model))
    environment = environment_identity(torch, transformers)
    invocation = [
        sys.executable,
        *(sys.argv if argv is None else [str(Path(__file__).resolve()), *argv]),
    ]
    result = base_result(KIND, invocation, pin, model, environment)
    result["schema_version"] = SCHEMA_VERSION
    result["protocol"] = PROTOCOL
    result["arguments"] = vars(args) | {"output": str(args.output)}
    result["randomness"] = {
        "seed": args.seed,
        "python_random_seed": args.seed,
        "torch_manual_seed": args.seed,
        "torch_cuda_manual_seed_all": args.seed,
        "torch_deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
    }

    spec = certification_workload(args.tau)
    gc_before = gc.isenabled()
    engine_started = perf_counter()
    llm = LLM(
        args.model,
        max_num_batched_tokens=spec["tau"],
        max_num_seqs=spec["max_num_seqs"],
        max_model_len=spec["max_model_len"],
        gpu_memory_utilization=spec["gpu_memory_utilization"],
        enforce_eager=spec["enforce_eager"],
        tensor_parallel_size=spec["tensor_parallel_size"],
        disable_python_gc=spec["disable_python_gc"],
    )
    register_engine(llm)
    torch.cuda.synchronize()
    gc_after_init = gc.isenabled()
    if gc_after_init:
        raise RuntimeError("engine-owned GC lease was not active after initialization")
    config = llm.model_runner.config
    actual_config = {
        "max_num_batched_tokens": config.max_num_batched_tokens,
        "max_num_seqs": config.max_num_seqs,
        "max_model_len": config.max_model_len,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "enforce_eager": config.enforce_eager,
        "tensor_parallel_size": config.tensor_parallel_size,
        "disable_python_gc": config.disable_python_gc,
        "kvcache_block_size": config.kvcache_block_size,
        "num_kvcache_blocks": config.num_kvcache_blocks,
    }
    for name in (
        "max_num_batched_tokens",
        "max_num_seqs",
        "max_model_len",
        "gpu_memory_utilization",
        "enforce_eager",
        "tensor_parallel_size",
        "disable_python_gc",
    ):
        if actual_config[name] != spec[
            "tau" if name == "max_num_batched_tokens" else name
        ]:
            raise RuntimeError(f"engine config differs from pinned workload: {name}")
    result["engine"] = {
        "initialization_ms": (perf_counter() - engine_started) * 1000.0,
        "config": actual_config,
    }
    result.update(_run_workload(llm, SamplingParams, torch, args.seed))

    llm.exit()
    gc_after_exit = gc.isenabled()
    if gc_after_exit != gc_before:
        raise RuntimeError("engine exit did not restore the pre-init Python GC state")
    result["python_gc"] = {
        "disable_requested": True,
        "enabled_before_engine": gc_before,
        "enabled_after_engine_init": gc_after_init,
        "enabled_after_engine_exit": gc_after_exit,
    }

    model_after = model_identity(Path(args.model))
    if model_after != model:
        raise RuntimeError("model manifest changed during certification run")
    result["model_after_run_matches"] = True
    result["provenance_after_run"] = validate_release_pin(
        args.expected_commit, args.expected_source_sha256
    )
    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    immutable_write_json(args.output, result)
    print(json.dumps({
        "output": str(args.output),
        "seed": args.seed,
        "tau": args.tau,
        "summary": result["summary"],
    }, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    engine = None

    def register_engine(value) -> None:
        nonlocal engine
        engine = value

    try:
        return _main_impl(argv, register_engine)
    finally:
        if engine is not None:
            engine.exit()


if __name__ == "__main__":
    raise SystemExit(main())
