"""Retained, release-pinned diagnostics for the chunked-prefill tail.

This is deliberately an external observer: it wraps methods on one runner
instance and never changes scheduler or model-runner production code.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

from benchmarks.chunked_prefill_tail.common import (
    base_result,
    environment_identity,
    handle_pin_query,
    immutable_write_json,
    model_identity,
    timing_summary,
    validate_release_pin,
)
from benchmarks.chunked_prefill_tail.release_policy import (
    evaluate_chunk_tail_release_profile,
)


DEFAULT_MODEL = "/workspace/models/Qwen3-0.6B"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure full engine steps at the decode/chunked-prefill tail."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tau", type=int, default=128)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--interactive-count", type=int, default=16)
    parser.add_argument("--interactive-prompt-len", type=int, default=64)
    parser.add_argument("--long-count", type=int, default=2)
    parser.add_argument("--long-prompt-len", type=int, default=2048)
    parser.add_argument("--pre-long-steps", type=int, default=40)
    parser.add_argument("--measured-steps", type=int, default=96)
    parser.add_argument("--cold-steps", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--disable-python-gc",
        action="store_true",
        help="opt in to engine-owned process-wide cyclic-GC suppression",
    )
    parser.add_argument("--expected-commit")
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--print-source-sha256", action="store_true")
    parser.add_argument("--show-pin", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    missing = [
        name
        for name in ("expected_commit", "expected_source_sha256", "output")
        if getattr(args, name) is None
    ]
    if missing:
        raise ValueError(
            "runtime diagnostics require "
            + ", ".join("--" + name.replace("_", "-") for name in missing)
        )
    positive = (
        "tau",
        "max_num_seqs",
        "max_model_len",
        "interactive_count",
        "interactive_prompt_len",
        "long_count",
        "long_prompt_len",
        "pre_long_steps",
        "measured_steps",
        "max_tokens",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.tau < args.max_num_seqs:
        raise ValueError("--tau must be greater than or equal to --max-num-seqs")
    total_requests = args.interactive_count + args.long_count
    if total_requests > args.max_num_seqs:
        raise ValueError(
            "interactive-count + long-count exceeds --max-num-seqs: "
            f"{total_requests} > {args.max_num_seqs}"
        )
    if args.interactive_prompt_len >= args.max_model_len:
        raise ValueError("interactive prompts must be shorter than --max-model-len")
    if args.long_prompt_len >= args.max_model_len:
        raise ValueError("long prompts must be shorter than --max-model-len")
    if args.interactive_prompt_len + args.max_tokens > args.max_model_len:
        raise ValueError("interactive prompt + max tokens exceeds --max-model-len")
    if args.long_prompt_len + args.max_tokens > args.max_model_len:
        raise ValueError("long prompt + max tokens exceeds --max-model-len")
    if args.max_tokens <= args.pre_long_steps + args.measured_steps:
        raise ValueError(
            "--max-tokens must exceed pre-long-steps + measured-steps so the "
            "interactive decode cohort remains live"
        )
    if not 0 <= args.cold_steps <= args.measured_steps:
        raise ValueError("--cold-steps must be between zero and --measured-steps")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        raise ValueError("--gpu-memory-utilization must be in (0, 1)")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite retained output: {args.output}")


def _prompts(
    rng: random.Random,
    count: int,
    length: int,
    vocab_size: int,
) -> list[list[int]]:
    if vocab_size <= 16:
        raise ValueError(f"unexpectedly small vocabulary: {vocab_size}")
    return [
        [rng.randrange(8, vocab_size) for _ in range(length)]
        for _ in range(count)
    ]


def _key(value) -> list[int] | None:
    if value is None:
        return None
    return list(value) if isinstance(value, tuple) else [int(value)]


def _summaries(steps: list[dict]) -> dict[str, object]:
    groups: dict[str, list[dict]] = {
        "all": steps,
        "cold": [row for row in steps if row["phase"] == "cold"],
        "steady": [row for row in steps if row["phase"] == "steady"],
    }
    for route in ("mixed", "prefill", "decode"):
        groups[f"route:{route}"] = [row for row in steps if row["route"] == route]
    return {name: timing_summary(rows) for name, rows in groups.items()}


def _main_impl(
    argv: list[str] | None,
    register_engine: Callable[[object], None],
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if handle_pin_query(args.print_source_sha256, args.show_pin):
        return 0
    _validate_args(args)

    # Imports stay below the source-pin query so provenance can be inspected on
    # CPU-only hosts without initializing CUDA or nano-vllm.
    import torch
    import transformers

    from nanovllm import LLM, SamplingParams
    from nanovllm.utils.context import get_context

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the retained step diagnostic")

    started = perf_counter()
    pin = validate_release_pin(args.expected_commit, args.expected_source_sha256)
    model = model_identity(Path(args.model))
    environment = environment_identity(torch, transformers)
    result = base_result(
        "chunked_prefill_tail_step_diagnostic",
        [sys.executable, *sys.argv],
        pin,
        model,
        environment,
    )
    result["arguments"] = vars(args) | {"output": str(args.output)}
    result["release_policy"] = evaluate_chunk_tail_release_profile(
        args.tau,
        {
            "model_resolved_path": model["resolved_path"],
            "gpu": environment.get("gpu"),
            "torch": environment.get("torch"),
            "cuda": environment.get("cuda_build"),
            "python_gc_mode": (
                "engine_option_after_successful_initialization"
                if args.disable_python_gc
                else "engine_default_no_gc_change"
            ),
            # This harness measures bounded engine steps rather than complete
            # request-metric ITLs, so it must not self-certify from reference
            # evidence even when every configuration field matches.
            "measurement_protocol": "bounded_per_step_diagnostic",
            "interactive_count": args.interactive_count,
            "interactive_prompt_len": args.interactive_prompt_len,
            "pre_long_steps": args.pre_long_steps,
            "long_count": args.long_count,
            "long_prompt_len": args.long_prompt_len,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_num_seqs": args.max_num_seqs,
        },
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rng = random.Random(args.seed)
    gc_enabled_before_engine = gc.isenabled()
    engine_started = perf_counter()
    llm = LLM(
        args.model,
        max_num_batched_tokens=args.tau,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False,
        disable_python_gc=args.disable_python_gc,
    )
    # Register immediately so an embedding caller that catches a later
    # diagnostic failure still gets deterministic engine/GC cleanup.
    register_engine(llm)
    torch.cuda.synchronize()
    gc_enabled_after_engine_init = gc.isenabled()
    if args.disable_python_gc and gc_enabled_after_engine_init:
        raise RuntimeError("engine did not disable Python GC after successful init")
    if not args.disable_python_gc and (
        gc_enabled_after_engine_init != gc_enabled_before_engine
    ):
        raise RuntimeError("default engine initialization changed Python GC state")
    engine_init_ms = (perf_counter() - engine_started) * 1000.0
    runner = llm.model_runner
    scheduler = llm.scheduler
    block_manager = scheduler.block_manager

    run_observations: list[dict] = []
    model_observations: list[dict] = []
    original_run = runner.run
    original_run_model = runner.run_model

    def observed_run(seqs, is_prefill):
        wall_started = perf_counter()
        cuda_started = torch.cuda.Event(enable_timing=True)
        cuda_finished = torch.cuda.Event(enable_timing=True)
        cuda_started.record()
        observation = {
            "is_ragged": bool(is_prefill),
            "segment_count": len(seqs),
            "scheduled_prefill_tokens": sum(
                seq.num_scheduled_tokens for seq in seqs if seq.is_prefill
            ),
            "scheduled_decode_tokens": sum(
                1 for seq in seqs if not seq.is_prefill
            ),
            "scheduled_rows": [
                {
                    "seq_id": seq.seq_id,
                    "is_prefill": bool(seq.is_prefill),
                    "scheduled_tokens": seq.num_scheduled_tokens,
                    "cached_tokens_before": seq.num_cached_tokens,
                    "sequence_tokens_before": len(seq),
                }
                for seq in seqs
            ],
            "_wall_started": wall_started,
            "_cuda_started": cuda_started,
            "_cuda_finished": cuda_finished,
        }
        run_observations.append(observation)
        output = original_run(seqs, is_prefill)
        cuda_finished.record()
        observation["_wall_finished"] = perf_counter()
        return output

    def observed_run_model(input_ids, positions, is_prefill):
        wall_started = perf_counter()
        cuda_started = torch.cuda.Event(enable_timing=True)
        cuda_finished = torch.cuda.Event(enable_timing=True)
        cuda_started.record()
        num_tokens = int(input_ids.size(0))
        misses_before = int(getattr(runner, "varlen_miss", 0))
        candidate = selected = None
        segment_count = int(input_ids.size(0))
        route = "decode_eager"
        if is_prefill:
            context = get_context()
            segment_count = int(context.cu_seqlens_q.numel() - 1)
            candidate = runner._select_varlen_graph_key(num_tokens, segment_count)
            if candidate is not None and runner._varlen_context_fits_graph(
                num_tokens, segment_count, context, candidate
            ):
                selected = candidate
                route = "varlen_cuda_graph"
            else:
                route = "prefill_eager"
        elif not runner.enforce_eager and num_tokens <= 512:
            selected = next(key for key in runner.graph_bs if key >= num_tokens)
            route = "decode_cuda_graph"

        output = original_run_model(input_ids, positions, is_prefill)
        cuda_finished.record()
        wall_finished = perf_counter()
        model_observations.append({
            "input_tokens": num_tokens,
            "segment_count": segment_count,
            "candidate_graph_key": _key(candidate),
            "selected_graph_key": _key(selected),
            "model_route": route,
            "varlen_miss_before": misses_before,
            "varlen_miss_after": int(getattr(runner, "varlen_miss", 0)),
            "varlen_miss_delta": int(getattr(runner, "varlen_miss", 0))
            - misses_before,
            "_wall_started": wall_started,
            "_wall_finished": wall_finished,
            "_cuda_started": cuda_started,
            "_cuda_finished": cuda_finished,
        })
        return output

    # Instance-only wrappers: the benchmark observes production behavior but
    # leaves the class and repository source untouched.
    runner.run = observed_run
    runner.run_model = observed_run_model

    vocab_size = int(runner.config.hf_config.vocab_size)
    params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )
    interactive_prompts = _prompts(
        rng,
        args.interactive_count,
        args.interactive_prompt_len,
        vocab_size,
    )
    long_prompts = _prompts(
        rng,
        args.long_count,
        args.long_prompt_len,
        vocab_size,
    )
    interactive_ids = [llm.add_request(prompt, params) for prompt in interactive_prompts]

    # Put the interactive cohort into steady decode before introducing long
    # prefills. These conditioning steps are explicitly excluded from timing.
    for _ in range(args.pre_long_steps):
        if llm.is_finished():
            raise RuntimeError("interactive cohort finished during conditioning")
        llm._step()
    torch.cuda.synchronize()
    conditioning_observations = {
        "steps": args.pre_long_steps,
        "runner_calls": len(run_observations),
        "model_calls": len(model_observations),
    }
    run_observations.clear()
    model_observations.clear()

    long_ids = [llm.add_request(prompt, params) for prompt in long_prompts]
    free_blocks_before = len(block_manager.free_block_ids)
    used_blocks_before = len(block_manager.used_block_ids)
    varlen_miss_before_measured = runner.varlen_miss
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    reserved_before = torch.cuda.memory_reserved()

    steps: list[dict] = []
    for index in range(args.measured_steps):
        if llm.is_finished():
            break
        if run_observations or model_observations:
            raise RuntimeError("observer state leaked between engine steps")
        torch.cuda.synchronize()
        cuda_start = torch.cuda.Event(enable_timing=True)
        cuda_end = torch.cuda.Event(enable_timing=True)
        wall_start = perf_counter()
        cuda_start.record()
        step_output = llm._step()
        cuda_end.record()
        cuda_end.synchronize()
        wall_ms = (perf_counter() - wall_start) * 1000.0
        cuda_ms = float(cuda_start.elapsed_time(cuda_end))

        if len(run_observations) != 1 or len(model_observations) != 1:
            raise RuntimeError(
                "expected exactly one runner/model call per engine step; found "
                f"{len(run_observations)}/{len(model_observations)}"
            )
        run = run_observations.pop()
        model_call = model_observations.pop()
        if run["scheduled_prefill_tokens"] != step_output.num_prefill_tokens:
            raise RuntimeError("observed prefill-token count disagrees with StepOutput")
        if run["scheduled_decode_tokens"] != step_output.num_decode_tokens:
            raise RuntimeError("observed decode-token count disagrees with StepOutput")
        if run["segment_count"] != model_call["segment_count"]:
            raise RuntimeError("scheduled segment count disagrees with model context")

        run_wall_started = run.pop("_wall_started")
        run_wall_finished = run.pop("_wall_finished")
        run_cuda_started = run.pop("_cuda_started")
        run_cuda_finished = run.pop("_cuda_finished")
        model_wall_started = model_call.pop("_wall_started")
        model_wall_finished = model_call.pop("_wall_finished")
        model_cuda_started = model_call.pop("_cuda_started")
        model_cuda_finished = model_call.pop("_cuda_finished")
        phase_timings = {
            # The boundary spans include host dispatch gaps between enqueued
            # GPU work. That is intentional: an outer CUDA-event outlier with
            # normal model kernels becomes attributable instead of invisible.
            "prepare_to_model_wall_ms": (
                model_wall_started - run_wall_started
            ) * 1000.0,
            "model_wall_ms": (
                model_wall_finished - model_wall_started
            ) * 1000.0,
            "post_model_sampler_wall_ms": (
                run_wall_finished - model_wall_finished
            ) * 1000.0,
            "runner_wall_ms": (run_wall_finished - run_wall_started) * 1000.0,
            "prepare_to_model_cuda_span_ms": float(
                run_cuda_started.elapsed_time(model_cuda_started)
            ),
            "model_cuda_span_ms": float(
                model_cuda_started.elapsed_time(model_cuda_finished)
            ),
            "post_model_sampler_cuda_span_ms": float(
                model_cuda_finished.elapsed_time(run_cuda_finished)
            ),
            "runner_cuda_span_ms": float(
                run_cuda_started.elapsed_time(run_cuda_finished)
            ),
        }

        prefill = int(step_output.num_prefill_tokens)
        decode = int(step_output.num_decode_tokens)
        if prefill + decode > args.tau:
            raise RuntimeError(
                f"scheduled step exceeds tau: {prefill} + {decode} > {args.tau}"
            )
        route = "mixed" if prefill and decode else "prefill" if prefill else "decode"
        steps.append({
            "step": index,
            "phase": "cold" if index < args.cold_steps else "steady",
            "route": route,
            "wall_ms": wall_ms,
            "cuda_ms": cuda_ms,
            "actual_prefill_tokens": prefill,
            "actual_decode_tokens": decode,
            "actual_total_tokens": prefill + decode,
            "phase_timings": phase_timings,
            **run,
            **model_call,
            "emitted_events": [event._asdict() for event in step_output.events],
            "finished_seq_ids": [seq.seq_id for seq in step_output.finished],
            "waiting_after": len(scheduler.waiting),
            "running_after": len(scheduler.running),
            "mid_chunk_seq_id_after": (
                scheduler.mid_chunk_seq.seq_id if scheduler.mid_chunk_seq else None
            ),
            "kv_blocks_free_after": len(block_manager.free_block_ids),
            "kv_blocks_used_after": len(block_manager.used_block_ids),
            "memory_allocated_bytes_after": torch.cuda.memory_allocated(),
            "memory_reserved_bytes_after": torch.cuda.memory_reserved(),
        })

    torch.cuda.synchronize()
    result.update({
        "engine": {
            "initialization_ms": engine_init_ms,
            "config": {
                "max_num_batched_tokens": runner.config.max_num_batched_tokens,
                "max_num_seqs": runner.config.max_num_seqs,
                "max_model_len": runner.config.max_model_len,
                "gpu_memory_utilization": runner.config.gpu_memory_utilization,
                "disable_python_gc": runner.config.disable_python_gc,
                "kvcache_block_size": runner.config.kvcache_block_size,
                "num_kvcache_blocks": runner.config.num_kvcache_blocks,
            },
            "varlen_graph_keys": [list(key) for key in sorted(runner.varlen_graphs)],
            "varlen_graph_count": len(runner.varlen_graphs),
            "varlen_miss_before_measured": varlen_miss_before_measured,
            "varlen_miss_final": runner.varlen_miss,
            "varlen_miss_measured_delta": (
                runner.varlen_miss - varlen_miss_before_measured
            ),
        },
        "workload": {
            "conditioning": conditioning_observations,
            "interactive_seq_ids": interactive_ids,
            "long_seq_ids": long_ids,
            "measured_steps_requested": args.measured_steps,
            "measured_steps_completed": len(steps),
            "scheduler_finished": llm.is_finished(),
        },
        "memory": {
            "allocated_before_bytes": allocated_before,
            "reserved_before_bytes": reserved_before,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "incremental_peak_allocated_bytes": (
                torch.cuda.max_memory_allocated() - allocated_before
            ),
            "incremental_peak_reserved_bytes": (
                torch.cuda.max_memory_reserved() - reserved_before
            ),
            "kv_blocks_free_before": free_blocks_before,
            "kv_blocks_used_before": used_blocks_before,
            "kv_blocks_free_after": len(block_manager.free_block_ids),
            "kv_blocks_used_after": len(block_manager.used_block_ids),
        },
        "steps": steps,
        "timing_summaries": _summaries(steps),
        "elapsed_ms_before_write": (perf_counter() - started) * 1000.0,
    })
    # Explicit exit proves the opt-in restoration contract in the same retained
    # artifact. The registered atexit callback is idempotent and becomes a no-op.
    llm.exit()
    if gc.isenabled() != gc_enabled_before_engine:
        raise RuntimeError("engine exit did not restore the prior Python GC state")
    result["python_gc"] = {
        "disable_requested": args.disable_python_gc,
        "enabled_before_engine": gc_enabled_before_engine,
        "enabled_after_engine_init": gc_enabled_after_engine_init,
        "enabled_after_engine_exit": gc.isenabled(),
    }
    # Recompute immediately before release: this is both retained evidence and
    # a guard against source mutation during a long diagnostic run.
    result["provenance_after_run"] = validate_release_pin(
        args.expected_commit, args.expected_source_sha256
    )
    immutable_write_json(args.output, result)
    print(json.dumps({
        "output": str(args.output),
        "steps": len(steps),
        "steady": result["timing_summaries"]["steady"],
        "peak_allocated_bytes": result["memory"]["peak_allocated_bytes"],
    }, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    engine = None

    def register_engine(created_engine) -> None:
        nonlocal engine
        engine = created_engine

    try:
        return _main_impl(argv, register_engine)
    finally:
        if engine is not None:
            engine.exit()


if __name__ == "__main__":
    raise SystemExit(main())
