#!/usr/bin/env python3
"""Intrusive, non-certifying attribution for rare decode-only ITL spikes.

The full-completion certification path intentionally avoids per-step observers.
This separate diagnostic trades a small amount of observer overhead for phase
attribution.  It never changes production classes: instance methods are wrapped
for one TP1 engine and restored before the engine exits.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import resource
import statistics
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter_ns, thread_time_ns

from benchmarks.chunked_prefill_tail.common import (
    ROOT,
    base_result,
    environment_identity,
    handle_pin_query,
    immutable_write_json,
    model_identity,
    nearest_rank_percentile,
    validate_release_pin,
)


SCHEMA_VERSION = 1
KIND = "chunked_prefill_decode_jitter_diagnostic"
PROTOCOL = "decode_phase_attribution_bs16_bs18_v1"
DEFAULT_MODEL = "/workspace/models/Qwen3-0.6B"
BATCH_SIZES = (16, 18)
MIN_STEPS_PER_BATCH = 1000
MAX_NUM_SEQS = 32
MAX_COMPLETION_SLACK = 16


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Attribute decode-only wall-time spikes at batch sizes 16 and 18; "
            "output is intrusive diagnostics, never certification evidence."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tau", type=int, choices=(256, 512), default=256)
    parser.add_argument(
        "--steps-per-batch", type=int, default=MIN_STEPS_PER_BATCH
    )
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260836)
    parser.add_argument("--spike-threshold-ms", type=float, default=10.0)
    parser.add_argument("--expected-commit")
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--print-source-sha256", action="store_true")
    parser.add_argument("--show-pin", action="store_true")
    return parser


def workload_spec(args: argparse.Namespace) -> dict[str, object]:
    max_tokens = 2 * args.steps_per_batch + MAX_COMPLETION_SLACK
    return {
        "tau": args.tau,
        "batch_sizes": list(BATCH_SIZES),
        "steps_per_batch": args.steps_per_batch,
        "prompt_tokens": args.prompt_len,
        "max_completion_tokens": max_tokens,
        "temperature": 0.6,
        "top_k": -1,
        "top_p": 1.0,
        "ignore_eos": True,
        "prompt_token_min_inclusive": 100,
        "prompt_token_max_exclusive": 10_000,
        "max_model_len": args.max_model_len,
        "max_num_seqs": MAX_NUM_SEQS,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": False,
        "tensor_parallel_size": 1,
        "disable_python_gc": True,
    }


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
    if args.steps_per_batch < MIN_STEPS_PER_BATCH:
        raise ValueError(
            f"--steps-per-batch must be at least {MIN_STEPS_PER_BATCH}"
        )
    if args.prompt_len <= 0:
        raise ValueError("--prompt-len must be positive")
    if args.max_model_len <= 0:
        raise ValueError("--max-model-len must be positive")
    spec = workload_spec(args)
    if args.prompt_len + spec["max_completion_tokens"] > args.max_model_len:
        raise ValueError(
            "prompt plus the completion lifetime needed for both cohorts "
            "exceeds --max-model-len"
        )
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        raise ValueError("--gpu-memory-utilization must be in (0, 1)")
    if not 0 <= args.seed < 2**63:
        raise ValueError("--seed must be in [0, 2**63)")
    if args.spike_threshold_ms <= 0:
        raise ValueError("--spike-threshold-ms must be positive")
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


@dataclass(frozen=True)
class HostSnapshot:
    wall_ns: int
    thread_cpu_ns: int
    voluntary_context_switches: int
    involuntary_context_switches: int


def host_snapshot(
    wall_clock: Callable[[], int] = perf_counter_ns,
    thread_clock: Callable[[], int] = thread_time_ns,
    getrusage: Callable[[int], object] = resource.getrusage,
) -> HostSnapshot:
    usage = getrusage(resource.RUSAGE_THREAD)
    return HostSnapshot(
        wall_ns=int(wall_clock()),
        thread_cpu_ns=int(thread_clock()),
        voluntary_context_switches=int(usage.ru_nvcsw),
        involuntary_context_switches=int(usage.ru_nivcsw),
    )


def host_delta(before: HostSnapshot, after: HostSnapshot) -> dict[str, object]:
    fields = (
        ("wall_ns", before.wall_ns, after.wall_ns),
        ("thread_cpu_ns", before.thread_cpu_ns, after.thread_cpu_ns),
        (
            "voluntary_context_switches",
            before.voluntary_context_switches,
            after.voluntary_context_switches,
        ),
        (
            "involuntary_context_switches",
            before.involuntary_context_switches,
            after.involuntary_context_switches,
        ),
    )
    for name, start, finish in fields:
        if finish < start:
            raise RuntimeError(f"non-monotonic host counter: {name}")
    wall_ms = (after.wall_ns - before.wall_ns) / 1_000_000.0
    thread_ms = (after.thread_cpu_ns - before.thread_cpu_ns) / 1_000_000.0
    return {
        "wall_ms": wall_ms,
        "thread_cpu_ms": thread_ms,
        "non_thread_cpu_wall_ms": wall_ms - thread_ms,
        "voluntary_context_switches": (
            after.voluntary_context_switches
            - before.voluntary_context_switches
        ),
        "involuntary_context_switches": (
            after.involuntary_context_switches
            - before.involuntary_context_switches
        ),
    }


def _number_summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    return {
        "median": statistics.median(values),
        "p95": nearest_rank_percentile(values, 0.95),
        "p99": nearest_rank_percentile(values, 0.99),
        "min": min(values),
        "max": max(values),
    }


def summarize_profile(
    rows: list[dict[str, object]], spike_threshold_ms: float
) -> dict[str, object]:
    if not rows:
        raise ValueError("cannot summarize an empty decode profile")
    timing_names = (
        "api_wall_ms",
        "api_thread_cpu_ms",
        "api_non_thread_cpu_wall_ms",
        "api_residual_wall_ms",
        "scheduler_wall_ms",
        "prepare_wall_ms",
        "model_wall_ms",
        "sampler_wall_ms",
        "postprocess_wall_ms",
        "runner_cuda_ms",
        "prepare_cuda_ms",
        "model_cuda_ms",
        "sampler_cuda_ms",
        "runner_cuda_residual_ms",
    )
    summaries = {
        name: _number_summary([float(row["timings_ms"][name]) for row in rows])
        for name in timing_names
    }
    spikes = [
        {
            "step": row["step"],
            "api_wall_ms": row["timings_ms"]["api_wall_ms"],
            "api_thread_cpu_ms": row["timings_ms"]["api_thread_cpu_ms"],
            "runner_cuda_ms": row["timings_ms"]["runner_cuda_ms"],
            "api_residual_wall_ms": row["timings_ms"]["api_residual_wall_ms"],
            "context_switches": row["context_switches"]["api"],
        }
        for row in rows
        if float(row["timings_ms"]["api_wall_ms"]) > spike_threshold_ms
    ]
    top_wall = sorted(
        rows,
        key=lambda row: float(row["timings_ms"]["api_wall_ms"]),
        reverse=True,
    )[:20]
    return {
        "step_count": len(rows),
        "spike_threshold_ms_strictly_greater_than": spike_threshold_ms,
        "spike_count": len(spikes),
        "spikes": spikes,
        "timings_ms": summaries,
        "context_switch_totals": {
            "voluntary": sum(
                int(row["context_switches"]["api"]["voluntary"]) for row in rows
            ),
            "involuntary": sum(
                int(row["context_switches"]["api"]["involuntary"]) for row in rows
            ),
        },
        "top_20_api_wall_steps": [
            {
                "step": row["step"],
                "api_wall_ms": row["timings_ms"]["api_wall_ms"],
                "runner_cuda_ms": row["timings_ms"]["runner_cuda_ms"],
                "api_thread_cpu_ms": row["timings_ms"]["api_thread_cpu_ms"],
                "context_switches": row["context_switches"]["api"],
            }
            for row in top_wall
        ],
    }


def _prompt_manifest(label: str, prompts: list[list[int]]) -> dict[str, object]:
    aggregate = hashlib.sha256()
    rows = []
    for index, prompt in enumerate(prompts):
        encoded = json.dumps(prompt, separators=(",", ":")).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        aggregate.update(index.to_bytes(8, "big"))
        aggregate.update(len(prompt).to_bytes(8, "big"))
        aggregate.update(bytes.fromhex(digest))
        rows.append({"index": index, "tokens": len(prompt), "sha256": digest})
    return {
        "label": label,
        "count": len(prompts),
        "aggregate_sha256": aggregate.hexdigest(),
        "prompts": rows,
    }


class DecodeStepTracer:
    """Instance-only phase observer for one TP1 engine."""

    PHASES = ("scheduler", "prepare", "model", "sampler", "postprocess")

    def __init__(self, scheduler, runner, torch_module):
        self.scheduler = scheduler
        self.runner = runner
        self.torch = torch_module
        self._active: dict[str, object] | None = None
        self._records: list[dict[str, object]] = []
        self._previous_api_end: HostSnapshot | None = None
        self._profile_batch_size: int | None = None
        self._installed = False
        self._original_schedule = scheduler.schedule
        self._original_prepare_decode = runner.prepare_decode
        self._original_run_model = runner.run_model
        self._original_sampler_forward = runner.sampler.forward
        self._original_postprocess = scheduler.postprocess

    def install(self) -> None:
        if self._installed:
            raise RuntimeError("decode observer is already installed")
        self.scheduler.schedule = self._observed_schedule
        self.runner.prepare_decode = self._observed_prepare_decode
        self.runner.run_model = self._observed_run_model
        self.runner.sampler.forward = self._observed_sampler
        self.scheduler.postprocess = self._observed_postprocess
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        self.scheduler.schedule = self._original_schedule
        self.runner.prepare_decode = self._original_prepare_decode
        self.runner.run_model = self._original_run_model
        self.runner.sampler.forward = self._original_sampler_forward
        self.scheduler.postprocess = self._original_postprocess
        self._installed = False

    def begin_profile(self, batch_size: int) -> None:
        if self._active is not None or self._records:
            raise RuntimeError("decode observer state is not empty")
        self._profile_batch_size = batch_size
        self._previous_api_end = None

    def begin_step(self, index: int) -> None:
        if self._profile_batch_size is None:
            raise RuntimeError("begin_profile must precede begin_step")
        if self._active is not None:
            raise RuntimeError("previous decode step is still active")
        entered = host_snapshot()
        gap = (
            None
            if self._previous_api_end is None
            else host_delta(self._previous_api_end, entered)
        )
        self._active = {
            "step": index,
            "batch_size": self._profile_batch_size,
            "_api_enter": entered,
            "caller_gap_before": gap,
            "_phases": {},
            "_cuda_events": {},
            "route": None,
        }

    def _timed(self, name: str, function: Callable, *args, **kwargs):
        if self._active is None:
            return function(*args, **kwargs)
        phases = self._active["_phases"]
        if name in phases:
            raise RuntimeError(f"phase observed more than once in one step: {name}")
        before = host_snapshot()
        output = function(*args, **kwargs)
        after = host_snapshot()
        phases[name] = host_delta(before, after)
        return output

    def _event(self, name: str):
        if self._active is None:
            raise RuntimeError("CUDA event requested outside an active step")
        events = self._active["_cuda_events"]
        if name in events:
            raise RuntimeError(f"duplicate CUDA event: {name}")
        event = self.torch.cuda.Event(enable_timing=True)
        event.record()
        events[name] = event
        return event

    def _observed_schedule(self, *args, **kwargs):
        result = self._timed("scheduler", self._original_schedule, *args, **kwargs)
        if self._active is not None:
            seqs, is_ragged = result
            prefill = sum(seq.num_scheduled_tokens for seq in seqs if seq.is_prefill)
            decode = sum(1 for seq in seqs if not seq.is_prefill)
            graph_key = (
                None
                if is_ragged
                else next(key for key in self.runner.graph_bs if key >= decode)
            )
            self._active["route"] = {
                "is_ragged": bool(is_ragged),
                "scheduled_prefill_tokens": prefill,
                "scheduled_decode_tokens": decode,
                "scheduled_seq_ids": [seq.seq_id for seq in seqs],
                "selected_decode_graph_batch_size": graph_key,
            }
        return result

    def _observed_prepare_decode(self, *args, **kwargs):
        if self._active is None:
            return self._original_prepare_decode(*args, **kwargs)
        self._event("runner_start")
        self._event("prepare_start")
        output = self._timed(
            "prepare", self._original_prepare_decode, *args, **kwargs
        )
        self._event("prepare_end")
        return output

    def _observed_run_model(self, *args, **kwargs):
        if self._active is None:
            return self._original_run_model(*args, **kwargs)
        self._event("model_start")
        output = self._timed("model", self._original_run_model, *args, **kwargs)
        self._event("model_end")
        return output

    def _observed_sampler(self, *args, **kwargs):
        if self._active is None:
            return self._original_sampler_forward(*args, **kwargs)
        self._event("sampler_start")
        output = self._timed(
            "sampler", self._original_sampler_forward, *args, **kwargs
        )
        self._event("sampler_end")
        # Production runner.run calls tokens.tolist() immediately after this
        # wrapper returns.  That existing synchronization completes this event;
        # the diagnostic never adds a per-step CUDA synchronize.
        self._event("runner_end")
        return output

    def _observed_postprocess(self, *args, **kwargs):
        return self._timed("postprocess", self._original_postprocess, *args, **kwargs)

    def finish_step(self, outputs, signed_tokens: int) -> None:
        if self._active is None:
            raise RuntimeError("no active decode step to finish")
        exited = host_snapshot()
        row = self._active
        self._active = None
        row["_api_exit"] = exited
        row["_outputs_count"] = len(outputs)
        row["signed_tokens"] = signed_tokens
        self._records.append(row)
        self._previous_api_end = exited

    @staticmethod
    def _cuda_elapsed(events: dict[str, object], start: str, end: str) -> float:
        return float(events[start].elapsed_time(events[end]))

    def _resolve_record(self, row: dict[str, object]) -> dict[str, object]:
        entered = row.pop("_api_enter")
        exited = row.pop("_api_exit")
        outputs_count = row.pop("_outputs_count")
        phases = row.pop("_phases")
        events = row.pop("_cuda_events")
        missing = set(self.PHASES) - phases.keys()
        if missing:
            raise RuntimeError(f"decode step is missing phases: {sorted(missing)}")
        if not events["runner_end"].query():
            raise RuntimeError(
                "runner_end CUDA event was not complete after the production "
                "tokens.tolist() synchronization"
            )
        route = row["route"]
        batch_size = int(row["batch_size"])
        if route is None:
            raise RuntimeError("decode step is missing route telemetry")
        if route["is_ragged"] or route["scheduled_prefill_tokens"]:
            raise RuntimeError("measured profile left the decode-only route")
        if route["scheduled_decode_tokens"] != batch_size:
            raise RuntimeError(
                "measured decode count differs from the requested batch size"
            )
        if row["signed_tokens"] != -batch_size:
            raise RuntimeError("legacy signed-token result disagrees with decode count")
        if outputs_count:
            raise RuntimeError("a measured request finished before the profile ended")

        api = host_delta(entered, exited)
        phase_wall = sum(float(value["wall_ms"]) for value in phases.values())
        phase_thread = sum(
            float(value["thread_cpu_ms"]) for value in phases.values()
        )
        cuda = {
            "prepare_cuda_ms": self._cuda_elapsed(
                events, "prepare_start", "prepare_end"
            ),
            "model_cuda_ms": self._cuda_elapsed(
                events, "model_start", "model_end"
            ),
            "sampler_cuda_ms": self._cuda_elapsed(
                events, "sampler_start", "sampler_end"
            ),
            "runner_cuda_ms": self._cuda_elapsed(
                events, "runner_start", "runner_end"
            ),
        }
        cuda["runner_cuda_residual_ms"] = cuda["runner_cuda_ms"] - (
            cuda["prepare_cuda_ms"]
            + cuda["model_cuda_ms"]
            + cuda["sampler_cuda_ms"]
        )
        timings = {
            "api_wall_ms": api["wall_ms"],
            "api_thread_cpu_ms": api["thread_cpu_ms"],
            "api_non_thread_cpu_wall_ms": api["non_thread_cpu_wall_ms"],
            "api_residual_wall_ms": api["wall_ms"] - phase_wall,
            "api_residual_thread_cpu_ms": api["thread_cpu_ms"] - phase_thread,
            **{
                f"{name}_wall_ms": phases[name]["wall_ms"]
                for name in self.PHASES
            },
            **{
                f"{name}_thread_cpu_ms": phases[name]["thread_cpu_ms"]
                for name in self.PHASES
            },
            **cuda,
        }
        row["timings_ms"] = timings
        row["context_switches"] = {
            "scope": "RUSAGE_THREAD",
            "api": {
                "voluntary": api["voluntary_context_switches"],
                "involuntary": api["involuntary_context_switches"],
            },
            "phases": {
                name: {
                    "voluntary": phases[name]["voluntary_context_switches"],
                    "involuntary": phases[name]["involuntary_context_switches"],
                }
                for name in self.PHASES
            },
        }
        row["finished_outputs"] = outputs_count
        row["cuda_events_queried_after_api_return"] = True
        return row

    def finish_profile(self) -> list[dict[str, object]]:
        if self._active is not None:
            raise RuntimeError("cannot finish a profile with an active step")
        records = self._records
        self._records = []
        self._profile_batch_size = None
        self._previous_api_end = None
        return [self._resolve_record(row) for row in records]


def _condition_to_decode(llm, expected_batch_size: int) -> int:
    scheduler = llm.scheduler
    for count in range(1, 129):
        if (
            not scheduler.waiting
            and len(scheduler.running) == expected_batch_size
            and all(not seq.is_prefill for seq in scheduler.running)
        ):
            return count - 1
        output = llm._step()
        if output.finished:
            raise RuntimeError("request finished while conditioning decode profile")
    raise RuntimeError("decode cohort did not become steady within 128 steps")


def _run_profile(llm, tracer, torch_module, batch_size: int, steps: int) -> dict:
    torch_module.cuda.synchronize()
    torch_module.cuda.reset_peak_memory_stats()
    allocated_before = int(torch_module.cuda.memory_allocated())
    reserved_before = int(torch_module.cuda.memory_reserved())
    varlen_miss_before = int(getattr(llm.model_runner, "varlen_miss", 0))
    tracer.begin_profile(batch_size)
    for index in range(steps):
        tracer.begin_step(index)
        outputs, signed_tokens = llm.step_with_metrics()
        tracer.finish_step(outputs, signed_tokens)
    rows = tracer.finish_profile()
    if len(rows) != steps:
        raise RuntimeError(f"expected {steps} rows, retained {len(rows)}")
    varlen_miss_after = int(getattr(llm.model_runner, "varlen_miss", 0))
    return {
        "batch_size": batch_size,
        "selected_decode_graph_batch_size": rows[0]["route"][
            "selected_decode_graph_batch_size"
        ],
        "routing": {
            "decode_only": True,
            "decode_graph_eager_fallback_possible": False,
            "varlen_miss_before": varlen_miss_before,
            "varlen_miss_after": varlen_miss_after,
            "varlen_miss_delta": varlen_miss_after - varlen_miss_before,
        },
        "steps": rows,
        "memory": {
            "allocated_before_bytes": allocated_before,
            "allocated_after_bytes": int(torch_module.cuda.memory_allocated()),
            "reserved_before_bytes": reserved_before,
            "reserved_after_bytes": int(torch_module.cuda.memory_reserved()),
            "peak_allocated_bytes": int(torch_module.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch_module.cuda.max_memory_reserved()),
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

    # Keep source-pin inspection available on CPU-only hosts.
    import torch
    import transformers

    from nanovllm import LLM, SamplingParams

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for decode-jitter diagnostics")
    if not hasattr(resource, "RUSAGE_THREAD"):
        raise RuntimeError("RUSAGE_THREAD is required for per-thread attribution")
    if not gc.isenabled():
        raise RuntimeError("start diagnostics in a fresh process with Python GC enabled")

    pin = validate_release_pin(args.expected_commit, args.expected_source_sha256)
    model = model_identity(Path(args.model))
    environment = environment_identity(torch, transformers)
    invocation = [
        sys.executable,
        *(sys.argv if argv is None else [str(Path(__file__).resolve()), *argv]),
    ]
    result = base_result(KIND, invocation, pin, model, environment)
    result.update({
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "arguments": vars(args) | {"output": str(args.output)},
        "certification": {
            "eligible": False,
            "classification": "intrusive_diagnostic_only",
            "reason": (
                "per-step host snapshots and CUDA events intentionally perturb "
                "the timed path; only uninstrumented full-completion artifacts "
                "can certify latency"
            ),
        },
        "randomness": {
            "seed": args.seed,
            "python_random_seed": args.seed,
            "torch_manual_seed": args.seed,
            "torch_cuda_manual_seed_all": args.seed,
        },
        "measurement_contract": {
            "host_wall_clock": "time.perf_counter_ns",
            "thread_cpu_clock": "time.thread_time_ns",
            "context_switch_scope": "resource.RUSAGE_THREAD",
            "cuda_event_completion_boundary": (
                "queried only after production sampler output tokens.tolist() "
                "has synchronized the runner_end event"
            ),
            "cuda_span_interpretation": (
                "stream-boundary elapsed spans include any idle interval while "
                "the host is delayed between event records; correlate them with "
                "phase wall, thread CPU, and context-switch deltas"
            ),
            "added_per_step_cuda_synchronize": False,
            "profiles": list(BATCH_SIZES),
        },
    })
    spec = workload_spec(args)
    result["workload"] = {"spec": spec}

    random_generator = random.Random(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    gc_before = gc.isenabled()
    engine_start_ns = perf_counter_ns()
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
    if gc.isenabled():
        raise RuntimeError("engine-owned Python GC lease was not active")
    actual_config = {
        "max_num_batched_tokens": llm.model_runner.config.max_num_batched_tokens,
        "max_num_seqs": llm.model_runner.config.max_num_seqs,
        "max_model_len": llm.model_runner.config.max_model_len,
        "gpu_memory_utilization": llm.model_runner.config.gpu_memory_utilization,
        "enforce_eager": llm.model_runner.config.enforce_eager,
        "tensor_parallel_size": llm.model_runner.config.tensor_parallel_size,
        "disable_python_gc": llm.model_runner.config.disable_python_gc,
    }
    for name, expected in (
        ("max_num_batched_tokens", spec["tau"]),
        ("max_num_seqs", spec["max_num_seqs"]),
        ("max_model_len", spec["max_model_len"]),
        ("gpu_memory_utilization", spec["gpu_memory_utilization"]),
        ("enforce_eager", spec["enforce_eager"]),
        ("tensor_parallel_size", spec["tensor_parallel_size"]),
        ("disable_python_gc", spec["disable_python_gc"]),
    ):
        if actual_config[name] != expected:
            raise RuntimeError(f"engine config differs from pinned workload: {name}")
    result["engine"] = {
        "initialization_ms": (perf_counter_ns() - engine_start_ns) / 1_000_000.0,
        "config": actual_config | {
            "decode_graph_batch_sizes": list(llm.model_runner.graph_bs),
        },
    }

    vocabulary = int(llm.model_runner.config.hf_config.vocab_size)
    if vocabulary < spec["prompt_token_max_exclusive"]:
        raise RuntimeError("model vocabulary is smaller than the pinned prompt range")

    def prompts(count: int) -> list[list[int]]:
        return [
            [
                random_generator.randrange(
                    spec["prompt_token_min_inclusive"],
                    spec["prompt_token_max_exclusive"],
                )
                for _ in range(spec["prompt_tokens"])
            ]
            for _ in range(count)
        ]

    initial_prompts = prompts(BATCH_SIZES[0])
    added_prompts = prompts(BATCH_SIZES[1] - BATCH_SIZES[0])
    result["workload"]["prompt_manifests"] = {
        "initial_16": _prompt_manifest("initial_16", initial_prompts),
        "added_2": _prompt_manifest("added_2", added_prompts),
    }
    params = SamplingParams(
        temperature=spec["temperature"],
        max_tokens=spec["max_completion_tokens"],
        ignore_eos=spec["ignore_eos"],
        top_k=spec["top_k"],
        top_p=spec["top_p"],
    )
    initial_ids = [llm.add_request(prompt, params) for prompt in initial_prompts]
    initial_conditioning = _condition_to_decode(llm, BATCH_SIZES[0])

    tracer = DecodeStepTracer(llm.scheduler, llm.model_runner, torch)
    tracer.install()
    try:
        profile16 = _run_profile(
            llm, tracer, torch, BATCH_SIZES[0], args.steps_per_batch
        )
        added_ids = [llm.add_request(prompt, params) for prompt in added_prompts]
        added_conditioning = _condition_to_decode(llm, BATCH_SIZES[1])
        profile18 = _run_profile(
            llm, tracer, torch, BATCH_SIZES[1], args.steps_per_batch
        )
    finally:
        tracer.uninstall()

    profiles = {"bs16": profile16, "bs18": profile18}
    for profile in profiles.values():
        profile["summary"] = summarize_profile(
            profile["steps"], args.spike_threshold_ms
        )
    result["profiles"] = profiles
    result["workload"].update({
        "initial_seq_ids": initial_ids,
        "added_seq_ids": added_ids,
        "conditioning_steps": {
            "to_bs16_decode": initial_conditioning,
            "from_bs16_to_bs18_decode": added_conditioning,
        },
    })

    llm.exit()
    if gc.isenabled() != gc_before:
        raise RuntimeError("engine exit did not restore the pre-init Python GC state")
    result["python_gc"] = {
        "disable_requested": True,
        "enabled_before_engine": gc_before,
        "enabled_after_engine_init": False,
        "enabled_after_engine_exit": gc.isenabled(),
    }
    if model_identity(Path(args.model)) != model:
        raise RuntimeError("model manifest changed during the diagnostic")
    result["model_after_run_matches"] = True
    result["provenance_after_run"] = validate_release_pin(
        args.expected_commit, args.expected_source_sha256
    )
    result["completed_at_utc"] = datetime.now(UTC).isoformat()
    immutable_write_json(args.output, result)
    print(json.dumps({
        "output": str(args.output),
        "profiles": {
            name: {
                "steps": profile["summary"]["step_count"],
                "spikes": profile["summary"]["spike_count"],
                "api_wall_ms": profile["summary"]["timings_ms"]["api_wall_ms"],
            }
            for name, profile in profiles.items()
        },
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
