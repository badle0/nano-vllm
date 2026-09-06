#!/usr/bin/env python3
"""Self-pinned CPU roofline for repaired chunk scheduler admission and cost."""

from __future__ import annotations

import argparse
from collections import deque
import gc
import json
import os
import platform
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns
from types import SimpleNamespace

from benchmarks.chunked_prefill_tail.common import (
    ROOT,
    handle_pin_query,
    immutable_write_json,
    nearest_rank_percentile,
    validate_release_pin,
)
from nanovllm import SamplingParams, SchedulerCapacityError
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus


BACKLOG_SIZES = (0, 100_000, 500_000)
DEFAULT_WARMUP = 2_000
DEFAULT_SAMPLES = 20_000
ABSOLUTE_MEDIAN_GATE_US = 5.0
RELATIVE_MEDIAN_GATE = 2.0


def _config(max_num_seqs: int = 2, tau: int = 2):
    return SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=tau,
        eos=-1,
        kvcache_block_size=256,
        num_kvcache_blocks=32,
    )


def _sequence(token_count: int = 2) -> Sequence:
    return Sequence(
        list(range(1, token_count + 1)),
        SamplingParams(max_tokens=8, ignore_eos=True),
    )


def _saturated_scheduler(backlog_size: int) -> Scheduler:
    scheduler = Scheduler(_config())
    waiter = _sequence()
    scheduler.waiting = deque([waiter] * backlog_size)
    for _ in range(2):
        sequence = _sequence()
        sequence.status = SequenceStatus.RUNNING
        sequence.is_prefill = False
        scheduler.running.append(sequence)
    scheduler.block_manager.can_append = lambda sequence: True
    scheduler.block_manager.may_append = lambda sequence: None
    return scheduler


def _measure(backlog_size: int, warmup: int, samples: int) -> dict:
    scheduler = _saturated_scheduler(backlog_size)
    for _ in range(warmup):
        scheduled, is_prefill = scheduler.schedule()
        if is_prefill or len(scheduled) != 2:
            raise AssertionError("saturated scheduler left the decode-only route")

    timings_ns = []
    for _ in range(samples):
        started = perf_counter_ns()
        scheduled, is_prefill = scheduler.schedule()
        timings_ns.append(perf_counter_ns() - started)
        if is_prefill or len(scheduled) != 2:
            raise AssertionError("saturated scheduler left the decode-only route")

    timings_us = [value / 1_000.0 for value in timings_ns]
    return {
        "backlog_size": backlog_size,
        "warmup_calls": warmup,
        "sample_count": samples,
        "raw_nanoseconds": timings_ns,
        "microseconds": {
            "min": min(timings_us),
            "median": statistics.median(timings_us),
            "p95": nearest_rank_percentile(timings_us, 0.95),
            "p99": nearest_rank_percentile(timings_us, 0.99),
            "max": max(timings_us),
        },
        "post_state": {
            "running": len(scheduler.running),
            "waiting": len(scheduler.waiting),
            "mid_chunk_seq": scheduler.mid_chunk_seq is not None,
            "all_scheduled_counts_nonnegative": all(
                sequence.num_scheduled_tokens >= 0 for sequence in scheduler.running
            ),
        },
    }


def _correctness_contracts() -> dict:
    bounded = Scheduler(_config())
    accepted = [_sequence(), _sequence()]
    for sequence in accepted:
        bounded.add(sequence)
    rejected = _sequence()
    try:
        bounded.add(rejected)
    except SchedulerCapacityError as error:
        capacity = {
            "rejected": True,
            "requested": error.requested,
            "available": error.available,
            "capacity": error.capacity,
            "queue_unchanged": list(bounded.waiting) == accepted,
        }
    else:
        raise AssertionError("scheduler admitted beyond max_num_seqs")

    defensive = Scheduler(_config(max_num_seqs=4, tau=4))
    waiter = _sequence(token_count=3)
    defensive.add(waiter)
    for _ in range(3):
        sequence = _sequence()
        defensive.block_manager.allocate(sequence, 0)
        sequence.status = SequenceStatus.RUNNING
        sequence.is_prefill = False
        defensive.running.append(sequence)
    defensive._max_num_batched_tokens = 2
    scheduled, is_prefill = defensive.schedule()
    nonpositive = {
        "decode_only": not is_prefill,
        "scheduled_rows": len(scheduled),
        "waiter_unscheduled": waiter.num_scheduled_tokens == 0,
        "waiter_owns_no_blocks": not waiter.block_table,
        "all_scheduled_counts_nonnegative": all(
            sequence.num_scheduled_tokens >= 0 for sequence in scheduled
        ),
    }

    read_only = Scheduler(_config())
    try:
        read_only.max_num_batched_tokens = 1
    except AttributeError:
        budget_read_only = True
    else:
        budget_read_only = False

    gates = {
        "bounded_admission": capacity["rejected"] and capacity["queue_unchanged"],
        "nonpositive_remaining_schedules_no_waiter": all(nonpositive.values()),
        "constructor_budget_is_read_only": budget_read_only,
    }
    return {
        "capacity": capacity,
        "nonpositive_remaining": nonpositive,
        "constructor_budget_is_read_only": budget_read_only,
        "gates": gates | {"all_pass": all(gates.values())},
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--expected-commit")
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--print-source-sha256", action="store_true")
    parser.add_argument("--show-pin", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if handle_pin_query(args.print_source_sha256, args.show_pin):
        return 0
    if args.warmup < 1 or args.samples < 100:
        raise ValueError("--warmup must be positive and --samples must be at least 100")
    if args.output is None or not args.expected_commit or not args.expected_source_sha256:
        raise ValueError("release run requires --output and both expected pins")
    output = args.output.expanduser().resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("release output must be outside the repository worktree")

    pin = validate_release_pin(args.expected_commit, args.expected_source_sha256)
    rng = random.Random(args.seed)
    order = list(BACKLOG_SIZES)
    rng.shuffle(order)
    gc_enabled_before = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        rows_by_size = {
            backlog_size: _measure(backlog_size, args.warmup, args.samples)
            for backlog_size in order
        }
        contracts = _correctness_contracts()
    finally:
        if gc_enabled_before:
            gc.enable()
        else:
            gc.disable()

    rows = [rows_by_size[size] for size in BACKLOG_SIZES]
    zero_median = rows_by_size[0]["microseconds"]["median"]
    large_median = rows_by_size[500_000]["microseconds"]["median"]
    threshold = max(
        ABSOLUTE_MEDIAN_GATE_US,
        zero_median * RELATIVE_MEDIAN_GATE,
    )
    cost_gate = large_median <= threshold
    result = {
        "schema_version": 1,
        "kind": "chunk_scheduler_backlog_roofline",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "argv": list(sys.argv if argv is None else [sys.argv[0], *argv]),
        "seed": args.seed,
        "measurement_order": order,
        "python_gc": {
            "enabled_before": gc_enabled_before,
            "disabled_during_timing": True,
            "enabled_after": gc.isenabled(),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_model": next(
                (
                    line.split(":", 1)[1].strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")
                ),
                platform.processor(),
            ),
            "cpu_count": os.cpu_count(),
            "python_executable": sys.executable,
        },
        "protocol": {
            "backlog_sizes": list(BACKLOG_SIZES),
            "warmup_calls_per_size": args.warmup,
            "samples_per_size": args.samples,
            "backlog_construction_excluded": True,
            "running_decode_rows": 2,
        },
        "provenance": pin,
        "correctness": contracts,
        "measurements": rows,
        "roofline": {
            "zero_backlog_median_us": zero_median,
            "five_hundred_thousand_median_us": large_median,
            "relative_ratio": large_median / zero_median,
            "gate_threshold_us": threshold,
            "absolute_gate_us": ABSOLUTE_MEDIAN_GATE_US,
            "relative_gate": RELATIVE_MEDIAN_GATE,
            "passes": cost_gate,
        },
        "gates": {
            "correctness": contracts["gates"]["all_pass"],
            "approximately_constant_schedule_cost": cost_gate,
            "python_gc_state_restored": gc.isenabled() == gc_enabled_before,
        },
    }
    result["gates"]["all_pass"] = all(result["gates"].values())
    result["provenance_after_run"] = validate_release_pin(
        args.expected_commit, args.expected_source_sha256
    )
    immutable_write_json(output, result)
    print(json.dumps({
        "output": str(output),
        "medians_us": {
            str(row["backlog_size"]): row["microseconds"]["median"] for row in rows
        },
        "p95_us": {
            str(row["backlog_size"]): row["microseconds"]["p95"] for row in rows
        },
        "gates": result["gates"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
