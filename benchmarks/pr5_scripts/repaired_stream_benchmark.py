#!/usr/bin/env python3
"""Fresh-process evidence for the repaired synchronous streaming API.

The parent process never imports torch.  It starts one worker process per
observation so CUDA/model state cannot leak across observations.  Every worker
warms both public delivery paths, measures generate/stream in an alternating
order, rotates the slow-consumer order, and exercises correction-capable
TextUpdate application through one exact detokenizer flush.

This benchmark measures caller-visible delivery boundaries.  In particular,
``stream_first_event_seconds`` is not model TTFT: it includes public API work
through delivery of the first event and deliberately excludes the subsequent
stream drain.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter, sleep
from typing import Any


SCHEMA_VERSION = 1
DEFAULT_MODEL = "/workspace/models/Qwen3-0.6B"
PROMPT_BANK = (
    "The capital of France is",
    "def fibonacci(n):",
    "In 1969, humans first",
    "A short proof by induction begins",
    "The main difference between RAM and storage is",
    "Translate to French: good morning",
    "A recipe for tomato soup includes",
    "The next number in the sequence 2, 3, 5, 8 is",
)


def _run_git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _token_hash(token_ids: list[list[int]]) -> str:
    payload = json.dumps(token_ids, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty observation list")
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _seed_all(torch: Any, seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _cuda_start(torch: Any) -> float:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return perf_counter()


def _cuda_finish(torch: Any, started_at: float) -> tuple[float, dict[str, int]]:
    torch.cuda.synchronize()
    elapsed = perf_counter() - started_at
    return elapsed, {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def _measure_generate(llm: Any, torch: Any, prompts: list[str], params: Any, seed: int):
    _seed_all(torch, seed)
    started_at = _cuda_start(torch)
    outputs = llm.generate(prompts, params, use_tqdm=False)
    elapsed, memory = _cuda_finish(torch, started_at)
    token_ids = [output["token_ids"] for output in outputs]
    num_tokens = sum(len(tokens) for tokens in token_ids)
    if num_tokens != len(prompts) * params.max_tokens:
        raise AssertionError(f"generate produced {num_tokens} tokens unexpectedly")
    return {
        "return_seconds": elapsed,
        "tokens_per_second": num_tokens / elapsed,
        "num_tokens": num_tokens,
        "token_sha256": _token_hash(token_ids),
        "token_ids": token_ids,
        "memory": memory,
    }


def _metric_medians(metrics: dict[int, dict[str, Any]]) -> dict[str, float]:
    keys = (
        "caller_ttft",
        "caller_e2e",
        "engine_ttft",
        "engine_e2e",
        "first_token_to_delivery",
        "engine_finish_to_delivery",
    )
    return {
        key: statistics.median(item[key] for item in metrics.values())
        for key in keys
    }


def _measure_stream(llm: Any, torch: Any, prompts: list[str], params: Any, seed: int):
    _seed_all(torch, seed)
    started_at = _cuda_start(torch)
    first_event_seconds = None
    token_ids_by_seq: dict[int, list[int]] = {}
    with llm.stream(prompts, params) as session:
        seq_ids = session.seq_ids
        for event in session:
            if first_event_seconds is None:
                first_event_seconds = perf_counter() - started_at
            token_ids_by_seq.setdefault(event.seq_id, []).append(event.token_id)
        metrics = dict(session.metrics)
    elapsed, memory = _cuda_finish(torch, started_at)

    if first_event_seconds is None:
        raise AssertionError("stream produced no events")
    token_ids = [token_ids_by_seq[seq_id] for seq_id in seq_ids]
    num_tokens = sum(len(tokens) for tokens in token_ids)
    if num_tokens != len(prompts) * params.max_tokens:
        raise AssertionError(f"stream produced {num_tokens} events unexpectedly")
    if set(metrics) != set(seq_ids):
        raise AssertionError("stream delivery metrics did not cover every sequence")
    return {
        "first_event_seconds": first_event_seconds,
        "drained_seconds": elapsed,
        "tokens_per_second": num_tokens / elapsed,
        "num_events": num_tokens,
        "token_sha256": _token_hash(token_ids),
        "token_ids": token_ids,
        "metric_medians_seconds": _metric_medians(metrics),
        "memory": memory,
    }


def _measure_slow_stream(
    llm: Any,
    torch: Any,
    prompts: list[str],
    params: Any,
    seed: int,
    consumer_sleep_seconds: float,
):
    _seed_all(torch, seed)
    started_at = _cuda_start(torch)
    step_starts = []
    step_ids: list[int] = []
    max_pending_events = 0
    event_count = 0
    with llm.stream(prompts, params) as session:
        expected_ids = set(session.seq_ids)
        for event in session:
            if event_count % len(prompts) == 0:
                if step_ids and set(step_ids) != expected_ids:
                    raise AssertionError("stream step did not contain each sequence once")
                step_ids = []
                step_starts.append(perf_counter())
            step_ids.append(event.seq_id)
            event_count += 1
            max_pending_events = max(max_pending_events, len(session._pending))
            if consumer_sleep_seconds:
                sleep(consumer_sleep_seconds)
        if step_ids and set(step_ids) != expected_ids:
            raise AssertionError("final stream step did not contain each sequence once")
    elapsed, memory = _cuda_finish(torch, started_at)

    expected_events = len(prompts) * params.max_tokens
    if event_count != expected_events:
        raise AssertionError(f"slow stream produced {event_count}, expected {expected_events}")
    gaps = [later - earlier for earlier, later in zip(step_starts, step_starts[1:])]
    return {
        "consumer_sleep_seconds_per_event": consumer_sleep_seconds,
        "consumer_sleep_budget_seconds_per_step": (
            consumer_sleep_seconds * len(prompts)
        ),
        "elapsed_seconds": elapsed,
        "events_per_second": event_count / elapsed,
        "num_events": event_count,
        "num_steps": len(step_starts),
        "median_inter_step_gap_seconds": statistics.median(gaps),
        "p95_inter_step_gap_seconds": _percentile(gaps, 0.95),
        "max_pending_events_after_delivery": max_pending_events,
        "memory": memory,
    }


class _CountingTokenizer:

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self.decode_lengths: list[int] = []

    def decode(self, token_ids: list[int]) -> str:
        self.decode_lengths.append(len(token_ids))
        return self.tokenizer.decode(token_ids)


def _measure_detokenizer(
    tokenizer: Any,
    detokenizer_type: Any,
    lengths: list[int],
    repeats: int,
):
    source = (
        "The quick brown fox jumps over the lazy dog. "
        "Tokenizer cleanup can rewrite spaces and punctuation. "
    ) * 1000
    source_ids = tokenizer.encode(source)
    if len(source_ids) < max(lengths):
        raise AssertionError("detokenizer source did not contain enough tokens")

    results = []
    for num_tokens in lengths:
        timings = []
        structural = None
        final_hash = None
        for _ in range(repeats):
            counted = _CountingTokenizer(tokenizer)
            detokenizer = detokenizer_type(counted)
            rendered = ""
            corrections = 0
            started_at = perf_counter()
            for token_id in source_ids[:num_tokens]:
                update = detokenizer.feed(0, token_id)
                corrections += int(update.delete_count > 0)
                rendered = update.apply(rendered)
            feed_decode_lengths = list(counted.decode_lengths)
            final_update = detokenizer.flush(0)
            rendered = final_update.apply(rendered)
            elapsed = perf_counter() - started_at
            exact = tokenizer.decode(source_ids[:num_tokens])
            if rendered != exact or not final_update.final:
                raise AssertionError("TextUpdate application did not reconstruct exact text")

            timings.append(elapsed / num_tokens * 1e6)
            current_structural = {
                "max_feed_decode_tokens": max(feed_decode_lengths),
                "total_feed_decode_tokens": sum(feed_decode_lengths),
                "feed_decode_tokens_per_input_token": (
                    sum(feed_decode_lengths) / num_tokens
                ),
                "flush_decode_tokens": counted.decode_lengths[-1],
                "full_length_decode_calls": counted.decode_lengths.count(num_tokens),
                "correction_updates": corrections,
            }
            if structural is not None and structural != current_structural:
                raise AssertionError("detokenizer structural work changed across repeats")
            structural = current_structural
            final_hash = hashlib.sha256(rendered.encode()).hexdigest()

        results.append({
            "num_tokens": num_tokens,
            "us_per_token_repeats": timings,
            "median_us_per_token": statistics.median(timings),
            "text_sha256": final_hash,
            **structural,
        })
    return results


def _worker_environment(torch: Any, repo: Path, model: Path) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    model_config = model / "config.json"
    driver_version = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()[0]
    return {
        "repository": {
            "commit": _run_git(repo, "rev-parse", "HEAD"),
            "branch": _run_git(repo, "branch", "--show-current"),
            "worktree_dirty": bool(_run_git(repo, "status", "--porcelain")),
        },
        "benchmark_script_sha256": _sha256_file(Path(__file__).resolve()),
        "model": {
            "path": str(model.resolve()),
            "config_sha256": _sha256_file(model_config),
        },
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": _package_version("transformers"),
            "flash_attn": _package_version("flash-attn"),
        },
        "gpu": {
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "driver_version": driver_version,
            "total_memory_bytes": properties.total_memory,
            "device_count": torch.cuda.device_count(),
        },
    }


def _run_worker(args: argparse.Namespace) -> None:
    import torch
    from nanovllm import LLM, SamplingParams, StreamingDetokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("the repaired streaming benchmark requires CUDA")
    repo = Path(__file__).resolve().parents[2]
    model = Path(args.model)
    prompts = [PROMPT_BANK[index % len(PROMPT_BANK)] for index in range(args.batch_size)]
    slow_prompts = list(PROMPT_BANK[:args.slow_batch_size])
    llm = LLM(
        str(model),
        enforce_eager=False,
        max_model_len=args.max_model_len,
    )

    warmup_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.warmup_tokens,
        ignore_eos=True,
    )
    _measure_generate(llm, torch, prompts, warmup_params, args.seed)
    _measure_stream(llm, torch, prompts, warmup_params, args.seed)

    params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )
    core = {}
    order = ("generate", "stream") if args.order == "generate-stream" else ("stream", "generate")
    for route in order:
        if route == "generate":
            core[route] = _measure_generate(llm, torch, prompts, params, args.seed)
        else:
            core[route] = _measure_stream(llm, torch, prompts, params, args.seed)
    tokens_equivalent = core["generate"]["token_ids"] == core["stream"]["token_ids"]
    if not tokens_equivalent:
        raise AssertionError("seeded generate and stream token IDs diverged")
    del core["generate"]["token_ids"]
    del core["stream"]["token_ids"]

    slow_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.slow_tokens,
        ignore_eos=True,
    )
    _measure_slow_stream(
        llm,
        torch,
        slow_prompts,
        SamplingParams(
            temperature=args.temperature,
            max_tokens=min(4, args.slow_tokens),
            ignore_eos=True,
        ),
        args.seed,
        0.0,
    )
    sleep_values = [value / 1000.0 for value in args.sleep_ms]
    rotation = args.trial_index % len(sleep_values)
    sleep_order = sleep_values[rotation:] + sleep_values[:rotation]
    slow = [
        _measure_slow_stream(
            llm,
            torch,
            slow_prompts,
            slow_params,
            args.seed,
            sleep_seconds,
        )
        for sleep_seconds in sleep_order
    ]
    detokenizer = _measure_detokenizer(
        llm.tokenizer,
        StreamingDetokenizer,
        args.detok_lengths,
        args.detok_repeats,
    )

    result = {
        "trial_index": args.trial_index,
        "measurement_order": list(order),
        "slow_consumer_order_ms": [value * 1000.0 for value in sleep_order],
        "environment": _worker_environment(torch, repo, model),
        "core": {
            **core,
            "tokens_equivalent": tokens_equivalent,
            "paired_stream_throughput_delta_percent": (
                (core["stream"]["tokens_per_second"]
                 / core["generate"]["tokens_per_second"] - 1.0) * 100.0
            ),
            "caller_exposure_factor": (
                core["generate"]["return_seconds"]
                / core["stream"]["first_event_seconds"]
            ),
        },
        "slow_consumer": slow,
        "detokenizer": detokenizer,
    }
    output = Path(args.worker_output)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def _aggregate(workers: list[dict[str, Any]]) -> dict[str, Any]:
    generate_tps = [item["core"]["generate"]["tokens_per_second"] for item in workers]
    stream_tps = [item["core"]["stream"]["tokens_per_second"] for item in workers]
    generate_return = [item["core"]["generate"]["return_seconds"] for item in workers]
    stream_first = [item["core"]["stream"]["first_event_seconds"] for item in workers]

    sleep_values = sorted({
        item["consumer_sleep_seconds_per_event"]
        for worker in workers
        for item in worker["slow_consumer"]
    })
    slow = {}
    for sleep_value in sleep_values:
        observations = [
            item
            for worker in workers
            for item in worker["slow_consumer"]
            if item["consumer_sleep_seconds_per_event"] == sleep_value
        ]
        slow[f"{sleep_value * 1000.0:g}"] = {
            "consumer_sleep_ms_per_event": sleep_value * 1000.0,
            "inter_step_gap_ms": _summary([
                item["median_inter_step_gap_seconds"] * 1000.0
                for item in observations
            ]),
            "elapsed_seconds": _summary([item["elapsed_seconds"] for item in observations]),
            "max_pending_events_after_delivery": max(
                item["max_pending_events_after_delivery"] for item in observations
            ),
        }

    detok_lengths = [item["num_tokens"] for item in workers[0]["detokenizer"]]
    detokenizer = {}
    for num_tokens in detok_lengths:
        observations = [
            item
            for worker in workers
            for item in worker["detokenizer"]
            if item["num_tokens"] == num_tokens
        ]
        detokenizer[str(num_tokens)] = {
            "us_per_token": _summary([
                item["median_us_per_token"] for item in observations
            ]),
            "max_feed_decode_tokens": max(
                item["max_feed_decode_tokens"] for item in observations
            ),
            "max_feed_decode_tokens_per_input_token": max(
                item["feed_decode_tokens_per_input_token"] for item in observations
            ),
            "flush_decode_tokens": sorted({
                item["flush_decode_tokens"] for item in observations
            }),
            "full_length_decode_calls": sorted({
                item["full_length_decode_calls"] for item in observations
            }),
        }

    return {
        "null_consumer": {
            "generate_tokens_per_second": _summary(generate_tps),
            "stream_tokens_per_second": _summary(stream_tps),
            "paired_stream_delta_percent": _summary([
                item["core"]["paired_stream_throughput_delta_percent"]
                for item in workers
            ]),
        },
        "caller_delivery": {
            "generate_return_ms": _summary([value * 1000.0 for value in generate_return]),
            "stream_first_event_ms": _summary([value * 1000.0 for value in stream_first]),
            "paired_exposure_factor": _summary([
                item["core"]["caller_exposure_factor"] for item in workers
            ]),
            "boundary_note": (
                "Caller first-event exposure versus generate API return; "
                "this is not a model-TTFT comparison."
            ),
        },
        "slow_consumer": slow,
        "detokenizer": detokenizer,
        "gates": {
            "all_seeded_tokens_equivalent": all(
                item["core"]["tokens_equivalent"] for item in workers
            ),
            "all_worker_pins_identical": all(
                item["environment"] == workers[0]["environment"] for item in workers
            ),
        },
    }


def _parent_result(args: argparse.Namespace, workers: list[dict[str, Any]]) -> dict[str, Any]:
    environment = workers[0]["environment"]
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "repaired synchronous StreamSession and TextUpdate",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "fresh_worker_processes": args.runs,
            "paired_core_order": [worker["measurement_order"] for worker in workers],
            "slow_consumer_order_ms": [worker["slow_consumer_order_ms"] for worker in workers],
            "warmup_tokens_per_route": args.warmup_tokens,
            "batch_size": args.batch_size,
            "max_tokens": args.max_tokens,
            "slow_batch_size": args.slow_batch_size,
            "slow_tokens": args.slow_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
            "detokenizer_repeats_per_worker": args.detok_repeats,
            "detokenizer_lengths": args.detok_lengths,
        },
        "environment": environment,
        "aggregate": _aggregate(workers),
        "observations": workers,
    }


def _run_parent(args: argparse.Namespace) -> None:
    if args.runs < 3:
        raise ValueError("release evidence requires at least three fresh worker processes")
    if not 1 <= args.batch_size:
        raise ValueError("batch size must be positive")
    if not 1 <= args.slow_batch_size <= len(PROMPT_BANK):
        raise ValueError(f"slow batch size must be in [1, {len(PROMPT_BANK)}]")
    if args.slow_tokens < 2:
        raise ValueError("slow tokens must be at least two for inter-step gaps")
    if any(value < 0 for value in args.sleep_ms):
        raise ValueError("consumer sleep values cannot be negative")
    if any(value <= 0 for value in args.detok_lengths):
        raise ValueError("detokenizer lengths must be positive")
    if args.detok_repeats < 1:
        raise ValueError("detokenizer repeats must be positive")
    output = Path(args.output).resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output}; pass --overwrite explicitly")
    output.parent.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]

    workers = []
    with tempfile.TemporaryDirectory(prefix="repaired-stream-bench-") as temporary:
        for trial_index in range(args.runs):
            worker_output = Path(temporary) / f"worker-{trial_index}.json"
            order = "generate-stream" if trial_index % 2 == 0 else "stream-generate"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--worker-output", str(worker_output),
                "--trial-index", str(trial_index),
                "--order", order,
                "--model", args.model,
                "--runs", str(args.runs),
                "--batch-size", str(args.batch_size),
                "--max-tokens", str(args.max_tokens),
                "--slow-batch-size", str(args.slow_batch_size),
                "--slow-tokens", str(args.slow_tokens),
                "--warmup-tokens", str(args.warmup_tokens),
                "--temperature", str(args.temperature),
                "--seed", str(args.seed),
                "--max-model-len", str(args.max_model_len),
                "--detok-repeats", str(args.detok_repeats),
                "--sleep-ms", *[str(value) for value in args.sleep_ms],
                "--detok-lengths", *[str(value) for value in args.detok_lengths],
            ]
            environment = os.environ.copy()
            python_path = environment.get("PYTHONPATH")
            environment["PYTHONPATH"] = (
                str(repo) if not python_path else f"{repo}{os.pathsep}{python_path}"
            )
            completed = subprocess.run(
                command,
                cwd=repo,
                env=environment,
                capture_output=True,
                text=True,
            )
            if completed.returncode:
                raise RuntimeError(
                    f"worker {trial_index} failed with {completed.returncode}\n"
                    f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
                )
            workers.append(json.loads(worker_output.read_text()))
            print(f"completed fresh worker {trial_index + 1}/{args.runs}", flush=True)

    result = _parent_result(args, workers)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))
    print(f"wrote {output}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--slow-batch-size", type=int, default=8)
    parser.add_argument("--slow-tokens", type=int, default=32)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--sleep-ms", type=float, nargs="+", default=[0.0, 1.0, 4.0])
    parser.add_argument("--detok-lengths", type=int, nargs="+", default=[64, 256, 1024, 2048])
    parser.add_argument("--detok-repeats", type=int, default=3)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    parser.add_argument("--trial-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--order",
        choices=("generate-stream", "stream-generate"),
        default="generate-stream",
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.worker:
        if not args.worker_output:
            parser.error("--worker-output is required for a worker")
        _run_worker(args)
    else:
        if not args.output:
            parser.error("--output is required")
        _run_parent(args)


if __name__ == "__main__":
    main()
