#!/usr/bin/env python3
"""Fresh-process evidence for the repaired synchronous streaming API.

The parent process never imports torch.  It starts one worker process per
statistical unit so CUDA/model state cannot leak across units.  Every worker
warms both public delivery paths at the full measured length, then takes the
median of four cache-neutral paired generate/stream efficiency deltas with two
rounds in each route order and distinct prompt/seed pairs.  It also rotates the
slow-consumer order and exercises correction-capable TextUpdate application
through one exact detokenizer flush.

This benchmark measures caller-visible delivery boundaries.  In particular,
``stream_first_event_seconds`` is not model TTFT: it includes public API work
through delivery of the first event and deliberately excludes the subsequent
stream drain.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
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
from time import perf_counter, process_time, sleep
from typing import Any


SCHEMA_VERSION = 3
DEFAULT_MODEL = "/workspace/models/Qwen3-0.6B"
SOURCE_PATHS = ("nanovllm", "benchmarks/pr5_scripts/repaired_stream_benchmark.py")
CORE_TIMED_ROUNDS = 4
CORE_PREFIX_CACHE_POLICY = "fresh_block_manager_before_each_timed_route"
CORRECTION_WINDOW_SIZE = 32
CORRECTION_BOUNDARY_OVERLAP = 8
CORRECTION_TIME_SLOPE_LIMIT = 1.5
CORRECTION_STATE_SLOPE_LIMIT = 1.125
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

PROMPT_VARIANTS = (
    "Give one concise continuation.",
    "Continue with a concrete detail.",
    "Complete the thought in plain language.",
    "Respond with a short factual continuation.",
    "Add the most likely next sentence.",
    "Continue without a preamble.",
    "Supply a compact completion.",
    "Finish this prompt directly.",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _trial_seed(base_seed: int, trial_index: int) -> int:
    return base_seed + 1009 * trial_index


def _trial_prompts(batch_size: int, trial_index: int) -> list[str]:
    prompts = []
    for index in range(batch_size):
        base = PROMPT_BANK[(trial_index + index) % len(PROMPT_BANK)]
        variant = PROMPT_VARIANTS[(trial_index * 3 + index) % len(PROMPT_VARIANTS)]
        prompts.append(f"{base}. {variant} [case {trial_index:02d}-{index:03d}]")
    return prompts


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


def _source_fingerprint(repo: Path) -> dict[str, Any]:
    lines = _run_git(repo, "ls-files", "-s", "--", *SOURCE_PATHS).splitlines()
    blobs = []
    for line in lines:
        metadata, relative_path = line.split("\t", 1)
        mode, object_id, stage = metadata.split()
        blobs.append({
            "path": relative_path,
            "mode": mode,
            "git_blob": object_id,
            "stage": int(stage),
        })
    return {
        "git_tree": _run_git(repo, "rev-parse", "HEAD^{tree}"),
        "manifest_sha256": _canonical_sha256(blobs),
        "tracked_blobs": blobs,
    }


def _repository_fingerprint(repo: Path) -> dict[str, Any]:
    status = _run_git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    return {
        "commit": _run_git(repo, "rev-parse", "HEAD"),
        "branch": _run_git(repo, "branch", "--show-current"),
        "worktree_dirty": bool(status),
        "status_porcelain": status.splitlines(),
        "source": _source_fingerprint(repo),
    }


def _model_fingerprint(model: Path) -> dict[str, Any]:
    files = []
    for path in sorted(item for item in model.rglob("*") if item.is_file()):
        files.append({
            "path": path.relative_to(model).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        })
    if not files:
        raise ValueError(f"model directory contains no files: {model}")
    return {
        "path": str(model.resolve()),
        "files": files,
        "manifest_sha256": _canonical_sha256(files),
        "total_size_bytes": sum(item["size_bytes"] for item in files),
    }


def _write_json_exclusive(path: Path, value: Any) -> None:
    """Atomically publish JSON without ever replacing an existing artifact."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite immutable result: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _token_hash(token_ids: list[list[int]]) -> str:
    payload = json.dumps(token_ids, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _text_hash(texts: list[str]) -> str:
    return _canonical_sha256(texts)


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
        "p99": _percentile(values, 0.99),
        "max": max(values),
    }


def _stream_peak_within_one_percent_of_generate(
    stream_peak_bytes: int,
    generate_peak_bytes: int,
) -> bool:
    """Compare paired peaks without turning a zero baseline into a free allowance."""
    if stream_peak_bytes < 0 or generate_peak_bytes < 0:
        return False
    if generate_peak_bytes == 0:
        return stream_peak_bytes == 0
    return stream_peak_bytes <= generate_peak_bytes * 1.01


def _reset_core_prefix_cache(
    llm: Any,
    block_manager_type: type,
    route: str,
    pair_position: int,
) -> dict[str, Any]:
    """Install fresh prefix metadata only when no request owns a KV block."""
    scheduler = llm.scheduler
    if not scheduler.is_finished():
        raise AssertionError("core prefix-cache reset requires an idle scheduler")
    if getattr(llm, "_active_session", None) is not None:
        raise AssertionError("core prefix-cache reset requires no active API session")
    previous = scheduler.block_manager
    if previous.used_block_ids:
        raise AssertionError("idle scheduler still owns KV-cache blocks")
    if len(previous.free_block_ids) != len(previous.blocks):
        raise AssertionError("idle block manager does not expose every block as free")
    if any(block.ref_count != 0 for block in previous.blocks):
        raise AssertionError("idle block manager retained a nonzero block reference")

    replacement = block_manager_type(len(previous.blocks), previous.block_size)
    if replacement.used_block_ids or replacement.hash_to_block_id:
        raise AssertionError("fresh block manager unexpectedly contains cache state")
    scheduler.block_manager = replacement
    return {
        "policy": CORE_PREFIX_CACHE_POLICY,
        "route": route,
        "pair_position": pair_position,
        "num_blocks": len(previous.blocks),
        "block_size": previous.block_size,
        "discarded_cached_block_hashes": len(previous.hash_to_block_id),
        "used_blocks_before_reset": 0,
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _seed_all(torch: Any, seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _rss_bytes() -> int:
    resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def _cuda_start(torch: Any) -> dict[str, Any]:
    gc.collect()
    torch.cuda.synchronize()
    gpu_start = {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
    }
    torch.cuda.reset_peak_memory_stats()
    return {
        "perf_counter": perf_counter(),
        "started_at_utc": _utc_now(),
        "gpu": gpu_start,
        "rss_bytes": _rss_bytes(),
    }


def _cuda_finish(
    torch: Any,
    started_at: dict[str, Any],
) -> tuple[float, dict[str, Any], str]:
    torch.cuda.synchronize()
    elapsed = perf_counter() - started_at["perf_counter"]
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    memory = {
        "start": started_at["gpu"],
        "end": {
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
        },
        "peak": {
            "allocated_bytes": peak_allocated,
            "reserved_bytes": peak_reserved,
        },
        "peak_increment_from_start": {
            "allocated_bytes": max(
                0, peak_allocated - started_at["gpu"]["allocated_bytes"]
            ),
            "reserved_bytes": max(
                0, peak_reserved - started_at["gpu"]["reserved_bytes"]
            ),
        },
        "rss": {
            "start_bytes": started_at["rss_bytes"],
            "end_bytes": _rss_bytes(),
        },
    }
    return elapsed, memory, _utc_now()


def _measure_generate(
    llm: Any,
    torch: Any,
    prompts: list[str],
    params: Any,
    seed: int,
    pair_position: int | None = None,
):
    _seed_all(torch, seed)
    started_at = _cuda_start(torch)
    outputs = llm.generate(prompts, params, use_tqdm=False)
    elapsed, memory, finished_at_utc = _cuda_finish(torch, started_at)
    token_ids = [output["token_ids"] for output in outputs]
    texts = [output["text"] for output in outputs]
    num_tokens = sum(len(tokens) for tokens in token_ids)
    if num_tokens != len(prompts) * params.max_tokens:
        raise AssertionError(f"generate produced {num_tokens} tokens unexpectedly")
    return {
        "pair_position": pair_position,
        "started_at_utc": started_at["started_at_utc"],
        "finished_at_utc": finished_at_utc,
        "matched_detokenization": "one final full decode per sequence",
        "return_seconds": elapsed,
        "tokens_per_second": num_tokens / elapsed,
        "num_tokens": num_tokens,
        "token_sha256": _token_hash(token_ids),
        "text_sha256": _text_hash(texts),
        "token_ids": token_ids,
        "texts": texts,
        "memory": memory,
    }


def _metric_distributions(
    metrics: dict[int, dict[str, Any]],
    seq_ids: tuple[int, ...],
) -> dict[str, dict[str, Any]]:
    keys = (
        "caller_ttft",
        "caller_e2e",
        "engine_ttft",
        "engine_e2e",
        "first_token_to_delivery",
        "engine_finish_to_delivery",
    )
    return {
        key: {
            "values": [metrics[seq_id][key] for seq_id in seq_ids],
            "summary": _summary([metrics[seq_id][key] for seq_id in seq_ids]),
        }
        for key in keys
    }


def _measure_stream(
    llm: Any,
    torch: Any,
    prompts: list[str],
    params: Any,
    seed: int,
    pair_position: int | None = None,
):
    _seed_all(torch, seed)
    started_at = _cuda_start(torch)
    first_event_seconds = None
    token_ids_by_seq: dict[int, list[int]] = {}
    event_delivery_rows = []
    with llm.stream(prompts, params) as session:
        seq_ids = session.seq_ids
        logical_index = {seq_id: index for index, seq_id in enumerate(seq_ids)}
        for event in session:
            received_at = perf_counter()
            if first_event_seconds is None:
                first_event_seconds = received_at - started_at["perf_counter"]
            sequence_tokens = token_ids_by_seq.setdefault(event.seq_id, [])
            token_index = len(sequence_tokens)
            engine_token_at = session._sequences[event.seq_id].token_times[token_index]
            event_delivery_rows.append((
                logical_index[event.seq_id],
                token_index,
                event.finished,
                engine_token_at,
                received_at,
            ))
            sequence_tokens.append(event.token_id)
        metrics = dict(session.metrics)

    if first_event_seconds is None:
        raise AssertionError("stream produced no events")
    token_ids = [token_ids_by_seq[seq_id] for seq_id in seq_ids]
    # generate() performs one full decode per completed sequence before return.
    # Do the same work before stopping the stream route timer.
    texts = [llm.tokenizer.decode(tokens) for tokens in token_ids]
    elapsed, memory, finished_at_utc = _cuda_finish(torch, started_at)
    event_deliveries = [
        {
            "sequence_index": sequence_index,
            "token_index": token_index,
            "finished": finished,
            "engine_token_seconds_since_route_start": (
                engine_token_at - started_at["perf_counter"]
            ),
            "caller_receive_seconds_since_route_start": (
                received_at - started_at["perf_counter"]
            ),
            "engine_to_caller_seconds": received_at - engine_token_at,
        }
        for (
            sequence_index,
            token_index,
            finished,
            engine_token_at,
            received_at,
        ) in event_delivery_rows
    ]
    num_tokens = sum(len(tokens) for tokens in token_ids)
    if num_tokens != len(prompts) * params.max_tokens:
        raise AssertionError(f"stream produced {num_tokens} events unexpectedly")
    if set(metrics) != set(seq_ids):
        raise AssertionError("stream delivery metrics did not cover every sequence")
    event_delays = [item["engine_to_caller_seconds"] for item in event_deliveries]
    return {
        "pair_position": pair_position,
        "started_at_utc": started_at["started_at_utc"],
        "finished_at_utc": finished_at_utc,
        "matched_detokenization": "one final full decode per sequence",
        "first_event_seconds": first_event_seconds,
        "drained_seconds": elapsed,
        "tokens_per_second": num_tokens / elapsed,
        "num_events": num_tokens,
        "token_sha256": _token_hash(token_ids),
        "text_sha256": _text_hash(texts),
        "token_ids": token_ids,
        "texts": texts,
        "request_metric_distributions_seconds": _metric_distributions(
            metrics, seq_ids
        ),
        "event_delivery": {
            "values": event_deliveries,
            "engine_to_caller_seconds": _summary(event_delays),
        },
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
    elapsed, memory, finished_at_utc = _cuda_finish(torch, started_at)

    expected_events = len(prompts) * params.max_tokens
    if event_count != expected_events:
        raise AssertionError(f"slow stream produced {event_count}, expected {expected_events}")
    gaps = [later - earlier for earlier, later in zip(step_starts, step_starts[1:])]
    return {
        "started_at_utc": started_at["started_at_utc"],
        "finished_at_utc": finished_at_utc,
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


class _CorrectionTokenizer:

    def __init__(self):
        self.decode_lengths: list[int] = []

    @staticmethod
    def render(token_ids: list[int]) -> str:
        pieces = "".join(" " if token_id == 0 else "." for token_id in token_ids)
        return pieces.replace(" .", ".")

    def decode(self, token_ids: list[int]) -> str:
        self.decode_lengths.append(len(token_ids))
        return self.render(token_ids)


def _detokenizer_state_bytes(detokenizer: Any, seq_id: int) -> int:
    state = detokenizer._states[seq_id]
    return sum((
        sys.getsizeof(detokenizer._states),
        sys.getsizeof(state),
        sys.getsizeof(state.token_ids),
        sys.getsizeof(state.window_text),
    ))


def _measure_correction_scaling(
    detokenizer_type: Any,
    lengths: list[int],
    repeats: int,
) -> dict[str, Any]:
    if len(lengths) != 2 or lengths != sorted(set(lengths)):
        raise ValueError("correction lengths must be two distinct ascending values")
    hard_limit = CORRECTION_WINDOW_SIZE + 2 * CORRECTION_BOUNDARY_OVERLAP
    results = []
    for num_tokens in lengths:
        token_ids = [index % 2 for index in range(num_tokens)]
        timings = []
        structural = None
        for _ in range(repeats):
            tokenizer = _CorrectionTokenizer()
            detokenizer = detokenizer_type(
                tokenizer,
                window_size=CORRECTION_WINDOW_SIZE,
                boundary_overlap=CORRECTION_BOUNDARY_OVERLAP,
            )
            rendered = ""
            corrections = 0
            started_at = process_time()
            for token_id in token_ids:
                update = detokenizer.feed(0, token_id)
                corrections += int(update.delete_count > 0)
                rendered = update.apply(rendered)
            feed_process_seconds = process_time() - started_at
            state_bytes = _detokenizer_state_bytes(detokenizer, 0)
            state_token_count = len(detokenizer._states[0].token_ids)
            feed_decode_lengths = list(tokenizer.decode_lengths)
            flush_started_at = process_time()
            final_update = detokenizer.flush(0)
            rendered = final_update.apply(rendered)
            process_seconds = (
                feed_process_seconds + process_time() - flush_started_at
            )
            exact = _CorrectionTokenizer.render(token_ids)
            current_structural = {
                "exact_final_text": rendered == exact and final_update.final,
                "correction_updates": corrections,
                "correction_fraction": corrections / num_tokens,
                "max_feed_decode_tokens": max(feed_decode_lengths),
                "total_feed_decode_tokens": sum(feed_decode_lengths),
                "flush_decode_tokens": tokenizer.decode_lengths[-1],
                "full_length_decode_calls": tokenizer.decode_lengths.count(num_tokens),
                "state_bytes_before_flush": state_bytes,
                "state_token_count_before_flush": state_token_count,
                "state_released_after_flush": not detokenizer._states,
                "rendered_code_points": len(rendered),
            }
            if structural is not None and structural != current_structural:
                raise AssertionError("correction scaling structure changed across repeats")
            structural = current_structural
            timings.append(process_seconds)
        results.append({
            "num_tokens": num_tokens,
            "process_seconds_repeats": timings,
            "median_process_seconds": statistics.median(timings),
            "median_us_per_token": statistics.median(timings) / num_tokens * 1e6,
            **structural,
        })

    small, large = results
    token_ratio = large["num_tokens"] / small["num_tokens"]
    time_ratio = large["median_process_seconds"] / small["median_process_seconds"]
    state_ratio = (
        large["state_bytes_before_flush"] / small["state_bytes_before_flush"]
    )
    gates = {
        "exact_and_state_released": all(
            item["exact_final_text"] and item["state_released_after_flush"]
            for item in results
        ),
        "correction_fraction_at_least_0_49": all(
            item["correction_fraction"] >= 0.49 for item in results
        ),
        "incremental_decode_length_bounded": all(
            item["max_feed_decode_tokens"] <= hard_limit for item in results
        ),
        "exactly_one_full_length_flush": all(
            item["flush_decode_tokens"] == item["num_tokens"]
            and item["full_length_decode_calls"] == 1
            for item in results
        ),
        "process_time_scaling_within_roofline": (
            time_ratio <= token_ratio * CORRECTION_TIME_SLOPE_LIMIT
        ),
        "state_scaling_within_linear_roofline": (
            state_ratio <= token_ratio * CORRECTION_STATE_SLOPE_LIMIT
        ),
    }
    return {
        "description": (
            "alternating space/punctuation forces a suffix correction on half "
            "of feeds; timing includes every immutable TextUpdate.apply"
        ),
        "window_size": CORRECTION_WINDOW_SIZE,
        "boundary_overlap": CORRECTION_BOUNDARY_OVERLAP,
        "hard_incremental_decode_limit": hard_limit,
        "results": results,
        "scaling": {
            "token_ratio": token_ratio,
            "process_time_ratio": time_ratio,
            "state_bytes_ratio": state_ratio,
            "process_time_ratio_limit": (
                token_ratio * CORRECTION_TIME_SLOPE_LIMIT
            ),
            "state_bytes_ratio_limit": (
                token_ratio * CORRECTION_STATE_SLOPE_LIMIT
            ),
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def _load_detokenizer_type(repo: Path) -> Any:
    module_path = repo / "nanovllm/utils/streaming_detokenizer.py"
    module_name = "_streaming_detokenizer_certification"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load detokenizer from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.StreamingDetokenizer


def _cpu_model() -> str | None:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return None


def _worker_environment(
    torch: Any,
    repo: Path,
    model: Path,
    expected_model_manifest_sha256: str,
) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    model_config = model / "config.json"
    gpu_row = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,driver_version,name",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()[0]
    gpu_uuid, driver_version, nvidia_smi_name = (
        item.strip() for item in gpu_row.split(",", 2)
    )
    return {
        "repository": _repository_fingerprint(repo),
        "benchmark_script_sha256": _sha256_file(Path(__file__).resolve()),
        "model": {
            "path": str(model.resolve()),
            "config_sha256": _sha256_file(model_config),
            "parent_manifest_sha256": expected_model_manifest_sha256,
        },
        "software": {
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": _package_version("transformers"),
            "tokenizers": _package_version("tokenizers"),
            "flash_attn": _package_version("flash-attn"),
            "triton": _package_version("triton"),
        },
        "cpu": {"model": _cpu_model()},
        "gpu": {
            "name": properties.name,
            "nvidia_smi_name": nvidia_smi_name,
            "uuid": gpu_uuid,
            "compute_capability": [properties.major, properties.minor],
            "driver_version": driver_version,
            "total_memory_bytes": properties.total_memory,
            "device_count": torch.cuda.device_count(),
        },
        "selected_environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "PYTHONHASHSEED",
                "PYTHONPATH",
                "TORCHINDUCTOR_CACHE_DIR",
            )
        },
        "argv": list(sys.argv),
    }


def _assert_subblock_route_released(
    llm: Any,
    reset_record: dict[str, Any],
) -> None:
    scheduler = llm.scheduler
    if not scheduler.is_finished() or scheduler.block_manager.used_block_ids:
        raise AssertionError("completed core route retained scheduler or KV-block ownership")
    cached_hashes = len(scheduler.block_manager.hash_to_block_id)
    if cached_hashes:
        raise AssertionError("sub-block core route unexpectedly populated prefix cache")
    reset_record["used_blocks_after_route"] = 0
    reset_record["cached_block_hashes_after_route"] = cached_hashes


def _run_worker(args: argparse.Namespace) -> None:
    worker_started_at_utc = _utc_now()
    import torch
    from nanovllm import LLM, SamplingParams, StreamingDetokenizer
    from nanovllm.engine.block_manager import BlockManager

    if not torch.cuda.is_available():
        raise RuntimeError("the repaired streaming benchmark requires CUDA")
    repo = Path(__file__).resolve().parents[2]
    model = Path(args.model)
    trial_seed = _trial_seed(args.seed, args.trial_index)
    slow_prompts = _trial_prompts(
        args.batch_size, 50_000 + args.trial_index
    )[:args.slow_batch_size]
    environment = _worker_environment(
        torch,
        repo,
        model,
        args.expected_model_manifest_sha256,
    )
    repository = environment["repository"]
    if repository["worktree_dirty"]:
        raise RuntimeError("release worker requires a clean repository")
    if repository["commit"] != args.expected_commit:
        raise RuntimeError("worker HEAD changed from the parent release commit")
    if (
        repository["source"]["manifest_sha256"]
        != args.expected_source_manifest_sha256
    ):
        raise RuntimeError("worker source manifest changed from the parent")
    if environment["benchmark_script_sha256"] != args.expected_script_sha256:
        raise RuntimeError("worker benchmark script changed from the parent")
    llm = LLM(
        str(model),
        enforce_eager=False,
        max_model_len=args.max_model_len,
    )

    warmup_prompts = _trial_prompts(args.batch_size, 10_000 + args.trial_index)
    warmup_prompt_lengths = [len(llm.tokenizer.encode(item)) for item in warmup_prompts]
    if max(warmup_prompt_lengths) + args.max_tokens >= llm.scheduler.block_size:
        raise AssertionError("warmup prompt plus completion must remain below one KV block")
    warmup_seed = trial_seed ^ 0x5A5A5A5A
    warmup_records = []
    for warmup_tokens in (args.warmup_tokens, args.max_tokens):
        warmup_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=warmup_tokens,
            ignore_eos=True,
        )
        for pair_position, route in enumerate(("generate", "stream")):
            reset_record = _reset_core_prefix_cache(
                llm, BlockManager, route, pair_position
            )
            if route == "generate":
                _measure_generate(
                    llm, torch, warmup_prompts, warmup_params, warmup_seed
                )
            else:
                _measure_stream(
                    llm, torch, warmup_prompts, warmup_params, warmup_seed
                )
            _assert_subblock_route_released(llm, reset_record)
            warmup_records.append({
                "route": route,
                "max_tokens": warmup_tokens,
                "prefix_cache_reset": reset_record,
            })

    params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )
    initial_order = (
        ("generate", "stream")
        if args.order == "generate-stream"
        else ("stream", "generate")
    )
    timed_rounds = []
    for round_index in range(args.core_rounds):
        prompt_index = args.trial_index * args.core_rounds + round_index
        prompts = _trial_prompts(args.batch_size, prompt_index)
        prompt_lengths = [len(llm.tokenizer.encode(item)) for item in prompts]
        max_prompt_plus_completion = max(prompt_lengths) + args.max_tokens
        if max_prompt_plus_completion >= llm.scheduler.block_size:
            raise AssertionError("timed prompt plus completion must remain below one KV block")
        round_seed = trial_seed + 313 * round_index
        order = initial_order if round_index % 2 == 0 else tuple(reversed(initial_order))
        routes = {}
        resets = []
        for pair_position, route in enumerate(order):
            reset_record = _reset_core_prefix_cache(
                llm, BlockManager, route, pair_position
            )
            if route == "generate":
                routes[route] = _measure_generate(
                    llm, torch, prompts, params, round_seed, pair_position
                )
            else:
                routes[route] = _measure_stream(
                    llm, torch, prompts, params, round_seed, pair_position
                )
            _assert_subblock_route_released(llm, reset_record)
            resets.append(reset_record)

        tokens_equivalent = (
            routes["generate"]["token_ids"] == routes["stream"]["token_ids"]
        )
        texts_equivalent = (
            routes["generate"]["texts"] == routes["stream"]["texts"]
        )
        if not tokens_equivalent or not texts_equivalent:
            raise AssertionError("seeded generate and stream outputs diverged")
        for route in ("generate", "stream"):
            del routes[route]["token_ids"]
            del routes[route]["texts"]
        timed_rounds.append({
            "round_index": round_index,
            "round_seed": round_seed,
            "prompts": prompts,
            "prompt_sha256": _canonical_sha256(prompts),
            "prompt_token_lengths": prompt_lengths,
            "max_prompt_plus_completion_tokens": max_prompt_plus_completion,
            "block_size": llm.scheduler.block_size,
            "measurement_order": list(order),
            "prefix_cache_resets": resets,
            **routes,
            "tokens_equivalent": tokens_equivalent,
            "texts_equivalent": texts_equivalent,
            "paired_stream_throughput_delta_percent": (
                (routes["stream"]["tokens_per_second"]
                 / routes["generate"]["tokens_per_second"] - 1.0) * 100.0
            ),
            "caller_exposure_factor": (
                routes["generate"]["return_seconds"]
                / routes["stream"]["first_event_seconds"]
            ),
        })

    generate_tps_median = statistics.median(
        item["generate"]["tokens_per_second"] for item in timed_rounds
    )
    stream_tps_median = statistics.median(
        item["stream"]["tokens_per_second"] for item in timed_rounds
    )
    core = {
        "timed_rounds": timed_rounds,
        "route_medians": {
            "generate": {
                "tokens_per_second": generate_tps_median,
                "return_seconds": statistics.median(
                    item["generate"]["return_seconds"] for item in timed_rounds
                ),
            },
            "stream": {
                "tokens_per_second": stream_tps_median,
                "drained_seconds": statistics.median(
                    item["stream"]["drained_seconds"] for item in timed_rounds
                ),
                "first_event_seconds": statistics.median(
                    item["stream"]["first_event_seconds"] for item in timed_rounds
                ),
            },
        },
        "tokens_equivalent": all(item["tokens_equivalent"] for item in timed_rounds),
        "texts_equivalent": all(item["texts_equivalent"] for item in timed_rounds),
        "matched_work": "both routes perform one final full tokenizer decode per sequence",
        "prefix_cache_policy": CORE_PREFIX_CACHE_POLICY,
        "paired_stream_throughput_delta_percent": statistics.median(
            item["paired_stream_throughput_delta_percent"]
            for item in timed_rounds
        ),
        "caller_exposure_factor": statistics.median(
            item["caller_exposure_factor"] for item in timed_rounds
        ),
    }

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
        trial_seed,
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
            trial_seed,
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

    repository_after = _repository_fingerprint(repo)
    if repository_after != repository:
        raise RuntimeError("repository changed while a release worker was running")
    prompt_sha256 = _canonical_sha256([
        item["prompt_sha256"] for item in timed_rounds
    ])
    result = {
        "schema_version": SCHEMA_VERSION,
        "worker_started_at_utc": worker_started_at_utc,
        "worker_finished_at_utc": _utc_now(),
        "trial_index": args.trial_index,
        "pair_id": f"{args.expected_commit[:12]}-{args.trial_index:02d}-{prompt_sha256[:12]}",
        "trial_seed": trial_seed,
        "timed_prompt_sets": [item["prompts"] for item in timed_rounds],
        "prompt_sha256": prompt_sha256,
        "timed_round_orders": [item["measurement_order"] for item in timed_rounds],
        "timed_round_seeds": [item["round_seed"] for item in timed_rounds],
        "slow_consumer_order_ms": [value * 1000.0 for value in sleep_order],
        "environment": environment,
        "warmup": {
            "short_tokens_per_route": args.warmup_tokens,
            "full_length_tokens_per_route": args.max_tokens,
            "prompt_sha256": _canonical_sha256(warmup_prompts),
            "max_prompt_plus_completion_tokens": (
                max(warmup_prompt_lengths) + args.max_tokens
            ),
            "block_size": llm.scheduler.block_size,
            "prefix_cache_policy": CORE_PREFIX_CACHE_POLICY,
            "records": warmup_records,
        },
        "core": core,
        "slow_consumer": slow,
        "detokenizer": detokenizer,
    }
    output = Path(args.worker_output)
    _write_json_exclusive(output, result)


def _mean_90ci_for_eight(values: list[float]) -> dict[str, float]:
    if len(values) != 8:
        raise ValueError("the predeclared equivalence interval requires eight pairs")
    mean = statistics.mean(values)
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    margin = 1.894579 * standard_error  # Student t, df=7, central 90% interval
    return {"mean": mean, "low": mean - margin, "high": mean + margin}


def _stable_environment_pin(environment: dict[str, Any]) -> dict[str, Any]:
    return {
        "repository": environment["repository"],
        "benchmark_script_sha256": environment["benchmark_script_sha256"],
        "model": environment["model"],
        "software": environment["software"],
        "cpu": environment["cpu"],
        "gpu": environment["gpu"],
    }


def _aggregate(workers: list[dict[str, Any]]) -> dict[str, Any]:
    timed_rounds = [
        timed_round
        for worker in workers
        for timed_round in worker["core"]["timed_rounds"]
    ]
    generate_tps = [
        item["core"]["route_medians"]["generate"]["tokens_per_second"]
        for item in workers
    ]
    stream_tps = [
        item["core"]["route_medians"]["stream"]["tokens_per_second"]
        for item in workers
    ]
    generate_return = [
        item["core"]["route_medians"]["generate"]["return_seconds"]
        for item in workers
    ]
    stream_first = [
        item["core"]["route_medians"]["stream"]["first_event_seconds"]
        for item in workers
    ]
    paired_deltas = [
        item["core"]["paired_stream_throughput_delta_percent"] for item in workers
    ]
    paired_delta_90ci = _mean_90ci_for_eight(paired_deltas)
    event_delivery_seconds = [
        event["engine_to_caller_seconds"]
        for timed_round in timed_rounds
        for event in timed_round["stream"]["event_delivery"]["values"]
    ]
    request_delivery = {}
    for key in ("first_token_to_delivery", "engine_finish_to_delivery"):
        values = [
            value
            for timed_round in timed_rounds
            for value in timed_round["stream"]
            ["request_metric_distributions_seconds"][key]["values"]
        ]
        request_delivery[key] = {
            "seconds": _summary(values),
            "milliseconds": _summary([value * 1000.0 for value in values]),
        }

    generate_peak = [
        item["generate"]["memory"]["peak"]["allocated_bytes"]
        for item in timed_rounds
    ]
    stream_peak = [
        item["stream"]["memory"]["peak"]["allocated_bytes"]
        for item in timed_rounds
    ]
    memory_delta = [
        stream - generate
        for stream, generate in zip(stream_peak, generate_peak, strict=True)
    ]

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

    baseline_gap_ms = slow["0"]["inter_step_gap_ms"]["median"]
    slow_batch_size = (
        workers[0]["slow_consumer"][0]["num_events"]
        // workers[0]["slow_consumer"][0]["num_steps"]
    )
    backpressure_roofline = {}
    for key, item in slow.items():
        sleep_ms = item["consumer_sleep_ms_per_event"]
        expected_gap_ms = baseline_gap_ms + slow_batch_size * sleep_ms
        observed_gap_ms = item["inter_step_gap_ms"]["median"]
        residual_ms = observed_gap_ms - expected_gap_ms
        tolerance_ms = max(2.0, 0.10 * slow_batch_size * sleep_ms)
        backpressure_roofline[key] = {
            "observed_gap_ms": observed_gap_ms,
            "expected_gap_ms": expected_gap_ms,
            "residual_ms": residual_ms,
            "tolerance_ms": tolerance_ms,
            "within_tolerance": abs(residual_ms) <= tolerance_ms,
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
        "matched_final_decode_consumer": {
            "worker_statistical_unit": (
                "one fresh worker's median of four within-round paired stream "
                "throughput deltas across two rounds in each route order"
            ),
            "generate_tokens_per_second": _summary(generate_tps),
            "stream_tokens_per_second": _summary(stream_tps),
            "paired_stream_delta_percent": _summary(paired_deltas),
            "paired_mean_delta_percent_90ci": paired_delta_90ci,
            "equivalence_band_percent": [-2.0, 2.0],
            "raw_timed_round_stream_delta_percent": _summary([
                item["paired_stream_throughput_delta_percent"]
                for item in timed_rounds
            ]),
            "raw_by_stream_position": {
                str(position): _summary([
                    item["paired_stream_throughput_delta_percent"]
                    for item in timed_rounds
                    if item["stream"]["pair_position"] == position
                ])
                for position in (0, 1)
            },
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
            "all_event_engine_to_caller_ms": _summary([
                value * 1000.0 for value in event_delivery_seconds
            ]),
            "request_boundaries": request_delivery,
        },
        "memory": {
            "generate_peak_allocated_bytes": _summary(generate_peak),
            "stream_peak_allocated_bytes": _summary(stream_peak),
            "stream_minus_generate_peak_allocated_bytes": _summary(memory_delta),
        },
        "slow_consumer": slow,
        "backpressure_roofline": backpressure_roofline,
        "detokenizer": detokenizer,
        "gates": {
            "all_seeded_tokens_equivalent": all(
                item["core"]["tokens_equivalent"] for item in workers
            ),
            "all_seeded_text_equivalent": all(
                item["core"]["texts_equivalent"] for item in workers
            ),
            "eight_distinct_seeds": len({item["trial_seed"] for item in workers}) == 8,
            "eight_distinct_prompt_sets": len({
                item["prompt_sha256"] for item in workers
            }) == 8,
            "all_timed_round_seeds_distinct": len({
                item["round_seed"] for item in timed_rounds
            }) == len(timed_rounds),
            "all_timed_prompt_sets_distinct": len({
                item["prompt_sha256"] for item in timed_rounds
            }) == len(timed_rounds),
            "balanced_pair_order": (
                len(timed_rounds) == len(workers) * CORE_TIMED_ROUNDS
                and all(
                    sum(
                        item["measurement_order"] == ["generate", "stream"]
                        for item in worker["core"]["timed_rounds"]
                    ) == CORE_TIMED_ROUNDS // 2
                    and sum(
                        item["measurement_order"] == ["stream", "generate"]
                        for item in worker["core"]["timed_rounds"]
                    ) == CORE_TIMED_ROUNDS // 2
                    for worker in workers
                )
                and sum(
                    item["measurement_order"] == ["generate", "stream"]
                    for item in timed_rounds
                ) == len(timed_rounds) // 2
                and sum(
                    item["measurement_order"] == ["stream", "generate"]
                    for item in timed_rounds
                ) == len(timed_rounds) // 2
            ),
            "route_positions_match_declared_order": all(
                timed_round[route]["pair_position"] == position
                for timed_round in timed_rounds
                for position, route in enumerate(timed_round["measurement_order"])
            ),
            "full_length_warmup_covers_both_routes": all(
                worker["warmup"]["full_length_tokens_per_route"]
                == max(
                    timed_round["generate"]["num_tokens"]
                    // len(timed_round["prompts"])
                    for timed_round in worker["core"]["timed_rounds"]
                )
                and {
                    item["route"]
                    for item in worker["warmup"]["records"]
                    if item["max_tokens"]
                    == worker["warmup"]["full_length_tokens_per_route"]
                } == {"generate", "stream"}
                for worker in workers
            ),
            "cache_neutral_core_policy_asserted": all(
                worker["core"]["prefix_cache_policy"]
                == CORE_PREFIX_CACHE_POLICY
                and worker["warmup"]["prefix_cache_policy"]
                == CORE_PREFIX_CACHE_POLICY
                for worker in workers
            ) and all(
                timed_round["max_prompt_plus_completion_tokens"]
                < timed_round["block_size"]
                and [
                    item["route"] for item in timed_round["prefix_cache_resets"]
                ] == timed_round["measurement_order"]
                and all(
                    item["policy"] == CORE_PREFIX_CACHE_POLICY
                    and item["pair_position"] == position
                    and item["used_blocks_before_reset"] == 0
                    and item["used_blocks_after_route"] == 0
                    and item["cached_block_hashes_after_route"] == 0
                    for position, item in enumerate(
                        timed_round["prefix_cache_resets"]
                    )
                )
                for timed_round in timed_rounds
            ),
            "all_worker_pins_identical": all(
                _stable_environment_pin(item["environment"])
                == _stable_environment_pin(workers[0]["environment"])
                for item in workers
            ),
            "paired_mean_90ci_inside_plus_or_minus_2_percent": (
                paired_delta_90ci["low"] >= -2.0
                and paired_delta_90ci["high"] <= 2.0
            ),
            "event_delivery_p95_at_most_1ms": (
                _percentile(event_delivery_seconds, 0.95) <= 0.001
            ),
            "minimum_caller_exposure_at_least_10x": min(
                item["core"]["caller_exposure_factor"] for item in workers
            ) >= 10.0,
            "stream_peak_memory_within_1_percent_of_generate": all(
                _stream_peak_within_one_percent_of_generate(stream, generate)
                for stream, generate in zip(
                    stream_peak, generate_peak, strict=True
                )
            ),
            "pending_storage_bounded_by_slow_batch_minus_one": all(
                item["max_pending_events_after_delivery"]
                <= item["num_events"] // item["num_steps"] - 1
                for worker in workers
                for item in worker["slow_consumer"]
            ),
            "backpressure_matches_synchronous_roofline": all(
                item["within_tolerance"] for item in backpressure_roofline.values()
            ),
            "incremental_detokenizer_decode_bounded": all(
                item["max_feed_decode_tokens"]
                <= CORRECTION_WINDOW_SIZE + 2 * CORRECTION_BOUNDARY_OVERLAP
                for worker in workers
                for item in worker["detokenizer"]
            ),
        },
    }


def _parent_result(
    args: argparse.Namespace,
    workers: list[dict[str, Any]],
    repository: dict[str, Any],
    model: dict[str, Any],
    correction_scaling: dict[str, Any],
    worker_commands: list[list[str]],
    started_at_utc: str,
) -> dict[str, Any]:
    environment = workers[0]["environment"]
    aggregate = _aggregate(workers)
    aggregate["gates"]["correction_heavy_cpu_scaling"] = correction_scaling[
        "all_gates_pass"
    ]
    aggregate["gates"]["all_required_gates_pass"] = all(
        aggregate["gates"].values()
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "repaired synchronous StreamSession and TextUpdate",
        "started_at_utc": started_at_utc,
        "recorded_at_utc": _utc_now(),
        "protocol": {
            "fresh_worker_processes": args.runs,
            "fresh_process_pairs": args.runs,
            "raw_timed_route_pairs": args.runs * args.core_rounds,
            "timed_rounds_per_worker": args.core_rounds,
            "paired_core_order": [
                worker["timed_round_orders"] for worker in workers
            ],
            "pair_ids": [worker["pair_id"] for worker in workers],
            "slow_consumer_order_ms": [worker["slow_consumer_order_ms"] for worker in workers],
            "short_warmup_tokens_per_route": args.warmup_tokens,
            "full_length_warmup_tokens_per_route": args.max_tokens,
            "core_prefix_cache_policy": CORE_PREFIX_CACHE_POLICY,
            "batch_size": args.batch_size,
            "max_tokens": args.max_tokens,
            "slow_batch_size": args.slow_batch_size,
            "slow_tokens": args.slow_tokens,
            "temperature": args.temperature,
            "base_seed": args.seed,
            "trial_seeds": [worker["trial_seed"] for worker in workers],
            "timed_round_seeds": [
                worker["timed_round_seeds"] for worker in workers
            ],
            "prompt_sha256": [worker["prompt_sha256"] for worker in workers],
            "timed_prompt_sha256": [
                [
                    timed_round["prompt_sha256"]
                    for timed_round in worker["core"]["timed_rounds"]
                ]
                for worker in workers
            ],
            "matched_route_work": (
                "both generate and stream perform one final full tokenizer "
                "decode per sequence before their route timer stops"
            ),
            "detokenizer_repeats_per_worker": args.detok_repeats,
            "detokenizer_lengths": args.detok_lengths,
            "correction_scaling_repeats": args.correction_repeats,
            "correction_scaling_lengths": args.correction_lengths,
        },
        "provenance": {
            "repository": repository,
            "benchmark_script_sha256": _sha256_file(Path(__file__).resolve()),
            "model": model,
            "core_timing_policy": {
                "statistical_unit": (
                    "fresh-worker median of four within-round paired deltas"
                ),
                "full_length_warmup_per_route": True,
                "two_rounds_per_route_order": True,
                "distinct_prompt_and_seed_per_round": True,
                "prefix_cache_policy": CORE_PREFIX_CACHE_POLICY,
                "prompt_plus_completion_below_one_block": True,
            },
            "post_run_verification": {
                "repository_unchanged": True,
                "model_manifest_unchanged": True,
            },
            "parent_argv": list(sys.argv),
            "worker_commands": worker_commands,
        },
        "environment": environment,
        "aggregate": aggregate,
        "correction_heavy_cpu_scaling": correction_scaling,
        "observations": workers,
    }


def _run_parent(args: argparse.Namespace) -> None:
    started_at_utc = _utc_now()
    if args.runs != 8:
        raise ValueError("release evidence requires exactly eight fresh process pairs")
    if args.core_rounds != CORE_TIMED_ROUNDS:
        raise ValueError(
            f"release evidence requires exactly {CORE_TIMED_ROUNDS} timed rounds per worker"
        )
    if args.max_tokens < 1:
        raise ValueError("max tokens must be positive")
    if not 1 <= args.warmup_tokens < args.max_tokens:
        raise ValueError("short warmup tokens must be in [1, max_tokens)")
    if not 1 <= args.batch_size:
        raise ValueError("batch size must be positive")
    if not 1 <= args.slow_batch_size <= args.batch_size:
        raise ValueError("slow batch size must be in [1, batch_size]")
    if args.slow_tokens < 2:
        raise ValueError("slow tokens must be at least two for inter-step gaps")
    if any(value < 0 for value in args.sleep_ms):
        raise ValueError("consumer sleep values cannot be negative")
    if 0.0 not in args.sleep_ms or len(set(args.sleep_ms)) != len(args.sleep_ms):
        raise ValueError("consumer sleep values must be unique and include zero")
    if any(value <= 0 for value in args.detok_lengths):
        raise ValueError("detokenizer lengths must be positive")
    if args.detok_repeats < 1:
        raise ValueError("detokenizer repeats must be positive")
    if args.correction_repeats < 1:
        raise ValueError("correction repeats must be positive")
    if (
        len(args.correction_lengths) != 2
        or args.correction_lengths != sorted(set(args.correction_lengths))
    ):
        raise ValueError("correction lengths must be two distinct ascending values")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable result: {output}")
    repo = Path(__file__).resolve().parents[2]
    if output.is_relative_to(repo):
        raise ValueError("release output must be outside the repository worktree")
    repository = _repository_fingerprint(repo)
    if repository["worktree_dirty"]:
        raise RuntimeError("release evidence requires a clean repository")
    script_sha256 = _sha256_file(Path(__file__).resolve())
    model_path = Path(args.model).resolve()
    model = _model_fingerprint(model_path)
    correction_scaling = _measure_correction_scaling(
        _load_detokenizer_type(repo),
        args.correction_lengths,
        args.correction_repeats,
    )
    if not correction_scaling["all_gates_pass"]:
        raise RuntimeError("correction-heavy CPU scaling gate failed")

    workers = []
    worker_commands = []
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
                "--core-rounds", str(args.core_rounds),
                "--slow-batch-size", str(args.slow_batch_size),
                "--slow-tokens", str(args.slow_tokens),
                "--warmup-tokens", str(args.warmup_tokens),
                "--temperature", str(args.temperature),
                "--seed", str(args.seed),
                "--max-model-len", str(args.max_model_len),
                "--detok-repeats", str(args.detok_repeats),
                "--expected-commit", repository["commit"],
                "--expected-source-manifest-sha256",
                repository["source"]["manifest_sha256"],
                "--expected-script-sha256", script_sha256,
                "--expected-model-manifest-sha256", model["manifest_sha256"],
                "--sleep-ms", *[str(value) for value in args.sleep_ms],
                "--detok-lengths", *[str(value) for value in args.detok_lengths],
            ]
            worker_commands.append(command)
            environment = os.environ.copy()
            python_path = environment.get("PYTHONPATH")
            environment["PYTHONPATH"] = (
                str(repo) if not python_path else f"{repo}{os.pathsep}{python_path}"
            )
            environment["PYTHONHASHSEED"] = str(_trial_seed(args.seed, trial_index))
            environment["TORCHINDUCTOR_CACHE_DIR"] = str(
                Path(temporary) / f"inductor-{trial_index}"
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

    if _repository_fingerprint(repo) != repository:
        raise RuntimeError("repository changed during the release benchmark")
    if _model_fingerprint(model_path) != model:
        raise RuntimeError("model snapshot changed during the release benchmark")
    result = _parent_result(
        args,
        workers,
        repository,
        model,
        correction_scaling,
        worker_commands,
        started_at_utc,
    )
    _write_json_exclusive(output, result)
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))
    print(f"wrote {output}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output")
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--core-rounds", type=int, default=CORE_TIMED_ROUNDS)
    parser.add_argument("--slow-batch-size", type=int, default=8)
    parser.add_argument("--slow-tokens", type=int, default=32)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--sleep-ms", type=float, nargs="+", default=[0.0, 1.0, 4.0])
    parser.add_argument("--detok-lengths", type=int, nargs="+", default=[64, 256, 1024, 2048])
    parser.add_argument("--detok-repeats", type=int, default=3)
    parser.add_argument("--correction-lengths", type=int, nargs="+", default=[8000, 32000])
    parser.add_argument("--correction-repeats", type=int, default=3)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    parser.add_argument("--trial-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--expected-commit", help=argparse.SUPPRESS)
    parser.add_argument("--expected-source-manifest-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--expected-script-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--expected-model-manifest-sha256", help=argparse.SUPPRESS)
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
        required = (
            "worker_output",
            "expected_commit",
            "expected_source_manifest_sha256",
            "expected_script_sha256",
            "expected_model_manifest_sha256",
        )
        missing = [name for name in required if not getattr(args, name)]
        if missing:
            parser.error(f"worker arguments missing: {', '.join(missing)}")
        _run_worker(args)
    else:
        if not args.output:
            parser.error("--output is required")
        _run_parent(args)


if __name__ == "__main__":
    main()
