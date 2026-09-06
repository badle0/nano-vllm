#!/usr/bin/env python3
"""Validate exactly five fresh full-completion runs and retain their verdict."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import stat
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from benchmarks.chunked_prefill_tail.common import (
    ROOT,
    handle_pin_query,
    immutable_write_bytes,
    immutable_write_json,
    validate_release_pin,
)
from benchmarks.chunked_prefill_tail.full_completion_cert import (
    KIND as RUN_KIND,
    MAX_ITL_SLO_MS,
    PROTOCOL as RUN_PROTOCOL,
    SCHEMA_VERSION as RUN_SCHEMA_VERSION,
    SUPPORTED_TAUS,
    certification_workload,
    prompt_manifest,
    summarize_requests,
)


SCHEMA_VERSION = 1
KIND = "chunked_prefill_full_completion_aggregate"
REQUIRED_RUNS = 5
SHA256_HEX = set("0123456789abcdef")
COMMIT_HEX = SHA256_HEX


@dataclass(frozen=True)
class LoadedArtifact:
    path: Path
    raw: bytes
    sha256: str
    payload: dict[str, object]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate exactly five self-pinned full-completion runs."
    )
    parser.add_argument("--tau", type=int, choices=SUPPORTED_TAUS, default=256)
    parser.add_argument("--input", type=Path, nargs="+")
    parser.add_argument("--expected-commit")
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--archive-dir", type=Path)
    parser.add_argument("--print-source-sha256", action="store_true")
    parser.add_argument("--show-pin", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    missing = [
        name
        for name in ("input", "expected_commit", "expected_source_sha256")
        if getattr(args, name) is None
    ]
    if missing:
        raise ValueError(
            "aggregation requires "
            + ", ".join("--" + name.replace("_", "-") for name in missing)
        )
    if len(args.input) != REQUIRED_RUNS:
        raise ValueError(f"--input requires exactly {REQUIRED_RUNS} run artifacts")
    if (args.output is None) == (args.archive_dir is None):
        raise ValueError("choose exactly one of --output or --archive-dir")
    destination = args.output if args.output is not None else args.archive_dir
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite retained output: {destination}")
    if args.archive_dir is not None:
        complete_marker = args.archive_dir.with_name(
            args.archive_dir.name + ".COMPLETE"
        )
        if complete_marker.exists():
            raise FileExistsError(
                f"refusing to overwrite archive marker: {complete_marker}"
            )
    try:
        destination.expanduser().resolve().relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise ValueError("retained destination must be outside the pinned worktree")


def _is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= SHA256_HEX
    )


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _argv_option(argv: list[str], option: str, *, required: bool) -> str | None:
    values = []
    for index, value in enumerate(argv):
        if value == option:
            if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                raise ValueError(f"argv option {option} has no value")
            values.append(argv[index + 1])
        elif value.startswith(option + "="):
            values.append(value.split("=", 1)[1])
    if len(values) > 1:
        raise ValueError(f"argv option {option} is duplicated")
    if required and not values:
        raise ValueError(f"argv option {option} is missing")
    return values[0] if values else None


def load_artifact(path: Path) -> LoadedArtifact:
    resolved = path.expanduser().resolve(strict=True)
    info = resolved.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"run artifact is not a regular file: {resolved}")
    if info.st_mode & 0o222:
        raise ValueError(f"run artifact must be immutable/read-only: {resolved}")
    raw = resolved.read_bytes()
    try:
        payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid run artifact JSON: {resolved}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"run artifact root must be an object: {resolved}")
    return LoadedArtifact(
        path=resolved,
        raw=raw,
        sha256=hashlib.sha256(raw).hexdigest(),
        payload=payload,
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_nonnegative(value, name: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{name} must be numeric",
    )
    number = float(value)
    _require(math.isfinite(number) and number >= 0.0, f"{name} must be finite >= 0")
    return number


def _same_number(actual, expected, name: str) -> None:
    actual_number = _finite_nonnegative(actual, name)
    _require(
        math.isclose(
            actual_number,
            float(expected),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ),
        f"{name} does not match recomputed raw metrics",
    )


def _validate_source_manifest(source: dict[str, object], name: str) -> str:
    files = source.get("files")
    _require(isinstance(files, list) and files, f"{name}.source files missing")
    _require(
        source.get("file_count") == len(files),
        f"{name}.source file count mismatch",
    )
    aggregate = hashlib.sha256()
    paths = []
    for row in files:
        _require(isinstance(row, dict), f"{name}.source file row invalid")
        relative = row.get("path")
        size = row.get("bytes")
        digest = row.get("sha256")
        _require(isinstance(relative, str) and relative, f"{name}.source path invalid")
        pure = PurePosixPath(relative)
        _require(
            not pure.is_absolute() and ".." not in pure.parts,
            f"{name}.source path is not relative",
        )
        _require(type(size) is int and size >= 0, f"{name}.source size invalid")
        _require(_is_sha256(digest), f"{name}.source hash invalid")
        paths.append(relative)
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(bytes.fromhex(digest))
    _require(
        paths == sorted(paths) and len(paths) == len(set(paths)),
        f"{name}.source paths not unique/sorted",
    )
    retained = source.get("aggregate_sha256")
    _require(_is_sha256(retained), f"{name} source hash is invalid")
    _require(
        retained == aggregate.hexdigest(),
        f"{name}.source aggregate hash mismatch",
    )
    return retained


def _validate_pin(pin: dict[str, object], name: str) -> tuple[str, str, str]:
    _require(isinstance(pin, dict), f"{name} must be an object")
    git = pin.get("git")
    source = pin.get("source")
    _require(isinstance(git, dict), f"{name}.git must be an object")
    _require(isinstance(source, dict), f"{name}.source must be an object")
    _require(git.get("clean") is True, f"{name} must record a clean worktree")
    _require(git.get("status") == [], f"{name} status must be empty")
    commit = git.get("commit")
    _require(
        isinstance(commit, str)
        and len(commit) == 40
        and set(commit) <= COMMIT_HEX,
        f"{name} commit must be full length",
    )
    tree = git.get("tree")
    _require(
        isinstance(tree, str) and len(tree) == 40 and set(tree) <= COMMIT_HEX,
        f"{name} tree must be full length",
    )
    _require(isinstance(git.get("branch"), str), f"{name} branch missing")
    source_hash = _validate_source_manifest(source, name)
    return commit, source_hash, tree


def _validate_model_manifest(model: dict[str, object]) -> None:
    _require(isinstance(model, dict), "model manifest must be an object")
    files = model.get("files")
    _require(isinstance(files, list) and files, "model manifest files must be nonempty")
    _require(model.get("file_count") == len(files), "model file count mismatch")
    aggregate = hashlib.sha256()
    total = 0
    paths = []
    for row in files:
        _require(isinstance(row, dict), "model file row must be an object")
        relative = row.get("path")
        size = row.get("bytes")
        digest = row.get("sha256")
        _require(isinstance(relative, str) and relative, "model file path is invalid")
        pure = PurePosixPath(relative)
        _require(
            not pure.is_absolute() and ".." not in pure.parts,
            f"model file path is not relative: {relative}",
        )
        _require(type(size) is int and size >= 0, "model file size is invalid")
        _require(_is_sha256(digest), "model file hash is invalid")
        paths.append(relative)
        total += size
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(bytes.fromhex(digest))
    _require(paths == sorted(paths) and len(paths) == len(set(paths)), "model paths not unique/sorted")
    _require(model.get("total_bytes") == total, "model byte count mismatch")
    _require(
        model.get("aggregate_sha256") == aggregate.hexdigest(),
        "model aggregate hash mismatch",
    )


def _validate_prompt_manifest(
    manifest: dict[str, object],
    label: str,
    expected_count: int,
    expected_tokens: int,
) -> None:
    _require(isinstance(manifest, dict), f"{label} prompt manifest missing")
    _require(manifest.get("label") == label, f"{label} prompt label mismatch")
    rows = manifest.get("prompts")
    _require(isinstance(rows, list), f"{label} prompt rows missing")
    _require(len(rows) == expected_count, f"{label} prompt count mismatch")
    aggregate = hashlib.sha256()
    for index, row in enumerate(rows):
        _require(row.get("index") == index, f"{label} prompt index mismatch")
        _require(row.get("tokens") == expected_tokens, f"{label} prompt length mismatch")
        digest = row.get("sha256")
        _require(_is_sha256(digest), f"{label} prompt hash invalid")
        aggregate.update(index.to_bytes(8, "big"))
        aggregate.update(expected_tokens.to_bytes(8, "big"))
        aggregate.update(bytes.fromhex(digest))
    _require(manifest.get("count") == expected_count, f"{label} manifest count mismatch")
    _require(
        manifest.get("aggregate_sha256") == aggregate.hexdigest(),
        f"{label} aggregate hash mismatch",
    )


def validate_run_payload(payload: dict[str, object], expected_tau: int) -> dict[str, object]:
    _require(payload.get("schema_version") == RUN_SCHEMA_VERSION, "run schema mismatch")
    _require(payload.get("kind") == RUN_KIND, "input is not a full-completion run")
    _require(payload.get("protocol") == RUN_PROTOCOL, "run protocol mismatch")
    argv = payload.get("argv")
    _require(
        isinstance(argv, list) and argv and all(isinstance(value, str) for value in argv),
        "run argv must be a complete string vector",
    )
    _require(
        len(argv) >= 2 and Path(argv[1]).name == "full_completion_cert.py",
        "run argv is not a direct full-completion harness invocation",
    )
    _require(
        "--print-source-sha256" not in argv and "--show-pin" not in argv,
        "runtime argv contains a pin-query flag",
    )
    before_commit, before_source, before_tree = _validate_pin(
        payload.get("provenance"), "provenance"
    )
    after_commit, after_source, after_tree = _validate_pin(
        payload.get("provenance_after_run"), "provenance_after_run"
    )
    _require(
        (before_commit, before_source, before_tree)
        == (after_commit, after_source, after_tree),
        "source pin changed during run",
    )
    _require(
        payload.get("provenance") == payload.get("provenance_after_run"),
        "full provenance manifest changed during run",
    )

    arguments = payload.get("arguments")
    randomness = payload.get("randomness")
    _require(isinstance(arguments, dict), "run arguments missing")
    _require(isinstance(randomness, dict), "run randomness missing")
    tau = arguments.get("tau")
    seed = arguments.get("seed")
    _require(tau == expected_tau, f"run tau {tau} does not match {expected_tau}")
    _require(type(seed) is int and 0 <= seed < 2**63, "run seed is invalid")
    for name in (
        "seed",
        "python_random_seed",
        "torch_manual_seed",
        "torch_cuda_manual_seed_all",
    ):
        _require(randomness.get(name) == seed, f"randomness field {name} mismatch")
    _require(
        randomness.get("torch_deterministic_algorithms_enabled") is False,
        "historical protocol requires default deterministic-algorithms=False",
    )
    _require(_argv_option(argv, "--seed", required=True) == str(seed), "argv seed mismatch")
    _require(
        _argv_option(argv, "--expected-commit", required=True) == before_commit,
        "argv commit pin mismatch",
    )
    _require(
        _argv_option(argv, "--expected-source-sha256", required=True)
        == before_source,
        "argv source pin mismatch",
    )
    _require(
        _argv_option(argv, "--output", required=True) == arguments.get("output"),
        "argv output mismatch",
    )
    argv_tau = _argv_option(argv, "--tau", required=False)
    _require(argv_tau is None or argv_tau == str(tau), "argv tau mismatch")
    argv_model = _argv_option(argv, "--model", required=False)
    _require(
        argv_model is None or argv_model == arguments.get("model"),
        "argv model mismatch",
    )
    _require(arguments.get("expected_commit") == before_commit, "argument commit pin mismatch")
    _require(
        arguments.get("expected_source_sha256") == before_source,
        "argument source pin mismatch",
    )

    model = payload.get("model")
    _validate_model_manifest(model)
    _require(payload.get("model_after_run_matches") is True, "model was not revalidated")
    _require(arguments.get("model") == model.get("argument"), "model argument mismatch")
    try:
        argument_model_path = str(Path(arguments["model"]).expanduser().resolve(strict=True))
    except (KeyError, OSError, TypeError) as error:
        raise ValueError("recorded model path is not currently resolvable") from error
    _require(
        argument_model_path == model.get("resolved_path"),
        "resolved model path mismatch",
    )
    environment = payload.get("environment")
    _require(isinstance(environment, dict), "environment manifest missing")
    for name in (
        "python",
        "platform",
        "torch",
        "cuda_build",
        "transformers",
        "flash_attn",
        "gpu",
        "gpu_total_memory_bytes",
        "compute_capability",
        "driver",
        "environment_variables",
    ):
        _require(name in environment, f"environment field {name} missing")
    _require(
        environment.get("python_gc_enabled") is True,
        "environment must be captured before the engine disables GC",
    )
    _require(
        isinstance(environment.get("environment_variables"), dict),
        "environment variable manifest must be an object",
    )

    expected_spec = certification_workload(expected_tau)
    workload = payload.get("workload")
    _require(isinstance(workload, dict), "workload manifest missing")
    _require(workload.get("spec") == expected_spec, "workload spec is not exact")
    prompt_manifests = workload.get("prompt_manifests")
    _require(isinstance(prompt_manifests, dict), "prompt manifests missing")
    _validate_prompt_manifest(
        prompt_manifests.get("warmup"),
        "warmup",
        expected_spec["warmup_requests"],
        expected_spec["warmup_prompt_tokens"],
    )
    rng = random.Random(seed)

    def reproduced_prompts(count: int, length: int) -> list[list[int]]:
        return [
            [
                rng.randrange(
                    expected_spec["prompt_token_min_inclusive"],
                    expected_spec["prompt_token_max_exclusive"],
                )
                for _ in range(length)
            ]
            for _ in range(count)
        ]

    reproduced = {
        "warmup": prompt_manifest(
            "warmup",
            reproduced_prompts(
                expected_spec["warmup_requests"],
                expected_spec["warmup_prompt_tokens"],
            ),
        ),
        "interactive": prompt_manifest(
            "interactive",
            reproduced_prompts(
                expected_spec["interactive_requests"],
                expected_spec["interactive_prompt_tokens"],
            ),
        ),
        "long": prompt_manifest(
            "long",
            reproduced_prompts(
                expected_spec["long_requests"],
                expected_spec["long_prompt_tokens"],
            ),
        ),
    }
    _require(
        prompt_manifests == reproduced,
        "prompt manifests do not reproduce from the pinned seed",
    )
    _validate_prompt_manifest(
        prompt_manifests.get("interactive"),
        "interactive",
        expected_spec["interactive_requests"],
        expected_spec["interactive_prompt_tokens"],
    )
    _validate_prompt_manifest(
        prompt_manifests.get("long"),
        "long",
        expected_spec["long_requests"],
        expected_spec["long_prompt_tokens"],
    )
    warmup_hashes = workload.get("warmup_completion_token_ids_sha256")
    _require(
        isinstance(warmup_hashes, list)
        and len(warmup_hashes) == expected_spec["warmup_requests"]
        and all(_is_sha256(value) for value in warmup_hashes),
        "warmup output hashes are incomplete",
    )

    engine = payload.get("engine")
    _require(isinstance(engine, dict), "engine manifest missing")
    config = engine.get("config")
    _require(isinstance(config, dict), "engine config missing")
    pinned_config = {
        "max_num_batched_tokens": expected_tau,
        "max_num_seqs": expected_spec["max_num_seqs"],
        "max_model_len": expected_spec["max_model_len"],
        "gpu_memory_utilization": expected_spec["gpu_memory_utilization"],
        "enforce_eager": expected_spec["enforce_eager"],
        "tensor_parallel_size": expected_spec["tensor_parallel_size"],
        "disable_python_gc": True,
    }
    for name, expected in pinned_config.items():
        _require(config.get(name) == expected, f"engine config {name} mismatch")

    python_gc = payload.get("python_gc")
    _require(isinstance(python_gc, dict), "Python GC lifecycle missing")
    _require(python_gc.get("disable_requested") is True, "engine GC option not requested")
    _require(python_gc.get("enabled_before_engine") is True, "run did not start with GC enabled")
    _require(python_gc.get("enabled_after_engine_init") is False, "engine did not disable GC")
    _require(python_gc.get("enabled_after_engine_exit") is True, "engine did not restore GC")

    requests = payload.get("requests")
    _require(isinstance(requests, list), "raw request metrics missing")
    _require(
        len(requests) == expected_spec["interactive_requests"] + expected_spec["long_requests"],
        "raw request count mismatch",
    )
    seen_ids = set()
    cohorts = {"interactive": [], "long": []}
    for row in requests:
        _require(isinstance(row, dict), "request row must be an object")
        seq_id = row.get("seq_id")
        cohort = row.get("cohort")
        _require(type(seq_id) is int and seq_id not in seen_ids, "request seq_id invalid/duplicate")
        _require(cohort in cohorts, "request cohort invalid")
        seen_ids.add(seq_id)
        token_ids = row.get("completion_token_ids")
        _require(
            isinstance(token_ids, list)
            and len(token_ids) == expected_spec["max_completion_tokens_per_request"]
            and all(type(token) is int for token in token_ids),
            "raw completion token vector is invalid",
        )
        expected_token_hash = hashlib.sha256(
            json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        _require(
            row.get("completion_token_ids_sha256") == expected_token_hash,
            "completion token hash mismatch",
        )
        metrics = row.get("metrics")
        _require(isinstance(metrics, dict), "request metrics must be an object")
        prompt_tokens = (
            expected_spec["interactive_prompt_tokens"]
            if cohort == "interactive"
            else expected_spec["long_prompt_tokens"]
        )
        _require(metrics.get("num_prompt_tokens") == prompt_tokens, "prompt metric mismatch")
        _require(
            metrics.get("num_completion_tokens")
            == expected_spec["max_completion_tokens_per_request"],
            "completion metric mismatch",
        )
        itls = metrics.get("engine_itls")
        _require(
            isinstance(itls, list)
            and len(itls) == expected_spec["max_completion_tokens_per_request"] - 1,
            "raw ITL vector is incomplete",
        )
        normalized_itls = [
            _finite_nonnegative(value, "engine_itl") for value in itls
        ]
        for name in (
            "engine_queue_time",
            "engine_ttft",
            "engine_e2e",
            "engine_mean_itl",
            "engine_max_itl",
            "submission_to_engine",
            "submission_to_first_token",
            "submission_to_engine_finish",
        ):
            _finite_nonnegative(metrics.get(name), name)
        _same_number(metrics["engine_max_itl"], max(normalized_itls), "engine_max_itl")
        _same_number(
            metrics["engine_mean_itl"],
            sum(normalized_itls) / len(normalized_itls),
            "engine_mean_itl",
        )
        _same_number(
            metrics["engine_e2e"],
            metrics["engine_ttft"] + sum(normalized_itls),
            "engine_e2e",
        )
        _same_number(
            metrics["submission_to_first_token"],
            metrics["submission_to_engine"] + metrics["engine_ttft"],
            "submission_to_first_token",
        )
        _same_number(
            metrics["submission_to_engine_finish"],
            metrics["submission_to_engine"] + metrics["engine_e2e"],
            "submission_to_engine_finish",
        )
        cohorts[cohort].append(row)
    _require(
        len(cohorts["interactive"]) == expected_spec["interactive_requests"],
        "interactive raw metric count mismatch",
    )
    _require(
        len(cohorts["long"]) == expected_spec["long_requests"],
        "long raw metric count mismatch",
    )
    _require(
        set(workload.get("interactive_seq_ids", ()))
        == {row["seq_id"] for row in cohorts["interactive"]},
        "interactive sequence-id manifest mismatch",
    )
    _require(
        set(workload.get("long_seq_ids", ()))
        == {row["seq_id"] for row in cohorts["long"]},
        "long sequence-id manifest mismatch",
    )

    summary = payload.get("summary")
    _require(isinstance(summary, dict), "run summary missing")
    _require(summary.get("single_run_latency_certified") is False, "single run self-certified")
    for cohort, rows in (*cohorts.items(), ("all", requests)):
        recomputed = summarize_requests(rows)
        retained = summary.get(cohort)
        _require(isinstance(retained, dict), f"{cohort} summary missing")
        for name, expected in recomputed.items():
            if isinstance(expected, float):
                _same_number(retained.get(name), expected, f"{cohort}.{name}")
            else:
                _require(retained.get(name) == expected, f"{cohort}.{name} mismatch")
    whole_run_s = _finite_nonnegative(summary.get("whole_run_s"), "whole_run_s")
    _require(whole_run_s > 0.0, "whole_run_s must be positive")
    after_long_s = _finite_nonnegative(
        summary.get("after_long_admission_s"), "after_long_admission_s"
    )
    _require(after_long_s <= whole_run_s, "post-admission time exceeds whole run")
    expected_tokens = (
        expected_spec["interactive_requests"] + expected_spec["long_requests"]
    ) * expected_spec["max_completion_tokens_per_request"]
    _require(summary.get("total_completion_tokens") == expected_tokens, "TPS token count mismatch")
    _same_number(
        summary.get("completion_tokens_per_s"),
        expected_tokens / whole_run_s,
        "completion_tokens_per_s",
    )

    telemetry = payload.get("telemetry")
    _require(isinstance(telemetry, dict), "telemetry missing")
    routing = telemetry.get("routing")
    memory = telemetry.get("memory")
    _require(isinstance(routing, dict), "routing telemetry missing")
    _require(isinstance(memory, dict), "memory telemetry missing")
    _require(
        routing.get("per_step_routes_observed") is False,
        "certification timing path must not observe per-step routes",
    )
    _require("steps" not in routing, "per-step routes leaked into certification")
    _require(routing.get("selected_graph_keys_observed") is False, "timed model path was wrapped")
    miss_before = routing.get("varlen_miss_before")
    miss_after = routing.get("varlen_miss_after")
    miss_delta = routing.get("varlen_miss_delta")
    _require(
        type(miss_before) is int
        and type(miss_after) is int
        and type(miss_delta) is int,
        "graph-miss endpoints must be integers",
    )
    _require(
        miss_before >= 0 and miss_after - miss_before == miss_delta,
        "graph-miss endpoints disagree with per-step deltas",
    )
    required_memory = {
        "allocated_before_bytes",
        "reserved_before_bytes",
        "allocated_after_bytes",
        "reserved_after_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "kv_blocks_free_before",
        "kv_blocks_used_before",
        "kv_blocks_free_after",
        "kv_blocks_used_after",
    }
    _require(required_memory <= memory.keys(), "memory telemetry fields missing")
    for name in required_memory:
        value = memory[name]
        _require(type(value) is int and value >= 0, f"memory telemetry {name} invalid")
    _require(
        memory.get("peak_allocated_bytes", -1)
        >= max(
            memory.get("allocated_before_bytes", -1),
            memory.get("allocated_after_bytes", -1),
        ),
        "peak allocated memory is below an endpoint",
    )
    _require(
        memory.get("peak_reserved_bytes", -1)
        >= max(
            memory.get("reserved_before_bytes", -1),
            memory.get("reserved_after_bytes", -1),
        ),
        "peak reserved memory is below an endpoint",
    )

    return {
        "tau": expected_tau,
        "seed": seed,
        "commit": before_commit,
        "tree": before_tree,
        "source_sha256": before_source,
        "model": model,
        "environment": environment,
        "workload": workload,
        "interactive_max_itl_ms": float(
            summary["interactive"]["engine_max_itl_max_ms"]
        ),
        "long_max_ttft_ms": float(summary["long"]["engine_ttft_max_ms"]),
        "completion_tokens_per_s": float(summary["completion_tokens_per_s"]),
    }


def aggregate_artifacts(
    artifacts: list[LoadedArtifact],
    tau: int,
    current_pin: dict[str, object],
    argv: list[str],
) -> dict[str, object]:
    if len(artifacts) != REQUIRED_RUNS:
        raise ValueError(f"exactly {REQUIRED_RUNS} artifacts are required")
    facts = [validate_run_payload(artifact.payload, tau) for artifact in artifacts]
    seeds = [fact["seed"] for fact in facts]
    if len(set(seeds)) != REQUIRED_RUNS:
        raise ValueError("five distinct fresh seeds are required")
    if len({artifact.sha256 for artifact in artifacts}) != REQUIRED_RUNS:
        raise ValueError("duplicate run artifact content is not fresh evidence")
    if len({artifact.path for artifact in artifacts}) != REQUIRED_RUNS:
        raise ValueError("duplicate run artifact path")

    current_commit, current_source, current_tree = _validate_pin(
        current_pin, "aggregate_provenance"
    )
    for fact in facts:
        if (fact["commit"], fact["source_sha256"], fact["tree"]) != (
            current_commit,
            current_source,
            current_tree,
        ):
            raise ValueError("run source pin does not match the current clean HEAD")
    first = facts[0]
    for fact in facts[1:]:
        if fact["model"] != first["model"]:
            raise ValueError("model manifests differ across runs")
        if fact["environment"] != first["environment"]:
            raise ValueError("environment manifests differ across runs")
        if fact["workload"]["spec"] != first["workload"]["spec"]:
            raise ValueError("workload specs differ across runs")

    ordered = sorted(zip(artifacts, facts, strict=True), key=lambda pair: pair[1]["seed"])
    max_itls = [pair[1]["interactive_max_itl_ms"] for pair in ordered]
    long_ttfts = [pair[1]["long_max_ttft_ms"] for pair in ordered]
    throughputs = [pair[1]["completion_tokens_per_s"] for pair in ordered]
    per_run_pass = [value < MAX_ITL_SLO_MS for value in max_itls]
    tau256_all_pass = tau == 256 and all(per_run_pass)
    if tau == 256:
        classification = (
            "latency_certified_current_self_pinned"
            if tau256_all_pass
            else "latency_not_certified"
        )
        reason = (
            "all five full-completion run maxima are strictly below 10 ms"
            if tau256_all_pass
            else "at least one full-completion run maximum is not below 10 ms"
        )
    else:
        classification = "throughput_ttft_only_not_latency_certified"
        reason = (
            "tau 512 is excluded from latency certification policy; phase or "
            "single-run diagnostics never promote it"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "argv": argv,
        "provenance": current_pin,
        "protocol": RUN_PROTOCOL,
        "tau": tau,
        "required_runs": REQUIRED_RUNS,
        "seeds": [pair[1]["seed"] for pair in ordered],
        "workload": first["workload"]["spec"],
        "model": first["model"],
        "environment": first["environment"],
        "inputs": [
            {
                "path": str(artifact.path),
                "bytes": len(artifact.raw),
                "sha256": artifact.sha256,
                "seed": fact["seed"],
                "interactive_max_itl_ms": fact["interactive_max_itl_ms"],
                "long_max_ttft_ms": fact["long_max_ttft_ms"],
                "completion_tokens_per_s": fact["completion_tokens_per_s"],
                "strict_max_itl_slo_pass": fact["interactive_max_itl_ms"]
                < MAX_ITL_SLO_MS,
            }
            for artifact, fact in ordered
        ],
        "summary": {
            "run_interactive_max_itl_ms": max_itls,
            "median_run_interactive_max_itl_ms": statistics.median(max_itls),
            "worst_run_interactive_max_itl_ms": max(max_itls),
            "run_long_max_ttft_ms": long_ttfts,
            "median_run_long_max_ttft_ms": statistics.median(long_ttfts),
            "worst_run_long_max_ttft_ms": max(long_ttfts),
            "run_completion_tokens_per_s": throughputs,
            "median_completion_tokens_per_s": statistics.median(throughputs),
        },
        "policy": {
            "max_itl_slo_ms_exclusive": MAX_ITL_SLO_MS,
            "runs_meeting_slo": sum(per_run_pass),
            "all_five_runs_meet_slo": all(per_run_pass),
            "latency_certified": tau256_all_pass,
            "classification": classification,
            "reason": reason,
            "evidence_scope": "full_completion_request_metrics_only",
            "phase_diagnostics_can_certify": False,
            "single_run_can_certify": False,
            "tau512_can_certify": False,
        },
    }


def write_archive(
    archive_dir: Path,
    aggregate: dict[str, object],
    artifacts: list[LoadedArtifact],
) -> None:
    target = archive_dir.expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(mode=0o755)
    runs_dir = target / "runs"
    runs_dir.mkdir(mode=0o755)
    files = []
    by_seed = sorted(
        artifacts,
        key=lambda artifact: int(artifact.payload["arguments"]["seed"]),
    )
    for index, artifact in enumerate(by_seed, start=1):
        seed = artifact.payload["arguments"]["seed"]
        relative = Path("runs") / f"run-{index:02d}-seed{seed}.json"
        immutable_write_bytes(target / relative, artifact.raw)
        files.append({
            "path": relative.as_posix(),
            "bytes": len(artifact.raw),
            "sha256": artifact.sha256,
        })
    aggregate_raw = (
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    immutable_write_bytes(target / "aggregate.json", aggregate_raw)
    files.append({
        "path": "aggregate.json",
        "bytes": len(aggregate_raw),
        "sha256": hashlib.sha256(aggregate_raw).hexdigest(),
    })
    immutable_write_json(target / "manifest.json", {
        "schema_version": 1,
        "kind": "chunked_prefill_certification_archive",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
    })
    runs_dir.chmod(0o555)
    target.chmod(0o555)
    for path in (target, *target.rglob("*")):
        if path.stat().st_mode & 0o222:
            raise RuntimeError(f"archive path remained writable: {path}")
    complete_marker = target.with_name(target.name + ".COMPLETE")
    immutable_write_bytes(
        complete_marker,
        (
            json.dumps({
                "archive": str(target.resolve()),
                "aggregate_sha256": hashlib.sha256(aggregate_raw).hexdigest(),
                "manifest": "manifest.json",
            }, sort_keys=True)
            + "\n"
        ).encode("utf-8"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if handle_pin_query(args.print_source_sha256, args.show_pin):
        return 0
    _validate_args(args)
    pin = validate_release_pin(args.expected_commit, args.expected_source_sha256)
    artifacts = [load_artifact(path) for path in args.input]
    invocation = [
        sys.executable,
        *(sys.argv if argv is None else [str(Path(__file__).resolve()), *argv]),
    ]
    aggregate = aggregate_artifacts(artifacts, args.tau, pin, invocation)
    destination = args.output if args.output is not None else args.archive_dir
    model_root = Path(aggregate["model"]["resolved_path"])
    try:
        destination.expanduser().resolve().relative_to(model_root)
    except ValueError:
        pass
    else:
        raise ValueError("retained destination must be outside the model directory")
    aggregate["provenance_after_validation"] = validate_release_pin(
        args.expected_commit, args.expected_source_sha256
    )
    if args.output is not None:
        immutable_write_json(args.output, aggregate)
    else:
        write_archive(args.archive_dir, aggregate, artifacts)
    print(json.dumps({
        "destination": str(destination),
        "policy": aggregate["policy"],
        "seeds": aggregate["seeds"],
        "summary": aggregate["summary"],
        "tau": args.tau,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
