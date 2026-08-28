"""Offline validator for retained speculative-V3 route/compile evidence.

The GPU producer writes one JSON artifact and one stderr log per execution
mode.  This validator consumes the eager and graph pairs together.  It treats
both files as hostile input, validates the complete canonical workload and
binds every JSON record to an exact, run-specific stderr interval.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from _speculative_v3_evidence import (
    finalize_evidence,
    load_strict_json_bytes,
    path_is_within,
    prepare_evidence,
    read_registered_file,
    require_independent_compiler_cache_roots,
    sha256_bytes,
    validate_output_paths,
    validate_retained_provenance,
    write_json_exclusive,
)


INPUT_SCHEMA = "nano-vllm-speculative-v3-route-compile-v2"
OUTPUT_SCHEMA = "nano-vllm-speculative-v3-route-compile-validation-v2"
RUNNER_PATH = "tests/run_speculative_v3_route_compile.py"
SEED = 20260828
VOCAB_SIZE = 151936
RECORD_COUNT = 32
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_LOG_BYTES = 64 * 1024 * 1024
SHA256_RE = re.compile(r"[0-9a-f]{64}")
RUN_ID_RE = re.compile(r"[0-9a-f]{12}4[0-9a-f]{3}[89ab][0-9a-f]{15}")
MARKER_FRAGMENT = "V3_DRAFT_INTERVAL_"
BEGIN_PREFIX = "V3_DRAFT_INTERVAL_BEGIN "
END_PREFIX = "V3_DRAFT_INTERVAL_END "
EXPECTED_RUNTIME_IMPORTS = {
    "nanovllm": "nanovllm/__init__.py",
    "LLM": "nanovllm/llm.py",
    "LLMEngine": "nanovllm/engine/llm_engine.py",
    "ModelRunner": "nanovllm/engine/model_runner.py",
    "Scheduler": "nanovllm/engine/scheduler.py",
    "Sampler": "nanovllm/layers/sampler.py",
    "SamplingParams": "nanovllm/sampling_params.py",
    "DraftRouteRegistry": "nanovllm/engine/speculative_routes.py",
}
DISABLED_COMPILER_CACHE_FLAGS = (
    "TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE",
    "TORCHINDUCTOR_AUTOGRAD_CACHE",
    "TORCHINDUCTOR_AUTOGRAD_REMOTE_CACHE",
    "TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE",
    "TORCHINDUCTOR_BUNDLED_AUTOTUNE_REMOTE_CACHE",
    "TORCH_DYNAMO_AUTOMATIC_DYNAMIC_LOCAL_PGO",
    "TORCH_DYNAMO_AUTOMATIC_DYNAMIC_REMOTE_PGO",
)
ROOT_KEYS = {
    "schema",
    "run_id",
    "generated_at",
    "mode",
    "seed",
    "model",
    "draft_model",
    "configured_k",
    "configuration",
    "registry_cardinality",
    "visited_cardinality",
    "route_pretouch_peak_bytes",
    "capture_ledger_after_init",
    "capture_ledger_after_runtime",
    "all_draft_intervals_compiler_state_unchanged",
    "compiler_state_after_init",
    "compiler_state_after_init_sha256",
    "compiler_state_after_runtime",
    "compiler_state_after_runtime_sha256",
    "post_init_to_runtime_compiler_delta",
    "records",
    "provenance",
    "retention_eligible",
}
RECORD_KEYS = {
    "route",
    "live_batch_size",
    "catchup_tokens",
    "q_shape",
    "q_stride",
    "graph_decode_steps",
    "eager_decode_steps",
    "target_token_ids",
    "proposed_token_ids",
    "compiler_delta",
    "compiler_snapshot_before_sha256",
    "compiler_snapshot_after_sha256",
    "rng_neutral",
    "context_reset",
    "host_result_cuda_free",
    "repetition",
    "phase",
}
COMPILER_KEYS = {
    "counters",
    "guard_failures",
    "graph_break_reasons",
    "cache_manifest",
    "cuda_graph_objects",
    "cuda_graph_contexts",
}


def require(condition: Any, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def exact_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    require(type(value) is int, f"{name} must be a JSON integer")
    if minimum is not None:
        require(value >= minimum, f"{name} must be >= {minimum}")
    return value


def exact_bool(value: Any, name: str) -> bool:
    require(type(value) is bool, f"{name} must be a JSON boolean")
    return value


def require_sha256(value: Any, name: str) -> str:
    require(
        isinstance(value, str) and SHA256_RE.fullmatch(value),
        f"{name} must be lowercase SHA-256",
    )
    return value


def payload_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return sha256_bytes(payload.encode("utf-8"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate paired eager/graph V3 route JSON and stderr logs."
    )
    parser.add_argument("--eager-json", required=True)
    parser.add_argument("--eager-log", required=True)
    parser.add_argument("--graph-json", required=True)
    parser.add_argument("--graph-log", required=True)
    parser.add_argument("--evidence-root")
    parser.add_argument(
        "--expected-commit",
        help="full comparator/producer commit required by --retained",
    )
    parser.add_argument(
        "--retained",
        action="store_true",
        help="enforce exact clean-SHA producer and validator provenance",
    )
    parser.add_argument("--output")
    return parser.parse_args(argv)


def is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def require_no_symlink_components(path: Path, stop: Path) -> None:
    current = path.absolute()
    stop = stop.absolute()
    require(is_within(current, stop), f"artifact escapes evidence root: {path}")
    while True:
        require(
            not current.is_symlink(),
            f"artifact path contains a symlink: {current}",
        )
        if current == stop:
            break
        current = current.parent


def confined_existing_path(path: Path, root: Path) -> Path:
    lexical = path.expanduser()
    if not lexical.is_absolute():
        lexical = Path.cwd() / lexical
    require_no_symlink_components(lexical, root)
    resolved = lexical.resolve()
    require(is_within(resolved, root), f"artifact resolves outside evidence root: {path}")
    return resolved


def canonical_batch_bucket(mode: str, batch_size: int) -> int:
    if mode == "eager":
        return 4
    return 1 if batch_size == 1 else 2 if batch_size == 2 else 4


def canonical_route(
    mode: str,
    *,
    batch_size: int,
    effective_k: int,
    phase: str,
) -> dict[str, Any]:
    return {
        "schema": "draft-discard-v1",
        "execution_mode": "eager_dynamic" if mode == "eager" else "cuda_graph",
        "batch_bucket": canonical_batch_bucket(mode, batch_size),
        "effective_k": effective_k,
        "catchup_family": "paged_eager_dynamic_v1" if phase == "cold" else "none",
        "sampler_envelope": "exact_all_compositions_worst_case_v1",
    }


def canonical_record_specs():
    return tuple(
        (batch_size, effective_k, repetition, phase)
        for batch_size in range(1, 5)
        for effective_k in range(1, 3)
        for repetition in range(2)
        for phase in ("cold", "warm")
    )


def canonical_registry(mode: str) -> set[tuple[tuple[str, Any], ...]]:
    batch_sizes = (1, 2, 3, 4) if mode == "graph" else (1,)
    return {
        tuple(
            sorted(
                canonical_route(
                    mode,
                    batch_size=batch_size,
                    effective_k=effective_k,
                    phase=phase,
                ).items()
            )
        )
        for batch_size in batch_sizes
        for effective_k in (1, 2)
        for phase in ("cold", "warm")
    }


def capture_ledger(value: Any, *, mode: str, name: str) -> dict[str, int]:
    expected_count = 0 if mode == "eager" else 18
    require(
        isinstance(value, dict)
        and set(value) == {"cuda_graph_objects", "cuda_graph_contexts"},
        f"{name} capture ledger schema drifted",
    )
    for field in ("cuda_graph_objects", "cuda_graph_contexts"):
        exact_int(value[field], f"{name}.{field}", minimum=0)
        require(
            value[field] == expected_count,
            f"{name}.{field} must be {expected_count} for canonical {mode}",
        )
    return value


def validate_cache_manifest(value: Any) -> list[dict[str, Any]]:
    require(isinstance(value, list) and value, "compiler cache manifest must be non-empty")
    identities = []
    for index, item in enumerate(value):
        name = f"compiler cache manifest[{index}]"
        require(
            isinstance(item, dict)
            and set(item) == {"root", "path", "bytes", "sha256"},
            f"{name} schema drifted",
        )
        require(item["root"] in ("inductor", "triton"), f"{name} root drifted")
        raw_path = item["path"]
        require(isinstance(raw_path, str) and raw_path, f"{name} path is invalid")
        require("\\" not in raw_path, f"{name} path must use POSIX separators")
        relative = PurePosixPath(raw_path)
        require(
            not relative.is_absolute()
            and relative.parts
            and all(part not in ("", ".", "..") for part in relative.parts)
            and relative.as_posix() == raw_path,
            f"{name} path is not normalized and relative",
        )
        exact_int(item["bytes"], f"{name}.bytes", minimum=0)
        require_sha256(item["sha256"], f"{name}.sha256")
        identities.append((item["root"], raw_path))
    require(len(set(identities)) == len(identities), "compiler cache manifest has duplicate paths")
    require(set(root for root, _ in identities) == {"inductor", "triton"}, "compiler cache manifest omitted a cache root")
    require(
        identities == sorted(identities, key=lambda item: ((0 if item[0] == "inductor" else 1), item[1])),
        "compiler cache manifest order drifted",
    )
    return value


def cache_manifest_summary(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "file_count": len(manifest),
        "total_bytes": sum(item["bytes"] for item in manifest),
        "manifest_sha256": payload_sha256(manifest),
    }


def validate_initial_compiler_state(
    value: Any,
    *,
    ledger: Mapping[str, int],
) -> dict[str, Any]:
    require(
        isinstance(value, dict) and set(value) == COMPILER_KEYS,
        "initial compiler snapshot schema drifted",
    )
    require(isinstance(value["counters"], dict), "initial compiler counters must be an object")
    require(isinstance(value["guard_failures"], dict), "initial guard failures must be an object")
    require(isinstance(value["graph_break_reasons"], list), "initial graph-break reasons must be a list")
    stats = value["counters"].get("'stats'", {})
    require(
        isinstance(stats, dict)
        and exact_int(
            stats.get("'unique_graphs'"),
            "initial compiler unique graphs",
            minimum=1,
        )
        >= 1,
        "initial compiler counters prove no compiled graph",
    )
    manifest = validate_cache_manifest(value["cache_manifest"])
    require(
        sum(item["bytes"] for item in manifest) > 0,
        "initial compiler cache manifest is byte-empty",
    )
    for field in ("cuda_graph_objects", "cuda_graph_contexts"):
        exact_int(value[field], f"initial compiler {field}", minimum=0)
        require(value[field] == ledger[field], f"initial compiler {field} disagrees with capture ledger")
    return value


def validate_runtime_compiler_state(
    value: Any,
    *,
    ledger: Mapping[str, int],
) -> dict[str, Any]:
    require(
        isinstance(value, dict) and set(value) == COMPILER_KEYS,
        "runtime compiler snapshot schema drifted",
    )
    require(isinstance(value["counters"], dict), "runtime compiler counters must be an object")
    require(isinstance(value["guard_failures"], dict), "runtime guard failures must be an object")
    require(isinstance(value["graph_break_reasons"], list), "runtime graph-break reasons must be a list")
    manifest = validate_cache_manifest(value["cache_manifest"])
    require(
        sum(item["bytes"] for item in manifest) > 0,
        "runtime compiler cache manifest is byte-empty",
    )
    stats = value["counters"].get("'stats'", {})
    require(
        isinstance(stats, dict)
        and exact_int(
            stats.get("'unique_graphs'"),
            "runtime compiler unique graphs",
            minimum=1,
        )
        >= 1,
        "runtime compiler counters prove no compiled graph",
    )
    for field in ("cuda_graph_objects", "cuda_graph_contexts"):
        exact_int(value[field], f"runtime compiler {field}", minimum=0)
        require(value[field] == ledger[field], f"runtime compiler {field} disagrees with capture ledger")
    return value


def expected_compiler_delta(
    initial: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    result = {}
    for key in COMPILER_KEYS:
        before = (
            cache_manifest_summary(initial[key])
            if key == "cache_manifest"
            else initial[key]
        )
        after = runtime[key]
        if key == "cache_manifest":
            after = cache_manifest_summary(after)
        if before != after:
            result[key] = {"before": before, "after": after}
    return result


def validate_token_ids(value: Any, *, rows: int, columns: int | None, name: str) -> None:
    require(isinstance(value, list) and len(value) == rows, f"{name} row count drifted")
    if columns is None:
        require(
            all(type(token) is int and 0 <= token < VOCAB_SIZE for token in value),
            f"{name} token IDs are invalid",
        )
        return
    require(
        all(
            isinstance(row, list)
            and len(row) == columns
            and all(type(token) is int and 0 <= token < VOCAB_SIZE for token in row)
            for row in value
        ),
        f"{name} proposal matrix drifted",
    )


def validate_record(
    value: Any,
    *,
    mode: str,
    batch_size: int,
    effective_k: int,
    repetition: int,
    phase: str,
    record_index: int,
) -> str:
    label = f"B={batch_size},K={effective_k},rep={repetition},phase={phase}"
    require(isinstance(value, dict) and set(value) == RECORD_KEYS, f"{label} record schema drifted")
    route = value["route"]
    require(route == canonical_route(mode, batch_size=batch_size, effective_k=effective_k, phase=phase), f"{label} route drifted")
    require(
        isinstance(route, dict)
        and type(route["batch_bucket"]) is int
        and type(route["effective_k"]) is int,
        f"{label} route integer types drifted",
    )
    exact_int(value["live_batch_size"], f"{label}.live_batch_size")
    require(value["live_batch_size"] == batch_size, f"{label} live batch size drifted")
    exact_int(value["repetition"], f"{label}.repetition")
    require(value["repetition"] == repetition, f"{label} repetition drifted")
    require(value["phase"] == phase, f"{label} phase drifted")
    catchup_tokens = exact_int(value["catchup_tokens"], f"{label}.catchup_tokens", minimum=0)
    require(catchup_tokens == (4 * batch_size if phase == "cold" else 0), f"{label} catch-up count drifted")
    require(
        value["q_shape"] == [batch_size, effective_k, VOCAB_SIZE]
        and all(type(item) is int for item in value["q_shape"]),
        f"{label} q shape drifted",
    )
    require(
        value["q_stride"] == [VOCAB_SIZE, batch_size * VOCAB_SIZE, 1]
        and all(type(item) is int for item in value["q_stride"]),
        f"{label} q stride drifted",
    )
    eager_steps = exact_int(value["eager_decode_steps"], f"{label}.eager_decode_steps", minimum=0)
    graph_steps = exact_int(value["graph_decode_steps"], f"{label}.graph_decode_steps", minimum=0)
    require(
        (eager_steps, graph_steps)
        == ((effective_k, 0) if mode == "eager" else (0, effective_k)),
        f"{label} decode-path accounting drifted",
    )
    validate_token_ids(value["target_token_ids"], rows=batch_size, columns=None, name=f"{label}.target_token_ids")
    validate_token_ids(value["proposed_token_ids"], rows=batch_size, columns=effective_k, name=f"{label}.proposed_token_ids")
    require(value["compiler_delta"] == {}, f"{label} draft compiler delta is not empty")
    before_sha = require_sha256(value["compiler_snapshot_before_sha256"], f"{label} compiler before SHA")
    after_sha = require_sha256(value["compiler_snapshot_after_sha256"], f"{label} compiler after SHA")
    require(before_sha == after_sha, f"{label} compiler snapshots differ")
    for field in ("rng_neutral", "context_reset", "host_result_cuda_free"):
        exact_bool(value[field], f"{label}.{field}")
        require(value[field] is True, f"{label}.{field} is not true")
    return (
        f"run={{run_id}},record={record_index},batch={batch_size},"
        f"bucket={route['batch_bucket']},"
        f"k={effective_k},catchup={catchup_tokens}"
    )


def validate_raw_artifact(value: Any, expected_mode: str) -> list[str]:
    require(isinstance(value, dict) and set(value) == ROOT_KEYS, f"{expected_mode} artifact root schema drifted")
    require(value["schema"] == INPUT_SCHEMA, f"unexpected {expected_mode} artifact schema")
    require(value["mode"] == expected_mode, f"expected {expected_mode} artifact")
    run_id = value["run_id"]
    require(isinstance(run_id, str) and RUN_ID_RE.fullmatch(run_id), f"{expected_mode} run_id is not UUID4 hex")
    generated_at = value["generated_at"]
    require(isinstance(generated_at, str), f"{expected_mode} generated_at is invalid")
    try:
        generated = datetime.fromisoformat(generated_at)
    except ValueError as error:
        raise AssertionError(f"{expected_mode} generated_at is not ISO-8601") from error
    require(generated.tzinfo is not None and generated.utcoffset() is not None, f"{expected_mode} generated_at must be timezone-aware")
    require(value["seed"] == SEED and type(value["seed"]) is int, f"{expected_mode} seed drifted")
    require(value["configured_k"] == 2 and type(value["configured_k"]) is int, f"{expected_mode} configured K drifted")
    require(isinstance(value["model"], str) and value["model"], f"{expected_mode} target model path is invalid")
    require(isinstance(value["draft_model"], str) and value["draft_model"], f"{expected_mode} draft model path is invalid")
    configuration = value["configuration"]
    require(
        isinstance(configuration, dict)
        and set(configuration)
        == {
            "max_num_seqs",
            "max_num_batched_tokens",
            "max_model_len",
            "gpu_memory_utilization",
            "repetitions",
            "tensor_parallel_size",
            "top_p_backend",
        },
        f"{expected_mode} configuration schema drifted",
    )
    expected_configuration = {
        "max_num_seqs": 4,
        "max_num_batched_tokens": 512,
        "max_model_len": 512,
        "gpu_memory_utilization": 0.5,
        "repetitions": 2,
        "tensor_parallel_size": 1,
        "top_p_backend": "exact",
    }
    require(configuration == expected_configuration, f"{expected_mode} canonical configuration drifted")
    for field in ("max_num_seqs", "max_num_batched_tokens", "max_model_len", "repetitions", "tensor_parallel_size"):
        exact_int(configuration[field], f"configuration.{field}")
    require(type(configuration["gpu_memory_utilization"]) is float, "gpu_memory_utilization must be a JSON float")

    expected_registry = canonical_registry(expected_mode)
    expected_cardinality = 4 if expected_mode == "eager" else 12
    exact_int(value["registry_cardinality"], f"{expected_mode}.registry_cardinality")
    exact_int(value["visited_cardinality"], f"{expected_mode}.visited_cardinality")
    require(value["registry_cardinality"] == expected_cardinality, f"{expected_mode} registry cardinality drifted")
    require(value["visited_cardinality"] == expected_cardinality, f"{expected_mode} visited cardinality drifted")
    exact_int(value["route_pretouch_peak_bytes"], f"{expected_mode}.route_pretouch_peak_bytes", minimum=1)
    init_ledger = capture_ledger(value["capture_ledger_after_init"], mode=expected_mode, name="after-init")
    runtime_ledger = capture_ledger(value["capture_ledger_after_runtime"], mode=expected_mode, name="after-runtime")
    require(runtime_ledger == init_ledger, f"{expected_mode} runtime created new CUDA graphs")
    exact_bool(value["all_draft_intervals_compiler_state_unchanged"], "all_draft_intervals_compiler_state_unchanged")
    require(value["all_draft_intervals_compiler_state_unchanged"] is True, f"{expected_mode} draft intervals are not compiler-neutral")

    initial = validate_initial_compiler_state(value["compiler_state_after_init"], ledger=init_ledger)
    require_sha256(value["compiler_state_after_init_sha256"], "initial compiler snapshot SHA-256")
    require(payload_sha256(initial) == value["compiler_state_after_init_sha256"], "initial compiler snapshot hash mismatch")
    runtime = validate_runtime_compiler_state(
        value["compiler_state_after_runtime"],
        ledger=runtime_ledger,
    )
    require_sha256(
        value["compiler_state_after_runtime_sha256"],
        "runtime compiler snapshot SHA-256",
    )
    require(
        payload_sha256(runtime)
        == value["compiler_state_after_runtime_sha256"],
        "runtime compiler snapshot hash mismatch",
    )
    require(
        len(runtime["cache_manifest"]) >= len(initial["cache_manifest"])
        and sum(item["bytes"] for item in runtime["cache_manifest"])
        >= sum(item["bytes"] for item in initial["cache_manifest"]),
        "runtime compiler cache summary shrank below constructor state",
    )
    delta = value["post_init_to_runtime_compiler_delta"]
    require(isinstance(delta, dict), "post-init compiler delta must be an object")
    require(delta == expected_compiler_delta(initial, runtime), "post-init compiler delta summary drifted")

    records = value["records"]
    require(isinstance(records, list) and len(records) == RECORD_COUNT, f"{expected_mode} requires exactly {RECORD_COUNT} records")
    marker_templates = []
    for record_index, (record, spec) in enumerate(
        zip(records, canonical_record_specs(), strict=True)
    ):
        marker_templates.append(
            validate_record(
                record,
                mode=expected_mode,
                batch_size=spec[0],
                effective_k=spec[1],
                repetition=spec[2],
                phase=spec[3],
                record_index=record_index,
            )
        )
    require(
        len(set(marker_templates)) == RECORD_COUNT,
        f"{expected_mode} marker labels are not unique per record",
    )
    visited = {tuple(sorted(record["route"].items())) for record in records}
    require(visited == expected_registry, f"{expected_mode} runtime route coverage is incomplete")
    exact_bool(value["retention_eligible"], f"{expected_mode}.retention_eligible")
    provenance = value["provenance"]
    require(isinstance(provenance, dict), f"{expected_mode} provenance is missing")
    require(provenance.get("retention_eligible") == value["retention_eligible"], f"{expected_mode} retention flag disagrees with provenance")
    return [template.format(run_id=run_id) for template in marker_templates]


def validate_interval_log(payload: bytes, *, expected_labels: list[str]) -> dict[str, Any]:
    require(b"\0" not in payload, "stderr log contains a NUL byte")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AssertionError("stderr log is not UTF-8") from error
    require(len(expected_labels) == RECORD_COUNT, "validator expected-label registry drifted")
    active_label = None
    next_index = 0
    intervals = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if line.startswith(BEGIN_PREFIX):
            require(active_label is None, f"nested BEGIN marker at stderr line {line_number}")
            require(next_index < len(expected_labels), f"extra BEGIN marker at stderr line {line_number}")
            label = line[len(BEGIN_PREFIX) :]
            require(label == expected_labels[next_index], f"BEGIN marker order/content drifted at interval {next_index}")
            active_label = label
            intervals.append({"label": label, "begin_line": line_number})
            continue
        if line.startswith(END_PREFIX):
            require(active_label is not None, f"orphan END marker at stderr line {line_number}")
            label = line[len(END_PREFIX) :]
            require(label == active_label, f"END marker does not match active interval at stderr line {line_number}")
            intervals[-1]["end_line"] = line_number
            active_label = None
            next_index += 1
            continue
        require(MARKER_FRAGMENT not in line, f"malformed draft marker at stderr line {line_number}")
        if active_label is not None:
            require(not line.strip(), f"unexpected compiler/log output inside draft interval {next_index} at stderr line {line_number}: {line[:200]!r}")
    require(active_label is None, "stderr log ended inside a draft interval")
    require(next_index == len(expected_labels), f"stderr log has {next_index} complete intervals, expected {len(expected_labels)}")
    require(all("end_line" in interval for interval in intervals), "stderr interval ledger is incomplete")
    return {
        "interval_count": len(intervals),
        "marker_count": len(intervals) * 2,
        "strict_nonnested_pairs": True,
        "record_order_exact": True,
        "draft_interval_output_empty": True,
        "intervals": intervals,
    }


def load_route_artifact(path: Path, *, mode: str, root: Path):
    resolved = confined_existing_path(path, root)
    payload, descriptor = read_registered_file(resolved, max_bytes=MAX_JSON_BYTES)
    value = load_strict_json_bytes(payload)
    labels = validate_raw_artifact(value, mode)
    return value, labels, {"path": resolved.relative_to(root).as_posix(), **descriptor}


def load_route_log(path: Path, *, labels: list[str], root: Path):
    resolved = confined_existing_path(path, root)
    payload, descriptor = read_registered_file(resolved, max_bytes=MAX_LOG_BYTES)
    validation = validate_interval_log(payload, expected_labels=labels)
    return validation, {"path": resolved.relative_to(root).as_posix(), **descriptor}


def normalized_validated_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(value)
    environment = dict(normalized["environment"])
    selected = dict(environment["selected_environment"])
    selected.pop("TORCHINDUCTOR_CACHE_DIR", None)
    selected.pop("TRITON_CACHE_DIR", None)
    environment["selected_environment"] = selected
    normalized["environment"] = environment
    return normalized


def validate_recorded_a100_environment(value: Mapping[str, Any]) -> None:
    hardware = value.get("hardware")
    require(isinstance(hardware, dict), "producer hardware environment is missing")
    require(hardware.get("cuda_available") is True, "producer did not record CUDA")
    require(hardware.get("device_count") == 1 and type(hardware.get("device_count")) is int, "producer did not record exactly one visible GPU")
    devices = hardware.get("devices")
    require(isinstance(devices, list) and len(devices) == 1, "producer device registry drifted")
    device = devices[0]
    require(isinstance(device, dict), "producer device record is invalid")
    require(device.get("index") == 0 and type(device.get("index")) is int, "producer device index drifted")
    require(device.get("name") == "NVIDIA A100-SXM4-40GB", "producer GPU is not NVIDIA A100-SXM4-40GB")
    require(device.get("compute_capability") == [8, 0], "producer GPU compute capability is not 8.0")
    memory = exact_int(device.get("total_memory_bytes"), "producer GPU memory", minimum=39 * 1024**3)
    require(memory < 48 * 1024**3, "producer GPU memory does not identify a 40GB device")
    uuid = device.get("uuid")
    require(isinstance(uuid, str) and uuid and not uuid.startswith("MIG-"), "producer GPU UUID is missing or MIG")


def validate_compiler_environment(
    value: Mapping[str, Any],
    *,
    repo_root: Path,
    model_roots: tuple[Path, ...],
) -> tuple[Path, Path]:
    selected = value.get("selected_environment")
    require(isinstance(selected, dict), "producer selected environment is missing")
    for name in DISABLED_COMPILER_CACHE_FLAGS:
        require(selected.get(name) == "0", f"producer {name} was not disabled")
    for name in ("TORCH_COMPILE_DISABLE", "TORCHDYNAMO_DISABLE"):
        require(selected.get(name) in (None, "", "0"), f"producer {name} disabled compilation")
    torch_logs = {item.strip() for item in (selected.get("TORCH_LOGS") or "").split(",")}
    require({"recompiles", "graph_breaks"}.issubset(torch_logs), "producer TORCH_LOGS omitted recompiles or graph_breaks")
    cache_roots = []
    for name in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        raw = selected.get(name)
        require(isinstance(raw, str) and raw, f"producer {name} is missing")
        path = Path(raw)
        require(path.is_absolute() and path.resolve() == path, f"producer {name} is not canonical")
        require(not path_is_within(path, repo_root), f"producer {name} was inside source checkout")
        require(not any(path_is_within(path, model_root) for model_root in model_roots), f"producer {name} was inside a model directory")
        cache_roots.append(path)
    first, second = cache_roots
    require(first != second and not path_is_within(first, second) and not path_is_within(second, first), "producer compiler cache roots overlap")
    return first, second


def producer_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model")
    parser.add_argument("--mode", choices=("eager", "graph"), required=True)
    parser.add_argument("--configured-k", type=int, default=2)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--retained", action="store_true")
    parser.add_argument("--output", required=True)
    return parser


def resolve_invocation_path(value: str, cwd: Path) -> Path:
    path = Path(value).expanduser()
    return (cwd / path if not path.is_absolute() else path).resolve()


def validate_producer_invocation(
    artifact: Mapping[str, Any],
    *,
    artifact_path: Path,
    expected_commit: str,
) -> None:
    provenance = artifact["provenance"]
    invocation = provenance.get("invocation")
    cwd_value = provenance.get("cwd")
    require(isinstance(invocation, list) and invocation, "producer invocation is missing")
    require(isinstance(cwd_value, str), "producer cwd is missing")
    cwd = Path(cwd_value)
    require(cwd.is_absolute() and cwd.resolve() == cwd, "producer cwd is not canonical")
    try:
        arguments = producer_argument_parser().parse_args(invocation[1:])
    except (argparse.ArgumentError, SystemExit) as error:
        raise AssertionError("producer invocation does not satisfy the registered CLI") from error
    require(arguments.retained, "producer invocation omitted --retained")
    require(arguments.expected_commit == expected_commit, "producer invocation commit drifted")
    require(arguments.mode == artifact["mode"], "producer invocation mode drifted")
    require(
        (
            arguments.configured_k,
            arguments.max_num_seqs,
            arguments.max_num_batched_tokens,
            arguments.max_model_len,
            arguments.gpu_memory_utilization,
            arguments.repetitions,
        )
        == (2, 4, 512, 512, 0.5, 2),
        "producer invocation canonical configuration drifted",
    )
    target = resolve_invocation_path(arguments.model, cwd)
    draft = resolve_invocation_path(arguments.draft_model or arguments.model, cwd)
    require(target == Path(artifact["model"]).resolve(), "producer target-model argument drifted")
    require(draft == Path(artifact["draft_model"]).resolve(), "producer draft-model argument drifted")
    require(resolve_invocation_path(arguments.output, cwd) == artifact_path, "producer output argument does not name this JSON artifact")


def main(argv=None):
    args = parse_args(argv)
    input_arguments = tuple(
        Path(value).expanduser()
        for value in (
            args.eager_json,
            args.eager_log,
            args.graph_json,
            args.graph_log,
        )
    )
    if args.evidence_root is None:
        require(not args.retained, "--retained requires --evidence-root")
        root = Path(os.path.commonpath([str(path.absolute().parent) for path in input_arguments]))
    else:
        root = Path(args.evidence_root).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
    require(not root.is_symlink(), "evidence root must not be a symlink")
    root = root.resolve()
    require(root.is_dir(), f"evidence root is not a directory: {root}")

    evidence_context = prepare_evidence(
        script_path=Path(__file__).resolve(),
        retained=args.retained,
        expected_commit=args.expected_commit,
        model_arguments={},
        runtime_imports={},
    )
    comparator_cache_roots = tuple(
        Path(value).expanduser().resolve()
        for value in (
            os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
            os.environ.get("TRITON_CACHE_DIR"),
        )
        if value
    )
    output_path = None
    if args.output is not None:
        output_path = validate_output_paths(
            (Path(args.output),),
            repo_root=evidence_context.repo_root,
            cache_roots=comparator_cache_roots,
        )[0]
        require(is_within(output_path, root), "validation output must be inside evidence root")
    elif args.retained:
        raise AssertionError("--retained requires --output")

    artifacts = {}
    artifact_descriptors = {}
    log_validations = {}
    log_descriptors = {}
    for mode, json_argument, log_argument in (
        ("eager", input_arguments[0], input_arguments[1]),
        ("graph", input_arguments[2], input_arguments[3]),
    ):
        artifact, labels, artifact_descriptor = load_route_artifact(json_argument, mode=mode, root=root)
        log_validation, log_descriptor = load_route_log(log_argument, labels=labels, root=root)
        artifacts[mode] = artifact
        artifact_descriptors[mode] = artifact_descriptor
        log_validations[mode] = log_validation
        log_descriptors[mode] = log_descriptor

    registered_paths = [
        artifact_descriptors[mode]["path"]
        for mode in ("eager", "graph")
    ] + [log_descriptors[mode]["path"] for mode in ("eager", "graph")]
    require(len(set(registered_paths)) == 4, "route JSON/log inputs must be four distinct files")
    require(artifacts["eager"]["run_id"] != artifacts["graph"]["run_id"], "eager and graph runs reused a run_id")

    if args.retained:
        expected_commit = args.expected_commit.lower()
        model_snapshot_cache = {}
        validated_provenance = {}
        producer_cache_roots = []
        for mode in ("eager", "graph"):
            artifact_path = root / artifact_descriptors[mode]["path"]
            identity = validate_retained_provenance(
                artifacts[mode]["provenance"],
                repo_root=evidence_context.repo_root,
                current_source=evidence_context.source_before,
                expected_commit=expected_commit,
                expected_runner_path=RUNNER_PATH,
                expected_model_roles=("target", "draft"),
                expected_runtime_imports=EXPECTED_RUNTIME_IMPORTS,
                model_snapshot_cache=model_snapshot_cache,
            )
            require(Path(artifacts[mode]["model"]).resolve() == Path(identity["models"]["target"]["resolved_path"]), f"{mode} target path disagrees with hashed provenance")
            require(Path(artifacts[mode]["draft_model"]).resolve() == Path(identity["models"]["draft"]["resolved_path"]), f"{mode} draft path disagrees with hashed provenance")
            model_roots = tuple(Path(item["resolved_path"]) for item in identity["models"].values())
            validate_recorded_a100_environment(identity["environment"])
            producer_cache_roots.extend(
                validate_compiler_environment(
                    identity["environment"],
                    repo_root=evidence_context.repo_root,
                    model_roots=model_roots,
                )
            )
            validate_producer_invocation(artifacts[mode], artifact_path=artifact_path, expected_commit=expected_commit)
            validated_provenance[mode] = identity
        require_independent_compiler_cache_roots(
            (validated_provenance["eager"], validated_provenance["graph"])
        )
        normalized = normalized_validated_provenance(validated_provenance["eager"])
        require(normalized == normalized_validated_provenance(validated_provenance["graph"]), "eager/graph validated producer provenance differs beyond compiler cache roots")
        producer_provenance = {
            "pair_identity": normalized,
            "eager": validated_provenance["eager"],
            "graph": validated_provenance["graph"],
        }
        if output_path is not None:
            validate_output_paths(
                (output_path,),
                repo_root=evidence_context.repo_root,
                model_roots=tuple(Path(item["resolved_path"]) for item in validated_provenance["eager"]["models"].values()),
                cache_roots=tuple(comparator_cache_roots) + tuple(producer_cache_roots),
            )
    else:
        producer_provenance = {
            mode: artifacts[mode]["provenance"] for mode in ("eager", "graph")
        }

    validator_provenance = finalize_evidence(evidence_context)
    retention_eligible = bool(
        artifacts["eager"]["retention_eligible"]
        and artifacts["graph"]["retention_eligible"]
        and validator_provenance["retention_eligible"]
    )
    if args.retained:
        require(retention_eligible, "route validation is not retention eligible")
    result = {
        "schema": OUTPUT_SCHEMA,
        "verdict": "pass",
        "retention_eligible": retention_eligible,
        "record_count_per_mode": RECORD_COUNT,
        "route_registry_cardinality": {"eager": 4, "graph": 12},
        "artifact_descriptors": artifact_descriptors,
        "log_descriptors": log_descriptors,
        "log_validations": log_validations,
        "producer_provenance": producer_provenance,
        "validator_provenance": validator_provenance,
        "all_routes_complete": True,
        "all_draft_intervals_compiler_state_unchanged": True,
        "all_stderr_intervals_strict_and_empty": True,
        "capture_ledger_stable": True,
    }
    if output_path is not None:
        write_json_exclusive(output_path, result)
        print(
            "PASS: eager/graph V3 route registries and 64 run-bound draft "
            f"intervals validated; retention_eligible={retention_eligible}, "
            f"output={output_path}",
            flush=True,
        )
    else:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
