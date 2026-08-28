import copy
import json
import sys
from pathlib import Path

import pytest


TESTS_ROOT = Path(__file__).resolve().parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

import validate_speculative_v3_route_compile as VALIDATOR


EAGER_RUN_ID = "0123456789ab4def8abc0123456789ab"
GRAPH_RUN_ID = "fedcba9876544321a987fedcba987654"


def compiler_fixture(mode):
    count = 0 if mode == "eager" else 18
    manifest = [
        {
            "root": "inductor",
            "path": "aa/kernel.py",
            "bytes": 17,
            "sha256": "1" * 64,
        },
        {
            "root": "triton",
            "path": "bb/kernel.json",
            "bytes": 23,
            "sha256": "2" * 64,
        },
    ]
    initial = {
        "counters": {
            "'frames'": {"'ok'": 1, "'total'": 1},
            "'stats'": {"'unique_graphs'": 1},
        },
        "guard_failures": {},
        "graph_break_reasons": [],
        "cache_manifest": manifest,
        "cuda_graph_objects": count,
        "cuda_graph_contexts": count,
    }
    runtime = copy.deepcopy(initial)
    return initial, runtime


def route_artifact(mode):
    run_id = EAGER_RUN_ID if mode == "eager" else GRAPH_RUN_ID
    initial, runtime = compiler_fixture(mode)
    records = []
    for batch_size, effective_k, repetition, phase in VALIDATOR.canonical_record_specs():
        catchup = 4 * batch_size if phase == "cold" else 0
        records.append(
            {
                "route": VALIDATOR.canonical_route(
                    mode,
                    batch_size=batch_size,
                    effective_k=effective_k,
                    phase=phase,
                ),
                "live_batch_size": batch_size,
                "catchup_tokens": catchup,
                "q_shape": [batch_size, effective_k, VALIDATOR.VOCAB_SIZE],
                "q_stride": [
                    VALIDATOR.VOCAB_SIZE,
                    batch_size * VALIDATOR.VOCAB_SIZE,
                    1,
                ],
                "graph_decode_steps": effective_k if mode == "graph" else 0,
                "eager_decode_steps": effective_k if mode == "eager" else 0,
                "target_token_ids": list(range(batch_size)),
                "proposed_token_ids": [
                    list(range(effective_k)) for _ in range(batch_size)
                ],
                "compiler_delta": {},
                "compiler_snapshot_before_sha256": "4" * 64,
                "compiler_snapshot_after_sha256": "4" * 64,
                "rng_neutral": True,
                "context_reset": True,
                "host_result_cuda_free": True,
                "repetition": repetition,
                "phase": phase,
            }
        )
    artifact = {
        "schema": VALIDATOR.INPUT_SCHEMA,
        "run_id": run_id,
        "generated_at": "2026-08-28T00:00:00+00:00",
        "mode": mode,
        "seed": VALIDATOR.SEED,
        "model": "/models/target",
        "draft_model": "/models/draft",
        "configured_k": 2,
        "configuration": {
            "max_num_seqs": 4,
            "max_num_batched_tokens": 512,
            "max_model_len": 512,
            "gpu_memory_utilization": 0.5,
            "repetitions": 2,
            "tensor_parallel_size": 1,
            "top_p_backend": "exact",
        },
        "registry_cardinality": 4 if mode == "eager" else 12,
        "visited_cardinality": 4 if mode == "eager" else 12,
        "route_pretouch_peak_bytes": 4096,
        "capture_ledger_after_init": {
            "cuda_graph_objects": 0 if mode == "eager" else 18,
            "cuda_graph_contexts": 0 if mode == "eager" else 18,
        },
        "capture_ledger_after_runtime": {
            "cuda_graph_objects": 0 if mode == "eager" else 18,
            "cuda_graph_contexts": 0 if mode == "eager" else 18,
        },
        "all_draft_intervals_compiler_state_unchanged": True,
        "compiler_state_after_init": initial,
        "compiler_state_after_init_sha256": VALIDATOR.payload_sha256(initial),
        "compiler_state_after_runtime": runtime,
        "compiler_state_after_runtime_sha256": VALIDATOR.payload_sha256(
            runtime
        ),
        "post_init_to_runtime_compiler_delta": {},
        "records": records,
        "provenance": {"retention_eligible": False},
        "retention_eligible": False,
    }
    return artifact


def interval_log(labels, *, outside_line=True):
    lines = []
    if outside_line:
        lines.append("[__recompiles] an initialization event outside all draft intervals")
    for label in labels:
        lines.append(f"{VALIDATOR.BEGIN_PREFIX}{label}")
        lines.append(f"{VALIDATOR.END_PREFIX}{label}")
    return ("\n".join(lines) + "\n").encode()


@pytest.mark.parametrize("mode", ("eager", "graph"))
def test_route_artifact_and_run_bound_log_accept_canonical_contract(mode):
    artifact = route_artifact(mode)

    labels = VALIDATOR.validate_raw_artifact(artifact, mode)
    result = VALIDATOR.validate_interval_log(
        interval_log(labels),
        expected_labels=labels,
    )

    assert len(labels) == 32
    assert len(set(labels)) == 32
    assert result["interval_count"] == 32
    assert result["marker_count"] == 64
    assert result["strict_nonnested_pairs"] is True
    assert result["draft_interval_output_empty"] is True


def test_route_artifact_accepts_honest_whole_run_compiler_delta():
    artifact = route_artifact("eager")
    runtime = artifact["compiler_state_after_runtime"]
    runtime["counters"] = {
        "'frames'": {"'ok'": 2, "'total'": 2},
        "'stats'": {"'unique_graphs'": 2},
    }
    artifact["compiler_state_after_runtime_sha256"] = (
        VALIDATOR.payload_sha256(runtime)
    )
    artifact["post_init_to_runtime_compiler_delta"] = (
        VALIDATOR.expected_compiler_delta(
            artifact["compiler_state_after_init"],
            runtime,
        )
    )

    VALIDATOR.validate_raw_artifact(artifact, "eager")


@pytest.mark.parametrize(
    "mutation,match",
    (
        (
            lambda value: value["records"][0].__setitem__(
                "q_shape", [1, 2, VALIDATOR.VOCAB_SIZE]
            ),
            "q shape drifted",
        ),
        (
            lambda value: value["records"].__setitem__(
                0, value["records"][1]
            ),
            "phase drifted|route drifted|catch-up",
        ),
        (
            lambda value: value["records"][0].__setitem__(
                "compiler_snapshot_after_sha256", "5" * 64
            ),
            "compiler snapshots differ",
        ),
        (
            lambda value: value["capture_ledger_after_runtime"].__setitem__(
                "cuda_graph_objects", 1
            ),
            "must be 0|created new CUDA graphs",
        ),
        (
            lambda value: value.__setitem__(
                "compiler_state_after_init_sha256", "6" * 64
            ),
            "initial compiler snapshot hash mismatch",
        ),
        (
            lambda value: value.__setitem__(
                "compiler_state_after_runtime_sha256", "7" * 64
            ),
            "runtime compiler snapshot hash mismatch",
        ),
        (
            lambda value: value.__setitem__(
                "post_init_to_runtime_compiler_delta",
                {"counters": {"before": {}, "after": {}}},
            ),
            "post-init compiler delta summary drifted",
        ),
    ),
)
def test_route_artifact_rejects_semantic_corruption(mutation, match):
    artifact = route_artifact("eager")
    mutation(artifact)

    with pytest.raises(AssertionError, match=match):
        VALIDATOR.validate_raw_artifact(artifact, "eager")


def test_route_artifact_rejects_boolean_integer_alias():
    artifact = route_artifact("eager")
    artifact["configuration"]["max_num_seqs"] = True

    with pytest.raises(AssertionError, match="canonical configuration drifted"):
        VALIDATOR.validate_raw_artifact(artifact, "eager")

    artifact = route_artifact("graph")
    artifact["records"][0]["route"]["batch_bucket"] = True
    artifact["records"][0]["q_shape"][0] = True
    with pytest.raises(AssertionError, match="route integer types drifted"):
        VALIDATOR.validate_raw_artifact(artifact, "graph")


def test_route_artifact_rejects_vacuous_compiler_state():
    artifact = route_artifact("eager")
    artifact["compiler_state_after_init"]["counters"].pop("'stats'")
    artifact["compiler_state_after_init_sha256"] = VALIDATOR.payload_sha256(
        artifact["compiler_state_after_init"]
    )
    with pytest.raises(AssertionError, match="unique graphs"):
        VALIDATOR.validate_raw_artifact(artifact, "eager")


def test_route_artifact_rejects_shrinking_runtime_compiler_cache():
    artifact = route_artifact("eager")
    runtime = artifact["compiler_state_after_runtime"]
    runtime["cache_manifest"][0]["bytes"] = 1
    runtime["cache_manifest"][1]["bytes"] = 1
    artifact["compiler_state_after_runtime_sha256"] = (
        VALIDATOR.payload_sha256(runtime)
    )
    artifact["post_init_to_runtime_compiler_delta"] = (
        VALIDATOR.expected_compiler_delta(
            artifact["compiler_state_after_init"],
            runtime,
        )
    )
    with pytest.raises(AssertionError, match="shrank below constructor"):
        VALIDATOR.validate_raw_artifact(artifact, "eager")


def test_route_artifact_rejects_incomplete_graph_registry():
    artifact = route_artifact("graph")
    for record in artifact["records"]:
        if record["live_batch_size"] == 1:
            record["route"]["batch_bucket"] = 2

    with pytest.raises(AssertionError, match="route drifted|coverage is incomplete"):
        VALIDATOR.validate_raw_artifact(artifact, "graph")


def test_interval_log_rejects_foreign_run_id():
    labels = VALIDATOR.validate_raw_artifact(route_artifact("eager"), "eager")
    payload = interval_log(labels).replace(EAGER_RUN_ID.encode(), GRAPH_RUN_ID.encode(), 1)

    with pytest.raises(AssertionError, match="BEGIN marker order/content drifted"):
        VALIDATOR.validate_interval_log(payload, expected_labels=labels)


def test_interval_log_rejects_nested_markers():
    labels = VALIDATOR.validate_raw_artifact(route_artifact("eager"), "eager")
    begin = f"{VALIDATOR.BEGIN_PREFIX}{labels[0]}\n".encode()
    payload = interval_log(labels).replace(begin, begin + begin, 1)

    with pytest.raises(AssertionError, match="nested BEGIN"):
        VALIDATOR.validate_interval_log(payload, expected_labels=labels)


def test_interval_log_rejects_reordered_record_intervals():
    labels = VALIDATOR.validate_raw_artifact(route_artifact("eager"), "eager")
    reordered = labels.copy()
    reordered[0], reordered[4] = reordered[4], reordered[0]

    with pytest.raises(AssertionError, match="BEGIN marker order/content drifted"):
        VALIDATOR.validate_interval_log(
            interval_log(reordered),
            expected_labels=labels,
        )


@pytest.mark.parametrize(
    "inside_line",
    (
        "[__recompiles] Recompiling function forward",
        "[__graph_breaks] Graph break from user code",
        "BackendCompilerFailed: compile error",
        "an otherwise unexplained line",
    ),
)
def test_interval_log_rejects_any_output_inside_guarded_interval(inside_line):
    labels = VALIDATOR.validate_raw_artifact(route_artifact("eager"), "eager")
    begin = f"{VALIDATOR.BEGIN_PREFIX}{labels[0]}\n".encode()
    payload = interval_log(labels).replace(
        begin,
        begin + inside_line.encode() + b"\n",
        1,
    )

    with pytest.raises(AssertionError, match="unexpected compiler/log output"):
        VALIDATOR.validate_interval_log(payload, expected_labels=labels)


def test_interval_log_rejects_missing_and_malformed_markers():
    labels = VALIDATOR.validate_raw_artifact(route_artifact("eager"), "eager")
    payload = interval_log(labels)
    end = f"{VALIDATOR.END_PREFIX}{labels[-1]}\n".encode()
    before_last_end, after_last_end = payload.rsplit(end, 1)
    with pytest.raises(AssertionError, match="ended inside a draft interval"):
        VALIDATOR.validate_interval_log(
            before_last_end + after_last_end,
            expected_labels=labels,
        )

    malformed = payload.replace(
        VALIDATOR.BEGIN_PREFIX.encode(),
        b"V3_DRAFT_INTERVAL_BEGUN ",
        1,
    )
    with pytest.raises(AssertionError, match="malformed draft marker"):
        VALIDATOR.validate_interval_log(malformed, expected_labels=labels)


def test_interval_log_rejects_non_utf8_and_nul():
    labels = VALIDATOR.validate_raw_artifact(route_artifact("eager"), "eager")
    with pytest.raises(AssertionError, match="not UTF-8"):
        VALIDATOR.validate_interval_log(b"\xff", expected_labels=labels)
    with pytest.raises(AssertionError, match="NUL"):
        VALIDATOR.validate_interval_log(b"\0", expected_labels=labels)


def test_artifact_json_fixture_round_trips_strictly():
    artifact = route_artifact("graph")
    payload = json.dumps(artifact, allow_nan=False).encode()
    loaded = VALIDATOR.load_strict_json_bytes(payload)

    assert VALIDATOR.validate_raw_artifact(loaded, "graph")
