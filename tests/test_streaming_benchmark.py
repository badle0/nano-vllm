import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.pr5_scripts import repaired_stream_benchmark as benchmark
from benchmarks.pr5_scripts.validate_repaired_stream_certificate import (
    ACCEPTED_ARTIFACT_SHA256,
    ACCEPTED_COMMIT,
    CertificateValidationError,
    SUPERSEDED_ARTIFACT_SHA256,
    SUPERSEDED_COMMIT,
    validate_archive,
    validate_evidence,
)
from nanovllm import StreamingDetokenizer


def _fake_worker(index):
    request_metrics = {
        key: {"values": [0.0001], "summary": benchmark._summary([0.0001])}
        for key in (
            "caller_ttft",
            "caller_e2e",
            "engine_ttft",
            "engine_e2e",
            "first_token_to_delivery",
            "engine_finish_to_delivery",
        )
    }
    initial_order = (
        ["generate", "stream"] if index % 2 == 0 else ["stream", "generate"]
    )
    timed_rounds = []
    for round_index in range(benchmark.CORE_TIMED_ROUNDS):
        order = initial_order if round_index % 2 == 0 else list(reversed(initial_order))
        resets = [{
            "policy": benchmark.CORE_PREFIX_CACHE_POLICY,
            "route": route,
            "pair_position": position,
            "used_blocks_before_reset": 0,
            "used_blocks_after_route": 0,
            "cached_block_hashes_after_route": 0,
        } for position, route in enumerate(order)]
        timed_rounds.append({
            "round_index": round_index,
            "round_seed": 10_000 + index * benchmark.CORE_TIMED_ROUNDS + round_index,
            "prompts": ["prompt"] * 8,
            "prompt_sha256": f"{index * benchmark.CORE_TIMED_ROUNDS + round_index:064x}",
            "max_prompt_plus_completion_tokens": 200,
            "block_size": 256,
            "measurement_order": order,
            "prefix_cache_resets": resets,
            "generate": {
                "tokens_per_second": 100.0,
                "return_seconds": 1.0,
                "num_tokens": 8 * 128,
                "pair_position": order.index("generate"),
                "memory": {"peak": {"allocated_bytes": 100}},
            },
            "stream": {
                "tokens_per_second": 100.0,
                "first_event_seconds": 0.05,
                "pair_position": order.index("stream"),
                "memory": {"peak": {"allocated_bytes": 100}},
                "event_delivery": {
                    "values": [{"engine_to_caller_seconds": 0.0001}],
                },
                "request_metric_distributions_seconds": request_metrics,
            },
            "tokens_equivalent": True,
            "texts_equivalent": True,
            "paired_stream_throughput_delta_percent": 0.0,
            "caller_exposure_factor": 20.0,
        })
    slow = []
    for sleep_seconds in (0.0, 0.001, 0.004):
        slow.append({
            "consumer_sleep_seconds_per_event": sleep_seconds,
            "median_inter_step_gap_seconds": 0.003 + 8 * sleep_seconds,
            "elapsed_seconds": 1.0,
            "max_pending_events_after_delivery": 7,
            "num_events": 16,
            "num_steps": 2,
        })
    environment = {
        "repository": {"commit": "a" * 40},
        "benchmark_script_sha256": "b" * 64,
        "model": {"parent_manifest_sha256": "c" * 64},
        "software": {"python": "3.12"},
        "cpu": {"model": "test"},
        "gpu": {"total_memory_bytes": 10000},
    }
    return {
        "trial_seed": 1000 + index,
        "prompt_sha256": f"{index:064x}",
        "timed_round_seeds": [item["round_seed"] for item in timed_rounds],
        "timed_round_orders": [item["measurement_order"] for item in timed_rounds],
        "warmup": {
            "full_length_tokens_per_route": 128,
            "prefix_cache_policy": benchmark.CORE_PREFIX_CACHE_POLICY,
            "records": [
                {"route": route, "max_tokens": 128}
                for route in ("generate", "stream")
            ],
        },
        "environment": environment,
        "core": {
            "timed_rounds": timed_rounds,
            "route_medians": {
                "generate": {"tokens_per_second": 100.0, "return_seconds": 1.0},
                "stream": {"tokens_per_second": 100.0, "first_event_seconds": 0.05},
            },
            "tokens_equivalent": True,
            "texts_equivalent": True,
            "prefix_cache_policy": benchmark.CORE_PREFIX_CACHE_POLICY,
            "paired_stream_throughput_delta_percent": 0.0,
            "caller_exposure_factor": 20.0,
        },
        "slow_consumer": slow,
        "detokenizer": [{
            "num_tokens": 64,
            "median_us_per_token": 10.0,
            "max_feed_decode_tokens": 40,
            "feed_decode_tokens_per_input_token": 30.0,
            "flush_decode_tokens": 64,
            "full_length_decode_calls": 1,
        }],
    }


def test_eight_trials_use_distinct_seeds_and_prompt_sets():
    seeds = [benchmark._trial_seed(20260818, index) for index in range(8)]
    prompts = [benchmark._trial_prompts(16, index) for index in range(8)]

    assert len(set(seeds)) == 8
    assert all(len(items) == 16 for items in prompts)
    assert len({benchmark._canonical_sha256(items) for items in prompts}) == 8


def test_result_publication_is_atomic_and_never_overwrites(tmp_path):
    output = tmp_path / "result.json"
    benchmark._write_json_exclusive(output, {"attempt": 1})

    with pytest.raises(FileExistsError, match="immutable result"):
        benchmark._write_json_exclusive(output, {"attempt": 2})

    assert json.loads(output.read_text()) == {"attempt": 1}
    assert list(tmp_path.iterdir()) == [output]


def test_correction_heavy_8k_32k_apply_and_state_gates_are_cpu_only():
    result = benchmark._measure_correction_scaling(
        StreamingDetokenizer,
        [8000, 32000],
        repeats=2,
    )

    assert result["all_gates_pass"]
    assert all(result["gates"].values())
    assert [item["correction_updates"] for item in result["results"]] == [
        4000,
        16000,
    ]
    assert [item["state_token_count_before_flush"] for item in result["results"]] == [
        8000,
        32000,
    ]


def test_release_parser_predeclares_eight_pairs_and_has_no_overwrite_escape_hatch():
    parser = benchmark._parser()
    args = parser.parse_args(["--output", "/tmp/unused-streaming-cert.json"])

    assert args.runs == 8
    assert args.core_rounds == 4
    assert args.correction_lengths == [8000, 32000]
    assert "--overwrite" not in parser.format_help()


def test_cpu_synthetic_aggregate_exercises_all_gpu_release_gate_shapes():
    aggregate = benchmark._aggregate([_fake_worker(index) for index in range(8)])

    assert aggregate["matched_final_decode_consumer"][
        "paired_mean_delta_percent_90ci"
    ] == {"mean": 0.0, "low": 0.0, "high": 0.0}
    assert all(aggregate["gates"].values())
    assert aggregate["backpressure_roofline"]["4"]["residual_ms"] == 0.0


def test_peak_memory_gate_is_paired_relative_and_zero_safe():
    workers = [_fake_worker(index) for index in range(8)]
    first_round = workers[0]["core"]["timed_rounds"][0]
    first_round["generate"]["memory"]["peak"]["allocated_bytes"] = 100
    first_round["stream"]["memory"]["peak"]["allocated_bytes"] = 102

    assert not benchmark._aggregate(workers)["gates"][
        "stream_peak_memory_within_1_percent_of_generate"
    ]
    assert benchmark._stream_peak_within_one_percent_of_generate(101, 100)
    assert not benchmark._stream_peak_within_one_percent_of_generate(102, 100)
    assert benchmark._stream_peak_within_one_percent_of_generate(0, 0)
    assert not benchmark._stream_peak_within_one_percent_of_generate(1, 0)


def test_core_prefix_reset_requires_idle_ownership_and_replaces_cache_metadata():
    class FakeBlock:
        ref_count = 0

    class FakeBlockManager:
        def __init__(self, num_blocks, block_size):
            self.block_size = block_size
            self.blocks = [FakeBlock() for _ in range(num_blocks)]
            self.free_block_ids = list(range(num_blocks))
            self.used_block_ids = set()
            self.hash_to_block_id = {}

    previous = FakeBlockManager(4, 256)
    previous.hash_to_block_id[123] = 2
    scheduler = SimpleNamespace(
        is_finished=lambda: True,
        block_manager=previous,
    )
    llm = SimpleNamespace(scheduler=scheduler, _active_session=None)

    record = benchmark._reset_core_prefix_cache(
        llm, FakeBlockManager, "generate", 0
    )

    assert scheduler.block_manager is not previous
    assert scheduler.block_manager.hash_to_block_id == {}
    assert record["policy"] == benchmark.CORE_PREFIX_CACHE_POLICY
    assert record["discarded_cached_block_hashes"] == 1

    scheduler.block_manager.used_block_ids.add(0)
    with pytest.raises(AssertionError, match="owns KV-cache blocks"):
        benchmark._reset_core_prefix_cache(llm, FakeBlockManager, "stream", 1)


def test_accepted_streaming_certificate_archive_recomputes_every_gate():
    results = Path(__file__).resolve().parents[1] / "benchmarks/pr5_results"
    summary = validate_archive(
        results / "repaired_streaming_cert_a100_2026-08-18_14002ae.json",
        results / "repaired_streaming_cert_a100_2026-08-18_14002ae.manifest.json",
    )

    assert ACCEPTED_COMMIT == "14002ae04102eef58aea09fa8a2a78eca0103b5f"
    assert ACCEPTED_ARTIFACT_SHA256 == (
        "df662215db769d0c93129b8d29fc9fbcda4998e41cc412d312864c437da7f8c7"
    )
    assert summary["gates"]["all_required_gates_pass"]
    assert summary["raw_timed_pairs"] == 32
    assert summary["event_count"] == 65_536


@pytest.mark.parametrize(
    ("document", "raw_sha256"),
    [
        ({"schema_version": 2}, None),
        ({
            "schema_version": 3,
            "provenance": {"repository": {"commit": SUPERSEDED_COMMIT}},
        }, None),
        ({
            "schema_version": 3,
            "provenance": {"repository": {"commit": ACCEPTED_COMMIT}},
        }, SUPERSEDED_ARTIFACT_SHA256),
    ],
)
def test_streaming_certificate_validator_rejects_superseded_cf6_evidence(
    document,
    raw_sha256,
):
    with pytest.raises(CertificateValidationError, match="superseded cf6da50"):
        validate_evidence(document, raw_sha256=raw_sha256)


def test_streaming_certificate_validator_rejects_unknown_artifact_hash():
    document = {
        "schema_version": 3,
        "provenance": {"repository": {"commit": ACCEPTED_COMMIT}},
    }

    with pytest.raises(CertificateValidationError, match="artifact SHA-256"):
        validate_evidence(document, raw_sha256="0" * 64)


def test_streaming_certificate_validator_rejects_flipped_recorded_gate():
    artifact = (
        Path(__file__).resolve().parents[1]
        / "benchmarks/pr5_results/repaired_streaming_cert_a100_2026-08-18_14002ae.json"
    )
    document = json.loads(artifact.read_text())
    document["aggregate"]["gates"]["event_delivery_p95_at_most_1ms"] = False

    with pytest.raises(CertificateValidationError, match="recorded accepted gates"):
        validate_evidence(document)
