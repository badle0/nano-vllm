import json
import runpy
import stat
from pathlib import Path

import pytest

from benchmarks.chunked_prefill_tail import common
from benchmarks.chunked_prefill_tail import step_diagnostics
from benchmarks.chunked_prefill_tail.contract_matrix import MATRIX
from benchmarks.chunked_prefill_tail.release_policy import (
    chunk_tail_release_profile,
    evaluate_chunk_tail_release_profile,
)


def test_timing_summary_uses_nearest_rank_p95_and_true_median():
    rows = [
        {"wall_ms": wall, "cuda_ms": cuda}
        for wall, cuda in ((4, 40), (1, 10), (3, 30), (2, 20))
    ]
    assert common.timing_summary(rows) == {
        "count": 4,
        "wall_ms": {"median": 2.5, "p95": 4.0, "min": 1.0, "max": 4.0},
        "cuda_ms": {
            "median": 25.0,
            "p95": 40.0,
            "min": 10.0,
            "max": 40.0,
        },
    }
    assert common.timing_summary([]) is None


def test_immutable_json_is_read_only_and_refuses_overwrite(tmp_path):
    output = tmp_path / "retained" / "result.json"
    common.immutable_write_json(output, {"answer": 42})
    assert json.loads(output.read_text()) == {"answer": 42}
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    with pytest.raises(FileExistsError):
        common.immutable_write_json(output, {"answer": 43})


def test_source_identity_is_deterministic_and_covers_contract_entrypoints():
    first = common.source_identity()
    second = common.source_identity()
    assert first == second
    paths = {entry["path"] for entry in first["files"]}
    assert {
        "benchmarks/chunked_prefill_tail/common.py",
        "benchmarks/chunked_prefill_tail/step_diagnostics.py",
        "benchmarks/chunked_prefill_tail/contract_matrix.py",
        "benchmarks/chunked_prefill_tail/release_policy.py",
        "tests/run_varlen_graph_config.py",
        "tests/run_varlen_511_contract.py",
        "pyproject.toml",
    } <= paths
    assert len(first["aggregate_sha256"]) == 64


def test_boundary_prompt_and_matrix_contracts():
    namespace = runpy.run_path(
        str(common.ROOT / "tests/run_varlen_511_contract.py"),
        run_name="chunk_tail_boundary_contract",
    )
    prompts = namespace["boundary_prompts"](32000)
    assert len(prompts) == 4
    assert all(len(prompt) == 511 for prompt in prompts)
    assert len({tuple(prompt) for prompt in prompts}) == 4
    assert all(8 <= token < 32000 for prompt in prompts for token in prompt)
    assert namespace["MAX_MODEL_LEN"] == 512
    assert namespace["EXPECTED_GRAPH_KEY"] == (2048, 5)
    assert MATRIX == (
        (64, 512),
        (64, 1024),
        (64, 4096),
        (128, 512),
        (128, 1024),
        (128, 4096),
    )


def test_model_identity_is_content_addressed(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n")
    (model / "weights.bin").write_bytes(b"weights")
    first = common.model_identity(model)
    second = common.model_identity(model)
    assert first == second
    assert first["file_count"] == 2
    assert first["total_bytes"] == 10
    assert Path(first["resolved_path"]) == model.resolve()


def test_release_policy_retains_history_without_certifying_a_current_run():
    tau256 = chunk_tail_release_profile(256)
    tau512 = chunk_tail_release_profile(512)
    assert tau256["classification"] == "historical_reference"
    assert tau256["historical_release_role"] == "latency_slo_met"
    assert tau256["historical_latency_slo_met"] is True
    assert tau256["current_run_latency_certified"] is False
    assert tau256["applies_to_current_run"] is False
    assert tau256["runs_meeting_slo"] == 5
    assert tau256["worst_run_max_itl_ms"] < 10.0
    assert tau512["classification"] == "historical_reference"
    assert tau512["historical_release_role"] == "throughput_ttft_only"
    assert tau512["historical_latency_slo_met"] is False
    assert tau512["current_run_latency_certified"] is False
    assert tau512["applies_to_current_run"] is False
    assert tau512["runs_meeting_slo"] == 2
    assert tau512["median_run_max_itl_ms"] > 10.0
    assert tau512["worst_run_max_itl_ms"] > 10.0
    unknown = chunk_tail_release_profile(128)
    assert unknown["classification"] == "unverified"
    assert unknown["current_run_classification"] == "unverified"
    assert unknown["matches_historical_recorded_configuration"] is False
    assert unknown["applies_to_current_run"] is False
    assert "evidence_files" not in unknown
    exact_context = {
        "model_resolved_path": "/workspace/models/Qwen3-0.6B",
        "gpu": "NVIDIA A100-SXM4-40GB",
        "torch": "2.10.0+cu128",
        "cuda": "12.8",
        "python_gc_mode": "external_manual_disable_before_engine_construction",
        "measurement_protocol": "full_completion_request_metrics",
        "interactive_count": 16,
        "interactive_prompt_len": 64,
        "pre_long_steps": 40,
        "long_count": 2,
        "long_prompt_len": 2048,
        "max_tokens": 256,
        "temperature": 0.6,
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.8,
        "max_num_seqs": 256,
    }
    applicable = evaluate_chunk_tail_release_profile(256, exact_context)
    assert applicable["matches_historical_recorded_configuration"] is True
    assert applicable["configuration_match_scope"] == (
        "historical_recorded_fields_only"
    )
    assert applicable["applies_to_current_run"] is False
    assert applicable["current_run_classification"] == "not_certified_by_reference"
    assert applicable["applicability_mismatches"] == {}
    assert applicable["applicability_blockers"]
    bounded = evaluate_chunk_tail_release_profile(
        256,
        exact_context | {
            "measurement_protocol": "bounded_per_step_diagnostic",
            "python_gc_mode": "engine_option_after_successful_initialization",
        },
    )
    assert bounded["applies_to_current_run"] is False
    assert bounded["classification"] == "historical_reference"
    assert bounded["current_run_classification"] == "not_certified_by_reference"
    assert "measurement_protocol" in bounded["applicability_mismatches"]
    assert "python_gc_mode" in bounded["applicability_mismatches"]


def test_diagnostic_main_finally_exits_a_registered_engine(monkeypatch):
    class FakeEngine:
        exits = 0

        def exit(self):
            self.exits += 1

    engine = FakeEngine()

    def fail_after_registration(argv, register_engine):
        register_engine(engine)
        raise RuntimeError("fake diagnostic failure")

    monkeypatch.setattr(step_diagnostics, "_main_impl", fail_after_registration)
    with pytest.raises(RuntimeError, match="fake diagnostic failure"):
        step_diagnostics.main([])
    assert engine.exits == 1
