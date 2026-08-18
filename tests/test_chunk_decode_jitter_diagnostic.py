import pytest

from benchmarks.chunked_prefill_tail import decode_jitter_diagnostic as diagnostic


def runtime_args(tmp_path, *extra):
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text("{}\n")
    return diagnostic.build_parser().parse_args([
        "--model", str(model),
        "--expected-commit", "a" * 40,
        "--expected-source-sha256", "b" * 64,
        "--output", str(tmp_path / "result.json"),
        *extra,
    ])


def test_defaults_pin_bs16_bs18_and_a_thousand_steps(tmp_path):
    args = runtime_args(tmp_path)
    diagnostic._validate_args(args)
    spec = diagnostic.workload_spec(args)
    assert spec["batch_sizes"] == [16, 18]
    assert spec["steps_per_batch"] == 1000
    assert spec["max_num_seqs"] == 32
    assert spec["max_completion_tokens"] == 2016
    assert spec["disable_python_gc"] is True
    assert spec["tensor_parallel_size"] == 1
    assert spec["top_p"] == 1.0
    assert spec["top_k"] == -1


def test_validation_rejects_short_soaks_and_insufficient_context(tmp_path):
    with pytest.raises(ValueError, match="at least 1000"):
        diagnostic._validate_args(
            runtime_args(tmp_path, "--steps-per-batch", "999")
        )
    with pytest.raises(ValueError, match="exceeds --max-model-len"):
        diagnostic._validate_args(
            runtime_args(tmp_path, "--max-model-len", "2048")
        )


def test_host_delta_splits_wall_cpu_and_context_switches():
    before = diagnostic.HostSnapshot(1_000_000, 400_000, 7, 3)
    after = diagnostic.HostSnapshot(6_000_000, 1_400_000, 9, 4)
    assert diagnostic.host_delta(before, after) == {
        "wall_ms": 5.0,
        "thread_cpu_ms": 1.0,
        "non_thread_cpu_wall_ms": 4.0,
        "voluntary_context_switches": 2,
        "involuntary_context_switches": 1,
    }
    with pytest.raises(RuntimeError, match="non-monotonic"):
        diagnostic.host_delta(after, before)


def fake_row(step, wall, cuda, voluntary=0, involuntary=0):
    timings = {
        "api_wall_ms": wall,
        "api_thread_cpu_ms": 1.0,
        "api_non_thread_cpu_wall_ms": wall - 1.0,
        "api_residual_wall_ms": 0.2,
        "scheduler_wall_ms": 0.1,
        "prepare_wall_ms": 0.2,
        "model_wall_ms": 0.3,
        "sampler_wall_ms": 0.4,
        "postprocess_wall_ms": 0.5,
        "runner_cuda_ms": cuda,
        "prepare_cuda_ms": 0.2,
        "model_cuda_ms": cuda - 0.7,
        "sampler_cuda_ms": 0.3,
        "runner_cuda_residual_ms": 0.2,
    }
    return {
        "step": step,
        "timings_ms": timings,
        "context_switches": {
            "api": {"voluntary": voluntary, "involuntary": involuntary}
        },
    }


def test_summary_retains_raw_spike_decomposition():
    rows = [
        fake_row(0, 4.0, 3.0),
        fake_row(1, 13.0, 3.1, involuntary=1),
        fake_row(2, 5.0, 4.0, voluntary=2),
    ]
    summary = diagnostic.summarize_profile(rows, 10.0)
    assert summary["step_count"] == 3
    assert summary["spike_count"] == 1
    assert summary["spikes"] == [{
        "step": 1,
        "api_wall_ms": 13.0,
        "api_thread_cpu_ms": 1.0,
        "runner_cuda_ms": 3.1,
        "api_residual_wall_ms": 0.2,
        "context_switches": {"voluntary": 0, "involuntary": 1},
    }]
    assert summary["timings_ms"]["api_wall_ms"]["median"] == 5.0
    assert summary["timings_ms"]["api_wall_ms"]["p95"] == 13.0
    assert summary["context_switch_totals"] == {
        "voluntary": 2,
        "involuntary": 1,
    }


class FakeEvent:
    def __init__(self, stamp, complete=True):
        self.stamp = stamp
        self.complete = complete

    def query(self):
        return self.complete

    def elapsed_time(self, other):
        return other.stamp - self.stamp


def internal_record(complete=True):
    phases = {
        name: {
            "wall_ms": 0.1,
            "thread_cpu_ms": 0.05,
            "non_thread_cpu_wall_ms": 0.05,
            "voluntary_context_switches": 0,
            "involuntary_context_switches": 0,
        }
        for name in diagnostic.DecodeStepTracer.PHASES
    }
    stamps = {
        "runner_start": 0.0,
        "prepare_start": 0.0,
        "prepare_end": 0.2,
        "model_start": 0.3,
        "model_end": 3.3,
        "sampler_start": 3.5,
        "sampler_end": 3.8,
        "runner_end": 4.0,
    }
    return {
        "step": 0,
        "batch_size": 18,
        "_api_enter": diagnostic.HostSnapshot(0, 0, 0, 0),
        "_api_exit": diagnostic.HostSnapshot(5_000_000, 1_000_000, 1, 0),
        "_outputs_count": 0,
        "signed_tokens": -18,
        "caller_gap_before": None,
        "_phases": phases,
        "_cuda_events": {
            name: FakeEvent(stamp, complete=(complete or name != "runner_end"))
            for name, stamp in stamps.items()
        },
        "route": {
            "is_ragged": False,
            "scheduled_prefill_tokens": 0,
            "scheduled_decode_tokens": 18,
            "scheduled_seq_ids": list(range(18)),
            "selected_decode_graph_batch_size": 32,
        },
    }


def test_cuda_events_are_rejected_until_existing_sync_completed():
    tracer = object.__new__(diagnostic.DecodeStepTracer)
    resolved = tracer._resolve_record(internal_record())
    assert resolved["timings_ms"]["runner_cuda_ms"] == 4.0
    assert resolved["timings_ms"]["model_cuda_ms"] == 3.0
    assert resolved["cuda_events_queried_after_api_return"] is True
    with pytest.raises(RuntimeError, match="was not complete"):
        tracer._resolve_record(internal_record(complete=False))


def test_diagnostic_is_explicitly_non_certifying_and_main_cleans_up(monkeypatch):
    assert diagnostic.KIND == "chunked_prefill_decode_jitter_diagnostic"
    assert diagnostic.PROTOCOL.endswith("_v1")

    class FakeEngine:
        exits = 0

        def exit(self):
            self.exits += 1

    engine = FakeEngine()

    def fail(argv, register_engine):
        register_engine(engine)
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(diagnostic, "_main_impl", fail)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        diagnostic.main([])
    assert engine.exits == 1


def test_source_identity_includes_decode_jitter_harness():
    from benchmarks.chunked_prefill_tail.common import source_identity

    paths = {entry["path"] for entry in source_identity()["files"]}
    assert "benchmarks/chunked_prefill_tail/decode_jitter_diagnostic.py" in paths
