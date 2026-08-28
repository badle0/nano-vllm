import importlib.util
from pathlib import Path

import pytest


RUNNER_PATH = Path(__file__).with_name(
    "run_speculative_v2_gpu_recovery.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_speculative_v2_gpu_recovery_for_tests",
    RUNNER_PATH,
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)
REPO_ROOT = Path(__file__).resolve().parents[1]


def test_runtime_import_origin_gate_binds_recovery_to_this_checkout():
    origins = RUNNER.validate_runtime_import_origins(REPO_ROOT)

    assert origins == {
        "lifecycle_contract": str(
            REPO_ROOT / "tests" / "run_speculative_v2_lifecycle.py"
        ),
        "nanovllm_package": str(
            REPO_ROOT / "nanovllm" / "__init__.py"
        ),
        "model_runner_class": str(
            REPO_ROOT / "nanovllm" / "engine" / "model_runner.py"
        ),
        "llm_class": str(REPO_ROOT / "nanovllm" / "llm.py"),
    }


def test_runtime_import_origin_gate_rejects_foreign_nanovllm(
    tmp_path,
    monkeypatch,
):
    foreign_package = tmp_path / "nanovllm" / "__init__.py"
    monkeypatch.setattr(
        RUNNER.CONTRACT.nanovllm,
        "__file__",
        str(foreign_package),
    )

    with pytest.raises(AssertionError, match="certified checkout"):
        RUNNER.validate_runtime_import_origins(REPO_ROOT)


def _gpu_snapshot(memory_by_pid):
    return {
        "compute_apps_returncode": 0,
        "compute_apps_stderr": "",
        "compute_apps": [
            {
                "pid": pid,
                "used_memory_mib": memory_mib,
            }
            for pid, memory_mib in memory_by_pid.items()
        ],
    }


def test_allocator_pid_challenge_rejects_wrong_candidate_delta():
    mib = 1024**2
    with pytest.raises(AssertionError, match="did not track"):
        RUNNER.CONTRACT.validate_gpu_pid_allocator_challenge(
            candidate_pid=300,
            baseline_snapshot=_gpu_snapshot({300: 416}),
            challenged_snapshot=_gpu_snapshot({300: 417}),
            allocated_delta_bytes=390 * mib,
            reserved_delta_bytes=390 * mib,
        )


def test_allocator_pid_challenge_rejects_foreign_memory_change():
    mib = 1024**2
    with pytest.raises(AssertionError, match="non-candidate"):
        RUNNER.CONTRACT.validate_gpu_pid_allocator_challenge(
            candidate_pid=300,
            baseline_snapshot=_gpu_snapshot({300: 416, 999: 100}),
            challenged_snapshot=_gpu_snapshot({300: 806, 999: 101}),
            allocated_delta_bytes=390 * mib,
            reserved_delta_bytes=390 * mib,
        )


def test_allocator_pid_challenge_cleans_up_before_rethrow(monkeypatch):
    contract = RUNNER.CONTRACT
    mib = 1024**2
    baseline = _gpu_snapshot({300: 416})
    snapshots = iter((
        _gpu_snapshot({300: 417}),
        baseline,
    ))
    allocated = iter((0, 390 * mib, 0))
    reserved = iter((0, 390 * mib, 0))
    cleanup = []

    monkeypatch.setattr(
        contract,
        "gpu_gate_snapshot",
        lambda **kwargs: next(snapshots),
    )
    monkeypatch.setattr(
        contract.torch,
        "empty",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        contract.torch.cuda,
        "memory_allocated",
        lambda: next(allocated),
    )
    monkeypatch.setattr(
        contract.torch.cuda,
        "memory_reserved",
        lambda: next(reserved),
    )
    monkeypatch.setattr(
        contract.torch.cuda,
        "synchronize",
        lambda: cleanup.append("synchronize"),
    )
    monkeypatch.setattr(
        contract.torch.cuda,
        "empty_cache",
        lambda: cleanup.append("empty_cache"),
    )
    monkeypatch.setattr(
        contract.gc,
        "collect",
        lambda: cleanup.append("gc"),
    )

    with pytest.raises(AssertionError, match="did not track"):
        contract.run_gpu_pid_allocator_challenge(300, baseline)

    assert cleanup[-3:] == ["gc", "empty_cache", "synchronize"]
    assert list(allocated) == []
    assert list(reserved) == []


def test_phase_parser_selects_graph_only_where_required():
    assert RUNNER.MAX_POST_EXECUTION_GLOBAL_ALLOCATED_BYTES == 32 * 1024**2
    assert RUNNER.MAX_POST_EXECUTION_GLOBAL_RESERVED_BYTES == 64 * 1024**2
    for phase in RUNNER.PHASE_METHODS:
        args = RUNNER.parse_args(["--phase", phase, "--model", "/model"])
        expected = "graph" if phase in RUNNER.GRAPH_PHASES else "eager"
        assert args.mode == expected
        assert args.configured_k == 2
        assert args.num_kvcache_blocks == 4


@pytest.mark.parametrize("phase", tuple(RUNNER.PHASE_METHODS))
def test_phase_injection_occurs_after_the_real_operation(phase):
    calls = []
    method_name = RUNNER.PHASE_METHODS[phase]

    class FakeRunner:
        pass

    def operation(self):
        calls.append(method_name)
        return "completed"

    setattr(FakeRunner, method_name, operation)
    restore, ledger = RUNNER.install_phase_injection(
        phase,
        runner_class=FakeRunner,
    )
    instance = FakeRunner()
    try:
        if phase == "draft_graph":
            assert getattr(instance, method_name)() == "completed"
        with pytest.raises(
            RUNNER.InjectedSpeculativePhaseError,
            match=f"phase: {phase}",
        ):
            getattr(instance, method_name)()
    finally:
        restore()

    expected_calls = 2 if phase == "draft_graph" else 1
    assert calls == [method_name] * expected_calls
    assert ledger == {"calls": expected_calls, "injections": 1}
    assert getattr(instance, method_name)() == "completed"
