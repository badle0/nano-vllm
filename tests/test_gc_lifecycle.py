import gc
import threading
from dataclasses import dataclass

import pytest

from benchmarks.chunked_prefill_tail.step_diagnostics import build_parser
from nanovllm.config import Config
from nanovllm.engine import llm_engine as engine_module


@dataclass
class FakeConfig:
    model: str
    tensor_parallel_size: int = 1
    top_p_backend: str = "exact"
    disable_python_gc: bool = False
    kvcache_block_size: int = 256
    eos: int = -1


class FakeRunner:
    instances = []
    fail_exit = False

    def __init__(self, config, rank, event):
        self.config = config
        self.rank = rank
        self.event = event
        self.calls = []
        type(self).instances.append(self)

    def call(self, method_name, *args):
        self.calls.append((method_name, args))
        if method_name == "exit" and type(self).fail_exit:
            raise RuntimeError("fake runner exit failed")


class FakeTokenizer:
    eos_token_id = 17


class FakeAutoTokenizer:
    @staticmethod
    def from_pretrained(model, use_fast):
        assert use_fast is True
        return FakeTokenizer()


class FakeScheduler:
    def __init__(self, config, clock):
        self.config = config
        self.clock = clock


@pytest.fixture(autouse=True)
def restore_gc_and_fakes():
    gc_was_enabled = gc.isenabled()
    assert engine_module._PYTHON_GC_LEASE_COUNT == 0
    assert engine_module._PYTHON_GC_PRE_FIRST_ENABLED is None
    FakeRunner.instances.clear()
    FakeRunner.fail_exit = False
    try:
        yield
    finally:
        FakeRunner.fail_exit = False
        assert engine_module._PYTHON_GC_LEASE_COUNT == 0
        assert engine_module._PYTHON_GC_PRE_FIRST_ENABLED is None
        if gc_was_enabled:
            gc.enable()
        else:
            gc.disable()


@pytest.fixture
def fake_engine(monkeypatch):
    monkeypatch.setattr(engine_module, "Config", FakeConfig)
    monkeypatch.setattr(engine_module, "ModelRunner", FakeRunner)
    monkeypatch.setattr(engine_module, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(engine_module, "Scheduler", FakeScheduler)
    return engine_module.LLMEngine


@pytest.mark.parametrize("invalid", [1, 0, None, "false"])
def test_disable_python_gc_requires_a_strict_bool(invalid):
    with pytest.raises(TypeError, match="disable_python_gc must be a bool"):
        Config("/path/is/not/consulted", disable_python_gc=invalid)


def test_gc_control_defaults_off_and_diagnostic_flag_is_explicit():
    assert Config.__dataclass_fields__["disable_python_gc"].default is False
    assert list(Config.__dataclass_fields__)[-1] == "disable_python_gc"
    assert build_parser().parse_args([]).disable_python_gc is False
    assert build_parser().parse_args(["--disable-python-gc"]).disable_python_gc is True


def test_gc_control_rejects_unverified_tensor_parallel_use():
    with pytest.raises(
        ValueError,
        match="disable_python_gc currently supports tensor_parallel_size=1 only",
    ):
        Config(
            "/path/is/not/consulted",
            tensor_parallel_size=2,
            disable_python_gc=True,
        )


def test_default_does_not_change_gc_and_exit_is_idempotent(fake_engine):
    gc.enable()
    engine = fake_engine("fake-model")
    runner = engine.model_runner
    assert gc.isenabled()
    assert engine._python_gc_was_enabled is None

    engine.exit()
    engine.exit()

    assert gc.isenabled()
    assert runner.calls == [("exit", ())]
    assert engine.model_runner is None


def test_opt_in_disables_only_after_success_and_restores_enabled_state(fake_engine):
    gc.enable()
    engine = fake_engine("fake-model", disable_python_gc=True)
    assert not gc.isenabled()
    assert engine._python_gc_was_enabled is True

    engine.exit()

    assert gc.isenabled()
    assert engine._python_gc_was_enabled is None


def test_opt_in_restores_prior_disabled_state(fake_engine):
    gc.disable()
    engine = fake_engine("fake-model", disable_python_gc=True)
    assert not gc.isenabled()
    assert engine._python_gc_was_enabled is False

    # Even if another caller changes the process-global state while the engine
    # is live, exit restores the exact state observed at successful init.
    gc.enable()
    engine.exit()

    assert not gc.isenabled()
    assert engine._python_gc_was_enabled is None


@pytest.mark.parametrize("first_exit", [0, 1])
def test_overlapping_opt_in_engines_hold_a_reference_counted_lease(
    fake_engine,
    first_exit,
):
    gc.enable()
    engines = [
        fake_engine("fake-model-a", disable_python_gc=True),
        fake_engine("fake-model-b", disable_python_gc=True),
    ]
    assert not gc.isenabled()
    assert engine_module._PYTHON_GC_LEASE_COUNT == 2
    assert all(engine._python_gc_was_enabled is True for engine in engines)

    engines[first_exit].exit()
    assert not gc.isenabled()
    assert engine_module._PYTHON_GC_LEASE_COUNT == 1

    engines[1 - first_exit].exit()
    assert gc.isenabled()
    assert engine_module._PYTHON_GC_LEASE_COUNT == 0
    assert engine_module._PYTHON_GC_PRE_FIRST_ENABLED is None


def test_failed_initialization_never_disables_gc(fake_engine, monkeypatch):
    class FailingScheduler:
        def __init__(self, config, clock):
            raise RuntimeError("fake scheduler init failed")

    monkeypatch.setattr(engine_module, "Scheduler", FailingScheduler)
    gc.enable()

    with pytest.raises(RuntimeError, match="fake scheduler init failed"):
        fake_engine("fake-model", disable_python_gc=True)

    assert gc.isenabled()


def test_failed_atexit_registration_never_disables_gc(fake_engine, monkeypatch):
    def fail_registration(callback):
        raise RuntimeError("fake atexit registration failed")

    monkeypatch.setattr(engine_module.atexit, "register", fail_registration)
    gc.enable()

    with pytest.raises(RuntimeError, match="fake atexit registration failed"):
        fake_engine("fake-model", disable_python_gc=True)

    assert gc.isenabled()
    assert FakeRunner.instances[-1].calls == [("exit", ())]


def test_failed_gc_lease_acquisition_rolls_back_state(fake_engine, monkeypatch):
    callbacks = []
    real_disable = gc.disable

    def fail_after_disabling():
        real_disable()
        raise RuntimeError("fake gc disable failed")

    monkeypatch.setattr(engine_module.atexit, "register", callbacks.append)
    monkeypatch.setattr(engine_module.gc, "disable", fail_after_disabling)
    gc.enable()

    with pytest.raises(RuntimeError, match="fake gc disable failed"):
        fake_engine("fake-model", disable_python_gc=True)

    assert gc.isenabled()
    assert engine_module._PYTHON_GC_LEASE_COUNT == 0
    assert engine_module._PYTHON_GC_PRE_FIRST_ENABLED is None
    assert len(callbacks) == 1
    assert FakeRunner.instances[-1].calls == [("exit", ())]
    callbacks[0]()


def test_constructor_failure_after_acquire_releases_lease(fake_engine, monkeypatch):
    callbacks = []

    class FailingLeaseHandoffEngine(fake_engine):
        def __setattr__(self, name, value):
            if name == "_python_gc_lease_active" and value is True:
                raise RuntimeError("fake post-acquire handoff failed")
            super().__setattr__(name, value)

    monkeypatch.setattr(engine_module.atexit, "register", callbacks.append)
    gc.enable()

    with pytest.raises(RuntimeError, match="fake post-acquire handoff failed"):
        FailingLeaseHandoffEngine("fake-model", disable_python_gc=True)

    assert gc.isenabled()
    assert engine_module._PYTHON_GC_LEASE_COUNT == 0
    assert engine_module._PYTHON_GC_PRE_FIRST_ENABLED is None
    assert len(callbacks) == 1
    assert FakeRunner.instances[-1].calls == [("exit", ())]
    callbacks[0]()


def test_exit_failure_still_restores_gc_and_cleanup_is_claimed_once(fake_engine):
    class FakeProcess:
        def __init__(self):
            self.joins = 0

        def join(self):
            self.joins += 1

    gc.enable()
    engine = fake_engine("fake-model", disable_python_gc=True)
    process = FakeProcess()
    engine.ps.append(process)
    runner = engine.model_runner
    FakeRunner.fail_exit = True

    with pytest.raises(RuntimeError, match="fake runner exit failed"):
        engine.exit()

    assert gc.isenabled()
    assert process.joins == 1
    assert runner.calls == [("exit", ())]
    assert engine.model_runner is None

    # A registered atexit callback or explicit retry must be a no-op.
    engine.exit()
    assert process.joins == 1
    assert runner.calls == [("exit", ())]


def test_concurrent_exit_waits_for_the_cleanup_owner(fake_engine):
    gc.enable()
    engine = fake_engine("fake-model", disable_python_gc=True)
    runner = engine.model_runner
    original_call = runner.call
    runner_entered = threading.Event()
    release_runner = threading.Event()
    second_finished = threading.Event()

    def blocking_call(method_name, *args):
        runner_entered.set()
        assert release_runner.wait(timeout=2.0)
        return original_call(method_name, *args)

    runner.call = blocking_call
    owner = threading.Thread(target=engine.exit)
    waiter = threading.Thread(
        target=lambda: (engine.exit(), second_finished.set())
    )
    owner.start()
    assert runner_entered.wait(timeout=2.0)
    waiter.start()
    assert not second_finished.wait(timeout=0.05)
    release_runner.set()
    owner.join(timeout=2.0)
    waiter.join(timeout=2.0)

    assert not owner.is_alive()
    assert not waiter.is_alive()
    assert second_finished.is_set()
    assert gc.isenabled()


def test_exit_attempts_every_process_join_after_one_fails(fake_engine):
    class FakeProcess:
        def __init__(self, fail=False):
            self.fail = fail
            self.joins = 0

        def join(self):
            self.joins += 1
            if self.fail:
                raise RuntimeError("fake process join failed")

    gc.enable()
    engine = fake_engine("fake-model", disable_python_gc=True)
    processes = [FakeProcess(fail=True), FakeProcess()]
    engine.ps.extend(processes)

    with pytest.raises(RuntimeError, match="fake process join failed"):
        engine.exit()

    assert [process.joins for process in processes] == [1, 1]
    assert gc.isenabled()
    engine.exit()
