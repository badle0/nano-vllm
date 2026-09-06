"""Burst delivery/lifetime contracts, complementary to real V5 commit tests."""
import gc
import weakref
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from nanovllm import SamplingParams
from nanovllm.engine.llm_engine import StepOutput
from nanovllm.engine.model_runner import ModelRunner, SpeculativeDraftExecutionError
from nanovllm.engine.sequence import Sequence, SequenceStatus, StreamOutput
from test_streaming import make_fake_engine


def burst_engine():
    engine, clock = make_fake_engine()
    def step():
        clock.advance(1.)
        events, finished = [], []
        rows = list(engine.scheduler.sequences)
        for seq in rows:
            if seq.first_scheduled_time is None:
                seq.first_scheduled_time = clock()
                seq.first_token_time = clock()
            count = min(3, seq.max_tokens - seq.num_completion_tokens)
            for _ in range(count):
                token = 20 + seq.num_completion_tokens
                seq.append_token(token)
                seq.token_times.append(clock())
                terminal = seq.num_completion_tokens == seq.max_tokens
                if terminal:
                    seq.status = SequenceStatus.FINISHED
                    seq.finish_time = clock()
                    finished.append(seq)
                    engine.scheduler.sequences.remove(seq)
                events.append(StreamOutput(seq.seq_id, token, terminal))
        return StepOutput(events, finished, 0, len(rows))
    engine._step = step
    return engine, clock


@pytest.mark.parametrize("consumed", [0, 1, 2, 3, 4, 6, 7, 8])
def test_close_at_every_burst_boundary(consumed):
    engine, clock = burst_engine()
    session = engine.stream(["hello"], SamplingParams(max_tokens=8))
    delivered = [next(session) for _ in range(consumed)]
    assert [e.token_id for e in delivered] == list(range(20, 20 + consumed))
    session.close()
    session.close()
    assert session.closed and not session._pending
    assert engine._active_session is None
    assert not engine.scheduler.sequences


@pytest.mark.parametrize("consumed", [0, 1, 3, 4, 7])
def test_abandonment_releases_empty_or_nonempty_pending_burst(consumed):
    engine, clock = burst_engine()
    session = engine.stream(["hello"], SamplingParams(max_tokens=8))
    for _ in range(consumed):
        next(session)
    reference = weakref.ref(session)
    with pytest.warns(RuntimeWarning, match="garbage-collected"):
        del session
        gc.collect()
    assert reference() is None and engine._active_session is None
    assert not engine.scheduler.sequences
    with engine.stream(["again"], SamplingParams(max_tokens=2)) as retry:
        assert len(list(retry)) == 2


def test_burst_compute_and_delivery_clocks_are_distinct():
    engine, clock = burst_engine()
    session = engine.stream(["hello"], SamplingParams(max_tokens=8))
    events = []
    for event in session:
        events.append(event)
        clock.advance(.5)
    metrics = session.metrics[events[0].seq_id]
    assert metrics["engine_itls"] == [0., 0., 2.5, 0., 0., 2.5, 0.]
    assert metrics["engine_finish_to_delivery"] == .5
    assert metrics["caller_e2e"] > metrics["submission_to_engine_finish"]
    assert sum(e.finished for e in events) == 1


def test_foreign_request_is_not_cancelled_when_burst_protocol_rejects_it():
    engine, _ = burst_engine()
    session = engine.stream(["owned"], SamplingParams(max_tokens=8))
    foreign = Sequence([99], SamplingParams(max_tokens=8))
    engine.scheduler.sequences.append(foreign)
    with pytest.raises(RuntimeError, match="another request session"):
        next(session)
    assert engine.scheduler.sequences == [foreign]
    assert engine._active_session is None


@pytest.mark.parametrize("name", ["recompile_limit", "cache_size_limit"])
@pytest.mark.parametrize("fail", [False, True])
def test_constructor_compile_budget_is_version_adaptive_and_scoped(monkeypatch, name, fail):
    settings = SimpleNamespace(**{name: 8})
    @contextmanager
    def patch(**kwargs):
        assert kwargs == {name: 32}
        setattr(settings, name, 32)
        try:
            yield
        finally:
            setattr(settings, name, 8)
    settings.patch = patch
    monkeypatch.setattr(torch._dynamo, "config", settings)
    def initialize(self, *args):
        assert getattr(settings, name) == 32
        if fail:
            raise RuntimeError("injected construction failure")
    monkeypatch.setattr(ModelRunner, "_initialize", initialize)
    if fail:
        with pytest.raises(RuntimeError, match="injected"):
            ModelRunner(SimpleNamespace(speculation_enabled=True), 0, [])
    else:
        ModelRunner(SimpleNamespace(speculation_enabled=True), 0, [])
    assert getattr(settings, name) == 8


def test_failed_verifier_does_not_retain_tensor_traceback(monkeypatch):
    import nanovllm.engine.speculative_execution as execution
    references = []
    def fail(*args):
        tensor = torch.ones(100)
        references.append(weakref.ref(tensor))
        error = RuntimeError("primary verifier error")
        error.__notes__ = ["secondary cleanup detail"]
        raise error
    monkeypatch.setattr(execution, "execute", fail)
    runner = object.__new__(ModelRunner)
    with pytest.raises(SpeculativeDraftExecutionError, match="primary verifier error") as caught:
        runner.run_speculative(None, [])
    gc.collect()
    assert references[0]() is None
    assert caught.value.__context__ is None
