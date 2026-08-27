"""Regression tests for public admission and abandoned-stream hardening."""

import gc
import warnings
import weakref
from types import SimpleNamespace

import pytest

import nanovllm.engine.llm_engine as engine_module
from nanovllm import SamplingParams
from nanovllm.engine.llm_engine import AdmissionLimits, StreamSession
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence

from test_streaming import make_fake_engine


def _assert_cleanup_detail(error, seen_warnings, text):
    notes = getattr(error, "__notes__", ())
    warning_messages = tuple(str(item.message) for item in seen_warnings)
    assert any(text in note for note in notes) or any(
        text in message for message in warning_messages
    )


@pytest.mark.parametrize("value", [0, -1, 1.5, "2", True, None])
def test_max_tokens_rejects_non_positive_and_non_int(value):
    with pytest.raises((TypeError, ValueError)):
        SamplingParams(max_tokens=value)


@pytest.mark.parametrize("value", ["yes", 0, 1, None])
def test_ignore_eos_rejects_non_bool(value):
    with pytest.raises(TypeError):
        SamplingParams(ignore_eos=value)


def test_finish_condition_survives_internally_corrupted_max_tokens():
    config = SimpleNamespace(
        max_num_seqs=4,
        max_num_batched_tokens=64,
        eos=99,
        kvcache_block_size=256,
        num_kvcache_blocks=16,
    )
    scheduler = Scheduler(config)
    sequence = Sequence([1, 2, 3], SamplingParams(max_tokens=5, ignore_eos=True))
    scheduler.add(sequence)
    sequence.max_tokens = 0
    seqs, _ = scheduler.schedule()
    scheduler.postprocess(seqs, [7] * len(seqs))
    assert scheduler.is_finished()


def _limited_engine(**limits):
    engine, _ = make_fake_engine()
    defaults = dict(
        max_model_len=32,
        vocab_size=1000,
        block_size=256,
        kvcache_blocks=4,
    )
    defaults.update(limits)
    engine._admission_limits = AdmissionLimits(**defaults)
    return engine


def test_explicit_and_tokenized_out_of_vocab_ids_are_rejected():
    engine = _limited_engine()
    with pytest.raises(ValueError, match="vocabulary"):
        engine.add_request([1000], SamplingParams(max_tokens=1))
    with pytest.raises(ValueError, match="non-negative"):
        engine.add_request([-1], SamplingParams(max_tokens=1))
    engine.tokenizer.encode = lambda prompt: [1000]
    with pytest.raises(ValueError, match="vocabulary"):
        engine.add_request("tokenized", SamplingParams(max_tokens=1))
    assert engine.scheduler.sequences == []


@pytest.mark.parametrize("prompt", [[1.5], [True]])
def test_non_integer_token_id_rejected(prompt):
    engine = _limited_engine()
    with pytest.raises(TypeError, match="integers"):
        engine.add_request(prompt, SamplingParams(max_tokens=1))


@pytest.mark.parametrize("prompt", [(1, 2), b"12", None, 3])
def test_prompt_outer_type_is_rejected(prompt):
    engine = _limited_engine()
    with pytest.raises(TypeError, match="string or a list"):
        engine.add_request(prompt, SamplingParams(max_tokens=1))


def test_model_length_uses_actual_processed_token_boundary():
    engine = _limited_engine(max_model_len=32)
    engine.add_request([1] * 32, SamplingParams(max_tokens=1))

    engine = _limited_engine(max_model_len=32)
    with pytest.raises(ValueError, match="reduce max_tokens"):
        engine.add_request([1] * 32, SamplingParams(max_tokens=2))

    engine = _limited_engine(max_model_len=32)
    engine.add_request([1] * 20, SamplingParams(max_tokens=13))

    engine = _limited_engine(max_model_len=32)
    with pytest.raises(ValueError, match="reduce max_tokens"):
        engine.add_request([1] * 20, SamplingParams(max_tokens=14))


def test_kv_pool_uses_actual_processed_token_boundary():
    engine = _limited_engine(
        max_model_len=4096, block_size=256, kvcache_blocks=1
    )
    engine.add_request([1] * 256, SamplingParams(max_tokens=1))

    engine = _limited_engine(
        max_model_len=4096, block_size=256, kvcache_blocks=1
    )
    with pytest.raises(ValueError, match="KV-cache blocks"):
        engine.add_request([1] * 256, SamplingParams(max_tokens=2))

    engine = _limited_engine(
        max_model_len=4096, block_size=256, kvcache_blocks=1
    )
    engine.add_request([1] * 255, SamplingParams(max_tokens=2))

    engine = _limited_engine(
        max_model_len=4096, block_size=256, kvcache_blocks=1
    )
    with pytest.raises(ValueError, match="KV-cache blocks"):
        engine.add_request([1] * 255, SamplingParams(max_tokens=3))


def test_independently_missing_limits_do_not_disable_known_checks():
    engine = _limited_engine(vocab_size=None)
    with pytest.raises(ValueError, match="max_model_len"):
        engine.add_request([1] * 33, SamplingParams(max_tokens=1))

    engine = _limited_engine(kvcache_blocks=None)
    with pytest.raises(ValueError, match="vocabulary"):
        engine.add_request([1000], SamplingParams(max_tokens=1))


@pytest.mark.parametrize("with_limits", [False, True])
@pytest.mark.parametrize("tokenized", [False, True])
def test_negative_ids_are_rejected_without_vocabulary_metadata(
    with_limits, tokenized
):
    engine, _ = make_fake_engine()
    if with_limits:
        engine._admission_limits = AdmissionLimits(
            max_model_len=32,
            vocab_size=None,
            block_size=256,
            kvcache_blocks=4,
        )
    if tokenized:
        engine.tokenizer.encode = lambda prompt: [-1]
        prompt = "negative"
    else:
        prompt = [-1]
    with pytest.raises(ValueError, match="non-negative"):
        engine.add_request(prompt, SamplingParams(max_tokens=1))


def test_batch_rollback_failure_does_not_mask_admission_error():
    class CleanupError(RuntimeError):
        pass

    engine = _limited_engine()
    original_cancel = engine.scheduler.cancel

    def fail_after_cancel(seq_ids):
        original_cancel(seq_ids)
        raise CleanupError("batch cancel failed")

    engine.scheduler.cancel = fail_after_cancel
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always", RuntimeWarning)
        with pytest.raises(TypeError, match="string or a list") as exc_info:
            engine.stream([[1], (2,)], SamplingParams(max_tokens=1))
    assert engine._active_session is None
    _assert_cleanup_detail(exc_info.value, seen, "batch cancel failed")


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("max_tokens", 0, ValueError),
        ("max_tokens", True, TypeError),
        ("ignore_eos", 1, TypeError),
    ],
)
def test_mutated_sampling_params_are_revalidated_at_admission(field, value, error):
    engine = _limited_engine()
    params = SamplingParams(max_tokens=2)
    setattr(params, field, value)
    with pytest.raises(error):
        engine.add_request([1], params)
    assert engine.scheduler.sequences == []


def test_sampling_params_subclass_cannot_bypass_admission_validation():
    class BypassParams(SamplingParams):
        def __init__(self):
            self.temperature = 1.0
            self.max_tokens = -5
            self.ignore_eos = False
            self.top_k = -1
            self.top_p = 1.0

        def validate(self):
            pass

    engine = _limited_engine()
    with pytest.raises(ValueError, match="max_tokens"):
        engine.add_request([1], BypassParams())
    assert engine.scheduler.sequences == []


def test_late_invalid_batch_member_rolls_back_earlier_admission():
    engine = _limited_engine()
    with pytest.raises(TypeError, match="string or a list"):
        engine.stream([[1], (2,)], SamplingParams(max_tokens=1))
    assert engine.scheduler.is_finished()
    assert engine._active_session is None


def test_engine_without_gpu_limits_still_admits_valid_prompt():
    engine, _ = make_fake_engine()
    engine.add_request([1, 2, 3], SamplingParams(max_tokens=1))
    assert len(engine.scheduler.sequences) == 1


def test_abandoned_cleanup_survives_warning_as_error():
    engine, _ = make_fake_engine()
    session = engine.stream([[1]], SamplingParams(max_tokens=10))
    ids = session.seq_ids
    ref = weakref.ref(session)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        del session
        gc.collect()
    assert ref() is None
    assert engine._active_session is None
    assert engine.scheduler.is_finished()
    assert engine.scheduler.cancel_calls == [ids]


def test_abandoned_finalizer_runs_exactly_once_and_not_at_exit():
    engine, _ = make_fake_engine()
    session = engine.stream([[1]], SamplingParams(max_tokens=10))
    finalizer = session._finalizer
    assert finalizer is not None and not finalizer.atexit
    ids = session.seq_ids
    with pytest.warns(RuntimeWarning, match="garbage-collected"):
        del session
        gc.collect()
    gc.collect()
    assert engine.scheduler.cancel_calls == [ids]
    assert not finalizer.alive


def test_explicit_early_close_is_idempotent_and_warning_free():
    engine, _ = make_fake_engine()
    session = engine.stream([[1]], SamplingParams(max_tokens=10))
    next(session)
    ids = session.seq_ids
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always", RuntimeWarning)
        session.close()
        session.close()
        del session
        gc.collect()
    assert [w for w in seen if issubclass(w.category, RuntimeWarning)] == []
    assert engine.scheduler.cancel_calls == [ids]
    assert engine._active_session is None


def test_finalizer_registration_failure_rolls_back(monkeypatch):
    engine, _ = make_fake_engine()

    def fail_finalize(*args, **kwargs):
        raise MemoryError("injected finalizer allocation failure")

    monkeypatch.setattr(engine_module.weakref, "finalize", fail_finalize)
    with pytest.raises(MemoryError, match="injected"):
        engine.stream([[1]], SamplingParams(max_tokens=2))
    assert engine.scheduler.is_finished()
    assert engine._active_session is None
    assert len(engine.scheduler.cancel_calls) == 1
    assert len(engine.scheduler.cancel_calls[0]) == 1


def test_registration_primary_survives_rollback_cancel_failure(monkeypatch):
    class CleanupError(RuntimeError):
        pass

    engine, _ = make_fake_engine()
    monkeypatch.setattr(
        engine_module.weakref,
        "finalize",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            MemoryError("primary registration failure")
        ),
    )
    engine.scheduler.cancel = lambda seq_ids: (_ for _ in ()).throw(
        CleanupError("rollback cancel failure")
    )

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always", RuntimeWarning)
        with pytest.raises(MemoryError, match="primary registration failure") as exc_info:
            engine.stream([[1]], SamplingParams(max_tokens=2))

    assert engine._active_session is None
    _assert_cleanup_detail(exc_info.value, seen, "rollback cancel failure")


def test_iteration_primary_survives_close_failure():
    class PrimaryError(RuntimeError):
        pass

    class CleanupError(RuntimeError):
        pass

    engine, _ = make_fake_engine()
    session = engine.stream([[1]], SamplingParams(max_tokens=2))
    engine._step = lambda: (_ for _ in ()).throw(PrimaryError("model failed"))
    engine.scheduler.cancel = lambda seq_ids: (_ for _ in ()).throw(
        CleanupError("cancel failed")
    )

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always", RuntimeWarning)
        with pytest.raises(PrimaryError, match="model failed") as exc_info:
            next(session)

    assert engine._active_session is None
    _assert_cleanup_detail(exc_info.value, seen, "cancel failed")


def test_context_body_primary_survives_close_failure():
    class PrimaryError(RuntimeError):
        pass

    class CleanupError(RuntimeError):
        pass

    engine, _ = make_fake_engine()
    session = engine.stream([[1]], SamplingParams(max_tokens=2))
    engine.scheduler.cancel = lambda seq_ids: (_ for _ in ()).throw(
        CleanupError("context cancel failed")
    )

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always", RuntimeWarning)
        with pytest.raises(PrimaryError, match="body failed") as exc_info:
            with session:
                raise PrimaryError("body failed")

    assert engine._active_session is None
    _assert_cleanup_detail(exc_info.value, seen, "context cancel failed")


def test_generate_primary_survives_cleanup_failure():
    class PrimaryError(RuntimeError):
        pass

    class CleanupError(RuntimeError):
        pass

    engine, _ = make_fake_engine()
    engine._step = lambda: (_ for _ in ()).throw(
        PrimaryError("generate model failed")
    )
    engine.scheduler.cancel = lambda seq_ids: (_ for _ in ()).throw(
        CleanupError("generate cancel failed")
    )

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always", RuntimeWarning)
        with pytest.raises(PrimaryError, match="generate model failed") as exc_info:
            engine.generate([[1]], SamplingParams(max_tokens=2), use_tqdm=False)

    assert engine._active_session is None
    _assert_cleanup_detail(exc_info.value, seen, "generate cancel failed")


def test_finalizer_cancel_failure_still_releases_lease():
    lease = object()

    class CancelError(RuntimeError):
        pass

    class DummyScheduler:
        def cancel(self, seq_ids):
            raise CancelError("injected")

    class DummyEngine:
        scheduler = DummyScheduler()
        released = False

        def _release_session(self, actual):
            assert actual is lease
            self.released = True

    engine = DummyEngine()
    with pytest.warns(RuntimeWarning, match="automatic cleanup"):
        with pytest.raises(CancelError, match="injected"):
            StreamSession._finalize_abandoned(engine, lease, (1,))
    assert engine.released


def test_empty_stream_is_closed_without_finalizer_or_warning():
    engine, _ = make_fake_engine()
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always", RuntimeWarning)
        session = engine.stream([], SamplingParams(max_tokens=1))
        assert session.closed
        assert session._finalizer is None
        del session
        gc.collect()
    assert seen == []
    assert engine._active_session is None
