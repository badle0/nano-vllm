from threading import Lock
from types import MethodType

import pytest
import torch

from nanovllm import LLM, SamplingParams
from nanovllm.engine.llm_engine import LLMEngine, StepOutput
from nanovllm.engine.sequence import SequenceStatus, StreamOutput


MODEL_PATH = "/workspace/models/Qwen3-0.6B"
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "In 1969, humans first",
]
SP = SamplingParams(temperature=0.6, max_tokens=32, ignore_eos=True)


class FakeClock:

    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class FakeTokenizer:

    def __init__(self, clock):
        self.clock = clock
        self.encoded = []

    def encode(self, prompt):
        self.encoded.append(prompt)
        self.clock.advance(0.25)
        if prompt == "bad":
            return []
        return [10 + len(self.encoded)]

    def decode(self, token_ids):
        return ",".join(str(token_id) for token_id in token_ids)


class FakeScheduler:

    def __init__(self):
        self.sequences = []
        self.cancel_calls = []

    def add(self, sequence):
        self.sequences.append(sequence)

    def is_finished(self):
        return not self.sequences

    def cancel(self, seq_ids):
        targets = set(seq_ids)
        self.cancel_calls.append(tuple(sorted(targets)))
        retained = []
        cancelled = []
        for sequence in self.sequences:
            if sequence.seq_id in targets:
                sequence.status = SequenceStatus.CANCELLED
                cancelled.append(sequence.seq_id)
            else:
                retained.append(sequence)
        self.sequences = retained
        return cancelled


def make_fake_engine():
    clock = FakeClock()
    engine = LLMEngine.__new__(LLMEngine)
    engine._clock = clock
    engine._session_lock = Lock()
    engine._active_session = None
    engine.tokenizer = FakeTokenizer(clock)
    engine.scheduler = FakeScheduler()

    def fake_step(self):
        clock.advance(1.0)
        now = clock()
        events = []
        finished = []
        for sequence in list(self.scheduler.sequences):
            if sequence.first_scheduled_time is None:
                sequence.first_scheduled_time = now
            token_id = 20 + sequence.num_completion_tokens
            sequence.append_token(token_id)
            if sequence.first_token_time is None:
                sequence.first_token_time = now
            sequence.token_times.append(now)
            is_finished = sequence.num_completion_tokens == sequence.max_tokens
            if is_finished:
                sequence.finish_time = now
                sequence.status = SequenceStatus.FINISHED
                self.scheduler.sequences.remove(sequence)
                finished.append(sequence)
            events.append(StreamOutput(sequence.seq_id, token_id, is_finished))
        return StepOutput(events, finished, -len(events))

    engine._step = MethodType(fake_step, engine)
    return engine, clock


def seed():
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)


def reassemble(events):
    per_sequence = {}
    for event in events:
        per_sequence.setdefault(event.seq_id, []).append(event.token_id)
    return [per_sequence[seq_id] for seq_id in sorted(per_sequence)]


def test_second_session_and_public_engine_drivers_are_rejected():
    engine, _ = make_fake_engine()
    session = engine.stream(["first"], SamplingParams(max_tokens=2))
    first_event = next(session)
    assert first_event.seq_id in session.seq_ids

    with pytest.raises(RuntimeError, match="active stream session"):
        engine.stream(["second"], SamplingParams(max_tokens=2))
    with pytest.raises(RuntimeError, match="active stream session"):
        engine.generate(["second"], SamplingParams(max_tokens=2), use_tqdm=False)
    with pytest.raises(RuntimeError, match="active stream session"):
        engine.add_request("second", SamplingParams(max_tokens=2))
    with pytest.raises(RuntimeError, match="active stream session"):
        engine.step()

    owned_ids = session.seq_ids
    session.close()
    assert engine.scheduler.cancel_calls[-1] == owned_ids
    assert engine.scheduler.is_finished()

    with engine.stream(["second"], SamplingParams(max_tokens=1)) as second:
        assert len(list(second)) == 1


def test_context_exit_cleans_up_a_retained_partial_stream():
    engine, _ = make_fake_engine()
    session = engine.stream(["first"], SamplingParams(max_tokens=4))
    with session:
        next(session)
    assert session.closed
    assert engine.scheduler.is_finished()
    assert engine._active_session is None
    session.close()


def test_stream_delivery_metrics_are_caller_visible_without_event_arity_change():
    engine, clock = make_fake_engine()
    session = engine.stream(["first"], SamplingParams(max_tokens=2))
    first = next(session)
    clock.advance(3.0)
    final = next(session)

    assert len(first) == len(final) == 3
    assert final.finished
    assert session.closed
    metrics = session.metrics[final.seq_id]
    assert metrics["caller_ttft"] == pytest.approx(1.25)
    assert metrics["first_token_to_delivery"] == pytest.approx(0.0)
    assert metrics["caller_e2e"] == pytest.approx(5.25)
    assert metrics["engine_finish_to_delivery"] == pytest.approx(0.0)


def test_stream_validates_lengths_before_admission():
    engine, _ = make_fake_engine()
    with pytest.raises(ValueError, match="same number"):
        engine.stream(
            ["first", "second"],
            [SamplingParams(max_tokens=1)],
        )
    assert engine.scheduler.sequences == []
    assert engine.tokenizer.encoded == []
    assert engine._active_session is None


@pytest.mark.parametrize("method", ["stream", "generate"])
def test_admission_failure_rolls_back_only_admitted_ids(method):
    engine, _ = make_fake_engine()
    params = SamplingParams(max_tokens=1)
    with pytest.raises(IndexError):
        if method == "stream":
            engine.stream([[1], []], params)
        else:
            engine.generate([[1], []], params, use_tqdm=False)

    assert len(engine.scheduler.cancel_calls) >= 1
    assert len(engine.scheduler.cancel_calls[0]) == 1
    assert engine.scheduler.is_finished()
    assert engine._active_session is None

    with engine.stream([[1]], params) as session:
        assert len(list(session)) == 1


def test_string_tokenization_failure_is_transactional():
    engine, _ = make_fake_engine()
    with pytest.raises(IndexError):
        engine.stream(["first", "bad"], SamplingParams(max_tokens=1))
    assert len(engine.scheduler.cancel_calls[0]) == 1
    assert engine.scheduler.is_finished()
    assert engine._active_session is None


def test_add_request_returns_seq_id_and_manual_step_keeps_pair_contract():
    engine, _ = make_fake_engine()
    seq_id = engine.add_request("first", SamplingParams(max_tokens=1))
    outputs, num_tokens = engine.step()
    assert outputs == [(seq_id, [20])]
    assert num_tokens == -1


def test_seeded_equivalence(llm):
    seed()
    batch = [
        output["token_ids"]
        for output in llm.generate(PROMPTS, SP, use_tqdm=False)
    ]
    seed()
    streamed = reassemble(llm.stream(PROMPTS, SP))
    assert streamed == batch


def test_finished_flags(llm):
    seed()
    events = list(llm.stream(PROMPTS, SP))
    finished = [event for event in events if event.finished]
    assert len(finished) == len(PROMPTS)
    last_by_sequence = {event.seq_id: event for event in events}
    assert all(
        last_by_sequence[event.seq_id] == event for event in finished
    )


def test_context_break_does_not_leak_gpu_blocks(llm):
    block_manager = llm.scheduler.block_manager
    baseline = len(block_manager.free_block_ids)
    session = llm.stream(PROMPTS, SP)
    with session:
        for _ in range(5):
            next(session)
    assert session.closed
    assert len(block_manager.free_block_ids) == baseline
    assert llm.scheduler.is_finished()

    seed()
    output = llm.generate(PROMPTS, SP, use_tqdm=False)
    assert all(len(item["token_ids"]) == SP.max_tokens for item in output)


def test_max_tokens_one(llm):
    seed()
    events = list(llm.stream(
        PROMPTS,
        SamplingParams(temperature=0.6, max_tokens=1, ignore_eos=True),
    ))
    assert len(events) == len(PROMPTS)
    assert all(event.finished for event in events)


def test_chunked_prefill_emission(llm, monkeypatch):
    monkeypatch.setattr(llm.scheduler, "max_num_batched_tokens", 64)
    long_prompt = [11] * 200
    params = SamplingParams(temperature=0.6, max_tokens=8, ignore_eos=True)
    events = list(llm.stream([long_prompt], params))
    assert len(events) == params.max_tokens
    assert events[-1].finished
    assert not any(event.finished for event in events[:-1])


def test_prefix_cache_stream_generate_equivalence(llm):
    prompt = [11] * 300
    params = SamplingParams(temperature=0.6, max_tokens=8, ignore_eos=True)
    seed()
    llm.generate([prompt], params, use_tqdm=False)
    seed()
    streamed = reassemble(llm.stream([prompt], params))[0]
    seed()
    generated = llm.generate([prompt], params, use_tqdm=False)[0]["token_ids"]
    assert streamed == generated
