from types import MethodType
from threading import Lock

import pytest

from nanovllm import SamplingParams
from nanovllm.engine.llm_engine import LLMEngine, StepOutput
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.metrics import compute_metrics

class StubSequence:

    def __init__(self):
        self.submission_time = 9.0
        self.engine_arrival_time = 10.0
        self.first_scheduled_time = 10.5
        self.first_token_time = 11.0
        self.finish_time = 12.0
        self.token_times = [11.0, 11.4, 12.0]
        self.num_prompt_tokens = 100
        self.num_completion_tokens = 3


class FakeClock:

    def __init__(self, value=0.0):
        self.value = value

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
        self.clock.advance(2.0)
        return [len(self.encoded)]

    def decode(self, token_ids):
        self.clock.advance(0.5)
        return "decoded:" + ",".join(str(token_id) for token_id in token_ids)


class FakeScheduler:

    def __init__(self):
        self.sequences = []
        self.finished = True
        self.max_num_seqs = 512

    @property
    def available_capacity(self):
        return self.max_num_seqs - len(self.sequences)

    def require_capacity(self, requested=1):
        if requested > self.available_capacity:
            raise RuntimeError("fake scheduler capacity exceeded")

    def add(self, seq):
        self.require_capacity()
        self.sequences.append(seq)
        self.finished = False

    def is_finished(self):
        return self.finished

    def cancel(self, seq_ids):
        targets = set(seq_ids)
        self.sequences = [
            seq for seq in self.sequences if seq.seq_id not in targets
        ]
        self.finished = not self.sequences


def make_finished_sequence():
    seq = Sequence(
        [10],
        SamplingParams(max_tokens=2),
        submission_time=1.0,
        engine_arrival_time=2.0,
    )
    seq.first_scheduled_time = 3.0
    seq.first_token_time = 4.0
    seq.token_times = [4.0, 6.0]
    seq.append_token(20)
    seq.append_token(21)
    seq.finish_time = 6.0
    seq.status = SequenceStatus.FINISHED
    return seq


def make_fake_engine(clock):
    engine = LLMEngine.__new__(LLMEngine)
    engine._clock = clock
    engine._session_lock = Lock()
    engine._active_session = None
    engine.tokenizer = FakeTokenizer(clock)
    engine.scheduler = FakeScheduler()
    return engine


def test_compute_metrics_uses_explicit_boundaries():
    metrics = compute_metrics(StubSequence(), delivery_time=12.75)

    assert metrics["engine_queue_time"] == pytest.approx(0.5)
    assert metrics["engine_ttft"] == pytest.approx(1.0)
    assert metrics["engine_e2e"] == pytest.approx(2.0)
    assert metrics["engine_mean_itl"] == pytest.approx(0.5)
    assert metrics["engine_max_itl"] == pytest.approx(0.6)
    assert metrics["engine_itls"] == pytest.approx([0.4, 0.6])
    assert metrics["submission_to_engine"] == pytest.approx(1.0)
    assert metrics["submission_to_first_token"] == pytest.approx(2.0)
    assert metrics["submission_to_engine_finish"] == pytest.approx(3.0)
    assert metrics["engine_finish_to_delivery"] == pytest.approx(0.75)
    assert metrics["caller_e2e"] == pytest.approx(3.75)
    assert metrics["num_prompt_tokens"] == 100
    assert metrics["num_completion_tokens"] == 3
    assert {"ttft", "queue_time", "e2e_latency", "itls"}.isdisjoint(metrics)


def test_single_token_sequence_has_zero_engine_itl():
    seq = StubSequence()
    seq.first_scheduled_time = 9.1
    seq.first_token_time = 9.2
    seq.finish_time = 9.2
    seq.token_times = [9.2]
    seq.num_completion_tokens = 1
    seq.engine_arrival_time = 9.0

    metrics = compute_metrics(seq)

    assert metrics["engine_itls"] == []
    assert metrics["engine_mean_itl"] == 0.0
    assert metrics["engine_max_itl"] == 0.0
    assert metrics["engine_ttft"] == metrics["engine_e2e"] == pytest.approx(0.2)
    assert "caller_e2e" not in metrics


def test_add_request_stamps_submission_before_tokenization():
    clock = FakeClock(10.0)
    engine = make_fake_engine(clock)

    engine.add_request("hello", SamplingParams())

    seq = engine.scheduler.sequences[0]
    assert seq.submission_time == 10.0
    assert seq.engine_arrival_time == 12.0


def test_step_preserves_pairs_and_step_with_metrics_is_opt_in():
    seq = make_finished_sequence()
    engine = LLMEngine.__new__(LLMEngine)
    engine._clock = lambda: 7.0
    engine._session_lock = Lock()
    engine._active_session = None
    engine._execute_step = lambda: ([seq], -1)

    outputs, num_tokens = engine.step()
    assert outputs == [(seq.seq_id, [20, 21])]
    assert num_tokens == -1
    assert len(outputs[0]) == 2

    outputs, num_tokens = engine.step_with_metrics()
    assert outputs[0][:2] == (seq.seq_id, [20, 21])
    assert len(outputs[0]) == 3
    assert outputs[0][2]["caller_e2e"] == pytest.approx(6.0)
    assert num_tokens == -1


def test_generate_uses_common_batch_submission_and_final_delivery_clock():
    clock = FakeClock()
    engine = make_fake_engine(clock)

    def execute_step(self):
        clock.advance(1.0)
        for seq in self.scheduler.sequences:
            seq.first_scheduled_time = clock()
        clock.advance(2.0)
        for seq in self.scheduler.sequences:
            seq.first_token_time = clock()
            seq.token_times.append(clock())
            seq.append_token(20)
        clock.advance(2.0)
        for seq in self.scheduler.sequences:
            seq.token_times.append(clock())
            seq.append_token(21)
            seq.finish_time = clock()
            seq.status = SequenceStatus.FINISHED
        self.scheduler.finished = True
        sequences = list(self.scheduler.sequences)
        return StepOutput([], sequences, 0, len(sequences))

    engine._step = MethodType(execute_step, engine)
    outputs = engine.generate(
        ["first", "second"],
        SamplingParams(max_tokens=2),
        use_tqdm=False,
    )

    assert [output["token_ids"] for output in outputs] == [[20, 21], [20, 21]]
    metrics = [output["metrics"] for output in outputs]
    assert [item["submission_to_engine"] for item in metrics] == [2.0, 4.0]
    assert [item["engine_ttft"] for item in metrics] == [5.0, 3.0]
    assert [item["caller_e2e"] for item in metrics] == [10.0, 10.0]
    assert [item["engine_finish_to_delivery"] for item in metrics] == [1.0, 1.0]


def test_generate_rejects_mismatched_parameter_list_before_admission():
    clock = FakeClock()
    engine = make_fake_engine(clock)

    with pytest.raises(ValueError, match="same number"):
        engine.generate(
            ["first", "second"],
            [SamplingParams()],
            use_tqdm=False,
        )

    assert engine.scheduler.sequences == []
    assert engine.tokenizer.encoded == []

def test_engine_metrics_invariants(llm):
    outputs = llm.generate(
        ["The capital of France is", "def fibonacci(n):", "In 1969"],
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=8),
        use_tqdm=False,
    )

    for output in outputs:
        metrics = output["metrics"]
        assert metrics["engine_queue_time"] >= 0
        assert metrics["engine_ttft"] >= metrics["engine_queue_time"]
        assert metrics["engine_e2e"] >= metrics["engine_ttft"]
        assert metrics["submission_to_engine"] >= 0
        assert metrics["submission_to_first_token"] >= metrics["engine_ttft"]
        assert metrics["caller_e2e"] >= metrics["submission_to_engine_finish"]
        assert metrics["num_completion_tokens"] == len(output["token_ids"]) == 8
        assert len(metrics["engine_itls"]) == 7
        assert all(gap >= 0 for gap in metrics["engine_itls"])

    submission_delays = [
        output["metrics"]["submission_to_engine"] for output in outputs
    ]
    assert submission_delays == sorted(submission_delays)
    assert len({output["metrics"]["caller_e2e"] for output in outputs}) == 1
