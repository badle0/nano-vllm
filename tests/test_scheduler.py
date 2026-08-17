from collections import deque
import subprocess
import sys
from types import SimpleNamespace

import pytest

from nanovllm import SamplingParams, SchedulerCapacityError
from nanovllm.config import Config
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus


def make_scheduler(*, max_num_seqs=2, max_num_batched_tokens=2):
    config = SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        eos=-1,
        kvcache_block_size=256,
        num_kvcache_blocks=32,
    )
    return Scheduler(config)


def make_sequence(token_count=1):
    return Sequence(
        list(range(1, token_count + 1)),
        SamplingParams(max_tokens=8, ignore_eos=True),
    )


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"max_num_batched_tokens": True}, TypeError),
        ({"max_num_batched_tokens": 0}, ValueError),
        ({"max_num_seqs": False}, TypeError),
        ({"max_num_seqs": 0}, ValueError),
        (
            {"max_num_batched_tokens": 1, "max_num_seqs": 2},
            ValueError,
        ),
    ],
)
def test_config_rejects_invalid_scheduler_limits_before_model_loading(kwargs, error):
    with pytest.raises(error):
        Config("/definitely/not/a/model", **kwargs)


def test_config_validation_survives_python_optimized_mode():
    script = """
from nanovllm.config import Config
try:
    Config("/definitely/not/a/model", max_num_batched_tokens=1, max_num_seqs=2)
except ValueError:
    raise SystemExit(0)
raise SystemExit(1)
"""
    result = subprocess.run(
        [sys.executable, "-O", "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_capacity_rejection_cancel_and_retry():
    scheduler = make_scheduler()
    first, second, retry = (make_sequence() for _ in range(3))
    scheduler.add(first)
    scheduler.add(second)

    with pytest.raises(SchedulerCapacityError) as exc_info:
        scheduler.add(retry)
    assert exc_info.value.requested == 1
    assert exc_info.value.available == 0
    assert list(scheduler.waiting) == [first, second]

    assert scheduler.cancel([first.seq_id]) == [first.seq_id]
    assert first.status is SequenceStatus.CANCELLED
    scheduler.add(retry)
    assert list(scheduler.waiting) == [second, retry]


def test_request_is_rejected_when_all_sequence_slots_are_running():
    scheduler = make_scheduler()
    first, second, blocked = (make_sequence() for _ in range(3))
    scheduler.add(first)
    scheduler.add(second)
    seqs, is_prefill = scheduler.schedule()

    assert seqs == [first, second]
    assert is_prefill
    assert list(scheduler.running) == [first, second]
    assert not scheduler.waiting
    with pytest.raises(SchedulerCapacityError):
        scheduler.add(blocked)


def test_token_budget_is_constructor_coherent_and_read_only():
    scheduler = make_scheduler(max_num_batched_tokens=4)
    assert scheduler.max_num_batched_tokens == 4
    with pytest.raises(AttributeError):
        scheduler.max_num_batched_tokens = 3


def test_mid_chunk_pointer_tracks_partial_completion():
    scheduler = make_scheduler(max_num_batched_tokens=2)
    sequence = make_sequence(token_count=5)
    scheduler.add(sequence)

    seqs, is_prefill = scheduler.schedule()
    assert seqs == [sequence] and is_prefill
    assert sequence.num_scheduled_tokens == 2
    assert scheduler.mid_chunk_seq is sequence
    assert scheduler.waiting[0] is sequence
    scheduler.postprocess(seqs, [100])

    seqs, _ = scheduler.schedule()
    assert sequence.num_scheduled_tokens == 2
    assert scheduler.mid_chunk_seq is sequence
    scheduler.postprocess(seqs, [101])

    seqs, _ = scheduler.schedule()
    assert sequence.num_scheduled_tokens == 1
    assert scheduler.mid_chunk_seq is None
    assert sequence in scheduler.running
    scheduler.cancel([sequence.seq_id])


def test_cancelled_mid_chunk_releases_capacity_and_blocks_for_retry():
    scheduler = make_scheduler(max_num_seqs=1, max_num_batched_tokens=1)
    baseline = len(scheduler.block_manager.free_block_ids)
    partial = make_sequence(token_count=3)
    scheduler.add(partial)
    scheduler.schedule()
    assert scheduler.mid_chunk_seq is partial
    assert len(scheduler.block_manager.free_block_ids) < baseline

    scheduler.cancel([partial.seq_id])
    assert scheduler.mid_chunk_seq is None
    assert scheduler.available_capacity == 1
    assert len(scheduler.block_manager.free_block_ids) == baseline

    retry = make_sequence()
    scheduler.add(retry)
    assert scheduler.waiting[0] is retry


def test_preempted_victim_stays_behind_mid_chunk_and_waiter():
    scheduler = make_scheduler(max_num_seqs=3, max_num_batched_tokens=3)
    mid = make_sequence(token_count=5)
    newcomer = make_sequence()
    scheduler.add(mid)
    seqs, _ = scheduler.schedule()
    scheduler.postprocess(seqs, [100])
    scheduler.add(newcomer)

    victim = make_sequence(token_count=2)
    scheduler.block_manager.allocate(victim, 0)
    victim.status = SequenceStatus.RUNNING
    victim.is_prefill = False
    scheduler.preempt(victim)

    assert list(scheduler.waiting) == [mid, newcomer, victim]
    assert scheduler.mid_chunk_seq is mid


class NoScanDeque(deque):

    def __iter__(self):
        raise AssertionError("schedule scanned the waiting deque")

    def remove(self, value):
        raise AssertionError("schedule removed from the waiting deque")


def test_saturated_schedule_does_not_scan_waiting_backlog():
    scheduler = make_scheduler(max_num_seqs=2, max_num_batched_tokens=2)
    scheduler.waiting = NoScanDeque([make_sequence()])
    for _ in range(2):
        sequence = make_sequence(token_count=2)
        sequence.status = SequenceStatus.RUNNING
        sequence.is_prefill = False
        scheduler.running.append(sequence)
    scheduler.block_manager.can_append = lambda sequence: True
    scheduler.block_manager.may_append = lambda sequence: None

    seqs, is_prefill = scheduler.schedule()
    assert len(seqs) == 2
    assert not is_prefill


def test_nonpositive_remaining_never_schedules_waiting_work():
    scheduler = make_scheduler(max_num_seqs=4, max_num_batched_tokens=4)
    waiter = make_sequence(token_count=3)
    scheduler.add(waiter)
    for _ in range(3):
        sequence = make_sequence(token_count=2)
        scheduler.block_manager.allocate(sequence, 0)
        sequence.status = SequenceStatus.RUNNING
        sequence.is_prefill = False
        scheduler.running.append(sequence)

    scheduler._max_num_batched_tokens = 2
    seqs, is_prefill = scheduler.schedule()
    assert len(seqs) == 3
    assert not is_prefill
    assert waiter.num_scheduled_tokens == 0
    assert not waiter.block_table
    assert all(sequence.num_scheduled_tokens == 1 for sequence in seqs)
