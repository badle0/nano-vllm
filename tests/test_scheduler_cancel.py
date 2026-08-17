from collections import deque
from types import SimpleNamespace

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import SequenceStatus


class FakeBlockManager:

    def __init__(self):
        self.deallocated = []

    def deallocate(self, seq):
        self.deallocated.append(seq.seq_id)
        seq.block_table.clear()


def make_sequence(seq_id, status, blocks=()):
    return SimpleNamespace(
        seq_id=seq_id,
        status=status,
        block_table=list(blocks),
        num_scheduled_tokens=7,
    )


def test_cancel_is_scoped_and_preserves_queue_order():
    keep_waiting_a = make_sequence(1, SequenceStatus.WAITING)
    cancel_mid_prefill = make_sequence(2, SequenceStatus.WAITING, [10, 11])
    keep_waiting_b = make_sequence(3, SequenceStatus.WAITING)
    cancel_running = make_sequence(4, SequenceStatus.RUNNING, [13])
    keep_running = make_sequence(5, SequenceStatus.RUNNING, [14])

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.waiting = deque([
        cancel_mid_prefill,
        keep_waiting_a,
        keep_waiting_b,
    ])
    scheduler.running = deque([cancel_running, keep_running])
    scheduler.block_manager = FakeBlockManager()
    scheduler.mid_chunk_seq = cancel_mid_prefill

    cancelled = scheduler.cancel([2, 4, 4, 999])

    assert cancelled == [2, 4]
    assert list(scheduler.waiting) == [keep_waiting_a, keep_waiting_b]
    assert list(scheduler.running) == [keep_running]
    assert scheduler.block_manager.deallocated == [2, 4]
    assert cancel_mid_prefill.status is SequenceStatus.CANCELLED
    assert cancel_running.status is SequenceStatus.CANCELLED
    assert cancel_mid_prefill.num_scheduled_tokens == 0
    assert cancel_running.num_scheduled_tokens == 0
    assert keep_waiting_b.block_table == []
    assert keep_running.block_table == [14]
    assert scheduler.mid_chunk_seq is None


def test_cancel_unknown_and_empty_ids_are_noops():
    sequence = make_sequence(1, SequenceStatus.WAITING, [10])
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.waiting = deque([sequence])
    scheduler.running = deque()
    scheduler.block_manager = FakeBlockManager()
    scheduler.mid_chunk_seq = sequence

    assert scheduler.cancel([]) == []
    assert scheduler.cancel([999]) == []
    assert list(scheduler.waiting) == [sequence]
    assert scheduler.block_manager.deallocated == []
    assert scheduler.mid_chunk_seq is sequence
