import pickle
from multiprocessing.reduction import ForkingPickler
from types import SimpleNamespace

import pytest
import torch

from nanovllm.engine.model_runner import (
    ModelRunner,
    TP_SHM_HEADER,
    TP_SHM_MAGIC,
    TensorParallelTransportError,
    tensor_parallel_shm_size,
)
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.tp_transport import ScheduledSequence, compact_run_args
from nanovllm.utils.context import get_context, reset_context


class FakeEvent:
    def __init__(self):
        self.set_calls = 0
        self.clear_calls = 0
        self.wait_calls = 0

    def set(self):
        self.set_calls += 1

    def clear(self):
        self.clear_calls += 1

    def wait(self):
        self.wait_calls += 1


class FakeSharedMemory:
    def __init__(self, size):
        self.buf = bytearray(size)


def make_sequence(total, cached, scheduled, *, block_base=0, is_prefill=True):
    seq = Sequence(list(range(total)))
    seq.num_cached_tokens = cached
    seq.num_scheduled_tokens = scheduled
    seq.is_prefill = is_prefill
    seq.block_table = list(
        range(block_base, block_base + (total + seq.block_size - 1) // seq.block_size)
    )
    return seq


def make_runner(*, rank, size, events):
    runner = object.__new__(ModelRunner)
    runner.world_size = 2
    runner.rank = rank
    runner.event = events
    runner.shm = FakeSharedMemory(size)
    return runner


def snapshot_context():
    context = get_context()
    return {
        "is_prefill": context.is_prefill,
        "cu_seqlens_q": context.cu_seqlens_q.clone()
        if context.cu_seqlens_q is not None
        else None,
        "cu_seqlens_k": context.cu_seqlens_k.clone()
        if context.cu_seqlens_k is not None
        else None,
        "max_seqlen_q": context.max_seqlen_q,
        "max_seqlen_k": context.max_seqlen_k,
        "slot_mapping": context.slot_mapping.clone()
        if context.slot_mapping is not None
        else None,
        "context_lens": context.context_lens.clone()
        if context.context_lens is not None
        else None,
        "block_tables": context.block_tables.clone()
        if context.block_tables is not None
        else None,
    }


def assert_contexts_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if isinstance(left[key], torch.Tensor):
            assert torch.equal(left[key], right[key]), key
        else:
            assert left[key] == right[key], key


def test_realistic_64_by_4096_compact_payload_fits_derived_transport():
    seqs = [
        make_sequence(4096, 3840, 256, block_base=row * 16)
        for row in range(64)
    ]
    worker_args = compact_run_args((seqs, True))
    payload = pickle.dumps(
        ["run", *worker_args], protocol=pickle.HIGHEST_PROTOCOL
    )
    config = SimpleNamespace(
        max_num_batched_tokens=64 * 256,
        max_num_seqs=64,
        max_model_len=4096,
        kvcache_block_size=256,
    )

    assert len(payload) <= tensor_parallel_shm_size(config) - TP_SHM_HEADER.size
    assert len(payload) < 2**20


def test_transport_sizing_rejects_configuration_above_explicit_bound():
    config = SimpleNamespace(
        max_num_batched_tokens=8 * 1024 * 1024,
        max_num_seqs=1,
        max_model_len=4096,
        kvcache_block_size=256,
    )

    with pytest.raises(ValueError, match="above.*safety limit"):
        tensor_parallel_shm_size(config)


def test_compact_payload_contains_only_scheduled_token_slice():
    seq = make_sequence(4096, 3840, 256)

    dto = ScheduledSequence.from_sequence(seq)

    assert dto.scheduled_token_ids == tuple(range(3840, 4096))
    assert len(dto.scheduled_token_ids) == seq.num_scheduled_tokens
    assert dto._fields == (
        "scheduled_token_ids",
        "is_prefill",
        "num_cached_tokens",
        "num_scheduled_tokens",
        "num_tokens",
        "last_token",
        "block_table",
    )
    restored = ForkingPickler.loads(bytes(ForkingPickler.dumps(dto)))
    assert restored == dto


def test_write_read_run_payload_uses_dto_without_mutating_rank_zero_sequence():
    event = FakeEvent()
    writer = make_runner(rank=0, size=64 * 1024, events=[event])
    seq = make_sequence(512, 256, 256)
    original_state = seq.__dict__.copy()

    writer.write_shm("run", [seq], True)

    assert event.set_calls == 1
    assert seq.__dict__ == original_state
    reader = make_runner(rank=1, size=len(writer.shm.buf), events=event)
    reader.shm = writer.shm
    method_name, args = reader.read_shm()
    remote_seqs, is_prefill = args
    assert method_name == "run"
    assert is_prefill is True
    assert remote_seqs == [ScheduledSequence.from_sequence(seq)]
    assert event.wait_calls == 1
    assert event.clear_calls == 1


def test_rank_zero_call_keeps_original_sequence_for_local_run():
    runner = object.__new__(ModelRunner)
    runner.world_size = 2
    runner.rank = 0
    seq = make_sequence(512, 256, 256)
    published = []
    locally_seen = []
    runner.write_shm = lambda method_name, *args: published.append(
        (method_name, args)
    )
    runner.run = lambda seqs, is_prefill: locally_seen.append(
        (seqs, is_prefill)
    ) or [123]

    result = runner.call("run", [seq], True)

    assert result == [123]
    assert published[0][1][0][0] is seq
    assert locally_seen == [([seq], True)]


def test_write_overflow_does_not_publish_or_signal_events():
    events = [FakeEvent(), FakeEvent()]
    runner = make_runner(rank=0, size=128, events=events)
    before = bytes(runner.shm.buf)
    seq = make_sequence(512, 0, 512)

    with pytest.raises(TensorParallelTransportError, match="exceeds"):
        runner.write_shm("run", [seq], True)

    assert bytes(runner.shm.buf) == before
    assert [event.set_calls for event in events] == [0, 0]


def test_read_rejects_invalid_length_header_and_clears_event():
    event = FakeEvent()
    runner = make_runner(rank=1, size=64, events=event)
    capacity = len(runner.shm.buf) - TP_SHM_HEADER.size
    runner.shm.buf[:TP_SHM_HEADER.size] = TP_SHM_HEADER.pack(
        TP_SHM_MAGIC, capacity + 1
    )

    with pytest.raises(TensorParallelTransportError, match="invalid.*length"):
        runner.read_shm()

    assert event.wait_calls == 1
    assert event.clear_calls == 1


def test_read_rejects_invalid_magic_header_and_clears_event():
    event = FakeEvent()
    runner = make_runner(rank=1, size=64, events=event)
    runner.shm.buf[:TP_SHM_HEADER.size] = TP_SHM_HEADER.pack(b"NOPE", 1)

    with pytest.raises(TensorParallelTransportError, match="invalid.*header"):
        runner.read_shm()

    assert event.wait_calls == 1
    assert event.clear_calls == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="preparation uses CUDA")
def test_scheduled_dto_prepare_ragged_parity_for_prefill_and_decode_rows():
    runner = object.__new__(ModelRunner)
    runner.block_size = 256
    prefill = make_sequence(512, 256, 256, block_base=10)
    decode = make_sequence(
        257, 256, 1, block_base=30, is_prefill=False
    )
    seqs = [prefill, decode]

    original_ids, original_positions = runner.prepare_ragged(seqs)
    original_context = snapshot_context()
    reset_context()
    remote_ids, remote_positions = runner.prepare_ragged(
        [ScheduledSequence.from_sequence(seq) for seq in seqs]
    )
    remote_context = snapshot_context()
    reset_context()

    assert torch.equal(original_ids, remote_ids)
    assert torch.equal(original_positions, remote_positions)
    assert_contexts_equal(original_context, remote_context)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="preparation uses CUDA")
def test_scheduled_dto_prepare_decode_parity():
    runner = object.__new__(ModelRunner)
    runner.block_size = 256
    seqs = [
        make_sequence(257, 256, 1, block_base=10, is_prefill=False),
        make_sequence(513, 512, 1, block_base=30, is_prefill=False),
    ]

    original_ids, original_positions = runner.prepare_decode(seqs)
    original_context = snapshot_context()
    reset_context()
    remote_ids, remote_positions = runner.prepare_decode(
        [ScheduledSequence.from_sequence(seq) for seq in seqs]
    )
    remote_context = snapshot_context()
    reset_context()

    assert torch.equal(original_ids, remote_ids)
    assert torch.equal(original_positions, remote_positions)
    assert_contexts_equal(original_context, remote_context)
