"""Host regressions for the chunked-prefill review (not GPU certification)."""

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.speculative_routes import draft_graph_batch_buckets
from nanovllm.sampling_params import SamplingParams


def _scheduler(*, budget, blocks, capacity=2, block_size=256):
    return Scheduler(SimpleNamespace(
        max_num_seqs=capacity,
        max_num_batched_tokens=budget,
        eos=-1,
        kvcache_block_size=block_size,
        num_kvcache_blocks=blocks,
    ))


def _assert_ownership(scheduler):
    manager = scheduler.block_manager
    live = [*scheduler.running, *scheduler.waiting]
    assert len({seq.seq_id for seq in live}) == len(live)
    owners = Counter(block for seq in live for block in seq.block_table)
    free = list(manager.free_block_ids)
    assert len(free) == len(set(free))
    assert set(free).isdisjoint(manager.used_block_ids)
    assert set(free) | manager.used_block_ids == set(range(len(manager.blocks)))
    assert manager.used_block_ids == set(owners)
    assert all(block.ref_count == owners[block.block_id] for block in manager.blocks)
    scheduler._check_mid_chunk_invariant()


@pytest.mark.parametrize("budget,expected_steps", [(64, 69), (128, 35), (256, 18)])
def test_incremental_prefill_avoids_interrupting_decode(
    budget, expected_steps, monkeypatch,
):
    scheduler = _scheduler(budget=budget, blocks=17)
    short = Sequence(list(range(1000, 1250)), SamplingParams(max_tokens=16, ignore_eos=True))
    long = Sequence(list(range(20000, 24096)), SamplingParams(max_tokens=1, ignore_eos=True))
    scheduler.add(short)
    scheduler.add(long)
    victims = []
    original_preempt = scheduler.preempt

    def record_preempt(seq):
        victims.append(seq.seq_id)
        original_preempt(seq)

    monkeypatch.setattr(scheduler, "preempt", record_preempt)
    emission_steps = {short.seq_id: [], long.seq_id: []}
    step = 0
    while not scheduler.is_finished():
        assert step < 200
        seqs, _ = scheduler.schedule()
        _assert_ownership(scheduler)
        events = scheduler.postprocess(seqs, [50000 + seq.seq_id for seq in seqs])
        for event in events:
            emission_steps[event.seq_id].append(step)
        _assert_ownership(scheduler)
        step += 1

    short_steps = emission_steps[short.seq_id]
    assert len(short_steps) == 16
    assert all(right - left == 1 for left, right in zip(short_steps, short_steps[1:]))
    assert len(emission_steps[long.seq_id]) == 1
    # Incremental physical allocation prevents the long prompt from holding
    # all 16 blocks before its chunks are scheduled.
    assert victims == []
    assert step == expected_steps
    assert short.num_completion_tokens == 16 and long.num_completion_tokens == 1
    assert len(scheduler.block_manager.free_block_ids) == 17


def test_incremental_kv_extension_is_atomic_on_capacity_failure(monkeypatch):
    from nanovllm.engine.block_manager import BlockManager

    monkeypatch.setattr(Sequence, "block_size", 4)
    manager = BlockManager(num_blocks=3, block_size=4)
    growing = Sequence(list(range(9)))
    assert manager.can_allocate(growing) == 0
    assert manager.allocate_incremental(growing, 0, through_tokens=4)
    assert len(growing.block_table) == 1

    competitor = Sequence(list(range(100, 108)))
    manager.allocate(competitor, num_cached_blocks=0)
    snapshot = (
        tuple(growing.block_table),
        tuple(manager.free_block_ids),
        frozenset(manager.used_block_ids),
        growing.num_cached_tokens,
    )
    assert manager.extend(growing, through_tokens=5) is False
    assert (
        tuple(growing.block_table),
        tuple(manager.free_block_ids),
        frozenset(manager.used_block_ids),
        growing.num_cached_tokens,
    ) == snapshot


def _allocation_state(manager, seq):
    return (
        tuple(seq.block_table),
        seq.num_cached_tokens,
        seq.num_draft_cached_tokens,
        tuple(manager.free_block_ids),
        frozenset(manager.used_block_ids),
        tuple(sorted(manager.hash_to_block_id.items())),
        tuple(
            (block.ref_count, block.hash, tuple(block.token_ids))
            for block in manager.blocks
        ),
    )


def test_incremental_initial_allocation_is_exception_atomic(monkeypatch):
    from nanovllm.engine.block_manager import BlockManager

    monkeypatch.setattr(Sequence, "block_size", 4)
    manager = BlockManager(num_blocks=5, block_size=4)
    seq = Sequence(list(range(9)))
    before = _allocation_state(manager, seq)
    original = manager._allocate_block
    calls = 0

    def fail_second_allocation():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected allocation failure")
        return original()

    monkeypatch.setattr(manager, "_allocate_block", fail_second_allocation)
    with pytest.raises(RuntimeError, match="injected allocation failure"):
        manager.allocate_incremental(seq, 0, through_tokens=9)

    assert _allocation_state(manager, seq) == before


def test_incremental_extension_is_exception_atomic(monkeypatch):
    from nanovllm.engine.block_manager import BlockManager

    monkeypatch.setattr(Sequence, "block_size", 4)
    manager = BlockManager(num_blocks=5, block_size=4)
    seq = Sequence(list(range(12)))
    assert manager.allocate_incremental(seq, 0, through_tokens=4)
    before = _allocation_state(manager, seq)
    original = manager._allocate_block
    calls = 0

    def fail_second_allocation():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected extension failure")
        return original()

    monkeypatch.setattr(manager, "_allocate_block", fail_second_allocation)


def test_scheduler_extends_physical_kv_only_across_chunk_boundaries(monkeypatch):
    monkeypatch.setattr(Sequence, "block_size", 4)
    scheduler = _scheduler(budget=3, blocks=4, capacity=1, block_size=4)
    seq = Sequence(list(range(10)), SamplingParams(max_tokens=1, ignore_eos=True))
    scheduler.add(seq)

    rows, _ = scheduler.schedule()
    assert rows == [seq]
    assert seq.num_scheduled_tokens == 3
    assert len(seq.block_table) == 1
    scheduler.postprocess(rows, [100])

    rows, _ = scheduler.schedule()
    assert seq.num_cached_tokens == 3
    assert seq.num_scheduled_tokens == 3
    assert len(seq.block_table) == 2
    scheduler.cancel_all()
    assert len(scheduler.block_manager.free_block_ids) == 4


def test_preemption_rechecks_capacity_and_preserves_waiting_fifo(monkeypatch):
    scheduler = _scheduler(budget=4, blocks=32, capacity=4)
    mid = Sequence([1] * 8)
    newcomer = Sequence([2])
    scheduler.add(mid)
    seqs, _ = scheduler.schedule()
    scheduler.postprocess(seqs, [100])
    scheduler.add(newcomer)
    decoders = [Sequence([3]), Sequence([4])]
    for seq in decoders:
        scheduler.block_manager.allocate(seq, 0)
        seq.status = SequenceStatus.RUNNING
        seq.is_prefill = False
        scheduler.running.append(seq)

    checks = iter([False, False, True])
    victims = []
    original_preempt = scheduler.preempt

    def record_preempt(seq):
        victims.append(seq)
        original_preempt(seq)

    # Do not assume a victim released sufficient capacity. This also exercises
    # the fallback order if a future allocator needs more than one free block.
    monkeypatch.setattr(scheduler.block_manager, "can_append", lambda seq: next(checks))
    monkeypatch.setattr(scheduler, "preempt", record_preempt)
    # Freeze subsequent FIFO admission so its order can be inspected separately.
    monkeypatch.setattr(scheduler.block_manager, "can_allocate", lambda seq: -1)
    rows, _ = scheduler.schedule()
    assert rows == decoders[:1]
    assert victims == [mid, decoders[1]]
    assert list(scheduler.waiting) == [newcomer, mid, decoders[1]]
    assert scheduler.mid_chunk_seq is None
    scheduler.cancel_all()
    _assert_ownership(scheduler)


def test_decode_bucket_policy_covers_every_legal_cap_and_batch():
    for cap in range(1, 513):
        buckets = draft_graph_batch_buckets(cap)
        assert buckets == tuple(sorted(set(buckets)))
        assert all(1 <= bucket <= cap for bucket in buckets)
        for batch in range(1, cap + 1):
            assert next((bucket for bucket in buckets if bucket >= batch), None) is not None


@pytest.mark.parametrize(
    "buckets,captured",
    [([1, 2, 4, 8, 16], [1, 2, 4, 8, 16]), ([17], []), ([], [])],
)
def test_decode_missing_capture_uses_eager(buckets, captured):
    runner = ModelRunner.__new__(ModelRunner)
    runner.enforce_eager = False
    runner.graph_bs = buckets
    runner.graphs = dict.fromkeys(captured, object())
    calls = []

    class FakeModel:
        def __call__(self, input_ids, positions):
            calls.append((input_ids, positions))
            return input_ids.float().unsqueeze(1)

        def compute_logits(self, hidden):
            return hidden + 100

    runner.model = FakeModel()
    ids = torch.arange(17)
    positions = torch.arange(17)
    # No persistent graph buffers/context are supplied: eager fallback must not
    # touch them, nor sample extra rows or alter the model's result.
    result = runner.run_model(ids, positions, False)
    assert len(calls) == 1
    assert torch.equal(result, ids.float().unsqueeze(1) + 100)


def test_randomized_pressure_preserves_prefix_tags_and_ownership(monkeypatch):
    """Exact logical KV tags are an independent host oracle, not attention math."""
    monkeypatch.setattr(Sequence, "block_size", 4)
    rng = random.Random(20260906)
    for trial in range(1000):
        capacity = rng.randint(1, 5)
        blocks = rng.randint(8, 24)
        scheduler = _scheduler(
            budget=rng.randint(capacity, 24), blocks=blocks,
            capacity=capacity, block_size=4,
        )
        requests = []
        for row in range(capacity):
            length = rng.randint(1, (blocks - 2) * 4)
            prompt = [1, 2, 3, 4] + [1000 + row * 100 + i for i in range(length)]
            seq = Sequence(prompt[:length], SamplingParams(
                max_tokens=rng.randint(1, 8), ignore_eos=True,
            ))
            scheduler.add(seq)
            requests.append(seq)
        tags = {}
        steps = 0
        while not scheduler.is_finished():
            assert steps < 10000, (trial, "scheduler failed to make progress")
            rows, _ = scheduler.schedule()
            assert sum(seq.num_scheduled_tokens for seq in rows) <= scheduler.max_num_batched_tokens
            _assert_ownership(scheduler)
            for seq in rows:
                for index in range(seq.num_cached_tokens):
                    slot = (seq.block_table[index // 4], index % 4)
                    assert tags[slot] == tuple(seq.token_ids[:index + 1]), (trial, seq.seq_id, index)
                end = seq.num_cached_tokens + seq.num_scheduled_tokens
                for index in range(seq.num_cached_tokens, end):
                    slot = (seq.block_table[index // 4], index % 4)
                    tags[slot] = tuple(seq.token_ids[:index + 1])
            scheduler.postprocess(rows, [50000 + seq.seq_id for seq in rows])
            if not scheduler.is_finished() and rng.random() < 0.02:
                victim = rng.choice([*scheduler.running, *scheduler.waiting])
                scheduler.cancel([victim.seq_id])
            _assert_ownership(scheduler)
            steps += 1
        assert len(scheduler.block_manager.free_block_ids) == blocks
        for seq in requests:
            assert seq.status in (SequenceStatus.FINISHED, SequenceStatus.CANCELLED)
            if seq.status is SequenceStatus.FINISHED:
                assert seq.num_completion_tokens == seq.max_tokens


def test_gpu_worker_rejects_optimized_python(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "must-not-be-certified.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-O", "tests/run_chunk_prefill_review.py", "--output", str(output)],
        cwd=root, env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "correctness validation requires assertions enabled" in result.stderr
    assert not output.exists()


def test_retained_gpu_review_preserves_negative_parity_evidence():
    root = Path(__file__).resolve().parents[1]
    archive = root / "benchmarks/chunked_prefill_tail/evidence/2026-09-08-review-fixes"
    pins = {
        "graph256.json": "60a6d5fc85a13bcb5c3b18c4c1a3d87acb115171a010ff55c02eb0d7ff36ae02",
        "eager256.json": "80794db7ccb0a96c8685dce3e6f39f51ebd75f0c383ae85a1a7835521b1808a8",
    }
    reports = {}
    for filename, digest in pins.items():
        raw = (archive / filename).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest
        report = reports[filename] = json.loads(raw)
        assert report["latency_certified"] is False
        assert report["provenance_before"]["source"] == report["provenance_after"]["source"]
        assert report["pressure"]["short_max_gap_steps"] == 1
    graph, eager = reports["graph256.json"], reports["eager256.json"]
    assert graph["provenance_before"]["source"] == eager["provenance_before"]["source"]
    assert graph["pressure"]["token_ids"] == eager["pressure"]["token_ids"]
    assert graph["passed"] is True
    assert eager["passed"] is False
    assert eager["graph_eager_tokens_equal_by_case"] == {"graph17": False, "pressure": True}
