import torch
from nanovllm import SamplingParams

# C3 pins (F2 verdict): decode-first admission, FIFO chunk fill, <=1 partial chunk
# per step, <=1 mid-chunk seq system-wide, emission predicate untouched, preemption
# machinery untouched. White-box budget overrides below exercise scheduling
# mechanics only; deployment budgets are immutable constructor configuration.
# Snapshots are taken AT schedule() time: seq fields (num_scheduled_tokens etc.)
# are mutated by postprocess, so live refs would assert against stale state.
# TRAP (shared engine): prompt fillers are unique per test — repeated multi-block
# content would hit the prefix cache hashed by an earlier test and silently change
# the chunk arithmetic these tests assert on (test_streaming TRAP 1).


def _capture_schedules(llm, monkeypatch):
    caps = []
    orig = llm.scheduler.schedule
    def spy():
        seqs, ragged = orig()
        caps.append(([{"id": s.seq_id, "is_prefill": s.is_prefill,
                       "scheduled": s.num_scheduled_tokens, "cached": s.num_cached_tokens,
                       "total": s.num_tokens} for s in seqs], ragged))
        return seqs, ragged
    monkeypatch.setattr(llm.scheduler, "schedule", spy)
    return caps


def _drain(llm):
    while not llm.is_finished():
        llm.step()


def test_mixed_step_decode_first(llm, monkeypatch):
    torch.manual_seed(1234); torch.cuda.manual_seed_all(1234)
    monkeypatch.setattr(llm.scheduler, "_max_num_batched_tokens", 256)
    llm.add_request([7] * 100, SamplingParams(temperature=0.6, max_tokens=32, ignore_eos=True))
    llm.step()                                   # full prefill -> seq is decoding
    caps = _capture_schedules(llm, monkeypatch)
    llm.add_request([9] * 600, SamplingParams(temperature=0.6, max_tokens=2, ignore_eos=True))
    llm.step()
    snap, ragged = caps[-1]
    assert ragged is True                        # ragged step: decode + chunk together
    assert snap[0]["is_prefill"] is False        # decode admitted first
    assert snap[0]["scheduled"] == 1             # charged exactly one token
    assert any(s["is_prefill"] for s in snap)    # prefill chunk in the same step
    assert sum(1 for s in llm.scheduler.waiting if s.block_table) <= 1
    _drain(llm)


def test_decode_never_skips_under_chunk_pressure(llm, monkeypatch):
    torch.manual_seed(1234); torch.cuda.manual_seed_all(1234)
    monkeypatch.setattr(llm.scheduler, "_max_num_batched_tokens", 128)
    sp = SamplingParams(temperature=0.6, max_tokens=24, ignore_eos=True)
    for _ in range(3):
        llm.add_request([5] * 40, sp)            # 120 tokens: all three prefill in one step
    llm.step()
    decode_ids = {s.seq_id for s in llm.scheduler.running}
    assert len(decode_ids) == 3
    caps = _capture_schedules(llm, monkeypatch)
    llm.add_request([21] * 500, SamplingParams(temperature=0.6, max_tokens=2, ignore_eos=True))
    while llm.scheduler.waiting:
        llm.step()                               # the 500-token prompt chunks across steps
    assert len(caps) >= 4                        # 500 tokens through a ~125-token budget
    for snap, ragged in caps:
        assert ragged is True
        scheduled_ids = {s["id"] for s in snap}
        assert decode_ids <= scheduled_ids       # F2: no live decoder ever skips a step
        partials = [s for s in snap
                    if s["is_prefill"] and s["cached"] + s["scheduled"] < s["total"]]
        assert len(partials) <= 1                # <=1 partial chunk per step
    _drain(llm)


def test_multi_seq_chunk_emission(llm, monkeypatch):
    # two long prompts chunking concurrently: mid-chunk steps emit nothing,
    # each seq emits exactly max_tokens events, finished flag on the last only
    monkeypatch.setattr(llm.scheduler, "_max_num_batched_tokens", 96)
    sp = SamplingParams(temperature=0.6, max_tokens=6, ignore_eos=True)
    events = list(llm.stream([[11] * 300, [13] * 250], sp))
    per_seq = {}
    for ev in events:
        per_seq.setdefault(ev.seq_id, []).append(ev)
    assert len(per_seq) == 2
    for evs in per_seq.values():
        assert len(evs) == sp.max_tokens
        assert evs[-1].finished and not any(e.finished for e in evs[:-1])


def test_preempt_under_mixing(llm, monkeypatch):
    torch.manual_seed(1234); torch.cuda.manual_seed_all(1234)
    monkeypatch.setattr(llm.scheduler, "_max_num_batched_tokens", 128)
    sp = SamplingParams(temperature=0.6, max_tokens=16, ignore_eos=True)
    for _ in range(2):
        llm.add_request([5] * 30, sp)
    llm.step()                                   # both decoding
    llm.add_request([23] * 400, SamplingParams(temperature=0.6, max_tokens=2, ignore_eos=True))
    bm = llm.scheduler.block_manager
    real = bm.can_append
    calls = {"n": 0}
    def flaky(seq):                              # deny the first decoder once, mid-mixing
        calls["n"] += 1
        return False if calls["n"] == 1 else real(seq)
    monkeypatch.setattr(bm, "can_append", flaky)
    llm.step()                                   # preemption fires inside a mixed step
    monkeypatch.setattr(bm, "can_append", real)
    assert sum(1 for s in llm.scheduler.waiting if s.block_table) <= 1
    _drain(llm)                                  # preempted seq recovers and completes
    assert llm.scheduler.is_finished()


def test_step_shim_convention(llm, monkeypatch):
    # C4 contract: step() keeps (finished, num_tokens); pure steps keep the legacy
    # signed value, a mixed step reports +num_prefill_tokens (decode rows excluded).
    torch.manual_seed(1234); torch.cuda.manual_seed_all(1234)
    monkeypatch.setattr(llm.scheduler, "_max_num_batched_tokens", 256)
    llm.add_request([27] * 100, SamplingParams(temperature=0.6, max_tokens=8, ignore_eos=True))
    _, n = llm.step()
    assert n == 100                              # pure prefill: +tokens (legacy)
    _, n = llm.step()
    assert n == -1                               # pure decode: -num_seqs (legacy)
    llm.add_request([29] * 600, SamplingParams(temperature=0.6, max_tokens=2, ignore_eos=True))
    _, n = llm.step()
    assert n == 255                              # mixed: budget 256 - 1 decode = 255 chunk
    _drain(llm)
