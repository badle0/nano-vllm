import torch
from nanovllm import SamplingParams

# TRAP 1 (prefix cache): equivalence runs generate() then stream() on ONE engine.
# Prompts must stay under one KV block (256 tokens): can_allocate only consults
# full blocks (range(num_blocks - 1)), so sub-block prompts get zero cache hits
# and run 2's prefill batching is identical to run 1's. Longer prompts would hit
# the prefix cache on the second run, change batch composition, and legitimately
# perturb sampled tokens — a flaky gate that looks like a streaming bug.
PROMPTS = ["The capital of France is", "def fibonacci(n):", "In 1969, humans first"]
SP = SamplingParams(temperature=0.6, max_tokens=32, ignore_eos=True)

def _seed():
    torch.manual_seed(1234); torch.cuda.manual_seed_all(1234)

def _reassemble(events):
    per_seq = {}
    for ev in events:
        per_seq.setdefault(ev.seq_id, []).append(ev.token_id)
    # TRAP 2 (seq_id offset): absolute ids are a global counter also consumed by
    # warmup; only their ORDER maps to submission order — same trick generate uses.
    return [per_seq[k] for k in sorted(per_seq)]

def test_seeded_equivalence(llm):
    _seed(); batch = [o["token_ids"] for o in llm.generate(PROMPTS, SP, use_tqdm=False)]
    _seed(); streamed = _reassemble(llm.stream(PROMPTS, SP))
    assert streamed == batch

def test_finished_flags(llm):
    _seed(); events = list(llm.stream(PROMPTS, SP))
    finished = [e for e in events if e.finished]
    assert len(finished) == len(PROMPTS)
    last_by_seq = {e.seq_id: e for e in events}
    assert all(last_by_seq[e.seq_id] == e for e in finished)   # finished is each seq's last event

def test_break_does_not_leak(llm):
    bm = llm.scheduler.block_manager
    baseline = len(bm.free_block_ids)
    it = llm.stream(PROMPTS, SP)
    for _ in range(5): next(it)
    it.close()                                   # GeneratorExit -> finally -> cancel_all
    assert len(bm.free_block_ids) == baseline
    assert llm.scheduler.is_finished()
    _seed(); out = llm.generate(PROMPTS, SP, use_tqdm=False)   # engine reusable
    assert all(len(o["token_ids"]) == SP.max_tokens for o in out)

def test_max_tokens_one(llm):
    _seed()
    events = list(llm.stream(PROMPTS, SamplingParams(temperature=0.6, max_tokens=1, ignore_eos=True)))
    assert len(events) == len(PROMPTS) and all(e.finished for e in events)
    # one event per seq, finished=True, emitted from its prefill-completing step

def test_step_public_contract(llm):
    _seed()
    llm.add_request(PROMPTS[0], SP)
    while not llm.is_finished():
        out, num_tokens = llm.step()
    assert out and len(out[0]) == 3          # (seq_id, token_ids, metrics) — bench_latency's shape

def test_chunked_prefill_emission(llm, monkeypatch):
    # Budget is a plain scheduler attribute read fresh each schedule() call —
    # shrink it at runtime to force multi-chunk prefill on the shared engine.
    monkeypatch.setattr(llm.scheduler, "max_num_batched_tokens", 64)
    long_prompt = [11] * 200                     # chunks: 64/64/64/8
    sp = SamplingParams(temperature=0.6, max_tokens=8, ignore_eos=True)
    events = list(llm.stream([long_prompt], sp))
    assert len(events) == sp.max_tokens          # 3 intermediate chunks emitted nothing
    assert events[-1].finished and not any(e.finished for e in events[:-1])