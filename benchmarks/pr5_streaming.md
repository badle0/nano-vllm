# PR 5 (token streaming API) — design and validation record

## 2026-08-17 repair addendum

The measurements below are retained as historical evidence, but the original
ownership and append-only text designs are superseded by the repaired API:

- `stream()` eagerly returns a single active `StreamSession`. A second stream or
  generate call is rejected until that session finishes or closes.
- `StreamSession` is an iterator and context manager. Early exit must use
  `with llm.stream(...) as stream:` or an explicit `stream.close()`; a retained
  iterator is not closed merely by breaking a loop.
- Admission is transactional. Cleanup calls `Scheduler.cancel(seq_ids)` with
  only IDs admitted by that request; it never performs global cancellation.
- `add_request()` returns its sequence ID. Public `step()` retains legacy pairs,
  while `step_with_metrics()` is the opt-in metrics API.
- `StreamOutput` remains the three-field token event. Caller first/final delivery
  metrics are exposed through `StreamSession.metrics` after each request finishes.
- `StreamingDetokenizer.feed()` returns a correction-capable `TextUpdate`, not an
  append-only string. It decodes a bounded unstable tail, advances its stable
  frontier only across an exactly reconstructible split, and performs one exact
  full decode at `flush()`.

The repaired suite passes 43 tests on the pinned A100/Qwen3-0.6B environment,
including ownership rejection, scoped cancellation, failed-admission rollback,
context cleanup, prefix-cache equivalence, tokenizer rewrites across window
shifts, and a structural bounded-decode-work gate. Throughput figures later in
this document describe the original branch and must not be relabeled as fresh
measurements of the repaired implementation.

The repaired bounded-window benchmark, including one exact final flush per
sequence, produced:

| tokens | repaired us/token |
|---:|---:|
| 64 | 19.6 |
| 256 | 21.1 |
| 1,024 | 22.4 |
| 2,048 | 23.4 |

The nearly flat per-token cost replaces the original cumulative path whose cost
rose to roughly 230 us/token at 2,048 tokens.

Branch: feat/token-streaming (two commits over feat/request-metrics 08a6e35; stacked on
the metrics PR, independent of the sampling series).
Hardware/env: Vast.ai A100 SXM4, torch 2.10 + cu128 + flash-attn 2.8.1, Qwen3-0.6B bf16.
Protocol: all comparisons interleaved within one process (see "One engine per process");
first run discarded; medians of 3 reported; each script run twice, second run kept.

## What this adds

`LLMEngine.stream(prompts, sampling_params)` — a generator yielding
`StreamOutput(seq_id, token_id, finished)` at the moment each token is produced, plus
`StreamingDetokenizer` for consumer-side incremental text assembly, and
`Scheduler.cancel_all()` for teardown when a consumer abandons the stream.

The engine already produced tokens one step at a time and already admitted requests one
step at a time (iteration-level scheduling, Orca OSDI '22). Only *delivery* was
job-granular: `generate()` buffered every token until the last sequence in the batch
finished. This PR removes that buffer. No new computation; a change in delivery
granularity only.

## Design decisions

### Payload: token IDs, not text (engine stays tokenizer-free)

The stream yields token IDs. Text assembly is a consumer-side utility
(`nanovllm/utils/streaming_detokenizer.py`), not engine machinery.

- Upstream already draws this boundary: the tokenizer appears at exactly two sites in
  the engine (`encode` in `add_request`, `decode` after the drain), both outside the
  step loop. In-engine text deltas would put B incremental decodes on every decode step.
- vLLM V1 moved detokenization out of the core loop for exactly this reason; the
  EngineCore's native output is token IDs, and an OutputProcessor detokenizes downstream.
- IDs make the equivalence gate integer-exact: reassembled stream ==
  `seq.completion_token_ids`, no text-diff ambiguity.
- Measured consequence: see "Detokenization cost" — cumulative decode reaches
  229 us/token at n=2048, which would have exceeded a token's entire production cost
  had it run inside the loop.

### Topology: `generate()` and `stream()` are two consumers of one loop

`_step()` is the single loop body returning `StepOutput(events, finished, num_tokens)`;
`_run_engine()` yields it; `generate()` and `stream()` both consume it. `step()` is
retained as a two-line shim returning `(finished, num_tokens)`.

- Two loops would drift. The chunked-prefill work (next PR) edits this loop; a duplicated
  loop means every scheduler change lands twice or the paths diverge silently.
- The loop already had two consumers before this PR: `generate()` and
  `benchmarks/bench_latency.py`, which drives `add_request`/`step()` directly. Making the
  core explicit is refactoring toward existing reality.
- `step()`'s signature is preserved deliberately: `bench_latency.py` is the metrics PR's
  stall instrument and the "before" half of the chunked-prefill comparison. Changing it
  would break the measurement this project depends on. A test
  (`test_step_public_contract`) pins the shim.
- Rejected: a callback hook (`generate(on_token=...)`). HF Transformers took that fork;
  the documented consequence is that every consumer wanting iteration must spawn a
  thread and drain a queue. A generator gives suspension and early termination for free.

### Concurrency: synchronous generator, coupling documented and measured

Consumer time lands on the critical path by design. The async alternative
(per-request generators, concurrent `add_request`, demux) is vLLM's `AsyncLLMEngine`: a
multi-process architecture with ZMQ IPC between API server and engine core. Out of scope
for this PR. The event contract is concurrency-agnostic — a keyed `(seq_id, ...)` stream
demuxes into per-request queues without changing shape — so the async version is additive.

Cost of the choice is quantified below rather than hand-waved.

### Emission predicate: `postprocess` is the single owner

Events are constructed in `Scheduler.postprocess`, below the partial-prefill `continue`.
That line — `if is_prefill and seq.num_cached_tokens < seq.num_tokens: continue` — is the
entire predicate, and it already existed. Consequences, all verified in source:

- A decode step emits one event per scheduled sequence.
- A prefill-completing step emits the sequence's *first* completion token (the same branch
  that sets `first_token_time`, i.e. the streaming yield point and the metrics stamp point
  are the same moment in code — the payoff for stacking on the metrics PR).
- Intermediate chunks emit nothing; the sampled token at a chunk boundary is discarded.
- EOS is appended before the finish check, so an EOS-terminated stream's final event
  carries the EOS id with `finished=True`, and reassembly == `completion_token_ids`
  holds by construction.

Deriving events anywhere else would duplicate the predicate — the exact place a future
generalized-chunking change would edit once and miss twice.

### Teardown on close (forced, not optional)

`stream()` wraps its loop in `try/finally: self.scheduler.cancel_all()`. Idiomatic
streaming use is `break`, which delivers `GeneratorExit` at the suspended yield. Without
teardown the scheduler retains unfinished sequences and the block manager retains their
KV blocks, so the next `generate()` on the same engine inherits ghost sequences and
leaked blocks — and `bench.py`-style warmup-then-measure usage calls `generate` repeatedly.

`cancel_all` sweeps `running` *and* `waiting`, deallocating any sequence with a non-empty
`block_table`. See findings: a mid-chunk sequence lives in `waiting` while holding
allocated blocks, so sweeping `running` alone would leak.

## Commit topology and the zero-behavior-change certification

Two commits, deliberately separable:

| commit | contents | claim |
|--------|----------|-------|
| C1 `refactor: single-source the engine loop` | 3 engine files, +46/-16 | behavior identical to parent |
| C2 `feat: token streaming API` | 6 files, +109/-16 (87 insertions are tests) | purely additive |

C1's claim is *measured*, not asserted. A fixed-seed `generate()` over three prompts
(32 tokens each, temperature 0.6) produces byte-identical `token_ids` at the pristine
parent (08a6e35) and at C1. Under a fixed seed the token stream is a pure function of
batch composition and RNG consumption order, so byte-equality certifies the refactor
changed neither. C1 was verified *standalone*, in a separate git worktree with C2's code
absent from the filesystem (`hasattr(LLMEngine, 'stream') == False` on that checkout),
so the certification applies to the commit a reviewer reads first, not to the merged tip.

End-to-end no-regression (`bench.py`, interleaved, first run discarded):

| tree | kept runs tok/s | median |
|------|-----------------|--------|
| parent 08a6e35 | 8646.2, 8654.7 | 8650.5 |
| C1+C2 branch   | 8607.0, 8564.6 | 8585.8 |

Delta -0.75%, within the ~3% cross-session variance documented for this host in the
sampling series (and see "Null consumer" below, where the same comparison run
*within* one process shows no measurable difference).

## Validation

### Test suite (18 total: 9 engine + 9 detokenizer)

Streaming (6, GPU, shared session engine):
- `test_seeded_equivalence` — per-sequence reassembly of the interleaved stream equals
  `generate()`'s `token_ids` exactly, same seed, temperature 0.6. This is the theorem:
  streaming is a delivery-schedule change, so for identical scheduling the sampled
  tokens must be identical.
- `test_finished_flags` — exactly one `finished=True` per sequence, and it is that
  sequence's last event.
- `test_break_does_not_leak` — break after 5 events, `close()`, assert free-block count
  restored to baseline, scheduler `is_finished()`, and a fresh `generate()` produces
  full-length outputs. Guards the ghost-sequence/leaked-block class directly.
- `test_chunked_prefill_emission` — 200-token prompt at a 64-token batch budget
  (chunks 64/64/64/8): exactly `max_tokens` events, none from intermediate chunks,
  last one `finished`.
- `test_max_tokens_one` — one event per sequence, `finished=True`, emitted from the
  prefill-completing step (the TTFT edge).
- `test_step_public_contract` — the `add_request`/`is_finished`/`step()` protocol that
  `bench_latency.py` uses still returns `(seq_id, token_ids, metrics)` triples.

Detokenizer (9, CPU, no engine): concatenated deltas equal the full decode over
adversarial inputs — emoji including ZWJ sequences, CJK, combining diacritics, mixed
scripts, leading/trailing whitespace — plus per-sequence state release and an
interleaved-two-sequence test (the stream multiplexes, so per-seq state must not
cross-contaminate).

### Caller-visible time-to-first-content (the headline)

Same engine, same seed, 3 prompts x 256 tokens. Batch = wall time until `generate()`
returns; stream = wall time until the first event.

| path | median first content |
|------|----------------------|
| `generate()` | 832.0 ms |
| `stream()`   |  31.7 ms |
| collapse     |  26.3x |

Reproduced across two runs (26.3x both). Cross-validation: the 31 ms figure matches the
interactive TTFT p50 of 31.5-31.7 ms measured independently in the metrics PR through a
different code path — two instruments, one number.

### Null consumer (throughput cost of the streaming path)

32 prompts x 256 tokens, `for _ in llm.stream(...): pass` vs `generate()`, same process,
interleaved. Event count asserted == 8192 (one per token) on every run.

| path | run 1 | run 2 |
|------|-------|-------|
| `generate()`    | 9174.3 | 9144.9 tok/s |
| `stream()` null | 9216.3 | 9233.5 tok/s |
| delta           | +0.46% | +0.97% |

No measurable cost. The delta is inside the 0.5-0.7% within-session spread documented in
the sampling series, so this is "no penalty," not "a speedup" — though the sign was
positive in all four measured runs, plausibly because the null stream skips `generate()`'s
final dict-sort and 8192-token `tokenizer.decode`. Sub-noise; stated as hypothesis.

This result is what licenses the topology: making `generate()` a consumer of the same
loop costs nothing measurable.

### Slow consumer (the sync-design coupling, quantified)

8 prompts x 64 tokens; a `sleep(delta)` in the consumer body; median *step* gap
(measured at each step boundary, not each event — with B=8, seven of every eight
inter-event gaps are intra-step and near zero).

Model, predicted before measurement: `T_step ~= T_base + B * delta`.

| consumer sleep | predicted | measured (run 1 / run 2) |
|----------------|-----------|--------------------------|
|  0 ms | ~3.5 ms | 3.06 / 3.01 ms |
|  2 ms | ~19.5 ms | 19.91 / 19.75 ms |
| 10 ms | ~83.5 ms | 84.75 / 84.87 ms |

Confirmed across a 28x range of induced load. Subtracting the sleep contribution recovers
the baseline (19.9 - 16 = 3.9; 84.7 - 80 = 4.7 ms). The multiplier is B, not 1: consumer
time is paid once per *event*, and a step emits B events. This is the documented price of
the synchronous design — a slow consumer is a producer stall — and the reason the async
engine is the stated follow-up.

### Detokenization cost (follow-up with a measured trigger)

Cumulative decode, per-token cost vs sequence length (CPU, no engine):

| n | us/token |
|---|----------|
|   64 |  10.2 |
|  256 |  32.6 |
| 1024 | 116 |
| 2048 | 229 |

Linear per-token growth = O(n^2) total, as designed and documented in the class docstring.
Compare against a token's production cost: at B=32 a ~5 ms decode step gives ~156 us per
sequence per token. Detokenization is therefore ~20% of production cost at n=256 and
*exceeds* it at n=2048. Sliding-window detokenization (decode the last k tokens, diff
against the previous window) is the known fix, deferred with the crossover measured
rather than guessed. Note this cost is consumer-side by construction — the engine loop
never pays it, which is the payoff of the IDs-only payload decision.

## Findings (discovered during implementation, not designed)

1. **Engines are one per process.** `ModelRunner.__init__` calls
   `dist.init_process_group` unconditionally; `LLMEngine` registers `atexit` cleanup that
   pins the engine (and its 0.9-utilization KV allocation) until interpreter exit; TP>1
   uses a fixed shared-memory name. A second `LLM()` in one process raises
   "trying to initialize the default process group twice". Guarding the init only moves
   the failure to `assert num_kvcache_blocks > 0`, since the first engine still holds the
   memory. Consequence for tests: a session-scoped engine fixture in `tests/conftest.py`,
   shared by the metrics and streaming suites. (The pre-existing metrics suite passed only
   by alphabetical luck before this change.)
2. **Chunked prefill is forced by runtime monkeypatch, not construction.**
   `max_num_batched_tokens` defaults to 16384, so chunking effectively never fires in
   tests. It is a plain scheduler attribute read fresh on each `schedule()` call, so
   `monkeypatch.setattr(llm.scheduler, "max_num_batched_tokens", 64)` on the shared
   engine forces deterministic multi-chunk prefill — no second engine needed.
3. **Abort must sweep `waiting`, not just `running`.** A sequence mid-chunked-prefill
   remains in `waiting` (it moves to `running` only when its final chunk lands) while
   already holding allocated blocks. `cancel_all` therefore deallocates any sequence with
   a non-empty `block_table` across both queues; the predicate is uniformly correct for
   never-scheduled (no blocks), mid-chunk (blocks, waiting), preempted (deallocated
   already, `block_table` empty), and running sequences.
4. **Latent upstream edge, inherited unchanged**: `Scheduler.schedule`'s decode branch
   asserts `scheduled_seqs` non-empty, which can fail if a sole running sequence cannot
   append and preempts itself under total KV exhaustion. Not triggered by streaming;
   noted because `stream()` inherits it.

## Reproduction

pip install -e . --no-deps
pytest tests/ -v                                   # 18 passed
python example_stream.py                           # live typing demo

All scripts run from the repo root; each was run twice with the first run discarded:

python benchmarks/pr5_scripts/ttft_caller.py       # caller-visible TTFT, batch vs stream
python benchmarks/pr5_scripts/null_consumer.py     # throughput cost, null consumer
python benchmarks/pr5_scripts/slow_consumer.py     # step-coupling model
python benchmarks/pr5_scripts/detok_cost.py        # cumulative-decode scaling (CPU only)

Raw output of the recorded session: benchmarks/pr5_raw_results.txt

## References

- Orca: A Distributed Serving System for Transformer-Based Generative Models (OSDI '22)
  — iteration-level scheduling; the step-at-a-time loop a generator encodes directly.
- vLLM V1 architecture (blog, Jan 2025; docs/design/arch_overview) — detokenization and
  request streaming moved off the core loop; async path as multi-process + ZMQ.
- huggingface/transformers `generation/streamers.py` — the callback fork, and the
  thread-plus-queue consumer burden it imposes.
- Sarathi-Serve (OSDI '24) — time-between-tokens as a first-class SLO; streaming is what
  makes TBT observable to a caller at all. Bridge to the chunked-prefill PR.
