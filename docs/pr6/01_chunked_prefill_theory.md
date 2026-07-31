# Chunked Prefill — Theory, From the Ground Up

*Sourced against `badle0/nano-vllm` branch `dev` @ `317e6f0`. Companion files: `02_chunked_prefill_code_map.md` (line-by-line code), `03_chunked_prefill_math_roofline.md` (formulas and roofline).*

---

## 1. Two phases with opposite personalities

Serving an LLM request has two phases that could hardly be more different.

**Prefill** processes the entire prompt at once. If the prompt is T tokens, one forward
pass computes attention and MLP outputs for all T positions and writes T tokens' worth
of K/V into the cache. There is enormous parallel work per pass — thousands of tokens
multiply through every weight matrix — so the GPU's compute units are saturated.
Prefill is **compute-bound**: its speed is set by how fast the tensor cores can do
matmuls.

**Decode** produces one token per sequence per step. Each step multiplies a handful of
token vectors (one per running sequence) through *every weight in the model*. The
weights — 1.2 GB for Qwen3-0.6B — must stream from HBM into the compute units every
single step, to be used once per sequence. Decode is **memory-bandwidth-bound**: its
speed is set by how fast HBM can deliver weights, and the compute units mostly idle.

An intuition that survives scrutiny: the GPU is a photocopier. Prefill is one person
with a 500-page job — the machine runs flat out. Decode is a queue of people copying
one page each — the machine spends most of its time waiting for the next original to
be placed on the glass (weights arriving from HBM), and duplicating a page for sixteen
people costs barely more than for one.

The engine alternates between these two regimes step by step. The question chunked
prefill answers is: *what happens when a 500-page job walks into a queue of
one-pagers?*

## 2. The problem, as nano-vllm exhibits it (and as you measured it)

nano-vllm's scheduler makes each engine step **homogeneous**: `schedule()` returns
either a pure-prefill batch or a pure-decode batch, signalled by a single boolean
(`scheduler.py:57-58` — if the prefill loop scheduled anything, that *is* the step).
And it is **prefill-first**: whenever any sequence is waiting and fits, the entire
step becomes prefill, and every in-flight decode freezes.

You have already measured the consequence. In the metrics PR's stall demonstration,
16 interactive requests are mid-decode at ~5 ms/token when two 2048-token prompts
arrive. The next steps are prefill-only; the interactive requests' inter-token latency
spikes ~10× (mean 4.95 → max 44.4 ms on the current host), and the identity you
verified through two independent measurement paths says exactly what happened:

> **interactive max_ITL ≈ long-request TTFT + one decode step.**

The long request's time-to-first-token is *paid by everyone else as latency*. This is
head-of-line blocking at the step granularity, and it is intrinsic to the
homogeneous-step + prefill-first design, not a bug.

## 3. What nano-vllm already has — and why it doesn't fix the stall

The name "chunked prefill" already appears in this codebase: the `bb823b3` refactor
(upstream PR #218) lets a prompt's prefill be split across steps. The mechanics, all
of which your streaming work built on:

- A step has a token budget, `max_num_batched_tokens` (default 16384, `config.py:9`).
- The **first waiting sequence only** may take a partial slice of that budget
  (`scheduler.py:43` — the gate you know as "only allow chunked prefill for the first
  seq"). It stays in `waiting`, holding its allocated blocks, with
  `num_cached_tokens` marking how far its prefill has advanced.
- A mid-chunk step samples a token at the chunk's last position but **discards** it
  (`scheduler.py:92-93` — the `continue` that your streaming emission predicate
  inherits). Only the chunk that completes the prompt appends a real token.
- Attention across chunks is correct because the chunk's queries attend to the *full
  cached prefix* via the paged KV cache, with FlashAttention's causal mask aligned
  bottom-right when `seqlen_q < seqlen_k` (`attention.py:64-70`).

Here is the crucial observation: **today's chunking does not relieve the stall at
all.** The scheduler's prefill branch still wins on every step while the long prompt
has remaining work, so the interactive decodes freeze for chunk 1, chunk 2, chunk 3…
— the same total prefill time as before, just sliced. What today's chunking buys is
*budget compliance* (a step never exceeds `max_num_batched_tokens` new tokens, which
also bounds activation memory, and matches the warmup shape used for KV sizing). It
was never asked to protect latency.

## 4. The generalization: mixed batches (the actual feature)

The idea, from SARATHI ("piggybacking") and Sarathi-Serve ("stall-free batching"):
put decode tokens **and** a bounded prefill chunk **into the same forward pass**.

Every step, the scheduler:

1. gives each running (decoding) sequence its 1 token — decodes never wait;
2. spends the *remaining* token budget on prefill chunks from waiting sequences
   (possibly several, not just the first);
3. hands the whole heterogeneous set to one forward pass.

Why this is nearly free, in one paragraph of physics: a decode-heavy step is
bandwidth-bound — the step's duration is essentially "time to stream 1.2 GB of
weights," and the compute units are mostly idle *while the weights are streaming
anyway*. A prefill chunk is exactly the kind of dense work those idle units can
absorb. Conversely, from the prefill's point of view, the decode tokens ride along on
weight traffic that the chunk was going to pay for regardless. Below the roofline
ridge (~200–240 tokens per step on this A100; see the math file), adding tokens to a
step barely changes its duration. The 500-page job and the one-page queue share the
copier's warm-up: everyone wins.

The observable consequences, which are also your evaluation predictions:

- **Interactive tail latency collapses.** max_ITL is no longer bounded by "one full
  prompt's prefill" but by "one mixed step at the budget" — from ~44 ms toward
  single-digit milliseconds in your stall scenario.
- **Long-request TTFT changes shape.** Instead of one big step, the prompt takes
  ⌈T/C⌉ mixed steps. Each step is slightly slower than a pure chunk (it carries the
  decodes), but the decodes were going to run anyway; with a sane chunk size the TTFT
  ends up comparable, sometimes better (the pure-prefill step it replaces was itself
  large).
- **Throughput is roughly conserved** when the budget is tuned: total FLOPs and total
  weight traffic are unchanged; what moves is *when* work happens.

## 5. The one true knob: the token budget τ

Everything trades through `max_num_batched_tokens` (which your test suite already
manipulates at runtime — it is a live scheduler attribute):

- **Small τ** → each step is short → the worst-case stall any decode can experience
  is small (TBT/ITL bound tight) → but prefill's chunk is small, so the prefill
  portion of each step sits below the roofline ridge, weights get re-streamed across
  more steps for the same prompt, and prompt throughput / TTFT degrade.
- **Large τ** → efficient, near-monolithic prefill → but each mixed step is long, and
  the ITL bound loosens back toward today's stall.

Sarathi-Serve's framing: pick the largest τ such that a full step at budget still
meets your time-between-tokens SLO. The math file gives the closed forms and a
predicted before/after table for your exact stall scenario, pre-registered so the
eventual measurement is a test, not a fit.

## 6. Why the hard part is *not* the kernel

A fact worth internalizing before design, because it inverts where the effort goes:
**the attention path already supports mixed batches.** `prepare_prefill` builds a
general ragged layout — per sequence, `seqlen_q = num_scheduled_tokens` queries
against `seqlen_k = cached + scheduled` keys, K/V read from the paged cache via
`block_table` (`model_runner.py:129-170`, `attention.py:64-70`). A decode token is
just the degenerate case `seqlen_q = 1` of that layout. FlashAttention's varlen
kernel with bottom-right causal alignment handles the whole heterogeneous batch in
one call.

What actually forbids mixing today is *plumbing*, in four places (mapped precisely in
the code-map file): the scheduler's either/or step shape and first-seq-only gate; the
`is_prefill` boolean that forks `run()` into two different input-preparation paths
and threads through the ambient `Context`; the CUDA-graph fast path, which is
captured for decode-only shapes and keyed by batch size; and the engine's
`num_tokens` sign convention, which encodes "a step is prefill XOR decode" into the
throughput accounting. The design doc's job is to dissolve that dichotomy without
breaking five shipped features that sit on top of it — sampling's per-sequence
parameters, the metrics stamps, the streaming emission predicate, prefix caching's
hash chain, and preemption.

## 7. The correctness pillars you already own

Chunked prefill is unusual in this codebase: its correctness rests on four invariants
you have personally derived or shipped tests for.

1. **Bottom-right causal alignment.** When a chunk's queries (few) meet the full
   prefix's keys (many), FlashAttention aligns the causal mask so the *last* query
   corresponds to the *last* key — each chunk token sees exactly the prefix plus its
   chunk-local predecessors. This is why splitting a prompt does not change any
   logit, up to floating-point tiling effects.
2. **The emission predicate.** "A sequence yields an event iff `postprocess` appends
   a completion token" — mid-chunk steps append nothing, the completing chunk appends
   the first token. Your streaming PR made `postprocess` the single owner of this
   rule precisely so that generalized chunking would change *who* gets gated without
   duplicating the gate.
3. **The T mod B ≡ 1 block-boundary congruence.** `may_append` allocates a fresh
   block exactly when a sequence's length has just crossed into one — the appended
   token is the new block's first occupant, and `prepare_decode`'s slot arithmetic
   targets it.
4. **Hash-before-advance.** Prefix-cache hashing runs on the pre-advance counters,
   chaining each newly filled block's hash from its predecessor; chunk boundaries and
   block boundaries interleave correctly because both are driven by the same
   `num_cached_tokens` bookkeeping.

One honest caveat to carry into validation: unlike streaming, chunked-vs-unchunked
equivalence is **not a bitwise theorem**. Chunking changes kernel tiling and batch
composition, so logits can differ in low-order bits and greedy argmax can flip on
near-ties. Expect token-level agreement almost everywhere, adjudicate divergences
with the fp64 tie methodology from PR 3, and write the gate accordingly.

## 8. What "done" looks like

The feature is done when `benchmarks/bench_latency.py` — unchanged, per the metrics
doc's own words — shows the interactive group's max_ITL bounded by a mixed step
instead of a full prefill, at comparable long-request TTFT and end-to-end throughput
within the documented noise band, with the 37-test suite green and the streaming
equivalence gates still holding under forced chunking. The "before" table exists; the
identity it measured is the thing this feature deletes.
