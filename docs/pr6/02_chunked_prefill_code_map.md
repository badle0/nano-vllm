# Chunked Prefill — Code Map at `dev` (317e6f0)

*Every line number below refers to branch `dev` @ `317e6f0` of `badle0/nano-vllm`,
read directly from the tree. Files are ordered by how central they are to chunking.
Blocks marked **[MIXING BLOCKER n]** are the places whose current shape forbids mixed
decode+prefill steps — constraints any design must resolve, listed as observations,
not prescriptions.*

---

## A. Directly involved — line-by-line

### A1. `nanovllm/engine/scheduler.py` (113 lines) — the chunk state machine

**1–18, construction.** Imports `StreamOutput` (5, your streaming addition). `__init__`
copies `max_num_seqs` (12) and `max_num_batched_tokens` (13) — the **token budget τ**,
a plain attribute read fresh every `schedule()` call, which is why your chunk test can
monkeypatch it. Two deques (17–18): `waiting` (not yet fully prefilled) and `running`
(decoding). Chunking's central state fact: *a mid-chunk sequence lives in `waiting`.*

**20–24.** `is_finished` = both deques empty (the generator's termination predicate);
`add` appends to `waiting`.

**26–58, the prefill branch.**
- 27–28: fresh accumulators each call — `num_batched_tokens` counts *scheduled new
  tokens*, i.e. budget is charged for Q-tokens, not attended keys. (Consequence for
  the math file: equal-budget steps are not equal-cost steps, because a chunk's
  attention cost scales with its prefix length.)
- 31: loop over `waiting`, bounded by `max_num_seqs`.
- 32: `seq = self.waiting[0]` — only ever the head. Combined with 43 this is the
  single-chunker restriction.
- 33–35: remaining budget; a zero budget ends the prefill branch.
- 36–40: **fresh sequence path.** No `block_table` yet → `can_allocate` probes the
  prefix cache and returns how many *full leading blocks* are already cached (or −1 if
  the free list can't cover the rest → 39 breaks; the seq waits whole). 40: the work
  to schedule is `num_tokens − cached_blocks·block_size` — prefix-cache hits shrink
  the chunk *for free* and are never charged to the budget.
- 41–42: **resumption path.** A `block_table` already exists (previous chunk) →
  remaining work is `num_tokens − num_cached_tokens`, using the counter `postprocess`
  advanced last time.
- 43–44: **[MIXING BLOCKER 1a]** `if remaining < num_tokens and scheduled_seqs: break`
  — only the *first* scheduled sequence may take a partial slice. Everyone behind a
  chunking head waits entirely. (Note the converse: the head may chunk even when other
  full prompts were scheduled before its turn didn't arise — the gate is positional,
  not per-sequence capability.)
- 45–46: allocation happens on the **first** chunk only, and for the **whole**
  sequence's blocks up front — which is why a mid-chunk sequence in `waiting` holds
  blocks (your `cancel_all` sweep exists for exactly this).
- 47–48: `num_scheduled_tokens = min(num_tokens, remaining)` — the chunk size — and
  the budget is charged.
- 49–52: **the residence rule.** Only when this chunk *completes* the prompt does the
  sequence flip to `RUNNING` and move `waiting → running`. Mid-chunk: stays at
  `waiting[0]` with status `WAITING`, guaranteeing it is re-picked next step (32).
- 53–54: `first_scheduled_time` stamp, `None`-guarded → idempotent across chunks and
  preemption re-entry (metrics correctness).
- 57–58: **[MIXING BLOCKER 1b]** if the prefill loop scheduled *anything*, the step
  returns immediately as `(seqs, True)` — decodes never share it. This pair of lines
  *is* the homogeneous-step, prefill-first policy; it is the proximate cause of the
  measured 10× stall.

**60–76, the decode branch** (reached only when prefill scheduled nothing).
- 62: pop each running seq (FIFO).
- 63–68: `can_append` loop with preemption — victim is `running.pop()` (LIFO: youngest
  first); if the seq is alone it preempts *itself* and breaks.
- 69–73: the `while`'s `else` (no break): `num_scheduled_tokens = 1`,
  `is_prefill = False` (see A5 on what this flag really gates), `may_append` handles
  the block-boundary congruence, schedule it.
- 74: `assert scheduled_seqs` — the latent KV-exhaustion edge documented in the
  streaming write-up; inherited unchanged.
- 75: `extendleft(reversed(...))` restores FIFO order after the pops.

**78–82, `preempt`.** Status → `WAITING`, `is_prefill = True`, **deallocate**
(recompute-on-resume; `deallocate` also zeroes `num_cached_tokens`, so resumption goes
back through the fresh-sequence path 36–40, where lazy prefix-cache survival may
rescue full blocks), `appendleft` → next in line.

**84–106, `postprocess`** — your streaming version; chunking's commit point.
- 86: one `perf_counter` per step (ITL resolution = step granularity, by design).
- 88: `strict=True` zip — enforces the 1-token-per-scheduled-seq alignment contract.
- 89: `hash_blocks(seq)` **before** 90 — hash-before-advance; the hash window is
  computed from pre-advance counters (see A6).
- 90–91: advance `num_cached_tokens` by the chunk, zero `num_scheduled_tokens`. (This
  zeroing is why `llm_engine._step` must read `num_tokens` first — footgun one.)
- 92–93: **the emission predicate.** Mid-chunk (`is_prefill` and cached < total):
  `continue` — the sampled token is discarded, no append, no event, no timestamps.
- 94–105: real token path — append, first-token stamp, per-token time, hoisted
  `finished` predicate, finish transition (deallocate + `running.remove`), event.
- 106 returns the events; 108–113 `cancel_all` sweeps both deques by the
  `block_table` predicate.

Chunk-relevance summary: generalizing chunking is, at this file's level, a rewrite of
26–76 while keeping 84–113's contracts byte-for-byte intact.

### A2. `nanovllm/engine/model_runner.py` (262 lines) — where chunks become tensors

**17–48, construction.** Line 26: the unconditional `init_process_group` (the
one-engine-per-process constraint). 34–37: warmup → KV allocation → CUDA-graph
capture, in that order (capture must see the final memory map).

**91–101, `warmup_model`.** Builds `min(τ, max_model_len)`-length dummy sequences —
at defaults, a 4×4096 prefill — sets their `num_scheduled_tokens` (99) and runs one
prefill. Two chunk couplings: (a) the peak recorded here feeds `allocate_kv_cache`'s
headroom formula (108, 113), so **changing the default budget changes KV capacity**;
(b) warmup sequences have no `block_table`, which is what the 149 `continue` below
exists for.

**103–121, `allocate_kv_cache`.** 112: `block_bytes = 2·L·block_size·n_kv·d_h·2` —
28 MiB per 256-token block for Qwen3-0.6B. 113: blocks = headroom // block_bytes.
115–121: one giant tensor, views wired into each `Attention` module's
`k_cache`/`v_cache`.

**123–127, `prepare_block_tables`.** Ragged per-seq block lists padded with −1 into a
rectangular int32 tensor. Used by both the prefix/chunk prefill path and decode.

**129–170, `prepare_prefill` — the general ragged layout.** Per sequence:
- 139–142: `start = num_cached_tokens`, `seqlen_q = num_scheduled_tokens`,
  `end = start + seqlen_q`, `seqlen_k = end`. *This is the whole chunk abstraction in
  four lines:* queries are the chunk, keys are the entire prefix-so-far including the
  chunk.
- 143–144: input ids are the chunk's tokens; positions are absolute
  (`range(start, end)`) — RoPE phases are therefore chunk-invariant.
- 145–148: ragged cu_seqlens for q and k accumulate independently; `seqlen_q ≠
  seqlen_k` is exactly the chunk/prefix-cache case, and `seqlen_q = 1` would be a
  decode token — the layout is already fully general.
- 149–150: warmup escape hatch (no block table → no slot mapping; KV writes for
  warmup are suppressed by the −1 sentinel further down the stack).
- 151–161: **slot-mapping window arithmetic** — maps the chunk's token span
  `[start, end)` onto physical cache slots, walking `block_table[start_block …
  end_block)`, trimming the first block by `start % block_size` (155–156) and the
  last by `end − i·block_size` (159–160). This is the code that makes "resume a
  prefill mid-block" physically land in the right slots; it already handles every
  alignment case a generalized chunker needs.
- 162–163: `block_tables` is attached **iff** any key predates the chunk
  (`cu_seqlens_k[-1] > cu_seqlens_q[-1]`) — the flag `attention.py:65` reads as
  "K/V must come from the cache."
- 169: `set_context(True, …)` — publishes the layout ambiently (see A4).

**172–188, `prepare_decode`.** The `seqlen_q = 1` special case, hand-rolled flat:
last token, position `len−1`, `context_lens = len(seq)`, and 181's slot formula
`block_table[-1]·B + last_block_num_tokens − 1` — the write target `may_append` just
guaranteed exists (the T mod B ≡ 1 congruence, A6). Publishes `is_prefill=False`.

**190–198, `prepare_sample`.** Per-seq temperature always; `top_ks`/`top_ps` tensors
only if any sequence deviates (the fast path measured in PR 2/3). Mixed-batch-ready
by construction: it is already per-sequence over an arbitrary `seqs` list.

**200–217, `run_model`.** **[MIXING BLOCKER 3]** Line 202: prefill (or eager, or
bs > 512) → plain forward; otherwise the **decode CUDA-graph fast path**: pick the
smallest captured batch-size bucket ≥ bs (207), copy inputs into the persistent
`graph_vars` buffers padded with the −1 slot sentinel (211 — which the store kernel
honors, A3), replay, gather logits. The graphs were captured for shape
"bs tokens, one query each" (see 227–262); a mixed step's ragged token dimension does
not fit this contract, so under the current design mixed steps run eager — a cost any
design must either accept and measure or engineer around.

**219–225, `run`.** **[MIXING BLOCKER 2]** Line 220: `prepare_prefill if is_prefill
else prepare_decode` — the boolean dichotomy in the flesh. 223: sampler over one
logits row per sequence, `.tolist()` = the per-step GPU→CPU sync (the metrics stamp
and streaming yield semantics hang off this). 224: `reset_context`.

**227–262, `capture_cudagraph`.** Buckets `[1,2,4,8,16,32,…,512]` (239), captured
largest-first sharing one memory pool (247–250), each with a decode-shaped context
(245). Persistent input buffers (233–238) are what `run_model` fills at replay. The
contract to remember: *graphs are keyed by batch size under the one-token-per-seq
assumption* — batch size and token count coincide only for pure decode.

### A3. `nanovllm/layers/attention.py` (75 lines) — where chunk correctness lives

**10–40, the store-KV Triton kernel.** One program per token; 22–23: `slot = −1`
skips the store — the sentinel that makes warmup (no slots) and CUDA-graph padding
(211) safe. 24–30: contiguous `[num_kv_heads·head_dim]` copy into the flat slot.
`store_kvcache` (33–40) asserts layout invariants and launches over N tokens.

**59–75, `Attention.forward` — three regimes.**
- 62–63: **store before attend.** The chunk's K/V land in the cache *first*, which is
  what permits 66 to read K/V *entirely* from cache and still see the current chunk.
- 64–70, prefill: if `block_tables` is present (prefix cache or chunk resumption),
  `k, v = k_cache, v_cache` (66) and `flash_attn_varlen_func` runs with the ragged
  cu_seqlens and `block_table=` paged gather; `causal=True` with `seqlen_q <
  seqlen_k` gives **bottom-right alignment** — the mathematical heart of chunk
  correctness: chunk queries see full prefix + chunk-local causal structure, so
  logits match the monolithic computation. With no cached prefix, k/v are the fresh
  tensors and it is ordinary varlen self-attention.
- 71–74, decode: `flash_attn_with_kvcache` on `q.unsqueeze(1)` with
  `cache_seqlens=context_lens` — a different kernel entry point than prefill's. A
  mixed step would route *everything* through the varlen path (a decode token is
  `seqlen_q=1` there), making this fork collapse naturally — the kernel is the one
  layer where mixing is already solved.

### A4. `nanovllm/utils/context.py` (27 lines) — the ambient dichotomy

A process-global dataclass (`is_prefill`, both cu_seqlens, max lens, slot mapping,
context_lens, block_tables) set by the prepare functions and read inside every
`Attention.forward`, reset after each run. Two chunk-relevant facts: the design
already tolerates "different fields populated per regime," and **the `is_prefill`
boolean is the single ambient bit any mixed design must generalize** — either by
making the varlen fields the universal representation (decode as `seqlen_q=1`) or by
carrying a per-step composition descriptor. Also note this is process-global state:
one step shape at a time, which the synchronous engine guarantees.

### A5. `nanovllm/engine/sequence.py` (96 lines) — the chunk counters

Fields (24–44): the chunk-critical trio is `num_cached_tokens` (31, "prefill progress
committed to cache"), `num_scheduled_tokens` (32, "this step's chunk, valid only
between schedule and postprocess"), and `is_prefill` (33). Note `is_prefill`'s real
job at dev: it is **not** read by the scheduler's branch choice (that's positional);
it gates `__getstate__` (86) — a prefill-mode sequence ships its **full
`token_ids`** to TP workers, a decode-mode one ships only `last_token`. Consequence:
under TP > 1, a T-token prompt chunked n ways serializes O(T) tokens **per chunk**,
O(n·T) total — a real (TP-only) cost of chunking worth a line in any design doc.
Properties 68–78 give the block arithmetic (`num_blocks`, `last_block_num_tokens`,
`block(i)`) that block-manager hashing and decode slot math consume. 85–96: the
allowlisted pickle — metrics fields and anything streaming added never cross the shm
boundary.

> **Repaired-state note (2026-08-17):** the paragraph above describes the pinned
> `dev@317e6f0` snapshot mapped by this document. The repaired chunk branch fixes
> this cost in `67d654f`: rank zero serializes a compact `ScheduledSequence` DTO
> with the scheduled token slice, mode/counts, last token, and block table. The
> transport now has derived page-rounded capacity, a validated frame header, and
> checked reads/writes. Commit `6a98622` exercises those frames across a real
> spawned process and OS shared-memory segment. TP2 model execution remains
> unverified on the one-GPU audit host.

### A6. `nanovllm/engine/block_manager.py` (120 lines) — paged memory under chunks

- 36–41 `compute_hash`: xxhash chained with the previous block's digest — the Merkle
  chain that makes prefix identity positional, not just content-based.
- 58–73 `can_allocate`: probes only **full leading blocks** (`range(num_blocks − 1)`,
  62) — partial tails are never cacheable — verifying token content against the
  stored block (66, guarding hash collisions and lazy invalidation), and counting how
  many new blocks the free list must cover. Returns cached-block count or −1.
- 75–92 `allocate`: re-references cached blocks (rescuing lazily-freed ones off the
  free list, 85–88), allocates the rest, and — line 92 — sets `num_cached_tokens =
  cached_blocks·block_size`: **prefix-cache hits and chunk progress share one
  counter**, which is exactly why `prepare_prefill`'s `start = num_cached_tokens`
  handles both uniformly.
- 94–101 `deallocate`: ref-count decrement in reverse, zero the counter, clear the
  table — freed blocks keep hash+tokens until reallocated (47–48), the lazy
  invalidation that lets preempted work partially resurrect.
- 103–108 `can_append`/`may_append`: the length ≡ 1 (mod block_size) congruence —
  allocate the new block exactly when the just-appended token is its first occupant.
- 110–120 `hash_blocks`: window `[cached//B, (cached+scheduled)//B)` over **pre-advance**
  counters — only blocks the current chunk *completed*; the chain resumes from the
  previous block's stored hash (114). This is why postprocess's 89-before-90 ordering
  is load-bearing, and it already handles chunk boundaries landing anywhere relative
  to block boundaries.

---

## B. Adjacent — how a chunked run touches them

**`nanovllm/engine/llm_engine.py`** (verified above at dev). `_step` (57–65) is a
pass-through — schedule → count → run → postprocess → finished triples — with two
chunk sensitivities: line 60's `num_tokens` **sign convention**
(**[MIXING BLOCKER 4]**: positive ⇒ prefill step, negative ⇒ decode step; a mixed
step fits neither, so the tqdm throughput accounting encodes the dichotomy), and the
read-before-postprocess ordering (footgun one). `step()` (67–69) is the public shim
`bench_latency.py` drives — the before/after instrument — and must keep its shape.
`stream`/`generate` are pure consumers of `StepOutput` and are chunk-agnostic so long
as `postprocess`'s event contract survives.

**`nanovllm/config.py`.** Line 9: τ's default (16384) and home. Note the coupling
chain: τ → warmup shape → recorded peak → KV block count. Any design that changes the
default budget shifts KV capacity as a side effect.

**`nanovllm/metrics.py` + the timestamp fields.** Chunk-proofed by construction:
`first_scheduled_time` is None-guarded per chunk, `first_token_time` stamps at the
append site *below* the mid-chunk `continue`. The metrics are also the feature's
*evaluation*: max_ITL is the number generalized chunking exists to shrink, and
`bench_latency.py` rerun unchanged is the experiment.

**`nanovllm/layers/sampler.py` + `sampling_params.py`.** Already mixed-batch-ready:
fully per-sequence over whatever `seqs` list arrives, greedy handled per-row, fast
path when knobs are off. Chunk interaction: mid-chunk sequences get sampled anyway
(one logits row each) and the result is discarded by the predicate — wasted-but-cheap
work today; a mixed step samples decodes, completing-prefills, and mid-chunks in one
batched call and relies on the same discard.

**Streaming (`stream`, `StreamOutput`, detokenizer, tests).** Unaffected *iff* the
emission predicate keeps its meaning under generalization — which is precisely why
your streaming PR centralized it in `postprocess`. The existing
`test_chunked_prefill_emission` (budget-monkeypatch) is the regression tripwire, and
the seeded equivalence gate must be re-run with chunking forced.

**`nanovllm/models/qwen3.py` + other `layers/`** (linear, RoPE, RMSNorm,
embed/head). Shape-agnostic over the flat ragged token dimension: they see
`[total_tokens, hidden]` and absolute positions, and never consult `is_prefill`.
Attention is the only layer that reads the Context's regime fields. One item flagged
for design-time verification rather than asserted here: the exact site of the
last-position logits gather (one row per sequence out of the ragged prefill batch,
keyed off `cu_seqlens_q`) lives in the model/logits path — confirm its indexing
generalizes when a step's sequence list is heterogeneous.

**`nanovllm/utils/loader.py`, `llm.py`, `__init__.py`.** Untouched by chunking:
weight loading, the thin `LLM(LLMEngine)` wrapper, and exports.

**`bench.py`, `example*.py`, `tests/`.** Consumers only. `bench_latency.py` is
sacred (the before/after); `bench.py` is the throughput no-regression gate;
`conftest.py`'s one-engine-per-process constraint means chunk experiments keep using
the runtime-budget monkeypatch rather than second engines.
