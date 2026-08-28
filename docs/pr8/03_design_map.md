# PR 8 — 03: speculative decoding v2 design map

This document records the architecture before implementation. It is a decision
ledger, not a claim that the feature exists. The code base is
`origin/fork-main` at `663753b99131945c297c1fbe02341108f422dce7`.
The old `feat/speculative-decoding` tip `a632b59` is prototype evidence only.
Throughout this packet, v1/v2 name milestones of this nano-vLLM fork; they do
not import API or fallback contracts from the separate vLLM project.

The selected first release is classic draft-target speculative decoding for
pure-decode work on one GPU. Its normal path uses the exact modified-rejection
law over the fork's temperature, top-k, and top-p distribution. A separately
counted target-distribution fallback handles a finite-precision branch that is
analytically unreachable; therefore an unqualified distribution-exactness claim
is valid only when that fallback rate is zero. It is disabled by default and
must not change the baseline path when disabled.

## 1. What the design optimizes for

The priority order is:

1. Preserve the target distribution and committed engine state.
2. Preserve all current admission, chunked-prefill, streaming, metrics,
   cancellation, and lifecycle guarantees when speculation is off and on.
3. Fail unsupported configurations before acquiring GPU or worker resources.
4. Improve low-concurrency decode latency when measurements justify it.
5. Bypass speculation where it loses; there is no universal-speedup claim.

The design does not optimize first for minimum diff size. Speculative decoding
changes one-token assumptions across scheduling, KV ownership, logits
selection, sampling, postprocessing, and accounting. Making those concepts
explicit is smaller in risk than hiding them behind booleans and mutable
`Sequence` fields.

## 2. Scope boundary

### In the complete v1 path

- A separate smaller autoregressive draft model with the same token-ID space.
- A fixed configured maximum lookahead `K`, reduced per cycle when necessary.
- Exact greedy verification and exact modified rejection on the normal path,
  plus an explicit, counted finite-precision recovery policy.
- Temperature, top-k, and exact top-p applied consistently to target and draft.
- Packed, pure-decode target verification.
- Separate target and draft KV tensors indexed by one logical block table.
- The standard bonus token when all proposals are accepted and output budget
  remains.
- Multi-token streaming, metrics, cancellation, preemption, prefix-cache, eager,
  and CUDA-graph certification.
- A measured low-batch routing envelope with baseline fallback.

### Explicitly deferred

- Tensor parallelism greater than one, pipeline parallelism, or multi-node work.
- FlashInfer top-p in sampled speculative mode.
- Trees, Medusa/EAGLE heads, self-speculation, prompt lookup, and retrieval
  proposal sources.
- Beam search and lossy/lenient acceptance policies.
- Per-request adapters, grammar constraints, penalties, or processors not
  currently supported by the baseline sampler.
- Online draft training and dynamic tree construction.

Deferred means fail fast or remain on baseline decoding; it never means silently
run an uncertified combination.

## 3. Proposed state machine

```text
admit committed request
        |
        v
baseline prefill / chunked prefill ----> ordinary one-token decode fallback
        |                                      ^
        | prompt complete                      | ineligible / no headroom /
        v                                      | over budget / policy bypass
select pure-decode candidates --> build SpecStepPlan + reserve provisional slots
        |                                      |
        |                                      | exception
        v                                      v
catch draft cache up to committed prefix    rollback reservation
        |
        v
draft K candidates and retain the actual q_i distributions
        |
        v
target verifies [last_committed, d1, ..., dK] in one ragged pass
        |
        v
left-to-right accept/reject using p_i and q_i
        |
        +-- first rejection j --> accepted prefix + residual correction
        |
        +-- all accepted ------> K drafts + target bonus (if budget permits)
        |
        v
validate every row's SpecResult before mutating any Sequence
        |
        v
atomically commit tokens/cache coverage; trim reservation; hash committed blocks
        |
        v
emit one StreamOutput per committed token; record work and delivery separately
```

The synchronous `_step()` call is the transaction boundary. A stream may be
closed before or after a cycle, never during target verification. GPU work may
write provisional KV slots, but public and logical state changes only in commit.

## 4. Non-negotiable invariants

### I1 — public tokens are committed tokens

`Sequence.token_ids`, `last_token`, completion output, detokenizer input, and
prefix hashes contain only tokens selected by the exact target-law algorithm.
Draft candidates remain cycle-local until accepted.

### I2 — separate cache coverage

For every running sequence, track target and draft coverage independently:

```text
0 <= target_cached_tokens <= committed_tokens
0 <= draft_cached_tokens  <= committed_tokens
```

At a normal decode-ready boundary, target coverage remains the current
`len(sequence) - 1` convention: the last emitted token is processed next. Draft
coverage may lag and must be caught up before proposing. Physical stale values
beyond logical coverage are never attended to and are overwritten before use.

### I3 — exact proposal law

`q_i` is the distribution that actually sampled proposal `d_i`, after the same
temperature/top-k/top-p transformation contract used for that draft row. Raw
draft logits or a reconstructed approximation are not interchangeable with
`q_i`. In the formulas, `p_i` and `q_i` are the row-normalized laws induced by
the canonical FP32 weight tensors actually used by nano-vLLM's categorical draw.
The reference oracle upcasts those retained FP32 tensors and their row masses;
it does not recompute softmax from logits in FP64 and thereby test a different
law. Before acceptance, the selected proposal metadata must satisfy finite
`q_i(d_i) > 0`; violation raises the typed invariant error without mutation.

### I4 — exact target law

`p_i` is the current target model's effective distribution at the candidate
prefix, after all supported transforms. Analytically, the first-rejection
residual is `normalize(max(p_i - q_i, 0))`.

Finite precision is part of this invariant rather than an implementation detail.
Validate both canonical rows as finite and non-negative with positive finite row
mass, then define normalized `p_i` and `q_i` from those same retained weights.
An invalid row raises the typed invariant error. Let
`r_i = max(p_i - q_i, 0)` and `z_i = sum(r_i)`.

The fast production path may compute `r_i` and `z_i` in its certified dtype. If
its mass is zero or non-finite, recompute by upcasting the retained canonical
FP32 rows to the FP64 reference path. Then:

- finite `z_i > 0`: sample the robust residual, even if the fast path failed;
- `z_i == 0` after a numerical rejection: robustly renormalize `p_i`, sample it
  with the independent corrective draw, and increment
  `spec_residual_numerical_fallbacks`;
- non-finite FP64/reference `z_i`: raise the typed invariant error and roll back;
  do not mask a broken invariant with target fallback.

The implementation must never divide by zero or silently choose an arbitrary or
uniform token.

The target-distribution fallback is a declared finite-precision availability
policy for a branch that is unreachable in exact arithmetic when `p_i == q_i`.
It is observable and must not be presented as an exact residual-law event. Under
a coupling that agrees off fallback events, its total-variation contribution is
at most `Pr(fallback)`; certification reports the event rate and a preregistered
upper confidence bound. An exact-release claim requires zero fallback events in
all non-injected certification workloads. Any nonzero rate qualifies the result
as machine-precision guarded and reports that bound. Any faster production path
must match the FP64 oracle's branch classification and categorical law on
adversarial zero, subnormal, and near-equal inputs.

### I5 — resource accounting precedes execution

A cycle starts only after proving that its target query positions, draft work,
graph buffers, model-position range, probability/sampling peak live set, and
provisional block writes fit. Expected acceptance is never used as an allocation
guarantee. The proof occurs before speculative tensors or blocks are allocated;
failure routes the whole selected batch through ordinary decode.

### I6 — failed work commits nothing

Failure during catch-up, drafting, verification, acceptance, result validation,
or commit rolls back cycle-local staging and provisional reservations. Existing
committed tokens, cache lengths, block references, session ownership, and prefix
hashes remain valid.

### I7 — speculation-off is the old path

With `draft_model=None` and `num_speculative_tokens=0`, construction, RNG use,
model loading, KV capacity, graph capture, scheduler decisions, outputs, metrics,
exceptions, and cleanup follow the current baseline path. No draft object is
created and no extra random draw is consumed.

### I8 — work and output are different units

Decode rows, draft positions, target verification positions, accepted proposals,
and committed output tokens are distinct counters. None may be relabeled as
another to manufacture throughput.

## 5. Data model

Names below are design sketches, not frozen APIs.

```python
@dataclass(frozen=True, slots=True)
class SpecStepPlan:
    rows: tuple[Sequence, ...]
    lookahead: int                 # common maximum K for this cycle
    target_query_tokens: int       # sum(K + 1) for verifier rows
    reservation: BlockReservation # owns only newly appended block IDs
    bypass_reason: str | None

@dataclass(frozen=True, slots=True)
class ProposalBatch:
    token_ids: Tensor              # [B, K], cycle-local
    probabilities: Tensor          # actual q_i; bounded by routing policy
    draft_positions: int

@dataclass(frozen=True, slots=True)
class SpecRowResult:
    seq_id: int
    committed_token_ids: tuple[int, ...]
    accepted_draft_tokens: int
    proposed_draft_tokens: int
    target_cache_advance: int
    draft_cache_advance: int
    used_bonus: bool

@dataclass(frozen=True, slots=True)
class SpecExecutionResult:
    rows: tuple[SpecRowResult, ...]
    target_verification_positions: int
    draft_positions: int
```

`ModelRunner` reads a plan and produces an execution result. It does not append
tokens or alter scheduler queues. `Scheduler` validates all rows and owns the
commit. A separate `BlockReservation` object makes rollback and trimming
idempotent and prevents a failed cycle from guessing which blocks it owns.

Cycle-local proposal tensors must not be added to `Sequence`. The only new
persistent per-request state should be draft cache coverage (and, only if needed,
a draft-validity generation/hash).

## 6. Design decisions and alternatives

### F1 — Algorithm family

Options: classic draft-target chain; tree verification; Medusa/EAGLE;
self-speculation; prompt/retrieval lookup.

Verdict: classic chain first. It gives a complete exactness proof, uses existing
off-the-shelf Qwen checkpoints, and exposes the engine problems we need to solve
without adding training or tree masks. The proposal seam should remain narrow so
other proposers can be added later, but v1 must not build an abstraction larger
than the measured need.

### F2 — Exact versus lenient acceptance

Options: exact modified rejection; greedy-match only; typical/threshold/entropy
acceptance.

Verdict: exact rejection sampling, with greedy as its optimized point-mass case.
Lossy policies are out of scope and must never share an `exact` flag. Correctness
is target-distribution equality within declared numerical tolerances, not an
informal quality comparison.

### F3 — Draft source and model pair

Verdict: a smaller Qwen-family causal LM supplied through `draft_model`. The
actual target/draft pair is a benchmark variable, not hard-coded policy. A pair
is release-eligible only after memory fit, tokenizer identity, acceptance, and
cost measurements.

The earlier PR7 pair measurements are hypotheses. They were recorded on an older
engine and cannot select the v2 default without rerunning on the current base.

### F4 — Tokenizer compatibility

Options: a canary encode/decode check; vocabulary-size check; full token-ID-space
identity; arbitrary-vocabulary mapping.

Verdict: require a verified identical token-ID space in v1. Compare tokenizer
class/config, serialized vocabulary and added-token mapping, special-token IDs,
normalization/pre-tokenization configuration, and stable fingerprints of the
tokenizer artifacts. A one-string canary is not proof.

An arbitrary `q` over the *target IDs* can still yield exact rejection sampling,
but mapping a different draft vocabulary into target IDs is a separate algorithm
and proof. It is deferred.

### F5 — Configuration contract

Append fields so existing positional construction stays compatible:

```python
draft_model: str | os.PathLike | None = None
num_speculative_tokens: int = 0
```

`K=0` and `draft_model=None` is disabled. `K>0` requires a valid draft directory.
Validation uses typed exceptions and happens before tokenizer, workers, process
groups, or GPU model allocation. Initial compatibility gates are TP=1 and
`top_p_backend="exact"`.

An optional batch crossover field should be added only after measurements define
its semantics. Until then, the scheduler's hard token budget and an internal
evidence-backed bypass rule are preferable to an arbitrary public knob.

### F6 — One resource owner versus two runners

Options: instantiate a second `ModelRunner`; embed target and draft under one
runner; put draft in another process/device.

Verdict: one `ModelRunner` owns both models on v1's GPU. A second runner would
duplicate process-group ownership, CUDA context, transport, graphs, memory
profiling, and cleanup, conflicting with the current sequential-engine lifecycle
guarantees. Target and draft are separate model states inside one transactional
owner.

Every draft attribute is initialized to a safe sentinel before fallible work.
Cleanup order is graphs and static buffers, module cache views, KV tensors,
samplers/models, process group, GC, then CUDA allocator cleanup.

### F7 — KV layout

Options: independent block managers; no draft cache; one logical block table with
parallel physical caches.

Verdict: one logical block-ID space, separate target and draft KV tensors, and a
joint per-block memory price:

```text
bytes_per_logical_block = target_KV_block_bytes + draft_KV_block_bytes
num_blocks = floor(profiled_usable_bytes / bytes_per_logical_block)
```

The draft tensor may have different layer/head geometry. Only block IDs and token
slots are shared; KV values are never shared between models. Explicit
`num_kvcache_blocks` remains an exact override checked against the joint cost.

### F8 — Prefix-cache validity

A target prefix-cache hit does not prove that the corresponding draft KV is
valid. v1 therefore treats draft coverage as independently unproven on allocation
or preemption and catches the draft up over the committed prefix. Reusing draft
prefix KV later requires per-block validity tied to the same token hash.

Rewriting identical draft KV into a shared prefix block is correct but may cost
performance; optimize it only after a block-reuse correctness test exists.

### F9 — Scheduler representation

Options: retain `(seqs, is_prefill)` and infer speculation; add flags to
`Sequence`; return an explicit plan.

Verdict: introduce a typed plan. The current boolean already means both
"prefill" and "ragged/mixed". A third meaning would make token charging,
reservations, TP transport, and runner dispatch implicit. The planner should
select rows/mode first and reserve exact write coverage second.

The ordinary plan remains representable without speculative fields so the same
scheduler can preserve current behavior.

### F10 — Interaction with chunked prefill

Options: speculate decodes inside a mixed decode+prefill ragged step; delay
prefill to speculate; use baseline whenever the step is mixed.

Verdict: pure-decode speculation only. If current fairness logic admits any
waiting/prefill work, execute the baseline ragged step. Speculation may not starve
prompt work or change the certified one-mid-chunk invariant.

A future unified mixed step is possible, but it requires separate packed logits
selection and a scheduler policy that prices both prefill and verification. It
is not a v1 shortcut.

### F11 — Token-budget accounting and speculative microbatching

A verifier for batch `B` and lookahead `K` evaluates `B*(K+1)` target query
positions, not `B`. It must fit `max_num_batched_tokens`, captured buffers, and
the selected graph tier.

Verdict: compute a common batch-wide effective `K`:

```text
K_budget = floor(max_num_batched_tokens / B) - 1
K_eff = min(configured_K, K_budget, per-request headroom minima)
```

If `K_eff < 1`, use ordinary decode. Initially, do not split a large decode batch
into speculative and ordinary subgroups in one engine step; that complicates
fairness and RNG identity. If high-batch measurements justify it, a separately
planned speculative microbatch can be added with explicit ordering tests.

Draft positions are counted separately even though the target token budget is
the graph/input safety bound.

### F12 — Remaining-length and model-position headroom

For sequence length `L`, remaining completion budget `R`, and a canonical
bonus-capable verifier with inputs `[x_last,d1,...,dK]`, require:

```text
K <= R - 1
L + K <= effective_max_model_len
```

Thus a request with only one output token remaining uses baseline decode. This
ensures the cycle can emit `K+1` tokens on full acceptance without crossing the
request limit and can address every verification position. Commit still truncates
defensively at the first EOS only when `ignore_eos` is false, and always at
`max_tokens`. With `ignore_eos=true`, EOS is an ordinary committed token.

A future per-row ragged `K_i` may recover tail efficiency. The first version uses
one common `K_eff` because it simplifies graph shapes, reservations, and proof.

### F13 — Multi-position reservation

Current `can_append/may_append` reserves a one-token write. v1 needs a
transactional API such as:

```text
reserve_through(sequence, highest_written_position) -> BlockReservation
commit(reservation, blocks_needed_by_committed_sequence)
rollback(reservation)
```

The lease records exact appended block IDs and verifies refcounts. On rejection,
physical rejected KV may remain dirty inside retained committed blocks, but
logical coverage stops before it. Blocks beyond the committed sequence length
are trimmed. Prefix hashing sees only complete, committed, logically cached
blocks.

### F14 — Verification compute path and logits selection

The attention path can represent verification as a causal ragged query over a
paged prefix. The current LM head, however, gathers only the final query row of
each prefill segment. Speculation needs every live verifier row.

Verdict: extend execution context with an explicit logits selection mode, for
example `LAST_PER_SEQUENCE`, `ALL_LIVE_ROWS`, or explicit row indices. Ordinary
prefill keeps `LAST_PER_SEQUENCE`; verifier uses `ALL_LIVE_ROWS`. Do not globally
remove the gather or overload `is_prefill` again.

Verification starts eager for correctness. Existing ragged CUDA graphs may be
reused only after live-row mapping and per-bucket equality tests pass. Dedicated
small token buckets are measurement-driven.

Every declared verifier route must also have an initialization-time pretouch or
capture for its production-shaped all-live-row output. A shape outside the
declared warm/capture envelope is not allowed to trigger an unplanned first-cycle
compile; it follows the explicitly certified eager route or deterministically
bypasses speculation.

### F15 — Distribution preparation

Options: duplicate warp code in the rejection sampler; mutate current logits;
extract one canonical transform-to-probabilities path.

Verdict: add canonical, non-mutating distribution preparation for speculative
rows while leaving the existing baseline `run()` behavior intact. The transform
order is temperature, top-k mask, top-p mask, then normalization. Greedy has a
separate point-mass path.

The exact top-p implementation's private FP32 workspace is part of the contract.
FlashInfer's sampling-only API, Philox use, and tie behavior do not currently
provide the full `q`/`p` representation needed by residual correction, so it is
rejected for v1 speculation.

### F16 — Probability storage and peak sampler workspace

Exact rejection needs `q_i(d_i)` for acceptance and the full `q_j` at the first
rejection. Options include storing `[B,K,V]`, recomputing the rejected position,
or compressing known sparse support. Verification also produces `K+1` target
rows per sequence. A vectorized implementation that materializes their
probabilities has `p` shaped `[B,K+1,V]`; counting only the draft side is not a
valid peak-memory model.

Verdict for correctness-first v1: retain the actual FP32 proposal distributions
`q[B,K,V]` and materialize the canonical FP32 target distributions
`p[B,K+1,V]` for a bounded speculative batch. This is simple and auditable. Their
simultaneously live lower bound is:

```text
W_probability_floor = sizeof(FP32) * V * (B*K + B*(K+1))
```

The routing and KV-sizing quantity is the larger measured-and-modeled peak live
set, not that floor alone:

```text
W_spec_live_peak(B,K,V,route_key) = peak_live(
    retained q,
    current draft hidden/logits, probability output, and transform scratch,
    verifier hidden/logits and materialized p,
    correction residual rows,
    acceptance masks/scalars,
    categorical RNG/noise,
    exact top-k/top-p private sort/cumsum/index/mask workspaces,
    eager or graph-static buffers,
    target/draft activation and library workspaces not counted as persistent,
)
W_spec_reservation = W_spec_live_peak + allocator_safety_margin
```

Use tensor-lifetime analysis to avoid double-counting buffers that provably do
not overlap, but do not assume in-place reuse until an actual implementation and
peak-memory measurement prove it. Persistent weights and KV tensors are modeled
separately and included exactly once in total capacity. Compare incremental CUDA
peak allocated bytes with `W_spec_live_peak`; use peak reserved/headroom evidence
to validate `W_spec_reservation`. Automatic KV sizing reserves
`W_spec_reservation` before selecting the cache block count. Explicit block-count
overrides are rejected with a typed capacity error when the same reservation
cannot fit. At scheduling time, an ineligible `route_key` routes the whole
selected batch through ordinary decode before allocating speculative workspace.

`route_key` is machine-readable and covers at least the draft batch bucket,
verifier ragged `(token_bucket, slot_tier)`, `effective_k`, execution mode, and
the heterogeneous sampler-plan signature. The signature distinguishes active
plain/temperature/top-k/top-p/combined row composition, or selects a proved
conservative worst case. A homogeneous mode label alone is insufficient.

After certification, streamed/recomputed target rows, recompute-on-rejection, or
an exact sparse-support representation may reduce memory. Each alternative must
publish its smaller live-set formula and reproduce the actual target/proposal law
at the rejection point. Storing only `q(d)` is insufficient.

### F17 — RNG, warmup, and compile contract

Speculation changes the number and order of random draws, so same-seed sampled
tokens are not a valid equality gate. The release contract is equality in law.

Verdict: keep all draft, acceptance-uniform, residual, and bonus draws on rank
zero and outside model CUDA graphs. Preserve the existing global CUDA RNG path
initially so tests can control it with the existing seed mechanism; save and
restore RNG around warmup/capture exactly as baseline initialization does.
Spec-off consumes no new draw.

Speculation adds draft decode, all-query verifier logits, probability preparation,
acceptance, corrective-residual, and bonus-sampling entry points. Initialization
must pretouch or capture every machine-readable `route_key` that the router may
select, including logits shaped `[B*(K+1),V]` and heterogeneous active-row
combinations for temperature/top-k/top-p. One versioned registry defines router,
workspace, and warm/capture keys, and tests enforce:

```text
router_admitted_keys <= workspace_certified_keys
router_admitted_keys <= warm_capture_keys
```

Pretouch runs only after production dtype and dispatch state are restored. It
snapshots and restores CPU and CUDA RNG state, and is part of the existing
transactional runner lifecycle.

After initialization, the first eligible speculative cycle for a declared route
must not cause a new Dynamo/Inductor compile, graph break, guard-triggered
recompile, or CUDA graph capture. Certification uses a unique, initially empty
`TORCHINDUCTOR_CACHE_DIR` for every subprocess. The compile-completeness gate
disables remote compiler caches; all other runs record their cache policy. It
records compile/unique-graph counters plus recompile and graph-break logs before
and after the first cycle; recompile-only logging is not sufficient. Repeat the
cycle to distinguish cold-path work from steady state.
Cold construction/warmup latency is reported separately; it is never hidden
inside warmed throughput. With speculation disabled, none of these new routes is
loaded, warmed, captured, or allowed to consume RNG.

Per-request generators can later improve batch-composition reproducibility, but
would be a separate public reproducibility feature with its own compatibility
tests.

### F18 — Bonus token

Options: omit it permanently; implement it after a no-bonus milestone; include
it from the first end-to-end commit.

Verdict: the complete v1 includes the bonus. The target verifier already computes
`p_(K+1)`. On full acceptance, emitting its sample improves expected tokens per
target call and makes target cache coverage align naturally with all verifier
inputs. A no-bonus path is acceptable only as an intermediate correctness rung
and must not be used for final performance claims.

If an honored EOS (`ignore_eos=false`) or the length budget ends the request
within accepted drafts, do not emit a bonus. With `ignore_eos=true`, EOS alone
does not suppress the bonus or finish the request.

### F19 — Atomic multi-token commit

For rejection after `A` accepted drafts (`A<K`):

```text
emit d1..dA, then one residual correction
advance target and draft logical caches by A+1 valid inputs:
  last, d1, ..., dA
retain no logical cache for the correction token itself
```

Although drafting may have physically written later candidates, both logical
coverages stop after `last,d1..dA`; the rejected suffix is stale and may not be
attended to. The correction remains the new unprocessed tail token.

For full acceptance with bonus:

```text
emit d1..dK, then bonus
advance target logical cache by K+1 inputs: last, d1, ..., dK
advance draft logical cache by K inputs: last, d1, ..., d(K-1)
retain no logical cache for the bonus itself
```

The `K` sequential draft steps have not processed `dK`: they consume `last` to
produce `d1`, then consume through `d(K-1)` to produce `dK`. Before proposing in
the next cycle, draft catch-up therefore processes both `dK` and the bonus so its
coverage reaches the new committed-prefix boundary.

The scheduler first validates every row: IDs, nonempty commits, acceptance
bounds, cache advances, the exact finish predicate, reservation ownership, and
unique sequence identity. The predicate remains the current nano-vLLM contract:

```text
finished = ((not ignore_eos) and token == eos) or
           (num_completion_tokens >= max_tokens)
```

When `ignore_eos=false`, commit truncates the burst at the first EOS and discards
later staged tokens. When `ignore_eos=true`, EOS is committed and processing
continues through the otherwise valid burst. Only after validating all rows does
the scheduler mutate state. It emits one `StreamOutput` per committed token; only
the last event for a finished row has `finished=True`.

### F20 — Prefix hashes and stale KV

Rejected proposals must never affect `hash_to_block_id`. Hash only complete blocks
whose tokens are committed and whose target coverage proves they were processed.
Draft validity metadata, if added, is separate from the target prefix hash.

Stale physical KV is allowed only under an overwrite-before-read invariant with
logical lengths and slot mappings that exclude it. Tests compare the next logits
against full recomputation after every possible acceptance length and around
block boundaries.

### F21 — Streaming and detokenization

`StreamSession` already has a pending deque, and `StreamingDetokenizer.feed`
accepts one token. Preserve that interface: a burst becomes several ordered
events, never a multi-token opaque event.

Closing or abandoning a stream after receiving part of a pending burst cancels
only still-scheduled work. Already committed but undelivered events are discarded
from the local pending deque, just as current `close()` discards pending output;
KV and request cleanup still occur exactly once. No proposal token can enter the
deque.

### F22 — Metrics

Keep current meanings:

- `num_decode_tokens` remains the number of scheduled decode rows.
- existing engine token timestamps are commit timestamps;
- delivery timestamps are taken when `StreamSession.__next__` yields an event.

Add explicit cycle counters:

- committed/emitted tokens;
- proposed and accepted draft tokens;
- target verification positions;
- draft positions;
- rejection position histogram;
- bonus count;
- `spec_residual_numerical_fallbacks`;
- speculative cycles and bypass reasons;
- draft, verify, rejection, commit, and rollback time.

Tokens committed in one cycle may share a compute timestamp, producing zero
intra-burst engine ITLs. That is truthful. Do not invent staggered timestamps.
Progress display may use committed tokens per second, while the legacy `step()`
return retains its existing row-based sign convention.

### F23 — Lifecycle and failure behavior

The current engine has transactional construction, idempotent exit, bounded TP
worker teardown, Python-GC leases, session ownership, and abandoned-stream
finalization. Draft load, cache allocation, graph capture, and pretouch join those
transactions; they do not create parallel cleanup systems. Every new sampler and
verifier warmup is registered in the same phase ledger, restores RNG even on
failure, and releases engine-owned graph/static-buffer/model references during
rollback. Global Dynamo/Inductor caches are not claimed to be per-engine
reclaimable; the required postcondition is a reusable process with no live engine
ownership leak.

Inject failures after each new phase. After failure, a new engine in the same
process must construct and run. Cancellation/finalization after a failed
speculative cycle must release provisional blocks and the session lease.

### F24 — Tensor parallelism

Verdict: reject `K>0` with `tensor_parallel_size>1` before workers spawn.

Current shared-memory compaction understands only `run(seqs,is_prefill)` and
workers receive compact committed sequence DTOs. Correct TP speculation would
need explicit plan/proposal transport and identical target/draft collectives on
every rank. It also requires real multi-GPU tests. Rank-zero-only support is not
support.

### F25 — Performance routing

Options: always speculate; static batch cutoff; analytical cost rule; online
adaptive `K`.

Verdict: never always speculate. Start with hard eligibility, token bounds, and
the complete `W_spec_reservation` workspace bound, collect phase timings and
acceptance, then choose a conservative static low-batch envelope from paired
A100 measurements. Keep baseline fallback cheap and decide it before
speculative allocation.

The decision model is:

```text
benefit = E[committed tokens per cycle] * baseline_decode_time(B,L)
cost = draft_time(B,L,K) + verify_time(B,L,K)
       + rejection_time(B,K,V) + scheduler/cache overhead
speculate only where benefit > cost with a declared safety margin
```

If static routing cannot contain regressions across domains and sampling
settings, reopen the decision for an EWMA acceptance/cost controller. Dynamic
policy comes after, not before, trustworthy counters.

### F26 — Evidence and claims

Every performance artifact records target/draft SHAs, tokenizer fingerprints,
code SHA, hardware, clocks/power policy, CUDA/Torch/FlashAttention versions,
precision, graph/eager mode, prompts, sampling parameters, warmup, random A/B
order, declared warm/capture shape matrix, compile/recompile counters, raw samples,
and failures. Cold initialization and the first eligible cycle are retained as a
separate record from warmed steady-state measurements.

PR7 evidence stays immutable and labeled prototype-only. v2 creates new dated
artifacts. A paper's reported speedup or the old branch's preview is never written
as a v2 result.

## 7. Main difficulties and why they are hard

| Difficulty | Why the obvious shortcut fails | Required proof |
|---|---|---|
| Rejection sampling | Support-mask equality does not establish probability-law equality; finite precision can produce a numerical rejection with zero residual mass | tiny-vocab FP64 oracle, explicit target-p fallback/counter, forced zero/subnormal cases, sequence-law and Monte Carlo tests |
| Verifier logits | current prefill LM head returns only the last row | all-live-row mapping and eager/graph equality |
| KV rollback | GPU writes occur before acceptance is known | logical coverage, reservation lease, next-logit full-recompute oracle |
| Prefix cache | target-valid block does not imply draft-valid block | block hash/validity audit and shared-prefix reuse tests |
| Budgeting | one decode row becomes `K+1` target queries | planner proves token, graph, position, and block bounds |
| Multi-token commit | current postprocess assumes one token/row | validate-all-then-commit transaction with every acceptance length |
| Sampling workspace | retained `q`, draft/verifier logits and activations, materialized `p`, heterogeneous transform scratch, and sampler noise can overlap | separate live-peak/reservation models, reserved headroom, measured reconciliation, and pre-allocation bypass |
| Streaming | one compute cycle yields several delivery events and EOS is conditional on `ignore_eos` | ordering, both `ignore_eos` modes, EOS-once, close/finalizer tests |
| Metrics | emitted tokens differ from physical work | separate immutable counters and clock semantics |
| Lifecycle | a second model doubles fallible resource phases | phase-injected failure and sequential-engine tests |
| Performance | high acceptance does not imply speedup | paired phase timings and workload-specific roofline/crossover |
| TP | every rank must execute the same collectives and cache updates | fail-fast v1; real TP certification later |

## 8. Compatibility matrix for v1

| Feature | Initial status | Behavior |
|---|---|---|
| speculation disabled | supported | exact current baseline path |
| greedy, TP1 | supported | longest target-argmax match plus target correction/bonus |
| temperature/top-k/exact top-p, TP1 | supported after statistical gates | exact modified rejection on the normal path; counted machine-precision recovery is reported and qualifies exactness if observed |
| FlashInfer top-p + speculation | unsupported | construction-time error or baseline-only explicit policy; never silent exact claim |
| chunked prefill | supported by coexistence | mixed steps use baseline; pure-decode steps may speculate |
| prefix cache | supported after cache-validity gates | target reuse plus independent draft catch-up |
| eager mode | correctness-supported | performance not promised |
| CUDA graphs | supported after per-bucket gates | graphed draft; measured verifier routing; no unplanned first-cycle compile/capture |
| streaming/generate/manual step | supported | same public token order and ownership contracts |
| cancellation/abandoned stream | supported | rollback/cleanup at cycle boundaries |
| TP>1 | unsupported | fail before worker/GPU ownership |
| tree/Medusa/EAGLE | out of scope | future proposer/verifier architecture |

## 9. Decisions deliberately left measurement-dependent

The following are not fixed in prose:

- release model pair;
- default maximum `K`;
- maximum speculative batch size;
- verifier eager-versus-ragged-graph crossover;
- new small ragged graph buckets;
- complete FP32 probability/sampler live-peak cap and separate safety margin;
- whether recompute-on-rejection beats retaining all `q_i`;
- whether a static router is sufficient.

Each is chosen only after the probes and paired benchmarks in
`04_implementation_validation_plan.md`. Changing one requires updating the
prediction, raw evidence, cost model, and compatibility statement together.

## 10. Release decision

The feature is a no-go if any of these remains true:

- a transformed-sampling oracle disagrees with the target law;
- a numerical rejection can produce NaN, an arbitrary token, or an unobservable
  residual fallback;
- a nonzero numerical fallback rate is reported under an unqualified exactness
  claim, or `q(d) <= 0` reaches acceptance arithmetic;
- rejected tokens enter sequence state, hashes, output, or metrics;
- target/draft next logits disagree with full recomputation after rollback;
- a failure leaks blocks, GPU state, a process group, or a stream lease;
- spec-off changes baseline output/RNG/capacity/lifecycle behavior;
- unsupported TP or FlashInfer combinations proceed silently;
- the router admits a key absent from the workspace/warm registries, whose
  measured peak exceeds `W_spec_reservation`, or whose first production cycle
  compiles, graph-breaks, recompiles, or captures;
- the intended workload has no statistically credible speedup and routing does
  not bypass the regression;
- evidence cannot be tied to a code SHA and reproducible protocol.

Passing unit tests is necessary but not sufficient. Release requires the
branch ladder, GPU cache/state gates, and A100 cost/crossover report in document
04.
