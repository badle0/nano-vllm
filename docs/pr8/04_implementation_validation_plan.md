# PR8 speculative decoding v2: implementation and validation plan

Status: **V0 frozen at `480a3b2`; V1 sampling-law implementation and CPU
certification are in progress on `feat/spec-v2-sampling-law`**. No dual-model or
engine execution path exists yet.

Base: `origin/fork-main` at `663753b`.

Work branch: `docs/speculative-decoding-v2` for the planning record, followed by
the implementation branches defined below. The existing
`feat/speculative-decoding` tip at `a632b59` is a historical prototype. Its PR7
documents, tests, and measurements are useful design inputs, but **PR7 evidence
is prototype-only and does not certify PR8 code, current fork contracts, current
dependencies, or current performance**.

Throughout this packet, v1/v2 name milestones of this nano-vLLM fork. No API,
kernel, numerical-fallback, or performance contract is inherited from the
separate vLLM project.

The committed PR7 status leaves C6 unchecked and incomplete; its evidence remains
a prototype record rather than PR8 certification.

This document is the execution contract for PR8. A failed hard gate stops the
branch ladder. It is not acceptable to proceed by widening a tolerance, changing
a workload after observing a result, or relabeling old PR7 evidence as a new
measurement.

## 1. Objective and scope

PR8 adds classic draft-model speculative decoding to the maintained fork. The
normal path implements exact modified rejection; the explicitly counted
finite-precision recovery is reported separately and qualifies any exactness
claim if observed. The feature preserves the contracts already present on
`fork-main`:

- typed and transactional request admission;
- bounded scheduler capacity and token budgets;
- decode-first chunked prefill and the single-`mid_chunk_seq` invariant;
- prefix-cache and KV-block ownership correctness;
- exact temperature/top-k/top-p sampling semantics;
- request-owned streaming, targeted cancellation, abandoned-session cleanup,
  and separate compute/delivery timing;
- exception-safe, idempotent engine and model-runner teardown;
- compact tensor-parallel transport, even though PR8 v1 is single-GPU only.

The v1 feature is deliberately narrow:

- a separate draft model;
- one tokenizer/token-ID space shared with the target;
- fixed configured maximum `K`, with a safe per-cycle effective `K`;
- exact modified rejection sampling on the normal path, with a declared and
  counted finite-precision target-distribution recovery;
- the standard one-token target bonus after all `effective_k` drafts are
  accepted;
- speculation on eligible pure-decode work only;
- tensor parallel size one;
- the current exact top-p backend only;
- correctness-supported eager draft decode with no performance promise, plus a
  graphed performance path; target verification remains correctness-first with a
  later measured routing decision.

### Explicit exclusions

The following are not PR8 completion requirements:

- FlashInfer speculative sampling semantics or performance;
- tensor-parallel speculative execution;
- adaptive or per-request configured `K`;
- n-gram, prompt-lookup, Medusa, EAGLE, tree, or self-speculative proposers;
- combining speculative rows and prefill chunks in one unified ragged step;
- asynchronous or concurrent sessions on one `LLMEngine`;
- claiming literal token identity for numerically tied greedy logits produced by
  different kernel compositions;
- preserving PR7 fixed-seed sampled output, CUDA-graph layout, or benchmark
  numbers.

If an excluded feature is needed to make the scoped implementation correct, stop
and revise this contract before writing more code.

## 2. Non-negotiable design contracts

### 2.1 Sampling law

For a draft token `d ~ q` and target distribution `p`, accept `d` with
`min(1, p(d) / q(d))`. On rejection, sample from normalized
`max(0, p - q)`. The emitted token must therefore be distributed as `p` for
any valid `q` in exact arithmetic.

In these formulas, `p` and `q` are the normalized laws induced by the canonical
FP32 weight rows actually used by nano-vLLM's categorical sampler. The FP64
oracle upcasts those retained FP32 rows and their row masses; it must not
recompute softmax from logits in FP64 and thereby define a different law. Before
acceptance, validate both rows as finite and non-negative with positive finite
mass, normalize consistently, and require finite `q(d) > 0`. Any violation
raises the typed speculative-sampling invariant error without caller-visible
mutation.

The implementation must define the finite-precision recovery branch explicitly.
For residual weights `r = max(p - q, 0)` and mass `z = sum(r)`, V1 always
upcasts the retained canonical FP32 rows and their row masses to the FP64
reference path. A later production path may use a faster dtype only after a
separate gate proves categorical-law equivalence for every routed input class;
finite positive fast mass alone is not such a proof because it can still lose
reference-positive support. Zero, subnormal, or non-finite fast mass always
routes to the reference path. Then:

1. finite `z > 0`: sample the robust residual;
2. `z == 0` after a numerical rejection: robustly renormalize `p`, sample it
   with the independent corrective draw, and increment
   `spec_residual_numerical_fallbacks` exactly once;
3. non-finite FP64/reference `z`: raise the typed invariant error and roll back.

Never divide by zero, return an implementation-selected token, or substitute a
uniform distribution. This target-`p` fallback is an observable
finite-precision availability policy, not an analytic residual-law claim. Under
a coupling that agrees off fallback events, its total-variation contribution is
at most `Pr(fallback)`. Certification reports the observed rate and a
preregistered upper confidence bound. An unqualified exact-release claim requires
zero fallback events in all non-injected certification workloads; any nonzero
rate must qualify the result as machine-precision guarded and report the bound.

`p` must be the distribution defined by the current exact sampler after the
same temperature, top-k, and top-p semantics used by ordinary generation. PR8
must factor or expose a canonical current distribution-building operation; it
must not copy the obsolete PR7 `warp_logits` implementation over the current
sampler. In particular, it must retain the FP32 private-workspace fix for top-p
and the current mixed-active-row support and tie behavior.

The FlashInfer backend must fail during typed configuration validation when
speculation is enabled. Silent fallback or approximate residual sampling is not
allowed.

### 2.2 Effective proposal length

The configured `K` is a maximum, not permission to read or write beyond a
request's licensed range. Before each speculative cycle, derive a batch-wide
`effective_k`:

```text
effective_k = min(
    configured_k,
    minimum (remaining completion-token budget - 1) in the selected batch,
    minimum target model-position headroom in the selected batch,
    floor(max_num_batched_tokens / selected batch size) - 1,
)
```

For v1, use one `effective_k` for the whole speculative group. This intentionally
trades a small amount of tail efficiency for simpler shapes, reservations,
acceptance logic, and proofs. `effective_k == 0` routes through ordinary decode.
The two `- 1` terms reserve different required headroom: one completion slot for
the bonus token and one target-work slot because verification feeds the previous
committed token plus `effective_k` drafts. Thus a fully accepted cycle may emit
`effective_k + 1` tokens while its verification row still contains exactly
`effective_k + 1` model inputs. The model-position term bounds those verification
inputs; the bonus itself is the new unprocessed last token and is processed by a
later cycle only if generation continues. EOS remains unknowable before
verification. During commit, stop at the first EOS if and only if
`ignore_eos` is false. With `ignore_eos=true`, EOS is an ordinary committed
token and the token-level finish predicate continues only until `max_tokens`.
Cancellation and errors remain separate lifecycle paths, not token finish
conditions.

### 2.3 Scheduler and work budget

A speculative verification row contains `effective_k + 1` target input tokens.
The scheduler may not charge it as one token while silently exceeding
`max_num_batched_tokens`, graph buffers, or captured-shape limits. A speculative
plan must explicitly record:

- selected sequence IDs;
- `effective_k`;
- target verification-token count;
- draft-step token counts;
- predicted `W_spec_live_peak`, `W_spec_reservation`, and machine-readable
  `route_key`;
- transient KV-block reservations;
- baseline fallback decision and reason.

For correctness-first v1, proposal probabilities are retained as FP32
`q[B,K,V]` and verifier probabilities are materialized as FP32
`p[B,K+1,V]`. The simultaneous probability floor is:

```text
W_probability_floor = sizeof(FP32) * V * (B*K + B*(K+1))
```

The model separates actual transient liveness from reserved safety headroom:

```text
W_spec_live_peak(B,K,V,route_key) = peak_live(
    retained q,
    current draft hidden/logits, probability output, and transform scratch,
    verifier hidden/logits and materialized p,
    correction rows, acceptance data, and categorical RNG/noise,
    exact top-k/top-p sort/cumsum/index/mask workspaces,
    eager or graph-static buffers,
    target/draft activation and library workspaces not counted as persistent,
)
W_spec_reservation = W_spec_live_peak + allocator_safety_margin
```

Persistent weights and KV tensors are modeled separately and counted exactly
once. Buffers may be excluded from the live peak only when an implementation
lifetime proof and measured peak show they do not overlap. Compare incremental
CUDA peak allocated bytes with `W_spec_live_peak`; validate
`W_spec_reservation` against peak reserved bytes and post-init headroom. The plan
must fit the reservation before allocating proposal tensors or provisional
blocks.

`route_key` includes at least the draft batch bucket, verifier ragged
`(token_bucket, slot_tier)`, `effective_k`, eager/graph execution mode, and a
heterogeneous sampler-plan signature. That signature distinguishes active
plain/temperature/top-k/top-p/combined row composition, or names a proved
conservative worst case.

Only eligible pure-decode work may speculate. If the whole selected decode batch
cannot use a positive common `effective_k` or does not fit the configured
verification workspace, v1 routes that whole batch through ordinary decode. It
must not split one scheduled batch into speculative and ordinary subgroups.
Speculative subgrouping or microbatching is a future V7-or-later extension that
requires a measured crossover benefit and separate scheduling gates. The v1
fallback must be deterministic, tested, documented, included in performance
results, and preserve the current decode-first, FIFO, preemption, and
`mid_chunk_seq` contracts.

### 2.4 KV ownership and transactionality

Target and draft models use separate KV tensors with the same logical block IDs.
A target prefix-cache hit is not evidence that the corresponding draft cache is
valid. Draft coverage must be tracked independently and reset whenever block
identity may change, including preemption and deallocation/reuse.

Speculative allocation is a transaction:

1. choose a plan;
2. reserve only the additional blocks required by that plan;
3. propose and verify;
4. commit only accepted/corrective tokens;
5. hash only committed complete blocks;
6. trim unused trailing reservations safely;
7. clear all transient state.

Any exception after step 2 must roll back reservations and transient proposal
state without hiding the primary exception. Rejected drafts must never enter
`Sequence.token_ids`, prefix hashes, metrics, or stream events. Refcount-safe
trimming is required; indefinitely retaining one speculative boundary block per
sequence is not accepted as a capacity tax.

### 2.5 Lifecycle ownership

There remains one `ModelRunner` owner. Draft resources are an owned sub-state of
that runner, not a second independent runner or process group. All draft
attributes must be initialized to a cleanup-safe value before fallible work.
Construction and teardown must integrate with the existing transactional paths:

- worker spawn rollback;
- distributed process-group teardown;
- shared-memory cleanup;
- graph and graph-pool release;
- target and draft KV release;
- target and draft model release;
- context reset and CUDA allocator cleanup;
- Python-GC lease restoration;
- idempotent engine exit and bounded worker joins.

Automatic KV sizing must account for both models, both cache tensors, warmup,
graphs, and the complete `W_spec_reservation` before choosing the block
count. An explicit `num_kvcache_blocks` override must be validated against
dual-cache plus speculative-workspace feasibility and fail with an actionable
typed error. PR7 peak-memory arithmetic is not assumed valid on the current
lifecycle.

### 2.6 Streaming and metrics

Commit emits one `StreamOutput` per committed token, in order. A burst may place
several events in the current `StreamSession` pending queue. Closing or abandoning
a stream while that queue is non-empty must cancel only the session's owned
requests, release its lease, and leave the engine reusable.

Do not silently redefine `StepOutput.num_decode_tokens`. Keep scheduled/model
work and emitted-token accounting distinct. Performance instrumentation must at
least expose:

- baseline decode rows;
- target verification tokens;
- draft tokens evaluated;
- accepted draft tokens;
- emitted tokens;
- numerical residual fallbacks;
- speculative cycles and fallback cycles.

Compute timestamps for tokens in one burst may be equal by design. Delivery
timestamps must continue to describe when the consumer received events.

### 2.7 Warmup and compilation

Every route that the speculative router may select must be ready before the first
request can enter it. A versioned, machine-readable registry defines the router,
workspace-certified, and warm/capture key sets. Tests enforce:

```text
router_admitted_keys <= workspace_certified_keys
router_admitted_keys <= warm_capture_keys
```

Each key records the actual nano-vLLM graph families rather than only a prose
`(B,K)` pair. The declared warm/capture matrix includes, as applicable:

- draft decode for every supported batch/graph bucket;
- all-query verifier hidden/logit outputs, including `[B*(K+1),V]` logits;
- target and draft distribution preparation for plain, temperature, top-k,
  top-p, and combined active-row plans;
- acceptance `[B,K]`, corrective residual `[B,V]`, and bonus `[B,V]` paths;
- each supported eager/graph mode and `(B,K)` bucket family.

Pretouch runs after production dtype, default-device, and dispatch state have
been restored. It saves and restores CPU and CUDA RNG state and participates in
transactional construction/rollback. A route outside this declared matrix must
take an explicitly pre-touched eager path or deterministically fall back to
ordinary decode; it may not cause opportunistic production-time compilation.

In each isolated certification subprocess, use a unique, initially empty
`TORCHINDUCTOR_CACHE_DIR`. The compile-completeness gate disables remote compiler
caches; every other run records its cache policy. After engine initialization,
snapshot compile/unique-graph counters and collect recompile plus graph-break
logs; `TORCH_LOGS=recompiles` alone is insufficient to prove absence of a
first-time compile. The first eligible speculative cycle for every admitted key
must add no compile, graph break, guard miss, recompile, or CUDA graph capture.
Repeat the same cycle to establish steady state. Report
construction/warmup and cold first-cycle latency separately from warmed benchmark
results. With speculation disabled, no speculative load, pretouch, capture,
compilation, or RNG draw is permitted.

## 3. Archive and rollback plan

Before implementation:

1. Fetch and record the exact refs for `origin/fork-main`,
   `feat/speculative-decoding`, and the PR7 base.
2. Leave `feat/speculative-decoding` and its existing worktree untouched.
3. Optionally create an immutable annotated archive tag at `a632b59`, but only
   with explicit repository-owner authorization. That tag preserves committed
   history only; it does not preserve the old worktree's current dirty state.
4. Before any old-worktree or old-branch deletion is considered, separately review
   and preserve the modified `benchmarks/pr7/status.md`, the untracked
   `benchmarks/feat_audit/` tree, and the untracked audit documents
   `docs/feat_audit_benchmark_and_fix_plan_2026-08-17.md` and
   `docs/nano-vllm_feat_audit_2026-08-14.md`. Use either an owner-approved WIP
   archive commit on a dedicated archive ref or a checksummed external archive;
   do not silently fold these files into immutable PR7 claims. Verify recovery by
   reconstructing the archive in a separate location, checking every manifest
   hash, and confirming the retained patch and file inventory before treating the
   worktree as disposable.
5. If PR7 documents or artifacts are copied into PR8, preserve their bytes and
   add a manifest containing source path, source commit or explicit dirty-worktree
   provenance, file SHA256, environment, date, and the statement that C6 was
   incomplete.
6. Store all PR8 results under a new dated directory with a manifest tied to the
   exact tested PR8 commit. Never overwrite PR7 JSON or text files.

Each implementation rung is a separate commit or short-lived branch. A red hard
gate stops later rungs. Roll back by abandoning or reverting only the failing
rung; do not rewrite the archive branch, amend retained evidence, or reset the
shared worktree. Speculation remains opt-in and defaults off throughout, so the
release branch always has a baseline escape hatch.

Recommended branch ladder from `origin/fork-main@663753b`:

| Rung | Suggested branch | Purpose |
|---|---|---|
| V0 | `docs/speculative-decoding-v2` | Frozen plan and PR7 provenance only |
| V1 | `feat/spec-v2-sampling-law` | CPU oracle plus current exact-sampler distribution seam |
| V2 | `feat/spec-v2-dual-runner` | Inert typed config and transactional draft lifecycle/KV sizing |
| V3 | `feat/spec-v2-draft-path` | Draft catch-up and compute-then-discard |
| V4 | `feat/spec-v2-scheduler-plan` | Effective-K planning and transactional reservation |
| V5 | `feat/spec-v2-verify-commit` | Target verification, rejection sampling, burst commit |
| V6 | `feat/spec-v2-stream-lifecycle` | Streaming, metrics, cancel/finalizer, failure certification |
| V7 | `feat/spec-v2-performance` | Verification routing, roofline and benchmark certification |
| RC | `release/speculative-decoding-v2-rc1` | Squashed policy only if repository convention requires it; evidence ancestry retained |

V1 starts only from the merged V0 documentation commit, or from its exact reviewed
SHA if merge timing requires it. Each V(n+1) starts from the latest green, merged
predecessor and opens against `fork-main`; this avoids a long cross-base PR stack
and makes every gate independently reviewable. Never base v2 implementation code
on, merge, or cherry-pick the old `feat/speculative-decoding` branch wholesale.

## 4. Implementation ladder and hard gates

### V0: plan and provenance

Changes:

- add PR8 design and validation documents;
- add a PR7 provenance manifest only if historical files are copied;
- record exclusions, known prototype limitations, and fresh evidence paths.

Hard gates:

- diff contains documentation/provenance only;
- every copied artifact hashes to its PR7 source;
- text states that implementation has not begun and PR7 is not certification;
- base is exactly `origin/fork-main@663753b`.

### V1: sampling law and current sampler seam

Changes:

- add a pure/reference modified-rejection implementation for tests;
- expose target/draft probabilities through a canonical exact sampling seam;
- implement vectorized acceptance with injectable random inputs;
- implement an explicit residual outcome carrying the sampled token, whether the
  numerical target-`p` fallback ran, and a typed error for invalid target rows;
- preserve the existing ordinary sampler's fixed-seed behavior when speculation
  is disabled.

Hard gates:

- analytic identity tests for residual normalization and acceptance probability;
- the FP64 scalar oracle upcasts the exact retained canonical FP32 weight rows,
  normalizes their row masses, and never recomputes a different softmax law from
  logits;
- that oracle checks production residual branch classification and sampled
  support on exact-zero, all-subnormal, and near-equal fixtures: a fast-path
  zero/non-finite mass that becomes finite-positive in FP64 uses the residual;
  robust zero uses target-`p` fallback; robust non-finite mass raises the typed
  invariant error;
- a valid FP32 softmax fixture for which the fast/raw `p(d)/q(d) < u` and
  `max(p-q,0)` appears to have zero mass is upcast and row-normalized by the FP64
  oracle; its positive robust residual is sampled, it never divides by zero, and
  `spec_residual_numerical_fallbacks` remains zero;
- the target-`p` fallback uses the injected independent corrective draw and is
  deterministic under controlled draws; invalid `p`/`q`, inconsistent row
  normalization, and selected `q(d) <= 0` or non-finite metadata raise the typed
  invariant error and mutate no caller-visible state;
- scaling raw canonical weight rows by arbitrary positive row constants does not
  change the normalized law, acceptance result, or corrective distribution;
- CPU Monte Carlo emitted-law tests for plain temperature, top-k, top-p, and
  combined warps, with tolerances registered before running;
- zero emitted probability outside the target support;
- deliberately mismatched `q` still emits according to `p`;
- `p == q` accepts all with controlled identical inputs;
- the normally reachable `p == q` path records zero numerical fallbacks, while a
  direct lower-level hook injects rejection only to exercise the analytically
  unreachable robust-zero branch; that injected branch samples target `p` and
  increments `spec_residual_numerical_fallbacks` exactly once;
- all non-injected certification workloads record zero numerical fallbacks for
  an unqualified exactness claim; otherwise evidence reports the observed rate,
  an upper confidence bound on `Pr(fallback)`, and the corresponding TV bound;
- greedy limit equals target argmax;
- vectorized acceptance equals a per-element oracle on adversarial rows;
- FP32 input top-p mutation regression remains green;
- all existing sampler tests, including FlashInfer exclusion behavior, pass;
- speculation-off fixed-seed output is unchanged from V0.

V1 fallback-rate certificate preregistration: use the four fixed Monte Carlo
cases in `tests/test_speculative_sampler.py` (temperature, top-k, top-p, and
combined), each with 100,000 one-token speculative cycles. The event is
`target_fallback == True` per cycle; the deliberately injected robust-zero
fallback fixture is excluded. Passing requires zero events in all `n = 400,000`
cycles. For zero events, report the one-sided 95% exact binomial
(Clopper-Pearson) upper bound
`1 - 0.05 ** (1 / n) = 7.489302638941098e-6`; the coupling argument gives the
same upper bound on the fallback contribution to total variation. Any event is a
failed V1 gate and forbids an unqualified exactness claim rather than triggering
a post-hoc tolerance change.

### V2: inert dual-runner lifecycle

Changes:

- add validated draft-model configuration and `configured_k`;
- reject TP greater than one, FlashInfer, and invalid paths/types/ranges before
  worker creation; explicitly validate and support both eager correctness mode and
  the graphed performance mode;
- load, warm, size, allocate, bind, graph, and release draft resources inside the
  current runner transaction;
- reserve the configured maximum `W_spec_reservation` before automatic KV block sizing
  and validate the same headroom for explicit block-count overrides;
- keep `configured_k == 0` fully inert.

Hard gates:

- speculation-off output and scheduler traces match V0;
- constructor-failure injection after every new phase leaves no workers, shared
  memory, process group, graphs, KV tensors, model references, context, or GC
  lease behind;
- `exit()` is idempotent after success and after partial construction;
- two sequential engines in one process pass on GPU;
- a retained healthy engine remains usable after a second engine's constructor
  fails;
- automatic and explicit dual-KV sizing boundaries produce typed errors rather
  than CUDA OOM or `AssertionError` where validation is possible;
- failure during any draft pretouch/compile/capture phase restores CPU/CUDA RNG
  and leaves the process able to construct another engine;
- current lifecycle, config, model-runner, GC, and TP transport tests remain
  green;
- an A100 memory audit records weights, warmup peak, graph allocation, per-block
  target/draft bytes, `W_spec_live_peak`, allocator margin,
  `W_spec_reservation`, selected block count, and post-init headroom; incremental
  allocated bytes reconcile with live peak and reserved/headroom bytes reconcile
  with the reservation without double-counting non-overlapping buffers.

### V3: draft path, compute then discard

Changes:

- track draft-cache coverage independently;
- catch up the draft cache for cold, prefix-hit, mixed-history, and preempted
  sequences;
- run `effective_k` draft steps through the selected eager or graphed path and
  retain tokens/probabilities only as
  ephemeral cycle state;
- pretouch every draft `route_key` that discard-mode routing may select,
  after restoring production dtype/dispatch state and without advancing RNG;
- discard every proposal and execute ordinary target decode.

Hard gates:

- speculation enabled in discard mode produces the same baseline output and
  stream events under deterministic, non-tied fixtures;
- draft logits/tokens are nontrivial and acceptance-rate instrumentation is sane;
- boundary cases around block positions 255/256/257 and multi-block K pass;
- cold versus shared-prefix versus block-reuse draft cache results agree within
  the registered numerical contract;
- preemption clears/rebuilds draft coverage correctly;
- injected draft catch-up/proposal failures clear ephemeral state and leave the
  engine reusable or cleanly closable;
- with the empty-cache protocol from §2.7, the first eligible discard-mode cycle
  for every admitted draft key produces no new compile, unique graph, graph
  break, recompile, guard miss, or graph capture after initialization;
- no PR8 field is added to the TP DTO unless a worker actually needs it;
- all current chunked-prefill, prefix-cache, scheduler-cancel, streaming, and
  lifecycle tests pass with speculation off and relevant tests with discard mode
  on.

### V4: scheduler plan and reservations

Changes:

- introduce an explicit speculative step plan;
- calculate batch-wide `effective_k` from completion and model-position limits;
- reserve completion and target-work headroom for the full-acceptance bonus;
- enforce the configured verification-token budget;
- compute and record `W_spec_live_peak` and `W_spec_reservation` for the exact
  machine-readable `route_key`, and reject the
  speculative plan before tensor/block allocation when reserved headroom is
  insufficient;
- reserve and trim additional blocks transactionally;
- define deterministic whole-selected-batch baseline fallback; explicitly defer
  speculative subgrouping and microbatching to a measured V7-or-later extension.

Hard gates:

- property sweep over prompt length, `max_tokens`, configured K, block size,
  model limit, pool size, batch size, and mixed waiting/running queues;
- no planned target position exceeds `max_model_len - 1`;
- a rejection cycle emits at most `effective_k` accepted/corrective tokens and a
  full-acceptance cycle emits exactly `effective_k + 1`, without exceeding the
  request's remaining completion budget;
- plan work never exceeds `max_num_batched_tokens` or allocated input buffers;
- the maximum eligible workspace case completes without OOM, while the first
  one-above-cap case routes the whole selected batch through baseline before any
  speculative allocation and preserves scheduler order/RNG contracts;
- adversarial heterogeneous active-row compositions cover the largest exact
  top-k/top-p index/select/copy/sort workspace, and every router-admitted key is
  present in the workspace-certified registry;
- failed reservations restore exact free-block counts/refcounts/tables;
- partial rejection and tail cycles trim unused trailing blocks;
- the current `mid_chunk_seq`, decode-first, FIFO, bounded-capacity, preemption,
  and targeted-cancel suites stay green;
- repeated speculative boundary cycles do not leak capacity;
- speculation-off scheduler traces remain unchanged.

### V5: verify, accept, and commit

Changes:

- execute one correctness-first target verification over staged proposals;
- construct canonical FP32 target probabilities `p[B,K+1,V]` alongside retained
  draft probabilities `q[B,K,V]`, with actual tensor lifetimes matching the
  registered live-peak/reservation model;
- apply vectorized modified rejection sampling;
- pretouch/capture the all-query verifier, probability preparation, acceptance,
  residual-correction, and bonus paths for every `route_key` V5 may select;
- commit accepted prefix plus one corrective token on rejection;
- first gate an isolated no-bonus intermediate commit if useful, then sample and
  commit the standard target bonus when every draft is accepted; the no-bonus
  checkpoint is not a complete v1;
- append, hash, account, trim, and clear state in the proved order.

Hard gates:

- deterministic greedy fixtures match baseline exactly when target margins are
  unambiguous;
- real-model greedy divergences are independently adjudicated; any mismatch not
  attributable to a registered numerical tie is a stop;
- EOS at every accepted, corrective, and bonus position is crossed with
  `ignore_eos=false/true`: false truncates at the first EOS, discards later
  staged tokens, and emits one terminal event; true commits through EOS and
  finishes at `max_tokens` under the token-level predicate;
- EOS appearing only in a rejected proposal or rejected suffix never finishes or
  deallocates the request, never suppresses or replaces the valid correction,
  and cannot affect later bonus eligibility, under either `ignore_eos` value;
- `max_tokens` at every position in a burst truncates correctly;
- full acceptance emits all `effective_k` drafts plus exactly one target bonus;
- bonus sampling uses the target distribution after the final accepted draft,
  including temperature/top-k/top-p, and the bonus becomes the sole unprocessed
  last token when the request continues;
- tail cases with one remaining completion slot route through baseline decode;
  tail cases with `R > 1` satisfy `effective_k <= R - 1` and never drop or
  over-emit the bonus;
- no rejected token appears in committed text, hashes, events, or metrics;
- numerical residual fallback is finite, target-supported, independently drawn,
  counted once, and transactionally clean in an end-to-end fixture that uses the
  explicit forced-rejection injection hook rather than claiming a naturally
  valid normalized `p`/`q` row reached robust-zero mass after rejection;
- measured incremental peak allocation for the maximum supported key reconciles
  with `W_spec_live_peak`, peak reservation/headroom reconciles with
  `W_spec_reservation`, and one-above-cap routing reaches baseline without a
  first-use allocator spike;
- with a unique empty Inductor cache and recorded/disabled remote caches, the
  first eligible cycle for every admitted verifier/sampler key adds no compile,
  unique graph, graph break, recompile, guard miss, or graph capture after
  initialization; router keys are a tested subset of warm/capture keys;
- live block tables, target/draft coverage, refcounts, and prefix hashes agree
  with a cold committed-text oracle after every cycle;
- failures in verify, acceptance, append, hash, trim, and event assembly exercise
  rollback and preserve the primary exception;
- statistical end-to-end tests cover temperature, top-k, top-p, and combined
  modes without requiring byte equality between sampled runs;
- speculation-off results remain unchanged.

### V6: stream, metrics, cancellation, and failure certification

Changes:

- expose explicit speculative work/emission counters;
- expose `spec_residual_numerical_fallbacks` without changing existing counter
  meanings;
- pass burst events through the current request-owned `StreamSession`;
- preserve compute and delivery timing semantics;
- complete cleanup integration for close, abandonment, cancellation, and exit.

Hard gates:

- `stream()` token IDs/text equal `generate()` for the same deterministic,
  non-tied execution path;
- one event is delivered for every committed token, in sequence order, with one
  terminal event per finished request;
- close before the first step, between cycles, midway through a pending burst,
  and after the terminal event releases only owned requests and the session lease;
- abandoned-session finalizer tests cover an empty and non-empty pending burst;
- foreign-session ownership checks remain intact;
- TTFT, compute ITL, end-to-end latency, first delivery, and final delivery use
  their documented clocks; intra-burst equal compute timestamps are accepted and
  explicitly tested;
- manual `step()`, `step_with_metrics()`, `generate()`, and `stream()` retain
  their public shapes except for an explicitly documented additive counter;
- the fallback counter is zero on ordinary fixtures, exactly one on an injected
  fallback, and unchanged when speculation is disabled;
- any observed non-injected fallback rate and its upper confidence/TV bound are
  present in metrics and release claims;
- all admission, streaming, metrics, GC, lifecycle, cancel, and TP transport tests
  pass under failure injection and ordinary execution.

### V7: performance routing and fresh certification

No performance optimization begins until V1-V6 hard gates are green.

Changes:

- measure eager verification and existing/small varlen graph buckets;
- choose routing from registered crossover criteria;
- eliminate avoidable per-draft-step host synchronization if correctness gates
  remain green;
- optionally auto-fallback to baseline above a measured batch/cost threshold;
- measure construction/warmup and cold first-cycle latency separately from
  warmed steady state, and retain compile/recompile/guard-miss evidence;
- add reproducible evidence scripts, manifests, and validators.

Hard gates:

- the routing policy was registered before headline runs;
- all results are fresh on the exact V7 commit and validation environment;
- quiet-host repetition and paired ordering requirements below are satisfied;
- baseline with speculation disabled shows no material regression beyond the
  registered noise band;
- speculative latency improves in at least one declared supported workload, or
  the feature is honestly shipped as experimental without a default performance
  claim;
- high-batch regressions are bounded by deterministic fallback or documented as
  an opt-in limitation;
- every key admitted by the final router is present in the versioned workspace
  and warm/capture registries, and its first eligible production cycle is free of
  compile, unique-graph creation, graph break, recompile, and graph capture;
- predicted and measured `alpha`, `c`, `E[N]`, steps/token, and speedup reconcile
  within a pre-registered error band or receive a mechanism-based explanation;
- retained evidence validation passes from a clean checkout.

## 5. Test matrix

All unit/property tests run CPU-only where possible. GPU tests use one A100 and
must run in isolated processes when allocator, process-group, graph, or lifecycle
state is under test.

| Area | Required cases | Primary oracle |
|---|---|---|
| Config | types, ranges, missing model, K 0/1/max/out-of-range, TP>1, exact/FlashInfer, eager and graph modes | valid eager/graph construction; typed invalid-config exception before worker spawn |
| Sampling math | greedy, temperature, top-k, top-p, combined, mismatched q, zero support, q(d)=0/non-finite, row scaling/normalization, boundary ties, zero/subnormal residual, fast-vs-reference non-finite mass, explicitly injected analytically unreachable rejection | retained-FP32-to-FP64 scalar oracle, robust residual recovery, explicit target-p fallback/error/counter, and Monte Carlo |
| Admission | empty/invalid tokens, P/N/K near model limit, KV pool boundary, full scheduler | current typed admission plus effective-K property oracle |
| Scheduling | waiting/running/mid-chunk, pure/mixed, preemption, fallback, cancellation | invariant checker and trace model |
| KV blocks | 255/256/257, rejection lengths 0..K, full accept, preemption, shared prefix, recycled block | committed-token cold reconstruction, refcount/free-count audit |
| Draft cache | cold, hit, partial coverage, mixed baseline history, preempt/re-admit | eager teacher-forced draft output |
| Verification | eager first, every declared K/batch bucket, all-live-row shape, model tail, first eligible cycle after initialization | sequential target distribution oracle plus compile/recompile/graph-capture ledger |
| Workspace | maximum eligible and first ineligible route key, heterogeneous plain/top-k/top-p/combined active rows, auto/explicit KV blocks | live-peak plus safety-reservation model, CUDA allocated/reserved peaks, pre-allocation fallback trace |
| Commit | rejection at each position, EOS at accepted/corrective/bonus positions with both `ignore_eos` values, max_tokens each position, exception each phase | committed-prefix state machine |
| Streaming | close/GC at every lifecycle point, slow/null consumer, pending burst | same-run committed events and ownership audit |
| Metrics | first token, intra/inter-burst ITL, finish/delivery, cancellation | injected clock timeline |
| Lifecycle | every constructor phase, repeated exit, sequential engines, retained healthy engine, worker failure | process/CUDA/context/GC resource audit |
| Warmup | every admitted draft/verifier/sampler key, registry-subset check, pretouch failure, speculation off, unique empty cache, cold then repeated cycle | RNG snapshots, compile/unique-graph counters, recompile/graph-break logs, and CUDA graph ledger |
| Compatibility | speculation off, manual step, request metrics, chunked prefill, exact sampler, TP transport | current fork-main regression suites |

Fuzz/property sweeps must print their seed on failure and retain the smallest
reproducer. Tests must not depend on global `Sequence` IDs beginning at zero.

## 6. Roofline and benchmark plan

### 6.1 Quantities and model

For each model pair and batch, measure rather than infer:

- target baseline step time `t_target(B, context)`;
- draft graph replay time `t_draft(B, context)` and ratio
  `c = t_draft / t_target`;
- eager and graphed target verification time for
  `T_verify = B * (effective_k + 1)`;
- mean per-position acceptance `alpha` and full acceptance-length distribution;
- emitted tokens per cycle `E[N]`;
- host preparation, graph replay, GPU execution, sampler/acceptance, commit, and
  synchronization time;
- target/draft weight bytes, live KV bytes, measured memory bandwidth, and peak
  allocated/reserved memory.

Compare the simple prediction

```text
speedup ~= E[N] / (K * c + t_verify / t_target + overhead / t_target)
```

with measured TPOT and throughput. Report arithmetic intensity and whether each
regime is weight-bandwidth-, KV-bandwidth-, compute-, padding-, or host-dispatch-
limited. Do not reuse PR7's A100 bandwidth or timing constants without measuring
them on the certification host.

### 6.2 Functional benchmark matrix

Models, subject to availability and memory audit:

- target Qwen3-4B with draft Qwen3-0.6B: required;
- target Qwen3-8B with draft Qwen3-0.6B: optional secondary evidence, not a v1
  completion blocker;
- target-only baseline for every target/prompt/sampling cell.

Sweep:

| Dimension | Values |
|---|---|
| Batch | `1, 2, 4, 8, 16, 32, 64, 128` where memory permits |
| Configured K | `1, 2, 3, 4, 5, 6` |
| Context length | approximately `32, 256, 1024, 2048`, capped safely by model/pool |
| Completion | `1`, `<K`, `K`, `K+1`, `64`, `256` where valid |
| Sampling | greedy; temperature 0.8; top-k 50; top-p 0.95; combined |
| Workload | chat/prose, code, repetitive/formulaic, adversarial low-acceptance |
| Mode | baseline; spec eager-verify; spec graph-route; automatic policy |
| Cache | cold; repeated-prefix hit; shared-prefix batch |
| Consumer | generate; null stream; deliberately slow stream |

At least three independent seeds are required for sampled-law/performance cells.
Headline latency cells require at least five paired repetitions after warmup.
Alternate mode order (`ABBA` or randomized balanced order), record load average,
GPU clocks/power state, temperature, memory occupancy, package versions, model
hashes, and exact commands. Reject runs with concurrent GPU processes or declared
host-noise limits exceeded; retain rejected runs with the rejection reason.

Warmup is a declared protocol, not an unspecified number of discarded samples.
The archive records the versioned route-key registry exercised during engine
construction, RNG state before/after pretouch, and compile/unique-graph counters
plus recompile/graph-break/CUDA-graph logs after initialization. A separate suite
uses a unique empty Inductor cache per subprocess, records remote-cache policy,
and measures construction plus warmup and the first eligible speculative cycle.
Only after that cycle is proved compile-free may steady-state samples be reported
as warmed results.

### 6.3 Primary reported metrics

- caller-visible TTFT and p50/p95/p99 TPOT;
- output tokens/s and requests/s;
- engine steps per emitted token;
- cycle latency and phase decomposition;
- acceptance by position and workload;
- `E[N]` predicted versus observed;
- baseline fallback rate and reason;
- `spec_residual_numerical_fallbacks`, its non-injected event rate, upper
  confidence bound, TV bound, and exactness-claim qualification;
- modeled `W_probability_floor`, `W_spec_live_peak`, allocator margin, and
  `W_spec_reservation` versus measured peak allocated/reserved GPU memory, plus
  available KV-token capacity;
- construction/warmup latency, cold first-cycle latency, compile/unique-graph
  counts, recompile/graph-break logs, and graph captures, reported separately
  from steady state;
- prefix-hit versus cold numerical and performance delta;
- speculation-off overhead with confidence interval;
- batch crossover where speculation ceases to improve TPOT or throughput.

The release note must distinguish latency wins from throughput wins. It must not
present a low-batch TPOT improvement as a general throughput improvement.

## 7. Evidence format and reproducibility

Every certification archive contains:

- exact repository commit and dirty-state check;
- branch and base commit;
- model paths plus config/tokenizer/weight fingerprints;
- Python, PyTorch, CUDA, driver, GPU, attention backend, and optional dependency
  versions;
- command line, environment variables, seed, warmup, versioned route-key
  registry, repetition, and ordering;
- the versioned tensor-liveness workspace model, allocator safety margin,
  predicted and observed peaks, and every memory-based bypass reason;
- unique empty Inductor-cache path, remote-cache policy, compile/unique-graph
  counters, recompile/graph-break/guard-miss logs, and CUDA graph-capture evidence
  for cold and repeated cycles;
- raw per-request/per-step samples, not only aggregates;
- aggregate script version and deterministic re-aggregation command;
- host/GPU health metadata and rejected-run ledger;
- expected schema and a validator that fails on missing or non-finite fields;
- SHA256 manifest over scripts and raw results.

Validation must work in a clean checkout without a GPU for schema, provenance,
hash, and aggregation checks. GPU reproduction commands are documented separately.

## 8. Completion and release criteria

PR8 is implementation-complete only when all of the following are true:

1. V1-V6 hard gates pass on the exact proposed code.
2. The complete current CPU test suite passes with speculation disabled.
3. Required isolated A100 lifecycle, graph, KV, prefix-cache, and end-to-end tests
   pass with speculation enabled.
4. No known path exceeds model position, completion, scheduler-token,
   graph-buffer, `W_spec_live_peak`, `W_spec_reservation`, or KV-pool bounds; the
   first ineligible workspace key falls back before speculative allocation.
5. Every speculative reservation and every draft resource has a tested rollback
   and idempotent teardown path.
6. Output under the current exact sampling backend passes independent law tests,
   including the
   standard full-acceptance bonus from the target distribution; every fully
   accepted cycle emits exactly `effective_k + 1` tokens within request and model
   bounds. Empty/subnormal residual recovery is finite, target-supported, counted,
   and transactionally safe. All non-injected workloads have zero fallbacks for
   an unqualified exactness claim; otherwise the claim and TV bound are qualified.
   FlashInfer and TP>1 fail early with documented typed errors.
7. Streaming close, cancellation, abandoned-session finalization, and delivery
   metrics remain correct for bursts, including EOS at every burst position with
   both values of `ignore_eos`.
8. Speculation-off behavior and performance remain within registered compatibility
   bands. Every admitted key is workspace-certified and pre-touched/captured,
   with no first-cycle compile, unique graph, graph break, recompile, or capture
   under the empty-cache protocol.
9. V7 produces fresh, validated evidence and states the measured crossover and
   credible limitations.
10. Documentation describes opt-in configuration, model/tokenizer compatibility,
    capacity effects, effective-K/fallback behavior, metrics semantics, unsupported
    modes, and cleanup guarantees.
11. A reviewer can reconstruct every headline number from retained raw evidence.
12. The PR7 archive remains unchanged and is clearly labeled prototype-only.

Release is a **no-go** if there is an unattributed greedy mismatch, statistical-law
failure, invalid `q(d)` acceptance, NaN/arbitrary-token/unobserved numerical
recovery, a nonzero fallback rate under an unqualified exactness claim, sampler
reservation overrun, an admitted key missing from the workspace/warm registries,
first-cycle compile/graph-break/recompile/capture under the empty-cache protocol,
block/refcount leak, model-limit overrun, scheduler invariant failure,
constructor/teardown leak, stream ownership failure, silent unsupported-backend
fallback, or unvalidated performance claim. If correctness is complete but no
workload demonstrates a credible speedup, the code may be retained behind an
experimental opt-in flag, but it must not be marketed as a performance release.

## 9. Required final report

The final PR8 report must include:

- a commit-by-commit gate table for V0-V7;
- all deviations from this plan and the alternatives considered;
- hits and misses against preregistered predictions;
- the complete correctness and failure-injection matrix;
- the measured roofline classification and batch crossover;
- speculation-off regression results;
- supported and excluded configurations;
- numerical residual-fallback counts and adversarial recovery results;
- predicted versus measured sampler-workspace peaks and memory-based bypasses;
- the versioned router/workspace/warm key registry plus empty-cache cold-cycle
  compile/unique-graph/recompile/graph-break/capture evidence;
- unresolved limitations and follow-up work;
- archive locations and validation commands.

Until that report and the criteria above are complete, PR8 remains a redesign in
progress, not a certified replacement for the historical PR7 prototype.
