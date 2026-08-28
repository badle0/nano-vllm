# PR 8 — Current code map and speculative-decoding landing points

Base: `origin/fork-main` at `663753b99131945c297c1fbe02341108f422dce7`.

This document describes the code that exists at that base, not the historical
PR7 implementation. PR7 remains useful for the rejection-sampling theory and
for identifying likely performance regimes, but its source map predates robust
admission, capacity-bounded sessions, streaming cleanup, request metrics,
compact tensor-parallel transport, the exact/FlashInfer top-p split, and ragged
CUDA graphs. Every design conclusion below is therefore derived from the
current source.

## 1. One current request, from construction to cleanup

### 1.1 Public construction and ownership

`nanovllm/__init__.py` exports `LLM`, `SamplingParams`, `StreamOutput`,
`StreamSession`, `SchedulerCapacityError`, and the streaming detokenizer types.
`nanovllm/llm.py::LLM` is a pass-through subclass of
`nanovllm/engine/llm_engine.py::LLMEngine`.

`LLMEngine.__init__` performs the complete construction transaction:

1. It builds `Config` from recognized keyword fields.
2. It verifies the optional FlashInfer sampler before taking GPU/process
   ownership.
3. It loads the tokenizer before starting tensor-parallel workers.
4. It sets `Sequence.block_size`, spawns nonzero TP ranks, and constructs rank
   zero's `ModelRunner`.
5. It snapshots immutable admission bounds after KV sizing writes the final
   block count into the config.
6. It constructs `Scheduler`, registers teardown, and optionally acquires the
   process-global Python-GC lease.
7. Any failure unwinds workers, runner resources, Torch defaults, process
   groups, block-size state, and the GC lease without replacing the primary
   exception.

`ModelRunner.__init__` currently initializes one target model. It sets the CUDA
device and NCCL process group, constructs `Qwen3ForCausalLM`, loads weights,
constructs `Sampler`, warms the model/sampling kernels, sizes and binds one KV
cache, captures ordinary decode and ragged graphs when enabled, restores Torch
defaults, pre-touches eager prefill, and finally starts the TP worker loop.

Speculative decoding must extend this same transaction. A second independent
`ModelRunner` is not viable: it would attempt to own another process group and
would split scheduling, block ownership, and cleanup. The draft model, draft KV
tensor, draft graphs, and speculative sampler state belong inside the existing
runner and must participate in the same failure rollback.

### 1.2 Admission

`LLMEngine.generate`, `stream`, and `add_request` converge on:

```text
_normalize_batch
  -> _admit_batch (batch transaction)
       -> _admit_request
            -> Scheduler.require_capacity
            -> exact-base SamplingParams snapshot and validation
            -> tokenize or copy explicit token IDs
            -> token-ID validation
            -> model-length and KV-pool validation
            -> Sequence(...)
            -> Scheduler.add
```

The current processed-context bound is exact:

```text
processed_tokens = prompt_length + max_tokens - 1
```

The final generated token is returned but is not itself processed by the model,
so admission must remain `P + N - 1`, not `P + N`. Speculation changes how many
positions are processed per engine cycle, not the maximum logical committed
context required by the request. Temporary verification positions and their KV
slots are scheduler reservations.

When speculation is enabled, admission additionally needs a construction-time
snapshot of the effective target/draft positional limit and a verified shared
token-ID space. Vocabulary-size equality or a single canary encoding is not
sufficient. A safe initial contract requires matching tokenizer artifacts, or
at least matching tokenizer class/config, complete vocabulary and added-token
maps, normalization/pre-tokenization configuration, and every special-token ID.

### 1.3 Scheduling

`LLMEngine._step` currently calls `Scheduler.schedule()`, counts scheduled
prefill tokens and decode rows, calls `model_runner.call("run", seqs,
is_prefill)`, and passes one returned token per row to
`Scheduler.postprocess`.

`Scheduler.schedule` is decode-first:

1. Each running sequence is removed in FIFO order.
2. `BlockManager.can_append` is checked; tail sequences are preempted until the
   next KV slot can be reserved.
3. The sequence is marked as a one-token decode and
   `BlockManager.may_append` reserves a boundary block if needed.
4. Decode rows each charge one unit of `max_num_batched_tokens`.
5. Remaining capacity is filled with FIFO chunked-prefill work. At most one
   waiting sequence is left mid-chunk.
6. The return boolean means “use the ragged path because this step contains
   prefill,” not simply “all rows are prefill.”

That pair-valued return is too weak for speculation. Introduce an explicit,
pickle-safe step plan with a mode such as `DECODE`, `RAGGED`, or `SPECULATIVE`,
row-level effective proposal widths, and the resource reservations used to
construct it. Do not further overload `is_prefill`.

The first implementation should speculate only on a pure-decode cycle. This
preserves chunked-prefill behavior structurally, but eligibility must be decided
before KV reservation. The scheduler cannot reserve one ordinary decode slot
and then silently upgrade the batch to a `K`-position verifier.

The existing budget also needs new accounting. A batch of `B` ordinary decode
rows charges `B`, whereas target verification processes `B*K` positions without
a bonus or `B*(K+1)` positions with one. Those positions may exceed
`max_num_batched_tokens`, ragged graph capacity, or static buffers even though
`B <= max_num_seqs`. The scheduler must reduce row-level `K`, select only a
subset of speculative rows, or explicitly microbatch verification. Effective
`K` is also capped by each request's remaining completion tokens and remaining
model positions.

### 1.4 Sequence and KV state

`Sequence` currently owns only committed state: `token_ids`, `last_token`,
prompt/completion counts, target cache/schedule counters, one block table,
sampling fields, status, and timestamps. `append_token` is the single committed
mutation primitive.

Draft proposals must not be appended to `token_ids`. Add explicitly staged
cycle state or, preferably, carry it in the runner plan/result DTO. A running
sequence also needs independently defined target and draft logical cache
coverage. Physical target and draft caches may share block IDs, but validity in
one cache does not imply validity in the other.

`BlockManager` currently provides:

- `can_allocate`/`allocate` for prompt blocks and target prefix-cache hits;
- `can_append`/`may_append` for one decode write;
- `deallocate` for all blocks owned by a sequence;
- `hash_blocks` for newly valid full committed blocks.

Speculation needs an exact multi-position reservation primitive. For a committed
prefix ending in `x`:

- no-bonus verification feeds `[x, d1, ..., d(K-1)]` and produces target
  distributions `p1 ... pK`;
- bonus verification feeds `[x, d1, ..., dK]` and additionally produces
  `p(K+1)`.

Reservation is based on the highest KV position written by that verifier, not
merely the number of tokens it may emit. At a rejection, verifier slots after
the accepted prefix contain stale draft-conditioned KV. They may remain
physically dirty only under a strict overwrite-before-read rule; logical target
coverage must be reduced to the committed boundary. Draft coverage must likewise
reflect only context that is valid for the next proposal cycle.

Only committed, logically valid full blocks may enter `hash_to_block_id`.
Rejected proposal tokens must never be prefix-hashed. Preemption, cancellation,
and failed-cycle rollback must clear both logical coverages and release a shared
physical reservation exactly once. A target prefix-cache hit does not create a
draft prefix-cache hit; v1 should catch the draft model up from committed tokens
unless a separately certified draft prefix cache is added.

### 1.5 Runner preparation, attention, and graphs

`ModelRunner.call` broadcasts rank-zero calls through shared memory.
`compact_run_args` converts owning `Sequence` objects into pickle-safe
`ScheduledSequence` values for workers. `ScheduledSequence` currently carries
only a committed scheduled token slice, offsets, the last token, and one block
table. Speculative TP therefore requires a new transport DTO containing the
step mode, staged tokens, per-row widths, both cache coverages, and reservation
metadata. If this transport is not implemented in v1, config must reject
speculation with `tensor_parallel_size > 1`; partial rank-zero support would
deadlock or desynchronize collectives.

For normal execution:

- `prepare_decode` supplies one `last_token` per row, its position, the target
  slot mapping, context lengths, and padded block tables.
- `prepare_ragged` supplies the scheduled committed slice for each row and
  constructs `cu_seqlens_q`, `cu_seqlens_k`, positions, slot mapping, and block
  tables.
- `set_context` publishes those tensors globally to attention and the LM head.
- `run_model` selects eager decode, fixed-batch decode graphs, eager ragged, or
  captured ragged graphs.
- `Attention.forward` first writes K/V to `context.slot_mapping`, then calls
  paged FlashAttention in ragged mode or the paged decode kernel in decode mode.

Verification is structurally ragged, but `prepare_ragged` cannot be reused
unchanged because its inputs come only from committed sequence slices. Add a
verifier preparation path accepting staged input IDs and variable row widths,
while retaining the same carefully checked positions, causal boundaries, block
tables, and slot mappings.

There is a critical output-selection issue in
`layers/embed_head.py::ParallelLMHead.forward`: whenever
`context.is_prefill` is true, it gathers only
`context.cu_seqlens_q[1:] - 1`, producing one last-row logit vector per
sequence. A verifier needs every target distribution corresponding to every
proposal position, plus the final distribution when producing a bonus token.
Globally removing this gather would regress ordinary prefill memory and behavior.
Add an explicit context output-selection mode (for example `LAST_PER_SEQUENCE`
versus `ALL_QUERY_ROWS`) or gather verifier-selected hidden-state rows before
the LM head. Do not overload the `is_prefill` flag again.

Draft generation consists of `K` sequential one-token model calls. Those calls
must use captured draft decode graphs on the intended performance hosts. Target
and draft graph capture should be parameterized rather than copied, and cleanup
must release graph objects before their static buffers, caches, and models.
Verifier execution can first use eager ragged execution for correctness, then
use measured ragged graph buckets. Its graph key and buffers must account for
total verification positions, not just number of sequences.

KV allocation currently computes one target block's byte cost after warmup and
uses remaining memory to choose `num_kvcache_blocks`. With a draft LM, warm up
both models first and size a block using:

```text
target KV bytes per block + draft KV bytes per block
```

Allocate parallel target/draft tensors with the same block count and block-ID
space, then bind each model's attention layers to the appropriate cache. A
positive explicit `num_kvcache_blocks` remains an exact override and must be
validated against the combined allocation.

### 1.6 Sampling and verification result

`ModelRunner.prepare_sample` currently builds heterogeneous row plans for
temperature, top-k, and top-p. `run` computes one target logit row per sequence,
then uses:

- `Sampler.greedy` for an all-greedy batch;
- `filter_top_k` for active top-k buckets;
- exact `filter_top_p` followed by `Sampler.forward`; or
- `sample_top_p_flashinfer` for the statistical FlashInfer backend.

Speculative sampling must prepare the same effective distributions defined by
the public sampling contract:

```text
temperature -> top-k mask -> top-p mask -> normalize
```

For proposal `d ~ q`, exact sampled verification accepts with
`min(1, p(d)/q(d))`; on rejection it samples from normalized
`max(p-q, 0)`. The full target and draft distributions at the rejection
position are required. Storing only `q(d)` cannot implement the residual.
Extract one reusable warp/distribution preparation path so ordinary and
speculative sampling cannot drift on top-k boundary ties, top-p support, or
temperature scaling.

Greedy mode should remain a separate exact path: accept proposal tokens while
they match target argmax; at the first mismatch emit target argmax. It is the
token-exact end-to-end correctness gate. Positive-temperature equivalence is
equality in distribution, not same-seed byte equality.

The current FlashInfer path owns different Philox draws and boundary-tie
semantics and exposes a sampled token rather than the complete warped
distribution required by residual correction. Because engine configuration
cannot know whether future requests will be greedy or sampled, the v1
construction-time contract rejects **any** enabled speculation with
`top_p_backend="flashinfer"`. Supporting greedy-only speculation in an engine
configured for FlashInfer is a possible later relaxation, after the API has an
explicit way to guarantee that every request remains greedy; full sampled
support still requires a distribution-consistent path and separate statistical
certification.

Runner execution should be pure with respect to `Sequence` and return a typed
result, conceptually:

```text
SpecResult:
  commits: list[list[int]]
  accepted_counts: list[int]
  proposed_counts: list[int]
  target_valid_through: list[int]
  draft_valid_through: list[int]
  target_positions_evaluated: int
  draft_positions_evaluated: int
```

This keeps committed mutation, EOS handling, prefix hashing, and deallocation
inside the scheduler rather than splitting ownership between GPU code and host
postprocessing.

### 1.7 Commit, streaming, generate, and metrics

`Scheduler.postprocess` currently advances the scheduled cache count, skips
mid-prefill emission, appends exactly one token, records one engine timestamp,
checks EOS/max tokens, deallocates finished sequences, and emits one
`StreamOutput`.

Generalize it transactionally to a commit list per row:

1. Validate result shape, row ownership, token types/ranges, cache boundaries,
   and nonempty speculative commits before mutating any sequence.
2. Advance only certified logical target/draft cache coverage.
3. Append committed tokens in order through `Sequence.append_token`.
4. Stop at the request's remaining `max_tokens`.
5. When `ignore_eos` is false, stop at the first EOS and discard every later
   token in the burst.
6. Emit one `StreamOutput` per committed token, marking only the final emitted
   event as finished.
7. Prefix-hash only committed full blocks and deallocate a finished sequence
   once.

`StreamSession.__next__` already owns a pending event deque, so a burst of
one-token `StreamOutput` values naturally preserves order and synchronous
backpressure. `StreamingDetokenizer.feed` remains unchanged because it still
receives individual committed token IDs. `generate` likewise continues to
construct final text from `Sequence.completion_token_ids`.

Metrics need explicit new counters. `StepOutput.num_decode_tokens` currently
means number of decode rows, which no longer represents either compute or
emission. Record at least prompt positions, ordinary target positions, target
verification positions, draft positions, committed tokens, and speculative
cycles. Existing per-token timestamps should remain honest: tokens committed in
one burst share the engine timestamp, yielding zero intra-burst ITL and a larger
inter-cycle gap. Add cycle metrics rather than inventing staggered times.
Streaming delivery timestamps remain per `__next__` call and therefore retain
their current caller-boundary meaning.

### 1.8 Cancellation and cleanup

`StreamSession.close` and its abandoned-session finalizer cancel only owned
request IDs before releasing the exclusive engine lease. `generate` uses a
`finally` block to cancel admitted work and preserve primary failures.
`Scheduler.cancel` knows that a mid-chunk waiting sequence can own KV blocks.

Speculative state must follow the same ownership rules. Cancellation or a runner
exception must not leave staged proposals, transient block reservations, draft
coverage, or graph-held buffers reachable. `ModelRunner._close` must clear draft
attention cache views, release target and draft graph objects before shared
buffers, delete both KV tensors and both models, reset global context, destroy
the process group, collect Python objects, and empty CUDA cache. Construction
failure and repeated/concurrent `LLMEngine.exit` must remain idempotent.

## 2. Core invariants

These invariants are the redesign's correctness contract:

1. **Committed-state invariant:** `Sequence.token_ids` contains prompt tokens
   plus emitted tokens only; no unaccepted proposal is ever visible there.
2. **Admission invariant:** a request is admissible only if its maximum logical
   target context satisfies `P + N - 1 <= effective_max_model_len` and its
   per-request target/draft cache demand can fit the configured pool.
3. **Target-cache invariant:** logical target coverage names only positions
   whose KV corresponds to the committed prefix used by the next target call.
4. **Draft-cache invariant:** draft coverage is tracked independently and names
   only positions valid for the next proposal call.
5. **Overwrite invariant:** stale KV from rejected proposals is never read. Every
   stale position is overwritten before it can enter an attention context.
6. **Hash invariant:** only committed, fully valid target blocks are entered in
   the prefix hash map.
7. **Reservation invariant:** before GPU execution, the block table covers every
   slot that target or draft execution will write; a cycle never allocates after
   launching kernels.
8. **Commit invariant:** every successful speculative row commits at least one
   token, no row exceeds remaining `max_tokens`, and no token follows a terminal
   EOS when EOS is honored.
9. **Distribution invariant:** greedy mode is token-equivalent to ordinary target
   decoding; sampled mode emits the target's fully warped distribution in law.
10. **Ownership invariant:** runner execution returns data but never commits
    sequence state; scheduler postprocessing is the single commit authority.
11. **Batch-transaction invariant:** malformed runner output or any failure before
    commit leaves every sequence and allocator in its pre-cycle logical state.
12. **TP invariant:** all ranks execute the identical target/draft collective
    sequence, or speculative mode is rejected at config time.

## 3. Spec-off compatibility

`num_speculative_tokens=0` must be an inert default:

- no draft tokenizer, config, weights, cache, graphs, or sampler buffers load;
- `Config` positional field order remains compatible by appending new fields;
- `Scheduler.schedule` returns the same ordinary plans and preserves decode-first
  chunked-prefill ordering;
- `BlockManager.can_append`/`may_append`, normal `ModelRunner.run`, sampling RNG
  consumption, event ordering, metrics, and cleanup retain their current paths;
- no additional global RNG draw occurs during initialization or inference;
- public return schemas remain unchanged;
- the existing full CPU suite and eager/graph GPU parity tests remain green;
- fixed-seed greedy and sampled outputs on the existing engine are byte-identical
  to base, because the off path executes the same code, not a generalized path
  that merely intends to be equivalent.

New abstractions may wrap the current path only if tests demonstrate this strict
off behavior. Performance measurements must also report spec-off initialization,
prefill, decode, streaming, and chunked-prefill deltas.

## 4. Change matrix

| File | Current responsibility | Required speculative-decoding change |
|---|---|---|
| `nanovllm/config.py` | Validates engine/model/KV/sampler options and loads target HF config | Append draft path and K; load/validate draft config only when enabled; enforce effective positional/tokenizer/TP/backend contracts with exceptions |
| `nanovllm/sampling_params.py` | Defines and validates public sampling semantics | Prefer no schema change; ensure speculative distribution preparation consumes the exact existing snapshot |
| `nanovllm/engine/llm_engine.py` | Construction transaction, admission, sessions, stepping, generate/stream, cleanup | Validate tokenizer identity before GPU ownership; snapshot draft limits; replace boolean runner contract with typed plan/result; expand step counters; preserve admission/session rollback |
| `nanovllm/engine/sequence.py` | Owns committed tokens, cache counters, status, sampling fields, timestamps | Add explicitly defined draft logical coverage if sequence-owned; keep proposals outside committed IDs; retain `append_token` as sole committed mutation |
| `nanovllm/engine/scheduler.py` | Decode-first/chunked-prefill planning, preemption, commit, cancel | Produce explicit step plans; choose pure-decode speculative rows and effective K; reserve transient KV; transactionally commit bursts; truncate at EOS/max tokens; roll back/cancel both cache states |
| `nanovllm/engine/block_manager.py` | Prefix reuse, physical block allocation, one-token append, committed-block hashing | Add exact multi-position reservation/release helpers; separate logical validity from physical dirty slots; never hash rejected proposals |
| `nanovllm/engine/tp_transport.py` | Compact pickle-safe worker input for `run(seqs, is_prefill)` | Add plan/spec row DTOs and staged-token/cache metadata, or reject spec with TP>1 |
| `nanovllm/engine/model_runner.py` | Model/KV ownership, input preparation, graphs, sampling, TP dispatch, cleanup | Load/warm/capture target+draft; jointly size parallel KV; add draft decode and target verify paths; return typed pure results; extend graph buffers, counters, transport, and cleanup |
| `nanovllm/utils/context.py` | Global attention/LM-head execution metadata | Add explicit execution/output-selection mode and verifier metadata; avoid overloading `is_prefill` |
| `nanovllm/layers/attention.py` | Writes paged KV and dispatches ragged/decode FlashAttention | Reuse ragged causal execution with certified verifier slot mappings and cache binding; no rejected slot may be read |
| `nanovllm/layers/embed_head.py` | Computes distributed logits; gathers one last row per prefill sequence | Add verifier `ALL_QUERY_ROWS` or selected-row mode while preserving ordinary last-row prefill behavior |
| `nanovllm/models/qwen3.py` | Target/draft transformer and LM-head composition | Usually no semantic change beyond exposing clear hidden-state/logit selection seams used by runner |
| `nanovllm/layers/sampler.py` | Greedy, exact top-k/top-p, FlashInfer top-p, ordinary sampling | Extract shared warp/probability preparation; add exact acceptance/residual sampler and controlled RNG; gate unsupported FlashInfer sampled speculation |
| `nanovllm/metrics.py` | Computes queue, TTFT, ITL, E2E, and delivery metrics from sequence timestamps | Preserve per-token timing; add cycle/accepted/proposed/target-position/draft-position counters through engine result data |
| `nanovllm/utils/streaming_detokenizer.py` | Converts individual token events into incremental text | No algorithmic change if postprocess continues emitting one ordered event per committed token |
| `tests/test_config.py` and admission tests | Validate construction and request boundaries | Add K/draft/tokenizer/backend/TP validation, exact `P+N-1` boundaries, effective draft limit, and spec-off construction tests |
| `tests/test_scheduler.py` | Validates decode-first scheduling, chunking, preemption, blocks | Add variable-K budgeting, block-boundary reservations, rejection stale-slot invariants, rollback, EOS/max-token burst truncation, preemption and cancel tests |
| sampler tests | Validate top-k/top-p/greedy semantics | Add CPU reference rejection tests, support invariants, p=q accept-all, greedy limit, statistical-law tests, precision/zero-residual cases |
| graph/runner/streaming/metrics/lifecycle tests | Validate execution parity and ownership | Add eager/graph verify parity, LM-head all-row shape, stream burst ordering, abandoned session cleanup, constructor failure injection, repeated engine construction, and spec-off parity |

## 5. Recommended landing sequence

1. Add validated config/tokenizer compatibility and inert dual-model plumbing.
2. Add joint target/draft KV sizing, binding, draft graphs, and complete cleanup.
3. Implement the sampler mathematics against an independent CPU oracle.
4. Introduce typed plans/results and multi-position KV reservation without
   changing emitted output.
5. Run draft proposals and discard them; prove spec-off and allocator invariants.
6. Add eager target verification and transactional greedy commits without a
   bonus token.
7. Certify greedy token equality, block boundaries, preemption, cancellation,
   EOS/max-token truncation, and eager/graph parity.
8. Add exact-backend sampled rejection and statistical certification.
9. Add bonus-token output with its additional verifier row and cache-state tests.
10. Optimize verifier graphs and microbatch policy from measured acceptance,
    draft/target cost, memory use, and batch crossover.
11. Treat TP support as a separately gated extension rather than an implicit
    consequence of rank-zero success.

The difficult part is not the rejection formula. The highest-risk integration
points are variable-width scheduler/KV reservation, target and draft logical
coverage after a rejection, the current LM head's last-row-only prefill gather,
and preserving transactional multi-token commit behavior through streaming,
metrics, cancellation, and teardown.
