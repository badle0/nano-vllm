# PR8 V2: inert dual-model lifecycle

## Status and claim boundary

V2 adds an opt-in draft model as an inert, transactionally owned sub-state of
the existing `ModelRunner`. It validates the target/draft pair, loads and warms
both models, jointly sizes and allocates separate target and draft KV caches,
captures draft decode graphs when graphs are enabled, and releases all owned
resources through the existing engine transaction.

V2 does **not** execute the draft model during request generation. Generation
continues through the ordinary target-only scheduler and runner path, and the
V2 lifecycle gate requires zero draft forward calls during that workload.
There is no proposal, target verification, acceptance/rejection, bonus-token,
multi-token commit, speculative streaming, or speculative-metrics path in this
rung. Consequently, V2 makes no acceptance-rate, latency, throughput, roofline,
or speedup claim.

The implementation commit is the commit containing this document. At the time
that commit is created, no pre-commit A100 output is retention-eligible: a dirty
or uncommitted tree cannot be tied to the exact source snapshot required by the
evidence protocol. Retained V2 evidence is separate post-commit work. Until that
work is run from a clean checkout and independently validated, V2's status is
**implemented but not retained-certified**.

This document is an implementation delta to the frozen source map and design in
documents 02 through 04. It does not modify the V1 sampling-law certificate in
document 05 and does not relabel PR7 evidence as PR8 evidence.

## 1. Exact V2 scope

### 1.1 Public configuration

`Config` appends these caller-facing fields after all pre-existing positional
fields:

```python
draft_model: str | os.PathLike | None = None
num_speculative_tokens: int = 0
```

`configured_k` is a read-only property that returns
`num_speculative_tokens`; it is not a separate constructor keyword.
`speculation_enabled` is true exactly when `num_speculative_tokens > 0`.

The pair is strict:

- `num_speculative_tokens == 0` requires `draft_model is None`;
- `num_speculative_tokens > 0` requires a draft-model path; and
- `num_speculative_tokens` must be an integer and must not be negative.

When speculation is enabled, configuration also requires:

- `tensor_parallel_size == 1`;
- `top_p_backend == "exact"`;
- target and draft model directories that each contain at least one real
  `*.safetensors` file;
- Qwen3 target and draft Hugging Face configurations;
- equal positive target and draft vocabulary sizes; and
- at least one globally usable proposal slot after applying target/draft
  position limits and the engine token budget.

The effective engine model limit is clamped to the minimum of the configured
limit and both models' positive `max_position_embeddings` values. A configured
K larger than the usable global maximum remains valid because K is a maximum
that later cycles may clip. Configuration rejects only the globally impossible
case where either the clamped model limit or `max_num_batched_tokens` leaves no
positive proposal depth.

These checks run before tokenizer construction, worker creation, process-group
ownership, or CUDA model allocation. The Hugging Face configuration files and
safetensors directory inventories are therefore part of the CPU-side preflight.

### 1.2 Token-ID-space identity

`LLMEngine` loads the target tokenizer and, only when speculation is enabled,
the draft tokenizer before it creates a runner or starts workers. Both must be
fast tokenizers with serializable backends. V2 fingerprints and compares:

- the implementation class;
- the complete vocabulary;
- the added-token vocabulary;
- every declared special-token ID;
- the complete special-token ID list;
- the extended special-token map;
- tokenizer initialization configuration after removing only path-location
  fields; and
- the serialized normalizer, pre-tokenizer, decoder, and backend document.

All token IDs must be integers in the model vocabulary, and EOS must be present
in the tokenizer's complete special-ID list. A missing, malformed, unsupported,
or unequal component fails closed before GPU/process ownership. On success,
`LLMEngine.speculative_tokenizer_fingerprint` records the shared identity.

Equal vocabulary size alone, a tokenizer class name, or a canary encoding is not
accepted as proof of one token-ID space.

### 1.3 Fail-closed weight loading

V2 hardens the common safetensors loader because partially initialized draft or
target models make lifecycle and memory evidence meaningless. The loader:

- discovers sorted, real `*.safetensors` files without glob interpretation;
- rejects missing, unreadable, malformed, or duplicate checkpoint tensors;
- rejects unknown destination names and ambiguous packed mappings;
- uses exact dotted path segments for packed QKV and gate/up mappings;
- requires every packed shard and every exact parameter view;
- requires exact full checkpoint geometry before TP slicing and exact local
  destination geometry after TP/packed slicing, so `Tensor.copy_` broadcasting
  or silently ignored oversized tails cannot initialize a parameter;
- permits one checkpoint name to cover exact tied aliases;
- rejects contradictory values supplied for tied aliases; and
- preserves `MemoryError` and CUDA out-of-memory exceptions instead of
  disguising them as checkpoint-format errors.

All other checkpoint/load failures use `ModelWeightLoadError` with the original
exception chained where applicable.

This is a deliberate baseline-visible hardening. Successful speculation-off
model execution remains on the established target path, but malformed or
incomplete speculation-off checkpoints can now fail earlier and with stricter
diagnostics than at the frozen V0 base.

## 2. One transactional resource owner

V2 does not instantiate a second `ModelRunner` and does not create a second
process group. One runner owns two model states. Every draft-owned attribute is
initialized to a cleanup-safe sentinel before the first fallible draft action.

The speculation-enabled construction sequence is:

1. establish the ordinary CUDA device and TP1 process group;
2. construct and load the target Qwen3 model and target sampler;
3. construct the draft Qwen3 model under its configured dtype;
4. load the draft weights;
5. run ordinary target warmup;
6. run an inert draft prefill/logit warmup;
7. in graph mode, temporarily profile target fixed-decode graphs, target ragged
   graphs, and draft fixed-decode graphs, then release those profile owners;
8. build the deterministic speculative-workspace plan;
9. jointly size and allocate the target and draft KV tensors;
10. bind each model's attention layers to its own physical KV tensor;
11. in graph mode, capture the final target fixed/ragged and draft fixed-decode
    graphs against the final caches;
12. restore Torch defaults;
13. in graph mode, pretouch the established target and draft eager-prefill
    paths; and
14. finalize the immutable construction-time memory audit.

Draft phases save and restore CPU and CUDA RNG state on success and failure.
Draft construction also restores the target/default dtype even if construction
raises. This V2 guarantee concerns initialization neutrality; it is not a claim
about future request-time speculative RNG behavior.

The dedicated `capture_draft_cudagraph` helper deliberately owns independent
draft buffers, graph objects, and a draft graph pool while using the target
decode graph's established batch tiers. It duplicates a small amount of target
capture structure instead of immediately generalizing the certified target
path. That is an intentional V2 compatibility choice and reviewable technical
debt for a later rung, not evidence that target verification graphs exist.

### 2.1 Cleanup order and recovery

Construction failure enters the same runner cleanup transaction as ordinary
engine failure. Cleanup:

1. closes and, when owned, unlinks shared memory;
2. resets the global model-execution context and synchronizes initialized CUDA;
3. resets draft, ragged, and target graph executables;
4. deletes graph dictionaries, pools, and static buffers;
5. clears attention-layer cache views from both models;
6. deletes both KV tensors, the sampler, both models, and speculative audit
   state;
7. clears the process-global RoPE module cache;
8. destroys the owned process group;
9. collects Python objects; and
10. empties the CUDA allocator cache.

`exit()` remains idempotent. The RoPE-cache clear is another deliberate
baseline-visible teardown hardening: it releases the cached CUDA buffer so a
later engine in the same process does not inherit an engine-owned module.

Failure injection is defined after each real V2 phase has completed, maximizing
the state that cleanup must unwind. The isolated GPU matrix contains:

```text
draft_construct
draft_load
draft_warmup
graph_profile
joint_allocate
draft_graph
draft_pretouch
memory_finalize
```

`draft_graph` is invoked once by disposable graph profiling and again for final
capture. Its registered injection occurs after the second call so the final
runtime graph owners, rather than only the profiling owners, are tested.

Each GPU phase runs in a fresh Python process. After the injected failure, that
same process must construct, run, and idempotently close a healthy speculative
engine. This proves same-process recovery for the selected phase; it does not
claim that compiler/runtime global caches are per-engine reclaimable. The
recovery protocol permits at most its registered post-execution ceilings of
32 MiB allocated and 64 MiB reserved above the fresh-process baseline for such
process-global compiled-runtime caches.

## 3. Joint KV and speculative-workspace policy

### 3.1 Configured maximum geometry

Let `V` be the shared vocabulary size. The pure planner derives:

```text
K_max = min(
    configured_k,
    max_num_batched_tokens - 1,
    max_model_len - 1,
)

B_max = min(
    max_num_seqs,
    floor(max_num_batched_tokens / (K_max + 1)),
)
```

`B_max` is a fixed maximum batch licensed by this configured plan for every
future effective K that it covers. Clipping K later does not license a larger B
without a new plan and reservation. The plan prices `B_max*K_max` retained
draft-probability rows and `B_max*(K_max+1)` verifier-probability rows.

The mandatory simultaneous FP32 probability floor is:

```text
W_probability_floor = 4 * V * (
    B_max*K_max + B_max*(K_max+1)
)
```

The complete modeled live peak also prices:

- draft and verifier logits in their model dtypes;
- canonical FP32 transform buffers;
- worst-case exact top-k selection outputs and indices;
- chunked exact top-p sort/cumsum/index/mask payloads plus a provisional opaque
  sort-scratch payload;
- FP32/FP64 categorical-race workspaces;
- the FP64 rejection-correction reference workspace; and
- proposal, sampler-routing, acceptance, and result metadata.

Tensor-lifetime phases that cannot overlap are combined with `max`, not summed.
The allocator margin is the larger of 64 MiB and ten percent of the modeled
live peak, rounded upward to a 2 MiB boundary:

```text
W_spec_reservation = W_spec_modeled_live_peak + allocator_margin
```

This plan allocates no proposal or verifier workspace in V2. It reserves a
future configured-maximum envelope so joint KV sizing does not consume memory
already licensed to later speculative work.

### 3.2 Parallel physical caches

Target and draft share a logical block count and block-ID/slot geometry, but
their KV values and tensors are physically separate. For each model:

```text
KV_block_bytes =
    2
    * num_hidden_layers
    * block_size
    * local_num_key_value_heads
    * head_dim
    * dtype_itemsize

joint_block_bytes = target_KV_block_bytes + draft_KV_block_bytes
```

The target and draft caches may therefore have different layer, head, head-dim,
and dtype geometry while retaining the same logical block count. V2 verifies
that the number of attention cache views bound for each model matches its
configuration. The two tensors must not alias.

V2 does not yet define logical draft-cache coverage for live requests. Its draft
KV tensor is owned and sized, but no request-time draft catch-up or proposal path
uses it. Independent coverage and rollback begin in V3 and V4.

### 3.3 Construction and runtime envelopes

Warmup executes target and draft phases sequentially, so their transient peaks
are combined by maximum. In graph mode, V2 first performs disposable graph
capture to measure persistent graph ownership and the graph-capture high-water
mark. It then releases those owners before allocating the final KV caches.

The provisional graph construction cushion is:

```text
graph_allocator_margin = max(64 MiB, joint_block_bytes)

graph_construction_reservation =
    profiled_graph_construction_peak + graph_allocator_margin
```

In eager mode, both quantities are zero.

The future runtime envelope is:

```text
runtime_reservation =
    profiled_graph_ownership
    + max(target_warmup_transient, draft_warmup_transient)
    + W_spec_reservation
```

Final graph capture and speculative request execution are mutually exclusive,
so automatic and explicit KV capacity checks reserve the larger envelope rather
than incorrectly summing both:

```text
sizing_overhead = max(
    graph_construction_reservation,
    runtime_reservation,
)

memory_budget = floor(total_device_memory * gpu_memory_utilization)
usable_for_joint_kv = memory_budget - used_before_kv - sizing_overhead
automatic_num_blocks = floor(usable_for_joint_kv / joint_block_bytes)
```

A non-positive automatic block count, or a positive explicit block count above
that boundary, raises `SpeculativeKVCacheCapacityError` before the dual
`torch.empty` allocation. A positive explicit count is exact; it is not silently
reduced or expanded.

After final graph capture, V2 records raw post-graph/pre-pretouch allocated and
reserved endpoints separately from any allocations retained by eager-prefill
pretouch. It also records the final capture high-water deltas relative to the
post-KV baseline. Construction fails if the final observed graph-capture peak
exceeds the profiled graph construction reservation.

Both pretouches synchronize before returning. At audit finalization V2 releases
only their now-unused caching-allocator segments with `torch.cuda.empty_cache()`
before querying driver free memory; otherwise recyclable pretouch cache would be
counted as permanent ownership while the future transient was reserved again.
Live weights, KV caches, and graph private pools remain owned. Post-init
device-budget headroom must then still cover the modeled runtime transient plus
`W_spec_reservation`. The audit records physical target/draft weight storage
without double-counting tied parameter views, both KV tensor sizes, graph
ownership and capture peaks, warmup transients, selected blocks, allocator
state, and independent arithmetic-reconciliation inputs.

### 3.4 Why `gpu_certified` remains false

The planner intentionally leaves these fields unknown rather than reporting a
false zero:

```text
graph_static_workspace_bytes = None
backend_library_workspace_bytes = None
```

The audit also names components requiring execution-time measurement:

- model activation and attention-library workspace;
- route-specific CUDA graph/static buffers;
- backend-internal top-k/top-p selection or sort scratch beyond the documented
  payload proxy; and
- allocator fragmentation beyond the fixed margin.

Accordingly, every V2 `SpeculativeMemoryAudit` has
`gpu_certified == false`. V2 certifies deterministic planner arithmetic,
construction-time graph/KV ownership, typed capacity boundaries, and retained
headroom for the modeled workspace. It does **not** certify measured runtime
`W_spec_live_peak`, because no V2 request allocates or executes that workspace.
Route-specific allocated/reserved peak reconciliation belongs to the rungs that
implement draft proposals and target verification, and ultimately to V7's A100
performance certificate.

An A100 construction audit does not change that claim boundary. It supplies
measured construction inputs and verifies the provisional reservation on that
run; it cannot turn an unexecuted request-time route into a measured route.

## 4. Speculation-off and inert-on compatibility

With both new options at their defaults, no draft tokenizer, draft Hugging Face
configuration, draft model, draft cache, draft graphs, or speculative-memory
audit is created. Scheduler, block-manager, request admission, ordinary runner
execution, streaming, metrics, and public result schemas remain unchanged.

V2 compares three distinct things rather than treating one as proof of all:

1. a canonical greedy golden from detached V0;
2. the current V2 tree with speculation disabled; and
3. the current V2 tree with draft ownership enabled but request-time drafting
   proven inert.

For the registered greedy fixture, outputs and scheduler/runner traces must
match, and construction plus generation RNG snapshots must match between the
current off/on cases. These checks do not establish sampled end-to-end parity,
performance parity, or every possible scheduler workload. The full existing
regression suite remains a separate compatibility gate.

Two intentional baseline-visible changes must not be hidden by the term
"inert":

- fail-closed safetensors validation changes malformed/incomplete checkpoint
  failure behavior; and
- model-runner close clears the global RoPE cache to support clean sequential
  engine construction.

Neither change enables speculative token execution.

## 5. CPU validation protocol

The pull-request CPU job runs on Python 3.10 and 3.12 with CPU PyTorch 2.4.1.
Its V2 step hides CUDA and Hugging Face network access and imports a narrow
CI-only `Qwen3ForCausalLM` placeholder through
`.github/ci/speculative_v2/sitecustomize.py`:

```bash
CUDA_VISIBLE_DEVICES='' \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=.github/ci/speculative_v2:. \
  /venv/main/bin/pytest -q -p no:cacheprovider \
  tests/test_config.py \
  tests/test_gc_lifecycle.py \
  tests/test_tokenizer_identity.py \
  tests/test_loader.py \
  tests/test_speculative_memory.py \
  tests/test_llm_engine.py \
  tests/test_model_runner.py \
  tests/test_speculative_v0_golden.py \
  tests/test_speculative_v2_lifecycle_harness.py \
  tests/test_speculative_v2_gpu_recovery_harness.py
```

The placeholder must fail if an unmocked test attempts real Qwen construction.
This job certifies CPU validation, planner, provenance, writer, and mocked
lifecycle contracts. It does not certify real Qwen kernels, CUDA graphs, NCCL,
physical GPU memory, or GPU cleanup.

The implementation commit must also run the repository-wide regression and
static gates in the actual project environment:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  /venv/main/bin/pytest -q -p no:cacheprovider

/venv/main/bin/python -m compileall -q nanovllm benchmarks tests
git diff --check
```

This document intentionally records no result count, duration, artifact hash, or
PASS statement before those commands finish on the exact implementation tree.

## 6. A100 lifecycle evidence protocol

### 6.1 Exploratory versus retained mode

`tests/run_speculative_v2_lifecycle.py` supports an exploratory dirty-tree mode
for debugging, but exploratory output is never retention-eligible and must not
be relabeled later. Retained mode requires:

- a clean source tree at an exact full 40-character implementation SHA;
- no `--allow-dirty` flag;
- a write-once output path outside the source and both model directories;
- the explicit KV-boundary/recovery gate;
- a SHA-pinned canonical V0 golden generated from detached commit
  `480a3b26c5a4e465aac06d1dabd34e1230686feb` by the same committed V2 runner;
- unchanged source and target/draft model manifests across the run; and
- the registered A100 and endpoint-isolation requirements below.

Output is written atomically without replacing an existing file. The writer
publishes only after all second provenance checks pass and prints the artifact's
SHA256. A printed hash is not, by itself, an archive validator or release
certificate.

### 6.2 Registered hardware and isolation

Retained V2 lifecycle and recovery evidence requires:

- `NVIDIA A100-SXM4-40GB`;
- compute capability 8.0;
- at least 39 GiB reported device memory;
- a canonical non-MIG GPU UUID;
- matching Torch and `nvidia-smi` device identity; and
- no foreign compute applications on the selected GPU at the before and after
  endpoints.

The runner establishes a CUDA context and binds its container/host namespace PID
before admitting that process as the sole owned GPU consumer. Endpoint checks
cannot prove that a transient foreign process did not appear in the middle of a
run. The evidence records that limitation and makes no stronger isolation claim.

### 6.3 Canonical V0 golden

After committing V2, create a separate detached checkout of canonical V0. The
runner script itself remains in the clean V2 implementation checkout, while
`PYTHONPATH` points to V0 so the imported engine is historical code:

```bash
SPEC_V2_COMMIT="$(git rev-parse HEAD)"
SPEC_V0_COMMIT="480a3b26c5a4e465aac06d1dabd34e1230686feb"

git worktree add --detach \
  /tmp/nano-vllm-spec-v0-480a3b2 \
  "${SPEC_V0_COMMIT}"

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/tmp/nano-vllm-spec-v0-480a3b2 \
  /venv/main/bin/python tests/run_speculative_v0_golden.py \
  --model /workspace/models/Qwen3-0.6B \
  --mode eager \
  --expected-commit "${SPEC_V0_COMMIT}" \
  --expected-runner-commit "${SPEC_V2_COMMIT}" \
  --output /workspace/spec-v2-evidence/v0-eager.json

sha256sum /workspace/spec-v2-evidence/v0-eager.json
```

Run the same protocol with `--mode graph` and a distinct write-once output for
the graph lifecycle gate. The V0 and V2 mode, model identity, engine settings,
greedy workload, tokenizer snapshot, outputs, and scheduler trace must match the
comparator's registered contract. Do not reuse the eager golden for graph mode.

### 6.4 Retained inert lifecycle gate

Supply the matching V0 artifact and its exact SHA256 to the V2 runner. The
following is the same-model ownership cell; it proves independent model/cache
ownership but not heterogeneous target/draft geometry:

```bash
SPEC_V2_COMMIT="$(git rev-parse HEAD)"
SPEC_V0_EAGER_SHA="<sha256 printed for v0-eager.json>"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  /venv/main/bin/python tests/run_speculative_v2_lifecycle.py \
  --model /workspace/models/Qwen3-0.6B \
  --draft-model /workspace/models/Qwen3-0.6B \
  --mode eager \
  --configured-k 2 \
  --gpu-memory-utilization 0.5 \
  --max-model-len 512 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 4 \
  --check-explicit-boundary \
  --expected-commit "${SPEC_V2_COMMIT}" \
  --retained \
  --v0-golden /workspace/spec-v2-evidence/v0-eager.json \
  --v0-golden-sha256 "${SPEC_V0_EAGER_SHA}" \
  --output /workspace/spec-v2-evidence/v2-eager.json
```

The graph cell uses `--mode graph`, its graph-mode V0 golden and hash, and a new
output path. This lifecycle harness deliberately runs the current speculation-off
control first. It therefore makes no cold-start, first-eligible-cycle compile,
or clean compiler-cache claim, even if a unique cache directory happens to be
used for the subprocess.

The retained lifecycle cell verifies, among other invariants:

- no draft-owned attributes exist when speculation is disabled;
- off/on greedy outputs, traces, and RNG endpoints match;
- generation with ownership enabled performs zero draft forwards;
- tokenizer fingerprints and target/draft model manifests remain stable;
- automatic and explicit joint-KV audits independently reconcile;
- an impossible explicit block request raises the typed capacity error;
- a healthy engine remains usable after another constructor is rejected;
- teardown is idempotent; and
- a smaller explicit engine can construct and run after the capacity failure.

### 6.5 Retained phase-recovery matrix

Run one fresh Python subprocess per registered phase. The runner chooses eager
mode for non-graph phases and graph mode for graph profiling, final draft graph
capture, and draft pretouch:

```bash
SPEC_V2_COMMIT="$(git rev-parse HEAD)"

for SPEC_PHASE in \
  draft_construct \
  draft_load \
  draft_warmup \
  graph_profile \
  joint_allocate \
  draft_graph \
  draft_pretouch \
  memory_finalize
do
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
    /venv/main/bin/python tests/run_speculative_v2_gpu_recovery.py \
    --phase "${SPEC_PHASE}" \
    --model /workspace/models/Qwen3-0.6B \
    --draft-model /workspace/models/Qwen3-0.6B \
    --num-kvcache-blocks 4 \
    --expected-commit "${SPEC_V2_COMMIT}" \
    --retained \
    --output "/workspace/spec-v2-evidence/recovery-${SPEC_PHASE}.json" \
    || exit 1
done
```

The loop is orchestration only; each Python invocation is the required fresh
process. The explicit exit-status check prevents a later successful phase from
hiding an earlier failure.

## 7. Evidence retention status

The V2 implementation contains evidence producers and CPU tests for their
schemas, provenance rules, write-once behavior, and failure paths. The schemas
are:

```text
nano-vllm-speculative-v0-golden-v1
nano-vllm-speculative-v2-lifecycle-v2
nano-vllm-speculative-v2-gpu-recovery-v1
```

At implementation-commit time, these scripts are protocol code, not retained
results. No lifecycle JSON produced from the dirty pre-commit worktree is a V2
certificate. No result should enter a retained archive until all of the
following are available:

1. the exact clean V2 implementation SHA;
2. matching eager and graph canonical-V0 goldens and hashes;
3. clean retained eager and graph lifecycle artifacts;
4. all eight clean retained phase-recovery artifacts;
5. an archive manifest that hashes the scripts, model manifests, and raw JSON;
6. a GPU-free offline validator that rejects missing, altered, inconsistent, or
   non-finite required fields; and
7. documentation of the exact commands and any rejected runs.

Those artifacts, the validator, and their measured results belong in a separate
post-implementation evidence/docs commit. That commit may state actual PASS/FAIL
results and artifact hashes. This implementation document intentionally does
not invent them.

## 8. Explicit limitations and remaining rungs

V2 leaves these capabilities unimplemented:

- request-time draft-cache coverage and catch-up;
- draft-token proposal and retained proposal probabilities;
- effective-K scheduling for individual cycles;
- speculative scheduler DTOs and multi-position block reservations;
- target all-query verification logits;
- greedy prefix verification and stochastic modified rejection;
- correction and full-acceptance bonus sampling;
- transactional multi-token commit and rollback;
- speculative prefix hashing, preemption, or cancellation;
- burst streaming and speculative metrics;
- route-key registries and first-cycle compile-completeness proofs;
- tensor-parallel or FlashInfer speculative execution; and
- any performance router, benchmark, roofline, or speedup claim.

Before V3 executes proposals, its probability path must either write each
canonical draft row directly into its preallocated `q[B,K,V]` destination or
expand the planner for the extra result/copy lifetime. The current sampler
returns a fresh probability tensor; a naive collect-and-stack implementation is
not covered by V2's reservation and must not be admitted under it.

The planned continuation is:

| Rung | Scope after V2 |
|---|---|
| V3 | Track draft-cache coverage, catch up committed prefixes, execute real draft proposals, and discard them while ordinary target decode remains authoritative. |
| V4 | Add typed scheduler plans, per-cycle effective K, deterministic baseline fallback, and transactional multi-slot reservations. |
| V5 | Add all-query target verification, exact rejection/bonus sampling, actual speculative-workspace allocation, and atomic burst commit. |
| V6 | Certify streaming, metrics, cancellation, abandoned-session finalization, and failure behavior for committed bursts. |
| V7 | Define the route/workspace/warm registry, certify cold and repeated compile behavior, measure runtime memory peaks, build the performance router, and publish the A100 roofline/crossover archive. |

V2's draft warmup, draft cache allocation, and draft decode graph capture are
prerequisites for V3; they are not substitutes for V3 execution. Likewise,
V2's modeled workspace reservation is a capacity policy, not V5/V7 runtime
memory evidence.
