# Invariant numerics and speculative-decoding optimization

> The completed high-risk source review and expanded forced-history matrix are
> recorded in [High-risk and invariant-numerical review](RISK_AND_NUMERICAL_QUALIFICATION_REVIEW.md).
> That review supersedes the smaller qualification summary below and retains the
> current fast-mode negative controls.

This document describes the follow-up implementation on `fork-main`. The normal
runtime remains `numerical_mode="fast"`; its kernels and ordinary sampling path
retain their existing behavior. The new backend is opt-in:

```python
llm = LLM(
    model_path,
    numerical_mode="invariant",
    tensor_parallel_size=1,
    top_p_backend="exact",
    max_model_len=4096,
)
```

## Invariant execution contract

Invariant mode currently accepts BF16 Qwen3-0.6B and Qwen3-4B geometry, TP=1,
exact sampling, CUDA compute capability 8.0 or newer, and contexts through 4,096
tokens. Unsupported configurations fail during construction. The mode is copied
into the runner and every numerical module during construction, and the backend
identity is part of speculative graph/workspace fingerprints.

The backend uses fixed-geometry Triton linear and LM-head kernels with FP32
accumulators, one fixed-width FP32 RMSNorm reduction per logical row (including
Q/K normalization), direct elementwise rotary/SiLU execution, and one common
paged-attention implementation for prefill, decode, and speculative
verification. KV is stored before attention. Each query gathers its logical
paged prefix and uses its absolute context length as the causal boundary, so its
reduction does not depend on other live rows or earlier chunk boundaries.
Invariant mode currently runs eagerly because its query ownership and context
lengths are host metadata; there is no claim that it is a low-latency backend.

Intermediate prefill chunks no longer run an LM head or consume sampling RNG in
invariant mode. Explicit emission indices select only rows whose prompt has
finished. Speculative verification retains all query positions through a
separate all-position LM-head entry point.

The forced-history diagnostic in `benchmarks/invariant_numerics.py` gives every
execution an independently initialized engine/KV state. It forces identical
token histories, then compares corresponding logical KV and emission logits by
SHA-256. The A100 archive under
`benchmarks/invariant_numerics_evidence/2026-09-08-a100/` compares token budgets
64, 128, and 256 against an unchunked budget-512 reference. Each comparison has
six corresponding KV states and six corresponding logit states with zero
mismatches. Direct kernel tests also compare 19 live rows against the same rows
run individually, bit for bit.

This is a Qwen3-0.6B qualification result. The configured Qwen3-4B envelope was
not rerun because the historical temporary checkpoint is absent from this
machine. Do not treat the configuration allow-list as a 4B performance or
numerical certificate.

## Chunk scheduling and allocation

The reviewed missing-decode-graph fallback and partial-prefill reclamation fixes
are integrated. Physical KV allocation is now incremental: whole-request
admission still proves that a request can fit, but the allocator pins a reusable
prefix and owns blocks only through the scheduled chunk. Extension preflights
all required blocks and leaves ownership unchanged on failure. The scheduler
then applies its existing preemption/requeue policy. Speculative reservation,
rollback, prefix hashes, cancellation, and ownership regressions run against the
same allocator.

The former 17-block pressure workload no longer needs to evict the 4,096-token
prefill or an active decoder merely because the prefill reserved unscheduled
blocks. Exact token-budget graph endpoints and the existing sequence-slot tiers
remain available in fast mode; an absent compatible decode graph executes eager
instead of raising.

## Speculative execution changes

The optimized cycle keeps q probabilities, proposal IDs, target probabilities,
and bonus noise in engine-owned bounded buffers for B<=4 and K<=4. Static
sampling metadata is checked on the host. Internal trusted sampler and modified
rejection paths preserve the exact transformed FP32 p/q law and FP64
acceptance/residual arithmetic while removing repeated device-to-host validation
checks. Draft IDs are accumulated into one device error flag, clamped before a
possible embedding access, and checked once before target verification. Cycle
results use one packed device-to-host transfer before transactional commit.

Draft decode CUDA graphs now include the LM head. A draft that is behind by one
token reuses its decode graph; longer or irregular catch-up remains a bounded
eager ragged route. Parallel stochastic target verification uses the existing
ragged target CUDA graphs when a captured token/slot bucket fits and otherwise
falls back to eager. Its all-position logits remain distinct from ordinary
emission-only logits.

Fast-mode batches containing any greedy row retain sequential K+1 target calls
for numerical compatibility. Invariant homogeneous-greedy batches perform one
causal target pass, compare proposal IDs directly with target argmax IDs, and do
not allocate dense target p/residual tensors for acceptance. Invariant mixed
batches use parallel target probabilities.

The A100 functional archive under
`benchmarks/speculative_optimization_evidence/2026-09-08-a100/` contains matched
ordinary/speculative reports. Fast graph mode completed 50 speculative cycles
across K=1..4 with the causality, streaming, prefix-cache, cancellation,
injected-failure/RNG rollback, and memory cleanup checks passing; timed cycles
compiled no new graph. Invariant eager mode completed 44 cycles and all four
matched greedy workloads produced exactly the ordinary invariant tokens.

Those smoke timings are intentionally not a promotion result. They use
Qwen3-0.6B as both target and draft, one timed sample per workload, and include
large cold/path effects. Some cells won and others lost. The required five-pair
Qwen3-4B target/Qwen3-0.6B draft qualification cannot run without the 4B weights.

## Adaptive routing

Set `speculative_policy="adaptive"` together with the normal speculative model
options, then load calibration before admitting requests:

```python
llm = LLM(
    target_path,
    draft_model=draft_path,
    num_speculative_tokens=4,
    speculative_policy="adaptive",
)
llm.load_speculative_calibration([
    {
        "batch_size": 1,
        "sampling_family": "temperature",
        "context_bucket": 256,
        "catchup_bucket": 0,
        "k": 4,
        "ordinary_ms_per_token": 8.7,
        "cycle_ms": 20.0,
        "expected_committed_tokens": 3.0,
    },
])
```

Calibration keys bind exact batch size, sampling family, power-of-two context
and catch-up buckets, and K. The policy evaluates every legal K<=4 using
`ordinary_ms_per_token / (cycle_ms / expected_committed_tokens)`. It speculates
only when the predicted speedup is at least 1.10. Unknown cells, losing cells,
fast greedy compatibility routes, request/context tails, and insufficient token
budgets use ordinary decoding before proposal generation or RNG mutation.
Observed accepted-prefix length updates a per-cell EMA. Call
`llm.speculative_routing_metrics()` to inspect chosen K values and bypass
reasons. Calibration workloads must be separate from final qualification.

`speculative_policy="fixed"` is the default for compatibility, and speculation
remains disabled unless both draft options are supplied.

## Prefill/decode disaggregation gate

`benchmarks/pd_disaggregation_feasibility.py` implements the agreed analysis and
handoff contract; it does not implement distributed serving. The descriptor
binds model and numerical identities, full token history, processed-token
coverage, logical KV ordering/layout, and first-token ownership. A separate
progress record validates receiver allocation, transfer completion, receiver
install, cancellation, retry, and source-release acknowledgments. Physical
block IDs are never part of the portable identity.

The transfer calculator reproduces 147,456 target-KV bytes/token for Qwen3-4B:
a 4,096-token prompt is 576 MiB; Qwen3-0.6B draft KV adds 448 MiB. The
matched-resource comparison requires equal GPU counts and reports TTFT,
p95/p99/max ITL, throughput, latency-constrained goodput, and goodput per dollar.
Its default serving-plan gate requires at least 10% goodput improvement, no
regression in the recorded latency statistics, and no loss in goodput per
dollar.

This machine has one A100, so it can validate serialization and transfer
arithmetic but cannot establish resource-isolation gains. Run the experiment on
two GPUs, comparing one prefill plus one decode worker against two colocated
replicas, with speculation disabled first. Disaggregation does not repair
cross-shape numerical differences, sequential verification, or sampler
synchronization; it can only isolate prefill work and capacity pressure when the
measured handoff cost permits it.
