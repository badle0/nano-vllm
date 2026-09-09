# Experimental speculative decoding

This is an opt-in experimental implementation. It performs actual proposal,
target verification, rejection/correction or bonus sampling, and atomic token/KV
commit. The retained baseline routes measured **1.92–3.62x slower** than
speculation off; see [the benchmark report](SPECULATIVE_BENCHMARKS.md). The
follow-up adds reusable workspaces, graph-backed verification, invariant parallel
greedy verification, and adaptive bypass. Its implementation and current
qualification limits are recorded in
[invariant and speculative optimization](NUMERICAL_AND_SPECULATIVE_OPTIMIZATIONS.md).

## Enable or disable

Use compatible local Qwen3 checkpoints with matching vocabulary and tokenizer
semantics. Target and draft may have different model sizes. Both checkpoints
must have supported safetensors weights. Compatibility and resource checks run
before requests execute.

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    "/path/to/Qwen3-4B",
    draft_model="/path/to/Qwen3-0.6B",
    num_speculative_tokens=4,
    speculative_policy="fixed",  # or "adaptive" with loaded calibration
    tensor_parallel_size=1,
    top_p_backend="exact",
    max_num_seqs=4,
    max_num_batched_tokens=4096,
    max_model_len=4096,
    gpu_memory_utilization=0.8,
    enforce_eager=False,
)
try:
    outputs = llm.generate(
        ["Explain how a computer predicts the next word."],
        SamplingParams(temperature=0.8, max_tokens=64),
    )
    print(outputs[0]["text"])
finally:
    llm.exit()
```

Omit **both** `draft_model` and `num_speculative_tokens` to use ordinary decoding.
Speculation is disabled by default. Passing only one option is invalid. The
`fixed` policy preserves configured behavior. The `adaptive` policy requires
calibration loaded through `llm.load_speculative_calibration(...)` and bypasses
unknown or predicted-losing cells. Using the same checkpoint for both models is
a functional test, not an acceleration strategy. Memory requirements include
both models and separate physical KV pools.

## Supported envelope and bypass

- Speculation supports TP=1 and `top_p_backend="exact"` only. TP>1 and
  FlashInfer speculation fail early; they are not silently substituted.
- At most four live decode rows speculate together. Larger live batches bypass
  speculation as a whole; this is different from the public admission limit.
- Effective proposal length K is at most four. Configured K>4 is capped.
  Remaining output/context length, the full-batch input budget, free KV blocks,
  workspace and warmed routes can reduce K or cause ordinary decoding.
- Draft catch-up consumes the aggregate input budget. A large uncached prefix
  can prevent speculation even at a supported batch size.
- Draft weights and KV remain resident while bypassing. These caps are safety
  and validation limits, not measured optimal crossover points.
- Existing admission, synchronous session ownership and context rules remain;
  see [README](../README.md) and [robustness contracts](ROBUSTNESS_FIXES.md).

## Algorithm and code map

For draft proposal y, accept with probability `min(1, p(y)/q(y))`. On the first
rejection sample from normalized `max(p-q, 0)` and discard the proposal suffix.
If all K proposals are accepted, draw an independent bonus from the next target
distribution. Here p and q are the distributions **after** temperature, top-k
and top-p transformation. Applying rejection to raw softmax when filtering was
used would implement a different law.

The implementation retains canonical FP32 probabilities and evaluates modified
rejection robustly in FP64, with an explicit numerical empty-residual recovery
and accounting. This is exactness for the represented p/q law, not a guarantee
of identical probabilities across differently batched BF16 model kernels or
identical sampled output for the same random seed as ordinary decoding.

| Component | Responsibility |
|---|---|
| `config.py`, `utils/tokenizer_identity.py`, `utils/loader.py` | Option, checkpoint, tokenizer and tensor-shape validation |
| `engine/model_runner.py` | Dual model/KV ownership, memory sizing, draft catch-up/proposals, warmup and graph routes |
| `engine/speculative_memory.py`, `speculative_routes.py`, `speculative_plan.py` | Workspace lifetimes, finite route readiness and complete-batch admission |
| `engine/speculative_execution.py` | Target verification and proposal acceptance/correction/bonus orchestration |
| `layers/sampler.py` | Canonical sampling distributions and rejection law |
| `engine/speculative_result.py` | Host-only results validated before mutation |
| `engine/scheduler.py`, `block_manager.py` | Reservations, atomic burst commit, trimming, prefix hashes and rollback |
| `engine/llm_engine.py`, `metrics.py` | Execution, failure/RNG handling, public delivery and counters |
| `models/qwen3.py`, `layers/embed_head.py` | Explicit all-query logits path; ordinary prefill still selects its final query |

Stochastic batches verify `[last committed token, d1, ..., dK]` with one causal
paged target pass producing B*(K+1) logits rows. Fast-mode batches containing any
greedy row retain K+1 ordinary target decode calls because cross-shape BF16
verification can change a non-tied greedy decision. Invariant homogeneous-greedy
batches use one causal pass and compare target argmax IDs directly, without
dense target probability or residual tensors. Invariant mixed batches use the
parallel probability path.

Graph mode captures the draft transformer and LM head. A one-token draft
catch-up reuses the decode graph; longer ragged catch-up remains eager. Parallel
target verification uses warmed ragged target graphs when its token/slot shape
fits and otherwise executes eager. Exact probability and rejection sampling
remain eager ATen operations over reusable engine buffers. Warmup covers target
lanes and admitted B/K and sampling families. Old shadow/discard entry points
remain for contract tests; initialized speculative engines select verified
execution.

## Commit, rollback, streaming and metrics

For n committed tokens at old logical length L, target coverage becomes L+n-1
and draft coverage becomes `min(L+n-1, L+K-1)`. The final correction or bonus is
an unprocessed tail. Only committed full target blocks enter the prefix cache.
EOS ends a burst only if `ignore_eos=False`; `max_tokens` always applies.

All rows are validated before mutation. Recoverable failures undo logical and
physical ownership changes, queues, metrics and event creation, and restore
CPU/CUDA RNG. Successful proposals consume RNG normally. Reclaimed target blocks
have their old hashes invalidated before writes: rollback cannot restore hashes
for overwritten KV contents. This conservative cache eviction is intentional.
Recovery from a device-fatal CUDA error is not promised.

`stream()` delivers one event per committed token, never unverified proposals.
A burst preserves per-request order; only the terminal token is marked finished.
Closing or abandoning a session cancels its owned requests and drops undelivered
events. Compute timestamps may be equal within a burst, so zero intra-burst
compute ITL is expected; delivery latency remains separate.

Enabled requests expose additive counters (zero even if speculation never runs):

| Counter | Meaning |
|---|---|
| `spec_cycles` | Successfully committed cycles |
| `spec_proposed_draft_tokens` | Proposals, including rejected suffixes |
| `spec_accepted_draft_tokens` | Accepted proposals actually committed |
| `spec_committed_tokens` | All committed speculative emissions |
| `spec_bonus_tokens` | Bonus tokens actually committed |
| `spec_draft_positions` | Catch-up plus proposal input positions |
| `spec_target_verification_positions` | Target query positions |
| `spec_residual_numerical_fallbacks` | Numerical residual recoveries |

Failed cycles add no counters. Speculation-off metrics are unchanged.
`StepOutput.num_decode_tokens` remains a decode-row count, not a burst-token count.
Adaptive engines additionally expose `llm.speculative_routing_metrics()`, which
reports selected K counts, bypass reasons, the last decision, and calibration
cell count without conflating bypass with speculation disabled.

## Maintained tests and archive boundary

Runtime tests cover sampling laws and FP32 aliasing, config/load/tokenizer
validation, memory-owner accounting, route budgets, temporary blocks, rejection
positions, bonus and EOS tails, whole-batch commit faults, cache invalidation,
RNG rollback and stream lifecycle. V3/V4 names identify test origins, not obsolete
behavior. CI runs the maintained CPU suite with a fail-on-construction Qwen shim
and excludes the CUDA-only histogram module (which imports Triton at collection);
that job cannot certify attention kernels or model execution. The full GPU test
invocation retains the histogram tests.

The maintained CPU job uses Torch 2.10.0 and downloads only three files for a
pinned Qwen3-0.6B tokenizer before running tests offline. For a machine without
the local checkpoint, prepare those files with
`python .github/ci/prepare_tokenizer.py --output /tmp/nano-vllm-tokenizer`, then
set `NANOVLLM_TEST_TOKENIZER_PATH=/tmp/nano-vllm-tokenizer` when running pytest.
Missing tokenizer fixtures fail explicitly rather than silently skipping the
real-tokenizer tests. No model weights are required for the CPU contract suite.

Torch 2.4.1 CPU is not a passing compiled-sampler lane: remote CI hit an Inductor
graph-rewrite error in four ordinary sampler tests. Selecting Torch 2.10 for CI
does not repair or certify the older compiler; see the
[CI follow-up record](SPECULATIVE_BENCHMARKS.md). Compilation remains enabled,
with neither error suppression nor an eager-test substitution.

The GPU integration tool is `tests/run_speculative_v5_gpu.py`; use
`--numerical-mode invariant --mode eager` for the invariant route. Its
compiler/cache instrumentation still imports `run_speculative_v3_route_compile.py` and
`_speculative_v3_evidence.py`; these are retained dependencies, not disposable
logs. Use the V5 entry point for the current implementation, not the historical
V3 certification entry point. Current benchmark and phase tools are
`tests/run_speculative_v7_benchmark.py` and `tests/run_speculative_v7_phases.py`.

Historical artifact validators, their dedicated tests and the chronological
design bundle stay on the pinned experiment branch. They are not silently
skipped by CI here: they are outside this branch's maintained test set. See
[branch policy](BRANCHES.md) and the [benchmark reproduction guide](SPECULATIVE_BENCHMARKS.md).
Runtime `gpu_certified=False` remains intentional: bounded A100 experiments do
not certify every model, device, dtype or configuration.
