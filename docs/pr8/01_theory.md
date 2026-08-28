# Speculative decoding: theory and engine invariants

## Scope and claims

Speculative decoding accelerates autoregressive generation by proposing several
tokens with a cheap **draft** mechanism and checking them in parallel with the
**target** model. The exact algorithm below preserves the target model's output
distribution. This means equality in law: for every output sequence, its
probability is the same as ordinary target-model sampling. It does **not** imply
same-seed token equality, identical floating-point results, or bitwise equality
with a non-speculative run. Speculation consumes random numbers along a different
control flow, and batched kernels can use different reduction orders.

No v2 speedup is claimed here. Speedups cited in the bibliography are results
reported by their respective authors on their hardware, models, and workloads.
The gates below must be run before making a performance claim for nano-vLLM v2.

## Notation

At a committed prefix `x`, let `p_i(. | x, y_<i)` be the target distribution and
`q_i(. | x, y_<i)` the distribution that actually generated draft token `y_i`.
Both distributions include every transformation that affects sampling: temperature,
top-k/top-p, token bans, repetition penalties, grammar constraints, and any other
logit processor. Let `[z]_+ = max(z, 0)`. A speculation round proposes `gamma`
tokens and performs one target verification pass over their positions.

## Exact stochastic algorithm

For each proposed token `y_i`, draw an independent `u_i ~ Uniform(0, 1)` and
accept the token when

```text
u_i <= a_i(y_i),       a_i(y) = min(1, p_i(y) / q_i(y)).
```

Evaluation is sequential even though target logits are computed in parallel.
Stop at the first rejection. If `y_i` is rejected, sample a replacement from

```text
r_i(z) = [p_i(z) - q_i(z)]_+ / Z_i,
Z_i    = sum_z [p_i(z) - q_i(z)]_+.
```

Discard `y_i` and all later proposals. If all `gamma` proposals are accepted,
sample one bonus token from the target distribution at the next position. Thus a
round emits between one and `gamma + 1` tokens, subject to EOS and length limits.

The implementation must retain, or exactly reconstruct, `q_i(y_i)` and the full
`q_i` distribution required by the residual. Substituting a pre-processor draft
distribution, stale logits, or a different precision path invalidates the proof.
If `q_i(y_i) == 0` for a token said to have been sampled from `q_i`, metadata is
inconsistent and the round should fail rather than silently approximate.

### One-step proof

For any token `z`, the unconditional mass contributed by accepting the draft is

```text
q(z) min(1, p(z)/q(z)) = min(q(z), p(z)).
```

The total rejection probability is

```text
1 - sum_z min(p(z), q(z))
  = sum_z [p(z) - q(z)]_+
  = Z.
```

On rejection, the replacement contributes `Z r(z) = [p(z)-q(z)]_+`.
Consequently the emitted-token probability is

```text
min(p(z), q(z)) + [p(z)-q(z)]_+ = p(z).
```

Conditioning on each accepted prefix repeats the same argument at the next
position. Induction therefore gives exactly the target distribution over complete
sequences, including variable stopping at EOS. If `p == q`, then `Z == 0`, but a
rejection has probability zero and the residual is never sampled. Numerically,
that branch should be guarded explicitly.

The marginal acceptance probability at a position is

```text
alpha = sum_z min(p(z), q(z)) = 1 - TV(p, q),
```

where `TV` is total-variation distance. Draft quality matters through distributional
overlap, not merely top-1 agreement.

## Greedy limit

With greedy decoding, the target distribution is a point mass at its selected
token (including the production tie-breaking rule). Verification accepts the
longest draft prefix whose tokens equal the target argmax tokens. At the first
mismatch, emit the target argmax at that position and discard the remaining draft
suffix. If every proposal matches, emit the next target argmax as the bonus token.

This deterministic algorithm should be token-for-token equal to ordinary greedy
decoding when both paths use the same logits, processors, precision, and tie-break.
That stronger equality is specific to deterministic decoding; it must not be
advertised for stochastic sampling. A one-hot greedy draft combined with a
stochastic target is not the simple prefix-match case: it remains exact only when
treated as `q` by the rejection-sampling algorithm.

## Performance model and roofline interpretation

Let `s_k` be the probability that at least the first `k` proposals are accepted.
The expected tokens emitted per target verification is

```text
E[N] = 1 + sum_(k=1)^gamma s_k.
```

Under the simplifying assumption of independent, stationary acceptance `alpha`,

```text
E[N] = 1 + alpha + ... + alpha^gamma
     = (1 - alpha^(gamma+1)) / (1 - alpha).
```

If one target decode step costs one time unit and each sequential draft step costs
`c`, the idealized model from Leviathan et al. gives

```text
speedup_ideal = (1 - alpha^(gamma+1))
                / ((1 - alpha) (1 + gamma c)).
```

This is a reasoning model, not an engine prediction. A deployable model is

```text
T_round = T_draft(gamma)
        + T_verify(batch shape, gamma, context)
        + T_accept/reject
        + T_scheduler/cache/communication,

throughput = E[committed tokens per round] / T_round.
```

All terms must be measured. In particular, verification of `gamma` positions is
not universally the cost of one decode token.

The roofline explanation is workload-dependent. Small-batch decode commonly has
low arithmetic intensity because model weights are read for very little useful
work; it is often memory-bandwidth bound. Verifying several positions in one
target pass can reuse weights and increase useful operations per byte, converting
otherwise idle compute capacity into extra accepted tokens. Speculation can lose
when draft overhead is large, acceptance is low, verification becomes compute
bound, attention/KV traffic dominates at long context, continuous batches already
saturate the target, tensor-parallel communication grows, or ragged shapes defeat
efficient kernels/graphs.

Accordingly, tuning `gamma` is a constrained optimization over acceptance,
latency, memory, and batch shape. Adaptive speculation may choose a smaller depth
or disable speculation at high load. It must preserve the exact algorithm; tuning
may change work, not acceptance semantics.

## KV-cache correctness

Draft and target states are logically separate, even if storage is shared by an
explicitly safe optimization. A target verification pass creates provisional KV
entries for proposed positions. Only the accepted prefix may become committed.
At a rejection, the rejected token and every later proposal must be rolled back.
The replacement token is output but normally becomes an input—and receives its KV
entry—on the next target step. The same applies to the all-accepted bonus token.
An implementation may cache either eagerly only if it proves that the cached state
is exactly the state ordinary decoding would create and avoids double insertion.

Required invariants are:

1. committed sequence length, block table, positions, RoPE indices, and target KV
   length always describe the same prefix;
2. no rejected proposal is visible to a later attention operation;
3. rollback cannot mutate shared prefix-cache blocks; copy-on-write happens before
   provisional writes;
4. provisional capacity is reserved before launching verification and released on
   rejection, cancellation, preemption, admission failure, and exceptions;
5. block-boundary rollback is correct for acceptance lengths from zero through
   `gamma`, including partially filled blocks;
6. EOS and `max_tokens` truncate both emitted tokens and committed state; and
7. target and draft caches are independently advanced or rolled back according to
   what each model has consumed.

Tree verification additionally requires an ancestor-only attention mask. Nodes
must not attend to siblings or descendants, position IDs must represent their path,
and only one selected root-to-leaf path may be committed.

## Continuous batching and scheduling

Speculation makes “token count” ambiguous. The scheduler must distinguish committed
output tokens, draft tokens, target verification query positions, and provisional
KV slots. Admission should reserve a defensible worst case or use transactional
allocation with safe rollback. Counting only expected accepted tokens can
overcommit memory.

A target batch may pack ragged proposal lengths, but each request retains its own
acceptance scan, residual distribution, stopping conditions, RNG identity, and
commit length. Compaction must not associate logits or random draws with a different
request. Stable per-request RNG streams are desirable for reproducibility within
the speculative implementation, although they do not create same-seed equality
with baseline sampling.

Fairness must be based on committed service as well as speculative work. Otherwise
a low-acceptance request can consume disproportionate draft and verification
capacity. Preemption and cancellation are transactions: provisional blocks and
draft state are discarded, while the last committed prefix remains restartable.
Streaming exposes only committed tokens, never tentative proposals.

## Variant taxonomy

### Draft-model chain

Classic speculative decoding uses a smaller autoregressive model to produce one
chain. Leviathan et al. and Chen et al. independently established exact verification
schemes. Model/tokenizer compatibility and the draft-to-target cost ratio constrain
the attainable gain.

### Trees and multiple candidates

SpecInfer verifies a token tree produced by one or more speculative models in a
single target pass, using tree attention and a tree-aware verification procedure.
Sequoia optimizes tree structure under hardware-dependent cost and acceptance.
Trees improve coverage but increase verification width, masking complexity, KV
pressure, and graph-shape diversity. Exactness depends on the variant's published
verification rule; selecting any target-plausible branch is not sufficient.

### Medusa

Medusa adds heads that predict multiple future tokens and verifies a candidate
tree with tree attention. Medusa-1 describes rejection sampling for lossless
sampling as well as a typical-acceptance heuristic. The latter is intentionally
approximate and must be labeled lossy. Medusa-2 jointly fine-tunes the backbone and
heads; exactness claims then refer to the resulting fine-tuned target, not automatic
bitwise or distributional identity to the original checkpoint.

### EAGLE and EAGLE-2

EAGLE drafts in the feature space while also conditioning on sampled tokens,
addressing uncertainty in future features. EAGLE-2 builds draft trees dynamically
from confidence estimates. Both still require correct target verification; feature
prediction does not remove the rejection/selection correctness obligation.

### Prompt/reference lookup and retrieval

LLMA/reference decoding copies candidate continuations from matching text in the
prompt or a reference. REST retrieves candidate continuations from a datastore.
These approaches avoid a learned draft model and can excel on repetitive or
retrieval-aligned text. For greedy decoding, longest-prefix target verification is
exact. For stochastic decoding, candidates need a defined proposal law and exact
correction; “appears in the prompt” or “passes a probability threshold” alone is
not an exactness proof.

### Self-speculation and early exit

Draft & Verify skips selected layers during drafting and uses the full model for
verification. LayerSkip trains/uses early-exit predictions with self-speculative
verification. These share weights and can reduce memory, but cache reuse, skipped
layers, and the draft cost require careful implementation. Exactness again derives
from target correction, not architectural similarity.

### Blockwise, multi-token, and Jacobi-style methods

Blockwise Parallel Decoding predicts several positions with auxiliary models and
accepts a verified prefix. Multi-token-prediction heads are proposal mechanisms,
not by themselves exact decoders. Lookahead decoding uses Jacobi-style parallel
candidate generation and verification without a separate draft model. These belong
to the broader speculative family but have different cost models and candidate
dependencies; their correctness arguments must not be replaced by the chain proof
without showing the required proposal probabilities and verification equivalence.

## Limitations and non-goals

- Exact sampling preserves distributions, not baseline RNG traces or floating-point
  bit patterns. Statistical tests are required for stochastic modes.
- Approximate acceptance rules (typical thresholds, relaxed top-k agreement, or
  uncorrected candidate selection) trade quality for speed and require a separate,
  explicit product mode and quality evaluation.
- Beam search, constrained search with hidden state, and tokenizer-mismatched draft
  models need separate correctness designs. The chain proof does not automatically
  cover them.
- Acceptance is domain-, prompt-, temperature-, processor-, and model-pair-dependent.
  A single average acceptance rate is not a release claim.
- Speculation consumes extra model memory, provisional KV capacity, scheduler work,
  and potentially tensor-parallel communication. It can reduce maximum concurrency.
- Dynamic lengths and trees can cause compilation/graph proliferation. Bounds and
  fallbacks are part of the design, not incidental optimizations.

## Validation and release gates

### Distributional correctness

1. Exhaustively enumerate tiny-vocabulary `p` and `q`, including zero support,
   disjoint support, `p == q`, and extreme ratios; compare the algorithm's analytic
   emitted law with `p` to tight numerical tolerance.
2. Enumerate complete sequences from a tiny Markov model and verify sequence-level
   probabilities, EOS, and length truncation.
3. Run seeded Monte Carlo tests for plain, temperature, top-k, top-p, combined
   filters, token bans, repetition penalties, and grammar masks. Predeclare sample
   counts and thresholds; report total variation, maximum absolute error, and a
   multiple-testing-aware goodness-of-fit result.
4. Fail on mismatched/stale proposal metadata rather than silently using a wrong
   `q`. Exercise zero residual and underflow/overflow paths.
5. For greedy mode, require exact token equality with baseline across random logits,
   ties, processors, EOS, and every accepted-prefix length.

### State correctness

After every possible acceptance length `0..gamma`, compare the next target logits
and all subsequent tokens against a full-recompute oracle. Cover block boundaries,
prefix-cache hits, shared-prefix copy-on-write, chunked prefill, eager and captured
graphs, preemption/restart, cancellation, admission failure, exceptions, EOS,
replacement/bonus off-by-one cases, and repeated reject/accept cycles. Assert block
allocator accounting and absence of provisional-state leaks after each case. Run
equivalent tensor-parallel tests when multi-GPU hardware is available.

### Batching and API behavior

Test heterogeneous proposal depths, contexts, sampling parameters, stop conditions,
and accept lengths in the same packed batch. Randomly cancel, preempt, and admit
requests while checking per-request RNG/state identity, scheduler fairness, stream
versus non-stream output parity, and the rule that tentative tokens are never
published. Unsupported modes must fail before mutating engine state.

### Performance and roofline evidence

Use paired baseline/speculative runs on identical commits, weights, prompts, output
limits, seeds, warm-up policy, and GPU state. Sweep model/draft pairs, context and
output lengths, batch/concurrency, prompt domains, temperature/filter settings,
`gamma`, graph/eager mode, cache pressure, and tensor parallelism. Report at least:

- request throughput and committed output-token throughput;
- TTFT, TPOT/inter-token latency, and end-to-end latency at p50/p95/p99;
- acceptance-length histogram, `s_k`, and committed tokens per target invocation;
- draft, verification, correction, scheduler, cache, and communication time;
- proposed/verified/committed token counts and work amplification;
- peak model/KV/provisional memory, allocator churn, preemptions, and leaks;
- kernel time, memory bandwidth, FLOPs/utilization, graph hit rate, and TP traffic.

A release claim must include confidence intervals over repeated trials and a clearly
named operating region. Gates should require no material regression outside that
region, tail-latency and memory limits, correctness gates above, and a demonstrated
benefit over simply increasing ordinary continuous-batch concurrency. Source-paper
speedups are context, never substitutes for these measurements.

## Glossary

- **Acceptance length**: number of consecutive proposals accepted before rejection,
  or `gamma` when all are accepted.
- **Bonus token**: target token emitted after all proposals in a round are accepted.
- **Committed prefix**: tokens and KV state visible to future decoding and clients.
- **Draft/proposal distribution (`q`)**: the actual law used to sample candidates.
- **Exact/lossless**: output distribution equals target decoding's distribution.
- **Provisional KV**: verification state not yet authorized for commitment.
- **Residual distribution**: normalized positive part `[p-q]_+` used after rejection.
- **Target distribution (`p`)**: production next-token law after all processors.
- **Tree attention**: mask allowing a candidate node to attend only to its ancestors.
- **Verification**: target evaluation and exact acceptance/replacement procedure.

## Primary sources

1. Leviathan, Kalman, and Matias, “Fast Inference from Transformers via
   Speculative Decoding,” ICML 2023. <https://arxiv.org/abs/2211.17192>
2. Chen et al., “Accelerating Large Language Model Decoding with Speculative
   Sampling,” 2023. <https://arxiv.org/abs/2302.01318>
3. Miao et al., “SpecInfer: Accelerating Generative Large Language Model Serving
   with Tree-based Speculative Inference and Verification,” 2023.
   <https://arxiv.org/abs/2305.09781>
4. Cai et al., “Medusa: Simple LLM Inference Acceleration Framework with Multiple
   Decoding Heads,” 2024. <https://arxiv.org/abs/2401.10774>
5. Li et al., “EAGLE: Speculative Sampling Requires Rethinking Feature
   Uncertainty,” 2024. <https://arxiv.org/abs/2401.15077>
6. Li et al., “EAGLE-2: Faster Inference of Language Models with Dynamic Draft
   Trees,” 2024. <https://arxiv.org/abs/2406.16858>
7. Zhang et al., “Draft & Verify: Lossless Large Language Model Acceleration via
   Self-Speculative Decoding,” 2023. <https://arxiv.org/abs/2309.08168>
8. He et al., “REST: Retrieval-Based Speculative Decoding,” 2023.
   <https://arxiv.org/abs/2311.08252>
9. Yang et al., “Inference with Reference: Lossless Acceleration of Large Language
   Models,” 2023. <https://arxiv.org/abs/2304.04487>
10. Stern et al., “Blockwise Parallel Decoding for Deep Autoregressive Models,”
    NeurIPS 2018. <https://arxiv.org/abs/1811.03115>
11. Fu et al., “Break the Sequential Dependency of LLM Inference Using Lookahead
    Decoding,” 2024. <https://arxiv.org/abs/2402.02057>
12. Chen et al., “Sequoia: Scalable, Robust, and Hardware-aware Speculative
    Decoding,” 2024. <https://arxiv.org/abs/2402.12374>
13. Mamou et al., “Dynamic Speculation Lookahead Accelerates Speculative Decoding
    of Large Language Models,” 2024. <https://arxiv.org/abs/2405.04304>
14. Elhoushi et al., “LayerSkip: Enabling Early Exit Inference and Self-Speculative
    Decoding,” 2024. <https://arxiv.org/abs/2404.16710>

Official implementation documentation useful for integration constraints:

- Hugging Face Transformers, “Speculative decoding / assisted generation.”
  <https://huggingface.co/docs/transformers/main/generation_strategies#speculative-decoding>
- vLLM project, “Speculative Decoding.”
  <https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/README.md>
