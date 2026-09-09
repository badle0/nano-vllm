# High-risk and invariant-numerical review

## Scope and reproducible baseline

This review covers the first two delivery steps on `fork-main`, based on commit
`198e1e6a084dfb1f59d178a8ca3bdc1d0b094bed` and its then-dirty working tree.
Before making review fixes, the complete source state was preserved at
`/workspace/nano-vllm-review-snapshots/2026-09-08-pre-risk-review-198e1e6/`.
Its `source.tar.gz` SHA-256 is
`5aa9c4057f96565d407e8c994a3612ea082389c8a92589405c9da4f20c59d3f7`.
The final evidence is in
[`2026-09-08-a100-post-review`](../benchmarks/invariant_qualification_evidence/2026-09-08-a100-post-review),
and all invariant reports bind implementation SHA-256
`fe0431d97f7f1f0ea1c1063c5339d6dabd174e332d93586cda18ddc553c4f266`.

## High-risk review results

| Area | Finding | Disposition and evidence |
| --- | --- | --- |
| Exact rejection sampling | The trusted path used `p(d)/q(d)` on rounded FP32 rows, while the validating oracle used `(p(d)/sum(p))/(q(d)/sum(q))` in FP64. FP32 row sums are not guaranteed to equal one, and a crafted one-ULP boundary changes accept/reject. | Normalize both row masses in FP64 before the trusted ratio. Fixed-randomness tests compare the trusted path with the validating oracle, including the boundary case. Residual correction remains `normalize(max(p-q, 0))` in FP64 with the target-law recovery path. |
| Reusable buffers | Cycle result packing allocated a fresh `torch.cat`; readiness did not prove exclusive ownership of all p/q/result workspaces. | Add one runner-owned integer result buffer, fill it in place, and retain one consolidated device-to-host result transfer. Readiness checks dtype, shape, device, contiguity, and pairwise non-aliasing. The certificate binds the exact workspace storage identities, so replacement fails closed. |
| KV allocation and rollback | Capacity-failure returns were atomic, but an exception from `_allocate_block` could leave a partially attached initial allocation or extension. | Snapshot affected block metadata, free-list order, used membership, hash ownership, sequence block table, and target/draft coverage; restore them before re-raising. Fault-injection tests cover both initial allocation and extension. Existing speculative temporary reservations were reviewed and retain their validated rollback/commit transaction. |
| Graph readiness | A warmed `(batch, K)` set was insufficient: the certificate used graph keys but could survive same-key graph or buffer replacement. A failed re-warm could also leave an old certificate live. | Bind schema, numerical mode, route-plan identity, exact graph objects, graph input buffers, target/draft KV, and speculative workspace storage into the readiness fingerprint. Re-warm revokes the prior certificate before doing work. The engine checks readiness before KV reservation or RNG mutation. Tests replace each resource class and require fail-closed behavior. |

The graph-mode GPU smoke completed 140 cycles and the invariant eager smoke
completed 124. Both passed cache-repeat/stream parity, future-token causality,
injected verification failure followed by identical retry, cancellation cleanup,
and stable per-cycle compiler counters. Graph fast mode exercised sequential
greedy and parallel stochastic verification; invariant mode exercised parallel
greedy and parallel stochastic verification. These are correctness and lifecycle
checks, not speed claims.

## Numerical qualification result

Fast mode still reproduces the original graph/eager disagreement, which is the
expected negative control. Exactly two of 17 synthetic requests diverge:

| Request | Identical-prefix point | Graph | Eager |
| --- | --- | --- | --- |
| 9 | First prompt output | token 3988 at 8.0; tokens 353/9 at 7.96875 | tokens 3988/353/9 tie at 8.375; argmax selects 9 |
| 12 | First decode after prefill | token 38297 at 14.0625; token 25 at 13.9375 | tokens 38297/25 tie at 14.0; argmax selects 25 |

Invariant mode was then tested with independent engines and forced token
histories, so a first divergent generated token could not contaminate later
comparisons:

| Scenario | Reference → candidate | Coverage | Result |
| --- | --- | --- | --- |
| Known failures | packed budget 68 → serial budget 4 | 17 prompts, 34 KV and 34 logit states | Bitwise match |
| Mixed batches and block boundaries | packed budget 2305 → packed budget 257 | lengths 1, 255, 256, 257, 511, 512, 513; 14 KV/logit states | Bitwise match |
| Prefix-cache reuse | serial budget 1024 → serial budget 257 | cold seed plus 512-token reuse; 2 KV/logit states and repeated logical-state check | Bitwise match |
| Eviction/resumption | uninterrupted budget 769 → budget 257 with injected eviction after 257 processed tokens | final two forced states after release and resumed prefix lookup | Bitwise match |
| Long contexts | serial budget 4096 → irregular budget 511 | lengths 1024, 2048, and 4096; 3 KV/logit states | Bitwise match |

Across the matrix, all 55 corresponding live-KV hashes and all 55 corresponding
emission-logit hashes agree. Positions, logical prefixes, emission ownership,
and physical block tables are recorded in each report; comparisons key on the
logical token history rather than physical block IDs.

Independent FP64 checks cover fixed-geometry linear operations, RMSNorm plus
residual, and paged attention at context lengths 1, 255, 256, 257, and 4096.
All pass. Maximum peak-relative errors are 0.0023494 for linear, 0.0024396 for
RMSNorm, and 0.0021510 for attention. This detects agreement with a higher
precision computation, while the cross-shape hashes detect execution-shape
invariance. The FP64 work is a primitive reference, not a full model executed
end to end in FP64.

## Validation and limits

The final test suite result is **1,080 passed, one skipped** on A100-SXM4-40GB
with Torch 2.10.0+cu128. The expected nonzero exit from the fast eager negative
control is retained as evidence, not counted as a suite failure. `git diff
--check` is clean.

Only `/workspace/models/Qwen3-0.6B` is present. Qwen3-4B could not be rerun, so
this review does not certify the declared 4B envelope. The numerical matrix is
BF16, TP=1, exact sampling, A100, and contexts through 4096. Fast mode remains
the default and keeps its known graph/eager differences. The invariant and
speculative smoke timings are not performance-promotion evidence; the five-pair
Qwen3-4B/Qwen3-0.6B benchmark gate remains outstanding.

Reproduce the core numerical pair with new output paths:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /venv/main/bin/python \
  benchmarks/invariant_qualification.py \
  --model /workspace/models/Qwen3-0.6B --scenario known_failures \
  --budget 68 --layout packed --output /tmp/known-reference.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /venv/main/bin/python \
  benchmarks/invariant_qualification.py \
  --model /workspace/models/Qwen3-0.6B --scenario known_failures \
  --budget 4 --layout serial --compare /tmp/known-reference.json \
  --output /tmp/known-candidate.json
```
