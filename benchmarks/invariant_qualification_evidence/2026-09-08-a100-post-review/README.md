# A100 risk review and invariant numerical qualification

This directory is the final evidence set for the 2026-09-08 review. All
invariant JSON reports carry implementation SHA-256
`fe0431d97f7f1f0ea1c1063c5339d6dabd174e332d93586cda18ddc553c4f266`.
The baseline source was preserved before review at
`/workspace/nano-vllm-review-snapshots/2026-09-08-pre-risk-review-198e1e6/`.

The `known`, `mixed`, `cache`, `eviction`, and `long` pairs compare independent
fresh engines with forced token histories. Candidate reports contain the
comparison verdict. There are 55 corresponding live-KV states and 55 emission
logit states across the five comparisons; every hash matches bit for bit.
`fp64_primitives.json` independently checks invariant linear, RMSNorm/residual,
and paged attention against FP64 computations.

`fast_known_graph_b256.json` and `fast_known_eager_b256.json` are a deliberately
negative control. The eager command exits nonzero after retaining its report:
requests 9 and 12 still diverge in fast mode. This does not invalidate invariant
mode, whose reference is its own ordinary execution contract.

The speculative reports exercise final-source graph and invariant routes. They
cover 140 and 124 cycles respectively, cache-repeat and streaming parity,
future-token causality, injected failure/retry, cancellation cleanup, and
unchanged compiler counters in every measured cycle. They are lifecycle smoke
tests rather than performance-promotion results.

Only Qwen3-0.6B was available. Qwen3-4B remains unqualified. See
`review_summary.json` for a machine-readable result and `SHA256SUMS` for artifact
integrity.
