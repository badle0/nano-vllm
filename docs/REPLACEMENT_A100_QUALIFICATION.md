# Replacement A100 qualification — 2026-09-09

The invariant-prefill and speculative repair bundle passes the tested fixed-KV
numerical and lifecycle checks and improves speculative throughput over its
pre-fix implementation. **Full qualification remains incomplete: automatic KV
sizing regressed and can refuse initialization.** Integration of this repair
branch must not be described as an unrestricted release qualification.

## Provenance

Before: `198e1e6a084dfb1f59d178a8ca3bdc1d0b094bed` (`fork-main`).
After: `74eac2f9512e737d7eb1f6a1d2c97eebb5efc9ee`, containing repair commit
`3710c76e246d67021cfb84f114af7d26b7be6841`. No runtime edits were made during
qualification. Later documentation/evidence commits retain that runtime.

Hardware: one A100-SXM4-40GB, driver 570.133.20. Python 3.12.13,
Torch 2.10.0+cu128, Triton 3.6.0, FlashAttention 2.8.1, Transformers 5.14.1.
BF16, TP1, exact sampling. Qwen3-4B target revision
`1cfa9a7208912126459214e8b04321603b3df60c`; Qwen3-0.6B draft revision
`c1899de289a04d12100db370d81485cdf75e47ca`. Runtime SHA256:
`fe0431d97f7f1f0ea1c1063c5339d6dabd174e332d93586cda18ddc553c4f266`.

## Correctness and lifecycle

- Invariant eager numerical mode: 55 paired KV hashes and 55 paired logit
  hashes matched bitwise across known failures, mixed boundaries, cache reuse,
  eviction/resume and contexts through 4096. FP64 primitive checks passed;
  this is not a full-model FP64 comparison.
- Initial target/draft smokes passed 254 graph-fast and 248 eager-invariant
  cycles, each sweeping 80 cells. Expanded strict graph verification passed
  261 cycles with unchanged guarded compiler/cache/capture state.
- Graph-fast, eager-fast and eager-invariant runs matched their ordinary
  decoding controls. Cache reuse, streaming, cancellation/GC, fault rollback
  and retry, RNG restoration, causality and scratch checks passed with fixed KV.
- The one-ULP acceptance reproducer failed on the archived pre-review trusted
  sampler and passed on the repaired sampler against the validated reference.
  That pre-review snapshot is distinct from the performance baseline.
- Adaptive routing checks passed with synthetic calibration fixtures; they
  establish routing behavior, not adaptive performance qualification.
- Final full pytest: **1080 passed, 1 skipped**, 14 deprecation warnings,
  185.54 seconds. The earlier five-pair rerun also passed 1080/1.

## Performance

These are complete speculative decoding runs across sampling families, not
isolated sampler timings. Primary runs use fixed K=4, 64 KV blocks, batch
1/4/8, context 32/256, five sampling families and 64 output tokens. Each
comparison uses five fresh process pairs in alternating order, two warmups,
and seeds 17/23/41. Ratios use median seed ratios within each pair followed
by the median of five pairs; descriptive 95% intervals bootstrap those pairs.

| Comparison | Result |
| --- | --- |
| Old speculation vs repaired speculation | All 20 active B1/B4 cells improved **1.41–2.77×**; every interval above 1 |
| Old vs repaired B8 bypass | Ratios 0.996–1.005, approximately unchanged |
| Repaired speculation vs repaired ordinary decoding | **8/30 cells faster, 1.17–1.28×**; active greedy remains **1.42–1.62× slower** |
| Speculation disabled, old vs repaired | B1/B4/B8 latency changes +0.06%/+0.20%/+0.35%, within the declared ±5% band |

The initial on/off run retained 900 timings; the expanded run retained 1110,
with zero exclusions and no greedy/ordinary-control token mismatches. Extended
batch/context/stream/tail and configured-K checks passed, but have only one
process per side. Timings use warmed synthetic prompts; cache hits are possible.
The before/after result measures the whole repair bundle, not individual edits.

## Known automatic-KV regression

At `gpu_memory_utilization=0.5`, the matched batch-5 automatic-sizing constructor
succeeds before the bundle and fails afterward with an 18,972,055-byte modeled
headroom shortfall. Current batch-4 automatic sizing also fails with a
30,621,612-byte shortfall. Initialization is refused by the memory audit.

The matched probe measures 24,310,112 bytes of new persistent speculative
buffers allocated after KV sizing. The final audit still subtracts the full
workspace reservation after those buffers are resident. This is a likely
accounting contributor, not a separately repaired or validated root cause.
Fixed 64-block configurations passed; the automatic-sizing defect needs a
follow-up repair and matched GPU rerun.

## Retained evidence and limits

[Raw archives and checksums](../benchmarks/speculative_optimization_evidence/2026-09-09-a100-migration/README.md)
include execution scripts, per-cell results, failure logs and pytest XML.
The complete-arrival archive contains the detailed before/after table and
baseline source. Earlier benchmark documents remain historical records;
this dated report provides the replacement-instance results.

Claims are limited to this single-GPU Qwen3 configuration. TP>1, other models,
FlashInfer speculation, production traffic and adaptive speedups are not
qualified by these runs. No new runtime fix for automatic sizing is included.
