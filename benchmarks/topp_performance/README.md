# Top-p performance development

This directory contains development-only gates for replacing the exact
Transformers-compatible full-vocabulary top-p sort.  Runtime changes remain on
`fix/topp-performance`; release evidence must be produced later from a clean,
pinned runtime commit on a separate evidence descendant.

The first primitive is a CUDA BF16 counting-sort reconstruction.  It must return
the exact same FP32 ascending value tensor as:

```python
torch.sort(logits.float() / temperatures[:, None], dim=-1).values
```

It is not yet integrated into `Sampler`: recovering a cutoff tie's original
token IDs must be separately proven before production use.

## Exact-path gate results

The exact Transformers-compatible route has two independently measured
development gates on the pinned A100 / Torch 2.10 / CUDA 12.8 stack:

| Primitive, B256 x V151936 BF16 | CUDA median | Result |
|---|---:|---|
| Histogram + prefix + tuned range fill + FP32 divide | 2.57 ms | Exact values, already over the 1.6 ms release budget |
| Same reconstructed values + PyTorch softmax + cumsum | 3.99 ms | Bitwise-exact cumulative tensor, before boundary masking |
| CUB segmented BF16 keys-only sort | 1.738 ms | Exact values, failed the 0.75 ms go/no-go gate |
| PyTorch BF16 values-only sort reference | 3.492 ms | Reference only |

The CUB gate deliberately stops before a fused cutoff kernel: its sort alone is
slower than the complete 1.6 ms target.  The histogram implementation remains a
development oracle/building block, not a production fast path.

A grouped histogram cutoff was also rejected.  It ran in 2.894 ms and matched
255/256 production-shaped Gaussian rows, but the remaining row differed by one
token and forced-tie inputs produced larger errors.  FP32/FP64 agreement and
simple margin checks did not reliably identify those failures, so it must not
be described as Transformers-exact.

## Sorting-free Qrita prototype

`qrita_topp_proto.py` is a development-only adaptation of vLLM's Apache-2.0
standalone top-p Triton path, pinned to vLLM commit
`e0e5a7fb2808504ba86c94f7b379e38496002fd0`.  It keeps the Qrita
sample-statistics, outlier-compaction, and probability-pivot-search structure,
but accepts raw CUDA BF16 logits, applies per-row temperature internally, and
writes masked raw BF16 logits.  It is intentionally specialized to `p=0.9` and
is not imported by the production sampler.

The B256 x V151936, temperature-0.6 A100 gate was also a no-go:

| Complete filter | CUDA median | p95 | Filter scratch/transient |
|---|---:|---:|---:|
| Qrita development prototype | 5.955 ms | 5.967 ms | 62.60 MiB persistent |
| Current exact full-sort filter | 9.155 ms | 9.344 ms | 616.96 MiB peak transient |

Although the prototype was 1.54x faster and filtering was RNG-neutral, it
remained over twice the 2.6 ms interim budget and was not exact.  The production
Gaussian matrix had 14,720 support-bit differences across 38,895,616 logits;
all retained raw values were unchanged.  On the smaller characterization
matrix, Gaussian and peaked inputs had no differing sampled tokens in 512 paired
draws each, forced ties had 4/512, and uniform logits had 56/512 because the
prototype kept the full row while the exact filter retained 7,373/8,192 tokens.
These results rule out production integration without an explicit semantic
contract change and substantially more kernel work.

## Adaptive top-M prototype

`adaptive_topm_bench.py` characterizes a second development-only route.  It
computes full-vocabulary normalization mass, sorts only the largest M BF16
logits, and uses the exact full-sort filter for rows whose nucleus is not
certified to fit.  The certificate also falls back for a numerically marginal
mass decision, an incomplete kept tie at the M boundary, or a cutoff that
splits equal BF16 logits.  It is not imported by the production sampler.

The bounded A100 run used B256 x V151936 BF16, temperature 0.6, p=0.9, three
warmups, and ten measured iterations.  `peaked_model_like` is synthetic: a
broad Gaussian tail plus a sparse 512-logit boosted head, not captured model
output.

| Input | M | Exact fallbacks | Adaptive median / p95 | Exact median / p95 | Adaptive / exact peak transient |
|---|---:|---:|---:|---:|---:|
| Gaussian | 4,096 | 256/256 | 12.764 / 14.868 ms | 10.869 / 10.976 ms | 860.54 / 691.15 MiB |
| Gaussian | 8,192 | 256/256 | 13.117 / 13.155 ms | 9.324 / 9.509 ms | 880.63 / 691.15 MiB |
| Peaked/model-like | 4,096 | 193/256 | 10.510 / 10.686 ms | 9.312 / 9.404 ms | 824.03 / 691.15 MiB |
| Peaked/model-like | 8,192 | 193/256 | 10.899 / 11.012 ms | 9.304 / 9.558 ms | 844.12 / 691.15 MiB |

All four configurations had zero support differences, zero retained-value
differences, and zero false-certified rows because failed rows took the exact
fallback.  Across eight fixed seeds per configuration (8,192 row-level token
comparisons total), sampled tokens and final CUDA RNG state also matched.  The
Gaussian input never used the fast route; only 63/256 peaked rows did so.  The
dominant peaked fallback was the pinned unstable `torch.sort` cutoff-tie
identity, which cannot be reproduced safely by a separate unstable `topk`
ordering.

This route is a no-go: it is slower and consumes more transient memory even on
the peaked input, and M=8,192 makes both worse.  The equality results validate
the fallback, not the proposed optimization.

## FlashInfer complete-sampling candidate

`flashinfer_sampling_bench.py` measures a different contract rather than an
approximate mask.  It compares the complete current path (exact
Transformers-compatible `filter_top_p` followed by the existing compiled
`Sampler.forward`) with FlashInfer's sorting-free direct sampler.  The original
development run measured `sampling.softmax` followed by
`top_p_sampling_from_probs(deterministic=True)`.  The current harness also
measures the production `Sampler.sample_top_p_flashinfer` wrapper—including its
greedy argmax, dtype normalization, and final `torch.where`—as a separate
acceptance route.  The exact work buffer is restored before each event as
benchmark setup; the restore is excluded because production receives fresh
model logits and does not clone them.  All routes are JIT-warmed before the
five warmups and 25 measured iterations.

The pre-integration B256 x V151936 BF16, temperature-0.6, p=0.9 A100 run
established enough primitive headroom for integration:

| Complete sampling route | CUDA median | p95 | Peak transient allocated |
|---|---:|---:|---:|
| Current exact filter + sampler | 11.648 ms | 11.768 ms | 616.96 MiB |
| FlashInfer direct primitive | 0.913 ms | 1.058 ms | 296.75 MiB |

That is a 12.75x primitive speedup, but it is not by itself the shipped-wrapper
release gate.  FlashInfer 0.6.17 was loaded from an isolated development target
and identifies source commit
`a0a6b019b9b27d49d209f85d028a1ae5a9b347d7` (Apache-2.0).

This result is explicitly **not exact-backend equivalence**.  Resetting to the
same seed reproduced tokens and final RNG state within each route, but the two
routes differed on 256/256 production-shape token draws.  From the common CUDA
generator state at offset 0, the current compiled sampler ended at offset 4
and FlashInfer ended at offset 8,192.  Boundary ties also have different token
identity rules: for the eight-way uniform p=0.6 case, the pinned current
`torch.sort(stable=False)` path retained token IDs `[0, 1, 2, 3, 5]`, while
FlashInfer sampled all eight symmetrically; for three tied maxima at p=0.5,
the exact route retained `[0, 1]`, while FlashInfer sampled `[0, 1, 2]`.

The statistical contract passed its focused sanity check.  On a known
eight-token unique-logit distribution, 131,072 FlashInfer draws had zero draws
outside the exact three-token nucleus and a maximum absolute standardized
residual of 1.79.  The per-row heterogeneous-temperature softmax API matched
PyTorch within `1.1921e-7`.  These results support an opt-in statistical backend
with the exact backend remaining the default; they do not support silently
replacing the exact fixed-seed/tie contract.

Raw evidence is
`/workspace/.feat_bench/results/flashinfer_topp_sampling_b256_v151936.json`
(12,011 bytes, SHA-256
`fd3e68045b7bb7177f354b90a1d9ecd2b2c99d5d42efe7056466299e60fe4e08`),
recorded from base commit
`42affe47b3a97bb0ef2470f31c1e8b311cbc7a21` on
`fix/topp-performance`.  The JSON embeds all CUDA samples, environment and GPU
versions, resolved FlashInfer cache/package paths, git status, API provenance,
RNG states, and statistical counts.  A clean-commit rerun of the updated
harness is required to claim the production wrapper's 1.6 ms gate.

Run its focused tests and development benchmark with:

```bash
PYTHONPATH=. pytest -q tests/test_topp_histogram.py

PYTHONPATH=. /venv/main/bin/python \
  benchmarks/topp_performance/histogram_sort_bench.py \
  --output /tmp/topp_histogram_sort.json

PYTHONPATH=. /venv/main/bin/python \
  benchmarks/topp_performance/cub_keys_sort_bench.py \
  --output /tmp/topp_cub_keys_sort.json

PYTHONPATH=. /venv/main/bin/python \
  benchmarks/topp_performance/qrita_topp_bench.py \
  --output /tmp/topp_qrita_p90_bf16.json

PYTHONPATH=. /venv/main/bin/python \
  benchmarks/topp_performance/adaptive_topm_bench.py

PYTHONPATH=/tmp/nv_flashinfer_proto_nodeps:. \
FLASHINFER_WORKSPACE_BASE=/tmp/nv_flashinfer_cache \
/venv/main/bin/python \
  benchmarks/topp_performance/flashinfer_sampling_bench.py \
  --batch 256 --vocab 151936 --temperature 0.6 --top-p 0.9 \
  --warmups 5 --iterations 25 --statistical-draws 131072 \
  --expected-commit "$(git rev-parse HEAD)" \
  --output \
  /workspace/.feat_bench/results/flashinfer_topp_sampling_b256_v151936.json
```

The resulting production policy is therefore two explicit backends: retain the
existing full-sort implementation as the default `exact` contract, and expose
FlashInfer only through an opt-in `flashinfer` contract.  The grouped,
histogram, CUB full-sort, Qrita-mask, and adaptive top-M candidates remain
development evidence; none is imported by the production sampler.
