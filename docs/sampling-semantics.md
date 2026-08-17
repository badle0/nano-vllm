# Sampling semantics and performance gates

## Top-k

`top_k=-1` disables top-k filtering. Positive values keep every token whose
logit is greater than or equal to the kth-largest logit, so ties at the cutoff
remain eligible. Filtering is applied only to stochastic rows with an effective
`k` smaller than the vocabulary.

## Top-p

`top_p=1.0` disables nucleus filtering. Enabled filtering follows the
`transformers` `TopPLogitsWarper` contract validated with version 5.14.1:

1. Divide active-row logits by the effective temperature in FP32.
2. Sort ascending with the default `torch.sort` tie behavior.
3. Remove tokens while cumulative probability is less than or equal to
   `1 - top_p`.
4. Always keep at least the final sorted token.

The `1 - top_p` cutoff is computed from the Python parameter before FP32 device
transfer. Recomputing it from an already-rounded FP32 `top_p` can change support
at equality-adjacent boundaries. When both filters are enabled, top-k runs
before top-p, matching the corresponding Transformers processor order.

Only active rows enter either optional filter. The final stochastic sampler is
called once with the original full batch shape, so optional filtering does not
alter the number or order of random draws for inactive rows. All-active top-p
work is processed in chunks of 64 rows to bound temporary allocation without
changing per-row support.

## Greedy and top-k repair evidence

The greedy repair is pinned to `ec98870`; the top-k repair is pinned to its
descendant `8759c87`. On the declared A100 environment, three fresh processes
per B=256 sampler route produced the following aggregate values. Cold is the
median first call from unique empty Inductor caches; steady median and p95 are
medians of the per-process statistics; peak is the maximum per-process
incremental allocated-memory peak.

| Route | Cold wall | Steady median | Steady p95 | Peak incremental allocation |
|---|---:|---:|---:|---:|
| homogeneous greedy | 1,171.37 ms | 0.177 ms | 0.185 ms | 0.002 MiB |
| top-k disabled | 1,893.14 ms | 1.011 ms | 1.026 ms | 148.38 MiB |
| one row at top-k 50 | 2,003.54 ms | 1.069 ms | 1.112 ms | 148.38 MiB |
| all rows at top-k 50 | 1,943.67 ms | 2.038 ms | 2.043 ms | 148.38 MiB |

The common stochastic sampler dominates the allocated-memory peak. Activating
one top-k row adds only 5.78% sampler latency over disabled, while all-active
work is 1.91x the one-active route, confirming active-row scaling.

Four additional fresh B=256 end-to-end processes balanced which scenario ran
first. Their paired top-k-50 throughput changes were -6.31%, -5.98%, -6.48%,
and -5.93%; the median paired loss was -6.15%, inside the suggested 10% budget.
The exact harnesses, process order, environment/model pins, raw samples, hashes,
and validator are in
[`benchmarks/sampling_evidence/`](../benchmarks/sampling_evidence/README.md).

## A100 regression gates

The 2026-08-17 repair was measured on an NVIDIA A100-SXM4-40GB with PyTorch
2.10.0, CUDA 12.8, batch 256, vocabulary 151,936, five warmups, and 25 measured
iterations. These gates are intended to detect regressions on comparable
hardware, not to predict latency on other GPUs.

| Route | Steady median ceiling | Incremental scratch ceiling |
|---|---:|---:|
| Sampling disabled | 1.30 ms | 1 MiB |
| One row at top-p 0.9 | 1.75 ms | 16 MiB |
| All rows at top-p 0.9 | 12.0 ms | 650 MiB |

The measured medians were 1.15 ms, 1.41 ms, and 10.18 ms respectively. Active
row isolation therefore fixes heterogeneous-batch scaling, and chunking reduces
the old 1.41 GiB transient. Exact all-active nucleus sampling remains dominated
by the full-vocabulary sort: its measured end-to-end throughput loss was 38.3%
at batch 256. That all-active result is still a performance release blocker
until the project agrees to that cost or adopts an oracle-equivalent fused
selection implementation.
