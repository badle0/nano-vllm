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
