# Request-metrics A100 certification — 2026-08-18

This directory preserves byte-for-byte the original manifest and 16 raw process
outputs from `metrics-cert-b3efbdb-a100-20260818`. The comparison used main
`bb823b3e06983d71485a8e1f23715ebd87d98ef8` and metrics repair/certification
`b3efbdba905aa8006f8f586204edf19bd5670186` on one A100-SXM4-40GB.

Eight fresh-process pairs alternated baseline-first and repair-first. Each side
measured batches 64 and 256, 128 prompt tokens, 32 output tokens, one cold plus
five steady observations. Prompt, RNG, and output-token hashes match within
every pair.

| Case | Median paired efficiency | Paired-log Student-t 90% CI | TOST within `[0.98, 1.02]` |
|---|---:|---:|---:|
| B64 | 0.997971 | [0.989955, 1.002648] | pass |
| B256 | 0.998272 | [0.996415, 1.001430] | pass |

The 90% interval is the standard confidence interval for two one-sided tests at
alpha 0.05. It is computed on `log(T_repair / T_baseline)` with 8 paired process
ratios, 7 degrees of freedom, and `t(0.95, 7) = 1.894578605061305`, then
transformed back to the efficiency scale. Both entire intervals lie inside the
predeclared equivalence bounds.

Validate all archived hashes, raw observations, summaries, pair ratios, process
ordering, token parity, aggregate gates, equivalence intervals, and the current
model fingerprint with:

```bash
/venv/main/bin/python -B benchmarks/request_metrics/validate_certification.py \
  benchmarks/request_metrics/certification/2026-08-18-a100-b3efbdb/manifest.json \
  --check-model
```

`archive_provenance.json` records the original external directory, manifest
digest, every archived artifact digest, and exact recomputed statistics. The
original manifest and raw files were not edited to add the later TOST analysis.
