# Chunk graph contracts and phase diagnostics — 2026-08-18

This directory preserves byte-for-byte four immutable A100 artifacts from clean
commit `e50e732568727ae9127eac22d23c7dd7054e6b71`, tree
`1fa18dea8d797d80cd8812d5c995c951030936d2`, source SHA-256
`ba43d751ae343d47f5efd642bf070bacf64f86931e10c205474dbfa2d8e74db8`,
and model SHA-256
`0c659d1dba2804b0943c24bbece1858e93273147f7a11693fa99062df8c5997b`.

The 4x511 boundary contract passed with 2,044 real unpadded tokens, selected
graph key `(2048, 5)`, zero graph misses, identical eager/graph/end-to-end
tokens, and zero maximum logit difference. The tau `{64,128}` by
`max_model_len={512,1024,4096}` matrix passed all six cells. Tau 64 correctly
uses eager routing with no captured bucket and one routed miss; tau 128 uses
graph key `(128, 9)` with zero misses. Every cell matched eager output.

| tau | steady wall median / p95 / max (ms) | measured routes | selected keys | graph misses |
| ---: | ---: | --- | --- | ---: |
| 256 | 5.254 / 6.514 / 7.209 | 85 decode graph, 11 varlen graph | `(64)`, `(256,64)` | 0 |
| 512 | 5.252 / 5.441 / 9.085 | 91 decode graph, 5 varlen graph | `(64)`, `(512,64)` | 0 |

Both phase runs record the engine-owned GC lifecycle `enabled → disabled →
enabled`. They are intrusive, bounded step diagnostics. They do not measure the
full request ITL contract, and both explicitly retain
`current_run_latency_certified=false`. They therefore cannot override the
negative full-completion verdict.

`archive_provenance.json` records every original path, size, read-only mode, and
SHA-256. Validate the retained copies from the checkout root:

```bash
PYTHONPATH=. /venv/main/bin/python -B \
  benchmarks/chunked_prefill_tail/validate_retained_evidence.py \
  benchmarks/chunked_prefill_tail/evidence/2026-08-18-a100-contract-phase-e50e732
```
