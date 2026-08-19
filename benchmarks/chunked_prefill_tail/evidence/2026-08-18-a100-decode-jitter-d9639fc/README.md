# A100 decode-jitter attribution at `d9639fc`

This directory retains the exact read-only bytes produced by the intrusive
decode-only diagnostic. The artifact SHA-256 is
`6e8236dfb68c4ab42047c39ade251bad27d0697d2767d8bac215abd57092de79`.
It is pinned to commit `d9639fc5604f88069edbf0c36e71d0f0c83b3c91`, source
SHA-256 `5b2a4946700b72a0e1472f8dc21410cd7477396f6c975a8c86a9311d44b7b55e`,
Qwen3-0.6B model SHA-256
`0c659d1dba2804b0943c24bbece1858e93273147f7a11693fa99062df8c5997b`,
and one A100-SXM4-40GB run with 1,000 decode steps at each batch size.

Eight of 2,000 API steps were strictly above 10 ms:

| Batch | Step | API wall ms | Thread CPU ms | Runner CUDA ms | Model CUDA ms | API residual ms | Attribution |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 16 | 519 | 12.799 | 12.799 | 3.879 | 3.386 | 12.204 | post-enqueue/token-sync residual |
| 16 | 622 | 10.676 | 10.676 | 10.555 | 10.027 | 0.417 | model CUDA span |
| 16 | 787 | 13.469 | 13.469 | 4.341 | 3.750 | 12.736 | post-enqueue/token-sync residual |
| 16 | 952 | 11.883 | 11.883 | 4.416 | 3.926 | 11.295 | post-enqueue/token-sync residual |
| 18 | 49 | 11.811 | 11.811 | 4.307 | 3.804 | 11.209 | post-enqueue/token-sync residual |
| 18 | 119 | 13.427 | 13.427 | 4.432 | 3.931 | 12.823 | post-enqueue/token-sync residual |
| 18 | 677 | 10.833 | 10.833 | 5.260 | 4.744 | 10.226 | post-enqueue/token-sync residual |
| 18 | 757 | 15.279 | 15.279 | 5.388 | 4.871 | 14.662 | post-enqueue/token-sync residual |

For seven rows, runner CUDA remained below that profile's p99 while more than
10 ms remained outside the individually timed host phases, after runner-end was
enqueued and across the production `tokens.tolist()` synchronization/API glue.
The remaining row, batch 16 step 622, had a 10.027 ms model CUDA span and only
0.417 ms API residual.

All eight stall rows recorded zero voluntary and involuntary context switches;
thread CPU and wall differed by at most 0.000209 ms. That rules against a
recorded scheduler deschedule or sleeping wait and is most consistent with
busy CPU/CUDA-driver polling. The retained observer cannot identify the exact
driver call without a lower-level trace.

This result is **intrusive diagnostic evidence only**. It does not certify the
10 ms request-ITL SLO, alter the negative full-completion verdict, establish a
production regression, or authorize a production-code change.

Validate the exact bytes, pins, workload, raw route invariants, and recomputed
eight-row attribution with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /venv/main/bin/python \
  benchmarks/chunked_prefill_tail/validate_decode_jitter_evidence.py
```
