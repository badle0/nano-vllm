# Incremental-allocation graph smoke

`graph256.json` is a source/model-pinned A100 correctness run after incremental
KV allocation. It passed real batch-17 decode capture, a 17-block mixed pressure
workload, complete ownership release, and forced missing-graph eager fallback.
The short request emitted on every consecutive scheduler step; the long prefill
required no preemption, and the workload drained in 18 steps.

The report deliberately sets `latency_certified=false`. Torch Dynamo emitted the
retained RMSNorm rank-specialization recompile-limit warning during setup, so
this artifact is neither recompile-free nor a replacement for the five-process
tau-256 latency gate. `SHA256SUMS` covers the JSON report.
