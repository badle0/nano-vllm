# Automatic-KV accounting repair — 2026-09-09

The automatic-sizing failures recorded in the [replacement A100 qualification](REPLACEMENT_A100_QUALIFICATION.md) are repaired in `c15dd52b030396466ca7e632aea149552edb6413`. All five fresh GPU lifecycle sweeps and the full pytest rerun pass within the tested Qwen3 single-GPU envelope.

## Change

The optimized speculative decoder keeps five reusable buffers resident across runtime phases. KV sizing previously reserved the legacy phase workspace but omitted this additional permanent ownership. The final audit then measured those live buffers and correctly refused initialization near the automatic KV boundary.

Sizing now reserves the persistent buffers separately from the conservative legacy phase workspace. Each buffer receives a 2 MiB allocator-segment allowance, then the total is rounded up to whole joint target/draft KV blocks. For Qwen3-4B / Qwen3-0.6B, 24,310,112 bytes of tensor storage require a 30 MiB segment allowance, rounded to one 64 MiB joint block. The audit records this as `persistent_workspace_reservation_bytes`.

This future-allocation reserve participates in the runtime sizing envelope. The final audit measures resident ownership directly and does not subtract the new reserve again. Its existing transient, workspace, graph-peak and headroom checks remain intact. Buffer allocation, decoding, acceptance and sampling execution are unchanged from the previously benchmarked runtime.

The policy deliberately trades some KV capacity for conservative accounting. At batch limit 5 in graph mode it selects 157 blocks instead of the failing 158-block configuration. It does not claim a minimal allocator-footprint model or a new throughput improvement.

## Fresh GPU verification

A100-SXM4-40GB, driver 570.133.20, Python 3.12.13, Torch 2.10.0+cu128, BF16, TP1. Target Qwen3-4B / draft Qwen3-0.6B revisions are unchanged from the linked qualification. Utilization 0.5, configured K4, context limit 512 and token budget 1024. Each run uses a fresh process with isolated compiler caches and remote compiler caches disabled.

| Run | KV blocks | Speculative cycles | Sweep cells | Final modeled headroom (bytes) |
| --- | ---: | ---: | ---: | ---: |
| graph-auto-b5 | 157 | 261 | 80 | 48,136,809 |
| graph-auto-b4 | 158 | 254 | 80 | 36,487,252 |
| eager-auto-b5 | 161 | 248 | 80 | 33,456,745 |
| invariant-auto-b4 | 162 | 248 | 80 | 62,177,364 |
| graph-fixed-b5 | 64 | 261 | 80 | 6,289,261,161 |

Graph runs use strict retained compiler/cache/capture snapshots; all guarded cycles are unchanged. All five sweeps cover lifecycle rollback/retry, cancellation/GC, cache reuse, streaming parity, causality and memory checks. Eager and invariant cycles also show no compile-counter changes. The fixed-KV run uses 64 blocks to check the previously qualified configuration. All 20 greedy-control workloads match the retained ordinary-decoding outputs in their corresponding graph, eager or invariant mode; these reference runs were produced before the accounting-only change.

Focused memory/runner tests: 91 passed. Final full pytest: **1086 passed, 1 skipped**, zero failures/errors. Earlier five-pair performance results remain measurements of the earlier runtime; this repair received fresh lifecycle/regression verification, not another five-pair timing qualification.

## Retained attempts and limits

Moving allocations before sizing alone (`6ec2e3d`) still failed with a 10,583,447-byte shortfall because allocator reuse did not reliably expose additional ownership in the sizing snapshot. Explicit segment reservation without whole-block rounding (`d48141d`) passed graph batch 4/5 and eager batch 5 but failed invariant automatic sizing by 4,931,500 bytes. Both attempts are superseded; their logs and the original constructor probes are retained alongside the final passing evidence.

[Raw evidence, analysis and checksums](../benchmarks/speculative_optimization_evidence/2026-09-09-automatic-kv-repair/README.md) preserve the exact repair diff and source hashes. This verifies the recorded A100/Qwen3 configuration, not all GPU allocators, model geometries, memory-utilization settings or tensor-parallel deployments. The audit continues to fail safely if measured ownership exceeds the modeled budget.
