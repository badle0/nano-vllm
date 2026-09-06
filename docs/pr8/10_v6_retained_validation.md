# V5/V6 retained A100 certificate

Runtime and harness producer: `a165b654660e60ca60cecd47b89c95de1d65735b`.
This supersedes the V4 shadow-only scope with actual verification, rejection,
correction/bonus and burst commit. It does not certify performance or every
possible checkpoint/configuration. See [the execution contract](09_v5_verified_execution.md).

Archive: [2026-09-06-a100-v5-v6-a165b65](../../benchmarks/speculative_v5/evidence/2026-09-06-a100-v5-v6-a165b65/manifest.json).
Manifest SHA256:
`1c8e3b81f18e74a1d2f6b89601213c5d239f75b771646e88ef13f6b06f79a302`.

## Environment and protocol

One A100-SXM4-40GB; Torch 2.10.0+cu128, Python 3.12.13. Exact package/driver
versions, model hashes, source hashes and compiler environment are in each raw
JSON. Target and draft are the local Qwen3-0.6B checkpoint with weight SHA256
`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.

Every cell ran in a fresh process from a clean producer commit with empty,
distinct Inductor/Triton directories. Remote compilation/autotuning/PGO caches
were disabled. Every speculative interval compared full compiler counters,
guard-failure and graph-break records, compiler-file hashes and graph-capture
counts before/after. Aggregate ordinary-path compilation outside these intervals
is not claimed unchanged; those logs are retained too. Instrumented interval
times are **not** headline performance measurements.

Explicit-pool runs use 64 KV blocks, model length 512, token budget 1024, memory
utilization 0.5, and configured K=4. Eager uses max batch 4; graph uses max batch
5 so the first above-cap batch can be tested. Matching off controls use the same
mode/configuration. The separate automatic-pool graph regression uses K=2.

## Results

The complete CPU suite at this checkpoint: **1,224 passed, 31 skipped**; the
skips are not reported as GPU passes. Fourteen upstream Torch deprecation
warnings were emitted.

| Enabled cell | Cycles | KV blocks | Maximum normal-cycle incremental allocation | Modeled live peak | Reserved workspace |
|---|---:|---:|---:|---:|---:|
| Eager, explicit | 129 | 64 | 169,925,632 B | 179,894,188 B | 247,003,052 B |
| Graph, explicit | 135 | 64 | 169,925,632 B | 224,867,735 B | 291,976,599 B |
| Graph, automatic, K=2 | 71 | 301 | 58,772,480 B | 106,964,092 B | 174,072,956 B |

Both explicit enabled cells cover all 80 combinations of B=1..4, K=1..4 and
greedy/plain/top-k/top-p/combined sampling: **160 cells total**. All **335**
recorded speculative intervals were compiler/cache/guard/capture-neutral. Both
target lanes ran. All normal-cycle peaks fit the corrected live estimate,
not merely the allocator margin. The isolated sort-scratch probes also fit the
new two-payload allowance. Graph auto-sizing retained 52,794,756 B of modeled
headroom after reservations; the older failed V4 experiment remains historical
evidence, not a claim that auto-sizing can never fail on another configuration.

The following passed:

- Same-mode speculation-off/on greedy token equality; eager and graph are not
  required to be bit-identical to each other.
- Prompt boundaries 255/256/257, ragged B=1..4, prefix-cache replay, streaming
  token and real-tokenizer `TextUpdate` reconstruction parity.
- Pure stochastic parallel verification and mixed greedy/stochastic batches.
- Perturbing the final draft proposal leaves every earlier target distribution
  bit-identical while changing the bonus distribution: a non-vacuous causal test.
- One-above-cap graph batch falls back without any speculative cycle.
- Close before first step, midway through pending events, and after completion;
  abandoned pending-burst finalization and lease reuse.
- Compute/delivery timing separation and zero intra-burst compute ITLs.
- Forced empty-residual recovery exactly once per enabled run, with no natural
  fallback elsewhere. Zero observed events is not a universal statistical bound
  on unseen models or future runs; no global TV-error guarantee is inferred.
- Failure after real target verification restores host state and CPU/CUDA RNG;
  a manual-step retry yields the expected greedy continuation and releases KV.

The independent CPU law and state-machine tests additionally cover frequent
rejection with deliberately mismatched toy target/draft models, every rejection
position, EOS/ignore-EOS and completion tails, stale result rejection, append/
hash/trim/event failures, cache-eviction rollback and tensor-traceback lifetime.

## Recheck

```bash
python benchmarks/speculative_v5/validate_retained_evidence.py \
  benchmarks/speculative_v5/evidence/2026-09-06-a100-v5-v6-a165b65
```

This needs Git history but no Torch, GPU, model files or network. The validator
pins the manifest, verifies raw JSON/log hashes and historical runtime/harness
identity, checks route coverage/work counts/memory and residual results, and
compares matching-mode controls. Tamper tests cover both integrity and semantic
failures. Old V2/V3/V4 archives remain independently pinned to their own source.

V7 must separately qualify the required 4B/0.6B pair, uninstrumented performance,
roofline assumptions and opt-in limitations. No V7 performance claim is made by
this certificate.
