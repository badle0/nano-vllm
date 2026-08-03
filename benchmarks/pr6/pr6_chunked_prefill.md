# PR 6 (chunked prefill via graphed mixed steps) — design and validation record

Branch: feat/chunked-prefill (C1–C5 over dev 317e6f0, stacked on the streaming PR).
Hardware/env: Vast.ai A100 SXM4-40GB (host1, EPYC 7713 32 vCPU, C.45901419),
torch 2.10.0+cu128, flash-attn 2.8.1, Qwen3-0.6B bf16.
Protocol: byte-pinned instruments (bench_latency.py fetched from metrics-artifacts
5b6f013 via `git show`, never edited); one host per comparison; first run discarded;
interleaved A/B for every throughput claim; predictions registered with falsifiers
before each measurement and scored after (docs/pr6/05 §3/§3b).

## What this adds

Decode-first mixed-step scheduling (Sarathi-style) with CUDA-graphed ragged steps:
every step admits all running decoders first (1 token each), then fills the
remaining token budget τ with FIFO prompt chunks — at most one partial chunk per
step, at most one mid-chunk sequence system-wide. Any step containing prefill work
uses the varlen layout and replays a bucketed CUDA graph (T_pad ∈ {128…2048},
two slot tiers per bucket); the pure-decode path is untouched, byte-for-byte.

The metric this exists to move: a long prompt's prefill no longer inserts itself
into every running decoder's inter-token gap. The worst gap becomes ~one step at τ
— a chosen constant, not a function of arriving traffic.

## Headline (pinned instrument, before_host1 → after_host1)

| metric (default τ=16384)  | before | after  |
|---|---|---|
| interactive TTFT p50      | 27.4 ms | **9.2 ms** (3.0×) |
| interactive max_ITL p50   | 45.8 ms | 34.9 ms (spike 8.9× → 7.2×) |
| long-prompt TTFT          | 39.4 ms | 34.9 ms |
| interactive mean_ITL      | 5.0 ms  | 4.9 ms (untouched, by design) |

The ITL win is τ-tunable (the point of the feature): **max_ITL 35.9 / 30.2 / 18.0 /
7.8 ms at τ = 16384 / 2048 / 1024 / 512** — monotone in τ where the pre-PR curve was
flat (~40–49 ms regardless). At τ=512 the stall spike is 1.6×. Long-prompt TTFT
pays the designed price as chunks serialize (64.3 ms at τ=512, +63% on this host).

The mechanism's signature: before, max_ITL ≈ TTFT_long + ITL (decoders waited out
the whole prefill plus their own step); after, **max_ITL == TTFT_long exactly**
(34.9 == 34.9) — the stall and the prefill are the same mixed step.

Throughput: −0.5% at default τ on bench.py, attributed by paired phase-split to the
short-segment M-block tax decode rows pay inside 16k-token eager steps (+13 ms per
mixed step; decode phase byte-identical). Accepted as the classic chunked-prefill
trade; quiet-host band re-baselined 8600–8650 → 8560–8610 and confirmed
(8595/8586/8577 at loadavg ~6). Streaming (PR 5) claims re-validated under mixing:
TTFT collapse 26.3× → 230.3×, null-consumer +0.82%, slow-consumer shape identical.

## Design decisions (fork verdicts, docs/pr6/04; deviations evidence-driven)

- **F1 hybrid step representation**: pure-decode path untouched forever; any step
  with prefill work uses the varlen layout with a per-seq `is_prefill` branch.
- **F2 decode-first budget**: dev's decode-admission loop verbatim, run first; FIFO
  chunk fill at `min(work, remaining)`; ≤1 partial and ≤1 mid-chunk fall out as
  invariants (asserted, not tuned). One addition beyond the design doc: preemption's
  `appendleft` can land in front of the mid-chunk head, so `schedule()` rotates the
  unique live-block_table waiting seq back to the head.
- **F3 bucketing, corrected by measurement**: 1-D T_pad buckets {128…2048} — the
  drafted 4096 bucket was dropped (P12: its replay floor ≈ paged-eager; host1 is
  past the E2 dispatch/GPU crossover at that size, and C3's chunking bounds steps to
  τ anyway). Zero-length slot padding is NOT free (~0.006–0.043 ms/slot, scaling
  with M-blocks — P12/P13, correcting P1's correctness-only evidence): each bucket
  captures two slot tiers, lean 64 and full max_num_seqs+1, routed by live segment
  count. Buckets join the decode graphs' shared memory pool.
- **F4 surface**: StepOutput carries explicit num_prefill_tokens/num_decode_tokens;
  step() keeps its legacy (finished, num_tokens) shape via a documented shim.
- **F5 two-axis equivalence**: bitwise only within one kernel launch shape (graph vs
  its capture-identical eager launch — held per bucket per tier); everything
  cross-template/cross-composition judged at token level with fp64 tie adjudication.
  Measured outcomes: C2's branch change produced only EXACT fp64 ties (gap
  0.000000); C3's decode-row kernel swap produced one 2-ULP tie flip (gap 0.125 at
  logit 19.625, pair top-2 under both numerics — inside the P4d envelope).

## Incidental find, upstreamable independently

`ModelRunner.__init__` compiles all `@torch.compile` modules inside its bf16
default-dtype window and restores fp32 after; every init-era Dynamo cache entry
carries a dead `GLOBAL_STATE default_dtype` guard. Upstream pays a hidden ~multi-
hundred-ms recompile storm on the first real request (written off as cold start);
varlen graphs deferred it onto the first bucket-miss step where the benchmark could
see it (P10/P11: 1075 → 293 ms). Fixed by a post-restore pre-touch forward with
production-shaped inputs — including tensor provenance: pre-warm inputs must be
created OUTSIDE inference_mode or rotary compiles an ADInplaceOrView dispatch-key
flavor production never replays.

## Caveats, stated plainly

- **Host-class framing**: every number above is host1/EPYC. The graph win is
  dispatch-bound and host3 (Xeon, dispatch ~45 ms, stall 11.2×) should see larger
  gains, but post-fix host3 numbers are unmeasured.
- **TP > 1 is designed but unvalidated** (world_size=1 throughout); the per-seq
  serialization branch makes it structurally safe; the O(n·T) token-shipping cost
  per chunk is a known, documented TP-only tax.
- **Long-prompt TTFT rises at small τ** — the explicit dial: τ=16384 preserves
  throughput-optimal behavior (mixing costs ~0.5%), τ≈512 buys the 7.8 ms ITL bound.
- Prediction misses are recorded alongside hits in docs/pr6/05 §3b (notably:
  "long TTFT is a wash" missed on host1; the C2-stage TTFT band was optimistic).

## Evidence index

status.md (fork/commit/gate ledger) · c2/c3/c4/c5_gates_host1.txt ·
p11_host1.txt (recompile storm) · after_host1.txt (headline) ·
pr5_noregression_host1.txt · probes p1–p13 with host-labeled outputs.

## Freeze smoke record (every script executed on the frozen checkout, 2026-08-03)

19/21 scripts pass as-is (probes p1-p3, p4b, p5-p13, dispatch/graph-feasibility
pair, sweep-era instruments, all four gate scripts, all three arm instruments,
bench_latency_tau). Two findings, both resolved before the cut:
- p4_paged_vs_fresh.py asserts BY DESIGN on post-C2 trees (its premise — the fresh
  branch — was abolished by F3b); header now pins it historical, evidence-era
  checkout <= 9f7bffe. Not a defect.
- p13_small_bucket_tax.py Part B indexed varlen_graphs by bare bucket (pre-two-tier
  key); now routes the tier exactly as run_model does. Post-fix it measures the
  CURRENT production path: replay 4.89 / run_model 5.27 ms at the 512 bucket —
  independently re-confirming the ~2.6 ms two-tier recovery on the production path.
Notable smoke numbers on this checkout: p9 varlen 8646.96 tok/s; p7 graphed(1024)
9.1 ms; bench_latency_tau 512 long TTFT 63.9 ms — all consistent with the ledger.
