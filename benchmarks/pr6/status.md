# PR 6 status — chunked prefill via graphed mixed steps

Living tracker; update per commit. Terminology: forks F1–F5 and commits C1–C5 per the
design doc set (04_design_forks, 05_evidence_predictions, 06_implementation_plan).
Branch: feat/chunked-prefill off dev @ 317e6f0.
Before/after instrument: bench_latency.py byte-pinned at metrics-artifacts 5b6f013
(always fetched via `git show`, never copy-edited; one host per comparison).

## Fork resolutions

| fork | verdict | locked by |
|---|---|---|
| F1 step representation | hybrid — pure-decode path untouched forever; any step containing prefill work uses the varlen layout; per-seq `seq.is_prefill` branch for TP-safe token extraction | derivation + test_ragged (C1) |
| F2 scheduler policy | decode-first budget; FIFO chunk fill; ≤1 partial chunk per step; ≤1 mid-chunk seq system-wide; emission predicate survives unchanged | proof sketch (doc 04 §F2); pinned by C3 tests |
| F3 graph bucketing | 1-D T_pad buckets {128,256,512,1024,2048,4096}; segment slots fixed at max_num_seqs+1 with zero-length padding; max_seqlen_q baked = T_pad; capture joins the shared decode graph_pool | P1, P2, P3a, P3b |
| F3b branch unification | always-paged for every real ragged step, eager AND graphed (warmup exempt — runs pre-allocation); paged ≡ fresh functionally but NOT bitwise (cross-kernel-template rounding); bucket-miss fallback is bitwise-transparent by the same-template argument | P4 falsified as posed (bf16-tolerance artifact); P4b-2b (cu_seqlens_k governs paged key count), 2c (block_table honored), P4b-3 (production chunked vs monolithic: 16/16 greedy); P4c magnitudes: kernel max <fill>, model max <fill> |
| F4 public surface | step() shape sacred; StepOutput grows num_prefill_tokens / num_decode_tokens | bench_latency.py contract |
| F5 equivalence gates | axis 1 (graph vs eager, both paged — same template): bitwise; axis 2 (any cross-template or cross-composition comparison): token-level with fp64 tie adjudication. The C2 branch change is itself an axis-2 event: one-time greedy token-gate at C2 entry, then base.json re-baselined with a numerics note | P3a licenses axis 1; P4/P4b define axis 2's scope |

## Commit ledger

| commit | contents | status | gate record |
|---|---|---|---|
| C1 | prepare_prefill → prepare_ragged; dormant decode-mode extraction branch; class alias | DONE `<c1-sha>` | byte-gate: base.json == c1.json, IDENTICAL; pytest 38 incl. test_ragged; bench 8632.15 / 8597.40 (band) |
| C2 | bucketed varlen graphs + run_model routing + miss counter; prepare_ragged block_tables condition | wip `523c19e` — additionally gated on the P10/P11 step-1 compile investigation (probes in-tree; arm results pending) | targets: per-bucket bitwise; byte-gate continuity vs base.json; 39 tests; bench band; C2-stage bench_latency (predict: interactive TTFT 27→7–10 ms, long TTFT 39→8–14, max_ITL only ~15–20 — stall residual is C3's job) |
| C3 | decode-first mixed-step scheduler | pending | two-axis equivalence; ≤1-mid-chunk assert; streaming gates under forced mixing; preemption-under-mixing |
| C4 | StepOutput token fields; tqdm; shim convention | pending | test_step_public_contract; pinned bench_latency runs unmodified |
| C5 (optional) | drop redundant is_prefill conjunct from emission gate | pending | own byte-gate |

## Evidence index (benchmarks/pr6/, host-labeled)

| id | file(s) | one-line result |
|---|---|---|
| E1 before | before_host1.txt, before_host3.txt | stall 8.9× (EPYC) / 11.2× (Xeon); identity holds on both |
| E2 sweep | sweep_host1.txt, sweep_host3.txt | step time flat in C (~26 / ~47 ms); dispatch→GPU crossover at C≈2–4k visible on host1 only |
| E3 dispatch probe | probe_host1.txt, probe_host3.txt | dispatch == wall (~25 / ~45 ms); prepare/sample plumbing 1–3 ms |
| E4/E5 graph feasibility | graph_host1.txt, graph_probe_host3.txt | both attention branches capture; replay 6.29–6.51 / 6.06–6.26 ms |
| P1–P3 pre-code probes | probes2_host1.txt | zero-length segments bitwise-OK; oversized-M tax ~3%; graph==eager bitwise; decode pool intact after pooled varlen capture |
| P4/P4b/P4c/P4d | p4_paged_vs_fresh.py, p4b_decompose.py | paged vs fresh: not bitwise, not fp32-allclose — cross-template rounding, NOT divergence (P4c: kernel max 0.00390625 (= 2⁻⁸ exactly, one bf16 ULP; 84.5% elements bitwise-equal), model max 1.25 / mean 0.021 (28-layer amplification; scale adjudicated by P4d)); 
(p4d: fresh |max| 72.5, the residual stream carries the outlier channels; row profile (min 0.0 / median 0.25 / max 1.25) means some tokens came through entirely bitwise-identical while the median token drifted ~0.3%; worst row 320 is an unremarkable mid-block position (256+64), ruling out boundary artifacts)
kernel semantics proven: key count from cu_seqlens_k (2b), pages gathered per block_table (2c); production chunked==monolithic 16/16 greedy (2-3, first-ever output-correctness coverage of the paged branch) |

## Standing rules in force
- Instruments live in benchmarks/pr6/, never /tmp-only; content committed within minutes of existing.
- Before/after comparisons: byte-pinned instrument, single host, first run discarded.
- No --amend on this branch once pr6-artifacts is cut (artifacts convention restored for PR 6).
- Any torch/flash-attn version bump invalidates graph evidence: re-run E4/E5 and P1–P4 first.
- Bitwise comparisons are valid only within one kernel template; any cross-template or
  cross-composition claim is judged at token level with fp64 tie adjudication.
- bf16 tolerance discipline: one ULP ≈ value/256 (~0.004 at magnitude 0.5). Never
  allclose bf16 against fp32-scale atol; P4's "failure" was the ruler, not the math.
- Paged-prefill output correctness had zero test coverage before P4b-3; its
  production-path check graduates into the suite (natural prompts, longer outputs,
  tie adjudication) alongside C2/C3.
- Probe cleanup (p5+): shared plumbing lives in probe_common.py (LLM ctor, timed,
  arm parsing/stub, bench workload, ragged-step setup, clean exit). P8 folded into
  P9 (same workload/AB; P9 prints the P8' tok/s line). P6b/P6c retired and p7's
  "graphed(4096)" relabeled: varlen_ts caps at 2048, so the 4096-token step takes
  run_model's miss-fallback (verified: miss counter increments, run_model ≈
  paged-eager). P5b's padded-launch bitwise gate graduated to tests/test_varlen_graphs.py.