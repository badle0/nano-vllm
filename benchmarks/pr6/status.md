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
| F3b branch unification | always-paged varlen (fresh prefills attach block_tables; warmup stays fresh — runs pre-allocation) | P4 — PENDING |
| F4 public surface | step() shape sacred; StepOutput grows num_prefill_tokens / num_decode_tokens | bench_latency.py contract |
| F5 equivalence gates | axis 1 (graph vs eager): bitwise; axis 2 (chunked/mixed vs monolithic): token-level with fp64 tie adjudication | P3a licenses axis 1 |

## Commit ledger

| commit | contents | status | gate record |
|---|---|---|---|
| C1 | prepare_prefill → prepare_ragged; dormant decode-mode extraction branch; class alias | DONE `<c1-sha>` | byte-gate: base.json == c1.json, IDENTICAL; pytest 38 incl. test_ragged; bench 8632.15 / 8597.40 (band) |
| C2 | bucketed varlen graphs + run_model routing + miss counter; prepare_ragged block_tables condition | next — entry gated on P4 | targets: per-bucket bitwise; byte-gate continuity vs base.json; 39 tests; bench band; C2-stage bench_latency (predict: interactive TTFT 27→7–10 ms, long TTFT 39→8–14, max_ITL only ~15–20 — stall residual is C3's job) |
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
| P4 branch unification | p4_paged_vs_fresh.py | PENDING — gates C2 entry |

## Standing rules in force
- Instruments live in benchmarks/pr6/, never /tmp-only; content committed within minutes of existing.
- Before/after comparisons: byte-pinned instrument, single host, first run discarded.
- No --amend on this branch once pr6-artifacts is cut (artifacts convention restored for PR 6).
- Any torch/flash-attn version bump invalidates graph evidence: re-run E4/E5 and P1–P4 first.