# PR 6 status — chunked prefill via graphed mixed steps

Living tracker; update per commit. Terminology: forks F1–F5 and commits C1–C5 per the
design doc set (04_design_forks, 05_evidence_predictions, 06_implementation_plan).
The ledger below is the historical feature-arc record for `feat/chunked-prefill`
off `dev@317e6f0`; its commit IDs, test counts, and host outputs are retained as
provenance rather than rewritten as if they came from the repaired branch.
Before/after instrument: bench_latency.py byte-pinned at metrics-artifacts 5b6f013
(always fetched via `git show`, never copy-edited; one host per comparison).

## Repaired-state addendum — 2026-08-17

The merge candidate is now split between `fix/chunked-prefill-upstream` (code and
tests) and `fix/chunked-prefill` (the same source/tests plus this evidence). The
repair sequence is:

- `cb65ce2`: bounded transactional admission, explicit retryable capacity error,
  O(1) `mid_chunk_seq` state, constructor-coherent/read-only tau, and the
  `remaining <= 0` guard;
- `ea15b73` + `631a346`: safe low-tau eager routing, sparse graph-key selection,
  structural replay guards, and the tau 64/128 × max-model-length
  512/1024/4096 fresh-process matrix;
- `67d654f` + `6a98622`: compact scheduled-slice TP frames with checked bounds,
  followed by a real spawned-process/shared-memory/event handoff test;
- `73dce74`: seeded variable-length graph replay compared with valid unpadded
  eager execution on live greedy token IDs across the 128/256/512 buckets.

At `73dce74`, the full upstream suite is **149 passed**. The full evidence branch
has the same `nanovllm/` and `tests/` content after merging that commit. A real
two-GPU inference/NCCL run remains unavailable on this one-A100 host, so TP safety
is still explicitly **unverified**; spawn-boundary transport coverage is not a
substitute for that gate.

The old post-construction tau mutation used by eleven scripts became invalid when
the repaired scheduler made its configured budget read-only. Commit `66c735c`
updates those instruments (and the shared P5 caller) to pass tau through `LLM(...)`,
sets `max_num_seqs <= tau`, and constructs a fresh engine for each point in a
multi-tau sweep. Historical host output files below remain historical; they have
not been relabeled as regenerated repaired-tip measurements.

Fresh repaired-branch gates found one numerical tie-class divergence in eight
prompts, not universal byte identity:

| gate | repaired dev SHA-256 | repaired chunk SHA-256 | first divergence |
|---|---|---|---|
| stochastic | `5919fcde302485f1451f6be5c23a4ce109a31a8da099e5be50a5fd962d88fb4e` | `b8526e490891b21111320c240cc3847c2cb7554d5cfeb133d2c1238afb8312a1` | one sequence, completion position 15; seven later tokens cascade |
| greedy | `1e10d050c87fd8b7fc49932a31c2cc28cb959737b7ebe73569315c7ba3b73d51` | `6d9cd726cee967d843af37faa7ac07f701e7c1716c7d9c615968f19be358ca0b` | one sequence, completion position 28 |

Top-two capture classified the first greedy difference in the documented BF16
tie class: repaired dev had tokens 4024 and 1196 tied at 18.125 (argmax chose
1196), while chunked execution produced 18.375 and 18.250 (chose 4024). These are
declared numerical-compatibility limits across kernel composition, not invalid
tokens and not evidence for a bitwise-equivalence claim.

## Fork resolutions

| fork | verdict | locked by |
|---|---|---|
| F1 step representation | hybrid — pure-decode path untouched forever; any step containing prefill work uses the varlen layout; per-seq `seq.is_prefill` branch for TP-safe token extraction | derivation + test_ragged (C1) |
| F2 scheduler policy | decode-first budget; FIFO chunk fill; ≤1 partial chunk per step; ≤1 mid-chunk seq system-wide; emission predicate survives unchanged | proof sketch (doc 04 §F2); pinned by C3 tests |
| F3 graph bucketing | 1-D T_pad buckets {128,256,512,1024,2048} — 4096 dropped by P12 (replay floor 32 ms ~= paged-eager on host1: past the E2 crossover; C3 chunking bounds steps to the budget anyway); segment slots: TWO TIERS per bucket {min(64, max_num_seqs+1), max_num_seqs+1} routed by live segment count (P13 correction — originally fixed at max_num_seqs+1; zero-length padding tax measured ~0.006–0.043 ms/slot scaling with M-blocks, P12/P13); max_seqlen_q baked = T_pad; capture joins the shared decode graph_pool | P1, P2, P3a, P3b; P12 corrects the zero-cost-padding assumption |
| F3b branch unification | always-paged for every real ragged step, eager AND graphed (warmup exempt — runs pre-allocation); paged ≡ fresh functionally but NOT bitwise (cross-kernel-template rounding); bucket-miss fallback is bitwise-transparent by the same-template argument | P4 falsified as posed (bf16-tolerance artifact); P4b-2b (cu_seqlens_k governs paged key count), 2c (block_table honored), P4b-3 (production chunked vs monolithic: 16/16 greedy); P4c magnitudes: kernel max 0.00390625 (= 2⁻⁸, one bf16 ULP), model max 1.25 (P4d: amplification, not divergence) |
| F4 public surface | step() shape sacred; StepOutput grows num_prefill_tokens / num_decode_tokens | bench_latency.py contract |
| F5 equivalence gates | axis 1 (graph vs eager, both paged — same template): bitwise; axis 2 (any cross-template or cross-composition comparison): token-level with fp64 tie adjudication. The C2 branch change is itself an axis-2 event: one-time greedy token-gate at C2 entry, then base.json re-baselined with a numerics note | P3a licenses axis 1; P4/P4b define axis 2's scope |

## Commit ledger

| commit | contents | status | gate record |
|---|---|---|---|
| C1 | prepare_prefill → prepare_ragged; dormant decode-mode extraction branch; class alias | DONE 9f7bffe | byte-gate: base.json == c1.json, IDENTICAL (c1.json was a /tmp-era run artifact, not committed; the in-tree base.json is the later C2 re-baseline per its numerics note); pytest 38 incl. test_ragged; bench 8632.15 / 8597.40 (band) |
| C2 | bucketed varlen graphs + run_model routing + miss counter; prepare_ragged block_tables condition; post-restore pre-touch (P10 fix); F3 slot alignment S+1; bucket set {128..2048} (P12) | GATES COMPLETE — pending commit. Token-gate at C2 entry: 1 byte-match + 3 exact-fp64-tie flips, PASS; base.json re-baselined (numerics note: differs from C1 only at exact argmax ties). Byte self-consistency across processes: True. 39 tests. Band: 8604.61/8602.87 IN BAND. C2-stage bench_latency scored (c2_gates_host1.txt): interactive TTFT 27.4→14.7 ms (direction hit, 7–10 band missed — floor is the graphed step itself); long TTFT & max_ITL unchanged BY DESIGN (P12: 4096 is past the E2 crossover on host1; stall residual is C3's job; identity holds) | targets: per-bucket bitwise; byte-gate continuity vs base.json; 39 tests; bench band; C2-stage bench_latency (predict: interactive TTFT 27→7–10 ms, long TTFT 39→8–14, max_ITL only ~15–20 — stall residual is C3's job) |
| C3 | decode-first mixed-step scheduler (F2: decode admission verbatim-first, FIFO chunk fill, ≤1 partial, mid-chunk head rotation vs preempt-appendleft) | IMPLEMENTED, gates run (c3_gates_host1.txt) — pending commit + ONE OPEN DECISION | 43 tests (39 legacy unmodified + 4 pins); axis-2 gate PASS (1 flip = 2 bf16 ULPs, in P4d envelope); E2 inversion measured: max_ITL 35.9/30.2/18.0/10.6 ms at τ=16384/2048/1024/512 (was flat ~40-49); cost ~0.5% bench @ default τ, attributed (+13 ms/mixed-step short-segment tax, P12 mechanism) — band decision ACCEPTED (2026-07-31): provisional post-C3 band 8560–8610 pending quiet-host confirmation (CLAUDE.md updated); τ=512 residual 10.6 vs 6-8 headline → P13 probes the slot-tax rate at small T before any slot-policy change |
| C4 | StepOutput num_prefill_tokens/num_decode_tokens REPLACE the signed scalar (split computed pre-postprocess); tqdm updates both rates on mixed steps; step() shim keeps (finished, num_tokens) — legacy signed for pure steps, +num_prefill_tokens for mixed | DONE e6c3597 (c4_gates_host1.txt) | 45 tests incl. test_step_shim_convention (+100/-1/+255 deterministic); byte-gate IDENTICAL 4/4; pinned bench_latency runs unmodified; bench A/B delta ~-8 tok/s = pair noise, sub-band absolutes attributed to host load |
| C5 | drop redundant is_prefill conjunct from emission gate + its now-dead postprocess param (single caller updated) | DONE (c5_gates_host1.txt) | byte-gate vs base.json IDENTICAL 4/4; 45 tests unmodified; bench 8595.47/8586.03/8576.83 in band — run doubles as the quiet-host band confirmation: provisional 8560–8610 CONFIRMED (background loadavg ~6 vs C3-era 9–14), C3 caveat closed |

## Evidence index (benchmarks/pr6/, host-labeled)

| id | file(s) | one-line result |
|---|---|---|
| E1 before | before_host1.txt, before_host3.txt | stall 8.9× (EPYC) / 11.2× (Xeon); identity holds on both |
| E1 after (headline) | after_host1.txt | pinned instrument @ 3fbf72f: interactive TTFT 27.4→9.2 ms (3.0×); max_ITL 45.8→34.9 (spike 8.9→7.2× at default τ; 1.6× at τ=512 per C3/P13); mean_ITL unchanged; identity SHIFTS by design — max_ITL == long TTFT exactly (stall and prefill are now the same mixed step); P-E1r.1–4 HIT, .5 falsified-as-posed (the sum-identity's collapse IS the mechanism); throughput lines cite c5 band confirmation (external load hit loadavg 74 mid-session, latency metrics load-stable 11→30) |
| E2 sweep | sweep_host1.txt, sweep_host3.txt | step time flat in C (~26 / ~47 ms); dispatch→GPU crossover at C≈2–4k visible on host1 only |
| E3 dispatch probe | probe_host1.txt, probe_host3.txt | dispatch == wall (~25 / ~45 ms); prepare/sample plumbing 1–3 ms |
| E4/E5 graph feasibility | graph_host1.txt, graph_probe_host3.txt | both attention branches capture; replay 6.29–6.51 / 6.06–6.26 ms |
| P1–P3 pre-code probes | probes2_host1.txt | zero-length segments bitwise-OK; oversized-M tax ~3%; graph==eager bitwise; decode pool intact after pooled varlen capture |
| P10/P11 | p10_profile_step1.py, p11_step1_compile.py, p11_host1.txt | step-1 recompile storm = GLOBAL_STATE default_dtype guard on init-era Dynamo entries, deferred to the first bucket-miss step because bucket replays hide the warmup's eager call; fixed by post-restore pre-touch with production-shaped (non-inference-tensor) inputs; 1075→293 ms, dev-equivalent counters |
| P12 | p12_slot_tax.py, c2_gates_host1.txt | 4096-bucket replay floor 32 ms ~= paged-eager (host1 past E2 crossover) → bucket dropped; zero-length slot tax ~0.04 ms/slot at T=4096, K-ceiling free; corrects F3's padding-is-free assumption (P1 was correctness-only). Correction: production slots are 513, and the linear model then exactly explains the 54.6 ms in-engine replay (32.4 + 513×0.043) |
| P13 | p13_small_bucket_tax.py, c3_gates_host1.txt | slot-tax rate scales with M-blocks as predicted: 0.0058 ms/slot @ T=512, 0.0114 @ T=1024 (P12's 0.043 @ 4096 ÷ 8/4); bucket-512 decomposition: replay 7.46 of run_model 7.84 ms — replay dominates. RESOLVED by two-tier captures (slots {64, 513} per bucket, routed by live ns; options 1/3/4/5 rejected on record): τ=512 max_ITL 10.6 → 7.8 ms (median-3, HIT, inside 6–8 headline band); τ=1024 18.0 → 12.4; default-τ interactive TTFT 14.6 → 9.3; 44 tests (full-tier bitwise added); token-gate PASS; band in provisional range; init +0.2 s |
| Review fixes | review_fixes_host1.txt | post-freeze review (36 claims hunted, 10 confirmed): __setstate__ now restores is_prefill (real latent TP bug — AttributeError at any world_size>1; + pickle round-trip test), run_model predicate hoist, init capture-block merge, test hygiene, ledger corrections (C1 sha, F3 two-tier). 46 tests; byte-gate 4/4; band 8589/8582 IN BAND |
| PR5 no-regression | pr5_noregression_host1.txt | all three pinned streaming claims survive: TTFT collapse 26.3×→230.3× (stream first event 3.5 ms via lean 128 bucket), null-consumer +0.82% (inside pr5's own range), slow-consumer shape byte-near-identical; example_stream visual PASS. Eval-plan item 5 closed |
| C3 gates | test_mixed_steps.py, gate_generate_tau.py, bench_latency_tau.py, c3_gates_host1.txt | mixed-step scheduler: 43 tests; forced-mixing token-gate PASS (one 2-ULP tie flip); bench cost ~0.5% @ τ=16384 attributed to short-segment tax in eager mega-steps (p9 paired split: prefill 1.19→1.31 s, decode phase unchanged); τ sweep shows the E2 inversion — max_ITL ≈ one step at τ, monotone 35.9→10.6 ms |
| C2 gates | gate_generate.py, gate_adjudicate.py, base.json, c2_gates_host1.txt | token-gate vs C1: 3 divergences, all exact fp64 ties (gap +0.000000 both compositions) → PASS; byte self-consistent; bench_latency scored: interactive TTFT 27.4→14.7, long/max_ITL to C3 |
| C5 gates + band | c5_gates_host1.txt | emission-gate conjunct removal byte-identical (4/4) at 45 tests; quiet-host bench 8595/8586/8577 (median 8586) confirms the post-C3 band 8560–8610 — background loadavg ~6, quietest regime on record; C3's provisional-band caveat closed |
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
