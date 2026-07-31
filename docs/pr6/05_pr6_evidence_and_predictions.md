# PR 6 — Evidence Ledger, Predictions, and Evaluation Plan

*Everything measured so far, what it rules out, what is now predicted, and how the
feature will be judged. Host for all numbers below: instance C.46161205 —
A100-SXM4-40GB, 12 vCPU Intel Xeon @ 2.20 GHz, torch 2.10+cu128, flash-attn 2.8.1,
`dev` @ 317e6f0. All instruments committed under `benchmarks/pr6/`.*

## 1. The evidence ledger (chronological, five measurements)

**E1 — The fresh same-host "before"** (`before_host3.txt`; instrument =
`bench_latency.py` pinned at metrics-artifacts `5b6f013`, fetched via `git show`).
16 interactive decoders, two 2048-token prompts injected mid-decode:

    interactive: mean_ITL 5.19–5.20 ms, max_ITL p50 57.5–58.3 ms  → 11.2× spike
    long: TTFT 50.9–51.0 ms
    identity: 51.0 + 5.2 = 56.2 ≈ 57.5–58.3 (holds, two independent paths)

Host note discovered here: interactive TTFT is ~47 ms on this Xeon vs ~31.5 ms on
the prior EPYC hosts at equal ITL/throughput — the first hint that prefill carries a
CPU-side constant.

**E2 — Step-time-vs-C sweep** (`sweep_c.py`, full engine path, pure prefill of a
3,968-token prompt at seven budgets):

    C:        64    128    256    512   1024   2048   3968
    step ms: 47.5   46.3   46.7   47.6   47.7   49.0   49.2

Median step time is *flat in C*. Total prefill = steps × ~47.5 ms. **This refutes
the roofline model of file 03 as stated** — its T₀ was modeled at 1–2 ms; the
machine's is ~47. Direct consequence: naive mixed batching (eager) is a net loss on
this host-class, and today's monolithic prefill-first is accidentally optimal among
eager designs.

**E3 — Dispatch probe** (`dispatch_probe.py`, bare forward, inference_mode,
warmed, CPU-issue time vs GPU-completion time):

    C=64:   dispatch 46.0–46.2 | wall 46.1–46.2
    C=512:  dispatch 44.2–44.8 | wall 44.3–44.9
    C=3968: dispatch 45.4–45.6 | wall 46.0–46.1

dispatch ≈ wall at every size: the CPU kernel-launch stream is the bottleneck
(≈420 ops × ~108 µs on this 2.2 GHz vCPU); even 15+ ms of genuine GPU work at
C=3968 hides entirely under it. Bare forward 45–46 ms vs full-engine 47–49 ms ⇒
prepare/sample/postprocess plumbing is only 1–3 ms — the Python *around* the model
is innocent.

**E4 — Graph feasibility, fresh-KV branch** (`graph_feasibility.py`; C=512 first
chunk, `cu_seqlens_q == cu_seqlens_k`, `block_tables=None`):

    capture: OK    replay: 6.55 (first-touch), 6.06, 6.06 ms

7.4× vs eager; ~33% of the ~2 ms compute-roofline floor — honest small-GEMM
efficiency, not hidden overhead.

**E5 — Graph feasibility, paged branch** (`graph_feasibility2.py`; chunk 2 through
the real engine, `block_tables` gather, `k,v ← cache`):

    capture: OK    replay: 6.69 (first-touch), 6.27, 6.26 ms

Both attention branches capture. The +0.20 ms over E4 sits at the bottom of the
0.19–0.4 ms marginal-attention band for 512 extra keys/query — within noise, but the
cost model and the machine agree again once the dispatch constant is removed.

## 2. What the ledger rules out (design space, pruned by measurement)

- **Naive SARATHI mixing on eager steps** — E2/E3: every mixed step costs ~45 ms;
  max_ITL 58→~47 (1.2×), TTFT_long ×8, interactive service degraded for the whole
  prefill window. Dead.
- **Chunking as a standalone latency feature** — E2: chunk steps are still
  prefill-only under today's policy and each costs the flat constant; the stall is
  sliced, not shrunk.
- **Graphs as a standalone feature** — E4 arithmetic: 8 × 6 ms ≈ 48 ms stall ≈
  today's 51. A wash without mixing.
- **`torch.compile(mode="reduce-overhead")` as the dispatch fix (v1)** — the
  project's own `repro_inductor.py` (re-confirmed this month on this torch build)
  crashes Inductor on dynamic leading dimensions, which mixed varlen steps have by
  construction. Manual capture is feasibility-proven; compile is not. Deferred, not
  forgotten: re-probe on the next torch bump.
- **The unqualified §6 predictions of file 03** — superseded below. Their falsifier
  ("T₀ larger than modeled") pre-fired; the file stands as the record of a model
  honestly refuted.

Scope honesty carried into the write-up: the 45 ms constant is a property of
*eager PyTorch × slow cloud vCPU* (EPYC hosts: ~31 ms — same disease, milder). The
conclusions are host-class-dependent, and cheap-vCPU boxes are exactly what
rented-GPU serving runs on — the dependency is itself a finding, not a caveat to
hide.

## 3. Re-registered predictions (written before implementation; the after-run
tests these, it does not fit them)

Scenario = E1's exactly, on this host, with graphed mixed steps at τ = 512:

| quantity | today (measured) | predicted after | basis |
|---|---|---|---|
| interactive max_ITL | 57.5–58.3 ms | **6–8 ms** (~8–9× collapse) | one graphed mixed step ≈ E5's 6.3 ms + margin |
| interactive mean_ITL | 5.2 ms | 5.2–6.5 ms | decodes ride mixed steps during prefill windows |
| long TTFT | 50.9–51.0 ms | **≈ 48–60 ms** (a wash) | ⌈4096/494⌉ = 9 steps × 6–7 ms − pipeline overlap |
| identity | max_ITL ≈ TTFT_long + step | **breaks by design** | the stall the identity described no longer exists |
| bench.py throughput | 8465–8468 tok/s | unchanged ± 1% | pure-decode path untouched (Fork 1 verdict) |
| chunked prefill total (3,968 tok) | 46–49 ms eager monolithic | 48–56 ms graphed 9-chunk | 9 × 6.06–6.26 — **chunking becomes ~free** |
| graphed mixed-step time | — | 6–8 ms at T=512-bucket | E4/E5 band + decode rows |
| bucket hit rate, steady state | — | ≈ 100% (counter-verified) | Fork 3 design goal |

**Falsifiers, in advance.** max_ITL stuck above ~12 ms ⇒ a serialization survives
(scheduler still homogenizing, or fallback-eager steps leaking into the window —
check the miss counter first). TTFT_long > ~2× today ⇒ per-step overhead beyond the
6–7 ms band — suspect oversized-`M` grid waste or bucket thrash. bench.py drop
beyond noise ⇒ the pure-decode path was touched — a hard stop, per Fork 1. Bitwise
graph-vs-eager mismatch ⇒ buffer-rewrite bug — stop before interpreting anything
else.

## 4. Evaluation plan (the gates, in run order)

1. **Refactor gates (per commit, file 06):** byte-identical fixed-seed `generate()`
   at each zero-behavior commit; pure-decode steps byte-identical throughout;
   `pytest tests/ -v` = 37 green continuously.
2. **Graph gates:** per-bucket bitwise logits equality, graph vs eager, both
   attention branches; padded-row hygiene (no gather of garbage rows, no writes to
   −1 slots); replay-time table per bucket committed.
3. **Equivalence gates (two-axis, Fork 5):** greedy chunked/mixed vs monolithic —
   token-match with fp64 tie adjudication of every divergence, classification
   recorded; seeded stochastic equivalence; all streaming gates re-run with τ
   monkeypatched to force mixing; the ≤1-mid-chunk invariant asserted.
4. **The headline:** `bench_latency.py` at pinned ref `5b6f013`, byte-identical
   instrument, same host, run twice, first discarded — the "after" against E1's
   "before." This is the experiment the metrics PR promised in writing.
5. **No-regression:** `bench.py` ×2 interleaved against a `dev` checkout in the
   same session; `example_stream.py` visual check; the pr5 measurement scripts
   re-run (streaming's TTFT/null-consumer/slow-consumer claims must survive).
6. **Robustness:** preemption under mixed load; KV-exhaustion behavior unchanged;
   `cancel_all` mid-mixed-step (blocks restored); τ sweep re-run post-feature (the
   new curve *should* now fall with C in dedicated-prefill mode and be flat-in-C
   for latency — the exact inversion of E2, and a satisfying figure).

Every gate that produces numbers writes them to `benchmarks/pr6/`; the artifacts
branch (per the Fork-4 meta-decision) freezes the final record after the last
amend, scripts smoke-run on their own checkout, per the standing rules the audit
bought.
