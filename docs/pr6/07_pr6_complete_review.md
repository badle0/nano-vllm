# PR 6 — Complete Review

*Post-freeze review of the full chunked-prefill arc (feat/chunked-prefill, dev 317e6f0 → freeze d2380be → review fixes 3bc102a). Produced by a multi-agent audit: independent deep-readers over the commit history, the benchmarks inventory, the mechanism docs+code, and the artifacts-branch precedent, plus a redundancy hunt in which all 36 claims were adversarially verified (10 confirmed → fixed, 16 refuted as load-bearing). Companion evidence: `benchmarks/pr6/review_fixes_host1.txt`.*

---

## 1. What each of C1–C5 did

Verified commit map (9 commits touch production code; everything else is evidence/docs):

```
C1  9f7bffe  model_runner.py + tests/test_ragged.py
C2  08a03b4  tests/test_varlen_graphs.py     (gate committed BEFORE the implementation)
    523c19e  model_runner.py                 (capture/routing/_fill_varlen; always-paged flip)
    a18b258  model_runner.py                 (P10 pre-touch fix)
    8b6cbec  model_runner.py                 (bucket set finalized; S+1 slots)
C3  d18a6f4  scheduler.py + tests/test_mixed_steps.py
    1fdf9ac  model_runner.py + tests/test_varlen_graphs.py   (two-tier P13)
C4  e6c3597  llm_engine.py + tests/test_mixed_steps.py
C5  3fbf72f  llm_engine.py + scheduler.py
```

### C1 — `prepare_prefill` → `prepare_ragged` (zero behavior change)

The insight: **a decode is just a prefill chunk of length 1**, so the existing ragged
varlen layout already expresses a mixed step — except for one asymmetry: TP workers
only ship `last_token`, not the full token list, so slicing `seq[start:end]` for a
decode row would silently break on a worker. C1 adds a per-sequence `is_prefill`
branch extracting decode rows explicitly, and leaves it **dormant**: the
representation equivalence is proven in isolation (byte-identical outputs;
`test_ragged` asserts `prepare_ragged` ≡ `prepare_decode` on real decode steps)
before any scheduling or graph work depends on it.

### C2 — bucketed varlen CUDA graphs

The mechanism-carrying stage, in four commits:

- The per-bucket **bitwise gate test was committed 21 seconds before the
  implementation** (deliberate test-first; it references `varlen_vars` before it
  exists).
- The body: persistent buffers; per-bucket capture with one segment owning all
  tokens and the rest zero-length — every padding trick individually probed first
  (P1 zero-length segments, P2 oversized-M ≈3%, P3 bitwise + pool cohabitation);
  `_fill_varlen` as the **single** fill path shared by production and the bitwise
  test ("the certified code IS the production code"); and the **always-paged flip**
  (`prepare_ragged` sets block_tables for any real step) so bucket hits and misses
  share one kernel template — the bucket boundary can never change tokens.
- The **P10 fix**: `ModelRunner.__init__` compiles all `@torch.compile` modules
  inside its bf16 default-dtype window; every init-era Dynamo entry carries a dead
  `GLOBAL_STATE default_dtype` guard. Varlen graphs deferred the resulting
  recompile storm onto the first bucket-*miss* step (P10/P11: 1075 → 293 ms once
  fixed by a post-restore pre-touch with production-provenance inputs — created
  outside `inference_mode`, or rotary compiles an `ADInplaceOrView` dispatch-key
  flavor production never replays). This defect exists verbatim in upstream, hidden
  as first-request "cold start" — upstreamable independently.
- `8b6cbec`: the F3-drafted 4096 bucket dropped **by measurement** (P12: replay
  floor ≈ paged-eager; host1 is past the E2 dispatch/GPU crossover at that size);
  S+1 segment slots pre-provisioned for C3's partial chunk.

Gate: axis-2 token-gate vs C1 — 3 divergences, all **exact fp64 ties**
(gap +0.000000); per-bucket bitwise; band held.

### C3 — decode-first mixed steps (+ two-tier captures)

The commit that moves the target metric. `schedule()` inverted: dev's decode loop
verbatim but **first** (decodes charge 1 token each), then FIFO chunk fill at
`min(work, remaining)` — from which **≤1 partial chunk per step** and **≤1
mid-chunk sequence system-wide** fall out as theorems (asserted, not tuned), plus
one fix beyond the design doc: the mid-chunk **head rotation** (preemption's
`appendleft` can jump the queue; without the rotation two half-prefilled seqs
could coexist). The worst decoder gap becomes *one step at τ* — a chosen constant,
not a function of arriving traffic.

The two-tier follow-up exists because P13 measured zero-length slot padding at
~0.006–0.011 ms/slot: two prefix-sliced captures per bucket
(`{64, max_num_seqs+1}` slots, routed by live segment count) give the few-decoder
*and* many-decoder regimes their fast path — the alternative (lean slots + eager
fallback) would break the ITL bound exactly in the regime it exists for.

Gates: 43→44 tests; forced-mixing token-gate PASS (one flip = exactly **2 bf16
ULPs** at logit 19.625, inside the P4d envelope); the measured **E2 inversion**
(max_ITL 35.9/30.2/18.0/**7.8** ms at τ=16384/2048/1024/512, flat ~40–49 before);
−0.5% default-τ throughput attributed by paired phase-split to the short-segment
M-block tax and accepted (band re-baselined 8560–8610, later quiet-host
confirmed).

### C4 — honest accounting, sacred surface

Mixed steps made the signed `num_tokens` scalar a lie (it counted decode rows as
prefill throughput). `StepOutput` now carries explicit
`num_prefill_tokens`/`num_decode_tokens` — computed *before* `postprocess` zeroes
the counters — while `step()` keeps its legacy tuple shape via a documented shim,
because the byte-pinned `bench_latency.py` consumes that shape and changing it
would collapse the before/after methodology. Gate: deterministic shim test
(+100 / −1 / +255), byte-gate IDENTICAL 4/4.

### C5 — proof-backed deletion

After postprocess's increment, `cached < total` is *exactly* "mid-prefill row" —
decode rows always arrive at equality; preemption zeroes and re-enters via
prefill — so the emission gate's `is_prefill` conjunct (and parameter) was
provably redundant. Deleting it made the proof executable: any error in the
invariant reasoning would surface as a swallowed emission in the byte-gate
(IDENTICAL 4/4). Its bench run doubled as the quiet-host band confirmation.

---

## 2. What every `benchmarks/pr6/` file was for

**Era instruments (E2–E5)** — the founding observations:
- `sweep_c.py`: step time **flat** in chunk size (~26/~47 ms host1/host3) — chunking is free until the crossover.
- `dispatch_probe.py`: dispatch == wall — the stall is CPU-launch-bound → CUDA graphs are the remedy.
- `graph_feasibility{,2}.py`: both attention branches capture; replay ~6.3 ms — the design is licensed.
- Outputs: `sweep_host{1,3}.txt`, `probe_host{1,3}.txt`, `graph_host1.txt`, `graph_probe_host3.txt`.

**Pre-code probes (P1–P4b)**:
- `p1/p2/p3_*.py`: licensed F3's padding tricks (zero-length correctness, M-tax ≈3%, pool cohabitation) → `probes2_host1.txt`.
- `p4_paged_vs_fresh.py` + `p4b_decompose.py`: adjudicated paged-vs-fresh numerics to *exactly one bf16 ULP* at kernel level; locked F3b always-paged; P4b-3 was the first output-correctness coverage of the paged branch. `p4` is **historical-pinned** (its fresh-branch premise was abolished by C2; its assert fires by design on modern trees — run at ≤ 9f7bffe).

**Investigation probes (P5–P13)** on `probe_common.py` plumbing:
- `p5_kernel_config.py`: launch-shape mechanism; its P5b gate graduated into `tests/test_varlen_graphs.py`.
- `p6_paged_tax.py` / `p7_shape_policy.py`: paged-tax yardstick; shape policy data (backed the 4096-bucket drop and TTFT-floor attribution).
- `p9_phase_split.py`: prefill/decode phase split — produced the C3 −0.5% attribution (prefill 1.19→1.31 s, decode phase byte-identical).
- `p10_profile_step1.py` / `p11_step1_compile.py`: the recompile-storm bisection (profiler view + Dynamo-counter view) → `p11_host1.txt`.
- `p12_slot_tax.py`: killed the 4096 bucket; priced slot tax (~0.043 ms/slot @ T=4096, K-ceiling free).
- `p13_small_bucket_tax.py`: slot-tax rates at small T + production-path decomposition → chose two-tier from five options; re-keyed at the freeze to the two-tier graph dict.

**Gate instruments**:
- `gate_generate.py`: the greedy token/byte gate feeding `base.json` (C2 entry gate, C4/C5 byte-gates).
- `gate_greedy.py`: byte-identical twin kept as byte-pinning provenance.
- `gate_generate_tau.py`: forced-mixing variant (τ argv) — the C3 axis-2 gate.
- `gate_adjudicate.py`: the fp64 tie-adjudication machine (both kernel compositions per divergence).
- `bench_latency_tau.py`: τ-sweep wrapper execing the *pinned* `bench_latency.py` bytes (git show 5b6f013) — the E2-inversion measurement.

**Evidence records**: `before/after_host1.txt` (the headline pair; `before_host3.txt` for host-class framing), `c2/c3/c4/c5_gates_host1.txt`, `p11_host1.txt`, `pr5_noregression_host1.txt`, `review_fixes_host1.txt`; `status.md` (the living ledger) and `pr6_chunked_prefill.md` (the frozen design+validation record).

**Baseline**: `base.json` — re-baselined at C2 with a numerics note; held 4/4 through C3-default, two-tier, C4, C5, and the review fixes.

Inventory note: the P4-family, p5, p6, p7 numbers never got standalone host-labeled
txts — they live in status.md prose, gate records, and commit messages. Defensible
(the gate records are host-labeled), but the weakest link in the ledger chain for
independent audit of P4c.

---

## 3. Redundancy / over-complication audit — and the fixes applied

36 claims hunted; every one adversarially verified. **10 confirmed, 16 refuted as
load-bearing.** All confirmed items were fixed in commit `3bc102a` (gates: 46
tests, byte-gate IDENTICAL 4/4, band 8589/8582 in 8560–8610; see
`review_fixes_host1.txt`).

| # | severity | finding | resolution |
|---|---|---|---|
| 1 | **HIGH** | `sequence.py __setstate__` never restored `is_prefill`, but `prepare_ragged` reads it per row: pickle bypasses `__init__`, so any `world_size > 1` run would `AttributeError` on its **first prefill step**. C1's whole purpose was TP-safe extraction — and the flag didn't survive the wire. Untestable on a single-GPU box, which is exactly why every gate missed it. | Fixed: `self.is_prefill = isinstance(last_state, list)` (the payload already encodes the mode) + a fixture-free pickle round-trip test in `test_ragged.py`. TP>1 end-to-end remains unvalidated; the caveat stands in weakened form. |
| 2 | medium | `run_model`'s 3-conjunct graphable predicate duplicated verbatim (gate + miss-counter guard); `not enforce_eager` subsumed by `hasattr(varlen_graphs)`. | Hoisted to a single `graphable`. |
| 3 | low | Two consecutive identical `if not self.enforce_eager:` capture blocks in `__init__` (the third — pre-touch — must stay separate, making the pair look meaningful). | Merged. |
| 4 | medium | `tests/test_streaming.py`: dead `pytest`, `LLM`, `PATH`/`os` (pre-conftest leftovers). | Removed. |
| 5 | medium | `tests/test_metrics.py`: dead `import pytest, os`. | Removed. |
| 6 | low | `tests/test_metrics.py`: vacuous assertion — `itls == [...] or len(itls) == 2` can never fail on wrong values. | Replaced with length + exact-value checks at 1e-12. |
| 7 | medium | `status.md` C1 row: literal `<c1-sha>` placeholder; cites `c1.json`, a /tmp-era artifact never committed. | Filled (9f7bffe) + provenance note. |
| 8 | medium | `status.md` F3 verdict cell still claimed fixed S+1 slots — contradicted by the shipped two-tier code. | Corrected to the two-tier verdict with P12/P13 citations. |

(Item 1 appeared twice in the raw findings — two hunters found it independently; deduped here.)

**Notable refutations** (flagged, then killed by verification — kept on record so
they aren't re-flagged): `gate_greedy.py`'s byte-duplication of `gate_generate.py`
is deliberate byte-pinning provenance; the scheduler's trailing partial-`break`
and its two O(n) waiting scans are load-bearing or below measurement noise;
`p12`/`p13` scaffolding overlap is era-pinned evidence; the decode router's
literal `512` matches `capture_cudagraph`'s `max_bs` clamp.

---

## 4. How chunked prefill was achieved — and the most significant difference vs vanilla

**The one-line framing: make step cost proportional to work, *then* chunk.**

Vanilla nano-vllm's scheduler is prefill-first either/or — if any prefill is
admitted, that is the whole step, and every in-flight decoder freezes for the
prompt's duration. The naive fix (Sarathi mixing alone) would have been a *loss*
here, because E2/E3 established the deep fact the PR is built on: **an eager
prefill step costs a flat ~25–45 ms of CPU kernel dispatch regardless of token
count** (dispatch == wall; ~420 launches). Slicing a prompt into n chunks on an
eager engine pays that constant n times — τ was a dead knob (max_ITL flat in τ
pre-PR).

So the implementation is two co-dependent halves:

- **Graphs change what a step costs.** Any step with prefill work is laid out as
  one ragged varlen batch (decode rows are 1-token segments — C1's branch), padded
  up to a bucket (zero-length `cu_seqlens` tail segments; −1 slot/block sentinels)
  and replayed as one of 10 captured CUDA graphs (~6 ms instead of ~45), with a
  priced eager fallback and a miss counter. One kernel template for hit and miss
  (always-paged) makes the bucket boundary numerics-transparent.
- **Mixing changes who pays.** Decodes are admitted first every step; prompts fill
  the leftover budget τ; the worst decoder gap is one *step*, not one *prompt*.
  Downstream nothing else had to change: the sampler emits one token per segment
  (mid-chunk tokens discarded by the emission gate, which keys purely on sequence
  state — C5 made that structural), and the public surface is shimmed.

**Headline** (pinned instrument, same host, discard-first): interactive TTFT
27.4 → **9.2 ms** (3.0×); max_ITL 45.8 → 34.9 at default τ, **7.8 ms at τ=512**
(spike 8.9× → 1.6×); mean_ITL unchanged (decode path untouched, byte-for-byte);
throughput −0.5% at default τ, attributed and accepted.

**The most significant single difference** — ranked honestly, since neither half
ships alone ("graphs alone: a wash; mixing alone: everything regresses"): **the
graphed varlen step**. Three measured chains: (1) the dispatch-floor finding
nullifies any scheduling-only fix — upstream's monolithic prefill was
*accidentally optimal among eager designs*; (2) the largest headline number
(TTFT 3×) is a pure graph effect, isolatable by commit — the two-tier capture
change alone, zero scheduler edits, moved TTFT 14.6 → 9.3 ms; (3) the τ curve is
monotone *only because* graphs made cost ∝ tokens. Mixing supplies the bound's
**form** — certified by the identity shift: before,
`max_ITL ≈ TTFT_long + ITL` (decoders waited out the whole prefill *plus* their
own step); after, **`max_ITL == TTFT_long` exactly** (34.9 == 34.9) — the stall
and the prefill are literally the same mixed step. Graphs supply the bound's
**magnitude**.

---

## 5. Upstream-PR assembly: what ships, what stays

**House precedent** (pr1/pr2/pr3/metrics artifacts): feat branch = **code + tests
only**; every script, raw output, and record lives on the `*-artifacts` branch,
whose tip records code byte-identity with the shipped SHA. PR5 is the cautionary
counter-example — its benchmarks merged into dev and ride every branch since;
status.md explicitly restores the convention for PR6.

- **MUST ship**: `nanovllm/engine/{model_runner,scheduler,llm_engine,sequence}.py`
  deltas (~230 net lines) and the three new test files plus the `test_ragged.py`
  addition (~215 lines). `test_varlen_graphs.py` is part of the equivalence
  *contract* (the graduated P5b gate), not evidence.
- **SHOULD ship as PR description, not files**: the trimmed record — what-this-adds,
  the headline + τ table + identity signature, the −0.5% trade, the caveats
  verbatim (TP flag now survives the wire; TP>1 end-to-end still unvalidated), a
  paragraph flagging the upstreamable `default_dtype` find, and a link to
  `pr6-artifacts` as the evidence index.
- **STAYS on pr6-artifacts**: all of `benchmarks/pr6/` (host-specific,
  instrument-pinned to a sibling branch), all of `docs/pr6/` (including this
  review), `CLAUDE.md`.

**Recipe — arc-end snapshot restore** (cherry-pick rejected: 7 of 9 code commits
are mixed with evidence; history rewriting rejected: 5 gated snapshots give the
series for free):

```bash
git checkout -b feat/chunked-prefill-upstream 317e6f0
# for each arc end, in order — 9f7bffe, 8b6cbec, 1fdf9ac, e6c3597, <current tip>:
git restore --source=<sha> -- nanovllm/ tests/
git add nanovllm tests && git commit -m "<upstream-grade message>"
# closing gate — shipped code must be byte-identical to the validated tip:
git diff feat/chunked-prefill -- nanovllm tests    # must be EMPTY
```

Every intermediate state in that 5-commit series was actually gated (38/39/44/45/46
tests per the ledger). The last snapshot is the post-review-fix tip (3bc102a or
later), not 3fbf72f — the closing empty-diff gate enforces that automatically.
Provenance closes with one normal commit on `pr6-artifacts` recording the upstream
tip SHA and the empty-diff check (the invariant is code byte-identity, not literal
ancestry — the frozen branch is never rewritten).
