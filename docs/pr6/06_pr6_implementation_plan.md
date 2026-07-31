# PR 6 — Implementation Plan

*Build order with per-commit gates, the file-by-file change map, the three probes
that must run before code, and the risk register. Assumes the Fork verdicts of
file 04; base = `dev` @ 317e6f0; branch = `feat/chunked-prefill` (exists, holds the
evidence commits).*

## 0. Pre-code probes (one session, ~30 minutes, all committed to `benchmarks/pr6/`)

- **P1 — zero-length segments.** Build a varlen call whose `cu_seqlens_q` contains
  repeated values (empty segments) and verify output correctness for the non-empty
  ones. Decides Fork 3's bucket-key rank: pass ⇒ one-axis buckets (`T_pad` only,
  `S_pad` fixed at `max_num_seqs+1`); fail ⇒ two-axis, more graphs.
- **P2 — oversized `max_seqlen_q`.** Capture at `M = T_pad`, replay with short real
  segments; compare replay time against a tight-`M` capture. Decides whether one
  graph per `T_pad` serves all segment shapes or `M` needs its own bucket axis.
- **P3 — bitwise determinism check.** One eager varlen forward vs one replay on
  identical inputs, `torch.equal` on logits. Expected true; this calibrates the
  Fork-5 axis-one gate before it's relied on. (While here: capture-order
  interaction — insert the varlen captures into the existing largest-first
  `graph_pool` sequence and confirm the decode graphs still replay bitwise.)

Each probe is ≤25 lines in the established style; outputs appended to
`graph_probe_host3.txt`.

## 1. Commit topology (the C-structure, extended)

**C1 — unified varlen prepare, zero behavior change.**
`prepare_prefill` generalized into `prepare_ragged`: per-sequence branch on
`seq.is_prefill` (prefill-mode: existing slot-window math verbatim; decode-mode:
last_token / len−1 / last-slot formula — the `prepare_decode` per-seq logic
embedded). `prepare_prefill` becomes an alias; nothing calls the decode-mode branch
yet. *Gate:* byte-identical fixed-seed `generate()` vs `dev`; 37 tests; bench ±
noise. Blast radius: `model_runner.py` only.

**C2 — varlen graph infrastructure, behavior-identical outputs, faster prefill.**
Bucketed capture per Fork 3 (key rank per P1/P2), persistent buffers grown from the
decode `graph_vars` pattern, −1 slot padding, shared `graph_pool`, capture order
integrated largest-first; `run_model` routes varlen steps to buckets with the
eager-fallback rule and a miss counter. Pure-decode routing untouched (Fork 1).
*Gates:* per-bucket bitwise graph-vs-eager (P3's gate, now in pytest); fixed-seed
`generate()` still byte-identical (replay is bitwise ⇒ this must hold exactly);
measured: monolithic-prompt TTFT drops ~45→~7 ms — the first user-visible win,
shippable alone as "fast prefill" even before mixing.

**C3 — scheduler policy: decode-first mixed steps.** The Fork-2 rewrite of
`schedule()` lines 26–76: decode admission first (existing loop verbatim), then
FIFO chunk fill to budget, ≤1 partial; `is_prefill` return semantics become "ragged
step"; postprocess untouched (the predicate survives by the Fork-2 proof). *Gates:*
the two-axis equivalence suite; ≤1-mid-chunk invariant; all 37 + new tests;
`test_chunked_prefill_emission` and every streaming gate under forced mixing;
preemption-under-mixing.

**C4 — accounting + surface.** `StepOutput` gains `num_prefill_tokens` /
`num_decode_tokens`; tqdm shows both; `step()` shim keeps `(finished, num_tokens)`
with the documented mixed-step convention. *Gates:* `test_step_public_contract`;
pinned-ref `bench_latency.py` runs unmodified.

**C5 (cleanup, separate, optional-order):** delete the now-provably-redundant
`is_prefill` conjunct from the emission gate, with its own byte-gate — the old
conjecture retired with evidence.

Then: **the headline run** (E1 rerun, after vs before), the pr6 validation record,
and the artifacts decision executed (recommendation: `pr6-artifacts` cut after the
final amend; smoke rule — every script executed on its own checkout).

Standing rules in force throughout: commit within minutes of content existing; push
as backup freely; no `--amend` after any artifacts branch is cut; every checkout is
preceded by `git status`; instruments live in `benchmarks/pr6/`, never `/tmp`-only.

## 2. File-by-file change map (expected blast radius)

| file | C | change | must NOT change |
|---|---|---|---|
| `engine/model_runner.py` | C1,C2 | `prepare_ragged`; bucketed capture; routing + fallback + miss counter | `prepare_decode`, decode graphs, `run()` decode path |
| `engine/scheduler.py` | C3 | `schedule()` decode-first rewrite | `postprocess` (byte-for-byte), `preempt`, `cancel_all` |
| `engine/llm_engine.py` | C4 | `StepOutput` fields; tqdm; shim convention note | `step()` shape; `stream`/`generate` contracts; footgun-one ordering |
| `engine/sequence.py` | — | none expected | `__getstate__` allowlist; `is_prefill` semantics |
| `layers/attention.py` | C2? | possibly nothing (branch already keys off context) | store-before-attend order; both kernel calls |
| `utils/context.py` | C1 | doc/semantics of `is_prefill` → "ragged" | field set (buffers reference these) |
| `config.py` | — | none (τ default unchanged ⇒ warmup/KV coupling dormant) | — |
| `tests/` | C2–C4 | graph gates, invariant asserts, mixing-forced reruns | shared-engine conftest pattern |
| `benchmarks/pr6/` | all | probes, tables, records | pinned-ref protocol for `bench_latency.py` (`5b6f013`) |

## 3. Risk register

- **Capture fragility across version bumps.** Both branches are proven on *this*
  pin (torch 2.10+cu128, flash-attn 2.8.1). Any torch/flash upgrade re-runs
  E4/E5/P1–P3 before trusting graphs — add to `setup_env.sh`'s doc block.
- **Bucket thrash / fallback leakage.** A workload straddling bucket boundaries
  pays capture-miss eagers at 45 ms each. Mitigation: miss counter asserted ≈0 in
  the headline run; bucket edges chosen from the τ policy, not round numbers.
- **Memory: graphs + buffers + KV.** Varlen buffers (`T_max=4096` outputs ≈ 8 MB)
  and 6–12 extra graphs in the shared pool, on a 40 GB card already at 0.9
  utilization; `allocate_kv_cache` runs before capture, so capture allocations must
  fit the residual — verify `num_kvcache_blocks` unchanged vs `dev` at startup, and
  budget the pool before widening bucket coverage.
- **Numerics drift on the decode rows of mixed steps** (varlen kernel vs
  `flash_attn_with_kvcache`): expected low-order-bit only; covered by the Fork-5
  axis-two gate, but if divergence rates exceed tie-class levels, suspect the mixed
  layout (positions/slots) before the kernel.
- **TP correctness is designed but untested** (world_size=1 throughout). The
  per-seq `is_prefill` serialization branch makes it *structurally* safe; state
  explicitly in the PR that TP>1 chunked-mixed is unvalidated. The O(n·T)
  full-token-ids-per-chunk shipping cost (sequence.py:86) is a known TP-only tax,
  documented, unfixed.
- **The latent decode assert** (scheduler:74, KV-exhaustion self-preempt) is
  inherited; mixed steps don't change its trigger, but the new scheduler must not
  *widen* it — covered by the robustness gates.
- **Host-class framing.** Every headline number is Xeon-12vCPU-specific; the EPYC
  constant is ~31 ms. The write-up quotes both and frames the graph win as
  host-class-dependent in magnitude, universal in direction.

## 4. Definition of done

The pinned-ref `bench_latency.py`, unchanged, on this host, shows interactive
max_ITL inside the predicted 6–8 ms band with the identity broken as designed;
`bench.py` within noise; 37+N tests green with the two-axis equivalence suite;
bucket-miss counter ≈0; the τ sweep re-run showing the E2 inversion; the validation
record frozen on `pr6-artifacts` with every script smoke-run on its own checkout;
and file 05's prediction table filled in with measured values beside each
prediction — hits and misses both, per house style.
