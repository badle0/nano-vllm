# Request-latency metrics — design and validation record

Branch: feat/request-metrics (single commit over upstream bb823b3; independent of the
sampling series — no stacking).
Hardware/env: Vast.ai A100, torch 2.10 + cu128 + flash-attn 2.8.1, Qwen3-0.6B bf16.

## What was added (~20 lines core + tests)

Four timestamps and a per-token time list on Sequence (rank 0 only):
arrival_time (construction), first_scheduled_time (leaves the waiting queue),
first_token_time (first real token appended), finish_time, token_times[].
Stamping sites: scheduler.schedule() for first_scheduled (guarded: set only if None);
scheduler.postprocess() for the rest, with ONE perf_counter() read per step shared by
all sequences in that step. nanovllm/metrics.py derives per-request metrics
(ttft, queue_time, e2e_latency, mean_itl, max_itl, itls, token counts); generate()
output dicts gain an additive "metrics" key. Field vocabulary mirrors vLLM's
RequestMetrics (arrival / first-scheduled / first-token) for reviewer recognition.

## Design decisions

- time.perf_counter() (monotonic): only intervals are ever reported; wall clock buys
  nothing in single-process rank-0 stamping and can step backwards.
- Stamp once per step, not per sequence: all sequences in a step receive tokens at the
  same batched .tolist(); one stamp is cheaper and MORE honest — ITL resolution equals
  step granularity, which is exactly the resolution at which scheduling stalls exist.
  The stamp includes the GPU->CPU sync, i.e. the token's true arrival at the engine.
- Per-token token_times retained (one float per generated token, rank 0 only): a decode
  stall is ONE giant gap; mean ITL dilutes it, max/p99 ITL is its signature. This is the
  metric the chunked-prefill evaluation needs.
- Serialization untouched: __getstate__ is an explicit allowlist, so the new attributes
  never ship to TP workers; reading them on a worker raises (the standing tripwire).
- Chunked-prefill-proofed now: the first_scheduled guard survives preemption and future
  multi-chunk scheduling; first_token stamps at the append site, which the partial-
  prefill `continue` in postprocess skips — a discarded chunk-boundary sample can never
  masquerade as the first token.

## The stall demonstration (bench_latency.py)

Method: drive add_request/step directly (nano's generate is batch-synchronous; the
public step loop permits mid-flight arrivals). 16 interactive requests (64-token
prompts, max_tokens=256, ignore_eos) run 40 steps into decode; then 2 long requests
(2048-token prompts) are injected; run to completion; group metrics by prompt length.
Shape-matched warmup precedes measurement (see protocol below).

interactive  n=16 TTFT p50= 31.5ms p99= 31.5ms mean_ITL= 4.99ms max_ITL p50= 43.3ms
long         n= 2 TTFT p50= 37.4ms p99= 37.4ms mean_ITL= 4.82ms max_ITL p50= 12.3ms

Reading: the moment the long prompts arrive, the scheduler's prefill-first policy (any
waiting sequence that fits turns the entire step prefill-only) freezes all 16 in-flight
decodes for one 4096-token prefill step. The freeze appears as a ~10x ITL spike
(mean ~5.0 -> max ~47.0 ms) on every interactive request simultaneously. Consistency
identity, measured through two independent paths across three runs: interactive
max_ITL (47.0) ~= long TTFT (40.7) + one decode step (~5) — i.e. THE LONG REQUEST'S
TIME-TO-FIRST-TOKEN IS PAID BY THE INTERACTIVE REQUESTS AS INTER-TOKEN LATENCY.
(The 47.0 / 40.7 figures are the three-run aggregate; the single-run table above reads
43.3 / 37.4. The claim is the *relationship*, which holds within every run — absolute
values drift a few percent per session; see the reproducibility band below.)
This table is the "before" of the planned SARATHI-style mixed-batch chunked prefill;
rerunning this script unchanged after that lands is the "after".

## Measurement protocol (and the cold-start arc, kept honest)

TTFT of the first measured batch depends heavily on process shape history:
  no warmup at all:                interactive TTFT p50 = 1090 ms
  warmup, wrong shape (1x256):     interactive TTFT p50 =  211 ms
  warmup, measured shapes (16x64): interactive TTFT p50 = 37.4ms   <- reported
The ~177 ms recovered by shape-matched warmup is one-time compile/allocator work for
first-seen shapes (same cost class as the sampler-specialization compiles measured in
the sampling series). Protocol: report steady-state; note cold start adds ~0.2-1.1 s
to a process's first request. Steady-state is the right basis for the scheduling-policy
comparison this table exists for — cold start would be identical noise on both sides.
Two truthful artifacts: interactive TTFT p50=p99 (all 16 arrive together and share one
prefill step — degenerate by construction, not a bug); long max_ITL wobbles across runs
(extreme-value statistic over n=2 — not meaningful).

## Re-validation (2026-07-27, post-audit smoke)

Re-run unchanged on the same host, immediately after the artifact-branch provenance
audit. `python benchmarks/bench_latency.py` (from repo root):

interactive  n=16 TTFT p50= 31.7ms mean_ITL= 4.95ms max_ITL p50= 44.4ms
long         n= 2 TTFT p50= 38.1ms mean_ITL= 4.77ms max_ITL p50=  6.3ms

The stall signature reproduces: mean 4.95 -> max 44.4 ms is a ~9x spike on every
interactive request, and the consistency identity holds through both paths —
interactive max_ITL 44.4 ~= long TTFT 38.1 + one decode step (~5) = 43.1.

Reproducibility band (this run vs. the recorded run above): interactive TTFT
31.7 / 31.5, mean_ITL 4.95 / 4.99, max_ITL 44.4 / 43.3 — all within ~3%, the same
cross-session variance quoted for throughput in the sampling series. Long max_ITL is
the exception (6.3 / 12.3): a p50 over n=2, an extreme-value statistic on two samples,
already flagged above as not meaningful and carrying no weight in the argument.

This band is what makes the table usable as the "before" half of the chunked-prefill
comparison: a post-change delta must exceed ~3% on the interactive figures to be read
as signal rather than session drift. The spike itself (~9-10x) is an order of magnitude
clear of that threshold.

## Tests (3)

Pure (CPU, importlib-loads metrics.py, no engine): exact derived values on a fabricated
stamp set; single-token edge (empty ITLs -> zeros, not crashes).
Integration (GPU, real engine): ordering/count invariants only — queue_time >= 0,
ttft >= queue_time, e2e >= ttft, len(itls) == completion-1, all gaps >= 0 — plus a
per-request distinctness assert that specifically guards the silent bug class where
every request receives the last request's metrics (a leaked-loop-variable version of
generate was caught in review exactly here). Never assert magnitudes: wall-clock
magnitude tests flake; ordering tests don't.
Tests assume `pip install -e . --no-deps` (the --no-deps guard keeps pip from moving
torch out from under the pinned flash-attn wheel).

## Reproduction

pip install -e . --no-deps
pytest tests/ -v                      # 3 passed
python benchmarks/bench_latency.py    # prints the two-group table
                                      # run from the repo root; not importable as a module path