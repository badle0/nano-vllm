# nano-vllm contribution project — agent memory

## Working rules (non-negotiable — apply before any action)
- Before ANY code change: enumerate the alternative options (including "do
  nothing") and state why the chosen change is optimal — with a measurement
  when the deciding quantity is measurable (probe first if it isn't). State the
  theoretical mechanism before presenting the diff.
- Register predictions with falsifiers BEFORE any measurement run; score them after.
- Probes before code: mechanism confirmed by experiment, never assumed.
- Verify claims against source (read the file) before writing code that depends on them.
- Bitwise comparisons only within one kernel launch shape (template + tiles + split
  schedule). Cross-shape/composition claims: token-level, fp64 tie adjudication.
- Never allclose bf16 at fp32 tolerances (one ULP ≈ value/256).
- Instruments live in benchmarks/pr6/, committed within minutes of existing; never /tmp-only.
- Gate scripts: argument-check first, GPU work second.
- git: `rev-parse --short` takes ONE revision; no `--amend` after an artifacts branch
  is cut; named stashes only, verify with `stash show --stat`, prefer apply over pop;
  push freely as backup.
- Engines are one-per-process (unconditional init_process_group, atexit-pinned memory).
- bench.py band on host1 (EPYC, C.45901419): 8600–8650 tok/s through C2. Post-C3
  (decision accepted 2026-07-31: ~0.5% default-τ cost is the designed chunked-prefill
  trade): provisional band 8560–8610, confirm on a quiet host (C3-era samples ran at
  loadavg 9–14). A miss is a HARD STOP — no commit until attributed.
- bench_latency.py exists ONLY at metrics-artifacts 5b6f013; fetch via `git show`, never edit.

## Environment constants
torch 2.10.0+cu128 / flash-attn 2.8.1 / python 3.12 at /venv/main; editable install
→ this repo; weights ~/huggingface/Qwen3-0.6B; ./setup_env.sh rebuilds all of it.
Any torch/flash bump invalidates graph evidence: re-run E4/E5, P1–P5 first.

## Where the truth lives — read these before acting
- benchmarks/pr6/status.md — live fork/commit/gate state and correction ledger.
- docs/pr6/ — design docs 04–06 (forks, evidence & predictions, implementation plan).
- benchmarks/pr6/*_host{1,3}.txt — the evidence ledger, host-labeled.
Current phase: C2 blocked on a named Dynamo recompile investigation (P10); see
status.md for the exact pending command before doing anything else.