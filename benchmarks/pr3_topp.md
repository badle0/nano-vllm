# PR 3 (top-p / nucleus sampling) — validation record

Branch: feat/topp-sampling (single commit over feat/topk-sampling 93a4664).
Hardware/env: same instance and env as PR 2 (pr2_env.txt). Qwen3-0.6B bf16.
Protocol: interleaved same-session comparisons; first runs discarded; medians of 61 steps.

## Decode step-time (Event timers around run_model / sampler+tolist, B=256, 64 decode steps)

| configuration                         | model ms | sampler ms | share |
|---------------------------------------|----------|------------|-------|
| disabled (fast path, both knobs None) | 14.640   | 0.820      | 5.3%  |
| top_p=0.9 enabled                     | 14.643   | 10.560     | 41.9% |
| (reference: top_k=50, PR 2)           | 14.646   | 8.577      | 36.9% |
| (reference: fully eager sampler)      | ~18.5*   | 11.8       |  --   |

*TORCHDYNAMO_DISABLE de-compiles the whole model as well; listed for the sampler column only.
Fast path unchanged through all three PRs (0.820-0.821 ms): with both knobs None the
compiled graph is op-for-op the PR 1 sampler.

## Inductor workaround (why top-p's mask runs eager)

Compiling the cumsum-based nucleus mask inside the fused sampler graph crashes Inductor
(torch 2.10, CUDA): InductorError: TypeError in triton codegen get_block_shape
(tensor_dim=None). Trigger is a dynamic leading dim: a static-shape repro compiles clean;
the identical function crashes on the second distinct batch size (repro_inductor.py).
The same code compiles fine on CPU Inductor, which is why the unit suite passed.

Workaround: the mask is fenced with @torch.compiler.disable and runs eager. Measured
cost: +1.98 ms/step on top-p-enabled batches (10.560 vs 8.577 compiled-sort-only; fully
eager is 11.8, so the fence retains most of compilation's benefit). One-time ~1.7 s
compile on the first top-p batch per process (graph break + dynamic-dim recompile).
Disabled paths unaffected (Python-level branch: the fenced call is never traced).
Removable once the upstream codegen bug is fixed.

fp caveat: with top_p=1.0 passed as a tensor row in a mixed batch, fp32 cumsum error
over V=151,936 could in principle clip tokens whose exclusive cumulative rounds above
1.0 (unobserved at V=1000 over 16k draws); the None fast path is exact by construction.

## End-to-end no-regression (bench.py, knobs disabled, interleaved, first runs discarded)

| branch             | kept runs tok/s    | median |
|--------------------|--------------------|--------|
| feat/topk-sampling | 8945.33, 8879.91   | 8912.6 |
| feat/topp-sampling | 8940.59, 8912.05   | 8926.3 |

Delta +0.15%, within the 0.3-0.7% within-branch spread: no regression.

## Kept-set cross-validation vs HF TopPLogitsWarper

Tie-free fp32 logits: 0 / 600 mismatches (200 trials x p in {0.3, 0.8, 0.95} x 4 rows) —
kept sets exactly equal. The rules are equivalent: this implementation accumulates
head-descending with an exclusive cumsum; HF tail-ascending with an inclusive one.
bf16-quantized logits: 328/600 comparisons differ, every difference a symmetric
equal-probability swap inside a tie group straddling the nucleus boundary (kept counts
identical in all cases; kept mass equal to 1 ulp fp64; verified in fp64). The nucleus
definition is ambiguous under ties; the two implementations pick different, equally
valid representatives. Same bf16-tie root cause as the top-k value-threshold findings.

## Unit tests

19/19 (pytest tests/ -v, torch.compile enabled): PR1+PR2 suite plus top-p off-by-one
([0.5,0.3,0.2], p=0.6 keeps the crossing token), nucleus support vs independent
exclusive-cumsum oracle, adaptive collapse on peaked rows, combined k+p containment
(value-threshold oracle), greedy override with both knobs, top_p bounds validation.

## Combined-knobs smoke

topp_smoke.py: mixed batch (top_k=50+top_p=0.9 row; top_p=0.8-only row) through the
full engine with CUDA graphs on — coherent output; exercises the (tensor,tensor) and
(None,tensor) specializations and the fence in situ.
