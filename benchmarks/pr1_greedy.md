# PR 1 (greedy sampling) — validation record

Environment: see pr1_env.txt

## No-regression benchmark (bench.py, temperature=0.6)

| commit | run | prefill tok/s | decode tok/s | Throughput tok/s |
|---|---|---|---|---|
| main (bb823b3) | 1 (discard) | 3 | 337 | 8670.51 |
| main (bb823b3) | 2 (keep)    | 3 | 331 | 8686.52 |
| feat/greedy-sampling | 1 (discard) | 3 | 327 | 8668.83 |
| feat/greedy-sampling | 2 (keep)    | 3 | 328 | 8653.87 |

## G1b — greedy vs HF transformers

- prompt 0: MATCH for all 32 compared tokens (nv=32, hf=32)
- prompt 1: MATCH for all 32 compared tokens (nv=32, hf=32)
- prompt 2: diverges at index 31: nv=1096 hf=576
  shared prefix ids: [11105, 429, 279, 1372, 220, 16, 24, 21, 24, 374, 264, 10250, 1372, 13, 1096, 374, 279, 1156, 882, 304, 3840, 429, 264, 10250, 1372, 702, 1012, 11105, 553, 12677, 13]

  95/96 tokens match HF greedy exactly (prompts 0–1: 32/32; prompt 2: 31/32, final token only). At the divergence, candidates are ' This' (nano-vLLM) vs ' The' (HF cached decode). fp32 logit gap = 0.0573 — below one bf16 ulp (0.125) at logit magnitude, i.e., unrepresentable at execution precision. A full-sequence HF bf16 forward over the identical context, and the fp32 model, both rank nano-vLLM's token first. Conclusion: unresolvable bf16 near-tie between kernel paths; not a sampling defect.

## Unit tests

3 passed (pytest tests/ -v, torch.compile enabled, CPU tensors)

## Provenance correction (post-hoc)

This branch was originally cut on `bde1b8a`, whose `sampling_params.py` carried a
typo'd gate (`assert self.temperature > 0.0` beside a message reading
"non-negative"): the sampler-level greedy path worked, but
`SamplingParams(temperature=0.0)` was rejected at construction, so
`nv_greedy.py` could not have executed on that tree. The feature commit was later
amended to `e4a8c91` (`>=`, plus two API-level tests), leaving this branch pinned
to superseded code. Original base preserved as tag `archive/pr1-pre-amend`.

The branch is now rebased onto `e4a8c91`. Re-validation on the rebased tree:
`nv_greedy.py` runs to completion and reproduces the recorded `pr1_nv.json`
byte-for-byte; `pytest tests/` reports 5 passed (3 original + the 2 API-gate
tests carried in from `e4a8c91`). Layer-level validations are unaffected —
`sampler.py` is byte-identical across the amend — and no performance-relevant
code changed.

### Prompt 2 divergence (pre-existing, re-examined)

`compare_greedy.py`: prompts 0 and 1 match HF exactly for all 32 tokens; prompt 2
diverges only at the final token (index 31), nv=1096 `' This'` vs hf=576 `' The'`,
at a sentence boundary where both continuations are natural. Evaluating HF's own
logits on nano's 31-token context ranks the candidates identically in both
precisions — bf16 top-2 [1096, 576], gap 0.125; fp32 [1096, 576], gap 0.057 —
i.e. HF re-evaluated at nano's context prefers the token nano chose. This is a
small-margin, path-dependent branch at the `max_tokens=32` cutoff, not a
correctness defect in the greedy path. Residual: HF's recorded token differs from
HF's own re-evaluation; cause not isolated (candidate: HF-side prompt handling or
generation config). Caveat: `gap_check.py` measures HF logits, not nano's, so it
establishes the position is small-margin rather than measuring nano's own margin.

### Running these scripts

Scripts read and write `nv.json` / `hf.json` relative to the working directory and
are intended to run from the repo root: `nv_greedy.py`, then `hf_greedy.py`, then
`compare_greedy.py` / `gap_check.py <prompt_idx> <div_idx>` / `fp32_check.py`.
Results are promoted to `benchmarks/pr1_nv.json` and `benchmarks/pr1_hf.json`.
