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