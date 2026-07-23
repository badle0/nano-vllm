cat > benchmarks/pr1_greedy.md << 'EOF'
# PR 1 (greedy sampling) — validation record

Environment: see pr1_env.txt

## No-regression benchmark (bench.py, temperature=0.6)

| commit | run | prefill tok/s | decode tok/s | Throughput tok/s |
|---|---|---|---|---|
| main (bb823b3) | 1 (discard) | 3 | 337 | 8670.51 |
| main (bb823b3) | 2 (keep)    | 3 | 331 | 8686.52 |
| feat/greedy-sampling | 1 (discard) | 3 | 327 | 8668.83 |
| feat/greedy-sampling | 2 (keep)    | 3 | 328 | 8 |

## G1b — greedy vs HF transformers

- prompt 0:
- prompt 1:
- prompt 2:

## Unit tests

3 passed (pytest tests/ -v, torch.compile enabled, CPU tensors)
EOF