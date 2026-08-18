# Full-completion chunk-tail evidence — 2026-08-18

This directory preserves byte-for-byte the two completed external archives
produced by `aggregate_certification.py` from ten fresh A100 processes. Every
run pins clean commit `ba1bde44ce0735923ee41d7547121809dd1cfb91`, tree
`506d7305ee818dcbdf84354790d3bec711380554`, source SHA-256
`a0c6c56c73ad4784f3e2c8c5f133d0646bdde604ceb3216f3c716fdf3ebfd210`,
and model SHA-256
`0c659d1dba2804b0943c24bbece1858e93273147f7a11693fa99062df8c5997b`.

The workload used `disable_python_gc=True`, 16 interactive 64-token prompts,
40 steps before two 2,048-token requests were admitted, 256-token completions,
temperature 0.6, `max_model_len=4096`, and
`max_num_seqs=min(512,tau)`. The environment was one NVIDIA
A100-SXM4-40GB, Python 3.12.13, PyTorch 2.10.0+cu128, CUDA build 12.8,
and driver 570.133.20.

| tau | seeds | per-run maximum interactive ITL (ms) | strict `<10 ms` runs | retained policy verdict |
| ---: | --- | --- | ---: | --- |
| 256 | 20260826–20260830 | 6.982, 7.142, 9.177, 13.838, 12.854 | 3/5 | **latency not certified** |
| 512 | 20260831–20260835 | 9.380, 7.762, 13.705, 7.762, 9.312 | 4/5 | **throughput/TTFT only; not latency certified** |

Tau 256 fails because every one of five fresh processes must remain strictly
below 10 ms. Tau 512 is excluded from latency certification by policy even if
all observations pass. Neither a single run nor a bounded phase diagnostic can
promote either profile.

Each child `manifest.json` hashes its five raw files and aggregate. Its sibling
`.COMPLETE` marker binds the aggregate hash and records the original external
archive path. That absolute path is retained as immutable provenance, not as a
repository-relative locator. `archive_provenance.json` closes the remaining
chain by hashing the inner manifests and markers as well as every copied
payload; it also records the originals' read-only modes. Git does not preserve
source mode 0444/0555.

This is a pinned non-certification baseline, not evidence that latency is
fixed. It is committed before any later production latency optimization. A
later fix must use a new clean commit and source hash and must produce five new
fresh-process tau-256 runs; these observations must never be reused or
relabeled to certify changed code.

Run the repository validator from the checkout root:

```bash
PYTHONPATH=. /venv/main/bin/python -B \
  benchmarks/chunked_prefill_tail/validate_retained_evidence.py \
  benchmarks/chunked_prefill_tail/evidence/2026-08-18-a100-full-completion-ba1bde4
```
