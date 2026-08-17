# Repaired greedy and top-k sampling evidence

This artifact closes the release-evidence gaps for the repaired greedy and
top-k routes. It does not change sampler source code.

## Pinned scope

- greedy repair: `ec988708bbe1e4e84e97c3fc0599378c01920e3a`;
- top-k repair: `8759c877382f11ea16b40fdbd0dace7000b5e9ba`, whose parent stack includes
  the greedy repair; and
- evidence branch base: `b02f1efed6b33090fc6d23b1da0793c6365962c8`.

Every fresh release raw result embeds its resolved Git commit and aborts if the
checkout does not equal the expected repair. It also embeds the command line,
seed, working directory, unique Inductor cache, Python, PyTorch, CUDA build,
Transformers, GPU, GPU memory, and NVIDIA driver.

The E2E model was `/workspace/models/Qwen3-0.6B` (Qwen3, BF16, vocabulary
151,936). `provenance.json` pins SHA-256 hashes for the configuration, tokenizer,
and 1.5 GB safetensors file; those model hashes are manifest metadata rather
than fields emitted by the benchmark process.

## Protocol

### Sampler microbenchmark

Each scenario ran alone in three fresh Python processes with a unique empty
`TORCHINDUCTOR_CACHE_DIR`, at batch 256 and vocabulary 151,936. The exact input
commit was `ec98870` for greedy and `8759c87` for top-k. Each process records:

- the true first call before warmup, including compile latency and incremental
  allocated-memory peak;
- five warmups after that cold call; and
- 25 CUDA-event samples, nearest-rank p95, and incremental allocated-memory
  peak for the steady route.

Mutable top-k inputs are restored outside the timed region, modeling fresh
model logits without charging an artificial benchmark-only clone to the route.
The rotated process order is retained in `provenance.json`.

### Top-k end to end

Four fresh processes at `8759c87` generated 32 tokens for 256 distinct
128-token integer prompts. CUDA graphs were enabled with
`max_model_len=1024`, `max_num_seqs=256`, and 80% GPU-memory utilization. Each
process compared disabled top-k with all rows at top-k 50 over four alternating
observations, retaining observation zero as cold evidence and taking the median
of observations 1--3. First-scenario order was balanced
disabled/enabled/disabled/enabled across seeds 20260817--20260820.

To reproduce one micro process, run from the matching detached commit checkout:

```bash
PYTHONPATH=. TORCHINDUCTOR_CACHE_DIR=<new-empty-directory> \
  /venv/main/bin/python \
  <artifact>/benchmarks/sampling_evidence/sampler_bench.py \
  topk one_active_top_k_50 --seed 20260817 --output <result.json>
```

To reproduce one E2E pair, run from a checkout detached at `8759c87`:

```bash
PYTHONPATH=. TORCHINDUCTOR_CACHE_DIR=<new-empty-directory> \
  /venv/main/bin/python \
  <artifact>/benchmarks/sampling_evidence/topk_e2e_bench.py \
  --model /workspace/models/Qwen3-0.6B --seed 20260817 \
  --first disabled --output <result.json>
```

## Release results

Aggregate micro values use the median across three per-process cold/steady/p95
statistics and the maximum per-process incremental allocated-memory peak.

| Route | Cold wall | Steady median | Steady p95 | Peak incremental allocation |
|---|---:|---:|---:|---:|
| homogeneous greedy | 1,171.37 ms | 0.177 ms | 0.185 ms | 0.002 MiB |
| top-k disabled | 1,893.14 ms | 1.011 ms | 1.026 ms | 148.38 MiB |
| one row at top-k 50 | 2,003.54 ms | 1.069 ms | 1.112 ms | 148.38 MiB |
| all rows at top-k 50 | 1,943.67 ms | 2.038 ms | 2.043 ms | 148.38 MiB |

Greedy is near the audit's 0.173 ms direct-argmax reference. One-active top-k
adds 5.78% over disabled, while all-active work is 1.91x one-active; the former
full-sort implementation made one active row cost the same as all 256 and
peaked around 1.41 GiB.

| Seed | First scenario | Disabled tok/s | Top-k 50 tok/s | Paired change |
|---:|---|---:|---:|---:|
| 20260817 | disabled | 17,434.12 | 16,334.23 | -6.309% |
| 20260818 | enabled | 17,397.21 | 16,356.62 | -5.981% |
| 20260819 | disabled | 17,404.29 | 16,275.89 | -6.483% |
| 20260820 | enabled | 17,396.03 | 16,364.24 | -5.931% |
| median paired result | balanced | 17,400.75 | 16,345.42 | **-6.145%** |

All four paired E2E results pass the suggested maximum 10% throughput-loss
budget.

## Rejected top-p candidate (not release evidence)

`raw/rejected/topp_exact_checked_candidate_e2e.json` is preserved byte-for-byte
at the task owner's request. It belongs to the rejected checked top-p prototype
documented by `42affe47b3a97bb0ef2470f31c1e8b311cbc7a21`, relative to release baseline
`b02f1efed6b33090fc6d23b1da0793c6365962c8`. Its implementation was reverted
and is not retained as a benchmarkable commit.

The raw file embeds the model, Torch/CUDA/GPU, protocol sizes, and observations,
but does not embed a code commit, seed, Transformers version, driver, or command
line. The manifest records that provenance limitation explicitly. The candidate
lost 36.57% B=256 E2E throughput and remains rejected; it is not included in any
release aggregate or gate above, and no top-p rerun was performed here.

## Validation

From the repository root:

```bash
/venv/main/bin/python benchmarks/sampling_evidence/validate_provenance.py \
  --check-model
```

The validator checks every harness/raw hash, embedded commit/environment/seed
field, process counts/order balance, and all derived release aggregates. The
optional model check verifies the three large/local model identity hashes.
