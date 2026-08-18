# Chunked-prefill tail diagnostics

This directory contains release-pinned observers and contract gates. It does
not change scheduler, model, attention, or graph-routing production code. Run
each command from a clean committed checkout in a fresh process; nano-vllm owns
its process group and selected GPU allocation until process exit.

## Evidence contract

Every retained JSON contains the exact argv, clean Git commit/tree/branch,
aggregate SHA-256 plus per-file hashes for the executable source surface, model
path plus content hashes, Python/package/CUDA/GPU environment, and UTC start
time. The source identity is checked before and after execution. Output uses
exclusive creation (`O_EXCL`) and mode `0444`, so an existing result is never
silently replaced.

The step diagnostic records, for every full `LLMEngine._step()`:

- synchronized wall and CUDA-event time;
- actual prefill/decode/total tokens and scheduled rows/segment count;
- candidate and selected CUDA graph key, model route, and graph-miss delta;
- cold/steady label, queue state, mid-chunk owner, emitted events, and finishes;
- prepare-to-model, model, and post-model/sampler wall and CUDA-event spans;
- allocated/reserved CUDA memory and used/free KV blocks, plus run peaks.

The phase CUDA values are spans, not isolated kernel sums. In particular, they
intentionally retain host/runtime gaps between work enqueued at the two marker
events. This distinguishes two plausible tail causes: real model/attention
growth as a long prefix advances, and a slow outer step whose model and sampler
spans remain ordinary. A zero graph-miss counter alone cannot distinguish them.

`cold` means the first `--cold-steps` immediately after the long request is
admitted; later measured steps are `steady`. Conditioning the interactive cohort
before admission is not included in either timing population.

## Decision gates

Treat a cell as a correctness pass only when all of the following hold:

1. The selected key matches the recorded live token/segment shape; graph-backed
   cells have zero miss delta. Tau 64 deliberately has no captured bucket, so
   exactly one eager miss is expected instead.
2. The graph/eager greedy token vectors match. The 4x511 boundary additionally
   requires exactly 2,044 real tokens, four unpadded 511-token segments,
   `max_model_len=512`, selected key `(2048, 5)`, zero misses, and matching
   end-to-end tokens.
3. Actual per-step tokens never exceed tau, decodes remain present on mixed
   steps, and queue/mid-chunk state is consistent with the intended workload.
4. Judge a latency promise from steady route-specific p95/max, not a pooled
   median. Attribute a threshold breach using the phase spans and prefix state.
   In particular, do not certify tau 512 for a sub-10-ms bound merely because
   graph routing hits; compare tau 256 and tau 512 with the same workload.
5. Peak allocated/reserved memory and KV-block consumption must fit the release
   headroom, with no monotonic per-step growth unexplained by longer live KV.

These are diagnostic gates. They identify the limiting component before any
production optimization is proposed.

## Reproducible commands

First pin the exact committed source. `git status --porcelain` must print
nothing. Results belong outside this worktree.

```bash
cd /workspace/nano-vllm-chunk-tail
git status --porcelain
export PIN_COMMIT="$(git rev-parse HEAD)"
export PIN_SOURCE="$(PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 /venv/main/bin/python benchmarks/chunked_prefill_tail/step_diagnostics.py --print-source-sha256)"
mkdir -p /workspace/.feat_bench/chunk-tail
```

Run the high-segment mixed-tail comparison. These commands build 63 live decode
rows, then admit one 2,048-token prompt, reproducing 64-segment mixed steps while
holding every workload parameter except tau constant:

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_CACHE_DIR=/tmp/nv_chunk_tail_tau256 \
  /venv/main/bin/python benchmarks/chunked_prefill_tail/step_diagnostics.py \
  --model /workspace/models/Qwen3-0.6B --tau 256 --max-num-seqs 64 \
  --max-model-len 4096 --interactive-count 63 --long-count 1 \
  --long-prompt-len 2048 --pre-long-steps 80 --measured-steps 96 \
  --max-tokens 256 --cold-steps 3 --seed 20260822 \
  --expected-commit "$PIN_COMMIT" --expected-source-sha256 "$PIN_SOURCE" \
  --output /workspace/.feat_bench/chunk-tail/tail_tau256_seed20260822.json

PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_CACHE_DIR=/tmp/nv_chunk_tail_tau512 \
  /venv/main/bin/python benchmarks/chunked_prefill_tail/step_diagnostics.py \
  --model /workspace/models/Qwen3-0.6B --tau 512 --max-num-seqs 64 \
  --max-model-len 4096 --interactive-count 63 --long-count 1 \
  --long-prompt-len 2048 --pre-long-steps 80 --measured-steps 96 \
  --max-tokens 256 --cold-steps 3 --seed 20260822 \
  --expected-commit "$PIN_COMMIT" --expected-source-sha256 "$PIN_SOURCE" \
  --output /workspace/.feat_bench/chunk-tail/tail_tau512_seed20260822.json
```

Run the explicit maximum-length boundary regression:

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_CACHE_DIR=/tmp/nv_chunk_4x511 \
  /venv/main/bin/python tests/run_varlen_511_contract.py \
  --model /workspace/models/Qwen3-0.6B --seed 20260818 \
  --expected-commit "$PIN_COMMIT" --expected-source-sha256 "$PIN_SOURCE" \
  --output /workspace/.feat_bench/chunk-tail/contract_4x511_maxlen512.json
```

Run the retained tau `{64,128}` by max-model-length `{512,1024,4096}` matrix.
The orchestrator launches one fresh child process per cell and revalidates the
source pin on both sides of every child:

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_CACHE_DIR=/tmp/nv_chunk_contract_matrix \
  /venv/main/bin/python benchmarks/chunked_prefill_tail/contract_matrix.py \
  --model /workspace/models/Qwen3-0.6B \
  --expected-commit "$PIN_COMMIT" --expected-source-sha256 "$PIN_SOURCE" \
  --output /workspace/.feat_bench/chunk-tail/tau64_128_maxlen_matrix.json
```

Each output path must be new. Choose a new seed/name for a repeat; do not chmod
and overwrite retained evidence.
