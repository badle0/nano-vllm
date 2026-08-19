# Request-metrics overhead evidence

This directory preserves the three-pair A/B benchmark used to check that the
request-metrics repair did not materially reduce generation throughput.

## Provenance boundary

The six files under `raw/` are byte-for-byte copies of the historical outputs.
They embed the run label, seed, Torch version, CUDA wheel version, GPU name, and
measurements. They do **not** embed commit IDs, the model path, the Transformers
or Python version, the command line, or pair order.

`provenance.json` records those missing fields as a post-run reconstruction. It
was assembled later on the same still-running instance from the still-present
clean benchmark worktrees, the exact harness, raw-file timestamps, and the
unchanged `/venv/main` environment. This distinction matters: reconstructed
values are pinned for reproduction, but must not be mistaken for metadata
captured by the original process.

The checked-in `metrics_overhead.py` is byte-for-byte identical to the historical
harness (SHA-256 `52d717956323971a6fba0e193e04c5d839c6d199fa5036db6251cb38113942b4`).

## Protocol

Each seed pair ran the baseline first and the repair second, each in a fresh
Python process. The harness:

- loads `/workspace/models/Qwen3-0.6B` with CUDA graphs enabled,
  `max_model_len=1024`, `max_num_seqs=256`, and 80% GPU-memory utilization;
- warms the model with one three-token prompt and two generated tokens;
- measures batches 64 and 256 with 128-token integer prompts and 32 generated
  tokens at temperature 0.6 with EOS ignored;
- performs four observations, retains observation zero as cold evidence, and
  reports the median of observations 1--3; and
- seeds prompt generation with `seed + batch` and Torch with
  `seed * 1000 + batch * 10 + repetition`.

The following is the reconstructed reproduction command sequence. `<artifact>`
is a checkout containing this directory, while `<baseline>` and `<repair>` are
clean worktrees detached at the pinned commits in `provenance.json`.

```bash
for item in 1:20260817 2:20260818 3:20260819; do
  pair=${item%%:*}
  seed=${item##*:}
  (cd <baseline> && PYTHONPATH=. /venv/main/bin/python \
    <artifact>/benchmarks/request_metrics/metrics_overhead.py \
    --label main --model /workspace/models/Qwen3-0.6B --seed "$seed" \
    --output <results>/metrics_fix_main_seed"$pair".json)
  (cd <repair> && PYTHONPATH=. /venv/main/bin/python \
    <artifact>/benchmarks/request_metrics/metrics_overhead.py \
    --label fix-request-metrics --model /workspace/models/Qwen3-0.6B \
    --seed "$seed" \
    --output <results>/metrics_fix_branch_seed"$pair".json)
done
```

Run `/venv/main/bin/python benchmarks/request_metrics/validate_provenance.py`
from the repository root to validate every raw-file hash and embedded metadata
field.

## Result

The repair-versus-baseline change in steady median output throughput was:

| seed | batch 64 | batch 256 |
|---:|---:|---:|
| 20260817 | +1.847% | -0.414% |
| 20260818 | +0.973% | -0.213% |
| 20260819 | +0.616% | +0.145% |
| median paired change | +0.973% | -0.213% |

These are paired comparisons derived from the raw results; they are not extra
fields that were present in the original JSON.

## Prospective certification protocol

The historical artifacts above remain immutable. New release evidence uses:

- `metrics_certify.py` for one GPU process;
- `run_certification.py` to serialize an even number of process pairs in
  alternating `baseline-repair` / `repair-baseline` order; and
- `validate_certification.py` to recompute every observation summary, paired
  efficiency, aggregate gate, token-parity check, and process-order check.

Every raw process refuses to overwrite output and captures its pair/order IDs,
argv, source HEAD/tree/clean status, complete model-directory fingerprint,
selected environment and package versions, prompt/token hashes, cold and steady
measurements, current/peak host RSS, and CUDA allocated/reserved peaks. Output
lengths are configurable; batch and output-length lists form a Cartesian set of
cases. `p95` always refers to steady elapsed latency; the corresponding slow-tail
throughput field is `p05_output_tokens_per_s`.

The service roofline is the no-metrics baseline. The aggregate gate requires
median paired efficiency `T_repair / T_baseline >= 0.98`; a separate 0.95
individual-pair floor catches severe process outliers without treating ordinary
single-run noise as the headline result. Cross-branch token hashes must match for
every observation. Cold efficiency and p95 elapsed ratios are preserved but are
reported diagnostics, not release gates.

Run outputs outside either source worktree so source cleanliness can be checked
before and after every process. For example, after substituting the committed
repair SHA:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
  /venv/main/bin/python -B benchmarks/request_metrics/run_certification.py \
  --run-id metrics-cert-a100-20260818 \
  --baseline-root /workspace/nano-vllm-main-bench \
  --repair-root /workspace/nano-vllm-metrics-cert \
  --expected-baseline-head bb823b3e06983d71485a8e1f23715ebd87d98ef8 \
  --expected-repair-head '<repair-commit>' \
  --model /workspace/models/Qwen3-0.6B \
  --output-dir /workspace/.feat_bench/results/metrics-cert-a100-20260818 \
  --pairs 8 --seed-base 20260821 \
  --batches 64,256 --output-lengths 32 \
  --prompt-length 128 --repetitions 6 --max-model-len 1024

/venv/main/bin/python -B \
  benchmarks/request_metrics/validate_certification.py \
  /workspace/.feat_bench/results/metrics-cert-a100-20260818/manifest.json \
  --check-source --check-model
```

For long-output host-list scaling, use a separate fresh output directory and a
smaller batch set, for example `--batches 1,64 --output-lengths 32,256,2048`.
That separate probe must also set `--max-model-len 4096`. The runner deliberately
refuses to reuse an existing directory, including one left by an interrupted run.

The completed A100 certification is archived byte-for-byte under
`certification/2026-08-18-a100-b3efbdb/`, including all 16 raw process outputs,
the original manifest, recomputed equivalence statistics, and file hashes.
