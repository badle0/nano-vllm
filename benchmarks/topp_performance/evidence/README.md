# Fast top-p release evidence

This directory archives the evidence for nano-vLLM's optional FlashInfer
top-p backend.  It does **not** justify replacing the default exact backend:
the two backends intentionally expose different tie and fixed-seed contracts.

## Result

On the pinned A100-SXM4-40GB, Torch 2.10.0+cu128, Transformers 5.14.1,
FlashInfer 0.6.17 stack, with B256 x V151936 BF16 logits, temperature 0.6,
and p=0.9:

| Complete sampling route | CUDA median | p95 | Peak transient |
|---|---:|---:|---:|
| Exact filter + exponential sampler | 11.649 ms | 11.816 ms | 616.96 MiB |
| FlashInfer direct primitive | 0.912 ms | 0.971 ms | 296.75 MiB |
| Production `sample_top_p_flashinfer` wrapper | 1.017 ms | 1.067 ms | 296.75 MiB |

The production wrapper passes the 1.6 ms median gate and is 11.46x faster
than the exact complete route in this sampler benchmark. Routes were measured
sequentially as exact, direct primitive, then production wrapper; the exact
samples shifted late in that run, so 11.46x is descriptive rather than a
portable ratio. The absolute wrapper gate is stronger: its median, p95, and
maximum observed latency were 1.017, 1.067, and 1.295 ms, so all 25 samples
were below 1.6 ms.

The B256 Qwen3-0.6B end-to-end workload used 128 prompt tokens, 32 output
tokens, CUDA graphs, and six observations per fresh process. Observation zero
was retained but excluded as shape-cold; each process median uses its other
five observations. Exact and FlashInfer ran in separate processes, and each
paired ratio is one process-pair timing replicate.

| Pair | Order | Exact tok/s | FlashInfer tok/s | Ratio |
|---:|---|---:|---:|---:|
| 1 | exact -> FlashInfer | 10,742.5 | 17,097.0 | 1.5915x |
| 2 | FlashInfer -> exact | 10,587.8 | 17,109.5 | 1.6160x |
| 3 | FlashInfer -> exact | 10,702.6 | 16,889.1 | 1.5780x |
| 4 | exact -> FlashInfer | 10,731.8 | 17,184.6 | 1.6013x |
| 5 | exact -> FlashInfer | 10,745.8 | 17,105.1 | 1.5918x |
| 6 | FlashInfer -> exact | 10,732.0 | 17,214.2 | 1.6040x |
| 7 | FlashInfer -> exact | 10,735.9 | 17,156.6 | 1.5981x |
| 8 | exact -> FlashInfer | 10,717.5 | 17,075.1 | 1.5932x |

The median paired ratio is 1.5956x, or +59.56%. Every pair improved. The
mean ratio is 1.5967x; a two-sided Student t interval over the eight pair
ratios is 1.5875x to 1.6060x. This interval describes this pinned
workload/runtime and treats the eight fresh-process measurements as its units.
The protocol contains four prompt seeds, each mirrored in the opposite order,
so the interval is not an eight-corpus-sample generalization and is not a
universal model, batch, GPU, or traffic claim.
The order-stratified medians are 1.5925x with exact first and 1.6010x with
FlashInfer first.

Prompts and pre-run CUDA RNG-state hashes match within every pair. The four
mirrored-seed pair groups reproduce prompt, token, and final RNG-state hashes
within each backend across fresh processes. Each repetition uses a distinct
prompt set, capacity metadata proves that KV preemption was unnecessary, and
the raw files embed a five-file model execution fingerprint, source hashes,
environment, order, and observed package provenance. The fingerprint includes
the weights, configs, and fast `tokenizer.json`, but not a Hugging Face Hub
repository/revision or separate `vocab.json` and `merges.txt` hashes.
End-to-end timing includes prefill, host scheduling, generation, and tokenizer
decoding; it is not a sampler-only or model-TTFT metric. The two backends take
different token trajectories, so this is their actual production E2E behavior,
not a same-output kernel-only comparison.

The measured scope is warmed steady-shape TP1 on one A100, Qwen3-0.6B, B256,
p=0.9, and temperature 0.6. Other models, batches, p/temperature values,
hardware, and tensor-parallel configurations require their own measurements.

## Semantic boundary

`top_p_backend="exact"` remains the default. It preserves the audited
Transformers 5.14.1 ascending-sort boundary/tie behavior and nano-vLLM's
full-vocabulary exponential sampling stream.

`top_p_backend="flashinfer"` is an explicit opt-in statistical backend. It is
repeatable when the same seed and pinned runtime are reset, but it uses a
different Philox stream and treats boundary ties differently. In the clean
micro run, exact and FlashInfer differed on 256/256 fixed-seed rows and ended
at different CUDA RNG offsets. The unique-logit distribution test used
131,072 draws, sampled nothing outside the mathematical nucleus, and passed
its six-sigma residual gate. Uniform and forced-boundary-tie tests intentionally
record the support difference.

## Dependency note

The benchmark loaded FlashInfer 0.6.17 from an isolated `--no-deps` target, so
the E2E raw environment accurately records CUDA-Python as absent. The project
extra separately constrains the certified deployment lane:

```text
flashinfer-python==0.6.17
torch>=2.10,<2.11
cuda-python>=12,<13
```

A resolver dry run retained the installed Torch 2.10.0+cu128, Triton 3.6.0,
and FlashAttention 2.8.1. That dry run is post-run packaging evidence, not an
embedded benchmark observation. FlashInfer's standard wheel brings a broader
CUDA/JIT dependency set, so operators should prebuild the kernel cache.

## Validate

`provenance.json` pins every raw and harness SHA-256, commit, environment,
model fingerprint, pair order, configuration, and derived result. Validate
the archive with:

```bash
PYTHONPATH=. /venv/main/bin/python \
  benchmarks/topp_performance/evidence/validate_provenance.py

PYTHONPATH=. /venv/main/bin/python \
  benchmarks/topp_performance/evidence/validate_provenance.py --check-model
```

The stronger command reports:

```text
validated 1 micro artifact, 16 E2E raw files, 8 fresh-process pairs, 2 harnesses, and 5 model files
```
