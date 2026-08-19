<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

> [!NOTE]
> This repository is a maintained fork of
> [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm).
> The original project and authors retain their attribution. Fork-specific
> development is integrated on `fork-main`; `main` remains aligned with the
> original upstream repository.

A lightweight vLLM implementation built from scratch.

See [the branch policy](docs/BRANCHES.md) for the relationship between
`main`, `fork-main`, release branches, and the retained repair branches.

## Key Features

-  **Fast offline inference** — Comparable inference speeds to vLLM
-  **Readable implementation** — A compact core designed for learning and experimentation
-  **Optimization suite** — Prefix caching, tensor parallelism, Torch compilation, and CUDA graphs
-  **Sampling controls** — Greedy, top-k, and top-p sampling
-  **Request metrics** — Queue, first-token, inter-token, engine, and caller latency measurements
-  **Token streaming** — Synchronously backpressured, request-scoped streaming
-  **Chunked prefill** — Bounded admission, mixed-step scheduling, and ragged CUDA-graph routing

## Installation

Clone and install the maintained fork:

```bash
git clone --branch fork-main https://github.com/badle0/nano-vllm.git
cd nano-vllm
pip install .
```

FlashInfer is available as an optional, sorting-free top-p sampling backend:

```bash
pip install '.[fast-sampling]'
```

The optional extra pins the measured FlashInfer 0.6.17 release and the
certified Torch 2.10/CUDA-Python 12.x lane. This prevents pip from silently
replacing a CUDA 12 environment with FlashInfer's newer default CUDA 13 stack,
which can invalidate compiled extensions such as FlashAttention.

FlashInfer's wheel also brings a broader CUDA/JIT dependency footprint than
nano-vLLM's core installation. Treat it as a deployment-level choice rather
than a small sampler-only dependency.

## Model Download

To download the model weights manually:

```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for additional usage. The API mirrors vLLM's interface with
minor differences in `LLM.generate()`:

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    "/YOUR/MODEL/PATH",
    enforce_eager=True,
    tensor_parallel_size=1,
)

sampling_params = SamplingParams(
    temperature=0.6,
    max_tokens=256,
)

prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
print(outputs[0]["text"])
```

Call `llm.exit()` when the engine is no longer needed.

## Fork-Specific Features

### Request Metrics

Each `generate()` result includes a `metrics` dictionary. Fields prefixed with
`engine_` start when tokenization has finished and the sequence is ready for the
scheduler. `submission_to_*` fields start at the public API boundary, while
`caller_e2e` ends after the returned text has been decoded and assembled.

All prompts passed to one `generate()` call share one submission timestamp and
one final delivery timestamp. This models the call as a batch: later prompts do
not receive an artificially younger submission time merely because tokenization
is sequential.

Low-level callers retain the original `step()` contract of
`(seq_id, token_ids)` pairs. Use `step_with_metrics()` to opt in to completed
triples of `(seq_id, token_ids, metrics)`.

The reproducible request-metrics overhead protocol, provenance manifest, and
raw results are stored under `benchmarks/request_metrics/`.

### Token Streaming

Only one synchronous `generate()` or `stream()` session can own an engine at a
time. Use the stream as a context manager when iteration may stop early:

```python
from nanovllm import StreamingDetokenizer

detokenizer = StreamingDetokenizer(llm.tokenizer)
rendered = {}

with llm.stream(prompts, sampling_params) as stream:
    for event in stream:
        text = rendered.get(event.seq_id, "")
        text = detokenizer.feed(event.seq_id, event.token_id).apply(text)

        if event.finished:
            text = detokenizer.flush(event.seq_id).apply(text)

        rendered[event.seq_id] = text
```

`StreamOutput` contains `(seq_id, token_id, finished)`. Text updates can replace
an earlier suffix because tokenizer cleanup and normalization are not always
append-only.

Completed caller-delivery metrics are available in
`stream.metrics[seq_id]`. A retained iterator is not closed by `break` alone;
the context manager or an explicit `stream.close()` performs ID-scoped cleanup.

The ownership lock makes simultaneous `generate()` and `stream()` starts fail
atomically. It does not make the engine a generally thread-safe dispatcher.
Serialize public engine access in one application thread. Concurrent or
asynchronous dispatch requires a separate central step owner and is outside the
scope of this synchronous API.

### Sampling

Greedy sampling uses `temperature=0`. Positive temperatures use stochastic
sampling and can be combined with per-request top-k and top-p parameters.

Fresh-process repaired greedy and top-k evidence is documented under
`benchmarks/sampling_evidence/`.

#### Top-p Backends

The default `top_p_backend="exact"` matches the audited Transformers 5.14.1
ascending-sort boundary and tie behavior. It also preserves nano-vLLM's existing
full-vocabulary exponential fixed-seed sampling stream.

For workloads where all-active top-p throughput matters more than fixed-seed
parity with the exact implementation, construct the engine with the optional
sorting-free backend:

```python
llm = LLM(
    "/YOUR/MODEL/PATH",
    top_p_backend="flashinfer",
)
```

`top_p_backend="flashinfer"` is deterministic for a fixed FlashInfer runtime
and seed, but it is a different sampling contract:

- Boundary ties may produce a different support.
- Philox consumption differs.
- The same seed is not expected to produce the same tokens as `"exact"`.
- Downstream CUDA RNG state is not expected to match `"exact"`.

Existing top-k filtering is still applied before top-p. FlashInfer kernels are
warmed during engine construction. Production deployments should install and
prebuild the optional kernel cache instead of compiling it during the first
served request.

If any stochastic row enables top-p, the fast backend samples the whole batch
with FlashInfer. Rows with `top_p=1.0` remain mathematically unfiltered, but
they also use FlashInfer's RNG stream and therefore lose exact-backend
fixed-seed parity in that mixed batch.

The exact and rejected-candidate measurements are documented in
`benchmarks/topp_performance/README.md`. On the pinned A100 B256 gate, the
production wrapper measured 1.017 ms versus 11.649 ms for the complete exact
sampling path. Eight fresh-process Qwen3-0.6B pairs measured a 59.56% median
end-to-end throughput improvement. These are workload-specific results, not a
universal speedup.

### Chunked Prefill and GC Control

For workloads sensitive to process-wide cyclic-GC pauses, an engine can
explicitly opt in with:

```python
llm = LLM(
    "/YOUR/MODEL/PATH",
    disable_python_gc=True,
)
```

The default is `False`. Suppression begins only after successful engine
initialization. Overlapping opted-in engines share a locked, reference-counted
lease. The final `exit()`, including its `atexit` path, restores the state that
existed before the first lease was acquired.

The lease is cooperative: unrelated code must not toggle cyclic GC while it is
active. This option currently supports `tensor_parallel_size=1` only.

Historical manually GC-disabled phase evidence met the tau-256 latency target,
but those artifacts lack the model-content and source pins required to certify
current code.

The retained full-completion certification is intentionally stricter:

- Tau 256 passed only 3 of 5 fresh runs and is therefore **not latency certified**.
- Tau 512 is classified as a throughput/TTFT profile and is not eligible for
  latency certification.
- One run can never certify a configuration.

The retained workflow, immutable artifacts, and validators are stored under
`benchmarks/chunked_prefill_tail/`. Re-run certification after any source,
model, software, hardware, or workload change.

## Known Certification Limits

- Exact all-active top-p remains expensive. The fast backend is opt-in because
  it uses a different tie and RNG contract.
- Real two-GPU tensor-parallel NCCL inference remains unverified.
- Chunked-prefill tau-256 latency has not passed the strict five-run release
  gate.
- Tau 512 remains a throughput/TTFT tradeoff rather than a latency profile.
- Performance results apply only to their recorded hardware, model, software,
  and workload configurations.

## Benchmarks

See `bench.py` for the original benchmark entry point.

Fork-specific reproducibility material is stored in:

- `benchmarks/request_metrics/`
- `benchmarks/sampling_evidence/`
- `benchmarks/topp_performance/`
- `benchmarks/chunked_prefill_tail/`
- `benchmarks/pr5_results/`
- `benchmarks/pr6/`

### Original Upstream Benchmark

The following result is retained from the original upstream project.

**Configuration:**

- Hardware: RTX 4070 Laptop, 8 GB
- Model: Qwen3-0.6B
- Requests: 256 sequences
- Input length: Randomly sampled between 100 and 1,024 tokens
- Output length: Randomly sampled between 100 and 1,024 tokens

| Inference engine | Output tokens | Time (s) | Throughput (tokens/s) |
| --- | ---: | ---: | ---: |
| vLLM | 133,966 | 98.37 | 1,361.84 |
| Nano-vLLM | 133,966 | 93.41 | 1,434.13 |

## Attribution

This fork preserves the original project's MIT license, authorship, and project
history. Upstream development is available at
[GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm).

## Upstream Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)