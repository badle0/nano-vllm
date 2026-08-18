<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

FlashInfer is available as an optional, sorting-free top-p sampling backend:

```bash
pip install '.[fast-sampling]'
```

The optional extra pins the measured FlashInfer 0.6.17 release and the
certified Torch 2.10/CUDA-Python 12.x lane. This prevents pip from silently
replacing a CUDA 12 environment with FlashInfer's newer default CUDA 13 stack,
which can invalidate compiled extensions such as FlashAttention. FlashInfer's
wheel also brings a broader CUDA/JIT dependency footprint than nano-vLLM's core
installation, so treat this as a deployment-level choice rather than a tiny
sampler-only wheel.

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

### Request metrics

Each `generate()` result includes a `metrics` dictionary. Fields prefixed with
`engine_` start when tokenization has finished and the sequence is ready for the
scheduler. `submission_to_*` fields start at the public API boundary, while
`caller_e2e` ends after the returned text has been decoded and assembled.

All prompts passed to one `generate()` call share one submission timestamp and
one final delivery timestamp. This models the call as a batch: later prompts do
not get an artificially younger submission time merely because tokenization is
sequential.

Low-level callers retain the original `step()` contract of
`(seq_id, token_ids)` pairs. Use `step_with_metrics()` to opt in to completed
triples of `(seq_id, token_ids, metrics)`.

### Token streaming

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
append-only. Completed caller-delivery metrics are available in
`stream.metrics[seq_id]`. A retained iterator is not closed by `break` alone;
the context manager or explicit `stream.close()` performs ID-scoped cleanup.
The ownership lock makes simultaneous `generate()`/`stream()` starts fail
atomically; it does not make the engine a generally thread-safe dispatcher.
Serialize all public engine access in one application thread. Concurrent or
asynchronous request dispatch requires a separate central step owner and is not
part of this synchronous API.

### Top-p sampling backends

The default `top_p_backend="exact"` matches the audited Transformers 5.14.1
ascending-sort boundary and tie behavior. It also preserves nano-vLLM's existing
full-vocabulary exponential fixed-seed sampling stream.

For workloads where all-active top-p throughput matters more than fixed-seed
parity with that implementation, construct the engine with the optional
sorting-free backend:

```python
llm = LLM(
    "/YOUR/MODEL/PATH",
    top_p_backend="flashinfer",
)
```

`top_p_backend="flashinfer"` is deterministic for a fixed FlashInfer runtime
and seed, but it is a different sampling contract: boundary ties may select a
different support, its Philox consumption differs, and the same seed is not
expected to produce the same tokens or downstream CUDA RNG state as `"exact"`.
Existing top-k filtering is still applied before top-p. FlashInfer kernels are
warmed while the engine is constructed; production deployments should install
and prebuild the optional kernel cache rather than compile it on the first
served request.

If any stochastic row enables top-p, the fast backend samples the whole batch
with FlashInfer. Rows with `top_p=1.0` remain mathematically unfiltered, but
they also use FlashInfer's RNG stream and therefore lose exact-backend
fixed-seed parity in that mixed batch.

The exact and rejected-candidate measurements behind this choice are recorded
in `benchmarks/topp_performance/README.md`. On the pinned A100 B256 gate, the
production wrapper measured 1.017 ms versus 11.649 ms for the exact complete
sampling path; eight fresh-process Qwen3-0.6B pairs had a +59.56% median E2E
throughput gain. These are workload-specific results, not a universal speedup.

## Benchmark

See `bench.py` for benchmark.

The reproducible request-metrics overhead A/B protocol, provenance manifest,
and byte-for-byte raw results are in `benchmarks/request_metrics/`.

Fresh-process repaired greedy/top-k release evidence and retained raw results
are documented in `benchmarks/sampling_evidence/`.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
