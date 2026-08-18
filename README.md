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
in `benchmarks/topp_performance/README.md`.

## Benchmark

See `bench.py` for benchmark.

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
