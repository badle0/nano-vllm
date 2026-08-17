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

## Benchmark

See `bench.py` for benchmark.

The reproducible request-metrics overhead A/B protocol, provenance manifest,
and byte-for-byte raw results are in `benchmarks/request_metrics/`.

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
