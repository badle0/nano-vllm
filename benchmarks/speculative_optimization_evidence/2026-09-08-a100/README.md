# A100 optimization smoke evidence

These four reports were produced from the dirty implementation rooted at base
commit `198e1e6`; each JSON embeds SHA-256 values for every runtime source file,
the harness hash, model files, Torch/CUDA versions, and GPU identity.

- `fast_speculative_graph.json`: 50 successful cycles, K=1..4, parallel
  stochastic plus sequential compatibility greedy verification, no timed-cycle
  compilation, causality and injected failure/retry passed.
- `fast_ordinary_graph.json`: matched speculation-off smoke protocol.
- `invariant_speculative_eager.json`: 44 successful cycles including one-pass
  invariant greedy verification.
- `invariant_ordinary_eager.json`: matched invariant ordinary reference; all
  four recorded greedy token matrices equal the speculative report.

These are functional smoke artifacts, not the five-pair Qwen3-4B/Qwen3-0.6B
performance gate. They use Qwen3-0.6B for both target and draft and one timed
sample per cell. `SHA256SUMS` covers the immutable JSON artifacts.
