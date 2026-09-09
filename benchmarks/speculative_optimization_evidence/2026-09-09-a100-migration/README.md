# Replacement A100 qualification evidence — 2026-09-09

These archives preserve the fresh Qwen3-4B / Qwen3-0.6B qualification produced
from commit `74eac2f9512e737d7eb1f6a1d2c97eebb5efc9ee`. See the
[qualification record](../../../docs/REPLACEMENT_A100_QUALIFICATION.md).

`manifest.json` records SHA256 and byte length for every archive and member.
Each archive contains its original report, raw results, and SHA256SUMS;
benchmark archives also contain execution and analysis scripts. Failed
GPU automatic-sizing runs are retained. Model weights and compiler caches
are excluded. Extract all four archives into sibling directories named after
the archive stems to preserve relative evidence links. Original absolute paths
in raw reports describe the producer instance, not a portable installation.

| Archive | Contents |
| --- | --- |
| `qwen4-first-qualification.tar.gz` | Numerical matrix and FP64 primitive checks |
| `speculative-target-draft-arrival.tar.gz` | Initial graph/invariant lifecycle smokes |
| `performance-five-pair-arrival.tar.gz` | Five-pair repaired ordinary/speculative comparison and pytest |
| `speculative-complete-arrival.tar.gz` | Strict verification, before/after comparison, failures and final pytest |

Full verification did **not** pass: automatic KV sizing regressed. Fixed
64-block lifecycle and benchmark configurations passed. These artifacts do
not certify automatic sizing or every production workload.
