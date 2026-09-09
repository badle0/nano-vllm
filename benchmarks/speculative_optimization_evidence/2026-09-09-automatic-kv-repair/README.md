# Automatic-KV accounting repair evidence

See the [repair qualification](../../../docs/AUTOMATIC_KV_REPAIR.md).
`evidence.tar.gz` retains the original probes, two unsuccessful repair attempts,
the final passing lifecycle sweeps and pytest results, exact source patch,
source hashes, and execution/analysis scripts. `qualified/` contains the final
passing runs; `final/` is the preserved intermediate attempt and is superseded.
`analysis.json` identifies each revision and outcome.

`manifest.json` records the archive checksum and every payload file checksum;
the archive also includes SHA256SUMS. Compiler caches and model weights are
excluded. Scripts retain the producer instance's absolute model/environment
paths and require adapting those paths when reproducing elsewhere.

The final tests cover the recorded A100/Qwen3 single-GPU configuration.
This is lifecycle and memory-accounting verification, not a new five-pair
performance certificate.
