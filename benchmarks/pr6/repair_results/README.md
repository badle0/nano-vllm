# PR6 repaired chunked-prefill results

This directory preserves the repaired-branch evidence that previously existed
only under `/workspace/.feat_bench`. The 46 artifacts named in `manifest.json`
are byte-identical copies: 33 unique tau JSONs, six stock-throughput JSONs, four
token gates, two external harnesses, and one supplemental smoke output.

The raw result files are authoritative. Summaries below are recomputed from them
by `validate.py`; they are not new benchmark runs.

## Tau A/B adjudication

Each side has three fresh-process results. Tau 128 and 16384 are the final
interleaved reruns. Tau 1024 is a same-host sequential baseline/fixed comparison,
not timestamp-interleaved. Delta is `(fixed / baseline - 1) * 100`.

| tau | mixed tok/s | interactive TTFT | interactive max ITL | long TTFT |
|---:|---:|---:|---:|---:|
| 128 | +0.62% | +0.30% | -14.17% | -8.88% |
| 1,024 | -0.07% | +0.21% | +0.64% | +1.60% |
| 16,384 | +0.51% | +3.95% | +1.45% | +1.45% |

The selected files are explicit in `sets.tau_adjudication` in the manifest.
At tau 128 the fixed `seed1b`–`seed3b` files supersede the initial fixed runs
for A/B; at tau 16384 the `pair1`–`pair3` files supersede the initial branch sets.

## Fixed constructor sweep

The initial fixed `seed1`–`seed3` sets are retained separately because they
support the constructor-configured tau table. These are medians of three fresh
processes and are not baseline comparisons.

| tau | mixed tok/s | interactive TTFT | interactive max ITL | long TTFT |
|---:|---:|---:|---:|---:|
| 128 | 3,098.8 | 20.176 ms | 7.001 ms | 133.277 ms |
| 256 | 3,185.9 | 14.848 ms | 7.099 ms | 72.706 ms |
| 512 | 3,165.8 | 10.566 ms | 9.058 ms | 58.212 ms |
| 1,024 | 2,900.3 | 11.531 ms | 15.456 ms | 53.860 ms |
| 2,048 | 2,953.7 | 11.572 ms | 19.517 ms | 41.791 ms |
| 16,384 | 3,275.0 | 11.101 ms | 36.741 ms | 36.706 ms |

The initial fixed tau-128 and tau-16384 sets remain valid sweep observations,
but the later paired files above are the A/B adjudication. The preliminary
baseline tau-16384 `seed1`–`seed3` set is excluded because `pair1`–`pair3`
superseded it and it supports no constructor-sweep claim.

## Stock and token gates

The three alternating fresh-process stock runs produced medians of 8,779.672
tok/s for dev and 8,761.927 tok/s for chunk: -0.20% by ratio of medians and
-0.05% by median paired delta, inside the <=1% gate.

The four selected token files are named explicitly rather than using the older
ambiguous `token_gate_*` aliases:

- stochastic: dev `5919fcde...`, chunk `b8526e49...`; one sequence differs at
  seven completion positions, first at zero-based index 15;
- greedy: dev `1e10d050...`, chunk `6d9cd726...`; one sequence differs at four
  positions, first at zero-based index 28.

Those are payload hashes embedded in the token JSONs. The manifest separately
pins the SHA-256 of each complete file.

## Supplemental smoke

`smoke/tau128_post_repair.json` is the reported post-repair tau-128 smoke: four
48-token outputs, 15.54 seconds, file SHA-256
`34f5638cafba25e30bbbc56a4bea7e939794660437b3f4234e3cdcd7ec166827`.
It is a single-process generation smoke, not multi-process performance evidence.

## Provenance limits

- Tau and stock JSONs embed GPU, Torch, CUDA, seed, and a short commit label.
  The tau JSONs also embed the caller-supplied model path.
- Both exact off-tree harnesses accept `--commit`; they do not inspect Git HEAD.
  Therefore the labels are preserved as caller-supplied fields, not claims that
  execution was verified at those commits. Full SHA resolution is explicitly
  marked as post-run reconstruction.
- Stock JSONs omit the model path. Token JSONs omit commit, model, and software/
  hardware environment. Their context is labeled post-run reconstruction.
- Raw JSONs do not embed command lines, host IDs, or run timestamps. Command
  templates and source mtimes in the manifest are reconstruction, not embedded
  evidence.
- Local Hugging Face revision and model-file hashes were observed after the run.
  They identify the surviving snapshot but do not prove its run-time bytes.
- No authoritative token-gate generator was retained in the scoped off-tree
  evidence, so this bundle does not claim an exact token-gate command.

## Validation

From the repository root, run:

```bash
/venv/main/bin/python benchmarks/pr6/repair_results/validate.py
```

The validator uses only the Python standard library. It checks every artifact's
byte count and SHA-256, JSON schemas and embedded fields, token payload hashes,
set membership, recomputed medians/deltas, and the smoke shape/classification.
