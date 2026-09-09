# Move the reviewed runtime to a replacement Vast instance

The current rental has a 16 GiB container disk and approximately 215 MiB free.
The pinned target and draft checkpoint files need 9,580,135,869 bytes together.
Vast cannot resize an existing container disk; a replacement instance with more
disk is the practical route. See the official [instance storage FAQ](https://docs.vast.ai/guides/reference/faq/instances).

The source and migration tooling are preserved on
`fix/invariant-speculative-review`. The reviewed implementation was committed as
`3710c76`; `deployment/a100-2026-09-09/environment.json` records its full commit
ID. The handoff manifest records the later commit containing this procedure.
`fork-main` remains at `198e1e6`. No remote push or merge is part of this handoff.

## 1. Choose the replacement instance

Use one **NVIDIA A100-SXM4-40GB**, Linux x86_64, preferably Ubuntu 24.04 with the
Vast PyTorch template. Allocate **100 GB of container disk**. This is capacity
guidance allowing room for the Python environment, 9.58 GB of model files,
compiler caches, installation temporary files, and new evidence. The bootstrap
requires at least 30 GiB free before installing. Prefer at least 64 GB allocated
host RAM and 16 CPU threads for comfortable setup and diagnostics; record the
actual allocation before comparing performance.

The old machine has an AMD EPYC 7713 host CPU and NVIDIA driver 570.133.20.
Use a host whose driver supports CUDA 12.8. Keep the host-injected NVIDIA driver;
do not install the `cuda` or `cuda-drivers` metapackage. A different CPU, driver,
GPU form factor, power setting, or shared-host load requires new performance
qualification, even if the environment checks pass. An A100 80 GB or PCIe card
is a different benchmark baseline; the supplied checker deliberately requires
the recorded SXM4 40 GB model.

The exact old container image tag was not recorded. This procedure rebuilds the
recorded Python/package/native-wheel stack rather than claiming a byte-identical
container image. Record the replacement image tag or digest from the rental UI.

A local volume is optional if you want model data to outlive container deletion;
it is tied to one physical host and cannot follow you to another machine.
[Vast volume documentation](https://docs.vast.ai/guides/instances/storage/volumes)
describes creating and attaching it when renting. Confirm its actual mount path.
The current `/workspace` is **not** backed by a volume. Stopping preserves its
data; destroying or recycling this instance loses it. Keep an off-instance copy
of the handoff archive before retiring the old rental.

## 2. Copy the recovery archive off the old instance

The prepared files are:

```text
/workspace/nano-vllm-migration-2026-09-09.tar.gz
/workspace/nano-vllm-migration-2026-09-09.tar.gz.sha256
```

The archive contains a self-contained Git bundle of local refs and reachable
history, both source-review snapshots, the original standalone model downloader,
the runbook, a handoff manifest, validation records, and SHA-256 checksums.
It excludes model weights, virtual environments, credentials, editor data, and
compiler caches. The installed environment is reconstructed from the lock files.

On your laptop, replace `OLD_HOST` and `OLD_PORT` with the old rental's SSH details:

```bash
scp -P OLD_PORT root@OLD_HOST:/workspace/nano-vllm-migration-2026-09-09.tar.gz .
scp -P OLD_PORT root@OLD_HOST:/workspace/nano-vllm-migration-2026-09-09.tar.gz.sha256 .
sha256sum -c nano-vllm-migration-2026-09-09.tar.gz.sha256
```

On macOS, use `shasum -a 256 -c` in place of `sha256sum -c`.
Then upload those two files to the replacement instance:

```bash
scp -P NEW_PORT nano-vllm-migration-2026-09-09.tar.gz \
  nano-vllm-migration-2026-09-09.tar.gz.sha256 root@NEW_HOST:/workspace/
```

Do not delete the old instance until the archive checksum and replacement
checkout have been verified. Git commits on the old disk alone are not a backup.

## 3. Restore the exact checkout and snapshots

Run on the new instance, using a fresh destination:

```bash
set -e
cd /workspace
sha256sum -c nano-vllm-migration-2026-09-09.tar.gz.sha256
tar -xzf nano-vllm-migration-2026-09-09.tar.gz
cd /workspace/nano-vllm-migration-2026-09-09
sha256sum -c SHA256SUMS
git clone --branch fix/invariant-speculative-review \
  /workspace/nano-vllm-migration-2026-09-09/nano-vllm.bundle /workspace/nano-vllm
git -C /workspace/nano-vllm bundle verify \
  /workspace/nano-vllm-migration-2026-09-09/nano-vllm.bundle
cp -a nano-vllm-review-snapshots /workspace/
python3 - <<'PY'
import json, subprocess
manifest = json.load(open('/workspace/nano-vllm-migration-2026-09-09/manifest.json'))
actual = subprocess.check_output(
    ['git', '-C', '/workspace/nano-vllm', 'rev-parse', 'HEAD'], text=True
).strip()
assert actual == manifest['handoff_commit'], (actual, manifest['handoff_commit'])
print('Restored exact handoff commit:', actual)
PY
git -C /workspace/nano-vllm status --short --branch
```

The bundle clone's `origin` points at the bundle. Restore normal remote naming:

```bash
git -C /workspace/nano-vllm remote rename origin migration
git -C /workspace/nano-vllm remote add origin git@github.com:badle0/nano-vllm.git
```

This performs no network write. Add your SSH credentials separately if you later
want to push; the bundle intentionally contains no credentials. The historical
archive branches are available through the `migration/*` remote refs.

## 4. Rebuild the Python environment

The recorded stack includes:

| Component | Pin |
| --- | --- |
| Python | 3.12.13 |
| PyTorch | 2.10.0+cu128 |
| Triton | 3.6.0 |
| FlashAttention | 2.8.1, CUDA 12 / Torch 2.10 / C++11 ABI / CPython 3.12 wheel |
| Transformers | 5.14.1 |
| Hugging Face Hub | 1.18.0 |
| Safetensors | 0.8.0 |
| NumPy | 2.4.6 |
| pytest | 9.1.1 |

All 94 installed non-project package versions are captured in
[`environment.json`](../deployment/a100-2026-09-09/environment.json). The bootstrap
installs the exact Torch CUDA wheel separately, then the dependency lock, then
the FlashAttention wheel with its SHA-256, then this checkout with `--no-deps`.
Package versions are pinned; only the native FlashAttention wheel has a retained
upstream artifact checksum. This is not a fully offline wheelhouse.

```bash
cd /workspace/nano-vllm
bash deployment/a100-2026-09-09/bootstrap.sh
source /workspace/venvs/nano-vllm/bin/activate
python -m pip check
```

The script uses the Vast image's `uv` to install Python 3.12.13, creates an
isolated venv, checks versions and the reviewed runtime hash, and executes a
tiny native FlashAttention smoke. It does not install optional FlashInfer or
change the host driver. Existing JIT caches are regenerated on the new machine.
The standalone package version becomes the source's `0.3.0rc1`; stale old
editable-install metadata reported `0.2.0` and is deliberately not reproduced.

## 5. Download and verify the pinned models

```bash
mkdir -p /workspace/models
python deployment/a100-2026-09-09/download_models.py
python deployment/a100-2026-09-09/download_models.py --verify-only
```

| Role | Final directory | Hugging Face revision |
| --- | --- | --- |
| Target | `/workspace/models/Qwen3-4B` | `1cfa9a7208912126459214e8b04321603b3df60c` |
| Draft | `/workspace/models/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` |

The downloader verifies sizes, archived SHA-256 hashes, upstream Git blob IDs
for small files, and safetensors headers/index coverage. It uses hidden staging
directories, supports interrupted-download reuse, and renames a checkpoint to
its final location only after verification. Verification records live outside
the model directories under `/workspace/models/.verification/`. The 4B download
still needs to happen on the new instance; it was blocked by disk capacity here.

For a volume mounted elsewhere, use `--models-root /YOUR/VOLUME/models` after
creating that directory and pass the resulting absolute model paths to the
harnesses. The full pytest fixture currently assumes
`/workspace/models/Qwen3-0.6B`, so retaining `/workspace/models` as the common
parent is simplest. Do not restore the broken `/root/huggingface/Qwen3-4B` link
to `/dev/shm`; shared memory is not suitable for retaining the checkpoint.

The legacy model fingerprint hashes download-cache metadata and absolute model
paths as well as model content. Fresh downloads can therefore differ from old
aggregate fingerprints while their actual weights match. Leave historical
artifacts unchanged and make each reference/candidate pair on the replacement
instance against one unchanged checkpoint directory.

## 6. Validate the migration before continuing qualification

```bash
cd /workspace/nano-vllm
source /workspace/venvs/nano-vllm/bin/activate
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/workspace/nano-vllm
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NANOVLLM_TEST_TOKENIZER_PATH=/workspace/models/Qwen3-0.6B
export CUDA_VISIBLE_DEVICES=0
mkdir -p /workspace/verification
python deployment/a100-2026-09-09/check_environment.py \
  --output /workspace/verification/environment-arrival.json
python benchmarks/validate_risk_numerical_review.py \
  benchmarks/invariant_qualification_evidence/2026-09-08-a100-post-review
python -m pytest -q -p no:cacheprovider
```

The retained suite result was 1,080 passed and one skipped. Archive validation
only checks retained evidence integrity; it is not a fresh numerical or speed
result. The environment output refuses to overwrite an existing record: use a
new filename on repeat runs.

Then run independent forced-history comparisons, first for the already reviewed
draft-sized model and then for the outstanding target model. These commands
include both known prompts, mixed/block boundaries, prefix reuse, injected
eviction/resumption, contexts through 4096, and an independent FP64 primitive
reference. They can take substantial GPU time and produce new output directories:

```bash
bash deployment/a100-2026-09-09/qualify_numerics.sh \
  /workspace/models/Qwen3-0.6B /workspace/verification/qwen06-arrival
bash deployment/a100-2026-09-09/qualify_numerics.sh \
  /workspace/models/Qwen3-4B /workspace/verification/qwen4-first-qualification
```

Keep fast-mode negative controls separate. The original two disagreements were
observed on Qwen3-0.6B; running those token histories on Qwen3-4B does not imply
that it must reproduce the same divergent tokens. The invariant matrix should
agree within each model. The independent FP64 reference covers primitives, not
an entire model in FP64.

After numerical checks, use `tests/run_speculative_v5_gpu.py` with the explicit
target/draft paths for graph fast-mode and eager invariant-mode lifecycle
smokes. Follow [the numerical review](RISK_AND_NUMERICAL_QUALIFICATION_REVIEW.md)
and [the optimization plan](NUMERICAL_AND_SPECULATIVE_OPTIMIZATIONS.md) for the
remaining gates. The [five-pair performance protocol](SPECULATIVE_BENCHMARKS.md)
must be rerun with fresh environment/source/model pins, including an ordinary
fast-decoder control on the new host. Historical timings do not certify active
speedup on this checkout or replacement hardware.

## 7. What was checked while preparing this handoff

The preserved runtime and every file in the post-review source manifest match
the prior snapshot. The retained review archive validator passes (55 KV and
55 logit comparisons; 264 speculative smoke cycles). The new environment checker
passes against the original installation, and the existing Qwen3-0.6B checkpoint
passes content/header verification. Shell syntax, offline failure paths, and
Git-bundle restoration are checked separately in the handoff validation records.

The exact source-preservation commit retains two pre-existing extra blank lines
at the ends of the numerical harness files; a full patch whitespace check reports
those cosmetic warnings. The snapshot was preserved without editing the harnesses.
No new model inference, full-suite rerun, performance qualification, or complete
bootstrap on a replacement machine was performed as part of this migration task.
