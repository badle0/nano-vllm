#!/usr/bin/env bash
# Rebuild the nano-vllm environment on a fresh Vast.ai instance.
#
# Verified: A100-SXM4-40GB, Ubuntu image with /venv/main (Python 3.12).
# Order is load-bearing: torch pins the ABI, and flash-attn must not move it —
# hence --no-build-isolation (build against the installed torch, not a fetched one)
# and --no-deps on the editable install (keeps pip from upgrading torch under the
# flash-attn wheel). xxhash is a nano runtime dep that --no-deps therefore skips.
set -euo pipefail
PIP="pip install --break-system-packages"

# 1. Clear any preinstalled torch stack (images often ship cu130 builds).
pip uninstall -y --break-system-packages torch torchvision torchcodec || true

# 2. torch first — the ABI anchor. cu128 wheels run fine on newer drivers.
$PIP "torch==2.10.*" --index-url https://download.pytorch.org/whl/cu128

# 3. Light deps, including xxhash (block_manager prefix-cache hashing).
$PIP transformers accelerate tqdm pytest huggingface_hub xxhash

# 4. flash-attn LAST, built against the torch installed above.
$PIP flash-attn==2.8.1 --no-build-isolation

# 5. The package itself, deps guarded.
pip install -e . --no-deps --break-system-packages

# 6. Weights (~1.5 GB).
[ -d "$HOME/huggingface/Qwen3-0.6B" ] || \
  hf download Qwen/Qwen3-0.6B --local-dir "$HOME/huggingface/Qwen3-0.6B"

# 7. Verify and print an environment record for benchmark provenance.
python - <<'PY'
import torch, flash_attn, transformers
from nanovllm import LLM, SamplingParams, StreamOutput, StreamingDetokenizer
print("env ok")
print(f"torch {torch.__version__}  cuda {torch.version.cuda}  "
      f"flash-attn {flash_attn.__version__}  transformers {transformers.__version__}")
print(f"gpu {torch.cuda.get_device_name(0)}  "
      f"{torch.cuda.get_device_properties(0).total_memory // 2**20} MiB")
PY
echo "host: $(nproc) vCPU, $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"
df -h /workspace | tail -1
