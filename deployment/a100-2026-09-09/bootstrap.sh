#!/usr/bin/env bash
# Run on the replacement rental, after restoring the Git bundle.
set -euo pipefail
MIGRATION_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$MIGRATION_DIR/../.." && pwd)"
MIGRATION_VENV="${NANOVLLM_VENV:-/workspace/venvs/nano-vllm}"

python3 - <<'PY'
import platform, shutil
assert platform.machine() == 'x86_64', 'This wheel lock requires Linux x86_64'
free = shutil.disk_usage('/workspace').free
assert free >= 30 * 1024**3, f'Need 30 GiB free before setup; found {free / 1024**3:.2f} GiB'
PY
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [[ ! -x "$MIGRATION_VENV/bin/python" ]]; then
    if ! command -v uv >/dev/null 2>&1; then
        printf '%s\n' 'uv is required; use the Vast PyTorch image with uv installed.' >&2
        exit 1
    fi
    export UV_PYTHON_INSTALL_DIR=/workspace/.python
    uv python install 3.12.13
    uv venv --seed --python 3.12.13 "$MIGRATION_VENV"
fi
"$MIGRATION_VENV/bin/python" - <<'PY'
import platform
assert platform.python_version() == '3.12.13', 'Use Python 3.12.13 for this lock'
PY

# --no-deps prevents pip from replacing Torch while installing the native wheel.
# The complete dependency inventory is explicitly installed immediately after it.
"$MIGRATION_VENV/bin/python" -m pip install --no-cache-dir --no-deps \
    --index-url https://download.pytorch.org/whl/cu128 'torch==2.10.0+cu128'
"$MIGRATION_VENV/bin/python" -m pip install --no-cache-dir --no-deps \
    --index-url https://pypi.org/simple -r "$MIGRATION_DIR/requirements.lock"
"$MIGRATION_VENV/bin/python" -m pip install --no-cache-dir --no-deps \
    'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl#sha256=f47361af0ebe127db80040cb72e947e3a2ef97c59d641df0c167c5b48cafdc28'
"$MIGRATION_VENV/bin/python" -m pip install --no-deps --no-build-isolation -e "$REPO_DIR"
"$MIGRATION_VENV/bin/python" -m pip check
PYTHONDONTWRITEBYTECODE=1 "$MIGRATION_VENV/bin/python" "$MIGRATION_DIR/check_environment.py"
printf 'Setup complete. Activate with: source %s/bin/activate\n' "$MIGRATION_VENV"
