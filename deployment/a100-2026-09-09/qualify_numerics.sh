#!/usr/bin/env bash
# Fresh independent engines for both sides of every comparison. No performance claims.
set -euo pipefail
MIGRATION_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$MIGRATION_DIR/../.." && pwd)"
MODEL_PATH="${1:?Usage: qualify_numerics.sh MODEL_PATH NEW_OUTPUT_DIRECTORY}"
OUTPUT_DIR="${2:?Usage: qualify_numerics.sh MODEL_PATH NEW_OUTPUT_DIRECTORY}"
mkdir -- "$OUTPUT_DIR"
OUTPUT_DIR="$(cd -- "$OUTPUT_DIR" && pwd)"
cd -- "$REPO_DIR"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export TRITON_CACHE_DIR="$OUTPUT_DIR/triton-cache"
export TORCHINDUCTOR_CACHE_DIR="$OUTPUT_DIR/inductor-cache"
python "$MIGRATION_DIR/check_environment.py" --output "$OUTPUT_DIR/environment.json"

pair() {
    local scenario="$1" ref_budget="$2" ref_layout="$3" budget="$4" layout="$5"
    shift 5
    python benchmarks/invariant_qualification.py --model "$MODEL_PATH" \
        --scenario "$scenario" --budget "$ref_budget" --layout "$ref_layout" \
        --output "$OUTPUT_DIR/$scenario-reference.json"
    python benchmarks/invariant_qualification.py --model "$MODEL_PATH" \
        --scenario "$scenario" --budget "$budget" --layout "$layout" "$@" \
        --compare "$OUTPUT_DIR/$scenario-reference.json" \
        --output "$OUTPUT_DIR/$scenario-candidate.json"
}

pair known_failures 68 packed 4 serial
pair mixed_boundaries 2305 packed 257 packed
pair cache_reuse 1024 serial 257 serial
pair eviction_resume 769 serial 257 serial --inject-eviction
pair long_context 4096 serial 511 serial
python benchmarks/invariant_precision_reference.py --output "$OUTPUT_DIR/fp64-primitives.json"
(
    cd -- "$OUTPUT_DIR"
    sha256sum ./*.json > SHA256SUMS
)
printf 'Numerical matrix complete: %s\n' "$OUTPUT_DIR"
