#!/usr/bin/env python3
"""Validate the retained speculative-V2 archive without CUDA or nano-vLLM.

The validator deliberately uses only the Python standard library.  Artifact
hashes establish byte identity; the checks below then independently validate
the registered schemas, cross-artifact relationships, memory arithmetic, and
the deliberately narrow V2 claim boundary.

Input must be a quiescent local Git checkout.  Concurrent hostile mutation
during validation is outside this static archive/CI threat model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any


MANIFEST_SCHEMA_VERSION = 1
ARCHIVE_KIND = "nano_vllm_speculative_v2_retained_archive"
V0_SCHEMA = "nano-vllm-speculative-v0-golden-v1"
LIFECYCLE_SCHEMA = "nano-vllm-speculative-v2-lifecycle-v2"
RECOVERY_SCHEMA = "nano-vllm-speculative-v2-gpu-recovery-v1"

IMPLEMENTATION_COMMIT = "d87f168b804778fbb5888a662dc8a0defccfd660"
IMPLEMENTATION_TREE = "4e98646e71c4550f58df291cf8efbbab71d9e2cb"
IMPLEMENTATION_SOURCE_TREE_SHA256 = (
    "9d6fca30d553ccb6ee6e37b3d6eb755c20e1a0d27e0055eddaca63ac9bd74797"
)
CANONICAL_V0_COMMIT = "480a3b26c5a4e465aac06d1dabd34e1230686feb"
CANONICAL_V0_TREE = "b5982753a5d9d4e7512253002985962c13a2435d"

GPU_NAME = "NVIDIA A100-SXM4-40GB"
GPU_COMPUTE_CAPABILITY = [8, 0]
GPU_MIN_MEMORY_BYTES = 39 * 1024**3
GPU_PID_CHALLENGE_BYTES = 389 * 1024**2
GPU_PID_ROUNDING_TOLERANCE_MIB = 1
POST_EXECUTION_ALLOCATED_CEILING_BYTES = 32 * 1024**2
POST_EXECUTION_RESERVED_CEILING_BYTES = 64 * 1024**2
REGISTERED_MODEL_WEIGHT_BYTES = 1_192_099_840
REGISTERED_MODEL_BLOCK_BYTES = 29_360_128
REGISTERED_JOINT_BLOCK_BYTES = 58_720_256

FP32_BYTES = 4
FP64_BYTES = 8
INT64_BYTES = 8
BOOL_BYTES = 1
TOP_P_CHUNK_SIZE = 64
ALLOCATOR_MARGIN_MIN_BYTES = 64 * 1024**2
ALLOCATOR_MARGIN_ALIGNMENT_BYTES = 2 * 1024**2

AUDIT_REQUIRED_COMPONENTS = [
    "model activation and attention-library workspace",
    "CUDA graph-pool and route-specific static buffers",
    "top-k/top-p backend-internal selection/sort workspace beyond the "
    "documented payload proxy",
    "measured CUDA allocator fragmentation beyond the fixed margin",
]

EXPECTED_PROTOCOL = {
    "configured_k": 2,
    "seed": 20260828,
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.5,
    "max_model_len": 512,
    "max_num_batched_tokens": 512,
    "max_num_seqs": 4,
    "kvcache_block_size": 256,
    "recovery_num_kvcache_blocks": 4,
    "top_p_backend": "exact",
    "vocab_size": 151936,
    "target_logits_itemsize": 2,
    "draft_logits_itemsize": 2,
    "sampling": {
        "temperature": 0.0,
        "max_tokens": 4,
        "ignore_eos": True,
        "top_k": -1,
        "top_p": 1.0,
    },
}

EXPECTED_MANIFEST_PROTOCOL = {
    "seed": 20260828,
    "configured_k": 2,
    "tensor_parallel_size": 1,
    "top_p_backend": "exact",
    "gpu_memory_utilization": 0.5,
    "max_model_len": 512,
    "max_num_batched_tokens": 512,
    "max_num_seqs": 4,
    "kvcache_block_size": 256,
    "recovery_num_kvcache_blocks": 4,
    "workload": {
        "prompt_token_ids": [[1, 2, 3, 4], [7, 8, 9]],
        "sampling": EXPECTED_PROTOCOL["sampling"],
    },
    "recovery_phase_modes": {},
    "recovery_expected_real_phase_calls": {},
}

EXPECTED_CLAIM_BOUNDARY = {
    "certified": [
        "canonical V0 eager and graph greedy output/trace controls",
        "V2 inert dual-model construction and independent ownership",
        "current speculation-off/on output, scheduler-trace, and RNG identity",
        "exact joint-KV automatic/explicit/N+1 capacity preflight",
        "transactional cleanup and all eight registered constructor-failure recovery phases",
    ],
    "not_certified": [
        "draft proposal execution",
        "target verification or acceptance/rejection",
        "multi-token commit, speculative streaming, or speculative metrics",
        "heterogeneous target/draft model geometry",
        "tensor parallelism or FlashInfer",
        "latency, throughput, acceptance rate, or speedup",
    ],
    "feature_stage": "V2 inert dual-model ownership",
    "gpu_isolation_scope": "before/after endpoint checks only",
    "memory_workspace_gpu_certified": False,
}

EXPECTED_RUNNER_HASHES = {
    "nanovllm/engine/speculative_memory.py": (
        "de03b20aee633d3527868936841998b0d75cfb685c3f7ee38042e7b0fb3c6520"
    ),
    "tests/run_speculative_v0_golden.py": (
        "87cc0556daa866009a975ff5f9f38eaa2c182547f9847e51c9c07ad906693a72"
    ),
    "tests/run_speculative_v2_gpu_recovery.py": (
        "5513de7d220a6e5c20ebe95c679cf9ed40241da6a92246fbb64f0eed880eec45"
    ),
    "tests/run_speculative_v2_lifecycle.py": (
        "5eb7facfebbed315ebc2d50fcd8f2f71869e736c026eff1ea339deeea59bbafe"
    ),
}

RECOVERY_PHASE_MODES = {
    "draft_construct": "eager",
    "draft_load": "eager",
    "draft_warmup": "eager",
    "graph_profile": "graph",
    "joint_allocate": "eager",
    "draft_graph": "graph",
    "draft_pretouch": "graph",
    "memory_finalize": "eager",
}
EXPECTED_MANIFEST_PROTOCOL["recovery_phase_modes"] = RECOVERY_PHASE_MODES
EXPECTED_MANIFEST_PROTOCOL["recovery_expected_real_phase_calls"] = {
    phase: 2 if phase == "draft_graph" else 1
    for phase in RECOVERY_PHASE_MODES
}
EXPECTED_REGISTERED_SOFTWARE = {
    "python": "3.12.13",
    "python_implementation": "CPython",
    "torch": "2.10.0+cu128",
    "torch_cuda_build": "12.8",
    "cudnn": 91002,
    "nccl": [2, 27, 5],
    "transformers": "5.14.1",
}

ARTIFACT_SPECS = {
    "raw/v0-eager.json": ("canonical_v0", "eager", None, V0_SCHEMA),
    "raw/v0-graph.json": ("canonical_v0", "graph", None, V0_SCHEMA),
    "raw/v2-eager.json": ("v2_lifecycle", "eager", None, LIFECYCLE_SCHEMA),
    "raw/v2-graph.json": ("v2_lifecycle", "graph", None, LIFECYCLE_SCHEMA),
    **{
        f"raw/recovery-{phase}.json": (
            "recovery",
            mode,
            phase,
            RECOVERY_SCHEMA,
        )
        for phase, mode in RECOVERY_PHASE_MODES.items()
    },
}

# This is the immutable release registry.  The manifest is an index, not a
# trust root: changing an artifact and then updating its manifest row must not
# manufacture a new certificate.
TRUSTED_RAW_ARTIFACTS = {
    "raw/recovery-draft_construct.json": (
        28352,
        "8d9184185a5e29be192dd64063c40e1f6d3ba86ba3dbba36d230a9c061564516",
    ),
    "raw/recovery-draft_graph.json": (
        28468,
        "566d2f8a5e15484a0215e5e8c6d89a828714d55d477fb04296540bf9e9486ee9",
    ),
    "raw/recovery-draft_load.json": (
        28312,
        "44f33219000ce8c248f53b7bb431462f46f926baaa7bf218c85b4ea485cff59f",
    ),
    "raw/recovery-draft_pretouch.json": (
        28492,
        "29ac449e19a23642165de0664cda80fe697b91e73d0e04ea46419a5fe0059165",
    ),
    "raw/recovery-draft_warmup.json": (
        28341,
        "90efcf20415a720b9956990e85c89911451b2c853cc69fed068dae5d28a1daff",
    ),
    "raw/recovery-graph_profile.json": (
        28484,
        "01c61ffff5adbf04370c5ebaec369d9b1c113695766101c44e73b78d0ba46c3d",
    ),
    "raw/recovery-joint_allocate.json": (
        28371,
        "b11bf9a1be24b98bbd4dd03ab80bbbf2ffa2a9837a7afa7514893c76ae6b439e",
    ),
    "raw/recovery-memory_finalize.json": (
        28365,
        "21e25666cbaf3c3fbffae44b152a803db6f1237bc26b8851c42c7ba7c185cce2",
    ),
    "raw/v0-eager.json": (
        12225,
        "8e6e7e4edd500ca98010c7bae1b6265d270d9c952fabcb1cd85cdcf80efebcda",
    ),
    "raw/v0-graph.json": (
        12227,
        "24f5cfcaf667a61913a4018e3d7a9de48b9c38d5da03f0805f3a520eeb6c8a61",
    ),
    "raw/v2-eager.json": (
        47875,
        "f8d617ef25eb6389e4c9912efe1b893ea68abc883efa49afe54e51bfda25dba9",
    ),
    "raw/v2-graph.json": (
        48296,
        "462f14488147a7dae513043190401eb3ae4a36f2881169d06a2667d84d14d9d7",
    ),
}
TRUSTED_README = (
    6114,
    "497073472383ef017c504174aa8814516d7eabd104eb2ef5a7e9f9dbf3a1d4a2",
)
TRUSTED_MANIFEST = (
    10592,
    "2d561fe74038300f70e0f499e92bd04bde5567e3589e21e0567f134b7bce36e9",
)

REGISTERED_OUTPUTS = [
    ["$%$%", [3, 4, 3, 4]],
    [" ( ) = ", [320, 873, 284, 220]],
]
REGISTERED_SCHEDULER_TRACE = [
    [True, [[4, 0, 4, True], [3, 0, 3, True]]],
    [False, [[5, 4, 1, False], [4, 3, 1, False]]],
    [False, [[6, 5, 1, False], [5, 4, 1, False]]],
    [False, [[7, 6, 1, False], [6, 5, 1, False]]],
]
REGISTERED_TOKENIZER_MANIFEST_SHA256 = (
    "ae22ab112507b607364512e7b5aa1b912abe0cd730c5af8026a86ae14ac8976d"
)
REGISTERED_SPECULATIVE_TOKENIZER_FINGERPRINTS = {
    REGISTERED_TOKENIZER_MANIFEST_SHA256: (
        "54080f7622052c7ea95e2993366bd9ff624be5a1e9e894c49a08b04f4ed56fcd"
    ),
}
EXPECTED_RECOVERY_IMPORT_ORIGINS = {
    "lifecycle_contract": (
        "/workspace/nano-vllm-spec-v2/tests/run_speculative_v2_lifecycle.py"
    ),
    "nanovllm_package": "/workspace/nano-vllm-spec-v2/nanovllm/__init__.py",
    "model_runner_class": (
        "/workspace/nano-vllm-spec-v2/nanovllm/engine/model_runner.py"
    ),
    "llm_class": "/workspace/nano-vllm-spec-v2/nanovllm/llm.py",
}

IGNORED_ARCHIVE_FILES = {"README.md", "manifest.json"}
HEX_40 = re.compile(r"[0-9a-f]{40}")
HEX_64 = re.compile(r"[0-9a-f]{64}")


class EvidenceValidationError(ValueError):
    """A retained artifact or its manifest violates the registered contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceValidationError(message)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise EvidenceValidationError(f"non-finite JSON number: {value}")


def _validate_finite_tree(value: Any, *, path: str = "$") -> None:
    if isinstance(value, float):
        _require(math.isfinite(value), f"non-finite number at {path}")
    elif isinstance(value, dict):
        for key, child in value.items():
            _validate_finite_tree(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_finite_tree(child, path=f"{path}[{index}]")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_bytes(),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except EvidenceValidationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceValidationError(f"invalid JSON: {path}") from error
    _require(isinstance(payload, dict), f"JSON root must be an object: {path}")
    _validate_finite_tree(payload)
    return payload


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    """Hash sorted compact UTF-8 JSON with no trailing newline.

    The model-content digest uses exactly this function over the four-key
    dictionary returned by :func:`_model_content_identity`; notably it excludes
    the path spelling supplied as the original model ``argument``.
    """

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> list[Any]:
    _require(isinstance(value, list), f"{label} must be a list")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    _require(type(value) is int, f"{label} must be an integer")
    _require(value >= minimum, f"{label} must be >= {minimum}")
    return value


def _reject_claim_critical_aliases(document: dict[str, Any], *, label: str) -> None:
    for key in ("gpu_certified", "draft_proposals_executed"):
        _require(
            key not in document,
            f"{label} contains unknown claim-critical top-level field: {key}",
        )


def _validate_rng_snapshot(value: Any, *, label: str) -> dict[str, str]:
    snapshot = _mapping(value, label)
    _require(
        set(snapshot) == {"cpu_sha256", "cuda_sha256"},
        f"{label} has the wrong RNG digest fields",
    )
    cpu = _hex_digest(snapshot.get("cpu_sha256"), f"{label} CPU SHA256")
    cuda = _hex_digest(snapshot.get("cuda_sha256"), f"{label} CUDA SHA256")
    return {"cpu_sha256": cpu, "cuda_sha256": cuda}


def _validate_protocol_types(protocol_value: Any) -> dict[str, Any]:
    protocol = _mapping(protocol_value, "manifest protocol")
    for key in (
        "seed",
        "configured_k",
        "tensor_parallel_size",
        "max_model_len",
        "max_num_batched_tokens",
        "max_num_seqs",
        "kvcache_block_size",
        "recovery_num_kvcache_blocks",
    ):
        _integer(protocol.get(key), f"manifest protocol {key}", minimum=1)
    _require(type(protocol.get("gpu_memory_utilization")) is float,
             "manifest gpu_memory_utilization must be a float")
    _require(type(protocol.get("top_p_backend")) is str,
             "manifest top_p_backend must be a string")
    workload = _mapping(protocol.get("workload"), "manifest workload")
    prompts = _sequence(workload.get("prompt_token_ids"), "manifest prompts")
    _require(
        bool(prompts)
        and all(
            isinstance(prompt, list)
            and bool(prompt)
            and all(type(token) is int and token >= 0 for token in prompt)
            for prompt in prompts
        ),
        "manifest prompt token IDs must be nonempty integer lists",
    )
    sampling = _mapping(workload.get("sampling"), "manifest sampling")
    _require(type(sampling.get("temperature")) is float,
             "manifest sampling temperature must be a float")
    _integer(sampling.get("top_k"), "manifest sampling top_k", minimum=-1)
    _require(type(sampling.get("top_p")) is float,
             "manifest sampling top_p must be a float")
    _integer(sampling.get("max_tokens"), "manifest sampling max_tokens", minimum=1)
    _require(type(sampling.get("ignore_eos")) is bool,
             "manifest sampling ignore_eos must be a boolean")
    phase_modes = _mapping(
        protocol.get("recovery_phase_modes"),
        "manifest recovery phase modes",
    )
    _require(all(type(value) is str for value in phase_modes.values()),
             "manifest recovery modes must be strings")
    phase_calls = _mapping(
        protocol.get("recovery_expected_real_phase_calls"),
        "manifest recovery phase calls",
    )
    for phase, calls in phase_calls.items():
        _integer(calls, f"manifest recovery calls for {phase}", minimum=1)
    return protocol


def _hex_digest(value: Any, label: str, *, length: int = 64) -> str:
    pattern = HEX_64 if length == 64 else HEX_40
    _require(isinstance(value, str) and pattern.fullmatch(value) is not None,
             f"{label} must be a lowercase {length}-character hex digest")
    return value


def _resolve_payload(root: Path, relative: Any) -> Path:
    _require(isinstance(relative, str) and relative, "artifact path is invalid")
    pure = PurePosixPath(relative)
    _require(
        not pure.is_absolute() and ".." not in pure.parts,
        f"artifact path is not archive-relative: {relative}",
    )
    candidate = root
    try:
        for index, part in enumerate(pure.parts):
            candidate = candidate / part
            info = candidate.lstat()
            _require(
                not stat.S_ISLNK(info.st_mode),
                f"artifact path contains a symlink: {relative}",
            )
            if index < len(pure.parts) - 1:
                _require(
                    stat.S_ISDIR(info.st_mode),
                    f"artifact ancestor is not a directory: {relative}",
                )
        path = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise EvidenceValidationError(f"missing artifact: {relative}") from error
    _require(path.is_relative_to(root), f"artifact escapes archive root: {relative}")
    info = path.lstat()
    _require(stat.S_ISREG(info.st_mode), f"artifact is not regular: {relative}")
    _require(info.st_nlink == 1, f"artifact must not be hard-linked: {relative}")
    return path


def _model_content_identity(snapshot: Any) -> dict[str, Any]:
    snapshot = _mapping(snapshot, "model snapshot")
    required = (
        "resolved_path",
        "metadata_files",
        "weight_files",
        "total_weight_bytes",
    )
    for key in required:
        _require(key in snapshot, f"model snapshot is missing {key}")
    metadata = _mapping(snapshot["metadata_files"], "model metadata_files")
    _require("config.json" in metadata, "model manifest is missing config.json")
    for name, row in metadata.items():
        _require(isinstance(name, str) and name, "model metadata name is invalid")
        row = _mapping(row, f"model metadata {name}")
        _integer(row.get("size_bytes"), f"model metadata {name} size")
        _hex_digest(row.get("sha256"), f"model metadata {name} SHA256")
    weights = _sequence(snapshot["weight_files"], "model weight_files")
    _require(bool(weights), "model manifest has no safetensors weights")
    names: list[str] = []
    total = 0
    for row in weights:
        row = _mapping(row, "model weight row")
        name = row.get("name")
        _require(
            isinstance(name, str) and name.endswith(".safetensors"),
            "model weight name must identify a safetensors file",
        )
        names.append(name)
        total += _integer(row.get("size_bytes"), f"model weight {name} size", minimum=1)
        _hex_digest(row.get("sha256"), f"model weight {name} SHA256")
    _require(names == sorted(set(names)), "model weight inventory is not unique/sorted")
    _require(
        snapshot["total_weight_bytes"] == total,
        "model total_weight_bytes does not equal the weight inventory",
    )
    _require(
        isinstance(snapshot["resolved_path"], str)
        and PurePosixPath(snapshot["resolved_path"]).is_absolute(),
        "model resolved_path must be absolute",
    )
    return {key: snapshot[key] for key in required}


def _validate_tokenizer(snapshot: Any) -> dict[str, Any]:
    snapshot = _mapping(snapshot, "tokenizer snapshot")
    for key in ("length", "vocab_size", "vocab_entries"):
        _integer(snapshot.get(key), f"tokenizer {key}", minimum=1)
    _require(snapshot["length"] >= snapshot["vocab_entries"],
             "tokenizer length is smaller than its vocabulary")
    _hex_digest(snapshot.get("vocab_sha256"), "tokenizer vocab SHA256")
    backend_hash = snapshot.get("backend_json_sha256")
    backend_bytes = snapshot.get("backend_json_bytes")
    _require(
        (backend_hash is None and backend_bytes is None)
        or (
            isinstance(backend_bytes, int)
            and backend_bytes > 0
            and isinstance(backend_hash, str)
            and HEX_64.fullmatch(backend_hash) is not None
        ),
        "tokenizer backend JSON hash/size pair is invalid",
    )
    _require(isinstance(snapshot.get("class"), str), "tokenizer class is missing")
    _sequence(snapshot.get("all_special_ids"), "tokenizer all_special_ids")
    return snapshot


def _validate_software(
    software_value: Any,
    *,
    label: str,
    python_value: Any,
) -> dict[str, Any]:
    software = _mapping(software_value, f"{label} software")
    for key in ("torch", "torch_cuda_build", "cudnn", "nccl", "transformers"):
        _require(
            software.get(key) == EXPECTED_REGISTERED_SOFTWARE[key],
            f"{label} software mismatch: {key}",
        )
    python = _mapping(python_value, f"{label} Python")
    _require(
        str(python.get("version", "")).startswith(EXPECTED_REGISTERED_SOFTWARE["python"]),
        f"{label} Python version mismatch",
    )
    _require(
        python.get("implementation") == EXPECTED_REGISTERED_SOFTWARE["python_implementation"],
        f"{label} Python implementation mismatch",
    )
    _require(python.get("optimize") == 0, f"{label} used optimized Python")
    return software


def _validate_git_snapshot(
    snapshot: Any,
    *,
    label: str,
    commit: str,
    tree: str,
    source_tree_sha256: str | None = None,
    detached: bool | None = None,
) -> dict[str, Any]:
    snapshot = _mapping(snapshot, label)
    _require(snapshot.get("head") == commit, f"{label} commit mismatch")
    _require(snapshot.get("tree") == tree, f"{label} tree mismatch")
    _require(snapshot.get("dirty") is False, f"{label} is dirty")
    _require(snapshot.get("status_porcelain_v1") == [], f"{label} status is not empty")
    if source_tree_sha256 is not None:
        _require(
            snapshot.get("source_tree_sha256") == source_tree_sha256,
            f"{label} source-tree hash mismatch",
        )
    if detached is not None:
        _require(snapshot.get("detached") is detached, f"{label} detached state mismatch")
    _require(isinstance(snapshot.get("repository"), str), f"{label} repository is missing")
    return snapshot


def _validate_unchanged_pair(
    before: Any,
    after: Any,
    *,
    label: str,
    **snapshot_kwargs: Any,
) -> dict[str, Any]:
    before = _validate_git_snapshot(before, label=f"{label} before", **snapshot_kwargs)
    after = _validate_git_snapshot(after, label=f"{label} after", **snapshot_kwargs)
    _require(before == after, f"{label} changed during the run")
    return before


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _round_up(value: int, alignment: int) -> int:
    return _ceil_div(value, alignment) * alignment


def _top_k_workspace(rows: int, vocab: int, itemsize: int) -> int:
    dense = rows * vocab
    return dense * itemsize + dense * itemsize + dense * INT64_BYTES + rows * itemsize


def _top_p_workspace(rows: int, vocab: int, itemsize: int) -> int:
    dense = rows * vocab
    chunk = min(rows, TOP_P_CHUNK_SIZE) * vocab
    return (
        dense * itemsize
        + chunk * (4 * FP32_BYTES + INT64_BYTES + 2 * BOOL_BYTES)
        + chunk * (FP32_BYTES + INT64_BYTES)
    )


def expected_workspace_plan(protocol: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the independently recomputed registered workspace plan."""

    p = EXPECTED_PROTOCOL if protocol is None else protocol
    vocab = p["vocab_size"]
    configured_k = p["configured_k"]
    max_effective_k = min(
        configured_k,
        p["max_num_batched_tokens"] - 1,
        p["max_model_len"] - 1,
    )
    batch = min(
        p["max_num_seqs"],
        p["max_num_batched_tokens"] // (max_effective_k + 1),
    )
    draft_rows = batch
    verifier_rows = batch * (max_effective_k + 1)
    target_itemsize = p["target_logits_itemsize"]
    draft_itemsize = p["draft_logits_itemsize"]

    draft_probability = batch * max_effective_k * vocab * FP32_BYTES
    target_probability = verifier_rows * vocab * FP32_BYTES
    probability_floor = draft_probability + target_probability
    draft_logits = draft_rows * vocab * draft_itemsize
    verifier_logits = verifier_rows * vocab * target_itemsize
    draft_transform = draft_rows * vocab * (draft_itemsize + FP32_BYTES)
    verifier_transform = verifier_rows * vocab * (target_itemsize + FP32_BYTES)
    draft_top_k = _top_k_workspace(draft_rows, vocab, draft_itemsize)
    verifier_top_k = _top_k_workspace(verifier_rows, vocab, target_itemsize)
    draft_top_p = _top_p_workspace(draft_rows, vocab, draft_itemsize)
    verifier_top_p = _top_p_workspace(verifier_rows, vocab, target_itemsize)
    draft_sampling = draft_rows * vocab * (FP32_BYTES + 3 * FP64_BYTES)
    bonus_sampling = batch * vocab * (FP32_BYTES + 3 * FP64_BYTES)
    rejection = batch * vocab * (3 * FP32_BYTES + 7 * FP64_BYTES)
    proposal_elements = batch * configured_k
    proposal_metadata = proposal_elements * (
        INT64_BYTES
        + 2 * FP64_BYTES
        + 2 * FP32_BYTES
        + 2 * FP64_BYTES
        + FP64_BYTES
        + FP32_BYTES
        + 2 * BOOL_BYTES
        + 2 * INT64_BYTES
    )
    sampler_metadata = (batch + verifier_rows) * (2 * FP32_BYTES + 2 * INT64_BYTES)
    per_sequence = batch * (4 * INT64_BYTES + 3 * BOOL_BYTES)
    metadata = proposal_metadata + sampler_metadata + per_sequence

    draft_filter = (
        draft_probability
        + draft_logits
        + draft_logits
        + max(draft_top_k, draft_top_p)
        + metadata
    )
    draft_softmax = (
        draft_probability
        + draft_logits
        + draft_logits
        + batch * vocab * FP32_BYTES
        + metadata
    )
    draft_race = draft_probability + draft_logits + draft_sampling + metadata
    draft_phase = max(draft_filter, draft_softmax, draft_race)
    verifier_filter = (
        draft_probability
        + verifier_logits
        + verifier_logits
        + max(verifier_top_k, verifier_top_p)
        + metadata
    )
    verifier_softmax = (
        draft_probability
        + verifier_logits
        + verifier_logits
        + target_probability
        + target_probability
        + metadata
    )
    verifier_phase = max(verifier_filter, verifier_softmax)
    rejection_phase = probability_floor + rejection + metadata
    bonus_phase = probability_floor + bonus_sampling + metadata
    live_peak = max(draft_phase, verifier_phase, rejection_phase, bonus_phase)
    fractional_margin = _ceil_div(live_peak, 10)
    allocator_margin = _round_up(
        max(ALLOCATOR_MARGIN_MIN_BYTES, fractional_margin),
        ALLOCATOR_MARGIN_ALIGNMENT_BYTES,
    )
    return {
        "batch_size": batch,
        "configured_k": configured_k,
        "max_effective_k": max_effective_k,
        "vocab_size": vocab,
        "draft_rows": draft_rows,
        "verifier_rows": verifier_rows,
        "target_logits_itemsize": target_itemsize,
        "draft_logits_itemsize": draft_itemsize,
        "draft_probability_bytes": draft_probability,
        "target_probability_bytes": target_probability,
        "probability_floor_bytes": probability_floor,
        "draft_logits_bytes": draft_logits,
        "verifier_logits_bytes": verifier_logits,
        "draft_transform_bytes": draft_transform,
        "verifier_transform_bytes": verifier_transform,
        "draft_top_k_workspace_bytes": draft_top_k,
        "verifier_top_k_workspace_bytes": verifier_top_k,
        "draft_top_p_workspace_bytes": draft_top_p,
        "verifier_top_p_workspace_bytes": verifier_top_p,
        "draft_sampling_workspace_bytes": draft_sampling,
        "bonus_sampling_workspace_bytes": bonus_sampling,
        "rejection_correction_workspace_bytes": rejection,
        "metadata_bytes": metadata,
        "draft_filter_phase_bytes": draft_filter,
        "draft_softmax_phase_bytes": draft_softmax,
        "draft_race_phase_bytes": draft_race,
        "verifier_filter_phase_bytes": verifier_filter,
        "verifier_softmax_phase_bytes": verifier_softmax,
        "graph_static_workspace_bytes": None,
        "backend_library_workspace_bytes": None,
        "audit_required_components": AUDIT_REQUIRED_COMPONENTS,
        "draft_phase_bytes": draft_phase,
        "verifier_phase_bytes": verifier_phase,
        "rejection_phase_bytes": rejection_phase,
        "bonus_phase_bytes": bonus_phase,
        "modeled_live_peak_bytes": live_peak,
        "allocator_margin_bytes": allocator_margin,
        "reservation_bytes": live_peak + allocator_margin,
    }


def independent_memory_reconciliation(audit: dict[str, Any], *, mode: str) -> dict[str, int]:
    plan = _mapping(audit.get("workspace_plan"), "workspace plan")
    graph_ownership = max(
        audit["profiled_graph_allocated_bytes"],
        audit["profiled_graph_reserved_bytes"],
    )
    graph_peak = max(
        graph_ownership,
        audit["profiled_graph_peak_allocated_bytes"],
        audit["profiled_graph_peak_reserved_bytes"],
    )
    graph_margin = 0 if mode == "eager" else max(
        ALLOCATOR_MARGIN_MIN_BYTES,
        audit["joint_block_bytes"],
    )
    graph_construction = graph_peak + graph_margin
    runtime_reservation = (
        graph_ownership
        + audit["warmup_transient_bytes"]
        + plan["reservation_bytes"]
    )
    sizing_overhead = max(graph_construction, runtime_reservation)
    sizing_usable = (
        audit["memory_budget_bytes"]
        - audit["used_before_kv_bytes"]
        - sizing_overhead
    )
    automatic_blocks = sizing_usable // audit["joint_block_bytes"]
    modeled_headroom = (
        audit["post_init_budget_headroom_bytes"]
        - audit["warmup_transient_bytes"]
        - plan["reservation_bytes"]
    )
    kv_increment = (
        audit["allocated_after_kv_before_graph_bytes"]
        - audit["allocated_before_kv_bytes"]
    )
    kv_accounted = audit["target_kv_bytes"] + audit["draft_kv_bytes"]
    final_allocated = max(
        audit["allocated_after_graph_before_pretouch_bytes"]
        - audit["allocated_after_kv_before_graph_bytes"],
        0,
    )
    final_reserved = max(
        audit["reserved_after_graph_before_pretouch_bytes"]
        - audit["reserved_after_kv_before_graph_bytes"],
        0,
    )
    post_allocated = max(
        audit["post_init_allocated_bytes"]
        - audit["allocated_after_graph_before_pretouch_bytes"],
        0,
    )
    post_reserved = max(
        audit["post_init_reserved_bytes"]
        - audit["reserved_after_graph_before_pretouch_bytes"],
        0,
    )
    total_allocated = audit["post_init_allocated_bytes"] - audit["allocated_before_kv_bytes"]
    total_accounted = kv_accounted + audit["final_graph_allocated_bytes"] + post_allocated
    return {
        "profiled_graph_ownership_bytes": graph_ownership,
        "profiled_graph_peak_bytes": graph_peak,
        "graph_allocator_margin_bytes": graph_margin,
        "graph_construction_reservation_bytes": graph_construction,
        "runtime_reservation_bytes": runtime_reservation,
        "sizing_overhead_bytes": sizing_overhead,
        "sizing_usable_bytes": sizing_usable,
        "automatic_num_blocks": automatic_blocks,
        "modeled_runtime_headroom_bytes": modeled_headroom,
        "kv_allocated_increment_bytes": kv_increment,
        "kv_accounted_bytes": kv_accounted,
        "final_graph_allocated_increment_bytes": final_allocated,
        "final_graph_reserved_increment_bytes": final_reserved,
        "post_init_allocated_increment_bytes": post_allocated,
        "post_init_reserved_increment_bytes": post_reserved,
        "total_allocated_increment_bytes": total_allocated,
        "total_accounted_increment_bytes": total_accounted,
    }


def _validate_memory_audit(
    audit_value: Any,
    reconciliation_value: Any,
    *,
    label: str,
    mode: str,
    protocol: dict[str, Any],
    hardware: dict[str, Any],
    automatic: bool,
    expected_blocks: int | None = None,
) -> tuple[dict[str, Any], dict[str, int]]:
    audit = _mapping(audit_value, f"{label} audit")
    reconciliation = _mapping(reconciliation_value, f"{label} reconciliation")
    expected_plan = expected_workspace_plan(protocol)
    _require(
        audit.get("workspace_plan") == expected_plan,
        f"{label} workspace planner arithmetic mismatch",
    )
    _require(audit.get("gpu_certified") is False, f"{label} promoted gpu_certified")
    _require(
        audit.get("audit_required_components") == AUDIT_REQUIRED_COMPONENTS,
        f"{label} omitted audit-required components",
    )
    required_ints = (
        "selected_num_blocks", "target_block_bytes", "draft_block_bytes",
        "joint_block_bytes", "target_kv_bytes", "draft_kv_bytes",
        "target_weight_bytes", "draft_weight_bytes", "total_memory_bytes",
        "memory_budget_bytes", "free_before_kv_bytes", "used_before_kv_bytes",
        "allocated_before_kv_bytes", "reserved_before_kv_bytes",
        "peak_before_kv_bytes", "target_warmup_transient_bytes",
        "draft_warmup_transient_bytes", "warmup_transient_bytes",
        "profiled_graph_allocated_bytes", "profiled_graph_reserved_bytes",
        "profiled_graph_peak_allocated_bytes", "profiled_graph_peak_reserved_bytes",
        "profiled_graph_ownership_bytes", "profiled_graph_peak_bytes",
        "graph_allocator_margin_bytes", "graph_construction_reservation_bytes",
        "graph_reservation_bytes", "runtime_reservation_bytes",
        "sizing_overhead_bytes", "allocated_after_kv_before_graph_bytes",
        "reserved_after_kv_before_graph_bytes",
        "allocated_after_graph_before_pretouch_bytes",
        "reserved_after_graph_before_pretouch_bytes", "final_graph_allocated_bytes",
        "final_graph_reserved_bytes", "final_graph_peak_allocated_bytes",
        "final_graph_peak_reserved_bytes", "post_init_allocated_bytes",
        "post_init_reserved_bytes", "post_init_free_bytes",
        "post_init_budget_headroom_bytes", "modeled_runtime_headroom_bytes",
    )
    for key in required_ints:
        _integer(audit.get(key), f"{label}.{key}")
    selected = audit["selected_num_blocks"]
    _require(selected > 0, f"{label} selected no KV blocks")
    _require(
        audit["joint_block_bytes"]
        == audit["target_block_bytes"] + audit["draft_block_bytes"],
        f"{label} joint block bytes mismatch",
    )
    _require(
        audit["target_block_bytes"] == REGISTERED_MODEL_BLOCK_BYTES
        and audit["draft_block_bytes"] == REGISTERED_MODEL_BLOCK_BYTES
        and audit["joint_block_bytes"] == REGISTERED_JOINT_BLOCK_BYTES,
        f"{label} KV block geometry differs from the registered model pair",
    )
    _require(
        audit["target_weight_bytes"] == REGISTERED_MODEL_WEIGHT_BYTES
        and audit["draft_weight_bytes"] == REGISTERED_MODEL_WEIGHT_BYTES,
        f"{label} physical model-weight bytes mismatch",
    )
    _require(
        audit["target_kv_bytes"] == selected * audit["target_block_bytes"]
        and audit["draft_kv_bytes"] == selected * audit["draft_block_bytes"],
        f"{label} KV bytes do not match selected blocks",
    )
    _require(
        audit["total_memory_bytes"] == hardware["total_memory_bytes"],
        f"{label} GPU memory identity mismatch",
    )
    _require(
        audit["memory_budget_bytes"]
        == int(audit["total_memory_bytes"] * protocol["gpu_memory_utilization"]),
        f"{label} memory budget mismatch",
    )
    _require(
        audit["used_before_kv_bytes"]
        == audit["total_memory_bytes"] - audit["free_before_kv_bytes"],
        f"{label} free/used/total mismatch",
    )
    _require(
        audit["allocated_before_kv_bytes"] <= audit["reserved_before_kv_bytes"],
        f"{label} allocated exceeds reserved before KV",
    )
    _require(
        audit["allocated_before_kv_bytes"]
        >= audit["target_weight_bytes"] + audit["draft_weight_bytes"],
        f"{label} pre-KV allocation is below physical weights",
    )
    _require(
        audit["peak_before_kv_bytes"] >= audit["allocated_before_kv_bytes"],
        f"{label} pre-KV peak is too small",
    )
    _require(
        audit["warmup_transient_bytes"] >= max(
            audit["target_warmup_transient_bytes"],
            audit["draft_warmup_transient_bytes"],
        ),
        f"{label} warmup transient omits a model phase",
    )
    computed = independent_memory_reconciliation(audit, mode=mode)
    _require(reconciliation == computed, f"{label} independent reconciliation mismatch")
    for audit_key, reconciliation_key in (
        ("profiled_graph_ownership_bytes", "profiled_graph_ownership_bytes"),
        ("profiled_graph_peak_bytes", "profiled_graph_peak_bytes"),
        ("graph_allocator_margin_bytes", "graph_allocator_margin_bytes"),
        ("graph_construction_reservation_bytes", "graph_construction_reservation_bytes"),
        ("runtime_reservation_bytes", "runtime_reservation_bytes"),
        ("sizing_overhead_bytes", "sizing_overhead_bytes"),
        ("modeled_runtime_headroom_bytes", "modeled_runtime_headroom_bytes"),
    ):
        _require(
            audit[audit_key] == computed[reconciliation_key],
            f"{label} {audit_key} mismatch",
        )
    _require(
        audit["graph_reservation_bytes"] == audit["graph_construction_reservation_bytes"],
        f"{label} graph reservation alias mismatch",
    )
    _require(computed["sizing_usable_bytes"] >= 0, f"{label} has negative sizing capacity")
    _require(computed["modeled_runtime_headroom_bytes"] >= 0,
             f"{label} has negative modeled runtime headroom")
    _require(selected <= computed["automatic_num_blocks"],
             f"{label} selected blocks exceed capacity")
    if automatic:
        _require(selected == computed["automatic_num_blocks"],
                 f"{label} automatic block selection is not the capacity floor")
    if expected_blocks is not None:
        _require(selected == expected_blocks, f"{label} selected block count mismatch")
    _require(
        computed["kv_allocated_increment_bytes"] == computed["kv_accounted_bytes"],
        f"{label} KV allocation increment mismatch",
    )
    _require(
        computed["final_graph_allocated_increment_bytes"]
        == audit["final_graph_allocated_bytes"]
        and computed["final_graph_reserved_increment_bytes"]
        == audit["final_graph_reserved_bytes"],
        f"{label} final graph ownership mismatch",
    )
    _require(
        computed["total_allocated_increment_bytes"]
        == computed["total_accounted_increment_bytes"],
        f"{label} total allocation accounting mismatch",
    )
    _require(
        audit["post_init_allocated_bytes"] <= audit["post_init_reserved_bytes"],
        f"{label} post-init allocated exceeds reserved",
    )
    _require(
        audit["post_init_free_bytes"] <= audit["total_memory_bytes"],
        f"{label} post-init free memory exceeds total device memory",
    )
    _require(
        audit["post_init_budget_headroom_bytes"]
        == audit["memory_budget_bytes"]
        - (audit["total_memory_bytes"] - audit["post_init_free_bytes"]),
        f"{label} post-init budget headroom mismatch",
    )
    _require(
        audit["final_graph_peak_allocated_bytes"] >= audit["final_graph_allocated_bytes"]
        and audit["final_graph_peak_reserved_bytes"] >= audit["final_graph_reserved_bytes"],
        f"{label} graph peak is below owned graph memory",
    )
    _require(
        max(audit["final_graph_peak_allocated_bytes"], audit["final_graph_peak_reserved_bytes"])
        <= audit["graph_construction_reservation_bytes"],
        f"{label} final graph peak exceeds the construction envelope",
    )
    graph_fields = (
        "profiled_graph_allocated_bytes", "profiled_graph_reserved_bytes",
        "profiled_graph_peak_allocated_bytes", "profiled_graph_peak_reserved_bytes",
        "profiled_graph_ownership_bytes", "profiled_graph_peak_bytes",
        "final_graph_allocated_bytes", "final_graph_reserved_bytes",
        "final_graph_peak_allocated_bytes", "final_graph_peak_reserved_bytes",
    )
    if mode == "eager":
        _require(all(audit[key] == 0 for key in graph_fields),
                 f"{label} eager audit owns graph memory")
        _require(
            audit["allocated_after_kv_before_graph_bytes"]
            == audit["allocated_after_graph_before_pretouch_bytes"]
            and audit["reserved_after_kv_before_graph_bytes"]
            == audit["reserved_after_graph_before_pretouch_bytes"],
            f"{label} eager graph baselines changed",
        )
    else:
        _require(
            audit["profiled_graph_allocated_bytes"] > 0
            and audit["final_graph_allocated_bytes"] > 0,
            f"{label} graph audit records no graph ownership",
        )
    return audit, computed


def _process_memory(snapshot: dict[str, Any]) -> dict[int, int]:
    rows = _sequence(snapshot.get("compute_apps"), "GPU compute applications")
    result: dict[int, int] = {}
    for row in rows:
        row = _mapping(row, "GPU compute application")
        pid = _integer(row.get("pid"), "GPU process PID", minimum=1)
        memory = _integer(row.get("used_memory_mib"), "GPU process memory")
        _require(pid not in result, f"duplicate GPU process PID {pid}")
        result[pid] = memory
    return result


def _validate_gpu_snapshot(
    snapshot_value: Any,
    *,
    label: str,
    hardware: dict[str, Any],
) -> dict[str, Any]:
    snapshot = _mapping(snapshot_value, label)
    _require(type(snapshot.get("index")) is int and snapshot["index"] == 0,
             f"{label} GPU index must be integer device 0")
    _require(snapshot.get("name") == GPU_NAME, f"{label} GPU name mismatch")
    _require(snapshot.get("compute_capability") == GPU_COMPUTE_CAPABILITY,
             f"{label} compute capability mismatch")
    _require(snapshot.get("total_memory_bytes") == hardware["total_memory_bytes"],
             f"{label} total memory mismatch")
    _require(snapshot.get("multiprocessor_count") == hardware["multiprocessor_count"],
             f"{label} multiprocessor count mismatch")
    _require(snapshot.get("uuid") == hardware["uuid"], f"{label} UUID mismatch")
    query_id = snapshot.get("nvidia_smi_query_id")
    _require(query_id == f"GPU-{hardware['uuid']}", f"{label} query UUID mismatch")
    _require(not str(query_id).startswith("MIG-"), f"{label} unexpectedly used MIG")
    _require(snapshot.get("nvidia_smi_returncode") == 0, f"{label} nvidia-smi query failed")
    _require(snapshot.get("compute_apps_returncode") == 0,
             f"{label} compute-app query failed")
    _require(snapshot.get("nvidia_smi_stderr") == "", f"{label} nvidia-smi stderr is nonempty")
    _require(snapshot.get("compute_apps_stderr") == "", f"{label} compute-app stderr is nonempty")
    _require(snapshot.get("foreign_compute_apps") == [],
             f"{label} contains foreign GPU consumers")
    rows = _sequence(snapshot.get("nvidia_smi_parsed_rows"), f"{label} GPU identity rows")
    _require(len(rows) == 1, f"{label} must contain one GPU identity row")
    row = _mapping(rows[0], f"{label} GPU identity row")
    _require(row.get("name") == GPU_NAME, f"{label} parsed GPU name mismatch")
    _require(row.get("gpu_uuid") == query_id, f"{label} parsed GPU UUID mismatch")
    _require(row.get("driver_version") == hardware["driver_version"],
             f"{label} driver mismatch")
    own_rows = _sequence(snapshot.get("own_gpu_process_ids"), f"{label} owned PIDs")
    _require(all(type(pid) is int and pid > 0 for pid in own_rows),
             f"{label} owned PID list is invalid")
    own = set(own_rows)
    for app in _sequence(snapshot.get("compute_apps"), f"{label} compute apps"):
        app = _mapping(app, f"{label} compute app")
        _require(app.get("gpu_uuid") == query_id, f"{label} compute app GPU mismatch")
        _require(app.get("pid") in own, f"{label} contains an unowned process")
    return snapshot


def _validate_allocator_challenge(before: dict[str, Any], *, hardware: dict[str, Any]) -> None:
    context = _mapping(before.get("self_context_establishment"), "GPU ownership binding")
    _require(context.get("pre_context_compute_apps") == [],
             "GPU context was not established from an empty endpoint")
    candidate = _integer(context.get("bound_host_namespace_pid"), "bound host PID", minimum=1)
    post_sentinel = _sequence(context.get("post_sentinel_compute_apps"), "post-context apps")
    _require(len(post_sentinel) == 1 and post_sentinel[0].get("pid") == candidate,
             "GPU context did not introduce exactly the bound host PID")
    aliases = _sequence(context.get("visible_pid_namespace_aliases"), "PID namespace aliases")
    _require(all(type(pid) is int and pid > 0 for pid in aliases),
             "PID namespace aliases are invalid")
    _require(candidate in before.get("own_gpu_process_ids", []),
             "bound host PID is not the owned endpoint process")
    challenge = _mapping(context.get("allocator_challenge"), "allocator challenge")
    _require(challenge.get("candidate_pid") == candidate,
             "allocator challenge candidate differs from bound PID")
    _require(challenge.get("requested_bytes") == GPU_PID_CHALLENGE_BYTES,
             "allocator challenge size mismatch")
    _require(challenge.get("rounding_tolerance_mib") == GPU_PID_ROUNDING_TOLERANCE_MIB,
             "allocator challenge rounding tolerance mismatch")
    for key in (
        "baseline_allocated_bytes", "baseline_reserved_bytes",
        "allocated_delta_bytes", "reserved_delta_bytes",
        "released_allocated_bytes", "released_reserved_bytes",
        "candidate_nvidia_smi_delta_mib",
    ):
        _integer(challenge.get(key), f"allocator challenge {key}")
    _require(challenge["allocated_delta_bytes"] >= GPU_PID_CHALLENGE_BYTES,
             "allocator challenge allocated delta is too small")
    _require(challenge["reserved_delta_bytes"] >= challenge["allocated_delta_bytes"],
             "allocator challenge reserved delta is too small")
    registered = _mapping(
        hardware.get("allocator_challenge"),
        "registered allocator challenge",
    )
    _require(
        challenge["allocated_delta_bytes"]
        == challenge["reserved_delta_bytes"]
        == registered["allocated_and_reserved_delta_bytes"],
        "allocator CUDA delta differs from the registered challenge",
    )
    _require(
        challenge.get("reserved_delta_mib") == registered["reserved_delta_mib"]
        and challenge["candidate_nvidia_smi_delta_mib"]
        == registered["nvidia_smi_delta_mib"],
        "allocator MiB deltas differ from the registered challenge",
    )
    baseline = _validate_gpu_snapshot(
        challenge.get("baseline_snapshot"), label="allocator baseline", hardware=hardware
    )
    challenged = _validate_gpu_snapshot(
        challenge.get("challenged_snapshot"), label="allocator challenged", hardware=hardware
    )
    released = _validate_gpu_snapshot(
        challenge.get("released_snapshot"), label="allocator released", hardware=hardware
    )
    baseline_memory = _process_memory(baseline)
    challenged_memory = _process_memory(challenged)
    released_memory = _process_memory(released)
    _require(set(baseline_memory) == set(challenged_memory) == set(released_memory),
             "GPU PID set changed during allocator challenge")
    _require(candidate in baseline_memory, "allocator candidate disappeared")
    _require(set(before.get("own_gpu_process_ids", [])) == {candidate},
             "endpoint ownership is not exactly the challenged PID")
    for snapshot in (baseline, challenged, released):
        _require(set(snapshot.get("own_gpu_process_ids", [])) == {candidate},
                 "allocator snapshot ownership is not exactly the challenged PID")
    observed_delta = challenged_memory[candidate] - baseline_memory[candidate]
    _require(observed_delta == challenge["candidate_nvidia_smi_delta_mib"],
             "allocator candidate NVML delta mismatch")
    expected_delta = challenge["reserved_delta_bytes"] / 1024**2
    _require(abs(observed_delta - expected_delta) <= GPU_PID_ROUNDING_TOLERANCE_MIB,
             "allocator challenge did not track the candidate PID")
    for pid, memory in baseline_memory.items():
        if pid != candidate:
            _require(challenged_memory[pid] == memory,
                     "non-candidate GPU memory changed during binding")
    _require(released_memory == baseline_memory,
             "GPU process memory did not return to the binding baseline")
    _require(
        challenge["released_allocated_bytes"] == challenge["baseline_allocated_bytes"]
        and challenge["released_reserved_bytes"] == challenge["baseline_reserved_bytes"],
        "CUDA allocator did not return to the binding baseline",
    )


def _validate_gpu_provenance(
    gpu_value: Any,
    *,
    label: str,
    hardware: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    gpu = _mapping(gpu_value, f"{label} GPU provenance")
    requirements = _mapping(gpu.get("retained_requirements"), f"{label} GPU requirements")
    _require(requirements.get("enforced") is True, f"{label} did not enforce hardware requirements")
    _require(requirements.get("name") == GPU_NAME, f"{label} registered GPU name mismatch")
    _require(requirements.get("compute_capability") == GPU_COMPUTE_CAPABILITY,
             f"{label} registered compute capability mismatch")
    _require(requirements.get("minimum_total_memory_bytes") == GPU_MIN_MEMORY_BYTES,
             f"{label} minimum GPU memory mismatch")
    _require(requirements.get("foreign_consumers_at_endpoints") == 0,
             f"{label} foreign-consumer requirement mismatch")
    before = _validate_gpu_snapshot(gpu.get("before"), label=f"{label} GPU before", hardware=hardware)
    after = _validate_gpu_snapshot(gpu.get("after"), label=f"{label} GPU after", hardware=hardware)
    _validate_allocator_challenge(before, hardware=hardware)
    candidate = before["self_context_establishment"]["bound_host_namespace_pid"]
    _require(set(after.get("own_gpu_process_ids", [])) == {candidate},
             f"{label} after endpoint ownership differs from the challenged PID")
    _require(set(_process_memory(after)) == {candidate},
             f"{label} after endpoint contains an added owned PID")
    stable = ("index", "name", "nvidia_smi_query_id", "compute_capability",
              "total_memory_bytes", "multiprocessor_count", "uuid")
    _require(all(before.get(key) == after.get(key) for key in stable),
             f"{label} selected GPU changed between endpoints")
    _require(gpu.get("same_selected_device") is True,
             f"{label} same-device verdict is false")
    _require(gpu.get("foreign_compute_consumers_absent_at_endpoints") is True,
             f"{label} endpoint isolation verdict is false")
    _require("endpoint" in str(gpu.get("isolation_scope", "")).lower(),
             f"{label} overstates isolation scope")
    return before, after


def _validate_provenance(
    provenance_value: Any,
    *,
    label: str,
    hardware: dict[str, Any],
) -> dict[str, Any]:
    provenance = _mapping(provenance_value, f"{label} provenance")
    # Recovery embeds the lifecycle contract's provenance schema by design.
    _require(provenance.get("schema") == LIFECYCLE_SCHEMA,
             f"{label} nested provenance schema mismatch")
    source = _mapping(provenance.get("source"), f"{label} provenance source")
    source_snapshot = _validate_unchanged_pair(
        source.get("git_before"), source.get("git_after"),
        label=f"{label} source", commit=IMPLEMENTATION_COMMIT,
        tree=IMPLEMENTATION_TREE,
        source_tree_sha256=IMPLEMENTATION_SOURCE_TREE_SHA256,
    )
    _require(
        source_snapshot.get("repository") == "/workspace/nano-vllm-spec-v2"
        and source_snapshot.get("branch") == "feat/spec-v2-dual-runner"
        and source_snapshot.get("detached") is False,
        f"{label} source checkout identity mismatch",
    )
    _require(
        source.get("nanovllm_import_origin")
        == "/workspace/nano-vllm-spec-v2/nanovllm/__init__.py",
        f"{label} nano-vLLM import origin mismatch",
    )
    _require(source.get("unchanged_during_run") is True,
             f"{label} source unchanged verdict is false")
    _validate_software(
        provenance.get("software"),
        label=label,
        python_value=provenance.get("python"),
    )
    _validate_gpu_provenance(provenance.get("gpu"), label=label, hardware=hardware)
    return provenance


def _validate_v0(
    document_value: Any,
    *,
    mode: str,
    model_content: dict[str, Any],
    tokenizer: dict[str, Any],
    hardware: dict[str, Any],
) -> dict[str, Any]:
    document = _mapping(document_value, f"V0 {mode}")
    _reject_claim_critical_aliases(document, label=f"V0 {mode}")
    _require(document.get("schema") == V0_SCHEMA, f"V0 {mode} schema mismatch")
    source = _mapping(document.get("source"), f"V0 {mode} source")
    _require(source.get("expected_commit") == CANONICAL_V0_COMMIT,
             f"V0 {mode} expected commit mismatch")
    v0_source_snapshot = _validate_unchanged_pair(
        source.get("git_before"), source.get("git_after"),
        label=f"V0 {mode} source", commit=CANONICAL_V0_COMMIT,
        tree=CANONICAL_V0_TREE, detached=True,
    )
    _require(
        v0_source_snapshot.get("repository")
        == "/tmp/nano-vllm-spec-v0-480a3b2"
        and v0_source_snapshot.get("branch") is None,
        f"V0 {mode} detached source checkout identity mismatch",
    )
    _require(
        source.get("nanovllm_import_origin")
        == "/tmp/nano-vllm-spec-v0-480a3b2/nanovllm/__init__.py",
        f"V0 {mode} import origin mismatch",
    )
    _require(source.get("unchanged_during_run") is True,
             f"V0 {mode} source unchanged verdict is false")
    runner = _mapping(document.get("runner"), f"V0 {mode} runner")
    _require(runner.get("expected_commit") == IMPLEMENTATION_COMMIT,
             f"V0 {mode} runner commit mismatch")
    v0_runner_snapshot = _validate_unchanged_pair(
        runner.get("git_before"), runner.get("git_after"),
        label=f"V0 {mode} runner", commit=IMPLEMENTATION_COMMIT,
        tree=IMPLEMENTATION_TREE,
    )
    _require(
        v0_runner_snapshot.get("repository") == "/workspace/nano-vllm-spec-v2"
        and v0_runner_snapshot.get("branch") == "feat/spec-v2-dual-runner"
        and v0_runner_snapshot.get("detached") is False,
        f"V0 {mode} runner checkout identity mismatch",
    )
    _require(
        runner.get("script_path")
        == "/workspace/nano-vllm-spec-v2/tests/run_speculative_v0_golden.py",
        f"V0 {mode} runner script path mismatch",
    )
    _require(
        runner.get("script_sha256_before")
        == runner.get("script_sha256_after")
        == EXPECTED_RUNNER_HASHES["tests/run_speculative_v0_golden.py"],
        f"V0 {mode} runner script hash mismatch",
    )
    _require(runner.get("unchanged_during_run") is True,
             f"V0 {mode} runner unchanged verdict is false")
    model = _mapping(document.get("model"), f"V0 {mode} model")
    _require(model.get("before") == model.get("after"),
             f"V0 {mode} model changed during run")
    _require(model.get("unchanged_during_run") is True,
             f"V0 {mode} model unchanged verdict is false")
    _require(_model_content_identity(model.get("before")) == model_content,
             f"V0 {mode} model manifest mismatch")
    _require(document.get("tokenizer") == tokenizer,
             f"V0 {mode} tokenizer manifest mismatch")
    randomness = _mapping(document.get("randomness"), f"V0 {mode} randomness")
    _require(randomness.get("seed") == EXPECTED_PROTOCOL["seed"],
             f"V0 {mode} seed mismatch")
    _require(randomness.get("sampling_mode") == "greedy",
             f"V0 {mode} is not greedy")
    run = _mapping(document.get("run"), f"V0 {mode} run")
    _require(run.get("mode") == mode, f"V0 {mode} mode mismatch")
    requested = _mapping(run.get("requested_engine_config"), f"V0 {mode} requested config")
    expected_requested = {
        "disable_python_gc": False,
        "enforce_eager": mode == "eager",
        "gpu_memory_utilization": EXPECTED_PROTOCOL["gpu_memory_utilization"],
        "kvcache_block_size": EXPECTED_PROTOCOL["kvcache_block_size"],
        "max_model_len": EXPECTED_PROTOCOL["max_model_len"],
        "max_num_batched_tokens": EXPECTED_PROTOCOL["max_num_batched_tokens"],
        "max_num_seqs": EXPECTED_PROTOCOL["max_num_seqs"],
        "num_kvcache_blocks": -1,
        "tensor_parallel_size": EXPECTED_PROTOCOL["tensor_parallel_size"],
        "top_p_backend": EXPECTED_PROTOCOL["top_p_backend"],
    }
    _require(type(requested.get("disable_python_gc")) is bool
             and type(requested.get("enforce_eager")) is bool,
             f"V0 {mode} requested boolean config types are invalid")
    _require(type(requested.get("gpu_memory_utilization")) is float,
             f"V0 {mode} requested memory utilization must be a float")
    for key in (
        "kvcache_block_size", "max_model_len", "max_num_batched_tokens",
        "max_num_seqs", "num_kvcache_blocks", "tensor_parallel_size",
    ):
        _require(type(requested.get(key)) is int,
                 f"V0 {mode} requested {key} must be an integer")
    _require(requested == expected_requested, f"V0 {mode} requested config mismatch")
    effective = _mapping(run.get("effective_engine_config"), f"V0 {mode} effective config")
    for key, value in expected_requested.items():
        if key != "num_kvcache_blocks":
            _require(effective.get(key) == value, f"V0 {mode} effective {key} mismatch")
    _integer(effective.get("num_kvcache_blocks"), f"V0 {mode} KV blocks", minimum=1)
    _require(effective.get("model") == model_content["resolved_path"],
             f"V0 {mode} effective model path mismatch")
    hf_config = _mapping(effective.get("hf_config"), f"V0 {mode} HF config")
    _require(
        hf_config.get("model_type") == "qwen3"
        and hf_config.get("vocab_size") == EXPECTED_PROTOCOL["vocab_size"]
        and hf_config.get("dtype") == "torch.bfloat16"
        and _integer(
            hf_config.get("max_position_embeddings"),
            f"V0 {mode} max positions",
            minimum=EXPECTED_PROTOCOL["max_model_len"],
        ) >= EXPECTED_PROTOCOL["max_model_len"],
        f"V0 {mode} registered model geometry mismatch",
    )
    workload = {
        "prompts_token_ids": [[1, 2, 3, 4], [7, 8, 9]],
        "sampling_params": EXPECTED_PROTOCOL["sampling"],
    }
    _require(run.get("workload") == workload, f"V0 {mode} workload mismatch")
    outputs = _sequence(run.get("outputs"), f"V0 {mode} outputs")
    normalized = []
    for index, output in enumerate(outputs):
        output = _mapping(output, f"V0 {mode} output {index}")
        _require(output.get("index") == index, f"V0 {mode} output index mismatch")
        _require(isinstance(output.get("text"), str), f"V0 {mode} output text is invalid")
        tokens = _sequence(output.get("token_ids"), f"V0 {mode} output tokens")
        _require(len(tokens) == EXPECTED_PROTOCOL["sampling"]["max_tokens"],
                 f"V0 {mode} output token count mismatch")
        _require(all(type(token) is int and token >= 0 for token in tokens),
                 f"V0 {mode} output tokens are invalid")
        normalized.append([output["text"], tokens])
    _require(run.get("comparison_outputs") == normalized,
             f"V0 {mode} comparison output projection mismatch")
    _require(normalized == REGISTERED_OUTPUTS,
             f"V0 {mode} outputs differ from the registered control")
    trace = _sequence(run.get("scheduler_trace"), f"V0 {mode} scheduler trace")
    _require(trace == REGISTERED_SCHEDULER_TRACE,
             f"V0 {mode} scheduler trace differs from the registered control")
    _require(run.get("scheduler_trace_schema") == [
        "is_prefill",
        ["sequence_length", "num_cached_tokens", "num_scheduled_tokens", "sequence_is_prefill"],
    ], f"V0 {mode} scheduler trace schema mismatch")
    _require(run.get("teardown") == {"exit_calls_completed": 2},
             f"V0 {mode} teardown mismatch")
    runtime = _mapping(document.get("runtime"), f"V0 {mode} runtime")
    software = _validate_software(
        runtime.get("software"),
        label=f"V0 {mode}",
        python_value=runtime.get("python"),
    )
    gpu = _mapping(runtime.get("gpu"), f"V0 {mode} GPU")
    _require(gpu.get("name") == GPU_NAME, f"V0 {mode} GPU name mismatch")
    _require(gpu.get("compute_capability") == GPU_COMPUTE_CAPABILITY,
             f"V0 {mode} compute capability mismatch")
    _require(gpu.get("total_memory_bytes") == hardware["total_memory_bytes"],
             f"V0 {mode} GPU memory mismatch")
    _require(gpu.get("uuid") == hardware["uuid"], f"V0 {mode} GPU UUID mismatch")
    _require(gpu.get("nvidia_smi_returncode") == 0, f"V0 {mode} nvidia-smi failed")
    return {
        "outputs": normalized,
        "trace": trace,
        "tokenizer": document["tokenizer"],
        "model": model_content,
        "software": software,
    }


def _validate_model_wrapper(value: Any, *, label: str, model_content: dict[str, Any]) -> None:
    wrapper = _mapping(value, f"{label} model artifacts")
    _require(wrapper.get("same_resolved_model") is True,
             f"{label} target/draft fixture is not the registered same-model cell")
    for role in ("target", "draft"):
        row = _mapping(wrapper.get(role), f"{label} {role} model")
        _require(row.get("before") == row.get("after"),
                 f"{label} {role} model changed during run")
        _require(row.get("unchanged_during_run") is True,
                 f"{label} {role} model unchanged verdict is false")
        _require(_model_content_identity(row.get("before")) == model_content,
                 f"{label} {role} model manifest mismatch")


def _validate_lifecycle(
    document_value: Any,
    *,
    mode: str,
    v0_sha256: str,
    v0_record: dict[str, Any],
    model_content: dict[str, Any],
    tokenizer: dict[str, Any],
    protocol: dict[str, Any],
    hardware: dict[str, Any],
) -> dict[str, Any]:
    document = _mapping(document_value, f"V2 {mode}")
    _reject_claim_critical_aliases(document, label=f"V2 {mode}")
    _require(document.get("evidence_schema") == LIFECYCLE_SCHEMA,
             f"V2 {mode} schema mismatch")
    _require(document.get("mode") == mode, f"V2 {mode} mode mismatch")
    _require(document.get("certification_mode") == "retained",
             f"V2 {mode} is not retained")
    _require(document.get("retention_eligible") is True,
             f"V2 {mode} is not retention-eligible")
    _require(document.get("configured_k") == protocol["configured_k"],
             f"V2 {mode} K mismatch")
    _require(type(document.get("configured_k")) is int,
             f"V2 {mode} K must be an integer")
    _require(document.get("source_policy") == {
        "expected_commit": IMPLEMENTATION_COMMIT,
        "allow_dirty": False,
    }, f"V2 {mode} source policy mismatch")
    scope = _mapping(document.get("scope"), f"V2 {mode} scope")
    _require(scope.get("feature_stage") == EXPECTED_CLAIM_BOUNDARY["feature_stage"],
             f"V2 {mode} feature stage mismatch")
    _require(scope.get("tensor_parallel_size") == 1, f"V2 {mode} TP scope mismatch")
    limitations = _sequence(scope.get("limitations"), f"V2 {mode} limitations")
    limitations_text = " ".join(map(str, limitations)).lower()
    for phrase in ("draft forward", "no speculative proposal", "no latency", "gpu_certified=false"):
        _require(phrase in limitations_text, f"V2 {mode} omits limitation: {phrase}")
    historical = _mapping(document.get("historical_v0_comparison"), f"V2 {mode} V0 comparison")
    _require(historical.get("sha256") == v0_sha256,
             f"V2 {mode} historical V0 hash mismatch")
    _require(historical.get("source_commit") == CANONICAL_V0_COMMIT,
             f"V2 {mode} historical V0 commit mismatch")
    _require(historical.get("runner_commit") == IMPLEMENTATION_COMMIT,
             f"V2 {mode} historical runner commit mismatch")
    _require(historical.get("mode") == mode, f"V2 {mode} historical mode mismatch")
    _require(historical.get("outputs_match") is True
             and historical.get("scheduler_trace_match") is True,
             f"V2 {mode} historical parity verdict is false")
    _require(historical.get("outputs") == v0_record["outputs"],
             f"V2 {mode} historical outputs mismatch")
    _require(historical.get("scheduler_trace") == v0_record["trace"],
             f"V2 {mode} historical trace mismatch")
    _validate_model_wrapper(document.get("model_artifacts"), label=f"V2 {mode}",
                            model_content=model_content)
    runtime_tokenizers = _mapping(document.get("runtime_tokenizers"), f"V2 {mode} tokenizers")
    _require(runtime_tokenizers.get("speculation_off") == tokenizer
             and runtime_tokenizers.get("speculation_on") == tokenizer,
             f"V2 {mode} runtime tokenizer mismatch")
    tokenizer_manifest_sha256 = canonical_sha256(tokenizer)
    _require(
        tokenizer_manifest_sha256 == REGISTERED_TOKENIZER_MANIFEST_SHA256,
        f"V2 {mode} tokenizer pair is not the registered identity",
    )
    fingerprint = document.get("speculative_tokenizer_fingerprint")
    _hex_digest(fingerprint, f"V2 {mode} tokenizer fingerprint")
    _require(
        fingerprint
        == REGISTERED_SPECULATIVE_TOKENIZER_FINGERPRINTS[tokenizer_manifest_sha256],
        f"V2 {mode} tokenizer fingerprint is not tied to the tokenizer pair",
    )
    output_fields = (
        "speculation_off_outputs", "speculation_on_outputs", "post_second_engine_outputs"
    )
    for field in output_fields:
        _require(document.get(field) == v0_record["outputs"],
                 f"V2 {mode} {field} mismatch")
    trace_fields = (
        "speculation_off_scheduler_trace", "speculation_on_scheduler_trace",
        "post_second_engine_scheduler_trace",
    )
    for field in trace_fields:
        _require(document.get(field) == v0_record["trace"],
                 f"V2 {mode} {field} mismatch")
    for field in (
        "construction_rng_identity", "post_generation_rng_identity",
        "inert_output_identity", "inert_scheduler_trace_identity",
        "half_config_rejected_while_healthy",
        "valid_second_engine_rejected_while_healthy",
        "explicit_boundary_requested", "post_failure_recovery", "exit_idempotency",
    ):
        _require(document.get(field) is True, f"V2 {mode} {field} is false")
    _require(document.get("draft_forward_calls_during_generation") == 0,
             f"V2 {mode} executed the draft model")
    _require(type(document.get("draft_forward_calls_during_generation")) is int,
             f"V2 {mode} draft forward count must be an integer")
    rng = _mapping(document.get("rng_snapshots"), f"V2 {mode} RNG snapshots")
    expected_rng_keys = {
        "speculation_off_after_construction",
        "speculation_on_after_construction",
        "speculation_off_after_generation",
        "speculation_on_after_generation",
    }
    _require(set(rng) == expected_rng_keys,
             f"V2 {mode} RNG snapshot set mismatch")
    validated_rng = {
        key: _validate_rng_snapshot(value, label=f"V2 {mode} RNG {key}")
        for key, value in rng.items()
    }
    _require(validated_rng["speculation_off_after_construction"]
             == validated_rng["speculation_on_after_construction"],
             f"V2 {mode} construction RNG mismatch")
    _require(validated_rng["speculation_off_after_generation"]
             == validated_rng["speculation_on_after_generation"],
             f"V2 {mode} post-generation RNG mismatch")
    half_error = _mapping(document.get("half_config_error"), f"V2 {mode} half-config error")
    _require(half_error.get("type") == "builtins.ValueError"
             and half_error.get("message")
             == "num_speculative_tokens > 0 requires draft_model",
             f"V2 {mode} half-config diagnostic mismatch")
    second_error = _mapping(document.get("valid_second_engine_error"),
                            f"V2 {mode} second-engine error")
    _require(second_error.get("type") == "builtins.ValueError"
             and second_error.get("message")
             == "trying to initialize the default process group twice!",
             f"V2 {mode} second-engine diagnostic mismatch")
    automatic_audit, automatic_reconciliation = _validate_memory_audit(
        document.get("speculative_memory_audit"),
        document.get("independent_memory_reconciliation"),
        label=f"V2 {mode} automatic", mode=mode, protocol=protocol,
        hardware=hardware, automatic=True,
    )
    selected = automatic_audit["selected_num_blocks"]
    _require(document.get("explicit_selected_blocks_passed") == selected,
             f"V2 {mode} explicit N differs from automatic N")
    explicit_result = _mapping(
        document.get("explicit_boundary"),
        f"V2 {mode} explicit boundary result",
    )
    _require(
        explicit_result.get("selected_blocks") == selected
        and explicit_result.get("outputs") == v0_record["outputs"]
        and explicit_result.get("scheduler_trace") == v0_record["trace"]
        and explicit_result.get("tokenizer_fingerprint") == fingerprint,
        f"V2 {mode} explicit-N result mismatch",
    )
    explicit_audit, explicit_reconciliation = _validate_memory_audit(
        document.get("explicit_memory_audit"), document.get("explicit_memory_reconciliation"),
        label=f"V2 {mode} explicit", mode=mode, protocol=protocol,
        hardware=hardware, automatic=False, expected_blocks=selected,
    )
    _require(
        explicit_audit == automatic_audit
        and explicit_reconciliation == automatic_reconciliation,
        f"V2 {mode} explicit audit drifted from the automatic ceiling",
    )
    first_ineligible = document.get("first_ineligible_blocks_rejected")
    _require(first_ineligible == selected + 1,
             f"V2 {mode} first-ineligible boundary is not N+1")
    capacity_error = _mapping(document.get("first_ineligible_capacity_error"),
                              f"V2 {mode} capacity error")
    _require(capacity_error.get("type")
             == "nanovllm.engine.model_runner.SpeculativeKVCacheCapacityError",
             f"V2 {mode} capacity error type mismatch")
    requested_bytes = first_ineligible * automatic_audit["joint_block_bytes"]
    expected_capacity_message = (
        f"requested num_kvcache_blocks={first_ineligible} requires "
        f"{requested_bytes} joint KV bytes, but only "
        f"{automatic_reconciliation['sizing_usable_bytes']} modeled bytes "
        f"({selected} blocks) are usable after sizing_overhead="
        f"{automatic_reconciliation['sizing_overhead_bytes']} "
        f"(graph_construction="
        f"{automatic_reconciliation['graph_construction_reservation_bytes']}, "
        f"runtime={automatic_reconciliation['runtime_reservation_bytes']}) reservation"
    )
    _require(capacity_error.get("message") == expected_capacity_message,
             f"V2 {mode} N+1 capacity diagnostic mismatch")
    recovery_run = _mapping(document.get("post_failure_recovery_run"),
                            f"V2 {mode} post-failure recovery")
    recovery_blocks = min(selected, max(protocol["max_num_seqs"], 2))
    _require(recovery_run.get("selected_blocks") == recovery_blocks,
             f"V2 {mode} post-failure recovery block count mismatch")
    _require(recovery_run.get("outputs") == v0_record["outputs"]
             and recovery_run.get("scheduler_trace") == v0_record["trace"],
             f"V2 {mode} post-failure recovery parity mismatch")
    _require(recovery_run.get("tokenizer_fingerprint") == fingerprint,
             f"V2 {mode} post-failure tokenizer mismatch")
    _validate_memory_audit(
        document.get("recovery_memory_audit"), document.get("recovery_memory_reconciliation"),
        label=f"V2 {mode} recovery", mode=mode, protocol=protocol,
        hardware=hardware, automatic=False, expected_blocks=recovery_blocks,
    )
    provenance = _validate_provenance(
        document.get("provenance"), label=f"V2 {mode}", hardware=hardware
    )
    return {
        "outputs": v0_record["outputs"],
        "trace": v0_record["trace"],
        "tokenizer_fingerprint": fingerprint,
        "plan": automatic_audit["workspace_plan"],
        "software": provenance["software"],
        "selected_blocks": selected,
    }


def _validate_recovery(
    document_value: Any,
    *,
    phase: str,
    mode: str,
    lifecycle: dict[str, Any],
    model_content: dict[str, Any],
    protocol: dict[str, Any],
    hardware: dict[str, Any],
) -> dict[str, Any]:
    document = _mapping(document_value, f"recovery {phase}")
    _reject_claim_critical_aliases(document, label=f"recovery {phase}")
    _require(document.get("evidence_schema") == RECOVERY_SCHEMA,
             f"recovery {phase} schema mismatch")
    _require(document.get("certification_mode") == "retained"
             and document.get("retention_eligible") is True,
             f"recovery {phase} is not retained")
    _require(document.get("phase") == phase, f"recovery phase mismatch: {phase}")
    _require(document.get("mode") == mode, f"recovery {phase} mode mismatch")
    _require(document.get("configured_k") == protocol["configured_k"],
             f"recovery {phase} K mismatch")
    _require(type(document.get("configured_k")) is int,
             f"recovery {phase} K must be an integer")
    _require(document.get("explicit_num_kvcache_blocks")
             == protocol["recovery_num_kvcache_blocks"],
             f"recovery {phase} block count mismatch")
    _require(type(document.get("explicit_num_kvcache_blocks")) is int,
             f"recovery {phase} block count must be an integer")
    _require(document.get("injection_semantics")
             == "raised after the selected real phase completed successfully",
             f"recovery {phase} injection semantics mismatch")
    ledger = _mapping(document.get("injection_ledger"), f"recovery {phase} ledger")
    expected_calls = 2 if phase == "draft_graph" else 1
    _require(ledger == {"calls": expected_calls, "injections": 1},
             f"recovery {phase} injection ledger mismatch")
    caught = _mapping(document.get("caught_error"), f"recovery {phase} caught error")
    _require(str(caught.get("type", "")).endswith(".InjectedSpeculativePhaseError")
             and caught.get("message") == f"injected after real V2 phase: {phase}",
             f"recovery {phase} caught error mismatch")
    for key in (
        "process_group_destroyed_after_failure",
        "torch_defaults_and_context_restored_after_failure",
        "recovery_engine_succeeded",
    ):
        _require(document.get(key) is True, f"recovery {phase} {key} is false")
    origins = _mapping(document.get("runtime_import_origins"),
                       f"recovery {phase} import origins")
    _require(
        origins == EXPECTED_RECOVERY_IMPORT_ORIGINS,
        f"recovery {phase} import origins are not the certified checkout",
    )
    memory = _mapping(document.get("cuda_memory"), f"recovery {phase} CUDA memory")
    for key in (
        "baseline_allocated_bytes", "baseline_reserved_bytes",
        "after_failure_allocated_bytes", "after_failure_reserved_bytes",
        "after_recovery_exit_allocated_bytes", "after_recovery_exit_reserved_bytes",
    ):
        _integer(memory.get(key), f"recovery {phase} {key}")
    _require(
        memory["baseline_allocated_bytes"] == 0
        and memory["baseline_reserved_bytes"] == 0,
        f"recovery {phase} did not start from a fresh CUDA allocator baseline",
    )
    _require(memory.get("post_execution_global_allocated_ceiling_bytes")
             == POST_EXECUTION_ALLOCATED_CEILING_BYTES,
             f"recovery {phase} allocated ceiling mismatch")
    _require(memory.get("post_execution_global_reserved_ceiling_bytes")
             == POST_EXECUTION_RESERVED_CEILING_BYTES,
             f"recovery {phase} reserved ceiling mismatch")
    for prefix in ("after_failure", "after_recovery_exit"):
        _require(
            memory[f"{prefix}_allocated_bytes"]
            <= memory["baseline_allocated_bytes"] + POST_EXECUTION_ALLOCATED_CEILING_BYTES,
            f"recovery {phase} exceeded allocated-memory ceiling",
        )
        _require(
            memory[f"{prefix}_reserved_bytes"]
            <= memory["baseline_reserved_bytes"] + POST_EXECUTION_RESERVED_CEILING_BYTES,
            f"recovery {phase} exceeded reserved-memory ceiling",
        )
        _require(memory[f"{prefix}_allocated_bytes"] <= memory[f"{prefix}_reserved_bytes"],
                 f"recovery {phase} allocated memory exceeds reserved memory")
    _validate_model_wrapper(document.get("model_artifacts"), label=f"recovery {phase}",
                            model_content=model_content)
    recovery = _mapping(document.get("recovery"), f"recovery {phase} result")
    _require(recovery.get("outputs") == lifecycle["outputs"],
             f"recovery {phase} outputs mismatch")
    _require(recovery.get("scheduler_trace") == lifecycle["trace"],
             f"recovery {phase} scheduler trace mismatch")
    _require(recovery.get("tokenizer_fingerprint") == lifecycle["tokenizer_fingerprint"],
             f"recovery {phase} tokenizer fingerprint mismatch")
    audit, _ = _validate_memory_audit(
        recovery.get("memory_audit"), recovery.get("independent_memory_reconciliation"),
        label=f"recovery {phase}", mode=mode, protocol=protocol,
        hardware=hardware, automatic=False,
        expected_blocks=protocol["recovery_num_kvcache_blocks"],
    )
    _require(audit["workspace_plan"] == lifecycle["plan"],
             f"recovery {phase} workspace plan differs from lifecycle")
    provenance = _validate_provenance(
        document.get("provenance"), label=f"recovery {phase}", hardware=hardware
    )
    return {"software": provenance["software"]}


def _validate_manifest(root: Path, manifest: dict[str, Any]) -> dict[str, Path]:
    readme = _resolve_payload(root, "README.md")
    _require(
        (readme.stat().st_size, _sha256_file(readme)) == TRUSTED_README,
        "README.md differs from the trusted release registry",
    )
    _require(
        set(TRUSTED_RAW_ARTIFACTS) == set(ARTIFACT_SPECS),
        "trusted raw-artifact registry is incomplete",
    )
    _require(manifest.get("schema_version") == MANIFEST_SCHEMA_VERSION,
             "manifest schema version mismatch")
    _require(manifest.get("kind") == ARCHIVE_KIND, "manifest kind mismatch")
    _require(
        manifest.get("archive_id") == "2026-08-28-a100-v2-d87f168",
        "manifest archive ID mismatch",
    )
    _require(
        manifest.get("claim_boundary") == EXPECTED_CLAIM_BOUNDARY,
        "manifest claim boundary mismatch",
    )
    claim_boundary = _mapping(manifest.get("claim_boundary"), "manifest claim boundary")
    _require(claim_boundary.get("memory_workspace_gpu_certified") is False,
             "manifest workspace certification flag must be boolean false")
    _require(all(type(item) is str for item in claim_boundary.get("certified", []))
             and all(type(item) is str for item in claim_boundary.get("not_certified", [])),
             "manifest claim lists must contain only strings")
    _require(manifest.get("source") == {
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "implementation_tree": IMPLEMENTATION_TREE,
        "implementation_source_tree_sha256": IMPLEMENTATION_SOURCE_TREE_SHA256,
        "canonical_v0_commit": CANONICAL_V0_COMMIT,
        "canonical_v0_tree": CANONICAL_V0_TREE,
    }, "manifest source identity mismatch")
    manifest_protocol = _validate_protocol_types(manifest.get("protocol"))
    _require(
        manifest_protocol == EXPECTED_MANIFEST_PROTOCOL,
        "manifest protocol mismatch",
    )
    hardware = _mapping(manifest.get("registered_hardware"), "manifest hardware")
    _require(hardware.get("name") == GPU_NAME, "manifest GPU name mismatch")
    _require(hardware.get("compute_capability") == GPU_COMPUTE_CAPABILITY,
             "manifest compute capability mismatch")
    _require(all(type(value) is int for value in hardware["compute_capability"]),
             "manifest compute capability must contain integers")
    _integer(hardware.get("total_memory_bytes"), "manifest total GPU memory", minimum=GPU_MIN_MEMORY_BYTES)
    _require(hardware.get("minimum_total_memory_bytes") == GPU_MIN_MEMORY_BYTES,
             "manifest minimum GPU memory mismatch")
    _integer(hardware.get("multiprocessor_count"), "manifest multiprocessor count", minimum=1)
    _require(isinstance(hardware.get("uuid"), str) and hardware["uuid"],
             "manifest GPU UUID is missing")
    _require(isinstance(hardware.get("driver_version"), str) and hardware["driver_version"],
             "manifest driver version is missing")
    challenge = _mapping(hardware.get("allocator_challenge"),
                         "manifest allocator challenge")
    for key in (
        "requested_bytes",
        "allocated_and_reserved_delta_bytes",
        "nvidia_smi_delta_mib",
        "rounding_tolerance_mib",
    ):
        _integer(challenge.get(key), f"manifest allocator challenge {key}")
    _require(type(challenge.get("reserved_delta_mib")) is float,
             "manifest reserved_delta_mib must be a float")
    _require(challenge == {
        "requested_bytes": GPU_PID_CHALLENGE_BYTES,
        "allocated_and_reserved_delta_bytes": 390 * 1024**2,
        "reserved_delta_mib": 390.0,
        "nvidia_smi_delta_mib": 390,
        "rounding_tolerance_mib": GPU_PID_ROUNDING_TOLERANCE_MIB,
    }, "manifest allocator challenge mismatch")
    _require(
        manifest.get("registered_software") == EXPECTED_REGISTERED_SOFTWARE,
        "manifest registered software mismatch",
    )
    runner_map = _mapping(manifest.get("runner_files"), "manifest runner files")
    for path, digest in runner_map.items():
        _require(isinstance(path, str), "manifest runner path is invalid")
        _hex_digest(digest, f"manifest runner {path} SHA256")
    _require(runner_map == EXPECTED_RUNNER_HASHES, "manifest runner hash set mismatch")
    repo_root = Path(__file__).resolve().parents[2]
    for relative, digest in runner_map.items():
        path = (repo_root / relative).resolve(strict=True)
        _require(path.is_relative_to(repo_root), f"runner path escapes repository: {relative}")
        _require(_sha256_file(path) == digest, f"current runner hash mismatch: {relative}")
    identity = _mapping(manifest.get("model_identity"), "manifest model identity")
    _require(identity.get("fixture") == "/workspace/models/Qwen3-0.6B",
             "manifest model fixture mismatch")
    _require(identity.get("target_equals_draft") is True,
             "manifest target/draft identity mismatch")
    _require(identity.get("canonical_content_definition") == [
        "resolved_path", "metadata_files", "weight_files", "total_weight_bytes"
    ], "manifest model-content definition mismatch")
    _hex_digest(identity.get("canonical_content_sha256"),
                "manifest canonical model-content SHA256")
    _hex_digest(identity.get("tokenizer_manifest_sha256"),
                "manifest tokenizer SHA256")
    weight = _mapping(identity.get("weight_file"), "manifest model weight")
    _require(weight.get("name") == "model.safetensors",
             "manifest model weight name mismatch")
    _integer(weight.get("size_bytes"), "manifest model weight size", minimum=1)
    _hex_digest(weight.get("sha256"), "manifest model weight SHA256")
    _require(manifest.get("rejected_attempts") == {
        "count": 2,
        "retained_raw_artifacts": 0,
        "details": "README.md",
    }, "manifest rejected-attempt ledger mismatch")
    rows = _sequence(manifest.get("artifacts"), "manifest artifacts")
    _require(len(rows) == len(ARTIFACT_SPECS), "manifest artifact count mismatch")
    paths: list[str] = []
    retained: dict[str, Path] = {}
    for row in rows:
        row = _mapping(row, "manifest artifact row")
        relative = row.get("path")
        _require(relative in ARTIFACT_SPECS, f"unexpected artifact path: {relative}")
        role, mode, phase, schema = ARTIFACT_SPECS[relative]
        _require(row.get("role") == role, f"artifact role mismatch: {relative}")
        _require(row.get("mode") == mode, f"artifact mode mismatch: {relative}")
        _require(row.get("phase") == phase, f"artifact phase mismatch: {relative}")
        _require(row.get("schema") == schema, f"artifact schema mismatch: {relative}")
        original_path = row.get("original_output_path")
        _require(isinstance(original_path, str) and PurePosixPath(original_path).is_absolute(),
                 f"artifact original_output_path must be absolute: {relative}")
        expected_commit = CANONICAL_V0_COMMIT if role == "canonical_v0" else IMPLEMENTATION_COMMIT
        _require(row.get("source_commit") == expected_commit,
                 f"artifact source commit mismatch: {relative}")
        _require(row.get("retention_eligible") is True,
                 f"artifact is not retention-eligible: {relative}")
        trusted_size, trusted_sha256 = TRUSTED_RAW_ARTIFACTS[relative]
        _require(
            row.get("size_bytes") == trusted_size
            and row.get("sha256") == trusted_sha256,
            f"manifest row differs from trusted artifact registry: {relative}",
        )
        path = _resolve_payload(root, relative)
        _require(path.stat().st_size == trusted_size,
                 f"trusted artifact size mismatch: {relative}")
        digest = row.get("sha256")
        _hex_digest(digest, f"artifact {relative} SHA256")
        _require(_sha256_file(path) == trusted_sha256,
                 f"trusted artifact hash mismatch: {relative}")
        paths.append(relative)
        retained[relative] = path
    _require(paths == sorted(paths) and len(paths) == len(set(paths)),
             "manifest artifact paths must be unique and sorted")
    _require(set(paths) == set(ARTIFACT_SPECS), "manifest artifact set mismatch")
    descendants = list(root.rglob("*"))
    for path in descendants:
        info = path.lstat()
        relative = path.relative_to(root).as_posix()
        _require(not stat.S_ISLNK(info.st_mode),
                 f"archive contains an unmanifested symlink: {relative}")
        _require(stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode),
                 f"archive contains a non-file/non-directory entry: {relative}")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in descendants
        if stat.S_ISREG(path.lstat().st_mode)
        and path.relative_to(root).as_posix() not in IGNORED_ARCHIVE_FILES
    }
    actual_directories = {
        path.relative_to(root).as_posix()
        for path in descendants
        if stat.S_ISDIR(path.lstat().st_mode)
    }
    _require(actual_directories == {"raw"},
             "archive directory set must contain only raw/")
    _require(actual_files == set(ARTIFACT_SPECS),
             "archive file set differs from the manifest")
    return retained


def validate_archive(archive: str | Path) -> dict[str, Any]:
    requested = Path(archive).expanduser()
    try:
        requested_info = requested.lstat()
    except FileNotFoundError as error:
        raise EvidenceValidationError(f"archive path does not exist: {requested}") from error
    _require(not stat.S_ISLNK(requested_info.st_mode),
             f"archive/manifest path must not be a symlink: {requested}")
    if stat.S_ISDIR(requested_info.st_mode):
        root = requested.resolve(strict=True)
        manifest_path = root / "manifest.json"
    else:
        _require(stat.S_ISREG(requested_info.st_mode) and requested.name == "manifest.json",
                 "archive argument must be a directory or manifest.json")
        manifest_path = requested.resolve(strict=True)
        root = manifest_path.parent
    try:
        manifest_info = manifest_path.lstat()
    except FileNotFoundError as error:
        raise EvidenceValidationError(f"missing archive manifest: {manifest_path}") from error
    _require(stat.S_ISREG(manifest_info.st_mode) and not stat.S_ISLNK(manifest_info.st_mode),
             "manifest.json must be a regular non-symlink file")
    _require(manifest_info.st_nlink == 1, "manifest.json must not be hard-linked")
    _require(
        (manifest_info.st_size, _sha256_file(manifest_path)) == TRUSTED_MANIFEST,
        "manifest.json differs from the trusted release registry",
    )
    manifest = _load_json(manifest_path)
    retained = _validate_manifest(root, manifest)
    documents = {relative: _load_json(path) for relative, path in retained.items()}
    canonical_v0 = documents["raw/v0-eager.json"]
    canonical_model_snapshot = _mapping(canonical_v0.get("model"), "canonical V0 model").get("before")
    model_content = _model_content_identity(canonical_model_snapshot)
    tokenizer = _validate_tokenizer(canonical_v0.get("tokenizer"))
    identity = manifest["model_identity"]
    _require(
        canonical_sha256(model_content) == identity["canonical_content_sha256"],
        "canonical V0 model-content hash differs from the manifest",
    )
    _require(
        canonical_sha256(tokenizer) == identity["tokenizer_manifest_sha256"],
        "canonical V0 tokenizer hash differs from the manifest",
    )
    _require(model_content["resolved_path"] == identity["fixture"],
             "canonical V0 model path differs from the manifest")
    _require(len(model_content["weight_files"]) == 1
             and model_content["weight_files"][0] == identity["weight_file"],
             "canonical V0 weight manifest differs from the manifest")
    hardware = manifest["registered_hardware"]
    protocol = EXPECTED_PROTOCOL

    v0: dict[str, dict[str, Any]] = {}
    lifecycle: dict[str, dict[str, Any]] = {}
    artifact_rows = {row["path"]: row for row in manifest["artifacts"]}
    for mode in ("eager", "graph"):
        v0_path = f"raw/v0-{mode}.json"
        v0[mode] = _validate_v0(
            documents[v0_path], mode=mode, model_content=model_content,
            tokenizer=tokenizer, hardware=hardware,
        )
    _require(v0["eager"] == v0["graph"],
             "canonical V0 eager/graph outputs, trace, model, tokenizer, or software differ")
    for mode in ("eager", "graph"):
        v0_path = f"raw/v0-{mode}.json"
        lifecycle_path = f"raw/v2-{mode}.json"
        lifecycle[mode] = _validate_lifecycle(
            documents[lifecycle_path], mode=mode,
            v0_sha256=artifact_rows[v0_path]["sha256"],
            v0_record=v0[mode], model_content=model_content,
            tokenizer=tokenizer, protocol=protocol, hardware=hardware,
        )
    for key in ("outputs", "trace", "tokenizer_fingerprint", "plan", "software"):
        _require(lifecycle["eager"][key] == lifecycle["graph"][key],
                 f"V2 eager/graph cross-artifact mismatch: {key}")
    for phase, mode in RECOVERY_PHASE_MODES.items():
        _validate_recovery(
            documents[f"raw/recovery-{phase}.json"], phase=phase, mode=mode,
            lifecycle=lifecycle[mode], model_content=model_content,
            protocol=protocol, hardware=hardware,
        )
    return {
        "kind": ARCHIVE_KIND,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "artifact_count": len(retained),
        "lifecycle_modes": ["eager", "graph"],
        "recovery_phases": sorted(RECOVERY_PHASE_MODES),
        "claim_boundary": EXPECTED_CLAIM_BOUNDARY,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a GPU-free retained speculative-V2 evidence archive "
            "from a quiescent local checkout."
        )
    )
    parser.add_argument(
        "archive",
        nargs="+",
        help="archive directory or its manifest.json path",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    reports = [validate_archive(path) for path in args.archive]
    print(json.dumps(reports, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
