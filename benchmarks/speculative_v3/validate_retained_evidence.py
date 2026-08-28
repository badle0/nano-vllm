#!/usr/bin/env python3
"""Validate the retained speculative-V3 archive without CUDA, Torch, or a model.

The archive is a certificate for one deliberately narrow implementation stage:
transactional draft-model proposal/discard execution.  Validation has three
independent trust layers:

* a release registry pins every retained byte by size and SHA-256;
* strict schema and cross-artifact checks explain what those bytes prove; and
* historical runner/helper bytes are read from the immutable producer commit,
  never from the current worktree.

PyTorch ``.pt`` cache sidecars are *not* loaded with pickle or Torch.  A
restricted unpickler substitutes inert metadata objects for the three allowed
globals, and raw ZIP storage members are checked directly.  Validation is
therefore CPU/GPU/model-free and cannot execute an archive-supplied callable.

This validator assumes a quiescent local filesystem while it runs.  It rejects
symlinks, hard links, non-regular files, duplicate ZIP/JSON names, non-finite
JSON numbers, and any unregistered archive member.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import math
import os
import pickle
import pickletools
import re
import stat
import struct
import subprocess
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Mapping, Sequence


ARCHIVE_ID = "2026-08-28-a100-v3-e8e0452"
ARCHIVE_KIND = "nano_vllm_speculative_v3_retained_archive"
MANIFEST_SCHEMA_VERSION = 1

PRODUCER_COMMIT = "e8e0452f99727958077b51f340a5375a090e6884"
PRODUCER_TREE = "51a5a1f1ee5452c8f1de4b08dc4647c9fff6dcdf"
PRODUCER_SOURCE_TREE_SHA256 = (
    "c60fbec34960d522f7b9dc2437ed3a0c8eeccda2d90f2adb9cf4ed994b6e1079"
)
IMPLEMENTATION_COMMIT = "7fec9993d5e4e0e06fec22e3973dfc203fdbd2d8"
IMPLEMENTATION_TREE = "5820fb685d76b549fe21117baa31b3b32ebae14b"
IMPLEMENTATION_PARENT = "e252086ee625ea8aefdb63f3affc093380f40064"
IMPLEMENTATION_NANOVLLM_TREE = "52398af379f767708a0b804646f4b490fa8323ad"

SEED = 20260828
VOCAB_SIZE = 151936
ROUTE_RECORD_COUNT = 32
GPU_NAME = "NVIDIA A100-SXM4-40GB"
GPU_COMPUTE_CAPABILITY = [8, 0]
GPU_TOTAL_MEMORY_BYTES = 42406903808
GPU_UUID = "773e0633-edb0-6c38-1d2b-d232f9109126"
NVIDIA_DRIVER = "570.133.20"

ROUTE_SCHEMA = "nano-vllm-speculative-v3-route-compile-v2"
ROUTE_VALIDATION_SCHEMA = "nano-vllm-speculative-v3-route-compile-validation-v2"
CACHE_SCHEMA = "nano-vllm-speculative-v3-cache-neutrality-v2"
CACHE_COMPARISON_SCHEMA = "nano-vllm-speculative-v3-cache-neutrality-comparison-v2"
OUTPUT_SCHEMA = "nano-vllm-speculative-v3-output-control-v2"
OUTPUT_COMPARISON_SCHEMA = "nano-vllm-speculative-v3-output-control-comparison-v2"

BEGIN_PREFIX = "V3_DRAFT_INTERVAL_BEGIN "
END_PREFIX = "V3_DRAFT_INTERVAL_END "
MARKER_FRAGMENT = "V3_DRAFT_INTERVAL_"
UUID4_HEX = re.compile(r"[0-9a-f]{12}4[0-9a-f]{3}[89ab][0-9a-f]{15}")
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")

HISTORICAL_SOURCE_HASHES = {
    "tests/_speculative_v3_evidence.py": (
        "73319d770cfb3650f3c66cd26141e73ca2eb26a4fe8568cf18ab79a5ec099b35"
    ),
    "tests/run_speculative_v3_route_compile.py": (
        "ec537370bc9b20642a4ec896413aa658b83ed1572c57f9e4a3c9c408ef36f35c"
    ),
    "tests/validate_speculative_v3_route_compile.py": (
        "1ee54a4040e2d6377e2c989a4f99ba55bc7d3f6fe1a00e65ccfcadd6453aa40b"
    ),
    "tests/run_speculative_v3_cache_neutrality.py": (
        "d38b87620de99f6adb096d9ee84a786c059b7297bbe5ed1f2fe8fc2309f3d076"
    ),
    "tests/compare_speculative_v3_cache_neutrality.py": (
        "f722e29dc2773e14ff20ded3ab4321e919285b8e87b59c14e2ce5f2d4f923634"
    ),
    "tests/run_speculative_v3_output_control.py": (
        "6b8ac128d5ef60c715f7a9d58c2dc075ad177e8d1faca27fbdda26a47477fd21"
    ),
    "tests/compare_speculative_v3_output_control.py": (
        "6a1529f6154b27a5757fbe4c107a6a71cc50debf44906d7c1666c2665a3251d6"
    ),
}

# Immutable release registry.  The manifest is an index, not a trust root:
# changing an artifact and then editing its manifest row cannot manufacture a
# new certificate.
TRUSTED_PAYLOADS: dict[str, tuple[int, str]] = {
    "comparisons/cache-eager.json": (38810, "cadee6a5a1097acd9d3355340764229f82830b8df9d74247cdddbed6aedbb230"),
    "comparisons/cache-graph.json": (38830, "9c27d3412058ff1c552391bc213a0d1e4406f82ca2df8632b8e2cb24bc7da2a1"),
    "comparisons/output-eager.json": (36251, "b820290b25e97f83ea1c58dd44a6e289fe59448e087261295ce73f086b56a864"),
    "comparisons/output-graph.json": (36251, "683d798aa40386485b5315a998874ee4d8b6cbe8bb1d736fe9b87ca92a88ec21"),
    "raw/cache-eager-nan.json": (32144, "39e6351eb5b49574de553c36df2826adff079580e8e07a7e54cbfa6ea04a29a2"),
    "raw/cache-eager-nan.tensors.pt": (8210413, "184c4b26a1b4e8f48ef2b2247609922aadc81f4c9400c84155e198f7436d8e69"),
    "raw/cache-eager-zero.json": (32151, "10049622dddc60973f104c3211bba0ebfade4b292acc0366487cb2657d99084e"),
    "raw/cache-eager-zero.tensors.pt": (8210413, "71aa51263cf9df7074b884177583c26aa2f9f4939777647ee20b2873a726db99"),
    "raw/cache-graph-nan.json": (32155, "ad862ed1af7ca35bb49b71adab2fd9a0fa0b27b5fd93f92362967885b1e55ac6"),
    "raw/cache-graph-nan.tensors.pt": (8210413, "9f1dbb022600cb246377013cf91ee014568e640c7aba3ece3ed96ae019e96a77"),
    "raw/cache-graph-zero.json": (32162, "eb4a70f8114473566a93ddfaf7e4183b2cdd1af5044d5d47ac898cda172d16d9"),
    "raw/cache-graph-zero.tensors.pt": (8210413, "f1424cbd0e61d24da541043a9660ca0cc53b972fd5d64e6e12ba0df497a0826c"),
    "raw/output-eager-off.json": (24816, "b423ac9d93b2adde4864865f833e2eba0c5e928e23bb895d2c51fa77621fc054"),
    "raw/output-eager-on.json": (27041, "07607d76e5e1e8d8de4da6d39c2ccaa0658492d150c71db1054af238ae9ee12b"),
    "raw/output-graph-off.json": (24816, "065b05aed8a3819d276f65b0d9fd8be4975af302de0a91f38d9bfa6bf90a3514"),
    "raw/output-graph-on.json": (27035, "a8d5d9010e6168df9b1e9b7f93b74f6d086582ba837cad8302a07e4229b8a8f5"),
    "raw/route-eager.json": (326198, "c02cc201dce99918e465bb8aa7be483192d012318afdaba5318a3f74463ded33"),
    "raw/route-eager.log": (19962, "3b4d002a239e3080f203d5beaaf80ec7f9b51c26341a07c396168b65a6f9418a"),
    "raw/route-graph.json": (367425, "53fb2fbce216168c13847a8ffaef65c96f37d2c887133d862e5293ab9bf95225"),
    "raw/route-graph.log": (32369, "76f3c08f0d187f89571715c5fb015e94a1ba9499bbdf4142ffac1ab4b750a22f"),
    "validations/route.json": (48611, "0c9f057031b99277ec17af31f428de60cf4d13323fb034894d285cfbcbda94e3"),
}

# Filled after README.md and manifest.json are finalized.  Validation fails
# closed while either value is None.  ``--print-registry`` prints the exact two
# tuples to paste here; that operation does not claim to validate the archive.
TRUSTED_README: tuple[int, str] | None = (
    7101,
    "f51520a3a09a4ae003b83f8e0f4d2bb78adbec40acd245ae45a5cfad3530d0f0",
)
TRUSTED_MANIFEST: tuple[int, str] | None = (
    11188,
    "297d992135dea071bc701f860a14ac4c001d040fd033562b17cc2005e2c57d5f",
)

EXPECTED_SOFTWARE = {
    "cudnn": 91002,
    "platform": "Linux-5.15.0-1079-nvidia-x86_64-with-glibc2.39",
    "python": "3.12.13",
    "python_executable": "/venv/main/bin/python3.12",
    "python_optimize": 0,
    "torch": "2.10.0+cu128",
    "torch_cuda": "12.8",
    "torch_dynamo_disable": False,
    "torch_dynamo_suppress_errors": False,
}

EXPECTED_MODEL_CONTENT = {
    "metadata_files": {
        "config.json": {"sha256": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd", "size_bytes": 726},
        "generation_config.json": {"sha256": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2", "size_bytes": 239},
        "merges.txt": {"sha256": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5", "size_bytes": 1671853},
        "tokenizer.json": {"sha256": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4", "size_bytes": 11422654},
        "tokenizer_config.json": {"sha256": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101", "size_bytes": 9732},
        "vocab.json": {"sha256": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910", "size_bytes": 2776833},
    },
    "resolved_path": "/workspace/models/Qwen3-0.6B",
    "total_weight_bytes": 1503300328,
    "weight_files": [
        {"name": "model.safetensors", "sha256": "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b", "size_bytes": 1503300328}
    ],
}

CLAIM_BOUNDARY = {
    "feature_stage": "V3 transactional draft-model proposal/discard execution",
    "certified": [
        "complete eager and CUDA-graph draft route coverage for B=1..4 and K=1..2",
        "no compiler, graph-capture, RNG, or attention-context mutation inside 64 run-bound draft intervals",
        "temporary draft KV reservation rollback at a block boundary and exact zero/NaN cache-fill independence",
        "speculation-off/on authoritative target events, target tokens, and four target RNG endpoints are exact",
    ],
    "not_certified": [
        "target verification or speculative acceptance/rejection",
        "multi-token speculative commit, speculative streaming, or speculative metrics",
        "latency, throughput, acceptance rate, or speedup",
        "heterogeneous target/draft model geometry",
        "tensor parallelism, multi-GPU execution, or FlashInfer",
    ],
    "gpu_isolation_scope": "recorded single visible retained A100 identity; not continuous external-process isolation",
}

RUNTIME_IMPORT_PATHS = {
    "nanovllm": "nanovllm/__init__.py",
    "LLM": "nanovllm/llm.py",
    "LLMEngine": "nanovllm/engine/llm_engine.py",
    "ModelRunner": "nanovllm/engine/model_runner.py",
    "Scheduler": "nanovllm/engine/scheduler.py",
    "Sampler": "nanovllm/layers/sampler.py",
    "SamplingParams": "nanovllm/sampling_params.py",
}

CACHE_RECORD_IDS = (
    "boundary/step-0",
    "boundary/step-1",
    "boundary/step-2",
    "shared-prefix/cold/step-0",
    "shared-prefix/cold/step-1",
    "shared-prefix/cold/step-2",
    "shared-prefix/hit/step-0",
    "shared-prefix/hit/step-1",
    "shared-prefix/hit/step-2",
)

DRAFT_PHASES = (
    "_construct_draft_model",
    "warmup_draft_model",
    "capture_draft_cudagraph",
    "_pretouch_draft_eager_prefill",
    "_pretouch_draft_routes",
)
DRAFT_RESOURCE_ATTRIBUTES = (
    "draft_model", "draft_kv_cache", "draft_graphs", "draft_graph_vars",
    "draft_graph_pool", "draft_graph_bs", "speculative_memory_plan",
    "speculative_memory_audit", "draft_route_registry",
    "_speculative_memory_audit_inputs", "_profiled_graph_allocated_bytes",
    "_profiled_graph_reserved_bytes", "_profiled_graph_peak_allocated_bytes",
    "_profiled_graph_peak_reserved_bytes", "_final_graph_allocated_baseline",
    "_final_graph_reserved_baseline", "_allocated_after_graph_before_pretouch",
    "_reserved_after_graph_before_pretouch", "_final_graph_peak_allocated_bytes",
    "_final_graph_peak_reserved_bytes", "_target_warmup_transient_bytes",
    "_draft_warmup_transient_bytes", "_warmup_transient_bytes",
    "_draft_route_pretouch_peak_bytes",
)
DRAFT_OWNED_ATTRIBUTES = (
    "_draft_route_pretouch_peak_bytes", "_draft_warmup_transient_bytes",
    "_speculative_memory_audit_inputs", "draft_graph_bs", "draft_graph_pool",
    "draft_graph_vars", "draft_graphs", "draft_kv_cache", "draft_model",
    "draft_route_registry", "speculative_memory_audit", "speculative_memory_plan",
)

EXPECTED_PROTOCOL = {
    "seed": SEED,
    "route_compile": {
        "configured_k": 2,
        "batch_sizes": [1, 2, 3, 4],
        "effective_k": [1, 2],
        "phases": ["cold", "warm"],
        "repetitions": 2,
        "records_per_mode": 32,
        "registry_cardinality": {"eager": 4, "graph": 12},
    },
    "cache_neutrality": {
        "configured_k": 3,
        "fills": ["zero", "nan"],
        "record_ids": list(CACHE_RECORD_IDS),
        "vocab_size": VOCAB_SIZE,
        "comparison": "bitwise exact",
    },
    "output_control": {
        "configured_k": 2,
        "prompt_token_ids": [[10, 11, 12, 13], [20, 21, 22]],
        "sampling": {
            "temperature": 0.8,
            "top_k": 8,
            "top_p": 0.9,
            "max_tokens": 6,
            "ignore_eos": True,
        },
        "target_steps": ["prefill", "first_target_decode", "repeated_target_decode"],
    },
}


def _artifact_row_metadata(relative: str) -> tuple[str, str | None, str | None]:
    parts = PurePosixPath(relative).parts
    name = parts[-1]
    if name.startswith("route-"):
        mode = "eager" if "eager" in name else "graph"
        kind = "route_log" if name.endswith(".log") else "route_json"
        return kind, mode, None
    if name.startswith("cache-"):
        mode = "eager" if "eager" in name else "graph"
        if parts[0] == "comparisons":
            return "cache_comparison", mode, None
        fill = "zero" if "zero" in name else "nan"
        kind = "cache_tensor_sidecar" if name.endswith(".pt") else "cache_json"
        return kind, mode, fill
    if name.startswith("output-"):
        mode = "eager" if "eager" in name else "graph"
        if parts[0] == "comparisons":
            return "output_comparison", mode, None
        side = "off" if "-off." in name else "on"
        return "output_json", mode, side
    _require(relative == "validations/route.json", f"unknown registered artifact: {relative}")
    return "route_validation", None, None


def expected_manifest() -> dict[str, Any]:
    artifacts = []
    for relative, (size_bytes, digest) in sorted(TRUSTED_PAYLOADS.items()):
        kind, mode, variant = _artifact_row_metadata(relative)
        artifacts.append(
            {
                "path": relative,
                "size_bytes": size_bytes,
                "sha256": digest,
                "kind": kind,
                "mode": mode,
                "variant": variant,
            }
        )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": ARCHIVE_KIND,
        "archive_id": ARCHIVE_ID,
        "producer_commit": PRODUCER_COMMIT,
        "producer_tree": PRODUCER_TREE,
        "producer_source_tree_sha256": PRODUCER_SOURCE_TREE_SHA256,
        "implementation": {
            "commit": IMPLEMENTATION_COMMIT,
            "tree": IMPLEMENTATION_TREE,
            "parent": IMPLEMENTATION_PARENT,
            "nanovllm_tree": IMPLEMENTATION_NANOVLLM_TREE,
        },
        "artifact_count": len(TRUSTED_PAYLOADS),
        "artifacts": artifacts,
        "historical_sources": dict(sorted(HISTORICAL_SOURCE_HASHES.items())),
        "registered_hardware": {
            "name": GPU_NAME,
            "compute_capability": GPU_COMPUTE_CAPABILITY,
            "total_memory_bytes": GPU_TOTAL_MEMORY_BYTES,
            "uuid": GPU_UUID,
            "driver_version": NVIDIA_DRIVER,
        },
        "registered_software": EXPECTED_SOFTWARE,
        "model_identity": EXPECTED_MODEL_CONTENT,
        "protocol": EXPECTED_PROTOCOL,
        "claim_boundary": CLAIM_BOUNDARY,
        "validator": {
            "path": "benchmarks/speculative_v3/validate_retained_evidence.py",
            "runtime": "Python standard library only",
            "command": "python benchmarks/speculative_v3/validate_retained_evidence.py <archive>",
        },
    }


class EvidenceValidationError(ValueError):
    """A retained artifact violates the immutable V3 certificate contract."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise EvidenceValidationError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> list[Any]:
    _require(isinstance(value, list), f"{label} must be a list")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    _require(type(value) is int, f"{label} must be a JSON integer")
    _require(value >= minimum, f"{label} must be >= {minimum}")
    return value


def _boolean(value: Any, label: str) -> bool:
    _require(type(value) is bool, f"{label} must be a JSON boolean")
    return value


def _sha(value: Any, label: str, *, bits: int = 256) -> str:
    pattern = HEX64 if bits == 256 else HEX40
    _require(isinstance(value, str) and pattern.fullmatch(value), f"{label} is not lowercase SHA-{bits}")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    _require(set(value) == expected, f"{label} schema drifted")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise EvidenceValidationError(f"non-finite JSON number: {value}")


def _finite_tree(value: Any, path: str = "$") -> None:
    if isinstance(value, float):
        _require(math.isfinite(value), f"non-finite JSON number at {path}")
    elif isinstance(value, dict):
        for key, child in value.items():
            _finite_tree(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite_tree(child, f"{path}[{index}]")


def load_strict_json_bytes(payload: bytes, label: str = "artifact") -> dict[str, Any]:
    _require(len(payload) <= 64 * 1024 * 1024, f"{label} exceeds JSON size limit")
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except EvidenceValidationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceValidationError(f"invalid JSON: {label}") from error
    _require(isinstance(value, dict), f"{label} JSON root must be an object")
    _finite_tree(value)
    return value


def _safe_relative(value: str, label: str) -> PurePosixPath:
    _require(isinstance(value, str) and value and "\\" not in value, f"{label} is not a POSIX relative path")
    path = PurePosixPath(value)
    _require(
        not path.is_absolute()
        and path.parts
        and all(part not in ("", ".", "..") for part in path.parts)
        and path.as_posix() == value,
        f"{label} is not normalized and confined",
    )
    return path


def _open_confined_regular(root: Path, relative: str) -> tuple[BinaryIO, os.stat_result]:
    pure = _safe_relative(relative, "archive member path")
    current = root
    try:
        for part in pure.parts[:-1]:
            current = current / part
            info = current.lstat()
            _require(stat.S_ISDIR(info.st_mode), f"archive ancestor is not a directory: {relative}")
            _require(not stat.S_ISLNK(info.st_mode), f"archive path contains a symlink: {relative}")
        target = current / pure.name
        before = target.lstat()
    except FileNotFoundError as error:
        raise EvidenceValidationError(f"missing archive member: {relative}") from error
    _require(stat.S_ISREG(before.st_mode) and not stat.S_ISLNK(before.st_mode), f"archive member is not regular: {relative}")
    _require(before.st_nlink == 1, f"archive member must not be hard-linked: {relative}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(target, flags)
    except OSError as error:
        raise EvidenceValidationError(f"cannot safely open archive member: {relative}") from error
    stream = os.fdopen(fd, "rb")
    opened = os.fstat(stream.fileno())
    _require(
        (opened.st_dev, opened.st_ino, opened.st_size, opened.st_nlink)
        == (before.st_dev, before.st_ino, before.st_size, before.st_nlink),
        f"archive member changed while opening: {relative}",
    )
    return stream, opened


def _read_registered(root: Path, relative: str, expected: tuple[int, str]) -> bytes:
    stream, opened = _open_confined_regular(root, relative)
    try:
        payload = stream.read(expected[0] + 1)
        after = os.fstat(stream.fileno())
    finally:
        stream.close()
    _require(len(payload) == expected[0], f"archive member size mismatch: {relative}")
    _require(opened.st_size == expected[0] and after.st_size == expected[0], f"archive member size changed: {relative}")
    _require((opened.st_dev, opened.st_ino) == (after.st_dev, after.st_ino), f"archive member identity changed: {relative}")
    _require(_sha256_bytes(payload) == expected[1], f"archive member SHA-256 mismatch: {relative}")
    return payload


def _enumerate_archive_files(root: Path) -> set[str]:
    result: set[str] = set()
    stack = [(root, PurePosixPath())]
    while stack:
        directory, prefix = stack.pop()
        for entry in os.scandir(directory):
            relative = (prefix / entry.name).as_posix()
            _safe_relative(relative, "archive filesystem path")
            info = entry.stat(follow_symlinks=False)
            _require(not stat.S_ISLNK(info.st_mode), f"archive contains a symlink: {relative}")
            if stat.S_ISDIR(info.st_mode):
                stack.append((Path(entry.path), prefix / entry.name))
            else:
                _require(stat.S_ISREG(info.st_mode), f"archive contains a non-regular file: {relative}")
                _require(info.st_nlink == 1, f"archive contains a hard-linked file: {relative}")
                result.add(relative)
    return result


def _archive_root(argument: str | Path) -> Path:
    requested = Path(argument).expanduser()
    try:
        info = requested.lstat()
    except FileNotFoundError as error:
        raise EvidenceValidationError(f"archive path does not exist: {requested}") from error
    _require(not stat.S_ISLNK(info.st_mode), "archive/manifest path must not be a symlink")
    if stat.S_ISDIR(info.st_mode):
        root = requested.resolve(strict=True)
    else:
        _require(stat.S_ISREG(info.st_mode) and requested.name == "manifest.json", "archive argument must be a directory or manifest.json")
        root = requested.resolve(strict=True).parent
    _require(root.name == ARCHIVE_ID, f"archive directory must be named {ARCHIVE_ID}")
    return root


def _git_bytes(repo_root: Path, *arguments: str, label: str) -> bytes:
    environment = os.environ.copy()
    for name in (
        "GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_NAMESPACE", "GIT_INDEX_FILE",
    ):
        environment.pop(name, None)
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        completed = subprocess.run(
            ("git", "--no-replace-objects", "-C", str(repo_root), *arguments),
            check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=environment, shell=False, timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        raise EvidenceValidationError(f"cannot read historical Git object: {label}") from error
    _require(completed.returncode == 0, f"historical Git object unavailable: {label}; fetch {PRODUCER_COMMIT}")
    return completed.stdout


def _validate_historical_sources(repo_root: Path) -> dict[str, dict[str, str]]:
    object_type = _git_bytes(repo_root, "cat-file", "-t", PRODUCER_COMMIT, label="producer commit").strip()
    _require(object_type == b"commit", "registered producer object is not a commit")
    tree = _git_bytes(repo_root, "rev-parse", f"{PRODUCER_COMMIT}^{{tree}}", label="producer tree").decode().strip()
    _require(tree == PRODUCER_TREE, "registered producer tree mismatch")
    implementation_tree = _git_bytes(repo_root, "rev-parse", f"{IMPLEMENTATION_COMMIT}^{{tree}}", label="implementation tree").decode().strip()
    _require(implementation_tree == IMPLEMENTATION_TREE, "registered implementation tree mismatch")
    parent = _git_bytes(repo_root, "rev-parse", f"{IMPLEMENTATION_COMMIT}^", label="implementation parent").decode().strip()
    _require(parent == IMPLEMENTATION_PARENT, "registered implementation parent mismatch")
    nanovllm_tree = _git_bytes(repo_root, "rev-parse", f"{IMPLEMENTATION_COMMIT}:nanovllm", label="implementation nanovllm tree").decode().strip()
    _require(nanovllm_tree == IMPLEMENTATION_NANOVLLM_TREE, "registered nanovllm tree mismatch")
    result = {}
    for relative, expected_sha in HISTORICAL_SOURCE_HASHES.items():
        payload = _git_bytes(repo_root, "cat-file", "blob", f"{PRODUCER_COMMIT}:{relative}", label=relative)
        _require(_sha256_bytes(payload) == expected_sha, f"historical source SHA-256 mismatch: {relative}")
        blob = _git_bytes(repo_root, "rev-parse", f"{PRODUCER_COMMIT}:{relative}", label=f"blob {relative}").decode().strip()
        _sha(blob, f"historical source blob {relative}", bits=160)
        result[relative] = {"sha256": expected_sha, "blob": blob}
    return result


def _historical_runtime_blobs(repo_root: Path) -> dict[str, dict[str, str]]:
    paths = set(RUNTIME_IMPORT_PATHS.values()) | {"nanovllm/engine/speculative_routes.py"}
    result = {}
    for relative in sorted(paths):
        payload = _git_bytes(repo_root, "cat-file", "blob", f"{PRODUCER_COMMIT}:{relative}", label=relative)
        blob = _git_bytes(repo_root, "rev-parse", f"{PRODUCER_COMMIT}:{relative}", label=f"blob {relative}").decode().strip()
        result[relative] = {"sha256": _sha256_bytes(payload), "blob": blob}
    return result


def _validate_identity_row(
    value: Any,
    *,
    label: str,
    relative: str,
    registered: Mapping[str, str],
) -> dict[str, Any]:
    row = _mapping(value, label)
    _exact_keys(
        row,
        {"path", "sha256", "head_blob", "head_blob_sha256", "matches_head"},
        label,
    )
    _require(row["path"] == relative, f"{label} path mismatch")
    _require(row["sha256"] == registered["sha256"], f"{label} runtime hash mismatch")
    _require(row["head_blob"] == registered["blob"], f"{label} Git blob mismatch")
    _require(row["head_blob_sha256"] == registered["sha256"], f"{label} Git blob hash mismatch")
    _require(row["matches_head"] is True, f"{label} did not match the producer commit")
    return row


def _validate_model_snapshot(value: Any, label: str) -> dict[str, Any]:
    snapshot = _mapping(value, label)
    _exact_keys(
        snapshot,
        {"argument", "metadata_files", "resolved_path", "total_weight_bytes", "weight_files"},
        label,
    )
    _require(isinstance(snapshot["argument"], str) and snapshot["argument"], f"{label} argument is invalid")
    content = {key: snapshot[key] for key in EXPECTED_MODEL_CONTENT}
    _require(content == EXPECTED_MODEL_CONTENT, f"{label} model-content manifest mismatch")
    metadata = _mapping(snapshot["metadata_files"], f"{label} metadata")
    for name, descriptor in metadata.items():
        _safe_relative(name, f"{label} metadata name")
        descriptor = _mapping(descriptor, f"{label} metadata {name}")
        _exact_keys(descriptor, {"size_bytes", "sha256"}, f"{label} metadata {name}")
        _integer(descriptor["size_bytes"], f"{label} metadata {name} size", minimum=1)
        _sha(descriptor["sha256"], f"{label} metadata {name} SHA-256")
    total = 0
    names = []
    for row in _sequence(snapshot["weight_files"], f"{label} weights"):
        row = _mapping(row, f"{label} weight row")
        _exact_keys(row, {"name", "size_bytes", "sha256"}, f"{label} weight row")
        names.append(row["name"])
        total += _integer(row["size_bytes"], f"{label} weight bytes", minimum=1)
        _sha(row["sha256"], f"{label} weight SHA-256")
    _require(names == sorted(set(names)), f"{label} weight inventory is not sorted/unique")
    _require(total == snapshot["total_weight_bytes"], f"{label} total weight bytes mismatch")
    return snapshot


def _validate_recorded_environment(
    value: Any,
    *,
    route: bool,
    gpu_expected: bool,
) -> dict[str, Any]:
    environment = _mapping(value, "recorded environment")
    _exact_keys(environment, {"hardware", "selected_environment", "software"}, "recorded environment")
    hardware = _mapping(environment["hardware"], "recorded hardware")
    _exact_keys(hardware, {"cuda_available", "device_count", "devices", "nvidia_smi"}, "recorded hardware")
    devices = _sequence(hardware["devices"], "recorded devices")
    if gpu_expected:
        _require(hardware["cuda_available"] is True, "GPU producer did not record CUDA")
        _require(type(hardware["device_count"]) is int and hardware["device_count"] == 1, "GPU producer did not record one visible GPU")
        _require(len(devices) == 1, "GPU producer device registry must contain one GPU")
        device = _mapping(devices[0], "recorded GPU")
        _exact_keys(device, {"index", "name", "compute_capability", "total_memory_bytes", "uuid"}, "recorded GPU")
        _require(
            device
            == {
                "index": 0,
                "name": GPU_NAME,
                "compute_capability": GPU_COMPUTE_CAPABILITY,
                "total_memory_bytes": GPU_TOTAL_MEMORY_BYTES,
                "uuid": GPU_UUID,
            },
            "recorded GPU identity differs from the registered A100",
        )
    else:
        _require(
            hardware["cuda_available"] is False
            and type(hardware["device_count"]) is int
            and hardware["device_count"] == 0
            and devices == [],
            "offline comparator did not run with CUDA disabled",
        )
    _require(
        hardware["nvidia_smi"]
        == f"0, GPU-{GPU_UUID}, {GPU_NAME}, {NVIDIA_DRIVER}, 40960",
        "recorded nvidia-smi identity mismatch",
    )
    _require(environment["software"] == EXPECTED_SOFTWARE, "recorded software environment mismatch")
    selected = _mapping(environment["selected_environment"], "selected environment")
    _require(
        selected.get("CUDA_VISIBLE_DEVICES") == ("0" if gpu_expected else ""),
        "recorded CUDA visibility differs from producer role",
    )
    _require(selected.get("PYTHONDONTWRITEBYTECODE") == "1", "producer did not disable bytecode writes")
    _require(selected.get("PYTHONPATH") == ".", "producer PYTHONPATH drifted")
    _require(selected.get("TORCH_COMPILE_DISABLE") in (None, "", "0"), "producer disabled compilation")
    _require(selected.get("TORCHDYNAMO_DISABLE") in (None, "", "0"), "producer disabled Dynamo")
    _require(selected.get("TORCHDYNAMO_SUPPRESS_ERRORS") in (None, "", "0"), "producer suppressed Dynamo errors")
    if route:
        for name in (
            "TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE",
            "TORCHINDUCTOR_AUTOGRAD_CACHE",
            "TORCHINDUCTOR_AUTOGRAD_REMOTE_CACHE",
            "TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE",
            "TORCHINDUCTOR_BUNDLED_AUTOTUNE_REMOTE_CACHE",
            "TORCH_DYNAMO_AUTOMATIC_DYNAMIC_LOCAL_PGO",
            "TORCH_DYNAMO_AUTOMATIC_DYNAMIC_REMOTE_PGO",
        ):
            _require(selected.get(name) == "0", f"route producer did not disable {name}")
        logs = {item.strip() for item in str(selected.get("TORCH_LOGS") or "").split(",")}
        _require({"recompiles", "graph_breaks"}.issubset(logs), "route producer omitted compiler diagnostics")
    for name in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        cache = selected.get(name)
        if not gpu_expected and name == "TRITON_CACHE_DIR":
            _require(cache is None, "offline comparator unexpectedly selected a Triton cache")
        else:
            _require(isinstance(cache, str) and PurePosixPath(cache).is_absolute(), f"producer {name} is not absolute")
    return environment


def _validate_source_snapshot(value: Any, label: str) -> dict[str, Any]:
    snapshot = _mapping(value, label)
    _exact_keys(
        snapshot,
        {
            "branch", "clean", "detached", "head", "nanovllm_tree",
            "repository", "source_tree_sha256", "status_porcelain_v1",
            "status_sha256", "tree",
        },
        label,
    )
    _require(snapshot["head"] == PRODUCER_COMMIT, f"{label} commit mismatch")
    _require(snapshot["tree"] == PRODUCER_TREE, f"{label} tree mismatch")
    _require(snapshot["nanovllm_tree"] == IMPLEMENTATION_NANOVLLM_TREE, f"{label} nanovllm tree mismatch")
    _require(snapshot["source_tree_sha256"] == PRODUCER_SOURCE_TREE_SHA256, f"{label} source-tree hash mismatch")
    _require(snapshot["clean"] is True and snapshot["status_porcelain_v1"] == [], f"{label} source was dirty")
    _require(snapshot["status_sha256"] == hashlib.sha256(b"").hexdigest(), f"{label} clean-status hash mismatch")
    _require(snapshot["detached"] is False and snapshot["branch"] == "feat/spec-v2-draft-path", f"{label} branch identity mismatch")
    _require(snapshot["repository"] == "/workspace/nano-vllm-spec-v2", f"{label} repository identity mismatch")
    return snapshot


def _validate_invocation(
    value: Any,
    *,
    runner_path: str,
    required_options: Mapping[str, str | None],
) -> list[str]:
    invocation = _sequence(value, "producer invocation")
    _require(invocation and all(isinstance(item, str) for item in invocation), "producer invocation is invalid")
    _require(invocation[0] == f"/workspace/nano-vllm-spec-v2/{runner_path}", "producer invocation runner mismatch")
    arguments = invocation[1:]
    _require(arguments.count("--retained") == 1, "producer invocation must contain --retained exactly once")
    for option, expected in required_options.items():
        _require(arguments.count(option) == 1, f"producer invocation must contain {option} exactly once")
        index = arguments.index(option)
        if expected is not None:
            _require(index + 1 < len(arguments) and arguments[index + 1] == expected, f"producer invocation {option} drifted")
    _require("--expected-commit" in arguments, "producer invocation omitted expected commit")
    index = arguments.index("--expected-commit")
    _require(arguments[index + 1] == PRODUCER_COMMIT, "producer invocation commit mismatch")
    return invocation


def _validate_provenance(
    value: Any,
    *,
    runner_path: str,
    runtime_extra: tuple[str, str] | None,
    has_models: bool,
    route_environment: bool,
    required_options: Mapping[str, str | None],
    historical_sources: Mapping[str, Mapping[str, str]],
    runtime_blobs: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    provenance = _mapping(value, "provenance")
    _exact_keys(
        provenance,
        {
            "cwd", "environment", "implementation", "invocation", "models",
            "producer_commit", "retention_eligible", "retention_requested",
            "runtime_imports", "source", "source_files",
        },
        "provenance",
    )
    _require(provenance["cwd"] == "/workspace/nano-vllm-spec-v2", "provenance cwd mismatch")
    _require(provenance["producer_commit"] == PRODUCER_COMMIT, "provenance producer commit mismatch")
    _require(provenance["retention_requested"] is True and provenance["retention_eligible"] is True, "provenance is not retention eligible")
    _require(
        provenance["implementation"]
        == {
            "commit": IMPLEMENTATION_COMMIT,
            "tree": IMPLEMENTATION_TREE,
            "parent": IMPLEMENTATION_PARENT,
            "nanovllm_tree": IMPLEMENTATION_NANOVLLM_TREE,
        },
        "provenance implementation identity mismatch",
    )

    source = _mapping(provenance["source"], "source endpoints")
    _exact_keys(source, {"before", "after", "unchanged"}, "source endpoints")
    before_source = _validate_source_snapshot(source["before"], "source before")
    after_source = _validate_source_snapshot(source["after"], "source after")
    _require(source["unchanged"] is True and before_source == after_source, "source changed during retained run")

    source_files = _mapping(provenance["source_files"], "source-file endpoints")
    _exact_keys(source_files, {"before", "after", "unchanged"}, "source-file endpoints")
    expected_files = {
        "helper": "tests/_speculative_v3_evidence.py",
        "runner": runner_path,
    }
    for endpoint in ("before", "after"):
        rows = _mapping(source_files[endpoint], f"source files {endpoint}")
        _exact_keys(rows, set(expected_files), f"source files {endpoint}")
        for role, relative in expected_files.items():
            _validate_identity_row(
                rows[role], label=f"source {role} {endpoint}", relative=relative,
                registered=historical_sources[relative],
            )
    _require(source_files["unchanged"] is True and source_files["before"] == source_files["after"], "source files changed during retained run")

    runtime = _mapping(provenance["runtime_imports"], "runtime-import endpoints")
    _exact_keys(runtime, {"before", "after", "unchanged"}, "runtime-import endpoints")
    expected_imports = dict(RUNTIME_IMPORT_PATHS)
    if runtime_extra is not None:
        expected_imports[runtime_extra[0]] = runtime_extra[1]
    if not has_models:
        expected_imports = {}
    for endpoint in ("before", "after"):
        rows = _mapping(runtime[endpoint], f"runtime imports {endpoint}")
        _exact_keys(rows, set(expected_imports), f"runtime imports {endpoint}")
        for role, relative in expected_imports.items():
            _validate_identity_row(
                rows[role], label=f"runtime import {role} {endpoint}",
                relative=relative, registered=runtime_blobs[relative],
            )
    _require(runtime["unchanged"] is True and runtime["before"] == runtime["after"], "runtime imports changed during retained run")

    models = _mapping(provenance["models"], "model provenance")
    if has_models:
        _exact_keys(models, {"target", "draft"}, "model provenance")
        for role in ("target", "draft"):
            record = _mapping(models[role], f"{role} model endpoints")
            _exact_keys(record, {"before", "after", "unchanged"}, f"{role} model endpoints")
            before = _validate_model_snapshot(record["before"], f"{role} model before")
            after = _validate_model_snapshot(record["after"], f"{role} model after")
            _require(record["unchanged"] is True and before == after, f"{role} model changed during retained run")
        _require(
            {key: models["target"]["before"][key] for key in EXPECTED_MODEL_CONTENT}
            == {key: models["draft"]["before"][key] for key in EXPECTED_MODEL_CONTENT},
            "target/draft model bytes differ",
        )
    else:
        _require(models == {}, "CPU validator/comparator provenance unexpectedly loaded models")

    environment = _mapping(provenance["environment"], "environment endpoints")
    _exact_keys(environment, {"before", "after", "unchanged"}, "environment endpoints")
    before_env = _validate_recorded_environment(
        environment["before"], route=route_environment, gpu_expected=has_models
    )
    after_env = _validate_recorded_environment(
        environment["after"], route=route_environment, gpu_expected=has_models
    )
    _require(environment["unchanged"] is True and before_env == after_env, "recorded environment changed during run")
    _validate_invocation(provenance["invocation"], runner_path=runner_path, required_options=required_options)
    return provenance


ROUTE_ROOT_KEYS = {
    "schema", "run_id", "generated_at", "mode", "seed", "model",
    "draft_model", "configured_k", "configuration", "registry_cardinality",
    "visited_cardinality", "route_pretouch_peak_bytes",
    "capture_ledger_after_init", "capture_ledger_after_runtime",
    "all_draft_intervals_compiler_state_unchanged",
    "compiler_state_after_init", "compiler_state_after_init_sha256",
    "compiler_state_after_runtime", "compiler_state_after_runtime_sha256",
    "post_init_to_runtime_compiler_delta", "records", "provenance",
    "retention_eligible",
}
ROUTE_RECORD_KEYS = {
    "route", "live_batch_size", "catchup_tokens", "q_shape", "q_stride",
    "graph_decode_steps", "eager_decode_steps", "target_token_ids",
    "proposed_token_ids", "compiler_delta",
    "compiler_snapshot_before_sha256", "compiler_snapshot_after_sha256",
    "rng_neutral", "context_reset", "host_result_cuda_free", "repetition",
    "phase",
}
COMPILER_KEYS = {
    "counters", "guard_failures", "graph_break_reasons", "cache_manifest",
    "cuda_graph_objects", "cuda_graph_contexts",
}


def _batch_bucket(mode: str, batch_size: int) -> int:
    if mode == "eager":
        return 4
    return 1 if batch_size == 1 else 2 if batch_size == 2 else 4


def _route(mode: str, batch_size: int, effective_k: int, phase: str) -> dict[str, Any]:
    return {
        "schema": "draft-discard-v1",
        "execution_mode": "eager_dynamic" if mode == "eager" else "cuda_graph",
        "batch_bucket": _batch_bucket(mode, batch_size),
        "effective_k": effective_k,
        "catchup_family": "paged_eager_dynamic_v1" if phase == "cold" else "none",
        "sampler_envelope": "exact_all_compositions_worst_case_v1",
    }


def _route_specs() -> tuple[tuple[int, int, int, str], ...]:
    return tuple(
        (batch_size, effective_k, repetition, phase)
        for batch_size in range(1, 5)
        for effective_k in range(1, 3)
        for repetition in range(2)
        for phase in ("cold", "warm")
    )


def _validate_cache_manifest(value: Any, label: str) -> list[dict[str, Any]]:
    manifest = _sequence(value, label)
    _require(manifest, f"{label} is empty")
    identities = []
    for index, raw in enumerate(manifest):
        row = _mapping(raw, f"{label}[{index}]")
        _exact_keys(row, {"root", "path", "bytes", "sha256"}, f"{label}[{index}]")
        _require(row["root"] in ("inductor", "triton"), f"{label}[{index}] cache root drifted")
        path = _safe_relative(row["path"], f"{label}[{index}] path")
        _integer(row["bytes"], f"{label}[{index}] bytes")
        _sha(row["sha256"], f"{label}[{index}] SHA-256")
        identities.append((row["root"], path.as_posix()))
    expected_order = sorted(identities, key=lambda item: ((0 if item[0] == "inductor" else 1), item[1]))
    _require(identities == expected_order and len(set(identities)) == len(identities), f"{label} order/uniqueness drifted")
    _require({item[0] for item in identities} == {"inductor", "triton"}, f"{label} omitted a cache root")
    _require(sum(row["bytes"] for row in manifest) > 0, f"{label} is byte-empty")
    return manifest


def _validate_compiler_snapshot(value: Any, *, mode: str, label: str) -> dict[str, Any]:
    snapshot = _mapping(value, label)
    _exact_keys(snapshot, COMPILER_KEYS, label)
    _mapping(snapshot["counters"], f"{label} counters")
    _mapping(snapshot["guard_failures"], f"{label} guard failures")
    _sequence(snapshot["graph_break_reasons"], f"{label} graph-break reasons")
    stats = _mapping(snapshot["counters"].get("'stats'"), f"{label} stats")
    _integer(stats.get("'unique_graphs'"), f"{label} unique graphs", minimum=1)
    _validate_cache_manifest(snapshot["cache_manifest"], f"{label} cache manifest")
    capture_count = 0 if mode == "eager" else 18
    for name in ("cuda_graph_objects", "cuda_graph_contexts"):
        _require(type(snapshot[name]) is int and snapshot[name] == capture_count, f"{label} {name} drifted")
    return snapshot


def _compiler_summary(snapshot: Mapping[str, Any], key: str) -> Any:
    value = snapshot[key]
    if key != "cache_manifest":
        return value
    return {
        "file_count": len(value),
        "total_bytes": sum(row["bytes"] for row in value),
        "manifest_sha256": canonical_sha256(value),
    }


def _compiler_delta(initial: Mapping[str, Any], runtime: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for key in COMPILER_KEYS:
        before = _compiler_summary(initial, key)
        after = _compiler_summary(runtime, key)
        if before != after:
            result[key] = {"before": before, "after": after}
    return result


def _validate_route_artifact(
    value: Any,
    *,
    mode: str,
    historical_sources: Mapping[str, Mapping[str, str]],
    runtime_blobs: Mapping[str, Mapping[str, str]],
) -> tuple[list[str], dict[str, Any]]:
    artifact = _mapping(value, f"{mode} route artifact")
    _exact_keys(artifact, ROUTE_ROOT_KEYS, f"{mode} route artifact")
    _require(artifact["schema"] == ROUTE_SCHEMA and artifact["mode"] == mode, f"{mode} route schema/mode mismatch")
    _require(isinstance(artifact["run_id"], str) and UUID4_HEX.fullmatch(artifact["run_id"]), f"{mode} run ID is not UUID4 hex")
    try:
        generated = datetime.fromisoformat(artifact["generated_at"])
    except (TypeError, ValueError) as error:
        raise EvidenceValidationError(f"{mode} generated_at is not ISO-8601") from error
    _require(generated.tzinfo is not None and generated.utcoffset() is not None, f"{mode} generated_at is not timezone-aware")
    _require(type(artifact["seed"]) is int and artifact["seed"] == SEED, f"{mode} route seed drifted")
    _require(type(artifact["configured_k"]) is int and artifact["configured_k"] == 2, f"{mode} configured K drifted")
    _require(artifact["model"] == artifact["draft_model"] == EXPECTED_MODEL_CONTENT["resolved_path"], f"{mode} model arguments drifted")
    expected_configuration = {
        "max_num_seqs": 4,
        "max_num_batched_tokens": 512,
        "max_model_len": 512,
        "gpu_memory_utilization": 0.5,
        "repetitions": 2,
        "tensor_parallel_size": 1,
        "top_p_backend": "exact",
    }
    _require(artifact["configuration"] == expected_configuration, f"{mode} route configuration drifted")
    _require(type(artifact["configuration"]["gpu_memory_utilization"]) is float, "route utilization is not a JSON float")
    expected_cardinality = 4 if mode == "eager" else 12
    for name in ("registry_cardinality", "visited_cardinality"):
        _require(type(artifact[name]) is int and artifact[name] == expected_cardinality, f"{mode} {name} drifted")
    _integer(artifact["route_pretouch_peak_bytes"], f"{mode} route pretouch peak", minimum=1)
    capture_count = 0 if mode == "eager" else 18
    expected_ledger = {"cuda_graph_objects": capture_count, "cuda_graph_contexts": capture_count}
    _require(artifact["capture_ledger_after_init"] == expected_ledger, f"{mode} constructor capture ledger drifted")
    _require(artifact["capture_ledger_after_runtime"] == expected_ledger, f"{mode} runtime capture ledger drifted")
    _require(artifact["all_draft_intervals_compiler_state_unchanged"] is True, f"{mode} compiler-neutral verdict false")

    initial = _validate_compiler_snapshot(artifact["compiler_state_after_init"], mode=mode, label=f"{mode} initial compiler state")
    runtime = _validate_compiler_snapshot(artifact["compiler_state_after_runtime"], mode=mode, label=f"{mode} runtime compiler state")
    _require(canonical_sha256(initial) == artifact["compiler_state_after_init_sha256"], f"{mode} initial compiler hash mismatch")
    _require(canonical_sha256(runtime) == artifact["compiler_state_after_runtime_sha256"], f"{mode} runtime compiler hash mismatch")
    _require(len(runtime["cache_manifest"]) >= len(initial["cache_manifest"]), f"{mode} compiler cache shrank")
    _require(sum(row["bytes"] for row in runtime["cache_manifest"]) >= sum(row["bytes"] for row in initial["cache_manifest"]), f"{mode} compiler cache bytes shrank")
    _require(artifact["post_init_to_runtime_compiler_delta"] == _compiler_delta(initial, runtime), f"{mode} whole-run compiler delta summary mismatch")

    records = _sequence(artifact["records"], f"{mode} route records")
    _require(len(records) == ROUTE_RECORD_COUNT, f"{mode} must contain 32 route records")
    labels = []
    visited = set()
    for index, (record, spec) in enumerate(zip(records, _route_specs(), strict=True)):
        batch_size, effective_k, repetition, phase = spec
        label = f"{mode} route record {index}"
        record = _mapping(record, label)
        _exact_keys(record, ROUTE_RECORD_KEYS, label)
        route = _route(mode, batch_size, effective_k, phase)
        _require(record["route"] == route, f"{label} route drifted")
        _require(type(record["route"]["batch_bucket"]) is int and type(record["route"]["effective_k"]) is int, f"{label} route integer type drifted")
        _require(type(record["live_batch_size"]) is int and record["live_batch_size"] == batch_size, f"{label} batch size drifted")
        _require(type(record["repetition"]) is int and record["repetition"] == repetition, f"{label} repetition drifted")
        _require(record["phase"] == phase, f"{label} phase drifted")
        catchup = 4 * batch_size if phase == "cold" else 0
        _require(type(record["catchup_tokens"]) is int and record["catchup_tokens"] == catchup, f"{label} catch-up count drifted")
        _require(record["q_shape"] == [batch_size, effective_k, VOCAB_SIZE] and all(type(item) is int for item in record["q_shape"]), f"{label} q shape drifted")
        _require(record["q_stride"] == [VOCAB_SIZE, batch_size * VOCAB_SIZE, 1] and all(type(item) is int for item in record["q_stride"]), f"{label} q stride drifted")
        steps = (record["eager_decode_steps"], record["graph_decode_steps"])
        _require(steps == ((effective_k, 0) if mode == "eager" else (0, effective_k)), f"{label} decode-path accounting drifted")
        target_ids = _sequence(record["target_token_ids"], f"{label} target tokens")
        _require(len(target_ids) == batch_size and all(type(token) is int and 0 <= token < VOCAB_SIZE for token in target_ids), f"{label} target tokens invalid")
        proposal_ids = _sequence(record["proposed_token_ids"], f"{label} proposals")
        _require(len(proposal_ids) == batch_size and all(isinstance(row, list) and len(row) == effective_k and all(type(token) is int and 0 <= token < VOCAB_SIZE for token in row) for row in proposal_ids), f"{label} proposal matrix invalid")
        _require(record["compiler_delta"] == {}, f"{label} compiler delta is nonempty")
        before_sha = _sha(record["compiler_snapshot_before_sha256"], f"{label} compiler before")
        after_sha = _sha(record["compiler_snapshot_after_sha256"], f"{label} compiler after")
        _require(before_sha == after_sha, f"{label} compiler snapshots differ")
        for name in ("rng_neutral", "context_reset", "host_result_cuda_free"):
            _require(record[name] is True, f"{label} {name} is not true")
        marker = (
            f"run={artifact['run_id']},record={index},batch={batch_size},"
            f"bucket={route['batch_bucket']},k={effective_k},catchup={catchup}"
        )
        labels.append(marker)
        visited.add(tuple(sorted(route.items())))
    expected_registry = {
        tuple(sorted(_route(mode, batch, k, phase).items()))
        for batch in ((1, 2, 3, 4) if mode == "graph" else (1,))
        for k in (1, 2)
        for phase in ("cold", "warm")
    }
    _require(visited == expected_registry, f"{mode} route registry coverage incomplete")
    _require(len(set(labels)) == ROUTE_RECORD_COUNT, f"{mode} route markers are not unique")
    _require(artifact["retention_eligible"] is True, f"{mode} route artifact is not retention eligible")
    provenance = _validate_provenance(
        artifact["provenance"],
        runner_path="tests/run_speculative_v3_route_compile.py",
        runtime_extra=("DraftRouteRegistry", "nanovllm/engine/speculative_routes.py"),
        has_models=True,
        route_environment=True,
        required_options={"--mode": mode, "--output": None},
        historical_sources=historical_sources,
        runtime_blobs=runtime_blobs,
    )
    _require(provenance["retention_eligible"] == artifact["retention_eligible"], f"{mode} route retention flag mismatch")
    return labels, artifact


def _validate_route_log(payload: bytes, labels: Sequence[str]) -> dict[str, Any]:
    _require(b"\0" not in payload, "route stderr contains a NUL byte")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise EvidenceValidationError("route stderr is not UTF-8") from error
    active: str | None = None
    next_index = 0
    intervals = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if line.startswith(BEGIN_PREFIX):
            _require(active is None, f"nested route BEGIN at line {line_number}")
            _require(next_index < len(labels), f"extra route BEGIN at line {line_number}")
            active = line[len(BEGIN_PREFIX):]
            _require(active == labels[next_index], f"route BEGIN order/content drifted at interval {next_index}")
            intervals.append({"label": active, "begin_line": line_number})
        elif line.startswith(END_PREFIX):
            _require(active is not None, f"orphan route END at line {line_number}")
            label = line[len(END_PREFIX):]
            _require(label == active, f"route END mismatch at line {line_number}")
            intervals[-1]["end_line"] = line_number
            active = None
            next_index += 1
        else:
            _require(MARKER_FRAGMENT not in line, f"malformed route marker at line {line_number}")
            if active is not None:
                _require(not line.strip(), f"unexpected output inside route interval {next_index}")
    _require(active is None, "route stderr ended inside an interval")
    _require(next_index == len(labels) == ROUTE_RECORD_COUNT, "route stderr interval count drifted")
    return {
        "interval_count": len(intervals),
        "marker_count": len(intervals) * 2,
        "strict_nonnested_pairs": True,
        "record_order_exact": True,
        "draft_interval_output_empty": True,
        "intervals": intervals,
    }


@dataclass(frozen=True)
class _StorageType:
    dtype: str
    itemsize: int


@dataclass(frozen=True)
class _StorageRef:
    dtype: _StorageType
    key: str
    count: int


@dataclass(frozen=True)
class _TensorRef:
    storage: _StorageRef
    offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]


BF16_STORAGE = _StorageType("torch.bfloat16", 2)
FP32_STORAGE = _StorageType("torch.float32", 4)


def _inert_rebuild_tensor_v2(
    storage: _StorageRef,
    offset: int,
    shape: Sequence[int],
    stride: Sequence[int],
    requires_grad: bool,
    backward_hooks: Any,
    *extra: Any,
) -> _TensorRef:
    _require(isinstance(storage, _StorageRef), "sidecar tensor storage reference is invalid")
    _require(type(offset) is int and offset >= 0, "sidecar tensor offset is invalid")
    _require(isinstance(shape, tuple) and all(type(item) is int and item >= 0 for item in shape), "sidecar tensor shape is invalid")
    _require(isinstance(stride, tuple) and all(type(item) is int and item >= 0 for item in stride), "sidecar tensor stride is invalid")
    _require(requires_grad is False, "sidecar tensor unexpectedly requires gradients")
    _require(isinstance(backward_hooks, collections.OrderedDict) and not backward_hooks, "sidecar tensor contains backward hooks")
    _require(not extra, "sidecar tensor rebuild signature drifted")
    return _TensorRef(storage, offset, tuple(shape), tuple(stride))


class _RestrictedTorchMetadataUnpickler(pickle.Unpickler):
    _GLOBALS = {
        ("torch._utils", "_rebuild_tensor_v2"): _inert_rebuild_tensor_v2,
        ("torch", "BFloat16Storage"): BF16_STORAGE,
        ("torch", "FloatStorage"): FP32_STORAGE,
        ("collections", "OrderedDict"): collections.OrderedDict,
    }

    def find_class(self, module: str, name: str) -> Any:
        try:
            return self._GLOBALS[(module, name)]
        except KeyError as error:
            raise EvidenceValidationError(f"disallowed sidecar pickle global: {module}.{name}") from error

    def persistent_load(self, pid: Any) -> _StorageRef:
        _require(isinstance(pid, tuple) and len(pid) == 5, "sidecar persistent storage ID schema drifted")
        tag, storage_type, key, location, count = pid
        _require(tag == "storage", "sidecar persistent ID is not a storage")
        _require(storage_type in (BF16_STORAGE, FP32_STORAGE), "sidecar storage dtype is not BF16/FP32")
        _require(isinstance(key, str) and key.isdecimal(), "sidecar storage key is invalid")
        _require(location == "cpu", "sidecar storage is not on CPU")
        _integer(count, "sidecar storage element count", minimum=1)
        return _StorageRef(storage_type, key, count)


def _scan_pickle(payload: bytes) -> None:
    allowed_globals = {
        "torch._utils _rebuild_tensor_v2",
        "torch BFloat16Storage",
        "torch FloatStorage",
        "collections OrderedDict",
    }
    globals_seen = set()
    stop_count = 0
    try:
        for opcode, argument, _position in pickletools.genops(payload):
            if opcode.name == "GLOBAL":
                _require(argument in allowed_globals, f"disallowed sidecar pickle global: {argument}")
                globals_seen.add(argument)
            _require(
                opcode.name not in {
                    "STACK_GLOBAL", "EXT1", "EXT2", "EXT4", "INST", "OBJ",
                    "NEWOBJ", "NEWOBJ_EX",
                },
                f"disallowed sidecar pickle opcode: {opcode.name}",
            )
            if opcode.name == "STOP":
                stop_count += 1
    except EvidenceValidationError:
        raise
    except (ValueError, pickle.UnpicklingError) as error:
        raise EvidenceValidationError("invalid sidecar pickle bytecode") from error
    _require(stop_count == 1, "sidecar pickle must contain exactly one STOP")
    _require(globals_seen == allowed_globals, "sidecar pickle global registry drifted")


def _tensor_domain_hash(dtype: str, shape: Sequence[int], payload: bytes) -> str:
    metadata = json.dumps(
        {"dtype": dtype, "shape": list(shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(metadata + b"\0" + payload)


def _bf16_argmax(payload: bytes) -> int:
    _require(len(payload) == VOCAB_SIZE * 2, "BF16 storage byte length drifted")
    values = struct.unpack(f"<{VOCAB_SIZE}H", payload)
    best_index = -1
    best_key = -1
    for index, bits in enumerate(values):
        exponent = (bits >> 7) & 0xFF
        _require(exponent != 0xFF, "BF16 logits contain NaN or infinity")
        key = ((~bits) & 0xFFFF) if (bits & 0x8000) else (bits | 0x8000)
        if key > best_key:
            best_key = key
            best_index = index
    return best_index


def _fp32_one_hot_index(payload: bytes) -> int:
    _require(len(payload) == VOCAB_SIZE * 4, "FP32 storage byte length drifted")
    values = struct.unpack(f"<{VOCAB_SIZE}f", payload)
    one_index = -1
    total = 0.0
    for index, value in enumerate(values):
        _require(math.isfinite(value) and value >= 0.0, "probabilities contain a negative/non-finite value")
        _require(value in (0.0, 1.0), "greedy probability storage is not exact one-hot")
        if value == 1.0:
            _require(one_index < 0, "greedy probability storage has multiple ones")
            one_index = index
        total += value
    _require(one_index >= 0 and total == 1.0, "greedy probability storage mass is not exactly one")
    return one_index


def _validate_tensor_ref(tensor: Any, *, dtype: _StorageType, label: str) -> _TensorRef:
    _require(isinstance(tensor, _TensorRef), f"{label} is not an inert tensor reference")
    _require(tensor.storage.dtype == dtype, f"{label} dtype drifted")
    _require(tensor.storage.count == VOCAB_SIZE, f"{label} storage count drifted")
    _require(tensor.offset == 0, f"{label} storage offset drifted")
    _require(tensor.shape == (1, VOCAB_SIZE), f"{label} shape drifted")
    _require(tensor.stride == (VOCAB_SIZE, 1), f"{label} stride drifted")
    return tensor


def _validate_pt_sidecar(
    payload: bytes,
    *,
    mode: str,
    fill: str,
    raw_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, bytes | str]]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload), "r")
    except (zipfile.BadZipFile, OSError) as error:
        raise EvidenceValidationError("cache sidecar is not a valid ZIP archive") from error
    with archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        _require(len(names) == len(set(names)), "cache sidecar ZIP has duplicate names")
        expected_names = {
            "archive/data.pkl", "archive/.format_version", "archive/.storage_alignment",
            "archive/byteorder", "archive/version", "archive/.data/serialization_id",
            *(f"archive/data/{index}" for index in range(18)),
        }
        _require(set(names) == expected_names, "cache sidecar ZIP member registry drifted")
        for info in infos:
            _safe_relative(info.filename, "cache sidecar ZIP member")
            _require(not info.is_dir(), "cache sidecar ZIP contains a directory entry")
            _require(not (info.flag_bits & 0x1), "cache sidecar ZIP contains an encrypted member")
            _require(info.compress_type == zipfile.ZIP_STORED, "cache sidecar ZIP member is unexpectedly compressed")
            mode_bits = (info.external_attr >> 16) & 0xFFFF
            _require(not stat.S_ISLNK(mode_bits), "cache sidecar ZIP contains a symlink")
        _require(archive.testzip() is None, "cache sidecar ZIP CRC failure")
        _require(archive.read("archive/.format_version") == b"1", "cache sidecar format version drifted")
        _require(archive.read("archive/.storage_alignment") == b"64", "cache sidecar storage alignment drifted")
        _require(archive.read("archive/byteorder") == b"little", "cache sidecar byte order drifted")
        _require(archive.read("archive/version") == b"3\n", "cache sidecar serialization version drifted")
        serialization_id = archive.read("archive/.data/serialization_id")
        _require(len(serialization_id) == 40 and serialization_id.isdigit(), "cache sidecar serialization ID drifted")
        pickle_payload = archive.read("archive/data.pkl")
        _require(len(pickle_payload) <= 64 * 1024, "cache sidecar pickle metadata is oversized")
        _scan_pickle(pickle_payload)
        try:
            root = _RestrictedTorchMetadataUnpickler(io.BytesIO(pickle_payload)).load()
        except EvidenceValidationError:
            raise
        except Exception as error:
            raise EvidenceValidationError("cache sidecar metadata cannot be decoded safely") from error
        root = _mapping(root, "cache sidecar metadata root")
        _exact_keys(root, {"schema", "mode", "draft_cache_fill", "records"}, "cache sidecar metadata root")
        _require(root["schema"] == CACHE_SCHEMA and root["mode"] == mode and root["draft_cache_fill"] == fill, "cache sidecar metadata identity drifted")
        records = _sequence(root["records"], "cache sidecar records")
        _require(len(records) == len(CACHE_RECORD_IDS), "cache sidecar record count drifted")
        validated = []
        storage_keys = []
        for index, (raw_tensor_record, raw_json_record, expected_id) in enumerate(
            zip(records, raw_records, CACHE_RECORD_IDS, strict=True)
        ):
            record = _mapping(raw_tensor_record, f"cache sidecar record {index}")
            _exact_keys(record, {"record_id", "logits", "probabilities"}, f"cache sidecar record {index}")
            _require(record["record_id"] == expected_id, f"cache sidecar record {index} ID drifted")
            logits_ref = _validate_tensor_ref(record["logits"], dtype=BF16_STORAGE, label=f"{expected_id} logits")
            probability_ref = _validate_tensor_ref(record["probabilities"], dtype=FP32_STORAGE, label=f"{expected_id} probabilities")
            _require(logits_ref.storage.key == str(2 * index), f"{expected_id} logits storage key drifted")
            _require(probability_ref.storage.key == str(2 * index + 1), f"{expected_id} probability storage key drifted")
            storage_keys.extend((logits_ref.storage.key, probability_ref.storage.key))
            logits = archive.read(f"archive/data/{logits_ref.storage.key}")
            probabilities = archive.read(f"archive/data/{probability_ref.storage.key}")
            logits_argmax = _bf16_argmax(logits)
            probability_argmax = _fp32_one_hot_index(probabilities)
            _require(logits_argmax == probability_argmax, f"{expected_id} probability token is not logits argmax")
            _require(raw_json_record["token_ids"] == [logits_argmax], f"{expected_id} JSON token does not match sidecar")
            _require(_tensor_domain_hash(BF16_STORAGE.dtype, (1, VOCAB_SIZE), logits) == raw_json_record["logits_sha256"], f"{expected_id} logits hash does not bind sidecar bytes")
            _require(_tensor_domain_hash(FP32_STORAGE.dtype, (1, VOCAB_SIZE), probabilities) == raw_json_record["probabilities_sha256"], f"{expected_id} probability hash does not bind sidecar bytes")
            validated.append({"record_id": expected_id, "logits": logits, "probabilities": probabilities})
        _require(storage_keys == [str(index) for index in range(18)], "cache sidecar storage registry drifted")
        return validated


def _nested(value: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = value
    for part in path:
        _require(isinstance(current, dict) and part in current, f"missing cache cycle {'/'.join(path)}")
        current = current[part]
    return current


def _validate_cache_cycle(
    artifact: Mapping[str, Any],
    *,
    mode: str,
    cycle_name: str,
    path: tuple[str, ...],
    positions: list[int],
    catchup: int,
    offset: int,
) -> list[dict[str, Any]]:
    cycle = _mapping(_nested(artifact, path), f"cache cycle {cycle_name}")
    _require(cycle.get("effective_k") == 3 and type(cycle.get("effective_k")) is int, f"{cycle_name} effective K drifted")
    _require(cycle.get("proposal_input_positions") == positions, f"{cycle_name} positions drifted")
    _require(cycle.get("catchup_tokens") == catchup and type(cycle.get("catchup_tokens")) is int, f"{cycle_name} catch-up drifted")
    expected_route = {
        "schema": "draft-discard-v1",
        "execution_mode": "eager_dynamic" if mode == "eager" else "cuda_graph",
        "batch_bucket": 1,
        "effective_k": 3,
        "catchup_family": "paged_eager_dynamic_v1",
        "sampler_envelope": "exact_all_compositions_worst_case_v1",
    }
    _require(cycle.get("route") == expected_route, f"{cycle_name} route drifted")
    expected_steps = (3, 0) if mode == "eager" else (0, 3)
    _require((cycle.get("eager_decode_steps"), cycle.get("graph_decode_steps")) == expected_steps, f"{cycle_name} decode path drifted")
    rng_before = _mapping(cycle.get("rng_before_draft"), f"{cycle_name} RNG before")
    _exact_keys(rng_before, {"cpu", "cuda"}, f"{cycle_name} RNG before")
    for name, digest in rng_before.items():
        _sha(digest, f"{cycle_name} RNG {name}")
    _require(cycle.get("rng_after_draft") == rng_before, f"{cycle_name} draft changed RNG")
    proposals = _sequence(cycle.get("proposed_token_ids"), f"{cycle_name} proposals")
    _require(len(proposals) == 1 and isinstance(proposals[0], list) and len(proposals[0]) == 3 and all(type(token) is int for token in proposals[0]), f"{cycle_name} proposal shape drifted")
    target = _sequence(cycle.get("target_token_ids"), f"{cycle_name} target tokens")
    _require(len(target) == 1 and type(target[0]) is int, f"{cycle_name} target token shape drifted")
    records = _sequence(cycle.get("sampler_records"), f"{cycle_name} sampler records")
    _require(len(records) == 3, f"{cycle_name} must contain three sampler records")
    expected_ids = list(CACHE_RECORD_IDS[offset:offset + 3])
    _require([row.get("record_id") for row in records] == expected_ids, f"{cycle_name} sampler record IDs drifted")
    for step, record in enumerate(records):
        record = _mapping(record, f"{cycle_name} sampler record {step}")
        _require(record.get("token_ids") == [proposals[0][step]], f"{cycle_name} sampler/proposal token mismatch")
        _require(record.get("logits_finite") is True and record.get("probabilities_finite") is True, f"{cycle_name} sampler contains non-finite values")
        _sha(record.get("logits_sha256"), f"{cycle_name} logits SHA-256")
        _sha(record.get("probabilities_sha256"), f"{cycle_name} probability SHA-256")
        _require(record.get("logits_dtype") == "torch.bfloat16" and record.get("probabilities_dtype") == "torch.float32", f"{cycle_name} sampler dtype drifted")
        _require(record.get("logits_shape") == [1, VOCAB_SIZE] and record.get("probabilities_shape") == [1, VOCAB_SIZE], f"{cycle_name} sampler shape drifted")
        sums = record.get("probability_row_sums")
        _require(isinstance(sums, list) and len(sums) == 1 and type(sums[0]) is float and abs(sums[0] - 1.0) <= 1e-6, f"{cycle_name} probability mass drifted")
    target_tables = cycle.get("tables_after_target_schedule")
    reservation_tables = cycle.get("tables_during_reservation")
    handoff_tables = cycle.get("tables_after_handoff")
    for label, tables in (("target", target_tables), ("reservation", reservation_tables), ("handoff", handoff_tables)):
        _require(isinstance(tables, list) and len(tables) == 1 and isinstance(tables[0], list) and tables[0] and all(type(block) is int and block >= 0 for block in tables[0]) and len(set(tables[0])) == len(tables[0]), f"{cycle_name} {label} block table drifted")
    _require(handoff_tables == target_tables, f"{cycle_name} target block table was not restored")
    _require(reservation_tables[0][:len(target_tables[0])] == target_tables[0], f"{cycle_name} reservation table prefix drifted")
    temporary = cycle.get("temporary_blocks")
    _require(type(temporary) is int and temporary == (1 if cycle_name == "boundary" else 0), f"{cycle_name} temporary block count drifted")
    _require(temporary == len(set(reservation_tables[0]) - set(target_tables[0])), f"{cycle_name} temporary block accounting drifted")
    _require(cycle.get("filled_physical_blocks") == sorted(set(reservation_tables[0])), f"{cycle_name} did not poison every reserved block")
    return records


def _validate_cache_artifact(
    value: Any,
    *,
    mode: str,
    fill: str,
    sidecar_payload: bytes,
    sidecar_registry: tuple[int, str],
    historical_sources: Mapping[str, Mapping[str, str]],
    runtime_blobs: Mapping[str, Mapping[str, str]],
) -> tuple[dict[str, Any], list[dict[str, bytes | str]]]:
    artifact = _mapping(value, f"{mode}/{fill} cache artifact")
    _require(artifact.get("schema") == CACHE_SCHEMA and artifact.get("mode") == mode and artifact.get("draft_cache_fill") == fill, f"{mode}/{fill} cache identity drifted")
    _require(artifact.get("seed") == SEED and type(artifact.get("seed")) is int, f"{mode}/{fill} cache seed drifted")
    _require(artifact.get("model") == artifact.get("draft_model") == EXPECTED_MODEL_CONTENT["resolved_path"], f"{mode}/{fill} model arguments drifted")
    _require(
        artifact.get("configuration")
        == {
            "num_speculative_tokens": 3,
            "gpu_memory_utilization": 0.5,
            "max_model_len": 512,
            "max_num_batched_tokens": 512,
            "max_num_seqs": 1,
            "tensor_parallel_size": 1,
            "top_p_backend": "exact",
        },
        f"{mode}/{fill} cache configuration drifted",
    )
    _require(
        artifact.get("workload")
        == {
            "neutral_boundary_warmup": {"length": 255, "salt": 911, "fill": "zero", "proposal_positions": [255, 256, 257], "sampler_records_discarded": 3},
            "boundary_prompt": {"length": 255, "salt": 31, "proposal_positions": [255, 256, 257]},
            "shared_prefix_prompt": {"length": 257, "salt": 73, "proposal_positions": [257, 258, 259]},
            "sampling": {"temperature": 0.0, "top_k": -1, "top_p": 1.0, "ignore_eos": True},
        },
        f"{mode}/{fill} cache workload drifted",
    )
    warmup = _mapping(artifact.get("measurement_warmup"), f"{mode}/{fill} measurement warmup")
    _require(warmup.get("fill") == "zero" and warmup.get("prompt_length") == 255 and warmup.get("prompt_salt") == 911 and warmup.get("proposal_input_positions") == [255, 256, 257] and warmup.get("catchup_tokens") == 255 and warmup.get("temporary_blocks") == 1 and warmup.get("sampler_records_discarded") == 3, f"{mode}/{fill} cache warmup drifted")
    _require(artifact.get("all_logits_finite") is True and artifact.get("all_probabilities_finite") is True, f"{mode}/{fill} cache reports non-finite values")
    cycle_specs = (
        ("boundary", ("boundary",), [255, 256, 257], 255),
        ("shared-prefix/cold", ("shared_prefix", "cold"), [257, 258, 259], 257),
        ("shared-prefix/hit", ("shared_prefix", "shared_prefix"), [257, 258, 259], 257),
    )
    raw_records = []
    for index, (name, path, positions, catchup) in enumerate(cycle_specs):
        raw_records.extend(_validate_cache_cycle(artifact, mode=mode, cycle_name=name, path=path, positions=positions, catchup=catchup, offset=index * 3))
    _require([row["record_id"] for row in raw_records] == list(CACHE_RECORD_IDS), f"{mode}/{fill} cache record order drifted")
    descriptor = _mapping(artifact.get("tensor_artifact"), f"{mode}/{fill} tensor descriptor")
    _exact_keys(descriptor, {"path", "format", "size_bytes", "sha256", "record_count"}, f"{mode}/{fill} tensor descriptor")
    _require(descriptor["path"] == "result.tensors.pt", f"{mode}/{fill} tensor sidecar path drifted")
    _require(descriptor["format"] == "torch-save-weights-only" and descriptor["record_count"] == 9, f"{mode}/{fill} tensor format/count drifted")
    _require((descriptor["size_bytes"], descriptor["sha256"]) == sidecar_registry, f"{mode}/{fill} tensor descriptor does not bind archive sidecar")
    tensors = _validate_pt_sidecar(sidecar_payload, mode=mode, fill=fill, raw_records=raw_records)
    _require(artifact.get("retention_eligible") is True, f"{mode}/{fill} cache artifact is not retention eligible")
    provenance = _validate_provenance(
        artifact.get("provenance"),
        runner_path="tests/run_speculative_v3_cache_neutrality.py",
        runtime_extra=("DraftRouteAdmission", "nanovllm/engine/speculative_routes.py"),
        has_models=True,
        route_environment=False,
        required_options={"--mode": mode, "--draft-cache-fill": fill, "--output": None},
        historical_sources=historical_sources,
        runtime_blobs=runtime_blobs,
    )
    _require(provenance["retention_eligible"] is True, f"{mode}/{fill} cache provenance is not retained")
    return artifact, tensors


def _validate_rng(value: Any, label: str) -> dict[str, str]:
    rng = _mapping(value, label)
    _exact_keys(rng, {"cpu_sha256", "cuda_sha256"}, label)
    _sha(rng["cpu_sha256"], f"{label} CPU SHA-256")
    _sha(rng["cuda_sha256"], f"{label} CUDA SHA-256")
    return rng


def _validate_output_event(value: Any, *, stage: str, seq_id: int) -> dict[str, Any]:
    event = _mapping(value, f"{stage} event")
    _exact_keys(event, {"stage", "seq_id", "token_id", "finished"}, f"{stage} event")
    _require(event["stage"] == stage, f"{stage} event stage drifted")
    _require(type(event["seq_id"]) is int and event["seq_id"] == seq_id, f"{stage} sequence ID drifted")
    _require(type(event["token_id"]) is int and event["token_id"] >= 0, f"{stage} token ID drifted")
    _require(event["finished"] is False, f"{stage} event unexpectedly finished")
    return event


def _validate_output_artifact(
    value: Any,
    *,
    mode: str,
    side: str,
    historical_sources: Mapping[str, Mapping[str, str]],
    runtime_blobs: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    artifact = _mapping(value, f"{mode}/{side} output artifact")
    _require(artifact.get("schema") == OUTPUT_SCHEMA and artifact.get("mode") == mode and artifact.get("side") == side, f"{mode}/{side} output identity drifted")
    _require(artifact.get("seed") == SEED and type(artifact.get("seed")) is int, f"{mode}/{side} output seed drifted")
    _require(artifact.get("configured_k") == 2 and type(artifact.get("configured_k")) is int, f"{mode}/{side} configured K drifted")
    _require(artifact.get("model") == artifact.get("draft_model_argument") == EXPECTED_MODEL_CONTENT["resolved_path"], f"{mode}/{side} model arguments drifted")
    _require(
        artifact.get("configuration")
        == {
            "max_model_len": 512,
            "max_num_batched_tokens": 512,
            "max_num_seqs": 2,
            "gpu_memory_utilization": 0.5,
            "top_p_backend": "exact",
            "tensor_parallel_size": 1,
        },
        f"{mode}/{side} output configuration drifted",
    )
    _require(
        artifact.get("workload")
        == {
            "prompt_token_ids": [[10, 11, 12, 13], [20, 21, 22]],
            "sampling": {
                "temperature": 0.8,
                "top_k": 8,
                "top_p": 0.9,
                "max_tokens": 6,
                "ignore_eos": True,
            },
        },
        f"{mode}/{side} output workload drifted",
    )
    phase_calls = _mapping(artifact.get("draft_phase_calls"), f"{mode}/{side} draft phase calls")
    _exact_keys(phase_calls, set(DRAFT_PHASES), f"{mode}/{side} draft phase calls")
    expected_on_calls = {
        "_construct_draft_model": 1,
        "warmup_draft_model": 1,
        "capture_draft_cudagraph": 0 if mode == "eager" else 2,
        "_pretouch_draft_eager_prefill": 0 if mode == "eager" else 1,
        "_pretouch_draft_routes": 1,
    }
    expected_calls = {name: 0 for name in DRAFT_PHASES} if side == "off" else expected_on_calls
    _require(phase_calls == expected_calls, f"{mode}/{side} draft phase counts drifted")
    resources = _mapping(artifact.get("live_draft_resource_attributes"), f"{mode}/{side} draft resources")
    _exact_keys(resources, set(DRAFT_RESOURCE_ATTRIBUTES), f"{mode}/{side} draft resources")
    _require(all(type(item) is bool and item is (side == "on") for item in resources.values()), f"{mode}/{side} draft resource state drifted")
    expected_owned = [] if side == "off" else list(DRAFT_OWNED_ATTRIBUTES)
    _require(artifact.get("draft_owned_instance_attributes") == expected_owned, f"{mode}/{side} draft-owned attribute registry drifted")

    calls = _sequence(artifact.get("runtime_draft_calls"), f"{mode}/{side} runtime draft calls")
    if side == "off":
        _require(calls == [], f"{mode}/off unexpectedly ran draft work")
    else:
        _require(len(calls) == 2, f"{mode}/on must contain two draft intervals")
        for index, raw_call in enumerate(calls):
            call = _mapping(raw_call, f"{mode}/on draft call {index}")
            expected_route = {
                "schema": "draft-discard-v1",
                "execution_mode": "eager_dynamic" if mode == "eager" else "cuda_graph",
                "batch_bucket": 2,
                "effective_k": 2,
                "catchup_family": "paged_eager_dynamic_v1" if index == 0 else "none",
                "sampler_envelope": "exact_all_compositions_worst_case_v1",
            }
            _require(call.get("route") == expected_route, f"{mode}/on call {index} route drifted")
            _require(call.get("catchup_tokens") == (7 if index == 0 else 0), f"{mode}/on call {index} catch-up drifted")
            proposals = call.get("proposal_token_ids")
            _require(isinstance(proposals, list) and len(proposals) == 2 and all(isinstance(row, list) and len(row) == 2 and all(type(token) is int and token >= 0 for token in row) for row in proposals), f"{mode}/on call {index} proposal matrix drifted")
            expected_steps = (2, 0) if mode == "eager" else (0, 2)
            _require((call.get("eager_decode_steps"), call.get("graph_decode_steps")) == expected_steps, f"{mode}/on call {index} decode path drifted")
            before = _validate_rng(call.get("rng_before"), f"{mode}/on call {index} RNG before")
            after = _validate_rng(call.get("rng_after"), f"{mode}/on call {index} RNG after")
            _require(before == after and call.get("rng_neutral") is True, f"{mode}/on call {index} changed target RNG")

    snapshots = _mapping(artifact.get("rng_snapshots"), f"{mode}/{side} RNG endpoints")
    expected_rng_names = {"after_init", "after_prefill", "after_first_target_decode", "after_repeated_target_decode"}
    _exact_keys(snapshots, expected_rng_names, f"{mode}/{side} RNG endpoints")
    for name, snapshot in snapshots.items():
        _validate_rng(snapshot, f"{mode}/{side} {name}")
    steps = _sequence(artifact.get("steps"), f"{mode}/{side} target steps")
    stage_specs = (
        ("prefill", 7, 0),
        ("first_target_decode", 0, 2),
        ("repeated_target_decode", 0, 2),
    )
    _require(len(steps) == len(stage_specs), f"{mode}/{side} target step count drifted")
    flattened = []
    for step, (stage, prefill_tokens, decode_tokens) in zip(steps, stage_specs, strict=True):
        step = _mapping(step, f"{mode}/{side} {stage} step")
        _require(step.get("stage") == stage and step.get("num_prefill_tokens") == prefill_tokens and step.get("num_decode_tokens") == decode_tokens, f"{mode}/{side} {stage} accounting drifted")
        events = _sequence(step.get("events"), f"{mode}/{side} {stage} events")
        _require(len(events) == 2, f"{mode}/{side} {stage} event count drifted")
        for event, seq_id in zip(events, (1, 2), strict=True):
            flattened.append(_validate_output_event(event, stage=stage, seq_id=seq_id))
    _require(artifact.get("authoritative_target_events") == flattened, f"{mode}/{side} authoritative event ledger drifted")
    by_seq = _mapping(artifact.get("target_token_ids_by_seq"), f"{mode}/{side} per-sequence tokens")
    _exact_keys(by_seq, {"1", "2"}, f"{mode}/{side} per-sequence tokens")
    for seq_id, tokens in by_seq.items():
        expected_tokens = [event["token_id"] for event in flattened if event["seq_id"] == int(seq_id)]
        _require(tokens == expected_tokens, f"{mode}/{side} sequence {seq_id} target tokens drifted")
    _require(artifact.get("retention_eligible") is True, f"{mode}/{side} output artifact is not retention eligible")
    provenance = _validate_provenance(
        artifact.get("provenance"),
        runner_path="tests/run_speculative_v3_output_control.py",
        runtime_extra=("DraftRouteRegistry", "nanovllm/engine/speculative_routes.py"),
        has_models=True,
        route_environment=False,
        required_options={"--mode": mode, "--side": side, "--output": None},
        historical_sources=historical_sources,
        runtime_blobs=runtime_blobs,
    )
    _require(provenance["retention_eligible"] is True, f"{mode}/{side} output provenance is not retained")
    return artifact


def _validate_descriptor(
    value: Any,
    *,
    label: str,
    original_path: str,
    registry: tuple[int, str],
) -> dict[str, Any]:
    descriptor = _mapping(value, label)
    _exact_keys(descriptor, {"path", "size_bytes", "sha256"}, label)
    _require(descriptor["path"] == original_path, f"{label} original path mismatch")
    _require((descriptor["size_bytes"], descriptor["sha256"]) == registry, f"{label} does not bind archive payload")
    return descriptor


def _validate_cache_comparison(
    value: Any,
    *,
    mode: str,
    zero_tensors: Sequence[Mapping[str, Any]],
    nan_tensors: Sequence[Mapping[str, Any]],
    historical_sources: Mapping[str, Mapping[str, str]],
    runtime_blobs: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    comparison = _mapping(value, f"{mode} cache comparison")
    _require(comparison.get("schema") == CACHE_COMPARISON_SCHEMA and comparison.get("mode") == mode, f"{mode} cache comparison identity drifted")
    _require(comparison.get("verdict") == "pass" and comparison.get("retention_eligible") is True, f"{mode} cache comparison did not pass retention")
    _require(comparison.get("exact_cache_fill_independence") is True and comparison.get("host_oracles_equal") is True, f"{mode} cache comparison verdict flags drifted")
    _require(comparison.get("record_count") == 9 and comparison.get("record_ids") == list(CACHE_RECORD_IDS), f"{mode} cache comparison record registry drifted")
    for fill in ("zero", "nan"):
        descriptor = _mapping(comparison.get(f"{fill}_artifact"), f"{mode} {fill} comparison descriptor")
        _exact_keys(descriptor, {"path", "size_bytes", "sha256", "tensor_sidecar"}, f"{mode} {fill} comparison descriptor")
        json_relative = f"raw/cache-{mode}-{fill}.json"
        pt_relative = f"raw/cache-{mode}-{fill}.tensors.pt"
        _require(descriptor["path"] == f"cache-{mode}-{fill}/result.json", f"{mode}/{fill} original JSON path mismatch")
        _require((descriptor["size_bytes"], descriptor["sha256"]) == TRUSTED_PAYLOADS[json_relative], f"{mode}/{fill} comparison does not bind JSON")
        sidecar = _mapping(descriptor["tensor_sidecar"], f"{mode}/{fill} comparison tensor descriptor")
        _exact_keys(sidecar, {"path", "size_bytes", "sha256", "record_count"}, f"{mode}/{fill} comparison tensor descriptor")
        _require(sidecar["path"] == f"cache-{mode}-{fill}/result.tensors.pt" and sidecar["record_count"] == 9, f"{mode}/{fill} comparison tensor path/count drifted")
        _require((sidecar["size_bytes"], sidecar["sha256"]) == TRUSTED_PAYLOADS[pt_relative], f"{mode}/{fill} comparison does not bind tensor sidecar")
    tensor_comparisons = _sequence(comparison.get("tensor_comparisons"), f"{mode} tensor comparisons")
    _require(len(tensor_comparisons) == 9, f"{mode} tensor comparison count drifted")
    for expected_id, raw, zero, nan in zip(CACHE_RECORD_IDS, tensor_comparisons, zero_tensors, nan_tensors, strict=True):
        row = _mapping(raw, f"{mode} tensor comparison {expected_id}")
        _require(row == {"record_id": expected_id, "shape": [1, VOCAB_SIZE], "logits_bitwise_equal": True, "probabilities_bitwise_equal": True, "max_abs_difference": 0.0}, f"{mode} tensor comparison {expected_id} drifted")
        _require(zero["record_id"] == nan["record_id"] == expected_id, f"{mode} decoded sidecar IDs drifted")
        _require(zero["logits"] == nan["logits"], f"{mode}/{expected_id} zero/NaN logits are not bitwise equal")
        _require(zero["probabilities"] == nan["probabilities"], f"{mode}/{expected_id} zero/NaN probabilities are not bitwise equal")
    _validate_provenance(
        comparison.get("comparator_provenance"),
        runner_path="tests/compare_speculative_v3_cache_neutrality.py",
        runtime_extra=None,
        has_models=False,
        route_environment=False,
        required_options={"--output": None},
        historical_sources=historical_sources,
        runtime_blobs=runtime_blobs,
    )
    producer = _mapping(comparison.get("producer_provenance_identity"), f"{mode} cache producer provenance")
    _exact_keys(producer, {"pair_identity", "zero", "nan"}, f"{mode} cache producer provenance")
    for name in ("zero", "nan"):
        _require(producer[name].get("producer_commit") == PRODUCER_COMMIT, f"{mode} cache comparison {name} producer commit mismatch")
        _require(producer[name].get("implementation", {}).get("commit") == IMPLEMENTATION_COMMIT, f"{mode} cache comparison {name} implementation mismatch")
    return comparison


def _validate_output_comparison(
    value: Any,
    *,
    mode: str,
    off: Mapping[str, Any],
    on: Mapping[str, Any],
    historical_sources: Mapping[str, Mapping[str, str]],
    runtime_blobs: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    comparison = _mapping(value, f"{mode} output comparison")
    _require(comparison.get("schema") == OUTPUT_COMPARISON_SCHEMA and comparison.get("mode") == mode and comparison.get("seed") == SEED, f"{mode} output comparison identity drifted")
    _require(comparison.get("verdict") == "pass" and comparison.get("retention_eligible") is True, f"{mode} output comparison did not pass retention")
    for name in ("off_draft_phase_calls_zero", "off_draft_resources_absent", "on_cold_and_warm_routes_exercised", "target_events_exact", "target_tokens_exact", "rng_endpoints_exact"):
        _require(comparison.get(name) is True, f"{mode} output comparison {name} is false")
    _require(comparison.get("on_real_v3_intervals") == 2 and type(comparison.get("on_real_v3_intervals")) is int, f"{mode} output comparison interval count drifted")
    _validate_descriptor(comparison.get("off_artifact"), label=f"{mode} off descriptor", original_path=f"output-{mode}-off/result.json", registry=TRUSTED_PAYLOADS[f"raw/output-{mode}-off.json"])
    _validate_descriptor(comparison.get("on_artifact"), label=f"{mode} on descriptor", original_path=f"output-{mode}-on/result.json", registry=TRUSTED_PAYLOADS[f"raw/output-{mode}-on.json"])
    for field in (
        "mode", "seed", "model", "draft_model_argument", "configured_k",
        "configuration", "workload", "rng_snapshots", "steps",
        "authoritative_target_events", "target_token_ids_by_seq",
    ):
        _require(off[field] == on[field], f"{mode} off/on field mismatch: {field}")
    _validate_provenance(
        comparison.get("comparator_provenance"),
        runner_path="tests/compare_speculative_v3_output_control.py",
        runtime_extra=None,
        has_models=False,
        route_environment=False,
        required_options={"--output": None},
        historical_sources=historical_sources,
        runtime_blobs=runtime_blobs,
    )
    producer = _mapping(comparison.get("producer_provenance"), f"{mode} output producer provenance")
    _exact_keys(producer, {"pair_identity", "off", "on"}, f"{mode} output producer provenance")
    for name in ("off", "on"):
        _require(producer[name].get("producer_commit") == PRODUCER_COMMIT, f"{mode} output comparison {name} producer commit mismatch")
        _require(producer[name].get("implementation", {}).get("commit") == IMPLEMENTATION_COMMIT, f"{mode} output comparison {name} implementation mismatch")
    return comparison


def _validate_route_validation(
    value: Any,
    *,
    route_artifacts: Mapping[str, Mapping[str, Any]],
    route_logs: Mapping[str, Mapping[str, Any]],
    historical_sources: Mapping[str, Mapping[str, str]],
    runtime_blobs: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    validation = _mapping(value, "route validation")
    expected_keys = {
        "schema", "verdict", "retention_eligible", "record_count_per_mode",
        "route_registry_cardinality", "artifact_descriptors", "log_descriptors",
        "log_validations", "producer_provenance", "validator_provenance",
        "all_routes_complete", "all_draft_intervals_compiler_state_unchanged",
        "all_stderr_intervals_strict_and_empty", "capture_ledger_stable",
    }
    _exact_keys(validation, expected_keys, "route validation")
    _require(validation["schema"] == ROUTE_VALIDATION_SCHEMA and validation["verdict"] == "pass", "route validation did not pass")
    _require(validation["retention_eligible"] is True, "route validation is not retention eligible")
    _require(validation["record_count_per_mode"] == 32 and type(validation["record_count_per_mode"]) is int, "route validation record count drifted")
    _require(validation["route_registry_cardinality"] == {"eager": 4, "graph": 12}, "route validation cardinality drifted")
    for name in (
        "all_routes_complete", "all_draft_intervals_compiler_state_unchanged",
        "all_stderr_intervals_strict_and_empty", "capture_ledger_stable",
    ):
        _require(validation[name] is True, f"route validation {name} is false")
    original_json_paths = {
        "eager": "route-eager/result.json",
        "graph": "route-graph-retry/result.json",
    }
    original_log_paths = {
        "eager": "route-eager/stderr.log",
        "graph": "route-graph-retry/stderr.log",
    }
    artifact_descriptors = _mapping(validation["artifact_descriptors"], "route artifact descriptors")
    log_descriptors = _mapping(validation["log_descriptors"], "route log descriptors")
    log_validations = _mapping(validation["log_validations"], "route log validations")
    _exact_keys(artifact_descriptors, {"eager", "graph"}, "route artifact descriptors")
    _exact_keys(log_descriptors, {"eager", "graph"}, "route log descriptors")
    _exact_keys(log_validations, {"eager", "graph"}, "route log validations")
    for mode in ("eager", "graph"):
        _validate_descriptor(
            artifact_descriptors[mode],
            label=f"{mode} route validation JSON descriptor",
            original_path=original_json_paths[mode],
            registry=TRUSTED_PAYLOADS[f"raw/route-{mode}.json"],
        )
        _validate_descriptor(
            log_descriptors[mode],
            label=f"{mode} route validation log descriptor",
            original_path=original_log_paths[mode],
            registry=TRUSTED_PAYLOADS[f"raw/route-{mode}.log"],
        )
        _require(log_validations[mode] == route_logs[mode], f"{mode} retained log validation does not match independent replay")
    _require(route_artifacts["eager"]["run_id"] != route_artifacts["graph"]["run_id"], "eager/graph route runs reused a run ID")
    producer = _mapping(validation["producer_provenance"], "route validation producer provenance")
    _exact_keys(producer, {"pair_identity", "eager", "graph"}, "route validation producer provenance")
    for mode in ("eager", "graph"):
        _require(producer[mode].get("producer_commit") == PRODUCER_COMMIT, f"route validation {mode} producer commit mismatch")
        _require(producer[mode].get("implementation", {}).get("commit") == IMPLEMENTATION_COMMIT, f"route validation {mode} implementation mismatch")
    _validate_provenance(
        validation["validator_provenance"],
        runner_path="tests/validate_speculative_v3_route_compile.py",
        runtime_extra=None,
        has_models=False,
        route_environment=False,
        required_options={"--output": None},
        historical_sources=historical_sources,
        runtime_blobs=runtime_blobs,
    )
    return validation


def _cache_host_oracle(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key not in {
            "generated_at", "draft_cache_fill", "tensor_artifact", "provenance",
            "retention_eligible",
        }
    }


def _validate_manifest(root: Path, payload: bytes) -> dict[str, Any]:
    manifest = load_strict_json_bytes(payload, "manifest.json")
    expected = expected_manifest()
    _require(manifest == expected, "manifest content differs from the immutable V3 release schema")
    rows = _sequence(manifest["artifacts"], "manifest artifacts")
    _require([row["path"] for row in rows] == sorted(TRUSTED_PAYLOADS), "manifest artifact order drifted")
    _require(len({row["path"] for row in rows}) == len(rows), "manifest has duplicate artifact paths")
    _require(root.name == manifest["archive_id"], "manifest archive ID differs from directory name")
    return manifest


def _validate_readme(payload: bytes) -> None:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise EvidenceValidationError("README.md is not UTF-8") from error
    for required in (
        "# Speculative V3 draft-discard A100 evidence",
        IMPLEMENTATION_COMMIT,
        PRODUCER_COMMIT,
        "## Claim boundary",
        "It does **not** certify target verification",
        "Aggregate compiler state outside",
        "is not claimed unchanged.",
        "standard-library validator",
    ):
        _require(required in text, f"README.md omitted registered statement: {required}")


def validate_archive(archive: str | Path) -> dict[str, Any]:
    _require(TRUSTED_README is not None and TRUSTED_MANIFEST is not None, "V3 metadata registry is not finalized")
    root = _archive_root(archive)
    expected_files = set(TRUSTED_PAYLOADS) | {"README.md", "manifest.json"}
    actual_files = _enumerate_archive_files(root)
    _require(actual_files == expected_files, "archive file set differs from the immutable registry")

    readme_payload = _read_registered(root, "README.md", TRUSTED_README)
    manifest_payload = _read_registered(root, "manifest.json", TRUSTED_MANIFEST)
    _validate_readme(readme_payload)
    manifest = _validate_manifest(root, manifest_payload)
    payloads = {
        relative: _read_registered(root, relative, registered)
        for relative, registered in TRUSTED_PAYLOADS.items()
    }

    repo_root = Path(__file__).resolve().parents[2]
    historical_sources = _validate_historical_sources(repo_root)
    runtime_blobs = _historical_runtime_blobs(repo_root)

    json_documents = {
        relative: load_strict_json_bytes(payload, relative)
        for relative, payload in payloads.items()
        if relative.endswith(".json")
    }

    route_artifacts = {}
    route_logs = {}
    for mode in ("eager", "graph"):
        labels, route_artifacts[mode] = _validate_route_artifact(
            json_documents[f"raw/route-{mode}.json"],
            mode=mode,
            historical_sources=historical_sources,
            runtime_blobs=runtime_blobs,
        )
        route_logs[mode] = _validate_route_log(
            payloads[f"raw/route-{mode}.log"], labels
        )
    _validate_route_validation(
        json_documents["validations/route.json"],
        route_artifacts=route_artifacts,
        route_logs=route_logs,
        historical_sources=historical_sources,
        runtime_blobs=runtime_blobs,
    )

    cache_artifacts: dict[str, dict[str, dict[str, Any]]] = {}
    cache_tensors: dict[str, dict[str, list[dict[str, bytes | str]]]] = {}
    for mode in ("eager", "graph"):
        cache_artifacts[mode] = {}
        cache_tensors[mode] = {}
        for fill in ("zero", "nan"):
            json_relative = f"raw/cache-{mode}-{fill}.json"
            pt_relative = f"raw/cache-{mode}-{fill}.tensors.pt"
            artifact, tensors = _validate_cache_artifact(
                json_documents[json_relative],
                mode=mode,
                fill=fill,
                sidecar_payload=payloads[pt_relative],
                sidecar_registry=TRUSTED_PAYLOADS[pt_relative],
                historical_sources=historical_sources,
                runtime_blobs=runtime_blobs,
            )
            cache_artifacts[mode][fill] = artifact
            cache_tensors[mode][fill] = tensors
        _require(
            _cache_host_oracle(cache_artifacts[mode]["zero"])
            == _cache_host_oracle(cache_artifacts[mode]["nan"]),
            f"{mode} zero/NaN cache host oracle differs",
        )
        _validate_cache_comparison(
            json_documents[f"comparisons/cache-{mode}.json"],
            mode=mode,
            zero_tensors=cache_tensors[mode]["zero"],
            nan_tensors=cache_tensors[mode]["nan"],
            historical_sources=historical_sources,
            runtime_blobs=runtime_blobs,
        )

    output_artifacts: dict[str, dict[str, dict[str, Any]]] = {}
    for mode in ("eager", "graph"):
        output_artifacts[mode] = {}
        for side in ("off", "on"):
            output_artifacts[mode][side] = _validate_output_artifact(
                json_documents[f"raw/output-{mode}-{side}.json"],
                mode=mode,
                side=side,
                historical_sources=historical_sources,
                runtime_blobs=runtime_blobs,
            )
        _validate_output_comparison(
            json_documents[f"comparisons/output-{mode}.json"],
            mode=mode,
            off=output_artifacts[mode]["off"],
            on=output_artifacts[mode]["on"],
            historical_sources=historical_sources,
            runtime_blobs=runtime_blobs,
        )

    return {
        "kind": ARCHIVE_KIND,
        "archive_id": ARCHIVE_ID,
        "producer_commit": PRODUCER_COMMIT,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "artifact_count": len(TRUSTED_PAYLOADS),
        "route_records": {"eager": 32, "graph": 32},
        "route_intervals": {"eager": 32, "graph": 32},
        "cache_records": {"eager": 9, "graph": 9},
        "output_control_modes": ["eager", "graph"],
        "model_locality_required": False,
        "cuda_required": False,
        "claim_boundary": manifest["claim_boundary"],
    }


def _print_registry(archive: str | Path) -> None:
    root = _archive_root(archive)
    for name in ("README.md", "manifest.json"):
        stream, info = _open_confined_regular(root, name)
        try:
            payload = stream.read()
        finally:
            stream.close()
        print(f"{name}: ({info.st_size}, {_sha256_bytes(payload)!r})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate retained speculative-V3 evidence without CUDA, Torch, or a model."
    )
    parser.add_argument("archive", nargs="+", help="archive directory or manifest.json")
    parser.add_argument(
        "--print-registry",
        action="store_true",
        help="print README/manifest size and SHA only; does not validate",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.print_registry:
        for archive in args.archive:
            _print_registry(archive)
        return
    reports = [validate_archive(archive) for archive in args.archive]
    print(json.dumps(reports, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
