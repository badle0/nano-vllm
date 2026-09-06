"""Shared provenance primitives for retained speculative-V3 GPU evidence.

The V3 implementation is frozen at :data:`IMPLEMENTATION_COMMIT`.  Evidence
producers may evolve in later, harness-only commits, but a retainable run must
prove that the committed ``nanovllm`` tree still equals the implementation
tree.  These helpers deliberately keep certification machinery out of the
runtime package.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import platform
import re
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch


IMPLEMENTATION_COMMIT = "7fec9993d5e4e0e06fec22e3973dfc203fdbd2d8"
IMPLEMENTATION_TREE = "5820fb685d76b549fe21117baa31b3b32ebae14b"
IMPLEMENTATION_PARENT = "e252086ee625ea8aefdb63f3affc093380f40064"
IMPLEMENTATION_NANOVLLM_TREE = "52398af379f767708a0b804646f4b490fa8323ad"
FULL_SHA_RE = re.compile(r"[0-9a-f]{40}")
MODEL_METADATA_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "model.safetensors.index.json",
)
SELECTED_ENVIRONMENT = (
    "CUDA_VISIBLE_DEVICES",
    "PYTHONPATH",
    "PYTHONDONTWRITEBYTECODE",
    "TORCH_LOGS",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE",
    "TORCHINDUCTOR_AUTOGRAD_CACHE",
    "TORCHINDUCTOR_AUTOGRAD_REMOTE_CACHE",
    "TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE",
    "TORCHINDUCTOR_BUNDLED_AUTOTUNE_REMOTE_CACHE",
    "TORCH_DYNAMO_AUTOMATIC_DYNAMIC_LOCAL_PGO",
    "TORCH_DYNAMO_AUTOMATIC_DYNAMIC_REMOTE_PGO",
    "TORCH_COMPILE_DISABLE",
    "TORCHDYNAMO_DISABLE",
    "TORCHDYNAMO_SUPPRESS_ERRORS",
)
COMPILER_CACHE_ENVIRONMENT = (
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
)
_GIT_ENVIRONMENT_TO_REMOVE = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_NAMESPACE",
    "GIT_SHALLOW_FILE",
)


def require(condition: Any, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in tuple(environment):
        if name in _GIT_ENVIRONMENT_TO_REMOVE or name.startswith("GIT_CONFIG"):
            environment.pop(name, None)
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    environment["GIT_CONFIG_NOSYSTEM"] = "1"
    environment["GIT_CONFIG_GLOBAL"] = os.devnull
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    return environment


def git_run(
    repo_root: Path,
    *args: str,
    check: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        (
            "git",
            "--no-replace-objects",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            *args,
        ),
        cwd=repo_root,
        env=_git_environment(),
        check=check,
        capture_output=True,
        text=text,
    )


def git_output(repo_root: Path, *args: str, text: bool = True):
    return git_run(repo_root, *args, text=text).stdout


def source_tree_sha256(repo_root: Path) -> str:
    """Hash every tracked and untracked, non-ignored source path."""

    raw_paths = git_output(
        repo_root,
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
        text=False,
    )
    digest = hashlib.sha256()
    for raw_path in sorted(filter(None, raw_paths.split(b"\0"))):
        path = repo_root / os.fsdecode(raw_path)
        payload = (
            os.readlink(path).encode("utf-8", "surrogateescape")
            if path.is_symlink()
            else path.read_bytes()
        )
        digest.update(raw_path)
        digest.update(b"\0")
        digest.update(str(len(payload)).encode("ascii"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(payload).digest())
    return digest.hexdigest()


def git_snapshot(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    top_level = Path(
        git_output(repo_root, "rev-parse", "--show-toplevel").strip()
    ).resolve()
    require(top_level == repo_root, "evidence runner is not rooted at its repository")
    branch_result = git_run(
        repo_root,
        "symbolic-ref",
        "--quiet",
        "--short",
        "HEAD",
        check=False,
    )
    require(
        branch_result.returncode in (0, 1),
        f"git could not resolve branch state: {branch_result.stderr.strip()}",
    )
    status_raw = git_output(
        repo_root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        text=False,
    )
    status = [
        item.decode("utf-8", "surrogateescape")
        for item in status_raw.split(b"\0")
        if item
    ]
    branch = branch_result.stdout.strip() if branch_result.returncode == 0 else None
    return {
        "repository": str(repo_root),
        "head": git_output(repo_root, "rev-parse", "HEAD").strip(),
        "tree": git_output(repo_root, "rev-parse", "HEAD^{tree}").strip(),
        "nanovllm_tree": git_output(repo_root, "rev-parse", "HEAD:nanovllm").strip(),
        "branch": branch,
        "detached": branch is None,
        "clean": not bool(status),
        "status_porcelain_v1": status,
        "status_sha256": sha256_bytes(status_raw),
        "source_tree_sha256": source_tree_sha256(repo_root),
    }


def validate_implementation_binding(repo_root: Path, snapshot: Mapping[str, Any]) -> dict:
    require(
        git_output(repo_root, "cat-file", "-t", IMPLEMENTATION_COMMIT).strip()
        == "commit",
        "registered V3 implementation object is unavailable",
    )
    observed_tree = git_output(
        repo_root, "rev-parse", f"{IMPLEMENTATION_COMMIT}^{{tree}}"
    ).strip()
    observed_parent = git_output(
        repo_root, "rev-parse", f"{IMPLEMENTATION_COMMIT}^"
    ).strip()
    observed_nanovllm_tree = git_output(
        repo_root, "rev-parse", f"{IMPLEMENTATION_COMMIT}:nanovllm"
    ).strip()
    require(observed_tree == IMPLEMENTATION_TREE, "implementation tree identity drifted")
    require(observed_parent == IMPLEMENTATION_PARENT, "implementation parent drifted")
    require(
        observed_nanovllm_tree == IMPLEMENTATION_NANOVLLM_TREE,
        "implementation nanovllm tree identity drifted",
    )
    require(
        snapshot["nanovllm_tree"] == IMPLEMENTATION_NANOVLLM_TREE,
        "producer commit changes the registered V3 runtime tree",
    )
    return {
        "commit": IMPLEMENTATION_COMMIT,
        "tree": IMPLEMENTATION_TREE,
        "parent": IMPLEMENTATION_PARENT,
        "nanovllm_tree": IMPLEMENTATION_NANOVLLM_TREE,
    }


def _head_blob(repo_root: Path, relative_path: str) -> str | None:
    result = git_run(
        repo_root,
        "rev-parse",
        "--verify",
        f"HEAD:{relative_path}",
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def source_file_identity(path: Path, repo_root: Path) -> dict[str, Any]:
    path = path.resolve()
    repo_root = repo_root.resolve()
    require(path.is_file(), f"source file is missing: {path}")
    require(path == repo_root or repo_root in path.parents, f"source is outside repository: {path}")
    relative = path.relative_to(repo_root).as_posix()
    blob = _head_blob(repo_root, relative)
    actual_sha256 = sha256_file(path)
    committed_sha256 = None
    if blob is not None:
        committed = git_output(repo_root, "cat-file", "blob", blob, text=False)
        committed_sha256 = sha256_bytes(committed)
    return {
        "path": relative,
        "sha256": actual_sha256,
        "head_blob": blob,
        "head_blob_sha256": committed_sha256,
        "matches_head": committed_sha256 == actual_sha256,
    }


def runtime_import_identities(
    repo_root: Path,
    expected: Mapping[str, tuple[Any, str]],
) -> dict[str, dict[str, Any]]:
    identities = {}
    for name, (value, expected_relative) in expected.items():
        source = getattr(value, "__file__", None) or inspect.getsourcefile(value)
        require(source is not None, f"cannot resolve runtime import origin for {name}")
        path = Path(source).resolve()
        identity = source_file_identity(path, repo_root)
        require(
            identity["path"] == expected_relative,
            f"runtime import {name} came from {identity['path']}, expected {expected_relative}",
        )
        identities[name] = identity
    return identities


def model_snapshot(model_argument: str) -> tuple[Path, dict[str, Any]]:
    model_path = Path(model_argument).expanduser().resolve()
    require(model_path.is_dir(), f"model path is not a directory: {model_path}")
    require((model_path / "config.json").is_file(), "model config.json is missing")
    metadata = {}
    for name in MODEL_METADATA_FILES:
        path = model_path / name
        if path.is_file():
            metadata[name] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    weight_paths = sorted(path for path in model_path.glob("*.safetensors") if path.is_file())
    require(weight_paths, f"model has no safetensors weights: {model_path}")
    return model_path, {
        "argument": model_argument,
        "resolved_path": str(model_path),
        "metadata_files": metadata,
        "weight_files": [
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in weight_paths
        ],
        "total_weight_bytes": sum(path.stat().st_size for path in weight_paths),
    }


def model_content_identity(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "resolved_path": snapshot["resolved_path"],
        "metadata_files": snapshot["metadata_files"],
        "weight_files": snapshot["weight_files"],
        "total_weight_bytes": snapshot["total_weight_bytes"],
    }


def path_is_within(path: Path, directory: Path) -> bool:
    return path == directory or directory in path.parents


def compiler_cache_roots_from_environment(
    *, required: bool
) -> tuple[Path, ...]:
    """Return canonical, non-symlinked compiler roots in fixed env order."""

    values = [os.environ.get(name) for name in COMPILER_CACHE_ENVIRONMENT]
    if required:
        require(
            all(values),
            "retained producer requires explicit Inductor and Triton cache roots",
        )
    roots = []
    for name, value in zip(COMPILER_CACHE_ENVIRONMENT, values, strict=True):
        if not value:
            continue
        root = resolve_output_path(Path(value))
        os.environ[name] = str(root)
        roots.append(root)
    require(len(set(roots)) == len(roots), "compiler cache roots must be distinct")
    for index, root in enumerate(roots):
        for other in roots[index + 1 :]:
            require(
                not path_is_within(root, other)
                and not path_is_within(other, root),
                "compiler cache roots must not be nested",
            )
    return tuple(roots)


def validate_compiler_cache_root_isolation(
    roots: Iterable[Path],
    *,
    repo_root: Path,
    model_roots: Iterable[Path] = (),
) -> tuple[Path, ...]:
    roots = tuple(roots)
    require(len(set(roots)) == len(roots), "compiler cache roots must be distinct")
    for index, root in enumerate(roots):
        require(root.is_absolute(), "compiler cache roots must be absolute")
        require(
            not path_is_within(root, repo_root),
            "compiler cache root must be outside the source checkout",
        )
        require(
            not any(path_is_within(root, model) for model in model_roots),
            "compiler cache root must be outside model directories",
        )
        for other in roots[index + 1 :]:
            require(
                not path_is_within(root, other)
                and not path_is_within(other, root),
                "compiler cache roots must not be nested",
            )
    return roots


def require_independent_compiler_cache_roots(
    records: Iterable[Mapping[str, Any]],
) -> list[tuple[Path, ...]]:
    """Require every fresh producer record to own disjoint cache subtrees."""

    groups = []
    for record in records:
        selected = record["environment"]["selected_environment"]
        groups.append(
            tuple(Path(selected[name]) for name in COMPILER_CACHE_ENVIRONMENT)
        )
    for index, roots in enumerate(groups):
        for other_roots in groups[index + 1 :]:
            for root in roots:
                for other in other_roots:
                    require(
                        not path_is_within(root, other)
                        and not path_is_within(other, root),
                        "paired producers reused or nested compiler cache roots",
                    )
    return groups


def resolve_output_path(path: Path) -> Path:
    """Resolve an output path while rejecting existing symlink components."""

    lexical = path.expanduser()
    if not lexical.is_absolute():
        lexical = Path.cwd() / lexical
    current = lexical
    while True:
        require(
            not current.is_symlink(),
            f"evidence output path contains a symlink: {current}",
        )
        if current.parent == current:
            break
        current = current.parent
    return lexical.resolve()


def validate_output_paths(
    paths: Iterable[Path],
    *,
    repo_root: Path,
    model_roots: Iterable[Path] = (),
    cache_roots: Iterable[Path] = (),
) -> tuple[Path, ...]:
    resolved = tuple(resolve_output_path(path) for path in paths)
    require(len(set(resolved)) == len(resolved), "evidence output paths must be distinct")
    for path in resolved:
        require(not path.exists(), f"refusing to overwrite evidence output: {path}")
        require(not path_is_within(path, repo_root), "evidence output must be outside source checkout")
        require(
            not any(path_is_within(path, root) for root in model_roots),
            "evidence output must be outside model directories",
        )
        require(
            not any(path_is_within(path, root) for root in cache_roots),
            "evidence output must be outside compiler cache directories",
        )
    return resolved


def environment_snapshot() -> dict[str, Any]:
    from torch._dynamo import config as dynamo_config

    cuda_available = torch.cuda.is_available()
    devices = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "uuid": str(getattr(properties, "uuid", "")),
                    "compute_capability": [properties.major, properties.minor],
                    "total_memory_bytes": properties.total_memory,
                }
            )
    smi = subprocess.run(
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "software": {
            "python": platform.python_version(),
            "python_executable": str(Path(sys.executable).resolve()),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "python_optimize": sys.flags.optimize,
            "torch_dynamo_disable": bool(dynamo_config.disable),
            "torch_dynamo_suppress_errors": bool(
                dynamo_config.suppress_errors
            ),
        },
        "hardware": {
            "cuda_available": cuda_available,
            "device_count": torch.cuda.device_count() if cuda_available else 0,
            "devices": devices,
            "nvidia_smi": smi.stdout.strip() if smi.returncode == 0 else None,
        },
        "selected_environment": {
            name: os.environ.get(name) for name in SELECTED_ENVIRONMENT
        },
    }


def require_retained_a100_sxm4_40gb() -> None:
    require(torch.cuda.is_available(), "retained evidence requires CUDA")
    require(
        torch.cuda.device_count() == 1,
        "retained evidence requires exactly one visible GPU",
    )
    properties = torch.cuda.get_device_properties(0)
    require(
        properties.name == "NVIDIA A100-SXM4-40GB",
        "retained evidence requires NVIDIA A100-SXM4-40GB, "
        f"got {properties.name!r}",
    )
    require(
        (properties.major, properties.minor) == (8, 0),
        "retained evidence requires compute capability 8.0",
    )
    require(
        properties.total_memory >= 39 * 1024**3,
        "retained evidence requires at least 39 GiB visible device memory",
    )
    device_uuid = str(getattr(properties, "uuid", ""))
    require(
        not device_uuid.startswith("MIG-"),
        "retained evidence does not certify a MIG partition",
    )


def require_retained_runtime_environment() -> None:
    """Reject Python/Torch modes that can make evidence assertions vacuous."""

    from torch._dynamo import config as dynamo_config

    require(
        sys.flags.optimize == 0,
        "retained evidence refuses optimized Python",
    )
    require(
        os.environ.get("TORCHDYNAMO_SUPPRESS_ERRORS") in (None, ""),
        "TORCHDYNAMO_SUPPRESS_ERRORS must be unset for retained evidence",
    )
    require(
        dynamo_config.disable is False,
        "retained evidence requires TorchDynamo to be enabled",
    )
    require(
        dynamo_config.suppress_errors is False,
        "retained evidence requires TorchDynamo error suppression to be disabled",
    )


@dataclass(frozen=True)
class EvidenceContext:
    repo_root: Path
    script_path: Path
    helper_path: Path
    retained: bool
    expected_commit: str | None
    source_before: dict[str, Any]
    implementation: dict[str, Any]
    source_files_before: dict[str, dict[str, Any]]
    runtime_imports_before: dict[str, dict[str, Any]]
    model_paths: dict[str, Path]
    models_before: dict[str, dict[str, Any]]
    environment_before: dict[str, Any]
    invocation: list[str]
    cwd: str


def prepare_evidence(
    *,
    script_path: Path,
    retained: bool,
    expected_commit: str | None,
    model_arguments: Mapping[str, str],
    runtime_imports: Mapping[str, tuple[Any, str]],
) -> EvidenceContext:
    script_path = script_path.resolve()
    repo_root = script_path.parents[1]
    helper_path = Path(__file__).resolve()
    source_before = git_snapshot(repo_root)
    implementation = validate_implementation_binding(repo_root, source_before)
    if expected_commit is not None:
        expected_commit = expected_commit.lower()
        require(FULL_SHA_RE.fullmatch(expected_commit), "--expected-commit must be a full Git SHA")
        require(source_before["head"] == expected_commit, "checkout does not match --expected-commit")
    if retained:
        require(expected_commit is not None, "--retained requires --expected-commit")
        require_retained_runtime_environment()
        require(source_before["clean"], f"retained evidence requires a clean source tree: {source_before['status_porcelain_v1']!r}")
    source_files = {
        "runner": source_file_identity(script_path, repo_root),
        "helper": source_file_identity(helper_path, repo_root),
    }
    model_paths: dict[str, Path] = {}
    models: dict[str, dict[str, Any]] = {}
    by_path: dict[Path, dict[str, Any]] = {}
    for role, argument in model_arguments.items():
        resolved = Path(argument).expanduser().resolve()
        if resolved not in by_path:
            path, snapshot = model_snapshot(argument)
            by_path[path] = snapshot
        model_paths[role] = resolved
        models[role] = {**by_path[resolved], "argument": argument}
    return EvidenceContext(
        repo_root=repo_root,
        script_path=script_path,
        helper_path=helper_path,
        retained=retained,
        expected_commit=expected_commit,
        source_before=source_before,
        implementation=implementation,
        source_files_before=source_files,
        runtime_imports_before=runtime_import_identities(repo_root, runtime_imports),
        model_paths=model_paths,
        models_before=models,
        environment_before=environment_snapshot(),
        invocation=[str(script_path), *sys.argv[1:]],
        cwd=str(Path.cwd().resolve()),
    )


def finalize_evidence(context: EvidenceContext) -> dict[str, Any]:
    source_after = git_snapshot(context.repo_root)
    source_files_after = {
        "runner": source_file_identity(context.script_path, context.repo_root),
        "helper": source_file_identity(context.helper_path, context.repo_root),
    }
    runtime_imports_after = {
        name: source_file_identity(
            context.repo_root / identity["path"], context.repo_root
        )
        for name, identity in context.runtime_imports_before.items()
    }
    model_snapshots_by_path = {
        path: model_snapshot(str(path))[1]
        for path in set(context.model_paths.values())
    }
    models_after = {
        role: {
            **model_snapshots_by_path[path],
            "argument": context.models_before[role]["argument"],
        }
        for role, path in context.model_paths.items()
    }
    environment_after = environment_snapshot()
    source_unchanged = source_after == context.source_before
    source_files_unchanged = source_files_after == context.source_files_before
    imports_unchanged = runtime_imports_after == context.runtime_imports_before
    models_unchanged = all(
        model_content_identity(models_after[role])
        == model_content_identity(context.models_before[role])
        for role in context.models_before
    )
    environment_unchanged = environment_after == context.environment_before
    committed_sources = all(
        identity["matches_head"]
        for identity in context.source_files_before.values()
    ) and all(
        identity["matches_head"]
        for identity in context.runtime_imports_before.values()
    )
    retention_eligible = bool(
        context.retained
        and context.expected_commit == context.source_before["head"]
        and context.source_before["clean"]
        and source_after["clean"]
        and source_unchanged
        and source_files_unchanged
        and imports_unchanged
        and models_unchanged
        and environment_unchanged
        and committed_sources
        and context.source_before["nanovllm_tree"]
        == IMPLEMENTATION_NANOVLLM_TREE
    )
    if context.retained:
        require(retention_eligible, "retained evidence provenance gate failed")
    return {
        "implementation": context.implementation,
        "producer_commit": context.source_before["head"],
        "source": {
            "before": context.source_before,
            "after": source_after,
            "unchanged": source_unchanged,
        },
        "source_files": {
            "before": context.source_files_before,
            "after": source_files_after,
            "unchanged": source_files_unchanged,
        },
        "runtime_imports": {
            "before": context.runtime_imports_before,
            "after": runtime_imports_after,
            "unchanged": imports_unchanged,
        },
        "models": {
            role: {
                "before": context.models_before[role],
                "after": models_after[role],
                "unchanged": (
                    model_content_identity(models_after[role])
                    == model_content_identity(context.models_before[role])
                ),
            }
            for role in context.models_before
        },
        "environment": {
            "before": context.environment_before,
            "after": environment_after,
            "unchanged": environment_unchanged,
        },
        "invocation": context.invocation,
        "cwd": context.cwd,
        "retention_requested": context.retained,
        "retention_eligible": retention_eligible,
    }


def validate_retained_provenance(
    provenance: Any,
    *,
    repo_root: Path,
    current_source: Mapping[str, Any],
    expected_commit: str,
    expected_runner_path: str,
    expected_model_roles: Iterable[str],
    expected_runtime_imports: Mapping[str, str],
    model_snapshot_cache: dict[Path, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Independently validate a producer provenance record.

    Pair comparators must call this instead of trusting the producer
    retention_eligible boolean. Model content is rehashed from the paths named
    by the evidence, and every recorded source/import identity is checked
    against the clean comparator checkout.
    """

    require(isinstance(provenance, dict), "producer provenance must be an object")
    require(
        FULL_SHA_RE.fullmatch(expected_commit),
        "expected producer commit must be a full Git SHA",
    )
    require(
        provenance.get("implementation")
        == {
            "commit": IMPLEMENTATION_COMMIT,
            "tree": IMPLEMENTATION_TREE,
            "parent": IMPLEMENTATION_PARENT,
            "nanovllm_tree": IMPLEMENTATION_NANOVLLM_TREE,
        },
        "producer implementation identity is not registered V3",
    )
    require(
        provenance.get("producer_commit") == expected_commit,
        "producer commit mismatch",
    )
    require(
        provenance.get("retention_requested") is True,
        "producer was not run in retained mode",
    )
    require(
        provenance.get("retention_eligible") is True,
        "producer was not retention eligible",
    )

    source = provenance.get("source")
    require(isinstance(source, dict), "producer source provenance is missing")
    before = source.get("before")
    after = source.get("after")
    require(
        isinstance(before, dict) and isinstance(after, dict),
        "producer source endpoints are missing",
    )
    require(
        source.get("unchanged") is True and before == after,
        "producer source changed during the run",
    )
    require(before.get("head") == expected_commit, "producer source HEAD mismatch")
    require(before.get("clean") is True, "producer source was dirty")
    require(
        before.get("status_porcelain_v1") == [],
        "producer source status was not empty",
    )
    for name in (
        "head",
        "tree",
        "nanovllm_tree",
        "clean",
        "status_porcelain_v1",
        "status_sha256",
        "source_tree_sha256",
    ):
        require(
            before.get(name) == current_source.get(name),
            f"producer source does not match comparator checkout: {name}",
        )
    require(
        before["nanovllm_tree"] == IMPLEMENTATION_NANOVLLM_TREE,
        "producer runtime tree differs from registered V3",
    )

    source_files = provenance.get("source_files")
    require(
        isinstance(source_files, dict),
        "producer source-file provenance is missing",
    )
    recorded_files = source_files.get("before")
    require(
        source_files.get("unchanged") is True
        and recorded_files == source_files.get("after"),
        "producer source files changed during the run",
    )
    require(
        isinstance(recorded_files, dict)
        and set(recorded_files) == {"runner", "helper"},
        "producer source-file registry drifted",
    )
    expected_files = {
        "runner": source_file_identity(
            repo_root / expected_runner_path, repo_root
        ),
        "helper": source_file_identity(Path(__file__).resolve(), repo_root),
    }
    require(
        recorded_files == expected_files,
        "producer runner/helper identity mismatch",
    )
    require(
        all(identity["matches_head"] for identity in recorded_files.values()),
        "producer runner/helper was not committed",
    )

    runtime_imports = provenance.get("runtime_imports")
    require(
        isinstance(runtime_imports, dict),
        "producer runtime-import provenance is missing",
    )
    recorded_imports = runtime_imports.get("before")
    require(
        runtime_imports.get("unchanged") is True
        and isinstance(recorded_imports, dict)
        and recorded_imports == runtime_imports.get("after"),
        "producer runtime imports changed during the run",
    )
    require(
        set(recorded_imports) == set(expected_runtime_imports),
        "producer runtime-import registry is incomplete",
    )
    for name, identity in recorded_imports.items():
        require(
            isinstance(identity, dict),
            f"runtime import {name} identity is invalid",
        )
        relative = identity.get("path")
        require(
            isinstance(relative, str)
            and (
                relative == "nanovllm/__init__.py"
                or relative.startswith("nanovllm/")
            ),
            f"runtime import {name} is outside nanovllm",
        )
        require(
            relative == expected_runtime_imports[name],
            f"runtime import {name} path drifted",
        )
        require(
            identity
            == source_file_identity(repo_root / relative, repo_root),
            f"runtime import {name} does not match comparator checkout",
        )
        require(
            identity["matches_head"],
            f"runtime import {name} was not committed",
        )

    models = provenance.get("models")
    expected_roles = set(expected_model_roles)
    require(
        isinstance(models, dict) and set(models) == expected_roles,
        "producer model-role registry drifted",
    )
    model_identities = {}
    current_by_path = (
        {} if model_snapshot_cache is None else model_snapshot_cache
    )
    for role in sorted(expected_roles):
        record = models[role]
        require(isinstance(record, dict), f"{role} model provenance is invalid")
        model_before = record.get("before")
        model_after = record.get("after")
        require(
            record.get("unchanged") is True
            and isinstance(model_before, dict)
            and isinstance(model_after, dict)
            and model_content_identity(model_before)
            == model_content_identity(model_after),
            f"{role} model changed during the run",
        )
        resolved = Path(model_before["resolved_path"]).resolve()
        if resolved not in current_by_path:
            current_by_path[resolved] = model_snapshot(str(resolved))[1]
        require(
            model_content_identity(model_before)
            == model_content_identity(current_by_path[resolved]),
            f"{role} model bytes no longer match producer evidence",
        )
        model_identities[role] = model_content_identity(model_before)

    environment = provenance.get("environment")
    require(
        isinstance(environment, dict),
        "producer environment provenance is missing",
    )
    require(
        environment.get("unchanged") is True
        and environment.get("before") == environment.get("after"),
        "producer environment changed during the run",
    )
    recorded_environment = environment["before"]
    require(
        isinstance(recorded_environment, dict),
        "producer environment snapshot is invalid",
    )
    software = recorded_environment.get("software")
    selected_environment = recorded_environment.get("selected_environment")
    require(
        isinstance(software, dict)
        and software.get("python_optimize") == 0,
        "producer used optimized Python",
    )
    require(
        software.get("torch_dynamo_disable") is False,
        "producer disabled TorchDynamo",
    )
    require(
        software.get("torch_dynamo_suppress_errors") is False,
        "producer suppressed TorchDynamo errors",
    )
    require(
        isinstance(selected_environment, dict)
        and selected_environment.get("TORCHDYNAMO_SUPPRESS_ERRORS")
        in (None, ""),
        "producer set TORCHDYNAMO_SUPPRESS_ERRORS",
    )
    recorded_cache_roots = []
    for name in COMPILER_CACHE_ENVIRONMENT:
        value = selected_environment.get(name)
        require(
            isinstance(value, str) and value,
            f"producer omitted {name}",
        )
        root = Path(value)
        require(
            root.is_absolute() and root.resolve() == root,
            f"producer {name} is not a canonical path",
        )
        recorded_cache_roots.append(root)
    validate_compiler_cache_root_isolation(
        recorded_cache_roots,
        repo_root=repo_root,
        model_roots=(
            Path(identity["resolved_path"])
            for identity in model_identities.values()
        ),
    )
    invocation = provenance.get("invocation")
    require(
        isinstance(invocation, list)
        and invocation
        and Path(invocation[0]).resolve()
        == (repo_root / expected_runner_path).resolve(),
        "producer invocation does not name the registered runner",
    )
    require("--retained" in invocation, "producer invocation omitted --retained")
    require(
        expected_commit in invocation,
        "producer invocation omitted the expected producer commit",
    )
    return {
        "implementation": provenance["implementation"],
        "producer_commit": expected_commit,
        "source": before,
        "source_files": recorded_files,
        "runtime_imports": recorded_imports,
        "models": model_identities,
        "environment": recorded_environment,
    }


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path = resolve_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path = resolve_output_path(path)
    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(encoded)


def write_torch_exclusive(path: Path, payload: Any) -> None:
    """Serialize a weights-only-compatible payload without replacing a path."""

    path = resolve_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path = resolve_output_path(path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        torch.save(payload, handle)


def reject_duplicate_json_pairs(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, f"duplicate JSON key: {key}")
        value[key] = item
    return value


def reject_nonfinite_json(value: str):
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def load_strict_json_bytes(payload: bytes) -> Any:
    return json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=reject_duplicate_json_pairs,
        parse_constant=reject_nonfinite_json,
    )


def read_registered_file(path: Path, *, max_bytes: int) -> tuple[bytes, dict[str, Any]]:
    lexical_path = path.expanduser()
    if not lexical_path.is_absolute():
        lexical_path = Path.cwd() / lexical_path
    require(
        not lexical_path.is_symlink(),
        f"registered artifact is a symlink: {lexical_path}",
    )
    path = lexical_path.resolve()
    require(path.exists(), f"registered artifact is missing: {path}")
    metadata = path.stat()
    require(stat.S_ISREG(metadata.st_mode), f"registered artifact is not regular: {path}")
    require(metadata.st_nlink == 1, f"registered artifact has multiple hard links: {path}")
    require(metadata.st_size <= max_bytes, f"registered artifact exceeds byte cap: {path}")
    payload = path.read_bytes()
    require(len(payload) == metadata.st_size, f"artifact size changed while reading: {path}")
    return payload, {
        "size_bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }
