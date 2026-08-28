"""Isolated A100 gate for the inert V2 dual-model runner.

Run this file in a fresh process.  It intentionally creates and tears down
several NCCL process groups and CUDA owners in sequence.  V2 does not execute
draft tokens yet: the gate proves configuration, ownership, memory planning,
RNG neutrality, and current-tree speculation-off target execution while the
draft is inert.
"""

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
import transformers

import nanovllm
from nanovllm import LLM, SamplingParams
from nanovllm.engine import llm_engine as llm_engine_module
from nanovllm.engine.model_runner import (
    SpeculativeKVCacheCapacityError,
)
from nanovllm.utils.context import get_context


EVIDENCE_SCHEMA = "nano-vllm-speculative-v2-lifecycle-v2"
V0_EVIDENCE_SCHEMA = "nano-vllm-speculative-v0-golden-v1"
CANONICAL_V0_COMMIT = "480a3b26c5a4e465aac06d1dabd34e1230686feb"
MIN_GRAPH_ALLOCATOR_MARGIN_BYTES = 64 * 1024**2
GPU_PID_BINDING_CHALLENGE_BYTES = 389 * 1024**2
GPU_PID_BINDING_ROUNDING_TOLERANCE_MIB = 1
RETAINED_GPU_NAME = "NVIDIA A100-SXM4-40GB"
RETAINED_GPU_COMPUTE_CAPABILITY = (8, 0)
RETAINED_GPU_MIN_MEMORY_BYTES = 39 * 1024**3
MODEL_METADATA_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
PROVENANCE_ENVIRONMENT_KEYS = (
    "CONDA_DEFAULT_ENV",
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_LAUNCH_BLOCKING",
    "CUDA_MODULE_LOADING",
    "CUDA_VISIBLE_DEVICES",
    "HF_HOME",
    "LD_LIBRARY_PATH",
    "MKL_NUM_THREADS",
    "NCCL_DEBUG",
    "NCCL_P2P_DISABLE",
    "OMP_NUM_THREADS",
    "PYTHONPATH",
    "PYTHONHASHSEED",
    "PYTORCH_CUDA_ALLOC_CONF",
    "TOKENIZERS_PARALLELISM",
    "TORCH_CUDA_ARCH_LIST",
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE",
    "TORCHINDUCTOR_AUTOGRAD_CACHE",
    "TORCH_COMPILE_DISABLE",
    "TORCH_LOGS",
    "TRITON_CACHE_DIR",
    "VIRTUAL_ENV",
)


def require(condition, message):
    """Certification check that remains active under optimized Python."""

    if not condition:
        raise AssertionError(message)


def reject_optimized_python():
    require(
        __debug__,
        "the V2 lifecycle gate refuses optimized Python because -O removes "
        "third-party assertions used by dependencies",
    )


def physical_parameter_storages(model):
    """Map physical parameter allocations without tied-view inflation."""

    storages = {}
    for parameter in model.parameters():
        storage = parameter.untyped_storage()
        key = (storage.device, storage.data_ptr(), storage.nbytes())
        storages[key] = storage.nbytes()
    return storages


def physical_parameter_bytes(model):
    """Count independently owned parameter storages without tied-view inflation."""

    return sum(physical_parameter_storages(model).values())


def expected_cache_shape(hf_config, *, num_blocks, block_size, world_size):
    head_dim = getattr(hf_config, "head_dim", None)
    if head_dim is None:
        head_dim = hf_config.hidden_size // hf_config.num_attention_heads
    return (
        2,
        hf_config.num_hidden_layers,
        num_blocks,
        block_size,
        hf_config.num_key_value_heads // world_size,
        head_dim,
    )


def independent_memory_reconciliation(audit, *, enforce_eager):
    """Recompute runner capacity invariants without calling production helpers."""

    plan = audit.workspace_plan
    profiled_graph_ownership_bytes = max(
        audit.profiled_graph_allocated_bytes,
        audit.profiled_graph_reserved_bytes,
    )
    profiled_graph_peak_bytes = max(
        profiled_graph_ownership_bytes,
        audit.profiled_graph_peak_allocated_bytes,
        audit.profiled_graph_peak_reserved_bytes,
    )
    graph_allocator_margin_bytes = (
        0
        if enforce_eager
        else max(
            MIN_GRAPH_ALLOCATOR_MARGIN_BYTES,
            audit.joint_block_bytes,
        )
    )
    graph_construction_reservation_bytes = (
        profiled_graph_peak_bytes + graph_allocator_margin_bytes
    )
    runtime_reservation_bytes = (
        profiled_graph_ownership_bytes
        + audit.warmup_transient_bytes
        + plan.reservation_bytes
    )
    sizing_overhead_bytes = max(
        graph_construction_reservation_bytes,
        runtime_reservation_bytes,
    )
    sizing_usable_bytes = (
        audit.memory_budget_bytes
        - audit.used_before_kv_bytes
        - sizing_overhead_bytes
    )
    automatic_num_blocks = sizing_usable_bytes // audit.joint_block_bytes
    modeled_runtime_headroom_bytes = (
        audit.post_init_budget_headroom_bytes
        - audit.warmup_transient_bytes
        - plan.reservation_bytes
    )
    kv_allocated_increment_bytes = (
        audit.allocated_after_kv_before_graph_bytes
        - audit.allocated_before_kv_bytes
    )
    kv_accounted_bytes = audit.target_kv_bytes + audit.draft_kv_bytes
    final_graph_allocated_increment_bytes = max(
        audit.allocated_after_graph_before_pretouch_bytes
        - audit.allocated_after_kv_before_graph_bytes,
        0,
    )
    final_graph_reserved_increment_bytes = max(
        audit.reserved_after_graph_before_pretouch_bytes
        - audit.reserved_after_kv_before_graph_bytes,
        0,
    )
    post_init_allocated_increment_bytes = max(
        audit.post_init_allocated_bytes
        - audit.allocated_after_graph_before_pretouch_bytes,
        0,
    )
    post_init_reserved_increment_bytes = max(
        audit.post_init_reserved_bytes
        - audit.reserved_after_graph_before_pretouch_bytes,
        0,
    )
    total_allocated_increment_bytes = (
        audit.post_init_allocated_bytes - audit.allocated_before_kv_bytes
    )
    total_accounted_increment_bytes = (
        kv_accounted_bytes
        + audit.final_graph_allocated_bytes
        + post_init_allocated_increment_bytes
    )
    return {
        "profiled_graph_ownership_bytes": profiled_graph_ownership_bytes,
        "profiled_graph_peak_bytes": profiled_graph_peak_bytes,
        "graph_allocator_margin_bytes": graph_allocator_margin_bytes,
        "graph_construction_reservation_bytes": (
            graph_construction_reservation_bytes
        ),
        "runtime_reservation_bytes": runtime_reservation_bytes,
        "sizing_overhead_bytes": sizing_overhead_bytes,
        "sizing_usable_bytes": sizing_usable_bytes,
        "automatic_num_blocks": automatic_num_blocks,
        "modeled_runtime_headroom_bytes": modeled_runtime_headroom_bytes,
        "kv_allocated_increment_bytes": kv_allocated_increment_bytes,
        "kv_accounted_bytes": kv_accounted_bytes,
        "final_graph_allocated_increment_bytes": (
            final_graph_allocated_increment_bytes
        ),
        "final_graph_reserved_increment_bytes": (
            final_graph_reserved_increment_bytes
        ),
        "post_init_allocated_increment_bytes": (
            post_init_allocated_increment_bytes
        ),
        "post_init_reserved_increment_bytes": (
            post_init_reserved_increment_bytes
        ),
        "total_allocated_increment_bytes": total_allocated_increment_bytes,
        "total_accounted_increment_bytes": total_accounted_increment_bytes,
    }


def _git_output(repo_root, *args, text=True):
    return subprocess.run(
        ("git", *args),
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=text,
    ).stdout


def source_tree_sha256(repo_root):
    """Fingerprint tracked and untracked, non-ignored source tree contents."""

    raw_paths = _git_output(
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


def git_snapshot(repo_root):
    """Capture commit identity plus exact tracked/untracked source content."""

    top_level = Path(
        _git_output(repo_root, "rev-parse", "--show-toplevel").strip()
    ).resolve()
    branch_result = subprocess.run(
        ("git", "symbolic-ref", "--quiet", "--short", "HEAD"),
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    require(
        branch_result.returncode in (0, 1),
        "git could not determine whether the V2 checkout is detached: "
        f"{branch_result.stderr.strip()}",
    )
    branch = (
        branch_result.stdout.strip()
        if branch_result.returncode == 0
        else None
    )
    status = _git_output(
        repo_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).splitlines()
    return {
        "repository": str(top_level),
        "head": _git_output(repo_root, "rev-parse", "HEAD").strip(),
        "tree": _git_output(repo_root, "rev-parse", "HEAD^{tree}").strip(),
        "branch": branch,
        "detached": branch is None,
        "dirty": bool(status),
        "status_porcelain_v1": status,
        "source_tree_sha256": source_tree_sha256(repo_root),
    }


def validate_source_snapshot(snapshot, *, expected_commit, allow_dirty):
    require(
        Path(snapshot["repository"]).resolve()
        == Path(__file__).resolve().parents[1],
        "V2 source checkout differs from the evidence-runner checkout",
    )
    if expected_commit is not None:
        require(
            re.fullmatch(r"[0-9a-fA-F]{40}", expected_commit) is not None,
            "--expected-commit must be a full 40-character hexadecimal Git SHA",
        )
        require(
            snapshot["head"].lower() == expected_commit.lower(),
            "V2 checkout is not at the expected commit: "
            f"expected {expected_commit}, found {snapshot['head']}",
        )
    if not allow_dirty:
        require(
            not snapshot["dirty"],
            "retained V2 evidence requires a clean checkout; use "
            "--allow-dirty only for explicitly exploratory runs; status is "
            f"{snapshot['status_porcelain_v1']!r}",
        )


def validate_unchanged(before, after):
    for key in (
        "repository",
        "head",
        "tree",
        "branch",
        "detached",
        "status_porcelain_v1",
        "source_tree_sha256",
    ):
        require(
            after[key] == before[key],
            f"V2 source provenance changed during the run: {key}",
        )


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_snapshot(model_argument):
    model_path = Path(model_argument).expanduser().resolve()
    require(model_path.is_dir(), f"model path is not a directory: {model_path}")
    require(
        (model_path / "config.json").is_file(),
        f"model config is missing: {model_path / 'config.json'}",
    )
    metadata = {}
    for name in MODEL_METADATA_FILES:
        path = model_path / name
        if path.is_file():
            metadata[name] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    weight_files = sorted(
        path
        for path in model_path.glob("*.safetensors")
        if path.is_file()
    )
    require(weight_files, f"model has no safetensors weight files: {model_path}")
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
            for path in weight_files
        ],
        "total_weight_bytes": sum(path.stat().st_size for path in weight_files),
    }


def path_is_within(path, directory):
    return path == directory or directory in path.parents


def write_json_exclusive(path, payload):
    """Publish complete JSON atomically without replacing any existing path."""

    serialized = (json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    ) + "\n").encode("utf-8")
    temporary_path = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_path, path, follow_symlinks=False)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(path.parent, directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as error:
        raise RuntimeError(
            f"refusing to overwrite existing output: {path}"
        ) from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
    return hashlib.sha256(serialized).hexdigest()


def model_content_identity(snapshot):
    return {
        key: snapshot[key]
        for key in (
            "resolved_path",
            "metadata_files",
            "weight_files",
            "total_weight_bytes",
        )
    }


def validate_v0_golden(
    path,
    *,
    expected_sha256,
    expected_runner_commit,
    args,
    target_model_evidence,
    tokenizer_evidence,
    outputs,
    scheduler_trace,
):
    require(path.is_file(), f"V0 golden is not a file: {path}")
    require(
        re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256) is not None,
        "--v0-golden-sha256 must be a full SHA-256 digest",
    )
    actual_sha256 = sha256_file(path)
    require(
        actual_sha256 == expected_sha256.lower(),
        "V0 golden artifact hash mismatch: "
        f"expected {expected_sha256}, found {actual_sha256}",
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    require(
        document.get("schema") == V0_EVIDENCE_SCHEMA,
        "V0 golden has the wrong evidence schema",
    )
    source = document.get("source", {})
    require(
        source.get("expected_commit") == CANONICAL_V0_COMMIT,
        "V0 golden does not pin the canonical V0 commit",
    )
    for key in ("git_before", "git_after"):
        snapshot = source.get(key, {})
        require(
            snapshot.get("head") == CANONICAL_V0_COMMIT
            and snapshot.get("detached") is True
            and snapshot.get("dirty") is False,
            f"V0 golden {key} is not clean canonical detached V0",
        )
    require(
        source.get("unchanged_during_run") is True,
        "V0 source was not certified unchanged during its run",
    )
    runner = document.get("runner", {})
    require(
        runner.get("expected_commit") == expected_runner_commit,
        "V0 golden was not produced by the expected V2 runner commit",
    )
    for key in ("git_before", "git_after"):
        snapshot = runner.get(key, {})
        require(
            snapshot.get("head") == expected_runner_commit
            and snapshot.get("dirty") is False,
            f"V0 golden runner {key} is not the expected clean commit",
        )
    require(
        runner.get("script_sha256_before")
        == runner.get("script_sha256_after")
        and runner.get("unchanged_during_run") is True,
        "V0 golden runner script was not certified unchanged",
    )
    model = document.get("model", {})
    require(
        model.get("before") == model.get("after")
        and model.get("unchanged_during_run") is True,
        "V0 golden model was not certified unchanged",
    )
    require(
        model_content_identity(model["before"])
        == model_content_identity(target_model_evidence),
        "V0 golden model artifacts differ from the V2 target",
    )
    require(
        document.get("tokenizer") == tokenizer_evidence,
        "V0 golden runtime tokenizer differs from the V2 target tokenizer",
    )
    run = document.get("run", {})
    require(run.get("mode") == args.mode, "V0 golden mode differs from V2")
    requested = run.get("requested_engine_config", {})
    expected_requested = common_kwargs(args)
    for key in (
        "enforce_eager",
        "gpu_memory_utilization",
        "max_model_len",
        "max_num_batched_tokens",
        "max_num_seqs",
    ):
        require(
            requested.get(key) == expected_requested[key],
            f"V0 golden engine setting differs from V2: {key}",
        )
    for key, expected_value in {
        "tensor_parallel_size": 1,
        "kvcache_block_size": 256,
        "num_kvcache_blocks": -1,
        "top_p_backend": "exact",
        "disable_python_gc": False,
    }.items():
        require(
            requested.get(key) == expected_value,
            f"V0 golden fixed engine contract differs from V2: {key}",
        )
    randomness = document.get("randomness", {})
    require(
        randomness.get("seed") == 20260828
        and randomness.get("sampling_mode") == "greedy",
        "V0 golden randomness contract differs from V2",
    )
    expected_workload = {
        "prompts_token_ids": [[1, 2, 3, 4], [7, 8, 9]],
        "sampling_params": {
            "temperature": 0.0,
            "max_tokens": 4,
            "ignore_eos": True,
            "top_k": -1,
            "top_p": 1.0,
        },
    }
    require(
        run.get("workload") == expected_workload,
        "V0 golden workload differs from the registered V2 fixture",
    )
    expected_outputs = nested_dict(outputs)
    expected_trace = nested_dict(scheduler_trace)
    require(
        run.get("comparison_outputs") == expected_outputs,
        "current speculation-off outputs differ from canonical V0",
    )
    require(
        run.get("scheduler_trace") == expected_trace,
        "current speculation-off scheduler trace differs from canonical V0",
    )
    return {
        "artifact": str(path),
        "sha256": actual_sha256,
        "source_commit": CANONICAL_V0_COMMIT,
        "runner_commit": expected_runner_commit,
        "mode": args.mode,
        "outputs": run["comparison_outputs"],
        "scheduler_trace": run["scheduler_trace"],
        "outputs_match": True,
        "scheduler_trace_match": True,
    }


def normalize_gpu_uuid(value):
    if value is None:
        return None
    value = str(value)
    if value.startswith(("GPU-", "MIG-")):
        return value
    return f"GPU-{value}"


def parse_gpu_identity_rows(stdout):
    rows = []
    for fields in csv.reader(stdout.splitlines(), skipinitialspace=True):
        if not fields:
            continue
        require(
            len(fields) == 4,
            f"unexpected nvidia-smi GPU identity row: {fields!r}",
        )
        name, gpu_uuid, driver_version, total_memory_mib = (
            field.strip() for field in fields
        )
        rows.append({
            "name": name,
            "gpu_uuid": gpu_uuid,
            "driver_version": driver_version,
            "total_memory_mib": total_memory_mib,
        })
    return rows


def parse_compute_app_rows(stdout):
    rows = []
    for fields in csv.reader(stdout.splitlines(), skipinitialspace=True):
        if not fields:
            continue
        require(
            len(fields) == 4,
            f"unexpected nvidia-smi compute-app row: {fields!r}",
        )
        gpu_uuid, pid_text, process_name, used_memory_text = (
            field.strip() for field in fields
        )
        try:
            pid = int(pid_text)
        except ValueError:
            pid = None
        try:
            used_memory_mib = int(used_memory_text)
        except ValueError:
            used_memory_mib = None
        rows.append({
            "gpu_uuid": gpu_uuid,
            "pid": pid,
            "pid_raw": pid_text,
            "process_name": process_name,
            "used_memory_mib": used_memory_mib,
            "used_memory_raw": used_memory_text,
        })
    return rows


def own_process_namespace_ids():
    process_ids = {os.getpid()}
    try:
        status = Path("/proc/self/status").read_text(encoding="utf-8")
    except OSError:
        return sorted(process_ids)
    for line in status.splitlines():
        if line.startswith("NSpid:"):
            for value in line.split(":", 1)[1].split():
                try:
                    process_ids.add(int(value))
                except ValueError:
                    pass
            break
    return sorted(process_ids)


def compute_app_memory_mib(snapshot):
    """Return an unambiguous per-PID NVML memory ledger."""

    require(
        snapshot["compute_apps_returncode"] == 0,
        "GPU process query failed during allocator PID binding: "
        f"{snapshot['compute_apps_stderr']}",
    )
    result = {}
    for row in snapshot["compute_apps"]:
        pid = row["pid"]
        used_memory_mib = row["used_memory_mib"]
        require(
            type(pid) is int and type(used_memory_mib) is int,
            "allocator PID binding requires numeric nvidia-smi PID and "
            f"memory fields: {row!r}",
        )
        require(
            pid not in result,
            f"allocator PID binding saw duplicate nvidia-smi PID {pid}",
        )
        result[pid] = used_memory_mib
    return result


def validate_gpu_pid_allocator_challenge(
    *,
    candidate_pid,
    baseline_snapshot,
    challenged_snapshot,
    allocated_delta_bytes,
    reserved_delta_bytes,
):
    """Prove that the candidate PID owns this process's allocator growth."""

    baseline = compute_app_memory_mib(baseline_snapshot)
    challenged = compute_app_memory_mib(challenged_snapshot)
    require(
        set(challenged) == set(baseline),
        "GPU PID set changed during allocator binding challenge: "
        f"baseline={sorted(baseline)}, challenged={sorted(challenged)}",
    )
    require(
        candidate_pid in baseline,
        f"allocator binding candidate PID {candidate_pid} disappeared",
    )
    require(
        allocated_delta_bytes >= GPU_PID_BINDING_CHALLENGE_BYTES,
        "CUDA allocated-memory delta did not contain the registered PID "
        f"binding challenge: requested={GPU_PID_BINDING_CHALLENGE_BYTES}, "
        f"observed={allocated_delta_bytes}",
    )
    require(
        reserved_delta_bytes >= allocated_delta_bytes,
        "CUDA reserved-memory delta is smaller than allocated-memory delta "
        "during PID binding: "
        f"allocated={allocated_delta_bytes}, reserved={reserved_delta_bytes}",
    )
    mib = 1024**2
    reserved_delta_mib = reserved_delta_bytes / mib
    candidate_delta_mib = (
        challenged[candidate_pid] - baseline[candidate_pid]
    )
    require(
        abs(candidate_delta_mib - reserved_delta_mib)
        <= GPU_PID_BINDING_ROUNDING_TOLERANCE_MIB,
        "nvidia-smi candidate memory did not track this process's CUDA "
        "allocator challenge: "
        f"candidate_pid={candidate_pid}, "
        f"nvidia_smi_delta_mib={candidate_delta_mib}, "
        f"reserved_delta_mib={reserved_delta_mib}, "
        "tolerance_mib="
        f"{GPU_PID_BINDING_ROUNDING_TOLERANCE_MIB}",
    )
    for pid, baseline_mib in baseline.items():
        if pid != candidate_pid:
            require(
                challenged[pid] == baseline_mib,
                "non-candidate GPU process memory changed during allocator "
                f"binding: pid={pid}, baseline_mib={baseline_mib}, "
                f"challenged_mib={challenged[pid]}",
            )
    return {
        "candidate_pid": candidate_pid,
        "requested_bytes": GPU_PID_BINDING_CHALLENGE_BYTES,
        "allocated_delta_bytes": allocated_delta_bytes,
        "reserved_delta_bytes": reserved_delta_bytes,
        "reserved_delta_mib": reserved_delta_mib,
        "candidate_nvidia_smi_delta_mib": candidate_delta_mib,
        "rounding_tolerance_mib": (
            GPU_PID_BINDING_ROUNDING_TOLERANCE_MIB
        ),
    }


def validate_gpu_pid_allocator_release(
    *,
    baseline_snapshot,
    released_snapshot,
    baseline_allocated_bytes,
    baseline_reserved_bytes,
    released_allocated_bytes,
    released_reserved_bytes,
):
    """Require both NVML and the Torch allocator to return to baseline."""

    baseline = compute_app_memory_mib(baseline_snapshot)
    released = compute_app_memory_mib(released_snapshot)
    require(
        released == baseline,
        "GPU process memory did not return exactly to its pre-challenge "
        f"baseline: baseline={baseline}, released={released}",
    )
    require(
        released_allocated_bytes == baseline_allocated_bytes,
        "CUDA allocated memory did not return to the PID-binding baseline: "
        f"baseline={baseline_allocated_bytes}, "
        f"released={released_allocated_bytes}",
    )
    require(
        released_reserved_bytes == baseline_reserved_bytes,
        "CUDA reserved memory did not return to the PID-binding baseline: "
        f"baseline={baseline_reserved_bytes}, "
        f"released={released_reserved_bytes}",
    )


def run_gpu_pid_allocator_challenge(candidate_pid, baseline_snapshot):
    """Challenge one host PID and always unwind the temporary allocation."""

    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    challenge = None
    challenged_snapshot = None
    challenge_measurements = None
    primary_error = None
    try:
        challenge = torch.empty(
            GPU_PID_BINDING_CHALLENGE_BYTES,
            dtype=torch.uint8,
            device="cuda",
        )
        torch.cuda.synchronize()
        allocated_after = torch.cuda.memory_allocated()
        reserved_after = torch.cuda.memory_reserved()
        challenged_snapshot = gpu_gate_snapshot(
            phase="pid_binding_challenged",
            own_gpu_process_ids=(candidate_pid,),
        )
        challenge_measurements = validate_gpu_pid_allocator_challenge(
            candidate_pid=candidate_pid,
            baseline_snapshot=baseline_snapshot,
            challenged_snapshot=challenged_snapshot,
            allocated_delta_bytes=allocated_after - baseline_allocated,
            reserved_delta_bytes=reserved_after - baseline_reserved,
        )
    except BaseException as error:
        primary_error = error
    finally:
        challenge = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    released_allocated = torch.cuda.memory_allocated()
    released_reserved = torch.cuda.memory_reserved()
    released_snapshot = gpu_gate_snapshot(
        phase="pid_binding_released",
        own_gpu_process_ids=(candidate_pid,),
    )
    try:
        validate_gpu_pid_allocator_release(
            baseline_snapshot=baseline_snapshot,
            released_snapshot=released_snapshot,
            baseline_allocated_bytes=baseline_allocated,
            baseline_reserved_bytes=baseline_reserved,
            released_allocated_bytes=released_allocated,
            released_reserved_bytes=released_reserved,
        )
    except BaseException as release_error:
        if primary_error is not None:
            raise release_error from primary_error
        raise
    if primary_error is not None:
        raise primary_error
    return {
        **challenge_measurements,
        "baseline_snapshot": baseline_snapshot,
        "challenged_snapshot": challenged_snapshot,
        "released_snapshot": released_snapshot,
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "released_allocated_bytes": released_allocated,
        "released_reserved_bytes": released_reserved,
    }


def gpu_gate_snapshot(*, phase, own_gpu_process_ids=()):
    require(torch.cuda.is_available(), "V2 A100 gate requires CUDA")
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    gpu_uuid = getattr(properties, "uuid", None)
    gpu_uuid_text = None if gpu_uuid is None else str(gpu_uuid)
    nvidia_smi_id = normalize_gpu_uuid(gpu_uuid_text) or str(device_index)
    identity = subprocess.run(
        (
            "nvidia-smi",
            f"--id={nvidia_smi_id}",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    compute_apps = subprocess.run(
        (
            "nvidia-smi",
            f"--id={nvidia_smi_id}",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    identity_rows = (
        parse_gpu_identity_rows(identity.stdout)
        if identity.returncode == 0
        else []
    )
    app_rows = (
        parse_compute_app_rows(compute_apps.stdout)
        if compute_apps.returncode == 0
        else []
    )
    own_pids = sorted({
        *own_process_namespace_ids(),
        *own_gpu_process_ids,
    })
    return {
        "phase": phase,
        "index": device_index,
        "name": properties.name,
        "uuid": gpu_uuid_text,
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": properties.total_memory,
        "multiprocessor_count": properties.multi_processor_count,
        "nvidia_smi_query_id": nvidia_smi_id,
        "nvidia_smi_raw_lines": identity.stdout.strip().splitlines(),
        "nvidia_smi_parsed_rows": identity_rows,
        "nvidia_smi_returncode": identity.returncode,
        "nvidia_smi_stderr": identity.stderr.strip(),
        "compute_apps_raw_lines": (
            compute_apps.stdout.strip().splitlines()
        ),
        "compute_apps": app_rows,
        "compute_apps_returncode": compute_apps.returncode,
        "compute_apps_stderr": compute_apps.stderr.strip(),
        "own_process_namespace_ids": own_pids,
        "own_gpu_process_ids": sorted(set(own_gpu_process_ids)),
        "foreign_compute_apps": [
            row for row in app_rows if row["pid"] not in own_pids
        ],
    }


def establish_gpu_gate_before(*, retained):
    """Create one traceable CUDA context and bind its host-namespace PID."""

    pre_context = gpu_gate_snapshot(phase="before_cuda_context")
    if retained:
        validate_retained_gpu_snapshot(
            pre_context,
            phase="before CUDA context",
        )
    preexisting_pids = {
        row["pid"]
        for row in pre_context["compute_apps"]
        if row["pid"] is not None
    }
    sentinel = torch.empty(1, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    context_snapshot = gpu_gate_snapshot(phase="context_establishment")
    new_pids = sorted({
        row["pid"]
        for row in context_snapshot["compute_apps"]
        if row["pid"] is not None and row["pid"] not in preexisting_pids
    })
    del sentinel
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    require(
        len(new_pids) == 1,
        "could not bind exactly one new nvidia-smi process to this fresh "
        f"CUDA context: pre={pre_context['compute_apps']!r}, "
        f"post={context_snapshot['compute_apps']!r}",
    )
    candidate_pid = new_pids[0]
    context_pids = {
        row["pid"]
        for row in context_snapshot["compute_apps"]
        if row["pid"] is not None
    }
    challenge_baseline = gpu_gate_snapshot(
        phase="pid_binding_baseline",
        own_gpu_process_ids=(candidate_pid,),
    )
    challenge_baseline_pids = set(
        compute_app_memory_mib(challenge_baseline)
    )
    require(
        challenge_baseline_pids == context_pids,
        "GPU PID set changed before allocator binding challenge: "
        f"context={sorted(context_pids)}, "
        f"baseline={sorted(challenge_baseline_pids)}",
    )
    allocator_challenge = run_gpu_pid_allocator_challenge(
        candidate_pid,
        challenge_baseline,
    )
    before = gpu_gate_snapshot(
        phase="before",
        own_gpu_process_ids=(candidate_pid,),
    )
    before["self_context_establishment"] = {
        "pre_context_compute_apps": pre_context["compute_apps"],
        "post_sentinel_compute_apps": context_snapshot["compute_apps"],
        "bound_host_namespace_pid": candidate_pid,
        "visible_pid_namespace_aliases": own_process_namespace_ids(),
        "allocator_challenge": allocator_challenge,
    }
    if retained:
        validate_retained_gpu_snapshot(before, phase="before")
    return before


def validate_retained_gpu_snapshot(snapshot, *, phase):
    require(
        snapshot["nvidia_smi_returncode"] == 0,
        f"retained {phase} GPU identity query failed: "
        f"{snapshot['nvidia_smi_stderr']}",
    )
    require(
        snapshot["uuid"] is not None
        and not snapshot["nvidia_smi_query_id"].startswith("MIG-"),
        f"retained {phase} gate requires a non-MIG canonical GPU UUID",
    )
    require(
        snapshot["name"] == RETAINED_GPU_NAME,
        f"retained {phase} gate requires {RETAINED_GPU_NAME}, found "
        f"{snapshot['name']}",
    )
    require(
        tuple(snapshot["compute_capability"])
        == RETAINED_GPU_COMPUTE_CAPABILITY,
        f"retained {phase} gate requires compute capability "
        f"{RETAINED_GPU_COMPUTE_CAPABILITY}, found "
        f"{snapshot['compute_capability']}",
    )
    require(
        snapshot["total_memory_bytes"] >= RETAINED_GPU_MIN_MEMORY_BYTES,
        f"retained {phase} gate found too little device memory: "
        f"{snapshot['total_memory_bytes']}",
    )
    require(
        len(snapshot["nvidia_smi_parsed_rows"]) == 1,
        f"retained {phase} GPU identity query returned "
        f"{len(snapshot['nvidia_smi_parsed_rows'])} rows",
    )
    identity = snapshot["nvidia_smi_parsed_rows"][0]
    require(
        identity["gpu_uuid"] == snapshot["nvidia_smi_query_id"]
        and identity["name"] == snapshot["name"],
        f"retained {phase} torch and nvidia-smi identities differ: "
        f"torch=({snapshot['name']!r}, "
        f"{snapshot['nvidia_smi_query_id']!r}), nvidia-smi={identity!r}",
    )
    require(
        snapshot["compute_apps_returncode"] == 0,
        f"retained {phase} GPU consumer query failed: "
        f"{snapshot['compute_apps_stderr']}",
    )
    require(
        all(
            row["gpu_uuid"] == snapshot["nvidia_smi_query_id"]
            for row in snapshot["compute_apps"]
        ),
        f"retained {phase} compute query returned another GPU identity",
    )
    require(
        not snapshot["foreign_compute_apps"],
        f"retained {phase} gate found foreign compute consumers on the "
        f"selected GPU: {snapshot['foreign_compute_apps']!r}",
    )


def validate_same_gpu_endpoints(before, after):
    stable_fields = (
        "index",
        "name",
        "nvidia_smi_query_id",
        "compute_capability",
        "total_memory_bytes",
        "multiprocessor_count",
    )
    for field in stable_fields:
        require(
            after[field] == before[field],
            f"selected GPU identity changed between endpoints: {field}",
        )
    require(
        before["nvidia_smi_parsed_rows"]
        == after["nvidia_smi_parsed_rows"],
        "nvidia-smi GPU identity changed between endpoints",
    )


def collect_provenance(
    repo_root,
    started_at,
    git_before,
    git_after,
    *,
    retained,
    gpu_before,
    gpu_after,
):
    return {
        "schema": EVIDENCE_SCHEMA,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "command_argv": list(getattr(sys, "orig_argv", sys.argv)),
        "command_shell": shlex.join(getattr(sys, "orig_argv", sys.argv)),
        "cwd": str(Path.cwd().resolve()),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "optimize": sys.flags.optimize,
        },
        "platform": {
            "node": platform.node(),
            "platform": platform.platform(),
            "uname": list(platform.uname()),
        },
        "source": {
            "nanovllm_import_origin": str(Path(nanovllm.__file__).resolve()),
            "git_before": git_before,
            "git_after": git_after,
            "unchanged_during_run": True,
        },
        "environment": {
            key: os.environ.get(key)
            for key in PROVENANCE_ENVIRONMENT_KEYS
        },
        "software": {
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "nccl": list(torch.cuda.nccl.version()),
            "transformers": transformers.__version__,
        },
        "gpu": {
            "retained_requirements": {
                "enforced": retained,
                "name": RETAINED_GPU_NAME,
                "compute_capability": list(
                    RETAINED_GPU_COMPUTE_CAPABILITY
                ),
                "minimum_total_memory_bytes": (
                    RETAINED_GPU_MIN_MEMORY_BYTES
                ),
                "foreign_consumers_at_endpoints": 0,
            },
            "before": gpu_before,
            "after": gpu_after,
            "same_selected_device": (
                gpu_before["nvidia_smi_query_id"]
                == gpu_after["nvidia_smi_query_id"]
            ),
            "foreign_compute_consumers_absent_at_endpoints": (
                not gpu_before["foreign_compute_apps"]
                and not gpu_after["foreign_compute_apps"]
            ),
            "isolation_scope": (
                "before/after endpoint checks; no claim that a transient "
                "mid-run consumer could not appear"
            ),
        },
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model")
    parser.add_argument("--mode", choices=("eager", "graph"), default="eager")
    parser.add_argument("--configured-k", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--check-explicit-boundary", action="store_true")
    parser.add_argument(
        "--expected-commit",
        help="full 40-character V2 commit required for retained evidence",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="permit an unchanged dirty tree for explicitly exploratory runs",
    )
    parser.add_argument(
        "--retained",
        action="store_true",
        help="require the full clean/exact-SHA/boundary artifact contract",
    )
    parser.add_argument(
        "--v0-golden",
        help="canonical V0 JSON artifact to compare with current spec-off",
    )
    parser.add_argument(
        "--v0-golden-sha256",
        help="required exact SHA-256 of --v0-golden",
    )
    parser.add_argument("--output")
    return parser.parse_args(argv)


def common_kwargs(args):
    return dict(
        enforce_eager=args.mode == "eager",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
    )


def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_snapshot():
    return (
        torch.random.get_rng_state().clone(),
        torch.cuda.get_rng_state().clone(),
    )


def tensor_sha256(tensor):
    payload = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def rng_snapshot_evidence(snapshot):
    return {
        "cpu_sha256": tensor_sha256(snapshot[0]),
        "cuda_sha256": tensor_sha256(snapshot[1]),
    }


def runtime_tokenizer_snapshot(tokenizer):
    vocab = tokenizer.get_vocab()
    vocab_payload = json.dumps(
        sorted((token, int(token_id)) for token, token_id in vocab.items()),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    backend = getattr(tokenizer, "backend_tokenizer", None)
    backend_json = None if backend is None else backend.to_str()
    return {
        "class": (
            f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"
        ),
        "length": len(tokenizer),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "vocab_sha256": hashlib.sha256(vocab_payload).hexdigest(),
        "vocab_entries": len(vocab),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "unk_token_id": getattr(tokenizer, "unk_token_id", None),
        "all_special_ids": list(getattr(tokenizer, "all_special_ids", ())),
        "backend_json_bytes": (
            None if backend_json is None else len(backend_json.encode("utf-8"))
        ),
        "backend_json_sha256": (
            None
            if backend_json is None
            else hashlib.sha256(backend_json.encode("utf-8")).hexdigest()
        ),
    }


def assert_same_rng(left, right):
    require(
        torch.equal(left[0], right[0]),
        "V2 changed CPU RNG state",
    )
    require(
        torch.equal(left[1], right[1]),
        "V2 changed CUDA RNG state",
    )


def assert_process_state(device, dtype, gc_enabled):
    require(
        not dist.is_initialized(),
        "runner left a process group initialized",
    )
    require(
        torch.get_default_device() == device,
        "runner changed the process default device",
    )
    require(
        torch.get_default_dtype() == dtype,
        "runner changed the process default dtype",
    )
    require(
        gc.isenabled() == gc_enabled,
        "runner changed the caller's Python-GC state",
    )
    require(
        llm_engine_module._PYTHON_GC_LEASE_COUNT == 0,
        "runner leaked a Python-GC lease",
    )
    require(
        llm_engine_module._PYTHON_GC_PRE_FIRST_ENABLED is None,
        "runner leaked the Python-GC baseline",
    )
    context = get_context()
    require(not context.is_prefill, "runner left a prefill context active")
    require(context.max_seqlen_q == 0, "runner left max_seqlen_q active")
    require(context.max_seqlen_k == 0, "runner left max_seqlen_k active")
    for name in (
        "cu_seqlens_q",
        "cu_seqlens_k",
        "slot_mapping",
        "context_lens",
        "block_tables",
    ):
        require(
            getattr(context, name) is None,
            f"runner left context.{name} active",
        )


def exit_idempotently(llm, device, dtype, gc_enabled):
    llm.exit()
    llm.exit()
    assert_process_state(device, dtype, gc_enabled)


def attach_trace(llm):
    trace = []
    original_call = llm.model_runner.call

    def traced_call(method_name, *args):
        if method_name == "run":
            seqs, is_prefill = args
            trace.append(
                (
                    bool(is_prefill),
                    tuple(
                        (
                            len(seq),
                            seq.num_cached_tokens,
                            seq.num_scheduled_tokens,
                            bool(seq.is_prefill),
                        )
                        for seq in seqs
                    ),
                )
            )
        return original_call(method_name, *args)

    llm.model_runner.call = traced_call
    return trace


def generate_fixture(llm, *, prove_draft_inert=False):
    trace = attach_trace(llm)
    draft_calls = []
    hook = None
    if prove_draft_inert:
        hook = llm.model_runner.draft_model.register_forward_pre_hook(
            lambda model, args: draft_calls.append(True)
        )
    try:
        results = llm.generate(
            [[1, 2, 3, 4], [7, 8, 9]],
            SamplingParams(
                temperature=0.0,
                max_tokens=4,
                ignore_eos=True,
            ),
            use_tqdm=False,
        )
    finally:
        if hook is not None:
            hook.remove()
    require(
        not draft_calls,
        "inert V2 generation executed the draft model",
    )
    outputs = tuple(
        (result["text"], tuple(result["token_ids"]))
        for result in results
    )
    return outputs, tuple(trace)


def nested_dict(value):
    if hasattr(value, "_asdict"):
        return {
            key: nested_dict(item)
            for key, item in value._asdict().items()
        }
    if is_dataclass(value):
        return nested_dict(asdict(value))
    if isinstance(value, dict):
        return {
            key: nested_dict(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [nested_dict(item) for item in value]
    if isinstance(value, torch.dtype):
        return str(value)
    return value


def assert_audit_matches_live_runner(
    llm,
    *,
    automatic_blocks,
    expected_blocks=None,
):
    runner = llm.model_runner
    audit = runner.speculative_memory_audit
    require(audit is not None, "speculative memory audit is missing")
    plan = audit.workspace_plan
    config = runner.config
    require(
        plan.configured_k == config.configured_k,
        "workspace plan configured K differs from runner config",
    )
    expected_effective_k = min(
        config.configured_k,
        config.max_num_batched_tokens - 1,
        config.max_model_len - 1,
    )
    expected_batch_size = (
        0
        if expected_effective_k == 0
        else min(
            config.max_num_seqs,
            config.max_num_batched_tokens // (expected_effective_k + 1),
        )
    )
    require(
        plan.max_effective_k == expected_effective_k,
        "workspace plan effective K does not match configured limits",
    )
    require(
        plan.batch_size == expected_batch_size,
        "workspace plan batch size does not match configured limits",
    )
    require(
        plan.probability_floor_bytes
        == plan.draft_probability_bytes + plan.target_probability_bytes,
        "workspace probability floor omits target or draft probabilities",
    )
    require(
        audit.joint_block_bytes
        == audit.target_block_bytes + audit.draft_block_bytes,
        "joint block price does not equal target plus draft prices",
    )

    target_shape = expected_cache_shape(
        config.hf_config,
        num_blocks=audit.selected_num_blocks,
        block_size=runner.block_size,
        world_size=runner.world_size,
    )
    draft_shape = expected_cache_shape(
        config.draft_hf_config,
        num_blocks=audit.selected_num_blocks,
        block_size=runner.block_size,
        world_size=runner.world_size,
    )
    require(
        tuple(runner.kv_cache.shape) == target_shape,
        "target KV tensor has the wrong geometry",
    )
    require(
        tuple(runner.draft_kv_cache.shape) == draft_shape,
        "draft KV tensor has the wrong geometry",
    )
    require(
        runner.kv_cache.dtype == config.hf_config.dtype,
        "target KV tensor has the wrong dtype",
    )
    require(
        runner.draft_kv_cache.dtype == config.draft_hf_config.dtype,
        "draft KV tensor has the wrong dtype",
    )
    target_cache_storage = runner.kv_cache.untyped_storage().data_ptr()
    draft_cache_storage = runner.draft_kv_cache.untyped_storage().data_ptr()
    require(
        target_cache_storage != draft_cache_storage,
        "target and draft KV tensors share physical storage",
    )
    require(
        audit.target_kv_bytes
        == runner.kv_cache.numel() * runner.kv_cache.element_size(),
        "target KV audit bytes do not match the live tensor",
    )
    require(
        audit.draft_kv_bytes
        == runner.draft_kv_cache.numel()
        * runner.draft_kv_cache.element_size(),
        "draft KV audit bytes do not match the live tensor",
    )
    require(
        audit.target_kv_bytes
        == audit.selected_num_blocks * audit.target_block_bytes,
        "target KV bytes do not reconcile with selected blocks",
    )
    require(
        audit.draft_kv_bytes
        == audit.selected_num_blocks * audit.draft_block_bytes,
        "draft KV bytes do not reconcile with selected blocks",
    )

    for model_name, model, hf_config, cache_storage in (
        ("target", runner.model, config.hf_config, target_cache_storage),
        (
            "draft",
            runner.draft_model,
            config.draft_hf_config,
            draft_cache_storage,
        ),
    ):
        cache_modules = [
            module
            for module in model.modules()
            if hasattr(module, "k_cache") and hasattr(module, "v_cache")
        ]
        require(
            len(cache_modules) == hf_config.num_hidden_layers,
            f"{model_name} bound the wrong number of KV layers",
        )
        for module in cache_modules:
            require(
                module.k_cache.untyped_storage().data_ptr() == cache_storage,
                f"{model_name} K-cache view is bound to the wrong tensor",
            )
            require(
                module.v_cache.untyped_storage().data_ptr() == cache_storage,
                f"{model_name} V-cache view is bound to the wrong tensor",
            )

    target_weight_storages = physical_parameter_storages(runner.model)
    draft_weight_storages = physical_parameter_storages(runner.draft_model)
    target_weight_bytes = sum(target_weight_storages.values())
    draft_weight_bytes = sum(draft_weight_storages.values())
    require(
        target_weight_storages.keys().isdisjoint(
            draft_weight_storages.keys()
        ),
        "target and draft model parameters share physical storage",
    )
    require(
        audit.target_weight_bytes == target_weight_bytes,
        "target physical weight ownership does not reconcile",
    )
    require(
        audit.draft_weight_bytes == draft_weight_bytes,
        "draft physical weight ownership does not reconcile",
    )
    require(
        audit.allocated_before_kv_bytes
        >= target_weight_bytes + draft_weight_bytes,
        "pre-KV allocation is smaller than live physical weights",
    )
    device_properties = torch.cuda.get_device_properties(runner.rank)
    require(
        audit.total_memory_bytes == device_properties.total_memory,
        "audit total memory differs from the live CUDA device",
    )
    require(
        audit.memory_budget_bytes
        == int(
            audit.total_memory_bytes * config.gpu_memory_utilization
        ),
        "audit memory budget differs from configured utilization",
    )
    require(
        audit.used_before_kv_bytes
        == audit.total_memory_bytes - audit.free_before_kv_bytes,
        "pre-KV used/free/total bytes do not reconcile",
    )
    require(
        audit.allocated_before_kv_bytes <= audit.reserved_before_kv_bytes,
        "pre-KV CUDA allocation exceeds reserved memory",
    )
    require(
        audit.peak_before_kv_bytes >= audit.allocated_before_kv_bytes,
        "pre-KV peak is below its current allocation",
    )
    require(
        audit.warmup_transient_bytes
        >= max(
            audit.target_warmup_transient_bytes,
            audit.draft_warmup_transient_bytes,
        ),
        "joint warmup transient omits a target or draft phase",
    )
    require(
        torch.cuda.memory_allocated() == audit.post_init_allocated_bytes,
        "live allocated bytes differ from the post-init audit baseline",
    )
    require(
        torch.cuda.memory_reserved() == audit.post_init_reserved_bytes,
        "live reserved bytes differ from the post-init audit baseline",
    )

    reconciliation = independent_memory_reconciliation(
        audit,
        enforce_eager=runner.enforce_eager,
    )
    require(
        audit.profiled_graph_ownership_bytes
        == reconciliation["profiled_graph_ownership_bytes"],
        "profiled graph ownership does not reconcile from raw deltas",
    )
    require(
        audit.profiled_graph_peak_bytes
        == reconciliation["profiled_graph_peak_bytes"],
        "profiled graph peak does not reconcile from raw deltas",
    )
    require(
        audit.graph_allocator_margin_bytes
        == reconciliation["graph_allocator_margin_bytes"],
        "graph allocator margin differs from independent policy",
    )
    require(
        audit.graph_construction_reservation_bytes
        == reconciliation["graph_construction_reservation_bytes"],
        "graph construction envelope does not reconcile",
    )
    require(
        audit.graph_reservation_bytes
        == audit.graph_construction_reservation_bytes,
        "graph-reservation compatibility alias diverged",
    )
    require(
        audit.runtime_reservation_bytes
        == reconciliation["runtime_reservation_bytes"],
        "runtime ownership/activation/workspace envelope does not reconcile",
    )
    require(
        audit.sizing_overhead_bytes
        == reconciliation["sizing_overhead_bytes"],
        "capacity sizing did not choose the larger independent envelope",
    )
    require(
        reconciliation["sizing_usable_bytes"] >= 0,
        "successful runner reports negative sizing capacity",
    )
    require(
        audit.selected_num_blocks
        <= reconciliation["automatic_num_blocks"],
        "selected blocks exceed independently recomputed capacity",
    )
    if automatic_blocks:
        require(
            audit.selected_num_blocks
            == reconciliation["automatic_num_blocks"],
            "automatic block selection does not equal the capacity floor",
        )
    if expected_blocks is not None:
        require(
            audit.selected_num_blocks == expected_blocks,
            "explicit block selection differs from the requested value",
        )
    require(
        audit.modeled_runtime_headroom_bytes
        == reconciliation["modeled_runtime_headroom_bytes"],
        "modeled runtime headroom is not independently reproducible",
    )
    require(
        reconciliation["modeled_runtime_headroom_bytes"] >= 0,
        "post-init headroom cannot cover activation plus workspace",
    )
    require(
        reconciliation["kv_allocated_increment_bytes"]
        == reconciliation["kv_accounted_bytes"],
        "post-KV allocation increment does not reconcile with both caches",
    )
    require(
        reconciliation["final_graph_allocated_increment_bytes"]
        == audit.final_graph_allocated_bytes,
        "final graph allocated delta does not reconcile with raw baselines",
    )
    require(
        reconciliation["final_graph_reserved_increment_bytes"]
        == audit.final_graph_reserved_bytes,
        "final graph reserved delta does not reconcile with raw baselines",
    )
    require(
        reconciliation["total_allocated_increment_bytes"]
        == reconciliation["total_accounted_increment_bytes"],
        "post-init allocated increment does not reconcile with KV plus graphs",
    )
    require(
        audit.final_graph_peak_allocated_bytes
        >= audit.final_graph_allocated_bytes,
        "final graph allocated peak is below graph ownership",
    )
    require(
        audit.final_graph_peak_reserved_bytes
        >= audit.final_graph_reserved_bytes,
        "final graph reserved peak is below graph ownership",
    )
    require(
        max(
            audit.final_graph_peak_allocated_bytes,
            audit.final_graph_peak_reserved_bytes,
        )
        <= audit.graph_construction_reservation_bytes,
        "final graph construction peak exceeds its profiled envelope",
    )

    if runner.enforce_eager:
        require(
            audit.profiled_graph_allocated_bytes == 0
            and audit.profiled_graph_reserved_bytes == 0
            and audit.profiled_graph_peak_allocated_bytes == 0
            and audit.profiled_graph_peak_reserved_bytes == 0
            and audit.profiled_graph_ownership_bytes == 0
            and audit.profiled_graph_peak_bytes == 0
            and audit.final_graph_allocated_bytes == 0
            and audit.final_graph_reserved_bytes == 0
            and audit.final_graph_peak_allocated_bytes == 0
            and audit.final_graph_peak_reserved_bytes == 0,
            "eager mode unexpectedly owns CUDA graphs",
        )
        require(
            audit.allocated_after_kv_before_graph_bytes
            == audit.allocated_after_graph_before_pretouch_bytes
            and audit.reserved_after_kv_before_graph_bytes
            == audit.reserved_after_graph_before_pretouch_bytes,
            "eager post-KV baselines changed without graph capture",
        )
    else:
        require(
            audit.profiled_graph_allocated_bytes > 0,
            "graph profile recorded no allocated bytes",
        )
        require(
            audit.final_graph_allocated_bytes > 0,
            "final graph capture recorded no allocated bytes",
        )
        require(
            isinstance(runner.graphs, dict) and runner.graphs,
            "target decode graphs are missing",
        )
        require(
            isinstance(runner.draft_graphs, dict) and runner.draft_graphs,
            "draft decode graphs are missing",
        )
        require(
            set(runner.draft_graphs) == set(runner.graphs),
            "draft and target graph keys differ",
        )
        require(
            runner.draft_graph_pool != runner.graph_pool,
            "draft and target graphs share the same CUDA graph-pool handle",
        )
        for batch_size in runner.graphs:
            require(
                runner.draft_graphs[batch_size]
                is not runner.graphs[batch_size],
                f"draft and target graph objects alias at batch {batch_size}",
            )
        expected_graph_dtypes = {
            "input_ids": torch.int64,
            "positions": torch.int64,
            "slot_mapping": torch.int32,
            "context_lens": torch.int32,
            "block_tables": torch.int32,
        }
        require(
            set(runner.draft_graph_vars) == set(runner.graph_vars),
            "draft and target graph static-buffer keys differ",
        )
        for name, dtype in expected_graph_dtypes.items():
            require(
                runner.draft_graph_vars[name].dtype == dtype,
                f"draft graph {name} has the wrong dtype",
            )
            require(
                runner.graph_vars[name].dtype == dtype,
                f"target graph {name} has the wrong dtype",
            )
        require(
            runner.draft_graph_vars["outputs"].dtype
            == config.draft_hf_config.dtype,
            "draft graph output has the wrong dtype",
        )
        require(
            runner.graph_vars["outputs"].dtype == config.hf_config.dtype,
            "target graph output has the wrong dtype",
        )
        require(
            runner.draft_graph_vars["outputs"].shape[1]
            == config.draft_hf_config.hidden_size,
            "draft graph output has the wrong hidden geometry",
        )
        require(
            runner.graph_vars["outputs"].shape[1]
            == config.hf_config.hidden_size,
            "target graph output has the wrong hidden geometry",
        )
        for name in runner.graph_vars:
            require(
                runner.draft_graph_vars[name]
                .untyped_storage()
                .data_ptr()
                != runner.graph_vars[name].untyped_storage().data_ptr(),
                f"draft and target graph {name} buffers share storage",
            )

    require(
        audit.gpu_certified is False,
        "constructor must not self-certify provisional workspace modeling",
    )
    require(
        plan.graph_static_workspace_bytes is None
        and plan.backend_library_workspace_bytes is None,
        "unmeasured workspace components were mislabeled as measured",
    )
    require(
        bool(audit.audit_required_components),
        "audit-required GPU components were not disclosed",
    )
    return audit, reconciliation


def construct_speculative(args, draft_model, *, num_blocks=-1):
    return LLM(
        args.model,
        draft_model=draft_model,
        num_speculative_tokens=args.configured_k,
        num_kvcache_blocks=num_blocks,
        **common_kwargs(args),
    )


def main(argv=None):
    # This must run before argparse (and before any CUDA/process-group work) so
    # ``python -O`` cannot silently turn a certification run into a no-op.
    reject_optimized_python()
    started_at = datetime.now(timezone.utc).isoformat()
    repo_root = Path(__file__).resolve().parents[1]
    args = parse_args(argv)
    draft_model = args.draft_model or args.model
    package_origin = Path(nanovllm.__file__).resolve()
    package_root = package_origin.parent.parent
    require(
        package_root == repo_root,
        "imported nanovllm does not come from the V2 evidence-runner checkout: "
        f"package={package_origin}, runner={repo_root}",
    )
    output_path = (
        None
        if args.output is None
        else Path(args.output).expanduser().resolve()
    )
    if output_path is not None:
        require(
            not output_path.exists(),
            f"refusing to overwrite existing output: {output_path}",
        )
        require(
            not path_is_within(output_path, repo_root),
            "V2 evidence output must be outside the source checkout",
        )
    expected_commit = (
        None
        if args.expected_commit is None
        else args.expected_commit.lower()
    )
    if args.retained:
        require(output_path is not None, "--retained requires --output")
        require(
            expected_commit is not None,
            "--retained requires --expected-commit",
        )
        require(
            args.check_explicit_boundary,
            "--retained requires --check-explicit-boundary",
        )
        require(
            not args.allow_dirty,
            "--retained cannot be combined with --allow-dirty",
        )
        require(
            args.v0_golden is not None
            and args.v0_golden_sha256 is not None,
            "--retained requires --v0-golden and --v0-golden-sha256",
        )
    require(
        (args.v0_golden is None) == (args.v0_golden_sha256 is None),
        "--v0-golden and --v0-golden-sha256 must be provided together",
    )
    git_before = git_snapshot(repo_root)
    validate_source_snapshot(
        git_before,
        expected_commit=expected_commit,
        allow_dirty=args.allow_dirty,
    )
    target_model_argument = args.model
    target_model_path, target_model_evidence = model_snapshot(
        target_model_argument
    )
    unresolved_draft_model = draft_model
    if Path(draft_model).expanduser().resolve() == target_model_path:
        draft_model_path = target_model_path
        draft_model_evidence = {
            **target_model_evidence,
            "argument": unresolved_draft_model,
        }
    else:
        draft_model_path, draft_model_evidence = model_snapshot(draft_model)
    if output_path is not None:
        require(
            not path_is_within(output_path, target_model_path)
            and not path_is_within(output_path, draft_model_path),
            "V2 evidence output must be outside target/draft model directories",
        )
    args.model = str(target_model_path)
    draft_model = str(draft_model_path)
    original_device = torch.get_default_device()
    original_dtype = torch.get_default_dtype()
    original_gc_enabled = gc.isenabled()
    seed = 20260828
    gpu_before = establish_gpu_gate_before(retained=args.retained)

    # This is the current source tree with speculation disabled.  It is not a
    # historical V0 checkout, so the evidence must not label it as V0 parity.
    seed_all(seed)
    speculation_off = LLM(args.model, **common_kwargs(args))
    speculation_off_runner = speculation_off.model_runner
    require(
        not speculation_off_runner.config.speculation_enabled,
        "speculation-off control unexpectedly enabled speculation",
    )
    require(
        not hasattr(speculation_off_runner, "draft_model"),
        "K=0 path unexpectedly constructed a draft model",
    )
    require(
        not hasattr(speculation_off_runner, "draft_kv_cache"),
        "K=0 path unexpectedly allocated a draft KV cache",
    )
    require(
        not hasattr(speculation_off_runner, "speculative_memory_audit"),
        "K=0 path unexpectedly constructed a speculative memory audit",
    )
    speculation_off_construction_rng = rng_snapshot()
    speculation_off_outputs, speculation_off_trace = generate_fixture(
        speculation_off
    )
    speculation_off_tokenizer = runtime_tokenizer_snapshot(
        speculation_off.tokenizer
    )
    speculation_off_runtime_rng = rng_snapshot()
    speculation_off_blocks = (
        speculation_off.model_runner.config.num_kvcache_blocks
    )
    exit_idempotently(
        speculation_off,
        original_device,
        original_dtype,
        original_gc_enabled,
    )
    v0_comparison = None
    if args.v0_golden is not None:
        v0_comparison = validate_v0_golden(
            Path(args.v0_golden).expanduser().resolve(),
            expected_sha256=args.v0_golden_sha256,
            expected_runner_commit=(
                expected_commit or git_before["head"]
            ),
            args=args,
            target_model_evidence=target_model_evidence,
            tokenizer_evidence=speculation_off_tokenizer,
            outputs=speculation_off_outputs,
            scheduler_trace=speculation_off_trace,
        )

    seed_all(seed)
    speculative = construct_speculative(args, draft_model)
    tokenizer_fingerprint = getattr(
        speculative,
        "speculative_tokenizer_fingerprint",
        None,
    )
    require(
        isinstance(tokenizer_fingerprint, str)
        and re.fullmatch(r"[0-9a-f]{64}", tokenizer_fingerprint) is not None,
        "speculative runner did not retain a valid tokenizer fingerprint",
    )
    speculative_tokenizer = runtime_tokenizer_snapshot(speculative.tokenizer)
    require(
        speculative_tokenizer == speculation_off_tokenizer,
        "speculation-on and speculation-off runtime tokenizers differ",
    )
    speculative_construction_rng = rng_snapshot()
    assert_same_rng(
        speculation_off_construction_rng,
        speculative_construction_rng,
    )
    audit, reconciliation = assert_audit_matches_live_runner(
        speculative,
        automatic_blocks=True,
    )
    speculative_outputs, speculative_trace = generate_fixture(
        speculative,
        prove_draft_inert=True,
    )
    speculative_runtime_rng = rng_snapshot()
    assert_same_rng(speculation_off_runtime_rng, speculative_runtime_rng)
    require(
        speculative_outputs == speculation_off_outputs,
        "inert V2 changed greedy target outputs",
    )
    require(
        speculative_trace == speculation_off_trace,
        "inert V2 changed the target scheduler/run trace",
    )

    # Keep half-configuration rejection separate from the real one-runner-per-
    # process guard: the former fails in Config, before runner construction.
    try:
        LLM(
            args.model,
            num_speculative_tokens=args.configured_k,
            **common_kwargs(args),
        )
    except ValueError as error:
        half_config_error = {
            "type": f"{type(error).__module__}.{type(error).__qualname__}",
            "message": str(error),
        }
        require(
            "requires draft_model" in str(error),
            "half-configured rejection had the wrong diagnostic",
        )
    else:
        raise AssertionError("half-configured speculation unexpectedly succeeded")
    require(
        dist.is_initialized(),
        "half-configured rejection damaged the healthy process group",
    )

    # A fully valid second engine must fail while the first remains healthy.
    healthy_runner = speculative.model_runner
    try:
        construct_speculative(args, draft_model)
    except (RuntimeError, ValueError) as error:
        second_engine_error = str(error)
        second_engine_error_evidence = {
            "type": f"{type(error).__module__}.{type(error).__qualname__}",
            "message": second_engine_error,
        }
        require(
            "process group" in second_engine_error.lower()
            or "one engine" in second_engine_error.lower()
            or "already initialized" in second_engine_error.lower(),
            "valid second-engine rejection had the wrong diagnostic: "
            f"{second_engine_error}",
        )
    else:
        raise AssertionError("a valid second engine unexpectedly succeeded")
    require(
        dist.is_initialized(),
        "valid second-engine rejection destroyed the healthy process group",
    )
    require(
        speculative.model_runner is healthy_runner,
        "valid second-engine rejection replaced the healthy runner",
    )
    healthy_outputs, healthy_trace = generate_fixture(
        speculative,
        prove_draft_inert=True,
    )
    require(
        healthy_outputs == speculation_off_outputs,
        "healthy engine changed after valid second-engine rejection",
    )
    require(
        healthy_trace == speculation_off_trace,
        "healthy engine trace changed after second-engine rejection",
    )

    selected_blocks = audit.selected_num_blocks
    evidence = {
        "evidence_schema": EVIDENCE_SCHEMA,
        "mode": args.mode,
        "model": str(Path(args.model).resolve()),
        "draft_model": str(Path(draft_model).resolve()),
        "model_artifacts": {
            "target": target_model_evidence,
            "draft": draft_model_evidence,
        },
        "configured_k": args.configured_k,
        "certification_mode": (
            "retained" if args.retained else "exploratory"
        ),
        "speculative_tokenizer_fingerprint": tokenizer_fingerprint,
        "source_policy": {
            "expected_commit": expected_commit,
            "allow_dirty": args.allow_dirty,
        },
        "retention_eligible": args.retained,
        "parity_scope": (
            "current source tree speculation-off/on parity plus strict "
            "canonical historical V0 parity when historical_v0_comparison "
            "is present"
        ),
        "historical_v0_comparison": v0_comparison,
        "scope": {
            "feature_stage": "V2 inert dual-model ownership",
            "workload": {
                "prompt_token_ids": [[1, 2, 3, 4], [7, 8, 9]],
                "sampling": {
                    "temperature": 0.0,
                    "max_tokens": 4,
                    "ignore_eos": True,
                },
            },
            "tensor_parallel_size": 1,
            "limitations": [
                "draft forward is intentionally not executed during generation",
                "no speculative proposal, verification, acceptance, or commit path",
                "no latency, throughput, acceptance-rate, or speedup claim",
                "no tensor-parallel or FlashInfer certification",
                "memory workspace is modeled and reserved, not yet allocated",
                "memory audit remains gpu_certified=false",
                "the speculation-off control runs first, so this lifecycle gate makes no cold-start, first-cycle compile, or cache-isolation claim",
                "GPU process isolation is checked only at the before/after endpoints and cannot exclude a transient mid-run consumer",
            ],
        },
        "speculation_off_blocks": speculation_off_blocks,
        "speculative_memory_audit": nested_dict(audit),
        "independent_memory_reconciliation": reconciliation,
        "speculation_off_outputs": nested_dict(speculation_off_outputs),
        "speculation_off_scheduler_trace": nested_dict(
            speculation_off_trace
        ),
        "speculation_on_outputs": nested_dict(speculative_outputs),
        "speculation_on_scheduler_trace": nested_dict(speculative_trace),
        "post_second_engine_outputs": nested_dict(healthy_outputs),
        "post_second_engine_scheduler_trace": nested_dict(healthy_trace),
        "runtime_tokenizers": {
            "speculation_off": speculation_off_tokenizer,
            "speculation_on": speculative_tokenizer,
        },
        "rng_snapshots": {
            "speculation_off_after_construction": rng_snapshot_evidence(
                speculation_off_construction_rng
            ),
            "speculation_on_after_construction": rng_snapshot_evidence(
                speculative_construction_rng
            ),
            "speculation_off_after_generation": rng_snapshot_evidence(
                speculation_off_runtime_rng
            ),
            "speculation_on_after_generation": rng_snapshot_evidence(
                speculative_runtime_rng
            ),
        },
        "construction_rng_identity": True,
        "post_generation_rng_identity": True,
        "inert_output_identity": True,
        "inert_scheduler_trace_identity": True,
        "draft_forward_calls_during_generation": 0,
        "half_config_rejected_while_healthy": True,
        "half_config_error": half_config_error,
        "valid_second_engine_rejected_while_healthy": True,
        "valid_second_engine_error": second_engine_error_evidence,
        "explicit_boundary_requested": args.check_explicit_boundary,
        "exit_idempotency": True,
    }
    if draft_model_path == target_model_path:
        evidence["scope"]["limitations"].append(
            "the retained same-model target/draft fixture proves independent "
            "dual ownership but not heterogeneous-model GPU geometry"
        )
    exit_idempotently(
        speculative,
        original_device,
        original_dtype,
        original_gc_enabled,
    )

    if args.check_explicit_boundary:
        explicit = construct_speculative(
            args,
            draft_model,
            num_blocks=selected_blocks,
        )
        explicit_audit, explicit_reconciliation = (
            assert_audit_matches_live_runner(
                explicit,
                automatic_blocks=False,
                expected_blocks=selected_blocks,
            )
        )
        require(
            explicit.speculative_tokenizer_fingerprint
            == tokenizer_fingerprint,
            "explicit-boundary runner changed tokenizer identity",
        )
        explicit_outputs, explicit_trace = generate_fixture(
            explicit,
            prove_draft_inert=True,
        )
        require(
            explicit_outputs == speculation_off_outputs,
            "explicit-boundary runner changed greedy outputs",
        )
        require(
            explicit_trace == speculation_off_trace,
            "explicit-boundary runner changed scheduler trace",
        )
        exit_idempotently(
            explicit,
            original_device,
            original_dtype,
            original_gc_enabled,
        )

        # The exact first block beyond the certified automatic capacity must
        # be rejected by typed preflight rather than by torch.empty or a CUDA
        # OOM.  The diagnostic must confirm that the rerun measured the same
        # ceiling, otherwise this is not a first-ineligible certificate.
        first_ineligible_blocks = selected_blocks + 1
        try:
            construct_speculative(
                args,
                draft_model,
                num_blocks=first_ineligible_blocks,
            )
        except SpeculativeKVCacheCapacityError as error:
            first_ineligible_capacity_error = {
                "type": (
                    f"{type(error).__module__}.{type(error).__qualname__}"
                ),
                "message": str(error),
            }
            require(
                (
                    "requested num_kvcache_blocks="
                    f"{first_ineligible_blocks}"
                ) in str(error),
                "typed explicit-capacity failure had the wrong diagnostic",
            )
            require(
                f"({selected_blocks} blocks)" in str(error),
                "explicit-capacity rerun did not retain the certified "
                f"{selected_blocks}-block ceiling",
            )
        else:
            raise AssertionError(
                "first-ineligible explicit dual cache unexpectedly fit"
            )
        assert_process_state(
            original_device,
            original_dtype,
            original_gc_enabled,
        )

        recovery_blocks = min(
            selected_blocks,
            max(args.max_num_seqs, 2),
        )
        recovery = construct_speculative(
            args,
            draft_model,
            num_blocks=recovery_blocks,
        )
        recovery_audit, recovery_reconciliation = (
            assert_audit_matches_live_runner(
                recovery,
                automatic_blocks=False,
                expected_blocks=recovery_blocks,
            )
        )
        require(
            recovery.speculative_tokenizer_fingerprint
            == tokenizer_fingerprint,
            "post-failure recovery changed tokenizer identity",
        )
        recovery_outputs, recovery_trace = generate_fixture(
            recovery,
            prove_draft_inert=True,
        )
        require(
            recovery_outputs == speculation_off_outputs,
            "post-failure recovery changed greedy outputs",
        )
        require(
            recovery_trace == speculation_off_trace,
            "post-failure recovery changed scheduler trace",
        )
        exit_idempotently(
            recovery,
            original_device,
            original_dtype,
            original_gc_enabled,
        )
        evidence["explicit_selected_blocks_passed"] = selected_blocks
        evidence["explicit_boundary"] = {
            "selected_blocks": selected_blocks,
            "outputs": nested_dict(explicit_outputs),
            "scheduler_trace": nested_dict(explicit_trace),
            "tokenizer_fingerprint": (
                explicit.speculative_tokenizer_fingerprint
            ),
        }
        evidence["explicit_memory_audit"] = nested_dict(explicit_audit)
        evidence["explicit_memory_reconciliation"] = (
            explicit_reconciliation
        )
        evidence["first_ineligible_blocks_rejected"] = (
            first_ineligible_blocks
        )
        evidence["first_ineligible_capacity_error"] = (
            first_ineligible_capacity_error
        )
        evidence["post_failure_recovery"] = True
        evidence["post_failure_recovery_run"] = {
            "selected_blocks": recovery_blocks,
            "outputs": nested_dict(recovery_outputs),
            "scheduler_trace": nested_dict(recovery_trace),
            "tokenizer_fingerprint": (
                recovery.speculative_tokenizer_fingerprint
            ),
        }
        evidence["recovery_memory_audit"] = nested_dict(recovery_audit)
        evidence["recovery_memory_reconciliation"] = (
            recovery_reconciliation
        )

    gpu_after = gpu_gate_snapshot(
        phase="after",
        own_gpu_process_ids=gpu_before["own_gpu_process_ids"],
    )
    if args.retained:
        validate_retained_gpu_snapshot(gpu_after, phase="after")
        validate_same_gpu_endpoints(gpu_before, gpu_after)

    _, target_model_after = model_snapshot(target_model_argument)
    if draft_model_path == target_model_path:
        draft_model_after = {
            **target_model_after,
            "argument": unresolved_draft_model,
        }
    else:
        _, draft_model_after = model_snapshot(unresolved_draft_model)
    require(
        target_model_after == target_model_evidence,
        "target model artifacts changed during the V2 run",
    )
    require(
        draft_model_after == draft_model_evidence,
        "draft model artifacts changed during the V2 run",
    )
    evidence["model_artifacts"] = {
        "target": {
            "before": target_model_evidence,
            "after": target_model_after,
            "unchanged_during_run": True,
        },
        "draft": {
            "before": draft_model_evidence,
            "after": draft_model_after,
            "unchanged_during_run": True,
        },
        "same_resolved_model": draft_model_path == target_model_path,
    }

    git_after = git_snapshot(repo_root)
    validate_source_snapshot(
        git_after,
        expected_commit=expected_commit,
        allow_dirty=args.allow_dirty,
    )
    validate_unchanged(git_before, git_after)
    evidence["provenance"] = collect_provenance(
        repo_root,
        started_at,
        git_before,
        git_after,
        retained=args.retained,
        gpu_before=gpu_before,
        gpu_after=gpu_after,
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_sha256 = write_json_exclusive(output_path, evidence)
        print(f"artifact sha256: {artifact_sha256}")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    print("speculative V2 inert lifecycle fixture: PASS")


if __name__ == "__main__":
    main()
