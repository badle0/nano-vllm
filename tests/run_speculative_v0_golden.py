"""Produce an independent, write-once speculation-off golden from frozen V0.

Run this script itself from the implementation checkout, but put the detached
V0 checkout first on ``PYTHONPATH``.  For example::

    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace/nano-vllm-v0 \
    /venv/main/bin/python tests/run_speculative_v0_golden.py \
      --model /workspace/models/Qwen3-0.6B \
      --output /workspace/spec-evidence/v0-eager.json

The imported nano-vLLM tree, rather than this script's location, defines the
source checkout whose provenance is certified.  The evidence file is published
only after the run and the second provenance check succeed, and an existing
path is never overwritten.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


EVIDENCE_SCHEMA = "nano-vllm-speculative-v0-golden-v1"
CANONICAL_V0_COMMIT = "480a3b26c5a4e465aac06d1dabd34e1230686feb"
DEFAULT_SEED = 20260828
PROVENANCE_ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "HF_HOME",
    "PYTHONHASHSEED",
    "PYTHONPATH",
    "PYTHONDONTWRITEBYTECODE",
    "PYTORCH_CUDA_ALLOC_CONF",
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE",
    "TORCHINDUCTOR_AUTOGRAD_CACHE",
    "TORCH_COMPILE_DISABLE",
    "TORCH_LOGS",
    "TRITON_CACHE_DIR",
)
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


def require(condition, message):
    """Certification check that remains active under optimized Python."""

    if not condition:
        raise AssertionError(message)


def reject_optimized_python():
    require(
        __debug__,
        "the V0 golden runner refuses optimized Python because certification "
        "checks must remain active",
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate a clean detached-V0 greedy golden artifact."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--expected-commit",
        default=CANONICAL_V0_COMMIT,
        help="full detached V0 commit required for this run",
    )
    parser.add_argument(
        "--expected-runner-commit",
        required=True,
        help="full clean commit containing this evidence runner",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--mode",
        choices=("eager", "graph"),
        default="eager",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--kvcache-block-size", type=int, default=256)
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def _git(repo_root, *args, check=True):
    return subprocess.run(
        ("git", *args),
        cwd=repo_root,
        check=check,
        capture_output=True,
        text=True,
    )


def git_snapshot(repo_root):
    """Return exact commit, tree, branch, and worktree state."""

    top_level = Path(
        _git(repo_root, "rev-parse", "--show-toplevel").stdout.strip()
    ).resolve()
    branch_result = _git(
        repo_root,
        "symbolic-ref",
        "--quiet",
        "--short",
        "HEAD",
        check=False,
    )
    require(
        branch_result.returncode in (0, 1),
        "git could not determine whether the source checkout is detached: "
        f"{branch_result.stderr.strip()}",
    )
    branch = (
        branch_result.stdout.strip()
        if branch_result.returncode == 0
        else None
    )
    status = _git(
        repo_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).stdout.splitlines()
    return {
        "repository": str(top_level),
        "head": _git(repo_root, "rev-parse", "HEAD").stdout.strip(),
        "tree": _git(repo_root, "rev-parse", "HEAD^{tree}").stdout.strip(),
        "branch": branch,
        "detached": branch is None,
        "dirty": bool(status),
        "status_porcelain_v1": status,
        "commit_timestamp": _git(
            repo_root,
            "show",
            "-s",
            "--format=%cI",
            "HEAD",
        ).stdout.strip(),
        "commit_subject": _git(
            repo_root,
            "show",
            "-s",
            "--format=%s",
            "HEAD",
        ).stdout.strip(),
    }


def validate_v0_snapshot(snapshot, expected_commit):
    require(
        re.fullmatch(r"[0-9a-fA-F]{40}", expected_commit) is not None,
        "--expected-commit must be a full 40-character hexadecimal Git SHA",
    )
    require(
        snapshot["head"].lower() == expected_commit.lower(),
        "imported nano-vLLM is not at the expected V0 commit: "
        f"expected {expected_commit}, found {snapshot['head']}",
    )
    require(
        snapshot["detached"],
        "the V0 source checkout must be detached, not on branch "
        f"{snapshot['branch']!r}",
    )
    require(
        not snapshot["dirty"],
        "the V0 source checkout must be clean; status is "
        f"{snapshot['status_porcelain_v1']!r}",
    )


def validate_runner_snapshot(snapshot, expected_commit):
    require(
        re.fullmatch(r"[0-9a-fA-F]{40}", expected_commit) is not None,
        "--expected-runner-commit must be a full 40-character hexadecimal Git SHA",
    )
    require(
        snapshot["head"].lower() == expected_commit.lower(),
        "evidence runner checkout is not at the expected commit: "
        f"expected {expected_commit}, found {snapshot['head']}",
    )
    require(
        not snapshot["dirty"],
        "evidence runner checkout must be clean; status is "
        f"{snapshot['status_porcelain_v1']!r}",
    )


def validate_unchanged(before, after, *, role="V0 source"):
    for key in ("repository", "head", "tree", "branch", "detached"):
        require(
            after[key] == before[key],
            f"{role} provenance changed during the run: {key}",
        )
    require(
        after["status_porcelain_v1"] == before["status_porcelain_v1"],
        f"{role} worktree state changed during the run",
    )
    require(not after["dirty"], f"{role} checkout became dirty during the run")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_snapshot(model_argument):
    model_path = Path(model_argument).expanduser().resolve()
    require(model_path.is_dir(), f"model path is not a directory: {model_path}")
    config_path = model_path / "config.json"
    require(config_path.is_file(), f"model config is missing: {config_path}")

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
        "vocab_entries": len(vocab),
        "vocab_sha256": hashlib.sha256(vocab_payload).hexdigest(),
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


def path_is_within(path, directory):
    return path == directory or directory in path.parents


def attach_scheduler_trace(llm):
    """Observe the exact pre-run scheduler state through the frozen V0 seam."""

    trace = []
    original_call = llm.model_runner.call

    def traced_call(method_name, *args):
        if method_name == "run":
            require(
                len(args) == 2,
                "V0 ModelRunner.run observer received an unexpected signature",
            )
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
    return trace, original_call


def effective_config_snapshot(config):
    hf_config = config.hf_config
    return {
        "model": config.model,
        "max_num_batched_tokens": config.max_num_batched_tokens,
        "max_num_seqs": config.max_num_seqs,
        "max_model_len": config.max_model_len,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "tensor_parallel_size": config.tensor_parallel_size,
        "enforce_eager": config.enforce_eager,
        "eos": config.eos,
        "kvcache_block_size": config.kvcache_block_size,
        "num_kvcache_blocks": config.num_kvcache_blocks,
        "top_p_backend": config.top_p_backend,
        "disable_python_gc": config.disable_python_gc,
        "hf_config": {
            "model_type": getattr(hf_config, "model_type", None),
            "vocab_size": getattr(hf_config, "vocab_size", None),
            "max_position_embeddings": getattr(
                hf_config,
                "max_position_embeddings",
                None,
            ),
            "dtype": str(getattr(hf_config, "dtype", None)),
        },
    }


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_runtime(torch, transformers):
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    gpu_uuid = getattr(properties, "uuid", None)
    nvidia_smi = subprocess.run(
        (
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        nccl_version = torch.cuda.nccl.version()
        if isinstance(nccl_version, tuple):
            nccl_version = list(nccl_version)
    except Exception as error:
        nccl_version = {"unavailable": f"{type(error).__name__}: {error}"}
    return {
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
        "environment": {
            key: os.environ.get(key) for key in PROVENANCE_ENVIRONMENT_KEYS
        },
        "software": {
            "nano_vllm_distribution": package_version("nano-vllm"),
            "torch": str(torch.__version__),
            "torch_git_version": torch.version.git_version,
            "torch_cuda_build": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "nccl": nccl_version,
            "transformers": transformers.__version__,
            "triton": package_version("triton"),
            "flash_attn": package_version("flash-attn"),
        },
        "numeric_runtime": {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        },
        "gpu": {
            "current_device": torch.cuda.current_device(),
            "name": properties.name,
            "uuid": None if gpu_uuid is None else str(gpu_uuid),
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
            "multiprocessor_count": properties.multi_processor_count,
            "nvidia_smi": nvidia_smi.stdout.strip().splitlines(),
            "nvidia_smi_returncode": nvidia_smi.returncode,
            "nvidia_smi_stderr": nvidia_smi.stderr.strip(),
        },
    }


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
        raise RuntimeError(f"refusing to overwrite existing output: {path}") from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
    return hashlib.sha256(serialized).hexdigest()


def main(argv=None):
    reject_optimized_python()
    args = parse_args(argv)
    started_at = datetime.now(timezone.utc).isoformat()

    output_path = Path(args.output).expanduser().resolve()
    require(
        not output_path.exists(),
        f"refusing to overwrite existing output: {output_path}",
    )
    expected_commit = args.expected_commit.lower()
    require(
        re.fullmatch(r"[0-9a-f]{40}", expected_commit) is not None,
        "--expected-commit must be a full 40-character hexadecimal Git SHA",
    )
    require(
        expected_commit == CANONICAL_V0_COMMIT,
        "retained V0 golden must use the canonical frozen commit "
        f"{CANONICAL_V0_COMMIT}, not {expected_commit}",
    )
    expected_runner_commit = args.expected_runner_commit.lower()
    runner_root = Path(__file__).resolve().parents[1]
    runner_git_before = git_snapshot(runner_root)
    validate_runner_snapshot(runner_git_before, expected_runner_commit)
    runner_script_before = sha256_file(Path(__file__).resolve())
    model_path, model_evidence = model_snapshot(args.model)
    require(
        not path_is_within(output_path, runner_root),
        "the evidence output must be outside the runner checkout",
    )
    require(
        not path_is_within(output_path, model_path),
        "the evidence output must be outside the model directory",
    )

    import nanovllm
    import torch
    import torch.distributed as dist
    import transformers
    from nanovllm import LLM, SamplingParams

    package_origin = Path(nanovllm.__file__).resolve()
    source_candidate = package_origin.parent.parent
    source_root = Path(
        _git(source_candidate, "rev-parse", "--show-toplevel").stdout.strip()
    ).resolve()
    require(
        source_root == source_candidate,
        "imported nanovllm package is not rooted directly in its Git checkout: "
        f"package={package_origin}, git={source_root}",
    )
    require(
        source_root != runner_root,
        "canonical V0 source and the committed evidence runner must be "
        "separate checkouts",
    )
    require(
        not path_is_within(output_path, source_root),
        "the evidence output must be outside the detached V0 source checkout",
    )
    git_before = git_snapshot(source_root)
    validate_v0_snapshot(git_before, expected_commit)

    require(torch.cuda.is_available(), "the V0 golden runner requires CUDA")
    require(type(args.seed) is int, "--seed must be an integer")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    prompts = [[1, 2, 3, 4], [7, 8, 9]]
    sampling_config = {
        "temperature": 0.0,
        "max_tokens": 4,
        "ignore_eos": True,
        "top_k": -1,
        "top_p": 1.0,
    }
    engine_kwargs = {
        "enforce_eager": args.mode == "eager",
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "tensor_parallel_size": 1,
        "kvcache_block_size": args.kvcache_block_size,
        "num_kvcache_blocks": -1,
        "top_p_backend": "exact",
        "disable_python_gc": False,
    }

    llm = None
    original_call = None
    scheduler_trace = None
    effective_config = None
    normalized_outputs = None
    tokenizer_evidence = None
    exit_calls = 0
    try:
        llm = LLM(str(model_path), **engine_kwargs)
        config = llm.model_runner.config
        tokenizer_evidence = runtime_tokenizer_snapshot(llm.tokenizer)
        effective_config = effective_config_snapshot(config)
        require(
            config.enforce_eager == engine_kwargs["enforce_eager"],
            "V0 engine mode differs from the requested mode",
        )
        require(
            config.tensor_parallel_size == 1,
            "V0 golden must use tensor_parallel_size=1",
        )
        require(
            config.top_p_backend == "exact",
            "V0 golden must use the exact sampling backend",
        )
        require(
            type(config.num_kvcache_blocks) is int
            and config.num_kvcache_blocks > 0,
            "V0 automatic KV sizing did not select a positive block count",
        )
        vocab_size = config.hf_config.vocab_size
        require(
            all(
                type(token_id) is int and 0 <= token_id < vocab_size
                for prompt in prompts
                for token_id in prompt
            ),
            "golden fixture token IDs are outside the target vocabulary",
        )

        scheduler_trace, original_call = attach_scheduler_trace(llm)
        try:
            results = llm.generate(
                prompts,
                SamplingParams(**sampling_config),
                use_tqdm=False,
            )
        finally:
            if original_call is not None and llm.model_runner is not None:
                llm.model_runner.call = original_call
                original_call = None

        require(
            len(results) == len(prompts),
            "V0 generate returned the wrong number of results",
        )
        normalized_outputs = []
        for index, result in enumerate(results):
            require(isinstance(result, dict), "V0 result must be a dictionary")
            text = result.get("text")
            token_ids = result.get("token_ids")
            require(isinstance(text, str), "V0 result text must be a string")
            require(isinstance(token_ids, list), "V0 token_ids must be a list")
            require(
                len(token_ids) == sampling_config["max_tokens"],
                f"V0 result {index} emitted an unexpected token count",
            )
            require(
                all(type(token_id) is int for token_id in token_ids),
                "V0 completion token IDs must be integers",
            )
            require(
                text == llm.tokenizer.decode(token_ids),
                "V0 result text does not decode from its recorded token IDs",
            )
            normalized_outputs.append(
                {"index": index, "text": text, "token_ids": list(token_ids)}
            )
        require(bool(scheduler_trace), "V0 scheduler trace is empty")
        require(llm.is_finished(), "V0 scheduler is not empty after generate")
    finally:
        if llm is not None:
            if original_call is not None and llm.model_runner is not None:
                llm.model_runner.call = original_call
            llm.exit()
            exit_calls += 1
            llm.exit()
            exit_calls += 1

    require(exit_calls == 2, "V0 engine did not complete both exit() calls")
    require(
        not dist.is_initialized(),
        "V0 engine left a torch.distributed process group initialized",
    )
    require(normalized_outputs is not None, "V0 outputs were not captured")
    require(scheduler_trace is not None, "V0 scheduler trace was not captured")
    require(tokenizer_evidence is not None, "V0 tokenizer evidence was not captured")

    _, model_after = model_snapshot(args.model)
    require(
        model_after == model_evidence,
        "model artifacts changed during the V0 golden run",
    )
    git_after = git_snapshot(source_root)
    validate_v0_snapshot(git_after, expected_commit)
    validate_unchanged(git_before, git_after)
    runner_git_after = git_snapshot(runner_root)
    validate_runner_snapshot(runner_git_after, expected_runner_commit)
    validate_unchanged(
        runner_git_before,
        runner_git_after,
        role="evidence runner",
    )
    runner_script_after = sha256_file(Path(__file__).resolve())
    require(
        runner_script_after == runner_script_before,
        "V0 evidence runner script changed during the run",
    )
    runtime = collect_runtime(torch, transformers)
    finished_at = datetime.now(timezone.utc).isoformat()
    command_argv = list(getattr(sys, "orig_argv", sys.argv))
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "command": {
            "argv": command_argv,
            "shell": shlex.join(command_argv),
            "cwd": str(Path.cwd().resolve()),
        },
        "source": {
            "nanovllm_import_origin": str(package_origin),
            "expected_commit": expected_commit,
            "git_before": git_before,
            "git_after": git_after,
            "unchanged_during_run": True,
        },
        "runner": {
            "expected_commit": expected_runner_commit,
            "git_before": runner_git_before,
            "git_after": runner_git_after,
            "script_path": str(Path(__file__).resolve()),
            "script_sha256_before": runner_script_before,
            "script_sha256_after": runner_script_after,
            "unchanged_during_run": True,
        },
        "model": {
            "before": model_evidence,
            "after": model_after,
            "unchanged_during_run": True,
        },
        "tokenizer": tokenizer_evidence,
        "randomness": {
            "seed": args.seed,
            "seeded": ["python.random", "torch.cpu", "torch.cuda.all"],
            "sampling_mode": "greedy",
        },
        "run": {
            "mode": args.mode,
            "requested_engine_config": engine_kwargs,
            "effective_engine_config": effective_config,
            "workload": {
                "prompts_token_ids": prompts,
                "sampling_params": sampling_config,
            },
            "outputs": normalized_outputs,
            "comparison_outputs": [
                [output["text"], output["token_ids"]]
                for output in normalized_outputs
            ],
            "scheduler_trace_schema": [
                "is_prefill",
                [
                    "sequence_length",
                    "num_cached_tokens",
                    "num_scheduled_tokens",
                    "sequence_is_prefill",
                ],
            ],
            "scheduler_trace": scheduler_trace,
            "teardown": {"exit_calls_completed": exit_calls},
        },
        "runtime": runtime,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_sha256 = write_json_exclusive(output_path, evidence)
    print(
        "PASS: wrote write-once V0 golden "
        f"to {output_path} (sha256={artifact_sha256})"
    )


if __name__ == "__main__":
    main()
