#!/usr/bin/env python3
"""Fresh-process end-to-end benchmark for one top-p backend.

Run exact and FlashInfer in separate, serialized processes and rotate their
order across pairs.  Every repetition uses a distinct prompt set whose first
token is unique across the process, preventing prefix-cache reuse.  The first
observation is retained as settling evidence but excluded from the steady
median.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import torch
import transformers

from nanovllm import LLM, SamplingParams


ROOT = Path(__file__).resolve().parents[2]


def distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def command_output(*command: str) -> str:
    return subprocess.check_output(command, cwd=ROOT, text=True).strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(bytes(tensor.cpu().tolist())).hexdigest()


def model_fingerprint(model_dir: Path) -> dict[str, object]:
    paths = [
        model_dir / name
        for name in (
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        )
        if (model_dir / name).is_file()
    ]
    paths.extend(sorted(model_dir.glob("*.safetensors")))
    if not (model_dir / "config.json").is_file() or not any(
        path.suffix == ".safetensors" for path in paths
    ):
        raise SystemExit(
            "model fingerprint requires config.json, tokenizer.json, and "
            "at least one .safetensors file"
        )
    return {
        "resolved_path": str(model_dir),
        "files": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in paths
        ],
    }


def make_prompts(
    batch: int,
    prompt_len: int,
    seed: int,
    repetition: int,
    vocab_size: int,
) -> list[list[int]]:
    rng = random.Random(seed)
    prompts = []
    for row in range(batch):
        # The first token is disjoint across every row and repetition. Because
        # the cache hash is chained, no full prompt block can match a prior run.
        prompt = [repetition * batch + row]
        prompt.extend(rng.randrange(vocab_size) for _ in range(prompt_len - 1))
        prompts.append(prompt)
    return prompts


def prompt_digest(prompts: list[list[int]]) -> str:
    digest = hashlib.sha256()
    for prompt in prompts:
        digest.update(len(prompt).to_bytes(4, "little"))
        for token in prompt:
            digest.update(int(token).to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def token_digest(outputs: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for output in outputs:
        for token in output["token_ids"]:
            digest.update(int(token).to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def flashinfer_provenance(backend: str) -> dict[str, object] | None:
    if backend != "flashinfer":
        return None
    import flashinfer

    return {
        "version": flashinfer.__version__,
        "distribution_version": distribution_version("flashinfer-python"),
        "git_commit": flashinfer.__git_commit__,
        "module_file": str(Path(flashinfer.__file__).resolve()),
        "license": "Apache-2.0",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=("exact", "flashinfer"), required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--pair-id", type=int, required=True)
    parser.add_argument(
        "--pair-order",
        choices=("exact-flashinfer", "flashinfer-exact"),
        required=True,
    )
    parser.add_argument(
        "--position-in-pair", choices=("first", "second"), required=True
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if min(args.batch, args.prompt_len, args.output_len) < 1:
        parser.error("batch and token lengths must be positive")
    if args.repetitions < 4:
        parser.error("at least four repetitions are required")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing evidence: {args.output}")
    if args.output.resolve().is_relative_to(ROOT):
        raise SystemExit("raw evidence output must be outside the git checkout")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    ordered_backends = args.pair_order.split("-")
    expected_backend = ordered_backends[0 if args.position_in_pair == "first" else 1]
    if args.backend != expected_backend:
        parser.error(
            f"{args.position_in_pair} process in {args.pair_order} must use "
            f"backend={expected_backend}, not {args.backend}"
        )

    model_dir = Path(args.model).resolve()
    hf_config = transformers.AutoConfig.from_pretrained(model_dir)
    vocab_size = int(hf_config.vocab_size)
    effective_max_model_len = min(
        args.max_model_len, int(hf_config.max_position_embeddings)
    )
    if args.prompt_len + args.output_len > effective_max_model_len:
        parser.error(
            "prompt-len + output-len exceeds the effective max model length"
        )
    if args.repetitions * args.batch > vocab_size:
        parser.error(
            "repetitions * batch must not exceed vocab size; unique first "
            "prompt tokens are required"
        )

    commit = command_output("git", "rev-parse", "HEAD")
    if commit != args.expected_commit:
        raise SystemExit(
            f"wrong checkout: expected {args.expected_commit}, observed {commit}"
        )
    git_status = command_output("git", "status", "--short")
    if git_status:
        raise SystemExit(f"benchmark requires a clean checkout:\n{git_status}")

    started_at = datetime.now(UTC)
    params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        ignore_eos=True,
        max_tokens=args.output_len,
    )
    llm = LLM(
        args.model,
        enforce_eager=False,
        max_model_len=args.max_model_len,
        max_num_seqs=args.batch,
        gpu_memory_utilization=0.8,
        top_p_backend=args.backend,
    )
    effective_backend = llm.model_runner.config.top_p_backend
    if effective_backend != args.backend:
        raise RuntimeError(
            f"requested backend={args.backend}, effective backend={effective_backend}"
        )
    block_manager = llm.scheduler.block_manager
    effective_num_kvcache_blocks = len(block_manager.blocks)
    blocks_per_sequence = (
        args.prompt_len + args.output_len + block_manager.block_size - 1
    ) // block_manager.block_size
    required_peak_blocks_no_sharing = args.batch * blocks_per_sequence
    if effective_num_kvcache_blocks < required_peak_blocks_no_sharing:
        raise RuntimeError(
            "benchmark would allow KV-cache preemption: "
            f"{effective_num_kvcache_blocks} available < "
            f"{required_peak_blocks_no_sharing} required"
        )
    observations = []
    try:
        llm.generate(
            [[17, 19, 23]],
            SamplingParams(
                temperature=0.6,
                top_p=0.9,
                ignore_eos=True,
                max_tokens=2,
            ),
            use_tqdm=False,
        )
        for repetition in range(args.repetitions):
            run_seed = args.seed * 1000 + args.batch * 10 + repetition
            prompt_seed = args.seed * 10_000 + repetition
            prompts = make_prompts(
                args.batch,
                args.prompt_len,
                prompt_seed,
                repetition,
                vocab_size,
            )
            torch.manual_seed(run_seed)
            torch.cuda.manual_seed_all(run_seed)
            torch.cuda.synchronize()
            rng_state_before = tensor_sha256(torch.cuda.get_rng_state())
            begin = time.perf_counter()
            outputs = llm.generate(prompts, params, use_tqdm=False)
            torch.cuda.synchronize()
            elapsed_s = time.perf_counter() - begin
            rng_state_after = tensor_sha256(torch.cuda.get_rng_state())
            if len(outputs) != args.batch:
                raise RuntimeError(
                    f"expected {args.batch} outputs; observed {len(outputs)}"
                )
            if not all(
                len(output["token_ids"]) == args.output_len for output in outputs
            ):
                raise RuntimeError("an output did not contain the requested tokens")
            observations.append(
                {
                    "repetition": repetition,
                    "seed": run_seed,
                    "prompt_seed": prompt_seed,
                    "prompt_sha256": prompt_digest(prompts),
                    "cuda_rng_state_before_sha256": rng_state_before,
                    "cuda_rng_state_after_sha256": rng_state_after,
                    "elapsed_s": elapsed_s,
                    "output_tokens_per_s": (
                        args.batch * args.output_len / elapsed_s
                    ),
                    "token_sha256": token_digest(outputs),
                }
            )
    finally:
        atexit.unregister(llm.exit)
        llm.exit()

    steady = observations[1:]
    result = {
        "schema_version": 1,
        "benchmark": "topp_backend_e2e",
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "commit": commit,
        "git_status": git_status,
        "run_label": args.run_label,
        "pair_id": args.pair_id,
        "pair_order": args.pair_order,
        "position_in_pair": args.position_in_pair,
        "backend": args.backend,
        "seed": args.seed,
        "argv": sys.argv,
        "execution": {
            "cwd": str(Path.cwd()),
            "pythonpath": os.environ.get("PYTHONPATH"),
            "torchinductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
            "flashinfer_workspace_base": os.environ.get(
                "FLASHINFER_WORKSPACE_BASE"
            ),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "environment": {
            "platform": platform.platform(),
            "kernel": platform.release(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "triton": distribution_version("triton"),
            "flashinfer_python": distribution_version("flashinfer-python"),
            "cuda_python": distribution_version("cuda-python"),
            "flash_attn": distribution_version("flash-attn"),
            "cuda_build": torch.version.cuda,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_uuid": command_output(
                "nvidia-smi",
                "--query-gpu=uuid",
                "--format=csv,noheader",
            ),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(
                0
            ).total_memory,
            "nvidia_driver": command_output(
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ),
        },
        "flashinfer": flashinfer_provenance(args.backend),
        "source": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": file_sha256(Path(__file__).resolve()),
            "origin_url": command_output("git", "remote", "get-url", "origin"),
        },
        "model_fingerprint": model_fingerprint(model_dir),
        "configuration": {
            "model": str(model_dir),
            "model_vocab_size": vocab_size,
            "batch": args.batch,
            "prompt_tokens": args.prompt_len,
            "output_tokens": args.output_len,
            "temperature": 0.6,
            "top_p": 0.9,
            "effective_top_p_backend": effective_backend,
            "ignore_eos": True,
            "max_model_len": args.max_model_len,
            "effective_max_model_len": effective_max_model_len,
            "max_num_seqs": args.batch,
            "gpu_memory_utilization": 0.8,
            "cuda_graphs": True,
            "repetitions": args.repetitions,
            "shape_cold_observations_excluded": 1,
            "first_observation_classification": (
                "shape-cold after engine and sampler JIT warmup"
            ),
            "prompts_distinct_across_repetitions": True,
            "effective_num_kvcache_blocks": effective_num_kvcache_blocks,
            "kvcache_block_size": block_manager.block_size,
            "blocks_per_sequence_at_max_output": blocks_per_sequence,
            "required_peak_blocks_no_sharing": required_peak_blocks_no_sharing,
            "preemption_excluded_by_capacity": True,
        },
        "observations": observations,
        "steady": steady,
        "median_steady_elapsed_s": statistics.median(
            row["elapsed_s"] for row in steady
        ),
        "median_steady_output_tokens_per_s": statistics.median(
            row["output_tokens_per_s"] for row in steady
        ),
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
