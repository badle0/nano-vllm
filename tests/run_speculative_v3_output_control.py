"""Fresh-process A100 output/RNG control for V3 draft-discard execution.

Run this executable once with ``--side off`` and once with ``--side on`` from
otherwise identical fresh Python processes.  The companion
``compare_speculative_v3_output_control.py`` requires exact authoritative target
event and CPU/CUDA RNG endpoint parity between the two artifacts.

Dirty-worktree runs are exploratory only.  A retained artifact is eligible only
when the source snapshot taken before engine construction is clean.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import nanovllm

from nanovllm import LLM, SamplingParams
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.speculative_routes import DraftRouteRegistry
from nanovllm.layers.sampler import Sampler

from _speculative_v3_evidence import (
    compiler_cache_roots_from_environment,
    finalize_evidence,
    prepare_evidence,
    require_retained_a100_sxm4_40gb,
    sha256_bytes,
    validate_compiler_cache_root_isolation,
    validate_output_paths,
    write_json_exclusive,
)


SCHEMA = "nano-vllm-speculative-v3-output-control-v2"
SEED = 20260828
DRAFT_PHASES = (
    "_construct_draft_model",
    "warmup_draft_model",
    "capture_draft_cudagraph",
    "_pretouch_draft_eager_prefill",
    "_pretouch_draft_routes",
)
DRAFT_RESOURCE_ATTRIBUTES = (
    "draft_model",
    "draft_kv_cache",
    "draft_graphs",
    "draft_graph_vars",
    "draft_graph_pool",
    "draft_graph_bs",
    "speculative_memory_plan",
    "speculative_memory_audit",
    "draft_route_registry",
    "_speculative_memory_audit_inputs",
    "_profiled_graph_allocated_bytes",
    "_profiled_graph_reserved_bytes",
    "_profiled_graph_peak_allocated_bytes",
    "_profiled_graph_peak_reserved_bytes",
    "_final_graph_allocated_baseline",
    "_final_graph_reserved_baseline",
    "_allocated_after_graph_before_pretouch",
    "_reserved_after_graph_before_pretouch",
    "_final_graph_peak_allocated_bytes",
    "_final_graph_peak_reserved_bytes",
    "_target_warmup_transient_bytes",
    "_draft_warmup_transient_bytes",
    "_warmup_transient_bytes",
    "_draft_route_pretouch_peak_bytes",
)
PROMPTS = (
    (10, 11, 12, 13),
    (20, 21, 22),
)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run one fresh speculation-off/on V3 output-control side."
    )
    parser.add_argument("--side", choices=("off", "on"), required=True)
    parser.add_argument("--mode", choices=("eager", "graph"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--configured-k", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--expected-commit",
        help="full producer commit required by --retained",
    )
    parser.add_argument(
        "--retained",
        action="store_true",
        help="enforce clean exact-SHA retained-evidence provenance",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_sha256(tensor: torch.Tensor) -> str:
    payload = tensor.detach().cpu().contiguous().numpy().tobytes()
    return sha256_bytes(payload)


def rng_hashes() -> dict:
    torch.cuda.synchronize()
    return {
        "cpu_sha256": tensor_sha256(torch.random.get_rng_state()),
        "cuda_sha256": tensor_sha256(torch.cuda.get_rng_state()),
    }


def install_draft_phase_ledger() -> dict[str, int]:
    """Instrument every draft-only constructor phase before construction."""

    ledger = {name: 0 for name in DRAFT_PHASES}
    for name in DRAFT_PHASES:
        original = getattr(ModelRunner, name)

        def counted(self, *args, _name=name, _original=original, **kwargs):
            ledger[_name] += 1
            return _original(self, *args, **kwargs)

        setattr(ModelRunner, name, counted)
    return ledger


def route_dict(key) -> dict:
    return {
        "schema": key.schema,
        "execution_mode": str(key.execution_mode),
        "batch_bucket": key.batch_bucket,
        "effective_k": key.effective_k,
        "catchup_family": str(key.catchup_family),
        "sampler_envelope": str(key.sampler_envelope),
    }


def event_dict(stage: str, event) -> dict:
    return {
        "stage": stage,
        "seq_id": event.seq_id,
        "token_id": event.token_id,
        "finished": event.finished,
    }


def step_dict(stage: str, step) -> dict:
    return {
        "stage": stage,
        "num_prefill_tokens": step.num_prefill_tokens,
        "num_decode_tokens": step.num_decode_tokens,
        "events": [event_dict(stage, event) for event in step.events],
    }


def combined_sampling_params() -> list[SamplingParams]:
    return [
        SamplingParams(
            temperature=0.8,
            top_k=8,
            top_p=0.9,
            max_tokens=6,
            ignore_eos=True,
        )
        for _ in PROMPTS
    ]


def install_runtime_draft_ledger() -> list[dict]:
    """Record and independently prove RNG neutrality of each real V3 interval."""

    calls = []
    original = ModelRunner.run_speculative_discard

    def counted(self, plan, seqs):
        before = rng_hashes()
        result = original(self, plan, seqs)
        after = rng_hashes()
        require(before == after, "V3 draft interval changed target RNG state")
        calls.append(
            {
                "route": route_dict(result.route_key),
                "catchup_tokens": result.catchup_positions,
                "proposal_token_ids": [
                    list(row.proposed_token_ids) for row in result.rows
                ],
                "graph_decode_steps": result.graph_decode_steps,
                "eager_decode_steps": result.eager_decode_steps,
                "rng_before": before,
                "rng_after": after,
                "rng_neutral": True,
            }
        )
        return result

    ModelRunner.run_speculative_discard = counted
    return calls


def live_draft_resource_state(runner: ModelRunner) -> dict[str, bool]:
    return {name: hasattr(runner, name) for name in DRAFT_RESOURCE_ATTRIBUTES}


def draft_owned_instance_attributes(runner: ModelRunner) -> list[str]:
    """Return every runner-owned draft/speculative field except its off/on flag."""

    return sorted(
        name
        for name in vars(runner)
        if name != "speculation_enabled"
        and ("draft" in name or "speculative" in name)
    )


def main(argv=None):
    args = parse_args(argv)
    require(torch.cuda.is_available(), "CUDA is required")
    require(torch.cuda.device_count() == 1, "control requires exactly one visible GPU")
    device_name = torch.cuda.get_device_name(0)
    require("A100" in device_name, f"control requires an A100, got {device_name!r}")
    if args.retained:
        require_retained_a100_sxm4_40gb()
    cache_roots = compiler_cache_roots_from_environment(
        required=args.retained
    )
    require(args.configured_k >= 1, "configured K must be positive")
    require(args.max_num_seqs >= len(PROMPTS), "max_num_seqs is too small")
    require(
        args.max_num_batched_tokens >= sum(map(len, PROMPTS)),
        "token budget cannot hold the control prefill",
    )

    script_path = Path(__file__).resolve()
    evidence_context = prepare_evidence(
        script_path=script_path,
        retained=args.retained,
        expected_commit=args.expected_commit,
        model_arguments={"target": args.model, "draft": args.draft_model},
        runtime_imports={
            "nanovllm": (nanovllm, "nanovllm/__init__.py"),
            "LLM": (LLM, "nanovllm/llm.py"),
            "LLMEngine": (LLMEngine, "nanovllm/engine/llm_engine.py"),
            "ModelRunner": (ModelRunner, "nanovllm/engine/model_runner.py"),
            "Scheduler": (Scheduler, "nanovllm/engine/scheduler.py"),
            "Sampler": (Sampler, "nanovllm/layers/sampler.py"),
            "SamplingParams": (SamplingParams, "nanovllm/sampling_params.py"),
            "DraftRouteRegistry": (
                DraftRouteRegistry,
                "nanovllm/engine/speculative_routes.py",
            ),
        },
    )
    target_model_path = evidence_context.model_paths["target"]
    draft_model_path = evidence_context.model_paths["draft"]
    validate_compiler_cache_root_isolation(
        cache_roots,
        repo_root=evidence_context.repo_root,
        model_roots=evidence_context.model_paths.values(),
    )
    output_path = validate_output_paths(
        (Path(args.output),),
        repo_root=evidence_context.repo_root,
        model_roots=evidence_context.model_paths.values(),
        cache_roots=cache_roots,
    )[0]
    phase_ledger = install_draft_phase_ledger()
    runtime_draft_calls = install_runtime_draft_ledger()
    seed_all(args.seed)

    common = dict(
        enforce_eager=args.mode == "eager",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
    )
    if args.side == "on":
        common.update(
            draft_model=str(draft_model_path),
            num_speculative_tokens=args.configured_k,
        )

    engine = None
    output = None
    try:
        engine = LLM(str(target_model_path), **common)
        runner = engine.model_runner
        resource_state = live_draft_resource_state(runner)
        draft_owned_attributes = draft_owned_instance_attributes(runner)
        if args.side == "off":
            require(not runner.speculation_enabled, "off control enabled speculation")
            require(
                all(count == 0 for count in phase_ledger.values()),
                f"off control entered a draft phase: {phase_ledger}",
            )
            require(
                not any(resource_state.values()),
                f"off control retained draft resources: {resource_state}",
            )
            require(
                not draft_owned_attributes,
                "off control created draft-owned runner attributes: "
                f"{draft_owned_attributes}",
            )
        else:
            require(runner.speculation_enabled, "on control did not enable speculation")
            require(phase_ledger["_construct_draft_model"] == 1, "draft construction count drifted")
            require(phase_ledger["warmup_draft_model"] == 1, "draft warmup count drifted")
            require(phase_ledger["_pretouch_draft_routes"] == 1, "route pretouch count drifted")
            registry = getattr(runner, "draft_route_registry", None)
            require(isinstance(registry, DraftRouteRegistry), "on control lacks route registry")
            require(
                registry.ready_keys == registry.router_admitted_keys,
                "on control route registry is not fully ready",
            )

        rng_snapshots = {"after_init": rng_hashes()}
        params = combined_sampling_params()
        seq_ids = [
            engine.add_request(list(prompt), parameter)
            for prompt, parameter in zip(PROMPTS, params, strict=True)
        ]

        prefill = engine._step()
        require(
            prefill.num_prefill_tokens == sum(map(len, PROMPTS)),
            "control prompts did not complete in one prefill step",
        )
        require(prefill.num_decode_tokens == 0, "prefill mixed in decode work")
        rng_snapshots["after_prefill"] = rng_hashes()

        first_decode = engine._step()
        require(first_decode.num_prefill_tokens == 0, "first decode mixed in prefill work")
        require(
            first_decode.num_decode_tokens == len(PROMPTS),
            "first decode did not cover every control request",
        )
        rng_snapshots["after_first_target_decode"] = rng_hashes()

        repeated_decode = engine._step()
        require(repeated_decode.num_prefill_tokens == 0, "repeated decode mixed in prefill work")
        require(
            repeated_decode.num_decode_tokens == len(PROMPTS),
            "repeated decode did not cover every control request",
        )
        rng_snapshots["after_repeated_target_decode"] = rng_hashes()

        steps = [
            step_dict("prefill", prefill),
            step_dict("first_target_decode", first_decode),
            step_dict("repeated_target_decode", repeated_decode),
        ]
        authoritative_events = [
            event for step in steps for event in step["events"]
        ]
        target_tokens_by_seq = {
            str(seq_id): [
                event["token_id"]
                for event in authoritative_events
                if event["seq_id"] == seq_id
            ]
            for seq_id in seq_ids
        }

        if args.side == "on":
            require(len(runtime_draft_calls) == 2, "on control did not run two V3 intervals")
            require(
                runtime_draft_calls[0]["catchup_tokens"] > 0,
                "first V3 interval did not exercise cold catch-up",
            )
            require(
                runtime_draft_calls[1]["catchup_tokens"] == 0,
                "repeated V3 interval unexpectedly repeated catch-up",
            )
        else:
            require(not runtime_draft_calls, "off control executed a V3 interval")

        output = {
            "schema": SCHEMA,
            "side": args.side,
            "mode": args.mode,
            "seed": args.seed,
            "model": str(target_model_path),
            "draft_model_argument": str(draft_model_path),
            "configured_k": args.configured_k,
            "configuration": {
                "max_model_len": args.max_model_len,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "max_num_seqs": args.max_num_seqs,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "top_p_backend": "exact",
                "tensor_parallel_size": 1,
            },
            "workload": {
                "prompt_token_ids": [list(prompt) for prompt in PROMPTS],
                "sampling": {
                    "temperature": 0.8,
                    "top_k": 8,
                    "top_p": 0.9,
                    "max_tokens": 6,
                    "ignore_eos": True,
                },
            },
            "draft_phase_calls": dict(phase_ledger),
            "live_draft_resource_attributes": resource_state,
            "draft_owned_instance_attributes": draft_owned_attributes,
            "runtime_draft_calls": runtime_draft_calls,
            "rng_snapshots": rng_snapshots,
            "steps": steps,
            "authoritative_target_events": authoritative_events,
            "target_token_ids_by_seq": target_tokens_by_seq,
        }
    finally:
        if engine is not None:
            engine.exit()

    require(output is not None, "control did not produce output")
    provenance = finalize_evidence(evidence_context)
    output["provenance"] = provenance
    output["retention_eligible"] = provenance["retention_eligible"]
    write_json_exclusive(output_path, output)
    print(
        f"wrote {output_path}: side={args.side}, "
        f"mode={args.mode}, target_events={len(output['authoritative_target_events'])}, "
        f"draft_intervals={len(output['runtime_draft_calls'])}, "
        f"retention_eligible={output['retention_eligible']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
