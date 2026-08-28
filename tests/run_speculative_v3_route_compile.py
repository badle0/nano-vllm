"""Fresh-process GPU gate for V3 route completeness and cold compile behavior.

This is an executable certification harness, not a pytest module.  Give every
invocation initially empty, process-unique Inductor and Triton cache directories.
The harness enumerates the runner's immutable route registry, forces each exact
K/catch-up key twice through a real scheduler reservation, and proves that the
draft interval adds no Dynamo graph, compiler artifact, or CUDA graph capture.

Runs from a dirty worktree are exploratory only.  Retained evidence must be
produced from an exact clean commit and archived separately.
"""

import argparse
import copy
import hashlib
import json
import os
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import CodeType

import torch

import nanovllm

from nanovllm import LLM, SamplingParams
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.speculative_routes import (
    DraftRouteAdmission,
    DraftRouteRegistry,
)
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import get_context
from nanovllm.utils.errors import record_cleanup_failure

from _speculative_v3_evidence import (
    finalize_evidence,
    path_is_within,
    prepare_evidence,
    require_retained_a100_sxm4_40gb,
    validate_output_paths,
    write_json_exclusive,
)


SCHEMA = "nano-vllm-speculative-v3-route-compile-v2"
SEED = 20260828
RETAINED_CONFIGURATION = {
    "configured_k": 2,
    "max_num_seqs": 4,
    "max_num_batched_tokens": 512,
    "max_model_len": 512,
    "gpu_memory_utilization": 0.5,
    "repetitions": 2,
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model")
    parser.add_argument("--mode", choices=("eager", "graph"), required=True)
    parser.add_argument("--configured-k", type=int, default=2)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument(
        "--expected-commit",
        help="full producer commit required by --retained",
    )
    parser.add_argument(
        "--retained",
        action="store_true",
        help="enforce clean exact-SHA retained-evidence provenance",
    )
    parser.add_argument("--output")
    return parser.parse_args(argv)


def cache_root_path(name):
    value = os.environ.get(name)
    require(value, f"{name} must name an initially empty process-unique directory")
    lexical = Path(value).expanduser()
    require(not lexical.is_symlink(), f"{name} must not be a symlink")
    root = lexical.resolve()
    return root


def initialize_cache_root(name, root):
    root.mkdir(parents=True, exist_ok=True)
    require(root.is_dir() and not root.is_symlink(), f"{name} is not a real directory")
    require(not any(root.iterdir()), f"{name} is not initially empty: {root}")


def validate_cache_root_isolation(
    inductor_root,
    triton_root,
    *,
    repo_root,
    model_roots,
):
    require(inductor_root != triton_root, "compiler cache roots must be distinct")
    require(
        not path_is_within(inductor_root, triton_root)
        and not path_is_within(triton_root, inductor_root),
        "compiler cache roots must not be nested",
    )
    for name, root in (
        ("TORCHINDUCTOR_CACHE_DIR", inductor_root),
        ("TRITON_CACHE_DIR", triton_root),
    ):
        require(
            not path_is_within(root, repo_root),
            f"{name} must be outside the source checkout",
        )
        require(
            not any(path_is_within(root, model_root) for model_root in model_roots),
            f"{name} must be outside model directories",
        )


def require_compiler_environment():
    for name in ("TORCH_COMPILE_DISABLE", "TORCHDYNAMO_DISABLE"):
        require(
            os.environ.get(name) in (None, "", "0"),
            f"{name} must be unset or 0",
        )
    require(
        os.environ.get("TORCHDYNAMO_SUPPRESS_ERRORS") in (None, ""),
        "TORCHDYNAMO_SUPPRESS_ERRORS must be unset",
    )
    from torch._dynamo import config as dynamo_config

    require(
        dynamo_config.disable is False,
        "TorchDynamo must be enabled",
    )
    require(
        dynamo_config.suppress_errors is False,
        "TorchDynamo error suppression must be disabled",
    )
    disabled_cache_flags = (
        "TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE",
        "TORCHINDUCTOR_AUTOGRAD_CACHE",
        "TORCHINDUCTOR_AUTOGRAD_REMOTE_CACHE",
        "TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE",
        "TORCHINDUCTOR_BUNDLED_AUTOTUNE_REMOTE_CACHE",
        "TORCH_DYNAMO_AUTOMATIC_DYNAMIC_LOCAL_PGO",
        "TORCH_DYNAMO_AUTOMATIC_DYNAMIC_REMOTE_PGO",
    )
    for name in disabled_cache_flags:
        require(os.environ.get(name) == "0", f"{name} must be 0")
    torch_logs = {
        item.strip() for item in os.environ.get("TORCH_LOGS", "").split(",")
    }
    require(
        {"recompiles", "graph_breaks"}.issubset(torch_logs),
        "TORCH_LOGS must include recompiles,graph_breaks",
    )


def file_manifest(roots):
    manifest = []
    for label, root in roots:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            payload = path.read_bytes()
            manifest.append(
                {
                    "root": label,
                    "path": str(path.relative_to(root)),
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    return manifest


def stable(value):
    if isinstance(value, dict):
        return {
            stable_key(key): stable(item)
            for key, item in sorted(value.items(), key=lambda pair: stable_key(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [stable(item) for item in value]
    if isinstance(value, set):
        return sorted(stable(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def stable_key(key):
    if isinstance(key, CodeType):
        return f"{key.co_filename}:{key.co_firstlineno}:{key.co_name}"
    return repr(key)


def payload_sha256(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_manifest_summary(manifest):
    return {
        "file_count": len(manifest),
        "total_bytes": sum(item["bytes"] for item in manifest),
        "manifest_sha256": payload_sha256(manifest),
    }


def compiler_delta_summary(before, after):
    summary = {}
    for key in before:
        if before[key] == after[key]:
            continue
        if key == "cache_manifest":
            summary[key] = {
                "before": cache_manifest_summary(before[key]),
                "after": cache_manifest_summary(after[key]),
            }
        else:
            summary[key] = {"before": before[key], "after": after[key]}
    return summary


def compiler_snapshot(cache_roots, capture_ledger):
    from torch._dynamo import utils as dynamo_utils

    torch.cuda.synchronize()
    return {
        "counters": stable(copy.deepcopy(dynamo_utils.counters)),
        "guard_failures": stable(copy.deepcopy(dynamo_utils.guard_failures)),
        "graph_break_reasons": stable(
            copy.deepcopy(dynamo_utils.graph_break_reasons)
        ),
        "cache_manifest": file_manifest(cache_roots),
        "cuda_graph_objects": capture_ledger["cuda_graph_objects"],
        "cuda_graph_contexts": capture_ledger["cuda_graph_contexts"],
    }


def compiler_delta(before, after):
    return {
        key: {"before": before[key], "after": after[key]}
        for key in before
        if before[key] != after[key]
    }


def require_nonvacuous_compiler_snapshot(snapshot):
    counters = snapshot.get("counters", {})
    stats = counters.get("'stats'", {})
    unique_graphs = stats.get("'unique_graphs'", 0)
    require(
        isinstance(unique_graphs, int) and unique_graphs > 0,
        "constructor pretouch produced no compiled graphs",
    )
    manifest = snapshot.get("cache_manifest")
    require(
        isinstance(manifest, list)
        and manifest
        and sum(item.get("bytes", 0) for item in manifest) > 0,
        "constructor pretouch produced no compiler-cache artifacts",
    )


def require_retained_configuration(args):
    for name, expected in RETAINED_CONFIGURATION.items():
        require(
            getattr(args, name) == expected,
            f"retained route evidence requires {name}={expected}",
        )


def rng_hashes():
    cpu = torch.random.get_rng_state().cpu().numpy().tobytes()
    cuda = torch.cuda.get_rng_state().cpu().numpy().tobytes()
    return {
        "cpu": hashlib.sha256(cpu).hexdigest(),
        "cuda": hashlib.sha256(cuda).hexdigest(),
    }


def default_context():
    context = get_context()
    return bool(
        context.is_prefill is False
        and context.cu_seqlens_q is None
        and context.cu_seqlens_k is None
        and context.slot_mapping is None
        and context.context_lens is None
        and context.block_tables is None
    )


def contains_cuda_tensor(value, seen=None):
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return False
    seen.add(identity)
    if isinstance(value, torch.Tensor):
        return value.device.type == "cuda"
    if isinstance(value, dict):
        return any(
            contains_cuda_tensor(key, seen) or contains_cuda_tensor(item, seen)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(contains_cuda_tensor(item, seen) for item in value)
    return False


def sampling_params(batch_size, *, max_tokens):
    modes = (
        dict(temperature=0.0, top_k=-1, top_p=1.0),
        dict(temperature=0.8, top_k=-1, top_p=1.0),
        dict(temperature=0.8, top_k=8, top_p=1.0),
        dict(temperature=0.8, top_k=8, top_p=0.9),
    )
    # Put a sampled/combined row first so B=1 also exercises the race/filter path.
    order = (3, 0, 1, 2)
    return [
        SamplingParams(
            **modes[order[row % len(order)]],
            max_tokens=max_tokens,
            ignore_eos=True,
        )
        for row in range(batch_size)
    ]


def route_dict(key):
    return {
        "schema": key.schema,
        "execution_mode": str(key.execution_mode),
        "batch_bucket": key.batch_bucket,
        "effective_k": key.effective_k,
        "catchup_family": str(key.catchup_family),
        "sampler_envelope": str(key.sampler_envelope),
    }


@contextmanager
def guarded_draft_interval(interval_label):
    """Keep fail-on-recompile and stderr sentinels draft-only."""

    torch.compiler.set_stance("fail_on_recompile")
    try:
        print(
            f"V3_DRAFT_INTERVAL_BEGIN {interval_label}",
            file=sys.stderr,
            flush=True,
        )
        try:
            yield
        finally:
            print(
                f"V3_DRAFT_INTERVAL_END {interval_label}",
                file=sys.stderr,
                flush=True,
            )
    finally:
        torch.compiler.set_stance("default")


def force_route_cycle(
    engine,
    *,
    effective_k,
    cache_roots,
    capture_ledger,
    run_id,
    record_index,
):
    seqs, is_prefill = engine.scheduler.schedule()
    require(seqs and not is_prefill, "forced route cycle did not schedule decode")
    schedule_rollback = engine.scheduler.capture_decode_schedule_rollback(
        seqs,
        is_prefill=is_prefill,
    )
    plans = []
    try:
        return _force_scheduled_route_cycle(
            engine,
            seqs=seqs,
            effective_k=effective_k,
            cache_roots=cache_roots,
            capture_ledger=capture_ledger,
            plans=plans,
            run_id=run_id,
            record_index=record_index,
        )
    except BaseException as error:
        if plans:
            engine._rollback_failed_draft_state(plans[0], error)
        try:
            engine.scheduler.rollback_failed_decode_schedule(
                schedule_rollback
            )
        except BaseException as cleanup_error:
            record_cleanup_failure(
                error,
                "V3 route harness ordinary decode schedule rollback",
                cleanup_error,
            )
        raise


def _force_scheduled_route_cycle(
    engine,
    *,
    seqs,
    effective_k,
    cache_roots,
    capture_ledger,
    plans,
    run_id,
    record_index,
):
    admission = engine.model_runner.resolve_draft_route_admission(seqs)
    require(admission is not None, "ready registry returned a route miss")
    require(
        admission.max_effective_k >= effective_k,
        "requested K exceeds the resolved route admission",
    )
    forced = DraftRouteAdmission(
        registry_schema=admission.registry_schema,
        plan_fingerprint=admission.plan_fingerprint,
        batch_size=admission.batch_size,
        catchup_tokens=admission.catchup_tokens,
        route_keys=admission.route_keys[:effective_k],
    )
    plan = engine.scheduler.plan_draft_discard(
        seqs,
        route_admission=forced,
    )
    require(plan.uses_draft, f"forced route fell back: {plan.fallback_reason}")
    plans.append(plan)

    require(default_context(), "draft interval did not start from default attention context")
    rng_before = rng_hashes()
    compile_before = compiler_snapshot(cache_roots, capture_ledger)
    result = None
    interval_label = (
        f"run={run_id},record={record_index},batch={len(seqs)},"
        f"bucket={plan.route_key.batch_bucket},"
        f"k={effective_k},catchup={plan.draft_catchup_tokens}"
    )
    with guarded_draft_interval(interval_label):
        result = engine.model_runner.run_speculative_discard(plan, seqs)
        # This synchronizes CUDA while fail-on-recompile remains active.
        # The matching stderr sentinels therefore bracket only the guarded
        # draft call and its immediately following compiler snapshot.
        compile_after = compiler_snapshot(cache_roots, capture_ledger)
    rng_after = rng_hashes()
    delta = compiler_delta(compile_before, compile_after)
    require(not delta, f"draft interval changed compiler state: {delta}")
    require(rng_after == rng_before, "draft interval changed target RNG state")
    require(default_context(), "draft interval leaked attention context")
    require(not contains_cuda_tensor(result), "host result retained a CUDA tensor")
    require(result.route_key == plan.route_key, "result route key drifted")
    vocab_size = engine._admission_limits.vocab_size
    coverage = engine._validate_draft_discard_result(
        plan, result, seqs, vocab_size
    )
    engine.scheduler.handoff_draft_discard(plan)
    engine.scheduler.stage_draft_coverage(plan, seqs, coverage)
    target_tokens = engine.model_runner.run(seqs, False)
    events = engine.scheduler.postprocess(seqs, target_tokens)

    return {
        "route": route_dict(plan.route_key),
        "live_batch_size": len(seqs),
        "catchup_tokens": plan.draft_catchup_tokens,
        "q_shape": list(result.q_shape),
        "q_stride": list(result.q_stride),
        "graph_decode_steps": result.graph_decode_steps,
        "eager_decode_steps": result.eager_decode_steps,
        "target_token_ids": [event.token_id for event in events],
        "proposed_token_ids": [
            list(row.proposed_token_ids) for row in result.rows
        ],
        "compiler_delta": {},
        "compiler_snapshot_before_sha256": payload_sha256(compile_before),
        "compiler_snapshot_after_sha256": payload_sha256(compile_after),
        "rng_neutral": True,
        "context_reset": True,
        "host_result_cuda_free": True,
    }


def install_capture_ledger():
    ledger = {"cuda_graph_objects": 0, "cuda_graph_contexts": 0}
    original_graph_type = torch.cuda.CUDAGraph
    original_graph_context = torch.cuda.graph

    def counted_graph_type(*args, **kwargs):
        ledger["cuda_graph_objects"] += 1
        return original_graph_type(*args, **kwargs)

    @contextmanager
    def counted_graph_context(*args, **kwargs):
        ledger["cuda_graph_contexts"] += 1
        with original_graph_context(*args, **kwargs) as value:
            yield value

    torch.cuda.CUDAGraph = counted_graph_type
    torch.cuda.graph = counted_graph_context
    return ledger


def main(argv=None):
    require(
        sys.flags.optimize == 0,
        "route certification refuses optimized Python",
    )
    args = parse_args(argv)
    require(torch.cuda.is_available(), "CUDA is required")
    if args.retained:
        require_retained_a100_sxm4_40gb()
        require(args.output is not None, "--retained requires --output")
        require_retained_configuration(args)
    require(args.configured_k >= 1, "configured K must be positive")
    require(args.repetitions >= 2, "at least two repetitions are required")
    require_compiler_environment()
    inductor_root = cache_root_path("TORCHINDUCTOR_CACHE_DIR")
    triton_root = cache_root_path("TRITON_CACHE_DIR")
    # Canonicalize the environment consumed later by Torch/Triton so a
    # symlinked parent cannot be retargeted after path validation.
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_root)
    os.environ["TRITON_CACHE_DIR"] = str(triton_root)
    draft_model = args.draft_model or args.model
    script_path = Path(__file__).resolve()
    evidence_context = prepare_evidence(
        script_path=script_path,
        retained=args.retained,
        expected_commit=args.expected_commit,
        model_arguments={"target": args.model, "draft": draft_model},
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
    validate_cache_root_isolation(
        inductor_root,
        triton_root,
        repo_root=evidence_context.repo_root,
        model_roots=evidence_context.model_paths.values(),
    )
    for name, root in (
        ("TORCHINDUCTOR_CACHE_DIR", inductor_root),
        ("TRITON_CACHE_DIR", triton_root),
    ):
        initialize_cache_root(name, root)
    cache_roots = (
        ("inductor", inductor_root),
        ("triton", triton_root),
    )
    output_path = None
    if args.output is not None:
        output_path = validate_output_paths(
            (Path(args.output),),
            repo_root=evidence_context.repo_root,
            model_roots=evidence_context.model_paths.values(),
            cache_roots=(inductor_root, triton_root),
        )[0]
    capture_ledger = install_capture_ledger()
    run_id = uuid.uuid4().hex
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # Reset any sitecustomize/programmatic force-eager stance before the
    # constructor pretouch.  The post-init non-vacuity gate below proves this
    # process actually compiled the registered routes.
    torch.compiler.set_stance("default")

    engine = LLM(
        str(target_model_path),
        draft_model=str(draft_model_path),
        num_speculative_tokens=args.configured_k,
        enforce_eager=args.mode == "eager",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
    )
    records = []
    try:
        registry = engine.model_runner.draft_route_registry
        require(isinstance(registry, DraftRouteRegistry), "route registry missing")
        require(
            registry.ready_keys == registry.router_admitted_keys,
            "constructor published an incomplete route registry",
        )
        init_capture_ledger = dict(capture_ledger)
        init_compiler = compiler_snapshot(cache_roots, capture_ledger)
        require_nonvacuous_compiler_snapshot(init_compiler)

        route_batch_cap = max(
            key.batch_bucket for key in registry.ready_keys
        )
        route_k_cap = max(key.effective_k for key in registry.ready_keys)
        require(
            engine.model_runner.config.max_model_len >= route_k_cap + 6,
            "route-compile fixture requires effective max_model_len >= Kcap + 6",
        )
        require(
            args.max_num_batched_tokens
            >= route_batch_cap * (route_k_cap + 5),
            "route-compile fixture requires token budget >= Bcap*(Kcap+5)",
        )
        # Execute every live batch size, including dynamic/interior values such
        # as eager B=1..Bcap and graph B=3 -> bucket 4.  Key-set equality alone
        # would miss those leading-dimension routes.
        for batch_size in range(1, route_batch_cap + 1):
            for effective_k in range(1, route_k_cap + 1):
                for repetition in range(args.repetitions):
                    params = sampling_params(
                        batch_size,
                        max_tokens=route_k_cap + 3,
                    )
                    prompt = [10, 11, 12, 13]
                    seq_ids = [
                        engine.add_request(prompt, parameter)
                        for parameter in params
                    ]
                    prefill = engine._step()
                    require(
                        prefill.num_prefill_tokens == batch_size * len(prompt),
                        "route batch did not finish in one prefill step",
                    )
                    cold = force_route_cycle(
                        engine,
                        effective_k=effective_k,
                        cache_roots=cache_roots,
                        capture_ledger=capture_ledger,
                        run_id=run_id,
                        record_index=len(records),
                    )
                    warm = force_route_cycle(
                        engine,
                        effective_k=effective_k,
                        cache_roots=cache_roots,
                        capture_ledger=capture_ledger,
                        run_id=run_id,
                        record_index=len(records) + 1,
                    )
                    require(cold["catchup_tokens"] > 0, "cold route lacked catch-up")
                    require(warm["catchup_tokens"] == 0, "warm route repeated catch-up")
                    records.extend(
                        (
                            {**cold, "repetition": repetition, "phase": "cold"},
                            {**warm, "repetition": repetition, "phase": "warm"},
                        )
                    )
                    engine._cancel_requests(seq_ids)

        visited = {
            tuple(sorted(record["route"].items())) for record in records
        }
        expected = {
            tuple(sorted(route_dict(key).items())) for key in registry.ready_keys
        }
        require(visited == expected, "runtime did not visit every registered route key")
        require(
            capture_ledger == init_capture_ledger,
            "runtime performed a new CUDA graph construction/capture",
        )
        final_compiler = compiler_snapshot(cache_roots, capture_ledger)
        output = {
            "schema": SCHEMA,
            "run_id": run_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode,
            "seed": SEED,
            "model": str(target_model_path),
            "draft_model": str(draft_model_path),
            "configured_k": args.configured_k,
            "configuration": {
                "max_num_seqs": args.max_num_seqs,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "max_model_len": args.max_model_len,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "repetitions": args.repetitions,
                "tensor_parallel_size": 1,
                "top_p_backend": "exact",
            },
            "registry_cardinality": len(registry.ready_keys),
            "visited_cardinality": len(visited),
            "route_pretouch_peak_bytes": (
                engine.model_runner._draft_route_pretouch_peak_bytes
            ),
            "capture_ledger_after_init": init_capture_ledger,
            "capture_ledger_after_runtime": dict(capture_ledger),
            "all_draft_intervals_compiler_state_unchanged": True,
            "compiler_state_after_init": init_compiler,
            "compiler_state_after_init_sha256": payload_sha256(init_compiler),
            "compiler_state_after_runtime": final_compiler,
            "compiler_state_after_runtime_sha256": payload_sha256(
                final_compiler
            ),
            "post_init_to_runtime_compiler_delta": compiler_delta_summary(
                init_compiler, final_compiler
            ),
            "records": records,
        }
    finally:
        engine.exit()

    provenance = finalize_evidence(evidence_context)
    output["provenance"] = provenance
    output["retention_eligible"] = provenance["retention_eligible"]
    if output_path is not None:
        write_json_exclusive(output_path, output)
        # Retain complete compiler snapshots in the artifact, but keep terminal
        # and CI output bounded even when the cache manifest is large.
        print(
            json.dumps(
                {
                    "schema": output["schema"],
                    "mode": output["mode"],
                    "registry_cardinality": output["registry_cardinality"],
                    "visited_cardinality": output["visited_cardinality"],
                    "route_pretouch_peak_bytes": output[
                        "route_pretouch_peak_bytes"
                    ],
                    "all_draft_intervals_compiler_state_unchanged": output[
                        "all_draft_intervals_compiler_state_unchanged"
                    ],
                    "retention_eligible": output["retention_eligible"],
                    "output": str(output_path),
                },
                sort_keys=True,
            )
        )
    else:
        payload = json.dumps(output, indent=2, sort_keys=True) + "\n"
        print(payload, end="")


if __name__ == "__main__":
    main()
