"""Fresh-process A100 rollback gate for each V2 constructor phase.

Invoke this script once per phase.  The injected exception is raised only after
the selected real GPU operation succeeds, maximizing the owned state that the
constructor transaction must unwind.  The same process must then construct,
run, and tear down a healthy speculative engine.
"""

import argparse
import gc
import inspect
import importlib.util
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist

from nanovllm.engine.model_runner import ModelRunner


CONTRACT_PATH = Path(__file__).with_name(
    "run_speculative_v2_lifecycle.py"
)
CONTRACT_SPEC = importlib.util.spec_from_file_location(
    "run_speculative_v2_lifecycle_for_gpu_recovery",
    CONTRACT_PATH,
)
CONTRACT = importlib.util.module_from_spec(CONTRACT_SPEC)
CONTRACT_SPEC.loader.exec_module(CONTRACT)

EVIDENCE_SCHEMA = "nano-vllm-speculative-v2-gpu-recovery-v1"
MAX_POST_EXECUTION_GLOBAL_ALLOCATED_BYTES = 32 * 1024**2
MAX_POST_EXECUTION_GLOBAL_RESERVED_BYTES = 64 * 1024**2
PHASE_METHODS = {
    "draft_construct": "_construct_draft_model",
    "draft_load": "_load_draft_model",
    "draft_warmup": "warmup_draft_model",
    "graph_profile": "_profile_speculative_graph_memory",
    "joint_allocate": "_allocate_speculative_kv_cache",
    "draft_graph": "capture_draft_cudagraph",
    "draft_pretouch": "_pretouch_draft_eager_prefill",
    "memory_finalize": "_finalize_speculative_memory_audit",
}
GRAPH_PHASES = {"graph_profile", "draft_graph", "draft_pretouch"}


class InjectedSpeculativePhaseError(RuntimeError):
    pass


def install_phase_injection(phase, *, runner_class=ModelRunner):
    """Patch one runner method and return its restoration/call ledger."""

    method_name = PHASE_METHODS[phase]
    original = getattr(runner_class, method_name)
    ledger = {"calls": 0, "injections": 0}

    def injected(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        ledger["calls"] += 1
        # Graph profiling performs the first draft capture.  Let that complete
        # and inject after the second, final runtime capture.
        should_inject = phase != "draft_graph" or ledger["calls"] == 2
        if should_inject:
            ledger["injections"] += 1
            raise InjectedSpeculativePhaseError(
                f"injected after real V2 phase: {phase}"
            )
        return result

    setattr(runner_class, method_name, injected)

    def restore():
        setattr(runner_class, method_name, original)

    return restore, ledger


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=tuple(PHASE_METHODS))
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model")
    parser.add_argument("--num-kvcache-blocks", type=int, default=4)
    parser.add_argument("--expected-commit")
    parser.add_argument("--retained", action="store_true")
    parser.add_argument("--output")
    return parser


def parse_args(argv=None):
    args = build_parser().parse_args(argv)
    args.mode = "graph" if args.phase in GRAPH_PHASES else "eager"
    args.configured_k = 2
    args.gpu_memory_utilization = 0.5
    args.max_model_len = 512
    args.max_num_batched_tokens = 512
    args.max_num_seqs = 4
    return args


def validate_runtime_import_origins(repo_root):
    """Bind every executable recovery component to the certified checkout."""

    repo_root = Path(repo_root).resolve()

    def source_path(value, *, label):
        path = inspect.getsourcefile(value)
        CONTRACT.require(
            path is not None,
            f"cannot determine the source file for {label}",
        )
        return Path(path).resolve()

    package_file = getattr(CONTRACT.nanovllm, "__file__", None)
    CONTRACT.require(
        package_file is not None,
        "imported nanovllm package has no source file",
    )
    actual = {
        "lifecycle_contract": Path(CONTRACT.__file__).resolve(),
        "nanovllm_package": Path(package_file).resolve(),
        "model_runner_class": source_path(
            ModelRunner,
            label="ModelRunner",
        ),
        "llm_class": source_path(CONTRACT.LLM, label="LLM"),
    }
    expected = {
        "lifecycle_contract": CONTRACT_PATH.resolve(),
        "nanovllm_package": (
            repo_root / "nanovllm" / "__init__.py"
        ).resolve(),
        "model_runner_class": (
            repo_root / "nanovllm" / "engine" / "model_runner.py"
        ).resolve(),
        "llm_class": (repo_root / "nanovllm" / "llm.py").resolve(),
    }
    for label, expected_path in expected.items():
        CONTRACT.require(
            actual[label] == expected_path,
            "recovery runtime import does not come from the certified "
            f"checkout: {label}={actual[label]}, expected={expected_path}",
        )
    CONTRACT.require(
        ModelRunner is CONTRACT.llm_engine_module.ModelRunner,
        "recovery injection ModelRunner differs from the engine's "
        "ModelRunner class",
    )
    return {label: str(path) for label, path in actual.items()}


def main(argv=None):
    CONTRACT.reject_optimized_python()
    started_at = datetime.now(timezone.utc).isoformat()
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    runtime_import_origins = validate_runtime_import_origins(repo_root)
    expected_commit = (
        None
        if args.expected_commit is None
        else args.expected_commit.lower()
    )
    output_path = (
        None
        if args.output is None
        else Path(args.output).expanduser().resolve()
    )
    if args.retained:
        CONTRACT.require(output_path is not None, "--retained requires --output")
        CONTRACT.require(
            expected_commit is not None,
            "--retained requires --expected-commit",
        )
    if expected_commit is not None:
        CONTRACT.require(
            re.fullmatch(r"[0-9a-fA-F]{40}", expected_commit) is not None,
            "--expected-commit must be a full 40-character hexadecimal SHA",
        )

    source_before = CONTRACT.git_snapshot(repo_root)
    CONTRACT.validate_source_snapshot(
        source_before,
        expected_commit=expected_commit,
        allow_dirty=not args.retained,
    )
    target_argument = args.model
    draft_argument = args.draft_model or args.model
    target_path, target_before = CONTRACT.model_snapshot(target_argument)
    if Path(draft_argument).expanduser().resolve() == target_path:
        draft_path = target_path
        draft_before = {**target_before, "argument": draft_argument}
    else:
        draft_path, draft_before = CONTRACT.model_snapshot(draft_argument)
    if output_path is not None:
        CONTRACT.require(
            not output_path.exists(),
            f"refusing to overwrite existing output: {output_path}",
        )
        CONTRACT.require(
            not CONTRACT.path_is_within(output_path, repo_root)
            and not CONTRACT.path_is_within(output_path, target_path)
            and not CONTRACT.path_is_within(output_path, draft_path),
            "recovery evidence output must be outside source/model directories",
        )

    args.model = str(target_path)
    draft_model = str(draft_path)
    original_device = torch.get_default_device()
    original_dtype = torch.get_default_dtype()
    original_gc_enabled = gc.isenabled()
    gpu_before = CONTRACT.establish_gpu_gate_before(retained=args.retained)
    gc.collect()
    torch.cuda.empty_cache()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    CONTRACT.require(
        not dist.is_initialized(),
        "recovery gate must start without an active process group",
    )

    restore, ledger = install_phase_injection(args.phase)
    caught = None
    try:
        failed_engine = CONTRACT.construct_speculative(
            args,
            draft_model,
            num_blocks=args.num_kvcache_blocks,
        )
    except InjectedSpeculativePhaseError as error:
        caught = {
            "type": f"{type(error).__module__}.{type(error).__qualname__}",
            "message": str(error),
        }
    except BaseException as error:
        raise AssertionError(
            f"phase {args.phase} failed before the registered injection"
        ) from error
    else:
        failed_engine.exit()
        raise AssertionError(f"phase {args.phase} did not inject a failure")
    finally:
        restore()

    CONTRACT.require(ledger["injections"] == 1, "wrong injection count")
    expected_calls = 2 if args.phase == "draft_graph" else 1
    CONTRACT.require(
        ledger["calls"] == expected_calls,
        f"phase {args.phase} reached {ledger['calls']} calls, expected "
        f"{expected_calls}",
    )
    CONTRACT.assert_process_state(
        original_device,
        original_dtype,
        original_gc_enabled,
    )
    gc.collect()
    torch.cuda.empty_cache()
    failure_allocated = torch.cuda.memory_allocated()
    failure_reserved = torch.cuda.memory_reserved()
    CONTRACT.require(
        failure_allocated
        <= baseline_allocated + MAX_POST_EXECUTION_GLOBAL_ALLOCATED_BYTES,
        "failed constructor retained allocations above the registered global "
        f"compiled-runtime cache ceiling: baseline={baseline_allocated}, "
        f"after={failure_allocated}, "
        f"ceiling={MAX_POST_EXECUTION_GLOBAL_ALLOCATED_BYTES}",
    )
    CONTRACT.require(
        failure_reserved
        <= baseline_reserved + MAX_POST_EXECUTION_GLOBAL_RESERVED_BYTES,
        "failed constructor retained reservations above the registered global "
        f"allocator-cache ceiling: baseline={baseline_reserved}, "
        f"after={failure_reserved}, "
        f"ceiling={MAX_POST_EXECUTION_GLOBAL_RESERVED_BYTES}",
    )

    CONTRACT.seed_all(20260828)
    recovery = CONTRACT.construct_speculative(
        args,
        draft_model,
        num_blocks=args.num_kvcache_blocks,
    )
    recovery_audit, recovery_reconciliation = (
        CONTRACT.assert_audit_matches_live_runner(
            recovery,
            automatic_blocks=False,
            expected_blocks=args.num_kvcache_blocks,
        )
    )
    recovery_outputs, recovery_trace = CONTRACT.generate_fixture(
        recovery,
        prove_draft_inert=True,
    )
    tokenizer_fingerprint = recovery.speculative_tokenizer_fingerprint
    CONTRACT.exit_idempotently(
        recovery,
        original_device,
        original_dtype,
        original_gc_enabled,
    )
    del recovery
    gc.collect()
    torch.cuda.empty_cache()
    recovery_allocated = torch.cuda.memory_allocated()
    recovery_reserved = torch.cuda.memory_reserved()
    CONTRACT.require(
        recovery_allocated
        <= baseline_allocated + MAX_POST_EXECUTION_GLOBAL_ALLOCATED_BYTES,
        "recovery engine teardown exceeded the registered global compiled-"
        f"runtime cache ceiling: baseline={baseline_allocated}, "
        f"after={recovery_allocated}, "
        f"ceiling={MAX_POST_EXECUTION_GLOBAL_ALLOCATED_BYTES}",
    )
    CONTRACT.require(
        recovery_reserved
        <= baseline_reserved + MAX_POST_EXECUTION_GLOBAL_RESERVED_BYTES,
        "recovery engine teardown exceeded the registered global allocator-"
        f"cache ceiling: baseline={baseline_reserved}, "
        f"after={recovery_reserved}, "
        f"ceiling={MAX_POST_EXECUTION_GLOBAL_RESERVED_BYTES}",
    )

    gpu_after = CONTRACT.gpu_gate_snapshot(
        phase="after",
        own_gpu_process_ids=gpu_before["own_gpu_process_ids"],
    )
    if args.retained:
        CONTRACT.validate_retained_gpu_snapshot(gpu_after, phase="after")
        CONTRACT.validate_same_gpu_endpoints(gpu_before, gpu_after)

    _, target_after = CONTRACT.model_snapshot(target_argument)
    if draft_path == target_path:
        draft_after = {**target_after, "argument": draft_argument}
    else:
        _, draft_after = CONTRACT.model_snapshot(draft_argument)
    CONTRACT.require(
        target_after == target_before,
        "target model artifacts changed during recovery gate",
    )
    CONTRACT.require(
        draft_after == draft_before,
        "draft model artifacts changed during recovery gate",
    )
    source_after = CONTRACT.git_snapshot(repo_root)
    CONTRACT.validate_source_snapshot(
        source_after,
        expected_commit=expected_commit,
        allow_dirty=not args.retained,
    )
    CONTRACT.validate_unchanged(source_before, source_after)

    evidence = {
        "evidence_schema": EVIDENCE_SCHEMA,
        "certification_mode": "retained" if args.retained else "exploratory",
        "retention_eligible": args.retained,
        "phase": args.phase,
        "mode": args.mode,
        "injection_semantics": (
            "raised after the selected real phase completed successfully"
        ),
        "injection_ledger": ledger,
        "caught_error": caught,
        "process_group_destroyed_after_failure": True,
        "torch_defaults_and_context_restored_after_failure": True,
        "recovery_engine_succeeded": True,
        "configured_k": args.configured_k,
        "explicit_num_kvcache_blocks": args.num_kvcache_blocks,
        "runtime_import_origins": runtime_import_origins,
        "cuda_memory": {
            "baseline_allocated_bytes": baseline_allocated,
            "baseline_reserved_bytes": baseline_reserved,
            "after_failure_allocated_bytes": failure_allocated,
            "after_failure_reserved_bytes": failure_reserved,
            "after_recovery_exit_allocated_bytes": recovery_allocated,
            "after_recovery_exit_reserved_bytes": recovery_reserved,
            "post_execution_global_allocated_ceiling_bytes": (
                MAX_POST_EXECUTION_GLOBAL_ALLOCATED_BYTES
            ),
            "post_execution_global_reserved_ceiling_bytes": (
                MAX_POST_EXECUTION_GLOBAL_RESERVED_BYTES
            ),
        },
        "model_artifacts": {
            "target": {
                "before": target_before,
                "after": target_after,
                "unchanged_during_run": True,
            },
            "draft": {
                "before": draft_before,
                "after": draft_after,
                "unchanged_during_run": True,
            },
            "same_resolved_model": draft_path == target_path,
        },
        "recovery": {
            "outputs": CONTRACT.nested_dict(recovery_outputs),
            "scheduler_trace": CONTRACT.nested_dict(recovery_trace),
            "tokenizer_fingerprint": tokenizer_fingerprint,
            "memory_audit": CONTRACT.nested_dict(recovery_audit),
            "independent_memory_reconciliation": recovery_reconciliation,
        },
        "scope": {
            "claim": (
                "one injected V2 constructor failure rolls back GPU/process "
                "ownership and permits a healthy same-process recovery"
            ),
            "limitations": [
                "one fresh process is required per phase",
                "the draft remains inert during recovery generation",
                "no speculative-token or performance claim",
                "GPU isolation is checked only at before/after endpoints",
                "a late failure or healthy exit may retain at most the registered 32 MiB allocated / 64 MiB reserved process-global compiled-runtime cache ceilings",
            ],
        },
        "provenance": CONTRACT.collect_provenance(
            repo_root,
            started_at,
            source_before,
            source_after,
            retained=args.retained,
            gpu_before=gpu_before,
            gpu_after=gpu_after,
        ),
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_sha256 = CONTRACT.write_json_exclusive(
            output_path,
            evidence,
        )
        print(f"artifact sha256: {artifact_sha256}")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    print(f"speculative V2 GPU recovery phase {args.phase}: PASS")


if __name__ == "__main__":
    main()
