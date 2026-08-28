"""Strictly compare fresh speculation-off/on V3 output controls."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any, Mapping

from _speculative_v3_evidence import (
    finalize_evidence,
    load_strict_json_bytes,
    prepare_evidence,
    read_registered_file,
    require_independent_compiler_cache_roots,
    validate_output_paths,
    validate_retained_provenance,
    write_json_exclusive,
)


INPUT_SCHEMA = "nano-vllm-speculative-v3-output-control-v2"
OUTPUT_SCHEMA = "nano-vllm-speculative-v3-output-control-comparison-v2"
MAX_JSON_BYTES = 8 * 1024 * 1024
SHA256_RE = re.compile(r"[0-9a-f]{64}")
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
DRAFT_OWNED_ATTRIBUTES = (
    "_draft_route_pretouch_peak_bytes",
    "_draft_warmup_transient_bytes",
    "_speculative_memory_audit_inputs",
    "draft_graph_bs",
    "draft_graph_pool",
    "draft_graph_vars",
    "draft_graphs",
    "draft_kv_cache",
    "draft_model",
    "draft_route_registry",
    "speculative_memory_audit",
    "speculative_memory_plan",
)
EXPECTED_RUNTIME_IMPORTS = {
    "nanovllm": "nanovllm/__init__.py",
    "LLM": "nanovllm/llm.py",
    "LLMEngine": "nanovllm/engine/llm_engine.py",
    "ModelRunner": "nanovllm/engine/model_runner.py",
    "Scheduler": "nanovllm/engine/scheduler.py",
    "Sampler": "nanovllm/layers/sampler.py",
    "SamplingParams": "nanovllm/sampling_params.py",
    "DraftRouteRegistry": "nanovllm/engine/speculative_routes.py",
}


def require(condition: Any, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Require exact off/on V3 target-output and RNG parity."
    )
    parser.add_argument("--off", required=True)
    parser.add_argument("--on", required=True)
    parser.add_argument("--evidence-root")
    parser.add_argument(
        "--expected-commit",
        help="full comparator/producer commit required by --retained",
    )
    parser.add_argument(
        "--retained",
        action="store_true",
        help="enforce exact clean-SHA producer and comparator provenance",
    )
    parser.add_argument("--output")
    return parser.parse_args(argv)


def is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def confined_existing_path(path: Path, root: Path) -> Path:
    lexical = path.expanduser()
    if not lexical.is_absolute():
        lexical = Path.cwd() / lexical
    current = lexical.absolute()
    require(is_within(current, root), f"artifact escapes evidence root: {path}")
    while True:
        require(not current.is_symlink(), f"artifact path contains a symlink: {current}")
        if current == root:
            break
        current = current.parent
    resolved = lexical.resolve()
    require(is_within(resolved, root), f"artifact resolves outside evidence root: {path}")
    return resolved


def require_sha256(value: Any, name: str) -> str:
    require(
        isinstance(value, str) and SHA256_RE.fullmatch(value),
        f"{name} must be lowercase SHA-256",
    )
    return value


def validate_rng(value: Any, name: str) -> None:
    require(
        isinstance(value, dict)
        and set(value) == {"cpu_sha256", "cuda_sha256"},
        f"{name} RNG schema drifted",
    )
    require_sha256(value["cpu_sha256"], f"{name}.cpu_sha256")
    require_sha256(value["cuda_sha256"], f"{name}.cuda_sha256")


def validate_event(value: Any, *, stage: str, seq_id: int) -> None:
    require(
        isinstance(value, dict)
        and set(value) == {"stage", "seq_id", "token_id", "finished"},
        f"{stage} event schema drifted",
    )
    require(value["stage"] == stage, f"{stage} event stage drifted")
    require(type(value["seq_id"]) is int and value["seq_id"] == seq_id, f"{stage} sequence ID drifted")
    require(type(value["token_id"]) is int and value["token_id"] >= 0, f"{stage} token ID drifted")
    require(value["finished"] is False, f"{stage} event unexpectedly finished")


def validate_route(value: Any, *, mode: str, cold: bool) -> None:
    require(isinstance(value, dict), "runtime route must be an object")
    require(
        value
        == {
            "schema": "draft-discard-v1",
            "execution_mode": "eager_dynamic" if mode == "eager" else "cuda_graph",
            "batch_bucket": 2,
            "effective_k": 2,
            "catchup_family": "paged_eager_dynamic_v1" if cold else "none",
            "sampler_envelope": "exact_all_compositions_worst_case_v1",
        },
        "runtime route contract drifted",
    )


def validate_raw_artifact(value: Any, side: str) -> None:
    require(isinstance(value, dict), f"{side} artifact must be an object")
    require(value.get("schema") == INPUT_SCHEMA, f"invalid {side} schema")
    require(value.get("side") == side, f"expected {side} artifact")
    mode = value.get("mode")
    require(mode in ("eager", "graph"), f"invalid {side} mode")
    require(value.get("seed") == 20260828, f"{side} seed drifted")
    require(value.get("configured_k") == 2, f"{side} configured K drifted")
    require(
        value.get("configuration")
        == {
            "max_model_len": 512,
            "max_num_batched_tokens": 512,
            "max_num_seqs": 2,
            "gpu_memory_utilization": 0.5,
            "top_p_backend": "exact",
            "tensor_parallel_size": 1,
        },
        f"{side} configuration drifted",
    )
    require(
        value.get("workload")
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
        f"{side} workload drifted",
    )
    require(isinstance(value.get("model"), str), f"{side} target model path is invalid")
    require(isinstance(value.get("draft_model_argument"), str), f"{side} draft model path is invalid")
    require(type(value.get("retention_eligible")) is bool, f"{side} retention flag is invalid")
    provenance = value.get("provenance")
    require(isinstance(provenance, dict), f"{side} provenance is missing")
    require(
        provenance.get("retention_eligible") == value["retention_eligible"],
        f"{side} retention flag disagrees with provenance",
    )

    phase_calls = value.get("draft_phase_calls")
    require(
        isinstance(phase_calls, dict) and set(phase_calls) == set(DRAFT_PHASES),
        f"{side} phase ledger drifted",
    )
    expected_on_phase_calls = {
        "_construct_draft_model": 1,
        "warmup_draft_model": 1,
        "capture_draft_cudagraph": 0 if mode == "eager" else 2,
        "_pretouch_draft_eager_prefill": 0 if mode == "eager" else 1,
        "_pretouch_draft_routes": 1,
    }
    expected_phase_calls = (
        {name: 0 for name in DRAFT_PHASES}
        if side == "off"
        else expected_on_phase_calls
    )
    require(phase_calls == expected_phase_calls, f"{side} phase counts drifted")

    resources = value.get("live_draft_resource_attributes")
    require(
        isinstance(resources, dict)
        and set(resources) == set(DRAFT_RESOURCE_ATTRIBUTES)
        and all(type(item) is bool for item in resources.values()),
        f"{side} resource ledger drifted",
    )
    require(
        all(item is (side == "on") for item in resources.values()),
        f"{side} draft resource state drifted",
    )
    owned = value.get("draft_owned_instance_attributes")
    require(
        owned == ([] if side == "off" else list(DRAFT_OWNED_ATTRIBUTES)),
        f"{side} draft-owned attribute registry drifted",
    )

    calls = value.get("runtime_draft_calls")
    require(isinstance(calls, list), f"{side} runtime call ledger is invalid")
    if side == "off":
        require(not calls, "off artifact ran V3 draft work")
    else:
        require(len(calls) == 2, "on artifact needs two V3 intervals")
        for index, call in enumerate(calls):
            require(isinstance(call, dict), "runtime draft call must be an object")
            validate_route(call.get("route"), mode=mode, cold=index == 0)
            require(
                call.get("catchup_tokens") == (7 if index == 0 else 0),
                "runtime catch-up coverage drifted",
            )
            proposals = call.get("proposal_token_ids")
            require(
                isinstance(proposals, list)
                and len(proposals) == 2
                and all(
                    isinstance(row, list)
                    and len(row) == 2
                    and all(type(token) is int and token >= 0 for token in row)
                    for row in proposals
                ),
                "runtime proposal shape drifted",
            )
            require(
                call.get("eager_decode_steps") == (2 if mode == "eager" else 0)
                and call.get("graph_decode_steps") == (0 if mode == "eager" else 2),
                "runtime draft step count drifted",
            )
            validate_rng(call.get("rng_before"), "runtime before")
            validate_rng(call.get("rng_after"), "runtime after")
            require(call["rng_before"] == call["rng_after"], "runtime draft changed target RNG")
            require(call.get("rng_neutral") is True, "runtime draft is not marked RNG neutral")

    rng_snapshots = value.get("rng_snapshots")
    expected_rng_names = {
        "after_init",
        "after_prefill",
        "after_first_target_decode",
        "after_repeated_target_decode",
    }
    require(
        isinstance(rng_snapshots, dict)
        and set(rng_snapshots) == expected_rng_names,
        f"{side} RNG endpoint registry drifted",
    )
    for name, rng in rng_snapshots.items():
        validate_rng(rng, name)

    steps = value.get("steps")
    stage_specs = (
        ("prefill", 7, 0),
        ("first_target_decode", 0, 2),
        ("repeated_target_decode", 0, 2),
    )
    require(isinstance(steps, list) and len(steps) == 3, f"{side} step ledger drifted")
    flattened = []
    for step, (stage, prefill_tokens, decode_tokens) in zip(
        steps, stage_specs, strict=True
    ):
        require(
            isinstance(step, dict)
            and step.get("stage") == stage
            and step.get("num_prefill_tokens") == prefill_tokens
            and step.get("num_decode_tokens") == decode_tokens,
            f"{side} {stage} accounting drifted",
        )
        events = step.get("events")
        require(isinstance(events, list) and len(events) == 2, f"{side} {stage} events drifted")
        for event, seq_id in zip(events, (1, 2), strict=True):
            validate_event(event, stage=stage, seq_id=seq_id)
        flattened.extend(events)
    require(
        value.get("authoritative_target_events") == flattened,
        f"{side} authoritative event ledger drifted",
    )
    by_seq = value.get("target_token_ids_by_seq")
    require(
        isinstance(by_seq, dict)
        and set(by_seq) == {"1", "2"}
        and all(
            tokens
            == [
                event["token_id"]
                for event in flattened
                if event["seq_id"] == int(seq_id)
            ]
            for seq_id, tokens in by_seq.items()
        ),
        f"{side} per-sequence target token ledger drifted",
    )


def load_artifact(path: Path, side: str, root: Path):
    path = confined_existing_path(path, root)
    payload, descriptor = read_registered_file(path, max_bytes=MAX_JSON_BYTES)
    value = load_strict_json_bytes(payload)
    validate_raw_artifact(value, side)
    return value, {"path": path.relative_to(root).as_posix(), **descriptor}


def normalized_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(value)
    environment = dict(normalized["environment"])
    selected = dict(environment["selected_environment"])
    selected.pop("TORCHINDUCTOR_CACHE_DIR", None)
    selected.pop("TRITON_CACHE_DIR", None)
    environment["selected_environment"] = selected
    normalized["environment"] = environment
    return normalized


def main(argv=None):
    args = parse_args(argv)
    off_argument = Path(args.off).expanduser()
    on_argument = Path(args.on).expanduser()
    if args.evidence_root is None:
        require(not args.retained, "--retained requires --evidence-root")
        root = Path(
            os.path.commonpath(
                [
                    str(off_argument.absolute().parent),
                    str(on_argument.absolute().parent),
                ]
            )
        ).resolve()
    else:
        root = Path(args.evidence_root).expanduser().resolve()
    require(root.is_dir() and not root.is_symlink(), "evidence root must be a real directory")

    evidence_context = prepare_evidence(
        script_path=Path(__file__).resolve(),
        retained=args.retained,
        expected_commit=args.expected_commit,
        model_arguments={},
        runtime_imports={},
    )
    cache_roots = tuple(
        Path(value).expanduser().resolve()
        for value in (
            os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
            os.environ.get("TRITON_CACHE_DIR"),
        )
        if value
    )
    output_path = None
    if args.output is not None:
        output_path = validate_output_paths(
            (Path(args.output),),
            repo_root=evidence_context.repo_root,
            cache_roots=cache_roots,
        )[0]
        require(is_within(output_path, root), "comparison output must be inside evidence root")
    elif args.retained:
        raise AssertionError("--retained requires --output")

    off, off_descriptor = load_artifact(off_argument, "off", root)
    on, on_descriptor = load_artifact(on_argument, "on", root)
    require(off_descriptor["path"] != on_descriptor["path"], "off/on inputs must differ")
    for field in (
        "mode",
        "seed",
        "model",
        "draft_model_argument",
        "configured_k",
        "configuration",
        "workload",
        "rng_snapshots",
        "steps",
        "authoritative_target_events",
        "target_token_ids_by_seq",
    ):
        require(off[field] == on[field], f"off/on {field} mismatch")

    if args.retained:
        expected_commit = args.expected_commit.lower()
        model_snapshot_cache = {}
        off_provenance = validate_retained_provenance(
            off["provenance"],
            repo_root=evidence_context.repo_root,
            current_source=evidence_context.source_before,
            expected_commit=expected_commit,
            expected_runner_path="tests/run_speculative_v3_output_control.py",
            expected_model_roles=("target", "draft"),
            expected_runtime_imports=EXPECTED_RUNTIME_IMPORTS,
            model_snapshot_cache=model_snapshot_cache,
        )
        on_provenance = validate_retained_provenance(
            on["provenance"],
            repo_root=evidence_context.repo_root,
            current_source=evidence_context.source_before,
            expected_commit=expected_commit,
            expected_runner_path="tests/run_speculative_v3_output_control.py",
            expected_model_roles=("target", "draft"),
            expected_runtime_imports=EXPECTED_RUNTIME_IMPORTS,
            model_snapshot_cache=model_snapshot_cache,
        )
        require_independent_compiler_cache_roots(
            (off_provenance, on_provenance)
        )
        pair_identity = normalized_provenance(off_provenance)
        require(
            pair_identity == normalized_provenance(on_provenance),
            "off/on validated producer provenance differs",
        )
        for artifact, identity in ((off, off_provenance), (on, on_provenance)):
            require(
                Path(artifact["model"]).resolve()
                == Path(identity["models"]["target"]["resolved_path"]),
                "raw target model path disagrees with hashed provenance",
            )
            require(
                Path(artifact["draft_model_argument"]).resolve()
                == Path(identity["models"]["draft"]["resolved_path"]),
                "raw draft model path disagrees with hashed provenance",
            )
        producer_provenance = {
            "pair_identity": pair_identity,
            "off": off_provenance,
            "on": on_provenance,
        }
        if output_path is not None:
            validate_output_paths(
                (output_path,),
                repo_root=evidence_context.repo_root,
                model_roots=tuple(
                    Path(identity["resolved_path"])
                    for identity in off_provenance["models"].values()
                ),
                cache_roots=cache_roots,
            )
    else:
        producer_provenance = {
            "off": off["provenance"],
            "on": on["provenance"],
        }

    comparator_provenance = finalize_evidence(evidence_context)
    retention_eligible = bool(
        off["retention_eligible"]
        and on["retention_eligible"]
        and comparator_provenance["retention_eligible"]
    )
    if args.retained:
        require(retention_eligible, "off/on comparison is not retention eligible")
    result = {
        "schema": OUTPUT_SCHEMA,
        "verdict": "pass",
        "mode": off["mode"],
        "seed": off["seed"],
        "retention_eligible": retention_eligible,
        "off_artifact": off_descriptor,
        "on_artifact": on_descriptor,
        "producer_provenance": producer_provenance,
        "comparator_provenance": comparator_provenance,
        "off_draft_phase_calls_zero": True,
        "off_draft_resources_absent": True,
        "on_real_v3_intervals": len(on["runtime_draft_calls"]),
        "on_cold_and_warm_routes_exercised": True,
        "target_events_exact": True,
        "target_tokens_exact": True,
        "rng_endpoints_exact": True,
    }
    if output_path is not None:
        write_json_exclusive(output_path, result)
        print(
            "PASS: off/on authoritative target events, tokens, and all four "
            f"RNG endpoints match exactly ({off['mode']}); "
            f"retention_eligible={retention_eligible}, output={output_path}",
            flush=True,
        )
    else:
        import json

        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
