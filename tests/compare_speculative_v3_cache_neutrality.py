"""Strictly compare paired zero/NaN speculative-V3 cache artifacts."""

from __future__ import annotations

import argparse
import io
import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import torch

from _speculative_v3_evidence import (
    finalize_evidence,
    load_strict_json_bytes,
    prepare_evidence,
    read_registered_file,
    require_independent_compiler_cache_roots,
    sha256_bytes,
    validate_retained_provenance,
    validate_output_paths,
    write_json_exclusive,
)


INPUT_SCHEMA = "nano-vllm-speculative-v3-cache-neutrality-v2"
OUTPUT_SCHEMA = "nano-vllm-speculative-v3-cache-neutrality-comparison-v2"
EXPECTED_VOCAB_SIZE = 151936
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_TENSOR_BYTES = 64 * 1024 * 1024
SHA256_RE = re.compile(r"[0-9a-f]{64}")
RECORD_IDS = (
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
CYCLE_SPECS = (
    ("boundary", ("boundary",), [255, 256, 257], 255),
    (
        "shared-prefix/cold",
        ("shared_prefix", "cold"),
        [257, 258, 259],
        257,
    ),
    (
        "shared-prefix/hit",
        ("shared_prefix", "shared_prefix"),
        [257, 258, 259],
        257,
    ),
)


def require(condition: Any, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("zero_json")
    parser.add_argument("nan_json")
    parser.add_argument(
        "--evidence-root",
        help="root containing both JSON artifacts and their relative sidecars",
    )
    parser.add_argument(
        "--expected-commit",
        help="full comparator/producer commit required by --retained",
    )
    parser.add_argument(
        "--retained",
        action="store_true",
        help="enforce the exact clean-SHA retained-evidence contract",
    )
    parser.add_argument("--output")
    return parser.parse_args(argv)


def require_exact_bool(value: Any, name: str) -> bool:
    require(type(value) is bool, f"{name} must be a JSON boolean")
    return value


def require_exact_int(value: Any, name: str) -> int:
    require(type(value) is int, f"{name} must be a JSON integer")
    return value


def require_sha256(value: Any, name: str) -> str:
    require(isinstance(value, str) and SHA256_RE.fullmatch(value), f"{name} must be lowercase SHA-256")
    return value


def is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def require_no_symlink_components(path: Path, root: Path) -> None:
    path = path.absolute()
    root = root.absolute()
    require(is_within(path, root), f"artifact escapes evidence root: {path}")
    current = path
    while True:
        require(not current.is_symlink(), f"artifact path contains a symlink: {current}")
        if current == root:
            break
        current = current.parent


def confined_existing_path(path: Path, root: Path) -> Path:
    lexical = path.expanduser()
    if not lexical.is_absolute():
        lexical = Path.cwd() / lexical
    require_no_symlink_components(lexical, root)
    resolved = lexical.resolve()
    require(is_within(resolved, root), f"artifact resolves outside evidence root: {path}")
    return resolved


def resolve_relative_sidecar(raw_path: Path, value: Any, root: Path) -> Path:
    require(isinstance(value, str) and value, "tensor sidecar path must be a non-empty string")
    require("\\" not in value, "tensor sidecar path must use POSIX separators")
    relative = PurePosixPath(value)
    require(not relative.is_absolute(), "tensor sidecar path must be relative")
    require(
        relative.parts
        and all(part not in ("", ".", "..") for part in relative.parts),
        "tensor sidecar path must be normalized and confined",
    )
    require(relative.as_posix() == value, "tensor sidecar path is not normalized")
    lexical = raw_path.parent.joinpath(*relative.parts)
    return confined_existing_path(lexical, root)


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    metadata = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload = metadata + b"\0" + value.view(torch.uint8).numpy().tobytes()
    return sha256_bytes(payload)


def nested(value: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = value
    for name in path:
        require(isinstance(current, dict) and name in current, f"missing cycle field: {'/'.join(path)}")
        current = current[name]
    return current


def validate_route(route: Any, *, mode: str, cycle_name: str) -> None:
    require(isinstance(route, dict), f"{cycle_name} route must be an object")
    expected = {
        "schema": "draft-discard-v1",
        "execution_mode": "eager_dynamic" if mode == "eager" else "cuda_graph",
        "batch_bucket": 1,
        "effective_k": 3,
        "catchup_family": "paged_eager_dynamic_v1",
        "sampler_envelope": "exact_all_compositions_worst_case_v1",
    }
    require(route == expected, f"{cycle_name} route contract drifted")


def validate_cycle(
    artifact: Mapping[str, Any],
    *,
    mode: str,
    cycle_name: str,
    path: tuple[str, ...],
    expected_positions: list[int],
    expected_catchup: int,
    record_offset: int,
) -> list[dict[str, Any]]:
    cycle = nested(artifact, path)
    require(isinstance(cycle, dict), f"{cycle_name} must be an object")
    require_exact_int(cycle.get("effective_k"), f"{cycle_name}.effective_k")
    require(cycle["effective_k"] == 3, f"{cycle_name} K drifted")
    require(
        cycle.get("proposal_input_positions") == expected_positions,
        f"{cycle_name} proposal positions drifted",
    )
    require_exact_int(cycle.get("catchup_tokens"), f"{cycle_name}.catchup_tokens")
    require(cycle["catchup_tokens"] == expected_catchup, f"{cycle_name} catch-up drifted")
    validate_route(cycle.get("route"), mode=mode, cycle_name=cycle_name)
    if mode == "eager":
        require(cycle.get("eager_decode_steps") == 3, f"{cycle_name} eager step count drifted")
        require(cycle.get("graph_decode_steps") == 0, f"{cycle_name} graph step count drifted")
    else:
        require(cycle.get("eager_decode_steps") == 0, f"{cycle_name} eager step count drifted")
        require(cycle.get("graph_decode_steps") == 3, f"{cycle_name} graph step count drifted")
    rng_before = cycle.get("rng_before_draft")
    rng_after = cycle.get("rng_after_draft")
    require(
        isinstance(rng_before, dict)
        and set(rng_before) == {"cpu", "cuda"}
        and all(
            isinstance(value, str) and SHA256_RE.fullmatch(value)
            for value in rng_before.values()
        ),
        f"{cycle_name} RNG endpoint schema drifted",
    )
    require(rng_before == rng_after, f"{cycle_name} draft interval was not RNG neutral")
    proposals = cycle.get("proposed_token_ids")
    require(
        isinstance(proposals, list)
        and len(proposals) == 1
        and isinstance(proposals[0], list)
        and len(proposals[0]) == 3
        and all(type(token) is int for token in proposals[0]),
        f"{cycle_name} proposal shape drifted",
    )
    target = cycle.get("target_token_ids")
    require(
        isinstance(target, list)
        and len(target) == 1
        and type(target[0]) is int,
        f"{cycle_name} target token shape drifted",
    )
    records = cycle.get("sampler_records")
    require(isinstance(records, list) and len(records) == 3, f"{cycle_name} needs three sampler records")
    expected_ids = list(RECORD_IDS[record_offset : record_offset + 3])
    require(
        [record.get("record_id") for record in records] == expected_ids,
        f"{cycle_name} sampler record IDs drifted",
    )
    for step, record in enumerate(records):
        require(isinstance(record, dict), f"{cycle_name} sampler record must be an object")
        require(record.get("token_ids") == [proposals[0][step]], f"{cycle_name} sampler/proposal token mismatch")
        require_exact_bool(record.get("logits_finite"), f"{cycle_name}.logits_finite")
        require_exact_bool(record.get("probabilities_finite"), f"{cycle_name}.probabilities_finite")
        require(record["logits_finite"], f"{cycle_name} records non-finite logits")
        require(record["probabilities_finite"], f"{cycle_name} records non-finite probabilities")
        require_sha256(record.get("logits_sha256"), f"{cycle_name}.logits_sha256")
        require_sha256(record.get("probabilities_sha256"), f"{cycle_name}.probabilities_sha256")
        require(record.get("logits_dtype") == "torch.bfloat16", f"{cycle_name} logits dtype drifted")
        require(record.get("probabilities_dtype") == "torch.float32", f"{cycle_name} probability dtype drifted")
        require(record.get("logits_shape") == [1, EXPECTED_VOCAB_SIZE], f"{cycle_name} logits shape drifted")
        require(record.get("probabilities_shape") == [1, EXPECTED_VOCAB_SIZE], f"{cycle_name} probability shape drifted")
        row_sums = record.get("probability_row_sums")
        require(
            isinstance(row_sums, list)
            and len(row_sums) == 1
            and type(row_sums[0]) is float
            and abs(row_sums[0] - 1.0) <= 1e-6,
            f"{cycle_name} probability row mass drifted",
        )
    target_tables = cycle.get("tables_after_target_schedule")
    reservation_tables = cycle.get("tables_during_reservation")
    handoff_tables = cycle.get("tables_after_handoff")
    for name, tables in (
        ("target", target_tables),
        ("reservation", reservation_tables),
        ("handoff", handoff_tables),
    ):
        require(
            isinstance(tables, list)
            and len(tables) == 1
            and isinstance(tables[0], list)
            and tables[0]
            and all(type(block_id) is int and block_id >= 0 for block_id in tables[0])
            and len(set(tables[0])) == len(tables[0]),
            f"{cycle_name} {name} block table drifted",
        )
    require(
        handoff_tables == target_tables,
        f"{cycle_name} did not restore target block tables",
    )
    require(
        reservation_tables[0][: len(target_tables[0])] == target_tables[0],
        f"{cycle_name} reservation did not preserve the target table prefix",
    )
    temporary_blocks = require_exact_int(
        cycle.get("temporary_blocks"),
        f"{cycle_name}.temporary_blocks",
    )
    require(
        temporary_blocks
        == len(set(reservation_tables[0]) - set(target_tables[0])),
        f"{cycle_name} temporary block accounting drifted",
    )
    expected_temporary_blocks = 1 if cycle_name == "boundary" else 0
    require(
        temporary_blocks == expected_temporary_blocks,
        f"{cycle_name} temporary block count drifted",
    )
    require(
        cycle.get("filled_physical_blocks")
        == sorted(set(reservation_tables[0])),
        f"{cycle_name} did not poison every reserved physical block",
    )
    return records


def validate_raw_artifact(value: Any, expected_fill: str) -> list[dict[str, Any]]:
    require(isinstance(value, dict), "raw artifact must be a JSON object")
    require(value.get("schema") == INPUT_SCHEMA, "unexpected raw artifact schema")
    require(value.get("draft_cache_fill") == expected_fill, "unexpected cache fill side")
    mode = value.get("mode")
    require(mode in ("eager", "graph"), "invalid execution mode")
    require(value.get("seed") == 20260828, "cache evidence seed drifted")
    require(
        value.get("configuration")
        == {
            "num_speculative_tokens": 3,
            "gpu_memory_utilization": 0.5,
            "max_model_len": 512,
            "max_num_batched_tokens": 512,
            "max_num_seqs": 1,
            "tensor_parallel_size": 1,
            "top_p_backend": "exact",
        },
        "cache evidence configuration drifted",
    )
    require(
        value.get("workload")
        == {
            "neutral_boundary_warmup": {
                "length": 255,
                "salt": 911,
                "fill": "zero",
                "proposal_positions": [255, 256, 257],
                "sampler_records_discarded": 3,
            },
            "boundary_prompt": {
                "length": 255,
                "salt": 31,
                "proposal_positions": [255, 256, 257],
            },
            "shared_prefix_prompt": {
                "length": 257,
                "salt": 73,
                "proposal_positions": [257, 258, 259],
            },
            "sampling": {
                "temperature": 0.0,
                "top_k": -1,
                "top_p": 1.0,
                "ignore_eos": True,
            },
        },
        "cache evidence workload drifted",
    )
    warmup = value.get("measurement_warmup")
    require(
        isinstance(warmup, dict)
        and set(warmup)
        == {
            "fill",
            "prompt_length",
            "prompt_salt",
            "proposal_input_positions",
            "catchup_tokens",
            "temporary_blocks",
            "proposed_token_ids",
            "target_token_ids",
            "sampler_records_discarded",
        },
        "cache evidence warmup schema drifted",
    )
    require(
        warmup["fill"] == "zero"
        and warmup["prompt_length"] == 255
        and warmup["prompt_salt"] == 911
        and warmup["proposal_input_positions"] == [255, 256, 257]
        and warmup["catchup_tokens"] == 255
        and warmup["temporary_blocks"] == 1
        and warmup["sampler_records_discarded"] == 3,
        "cache evidence neutral warmup contract drifted",
    )
    require(
        isinstance(warmup["proposed_token_ids"], list)
        and len(warmup["proposed_token_ids"]) == 1
        and isinstance(warmup["proposed_token_ids"][0], list)
        and len(warmup["proposed_token_ids"][0]) == 3
        and all(type(token) is int for token in warmup["proposed_token_ids"][0])
        and isinstance(warmup["target_token_ids"], list)
        and len(warmup["target_token_ids"]) == 1
        and type(warmup["target_token_ids"][0]) is int,
        "cache evidence neutral warmup token shape drifted",
    )
    require_exact_bool(value.get("all_logits_finite"), "all_logits_finite")
    require_exact_bool(value.get("all_probabilities_finite"), "all_probabilities_finite")
    require(value["all_logits_finite"], "artifact reports non-finite logits")
    require(value["all_probabilities_finite"], "artifact reports non-finite probabilities")
    require_exact_bool(value.get("retention_eligible"), "retention_eligible")
    provenance = value.get("provenance")
    require(isinstance(provenance, dict), "raw artifact lacks provenance")
    require(
        provenance.get("retention_eligible") == value["retention_eligible"],
        "raw retention flag disagrees with provenance",
    )
    records = []
    for index, (cycle_name, path, positions, catchup) in enumerate(CYCLE_SPECS):
        records.extend(
            validate_cycle(
                value,
                mode=mode,
                cycle_name=cycle_name,
                path=path,
                expected_positions=positions,
                expected_catchup=catchup,
                record_offset=index * 3,
            )
        )
    require([record["record_id"] for record in records] == list(RECORD_IDS), "raw record order drifted")
    return records


def validate_tensor_record(
    tensor_record: Any,
    raw_record: Mapping[str, Any],
    expected_id: str,
) -> dict[str, Any]:
    require(
        isinstance(tensor_record, dict)
        and set(tensor_record) == {"record_id", "logits", "probabilities"},
        f"{expected_id} tensor record schema drifted",
    )
    require(tensor_record["record_id"] == expected_id, f"{expected_id} sidecar ID drifted")
    logits = tensor_record["logits"]
    probabilities = tensor_record["probabilities"]
    require(isinstance(logits, torch.Tensor), f"{expected_id} logits are not a tensor")
    require(isinstance(probabilities, torch.Tensor), f"{expected_id} probabilities are not a tensor")
    require(logits.device.type == "cpu" and probabilities.device.type == "cpu", f"{expected_id} sidecar tensors must be CPU")
    require(logits.layout == torch.strided and probabilities.layout == torch.strided, f"{expected_id} tensors must be strided")
    require(logits.is_contiguous() and probabilities.is_contiguous(), f"{expected_id} tensors must be contiguous")
    require(logits.dtype == torch.bfloat16, f"{expected_id} logits must retain BF16")
    require(probabilities.dtype == torch.float32, f"{expected_id} probabilities must be FP32")
    expected_shape = (1, EXPECTED_VOCAB_SIZE)
    require(tuple(logits.shape) == expected_shape, f"{expected_id} logits shape drifted")
    require(tuple(probabilities.shape) == expected_shape, f"{expected_id} probability shape drifted")
    require(bool(torch.isfinite(logits).all()), f"{expected_id} logits are non-finite")
    require(bool(torch.isfinite(probabilities).all()), f"{expected_id} probabilities are non-finite")
    require(not bool((probabilities < 0).any()), f"{expected_id} probabilities are negative")
    row_mass = probabilities.sum(dim=-1, dtype=torch.float64)
    require(bool(torch.allclose(row_mass, torch.ones_like(row_mass), atol=1e-6, rtol=0)), f"{expected_id} probability row is not normalized")
    require(tensor_sha256(logits) == raw_record["logits_sha256"], f"{expected_id} logits do not match JSON hash")
    require(tensor_sha256(probabilities) == raw_record["probabilities_sha256"], f"{expected_id} probabilities do not match JSON hash")
    require(list(logits.shape) == raw_record["logits_shape"], f"{expected_id} JSON logits shape drifted")
    require(list(probabilities.shape) == raw_record["probabilities_shape"], f"{expected_id} JSON probability shape drifted")
    require(str(logits.dtype) == raw_record["logits_dtype"], f"{expected_id} JSON logits dtype drifted")
    require(str(probabilities.dtype) == raw_record["probabilities_dtype"], f"{expected_id} JSON probability dtype drifted")
    require(
        probabilities.argmax(dim=-1).tolist() == raw_record["token_ids"],
        f"{expected_id} greedy token does not match retained probability law",
    )
    expected_probabilities = torch.zeros_like(probabilities)
    expected_probabilities.scatter_(
        -1,
        logits.argmax(dim=-1, keepdim=True),
        1.0,
    )
    require(
        torch.equal(probabilities, expected_probabilities),
        f"{expected_id} retained probability law is not exact greedy one-hot",
    )
    return {"record_id": expected_id, "logits": logits, "probabilities": probabilities}


def load_sidecar(
    raw_path: Path,
    value: Mapping[str, Any],
    raw_records: list[dict[str, Any]],
    root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    descriptor = value.get("tensor_artifact")
    require(isinstance(descriptor, dict), "tensor_artifact must be an object")
    require(
        set(descriptor)
        == {"path", "format", "size_bytes", "sha256", "record_count"},
        "tensor_artifact schema drifted",
    )
    require(descriptor["format"] == "torch-save-weights-only", "tensor format drifted")
    require_exact_int(descriptor["size_bytes"], "tensor_artifact.size_bytes")
    require_exact_int(descriptor["record_count"], "tensor_artifact.record_count")
    require(descriptor["record_count"] == len(RECORD_IDS), "tensor record count drifted")
    require_sha256(descriptor["sha256"], "tensor_artifact.sha256")
    sidecar_path = resolve_relative_sidecar(raw_path, descriptor["path"], root)
    payload, observed = read_registered_file(sidecar_path, max_bytes=MAX_TENSOR_BYTES)
    require(observed["size_bytes"] == descriptor["size_bytes"], "tensor sidecar size mismatch")
    require(observed["sha256"] == descriptor["sha256"], "tensor sidecar SHA-256 mismatch")
    loaded = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    require(isinstance(loaded, dict), "tensor sidecar root must be a dictionary")
    require(
        set(loaded) == {"schema", "mode", "draft_cache_fill", "records"},
        "tensor sidecar root schema drifted",
    )
    require(loaded["schema"] == INPUT_SCHEMA, "tensor sidecar schema drifted")
    require(loaded["mode"] == value["mode"], "tensor sidecar mode drifted")
    require(
        loaded["draft_cache_fill"] == value["draft_cache_fill"],
        "tensor sidecar fill mode drifted",
    )
    tensor_records = loaded["records"]
    require(isinstance(tensor_records, list) and len(tensor_records) == len(RECORD_IDS), "tensor sidecar needs exactly nine records")
    validated = [
        validate_tensor_record(tensor_record, raw_record, expected_id)
        for tensor_record, raw_record, expected_id in zip(
            tensor_records, raw_records, RECORD_IDS, strict=True
        )
    ]
    return validated, {
        "path": sidecar_path.relative_to(root).as_posix(),
        **observed,
        "record_count": len(validated),
    }


def load_artifact(path: Path, expected_fill: str, root: Path):
    path = confined_existing_path(path, root)
    raw_payload, raw_descriptor = read_registered_file(path, max_bytes=MAX_JSON_BYTES)
    value = load_strict_json_bytes(raw_payload)
    raw_records = validate_raw_artifact(value, expected_fill)
    tensor_records, sidecar_descriptor = load_sidecar(path, value, raw_records, root)
    return value, tensor_records, {
        "path": path.relative_to(root).as_posix(),
        **raw_descriptor,
        "tensor_sidecar": sidecar_descriptor,
    }


def stable_provenance_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    provenance = value["provenance"]
    environment_record = provenance["environment"]
    require(
        environment_record.get("unchanged") is True,
        "producer environment changed during the run",
    )
    environment = environment_record["before"]
    selected = dict(environment["selected_environment"])
    selected.pop("TORCHINDUCTOR_CACHE_DIR", None)
    selected.pop("TRITON_CACHE_DIR", None)
    return {
        "implementation": provenance["implementation"],
        "producer_commit": provenance["producer_commit"],
        "source": provenance["source"],
        "source_files": provenance["source_files"],
        "runtime_imports": provenance["runtime_imports"],
        "models": provenance["models"],
        "software": environment["software"],
        "hardware": environment["hardware"],
        "selected_environment": selected,
    }


def normalized_validated_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(value)
    environment = dict(normalized["environment"])
    selected = dict(environment["selected_environment"])
    selected.pop("TORCHINDUCTOR_CACHE_DIR", None)
    selected.pop("TRITON_CACHE_DIR", None)
    environment["selected_environment"] = selected
    normalized["environment"] = environment
    return normalized


def host_oracle(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key
        not in {
            "generated_at",
            "draft_cache_fill",
            "tensor_artifact",
            "provenance",
            "retention_eligible",
        }
    }


def compare_tensor_records(
    zero_records: list[dict[str, Any]],
    nan_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    require(len(zero_records) == len(nan_records) == len(RECORD_IDS), "paired record count drifted")
    comparisons = []
    for expected_id, zero, nan in zip(RECORD_IDS, zero_records, nan_records, strict=True):
        require(zero["record_id"] == nan["record_id"] == expected_id, f"{expected_id} pair ID drifted")
        zero_logits = zero["logits"]
        nan_logits = nan["logits"]
        zero_probabilities = zero["probabilities"]
        nan_probabilities = nan["probabilities"]
        require(torch.equal(zero_logits, nan_logits), f"{expected_id} zero/NaN logits differ")
        require(
            torch.equal(zero_probabilities, nan_probabilities),
            f"{expected_id} zero/NaN probabilities differ",
        )
        difference = (zero_logits.float() - nan_logits.float()).abs()
        comparisons.append(
            {
                "record_id": expected_id,
                "shape": list(zero_logits.shape),
                "logits_bitwise_equal": True,
                "probabilities_bitwise_equal": True,
                "max_abs_difference": float(difference.max().item()),
            }
        )
    return comparisons


def main(argv=None):
    args = parse_args(argv)
    zero_argument = Path(args.zero_json).expanduser()
    nan_argument = Path(args.nan_json).expanduser()
    if args.evidence_root is None:
        require(not args.retained, "--retained requires --evidence-root")
        root = Path(
            os.path.commonpath(
                [
                    str(zero_argument.absolute().parent),
                    str(nan_argument.absolute().parent),
                ]
            )
        ).resolve()
    else:
        root = Path(args.evidence_root).expanduser().resolve()
    require(root.is_dir(), f"evidence root is not a directory: {root}")
    require(not root.is_symlink(), "evidence root must not be a symlink")

    script_path = Path(__file__).resolve()
    evidence_context = prepare_evidence(
        script_path=script_path,
        retained=args.retained,
        expected_commit=args.expected_commit,
        model_arguments={},
        runtime_imports={},
    )
    comparator_cache_roots = tuple(
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
            cache_roots=comparator_cache_roots,
        )[0]
        require(is_within(output_path, root), "comparison output must be inside evidence root")
    elif args.retained:
        raise AssertionError("--retained requires --output")

    zero, zero_tensors, zero_descriptor = load_artifact(
        zero_argument, "zero", root
    )
    nan, nan_tensors, nan_descriptor = load_artifact(
        nan_argument, "nan", root
    )
    require(zero_descriptor["path"] != nan_descriptor["path"], "zero and NaN JSON paths must differ")
    require(
        zero_descriptor["tensor_sidecar"]["path"]
        != nan_descriptor["tensor_sidecar"]["path"],
        "zero and NaN tensor paths must differ",
    )
    require(zero["mode"] == nan["mode"], "zero/NaN execution mode drifted")
    require(host_oracle(zero) == host_oracle(nan), "zero/NaN host oracle differs")
    if args.retained:
        expected_commit = args.expected_commit.lower()
        model_snapshot_cache = {}
        zero_provenance = validate_retained_provenance(
            zero["provenance"],
            repo_root=evidence_context.repo_root,
            current_source=evidence_context.source_before,
            expected_commit=expected_commit,
            expected_runner_path="tests/run_speculative_v3_cache_neutrality.py",
            expected_model_roles=("target", "draft"),
            expected_runtime_imports={
                "nanovllm": "nanovllm/__init__.py",
                "LLM": "nanovllm/llm.py",
                "LLMEngine": "nanovllm/engine/llm_engine.py",
                "ModelRunner": "nanovllm/engine/model_runner.py",
                "Scheduler": "nanovllm/engine/scheduler.py",
                "Sampler": "nanovllm/layers/sampler.py",
                "SamplingParams": "nanovllm/sampling_params.py",
                "DraftRouteAdmission": "nanovllm/engine/speculative_routes.py",
            },
            model_snapshot_cache=model_snapshot_cache,
        )
        nan_provenance = validate_retained_provenance(
            nan["provenance"],
            repo_root=evidence_context.repo_root,
            current_source=evidence_context.source_before,
            expected_commit=expected_commit,
            expected_runner_path="tests/run_speculative_v3_cache_neutrality.py",
            expected_model_roles=("target", "draft"),
            expected_runtime_imports={
                "nanovllm": "nanovllm/__init__.py",
                "LLM": "nanovllm/llm.py",
                "LLMEngine": "nanovllm/engine/llm_engine.py",
                "ModelRunner": "nanovllm/engine/model_runner.py",
                "Scheduler": "nanovllm/engine/scheduler.py",
                "Sampler": "nanovllm/layers/sampler.py",
                "SamplingParams": "nanovllm/sampling_params.py",
                "DraftRouteAdmission": "nanovllm/engine/speculative_routes.py",
            },
            model_snapshot_cache=model_snapshot_cache,
        )
        require_independent_compiler_cache_roots(
            (zero_provenance, nan_provenance)
        )
        pair_provenance = normalized_validated_provenance(zero_provenance)
        require(
            pair_provenance
            == normalized_validated_provenance(nan_provenance),
            "zero/NaN validated producer provenance differs",
        )
        for artifact, identity in (
            (zero, zero_provenance),
            (nan, nan_provenance),
        ):
            require(
                Path(artifact["model"]).resolve()
                == Path(identity["models"]["target"]["resolved_path"]),
                "raw target model path disagrees with hashed provenance",
            )
            require(
                Path(artifact["draft_model"]).resolve()
                == Path(identity["models"]["draft"]["resolved_path"]),
                "raw draft model path disagrees with hashed provenance",
            )
        producer_provenance = {
            "pair_identity": pair_provenance,
            "zero": zero_provenance,
            "nan": nan_provenance,
        }
        if output_path is not None:
            validate_output_paths(
                (output_path,),
                repo_root=evidence_context.repo_root,
                model_roots=tuple(
                    Path(identity["resolved_path"])
                    for identity in zero_provenance["models"].values()
                ),
                cache_roots=comparator_cache_roots,
            )
    else:
        require(
            stable_provenance_identity(zero)
            == stable_provenance_identity(nan),
            "zero/NaN stable provenance differs",
        )
        producer_provenance = stable_provenance_identity(zero)
    comparisons = compare_tensor_records(zero_tensors, nan_tensors)
    comparator_provenance = finalize_evidence(evidence_context)
    retention_eligible = bool(
        zero["retention_eligible"]
        and nan["retention_eligible"]
        and comparator_provenance["retention_eligible"]
    )
    if args.retained:
        require(retention_eligible, "paired artifacts are not retention eligible")

    output = {
        "schema": OUTPUT_SCHEMA,
        "verdict": "pass",
        "mode": zero["mode"],
        "retention_eligible": retention_eligible,
        "exact_cache_fill_independence": True,
        "host_oracles_equal": True,
        "record_count": len(comparisons),
        "record_ids": list(RECORD_IDS),
        "zero_artifact": zero_descriptor,
        "nan_artifact": nan_descriptor,
        "tensor_comparisons": comparisons,
        "producer_provenance_identity": producer_provenance,
        "comparator_provenance": comparator_provenance,
    }
    if output_path is not None:
        write_json_exclusive(output_path, output)
        print(
            f"PASS: {zero['mode']} zero/NaN cache evidence is bitwise exact; "
            f"records={len(comparisons)}, retention_eligible={retention_eligible}, "
            f"output={output_path}",
            flush=True,
        )
    else:
        print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
