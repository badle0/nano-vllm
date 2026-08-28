"""Fresh-process A100 gate for V3 draft-KV cache neutrality.

Run paired processes with ``--draft-cache-fill zero`` and ``nan``.  The harness
fills every physical draft block named by the already-reserved immutable plan,
then executes the production compute-then-discard path.  Equality of full draft
logit/probability hashes proves cold catch-up overwrites every logically readable
slot instead of consuming constructor-pretouch residue or recycled KV bytes.

This is an executable certification harness, not a pytest module.  Dirty-tree
runs are exploratory and must not be retained as release evidence.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

import nanovllm

from nanovllm import LLM, SamplingParams
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.speculative_routes import DraftRouteAdmission
from nanovllm.layers.sampler import Sampler

from _speculative_v3_evidence import (
    compiler_cache_roots_from_environment,
    finalize_evidence,
    prepare_evidence,
    require_retained_a100_sxm4_40gb,
    sha256_file,
    validate_compiler_cache_root_isolation,
    validate_output_paths,
    write_json_exclusive,
    write_torch_exclusive,
)


SCHEMA = "nano-vllm-speculative-v3-cache-neutrality-v2"
SEED = 20260828
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


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model")
    parser.add_argument("--mode", choices=("eager", "graph"), required=True)
    parser.add_argument(
        "--draft-cache-fill", choices=("zero", "nan"), required=True
    )
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


def requested_output_paths(argument):
    """Derive JSON/tensor siblings without resolving attacker-controlled paths."""

    requested_output = Path(argument).expanduser()
    requested_tensor = requested_output.with_name(
        f"{requested_output.stem}.tensors.pt"
    )
    return requested_output, requested_tensor


def tensor_hash(tensor):
    value = tensor.detach().contiguous().cpu()
    # NumPy cannot expose torch.bfloat16 directly.  Hash the exact underlying
    # bytes instead of converting and weakening the bitwise oracle.  Domain
    # separate dtype and shape so equal byte strings cannot alias tensors with
    # different interpretations.
    metadata = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload = metadata + b"\0" + value.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def rng_hashes():
    cpu = torch.random.get_rng_state().cpu().numpy().tobytes()
    cuda = torch.cuda.get_rng_state().cpu().numpy().tobytes()
    return {
        "cpu": hashlib.sha256(cpu).hexdigest(),
        "cuda": hashlib.sha256(cuda).hexdigest(),
    }


def prompt_tokens(length, salt):
    require(length >= 1, "prompt length must be positive")
    # Stay far below the Qwen vocabulary while avoiding all-identical fixtures.
    return [10 + ((index * 17 + salt) % 997) for index in range(length)]


def route_dict(key):
    return {
        "schema": key.schema,
        "execution_mode": str(key.execution_mode),
        "batch_bucket": key.batch_bucket,
        "effective_k": key.effective_k,
        "catchup_family": str(key.catchup_family),
        "sampler_envelope": str(key.sampler_envelope),
    }


def install_sampler_recorder(runner):
    sampler = runner.sampler
    original = sampler.sample_exact_with_probabilities
    records = []
    tensor_records = []

    def recorded(*args, **kwargs):
        record_index = len(tensor_records)
        require(record_index < len(RECORD_IDS), "unexpected extra sampler call")
        record_id = RECORD_IDS[record_index]
        logits = args[0] if args else kwargs["logits"]
        sample = original(*args, **kwargs)
        probabilities = sample.probabilities
        logits_cpu = logits.detach().contiguous().cpu()
        probabilities_cpu = probabilities.detach().contiguous().cpu()
        tensor_records.append(
            {
                "record_id": record_id,
                "logits": logits_cpu,
                "probabilities": probabilities_cpu,
            }
        )
        records.append(
            {
                "record_id": record_id,
                "logits_sha256": tensor_hash(logits),
                "probabilities_sha256": tensor_hash(probabilities),
                "logits_dtype": str(logits.dtype),
                "probabilities_dtype": str(probabilities.dtype),
                "logits_shape": list(logits.shape),
                "probabilities_shape": list(probabilities.shape),
                "logits_finite": bool(torch.isfinite(logits).all().item()),
                "probabilities_finite": bool(
                    torch.isfinite(probabilities).all().item()
                ),
                "probability_row_sums": [
                    float(value)
                    for value in probabilities.sum(dim=-1).cpu().tolist()
                ],
                "token_ids": sample.token_ids.cpu().tolist(),
            }
        )
        return sample

    sampler.sample_exact_with_probabilities = recorded
    return records, tensor_records, original


def fill_plan_blocks(engine, plan, fill_mode):
    block_ids = sorted(
        {
            block_id
            for row in plan.rows
            for block_id in row.block_table
        }
    )
    require(block_ids, "draft plan names no physical blocks")
    indices = torch.tensor(
        block_ids,
        dtype=torch.int64,
        device=engine.model_runner.draft_kv_cache.device,
    )
    value = 0.0 if fill_mode == "zero" else float("nan")
    with torch.inference_mode():
        engine.model_runner.draft_kv_cache.index_fill_(2, indices, value)
    torch.cuda.synchronize()
    return block_ids


def execute_forced_cycle(engine, *, effective_k, fill_mode, sampler_records):
    scheduler = engine.scheduler
    seqs, is_prefill = scheduler.schedule()
    require(seqs and not is_prefill, "expected a pure-decode schedule")
    tables_after_target_schedule = [tuple(seq.block_table) for seq in seqs]
    free_before_plan = len(scheduler.block_manager.free_block_ids)
    admission = engine.model_runner.resolve_draft_route_admission(seqs)
    require(admission is not None, "V3 route unexpectedly missed")
    require(
        admission.max_effective_k >= effective_k,
        "requested K exceeds ready admission",
    )
    forced = DraftRouteAdmission(
        registry_schema=admission.registry_schema,
        plan_fingerprint=admission.plan_fingerprint,
        batch_size=admission.batch_size,
        catchup_tokens=admission.catchup_tokens,
        route_keys=admission.route_keys[:effective_k],
    )
    plan = scheduler.plan_draft_discard(seqs, route_admission=forced)
    require(plan.uses_draft, f"forced V3 route fell back: {plan.fallback_reason}")
    tables_during_reservation = [tuple(seq.block_table) for seq in seqs]
    free_during_reservation = len(scheduler.block_manager.free_block_ids)
    filled_blocks = fill_plan_blocks(engine, plan, fill_mode)
    sampler_start = len(sampler_records)
    rng_before = rng_hashes()

    try:
        result = engine.model_runner.run_speculative_discard(plan, seqs)
        rng_after_draft = rng_hashes()
        require(rng_after_draft == rng_before, "draft interval changed target RNG")
        proposal_positions = [
            row.committed_tokens - 1 + step
            for row in plan.rows
            for step in range(plan.effective_k)
        ]
        coverage = engine._validate_draft_discard_result(
            plan,
            result,
            seqs,
            engine._admission_limits.vocab_size,
        )
        require(scheduler.handoff_draft_discard(plan), "draft handoff failed")
        tables_after_handoff = [tuple(seq.block_table) for seq in seqs]
        free_after_handoff = len(scheduler.block_manager.free_block_ids)
        scheduler.stage_draft_coverage(plan, seqs, coverage)
        target_tokens = engine.model_runner.run(seqs, False)
        events = scheduler.postprocess(seqs, target_tokens)
    except BaseException as error:
        engine._rollback_failed_draft_state(plan, error)
        raise

    calls = sampler_records[sampler_start:]
    require(len(calls) == effective_k, "sampler call count does not match K")
    require(
        all(call["logits_finite"] and call["probabilities_finite"] for call in calls),
        "draft cache fill leaked a non-finite value into proposal sampling",
    )
    require(
        tables_after_handoff == tables_after_target_schedule,
        "handoff retained speculative-only block ownership",
    )
    require(
        free_after_handoff == free_before_plan,
        "handoff did not restore temporary block capacity",
    )
    return {
        "route": route_dict(plan.route_key),
        "effective_k": plan.effective_k,
        "catchup_tokens": plan.draft_catchup_tokens,
        "proposal_input_positions": proposal_positions,
        "filled_physical_blocks": filled_blocks,
        "tables_after_target_schedule": tables_after_target_schedule,
        "tables_during_reservation": tables_during_reservation,
        "tables_after_handoff": tables_after_handoff,
        "temporary_blocks": free_before_plan - free_during_reservation,
        "proposed_token_ids": [
            list(row.proposed_token_ids) for row in result.rows
        ],
        "target_token_ids": [event.token_id for event in events],
        "graph_decode_steps": result.graph_decode_steps,
        "eager_decode_steps": result.eager_decode_steps,
        "sampler_records": calls,
        "rng_before_draft": rng_before,
        "rng_after_draft": rng_after_draft,
    }


def run_boundary_cell(engine, fill_mode, sampler_records):
    prompt = prompt_tokens(255, salt=31)
    params = SamplingParams(
        temperature=0.0,
        top_k=-1,
        top_p=1.0,
        max_tokens=5,
        ignore_eos=True,
    )
    seq_id = engine.add_request(prompt, params)
    prefill = engine._step()
    require(prefill.num_prefill_tokens == 255, "boundary prefill was not cold")
    record = execute_forced_cycle(
        engine,
        effective_k=3,
        fill_mode=fill_mode,
        sampler_records=sampler_records,
    )
    require(
        record["proposal_input_positions"] == [255, 256, 257],
        "proposal positions did not cross the 255/256/257 boundary",
    )
    require(record["catchup_tokens"] == 255, "boundary catch-up charge drifted")
    require(record["temporary_blocks"] == 1, "boundary route reserved wrong block count")
    engine._cancel_requests([seq_id])
    return record


def run_neutral_boundary_warmup(engine, sampler_records, tensor_records):
    """Warm the exact 255/256/257 path before measuring fill sensitivity."""

    prompt = prompt_tokens(255, salt=911)
    params = SamplingParams(
        temperature=0.0,
        top_k=-1,
        top_p=1.0,
        max_tokens=5,
        ignore_eos=True,
    )
    seq_id = engine.add_request(prompt, params)
    prefill = engine._step()
    require(prefill.num_prefill_tokens == 255, "neutral warmup prefill was not cold")
    record = execute_forced_cycle(
        engine,
        effective_k=3,
        fill_mode="zero",
        sampler_records=sampler_records,
    )
    engine._cancel_requests([seq_id])
    require(
        record["proposal_input_positions"] == [255, 256, 257],
        "neutral warmup did not cross the target block boundary",
    )
    require(record["catchup_tokens"] == 255, "neutral warmup catch-up drifted")
    require(record["temporary_blocks"] == 1, "neutral warmup reservation drifted")
    require(
        len(sampler_records) == len(tensor_records) == 3,
        "neutral warmup sampler coverage drifted",
    )
    summary = {
        "fill": "zero",
        "prompt_length": 255,
        "prompt_salt": 911,
        "proposal_input_positions": record["proposal_input_positions"],
        "catchup_tokens": record["catchup_tokens"],
        "temporary_blocks": record["temporary_blocks"],
        "proposed_token_ids": record["proposed_token_ids"],
        "target_token_ids": record["target_token_ids"],
        "sampler_records_discarded": len(sampler_records),
    }
    sampler_records.clear()
    tensor_records.clear()
    return summary


def run_shared_prefix_cell(engine, fill_mode, sampler_records):
    prompt = prompt_tokens(257, salt=73)
    params = SamplingParams(
        temperature=0.0,
        top_k=-1,
        top_p=1.0,
        max_tokens=6,
        ignore_eos=True,
    )

    cold_id = engine.add_request(prompt, params)
    cold_prefill = engine._step()
    require(cold_prefill.num_prefill_tokens == 257, "seed prefix was not cold")
    cold = execute_forced_cycle(
        engine,
        effective_k=3,
        fill_mode=fill_mode,
        sampler_records=sampler_records,
    )
    engine._cancel_requests([cold_id])

    hit_id = engine.add_request(prompt, params)
    hit_prefill = engine._step()
    require(
        hit_prefill.num_prefill_tokens == 1,
        "second identical prompt did not reuse one complete prefix block",
    )
    hit = execute_forced_cycle(
        engine,
        effective_k=3,
        fill_mode=fill_mode,
        sampler_records=sampler_records,
    )
    engine._cancel_requests([hit_id])

    require(
        cold["proposed_token_ids"] == hit["proposed_token_ids"],
        "cold and shared-prefix proposals differ",
    )
    require(
        [item["logits_sha256"] for item in cold["sampler_records"]]
        == [item["logits_sha256"] for item in hit["sampler_records"]],
        "cold and shared-prefix draft logits differ",
    )
    require(
        cold["target_token_ids"] == hit["target_token_ids"],
        "cold and shared-prefix authoritative target tokens differ",
    )
    return {"cold": cold, "shared_prefix": hit}


def main(argv=None):
    args = parse_args(argv)
    require(torch.cuda.is_available(), "CUDA is required")
    if args.retained:
        require_retained_a100_sxm4_40gb()
    cache_roots = compiler_cache_roots_from_environment(
        required=args.retained
    )
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
            "DraftRouteAdmission": (
                DraftRouteAdmission,
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
    # Keep the requested path lexical until the hardened validator has checked
    # every existing component.  Resolving here would erase a dangling symlink
    # and could turn the sibling tensor path into an attacker-selected target.
    requested_output, requested_tensor = requested_output_paths(args.output)
    output_path, tensor_path = validate_output_paths(
        (requested_output, requested_tensor),
        repo_root=evidence_context.repo_root,
        model_roots=evidence_context.model_paths.values(),
        cache_roots=cache_roots,
    )
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    engine = LLM(
        str(target_model_path),
        draft_model=str(draft_model_path),
        num_speculative_tokens=3,
        enforce_eager=args.mode == "eager",
        gpu_memory_utilization=0.5,
        max_model_len=512,
        max_num_batched_tokens=512,
        max_num_seqs=1,
    )
    sampler_records, tensor_records, original_sampler = install_sampler_recorder(
        engine.model_runner
    )
    try:
        measurement_warmup = run_neutral_boundary_warmup(
            engine,
            sampler_records,
            tensor_records,
        )
        require(
            not sampler_records and not tensor_records,
            "neutral warmup records leaked into measured evidence",
        )
        boundary = run_boundary_cell(engine, args.draft_cache_fill, sampler_records)
        shared_prefix = run_shared_prefix_cell(
            engine, args.draft_cache_fill, sampler_records
        )
        output = {
            "schema": SCHEMA,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode,
            "seed": SEED,
            "draft_cache_fill": args.draft_cache_fill,
            "model": str(target_model_path),
            "draft_model": str(draft_model_path),
            "configuration": {
                "num_speculative_tokens": 3,
                "gpu_memory_utilization": 0.5,
                "max_model_len": 512,
                "max_num_batched_tokens": 512,
                "max_num_seqs": 1,
                "tensor_parallel_size": 1,
                "top_p_backend": "exact",
            },
            "workload": {
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
            "measurement_warmup": measurement_warmup,
            "boundary": boundary,
            "shared_prefix": shared_prefix,
            "all_logits_finite": all(
                record["logits_finite"] for record in sampler_records
            ),
            "all_probabilities_finite": all(
                record["probabilities_finite"] for record in sampler_records
            ),
        }
    finally:
        engine.model_runner.sampler.sample_exact_with_probabilities = (
            original_sampler
        )
        engine.exit()

    require(
        [record["record_id"] for record in tensor_records] == list(RECORD_IDS),
        "cache-neutrality sampler record coverage drifted",
    )
    provenance = finalize_evidence(evidence_context)
    output["provenance"] = provenance
    output["retention_eligible"] = provenance["retention_eligible"]
    write_torch_exclusive(
        tensor_path,
        {
            "schema": SCHEMA,
            "mode": args.mode,
            "draft_cache_fill": args.draft_cache_fill,
            "records": tensor_records,
        },
    )
    output["tensor_artifact"] = {
        "path": tensor_path.name,
        "format": "torch-save-weights-only",
        "size_bytes": tensor_path.stat().st_size,
        "sha256": sha256_file(tensor_path),
        "record_count": len(tensor_records),
    }
    write_json_exclusive(output_path, output)
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "mode": args.mode,
                "draft_cache_fill": args.draft_cache_fill,
                "boundary_positions": boundary["proposal_input_positions"],
                "boundary_proposals": boundary["proposed_token_ids"],
                "shared_prefix_equal": True,
                "retention_eligible": output["retention_eligible"],
                "output": str(output_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
