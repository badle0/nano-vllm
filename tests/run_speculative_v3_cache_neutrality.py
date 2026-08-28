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

from nanovllm import LLM, SamplingParams
from nanovllm.engine.speculative_routes import DraftRouteAdmission


SCHEMA = "nano-vllm-speculative-v3-cache-neutrality-v1"


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
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def tensor_hash(tensor):
    value = tensor.detach().contiguous().cpu()
    # NumPy cannot expose torch.bfloat16 directly.  Hash the exact underlying
    # bytes instead of converting and weakening the bitwise oracle.
    payload = value.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
        logits = args[0] if args else kwargs["logits"]
        sample = original(*args, **kwargs)
        probabilities = sample.probabilities
        tensor_records.append(
            {
                "logits": logits.detach().to(torch.float32).contiguous().cpu(),
                "probabilities": probabilities.detach().contiguous().cpu(),
            }
        )
        records.append(
            {
                "logits_sha256": tensor_hash(logits),
                "probabilities_sha256": tensor_hash(probabilities),
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
    torch.manual_seed(20260828)
    torch.cuda.manual_seed_all(20260828)
    engine = LLM(
        args.model,
        draft_model=args.draft_model or args.model,
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
        boundary = run_boundary_cell(engine, args.draft_cache_fill, sampler_records)
        shared_prefix = run_shared_prefix_cell(
            engine, args.draft_cache_fill, sampler_records
        )
        output = {
            "schema": SCHEMA,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode,
            "draft_cache_fill": args.draft_cache_fill,
            "model": str(Path(args.model).resolve()),
            "draft_model": str(
                Path(args.draft_model or args.model).resolve()
            ),
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

    tensor_path = Path(args.output).with_name(
        f"{Path(args.output).stem}.tensors.pt"
    )
    torch.save(
        {
            "schema": SCHEMA,
            "mode": args.mode,
            "draft_cache_fill": args.draft_cache_fill,
            "records": tensor_records,
        },
        tensor_path,
    )
    output["tensor_artifact"] = str(tensor_path.resolve())
    output["tensor_artifact_sha256"] = file_hash(tensor_path)
    payload = json.dumps(output, indent=2, sort_keys=True) + "\n"
    Path(args.output).write_text(payload, encoding="utf-8")
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "mode": args.mode,
                "draft_cache_fill": args.draft_cache_fill,
                "boundary_positions": boundary["proposal_input_positions"],
                "boundary_proposals": boundary["proposed_token_ids"],
                "shared_prefix_equal": True,
                "output": str(Path(args.output).resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
