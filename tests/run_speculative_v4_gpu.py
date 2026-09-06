"""Retained V4 shadow GPU controls. No verifier/acceptance implementation.

Run six fresh processes: eager/graph x off/zero/nan, each with empty independent
compiler caches. Enabled runs exercise the real V4 engine entry point, not the
legacy V3 planner. The K override only truncates ready route admission for the
route-coverage workload, and is explicitly recorded in each interval.
"""
import argparse
from dataclasses import asdict, replace
from enum import Enum
import hashlib
import json
from pathlib import Path
import sys
import uuid

import torch
import nanovllm
from nanovllm import LLM, SamplingParams
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.speculative_plan import SpecStepPlan
from nanovllm.layers.sampler import Sampler
from _speculative_v4_evidence import prepare, finalize
from _speculative_v3_evidence import validate_output_paths, write_json_exclusive
from run_speculative_v3_route_compile import (
    cache_root_path, initialize_cache_root, validate_cache_root_isolation,
    require_compiler_environment, install_capture_ledger, compiler_snapshot,
    require_nonvacuous_compiler_snapshot, default_context, rng_hashes,
    sampling_params, require,
)

SCHEMA = "nano-vllm-speculative-v4-gpu-v1"
SEED = 20260906
CONFIG = dict(max_num_seqs=4, max_num_batched_tokens=1024, max_model_len=512,
              gpu_memory_utilization=0.5)
IMPORTS = {
    "nanovllm": (nanovllm, "nanovllm/__init__.py"),
    "LLM": (LLM, "nanovllm/llm.py"),
    "LLMEngine": (LLMEngine, "nanovllm/engine/llm_engine.py"),
    "ModelRunner": (ModelRunner, "nanovllm/engine/model_runner.py"),
    "Scheduler": (Scheduler, "nanovllm/engine/scheduler.py"),
    "SpecStepPlan": (SpecStepPlan, "nanovllm/engine/speculative_plan.py"),
    "Sampler": (Sampler, "nanovllm/layers/sampler.py"),
}


def primitive(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): primitive(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [primitive(item) for item in value]
    return value


def digest(value):
    return hashlib.sha256(json.dumps(primitive(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def tensor_digest(value):
    raw = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def allocator_state(scheduler):
    manager = scheduler.block_manager
    return {
        "free": list(manager.free_block_ids), "used": sorted(manager.used_block_ids),
        "hashes": sorted(manager.hash_to_block_id.items()),
        "blocks": [(b.ref_count, b.hash, b.token_ids) for b in manager.blocks],
        "rows": [(s.seq_id, s.token_ids[:], s.num_cached_tokens,
                  s.num_draft_cached_tokens, s.num_scheduled_tokens, s.block_table[:])
                 for s in (*scheduler.running, *scheduler.waiting)],
    }


class Observer:
    def __init__(self, engine, fill, cache_roots, captures, run_id):
        self.engine, self.fill = engine, fill
        self.cache_roots, self.captures, self.run_id = cache_roots, captures, run_id
        self.label, self.cap, self.inject_target = "", None, False
        self.records, self.plans, self.samples = [], [], []
        scheduler, runner = engine.scheduler, engine.model_runner
        original_plan = scheduler.plan_speculative_step
        original_draft = runner.run_speculative_discard
        original_sample = runner.sampler.sample_exact_with_probabilities
        original_call = runner.call

        def plan(seqs, **kwargs):
            if self.cap is not None and kwargs["route_admission"] is not None:
                admission = kwargs["route_admission"]
                kwargs["route_admission"] = replace(admission, route_keys=admission.route_keys[:self.cap])
            before_tables = [seq.block_table[:] for seq in seqs]
            before_free = list(scheduler.block_manager.free_block_ids)
            result = original_plan(seqs, **kwargs)
            require(isinstance(result, SpecStepPlan), "engine did not use the V4 planner")
            self.plans.append({"label": self.label, "plan": primitive(asdict(result)),
                               "tables_before": before_tables, "free_before": before_free,
                               "tables_reserved": [seq.block_table[:] for seq in seqs],
                               "free_reserved": list(scheduler.block_manager.free_block_ids)})
            return result

        def sample(logits, *args, **kwargs):
            result = original_sample(logits, *args, **kwargs)
            q = result.probabilities
            require(bool(torch.isfinite(logits).all()), "nonfinite draft logits")
            require(bool(torch.isfinite(q).all()), "nonfinite draft probabilities")
            require(bool((q >= 0).all()), "negative draft probabilities")
            sums = q.sum(-1)
            require(bool(torch.allclose(sums, torch.ones_like(sums), atol=2e-6, rtol=0)), "q not normalized")
            self.samples.append({"logits_sha256": tensor_digest(logits), "q_sha256": tensor_digest(q),
                                 "shape": list(q.shape), "sums": sums.cpu().tolist(),
                                 "token_ids": result.token_ids.cpu().tolist()})
            return result

        def draft(plan, seqs):
            require(isinstance(plan, SpecStepPlan) and plan.uses_speculation, "not a positive V4 plan")
            # Only initialize completely cold draft rows. Warm coverage remains
            # valid and must not be poisoned. Target cache is never filled here.
            filled = sorted({bid for row in plan.rows if row.draft_cached_tokens == 0
                             for bid in row.block_table})
            if filled:
                indices = torch.tensor(filled, dtype=torch.int64, device=runner.draft_kv_cache.device)
                runner.draft_kv_cache.index_fill_(2, indices, 0.0 if fill == "zero" else float("nan"))
            index = len(self.records)
            interval = f"{self.run_id}:{index}"
            start = len(self.samples)
            before = compiler_snapshot(cache_roots, captures)
            rng_before = rng_hashes()
            torch.compiler.set_stance("fail_on_recompile")
            print(f"V4_DRAFT_INTERVAL_BEGIN {interval}", file=sys.stderr, flush=True)
            try:
                result = original_draft(plan, seqs)
            finally:
                print(f"V4_DRAFT_INTERVAL_END {interval}", file=sys.stderr, flush=True)
                torch.compiler.set_stance("default")
            after = compiler_snapshot(cache_roots, captures)
            rng_after = rng_hashes()
            require(before == after, "draft interval compiled, broke a graph or captured a new graph")
            require(rng_before == rng_after and default_context(), "draft RNG/context neutrality failed")
            require(len(self.samples) - start == plan.effective_k, "sampler calls do not cover K")
            self.records.append({
                "label": self.label, "interval": interval, "forced_route_cap": self.cap,
                "plan": primitive(asdict(plan)), "filled_blocks": filled,
                "compiler_before": before, "compiler_after": after,
                "rng_before": rng_before, "rng_after": rng_after, "context_default": True,
                "samples": self.samples[start:],
                "proposals": [list(row.proposed_token_ids) for row in result.rows],
                "graph_steps": result.graph_decode_steps, "eager_steps": result.eager_decode_steps,
            })
            return result

        def call(method, *args):
            if method == "run" and self.inject_target:
                require(scheduler._active_spec_transaction is not None, "target failure lacks V4 transaction")
                require(scheduler._active_draft_discard is None, "extra lease not handed off")
                self.inject_target = False
                raise RuntimeError("injected V4 pre-target failure")
            return original_call(method, *args)

        scheduler.plan_speculative_step = plan
        runner.run_speculative_discard = draft
        runner.sampler.sample_exact_with_probabilities = sample
        runner.call = call

    def step(self, label, cap=None):
        self.label, self.cap = label, cap
        output = self.engine._step()
        scheduler = self.engine.scheduler
        require(scheduler._active_spec_transaction is None, "transaction survived successful target commit")
        require(scheduler._active_draft_discard is None, "temporary lease survived commit")
        return output


def clean_capacity(engine):
    scheduler = engine.scheduler
    require(not scheduler.running and not scheduler.waiting, "queues not drained")
    require(not scheduler.block_manager.used_block_ids, "KV capacity leaked")
    require(len(scheduler.block_manager.free_block_ids) == len(scheduler.block_manager.blocks), "free capacity leaked")
    require(not scheduler.block_manager._active_temporary_reservations, "allocator lease survived cleanup")


def controls(engine, observer):
    prompts = [[10, 11, 12, 13], [20, 21, 22]]
    params = [SamplingParams(temperature=0.8, top_k=8, top_p=0.9,
                             max_tokens=6, ignore_eos=True) for _ in prompts]
    ids = [engine.add_request(prompt, param) for prompt, param in zip(prompts, params)]
    snapshots = [rng_hashes()]
    steps = []
    for index in range(6):
        output = observer.step(f"control/{index}") if observer else engine._step()
        steps.append([list(event) for event in output.events])
        snapshots.append(rng_hashes())
    clean_capacity(engine)
    return {"ids": ids, "prompts": prompts, "events": steps, "rng": snapshots}


def routes(engine, observer):
    registry = engine.model_runner.draft_route_registry
    require(registry.ready_keys == registry.router_admitted_keys, "incomplete warmed registry")
    for batch in range(1, 5):
        for k in (1, 2):
            for rep in range(2):
                ids = [engine.add_request([10, 11, 12, 13], p)
                       for p in sampling_params(batch, max_tokens=6)]
                engine._step()  # baseline prefill
                for phase in ("cold", "warm"):
                    label = f"route/{batch}/{k}/{rep}/{phase}"
                    before = len(observer.records)
                    observer.step(label, k)
                    require(len(observer.records) == before + 1, "forced route unexpectedly bypassed")
                    plan = observer.records[-1]["plan"]
                    require(plan["effective_k"] == k and len(plan["rows"]) == batch, "forced geometry drifted")
                    require((plan["draft_catchup_tokens"] > 0) == (phase == "cold"), "catchup phase drifted")
                engine._cancel_requests(ids)
                clean_capacity(engine)
    visited = {digest(record["plan"]["route_key"]) for record in observer.records if record["label"].startswith("route/")}
    expected = {digest(asdict(key)) for key in registry.ready_keys}
    require(visited == expected, "did not exercise every ready route")
    return [primitive(asdict(key)) for key in registry.ready_keys]


def boundary_and_rollback(engine, observer):
    records = []
    for length in (254, 255, 256, 257):
        prompt = [30 + (index * 7 + length) % 900 for index in range(length)]
        ids = [engine.add_request(prompt, SamplingParams(temperature=0.0, max_tokens=6, ignore_eos=True))]
        engine._step()
        before = allocator_state(engine.scheduler)
        observer.label, observer.cap, observer.inject_target = f"failure/{length}", None, True
        try:
            engine._step()
        except RuntimeError as error:
            require(str(error) == "injected V4 pre-target failure", f"unexpected error: {error}")
        else:
            raise AssertionError("target failure was not injected")
        after = allocator_state(engine.scheduler)
        require(before == after, "V4 failure did not restore exact logical/physical state")
        require(engine.scheduler._active_spec_transaction is None, "failed transaction not released")
        retry = observer.step(f"boundary/{length}")
        last_plan = observer.plans[-1]
        reserved = sum(len(row) for row in last_plan["tables_reserved"])
        original = sum(len(row) for row in last_plan["tables_before"])
        records.append({"length": length, "before_sha256": digest(before), "after_sha256": digest(after),
                        "extra_blocks": reserved - original,
                        "events": [list(event) for event in retry.events],
                        "tables_after": [seq.block_table[:] for seq in engine.scheduler.running]})
        require([seq.block_table for seq in engine.scheduler.running] == last_plan["tables_before"],
                "target commit retained speculative-only suffix")
        engine._cancel_requests(ids)
        clean_capacity(engine)
    return records


def prefix(engine, observer):
    prompt = [71 + (index * 13) % 800 for index in range(257)]
    outputs = []
    for phase in ("cold", "hit"):
        ids = [engine.add_request(prompt, SamplingParams(temperature=0.0, max_tokens=6, ignore_eos=True))]
        prefill = engine._step()
        require(prefill.num_prefill_tokens == (257 if phase == "cold" else 1), "prefix hit/cold oracle drifted")
        output = observer.step(f"prefix/{phase}")
        outputs.append({"prefill_tokens": prefill.num_prefill_tokens,
                        "target_tokens": [event.token_id for event in output.events],
                        "samples": observer.records[-1]["samples"]})
        engine._cancel_requests(ids)
        clean_capacity(engine)
    require(outputs[0]["samples"] == outputs[1]["samples"], "cache-hit draft numerics differ")
    require(outputs[0]["target_tokens"] == outputs[1]["target_tokens"], "cache-hit target tokens differ")
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("eager", "graph"), required=True)
    parser.add_argument("--side", choices=("off", "zero", "nan"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    require_compiler_environment()
    roots = tuple(cache_root_path(name) for name in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"))
    repo, before = prepare(args.model, args.expected_commit, IMPORTS)
    model = Path(args.model).resolve()
    validate_cache_root_isolation(*roots, repo_root=repo, model_roots=(model,))
    for name, root in zip(("inductor", "triton"), roots):
        initialize_cache_root(name, root)
    output = validate_output_paths((Path(args.output),), repo_root=repo, model_roots=(model,), cache_roots=roots)[0]
    captures = install_capture_ledger()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.compiler.set_stance("default")
    kwargs = dict(CONFIG, enforce_eager=args.mode == "eager")
    if args.side != "off":
        kwargs.update(draft_model=str(model), num_speculative_tokens=2)
    engine = LLM(str(model), **kwargs)
    try:
        cache_roots = tuple(zip(("inductor", "triton"), roots))
        initial_compiler = compiler_snapshot(cache_roots, captures)
        require_nonvacuous_compiler_snapshot(initial_compiler)
        initial_captures = dict(captures)
        observer = None if args.side == "off" else Observer(engine, args.side, cache_roots, captures, uuid.uuid4().hex)
        output_data = {"schema": SCHEMA, "mode": args.mode, "side": args.side,
                       "configuration": {**CONFIG, "configured_k": 2, "seed": SEED,
                                         "top_p_backend": "exact", "tensor_parallel_size": 1},
                       "control": controls(engine, observer)}
        if observer:
            output_data.update(registry=routes(engine, observer),
                               boundaries=boundary_and_rollback(engine, observer),
                               prefix=prefix(engine, observer),
                               records=observer.records, plans=observer.plans)
        require(captures == initial_captures, "post-init graph capture occurred")
        output_data.update(compiler_after_init=initial_compiler,
                           captures_after_init=initial_captures, captures_after_runtime=dict(captures))
    finally:
        engine.exit()
    output_data["provenance"] = finalize(repo, args.model, IMPORTS, before)
    write_json_exclusive(output, output_data)
    print(f"V4 retained producer completed: {args.mode}/{args.side}, intervals={len(output_data.get('records', []))}", flush=True)


if __name__ == "__main__":
    main()
