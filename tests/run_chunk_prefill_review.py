"""Fresh-process A100 correctness checks; not a latency benchmark.

Run once normally and once with --eager. Optional --compare checks the exact
greedy outputs against the other mode; a mismatch needs logit investigation.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import torch

from benchmarks.chunked_prefill_tail.common import (
    immutable_write_json,
    model_identity,
    release_pin,
    sha256_file,
)
from nanovllm import LLM, SamplingParams


def check(model_path: str, *, budget: int, eager: bool):
    if sys.flags.optimize:
        raise RuntimeError("correctness validation requires assertions enabled")
    before = release_pin()
    worker_before = sha256_file(Path(__file__))
    model = model_identity(Path(model_path))
    torch.manual_seed(20260908)
    llm = LLM(
        model_path, max_num_batched_tokens=budget, max_num_seqs=17,
        max_model_len=4096, num_kvcache_blocks=17,
        gpu_memory_utilization=0.5, enforce_eager=eager,
    )
    runner = llm.model_runner
    scheduler = llm.scheduler
    routes = []
    logit_trace = []
    active_rows = []
    case_name = None
    request_indices = {}
    original_run_model = runner.run_model
    original_run = runner.run

    def record_run(seqs, is_prefill):
        active_rows[:] = seqs
        return original_run(seqs, is_prefill)

    runner.run = record_run

    def record_run_model(input_ids, positions, is_prefill):
        if not is_prefill:
            batch = input_ids.size(0)
            key = None if eager else next(
                (size for size in runner.graph_bs if size >= batch and size in runner.graphs),
                None,
            )
            routes.append({"batch": batch, "decode_graph": key})
        logits = original_run_model(input_ids, positions, is_prefill)
        values, tokens = logits.topk(8, dim=-1)
        chosen = logits.argmax(dim=-1).tolist()
        for seq, ids, scores, token in zip(active_rows, tokens.tolist(), values.tolist(), chosen):
            logit_trace.append({
                "case": case_name, "request": request_indices[seq.seq_id],
                "prefix_sha256": hashlib.sha256(json.dumps(seq.token_ids).encode()).hexdigest(),
                "cached_tokens": seq.num_cached_tokens,
                "scheduled_tokens": seq.num_scheduled_tokens,
                "emits": seq.num_cached_tokens + seq.num_scheduled_tokens == len(seq),
                "top_ids": ids, "top_logits": scores, "chosen_token": token,
            })
        return logits

    runner.run_model = record_run_model

    def drain(name, prompts, limits):
        nonlocal case_name, request_indices
        case_name = name
        assert scheduler.is_finished()
        ids = [llm.add_request(prompt, SamplingParams(
            temperature=0.0, max_tokens=count, ignore_eos=True,
        )) for prompt, count in zip(prompts, limits)]
        request_indices = {seq_id: index for index, seq_id in enumerate(ids)}
        outputs = {seq_id: [] for seq_id in ids}
        emissions = {seq_id: [] for seq_id in ids}
        step = 0
        while not scheduler.is_finished():
            assert step < 200
            result = llm._step()
            for event in result.events:
                outputs[event.seq_id].append(event.token_id)
                emissions[event.seq_id].append(step)
            owners = Counter(
                block for seq in (*scheduler.running, *scheduler.waiting)
                for block in seq.block_table
            )
            manager = scheduler.block_manager
            assert all(block.ref_count == owners[block.block_id] for block in manager.blocks)
            assert set(manager.free_block_ids).isdisjoint(manager.used_block_ids)
            assert set(manager.free_block_ids) | manager.used_block_ids == set(range(17))
            scheduler._check_mid_chunk_invariant()
            step += 1
        assert [len(outputs[seq_id]) for seq_id in ids] == limits
        assert len(scheduler.block_manager.free_block_ids) == 17
        return {
            "token_ids": [outputs[seq_id] for seq_id in ids],
            "emission_steps": [emissions[seq_id] for seq_id in ids],
            "steps": step,
        }

    try:
        graph17 = drain("graph17", [[1000 + row] * 4 for row in range(17)], [4] * 17)
        graph17_routes = list(routes)
        assert graph17_routes and all(route["batch"] == 17 for route in graph17_routes)
        assert all(route["decode_graph"] == (None if eager else 17) for route in graph17_routes)
        if not eager:
            assert 17 in runner.graphs

        victims = []
        original_preempt = scheduler.preempt

        def record_preempt(seq):
            victims.append(seq.num_prompt_tokens)
            original_preempt(seq)

        scheduler.preempt = record_preempt
        pressure = drain("pressure", [list(range(1000, 1250)), list(range(20000, 24096))], [16, 1])
        short_steps = pressure["emission_steps"][0]
        assert all(y - x == 1 for x, y in zip(short_steps, short_steps[1:]))
        assert victims == [4096]
        pressure["preempted_prompt_lengths"] = victims
        pressure["short_max_gap_steps"] = max(y - x for x, y in zip(short_steps, short_steps[1:]))

        # Exercise a real eager fallback after successful initialization/capture,
        # not only a host double. Restore capture ownership before engine exit.
        fallback = None
        if not eager:
            saved_graphs = runner.graphs
            runner.graphs = {}
            start = len(routes)
            try:
                fallback = drain("missing_graph_fallback", [[7000] * 4], [4])
                assert routes[start:] and all(route["decode_graph"] is None for route in routes[start:])
            finally:
                runner.graphs = saved_graphs
        after = release_pin()
        assert before["source"] == after["source"], "runtime changed during GPU validation"
        assert worker_before == sha256_file(Path(__file__)), "worker changed during GPU validation"
        return {
            "kind": "chunk_prefill_review_gpu_correctness",
            "latency_certified": False,
            "provenance_before": before,
            "provenance_after": after,
            "worker_sha256": worker_before,
            "model": model,
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "budget": budget,
            "enforce_eager": eager,
            "graph17": graph17,
            "graph17_decode_routes": graph17_routes,
            "pressure": pressure,
            "missing_graph_fallback": fallback,
            "logit_trace": logit_trace,
            "passed": True,
        }
    finally:
        llm.exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/workspace/models/Qwen3-0.6B")
    parser.add_argument("--budget", type=int, choices=(64, 128, 256), default=256)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = check(args.model, budget=args.budget, eager=args.eager)
    if args.compare is not None:
        other = json.loads(args.compare.read_text())
        assert result["budget"] == other["budget"]
        assert result["model"] == other["model"]
        assert result["provenance_before"]["source"] == other["provenance_before"]["source"]
        assert result["enforce_eager"] != other["enforce_eager"]
        result["graph_eager_tokens_equal_by_case"] = {
            case: result[case]["token_ids"] == other[case]["token_ids"]
            for case in ("graph17", "pressure")
        }
        result["graph_eager_tokens_equal"] = all(result["graph_eager_tokens_equal_by_case"].values())
        result["passed"] = result["graph_eager_tokens_equal"]
    immutable_write_json(args.output, result)
    print(json.dumps({
        "output": str(args.output), "budget": args.budget,
        "enforce_eager": args.eager, "passed": result["passed"],
        "graph_eager_tokens_equal": result.get("graph_eager_tokens_equal"),
        "pressure_short_max_gap_steps": result["pressure"]["short_max_gap_steps"],
    }, sort_keys=True))
    if not result["passed"]:
        raise SystemExit("greedy comparison differs; retained both outputs for logit investigation")
