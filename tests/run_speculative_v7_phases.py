"""Intrusively synchronized phase diagnostics; never headline latency samples."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch
from nanovllm import LLM, SamplingParams
import nanovllm.engine.speculative_execution as execution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--mode", choices=("graph", "eager"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or sys_optimized():
        raise RuntimeError("fresh output path and enabled assertions required")
    torch.set_num_threads(1)
    source = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path("nanovllm").rglob("*.py"))}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    llm = LLM(args.model, draft_model=args.draft_model, num_speculative_tokens=4,
              max_num_seqs=8, max_num_batched_tokens=4096, max_model_len=4096,
              num_kvcache_blocks=64, gpu_memory_utilization=.8,
              enforce_eager=args.mode == "eager")
    active = None
    cells, cycles = [], []
    restores = []

    def install(owner, name, phase):
        original = getattr(owner, name)
        def observed(*positional, **keywords):
            if active is None:
                return original(*positional, **keywords)
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = original(*positional, **keywords)
            torch.cuda.synchronize()
            active.setdefault("phases", {}).setdefault(phase, []).append(time.perf_counter() - start)
            if phase == "run_speculative":
                active["result"] = asdict(result)
                active["batch"] = len(positional[1])
                active["k"] = positional[0].effective_k
            return result
        setattr(owner, name, observed)
        restores.append((owner, name, original))

    install(llm.model_runner, "_execute_draft_proposals_validated", "draft")
    install(execution, "target_probabilities", "verify_parallel")
    install(execution, "sequential_target_probabilities", "verify_greedy")
    install(llm.model_runner.speculative_rejection_sampler, "accept", "accept")
    install(execution, "_sample_exponential_race", "bonus")
    install(llm.model_runner, "run_speculative", "run_speculative")
    install(llm.scheduler, "commit_speculative", "commit")
    original_execute = llm._execute_speculative_verified
    current_cell = None
    def observe_cycle(*positional, **keywords):
        nonlocal active
        active = dict(cell=current_cell, phases={})
        torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            result = original_execute(*positional, **keywords)
            torch.cuda.synchronize()
            active["total_seconds"] = time.perf_counter() - start
            if "result" in active:
                cycles.append(active)
            return result
        finally:
            active = None
    try:
        for batch in (1, 4):
            for family, temperature in (("greedy", 0.), ("plain", .8)):
                for workload in ("prose", "code", "repetitive", "adversarial"):
                    text = {"prose": "Explain how a computer predicts the next word in a sentence. ",
                            "code": "def fibonacci(n):\n    ", "repetitive": "one two three "}.get(workload)
                    ids = llm.tokenizer.encode(text) if text else [42, 997, 80001, 12345, 70003, 321]
                    prompts = [(ids * 32)[:32] for _ in range(batch)]
                    params = SamplingParams(temperature=temperature, max_tokens=32, ignore_eos=True)
                    current_cell = [batch, family, workload]
                    # Warm without observers; diagnostic timer starts only after.
                    torch.manual_seed(17)
                    llm.generate(prompts, params, use_tqdm=False)
                    torch.manual_seed(17)
                    llm._execute_speculative_verified = observe_cycle
                    start = len(cycles)
                    outputs = llm.generate(prompts, params, use_tqdm=False)
                    llm._execute_speculative_verified = original_execute
                    assert all(len(r["token_ids"]) == 32 for r in outputs)
                    assert llm.is_finished() and not llm.scheduler.block_manager.used_block_ids
                    assert len(cycles) > start
                    cells.append(dict(cell=current_cell, cycles=len(cycles) - start, outputs=outputs))
        assert source == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path("nanovllm").rglob("*.py"))}
        with args.output.open("x") as handle:
            json.dump(dict(schema="speculative-v7-phase-diagnostics-v1", mode=args.mode,
                           revision=revision, source_sha256=source,
                           harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           synchronized=True, headline=False, cells=cells, cycles=cycles), handle, indent=2, allow_nan=False)
        print("PASS", args.output, "cycles", len(cycles), flush=True)
    finally:
        for owner, name, original in reversed(restores):
            setattr(owner, name, original)
        llm._execute_speculative_verified = original_execute
        llm.exit()


def sys_optimized():
    import sys
    return bool(sys.flags.optimize)


if __name__ == "__main__":
    main()
