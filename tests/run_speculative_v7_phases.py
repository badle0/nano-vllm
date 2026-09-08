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
from run_speculative_v7_benchmark import digest_file, hardware


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
    hardware_before = hardware()
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
    def weight_ledger(model):
        # Parameter objects can share storage (tied embedding / LM head). Count
        # physical storage once; count the LM head, not embedding lookup, as GEMM.
        storages = {p.untyped_storage().data_ptr(): p.untyped_storage().nbytes() for p in model.parameters()}
        return dict(parameter_object_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
                    unique_storage_bytes=sum(storages.values()),
                    linear_weight_elements=sum(p.numel() for name, p in model.named_parameters()
                                               if p.ndim == 2 and name != "model.embed_tokens.weight"),
                    embedding_bytes=model.model.embed_tokens.weight.numel() * model.model.embed_tokens.weight.element_size(),
                    tied_storage=model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr())
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
                           args={**vars(args), "output": str(args.output)},
                           gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                           hardware_before=hardware_before, hardware_after=hardware(),
                           revision=revision, source_sha256=source,
                           harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           weight_ledger={"target": weight_ledger(llm.model_runner.model),
                                          "draft": weight_ledger(llm.model_runner.draft_model)},
                           models={label: {p.name: dict(bytes=p.stat().st_size, sha256=digest_file(p))
                                           for p in sorted(Path(directory).iterdir())
                                           if p.is_file() and (p.suffix == ".safetensors" or p.name == "config.json")}
                                   for label, directory in (("target", args.model), ("draft", args.draft_model))},
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
