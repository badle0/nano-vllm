"""V7 fresh-process benchmark runner; no model/sampler timing hooks in headlines."""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def digest_file(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def hardware():
    gpu = subprocess.check_output([
        "nvidia-smi", "--query-gpu=uuid,clocks.sm,clocks.mem,power.draw,temperature.gpu,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True).strip()
    pids = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip().splitlines()
    return dict(gpu=gpu, gpu_processes=pids, load_average=os.getloadavg(), cpu_count=os.cpu_count())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--enabled", action="store_true")
    parser.add_argument("--mode", choices=("eager", "graph"), default="graph")
    parser.add_argument("--pair", type=int, default=0)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=("smoke", "primary", "extended", "off-regression"), default="primary")
    parser.add_argument("--configured-k", type=int, default=4)
    args = parser.parse_args()
    if sys.flags.optimize:
        raise RuntimeError("assertions must be enabled")
    if args.output.exists():
        raise FileExistsError(args.output)
    source = args.source_root.resolve()
    sys.path.insert(0, str(source))
    import torch
    import nanovllm
    from nanovllm import LLM, SamplingParams
    assert Path(nanovllm.__file__).resolve() == source / "nanovllm/__init__.py"
    torch.set_num_threads(1)
    source_hashes = {str(p.relative_to(source)): digest_file(p) for p in sorted((source / "nanovllm").rglob("*.py"))}
    maximum_batch = 128 if args.suite == "extended" else 8
    config = dict(max_num_seqs=maximum_batch, max_num_batched_tokens=4096,
                  max_model_len=4096, num_kvcache_blocks=64,
                  gpu_memory_utilization=.8, enforce_eager=args.mode == "eager")
    if args.enabled:
        config.update(draft_model=args.draft_model, num_speculative_tokens=args.configured_k)
    started = time.perf_counter()
    llm = LLM(args.model, **config)
    initialization_seconds = time.perf_counter() - started
    runner = llm.model_runner
    from torch._dynamo.utils import counters
    def compiled():
        return counters["stats"]["unique_graphs"]
    families = dict(greedy=(0., -1, 1.), plain=(.8, -1, 1.), topk=(.8, 50, 1.),
                    topp=(.8, -1, .95), combined=(.8, 50, .95))
    if args.suite == "primary":
        cells = [(b, length, family, "generate", 64)
                 for b in (1, 4, 8) for length in (32, 256) for family in families]
    elif args.suite == "extended":
        cells = [(b, 32, "plain", "generate", 64) for b in (2, 16, 32, 64, 128)]
        cells += [(b, length, "plain", "generate", 64) for b in (1, 4) for length in (1024, 2048)]
        cells += [(1, 32, "plain", consumer, 16) for consumer in ("stream", "slow-stream")]
        cells += [(1, 32, "plain", "generate", n) for n in (1, 2, 4, 5, 256)]
    elif args.suite == "off-regression":
        cells = [(b, 32, "plain", "generate", 64) for b in (1, 4, 8)]
    else:
        cells = [(b, 32, family, "generate", 16) for b in (1, 4) for family in ("greedy", "plain")]
    seeds = (17, 23, 41)
    records = []
    def generate(prompts, params, consumer):
        if consumer == "generate":
            return llm.generate(prompts, params, use_tqdm=False)
        tokens = {}
        with llm.stream(prompts, params) as session:
            for event in session:
                tokens.setdefault(event.seq_id, []).append(event.token_id)
                if consumer == "slow-stream":
                    time.sleep(.002)
        return [dict(token_ids=ids, metrics=session.metrics[seq_id]) for seq_id, ids in tokens.items()]
    try:
        for batch, length, family, consumer, completion in cells:
            prefix = ("def fibonacci(n):\n    " if family in ("plain", "topp")
                      else "Explain how a computer predicts the next word in a sentence. ")
            ids = llm.tokenizer.encode(prefix)
            prompts = [(ids * ((length + len(ids) - 1) // len(ids)))[:length] for _ in range(batch)]
            t, k, p = families[family]
            params = SamplingParams(temperature=t, top_k=k, top_p=p, max_tokens=completion, ignore_eos=True)
            # Exactly two full workload warmups, including both consumer modes.
            for seed in seeds[:2]:
                torch.manual_seed(seed)
                generate(prompts, params, consumer)
            for seed in seeds:
                torch.manual_seed(seed)
                before = hardware()
                compile_before = compiled()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                outputs = generate(prompts, params, consumer)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                after = hardware()
                reasons = []
                for state in (before, after):
                    if len(state["gpu_processes"]) != 1:
                        reasons.append("not exactly one visible GPU process")
                    if state["load_average"][0] > .75 * state["cpu_count"]:
                        reasons.append("host load above 0.75 per logical CPU")
                    if float(state["gpu"].split(",")[4].strip()) >= 85:
                        reasons.append("GPU temperature >=85 C")
                if compiled() != compile_before:
                    reasons.append("timed sample compiled a new graph")
                assert len(outputs) == batch and all(len(r["token_ids"]) == completion for r in outputs)
                assert llm.is_finished() and not llm.scheduler.block_manager.used_block_ids
                records.append(dict(batch=batch, context=length, family=family, consumer=consumer,
                                    completion=completion, seed=seed, pair=args.pair, seconds=elapsed,
                                    tokens_per_second=batch * completion / elapsed, accepted=not reasons,
                                    rejection_reasons=sorted(set(reasons)), hardware_before=before,
                                    hardware_after=after, outputs=outputs,
                                    peak_allocated=torch.cuda.max_memory_allocated(),
                                    peak_reserved=torch.cuda.max_memory_reserved(), gc_enabled=gc.isenabled()))
                print("sample", batch, length, family, seed, round(elapsed, 4), "accepted", not reasons, flush=True)
        # Attainable calibration roofs on this host, not copied A100 datasheet peaks.
        src = torch.ones(64 * 1024**2, dtype=torch.float32, device="cuda")
        dst = torch.empty_like(src)
        for _ in range(10):
            dst.copy_(src)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(50):
            dst.copy_(src)
        end.record()
        end.synchronize()
        copy_ms = start.elapsed_time(end) / 50
        bandwidth = (2 * src.numel() * src.element_size()) / (copy_ms / 1000)
        del src, dst
        a = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
        b = torch.randn_like(a)
        out = torch.empty_like(a)
        for _ in range(10):
            torch.mm(a, b, out=out)
        start.record()
        for _ in range(50):
            torch.mm(a, b, out=out)
        end.record()
        end.synchronize()
        matmul_ms = start.elapsed_time(end) / 50
        calibration = dict(copy_bytes=2 * 64 * 1024**2 * 4, copy_ms=copy_ms,
                           bandwidth_bytes_per_second=bandwidth, matmul_shape=[4096] * 3,
                           matmul_ms=matmul_ms, bf16_flops_per_second=2 * 4096**3 / (matmul_ms / 1000))
        payload = dict(schema="speculative-v7-benchmark-v1", args={**vars(args), "source_root": str(source), "output": str(args.output)},
                       source_sha256=source_hashes, torch=torch.__version__, cuda=torch.version.cuda,
                       python=platform.python_version(), gpu=torch.cuda.get_device_name(),
                       config=config, initialization_seconds=initialization_seconds,
                       warmups_per_cell=2, seeds=seeds, records=records, calibration=calibration,
                       target_parameter_count=sum(p.numel() for p in runner.model.parameters()),
                       target_weight_bytes=sum(p.numel() * p.element_size() for p in runner.model.parameters()),
                       target_kv_bytes=runner.kv_cache.numel() * runner.kv_cache.element_size(),
                       target_config=runner.config.hf_config.to_dict(),
                       draft_weight_bytes=sum(p.numel() * p.element_size() for p in runner.draft_model.parameters()) if args.enabled else 0,
                       workspace=asdict(runner.speculative_memory_plan) if args.enabled else None,
                       models={label: {p.name: dict(bytes=p.stat().st_size, sha256=digest_file(p))
                                       for p in sorted(Path(directory).iterdir()) if p.is_file() and (p.suffix == ".safetensors" or p.name == "config.json")}
                               for label, directory in (("target", args.model), ("draft", args.draft_model))})
        assert source_hashes == {str(p.relative_to(source)): digest_file(p) for p in sorted((source / "nanovllm").rglob("*.py"))}
        with args.output.open("x") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
        print("PASS", args.output, flush=True)
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
