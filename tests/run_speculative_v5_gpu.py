"""Fresh-process V5/V6 GPU integration and timing evidence (not a speedup claim)."""
import argparse
from dataclasses import asdict
from dataclasses import replace
import hashlib
import gc
import json
import math
import os
import platform
import sys
from importlib.metadata import version
from pathlib import Path
import subprocess
import time
import warnings
import weakref

import torch
from nanovllm import LLM, SamplingParams, StreamingDetokenizer
from run_speculative_v3_route_compile import (
    cache_root_path, initialize_cache_root, require_compiler_environment,
    install_capture_ledger, compiler_snapshot, compiler_delta_summary,
    require_nonvacuous_compiler_snapshot, payload_sha256,
    validate_cache_root_isolation,
)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/workspace/models/Qwen3-0.6B")
    parser.add_argument("--draft-model")
    parser.add_argument("--mode", choices=("eager", "graph"), required=True)
    parser.add_argument("--numerical-mode", choices=("fast", "invariant"), default="fast")
    parser.add_argument("--enabled", action="store_true")
    parser.add_argument("--auto-kv", action="store_true")
    parser.add_argument("--logit-trace", action="store_true")
    parser.add_argument("--max-batch", type=int, default=4)
    parser.add_argument("--configured-k", type=int, default=4)
    parser.add_argument("--model-length", type=int, default=512)
    parser.add_argument("--token-budget", type=int, default=1024)
    parser.add_argument("--memory-utilization", type=float, default=.5)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--retained", action="store_true")
    parser.add_argument("--expected-commit")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if sys.flags.optimize:
        raise RuntimeError("GPU validation requires assertions enabled (no python -O)")
    import nanovllm
    if Path(nanovllm.__file__).resolve() != Path("nanovllm/__init__.py").resolve():
        raise RuntimeError("validation imported nano-vLLM outside the checkout")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(Path("nanovllm").rglob("*.py"))}
    roots, captures = (), None
    if args.retained:
        assert args.expected_commit == revision
        assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        require_compiler_environment()
        roots = tuple((name, cache_root_path(name)) for name in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"))
        validate_cache_root_isolation(roots[0][1], roots[1][1], repo_root=Path.cwd(),
                                      model_roots=(Path(args.model).resolve(), Path(args.draft_model or args.model).resolve()))
        for name, root in roots:
            initialize_cache_root(name, root)
        captures = install_capture_ledger()
    config = dict(max_num_seqs=args.max_batch, max_num_batched_tokens=args.token_budget, max_model_len=args.model_length,
                  gpu_memory_utilization=args.memory_utilization, enforce_eager=args.mode == "eager",
                  numerical_mode=args.numerical_mode)
    if not args.auto_kv:
        config["num_kvcache_blocks"] = 64
    if args.enabled:
        config.update(draft_model=args.draft_model or args.model, num_speculative_tokens=args.configured_k)
    torch.manual_seed(20260906)
    started = time.perf_counter()
    llm = LLM(args.model, **config)
    init_seconds = time.perf_counter() - started
    runner, scheduler = llm.model_runner, llm.scheduler
    from torch._dynamo.utils import counters
    def compile_counts():
        return {f"{group}/{key}": value for group, values in counters.items()
                for key, value in values.items() if isinstance(value, int)}
    init_compiles = compile_counts()
    init_snapshot = compiler_snapshot(roots, captures) if args.retained else None
    if args.retained:
        require_nonvacuous_compiler_snapshot(init_snapshot)
    cycles, results = [], []
    purpose = "normal"
    logit_trace = []
    if args.logit_trace:
        original_run, original_forward, original_greedy = runner.run, runner.sampler.forward, runner.sampler.greedy
        current_rows = []
        def trace_run(seqs, is_prefill):
            current_rows[:] = seqs
            return original_run(seqs, is_prefill)
        def record_logits(logits):
            values, ids = logits.topk(5, dim=-1)
            for seq, row_values, row_ids in zip(current_rows, values.tolist(), ids.tolist(), strict=True):
                logit_trace.append(dict(prefix=seq.token_ids[:],
                                        values=[v if math.isfinite(v) else None for v in row_values], ids=row_ids))
        def trace_forward(logits, *positional, **keywords):
            record_logits(logits)
            return original_forward(logits, *positional, **keywords)
        def trace_greedy(logits):
            record_logits(logits)
            return original_greedy(logits)
        runner.run, runner.sampler.forward = trace_run, trace_forward
        runner.sampler.greedy = trace_greedy
    if args.enabled:
        original = runner.run_speculative
        def observe(plan, seqs):
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            compiles_before = compile_counts()
            snapshot_before = compiler_snapshot(roots, captures) if args.retained else None
            result = original(plan, seqs)
            torch.cuda.synchronize()
            snapshot_after = compiler_snapshot(roots, captures) if args.retained else None
            if args.retained:
                assert snapshot_before == snapshot_after, compiler_delta_summary(snapshot_before, snapshot_after)
            cycles.append(dict(batch=len(seqs), k=plan.effective_k,
                               purpose=purpose,
                               verifier=("invariant_parallel_greedy"
                                         if any(s.temperature == 0. for s in seqs)
                                         and runner.numerical_mode == "invariant"
                                         else "sequential_greedy"
                                         if any(s.temperature == 0. for s in seqs)
                                         else "paged_parallel"),
                               catchup=plan.draft_catchup_tokens,
                               seconds=time.perf_counter() - start,
                               peak_increment=torch.cuda.max_memory_allocated() - baseline,
                               reservation=runner.speculative_memory_plan.reservation_bytes,
                               compile_delta={key: value - compiles_before.get(key, 0)
                                              for key, value in compile_counts().items()
                                              if value != compiles_before.get(key, 0)},
                               compiler_sha256=payload_sha256(snapshot_before) if args.retained else None,
                               compiler_unchanged=snapshot_before == snapshot_after,
                               result=asdict(result)))
            return result
        runner.run_speculative = observe
    def check_drained():
        assert llm.is_finished()
        assert not scheduler.block_manager.used_block_ids
        assert not scheduler.block_manager._active_temporary_reservations
        assert scheduler._active_spec_transaction is None
    try:
        for length, batch in ((16, 1), (255, 2), (256, 3), (257, 4)):
            prompts = [[42 + i] * (length - i) for i in range(batch)]
            params = SamplingParams(temperature=0., max_tokens=17, ignore_eos=True)
            start = time.perf_counter()
            generated = llm.generate(prompts, params, use_tqdm=False)
            elapsed = time.perf_counter() - start
            tokens = [item["token_ids"] for item in generated]
            assert all(len(row) == 17 for row in tokens)
            check_drained()
            # Repeat with prefix cache hits, then compare the streaming protocol.
            repeated = llm.generate(prompts, params, use_tqdm=False)
            assert tokens == [item["token_ids"] for item in repeated]
            check_drained()
            streamed = {}
            rendered = {}
            detokenizer = StreamingDetokenizer(llm.tokenizer)
            terminal = set()
            with llm.stream(prompts, params) as session:
                for event in session:
                    assert event.seq_id not in terminal
                    streamed.setdefault(event.seq_id, []).append(event.token_id)
                    rendered[event.seq_id] = detokenizer.feed(event.seq_id, event.token_id).apply(rendered.get(event.seq_id, ""))
                    if event.finished:
                        rendered[event.seq_id] = detokenizer.flush(event.seq_id).apply(rendered[event.seq_id])
                        terminal.add(event.seq_id)
            assert list(streamed.values()) == tokens
            assert list(rendered.values()) == [item["text"] for item in generated]
            assert len(terminal) == batch
            check_drained()
            results.append(dict(length=length, batch=batch, tokens=tokens, seconds=elapsed,
                                cache_repeat=True, stream_parity=True))
        # Sampling mixtures hit exact temperature/top-k/top-p metadata expansion.
        mixed = [SamplingParams(temperature=t, top_k=k, top_p=p, max_tokens=13, ignore_eos=True)
                 for t, k, p in ((0., -1, 1.), (0.7, 8, 1.), (0.8, -1, 0.9), (1.1, 16, 0.8))]
        sampled = llm.generate([[31 + i] * (20 + i) for i in range(4)], mixed, use_tqdm=False)
        assert all(len(row["token_ids"]) == 13 for row in sampled)
        check_drained()
        stochastic = [SamplingParams(temperature=t, top_k=k, top_p=p, max_tokens=13, ignore_eos=True)
                      for t, k, p in ((1., -1, 1.), (.7, 8, 1.), (.8, -1, .9), (1.1, 16, .8))]
        parallel = llm.generate([[51 + i] * (20 + i) for i in range(4)], stochastic, use_tqdm=False)
        assert all(len(row["token_ids"]) == 13 for row in parallel)
        check_drained()
        if args.max_batch > 4:
            count = len(cycles)
            overflow = llm.generate([[42 + i] * 16 for i in range(5)], stochastic[0], use_tqdm=False)
            assert len(cycles) == count, "one-above-cap batch must use baseline"
            assert all(len(row["token_ids"]) == 13 for row in overflow)
            check_drained()
        sweep_cells = []
        if args.enabled and args.sweep:
            original_resolver = runner.resolve_draft_route_admission
            try:
                for batch, cap in sorted(runner.speculative_verifier_shapes):
                    runner.resolve_draft_route_admission = lambda seqs, cap=cap: replace(
                        original_resolver(seqs), route_keys=original_resolver(seqs).route_keys[:cap])
                    for name, t, k, p in (("greedy", 0., -1, 1.), ("plain", 1., -1, 1.),
                                           ("topk", .7, 8, 1.), ("topp", .8, -1, .9),
                                           ("combined", 1.1, 16, .8)):
                        start = len(cycles)
                        generated = llm.generate([[62 + i] * (9 + i) for i in range(batch)],
                                                 SamplingParams(temperature=t, top_k=k, top_p=p,
                                                                max_tokens=cap + 2, ignore_eos=True), use_tqdm=False)
                        assert all(len(row["token_ids"]) == cap + 2 for row in generated)
                        assert any(c["batch"] == batch and c["k"] == cap for c in cycles[start:])
                        check_drained()
                        sweep_cells.append(dict(batch=batch, k=cap, sampling=name, cycles=len(cycles) - start))
            finally:
                runner.resolve_draft_route_admission = original_resolver
        # Different models need not accept their first proposal. Find an actual
        # pending burst rather than assuming the second read produced one.
        lifecycle_prompt = [42] * 16
        if args.draft_model and Path(args.draft_model).resolve() != Path(args.model).resolve():
            phrase = llm.tokenizer.encode("Explain how a computer predicts the next word in a sentence. ")
            lifecycle_prompt = (phrase * 32)[:32]
        def consume_partial_burst(session):
            for _ in range(24 if args.enabled else 2):
                next(session)
                if session._pending:
                    return
            if args.enabled:
                raise AssertionError("lifecycle fixture produced no accepted burst")
        with llm.stream([lifecycle_prompt], SamplingParams(temperature=0., max_tokens=24, ignore_eos=True)) as session:
            consume_partial_burst(session)
            pending_before_close = len(session._pending)
        assert not session._pending
        check_drained()
        with llm.stream([[42] * 16], SamplingParams(max_tokens=4, ignore_eos=True)):
            pass  # close before first step
        check_drained()
        abandoned = llm.stream([lifecycle_prompt], SamplingParams(temperature=0., max_tokens=24, ignore_eos=True))
        consume_partial_burst(abandoned)
        abandoned_pending = len(abandoned._pending)
        reference = weakref.ref(abandoned)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            del abandoned
            gc.collect()
        assert reference() is None
        assert any("garbage-collected" in str(w.message) for w in caught)
        check_drained()
        # A fresh session proves finalizer released the exclusive engine lease.
        with llm.stream([lifecycle_prompt], SamplingParams(temperature=0., max_tokens=9, ignore_eos=True)) as completed:
            terminal_events = list(completed)
        metrics = next(iter(completed.metrics.values()))
        assert len(terminal_events) == metrics["num_completion_tokens"] == 9
        assert sum(event.finished for event in terminal_events) == 1
        if args.enabled:
            assert metrics["spec_residual_numerical_fallbacks"] == 0
            assert metrics["spec_committed_tokens"] > 0
            assert 0. in metrics["engine_itls"]
            assert abandoned_pending > 0
        else:
            assert not any(key.startswith("spec_") for key in metrics)
        check_drained()
        forced_metrics = None
        failure_retry = False
        causality_probe = False
        if args.enabled:
            import nanovllm.engine.speculative_execution as execution
            parallel_verifier = execution.target_probabilities
            def probe_causality(runner, rows, proposals):
                original = parallel_verifier(runner, rows, proposals).clone()
                altered = proposals.clone()
                altered[:, -1] = (altered[:, -1] + 1) % original.size(-1)
                perturbed = parallel_verifier(runner, rows, altered).clone()
                assert torch.equal(original[:, :-1], perturbed[:, :-1]), "future proposal leaked into earlier target law"
                assert not torch.equal(original[:, -1], perturbed[:, -1]), "causality probe was vacuous"
                # Restore actual proposal KV before acceptance/commit.
                return parallel_verifier(runner, rows, proposals)
            purpose = "instrumented_causality_probe"
            execution.target_probabilities = probe_causality
            try:
                llm.generate([[51] * 21, [52] * 23], SamplingParams(temperature=.8, max_tokens=7, ignore_eos=True), use_tqdm=False)
                causality_probe = True
            finally:
                execution.target_probabilities = parallel_verifier
            check_drained()
            from nanovllm.layers.sampler import ModifiedRejectionResult
            rejection = runner.speculative_rejection_sampler
            accept_name = (
                "accept_trusted" if hasattr(rejection, "accept_trusted") else "accept"
            )
            real_accept = getattr(rejection, accept_name)
            def forced_accept(p, tokens, q):
                setattr(rejection, accept_name, real_accept)  # force exactly one cycle
                correction = rejection.sample_correction(p[:, 0], p[:, 0])
                assert correction.target_fallback.all()
                return ModifiedRejectionResult(torch.zeros(p.size(0), dtype=torch.int64, device=p.device),
                                                correction.token_ids, correction.used_reference,
                                                correction.target_fallback)
            purpose = "forced_empty_residual"
            setattr(rejection, accept_name, forced_accept)
            try:
                # Invariant homogeneous greedy uses the direct argmax verifier
                # and therefore has no rejection sampler to inject. Exercise
                # numerical residual recovery through a stochastic exact row.
                forced_temperature = (
                    0.8 if runner.numerical_mode == "invariant" else 0.0
                )
                forced = llm.generate(
                    [[42] * 16],
                    SamplingParams(
                        temperature=forced_temperature,
                        max_tokens=9,
                        ignore_eos=True,
                    ),
                    use_tqdm=False,
                )
            finally:
                setattr(rejection, accept_name, real_accept)
            forced_metrics = forced[0]["metrics"]
            assert forced_metrics["spec_residual_numerical_fallbacks"] == 1
            check_drained()
            purpose = "failure_after_verifier"
            llm.add_request([42] * 16, SamplingParams(temperature=0., max_tokens=12, ignore_eos=True))
            llm._step()  # finish prefill before injecting a speculative failure
            def state():
                return [(s.token_ids[:], s.num_cached_tokens, s.num_draft_cached_tokens, s.block_table[:])
                        for s in scheduler.running], list(scheduler.block_manager.free_block_ids), set(scheduler.block_manager.used_block_ids)
            before_state = state()
            before_rng = runner.snapshot_speculative_rng()
            observed = runner.run_speculative
            def fail_after_verify(*positional):
                observed(*positional)
                raise RuntimeError("injected failure after target verification")
            runner.run_speculative = fail_after_verify
            try:
                try:
                    llm._step()
                except RuntimeError as error:
                    assert "injected failure" in str(error)
                else:
                    raise AssertionError("injected failure did not propagate")
            finally:
                runner.run_speculative = observed
            assert state() == before_state
            after_rng = runner.snapshot_speculative_rng()
            assert all(torch.equal(a, b) for a, b in zip(before_rng, after_rng, strict=True))
            purpose = "failure_retry"
            finished = []
            while not llm.is_finished():
                manual_outputs, count = llm.step()
                finished.extend(manual_outputs)
                assert type(count) is int
            assert len(finished) == 1 and finished[0][1] == results[0]["tokens"][0][:12]
            failure_retry = True
            check_drained()
            purpose = "normal"
        if args.enabled:
            assert pending_before_close > 0
            assert cycles and any(any(row["accepted_draft_tokens"] > 0 for row in c["result"]["rows"]) for c in cycles)
            assert all(c["peak_increment"] <= c["reservation"] + runner._warmup_transient_bytes for c in cycles)
            assert all(c["peak_increment"] <= runner.speculative_memory_plan.modeled_live_peak_bytes
                       for c in cycles if c["purpose"] == "normal"), "production live peak exceeds its model"
        sort_scratch = []
        if args.enabled:
            for rows in (4, 12, 16, 20):
                logits = torch.randn(rows, runner.config.hf_config.vocab_size, dtype=torch.bfloat16, device="cuda")
                temperatures = torch.ones(rows, device="cuda")
                cutoffs = torch.full((rows,), .1, device="cuda")
                torch.cuda.synchronize()
                baseline = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                runner.sampler.filter_top_p(logits, temperatures, None, cutoffs)
                torch.cuda.synchronize()
                measured = torch.cuda.max_memory_allocated() - baseline
                priced = rows * logits.size(1) * 52
                assert measured <= priced, "top-p private scratch exceeds its two-payload allowance"
                sort_scratch.append(dict(rows=rows, measured=measured, priced=priced))
                del logits, temperatures, cutoffs
        payload = dict(schema="nano-vllm-speculative-v5-gpu-v1", args=vars(args), config=config,
                       revision=revision, source_sha256=source_hashes,
                       torch=torch.__version__, cuda=torch.version.cuda,
                       gpu=torch.cuda.get_device_name(), init_seconds=init_seconds,
                       python=platform.python_version(),
                       packages={name: version(name) for name in ("torch", "transformers", "triton", "flash-attn")},
                       driver=subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip(),
                       model_files={p.name: dict(bytes=p.stat().st_size, sha256=file_sha256(p))
                                    for p in sorted(Path(args.model).iterdir())
                                    if p.is_file() and (p.suffix == ".safetensors" or p.name == "config.json")},
                       draft_model_files={p.name: dict(bytes=p.stat().st_size, sha256=file_sha256(p))
                                          for p in sorted(Path(args.draft_model or args.model).iterdir())
                                          if p.is_file() and (p.suffix == ".safetensors" or p.name == "config.json")},
                       compiler_environment={name: value for name, value in os.environ.items()
                                             if name.startswith(("TORCHINDUCTOR_", "TORCH_DYNAMO_", "TRITON_CACHE")) or name == "TORCH_LOGS"},
                       results=results, cycles=cycles, mixed_sample_lengths=[len(r["token_ids"]) for r in sampled],
                       pending_before_close=pending_before_close,
                       abandoned_pending=abandoned_pending, completed_metrics=metrics,
                       forced_metrics=forced_metrics, failure_retry=failure_retry, causality_probe=causality_probe,
                       sort_scratch=sort_scratch,
                       logit_trace=logit_trace,
                       stochastic_lengths=[len(r["token_ids"]) for r in parallel],
                       sweep_cells=sweep_cells,
                       init_compiler_snapshot=init_snapshot,
                       init_compiles=init_compiles, final_compiles=compile_counts(),
                       audit={**runner.speculative_memory_audit._asdict(),
                              "workspace_plan": asdict(runner.speculative_memory_plan)} if args.enabled else None)
        assert source_hashes == {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(Path("nanovllm").rglob("*.py"))}, "source changed while running"
        with output.open("x") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
        print("PASS", args.mode, args.enabled, "cycles", len(cycles), "output", output, flush=True)
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
