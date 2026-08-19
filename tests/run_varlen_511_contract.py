"""Explicit 4 x 511, max_model_len=512 graph-routing regression.

This is a fresh-process GPU contract rather than a pytest because a nano-vllm
engine owns its process group and most of its selected GPU allocation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

from benchmarks.chunked_prefill_tail.common import (
    base_result,
    environment_identity,
    handle_pin_query,
    immutable_write_json,
    model_identity,
    validate_release_pin,
)


DEFAULT_MODEL = "/workspace/models/Qwen3-0.6B"
PROMPT_COUNT = 4
PROMPT_LENGTH = 511
MAX_MODEL_LEN = 512
TOKEN_BUDGET = 2048
EXPECTED_GRAPH_KEY = (2048, 5)


def boundary_prompts(vocab_size: int) -> list[list[int]]:
    """Return four deterministic, distinct, valid 511-token prompts."""
    if vocab_size <= 16:
        raise ValueError(f"unexpectedly small vocabulary: {vocab_size}")
    span = vocab_size - 8
    prompts = [
        [8 + ((row * 7919 + column * 104729 + 17) % span)
         for column in range(PROMPT_LENGTH)]
        for row in range(PROMPT_COUNT)
    ]
    assert len({tuple(prompt) for prompt in prompts}) == PROMPT_COUNT
    return prompts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Certify the 4x511 unpadded ragged graph boundary."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--expected-commit")
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--print-source-sha256", action="store_true")
    parser.add_argument("--show-pin", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    missing = [
        name
        for name in ("expected_commit", "expected_source_sha256", "output")
        if getattr(args, name) is None
    ]
    if missing:
        raise ValueError(
            "runtime contract requires "
            + ", ".join("--" + name.replace("_", "-") for name in missing)
        )
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        raise ValueError("--gpu-memory-utilization must be in (0, 1)")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite retained output: {args.output}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if handle_pin_query(args.print_source_sha256, args.show_pin):
        return 0
    _validate_args(args)

    import torch
    import transformers

    from nanovllm import LLM, SamplingParams
    from nanovllm.utils.context import get_context, reset_context, set_context

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the retained boundary contract")

    started = perf_counter()
    pin = validate_release_pin(args.expected_commit, args.expected_source_sha256)
    model = model_identity(Path(args.model))
    environment = environment_identity(torch, transformers)
    result = base_result(
        "varlen_graph_4x511_maxlen512_contract",
        [sys.executable, *sys.argv],
        pin,
        model,
        environment,
    )
    result["arguments"] = vars(args) | {"output": str(args.output)}

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    engine_started = perf_counter()
    llm = LLM(
        args.model,
        max_num_batched_tokens=TOKEN_BUDGET,
        max_num_seqs=PROMPT_COUNT,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False,
    )
    torch.cuda.synchronize()
    engine_init_ms = (perf_counter() - engine_started) * 1000.0
    runner = llm.model_runner
    prompts = boundary_prompts(int(runner.config.hf_config.vocab_size))
    params = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    for prompt in prompts:
        llm.add_request(prompt, params)

    eager_logits = routed_logits = None
    try:
        seqs, is_prefill = llm.scheduler.schedule()
        if not is_prefill or len(seqs) != PROMPT_COUNT:
            raise AssertionError("4x511 requests were not scheduled as one ragged step")
        input_ids, positions = runner.prepare_ragged(seqs)
        live = get_context()
        num_tokens = int(input_ids.numel())
        num_segments = int(live.cu_seqlens_q.numel() - 1)
        q_lengths = (
            live.cu_seqlens_q[1:] - live.cu_seqlens_q[:-1]
        ).cpu().tolist()
        k_lengths = (
            live.cu_seqlens_k[1:] - live.cu_seqlens_k[:-1]
        ).cpu().tolist()
        if num_tokens != PROMPT_COUNT * PROMPT_LENGTH:
            raise AssertionError(f"expected 2044 real tokens, found {num_tokens}")
        if q_lengths != [PROMPT_LENGTH] * PROMPT_COUNT:
            raise AssertionError(f"unexpected query lengths: {q_lengths}")
        if k_lengths != [PROMPT_LENGTH] * PROMPT_COUNT:
            raise AssertionError(f"unexpected key lengths: {k_lengths}")

        candidate = runner._select_varlen_graph_key(num_tokens, num_segments)
        if candidate != EXPECTED_GRAPH_KEY:
            raise AssertionError(
                f"expected graph key {EXPECTED_GRAPH_KEY}, found {candidate}"
            )
        if not runner._varlen_context_fits_graph(
            num_tokens, num_segments, live, candidate
        ):
            raise AssertionError("4x511 live metadata does not fit its selected graph")

        misses_before = runner.varlen_miss
        with torch.inference_mode():
            eager_logits = runner.model.compute_logits(
                runner.model(input_ids, positions)
            ).clone()
        # Preserve the exact live unpadded context for the production graph
        # route and for the output head's last-token gather.
        set_context(
            True,
            live.cu_seqlens_q,
            live.cu_seqlens_k,
            live.max_seqlen_q,
            live.max_seqlen_k,
            live.slot_mapping,
            None,
            live.block_tables,
        )
        with torch.inference_mode():
            routed_logits = runner.run_model(input_ids, positions, True).clone()
        torch.cuda.synchronize()
        miss_delta = runner.varlen_miss - misses_before
        eager_tokens = eager_logits.argmax(dim=-1).cpu().tolist()
        routed_tokens = routed_logits.argmax(dim=-1).cpu().tolist()
        if miss_delta != 0:
            raise AssertionError(f"4x511 graph route incurred {miss_delta} miss(es)")
        if routed_tokens != eager_tokens:
            raise AssertionError(
                f"graph/eager argmax mismatch: {routed_tokens} != {eager_tokens}"
            )
        difference = (routed_logits.float() - eager_logits.float()).abs()
        comparison = {
            "eager_tokens": eager_tokens,
            "routed_tokens": routed_tokens,
            "argmax_equal": True,
            "max_abs_logit_difference": float(difference.max().item()),
            "mean_abs_logit_difference": float(difference.mean().item()),
            "varlen_miss_before": misses_before,
            "varlen_miss_after": runner.varlen_miss,
            "varlen_miss_delta": miss_delta,
        }
    finally:
        reset_context()
        llm.scheduler.cancel_all()

    outputs = llm.generate(prompts, params, use_tqdm=False)
    generated_tokens = [output["token_ids"][0] for output in outputs]
    if generated_tokens != comparison["routed_tokens"]:
        raise AssertionError(
            "end-to-end greedy tokens differ from the routed oracle: "
            f"{generated_tokens} != {comparison['routed_tokens']}"
        )

    result.update({
        "engine": {
            "initialization_ms": engine_init_ms,
            "config": {
                "max_num_batched_tokens": runner.config.max_num_batched_tokens,
                "max_num_seqs": runner.config.max_num_seqs,
                "max_model_len": runner.config.max_model_len,
                "kvcache_block_size": runner.config.kvcache_block_size,
                "num_kvcache_blocks": runner.config.num_kvcache_blocks,
            },
            "varlen_graph_keys": [list(key) for key in sorted(runner.varlen_graphs)],
        },
        "contract": {
            "prompt_count": PROMPT_COUNT,
            "prompt_length": PROMPT_LENGTH,
            "real_input_tokens": num_tokens,
            "segment_count": num_segments,
            "query_lengths": q_lengths,
            "key_lengths": k_lengths,
            "candidate_graph_key": list(candidate),
            "expected_graph_key": list(EXPECTED_GRAPH_KEY),
            "comparison": comparison,
            "generated_tokens": generated_tokens,
            "passed": True,
        },
        "elapsed_ms_before_write": (perf_counter() - started) * 1000.0,
        "provenance_after_run": validate_release_pin(
            args.expected_commit, args.expected_source_sha256
        ),
    })
    immutable_write_json(args.output, result)
    print(json.dumps({
        "output": str(args.output),
        "graph_key": list(candidate),
        "tokens": generated_tokens,
        "passed": True,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
