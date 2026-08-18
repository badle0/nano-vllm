"""Fresh-process GPU worker for low-budget ragged CUDA-graph routing.

Run one configuration per process because nano-vllm owns a process group and
most of the selected GPU memory for the lifetime of an engine::

    PYTHONPATH=. python tests/run_varlen_graph_config.py --tau 64 --max-model-len 512
    PYTHONPATH=. python tests/run_varlen_graph_config.py --tau 128 --max-model-len 512

The retained matrix orchestrator runs all six supported cells and owns release
pinning/output. This worker emits exactly one JSON object on stdout.
"""

import argparse
import json

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.utils.context import get_context, reset_context, set_context


DEFAULT_MODEL = "/workspace/models/Qwen3-0.6B"


def check(
    tau: int,
    max_model_len: int = 512,
    model_path: str = DEFAULT_MODEL,
) -> dict[str, object]:
    if tau not in (64, 128):
        raise ValueError("this focused check supports only tau 64 or 128")
    if max_model_len not in (512, 1024, 4096):
        raise ValueError("max_model_len must be 512, 1024, or 4096")

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    llm = LLM(
        model_path,
        max_num_batched_tokens=tau,
        max_num_seqs=8,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.5,
        enforce_eager=False,
    )
    runner = llm.model_runner

    if tau == 64:
        assert runner.varlen_ts == []
        assert runner.varlen_slots == []
        assert runner.varlen_graphs == {}
        assert runner.varlen_vars is None
    else:
        assert runner.varlen_ts == [128]
        assert runner.varlen_graphs
        assert runner._select_varlen_graph_key(124, 4) is not None

    params = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    prompt_len = 15 if tau == 64 else 31
    prompts = [
        [1000 + row * 100 + column for column in range(prompt_len)]
        for row in range(4)
    ]
    for prompt in prompts:
        llm.add_request(prompt, params)

    try:
        seqs, is_prefill = llm.scheduler.schedule()
        assert is_prefill and len(seqs) == len(prompts)
        input_ids, positions = runner.prepare_ragged(seqs)
        live = get_context()
        misses_before = runner.varlen_miss
        with torch.inference_mode():
            eager_logits = runner.model.compute_logits(
                runner.model(input_ids, positions)
            ).clone()
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

        candidate = runner._select_varlen_graph_key(
            int(input_ids.numel()), len(prompts)
        )
        eager_tokens = eager_logits.argmax(dim=-1).cpu().tolist()
        routed_tokens = routed_logits.argmax(dim=-1).cpu().tolist()
        miss_delta = runner.varlen_miss - misses_before
        misses_after_routed = runner.varlen_miss
        assert routed_tokens == eager_tokens
        if tau == 64:
            assert candidate is None
            assert miss_delta == 1
            expected_route = "prefill_eager_no_bucket"
        else:
            assert candidate is not None
            assert miss_delta == 0
            expected_route = "varlen_cuda_graph"
    finally:
        reset_context()
        llm.scheduler.cancel_all()

    outputs = llm.generate(prompts, params, use_tqdm=False)
    assert len(outputs) == len(prompts)
    assert all(len(output["token_ids"]) == 1 for output in outputs)
    generated_tokens = [output["token_ids"][0] for output in outputs]
    assert generated_tokens == routed_tokens
    return {
        "tau": tau,
        "max_model_len": max_model_len,
        "model": model_path,
        "prompt_count": len(prompts),
        "prompt_length": prompt_len,
        "real_input_tokens": int(input_ids.numel()),
        "candidate_graph_key": list(candidate) if candidate is not None else None,
        "expected_route": expected_route,
        "varlen_graph_keys": [list(key) for key in sorted(runner.varlen_graphs)],
        "varlen_graph_count": len(runner.varlen_graphs),
        "varlen_miss_before": misses_before,
        "varlen_miss_after_routed": misses_after_routed,
        "varlen_miss_delta": miss_delta,
        "varlen_miss_after_end_to_end": runner.varlen_miss,
        "eager_tokens": eager_tokens,
        "routed_tokens": routed_tokens,
        "generated_tokens": generated_tokens,
        "argmax_equal": True,
        "passed": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau", type=int, required=True)
    parser.add_argument(
        "--max-model-len",
        type=int,
        choices=(512, 1024, 4096),
        default=512,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    print(json.dumps(
        check(args.tau, args.max_model_len, args.model),
        sort_keys=True,
    ))
