"""Fresh-process GPU checks for low-budget ragged CUDA-graph routing.

Run one configuration per process because nano-vllm owns a process group and
most of the selected GPU memory for the lifetime of an engine::

    PYTHONPATH=. python tests/run_varlen_graph_config.py --tau 64
    PYTHONPATH=. python tests/run_varlen_graph_config.py --tau 128
"""

import argparse
import json

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.utils.context import get_context, reset_context, set_context


MODEL_PATH = "/workspace/models/Qwen3-0.6B"


def check(tau: int):
    if tau not in (64, 128):
        raise ValueError("this focused check supports only tau 64 or 128")

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    llm = LLM(
        MODEL_PATH,
        max_num_batched_tokens=tau,
        max_num_seqs=8,
        max_model_len=512,
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
            routed_logits = runner.run_model(input_ids, positions, True).clone()

        if tau == 64:
            assert runner.varlen_miss == misses_before + 1
        else:
            assert runner.varlen_miss == misses_before
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
                eager_logits = runner.model.compute_logits(
                    runner.model(input_ids, positions)
                )
            assert torch.equal(
                routed_logits.argmax(dim=-1), eager_logits.argmax(dim=-1)
            )
    finally:
        reset_context()
        llm.scheduler.cancel_all()

    outputs = llm.generate(prompts, params, use_tqdm=False)
    assert len(outputs) == len(prompts)
    assert all(len(output["token_ids"]) == 1 for output in outputs)
    print(json.dumps({
        "tau": tau,
        "varlen_graphs": len(runner.varlen_graphs),
        "varlen_miss": runner.varlen_miss,
        "tokens": [output["token_ids"][0] for output in outputs],
    }, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau", type=int, required=True)
    check(parser.parse_args().tau)
