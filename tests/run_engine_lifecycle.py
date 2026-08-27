"""Standalone TP1 GPU regression for failed and repeated engine construction.

Run this file in its own process; it intentionally creates several process
groups and CUDA-graph pools in sequence.
"""

import argparse
import torch
import torch.distributed as dist

from nanovllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--failure-utilization", type=float, default=0.001)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("eager", "graph"),
        default=("graph", "eager", "graph"),
    )
    return parser.parse_args()


def engine_kwargs(args, *, utilization, mode):
    return dict(
        enforce_eager=mode == "eager",
        gpu_memory_utilization=utilization,
        max_model_len=512,
        max_num_batched_tokens=512,
        max_num_seqs=4,
    )


def assert_process_state(device, dtype):
    assert not dist.is_initialized(), "engine left a process group initialized"
    assert torch.get_default_device() == device
    assert torch.get_default_dtype() == dtype


def main():
    args = parse_args()
    original_device = torch.get_default_device()
    original_dtype = torch.get_default_dtype()

    try:
        LLM(
            args.model,
            **engine_kwargs(
                args,
                utilization=args.failure_utilization,
                mode="eager",
            ),
        )
    except RuntimeError as error:
        message = str(error)
        assert "cannot allocate any KV-cache blocks" in message
        assert "usable=" in message and "block_bytes=" in message
    else:
        raise AssertionError("failure utilization unexpectedly constructed an engine")
    assert_process_state(original_device, original_dtype)

    # Retain exited engines and do not run caller-side GC/empty_cache between
    # constructors. The lifecycle implementation itself must release resources.
    exited_engines = []
    for index, mode in enumerate(args.modes, 1):
        llm = LLM(
            args.model,
            **engine_kwargs(
                args,
                utilization=args.gpu_memory_utilization,
                mode=mode,
            ),
        )
        model_calls = []
        original_call = llm.model_runner.call

        def counted_call(method_name, *call_args):
            model_calls.append(method_name)
            return original_call(method_name, *call_args)

        llm.model_runner.call = counted_call
        boundary_cases = (
            ([1] * 512, 1),
            ([2] * 511, 2),
        )
        for prompt, max_tokens in boundary_cases:
            outputs = llm.generate(
                [prompt],
                SamplingParams(
                    temperature=0.0,
                    max_tokens=max_tokens,
                    ignore_eos=True,
                ),
                use_tqdm=False,
            )
            assert len(outputs[0]["token_ids"]) == max_tokens

        calls_before_rejection = len(model_calls)
        try:
            llm.generate(
                [[3] * 512],
                SamplingParams(
                    temperature=0.0,
                    max_tokens=2,
                    ignore_eos=True,
                ),
                use_tqdm=False,
            )
        except ValueError as error:
            assert "model-processed tokens exceeds max_model_len" in str(error)
        else:
            raise AssertionError("P=max_model_len,N=2 was not rejected")
        assert len(model_calls) == calls_before_rejection
        blocks = len(llm.scheduler.block_manager.blocks)
        llm.exit()
        exited_engines.append(llm)
        assert_process_state(original_device, original_dtype)
        print(
            f"restart={index} mode={mode} blocks={blocks} "
            f"allocated={torch.cuda.memory_allocated()} "
            f"reserved={torch.cuda.memory_reserved()}"
        )

    assert len(exited_engines) == len(args.modes)
    print("engine lifecycle: PASS")


if __name__ == "__main__":
    main()
