"""Independent forced-history diagnostics for the invariant numerical backend."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from benchmarks.chunked_prefill_tail.common import model_identity
from nanovllm import LLM, SamplingParams


PROMPTS = (
    tuple(1000 + index for index in range(31)),
    tuple(2000 + index for index in range(257)),
)
FORCED_COMPLETIONS = ((17, 19, 23), (29, 31, 37))


def _tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _implementation_sha256(root: Path) -> str:
    """Bind evidence to the complete runtime, including future helper moves."""

    digest = hashlib.sha256()
    for path in sorted((root / "nanovllm").rglob("*.py")):
        filename = str(path.relative_to(root))
        digest.update(filename.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run(model_path: str, *, budget: int) -> dict:
    torch.manual_seed(20260908)
    llm = LLM(
        model_path,
        numerical_mode="invariant",
        enforce_eager=True,
        max_num_batched_tokens=budget,
        max_num_seqs=len(PROMPTS),
        max_model_len=512,
        num_kvcache_blocks=8,
        gpu_memory_utilization=0.5,
    )
    traces = []
    active_rows = []
    request_indices = {}
    original_run_model = llm.model_runner.run_model

    def traced_run_model(input_ids, positions, is_prefill):
        logits = original_run_model(input_ids, positions, is_prefill)
        cache = llm.model_runner.kv_cache.flatten(2, 3)
        emission_row = 0
        for seq in active_rows:
            processed = seq.num_cached_tokens + seq.num_scheduled_tokens
            prefix = tuple(seq.token_ids[:processed])
            slots = torch.tensor(
                [
                    seq.block_table[position // llm.model_runner.block_size]
                    * llm.model_runner.block_size
                    + position % llm.model_runner.block_size
                    for position in range(processed)
                ],
                dtype=torch.int64,
                device=cache.device,
            )
            live_kv = cache.index_select(2, slots)
            emits = processed == seq.num_tokens
            record = {
                "request": request_indices[seq.seq_id],
                "processed_tokens": processed,
                "prefix_sha256": hashlib.sha256(repr(prefix).encode()).hexdigest(),
                "kv_sha256": _tensor_sha256(live_kv),
                "emits": emits,
            }
            if emits:
                record["logits_sha256"] = _tensor_sha256(logits[emission_row])
                record["argmax"] = int(logits[emission_row].argmax().item())
                emission_row += 1
            traces.append(record)
        if emission_row != logits.size(0):
            raise RuntimeError("emission-row mapping disagrees with logits")
        return logits

    llm.model_runner.run_model = traced_run_model
    try:
        seqs = [
            llm._admit_request(
                list(prompt),
                SamplingParams(temperature=0.0, max_tokens=3, ignore_eos=True),
            )
            for prompt in PROMPTS
        ]
        request_indices = {seq.seq_id: index for index, seq in enumerate(seqs)}
        outputs = [[] for _ in seqs]
        while not llm.scheduler.is_finished():
            scheduled, is_prefill = llm.scheduler.schedule()
            active_rows[:] = scheduled
            computed = llm.model_runner.call("run", scheduled, is_prefill)
            forced = list(computed)
            for row_index, seq in enumerate(scheduled):
                if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                    request = request_indices[seq.seq_id]
                    forced[row_index] = FORCED_COMPLETIONS[request][
                        seq.num_completion_tokens
                    ]
            events = llm.scheduler.postprocess(scheduled, forced)
            for event in events:
                outputs[request_indices[event.seq_id]].append(event.token_id)
        root = Path(__file__).resolve().parents[1]
        return {
            "kind": "invariant_forced_history_numerics_v1",
            "model": model_identity(Path(model_path)),
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "numerical_mode": "invariant",
            "budget": budget,
            "prompts": [list(prompt) for prompt in PROMPTS],
            "forced_completions": [list(tokens) for tokens in FORCED_COMPLETIONS],
            "outputs": outputs,
            "implementation_sha256": _implementation_sha256(root),
            "traces": traces,
        }
    finally:
        llm.exit()


def compare(current: dict, reference: dict) -> dict:
    for field in (
        "kind",
        "model",
        "numerical_mode",
        "prompts",
        "forced_completions",
        "outputs",
        "implementation_sha256",
    ):
        if current[field] != reference[field]:
            raise AssertionError(f"comparison identity differs at {field}")

    def keyed(report, field):
        return {
            (
                item["request"],
                item["processed_tokens"],
                item["prefix_sha256"],
            ): item[field]
            for item in report["traces"]
            if field in item
        }

    current_logits = keyed(current, "logits_sha256")
    reference_logits = keyed(reference, "logits_sha256")
    current_kv = keyed(current, "kv_sha256")
    reference_kv = keyed(reference, "kv_sha256")
    logit_keys = sorted(current_logits.keys() & reference_logits.keys())
    kv_keys = sorted(current_kv.keys() & reference_kv.keys())
    if not logit_keys or not kv_keys:
        raise AssertionError("comparison has no corresponding live states")
    logit_mismatches = [
        key for key in logit_keys if current_logits[key] != reference_logits[key]
    ]
    kv_mismatches = [key for key in kv_keys if current_kv[key] != reference_kv[key]]
    return {
        "reference_budget": reference["budget"],
        "corresponding_logit_states": len(logit_keys),
        "corresponding_kv_states": len(kv_keys),
        "logit_mismatches": logit_mismatches,
        "kv_mismatches": kv_mismatches,
        "passed": not logit_mismatches and not kv_mismatches,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/workspace/models/Qwen3-0.6B")
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = run(args.model, budget=args.budget)
    if args.compare is not None:
        report["comparison"] = compare(
            report, json.loads(args.compare.read_text())
        )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "budget": args.budget,
        "comparison": report.get("comparison"),
    }, sort_keys=True))
    if args.compare is not None and not report["comparison"]["passed"]:
        raise SystemExit("invariant numerical comparison failed")


if __name__ == "__main__":
    main()
