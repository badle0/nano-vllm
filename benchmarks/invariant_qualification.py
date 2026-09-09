"""Forced-history qualification matrix for the invariant numerical backend."""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch

from benchmarks.chunked_prefill_tail.common import model_identity
from nanovllm import LLM, SamplingParams


SEED = 20260908
DEFAULT_SNAPSHOT = Path(
    "/workspace/nano-vllm-review-snapshots/"
    "2026-09-08-pre-risk-review-198e1e6/SHA256SUMS"
)


@dataclass(frozen=True)
class Case:
    label: str
    prompt: tuple[int, ...]
    forced: tuple[int, ...]


def _tokens(seed: int, length: int) -> tuple[int, ...]:
    return tuple(100 + ((seed + index * 7919) % 50000) for index in range(length))


def scenario_cases(name: str) -> tuple[Case, ...]:
    if name == "known_failures":
        return tuple(
            Case(
                f"known-{row}",
                (1000 + row,) * 4,
                (50000 + row, 50100 + row),
            )
            for row in range(17)
        )
    if name == "mixed_boundaries":
        return tuple(
            Case(
                f"length-{length}",
                _tokens(1000 + index * 101, length),
                (51000 + index, 51100 + index),
            )
            for index, length in enumerate((1, 255, 256, 257, 511, 512, 513))
        )
    if name == "cache_reuse":
        prompt = _tokens(7000, 513)
        return (
            Case("cache-seed", prompt, (52001,)),
            Case("cache-reuse", prompt, (52001,)),
        )
    if name == "eviction_resume":
        return (Case("eviction-resume", _tokens(11000, 769), (53001, 53002)),)
    if name == "long_context":
        return tuple(
            Case(
                f"length-{length}",
                _tokens(17000 + index * 313, length),
                (54000 + index,),
            )
            for index, length in enumerate((1024, 2048, 4096))
        )
    raise ValueError(f"unknown scenario: {name}")


def _tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _implementation_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "nanovllm").rglob("*.py")):
        filename = str(path.relative_to(root))
        digest.update(filename.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _snapshot_identity(path: Path | None) -> dict | None:
    if path is None:
        return None
    raw = path.read_bytes()
    entries = {}
    for line in raw.decode().splitlines():
        digest, filename = line.split(maxsplit=1)
        entries[Path(filename).name] = digest
    return {
        "manifest": str(path),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "head": entries.get("HEAD.txt"),
        "source_archive": entries.get("source.tar.gz"),
        "tracked_patch": entries.get("tracked-working-tree.patch"),
        "source_manifest": entries.get("source-sha256sums.txt"),
    }


def _state_consistency(traces: list[dict]) -> dict:
    grouped: dict[tuple[int, str], list[dict]] = {}
    for trace in traces:
        grouped.setdefault(
            (trace["processed_tokens"], trace["prefix_sha256"]), []
        ).append(trace)
    mismatches = []
    comparable = 0
    for key, rows in grouped.items():
        if len(rows) < 2:
            continue
        comparable += 1
        kv = {row["kv_sha256"] for row in rows}
        logits = {
            row["logits_sha256"] for row in rows if "logits_sha256" in row
        }
        if len(kv) != 1 or len(logits) > 1:
            mismatches.append(
                {
                    "state": key,
                    "labels": [row["label"] for row in rows],
                    "kv_hashes": sorted(kv),
                    "logit_hashes": sorted(logits),
                }
            )
    return {
        "comparable_repeated_states": comparable,
        "mismatches": mismatches,
        "passed": not mismatches,
    }


def run(
    model_path: str,
    *,
    scenario: str,
    budget: int,
    layout: str,
    inject_eviction: bool,
    snapshot_manifest: Path | None,
) -> dict:
    if not __debug__:
        raise RuntimeError("numerical qualification requires assertions enabled")
    cases = scenario_cases(scenario)
    if layout not in {"packed", "serial"}:
        raise ValueError("layout must be packed or serial")
    if inject_eviction and scenario != "eviction_resume":
        raise ValueError("eviction injection is valid only for eviction_resume")

    torch.manual_seed(SEED)
    llm = LLM(
        model_path,
        numerical_mode="invariant",
        enforce_eager=True,
        max_num_batched_tokens=budget,
        max_num_seqs=len(cases) if layout == "packed" else 1,
        max_model_len=4096,
        num_kvcache_blocks=96,
        gpu_memory_utilization=0.5,
    )
    traces: list[dict] = []
    outputs = {case.label: [] for case in cases}
    admission_cache_tokens: dict[str, int] = {}
    eviction_events = []
    labels_by_seq_id: dict[int, str] = {}
    cases_by_label = {case.label: case for case in cases}
    active_rows = []
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
                "label": labels_by_seq_id[seq.seq_id],
                "processed_tokens": processed,
                "cached_before": seq.num_cached_tokens,
                "scheduled_tokens": seq.num_scheduled_tokens,
                "prefix_sha256": hashlib.sha256(repr(prefix).encode()).hexdigest(),
                "kv_sha256": _tensor_sha256(live_kv),
                "emits": emits,
                "physical_blocks": list(seq.block_table),
            }
            if emits:
                row_logits = logits[emission_row]
                top_values, top_ids = torch.topk(row_logits.float(), k=3)
                record.update(
                    {
                        "logits_sha256": _tensor_sha256(row_logits),
                        "argmax": int(row_logits.argmax().item()),
                        "top_token_ids": [int(value) for value in top_ids.tolist()],
                        "top_logits": [float(value) for value in top_values.tolist()],
                    }
                )
                emission_row += 1
            traces.append(record)
        if emission_row != logits.size(0):
            raise RuntimeError("emission-row mapping disagrees with logits")
        return logits

    llm.model_runner.run_model = traced_run_model

    def drain(selected_cases: tuple[Case, ...]):
        nonlocal active_rows
        seqs = []
        for case in selected_cases:
            seq = llm._admit_request(
                list(case.prompt),
                SamplingParams(
                    temperature=0.0,
                    max_tokens=len(case.forced),
                    ignore_eos=True,
                ),
            )
            labels_by_seq_id[seq.seq_id] = case.label
            seqs.append(seq)

        evicted = False
        while not all(seq.is_finished for seq in seqs):
            scheduled, is_prefill = llm.scheduler.schedule()
            active_rows = scheduled
            for seq in scheduled:
                label = labels_by_seq_id[seq.seq_id]
                admission_cache_tokens.setdefault(label, seq.num_cached_tokens)
            computed = llm.model_runner.call("run", scheduled, is_prefill)
            forced = list(computed)
            for row_index, seq in enumerate(scheduled):
                processed = seq.num_cached_tokens + seq.num_scheduled_tokens
                if processed == seq.num_tokens:
                    label = labels_by_seq_id[seq.seq_id]
                    forced[row_index] = cases_by_label[label].forced[
                        seq.num_completion_tokens
                    ]
            events = llm.scheduler.postprocess(scheduled, forced)
            for event in events:
                outputs[labels_by_seq_id[event.seq_id]].append(event.token_id)

            if inject_eviction and not evicted:
                seq = seqs[0]
                if (
                    seq in llm.scheduler.waiting
                    and 0 < seq.num_cached_tokens < seq.num_tokens
                ):
                    if llm.scheduler.waiting[0] is not seq:
                        raise RuntimeError("eviction witness lost FIFO ownership")
                    llm.scheduler.waiting.popleft()
                    before = {
                        "cached_tokens": seq.num_cached_tokens,
                        "block_table": list(seq.block_table),
                    }
                    llm.scheduler.preempt(seq)
                    eviction_events.append(
                        {
                            **before,
                            "cached_tokens_after": seq.num_cached_tokens,
                            "block_table_after": list(seq.block_table),
                        }
                    )
                    evicted = True
        if inject_eviction and not evicted:
            raise RuntimeError("requested eviction was not exercised")

    try:
        if layout == "packed":
            drain(cases)
        else:
            for case in cases:
                drain((case,))
        root = Path(__file__).resolve().parents[1]
        repeated = _state_consistency(traces)
        known = {
            label: next(
                (
                    {
                        "argmax": trace["argmax"],
                        "top_token_ids": trace["top_token_ids"],
                        "top_logits": trace["top_logits"],
                    }
                    for trace in traces
                    if trace["label"] == label and trace["emits"]
                ),
                None,
            )
            for label in ("known-9", "known-12")
            if label in cases_by_label
        }
        return {
            "kind": "invariant_forced_history_qualification_v2",
            "model": model_identity(Path(model_path)),
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "numerical_mode": "invariant",
            "scenario": scenario,
            "budget": budget,
            "layout": layout,
            "inject_eviction": inject_eviction,
            "cases": [
                {
                    "label": case.label,
                    "prompt": list(case.prompt),
                    "forced": list(case.forced),
                }
                for case in cases
            ],
            "outputs": outputs,
            "admission_cache_tokens": admission_cache_tokens,
            "eviction_events": eviction_events,
            "known_failure_observations": known,
            "implementation_sha256": _implementation_sha256(root),
            "pre_review_snapshot": _snapshot_identity(snapshot_manifest),
            "internal_repeated_state_check": repeated,
            "traces": traces,
        }
    finally:
        llm.exit()


def compare(current: dict, reference: dict) -> dict:
    for field in (
        "kind",
        "model",
        "numerical_mode",
        "scenario",
        "cases",
        "outputs",
        "implementation_sha256",
        "pre_review_snapshot",
    ):
        if current[field] != reference[field]:
            raise AssertionError(f"comparison identity differs at {field}")

    def keyed(report, field):
        return {
            (
                item["label"],
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
    required_labels = {case["label"] for case in current["cases"]}
    emission_labels = {key[0] for key in logit_keys}
    if emission_labels != required_labels:
        raise AssertionError("comparison lacks a final emission for one or more cases")
    logit_mismatches = [
        key for key in logit_keys if current_logits[key] != reference_logits[key]
    ]
    kv_mismatches = [
        key for key in kv_keys if current_kv[key] != reference_kv[key]
    ]
    internal_passed = (
        current["internal_repeated_state_check"]["passed"]
        and reference["internal_repeated_state_check"]["passed"]
    )
    return {
        "reference_budget": reference["budget"],
        "reference_layout": reference["layout"],
        "reference_inject_eviction": reference["inject_eviction"],
        "corresponding_logit_states": len(logit_keys),
        "corresponding_kv_states": len(kv_keys),
        "logit_mismatches": logit_mismatches,
        "kv_mismatches": kv_mismatches,
        "required_emission_labels": sorted(required_labels),
        "internal_repeated_states_passed": internal_passed,
        "passed": not logit_mismatches and not kv_mismatches and internal_passed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/workspace/models/Qwen3-0.6B")
    parser.add_argument(
        "--scenario",
        required=True,
        choices=(
            "known_failures",
            "mixed_boundaries",
            "cache_reuse",
            "eviction_resume",
            "long_context",
        ),
    )
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--layout", choices=("packed", "serial"), required=True)
    parser.add_argument("--inject-eviction", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument(
        "--snapshot-manifest",
        type=Path,
        default=DEFAULT_SNAPSHOT if DEFAULT_SNAPSHOT.exists() else None,
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = run(
        args.model,
        scenario=args.scenario,
        budget=args.budget,
        layout=args.layout,
        inject_eviction=args.inject_eviction,
        snapshot_manifest=args.snapshot_manifest,
    )
    if args.compare is not None:
        report["comparison"] = compare(
            report, json.loads(args.compare.read_text())
        )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    summary = {
        "output": str(args.output),
        "scenario": args.scenario,
        "budget": args.budget,
        "layout": args.layout,
        "cache_tokens": report["admission_cache_tokens"],
        "evictions": len(report["eviction_events"]),
        "known": report["known_failure_observations"],
        "comparison": report.get("comparison"),
    }
    print(json.dumps(summary, sort_keys=True))
    if args.compare is not None and not report["comparison"]["passed"]:
        raise SystemExit("invariant numerical qualification failed")


if __name__ == "__main__":
    main()

