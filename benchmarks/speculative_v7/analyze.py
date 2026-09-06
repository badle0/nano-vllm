#!/usr/bin/env python3
"""Recompute paired V7 results from retained, uninstrumented raw samples."""
import argparse
import itertools
import json
from pathlib import Path
from statistics import median


def key(row):
    return tuple(row[name] for name in ("batch", "context", "family", "consumer", "completion"))


def interval(values):
    # Exact nonparametric bootstrap over independent process-pair summaries.
    # Five pairs -> 5**5 resamples. Seeds are NOT independent replicates.
    samples = sorted(median(sample) for sample in itertools.product(values, repeat=len(values)))
    return [samples[int((len(samples) - 1) * p)] for p in (.025, .975)]


def paired(runs, prefix, old="off", new="on"):
    cells = sorted({key(row) for run in runs.values() if run["args"]["suite"] in ("primary", "off-regression")
                    for row in run["records"]})
    results = []
    for cell in cells:
        ratios, old_seconds, new_seconds = [], [], []
        selected_pairs = []
        for pair in range(6 if prefix == "primary" else 5):
            a = {r["seed"]: r for r in runs[f"{prefix}-p{pair}-{old}"]["records"] if key(r) == cell}
            b = {r["seed"]: r for r in runs[f"{prefix}-p{pair}-{new}"]["records"] if key(r) == cell}
            if not a and not b:
                continue
            seeds = sorted(set(a) & set(b))
            accepted = [s for s in seeds if a[s]["accepted"] and b[s]["accepted"]]
            if len(accepted) != 3:
                continue  # no headline for an incomplete process pair
            ratios.append(median(a[s]["seconds"] / b[s]["seconds"] for s in accepted))
            selected_pairs.append(pair)
            old_seconds.extend(a[s]["seconds"] for s in accepted)
            new_seconds.extend(b[s]["seconds"] for s in accepted)
            if len(ratios) == 5:
                break
        if not ratios:
            continue
        results.append(dict(cell=list(cell), pairs=len(ratios), selected_pairs=selected_pairs, pair_ratios=ratios,
                            speed_ratio=median(ratios), bootstrap_95=interval(ratios),
                            control_seconds=median(old_seconds), candidate_seconds=median(new_seconds)))
    return results


def analyze(archive):
    manifest = json.loads((archive / "manifest.json").read_text())
    runs = {role: json.loads((archive / record["json"]["file"]).read_text())
            for role, record in manifest["runs"].items() if role.startswith(("primary-", "regression-", "extended-", "cap"))}
    primary = paired(runs, "primary")
    regression = paired(runs, "regression", "old", "new")
    work = []
    for result in primary:
        cell = tuple(result["cell"])
        rows = [row for name, run in runs.items() if name.startswith("primary-") and name.endswith("-on")
                for row in run["records"] if key(row) == cell and row["accepted"]]
        totals = {}
        itls, ttfts, caller_e2e = [], [], []
        for row in rows:
            for output in row["outputs"]:
                metrics = output["metrics"]
                for name, value in metrics.items():
                    if name.startswith("spec_"):
                        totals[name] = totals.get(name, 0) + value
                itls.extend(metrics["engine_itls"])
                ttfts.append(metrics["engine_ttft"])
                caller_e2e.append(metrics["caller_e2e"])
        cycles = totals.get("spec_cycles", 0)
        proposed = totals.get("spec_proposed_draft_tokens", 0)
        work.append(dict(cell=list(cell), totals=totals,
                         accepted_prefix_fraction=totals.get("spec_accepted_draft_tokens", 0) / proposed if proposed else None,
                         emitted_per_row_cycle=totals.get("spec_committed_tokens", 0) / cycles if cycles else None,
                         median_engine_itl=median(itls), median_engine_ttft=median(ttfts),
                         median_caller_e2e=median(caller_e2e)))
    calibrations = [run["calibration"] for name, run in runs.items() if name.startswith("primary-")]
    model = runs["primary-p0-on"]
    ledger = json.loads((archive / "phases-graph.json").read_text())["weight_ledger"]
    cfg = model["target_config"]
    kv_per_token = 2 * cfg["num_hidden_layers"] * cfg["num_key_value_heads"] * cfg["head_dim"] * 2
    bandwidth = median(c["bandwidth_bytes_per_second"] for c in calibrations)
    compute = median(c["bf16_flops_per_second"] for c in calibrations)
    roofs = dict(bandwidth_bytes_per_second=bandwidth, bf16_flops_per_second=compute,
                 parameter_objects_count=model["target_parameter_count"], weight_ledger=ledger,
                 target_weight_bytes=ledger["target"]["unique_storage_bytes"],
                 draft_weight_bytes=ledger["draft"]["unique_storage_bytes"], target_kv_bytes_per_token=kv_per_token,
                 target_weight_stream_ms=1000 * ledger["target"]["unique_storage_bytes"] / bandwidth,
                 draft_weight_stream_ms=1000 * ledger["draft"]["unique_storage_bytes"] / bandwidth,
                 probability_bytes_b4_k4=4 * (4 + 4 + 1) * cfg["vocab_size"] * 4,
                 target_dense_flops_per_query=2 * ledger["target"]["linear_weight_elements"])
    phases = []
    for mode in ("graph", "eager"):
        value = json.loads((archive / f"phases-{mode}.json").read_text())
        for cell in value["cells"]:
            cycles = [c for c in value["cycles"] if c["cell"] == cell["cell"]]
            totals = {p: sum(sum(c["phases"].get(p, [])) for c in cycles) for p in cycles[0]["phases"]}
            elapsed = sum(c["total_seconds"] for c in cycles)
            components = {k: v for k, v in totals.items() if k != "run_speculative"}
            histogram = {str(i): sum(row["accepted_draft_tokens"] == i for c in cycles for row in c["result"]["rows"]) for i in range(5)}
            phases.append(dict(mode=mode, cell=cell["cell"], cycles=len(cycles),
                               seconds=elapsed, component_seconds=components,
                               other_seconds=elapsed - sum(components.values()), acceptance_histogram=histogram,
                               emitted_per_row_cycle=sum(len(r["committed_token_ids"]) for c in cycles for r in c["result"]["rows"]) / sum(c["batch"] for c in cycles)))
    return dict(primary=primary, off_regression=regression, work=work, roofline=roofs,
                phases=phases,
                samples=sum(len(run["records"]) for run in runs.values()),
                rejected_samples=sum(not row["accepted"] for run in runs.values() for row in run["records"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    # Validation is mandatory before reporting numbers from an archive.
    from validate_retained_evidence import validate
    validate(args.archive)
    print(json.dumps(analyze(args.archive), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
