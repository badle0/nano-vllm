#!/usr/bin/env python3
"""Validate fast top-p raw evidence, protocol pins, and derived results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = Path(__file__).with_name("provenance.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_repo_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if path != ROOT and ROOT not in path.parents:
        raise SystemExit(f"manifest path escapes repository: {relative}")
    return path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def require_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise SystemExit(f"{label}: {actual} != {expected}")


def validate_hash(path: Path, expected: str) -> None:
    actual = sha256(path)
    require(actual == expected, f"SHA-256 mismatch for {path}: {actual}")


def nearest_rank(samples: list[float], quantile: float) -> float:
    ordered = sorted(samples)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def regenerated_prompt_sha256(
    batch: int,
    prompt_len: int,
    prompt_seed: int,
    repetition: int,
    vocab_size: int,
) -> str:
    rng = random.Random(prompt_seed)
    digest = hashlib.sha256()
    for row in range(batch):
        prompt = [repetition * batch + row]
        prompt.extend(rng.randrange(vocab_size) for _ in range(prompt_len - 1))
        digest.update(len(prompt).to_bytes(4, "little"))
        for token in prompt:
            digest.update(int(token).to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def stripped_backend_config(config: dict[str, object]) -> dict[str, object]:
    result = dict(config)
    result.pop("effective_top_p_backend", None)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-model",
        action="store_true",
        help="also hash the pinned local model files",
    )
    args = parser.parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text())

    harness_hashes = {}
    for harness in manifest["harnesses"]:
        path = checked_repo_path(harness["path"])
        validate_hash(path, harness["sha256"])
        harness_hashes[harness["path"]] = harness["sha256"]

    micro_artifact = manifest["micro_artifact"]
    micro_path = checked_repo_path(micro_artifact["path"])
    validate_hash(micro_path, micro_artifact["sha256"])
    micro = json.loads(micro_path.read_text())
    require(micro["git_status"] == "", "micro run was not from a clean checkout")
    require(micro["commit"] == micro_artifact["commit"], "micro commit mismatch")
    require(
        micro["source"]["sha256"]
        == harness_hashes[micro["source"]["path"]],
        "micro embedded source hash mismatch",
    )
    require(
        micro["configuration"] == manifest["protocol"]["micro"],
        "micro configuration mismatch",
    )
    require(
        len(micro["current_exact_complete"]["samples_cuda_ms"]) == 25,
        "micro exact sample count mismatch",
    )
    require(
        len(micro["flashinfer_production_complete"]["samples_cuda_ms"]) == 25,
        "micro production sample count mismatch",
    )
    micro_expected = manifest["release_results"]["micro"]
    for route, fields in micro_expected["routes"].items():
        samples = micro[route]["samples_cuda_ms"]
        require_close(
            statistics.median(samples),
            micro[route]["median_cuda_ms"],
            f"micro recomputed {route}.median_cuda_ms",
        )
        require_close(
            nearest_rank(samples, 0.95),
            micro[route]["p95_cuda_ms"],
            f"micro recomputed {route}.p95_cuda_ms",
        )
        for field, expected in fields.items():
            require_close(micro[route][field], expected, f"micro {route}.{field}")
    require(
        micro["flashinfer_production_complete"]["median_cuda_ms"]
        <= micro_expected["acceptance"]["maximum_median_cuda_ms"],
        "production wrapper failed its median latency gate",
    )
    maximum_production_sample = max(
        micro["flashinfer_production_complete"]["samples_cuda_ms"]
    )
    require_close(
        maximum_production_sample,
        micro_expected["acceptance"]["maximum_observed_cuda_ms"],
        "micro maximum production sample",
    )
    require(
        maximum_production_sample
        <= micro_expected["acceptance"]["maximum_allowed_observed_cuda_ms"],
        "production wrapper had a sample above the absolute latency gate",
    )
    require_close(
        micro["speedup_exact_over_flashinfer_production"],
        micro_expected["exact_over_production_speedup"],
        "micro exact/production speedup",
    )
    deterministic = micro["determinism_and_rng_contract"]
    require(
        deterministic["flashinfer_same_seed_tokens_repeat"]
        and deterministic["flashinfer_same_seed_rng_state_repeat"],
        "FlashInfer same-seed repeatability failed",
    )
    require(
        deterministic["fixed_seed_differing_tokens"] > 0
        and not deterministic["same_rng_state_after_routes"],
        "expected exact/FlashInfer contract divergence was not recorded",
    )
    require(
        micro["statistical_semantic_characterization"]["known_unique_logits"][
            "sanity_pass"
        ],
        "unique-logit statistical sanity check failed",
    )

    expected_environment = manifest["environment"]["e2e_embedded"]
    expected_model = manifest["model"]["embedded_fingerprint"]
    expected_e2e_config = manifest["protocol"]["e2e"]["configuration"]
    e2e_by_pair: dict[int, dict[str, dict[str, object]]] = defaultdict(dict)
    chronological = []
    cache_dirs = set()
    for artifact in manifest["e2e_artifacts"]:
        path = checked_repo_path(artifact["path"])
        validate_hash(path, artifact["sha256"])
        result = json.loads(path.read_text())
        require(result["benchmark"] == "topp_backend_e2e", f"tag mismatch: {path}")
        require(result["git_status"] == "", f"dirty checkout: {path}")
        require(
            result["commit"] == manifest["code_under_test"]["e2e_commit"],
            f"commit mismatch: {path}",
        )
        require(result["pair_id"] == artifact["pair_id"], f"pair mismatch: {path}")
        require(result["backend"] == artifact["backend"], f"backend mismatch: {path}")
        require(result["pair_order"] == artifact["pair_order"], f"order mismatch: {path}")
        require(
            result["position_in_pair"] == artifact["position"],
            f"position mismatch: {path}",
        )
        require(result["seed"] == artifact["seed"], f"seed mismatch: {path}")
        order = result["pair_order"].split("-")
        expected_backend = order[
            0 if result["position_in_pair"] == "first" else 1
        ]
        require(
            expected_backend == result["backend"],
            f"logical order/position mismatch: {path}",
        )
        require(result["environment"] == expected_environment, f"environment mismatch: {path}")
        require(result["model_fingerprint"] == expected_model, f"model mismatch: {path}")
        require(
            result["source"]["sha256"]
            == harness_hashes[result["source"]["path"]],
            f"source hash mismatch: {path}",
        )
        require(
            stripped_backend_config(result["configuration"])
            == expected_e2e_config,
            f"configuration mismatch: {path}",
        )
        require(
            result["configuration"]["effective_top_p_backend"] == result["backend"],
            f"effective backend mismatch: {path}",
        )
        require(
            result["configuration"]["preemption_excluded_by_capacity"],
            f"preemption was not excluded: {path}",
        )
        observations = result["observations"]
        require(len(observations) == 6, f"observation count mismatch: {path}")
        require(result["steady"] == observations[1:], f"steady slice mismatch: {path}")
        prompt_hashes = [row["prompt_sha256"] for row in observations]
        require(len(set(prompt_hashes)) == 6, f"prompt reuse within process: {path}")
        batch = result["configuration"]["batch"]
        prompt_len = result["configuration"]["prompt_tokens"]
        output_len = result["configuration"]["output_tokens"]
        vocab_size = result["configuration"]["model_vocab_size"]
        for repetition, observation in enumerate(observations):
            require(
                observation["repetition"] == repetition,
                f"repetition index mismatch: {path}",
            )
            expected_prompt_seed = result["seed"] * 10_000 + repetition
            expected_run_seed = result["seed"] * 1000 + batch * 10 + repetition
            require(
                observation["prompt_seed"] == expected_prompt_seed
                and observation["seed"] == expected_run_seed,
                f"derived seed mismatch: {path}, observation {repetition}",
            )
            require(
                observation["prompt_sha256"]
                == regenerated_prompt_sha256(
                    batch,
                    prompt_len,
                    expected_prompt_seed,
                    repetition,
                    vocab_size,
                ),
                f"regenerated prompt mismatch: {path}, observation {repetition}",
            )
            require_close(
                observation["output_tokens_per_s"],
                batch * output_len / observation["elapsed_s"],
                f"recomputed throughput: {path}, observation {repetition}",
            )
        require_close(
            result["median_steady_elapsed_s"],
            statistics.median(row["elapsed_s"] for row in observations[1:]),
            f"recomputed steady elapsed median: {path}",
        )
        require_close(
            result["median_steady_output_tokens_per_s"],
            statistics.median(
                row["output_tokens_per_s"] for row in observations[1:]
            ),
            f"recomputed steady throughput median: {path}",
        )
        cache_dir = result["execution"]["torchinductor_cache_dir"]
        require(cache_dir and cache_dir not in cache_dirs, f"reused Inductor cache: {path}")
        cache_dirs.add(cache_dir)
        if result["backend"] == "flashinfer":
            require(result["flashinfer"]["version"] == "0.6.17", f"FlashInfer mismatch: {path}")
        else:
            require(result["flashinfer"] is None, f"exact process imported fast backend: {path}")
        e2e_by_pair[result["pair_id"]][result["backend"]] = result
        chronological.append((result["started_at_utc"], result["run_label"]))

    require(set(e2e_by_pair) == set(range(1, 9)), "E2E pair IDs must be 1..8")
    expected_chronology = manifest["protocol"]["e2e"]["chronological_run_labels"]
    require(
        [label for _, label in sorted(chronological)] == expected_chronology,
        "E2E process chronology differs from the declared protocol",
    )
    time_ordered = sorted(
        (
            datetime.fromisoformat(result["started_at_utc"]),
            datetime.fromisoformat(result["finished_at_utc"]),
            result["run_label"],
        )
        for pair in e2e_by_pair.values()
        for result in pair.values()
    )
    for (_, previous_finish, previous_label), (next_start, _, next_label) in zip(
        time_ordered, time_ordered[1:]
    ):
        require(
            previous_finish <= next_start,
            f"overlapping E2E processes: {previous_label} and {next_label}",
        )

    ratios = []
    order_ratios: dict[str, list[float]] = defaultdict(list)
    for pair_id in range(1, 9):
        pair = e2e_by_pair[pair_id]
        require(set(pair) == {"exact", "flashinfer"}, f"incomplete pair {pair_id}")
        exact, fast = pair["exact"], pair["flashinfer"]
        require(exact["seed"] == fast["seed"], f"pair seed mismatch: {pair_id}")
        require(exact["pair_order"] == fast["pair_order"], f"pair order mismatch: {pair_id}")
        for index, (exact_row, fast_row) in enumerate(
            zip(exact["observations"], fast["observations"])
        ):
            require(
                exact_row["prompt_sha256"] == fast_row["prompt_sha256"]
                and exact_row["prompt_seed"] == fast_row["prompt_seed"],
                f"prompt mismatch in pair {pair_id}, observation {index}",
            )
            require(
                exact_row["cuda_rng_state_before_sha256"]
                == fast_row["cuda_rng_state_before_sha256"],
                f"pre-run RNG mismatch in pair {pair_id}, observation {index}",
            )
        ratio = (
            fast["median_steady_output_tokens_per_s"]
            / exact["median_steady_output_tokens_per_s"]
        )
        ratios.append(ratio)
        order_ratios[exact["pair_order"]].append(ratio)
        require_close(
            ratio,
            manifest["release_results"]["e2e"]["pair_ratios"][pair_id - 1],
            f"pair {pair_id} ratio",
        )

    for left, right in ((1, 2), (3, 4), (5, 6), (7, 8)):
        for backend in ("exact", "flashinfer"):
            first = e2e_by_pair[left][backend]
            second = e2e_by_pair[right][backend]
            require(first["seed"] == second["seed"], f"mirror seed mismatch: {left}/{right}")
            for index, (first_row, second_row) in enumerate(
                zip(first["observations"], second["observations"])
            ):
                require(
                    first_row["prompt_sha256"] == second_row["prompt_sha256"]
                    and first_row["token_sha256"] == second_row["token_sha256"]
                    and first_row["cuda_rng_state_after_sha256"]
                    == second_row["cuda_rng_state_after_sha256"],
                    f"fresh-process determinism mismatch: {left}/{right} "
                    f"{backend} observation {index}",
                )

    expected_results = manifest["release_results"]["e2e"]
    derived = {
        "median_pair_ratio": statistics.median(ratios),
        "median_pair_gain_percent": (statistics.median(ratios) - 1.0) * 100.0,
        "mean_pair_ratio": statistics.mean(ratios),
        "mean_pair_gain_percent": (statistics.mean(ratios) - 1.0) * 100.0,
        "sample_sd_pair_ratio": statistics.stdev(ratios),
        "minimum_pair_ratio": min(ratios),
        "maximum_pair_ratio": max(ratios),
        "exact_first_median_ratio": statistics.median(order_ratios["exact-flashinfer"]),
        "flashinfer_first_median_ratio": statistics.median(order_ratios["flashinfer-exact"]),
        "exact_process_median_tps": statistics.median(
            pair["exact"]["median_steady_output_tokens_per_s"]
            for pair in e2e_by_pair.values()
        ),
        "flashinfer_process_median_tps": statistics.median(
            pair["flashinfer"]["median_steady_output_tokens_per_s"]
            for pair in e2e_by_pair.values()
        ),
    }
    t_critical = expected_results["mean_ratio_t95"]["t_critical_df7"]
    half_width = t_critical * derived["sample_sd_pair_ratio"] / math.sqrt(len(ratios))
    derived["mean_ratio_t95_low"] = derived["mean_pair_ratio"] - half_width
    derived["mean_ratio_t95_high"] = derived["mean_pair_ratio"] + half_width
    derived["ratio_of_process_median_throughputs"] = (
        derived["flashinfer_process_median_tps"]
        / derived["exact_process_median_tps"]
    )
    for field, actual in derived.items():
        expected = expected_results["derived"][field]
        require_close(actual, expected, f"E2E {field}")
    require(
        min(ratios) > 1.0 and statistics.median(ratios) > 1.0,
        "not every fresh-process pair improved throughput",
    )

    model_status = "model hashes skipped"
    if args.check_model:
        model_root = Path(expected_model["resolved_path"])
        for item in expected_model["files"]:
            path = model_root / item["path"]
            require(path.stat().st_size == item["bytes"], f"model size mismatch: {path}")
            validate_hash(path, item["sha256"])
        model_status = f"{len(expected_model['files'])} model files"

    print(
        "validated 1 micro artifact, 16 E2E raw files, "
        f"8 fresh-process pairs, 2 harnesses, and {model_status}"
    )


if __name__ == "__main__":
    main()
