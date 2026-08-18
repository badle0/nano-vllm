"""Machine-readable chunk-tail release classification from retained A100 runs."""

from __future__ import annotations

from copy import deepcopy


MAX_ITL_SLO_MS = 10.0
EVIDENCE_COMMIT = "4a5742ab4003c1ecf7e442a812cbba3e04e06450"
PRODUCTION_BASE_COMMIT = "1ffe033bfc60dab6875634e90ffe816d772ec338"
EVIDENCE_SEEDS = (20260821, 20260822, 20260823, 20260824, 20260825)

_PROFILES = {
    256: {
        "historical_release_role": "latency_slo_met",
        "historical_latency_slo_met": True,
        "historical_recommended_use": "low_latency",
        "run_max_itl_ms": (
            8.057300001382828,
            6.923990324139595,
            7.43560865521431,
            7.575351744890213,
            7.30200856924057,
        ),
        "median_run_max_itl_ms": 7.43560865521431,
        "worst_run_max_itl_ms": 8.057300001382828,
        "runs_meeting_slo": 5,
        "completion_tokens_per_s": (
            3198.0314490999544,
            3173.9533912803627,
            3160.135594670068,
            3155.069639047207,
            3184.1669512508374,
        ),
        "long_ttft_max_ms": (
            113.37962187826633,
            113.04857023060322,
            119.85780857503414,
            122.73521721363068,
            109.33675616979599,
        ),
    },
    512: {
        "historical_release_role": "throughput_ttft_only",
        "historical_latency_slo_met": False,
        "historical_recommended_use": "throughput_or_ttft_not_sub_10ms_itl",
        "run_max_itl_ms": (
            9.75164957344532,
            17.763035371899605,
            9.20167937874794,
            13.002824038267136,
            16.355227679014206,
        ),
        "median_run_max_itl_ms": 13.002824038267136,
        "worst_run_max_itl_ms": 17.763035371899605,
        "runs_meeting_slo": 2,
        "completion_tokens_per_s": (
            3237.64085495712,
            3212.640789651801,
            3215.122157047536,
            3241.9212216789438,
            3177.331322410673,
        ),
        "long_ttft_max_ms": (
            78.51444371044636,
            65.94392657279968,
            76.17606595158577,
            67.95637123286724,
            77.28874497115612,
        ),
    },
}


def chunk_tail_release_profile(tau: int) -> dict[str, object]:
    """Return the evidence-backed role for tau, or an unverified classification."""
    if type(tau) is not int or tau <= 0:
        raise ValueError("tau must be a positive integer")
    profile = deepcopy(_PROFILES.get(tau))
    if profile is None:
        return {
            "tau": tau,
            "classification": "unverified",
            "current_run_classification": "unverified",
            "current_run_latency_certified": False,
            "current_run_recommended_use": "requires_retained_latency_evidence",
            "max_itl_slo_ms_exclusive": MAX_ITL_SLO_MS,
            "matches_historical_recorded_configuration": False,
            "configuration_match_scope": "historical_recorded_fields_only",
            "applies_to_current_run": False,
            "applicability_blockers": ("no retained evidence for tau",),
        }
    profile.update({
        "tau": tau,
        "classification": "historical_reference",
        "current_run_classification": "not_certified_by_reference",
        "current_run_latency_certified": False,
        "applies_to_current_run": False,
        "configuration_match_scope": "historical_recorded_fields_only",
        "max_itl_slo_ms_exclusive": MAX_ITL_SLO_MS,
        "evidence_commit": EVIDENCE_COMMIT,
        "production_base_commit": PRODUCTION_BASE_COMMIT,
        "production_source_relation": (
            "nanovllm source is identical at evidence and production-base commits"
        ),
        "gpu": "NVIDIA A100-SXM4-40GB",
        "torch": "2.10.0+cu128",
        "cuda": "12.8",
        "python_gc_enabled": False,
        "python_gc_mode": "external_manual_disable_before_engine_construction",
        "evidence_integrity": {
            "self_pinned_source_hash": False,
            "model_content_hash_recorded": False,
            "current_run_self_certification_supported": False,
            "limitation": (
                "legacy artifacts record commit and model path, not source/model hashes"
            ),
        },
        "seeds": EVIDENCE_SEEDS,
        "workload": {
            "measurement_protocol": "full_completion_request_metrics",
            "interactive_requests": 16,
            "interactive_prompt_tokens": 64,
            "interactive_steps_before_long_admission": 40,
            "long_requests": 2,
            "long_prompt_tokens": 2048,
            "max_completion_tokens_per_request": 256,
            "temperature": 0.6,
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.8,
            "max_num_seqs": tau,
        },
        "evidence_files": tuple(
            f"roofline_chunk_tip_tau{tau}_seed{seed}_gcdisabled.json"
            for seed in EVIDENCE_SEEDS
        ),
    })
    return profile


def evaluate_chunk_tail_release_profile(
    tau: int,
    current_run: dict[str, object],
) -> dict[str, object]:
    """Attach an exact-match verdict without broadening reference evidence."""
    profile = chunk_tail_release_profile(tau)
    if profile["classification"] == "unverified":
        profile.update({
            "applicability_mismatches": {
                "tau": {"expected": "retained evidence", "actual": tau}
            },
        })
        return profile

    workload = profile["workload"]
    required = {
        "model_resolved_path": "/workspace/models/Qwen3-0.6B",
        "gpu": profile["gpu"],
        "torch": profile["torch"],
        "cuda": profile["cuda"],
        "python_gc_mode": profile["python_gc_mode"],
        "measurement_protocol": workload["measurement_protocol"],
        "interactive_count": workload["interactive_requests"],
        "interactive_prompt_len": workload["interactive_prompt_tokens"],
        "pre_long_steps": workload["interactive_steps_before_long_admission"],
        "long_count": workload["long_requests"],
        "long_prompt_len": workload["long_prompt_tokens"],
        "max_tokens": workload["max_completion_tokens_per_request"],
        "temperature": workload["temperature"],
        "max_model_len": workload["max_model_len"],
        "gpu_memory_utilization": workload["gpu_memory_utilization"],
        "max_num_seqs": workload["max_num_seqs"],
    }
    mismatches = {
        name: {"expected": expected, "actual": current_run.get(name)}
        for name, expected in required.items()
        if current_run.get(name) != expected
    }
    historical_recorded_configuration_matches = not mismatches
    profile.update({
        "matches_historical_recorded_configuration": (
            historical_recorded_configuration_matches
        ),
        # Legacy evidence lacks model-content and source hashes. It can define
        # the release role at its recorded commit but can never self-promote a
        # new run, even when every recorded configuration field matches.
        "applies_to_current_run": False,
        "applicability_blockers": (
            "historical evidence has no model-content hash",
            "historical evidence has no self-pinned source hash",
        ),
        "applicability_required_fields": required,
        "applicability_mismatches": mismatches,
    })
    return profile
