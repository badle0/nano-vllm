from types import SimpleNamespace

import pytest

from nanovllm.engine.speculative_policy import AdaptiveSpeculativePolicy


class Row(SimpleNamespace):
    def __len__(self):
        return self.length


def row(*, temperature=0.7, top_k=-1, top_p=1.0, length=64):
    return Row(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        length=length,
        max_tokens=32,
        num_completion_tokens=0,
    )


def choose(policy, rows, *, mode="invariant"):
    return policy.choose(
        rows,
        candidate_ks=(1, 2, 3, 4),
        catchup_tokens=0,
        numerical_mode=mode,
        max_model_len=4096,
        max_num_batched_tokens=1024,
    )


def cell(k, *, cycle_ms, commits):
    return {
        "batch_size": 1,
        "sampling_family": "temperature",
        "context_bucket": 64,
        "catchup_bucket": 0,
        "k": k,
        "ordinary_ms_per_token": 10.0,
        "cycle_ms": cycle_ms,
        "expected_committed_tokens": commits,
    }


def test_adaptive_policy_bypasses_uncalibrated_and_fast_greedy_routes():
    policy = AdaptiveSpeculativePolicy()
    assert choose(policy, [row()]).reason == "uncalibrated_route"
    decision = choose(policy, [row(temperature=0.0)], mode="fast")
    assert decision.selected_k is None
    assert decision.reason == "fast_greedy_sequential_verifier"
    metrics = policy.snapshot()
    assert metrics["decision_counts"] == {
        "bypass_fast_greedy_sequential_verifier": 1,
        "bypass_uncalibrated_route": 1,
    }


def test_adaptive_policy_requires_and_selects_a_ten_percent_win():
    policy = AdaptiveSpeculativePolicy()
    policy.load([
        cell(1, cycle_ms=19.0, commits=2.0),
        cell(2, cycle_ms=25.0, commits=3.0),
        cell(3, cycle_ms=39.0, commits=4.0),
    ])
    decision = choose(policy, [row()])
    assert decision.selected_k == 2
    assert decision.reason == "selected"
    assert decision.predicted_speedup == pytest.approx(1.2)

    losing = AdaptiveSpeculativePolicy()
    losing.load([cell(1, cycle_ms=20.0, commits=2.0)])
    decision = choose(losing, [row()])
    assert decision.selected_k is None
    assert decision.reason == "below_10_percent_gate"


def test_adaptive_policy_rejects_duplicate_or_impossible_calibration():
    policy = AdaptiveSpeculativePolicy()
    with pytest.raises(ValueError, match="duplicate"):
        policy.load([cell(1, cycle_ms=10.0, commits=2.0)] * 2)
    invalid = cell(1, cycle_ms=10.0, commits=3.0)
    with pytest.raises(ValueError, match="K\\+1"):
        policy.load([invalid])
