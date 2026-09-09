"""Host-only performance gate for adaptive speculative decoding."""

from collections import Counter
from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Mapping


MIN_PREDICTED_SPEEDUP = 1.10


@dataclass(frozen=True, slots=True)
class CalibrationCell:
    batch_size: int
    sampling_family: str
    context_bucket: int
    catchup_bucket: int
    k: int
    ordinary_ms_per_token: float
    cycle_ms: float
    expected_committed_tokens: float

    def __post_init__(self):
        for name in ("batch_size", "context_bucket", "catchup_bucket", "k"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.batch_size < 1 or self.context_bucket < 1 or self.k < 1:
            raise ValueError("batch_size, context_bucket, and k must be positive")
        if not isinstance(self.sampling_family, str) or not self.sampling_family:
            raise ValueError("sampling_family must be a non-empty string")
        for name in (
            "ordinary_ms_per_token",
            "cycle_ms",
            "expected_committed_tokens",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.expected_committed_tokens > self.k + 1:
            raise ValueError("expected committed tokens cannot exceed K+1")

    @property
    def key(self):
        return (
            self.batch_size,
            self.sampling_family,
            self.context_bucket,
            self.catchup_bucket,
            self.k,
        )


@dataclass(frozen=True, slots=True)
class AdaptiveDecision:
    selected_k: int | None
    reason: str
    predicted_speedup: float | None
    calibration_key: tuple | None


class AdaptiveSpeculativePolicy:
    """Select only independently calibrated routes predicted to win by 10%."""

    def __init__(self):
        self._cells: dict[tuple, CalibrationCell] = {}
        self._accepted_prefix_ema: dict[tuple, float] = {}
        self._decision_counts = Counter()
        self._last_decision: AdaptiveDecision | None = None

    @staticmethod
    def _power_of_two_bucket(value: int) -> int:
        return 0 if value <= 0 else 1 << (value - 1).bit_length()

    @staticmethod
    def _sampling_family(rows) -> str:
        if all(row.temperature == 0.0 for row in rows):
            return "greedy"
        families = {
            "top_k_top_p" if row.top_k != -1 and row.top_p != 1.0
            else "top_k" if row.top_k != -1
            else "top_p" if row.top_p != 1.0
            else "temperature"
            for row in rows
            if row.temperature != 0.0
        }
        return next(iter(families)) if len(families) == 1 else "mixed"

    def load(self, cells: Iterable[CalibrationCell | Mapping]) -> None:
        loaded = {}
        for value in cells:
            cell = value if isinstance(value, CalibrationCell) else CalibrationCell(**value)
            if cell.key in loaded:
                raise ValueError("duplicate speculative calibration cell")
            loaded[cell.key] = cell
        if not loaded:
            raise ValueError("adaptive calibration must contain at least one cell")
        self._cells = loaded
        self._accepted_prefix_ema.clear()
        self._decision_counts.clear()
        self._last_decision = None

    def choose(
        self,
        rows,
        *,
        candidate_ks,
        catchup_tokens: int,
        numerical_mode: str,
        max_model_len: int,
        max_num_batched_tokens: int,
    ) -> AdaptiveDecision:
        rows = tuple(rows)
        candidate_ks = tuple(sorted(set(candidate_ks)))
        if not rows or not candidate_ks:
            return self._record(AdaptiveDecision(None, "route_unavailable", None, None))
        family = self._sampling_family(rows)
        if family == "greedy" and numerical_mode != "invariant":
            return self._record(
                AdaptiveDecision(None, "fast_greedy_sequential_verifier", None, None)
            )

        request_cap = min(
            max(row.max_tokens - row.num_completion_tokens - 1, 0)
            for row in rows
        )
        model_cap = min(max(max_model_len - len(row), 0) for row in rows)
        budget_cap = max(
            (max_num_batched_tokens - catchup_tokens) // len(rows) - 1,
            0,
        )
        legal_cap = min(max(candidate_ks), request_cap, model_cap, budget_cap)
        legal = tuple(k for k in candidate_ks if k <= legal_cap)
        if not legal:
            return self._record(AdaptiveDecision(None, "tail_or_budget", None, None))

        context_bucket = self._power_of_two_bucket(max(len(row) for row in rows))
        catchup_bucket = self._power_of_two_bucket(catchup_tokens)
        candidates = []
        for k in legal:
            key = (len(rows), family, context_bucket, catchup_bucket, k)
            cell = self._cells.get(key)
            if cell is None:
                continue
            observed = self._accepted_prefix_ema.get(key)
            expected_commits = (
                min(observed, float(k)) + 1.0
                if observed is not None
                else cell.expected_committed_tokens
            )
            speculative_ms_per_token = cell.cycle_ms / expected_commits
            speedup = cell.ordinary_ms_per_token / speculative_ms_per_token
            candidates.append((speedup, k, key))
        if not candidates:
            return self._record(
                AdaptiveDecision(None, "uncalibrated_route", None, None)
            )
        speedup, k, key = max(candidates)
        if speedup < MIN_PREDICTED_SPEEDUP:
            return self._record(
                AdaptiveDecision(None, "below_10_percent_gate", speedup, key)
            )
        return self._record(AdaptiveDecision(k, "selected", speedup, key))

    def observe(self, decision: AdaptiveDecision, result) -> None:
        if decision.selected_k is None or decision.calibration_key is None:
            return
        rows = getattr(result, "rows", ())
        if not rows:
            return
        observed = sum(row.accepted_draft_tokens for row in rows) / len(rows)
        key = decision.calibration_key
        previous = self._accepted_prefix_ema.get(key)
        self._accepted_prefix_ema[key] = (
            observed if previous is None else previous * 0.9 + observed * 0.1
        )

    def _record(self, decision: AdaptiveDecision) -> AdaptiveDecision:
        label = (
            f"chosen_k_{decision.selected_k}"
            if decision.selected_k is not None
            else f"bypass_{decision.reason}"
        )
        self._decision_counts[label] += 1
        self._last_decision = decision
        return decision

    def snapshot(self) -> dict:
        last = self._last_decision
        return {
            "policy": "adaptive",
            "calibration_cells": len(self._cells),
            "decision_counts": dict(sorted(self._decision_counts.items())),
            "last_decision": None if last is None else {
                "selected_k": last.selected_k,
                "reason": last.reason,
                "predicted_speedup": last.predicted_speedup,
                "calibration_key": last.calibration_key,
            },
        }
