"""Pure V4 speculative-step planning and modeled workspace certificates.

This module is intentionally separate from :mod:`scheduler`.  It accepts only
immutable row snapshots plus an already-selected V3 route key and returns a
pickle-safe plan.  No function here owns a ``Sequence``, CUDA tensor, KV-block
lease, or allocator snapshot.

V4 reserves the complete geometry that V5 verification will need even though
V4 executes the proposal path in compute-then-discard shadow mode.  For a live
batch ``B``, common proposal length ``K``, draft catch-up work ``C``, and hard
model-input budget ``M`` the certificate therefore enforces both::

    B * (K + 1) <= M
    C + B * K + B * (K + 1) <= M

The memory certificate is an exact evaluation of the existing conservative
workspace model for the selected route geometry.  It is not a measured CUDA
peak and must remain ``gpu_certified=False`` until a later rung executes and
reconciles every modeled owner.
"""

from dataclasses import dataclass, fields, replace
from hashlib import sha256

import torch

from nanovllm.engine.speculative_memory import (
    SpeculativeMemoryPlan,
    SpeculativeMemoryPlanningError,
    plan_speculative_workspace,
    speculative_route_fits_plan,
)
from nanovllm.engine.speculative_routes import (
    DRAFT_ROUTE_SCHEMA,
    DraftCatchupFamily,
    DraftExecutionMode,
    DraftRouteKey,
    DraftSamplerEnvelope,
)


SPECULATIVE_STEP_PLAN_SCHEMA = "speculative-step-v1"


def _strict_int(name: str, value: int, *, minimum: int = 0) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _optional_position(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    return _strict_int(name, value)


def _primitive_route_payload(route_key: DraftRouteKey | None) -> tuple | None:
    if route_key is None:
        return None
    return (
        route_key.schema,
        route_key.execution_mode.value,
        route_key.batch_bucket,
        route_key.effective_k,
        route_key.catchup_family.value,
        route_key.sampler_envelope.value,
    )


def _validate_route_key(route_key: DraftRouteKey) -> None:
    if not isinstance(route_key, DraftRouteKey):
        raise TypeError("route_key must be a DraftRouteKey")
    if route_key.schema != DRAFT_ROUTE_SCHEMA:
        raise ValueError("route_key uses an unsupported schema")
    if not isinstance(route_key.execution_mode, DraftExecutionMode):
        raise TypeError("route_key execution_mode is invalid")
    _strict_int("route_key.batch_bucket", route_key.batch_bucket, minimum=1)
    _strict_int("route_key.effective_k", route_key.effective_k, minimum=1)
    if not isinstance(route_key.catchup_family, DraftCatchupFamily):
        raise TypeError("route_key catchup_family is invalid")
    if not isinstance(route_key.sampler_envelope, DraftSamplerEnvelope):
        raise TypeError("route_key sampler_envelope is invalid")


@dataclass(frozen=True, slots=True)
class SpecPlanRow:
    """Primitive-only scheduler facts for one selected decode row."""

    seq_id: int
    committed_tokens: int
    target_cached_tokens: int
    draft_cached_tokens: int
    remaining_completion_tokens: int
    model_position_headroom: int
    highest_draft_write_position: int | None
    highest_target_write_position: int | None
    block_table: tuple[int, ...]

    def __post_init__(self) -> None:
        _strict_int("seq_id", self.seq_id)
        _strict_int("committed_tokens", self.committed_tokens, minimum=1)
        _strict_int("target_cached_tokens", self.target_cached_tokens)
        _strict_int("draft_cached_tokens", self.draft_cached_tokens)
        _strict_int(
            "remaining_completion_tokens",
            self.remaining_completion_tokens,
        )
        _strict_int("model_position_headroom", self.model_position_headroom)
        _optional_position(
            "highest_draft_write_position",
            self.highest_draft_write_position,
        )
        _optional_position(
            "highest_target_write_position",
            self.highest_target_write_position,
        )
        if self.target_cached_tokens > self.committed_tokens:
            raise ValueError("target cache coverage exceeds committed tokens")
        if self.draft_cached_tokens > self.target_cached_tokens:
            raise ValueError("draft cache coverage exceeds target cache coverage")
        if not isinstance(self.block_table, tuple):
            raise TypeError("block_table must be a tuple")
        for block_id in self.block_table:
            _strict_int("block_table entry", block_id)

    @property
    def highest_proposal_input_position(self) -> int | None:
        """V3 shadow-runner name for the last draft write position."""

        return self.highest_draft_write_position


@dataclass(frozen=True, slots=True)
class SpeculativeWorkspaceCertificate:
    """Primitive recomputation result for one exact V4 cycle geometry."""

    live_batch_size: int
    workspace_batch_size: int
    effective_k: int
    draft_catchup_tokens: int
    modeled_live_peak_bytes: int
    reservation_bytes: int
    workspace_fingerprint: str
    gpu_certified: bool

    def __post_init__(self) -> None:
        for name in (
            "live_batch_size",
            "workspace_batch_size",
            "effective_k",
            "draft_catchup_tokens",
            "modeled_live_peak_bytes",
            "reservation_bytes",
        ):
            _strict_int(name, getattr(self, name))
        if not isinstance(self.workspace_fingerprint, str) \
                or len(self.workspace_fingerprint) != 64:
            raise ValueError("workspace_fingerprint must be a SHA-256 hex digest")
        try:
            int(self.workspace_fingerprint, 16)
        except ValueError as error:
            raise ValueError(
                "workspace_fingerprint must be a SHA-256 hex digest"
            ) from error
        if type(self.gpu_certified) is not bool:
            raise TypeError("gpu_certified must be a bool")
        if self.gpu_certified:
            raise ValueError("V4 modeled certificates cannot be GPU-certified")
        if self.effective_k == 0:
            if self.workspace_batch_size != 0:
                raise ValueError("fallback certificates have no workspace batch")
            if self.draft_catchup_tokens != 0:
                raise ValueError("fallback certificates contain no draft catch-up")
            if self.modeled_live_peak_bytes or self.reservation_bytes:
                raise ValueError("fallback certificates contain no workspace bytes")
        else:
            if self.live_batch_size < 1:
                raise ValueError("positive certificates require a live batch")
            if self.workspace_batch_size < self.live_batch_size:
                raise ValueError("workspace batch does not cover the live batch")
            if self.modeled_live_peak_bytes < 1:
                raise ValueError("positive certificates require a modeled peak")
            if self.reservation_bytes < self.modeled_live_peak_bytes:
                raise ValueError("reservation is smaller than the modeled peak")


@dataclass(frozen=True, slots=True)
class SpecStepPlan:
    """Immutable, transport-safe V4 plan and modeled workspace certificate."""

    cycle_id: int
    rows: tuple[SpecPlanRow, ...]
    configured_k: int
    workspace_route_cap: int
    effective_k: int
    route_key: DraftRouteKey | None
    draft_catchup_tokens: int
    draft_query_tokens: int
    target_query_tokens: int
    total_model_positions: int
    modeled_live_peak_bytes: int
    reservation_bytes: int
    workspace_fingerprint: str
    gpu_certified: bool
    bypass_reason: str | None

    def __post_init__(self) -> None:
        _strict_int("cycle_id", self.cycle_id)
        if not isinstance(self.rows, tuple):
            raise TypeError("rows must be a tuple")
        if not all(isinstance(row, SpecPlanRow) for row in self.rows):
            raise TypeError("rows must contain SpecPlanRow values")
        if len({row.seq_id for row in self.rows}) != len(self.rows):
            raise ValueError("speculative plan rows must have unique sequence IDs")
        for name in (
            "configured_k",
            "workspace_route_cap",
            "effective_k",
            "draft_catchup_tokens",
            "draft_query_tokens",
            "target_query_tokens",
            "total_model_positions",
            "modeled_live_peak_bytes",
            "reservation_bytes",
        ):
            _strict_int(name, getattr(self, name))
        if not isinstance(self.workspace_fingerprint, str) \
                or len(self.workspace_fingerprint) != 64:
            raise ValueError("workspace_fingerprint must be a SHA-256 hex digest")
        try:
            int(self.workspace_fingerprint, 16)
        except ValueError as error:
            raise ValueError(
                "workspace_fingerprint must be a SHA-256 hex digest"
            ) from error
        if type(self.gpu_certified) is not bool:
            raise TypeError("gpu_certified must be a bool")
        if self.gpu_certified:
            raise ValueError("V4 modeled plans cannot be GPU-certified")
        if self.effective_k > min(self.configured_k, self.workspace_route_cap):
            raise ValueError("effective_k exceeds configured/workspace route cap")

        batch_size = len(self.rows)
        if self.effective_k == 0:
            if self.route_key is not None:
                raise ValueError("fallback plans cannot carry a route key")
            if not isinstance(self.bypass_reason, str) or not self.bypass_reason:
                raise ValueError("fallback plans require a bypass reason")
            if self.draft_catchup_tokens != 0 or self.draft_query_tokens != 0:
                raise ValueError("fallback plans cannot contain draft work")
            if self.target_query_tokens != batch_size:
                raise ValueError("fallback target work must equal batch size")
            if self.total_model_positions != batch_size:
                raise ValueError("fallback total work must equal batch size")
            if self.modeled_live_peak_bytes or self.reservation_bytes:
                raise ValueError("fallback plans cannot reserve speculative bytes")
            if any(
                row.highest_draft_write_position is not None
                or row.highest_target_write_position is not None
                for row in self.rows
            ):
                raise ValueError("fallback plans cannot name speculative writes")
            return

        if not self.rows:
            raise ValueError("positive speculative plans require rows")
        if self.bypass_reason is not None:
            raise ValueError("positive speculative plans cannot have a bypass reason")
        if self.route_key is None:
            raise ValueError("positive speculative plans require a route key")
        _validate_route_key(self.route_key)
        if self.route_key.effective_k != self.effective_k:
            raise ValueError("route K does not match plan effective_k")
        if self.route_key.batch_bucket < batch_size:
            raise ValueError("route batch bucket is smaller than the live batch")
        for row in self.rows:
            row.__post_init__()
            if row.target_cached_tokens != row.committed_tokens - 1:
                raise ValueError("positive plans require pure decode cache coverage")
            if row.remaining_completion_tokens <= self.effective_k:
                raise ValueError("proposal and bonus exceed request headroom")
            if row.model_position_headroom < self.effective_k:
                raise ValueError("target verification exceeds model headroom")
            if (row.highest_draft_write_position != row.committed_tokens + self.effective_k - 2
                    or row.highest_target_write_position != row.committed_tokens + self.effective_k - 1):
                raise ValueError("speculative write positions do not match plan geometry")
        if self.draft_catchup_tokens != sum(
            row.target_cached_tokens - row.draft_cached_tokens for row in self.rows
        ):
            raise ValueError("draft catch-up count does not match row coverage")
        if self.draft_query_tokens != batch_size * self.effective_k:
            raise ValueError("draft query count does not match B*K")
        if self.target_query_tokens != batch_size * (self.effective_k + 1):
            raise ValueError("target query count does not match B*(K+1)")
        expected_total = (
            self.draft_catchup_tokens
            + self.draft_query_tokens
            + self.target_query_tokens
        )
        if self.total_model_positions != expected_total:
            raise ValueError("total model-position count is inconsistent")
        if self.modeled_live_peak_bytes <= 0:
            raise ValueError("positive plans require a modeled live peak")
        if self.reservation_bytes < self.modeled_live_peak_bytes:
            raise ValueError("workspace reservation is smaller than its live peak")

    @property
    def uses_speculation(self) -> bool:
        return self.effective_k > 0

    @property
    def uses_draft(self) -> bool:
        """Compatibility with the shared proposal/coverage transaction."""
        return self.uses_speculation

    @property
    def fallback_reason(self) -> str | None:
        """V3-compatible spelling of the V4 bypass decision."""

        return self.bypass_reason

    @property
    def draft_step_token_counts(self) -> tuple[int, ...]:
        """One full live-batch proposal count for every shadow draft step."""

        return (len(self.rows),) * self.effective_k

    @property
    def shadow_target_query_tokens(self) -> int:
        """Target inputs V4 actually executes before V5 verification exists."""

        return len(self.rows)

    @property
    def total_scheduled_tokens(self) -> int:
        """V3 shadow-work count retained for the current draft runner.

        This compatibility value is deliberately *not* the V4/V5 reservation
        geometry in :attr:`total_model_positions`.  It counts catch-up, ``B*K``
        draft inputs, and the ordinary target batch that V4 actually executes.
        """

        if not self.uses_speculation:
            return len(self.rows)
        return (
            self.draft_catchup_tokens
            + self.draft_query_tokens
            + self.shadow_target_query_tokens
        )


def draft_catchup_tokens(rows: tuple[SpecPlanRow, ...]) -> int:
    """Return exact draft catch-up positions for immutable row snapshots."""

    if not isinstance(rows, tuple):
        raise TypeError("rows must be a tuple")
    if not all(isinstance(row, SpecPlanRow) for row in rows):
        raise TypeError("rows must contain SpecPlanRow values")
    return sum(
        row.target_cached_tokens - row.draft_cached_tokens
        for row in rows
    )


def speculative_k_budget(
    *,
    max_num_batched_tokens: int,
    batch_size: int,
    draft_catchup_tokens: int,
) -> int:
    """Return V4's zero-clamped aggregate proposal-length budget.

    This is exactly ``max(floor((M-C-B)/(2B)), 0)`` for ``B > 0``.  An
    empty batch has no speculative route and returns zero rather than dividing
    by zero.
    """

    max_num_batched_tokens = _strict_int(
        "max_num_batched_tokens", max_num_batched_tokens
    )
    batch_size = _strict_int("batch_size", batch_size)
    draft_catchup_tokens = _strict_int(
        "draft_catchup_tokens", draft_catchup_tokens
    )
    if batch_size == 0:
        return 0
    return max(
        (
            max_num_batched_tokens
            - draft_catchup_tokens
            - batch_size
        )
        // (2 * batch_size),
        0,
    )


def derive_speculative_effective_k(
    *,
    rows: tuple[SpecPlanRow, ...],
    configured_k: int,
    max_num_batched_tokens: int,
    workspace_route_cap: int,
) -> int:
    """Derive one common K without allocating speculative state.

    The independent verifier bound is retained even though the aggregate V4
    bound is currently at least as strict for non-negative catch-up work.  This
    makes buffer/graph input capacity explicit and prevents a later relaxed
    fairness budget from silently dropping verifier safety.
    """

    if not isinstance(rows, tuple):
        raise TypeError("rows must be a tuple")
    if not all(isinstance(row, SpecPlanRow) for row in rows):
        raise TypeError("rows must contain SpecPlanRow values")
    configured_k = _strict_int("configured_k", configured_k)
    max_num_batched_tokens = _strict_int(
        "max_num_batched_tokens", max_num_batched_tokens
    )
    workspace_route_cap = _strict_int(
        "workspace_route_cap", workspace_route_cap
    )
    if not rows:
        return 0
    batch_size = len(rows)
    catchup = draft_catchup_tokens(rows)
    completion_cap = min(
        max(row.remaining_completion_tokens - 1, 0)
        for row in rows
    )
    model_position_cap = min(
        max(row.model_position_headroom, 0)
        for row in rows
    )
    verifier_cap = max(
        max_num_batched_tokens // batch_size - 1,
        0,
    )
    aggregate_cap = speculative_k_budget(
        max_num_batched_tokens=max_num_batched_tokens,
        batch_size=batch_size,
        draft_catchup_tokens=catchup,
    )
    return min(
        configured_k,
        completion_cap,
        model_position_cap,
        verifier_cap,
        aggregate_cap,
        workspace_route_cap,
    )


def _canonical_dtype_for_itemsize(name: str, itemsize: int) -> torch.dtype:
    """Recover an arithmetic-equivalent dtype from V2's stored item size.

    ``SpeculativeMemoryPlan`` stores item sizes because every modeled tensor
    byte formula depends on size, not on FP16-vs-BF16 numerical semantics.  A
    canonical floating dtype therefore reproduces the exact byte plan without
    extending or mutating the frozen V2 dataclass.
    """

    _strict_int(name, itemsize, minimum=1)
    candidates = (
        getattr(torch, "float8_e4m3fn", None),
        torch.float16,
        torch.float32,
        torch.float64,
    )
    for dtype in candidates:
        if isinstance(dtype, torch.dtype) and dtype.itemsize == itemsize:
            return dtype
    raise SpeculativeMemoryPlanningError(
        f"{name}={itemsize} has no supported canonical floating dtype"
    )


def plan_exact_speculative_workspace(
    configured_plan: SpeculativeMemoryPlan,
    *,
    batch_size: int,
    effective_k: int,
) -> SpeculativeMemoryPlan:
    """Evaluate V2's conservative model at one exact positive ``(B, K)``.

    ``batch_size`` is the workspace shape, which may be a padded route bucket
    larger than the live selected batch.  The function fails if that shape was
    not covered by the configured reservation.
    """

    if not isinstance(configured_plan, SpeculativeMemoryPlan):
        raise TypeError("configured_plan must be a SpeculativeMemoryPlan")
    batch_size = _strict_int("batch_size", batch_size, minimum=1)
    effective_k = _strict_int("effective_k", effective_k, minimum=1)
    if not speculative_route_fits_plan(
        configured_plan,
        batch_size=batch_size,
        effective_k=effective_k,
    ):
        raise SpeculativeMemoryPlanningError(
            "route geometry exceeds the configured speculative workspace"
        )

    exact = plan_speculative_workspace(
        vocab_size=configured_plan.vocab_size,
        configured_k=effective_k,
        max_num_seqs=batch_size,
        max_num_batched_tokens=batch_size * (effective_k + 1),
        max_model_len=effective_k + 1,
        target_logits_dtype=_canonical_dtype_for_itemsize(
            "target_logits_itemsize",
            configured_plan.target_logits_itemsize,
        ),
        draft_logits_dtype=_canonical_dtype_for_itemsize(
            "draft_logits_itemsize",
            configured_plan.draft_logits_itemsize,
        ),
    )
    if exact.batch_size != batch_size or exact.max_effective_k != effective_k:
        raise RuntimeError("exact workspace planner changed requested geometry")
    if exact.reservation_bytes > configured_plan.reservation_bytes:
        raise SpeculativeMemoryPlanningError(
            "exact route reservation exceeds the configured reservation"
        )
    return exact


def speculative_workspace_fingerprint(
    exact_workspace: SpeculativeMemoryPlan | None,
    *,
    live_batch_size: int,
    draft_catchup_tokens: int,
    route_key: DraftRouteKey | None,
) -> str:
    """Bind modeled bytes to live geometry and the machine-readable route."""

    live_batch_size = _strict_int("live_batch_size", live_batch_size)
    draft_catchup_tokens = _strict_int(
        "draft_catchup_tokens", draft_catchup_tokens
    )
    if exact_workspace is None:
        if route_key is not None:
            raise ValueError("a fallback fingerprint cannot carry a route key")
        if draft_catchup_tokens != 0:
            raise ValueError("a fallback fingerprint cannot contain draft catch-up")
        payload = (
            SPECULATIVE_STEP_PLAN_SCHEMA,
            "ordinary_decode_fallback",
            live_batch_size,
            draft_catchup_tokens,
        )
    else:
        if not isinstance(exact_workspace, SpeculativeMemoryPlan):
            raise TypeError("exact_workspace must be a SpeculativeMemoryPlan")
        if route_key is None:
            raise ValueError("a positive workspace fingerprint requires a route")
        _validate_route_key(route_key)
        if route_key.effective_k != exact_workspace.max_effective_k:
            raise ValueError("route K does not match exact workspace geometry")
        if route_key.batch_bucket != exact_workspace.batch_size:
            raise ValueError("route batch bucket does not match workspace geometry")
        if not 1 <= live_batch_size <= exact_workspace.batch_size:
            raise ValueError("live batch does not fit the workspace geometry")
        workspace_payload = tuple(
            (field.name, repr(getattr(exact_workspace, field.name)))
            for field in fields(exact_workspace)
        )
        payload = (
            SPECULATIVE_STEP_PLAN_SCHEMA,
            "modeled_not_gpu_certified",
            live_batch_size,
            draft_catchup_tokens,
            _primitive_route_payload(route_key),
            workspace_payload,
        )
    return sha256(repr(payload).encode("utf-8")).hexdigest()


def certify_speculative_workspace(
    configured_workspace: SpeculativeMemoryPlan,
    *,
    live_batch_size: int,
    draft_catchup_tokens: int,
    effective_k: int,
    route_key: DraftRouteKey | None,
) -> SpeculativeWorkspaceCertificate:
    """Independently recompute the modeled certificate for ``B/C/K/route``."""

    if not isinstance(configured_workspace, SpeculativeMemoryPlan):
        raise TypeError("configured_workspace must be a SpeculativeMemoryPlan")
    live_batch_size = _strict_int("live_batch_size", live_batch_size)
    draft_catchup_tokens = _strict_int(
        "draft_catchup_tokens", draft_catchup_tokens
    )
    effective_k = _strict_int("effective_k", effective_k)
    if effective_k == 0:
        if route_key is not None:
            raise ValueError("fallback certificates cannot carry a route key")
        if draft_catchup_tokens != 0:
            raise ValueError("fallback certificates cannot contain draft catch-up")
        return SpeculativeWorkspaceCertificate(
            live_batch_size=live_batch_size,
            workspace_batch_size=0,
            effective_k=0,
            draft_catchup_tokens=0,
            modeled_live_peak_bytes=0,
            reservation_bytes=0,
            workspace_fingerprint=speculative_workspace_fingerprint(
                None,
                live_batch_size=live_batch_size,
                draft_catchup_tokens=0,
                route_key=None,
            ),
            gpu_certified=False,
        )
    if live_batch_size == 0:
        raise ValueError("positive certificates require a live batch")
    if route_key is None:
        raise ValueError("positive certificates require a route key")
    _validate_route_key(route_key)
    if route_key.effective_k != effective_k:
        raise ValueError("route K does not match effective_k")
    if route_key.batch_bucket < live_batch_size:
        raise ValueError("route batch bucket is smaller than the live batch")
    expected_catchup_family = (
        DraftCatchupFamily.NONE
        if draft_catchup_tokens == 0
        else DraftCatchupFamily.PAGED_EAGER_DYNAMIC
    )
    if route_key.catchup_family is not expected_catchup_family:
        raise ValueError("route catch-up family does not match cycle geometry")
    exact_workspace = plan_exact_speculative_workspace(
        configured_workspace,
        batch_size=route_key.batch_bucket,
        effective_k=effective_k,
    )
    return SpeculativeWorkspaceCertificate(
        live_batch_size=live_batch_size,
        workspace_batch_size=route_key.batch_bucket,
        effective_k=effective_k,
        draft_catchup_tokens=draft_catchup_tokens,
        modeled_live_peak_bytes=exact_workspace.modeled_live_peak_bytes,
        reservation_bytes=exact_workspace.reservation_bytes,
        workspace_fingerprint=speculative_workspace_fingerprint(
            exact_workspace,
            live_batch_size=live_batch_size,
            draft_catchup_tokens=draft_catchup_tokens,
            route_key=route_key,
        ),
        gpu_certified=False,
    )


def build_speculative_step_plan(
    *,
    cycle_id: int,
    rows: tuple[SpecPlanRow, ...],
    configured_k: int,
    workspace_route_cap: int,
    effective_k: int,
    max_num_batched_tokens: int,
    configured_workspace: SpeculativeMemoryPlan,
    route_key: DraftRouteKey | None,
    bypass_reason: str | None = None,
) -> SpecStepPlan:
    """Build and fully validate one V4 transport plan.

    Callers derive ``effective_k`` with :func:`derive_speculative_effective_k`,
    select the matching ready route, and invoke this builder before allocating
    proposal tensors or provisional KV blocks.
    """

    cycle_id = _strict_int("cycle_id", cycle_id)
    if not isinstance(rows, tuple):
        raise TypeError("rows must be a tuple")
    if not all(isinstance(row, SpecPlanRow) for row in rows):
        raise TypeError("rows must contain SpecPlanRow values")
    if len({row.seq_id for row in rows}) != len(rows):
        raise ValueError("speculative plan rows must have unique sequence IDs")
    effective_k = _strict_int("effective_k", effective_k)
    configured_k = _strict_int("configured_k", configured_k)
    workspace_route_cap = _strict_int(
        "workspace_route_cap", workspace_route_cap
    )
    max_num_batched_tokens = _strict_int(
        "max_num_batched_tokens", max_num_batched_tokens
    )
    if not isinstance(configured_workspace, SpeculativeMemoryPlan):
        raise TypeError("configured_workspace must be a SpeculativeMemoryPlan")
    if workspace_route_cap > configured_workspace.max_effective_k:
        raise SpeculativeMemoryPlanningError(
            "workspace route cap exceeds the configured reservation"
        )
    if effective_k > min(configured_k, workspace_route_cap):
        raise SpeculativeMemoryPlanningError(
            "effective_k exceeds configured/workspace route cap"
        )

    batch_size = len(rows)
    if effective_k == 0:
        if route_key is not None:
            raise ValueError("fallback plans cannot carry a route key")
        planned_rows = tuple(
            replace(
                row,
                highest_draft_write_position=None,
                highest_target_write_position=None,
            )
            for row in rows
        )
        return SpecStepPlan(
            cycle_id=cycle_id,
            rows=planned_rows,
            configured_k=configured_k,
            workspace_route_cap=workspace_route_cap,
            effective_k=0,
            route_key=None,
            draft_catchup_tokens=0,
            draft_query_tokens=0,
            target_query_tokens=batch_size,
            total_model_positions=batch_size,
            modeled_live_peak_bytes=0,
            reservation_bytes=0,
            workspace_fingerprint=speculative_workspace_fingerprint(
                None,
                live_batch_size=batch_size,
                draft_catchup_tokens=0,
                route_key=None,
            ),
            gpu_certified=False,
            bypass_reason=bypass_reason,
        )

    if not rows:
        raise ValueError("positive speculative plans require rows")
    if route_key is None:
        raise ValueError("positive speculative plans require a route key")
    _validate_route_key(route_key)
    if route_key.effective_k != effective_k:
        raise ValueError("route K does not match effective_k")
    if route_key.batch_bucket < batch_size:
        raise ValueError("route batch bucket is smaller than the live batch")
    if bypass_reason is not None:
        raise ValueError("positive speculative plans cannot have a bypass reason")

    catchup = draft_catchup_tokens(rows)
    draft_queries = batch_size * effective_k
    target_queries = batch_size * (effective_k + 1)
    total_positions = catchup + draft_queries + target_queries
    if target_queries > max_num_batched_tokens:
        raise SpeculativeMemoryPlanningError(
            "target verifier rows exceed max_num_batched_tokens"
        )
    if total_positions > max_num_batched_tokens:
        raise SpeculativeMemoryPlanningError(
            "aggregate speculative work exceeds max_num_batched_tokens"
        )
    for row in rows:
        if row.remaining_completion_tokens < effective_k + 1:
            raise SpeculativeMemoryPlanningError(
                f"sequence {row.seq_id} lacks K+1 completion headroom"
            )
        if row.model_position_headroom < effective_k:
            raise SpeculativeMemoryPlanningError(
                f"sequence {row.seq_id} lacks target position headroom"
            )

    certificate = certify_speculative_workspace(
        configured_workspace,
        live_batch_size=batch_size,
        draft_catchup_tokens=catchup,
        effective_k=effective_k,
        route_key=route_key,
    )
    planned_rows = tuple(
        replace(
            row,
            highest_draft_write_position=(
                row.committed_tokens + effective_k - 2
            ),
            highest_target_write_position=(
                row.committed_tokens + effective_k - 1
            ),
        )
        for row in rows
    )
    return SpecStepPlan(
        cycle_id=cycle_id,
        rows=planned_rows,
        configured_k=configured_k,
        workspace_route_cap=workspace_route_cap,
        effective_k=effective_k,
        route_key=route_key,
        draft_catchup_tokens=catchup,
        draft_query_tokens=draft_queries,
        target_query_tokens=target_queries,
        total_model_positions=total_positions,
        modeled_live_peak_bytes=certificate.modeled_live_peak_bytes,
        reservation_bytes=certificate.reservation_bytes,
        workspace_fingerprint=certificate.workspace_fingerprint,
        gpu_certified=False,
        bypass_reason=None,
    )


__all__ = [
    "SPECULATIVE_STEP_PLAN_SCHEMA",
    "SpecPlanRow",
    "SpecStepPlan",
    "SpeculativeWorkspaceCertificate",
    "build_speculative_step_plan",
    "certify_speculative_workspace",
    "derive_speculative_effective_k",
    "draft_catchup_tokens",
    "plan_exact_speculative_workspace",
    "speculative_k_budget",
    "speculative_workspace_fingerprint",
]
