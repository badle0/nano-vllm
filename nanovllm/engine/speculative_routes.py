"""Finite execution-route registry for speculative draft-discard cycles.

The registry is deliberately host-only.  It binds every runtime-admissible
``(execution mode, batch bucket, K, catch-up family)`` tuple to both the fixed
speculative memory reservation and the constructor phases that made that route
ready.  Runtime lookup therefore finishes before temporary KV blocks, RNG, or
CUDA proposal workspace can be touched.

Sampling parameters do not expand the key space.  The exact speculative sampler
is an eager sequence of ATen operations and distinct top-k buckets execute one
after another, so one conservative all-compositions envelope covers arbitrary
greedy/top-k/top-p mixtures without an exponential row-mask registry.
"""

from dataclasses import dataclass, fields, replace
from enum import Enum
from hashlib import sha256

from nanovllm.engine.speculative_memory import SpeculativeMemoryPlan


DRAFT_ROUTE_SCHEMA = "draft-discard-v2-lm-head"
MAX_CUDA_GRAPH_BATCH_SIZE = 512
# V3 is a correctness route, not yet the V7 performance router.  Bounding the
# enumerated K axis prevents a hostile-but-valid configuration from constructing
# millions of host route entries before CUDA capacity can fail closed.  Larger
# configured K values remain correct: admission exposes at most this ready K and
# the scheduler derives the batch-wide effective K from that cap.
MAX_DRAFT_ROUTE_EFFECTIVE_K = 32
FP32_BYTES = 4
INT64_BYTES = 8


class _StringEnum(str, Enum):
    """Python-3.10-compatible string enum with StrEnum-style formatting."""

    def __str__(self) -> str:
        return self.value


class DraftExecutionMode(_StringEnum):
    EAGER_DYNAMIC = "eager_dynamic"
    CUDA_GRAPH = "cuda_graph"


class DraftCatchupFamily(_StringEnum):
    NONE = "none"
    PAGED_EAGER_DYNAMIC = "paged_eager_dynamic_v1"


class DraftSamplerEnvelope(_StringEnum):
    EXACT_WORST_CASE = "exact_all_compositions_worst_case_v1"


@dataclass(frozen=True, slots=True, order=True)
class DraftRouteKey:
    schema: str
    execution_mode: DraftExecutionMode
    batch_bucket: int
    effective_k: int
    catchup_family: DraftCatchupFamily
    sampler_envelope: DraftSamplerEnvelope


@dataclass(frozen=True, slots=True, order=True)
class DraftWarmComponentKey:
    family: str
    batch_bucket: int = 0


@dataclass(frozen=True, slots=True)
class DraftWorkspaceCertificate:
    key: DraftRouteKey
    batch_capacity: int
    q_bytes: int
    proposal_id_bytes: int
    modeled_draft_live_bytes: int
    reserved_plan_bytes: int
    plan_fingerprint: str


@dataclass(frozen=True, slots=True)
class DraftRouteEntry:
    key: DraftRouteKey
    workspace: DraftWorkspaceCertificate
    required_warm_components: frozenset[DraftWarmComponentKey]


@dataclass(frozen=True, slots=True)
class DraftRouteAdmission:
    """Immutable pre-reservation lookup result for one exact decode batch."""

    registry_schema: str
    plan_fingerprint: str
    batch_size: int
    catchup_tokens: int
    route_keys: tuple[DraftRouteKey, ...]

    def __post_init__(self):
        if self.registry_schema != DRAFT_ROUTE_SCHEMA:
            raise ValueError("draft admission uses an unsupported registry schema")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("draft admission batch_size must be positive")
        if type(self.catchup_tokens) is not int or self.catchup_tokens < 0:
            raise ValueError("draft admission catchup_tokens must be non-negative")
        if not isinstance(self.plan_fingerprint, str) or not self.plan_fingerprint:
            raise ValueError("draft admission requires a plan fingerprint")
        if not isinstance(self.route_keys, tuple):
            raise TypeError("draft admission route_keys must be a tuple")
        if tuple(key.effective_k for key in self.route_keys) != tuple(
            range(1, len(self.route_keys) + 1)
        ):
            raise ValueError("draft admission route keys must cover contiguous K")

    @property
    def max_effective_k(self) -> int:
        return len(self.route_keys)

    def key_for(self, effective_k: int) -> DraftRouteKey | None:
        if type(effective_k) is not int:
            raise TypeError("effective_k must be an integer")
        if not 1 <= effective_k <= len(self.route_keys):
            return None
        return self.route_keys[effective_k - 1]


@dataclass(frozen=True, slots=True)
class DraftRouteRegistry:
    """Immutable desired/ready route ledger for one speculative memory plan."""

    schema: str
    plan_fingerprint: str
    entries: tuple[DraftRouteEntry, ...]
    warmed_components: frozenset[DraftWarmComponentKey] = frozenset()

    def __post_init__(self):
        if self.schema != DRAFT_ROUTE_SCHEMA:
            raise ValueError("unsupported draft route registry schema")
        keys = tuple(entry.key for entry in self.entries)
        if len(set(keys)) != len(keys):
            raise ValueError("draft route registry contains duplicate keys")
        if keys != tuple(sorted(keys)):
            raise ValueError("draft route registry entries must be sorted")
        desired = self.desired_warm_components
        if not self.warmed_components.issubset(desired):
            raise ValueError("draft registry names an unknown warmed component")

    @property
    def desired_warm_components(self) -> frozenset[DraftWarmComponentKey]:
        return frozenset(
            component
            for entry in self.entries
            for component in entry.required_warm_components
        )

    @property
    def router_admitted_keys(self) -> frozenset[DraftRouteKey]:
        return frozenset(entry.key for entry in self.entries)

    @property
    def workspace_certified_keys(self) -> frozenset[DraftRouteKey]:
        return frozenset(
            entry.key
            for entry in self.entries
            if entry.workspace.key == entry.key
            and entry.workspace.plan_fingerprint == self.plan_fingerprint
            and entry.workspace.batch_capacity == entry.key.batch_bucket
            and entry.workspace.q_bytes >= 0
            and entry.workspace.proposal_id_bytes
            == (
                entry.key.batch_bucket
                * entry.key.effective_k
                * INT64_BYTES
            )
            and (
                entry.workspace.q_bytes
                + entry.workspace.proposal_id_bytes
                <= entry.workspace.modeled_draft_live_bytes
            )
            and entry.workspace.modeled_draft_live_bytes
            <= entry.workspace.reserved_plan_bytes
        )

    @property
    def warm_capture_keys(self) -> frozenset[DraftRouteKey]:
        return frozenset(
            entry.key
            for entry in self.entries
            if entry.required_warm_components.issubset(self.warmed_components)
        )

    @property
    def ready_keys(self) -> frozenset[DraftRouteKey]:
        return (
            self.router_admitted_keys
            & self.workspace_certified_keys
            & self.warm_capture_keys
        )

    @property
    def graph_buckets(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    entry.key.batch_bucket
                    for entry in self.entries
                    if entry.key.execution_mode is DraftExecutionMode.CUDA_GRAPH
                }
            )
        )

    def with_warmed_components(
        self,
        components,
    ) -> "DraftRouteRegistry":
        warmed = frozenset(components)
        if not warmed.issubset(self.desired_warm_components):
            raise ValueError("cannot warm an undeclared draft route component")
        return replace(self, warmed_components=warmed)

    def resolve(
        self,
        *,
        batch_size: int,
        catchup_tokens: int,
    ) -> DraftRouteAdmission | None:
        """Resolve a contiguous ready-K range for an exact live batch."""

        if type(batch_size) is not int:
            raise TypeError("batch_size must be an integer")
        if type(catchup_tokens) is not int:
            raise TypeError("catchup_tokens must be an integer")
        if batch_size < 1 or catchup_tokens < 0:
            return None
        catchup_family = (
            DraftCatchupFamily.NONE
            if catchup_tokens == 0
            else DraftCatchupFamily.PAGED_EAGER_DYNAMIC
        )
        ready_keys = self.ready_keys
        candidates = tuple(
            entry.key
            for entry in self.entries
            if entry.key in ready_keys
            and entry.key.catchup_family is catchup_family
            and batch_size <= entry.key.batch_bucket
        )
        if not candidates:
            return None
        batch_bucket = min(key.batch_bucket for key in candidates)
        keys = tuple(
            sorted(
                (
                    key
                    for key in candidates
                    if key.batch_bucket == batch_bucket
                ),
                key=lambda key: key.effective_k,
            )
        )
        if tuple(key.effective_k for key in keys) != tuple(
            range(1, len(keys) + 1)
        ):
            raise RuntimeError("ready draft routes do not cover contiguous K")
        return DraftRouteAdmission(
            registry_schema=self.schema,
            plan_fingerprint=self.plan_fingerprint,
            batch_size=batch_size,
            catchup_tokens=catchup_tokens,
            route_keys=keys,
        )

    def validate_runtime_key(
        self,
        key: DraftRouteKey,
        *,
        batch_size: int,
        effective_k: int,
        catchup_tokens: int,
    ) -> bool:
        admission = self.resolve(
            batch_size=batch_size,
            catchup_tokens=catchup_tokens,
        )
        return admission is not None and admission.key_for(effective_k) == key


def draft_graph_batch_buckets(batch_cap: int) -> tuple[int, ...]:
    """Return the finite decode capture buckets, including the exact endpoint."""

    if type(batch_cap) is not int:
        raise TypeError("batch_cap must be an integer")
    if batch_cap < 0:
        raise ValueError("batch_cap must be non-negative")
    if batch_cap == 0:
        return ()
    return tuple(
        sorted(
            {
                *(
                    bucket
                    for bucket in (1, 2, 4, 8)
                    if bucket <= batch_cap
                ),
                *range(16, batch_cap + 1, 16),
                batch_cap,
            }
        )
    )


def speculative_plan_fingerprint(
    plan: SpeculativeMemoryPlan,
    numerical_backend: str = "fast",
) -> str:
    if not isinstance(plan, SpeculativeMemoryPlan):
        raise TypeError("plan must be a SpeculativeMemoryPlan")
    if not isinstance(numerical_backend, str) or not numerical_backend:
        raise ValueError("numerical_backend must be a non-empty string")
    payload = (
        numerical_backend,
        tuple(
            (field.name, repr(getattr(plan, field.name)))
            for field in fields(plan)
        ),
    )
    return sha256(repr(payload).encode("utf-8")).hexdigest()


def max_eligible_draft_catchup(
    *,
    max_num_batched_tokens: int,
    max_model_len: int,
    batch_cap: int,
) -> tuple[int, int]:
    """Return ``(aggregate catch-up tokens, live batch)`` for V3's K=1 edge.

    For ``B`` rows, the smallest admitted proposal cycle costs ``2*B`` model
    inputs in addition to catch-up.  Each row can be missing at most ``L-2``
    committed-prefix positions while retaining one proposal position.  Taking
    the maximum across the finite admitted batch domain gives the paged-ragged
    witness that constructor pretouch must execute.
    """

    for name, value in (
        ("max_num_batched_tokens", max_num_batched_tokens),
        ("max_model_len", max_model_len),
        ("batch_cap", batch_cap),
    ):
        if type(value) is not int:
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    if batch_cap == 0 or max_model_len == 0:
        return (0, 0)
    best = (0, 0)
    per_row_cap = max(max_model_len - 2, 0)
    quotient, remainder = divmod(max_num_batched_tokens, max_model_len)
    candidates = {
        1,
        batch_cap,
        min(max(quotient, 1), batch_cap),
        min(max(quotient + int(remainder > 0), 1), batch_cap),
    }
    # min(M-2B, B*(L-2)) increases before B=M/L and decreases
    # afterwards, so the clipped integer neighbors and endpoints are complete.
    for batch_size in sorted(candidates):
        token_budget_cap = max_num_batched_tokens - 2 * batch_size
        aggregate = min(token_budget_cap, batch_size * per_row_cap)
        candidate = (max(aggregate, 0), batch_size)
        if candidate[0] > best[0]:
            best = candidate
    return best


def build_draft_route_registry(
    plan: SpeculativeMemoryPlan,
    *,
    enforce_eager: bool,
    numerical_backend: str = "fast",
) -> DraftRouteRegistry:
    """Build the bounded desired registry before graph capture/pretouch."""

    if not isinstance(plan, SpeculativeMemoryPlan):
        raise TypeError("plan must be a SpeculativeMemoryPlan")
    if type(enforce_eager) is not bool:
        raise TypeError("enforce_eager must be a bool")
    fingerprint = speculative_plan_fingerprint(plan, numerical_backend)
    if plan.batch_size == 0 or plan.max_effective_k == 0:
        return DraftRouteRegistry(DRAFT_ROUTE_SCHEMA, fingerprint, ())

    execution_mode = (
        DraftExecutionMode.EAGER_DYNAMIC
        if enforce_eager
        else DraftExecutionMode.CUDA_GRAPH
    )
    route_k_cap = min(
        plan.max_effective_k,
        MAX_DRAFT_ROUTE_EFFECTIVE_K,
    )
    batch_cap = (
        plan.batch_size
        if enforce_eager
        else min(plan.batch_size, MAX_CUDA_GRAPH_BATCH_SIZE)
    )
    batch_buckets = (
        (batch_cap,)
        if enforce_eager
        else draft_graph_batch_buckets(batch_cap)
    )
    sampler_component = DraftWarmComponentKey(
        "exact_sampler_envelope", batch_cap
    )
    catchup_component = DraftWarmComponentKey(
        "draft_catchup_paged_dynamic", batch_cap
    )
    entries = []
    previous_bucket = 0
    for batch_bucket in batch_buckets:
        # This key is selected only for live B in (previous_bucket, bucket].
        # Prune K/family combinations that no B in that interval can execute.
        minimum_live_batch = previous_bucket + 1
        decode_component = DraftWarmComponentKey(
            (
                "draft_decode_eager_dynamic"
                if enforce_eager
                else "draft_decode_graph"
            ),
            batch_bucket,
        )
        for effective_k in range(1, route_k_cap + 1):
            if effective_k > max(plan.max_model_len - 2, 0):
                continue
            for catchup_family in DraftCatchupFamily:
                minimum_work = minimum_live_batch * (effective_k + 1)
                if catchup_family is DraftCatchupFamily.PAGED_EAGER_DYNAMIC:
                    minimum_work += 1
                if minimum_work > plan.max_num_batched_tokens:
                    continue
                key = DraftRouteKey(
                    schema=DRAFT_ROUTE_SCHEMA,
                    execution_mode=execution_mode,
                    batch_bucket=batch_bucket,
                    effective_k=effective_k,
                    catchup_family=catchup_family,
                    sampler_envelope=DraftSamplerEnvelope.EXACT_WORST_CASE,
                )
                required = {decode_component, sampler_component}
                if catchup_family is DraftCatchupFamily.PAGED_EAGER_DYNAMIC:
                    required.add(catchup_component)
                certificate = DraftWorkspaceCertificate(
                    key=key,
                    batch_capacity=batch_bucket,
                    q_bytes=(
                        batch_bucket
                        * effective_k
                        * plan.vocab_size
                        * FP32_BYTES
                    ),
                    proposal_id_bytes=(
                        batch_bucket * effective_k * INT64_BYTES
                    ),
                    # The configured-maximum plan is conservative for each
                    # smaller key.  Retain that full phase bound until the A100
                    # route-peak certificate replaces it with measured values.
                    modeled_draft_live_bytes=plan.draft_phase_bytes,
                    reserved_plan_bytes=plan.reservation_bytes,
                    plan_fingerprint=fingerprint,
                )
                entries.append(
                    DraftRouteEntry(key, certificate, frozenset(required))
                )
        previous_bucket = batch_bucket
    return DraftRouteRegistry(
        schema=DRAFT_ROUTE_SCHEMA,
        plan_fingerprint=fingerprint,
        entries=tuple(sorted(entries, key=lambda entry: entry.key)),
    )


__all__ = [
    "DRAFT_ROUTE_SCHEMA",
    "MAX_CUDA_GRAPH_BATCH_SIZE",
    "MAX_DRAFT_ROUTE_EFFECTIVE_K",
    "DraftCatchupFamily",
    "DraftExecutionMode",
    "DraftRouteAdmission",
    "DraftRouteEntry",
    "DraftRouteKey",
    "DraftRouteRegistry",
    "DraftSamplerEnvelope",
    "DraftWarmComponentKey",
    "DraftWorkspaceCertificate",
    "build_draft_route_registry",
    "draft_graph_batch_buckets",
    "max_eligible_draft_catchup",
    "speculative_plan_fingerprint",
]
