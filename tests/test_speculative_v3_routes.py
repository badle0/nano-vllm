from dataclasses import FrozenInstanceError, replace
from types import MethodType, SimpleNamespace

import pytest
import torch

from nanovllm.engine.speculative_memory import (
    INT64_BYTES,
    plan_speculative_workspace,
)
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.speculative_routes import (
    DRAFT_ROUTE_SCHEMA,
    DraftCatchupFamily,
    DraftExecutionMode,
    DraftRouteKey,
    DraftRouteRegistry,
    DraftSamplerEnvelope,
    MAX_CUDA_GRAPH_BATCH_SIZE,
    MAX_DRAFT_ROUTE_EFFECTIVE_K,
    build_draft_route_registry,
    draft_graph_batch_buckets,
    max_eligible_draft_catchup,
    speculative_plan_fingerprint,
)


def _memory_plan(*, batch_cap=4, k_cap=3, vocab_size=17):
    return plan_speculative_workspace(
        vocab_size=vocab_size,
        configured_k=k_cap,
        max_num_seqs=batch_cap,
        max_num_batched_tokens=batch_cap * (k_cap + 1),
        max_model_len=64,
        target_logits_dtype=torch.bfloat16,
        draft_logits_dtype=torch.bfloat16,
    )


@pytest.mark.parametrize(
    ("enforce_eager", "expected_entries", "expected_buckets"),
    [
        (False, 2 * 3 * 3, (1, 2, 4)),
        (True, 2 * 1 * 3, ()),
    ],
)
def test_route_registry_is_finite_versioned_and_immutable(
    enforce_eager,
    expected_entries,
    expected_buckets,
):
    registry = build_draft_route_registry(
        _memory_plan(), enforce_eager=enforce_eager
    )

    assert registry.schema == DRAFT_ROUTE_SCHEMA
    assert len(registry.entries) == expected_entries
    assert registry.graph_buckets == expected_buckets
    assert isinstance(registry.router_admitted_keys, frozenset)
    assert isinstance(registry.workspace_certified_keys, frozenset)
    assert isinstance(registry.warm_capture_keys, frozenset)
    assert not registry.warm_capture_keys
    assert len(registry.router_admitted_keys) == expected_entries
    assert all(hash(key) for key in registry.router_admitted_keys)
    assert all(key.effective_k <= 3 for key in registry.router_admitted_keys)
    assert all(key.batch_bucket <= 4 for key in registry.router_admitted_keys)
    with pytest.raises(FrozenInstanceError):
        registry.schema = "changed"
    assert str(DraftExecutionMode.EAGER_DYNAMIC) == "eager_dynamic"
    assert isinstance(DraftExecutionMode.EAGER_DYNAMIC, str)


def test_graph_bucket_policy_includes_nonmultiple_exact_endpoint():
    assert draft_graph_batch_buckets(0) == ()
    assert draft_graph_batch_buckets(1) == (1,)
    assert draft_graph_batch_buckets(3) == (1, 2, 3)
    assert draft_graph_batch_buckets(17) == (1, 2, 4, 8, 16, 17)


def test_graph_registry_caps_batch_and_k_cardinality_before_cuda():
    plan = plan_speculative_workspace(
        vocab_size=17,
        configured_k=10_000,
        max_num_seqs=10_000,
        max_num_batched_tokens=100_010_000,
        max_model_len=20_000,
        target_logits_dtype=torch.bfloat16,
        draft_logits_dtype=torch.bfloat16,
    )

    registry = build_draft_route_registry(plan, enforce_eager=False)

    assert registry.graph_buckets[-1] == MAX_CUDA_GRAPH_BATCH_SIZE
    assert max(key.effective_k for key in registry.router_admitted_keys) == (
        MAX_DRAFT_ROUTE_EFFECTIVE_K
    )
    assert len(registry.entries) == (
        2
        * len(draft_graph_batch_buckets(MAX_CUDA_GRAPH_BATCH_SIZE))
        * MAX_DRAFT_ROUTE_EFFECTIVE_K
    )


def test_registry_prunes_structurally_unreachable_family_and_k_endpoints():
    token_limited = plan_speculative_workspace(
        vocab_size=17,
        configured_k=2,
        max_num_seqs=1,
        max_num_batched_tokens=3,
        max_model_len=4,
        target_logits_dtype=torch.bfloat16,
        draft_logits_dtype=torch.bfloat16,
    )
    token_registry = build_draft_route_registry(
        token_limited, enforce_eager=True
    )
    assert {
        (key.effective_k, key.catchup_family)
        for key in token_registry.router_admitted_keys
    } == {
        (1, DraftCatchupFamily.NONE),
        (1, DraftCatchupFamily.PAGED_EAGER_DYNAMIC),
        (2, DraftCatchupFamily.NONE),
    }

    model_limited = plan_speculative_workspace(
        vocab_size=17,
        configured_k=5,
        max_num_seqs=1,
        max_num_batched_tokens=10,
        max_model_len=3,
        target_logits_dtype=torch.bfloat16,
        draft_logits_dtype=torch.bfloat16,
    )
    model_registry = build_draft_route_registry(
        model_limited, enforce_eager=True
    )
    assert {
        key.effective_k for key in model_registry.router_admitted_keys
    } == {1}


def test_max_eligible_catchup_counts_aggregate_rows_not_one_context():
    assert max_eligible_draft_catchup(
        max_num_batched_tokens=16_384,
        max_model_len=4_096,
        batch_cap=512,
    ) == (16_376, 4)


@pytest.mark.parametrize(
    "corruption",
    ("batch_capacity", "q_bytes", "proposal_id_bytes"),
)
def test_malformed_workspace_certificate_is_never_ready(corruption):
    desired = build_draft_route_registry(
        _memory_plan(), enforce_eager=True
    )
    entry = desired.entries[0]
    if corruption == "batch_capacity":
        certificate = replace(
            entry.workspace,
            batch_capacity=entry.workspace.batch_capacity + 1,
        )
    elif corruption == "q_bytes":
        certificate = replace(
            entry.workspace,
            q_bytes=entry.workspace.modeled_draft_live_bytes + 1,
        )
    else:
        certificate = replace(
            entry.workspace,
            proposal_id_bytes=entry.workspace.proposal_id_bytes + 1,
        )
    corrupted = DraftRouteRegistry(
        schema=desired.schema,
        plan_fingerprint=desired.plan_fingerprint,
        entries=(replace(entry, workspace=certificate), *desired.entries[1:]),
        warmed_components=desired.desired_warm_components,
    )

    assert entry.key not in corrupted.workspace_certified_keys
    assert entry.key not in corrupted.ready_keys


@pytest.mark.parametrize("enforce_eager", [False, True])
def test_every_published_route_is_workspace_certified_and_pretouched(
    enforce_eager,
):
    desired = build_draft_route_registry(
        _memory_plan(), enforce_eager=enforce_eager
    )
    ready = desired.with_warmed_components(
        desired.desired_warm_components
    )

    assert ready.router_admitted_keys == ready.workspace_certified_keys
    assert ready.router_admitted_keys == ready.warm_capture_keys
    assert ready.router_admitted_keys == ready.ready_keys
    for entry in ready.entries:
        certificate = entry.workspace
        assert certificate.key == entry.key
        assert certificate.plan_fingerprint == ready.plan_fingerprint
        assert certificate.proposal_id_bytes == (
            entry.key.batch_bucket
            * entry.key.effective_k
            * INT64_BYTES
        )
        assert (
            certificate.q_bytes + certificate.proposal_id_bytes
            <= certificate.modeled_draft_live_bytes
        )
        assert (
            certificate.modeled_draft_live_bytes
            <= certificate.reserved_plan_bytes
        )


def test_registry_does_not_publish_partial_warm_readiness():
    registry = build_draft_route_registry(
        _memory_plan(), enforce_eager=False
    )
    omitted = next(iter(registry.desired_warm_components))
    partial = registry.with_warmed_components(
        registry.desired_warm_components - {omitted}
    )

    assert partial.warm_capture_keys < partial.router_admitted_keys
    assert partial.ready_keys < partial.router_admitted_keys


@pytest.mark.parametrize("catchup_tokens", [0, 1, 257])
def test_registry_resolves_exact_smallest_batch_bucket_and_contiguous_k(
    catchup_tokens,
):
    desired = build_draft_route_registry(
        _memory_plan(), enforce_eager=False
    )
    registry = desired.with_warmed_components(
        desired.desired_warm_components
    )

    admission = registry.resolve(
        batch_size=3, catchup_tokens=catchup_tokens
    )

    assert admission.batch_size == 3
    assert admission.catchup_tokens == catchup_tokens
    assert admission.max_effective_k == 3
    assert tuple(key.effective_k for key in admission.route_keys) == (1, 2, 3)
    assert all(key.batch_bucket == 4 for key in admission.route_keys)
    assert all(
        key.execution_mode is DraftExecutionMode.CUDA_GRAPH
        for key in admission.route_keys
    )
    expected_family = (
        DraftCatchupFamily.NONE
        if catchup_tokens == 0
        else DraftCatchupFamily.PAGED_EAGER_DYNAMIC
    )
    assert all(key.catchup_family is expected_family for key in admission.route_keys)
    assert all(
        key.sampler_envelope is DraftSamplerEnvelope.EXACT_WORST_CASE
        for key in admission.route_keys
    )
    assert registry.validate_runtime_key(
        admission.key_for(2),
        batch_size=3,
        effective_k=2,
        catchup_tokens=catchup_tokens,
    )


def test_unwarmed_or_out_of_capacity_route_is_a_pre_cuda_miss():
    desired = build_draft_route_registry(
        _memory_plan(), enforce_eager=False
    )
    assert desired.resolve(batch_size=1, catchup_tokens=0) is None

    ready = desired.with_warmed_components(
        desired.desired_warm_components
    )
    assert ready.resolve(batch_size=5, catchup_tokens=0) is None
    assert ready.resolve(batch_size=0, catchup_tokens=0) is None
    assert ready.resolve(batch_size=1, catchup_tokens=-1) is None


def test_tampered_key_never_validates_against_ready_registry():
    desired = build_draft_route_registry(
        _memory_plan(), enforce_eager=True
    )
    registry = desired.with_warmed_components(
        desired.desired_warm_components
    )
    admission = registry.resolve(batch_size=2, catchup_tokens=0)
    real = admission.key_for(1)
    tampered = DraftRouteKey(
        schema=real.schema,
        execution_mode=real.execution_mode,
        batch_bucket=real.batch_bucket,
        effective_k=real.effective_k,
        catchup_family=DraftCatchupFamily.PAGED_EAGER_DYNAMIC,
        sampler_envelope=real.sampler_envelope,
    )

    assert not registry.validate_runtime_key(
        tampered,
        batch_size=2,
        effective_k=1,
        catchup_tokens=0,
    )


def test_all_legal_sampler_compositions_share_one_conservative_envelope():
    desired = build_draft_route_registry(
        _memory_plan(), enforce_eager=False
    )
    registry = desired.with_warmed_components(
        desired.desired_warm_components
    )
    # Route lookup intentionally does not depend on these row compositions.
    # They document the complete legal structural domain mapped to one proved
    # dense/sequential worst-case sampler envelope.
    modes = (
        (0.0, -1, 1.0),
        (0.8, -1, 1.0),
        (0.8, 2, 1.0),
        (0.8, -1, 0.9),
        (0.8, 2, 0.9),
    )
    for batch_size in range(1, 5):
        compositions = (
            tuple(modes[index % len(modes)] for index in range(batch_size)),
            tuple(modes[0] for _ in range(batch_size)),
            tuple(modes[-1] for _ in range(batch_size)),
        )
        keys = []
        for composition in compositions:
            assert len(composition) == batch_size
            admission = registry.resolve(
                batch_size=batch_size, catchup_tokens=0
            )
            keys.append(admission.key_for(1))
        assert len(set(keys)) == 1
        assert keys[0].sampler_envelope is DraftSamplerEnvelope.EXACT_WORST_CASE


def _pretouch_runner(monkeypatch, *, fail_at=None):
    runner = object.__new__(ModelRunner)
    runner.speculative_memory_plan = _memory_plan()
    runner.draft_route_registry = build_draft_route_registry(
        runner.speculative_memory_plan,
        enforce_eager=True,
    )
    runner.enforce_eager = True
    runner._warmup_transient_bytes = 0
    runner._draft_route_pretouch_peak_bytes = 0
    runner.config = SimpleNamespace(
        max_num_batched_tokens=16,
        max_model_len=64,
    )
    calls = []

    def record(name):
        def operation(self, *args, **kwargs):
            calls.append((name, args, kwargs))
            torch.rand(1)
            set_context = __import__(
                "nanovllm.utils.context", fromlist=["set_context"]
            ).set_context
            set_context(False, slot_mapping=torch.tensor([99], dtype=torch.int32))
            if fail_at == name:
                raise RuntimeError(f"injected {name} failure")
        return operation

    runner._pretouch_draft_decode_witness = MethodType(record("decode"), runner)
    runner._pretouch_draft_catchup_witness = MethodType(
        record("catchup"), runner
    )
    runner._pretouch_exact_sampler_envelope = MethodType(
        record("sampler"), runner
    )
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(
        torch.cuda,
        "memory_stats",
        lambda: {"allocated_bytes.all.peak": 0},
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    return runner, calls


def test_route_pretouch_publishes_readiness_only_after_every_component(monkeypatch):
    runner, calls = _pretouch_runner(monkeypatch)

    runner._pretouch_draft_routes()

    assert [name for name, _, _ in calls] == [
        "decode",
        "decode",
        "decode",
        "catchup",
        "catchup",
        "sampler",
    ]
    assert (
        runner.draft_route_registry.ready_keys
        == runner.draft_route_registry.router_admitted_keys
    )


def test_graph_route_pretouch_replays_every_bucket_and_an_interior_shape(
    monkeypatch,
):
    runner, calls = _pretouch_runner(monkeypatch)
    runner.enforce_eager = False
    runner.draft_route_registry = build_draft_route_registry(
        runner.speculative_memory_plan,
        enforce_eager=False,
    )
    runner.draft_graphs = {
        bucket: object() for bucket in runner.draft_route_registry.graph_buckets
    }

    def graph_witness(self, batch_size, route_key):
        calls.append(("graph", (batch_size, route_key), {}))

    runner._pretouch_draft_graph_decode_witness = MethodType(
        graph_witness,
        runner,
    )

    runner._pretouch_draft_routes()

    graph_calls = [args for name, args, _ in calls if name == "graph"]
    assert [batch_size for batch_size, _ in graph_calls] == [1, 2, 3, 4]
    assert [route_key.batch_bucket for _, route_key in graph_calls] == [1, 2, 4, 4]
    assert (
        runner.draft_route_registry.ready_keys
        == runner.draft_route_registry.router_admitted_keys
    )


def test_route_pretouch_executes_max_aggregate_and_single_row_catchup(
    monkeypatch,
):
    runner, calls = _pretouch_runner(monkeypatch)
    runner.speculative_memory_plan = plan_speculative_workspace(
        vocab_size=17,
        configured_k=1,
        max_num_seqs=512,
        max_num_batched_tokens=16_384,
        max_model_len=4_096,
        target_logits_dtype=torch.bfloat16,
        draft_logits_dtype=torch.bfloat16,
    )
    runner.draft_route_registry = build_draft_route_registry(
        runner.speculative_memory_plan,
        enforce_eager=True,
    )
    runner.config = SimpleNamespace(
        max_num_batched_tokens=16_384,
        max_model_len=4_096,
    )

    runner._pretouch_draft_routes()

    catchup_kwargs = [kwargs for name, _, kwargs in calls if name == "catchup"]
    assert {
        (kwargs["total_tokens"], kwargs["num_seqs"], kwargs["high_context"])
        for kwargs in catchup_kwargs
    } == {
        (1, 1, False),
        (257, 257, True),
        (4_094, 1, True),
        (16_376, 4, True),
    }


def test_sampler_pretouch_covers_dense_and_near_dense_max_top_k():
    calls = []

    class RecordingSampler:
        def sample_exact_with_probabilities(self, logits, temperatures, **kwargs):
            calls.append((logits.shape, temperatures.clone(), kwargs))
            probabilities = kwargs["probabilities_out"]
            probabilities.fill_(1.0 / logits.size(1))
            return SimpleNamespace(
                token_ids=torch.zeros(logits.size(0), dtype=torch.int64),
                probabilities=probabilities,
            )

    runner = object.__new__(ModelRunner)
    runner.draft_kv_cache = torch.empty(1)
    runner.config = SimpleNamespace(
        draft_hf_config=SimpleNamespace(
            vocab_size=7,
            dtype=torch.float32,
        )
    )
    runner.sampler = RecordingSampler()

    runner._pretouch_exact_sampler_envelope(5)

    assert calls[0][2]["top_k_buckets"] == ((6, None),)
    near_dense_top_k, near_dense_rows = calls[1][2]["top_k_buckets"][0]
    assert near_dense_top_k == 6
    assert near_dense_rows.tolist() == [0, 1, 2, 3]
    top_p_rows, _ = calls[1][2]["top_p_plan"]
    assert top_p_rows.tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize("fail_at", ["decode", "catchup", "sampler"])
def test_route_pretouch_failure_does_not_publish_partial_readiness(
    monkeypatch,
    fail_at,
):
    runner, _ = _pretouch_runner(monkeypatch, fail_at=fail_at)
    desired = runner.draft_route_registry

    with pytest.raises(RuntimeError, match=f"injected {fail_at} failure"):
        runner._pretouch_draft_routes()

    assert runner.draft_route_registry is desired
    assert not runner.draft_route_registry.warm_capture_keys


def test_route_pretouch_wrapper_restores_rng_and_context(monkeypatch):
    from nanovllm.utils.context import get_context

    runner, _ = _pretouch_runner(monkeypatch)
    cuda_state = torch.tensor([7, 8, 9], dtype=torch.uint8)
    restored_cuda = []
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: cuda_state.clone())
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state",
        lambda state: restored_cuda.append(state.clone()),
    )
    torch.manual_seed(20260828)
    cpu_before = torch.random.get_rng_state().clone()

    runner._run_draft_phase(
        "test V3 route pretouch",
        runner._pretouch_draft_routes,
    )

    assert torch.equal(torch.random.get_rng_state(), cpu_before)
    assert len(restored_cuda) == 1
    assert torch.equal(restored_cuda[0], cuda_state)
    context = get_context()
    assert context.is_prefill is False
    assert context.slot_mapping is None



def test_workspace_fingerprint_binds_numerical_backend_identity():
    plan = _memory_plan()
    fast = speculative_plan_fingerprint(plan, "fast")
    invariant = speculative_plan_fingerprint(plan, "invariant")

    assert fast != invariant
    assert build_draft_route_registry(
        plan, enforce_eager=True, numerical_backend="invariant"
    ).plan_fingerprint == invariant
