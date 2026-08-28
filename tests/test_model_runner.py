from types import SimpleNamespace
import warnings

import pytest
import torch

import nanovllm.engine.model_runner as runner_module
from nanovllm.engine.model_runner import (
    KVCacheBindingError,
    ModelRunner,
    SpeculativeKVCacheCapacityError,
)


def _assert_cleanup_detail(error, seen_warnings, text):
    notes = getattr(error, "__notes__", ())
    warning_messages = tuple(str(item.message) for item in seen_warnings)
    assert any(text in note for note in notes) or any(
        text in message for message in warning_messages
    )


def test_prepare_sample_detects_all_greedy_on_the_host():
    runner = object.__new__(ModelRunner)
    runner.config = SimpleNamespace(hf_config=SimpleNamespace(vocab_size=100))
    seqs = [SimpleNamespace(temperature=0.0) for _ in range(4)]

    temperatures, top_k_buckets, top_p_plan, all_greedy = runner.prepare_sample(seqs)

    assert temperatures is None
    assert top_k_buckets == ()
    assert top_p_plan is None
    assert all_greedy is True


def test_partial_constructor_failure_restores_defaults_and_process_group(
    monkeypatch,
):
    class BuildError(RuntimeError):
        pass

    config = SimpleNamespace(
        hf_config=SimpleNamespace(dtype=torch.bfloat16),
        kvcache_block_size=256,
        enforce_eager=True,
        tensor_parallel_size=1,
    )
    devices = []
    dtypes = []
    destroyed = []
    monkeypatch.setattr(torch, "get_default_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch, "get_default_dtype", lambda: torch.float32)
    monkeypatch.setattr(torch, "set_default_device", devices.append)
    monkeypatch.setattr(torch, "set_default_dtype", dtypes.append)
    monkeypatch.setattr(torch.cuda, "set_device", lambda rank: None)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(runner_module.dist, "init_process_group", lambda *a, **k: None)
    monkeypatch.setattr(runner_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        runner_module.dist,
        "destroy_process_group",
        lambda: destroyed.append(True),
    )
    monkeypatch.setattr(
        runner_module,
        "Qwen3ForCausalLM",
        lambda hf_config: (_ for _ in ()).throw(BuildError("injected")),
    )

    with pytest.raises(BuildError, match="injected"):
        ModelRunner(config, 0, [])

    assert devices == ["cuda", torch.device("cpu")]
    assert dtypes == [torch.bfloat16, torch.float32]
    assert destroyed == [True]


def test_close_releases_cuda_owners_and_is_idempotent(monkeypatch):
    class FakeGraph:
        def __init__(self):
            self.resets = 0

        def reset(self):
            self.resets += 1

    class CacheModule:
        k_cache = object()
        v_cache = object()

    cache_module = CacheModule()
    draft_cache_module = CacheModule()
    model = SimpleNamespace(modules=lambda: [cache_module])
    draft_model = SimpleNamespace(modules=lambda: [draft_cache_module])
    runner = object.__new__(ModelRunner)
    runner._closed = False
    runner._owns_process_group = False
    runner.shm = None
    runner.rank = 0
    graphs = [FakeGraph() for _ in range(3)]
    for name, value in {
        "model": model,
        "draft_model": draft_model,
        "sampler": object(),
        "kv_cache": object(),
        "draft_kv_cache": object(),
        "graphs": {1: graphs[0]},
        "draft_graphs": {1: graphs[1]},
        "graph_vars": object(),
        "draft_graph_vars": object(),
        "graph_pool": object(),
        "draft_graph_pool": object(),
        "draft_graph_bs": object(),
        "varlen_graphs": {(128, 1): graphs[2]},
        "varlen_vars": object(),
        "speculative_memory_plan": object(),
    }.items():
        setattr(runner, name, value)
    resets = []
    collections = []
    rope_cache_clears = []
    monkeypatch.setattr(runner_module, "reset_context", lambda: resets.append(True))
    monkeypatch.setattr(
        runner_module,
        "_clear_rope_cache",
        lambda: rope_cache_clears.append(True),
    )
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(runner_module.gc, "collect", lambda: collections.append(True))

    assert runner._close(abort=True) is None
    assert runner._close(abort=True) is None
    assert cache_module.k_cache is None and cache_module.v_cache is None
    assert (
        draft_cache_module.k_cache is None
        and draft_cache_module.v_cache is None
    )
    for name in (
        "model",
        "draft_model",
        "sampler",
        "kv_cache",
        "draft_kv_cache",
        "graphs",
        "draft_graphs",
        "graph_vars",
        "draft_graph_vars",
        "graph_pool",
        "draft_graph_pool",
        "draft_graph_bs",
        "varlen_graphs",
        "varlen_vars",
        "speculative_memory_plan",
    ):
        assert not hasattr(runner, name)
    assert resets == [True]
    assert collections == [True]
    assert rope_cache_clears == [True]
    assert [graph.resets for graph in graphs] == [1, 1, 1]


def test_draft_phase_restores_cpu_and_cuda_rng_after_success(monkeypatch):
    runner = object.__new__(ModelRunner)
    restored = []
    cpu_state = object()
    cuda_state = object()
    monkeypatch.setattr(torch.random, "get_rng_state", lambda: cpu_state)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: cuda_state)
    monkeypatch.setattr(
        torch.random,
        "set_rng_state",
        lambda state: restored.append(("cpu", state)),
    )
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state",
        lambda state: restored.append(("cuda", state)),
    )

    assert runner._run_draft_phase("test phase", lambda: 17) == 17
    assert restored == [("cpu", cpu_state), ("cuda", cuda_state)]


def test_draft_phase_restores_rng_without_masking_primary_failure(monkeypatch):
    class PhaseError(RuntimeError):
        pass

    runner = object.__new__(ModelRunner)
    restored = []
    monkeypatch.setattr(torch.random, "get_rng_state", lambda: "cpu-before")
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: "cuda-before")
    monkeypatch.setattr(
        torch.random,
        "set_rng_state",
        lambda state: restored.append(("cpu", state)),
    )
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state",
        lambda state: restored.append(("cuda", state)),
    )

    def fail():
        raise PhaseError("primary")

    with pytest.raises(PhaseError, match="primary"):
        runner._run_draft_phase("test phase", fail)
    assert restored == [
        ("cpu", "cpu-before"),
        ("cuda", "cuda-before"),
    ]


def test_draft_model_construction_uses_and_restores_explicit_dtype(
    monkeypatch,
):
    runner = object.__new__(ModelRunner)
    runner.config = SimpleNamespace(
        draft_hf_config=SimpleNamespace(dtype=torch.float16)
    )
    current_dtype = torch.float32
    transitions = []

    def set_dtype(dtype):
        nonlocal current_dtype
        current_dtype = dtype
        transitions.append(dtype)

    marker = object()
    monkeypatch.setattr(torch, "get_default_dtype", lambda: current_dtype)
    monkeypatch.setattr(torch, "set_default_dtype", set_dtype)
    monkeypatch.setattr(
        runner_module,
        "Qwen3ForCausalLM",
        lambda config: marker,
    )

    runner._construct_draft_model()

    assert runner.draft_model is marker
    assert transitions == [torch.float16, torch.float32]
    assert current_dtype == torch.float32


@pytest.mark.parametrize(
    "failure_phase",
    (
        "draft_construct",
        "draft_load",
        "draft_warmup",
        "route_registry",
        "graph_profile",
        "joint_allocate",
        "draft_graph",
        "draft_pretouch",
        "route_pretouch",
        "memory_finalize",
    ),
)
def test_each_draft_constructor_phase_rolls_back_one_runner_transaction(
    monkeypatch,
    failure_phase,
):
    class InjectedError(RuntimeError):
        pass

    graph_mode = failure_phase in {
        "graph_profile",
        "draft_graph",
        "draft_pretouch",
    }
    target_config = SimpleNamespace(dtype=torch.bfloat16)
    draft_config = SimpleNamespace(dtype=torch.float16)
    config = SimpleNamespace(
        hf_config=target_config,
        draft_hf_config=draft_config,
        kvcache_block_size=256,
        enforce_eager=not graph_mode,
        tensor_parallel_size=1,
        speculation_enabled=True,
        model="target",
        draft_model="draft",
    )
    models = []

    def build_model(hf_config):
        if failure_phase == "draft_construct" and hf_config is draft_config:
            raise InjectedError("draft_construct")
        model = _fake_model(0)
        models.append(model)
        return model

    load_calls = []

    def fake_load(model, path):
        load_calls.append(path)
        if failure_phase == "draft_load" and path == "draft":
            raise InjectedError("draft_load")

    dtype = torch.float32

    def set_dtype(value):
        nonlocal dtype
        dtype = value

    rng_restores = []
    destroyed = []
    resets = []
    monkeypatch.setattr(torch, "get_default_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch, "get_default_dtype", lambda: dtype)
    monkeypatch.setattr(torch, "set_default_device", lambda value: None)
    monkeypatch.setattr(torch, "set_default_dtype", set_dtype)
    monkeypatch.setattr(torch.cuda, "set_device", lambda rank: None)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: "cuda-state")
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state",
        lambda state: rng_restores.append(("cuda", state)),
    )
    monkeypatch.setattr(torch.random, "get_rng_state", lambda: "cpu-state")
    monkeypatch.setattr(
        torch.random,
        "set_rng_state",
        lambda state: rng_restores.append(("cpu", state)),
    )
    monkeypatch.setattr(runner_module.dist, "init_process_group", lambda *a, **k: None)
    monkeypatch.setattr(runner_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        runner_module.dist,
        "destroy_process_group",
        lambda: destroyed.append(True),
    )
    monkeypatch.setattr(runner_module, "Qwen3ForCausalLM", build_model)
    monkeypatch.setattr(runner_module, "load_model", fake_load)
    monkeypatch.setattr(runner_module, "Sampler", lambda: object())
    monkeypatch.setattr(runner_module, "reset_context", lambda: resets.append(True))
    monkeypatch.setattr(ModelRunner, "warmup_model", lambda self: None)
    monkeypatch.setattr(
        ModelRunner,
        "_profiled_transient_bytes",
        lambda self: 0,
    )

    def draft_warmup(self):
        if failure_phase == "draft_warmup":
            raise InjectedError("draft_warmup")

    def graph_profile(self):
        if failure_phase == "graph_profile":
            raise InjectedError("graph_profile")

    def initialize_routes(self):
        if failure_phase == "route_registry":
            raise InjectedError("route_registry")
        self.speculative_memory_plan = object()
        self.draft_route_registry = object()

    def allocate(self):
        self.kv_cache = object()
        self.draft_kv_cache = object()
        if failure_phase == "joint_allocate":
            for model in (self.model, self.draft_model):
                for module in model.cache_modules:
                    module.k_cache = object()
                    module.v_cache = object()
            raise InjectedError("joint_allocate")

    def capture_target(self):
        self.graphs = object()
        self.graph_pool = object()
        self.graph_vars = object()

    def capture_varlen(self):
        self.varlen_graphs = object()
        self.varlen_vars = object()

    def capture_draft(self):
        if failure_phase == "draft_graph":
            raise InjectedError("draft_graph")
        self.draft_graphs = object()
        self.draft_graph_pool = object()
        self.draft_graph_vars = object()

    def pretouch_draft(self):
        if failure_phase == "draft_pretouch":
            raise InjectedError("draft_pretouch")

    def pretouch_routes(self):
        assert dtype == torch.float32
        assert hasattr(self, "kv_cache")
        assert hasattr(self, "draft_kv_cache")
        assert self.speculative_memory_audit is None
        if failure_phase == "route_pretouch":
            raise InjectedError("route_pretouch")

    monkeypatch.setattr(ModelRunner, "warmup_draft_model", draft_warmup)
    monkeypatch.setattr(
        ModelRunner,
        "_initialize_speculative_route_registry",
        initialize_routes,
    )
    monkeypatch.setattr(
        ModelRunner,
        "_profile_speculative_graph_memory",
        graph_profile,
    )
    monkeypatch.setattr(ModelRunner, "allocate_kv_cache", allocate)
    monkeypatch.setattr(ModelRunner, "capture_cudagraph", capture_target)
    monkeypatch.setattr(ModelRunner, "capture_varlen_graphs", capture_varlen)
    monkeypatch.setattr(ModelRunner, "capture_draft_cudagraph", capture_draft)
    monkeypatch.setattr(
        ModelRunner,
        "_record_final_graph_memory_peaks",
        lambda self: None,
    )
    monkeypatch.setattr(ModelRunner, "_pretouch_eager_prefill", lambda self: None)
    monkeypatch.setattr(
        ModelRunner,
        "_pretouch_draft_eager_prefill",
        pretouch_draft,
    )
    monkeypatch.setattr(
        ModelRunner,
        "_pretouch_draft_routes",
        pretouch_routes,
    )
    monkeypatch.setattr(
        ModelRunner,
        "_finalize_speculative_memory_audit",
        lambda self: (
            (_ for _ in ()).throw(InjectedError("memory_finalize"))
            if failure_phase == "memory_finalize"
            else None
        ),
    )

    with pytest.raises(InjectedError, match=failure_phase):
        ModelRunner(config, 0, [])

    assert len(models) == (1 if failure_phase == "draft_construct" else 2)
    assert destroyed == [True]
    assert resets == [True]
    assert dtype == torch.float32
    assert ("cpu", "cpu-state") in rng_restores
    assert ("cuda", "cuda-state") in rng_restores
    for model in models:
        for module in model.cache_modules:
            assert module.k_cache is None and module.v_cache is None


def test_draft_warmup_resets_context_when_model_execution_fails(monkeypatch):
    class DraftError(RuntimeError):
        pass

    class FailingDraft:
        def __call__(self, input_ids, positions):
            raise DraftError("draft warmup")

    runner = object.__new__(ModelRunner)
    runner.config = SimpleNamespace(
        max_num_batched_tokens=2,
        max_model_len=2,
        max_num_seqs=1,
    )
    runner.draft_model = FailingDraft()
    runner.prepare_prefill = lambda seqs: (object(), object())
    resets = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(runner_module, "reset_context", lambda: resets.append(True))

    with pytest.raises(DraftError, match="draft warmup"):
        runner.warmup_draft_model()

    assert resets == [True]


def test_draft_warmup_does_not_consume_a_public_sequence_id(monkeypatch):
    class Draft:
        def __call__(self, input_ids, positions):
            return "hidden"

        def compute_logits(self, hidden_states):
            assert hidden_states == "hidden"

    runner = object.__new__(ModelRunner)
    runner.config = SimpleNamespace(
        max_num_batched_tokens=4,
        max_model_len=4,
        max_num_seqs=1,
    )
    runner.draft_model = Draft()
    captured = []

    def prepare_prefill(seqs):
        captured.extend(seqs)
        return object(), object()

    runner.prepare_prefill = prepare_prefill
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(runner_module, "reset_context", lambda: None)
    monkeypatch.setattr(
        runner_module,
        "Sequence",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("draft warmup allocated a public Sequence")
        ),
    )

    runner.warmup_draft_model()

    assert len(captured) == 1
    assert isinstance(captured[0], runner_module.ScheduledSequence)
    assert captured[0].scheduled_token_ids == (0, 0, 0, 0)
    assert captured[0].num_scheduled_tokens == 4


def test_draft_pretouch_resets_context_when_model_execution_fails(monkeypatch):
    class DraftError(RuntimeError):
        pass

    class FailingDraft:
        def __call__(self, input_ids, positions):
            raise DraftError("draft pretouch")

    runner = object.__new__(ModelRunner)
    runner.config = SimpleNamespace(
        max_num_batched_tokens=4,
        max_model_len=4,
    )
    runner.draft_model = FailingDraft()
    resets = []
    real_arange = torch.arange
    real_zeros = torch.zeros
    real_full = torch.full
    monkeypatch.setattr(
        torch,
        "arange",
        lambda *args, **kwargs: real_arange(
            *args,
            **{key: value for key, value in kwargs.items() if key != "device"},
        ),
    )
    monkeypatch.setattr(
        torch,
        "zeros",
        lambda *args, **kwargs: real_zeros(
            *args,
            **{key: value for key, value in kwargs.items() if key != "device"},
        ),
    )
    monkeypatch.setattr(
        torch,
        "full",
        lambda *args, **kwargs: real_full(
            *args,
            **{key: value for key, value in kwargs.items() if key != "device"},
        ),
    )
    monkeypatch.setattr(runner_module, "set_context", lambda *a, **k: None)
    monkeypatch.setattr(runner_module, "reset_context", lambda: resets.append(True))

    with pytest.raises(DraftError, match="draft pretouch"):
        runner._pretouch_draft_eager_prefill()

    assert resets == [True]


def test_strict_kv_binding_validates_every_layer_before_mutation():
    class CacheModule:
        k_cache = "original-k"
        v_cache = "original-v"

    modules = [CacheModule()]
    model = SimpleNamespace(modules=lambda: modules)
    config = SimpleNamespace(num_hidden_layers=2)

    with pytest.raises(KVCacheBindingError, match="target exposes 1"):
        ModelRunner._bind_kv_cache(
            model,
            object(),
            config,
            model_name="target",
        )

    assert modules[0].k_cache == "original-k"
    assert modules[0].v_cache == "original-v"


def test_default_restore_failure_does_not_mask_constructor_error(monkeypatch):
    class BuildError(RuntimeError):
        pass

    class RestoreError(RuntimeError):
        pass

    config = SimpleNamespace(
        hf_config=SimpleNamespace(dtype=torch.bfloat16),
        kvcache_block_size=256,
        enforce_eager=True,
        tensor_parallel_size=1,
    )
    restored_dtypes = []

    def set_device(device):
        if device == torch.device("cpu"):
            raise RestoreError("injected device restore failure")

    monkeypatch.setattr(torch, "get_default_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch, "get_default_dtype", lambda: torch.float32)
    monkeypatch.setattr(torch, "set_default_device", set_device)
    monkeypatch.setattr(torch, "set_default_dtype", restored_dtypes.append)
    monkeypatch.setattr(torch.cuda, "set_device", lambda rank: None)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(runner_module.dist, "init_process_group", lambda *a, **k: None)
    monkeypatch.setattr(runner_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(runner_module.dist, "destroy_process_group", lambda: None)
    monkeypatch.setattr(
        runner_module,
        "Qwen3ForCausalLM",
        lambda hf_config: (_ for _ in ()).throw(BuildError("original build error")),
    )

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always", RuntimeWarning)
        with pytest.raises(BuildError, match="original build error") as exc_info:
            ModelRunner(config, 0, [])

    assert restored_dtypes == [torch.bfloat16, torch.float32]
    _assert_cleanup_detail(exc_info.value, seen, "device restore failure")


def _allocation_runner(num_blocks):
    runner = object.__new__(ModelRunner)
    runner.block_size = 256
    runner.world_size = 1
    runner.model = SimpleNamespace(modules=lambda: [])
    runner.config = SimpleNamespace(
        hf_config=SimpleNamespace(
            num_key_value_heads=1,
            hidden_size=1,
            num_attention_heads=1,
            num_hidden_layers=1,
            dtype=torch.float16,
        ),
        gpu_memory_utilization=1.0,
        num_kvcache_blocks=num_blocks,
    )
    return runner


class _CacheModule:
    def __init__(self):
        self.k_cache = "unbound-k"
        self.v_cache = "unbound-v"


class _FakeAllocation:
    def __init__(self, shape, dtype, device):
        self.shape = shape
        self.dtype = dtype
        self.device = device

    def __getitem__(self, key):
        return (self, key)


def _fake_model(num_layers):
    modules = [_CacheModule() for _ in range(num_layers)]
    return SimpleNamespace(
        modules=lambda: modules,
        parameters=lambda: (),
        cache_modules=modules,
    )


def _spec_allocation_runner(num_blocks=-1, *, enforce_eager=True):
    runner = object.__new__(ModelRunner)
    runner.block_size = 256
    runner.world_size = 1
    runner.enforce_eager = enforce_eager
    runner.speculation_enabled = True
    runner.model = _fake_model(1)
    runner.draft_model = _fake_model(2)
    target_config = SimpleNamespace(
        num_key_value_heads=1,
        num_attention_heads=1,
        num_hidden_layers=1,
        head_dim=1,
        dtype=torch.float16,
        vocab_size=1,
    )
    draft_config = SimpleNamespace(
        num_key_value_heads=1,
        num_attention_heads=1,
        num_hidden_layers=2,
        head_dim=2,
        dtype=torch.float32,
    )
    runner.config = SimpleNamespace(
        hf_config=target_config,
        draft_hf_config=draft_config,
        configured_k=1,
        max_num_seqs=1,
        max_num_batched_tokens=2,
        max_model_len=2,
        gpu_memory_utilization=1.0,
        num_kvcache_blocks=num_blocks,
    )
    runner.speculative_memory_plan = None
    runner.draft_route_registry = None
    runner.speculative_memory_audit = None
    runner._speculative_memory_audit_inputs = None
    runner._target_warmup_transient_bytes = 100
    runner._draft_warmup_transient_bytes = 200
    runner._warmup_transient_bytes = 200
    runner._profiled_graph_allocated_bytes = 0
    runner._profiled_graph_reserved_bytes = 0
    runner._profiled_graph_peak_allocated_bytes = 0
    runner._profiled_graph_peak_reserved_bytes = 0
    runner._final_graph_allocated_baseline = 0
    runner._final_graph_reserved_baseline = 0
    runner._allocated_after_graph_before_pretouch = 0
    runner._reserved_after_graph_before_pretouch = 0
    runner._final_graph_peak_allocated_bytes = 0
    runner._final_graph_peak_reserved_bytes = 0
    return runner


def _patch_spec_allocation_cuda(monkeypatch, *, free, total):
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (free, total))
    monkeypatch.setattr(
        torch.cuda,
        "memory_stats",
        lambda: {
            "allocated_bytes.all.peak": 0,
            "allocated_bytes.all.current": 0,
            "reserved_bytes.all.peak": 0,
        },
    )
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 0)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)


def test_model_parameter_bytes_counts_shared_physical_storage_once():
    shared = torch.arange(8, dtype=torch.float32)
    first_view = torch.nn.Parameter(shared[:6])
    second_view = torch.nn.Parameter(shared[2:])
    independent = torch.nn.Parameter(torch.ones(3, dtype=torch.float16))
    model = SimpleNamespace(
        parameters=lambda: (first_view, second_view, independent)
    )

    assert ModelRunner._model_parameter_bytes(model) == 8 * 4 + 3 * 2


def test_explicit_kv_block_override_is_honored(monkeypatch):
    runner = _allocation_runner(3)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (4096, 4096))
    monkeypatch.setattr(
        torch.cuda,
        "memory_stats",
        lambda: {
            "allocated_bytes.all.peak": 0,
            "allocated_bytes.all.current": 0,
        },
    )
    allocation = object()
    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: allocation)

    runner.allocate_kv_cache()

    assert runner.config.num_kvcache_blocks == 3
    assert runner.kv_cache is allocation


def test_explicit_kv_block_override_above_budget_is_rejected(monkeypatch):
    runner = _allocation_runner(5)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (4096, 4096))
    monkeypatch.setattr(
        torch.cuda,
        "memory_stats",
        lambda: {
            "allocated_bytes.all.peak": 0,
            "allocated_bytes.all.current": 0,
        },
    )

    with pytest.raises(RuntimeError, match="requested num_kvcache_blocks=5"):
        runner.allocate_kv_cache()


def test_speculative_auto_sizing_subtracts_workspace_and_uses_joint_geometry(
    monkeypatch,
):
    runner = _spec_allocation_runner()
    plan = runner._plan_speculative_memory()
    target_block = runner_module.kv_cache_block_bytes(
        runner.config.hf_config,
        block_size=runner.block_size,
    )
    draft_block = runner_module.kv_cache_block_bytes(
        runner.config.draft_hf_config,
        block_size=runner.block_size,
    )
    joint_block = target_block + draft_block
    total = plan.reservation_bytes + 200 + 4 * joint_block - 1
    _patch_spec_allocation_cuda(monkeypatch, free=total, total=total)
    allocations = []

    def fake_empty(*shape, dtype, device):
        allocation = _FakeAllocation(shape, dtype, device)
        allocations.append(allocation)
        return allocation

    monkeypatch.setattr(torch, "empty", fake_empty)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    runner.allocate_kv_cache()

    assert runner.config.num_kvcache_blocks == 3
    assert [allocation.shape for allocation in allocations] == [
        (2, 1, 3, 256, 1, 1),
        (2, 2, 3, 256, 1, 2),
    ]
    assert [allocation.dtype for allocation in allocations] == [
        torch.float16,
        torch.float32,
    ]
    assert all(allocation.device == "cuda" for allocation in allocations)
    assert runner._speculative_memory_audit_inputs["joint_block_bytes"] == (
        joint_block
    )

    runner._finalize_speculative_memory_audit()
    audit = runner.speculative_memory_audit
    assert audit.workspace_plan == plan
    assert audit.target_warmup_transient_bytes == 100
    assert audit.draft_warmup_transient_bytes == 200
    assert audit.graph_allocator_margin_bytes == 0
    assert audit.graph_construction_reservation_bytes == 0
    assert audit.graph_reservation_bytes == 0
    assert audit.runtime_reservation_bytes == plan.reservation_bytes + 200
    assert audit.sizing_overhead_bytes == audit.runtime_reservation_bytes
    assert audit.reserved_before_kv_bytes == 0
    assert audit.allocated_after_kv_before_graph_bytes == 0
    assert audit.reserved_after_kv_before_graph_bytes == 0
    assert audit.target_kv_bytes == 3 * target_block
    assert audit.draft_kv_bytes == 3 * draft_block
    assert audit.gpu_certified is False
    with pytest.raises(AttributeError):
        audit.gpu_certified = True


@pytest.mark.parametrize(("requested", "passes"), [(3, True), (4, False)])
def test_speculative_explicit_override_has_exact_joint_boundary(
    monkeypatch,
    requested,
    passes,
):
    runner = _spec_allocation_runner(requested)
    plan = runner._plan_speculative_memory()
    joint_block = (
        runner_module.kv_cache_block_bytes(
            runner.config.hf_config,
            block_size=runner.block_size,
        )
        + runner_module.kv_cache_block_bytes(
            runner.config.draft_hf_config,
            block_size=runner.block_size,
        )
    )
    total = plan.reservation_bytes + 200 + 3 * joint_block
    _patch_spec_allocation_cuda(monkeypatch, free=total, total=total)
    monkeypatch.setattr(
        torch,
        "empty",
        lambda *shape, dtype, device: _FakeAllocation(shape, dtype, device),
    )

    if passes:
        runner.allocate_kv_cache()
        assert runner.config.num_kvcache_blocks == 3
    else:
        with pytest.raises(
            SpeculativeKVCacheCapacityError,
            match="requested num_kvcache_blocks=4",
        ):
            runner.allocate_kv_cache()


def test_speculative_graph_sizing_prices_disjoint_capture_and_runtime_peaks(
    monkeypatch,
):
    runner = _spec_allocation_runner(enforce_eager=False)
    runner._profiled_graph_allocated_bytes = 10
    runner._profiled_graph_reserved_bytes = 20
    runner._profiled_graph_peak_allocated_bytes = 30
    runner._profiled_graph_peak_reserved_bytes = 40
    plan = runner._plan_speculative_memory()
    joint_block = (
        runner_module.kv_cache_block_bytes(
            runner.config.hf_config,
            block_size=runner.block_size,
        )
        + runner_module.kv_cache_block_bytes(
            runner.config.draft_hf_config,
            block_size=runner.block_size,
        )
    )
    margin = runner_module.SPECULATIVE_GRAPH_ALLOCATOR_MARGIN_BYTES
    graph_construction = 40 + margin
    runtime = 20 + 200 + plan.reservation_bytes
    sizing_overhead = max(graph_construction, runtime)
    total = sizing_overhead + 3 * joint_block
    _patch_spec_allocation_cuda(monkeypatch, free=total, total=total)
    monkeypatch.setattr(
        torch,
        "empty",
        lambda *shape, dtype, device: _FakeAllocation(shape, dtype, device),
    )

    runner.allocate_kv_cache()

    assert runner.config.num_kvcache_blocks == 3
    inputs = runner._speculative_memory_audit_inputs
    assert inputs["warmup_transient_bytes"] == 200
    assert inputs["profiled_graph_ownership_bytes"] == 20
    assert inputs["profiled_graph_peak_bytes"] == 40
    assert inputs["graph_allocator_margin_bytes"] == margin
    assert inputs["graph_construction_reservation_bytes"] == (
        graph_construction
    )
    assert inputs["graph_reservation_bytes"] == graph_construction
    assert inputs["runtime_reservation_bytes"] == runtime
    assert inputs["sizing_overhead_bytes"] == sizing_overhead
    assert sizing_overhead < graph_construction + runtime


def test_speculative_graph_margin_is_at_least_one_joint_block(monkeypatch):
    runner = _spec_allocation_runner(enforce_eager=False)
    plan = runner._plan_speculative_memory()
    per_model_block = 40 * 1024 * 1024
    joint_block = 2 * per_model_block
    blocks = iter((per_model_block, per_model_block))
    monkeypatch.setattr(
        runner_module,
        "kv_cache_block_bytes",
        lambda *args, **kwargs: next(blocks),
    )
    sizing_overhead = max(
        joint_block,
        200 + plan.reservation_bytes,
    )
    total = sizing_overhead + 2 * joint_block
    _patch_spec_allocation_cuda(monkeypatch, free=total, total=total)
    monkeypatch.setattr(
        torch,
        "empty",
        lambda *shape, dtype, device: _FakeAllocation(shape, dtype, device),
    )

    runner.allocate_kv_cache()

    inputs = runner._speculative_memory_audit_inputs
    assert runner.config.num_kvcache_blocks == 2
    assert inputs["joint_block_bytes"] == joint_block
    assert inputs["graph_allocator_margin_bytes"] == joint_block


def test_dual_cache_second_allocation_failure_rolls_back_target_binding(
    monkeypatch,
):
    runner = _spec_allocation_runner(1)
    target_allocation = _FakeAllocation((), torch.float16, "cuda")
    calls = 0

    def fail_second(*shape, dtype, device):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise MemoryError("injected draft allocation failure")
        return target_allocation

    monkeypatch.setattr(torch, "empty", fail_second)

    with pytest.raises(MemoryError, match="draft allocation"):
        runner._allocate_and_bind_dual_kv_caches(1)

    assert runner.kv_cache is None
    assert runner.draft_kv_cache is None
    for module in runner.model.cache_modules:
        assert module.k_cache is None and module.v_cache is None


def test_dual_cache_draft_bind_failure_rolls_back_both_models(monkeypatch):
    runner = _spec_allocation_runner(1)
    runner.draft_model = _fake_model(1)
    monkeypatch.setattr(
        torch,
        "empty",
        lambda *shape, dtype, device: _FakeAllocation(shape, dtype, device),
    )

    with pytest.raises(KVCacheBindingError, match="draft exposes 1"):
        runner._allocate_and_bind_dual_kv_caches(1)

    assert runner.kv_cache is None
    assert runner.draft_kv_cache is None
    for model in (runner.model, runner.draft_model):
        for module in model.cache_modules:
            assert module.k_cache is None and module.v_cache is None


def test_graph_memory_profile_releases_graphs_pools_buffers_and_caches(
    monkeypatch,
):
    runner = _spec_allocation_runner(enforce_eager=False)

    def allocate(_):
        runner.kv_cache = object()
        runner.draft_kv_cache = object()
        for model in (runner.model, runner.draft_model):
            for module in model.cache_modules:
                module.k_cache = object()
                module.v_cache = object()

    def capture_target():
        runner.graphs = object()
        runner.graph_pool = object()
        runner.graph_vars = object()

    def capture_varlen():
        runner.varlen_graphs = object()
        runner.varlen_vars = object()

    def capture_draft():
        runner.draft_graphs = object()
        runner.draft_graph_pool = object()
        runner.draft_graph_vars = object()

    runner._allocate_and_bind_dual_kv_caches = allocate
    runner.capture_cudagraph = capture_target
    runner.capture_varlen_graphs = capture_varlen
    runner.capture_draft_cudagraph = capture_draft
    allocated = iter((100, 160))
    reserved = iter((200, 280))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: next(allocated))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: next(reserved))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(
        torch.cuda,
        "memory_stats",
        lambda: {
            "allocated_bytes.all.peak": 190,
            "reserved_bytes.all.peak": 330,
        },
    )
    monkeypatch.setattr(runner_module.gc, "collect", lambda: None)

    runner._profile_speculative_graph_memory()

    assert runner._profiled_graph_allocated_bytes == 60
    assert runner._profiled_graph_reserved_bytes == 80
    assert runner._profiled_graph_peak_allocated_bytes == 90
    assert runner._profiled_graph_peak_reserved_bytes == 130
    assert runner.kv_cache is None and runner.draft_kv_cache is None
    for name in (
        "graphs",
        "graph_pool",
        "graph_vars",
        "varlen_graphs",
        "varlen_vars",
    ):
        assert not hasattr(runner, name)
    assert runner.draft_graphs is None
    assert runner.draft_graph_pool is None
    assert runner.draft_graph_vars is None
    for model in (runner.model, runner.draft_model):
        for module in model.cache_modules:
            assert module.k_cache is None and module.v_cache is None


def test_final_graph_peak_ledger_uses_after_kv_baseline(monkeypatch):
    runner = _spec_allocation_runner(enforce_eager=False)
    runner._final_graph_allocated_baseline = 100
    runner._final_graph_reserved_baseline = 200
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 145)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 275)
    monkeypatch.setattr(
        torch.cuda,
        "memory_stats",
        lambda: {
            "allocated_bytes.all.peak": 160,
            "reserved_bytes.all.peak": 290,
        },
    )

    runner._record_final_graph_memory_peaks()

    assert runner._allocated_after_graph_before_pretouch == 145
    assert runner._reserved_after_graph_before_pretouch == 275
    assert runner._final_graph_peak_allocated_bytes == 60
    assert runner._final_graph_peak_reserved_bytes == 90


def test_final_graph_audit_freezes_post_capture_before_pretouch(monkeypatch):
    runner = _spec_allocation_runner(enforce_eager=False)
    plan = runner._plan_speculative_memory()
    total = plan.reservation_bytes + 80 * 1024**2
    _patch_spec_allocation_cuda(monkeypatch, free=total, total=total)
    monkeypatch.setattr(
        torch,
        "empty",
        lambda *shape, dtype, device: _FakeAllocation(shape, dtype, device),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    runner.allocate_kv_cache()
    runner._allocated_after_graph_before_pretouch = 145
    runner._reserved_after_graph_before_pretouch = 275
    runner._final_graph_peak_allocated_bytes = 60
    runner._final_graph_peak_reserved_bytes = 90
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 150)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 300)

    runner._finalize_speculative_memory_audit()

    audit = runner.speculative_memory_audit
    assert audit.allocated_after_kv_before_graph_bytes == 0
    assert audit.reserved_after_kv_before_graph_bytes == 0
    assert audit.allocated_after_graph_before_pretouch_bytes == 145
    assert audit.reserved_after_graph_before_pretouch_bytes == 275
    assert audit.final_graph_allocated_bytes == 145
    assert audit.final_graph_reserved_bytes == 275
    assert audit.post_init_allocated_bytes == 150
    assert audit.post_init_reserved_bytes == 300


def test_final_memory_audit_releases_pretouch_cache_before_snapshot(
    monkeypatch,
):
    runner = _spec_allocation_runner(enforce_eager=True)
    plan = runner._plan_speculative_memory()
    total = plan.reservation_bytes + 200 + 4 * 4096
    _patch_spec_allocation_cuda(monkeypatch, free=total, total=total)
    monkeypatch.setattr(
        torch,
        "empty",
        lambda *shape, dtype, device: _FakeAllocation(shape, dtype, device),
    )
    runner.allocate_kv_cache()

    events = []
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda: events.append("synchronize"),
    )
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: events.append("empty_cache"),
    )

    def mem_get_info():
        events.append("mem_get_info")
        return total, total

    monkeypatch.setattr(torch.cuda, "mem_get_info", mem_get_info)

    runner._finalize_speculative_memory_audit()

    assert events[:3] == ["synchronize", "empty_cache", "mem_get_info"]


def test_final_memory_audit_does_not_charge_recyclable_pretouch_cache_twice(
    monkeypatch,
):
    runner = _spec_allocation_runner(enforce_eager=True)
    plan = runner._plan_speculative_memory()
    total = plan.reservation_bytes + 200 + 4 * 4096
    _patch_spec_allocation_cuda(monkeypatch, free=total, total=total)
    monkeypatch.setattr(
        torch,
        "empty",
        lambda *shape, dtype, device: _FakeAllocation(shape, dtype, device),
    )
    runner.allocate_kv_cache()

    released = False
    required = plan.reservation_bytes + runner._warmup_transient_bytes
    cached_free = required - 1
    released_free = required + 1

    def empty_cache():
        nonlocal released
        released = True

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda: (released_free if released else cached_free, total),
    )

    assert cached_free - required == -1
    runner._finalize_speculative_memory_audit()

    assert released is True
    assert runner.speculative_memory_audit.modeled_runtime_headroom_bytes == 1


def test_final_memory_audit_rejects_post_init_headroom_shortfall(monkeypatch):
    runner = _spec_allocation_runner(enforce_eager=True)
    plan = runner._plan_speculative_memory()
    runner.speculative_memory_plan = plan
    runner._speculative_memory_audit_inputs = {
        "total_memory_bytes": 1_000,
        "free_before_kv_bytes": 1_000,
        "used_before_kv_bytes": 0,
        "memory_budget_bytes": 900,
        "allocated_before_kv_bytes": 0,
        "peak_before_kv_bytes": 0,
        "warmup_transient_bytes": 200,
        "target_warmup_transient_bytes": 100,
        "draft_warmup_transient_bytes": 200,
        "target_weight_bytes": 0,
        "draft_weight_bytes": 0,
        "graph_allocator_margin_bytes": 0,
        "graph_construction_reservation_bytes": 0,
        "graph_reservation_bytes": 0,
        "target_block_bytes": 1,
        "draft_block_bytes": 1,
        "joint_block_bytes": 2,
        "selected_num_blocks": 1,
    }
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (100, 1_000))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 0)

    with pytest.raises(
        SpeculativeKVCacheCapacityError,
        match="does not retain.*runtime headroom",
    ):
        runner._finalize_speculative_memory_audit()


def test_final_graph_audit_retains_workspace_and_activation_headroom(
    monkeypatch,
):
    runner = _spec_allocation_runner(enforce_eager=False)
    plan = runner._plan_speculative_memory()
    runner.speculative_memory_plan = plan
    runner._speculative_memory_audit_inputs = {
        "total_memory_bytes": 1_000_000_000,
        "free_before_kv_bytes": 1_000_000_000,
        "used_before_kv_bytes": 0,
        "memory_budget_bytes": 900_000_000,
        "allocated_before_kv_bytes": 0,
        "peak_before_kv_bytes": 0,
        "warmup_transient_bytes": 300,
        "target_warmup_transient_bytes": 100,
        "draft_warmup_transient_bytes": 200,
        "target_weight_bytes": 0,
        "draft_weight_bytes": 0,
        "graph_allocator_margin_bytes": 64 * 1024 * 1024,
        "graph_construction_reservation_bytes": 64 * 1024 * 1024,
        "graph_reservation_bytes": 64 * 1024 * 1024,
        "target_block_bytes": 1,
        "draft_block_bytes": 1,
        "joint_block_bytes": 2,
        "selected_num_blocks": 1,
    }
    # Workspace alone fits by one byte; the maximum live model activation does
    # not, so graph mode must reject the apparently positive headroom.
    budget_headroom = plan.reservation_bytes + 199
    post_used = 900_000_000 - budget_headroom
    post_free = 1_000_000_000 - post_used
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda: (post_free, 1_000_000_000),
    )
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 0)

    with pytest.raises(
        SpeculativeKVCacheCapacityError,
        match="does not retain.*runtime headroom",
    ):
        runner._finalize_speculative_memory_audit()


def test_final_memory_audit_rejects_unpriced_graph_construction_peak(
    monkeypatch,
):
    runner = _spec_allocation_runner(enforce_eager=False)
    runner.speculative_memory_plan = runner._plan_speculative_memory()
    runner._speculative_memory_audit_inputs = {
        "total_memory_bytes": 1_000,
        "memory_budget_bytes": 900,
        "warmup_transient_bytes": 0,
        "graph_allocator_margin_bytes": 3,
        "profiled_graph_peak_bytes": 7,
        "graph_construction_reservation_bytes": 10,
    }
    runner._final_graph_peak_allocated_bytes = 9
    runner._final_graph_peak_reserved_bytes = 11
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (500, 1_000))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 0)

    with pytest.raises(
        SpeculativeKVCacheCapacityError,
        match="graph construction exceeded.*reservation",
    ):
        runner._finalize_speculative_memory_audit()


def test_prepare_sample_metadata_buckets_only_active_stochastic_rows():
    seqs = [
        SimpleNamespace(temperature=0.0, top_k=3, top_p=1.0),
        SimpleNamespace(temperature=0.8, top_k=-1, top_p=1.0),
        SimpleNamespace(temperature=1.0, top_k=5, top_p=1.0),
        SimpleNamespace(temperature=0.7, top_k=3, top_p=1.0),
        SimpleNamespace(temperature=0.9, top_k=5, top_p=1.0),
        SimpleNamespace(temperature=1.0, top_k=100, top_p=1.0),
        SimpleNamespace(temperature=1.0, top_k=500, top_p=1.0),
    ]

    (
        temperatures,
        top_k_buckets,
        top_p_plan,
        all_greedy,
    ) = ModelRunner._prepare_sample_metadata(seqs, vocab_size=100)

    assert temperatures == (0.0, 0.8, 1.0, 0.7, 0.9, 1.0, 1.0)
    assert top_k_buckets == ((3, (3,)), (5, (2, 4)))
    assert top_p_plan is None
    assert all_greedy is False


def test_prepare_sample_metadata_uses_homogeneous_all_rows_route():
    seqs = [
        SimpleNamespace(temperature=1.0, top_k=50, top_p=1.0)
        for _ in range(4)
    ]

    _, top_k_buckets, top_p_plan, all_greedy = ModelRunner._prepare_sample_metadata(
        seqs, vocab_size=100
    )

    assert top_k_buckets == ((50, None),)
    assert top_p_plan is None
    assert all_greedy is False


def test_prepare_sample_metadata_isolates_active_top_p_rows():
    seqs = [
        SimpleNamespace(temperature=0.0, top_k=-1, top_p=0.5),
        SimpleNamespace(temperature=1.0, top_k=-1, top_p=0.9),
        SimpleNamespace(temperature=0.8, top_k=-1, top_p=1.0),
        SimpleNamespace(temperature=0.7, top_k=-1, top_p=0.8),
    ]

    _, top_k_buckets, top_p_plan, all_greedy = ModelRunner._prepare_sample_metadata(
        seqs, vocab_size=100
    )

    assert top_k_buckets == ()
    assert top_p_plan == ((1, 3), (0.9, 0.8))
    assert all_greedy is False


def test_expand_top_ps_fills_inactive_rows_with_disabled_value():
    assert ModelRunner._expand_top_ps(
        4, ((1, 3), (0.9, 0.8))
    ) == (1.0, 0.9, 1.0, 0.8)
    assert ModelRunner._expand_top_ps(
        2, (None, (0.7, 0.9))
    ) == (0.7, 0.9)


def test_fast_metadata_handles_greedy_active_inactive_and_topk_only_rows():
    seqs = [
        SimpleNamespace(temperature=0.0, top_k=3, top_p=0.5),
        SimpleNamespace(temperature=0.6, top_k=-1, top_p=0.9),
        SimpleNamespace(temperature=0.7, top_k=-1, top_p=1.0),
        SimpleNamespace(temperature=0.8, top_k=5, top_p=1.0),
    ]

    _, top_k_buckets, top_p_plan, all_greedy = (
        ModelRunner._prepare_sample_metadata(seqs, vocab_size=100)
    )

    assert top_k_buckets == ((5, (3,)),)
    assert top_p_plan == ((1,), (0.9,))
    assert ModelRunner._expand_top_ps(4, top_p_plan) == (1.0, 0.9, 1.0, 1.0)
    assert all_greedy is False


def test_expand_top_ps_rejects_incoherent_metadata():
    with pytest.raises(ValueError, match="same length"):
        ModelRunner._expand_top_ps(4, ((1, 3), (0.9,)))
    with pytest.raises(ValueError, match="cover the batch"):
        ModelRunner._expand_top_ps(4, (None, (0.9,)))


def test_fast_top_p_dispatch_runs_after_existing_top_k_filter(monkeypatch):
    calls = []

    class FakeSampler:
        def filter_top_k(self, logits, row_indices, top_k):
            calls.append(("top_k", top_k, row_indices))
            return logits

        def sample_top_p_flashinfer(self, logits, temperatures, top_ps):
            calls.append(("top_p", temperatures.clone(), top_ps.clone()))
            return torch.tensor([2, 3])

    runner = object.__new__(ModelRunner)
    runner.rank = 0
    runner.config = SimpleNamespace(top_p_backend="flashinfer")
    runner.sampler = FakeSampler()
    runner.prepare_decode = lambda seqs: (torch.tensor([1]), torch.tensor([0]))
    runner.prepare_sample = lambda seqs: (
        torch.tensor([0.6, 0.8]),
        ((5, None),),
        (None, torch.tensor([0.9, 0.7])),
        False,
    )
    runner.run_model = lambda input_ids, positions, is_prefill: torch.zeros(2, 8)
    monkeypatch.setattr(
        "nanovllm.engine.model_runner.reset_context", lambda: None
    )

    assert runner.run([object(), object()], is_prefill=False) == [2, 3]
    assert calls[0] == ("top_k", 5, None)
    assert calls[1][0] == "top_p"


def test_flashinfer_config_without_active_top_p_uses_legacy_sampler(monkeypatch):
    calls = []

    class FakeSampler:
        def __call__(self, logits, temperatures):
            calls.append("legacy")
            return torch.tensor([4, 5])

        def sample_top_p_flashinfer(self, logits, temperatures, top_ps):
            raise AssertionError("fast top-p must not run without an active row")

    runner = object.__new__(ModelRunner)
    runner.rank = 0
    runner.config = SimpleNamespace(top_p_backend="flashinfer")
    runner.sampler = FakeSampler()
    runner.prepare_decode = lambda seqs: (torch.tensor([1]), torch.tensor([0]))
    runner.prepare_sample = lambda seqs: (
        torch.tensor([0.6, 0.8]),
        (),
        None,
        False,
    )
    runner.run_model = lambda input_ids, positions, is_prefill: torch.zeros(2, 8)
    monkeypatch.setattr(
        "nanovllm.engine.model_runner.reset_context", lambda: None
    )

    assert runner.run([object(), object()], is_prefill=False) == [4, 5]
    assert calls == ["legacy"]
