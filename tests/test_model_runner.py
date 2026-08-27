from types import SimpleNamespace
import warnings

import pytest
import torch

import nanovllm.engine.model_runner as runner_module
from nanovllm.engine.model_runner import ModelRunner


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
    class CacheModule:
        k_cache = object()
        v_cache = object()

    cache_module = CacheModule()
    model = SimpleNamespace(modules=lambda: [cache_module])
    runner = object.__new__(ModelRunner)
    runner._closed = False
    runner._owns_process_group = False
    runner.shm = None
    runner.rank = 0
    for name, value in {
        "model": model,
        "sampler": object(),
        "kv_cache": object(),
        "graphs": object(),
        "graph_vars": object(),
        "graph_pool": object(),
        "varlen_graphs": object(),
        "varlen_vars": object(),
    }.items():
        setattr(runner, name, value)
    resets = []
    collections = []
    monkeypatch.setattr(runner_module, "reset_context", lambda: resets.append(True))
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(runner_module.gc, "collect", lambda: collections.append(True))

    assert runner._close(abort=True) is None
    assert runner._close(abort=True) is None
    assert cache_module.k_cache is None and cache_module.v_cache is None
    for name in (
        "model",
        "sampler",
        "kv_cache",
        "graphs",
        "graph_vars",
        "graph_pool",
        "varlen_graphs",
        "varlen_vars",
    ):
        assert not hasattr(runner, name)
    assert resets == [True]
    assert collections == [True]


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
