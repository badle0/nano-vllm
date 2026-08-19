from types import SimpleNamespace

import pytest
import torch

from nanovllm.engine.model_runner import ModelRunner


def test_prepare_sample_detects_all_greedy_on_the_host():
    runner = object.__new__(ModelRunner)
    runner.config = SimpleNamespace(hf_config=SimpleNamespace(vocab_size=100))
    seqs = [SimpleNamespace(temperature=0.0) for _ in range(4)]

    temperatures, top_k_buckets, top_p_plan, all_greedy = runner.prepare_sample(seqs)

    assert temperatures is None
    assert top_k_buckets == ()
    assert top_p_plan is None
    assert all_greedy is True


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
