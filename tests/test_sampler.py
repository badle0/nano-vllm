import dataclasses
import importlib.util
import math
import pathlib
import pickle
import subprocess
import sys

import pytest
import torch
from transformers.generation.logits_process import (
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from nanovllm.sampling_params import SamplingParams

_sp = pathlib.Path(__file__).resolve().parents[1] / "nanovllm/layers/sampler.py"
_ss = importlib.util.spec_from_file_location("sampler", _sp)
_sm = importlib.util.module_from_spec(_ss); _ss.loader.exec_module(_sm)
Sampler = _sm.Sampler

def _filter_top_k(logits, *buckets):
    sampler = Sampler()
    for top_k, rows in buckets:
        row_indices = torch.tensor(rows, dtype=torch.int64, device=logits.device)
        logits = sampler.filter_top_k(logits, row_indices, top_k)
    return logits

def _filter_top_p(logits, temperatures, top_ps, rows=None):
    row_indices = (
        None
        if rows is None
        else torch.tensor(rows, dtype=torch.int64, device=logits.device)
    )
    probability_cutoffs = torch.tensor(
        [1.0 - float(top_p) for top_p in top_ps],
        dtype=torch.float32,
        device=logits.device,
    )
    return Sampler().filter_top_p(
        logits, temperatures, row_indices, probability_cutoffs
    )

def _transformers_top_p_support(logits, temperatures, top_ps):
    support = []
    input_ids = torch.zeros(1, 1, dtype=torch.int64, device=logits.device)
    for row, top_p in enumerate(top_ps):
        scores = logits[row:row + 1].float() / temperatures[row]
        warped = TopPLogitsWarper(float(top_p))(input_ids, scores)
        support.append(torch.isfinite(warped))
    return torch.cat(support, dim=0)

def _transformers_top_k_top_p_support(
    logits, temperatures, top_ks, top_ps
):
    support = []
    input_ids = torch.zeros(1, 1, dtype=torch.int64, device=logits.device)
    for row, (top_k, top_p) in enumerate(zip(top_ks, top_ps)):
        scores = logits[row:row + 1].float() / temperatures[row]
        scores = TopKLogitsWarper(int(top_k))(input_ids, scores)
        scores = TopPLogitsWarper(float(top_p))(input_ids, scores)
        support.append(torch.isfinite(scores))
    return torch.cat(support, dim=0)

def test_temperature_zero_is_permitted():
    SamplingParams(temperature=0.0)

def test_negative_temperature_rejected():
    with pytest.raises(ValueError):
        SamplingParams(temperature=-1.0)

@pytest.mark.parametrize("temperature", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_temperature_rejected(temperature):
    with pytest.raises(ValueError, match="finite"):
        SamplingParams(temperature=temperature)

def test_legacy_positional_arguments_keep_their_mapping():
    params = SamplingParams(0.6, 128, True)
    assert params.temperature == 0.6
    assert params.max_tokens == 128
    assert params.ignore_eos is True
    assert params.top_k == -1
    assert params.top_p == 1.0

def test_sampling_params_round_trip():
    params = SamplingParams(0.6, 128, True, 17, 0.9)
    assert dataclasses.asdict(params) == {
        "temperature": 0.6,
        "max_tokens": 128,
        "ignore_eos": True,
        "top_k": 17,
        "top_p": 0.9,
    }
    assert pickle.loads(pickle.dumps(params)) == params

@pytest.mark.parametrize("top_k", [True, 1.9, "5", None])
def test_topk_rejects_non_integer_values(top_k):
    with pytest.raises(TypeError):
        SamplingParams(top_k=top_k)

@pytest.mark.parametrize("top_p", [True, "0.9", None])
def test_topp_rejects_non_numeric_values(top_p):
    with pytest.raises(TypeError):
        SamplingParams(top_p=top_p)

@pytest.mark.parametrize(
    "top_p",
    [0.0, -0.1, 1.1, float("nan"), float("inf")],
)
def test_topp_rejects_values_outside_finite_unit_interval(top_p):
    with pytest.raises(ValueError):
        SamplingParams(top_p=top_p)

def test_validation_survives_optimized_python():
    code = """
from nanovllm.sampling_params import SamplingParams

invalid = (
    {"temperature": -1.0},
    {"temperature": float("nan")},
    {"temperature": float("inf")},
    {"top_k": True},
    {"top_k": 1.9},
    {"top_k": 0},
    {"top_p": True},
    {"top_p": float("nan")},
    {"top_p": 0.0},
    {"top_p": 1.1},
)
for kwargs in invalid:
    try:
        SamplingParams(**kwargs)
    except (TypeError, ValueError):
        continue
    raise SystemExit(f"accepted invalid parameters: {kwargs}")
"""
    subprocess.run(
        [sys.executable, "-O", "-c", code],
        cwd=pathlib.Path(__file__).resolve().parents[1],
        check=True,
    )

def test_greedy_is_argmax():
    logits = torch.randn(8, 1000, dtype=torch.bfloat16)
    out = Sampler().greedy(logits.clone())
    assert torch.equal(out, logits.argmax(-1))

def test_greedy_argmax_preserves_exact_tie_order():
    logits = torch.tensor([[1.0, 4.0, 4.0, 2.0]])
    assert Sampler().greedy(logits).item() == logits.argmax(-1).item() == 1

def test_greedy_does_not_advance_cpu_rng():
    logits = torch.arange(8000, dtype=torch.float32).reshape(8, 1000)
    sampler = Sampler()
    sampler.greedy(logits)
    state = torch.get_rng_state().clone()
    sampler.greedy(logits)
    assert torch.equal(torch.get_rng_state(), state)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_greedy_does_not_advance_cuda_rng():
    logits = torch.arange(8000, dtype=torch.float32, device="cuda").reshape(8, 1000)
    sampler = Sampler().cuda()
    sampler.greedy(logits)
    torch.cuda.synchronize()
    state = torch.cuda.get_rng_state().clone()
    sampler.greedy(logits)
    torch.cuda.synchronize()
    assert torch.equal(torch.cuda.get_rng_state(), state)

def test_mixed_batch_routes_per_row():
    logits = torch.randn(6, 1000, dtype=torch.bfloat16)
    temps = torch.tensor([0., 0.6, 0., 1.0, 0., 0.6])
    out = Sampler()(logits.clone(), temps)
    g = temps == 0
    assert torch.equal(out[g], logits.argmax(-1)[g])

def test_stochastic_path_is_not_constant():
    logits = torch.randn(1, 1000, dtype=torch.bfloat16)
    draws = {Sampler()(logits.clone(), torch.ones(1)).item() for _ in range(200)}
    assert len(draws) > 1

def _main_sampler(logits, temperatures):
    logits = logits.float().div_(temperatures.unsqueeze(dim=1))
    probs = torch.softmax(logits, dim=-1)
    return probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)

def test_stochastic_path_is_fixed_seed_equivalent_to_main():
    logits = torch.randn(8, 1000, dtype=torch.bfloat16)
    temperatures = torch.tensor([0.6, 1.0, 0.8, 1.2, 0.5, 0.9, 1.1, 0.7])
    reference = torch.compile(_main_sampler)
    torch.manual_seed(42)
    expected = reference(logits.clone(), temperatures.clone())
    torch.manual_seed(42)
    actual = Sampler()(logits.clone(), temperatures.clone())
    assert torch.equal(actual, expected)

def _pr1_reference(logits, temperatures):
    greedy = logits.argmax(dim=-1)
    logits = logits.float().div_(temperatures.clamp_min(1e-10).unsqueeze(dim=1))
    probs = torch.softmax(logits, dim=-1)
    sampled = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
    return torch.where(temperatures == 0, greedy, sampled)

def test_disabled_topk_is_seed_equivalent_to_pr1():
    logits = torch.randn(8, 1000, dtype=torch.bfloat16)
    temps = torch.tensor([0., .6, 1., .6, 1.3, .9, .6, 0.])
    ref = torch.compile(_pr1_reference)
    torch.manual_seed(42); a = ref(logits.clone(), temps.clone())
    torch.manual_seed(42); b = Sampler()(logits.clone(), temps.clone())
    assert torch.equal(a, b)

def test_topk_prefilter_only_mutates_active_rows():
    logits = torch.randn(4, 1000, dtype=torch.bfloat16)
    original = logits.clone()
    filtered = _filter_top_k(logits, (5, (2,)))
    threshold = original[2].float().topk(5).values.amin()

    assert torch.equal(filtered[[0, 1, 3]], original[[0, 1, 3]])
    assert torch.equal(torch.isfinite(filtered[2]), original[2].float() >= threshold)

def test_topk_prefilter_selects_only_active_rows(monkeypatch):
    observed_shapes = []
    torch_topk = torch.topk

    def track_topk(tensor, *args, **kwargs):
        observed_shapes.append(tuple(tensor.shape))
        return torch_topk(tensor, *args, **kwargs)

    monkeypatch.setattr(torch, "topk", track_topk)
    _filter_top_k(torch.randn(8, 1000), (50, (3,)))

    assert observed_shapes == [(1, 1000)]

def test_topk_prefilter_accepts_model_inference_tensors():
    with torch.inference_mode():
        logits = torch.randn(2, 1000, dtype=torch.bfloat16)

    filtered = _filter_top_k(logits, (5, (0,)))

    assert filtered is logits
    assert torch.isfinite(filtered[0]).sum() >= 5

def test_topk_prefilter_buckets_heterogeneous_k_values():
    logits = torch.randn(4, 1000, dtype=torch.bfloat16)
    original = logits.clone()
    filtered = _filter_top_k(logits, (1, (0,)), (5, (1, 3)))

    assert torch.equal(filtered[2], original[2])
    for row, top_k in ((0, 1), (1, 5), (3, 5)):
        threshold = original[row].float().topk(top_k).values.amin()
        assert torch.equal(torch.isfinite(filtered[row]), original[row].float() >= threshold)

def test_one_active_topk_row_preserves_inactive_draws_and_cpu_rng_state():
    logits = torch.randn(8, 1000, dtype=torch.bfloat16)
    temperatures = torch.ones(8)
    sampler = Sampler()
    sampler(logits.clone(), temperatures)

    torch.manual_seed(17)
    disabled = sampler(logits.clone(), temperatures)
    disabled_state = torch.get_rng_state().clone()

    torch.manual_seed(17)
    filtered = _filter_top_k(logits.clone(), (50, (0,)))
    enabled = sampler(filtered, temperatures)
    enabled_state = torch.get_rng_state().clone()

    assert torch.equal(enabled[1:], disabled[1:])
    assert torch.equal(enabled_state, disabled_state)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_one_active_topk_row_preserves_inactive_draws_and_cuda_rng_state():
    logits = torch.arange(8000, dtype=torch.float32, device="cuda").reshape(8, 1000)
    temperatures = torch.ones(8, device="cuda")
    sampler = Sampler().cuda()
    sampler(logits.clone(), temperatures)

    torch.cuda.manual_seed(17)
    disabled = sampler(logits.clone(), temperatures)
    disabled_state = torch.cuda.get_rng_state().clone()

    torch.cuda.manual_seed(17)
    filtered = _filter_top_k(logits.clone(), (50, (0,)))
    enabled = sampler(filtered, temperatures)
    enabled_state = torch.cuda.get_rng_state().clone()

    assert torch.equal(enabled[1:], disabled[1:])
    assert torch.equal(enabled_state, disabled_state)

def test_topk_support():
    logits = torch.randn(4, 1000, dtype=torch.bfloat16)
    kth = logits.float().sort(-1, descending=True).values[:, 4:5]   # value threshold, not topk indices
    allowed = logits.float() >= kth
    filtered = _filter_top_k(logits.clone(), (5, range(4)))
    for _ in range(500):
        s = Sampler()(filtered.clone(), torch.ones(4))
        assert bool(allowed.gather(1, s.unsqueeze(1)).all())

def test_topk_tie_at_boundary_keeps_all():
    row = torch.full((1, 10), -10.0); row[0, :2] = 5.0; row[0, 2:5] = 3.0
    filtered = _filter_top_k(row.clone(), (3, (0,)))
    assert torch.isfinite(filtered).nonzero(as_tuple=False)[:, 1].tolist() == [0, 1, 2, 3, 4]
    seen = {int(Sampler()(filtered.clone(), torch.ones(1))) for _ in range(2000)}
    assert seen == {0, 1, 2, 3, 4}

def test_topk_one_samples_a_maximum():
    torch.manual_seed(3)
    logits = torch.randn(8, 1000, dtype=torch.bfloat16)
    filtered = _filter_top_k(logits.clone(), (1, range(8)))
    out = Sampler()(filtered, torch.ones(8))
    lf = logits.float()
    assert torch.equal(lf.gather(1, out.unsqueeze(1)).squeeze(1), lf.max(-1).values)

def test_topk_one_with_tied_maxima_samples_among_ties():
    row = torch.full((1, 10), -10.0); row[0, 2] = 5.0; row[0, 7] = 5.0
    filtered = _filter_top_k(row.clone(), (1, (0,)))
    seen = {int(Sampler()(filtered.clone(), torch.ones(1))) for _ in range(500)}
    assert seen == {2, 7}

def test_topk_params_validation():
    SamplingParams(top_k=-1); SamplingParams(top_k=5)
    with pytest.raises(ValueError):
        SamplingParams(top_k=0)

def test_topp_keeps_the_crossing_token():
    row = torch.tensor([[math.log(0.5), math.log(0.3), math.log(0.2)]])
    temperatures = torch.ones(1)
    filtered = _filter_top_p(row.clone(), temperatures, [0.6])
    seen = {int(Sampler()(filtered.clone(), temperatures)) for _ in range(2000)}
    assert seen == {0, 1}

def test_topp_exact_half_boundary_matches_transformers():
    row = torch.tensor([[math.log(0.5), math.log(0.5)]])
    temperatures = torch.ones(1)
    actual = torch.isfinite(
        _filter_top_p(row.clone(), temperatures, [0.5])
    )
    expected = _transformers_top_p_support(row, temperatures, [0.5])

    assert torch.equal(actual, expected)
    assert actual.sum() == 1

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_topp_host_cutoff_rounding_matches_transformers(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    host_cutoff = torch.tensor(1.0 - 0.9, dtype=torch.float32)
    probability = host_cutoff
    for _ in range(2):
        probability = torch.nextafter(
            probability, torch.tensor(float("inf"), dtype=torch.float32)
        )
    small_logit = math.log(probability.item() / (1.0 - probability.item()))
    logits = torch.tensor([[small_logit, 0.0]], device=device)
    temperatures = torch.ones(1, device=device)

    actual = torch.isfinite(
        _filter_top_p(logits.clone(), temperatures, [0.9])
    )
    expected = _transformers_top_p_support(logits, temperatures, [0.9])

    assert torch.equal(actual, expected)
    assert actual.all()

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_topp_random_support_matches_transformers(dtype):
    torch.manual_seed(29)
    logits = torch.randn(600, 257).to(dtype)
    temperatures = torch.linspace(0.5, 1.5, logits.size(0))
    top_ps = [0.1, 0.5, 0.8, 0.9, 0.95, 0.99] * 100

    actual = torch.isfinite(
        _filter_top_p(logits.clone(), temperatures, top_ps)
    )
    expected = _transformers_top_p_support(logits, temperatures, top_ps)

    assert torch.equal(actual, expected)

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_topp_forced_tie_support_matches_transformers(dtype):
    torch.manual_seed(31)
    logits = torch.randint(-3, 4, (600, 257)).to(dtype)
    temperatures = torch.linspace(0.5, 1.5, logits.size(0))
    top_ps = [0.2, 0.5, 0.8, 0.9] * 150

    actual = torch.isfinite(
        _filter_top_p(logits.clone(), temperatures, top_ps)
    )
    expected = _transformers_top_p_support(logits, temperatures, top_ps)

    assert torch.equal(actual, expected)

def test_topp_prefilter_selects_only_active_rows(monkeypatch):
    observed_shapes = []
    torch_sort = torch.sort

    def track_sort(tensor, *args, **kwargs):
        observed_shapes.append(tuple(tensor.shape))
        return torch_sort(tensor, *args, **kwargs)

    monkeypatch.setattr(torch, "sort", track_sort)
    logits = torch.randn(8, 1000)
    original_logits = logits.clone()
    temperatures = torch.ones(8)
    _filter_top_p(logits, temperatures, [0.9], rows=(3,))

    assert observed_shapes == [(1, 1000)]
    inactive_rows = torch.tensor([0, 1, 2, 4, 5, 6, 7])
    assert torch.equal(
        logits.index_select(0, inactive_rows),
        original_logits.index_select(0, inactive_rows),
    )

def test_topp_chunks_all_active_rows_without_changing_transformers_support(
    monkeypatch,
):
    observed_shapes = []
    torch_sort = torch.sort

    def track_sort(tensor, *args, **kwargs):
        observed_shapes.append(tuple(tensor.shape))
        return torch_sort(tensor, *args, **kwargs)

    monkeypatch.setattr(torch, "sort", track_sort)
    logits = torch.randint(-3, 4, (65, 257), dtype=torch.float32)
    temperatures = torch.linspace(0.5, 1.5, logits.size(0))
    top_ps = [0.9] * logits.size(0)
    actual = torch.isfinite(
        _filter_top_p(logits.clone(), temperatures, top_ps)
    )
    expected = _transformers_top_p_support(logits, temperatures, top_ps)

    assert observed_shapes[:2] == [(64, 257), (1, 257)]
    assert torch.equal(actual, expected)

def test_topp_prefilter_accepts_model_inference_tensors():
    with torch.inference_mode():
        logits = torch.randn(2, 1000, dtype=torch.bfloat16)
    temperatures = torch.ones(2)

    filtered = _filter_top_p(logits, temperatures, [0.9], rows=(0,))

    assert filtered is logits
    assert torch.isfinite(filtered[0]).sum() < filtered.size(1)

def test_one_active_topp_row_preserves_inactive_draws_and_cpu_rng_state():
    logits = torch.randn(8, 1000, dtype=torch.bfloat16)
    temperatures = torch.ones(8)
    sampler = Sampler()
    sampler(logits.clone(), temperatures)

    torch.manual_seed(37)
    disabled = sampler(logits.clone(), temperatures)
    disabled_state = torch.get_rng_state().clone()

    torch.manual_seed(37)
    filtered = _filter_top_p(
        logits.clone(), temperatures, [0.9], rows=(0,)
    )
    enabled = sampler(filtered, temperatures)
    enabled_state = torch.get_rng_state().clone()

    assert torch.equal(enabled[1:], disabled[1:])
    assert torch.equal(enabled_state, disabled_state)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_one_active_topp_row_preserves_inactive_draws_and_cuda_rng_state():
    logits = torch.arange(8000, dtype=torch.float32, device="cuda").reshape(8, 1000)
    temperatures = torch.ones(8, device="cuda")
    sampler = Sampler().cuda()
    sampler(logits.clone(), temperatures)

    torch.cuda.manual_seed(37)
    disabled = sampler(logits.clone(), temperatures)
    disabled_state = torch.cuda.get_rng_state().clone()

    torch.cuda.manual_seed(37)
    filtered = _filter_top_p(
        logits.clone(), temperatures, [0.9], rows=(0,)
    )
    enabled = sampler(filtered, temperatures)
    enabled_state = torch.cuda.get_rng_state().clone()

    assert torch.equal(enabled[1:], disabled[1:])
    assert torch.equal(enabled_state, disabled_state)

def test_topp_adaptive_collapse_on_peaked_row():
    row = torch.full((1, 100), -10.0)
    row[0, 7] = 10.0
    row[0, 3] = 2.0
    temperatures = torch.ones(1)
    filtered = _filter_top_p(row.clone(), temperatures, [0.8])
    seen = {int(Sampler()(filtered.clone(), temperatures)) for _ in range(500)}
    assert seen == {7}

def test_topp_tiny_probability_keeps_one_token_and_matches_transformers():
    logits = torch.zeros(1, 17)
    temperatures = torch.ones(1)
    actual = torch.isfinite(
        _filter_top_p(logits.clone(), temperatures, [1e-9])
    )
    expected = _transformers_top_p_support(logits, temperatures, [1e-9])

    assert torch.equal(actual, expected)
    assert actual.sum() == 1

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_topk_then_topp_support_matches_transformers(dtype):
    torch.manual_seed(41)
    logits = torch.randint(-20, 21, (120, 257)).to(dtype)
    temperatures = torch.linspace(0.5, 1.5, logits.size(0))
    top_ks = [5, 17, 50] * 40
    top_ps = [0.5, 0.8, 0.9, 0.95] * 30
    filtered = logits.clone()
    for top_k in sorted(set(top_ks)):
        rows = tuple(row for row, value in enumerate(top_ks) if value == top_k)
        filtered = _filter_top_k(filtered, (top_k, rows))
    actual = torch.isfinite(
        _filter_top_p(filtered, temperatures, top_ps)
    )
    expected = _transformers_top_k_top_p_support(
        logits, temperatures, top_ks, top_ps
    )

    assert torch.equal(actual, expected)

def test_topp_params_validation():
    SamplingParams(top_p=1.0)
    SamplingParams(top_p=0.9)
    with pytest.raises(ValueError):
        SamplingParams(top_p=0.0)
    with pytest.raises(ValueError):
        SamplingParams(top_p=1.5)
