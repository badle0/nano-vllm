import importlib.util
import pathlib
import subprocess
import sys

import pytest
import torch

from nanovllm.sampling_params import SamplingParams

_sp = pathlib.Path(__file__).resolve().parents[1] / "nanovllm/layers/sampler.py"
_ss = importlib.util.spec_from_file_location("sampler", _sp)
_sm = importlib.util.module_from_spec(_ss); _ss.loader.exec_module(_sm)
Sampler = _sm.Sampler

def _disabled(n):
    return torch.full((n,), -1, dtype=torch.int64)

def test_temperature_zero_is_permitted():
    SamplingParams(temperature=0.0)

def test_negative_temperature_rejected():
    with pytest.raises(ValueError):
        SamplingParams(temperature=-1.0)

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
    out = Sampler()(logits.clone(), temps, _disabled(6))
    g = temps == 0
    assert torch.equal(out[g], logits.argmax(-1)[g])

def test_stochastic_path_is_not_constant():
    logits = torch.randn(1, 1000, dtype=torch.bfloat16)
    draws = {Sampler()(logits.clone(), torch.ones(1), None).item() for _ in range(200)}
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
    actual = Sampler()(logits.clone(), temperatures.clone(), None)
    assert torch.equal(actual, expected)

def test_validation_survives_optimized_python():
    code = """
from nanovllm.sampling_params import SamplingParams

for temperature in (-1.0, True, "cold"):
    try:
        SamplingParams(temperature=temperature)
    except (TypeError, ValueError):
        continue
    raise SystemExit(f"accepted invalid temperature: {temperature!r}")
"""
    subprocess.run(
        [sys.executable, "-O", "-c", code],
        cwd=pathlib.Path(__file__).resolve().parents[1],
        check=True,
    )

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
    torch.manual_seed(42); b = Sampler()(logits.clone(), temps.clone(), torch.full((8,), -1, dtype=torch.int64))
    assert torch.equal(a, b)

def test_none_topk_is_seed_equivalent_to_pr1():
    # fast path: with top_ks=None the compiled graph is op-for-op the PR1 sampler
    logits = torch.randn(8, 1000, dtype=torch.bfloat16)
    temps = torch.tensor([0., .6, 1., .6, 1.3, .9, .6, 0.])
    ref = torch.compile(_pr1_reference)
    torch.manual_seed(42); a = ref(logits.clone(), temps.clone())
    torch.manual_seed(42); b = Sampler()(logits.clone(), temps.clone(), None)
    assert torch.equal(a, b)

def test_none_and_all_disabled_tensor_agree():
    logits = torch.randn(8, 1000, dtype=torch.bfloat16)
    torch.manual_seed(9); a = Sampler()(logits.clone(), torch.ones(8), None)
    torch.manual_seed(9); b = Sampler()(logits.clone(), torch.ones(8), torch.full((8,), -1, dtype=torch.int64))
    assert torch.equal(a, b)

def test_topk_support():
    logits = torch.randn(4, 1000, dtype=torch.bfloat16)
    kth = logits.float().sort(-1, descending=True).values[:, 4:5]   # value threshold, not topk indices
    allowed = logits.float() >= kth
    for _ in range(500):
        s = Sampler()(logits.clone(), torch.ones(4), torch.full((4,), 5, dtype=torch.int64))
        assert bool(allowed.gather(1, s.unsqueeze(1)).all())

def test_topk_tie_at_boundary_keeps_all():
    row = torch.full((1, 10), -10.0); row[0, :2] = 5.0; row[0, 2:5] = 3.0
    seen = {int(Sampler()(row.clone(), torch.ones(1), torch.tensor([3]))) for _ in range(2000)}
    assert seen == {0, 1, 2, 3, 4}

def test_topk_one_samples_a_maximum():
    torch.manual_seed(3)
    logits = torch.randn(8, 1000, dtype=torch.bfloat16)
    out = Sampler()(logits.clone(), torch.ones(8), torch.ones(8, dtype=torch.int64))
    lf = logits.float()
    assert torch.equal(lf.gather(1, out.unsqueeze(1)).squeeze(1), lf.max(-1).values)

def test_topk_one_with_tied_maxima_samples_among_ties():
    row = torch.full((1, 10), -10.0); row[0, 2] = 5.0; row[0, 7] = 5.0
    seen = {int(Sampler()(row.clone(), torch.ones(1), torch.tensor([1]))) for _ in range(500)}
    assert seen == {2, 7}

def test_topk_params_validation():
    SamplingParams(top_k=-1); SamplingParams(top_k=5)
    with pytest.raises(AssertionError):
        SamplingParams(top_k=0)
