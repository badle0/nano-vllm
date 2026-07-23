import importlib.util, pathlib, torch, pytest

from nanovllm.sampling_params import SamplingParams

_p = pathlib.Path(__file__).resolve().parents[1] / "nanovllm/layers/sampler.py"
_s = importlib.util.spec_from_file_location("sampler", _p)
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)
Sampler = _m.Sampler

def test_temperature_zero_is_permitted():
    SamplingParams(temperature=0.0)

def test_negative_temperature_rejected():
    with pytest.raises(AssertionError):
        SamplingParams(temperature=-1.0)

def test_greedy_is_argmax():
    logits = torch.randn(8, 1000, dtype=torch.bfloat16)
    out = Sampler()(logits.clone(), torch.zeros(8))
    assert torch.equal(out, logits.argmax(-1))

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