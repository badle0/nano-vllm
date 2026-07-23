from types import SimpleNamespace

from nanovllm.engine.model_runner import ModelRunner


def test_prepare_sample_detects_all_greedy_on_the_host():
    runner = object.__new__(ModelRunner)
    seqs = [SimpleNamespace(temperature=0.0) for _ in range(4)]

    temperatures, top_ks, all_greedy = runner.prepare_sample(seqs)

    assert temperatures is None
    assert top_ks is None
    assert all_greedy is True
