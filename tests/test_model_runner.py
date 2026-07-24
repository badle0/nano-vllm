from types import SimpleNamespace

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
