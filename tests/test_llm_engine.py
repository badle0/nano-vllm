import json
from types import SimpleNamespace

import pytest

import nanovllm.engine.llm_engine as llm_engine_module
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams


@pytest.mark.parametrize("num_params", [1, 3])
def test_generate_rejects_mismatched_sampling_params_before_admission(num_params):
    engine = object.__new__(LLMEngine)

    with pytest.raises(ValueError, match="same length"):
        engine.generate(
            ["first", "second"],
            [SamplingParams()] * num_params,
            use_tqdm=False,
        )


def test_flashinfer_dependency_fails_before_worker_processes_start(monkeypatch):
    monkeypatch.setattr(
        llm_engine_module,
        "fields",
        lambda config_type: [
            type("Field", (), {"name": "top_p_backend", "init": True})()
        ],
    )
    monkeypatch.setattr(
        llm_engine_module,
        "Config",
        lambda model, **kwargs: type(
            "ConfigResult", (), {"top_p_backend": "flashinfer"}
        )(),
    )

    def missing_dependency():
        raise RuntimeError("install nano-vllm[fast-sampling]")

    monkeypatch.setattr(
        llm_engine_module, "require_flashinfer_sampling", missing_dependency
    )
    monkeypatch.setattr(
        llm_engine_module.mp,
        "get_context",
        lambda method: pytest.fail("workers started before dependency validation"),
    )

    with pytest.raises(RuntimeError, match="fast-sampling"):
        LLMEngine("/unused", top_p_backend="flashinfer")


def _fake_tokenizer(vocab=None):
    return SimpleNamespace(
        SPECIAL_TOKENS_ATTRIBUTES=("eos_token",),
        eos_token_id=2,
        all_special_ids=[2],
        special_tokens_map_extended={"eos_token": "<eos>"},
        init_kwargs={
            "clean_up_tokenization_spaces": False,
            "name_or_path": "model",
        },
        get_vocab=lambda: dict(
            {"a": 0, "<eos>": 2} if vocab is None else vocab
        ),
        get_added_vocab=lambda: {"<eos>": 2},
        backend_tokenizer=SimpleNamespace(
            to_str=lambda: json.dumps(
                {
                    "normalizer": {"type": "NFC"},
                    "pre_tokenizer": {"type": "ByteLevel"},
                }
            )
        ),
    )


@pytest.mark.parametrize(
    ("component", "message"),
    [
        ("implementation", "implementation class differs"),
        ("vocabulary", "complete vocabulary differs"),
        ("added", "added-token mapping differs"),
        ("special_ids", "special-token IDs differ"),
        ("all_special_ids", "complete special-token ID list differs"),
        ("special_map", "special-token map differs"),
        ("config", "tokenizer configuration differs"),
        ("backend", "normalizer/pre-tokenizer/backend configuration differs"),
        ("negative_id", "outside model vocabulary"),
        ("upper_bound_id", "outside model vocabulary"),
        ("missing_special_attributes", "SPECIAL_TOKENS_ATTRIBUTES"),
        ("missing_all_special_ids", "all_special_ids"),
        ("missing_special_map", "special_tokens_map"),
    ],
)
def test_speculative_tokenizer_mismatch_fails_before_worker_creation(
    monkeypatch,
    component,
    message,
):
    monkeypatch.setattr(
        llm_engine_module,
        "fields",
        lambda config_type: [
            type("Field", (), {"name": name, "init": True})()
            for name in ("draft_model", "num_speculative_tokens")
        ],
    )
    monkeypatch.setattr(
        llm_engine_module,
        "Config",
        lambda model, **kwargs: SimpleNamespace(
            model=model,
            draft_model=kwargs["draft_model"],
            speculation_enabled=True,
            top_p_backend="exact",
            hf_config=SimpleNamespace(vocab_size=3),
        ),
    )
    target_tokenizer = _fake_tokenizer()
    draft_tokenizer = _fake_tokenizer()
    if component == "implementation":
        class DifferentTokenizer(SimpleNamespace):
            pass

        draft_tokenizer = DifferentTokenizer(**vars(draft_tokenizer))
    elif component == "vocabulary":
        draft_tokenizer.get_vocab = lambda: {"a": 1, "<eos>": 2}
    elif component == "added":
        draft_tokenizer.get_added_vocab = lambda: {"<eos>": 1}
    elif component == "special_ids":
        draft_tokenizer.eos_token_id = 1
        draft_tokenizer.all_special_ids = [1]
    elif component == "all_special_ids":
        draft_tokenizer.all_special_ids = [1, 2]
    elif component == "special_map":
        draft_tokenizer.special_tokens_map_extended = {"eos_token": "</s>"}
    elif component == "config":
        draft_tokenizer.init_kwargs = {"clean_up_tokenization_spaces": True}
    elif component == "backend":
        draft_tokenizer.backend_tokenizer = SimpleNamespace(
            to_str=lambda: json.dumps({"normalizer": {"type": "NFD"}})
        )
    elif component == "negative_id":
        draft_tokenizer.get_vocab = lambda: {"bad": -1, "<eos>": 2}
    elif component == "upper_bound_id":
        draft_tokenizer.get_vocab = lambda: {"bad": 3, "<eos>": 2}
    elif component == "missing_special_attributes":
        draft_tokenizer.SPECIAL_TOKENS_ATTRIBUTES = None
    elif component == "missing_all_special_ids":
        draft_tokenizer.all_special_ids = None
    elif component == "missing_special_map":
        draft_tokenizer.special_tokens_map_extended = None
    tokenizers = iter([target_tokenizer, draft_tokenizer])
    monkeypatch.setattr(
        llm_engine_module.AutoTokenizer,
        "from_pretrained",
        lambda model, use_fast: next(tokenizers),
    )
    monkeypatch.setattr(
        llm_engine_module.mp,
        "get_context",
        lambda method: pytest.fail("workers started before tokenizer validation"),
    )

    with pytest.raises(ValueError, match=message):
        LLMEngine(
            "target",
            draft_model="draft",
            num_speculative_tokens=4,
        )


def test_speculation_off_loads_only_the_target_tokenizer(monkeypatch):
    class StopBeforeWorkers(RuntimeError):
        pass

    monkeypatch.setattr(llm_engine_module, "fields", lambda config_type: [])
    monkeypatch.setattr(
        llm_engine_module,
        "Config",
        lambda model, **kwargs: SimpleNamespace(
            model=model,
            speculation_enabled=False,
            top_p_backend="exact",
            eos=-1,
            kvcache_block_size=256,
            tensor_parallel_size=1,
        ),
    )
    loaded = []

    def load_tokenizer(model, use_fast):
        loaded.append(model)
        return _fake_tokenizer({"a": 0, "<eos>": 2})

    monkeypatch.setattr(
        llm_engine_module.AutoTokenizer,
        "from_pretrained",
        load_tokenizer,
    )
    monkeypatch.setattr(
        llm_engine_module.mp,
        "get_context",
        lambda method: (_ for _ in ()).throw(StopBeforeWorkers()),
    )

    engine = object.__new__(LLMEngine)
    with pytest.raises(StopBeforeWorkers):
        engine.__init__("target")

    assert loaded == ["target"]
    assert not hasattr(engine, "speculative_tokenizer_fingerprint")


def test_matching_speculative_tokenizers_reach_worker_boundary(monkeypatch):
    class WorkerBoundaryReached(RuntimeError):
        pass

    monkeypatch.setattr(
        llm_engine_module,
        "fields",
        lambda config_type: [
            type("Field", (), {"name": name, "init": True})()
            for name in ("draft_model", "num_speculative_tokens")
        ],
    )
    config = SimpleNamespace(
        model="target",
        draft_model="draft",
        speculation_enabled=True,
        top_p_backend="exact",
        hf_config=SimpleNamespace(vocab_size=3),
        eos=-1,
        kvcache_block_size=256,
        tensor_parallel_size=1,
    )
    monkeypatch.setattr(
        llm_engine_module,
        "Config",
        lambda model, **kwargs: config,
    )
    loaded = []

    def load_tokenizer(model, use_fast):
        loaded.append(model)
        return _fake_tokenizer()

    monkeypatch.setattr(
        llm_engine_module.AutoTokenizer,
        "from_pretrained",
        load_tokenizer,
    )
    monkeypatch.setattr(
        llm_engine_module.mp,
        "get_context",
        lambda method: (_ for _ in ()).throw(WorkerBoundaryReached()),
    )

    engine = object.__new__(LLMEngine)
    with pytest.raises(WorkerBoundaryReached):
        engine.__init__(
            "target",
            draft_model="draft",
            num_speculative_tokens=4,
        )

    assert loaded == ["target", "draft"]
    assert config.eos == 2
    assert len(engine.speculative_tokenizer_fingerprint) == 64


def test_draft_tokenizer_load_failure_precedes_worker_creation(monkeypatch):
    class DraftTokenizerError(RuntimeError):
        pass

    monkeypatch.setattr(
        llm_engine_module,
        "fields",
        lambda config_type: [
            type("Field", (), {"name": name, "init": True})()
            for name in ("draft_model", "num_speculative_tokens")
        ],
    )
    monkeypatch.setattr(
        llm_engine_module,
        "Config",
        lambda model, **kwargs: SimpleNamespace(
            model=model,
            draft_model=kwargs["draft_model"],
            speculation_enabled=True,
            top_p_backend="exact",
            hf_config=SimpleNamespace(vocab_size=3),
        ),
    )
    loaded = []

    def load_tokenizer(model, use_fast):
        loaded.append(model)
        if model == "draft":
            raise DraftTokenizerError("injected draft tokenizer failure")
        return _fake_tokenizer()

    monkeypatch.setattr(
        llm_engine_module.AutoTokenizer,
        "from_pretrained",
        load_tokenizer,
    )
    monkeypatch.setattr(
        llm_engine_module.mp,
        "get_context",
        lambda method: pytest.fail("workers started after tokenizer failure"),
    )

    engine = object.__new__(LLMEngine)
    with pytest.raises(DraftTokenizerError, match="injected"):
        engine.__init__(
            "target",
            draft_model="draft",
            num_speculative_tokens=4,
        )

    assert loaded == ["target", "draft"]
    assert not hasattr(engine, "speculative_tokenizer_fingerprint")


def test_llm_engine_forwards_only_init_config_fields(monkeypatch):
    class ConfigReached(RuntimeError):
        pass

    monkeypatch.setattr(
        llm_engine_module,
        "fields",
        lambda config_type: [
            type("Field", (), {"name": "num_speculative_tokens", "init": True})(),
            type("Field", (), {"name": "draft_hf_config", "init": False})(),
        ],
    )

    def construct_config(model, **kwargs):
        assert kwargs == {"num_speculative_tokens": 0}
        raise ConfigReached()

    monkeypatch.setattr(llm_engine_module, "Config", construct_config)

    with pytest.raises(ConfigReached):
        LLMEngine(
            "target",
            num_speculative_tokens=0,
            draft_hf_config=object(),
        )
