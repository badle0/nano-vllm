import json

import pytest

from nanovllm.utils.tokenizer_identity import (
    require_same_token_id_space,
    tokenizer_identity,
)


VOCAB_SIZE = 3


class FakeBackend:
    def __init__(self, document):
        self.document = document

    def to_str(self):
        return json.dumps(self.document)


class FakeTokenizer:
    SPECIAL_TOKENS_ATTRIBUTES = (
        "bos_token",
        "eos_token",
        "additional_special_tokens",
    )

    def __init__(
        self,
        *,
        vocab=None,
        added=None,
        eos_token_id=2,
        all_special_ids=None,
        special_tokens_map=None,
        tokenizer_config=None,
        backend=None,
    ):
        self._vocab = {"a": 0, "b": 1, "<eos>": 2} if vocab is None else vocab
        self._added = {"<eos>": 2} if added is None else added
        self.bos_token_id = None
        self.eos_token_id = eos_token_id
        self.additional_special_tokens_ids = [2]
        self.all_special_ids = [2] if all_special_ids is None else all_special_ids
        self.special_tokens_map_extended = (
            {"eos_token": "<eos>"}
            if special_tokens_map is None
            else special_tokens_map
        )
        self.init_kwargs = (
            {"clean_up_tokenization_spaces": False, "name_or_path": "model"}
            if tokenizer_config is None
            else tokenizer_config
        )
        self.backend_tokenizer = FakeBackend(
            {
                "normalizer": {"type": "NFC"},
                "pre_tokenizer": {"type": "ByteLevel"},
            }
            if backend is None
            else backend
        )

    def get_vocab(self):
        return dict(self._vocab)

    def get_added_vocab(self):
        return dict(self._added)


def test_matching_tokenizers_have_one_stable_identity():
    target = FakeTokenizer()
    draft = FakeTokenizer()

    assert tokenizer_identity(
        target,
        role="target",
        vocab_size=VOCAB_SIZE,
    ) == tokenizer_identity(
        draft,
        role="draft",
        vocab_size=VOCAB_SIZE,
    )
    assert len(
        require_same_token_id_space(
            target,
            draft,
            vocab_size=VOCAB_SIZE,
        )
    ) == 64


@pytest.mark.parametrize(
    ("draft", "component"),
    [
        (FakeTokenizer(vocab={"a": 0, "b": 2}), "complete vocabulary"),
        (FakeTokenizer(added={"<eos>": 1}), "added-token mapping"),
        (
            FakeTokenizer(eos_token_id=1, all_special_ids=[1]),
            "special-token IDs",
        ),
        (
            FakeTokenizer(all_special_ids=[1, 2]),
            "complete special-token ID list",
        ),
        (
            FakeTokenizer(special_tokens_map={"eos_token": "</s>"}),
            "special-token map",
        ),
        (
            FakeTokenizer(
                tokenizer_config={"clean_up_tokenization_spaces": True}
            ),
            "tokenizer configuration",
        ),
        (
            FakeTokenizer(backend={"normalizer": {"type": "NFD"}}),
            "normalizer/pre-tokenizer/backend configuration",
        ),
    ],
)
def test_tokenizer_identity_names_the_mismatching_component(draft, component):
    with pytest.raises(ValueError, match=component):
        require_same_token_id_space(
            FakeTokenizer(),
            draft,
            vocab_size=VOCAB_SIZE,
        )


def test_tokenizer_identity_requires_a_fast_serializable_backend():
    tokenizer = FakeTokenizer()
    tokenizer.backend_tokenizer = None

    with pytest.raises(ValueError, match="fast tokenizer"):
        tokenizer_identity(
            tokenizer,
            role="draft",
            vocab_size=VOCAB_SIZE,
        )


def test_backend_json_key_order_does_not_change_identity():
    target = FakeTokenizer(
        backend={"normalizer": {"type": "NFC"}, "decoder": {"type": "ByteLevel"}}
    )
    draft = FakeTokenizer(
        backend={"decoder": {"type": "ByteLevel"}, "normalizer": {"type": "NFC"}}
    )

    require_same_token_id_space(
        target,
        draft,
        vocab_size=VOCAB_SIZE,
    )


def test_artifact_location_does_not_change_tokenizer_identity():
    target = FakeTokenizer(
        tokenizer_config={
            "name_or_path": "/models/target",
            "tokenizer_file": "/models/target/tokenizer.json",
            "clean_up_tokenization_spaces": False,
        }
    )
    draft = FakeTokenizer(
        tokenizer_config={
            "name_or_path": "/models/draft",
            "tokenizer_file": "/models/draft/tokenizer.json",
            "clean_up_tokenization_spaces": False,
        }
    )

    require_same_token_id_space(
        target,
        draft,
        vocab_size=VOCAB_SIZE,
    )


@pytest.mark.parametrize(
    "invalid_id",
    [-1, VOCAB_SIZE],
)
def test_tokenizer_identity_rejects_ids_outside_model_vocabulary(invalid_id):
    tokenizer = FakeTokenizer(vocab={"bad": invalid_id, "<eos>": 2})

    with pytest.raises(ValueError, match="outside model vocabulary"):
        tokenizer_identity(
            tokenizer,
            role="draft",
            vocab_size=VOCAB_SIZE,
        )


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("SPECIAL_TOKENS_ATTRIBUTES", None, "SPECIAL_TOKENS_ATTRIBUTES"),
        ("all_special_ids", None, "all_special_ids"),
        ("special_tokens_map_extended", None, "special_tokens_map"),
        ("eos_token_id", None, "non-integer token ID"),
        ("eos_token_id", VOCAB_SIZE, "outside model vocabulary"),
        ("all_special_ids", [], "absent from all_special_ids"),
    ],
)
def test_tokenizer_identity_requires_complete_special_metadata(
    attribute,
    value,
    message,
):
    tokenizer = FakeTokenizer()
    setattr(tokenizer, attribute, value)

    with pytest.raises(ValueError, match=message):
        tokenizer_identity(
            tokenizer,
            role="draft",
            vocab_size=VOCAB_SIZE,
        )
