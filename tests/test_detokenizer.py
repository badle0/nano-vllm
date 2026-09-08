import os
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from nanovllm import StreamingDetokenizer, TextUpdate


MODEL_PATH = "/workspace/models/Qwen3-0.6B"
CASES = [
    "hello world",
    "你好，世界！这是中文测试。",
    "emoji: 🎉🚀👨‍👩‍👧‍👦 done",
    "café naïve résumé",
    "mixed 中文 and English 🎉 text",
    "   leading and trailing   ",
    "日本語とKorean한국어のmix",
]


@pytest.fixture(scope="module")
def tok():
    path = Path(os.environ.get("NANOVLLM_TEST_TOKENIZER_PATH", MODEL_PATH))
    if not path.is_dir():
        pytest.fail(
            f"Tokenizer fixture directory is missing: {path}. "
            "Run .github/ci/prepare_tokenizer.py --output DIR and set "
            "NANOVLLM_TEST_TOKENIZER_PATH=DIR. Model weights are not required."
        )
    return AutoTokenizer.from_pretrained(str(path), local_files_only=True)


def test_tokenizer_fixture_uses_configured_local_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOVLLM_TEST_TOKENIZER_PATH", str(tmp_path))
    expected = object()
    calls = []

    def load(path, **kwargs):
        calls.append((path, kwargs))
        return expected

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load)
    assert tok.__wrapped__() is expected
    assert calls == [(str(tmp_path), {"local_files_only": True})]


def test_missing_tokenizer_fixture_fails_explicitly(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOVLLM_TEST_TOKENIZER_PATH", str(tmp_path / "missing"))
    with pytest.raises(pytest.fail.Exception, match="Tokenizer fixture directory is missing"):
        tok.__wrapped__()


def apply_update(rendered, update):
    return update.apply(rendered)


@pytest.mark.parametrize("text", CASES)
def test_incremental_updates_finish_at_exact_full_decode(tok, text):
    token_ids = tok.encode(text)
    detokenizer = StreamingDetokenizer(tok)
    rendered = ""
    for token_id in token_ids:
        rendered = apply_update(rendered, detokenizer.feed(0, token_id))
    rendered = apply_update(rendered, detokenizer.flush(0))
    assert rendered == tok.decode(token_ids)


class RewriteTokenizer:

    def decode(self, token_ids):
        outputs = {
            (1,): "hello ",
            (1, 2): "hello.",
            (3,): "cafe\u0301",
            (3, 4): "café!",
        }
        return outputs[tuple(token_ids)]


@pytest.mark.parametrize(
    ("token_ids", "prefixes"),
    [
        ([1, 2], ["hello ", "hello."]),
        ([3, 4], ["cafe\u0301", "café!"]),
    ],
)
def test_updates_repair_cleanup_and_normalization_rewrites(token_ids, prefixes):
    detokenizer = StreamingDetokenizer(RewriteTokenizer())
    rendered = ""
    for token_id, expected in zip(token_ids, prefixes, strict=True):
        update = detokenizer.feed(7, token_id)
        rendered = update.apply(rendered)
        assert rendered == expected
    rendered = detokenizer.flush(7).apply(rendered)
    assert rendered == prefixes[-1]


class BoundaryRewriteTokenizer:

    def decode(self, token_ids):
        pieces = {1: "A", 2: " ", 3: "."}
        text = "".join(pieces.get(token_id, "x") for token_id in token_ids)
        return text.replace(" .", ".")


def test_length_changing_rewrite_remains_exact_across_window_shifts():
    tokenizer = BoundaryRewriteTokenizer()
    token_ids = [1, 2, 3] + list(range(4, 40))
    detokenizer = StreamingDetokenizer(
        tokenizer,
        window_size=4,
        boundary_overlap=3,
    )
    rendered = ""
    for index, token_id in enumerate(token_ids, start=1):
        rendered = detokenizer.feed(0, token_id).apply(rendered)
        assert rendered == tokenizer.decode(token_ids[:index])
    rendered = detokenizer.flush(0).apply(rendered)
    assert rendered == tokenizer.decode(token_ids)


class NonSplittableTokenizer:

    def decode(self, token_ids):
        return f"<{','.join(str(token_id) for token_id in token_ids)}>"


def test_overlap_exhaustion_fails_before_admitting_token_and_can_flush_exactly():
    tokenizer = NonSplittableTokenizer()
    window_size = 4
    boundary_overlap = 2
    hard_limit = window_size + 2 * boundary_overlap
    detokenizer = StreamingDetokenizer(
        tokenizer,
        window_size=window_size,
        boundary_overlap=boundary_overlap,
    )
    rendered = ""
    for token_id in range(hard_limit):
        rendered = detokenizer.feed(0, token_id).apply(rendered)
        assert rendered == tokenizer.decode(list(range(token_id + 1)))

    with pytest.raises(RuntimeError, match="exceeded.*boundary overlap"):
        detokenizer.feed(0, hard_limit)

    state = detokenizer._states[0]
    assert state.token_ids == list(range(hard_limit))
    final = detokenizer.flush(0)
    assert final.final
    assert final.apply(rendered) == tokenizer.decode(list(range(hard_limit)))
    assert detokenizer._states == {}


class FragmentTokenizer:

    def decode(self, token_ids):
        if token_ids == [1]:
            return "\ufffd"
        if token_ids == [1, 2]:
            return "🎉"
        return ""


def test_incomplete_utf8_fragment_is_held_then_corrected():
    detokenizer = StreamingDetokenizer(FragmentTokenizer())
    rendered = detokenizer.feed(0, 1).apply("")
    assert rendered == ""
    rendered = detokenizer.feed(0, 2).apply(rendered)
    assert rendered == "🎉"
    assert detokenizer.flush(0).apply(rendered) == "🎉"


class PersistentReplacementTokenizer:

    def __init__(self):
        self.decode_lengths = []

    def decode(self, token_ids):
        self.decode_lengths.append(len(token_ids))
        if len(token_ids) >= 2 and token_ids[-2:] == [1, 2]:
            return "\ufffd" * (len(token_ids) - 2) + "🎉"
        return "\ufffd" * len(token_ids)


def test_persistent_replacement_suffix_makes_bounded_correctable_progress():
    tokenizer = PersistentReplacementTokenizer()
    window_size = 4
    boundary_overlap = 2
    detokenizer = StreamingDetokenizer(
        tokenizer,
        window_size=window_size,
        boundary_overlap=boundary_overlap,
    )

    token_ids = [1] * 64
    rendered = ""
    max_retained = 0
    for token_id in token_ids:
        rendered = detokenizer.feed(0, token_id).apply(rendered)
        state = detokenizer._states[0]
        retained = len(state.token_ids) - state.window_start_token
        max_retained = max(max_retained, retained)

    assert rendered == "\ufffd" * 64
    assert max_retained < window_size + boundary_overlap

    token_ids.append(2)
    rendered = detokenizer.feed(0, 2).apply(rendered)
    expected = "\ufffd" * 63 + "🎉"
    assert rendered == expected

    incremental_decode_lengths = list(tokenizer.decode_lengths)
    rendered = detokenizer.flush(0).apply(rendered)
    assert rendered == expected
    assert max(incremental_decode_lengths) <= (
        window_size + 2 * boundary_overlap
    )
    assert tokenizer.decode_lengths[-1] == len(token_ids)
    assert detokenizer._states == {}


def test_state_is_freed(tok):
    detokenizer = StreamingDetokenizer(tok)
    for token_id in tok.encode("你好🎉"):
        detokenizer.feed(0, token_id)
    detokenizer.flush(0)
    assert detokenizer._states == {}


def test_interleaved_sequences(tok):
    first = tok.encode("你好世界")
    second = tok.encode("🎉 party")
    detokenizer = StreamingDetokenizer(tok)
    rendered = {0: "", 1: ""}
    for index in range(max(len(first), len(second))):
        if index < len(first):
            rendered[0] = detokenizer.feed(0, first[index]).apply(rendered[0])
        if index < len(second):
            rendered[1] = detokenizer.feed(1, second[index]).apply(rendered[1])
    rendered[0] = detokenizer.flush(0).apply(rendered[0])
    rendered[1] = detokenizer.flush(1).apply(rendered[1])
    assert rendered[0] == tok.decode(first)
    assert rendered[1] == tok.decode(second)


class CountingTokenizer:

    def __init__(self):
        self.decode_lengths = []

    def decode(self, token_ids):
        self.decode_lengths.append(len(token_ids))
        return "x" * len(token_ids)


def test_decode_work_is_bounded_per_feed_and_linear_overall():
    tokenizer = CountingTokenizer()
    window_size = 16
    boundary_overlap = 8
    num_tokens = 4096
    detokenizer = StreamingDetokenizer(
        tokenizer,
        window_size=window_size,
        boundary_overlap=boundary_overlap,
    )
    rendered = ""
    for token_id in range(num_tokens):
        rendered = detokenizer.feed(0, token_id).apply(rendered)
    rendered = detokenizer.flush(0).apply(rendered)

    assert rendered == "x" * num_tokens
    assert max(tokenizer.decode_lengths[:-1]) <= (
        window_size + 2 * boundary_overlap
    )
    assert tokenizer.decode_lengths[-1] == num_tokens
    assert tokenizer.decode_lengths.count(num_tokens) == 1
    assert sum(tokenizer.decode_lengths) <= (
        num_tokens * 3 * (window_size + 2 * boundary_overlap)
    )


def test_text_update_uses_unicode_code_point_offsets():
    update = TextUpdate(0, replace_from=1, delete_count=1, insert="🚀")
    assert update.apply("a🎉b") == "a🚀b"


@pytest.mark.parametrize(
    ("window_size", "boundary_overlap"),
    [(True, 8), (1, 8), (32, True), (32, 0), (32, -1)],
)
def test_window_configuration_is_explicitly_validated(
    window_size,
    boundary_overlap,
):
    with pytest.raises(ValueError):
        StreamingDetokenizer(
            CountingTokenizer(),
            window_size=window_size,
            boundary_overlap=boundary_overlap,
        )
