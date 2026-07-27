import os, pytest
from transformers import AutoTokenizer
from nanovllm import StreamingDetokenizer

CASES = [
    "hello world",
    "你好，世界！这是中文测试。",
    "emoji: 🎉🚀👨‍👩‍👧‍👦 done",          # 4-byte + ZWJ sequences
    "café naïve résumé",                    # combining diacritics
    "mixed 中文 and English 🎉 text",
    "   leading and trailing   ",
    "日本語とKorean한국어のmix",
]

@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(os.path.expanduser("~/huggingface/Qwen3-0.6B"))

@pytest.mark.parametrize("s", CASES)
def test_incremental_equals_full(tok, s):
    ids = tok.encode(s)
    d = StreamingDetokenizer(tok)
    out = "".join(d.feed(0, t) for t in ids) + d.flush(0)
    assert out == tok.decode(ids)

def test_state_is_freed(tok):
    d = StreamingDetokenizer(tok)
    for t in tok.encode("你好🎉"): d.feed(0, t)
    d.flush(0)
    assert d._ids == {} and d._emitted == {}

def test_interleaved_sequences(tok):
    a, b = tok.encode("你好世界"), tok.encode("🎉 party")
    d = StreamingDetokenizer(tok)
    oa = ob = ""
    for x, y in zip(a, b):                   # simulate interleaved stream
        oa += d.feed(0, x); ob += d.feed(1, y)
    for x in a[len(b):]: oa += d.feed(0, x)
    for y in b[len(a):]: ob += d.feed(1, y)
    assert oa + d.flush(0) == tok.decode(a)
    assert ob + d.flush(1) == tok.decode(b)