class StreamingDetokenizer:
    """Incremental text assembly for streamed token IDs.

    decode(t1..tk) != decode(t1..tk-1) + decode(tk) in general: byte-level BPE
    can split one UTF-8 code point across tokens, and space markers attach at
    token boundaries. This keeps per-sequence state, decodes cumulatively, and
    holds back any trailing fragment while the decode ends in U+FFFD.

    Cumulative decode is O(n^2) over a sequence; fine at nano's output lengths.
    A sliding window is the production fix if the null-consumer benchmark shows it.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._ids: dict[int, list[int]] = {}
        self._emitted: dict[int, int] = {}      # chars already returned, per seq

    def feed(self, seq_id: int, token_id: int) -> str:
        ids = self._ids.setdefault(seq_id, [])
        ids.append(token_id)
        text = self.tokenizer.decode(ids)
        if text.endswith("\ufffd"):             # incomplete character: hold back
            return ""
        n = self._emitted.get(seq_id, 0)
        self._emitted[seq_id] = len(text)
        return text[n:]

    def flush(self, seq_id: int) -> str:
        ids = self._ids.pop(seq_id, [])
        n = self._emitted.pop(seq_id, 0)
        return self.tokenizer.decode(ids)[n:] if ids else ""