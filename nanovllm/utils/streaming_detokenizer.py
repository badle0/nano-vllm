from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class TextUpdate:
    """A correction to one rendered sequence.

    ``replace_from`` and ``delete_count`` are Python Unicode code-point
    offsets. Applying an update replaces the named span with ``insert``.
    """

    seq_id: int
    replace_from: int
    delete_count: int
    insert: str
    final: bool = False

    def apply(self, text: str) -> str:
        if self.replace_from < 0 or self.replace_from > len(text):
            raise ValueError("replace_from lies outside the current text")
        end = self.replace_from + self.delete_count
        if self.delete_count < 0 or end > len(text):
            raise ValueError("delete_count lies outside the current text")
        return text[:self.replace_from] + self.insert + text[end:]


@dataclass(slots=True)
class _DecodeState:
    token_ids: list[int] = field(default_factory=list)
    window_start_token: int = 0
    window_start_char: int = 0
    window_text: str = ""
    rendered_length: int = 0


class StreamingDetokenizer:
    """Bounded-window streamed decoding with correction-capable output.

    Tokenizer decoding is not generally prefix-monotone: a later token can
    replace a preceding space, normalization fragment, or incomplete UTF-8
    character. ``feed`` therefore returns a ``TextUpdate`` instead of an
    append-only string.

    State is split into a stable prefix, represented by its character length,
    and a decoded unstable tail. The frontier advances only when independently
    decoding the committed and retained token slices exactly reconstructs the
    current tail. ``flush`` performs one exact full decode and emits a final
    correction. For a fixed window this changes tokenizer work from quadratic
    to linear. Intermediate correctness assumes rewrites remain inside the
    configured overlap; final text is always exact.

    Short-lived trailing replacement characters are withheld. If one persists
    until the normal frontier threshold, it is emitted as correctable text so
    an exactly splittable invalid-byte run cannot prevent bounded progress.
    """

    def __init__(self, tokenizer, window_size: int = 32, boundary_overlap: int = 8):
        if type(window_size) is not int or window_size < 2:
            raise ValueError("window_size must be an integer of at least 2")
        if type(boundary_overlap) is not int or boundary_overlap < 1:
            raise ValueError("boundary_overlap must be a positive integer")
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.boundary_overlap = boundary_overlap
        self._states: dict[int, _DecodeState] = {}

    @staticmethod
    def _common_prefix_length(left: str, right: str) -> int:
        limit = min(len(left), len(right))
        index = 0
        while index < limit and left[index] == right[index]:
            index += 1
        return index

    def _advance_frontier(self, state: _DecodeState):
        window_tokens = len(state.token_ids) - state.window_start_token
        if window_tokens < self.window_size + self.boundary_overlap:
            return

        target = len(state.token_ids) - self.window_size
        lower = max(
            state.window_start_token + 1,
            target - self.boundary_overlap,
        )
        for candidate in range(target, lower - 1, -1):
            committed = self.tokenizer.decode(
                state.token_ids[state.window_start_token:candidate]
            )
            retained = self.tokenizer.decode(state.token_ids[candidate:])
            if committed + retained != state.window_text:
                continue
            state.window_start_token = candidate
            state.window_start_char += len(committed)
            state.window_text = retained
            return

    def feed(self, seq_id: int, token_id: int) -> TextUpdate:
        state = self._states.setdefault(seq_id, _DecodeState())
        hard_limit = self.window_size + 2 * self.boundary_overlap
        if len(state.token_ids) - state.window_start_token >= hard_limit:
            raise RuntimeError(
                "tokenizer rewrite exceeded the configured boundary overlap"
            )
        state.token_ids.append(token_id)
        new_window = self.tokenizer.decode(
            state.token_ids[state.window_start_token:]
        )

        window_tokens = len(state.token_ids) - state.window_start_token
        frontier_threshold = self.window_size + self.boundary_overlap
        # Hide ordinary incomplete UTF-8 fragments, but do not let a persistent
        # replacement suffix bypass frontier advancement until the hard guard.
        # TextUpdate can repair an emitted replacement when later tokens make
        # the tokenizer output complete.
        if (
            new_window.endswith("\ufffd")
            and window_tokens < frontier_threshold
        ):
            return TextUpdate(seq_id, state.rendered_length, 0, "")

        common = self._common_prefix_length(state.window_text, new_window)
        update = TextUpdate(
            seq_id=seq_id,
            replace_from=state.window_start_char + common,
            delete_count=len(state.window_text) - common,
            insert=new_window[common:],
        )
        state.window_text = new_window
        state.rendered_length = state.window_start_char + len(new_window)
        self._advance_frontier(state)
        return update

    def flush(self, seq_id: int) -> TextUpdate:
        state = self._states.pop(seq_id, None)
        if state is None:
            return TextUpdate(seq_id, 0, 0, "", final=True)
        exact = self.tokenizer.decode(state.token_ids)
        return TextUpdate(
            seq_id=seq_id,
            replace_from=0,
            delete_count=state.rendered_length,
            insert=exact,
            final=True,
        )

    def discard(self, seq_id: int):
        self._states.pop(seq_id, None)
