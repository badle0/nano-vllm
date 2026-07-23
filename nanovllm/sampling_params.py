from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    top_k: int = -1          # -1 = disabled (consider all tokens)
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature >= 0.0, "temperature must be non-negative"
        assert self.top_k == -1 or self.top_k >= 1, "top_k must be -1 (disabled) or >= 1"
