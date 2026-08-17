from dataclasses import dataclass
from math import isfinite


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    top_k: int = -1          # -1 = disabled (consider all tokens)
    top_p: float = 1.0       # 1.0 = disabled (full nucleus)
    max_tokens: int = 64
    ignore_eos: bool = False
    top_k: int = -1          # -1 = disabled (consider all tokens)
    top_p: float = 1.0       # 1.0 = disabled (full nucleus)

    def __post_init__(self):
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)):
            raise TypeError("temperature must be a number")
        if not isfinite(self.temperature):
            raise ValueError("temperature must be finite")
        if self.temperature < 0.0:
            raise ValueError("temperature must be non-negative")
        if type(self.top_k) is not int:
            raise TypeError("top_k must be an integer")
        if self.top_k != -1 and self.top_k < 1:
            raise ValueError("top_k must be -1 (disabled) or >= 1")
        if isinstance(self.top_p, bool) or not isinstance(self.top_p, (int, float)):
            raise TypeError("top_p must be a number")
        if not isfinite(self.top_p):
            raise ValueError("top_p must be finite")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
