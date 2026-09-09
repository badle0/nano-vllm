import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()
        self.numerical_mode = "fast"

    @staticmethod
    def _apply(x: torch.Tensor) -> torch.Tensor:
        gate, value = x.chunk(2, -1)
        return F.silu(gate) * value

    @torch.compile
    def compiled_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._apply(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.numerical_mode == "invariant":
            return self._apply(x)
        return self.compiled_forward(x)
