import os
from dataclasses import dataclass
from math import isfinite

from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str | os.PathLike
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    top_p_backend: str = "exact"
    # Appended to preserve Config's existing positional field order. This is an
    # opt-in, reference-counted process-global lease owned by LLMEngine.
    disable_python_gc: bool = False

    def __post_init__(self):
        if not isinstance(self.top_p_backend, str):
            raise TypeError("top_p_backend must be a string")
        if self.top_p_backend not in {"exact", "flashinfer"}:
            raise ValueError(
                "top_p_backend must be either 'exact' or 'flashinfer'"
            )
        if type(self.disable_python_gc) is not bool:
            raise TypeError("disable_python_gc must be a bool")
        for name in ("max_num_batched_tokens", "max_num_seqs"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError(
                "max_num_batched_tokens must be greater than or equal to "
                "max_num_seqs"
            )
        if type(self.kvcache_block_size) is not int:
            raise TypeError("kvcache_block_size must be an integer")
        if self.kvcache_block_size <= 0 or self.kvcache_block_size % 256 != 0:
            raise ValueError(
                "kvcache_block_size must be a positive multiple of 256"
            )
        if type(self.tensor_parallel_size) is not int:
            raise TypeError("tensor_parallel_size must be an integer")
        if not 1 <= self.tensor_parallel_size <= 8:
            raise ValueError("tensor_parallel_size must be between 1 and 8")
        if self.disable_python_gc and self.tensor_parallel_size != 1:
            raise ValueError(
                "disable_python_gc currently supports tensor_parallel_size=1 only"
            )
        if type(self.max_model_len) is not int:
            raise TypeError("max_model_len must be an integer")
        if self.max_model_len < 1:
            raise ValueError("max_model_len must be >= 1")
        if isinstance(self.gpu_memory_utilization, bool) or not isinstance(
            self.gpu_memory_utilization, (int, float)
        ):
            raise TypeError("gpu_memory_utilization must be a number")
        if not isfinite(self.gpu_memory_utilization):
            raise ValueError("gpu_memory_utilization must be finite")
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if type(self.enforce_eager) is not bool:
            raise TypeError("enforce_eager must be a bool")
        if type(self.num_kvcache_blocks) is not int:
            raise TypeError("num_kvcache_blocks must be an integer")
        if self.num_kvcache_blocks != -1 and self.num_kvcache_blocks < 1:
            raise ValueError(
                "num_kvcache_blocks must be -1 (auto) or a positive integer"
            )
        if isinstance(self.model, os.PathLike):
            self.model = os.fsdecode(os.fspath(self.model))
        elif not isinstance(self.model, str):
            raise TypeError("model must be a path string or os.PathLike")
        if not os.path.isdir(self.model):
            raise ValueError(f"model path is not a directory: {self.model!r}")
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
