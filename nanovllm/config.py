import os
from dataclasses import dataclass, field
from math import isfinite

from transformers import AutoConfig


def _positive_hf_integer(hf_config, role: str, name: str) -> int:
    value = getattr(hf_config, name, None)
    if type(value) is not int or value < 1:
        raise ValueError(
            f"{role} model must define a positive integer {name}"
        )
    return value


def _require_safetensors(path: str, role: str) -> None:
    try:
        files = [
            entry.path
            for entry in os.scandir(path)
            if entry.name.endswith(".safetensors") and entry.is_file()
        ]
    except OSError as error:
        raise ValueError(
            f"cannot inspect {role} model directory {path!r}: {error}"
        ) from error
    if not files:
        raise ValueError(
            f"{role} model directory must contain at least one "
            f"safetensors weight file: {path!r}"
        )


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
    # Speculative decoding is an engine-wide opt-in. Keep these public fields at
    # the end so every existing positional Config construction retains its
    # meaning. The loaded draft config is derived state, never caller input.
    draft_model: str | os.PathLike | None = None
    num_speculative_tokens: int = 0
    draft_hf_config: AutoConfig | None = field(
        init=False,
        default=None,
        repr=False,
    )

    @property
    def configured_k(self) -> int:
        return self.num_speculative_tokens

    @property
    def speculation_enabled(self) -> bool:
        return self.num_speculative_tokens > 0

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

        if self.draft_model is not None:
            if isinstance(self.draft_model, os.PathLike):
                self.draft_model = os.fsdecode(os.fspath(self.draft_model))
            elif not isinstance(self.draft_model, str):
                raise TypeError(
                    "draft_model must be None, a path string, or os.PathLike"
                )
        if type(self.num_speculative_tokens) is not int:
            raise TypeError("num_speculative_tokens must be an integer")
        if self.num_speculative_tokens < 0:
            raise ValueError("num_speculative_tokens must be >= 0")
        if self.speculation_enabled and self.draft_model is None:
            raise ValueError(
                "num_speculative_tokens > 0 requires draft_model"
            )
        if not self.speculation_enabled and self.draft_model is not None:
            raise ValueError(
                "draft_model requires num_speculative_tokens > 0"
            )

        if self.speculation_enabled:
            if self.tensor_parallel_size != 1:
                raise ValueError(
                    "speculative decoding currently supports "
                    "tensor_parallel_size=1 only"
                )
            if self.top_p_backend != "exact":
                raise ValueError(
                    "speculative decoding requires top_p_backend='exact'"
                )
            if not os.path.isdir(self.draft_model):
                raise ValueError(
                    "draft model path is not a directory: "
                    f"{self.draft_model!r}"
                )
            # V2 pays for two model owners. Fail before tokenizer, workers,
            # process groups, or CUDA allocation when either directory cannot
            # supply the only weight format understood by nano-vLLM's loader.
            _require_safetensors(self.model, "target")
            _require_safetensors(self.draft_model, "draft")

        self.hf_config = AutoConfig.from_pretrained(self.model)
        if not self.speculation_enabled:
            # Preserve the established target-only path and its compatibility.
            self.max_model_len = min(
                self.max_model_len,
                self.hf_config.max_position_embeddings,
            )
            return

        self.draft_hf_config = AutoConfig.from_pretrained(self.draft_model)
        target_model_type = getattr(self.hf_config, "model_type", None)
        draft_model_type = getattr(self.draft_hf_config, "model_type", None)
        if target_model_type != "qwen3" or draft_model_type != "qwen3":
            raise ValueError(
                "speculative decoding currently requires Qwen3 target and "
                "draft models; got "
                f"target={target_model_type!r}, draft={draft_model_type!r}"
            )
        target_vocab_size = getattr(self.hf_config, "vocab_size", None)
        draft_vocab_size = getattr(self.draft_hf_config, "vocab_size", None)
        if type(target_vocab_size) is not int or target_vocab_size < 1:
            raise ValueError(
                "target model must define a positive integer vocab_size"
            )
        if type(draft_vocab_size) is not int or draft_vocab_size < 1:
            raise ValueError(
                "draft model must define a positive integer vocab_size"
            )
        if target_vocab_size != draft_vocab_size:
            raise ValueError(
                "target and draft vocab_size must match; got "
                f"target={target_vocab_size!r}, draft={draft_vocab_size!r}"
            )
        target_position_limit = _positive_hf_integer(
            self.hf_config,
            "target",
            "max_position_embeddings",
        )
        draft_position_limit = _positive_hf_integer(
            self.draft_hf_config,
            "draft",
            "max_position_embeddings",
        )
        self.max_model_len = min(
            self.max_model_len,
            target_position_limit,
            draft_position_limit,
        )
        if min(
            self.configured_k,
            self.max_model_len - 1,
            self.max_num_batched_tokens - 1,
        ) < 1:
            raise ValueError(
                "speculative decoding has no globally usable proposal slot; "
                "max_model_len and max_num_batched_tokens must both be at "
                "least 2 after target/draft position-limit clamping"
            )
