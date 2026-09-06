"""Narrow CPU-CI shim for nano-vLLM's eager Qwen import.

The selected contract tests mock model construction and must never execute a
real Qwen kernel.  The production package imports Qwen (and therefore CUDA-only
attention dependencies) eagerly, so CPU CI preloads only that leaf module with
a fail-on-construction placeholder.  GPU/model integration is certified by the
separate A100 lifecycle harness.
"""

import sys
import types


MODULE_NAME = "nanovllm.models.qwen3"


if MODULE_NAME not in sys.modules:
    module = types.ModuleType(MODULE_NAME)

    class Qwen3ForCausalLM:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "CPU CI must mock Qwen3ForCausalLM construction"
            )

    module.Qwen3ForCausalLM = Qwen3ForCausalLM
    sys.modules[MODULE_NAME] = module
