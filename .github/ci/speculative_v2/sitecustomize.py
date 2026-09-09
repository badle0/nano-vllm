"""Narrow CPU-CI shim for nano-vLLM's eager Qwen import.

The selected contract tests mock model construction and must never execute a
real Qwen kernel.  The production package imports Qwen (and therefore CUDA-only
attention dependencies) eagerly, so CPU CI preloads that leaf module with
a fail-on-construction placeholder. Loader contract tests also import layer
classes whose invariant backend requires Triton; a fail-on-execution placeholder
keeps those CPU weight-loading tests independent of GPU dependencies.  GPU/model integration is certified by the
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


INVARIANT_MODULE_NAME = "nanovllm.layers.invariant_ops"

if INVARIANT_MODULE_NAME not in sys.modules:
    invariant_module = types.ModuleType(INVARIANT_MODULE_NAME)

    def unavailable_invariant_kernel(*args, **kwargs):
        raise RuntimeError("CPU CI must not execute invariant GPU kernels")

    invariant_module.invariant_linear = unavailable_invariant_kernel
    invariant_module.invariant_rms_norm = unavailable_invariant_kernel
    sys.modules[INVARIANT_MODULE_NAME] = invariant_module
