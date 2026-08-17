import pytest

@pytest.fixture(scope="session")
def llm():
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("engine tests need a GPU")
    except Exception:
        pytest.skip("engine tests need torch")
    from nanovllm import LLM
    # ONE engine per pytest process — nano-vllm engines cannot coexist:
    # unconditional dist.init_process_group (model_runner:26), atexit-pinned
    # 0.9 memory grab (llm_engine:36), fixed shm name for TP>1.
    return LLM(
        "/workspace/models/Qwen3-0.6B",
        enforce_eager=False,
        max_model_len=4096,
    )
