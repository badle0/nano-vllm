import torch

def test_prepare_ragged_matches_prepare_decode(llm):
    from nanovllm import SamplingParams
    from nanovllm.utils.context import get_context, reset_context
    llm.add_request("The capital of France is",
                    SamplingParams(temperature=0.6, max_tokens=4, ignore_eos=True))
    llm.step()                                        # prefill: seq now decoding
    seqs, is_prefill = llm.scheduler.schedule()
    assert not is_prefill
    ids_d, pos_d = llm.model_runner.prepare_decode(seqs)
    slot_d = get_context().slot_mapping.clone()
    ids_r, pos_r = llm.model_runner.prepare_ragged(seqs)
    ctx = get_context()
    assert torch.equal(ids_r, ids_d) and torch.equal(pos_r, pos_d)
    assert torch.equal(ctx.slot_mapping, slot_d)
    assert ctx.cu_seqlens_q[-1].item() == len(seqs)   # one query per seq
    assert ctx.block_tables is not None               # k > q ⇒ paged branch armed
    reset_context()
    while not llm.is_finished(): llm.step()           # drain; leave engine clean

def test_setstate_restores_is_prefill():
    # TP workers receive Sequences via pickle (bypasses __init__); prepare_ragged
    # branches on seq.is_prefill per row, so __setstate__ must restore it — the
    # payload encodes it (prefill ships token_ids list, decode ships last_token)
    import pickle
    from nanovllm.engine.sequence import Sequence
    pre = Sequence([1, 2, 3])
    assert pre.is_prefill
    rt = pickle.loads(pickle.dumps(pre))
    assert rt.is_prefill is True and rt.last_token == 3
    dec = Sequence([1, 2, 3])
    dec.is_prefill = False                     # what the scheduler does on decode admission
    rt = pickle.loads(pickle.dumps(dec))
    assert rt.is_prefill is False and rt.last_token == 3 and rt.token_ids == []
