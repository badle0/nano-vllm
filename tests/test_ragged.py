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

def test_sequence_pickle_keeps_ordinary_object_state():
    # Tensor-parallel compression belongs to its transport DTO, not Sequence's
    # general pickle protocol.  Check fields that the old custom state dropped.
    import pickle
    from nanovllm.engine.sequence import Sequence, SequenceStatus

    seq = Sequence([1, 2, 3])
    seq.is_prefill = False
    seq.status = SequenceStatus.RUNNING
    seq.block_table = [7]
    seq.first_scheduled_time = 12.5

    restored = pickle.loads(pickle.dumps(seq))

    assert restored.__dict__ == seq.__dict__
    assert restored.token_ids == [1, 2, 3]
    assert restored.is_prefill is False
    assert restored.status is SequenceStatus.RUNNING
