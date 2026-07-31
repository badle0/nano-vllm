from collections import deque
from time import perf_counter

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus, StreamOutput
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []

        # decode admission first, unconditionally (F2): the ITL bound exists only if
        # decodes never wait behind prefill work — the loop is dev's decode loop verbatim
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        num_decodes = len(scheduled_seqs)
        num_batched_tokens = num_decodes    # decodes charge 1 each (F2 budget accounting)
        self.running.extendleft(reversed(scheduled_seqs))

        # the unique mid-chunk seq (live block_table while waiting) re-takes the budget
        # head even if a preemption appendleft-ed in front of it — this is what keeps
        # the <=1-mid-chunk-system-wide invariant true under preemption (F2)
        mid = next((s for s in self.waiting if s.block_table), None)
        if mid is not None and self.waiting[0] is not mid:
            self.waiting.remove(mid)
            self.waiting.appendleft(mid)

        # FIFO chunk fill to the remaining budget: each seq takes min(work, remaining),
        # so only the last admitted seq can be partial (<=1 partial per step, F2)
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            seq = self.waiting[0]
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
                self.block_manager.allocate(seq, num_cached_blocks)
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            if seq.first_scheduled_time is None:
                seq.first_scheduled_time = perf_counter()
            scheduled_seqs.append(seq)
            if seq.num_scheduled_tokens < num_tokens:   # partial => budget exhausted
                break

        assert scheduled_seqs
        assert sum(1 for s in self.waiting if s.block_table) <= 1   # <=1 mid-chunk (F2)
        # is_prefill return semantics are now "ragged step": any prefill work present
        return scheduled_seqs, len(scheduled_seqs) > num_decodes

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int],
                    is_prefill: bool) -> list[StreamOutput]:
        now = perf_counter()
        events: list[StreamOutput] = []
        for seq, token_id in zip(seqs, token_ids, strict=True):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if seq.first_token_time is None:
                seq.first_token_time = now
            seq.token_times.append(now)
            finished = (not seq.ignore_eos and token_id == self.eos) \
                    or seq.num_completion_tokens == seq.max_tokens
            if finished:
                seq.finish_time = now
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
            events.append(StreamOutput(seq.seq_id, token_id, finished))
        return events
    
    def cancel_all(self):
        for seq in (*self.running, *self.waiting):
            if seq.block_table:
                self.block_manager.deallocate(seq)
        self.running.clear()
        self.waiting.clear()
