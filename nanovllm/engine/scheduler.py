from collections import deque
from time import perf_counter

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus, StreamOutput
from nanovllm.engine.block_manager import BlockManager


class SchedulerCapacityError(RuntimeError):

    def __init__(self, requested: int, available: int, capacity: int):
        self.requested = requested
        self.available = available
        self.capacity = capacity
        super().__init__(
            f"cannot admit {requested} request(s): only {available} of "
            f"{capacity} scheduler slots are available; split the batch or "
            "retry after requests finish or are cancelled"
        )


class Scheduler:

    def __init__(self, config: Config, clock=None):
        self._clock = perf_counter if clock is None else clock
        self.max_num_seqs = config.max_num_seqs
        self._max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.mid_chunk_seq: Sequence | None = None

    @property
    def max_num_batched_tokens(self) -> int:
        """Constructor-configured token budget used by scheduling and graphs."""
        return self._max_num_batched_tokens

    @property
    def available_capacity(self) -> int:
        return self.max_num_seqs - len(self.waiting) - len(self.running)

    def require_capacity(self, requested: int = 1):
        if requested < 0:
            raise ValueError("requested capacity must be non-negative")
        available = self.available_capacity
        if requested > available:
            raise SchedulerCapacityError(
                requested=requested,
                available=max(available, 0),
                capacity=self.max_num_seqs,
            )

    def _check_mid_chunk_invariant(self):
        mid = self.mid_chunk_seq
        if mid is None:
            if self.waiting and self.waiting[0].block_table:
                raise RuntimeError("waiting head owns KV blocks without mid_chunk_seq")
            return
        if not self.waiting or self.waiting[0] is not mid:
            raise RuntimeError("mid_chunk_seq must remain at the waiting head")
        if not mid.block_table:
            raise RuntimeError("mid_chunk_seq must own its allocated KV blocks")

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.require_capacity()
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        self._check_mid_chunk_invariant()
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

        # FIFO chunk fill to the remaining budget: each seq takes min(work, remaining),
        # so only the last admitted seq can be partial (<=1 partial per step, F2)
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining <= 0:
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
                if seq is self.mid_chunk_seq:
                    self.mid_chunk_seq = None
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            if seq.first_scheduled_time is None:
                seq.first_scheduled_time = self._clock()
            scheduled_seqs.append(seq)
            if seq.num_scheduled_tokens < num_tokens:   # partial => budget exhausted
                if self.mid_chunk_seq not in (None, seq):
                    raise RuntimeError("more than one sequence is mid-chunk")
                self.mid_chunk_seq = seq
                break

        assert scheduled_seqs
        self._check_mid_chunk_invariant()
        # is_prefill return semantics are now "ragged step": any prefill work present
        return scheduled_seqs, len(scheduled_seqs) > num_decodes

    def preempt(self, seq: Sequence):
        if seq is self.mid_chunk_seq:
            self.mid_chunk_seq = None
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        # A preempted victim must not jump ahead of an already-waiting request.
        self.waiting.append(seq)
        self._check_mid_chunk_invariant()

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[StreamOutput]:
        now = self._clock()
        events: list[StreamOutput] = []
        for seq, token_id in zip(seqs, token_ids, strict=True):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            # emission gate: skip mid-prefill rows. After the increment above,
            # cached < total is exact — decode rows always arrive at equality
            # (preemption zeroes cached and re-enters via the prefill path)
            if seq.num_cached_tokens < seq.num_tokens:
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

    def cancel(self, seq_ids) -> list[int]:
        """Cancel only the queued sequences named by ``seq_ids``.

        A partially prefetched sequence can still be in ``waiting`` while it
        owns KV blocks, so both scheduler queues must use the same deallocation
        rule. Unknown and duplicate IDs are harmless.
        """
        targets = set(seq_ids)
        if not targets:
            return []

        cancelled = []
        for queue in (self.waiting, self.running):
            retained = deque()
            while queue:
                seq = queue.popleft()
                if seq.seq_id not in targets:
                    retained.append(seq)
                    continue
                if seq.block_table:
                    self.block_manager.deallocate(seq)
                if seq is self.mid_chunk_seq:
                    self.mid_chunk_seq = None
                seq.num_scheduled_tokens = 0
                seq.status = SequenceStatus.CANCELLED
                cancelled.append(seq.seq_id)
            queue.extend(retained)
        self._check_mid_chunk_invariant()
        return cancelled

    def cancel_all(self):
        """Administrative compatibility wrapper; request cleanup uses cancel."""
        return self.cancel(
            seq.seq_id for seq in (*self.running, *self.waiting)
        )
