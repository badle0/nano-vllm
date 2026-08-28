from copy import copy
from enum import Enum, auto
from itertools import count
from time import perf_counter
from typing import NamedTuple

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()
    CANCELLED = auto()

class StreamOutput(NamedTuple):
    seq_id: int
    token_id: int
    finished: bool

class Sequence:
    block_size = 256
    counter = count()

    def __init__(
        self,
        token_ids: list[int],
        sampling_params=SamplingParams(),
        *,
        submission_time: float | None = None,
        engine_arrival_time: float | None = None,
    ):
        if len(token_ids) == 0:
            raise ValueError("prompt must contain at least one token")
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        # Target and draft models have separate KV tensors even though they use
        # one scheduler-owned block-ID table.  A target prefix-cache hit is not
        # evidence that the draft tensor contains the same prefix, so draft
        # coverage always starts cold and is reset whenever block identity is
        # released or replaced.
        self.num_draft_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.top_k = sampling_params.top_k
        self.top_p = sampling_params.top_p
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        if submission_time is None or engine_arrival_time is None:
            now = perf_counter()
            if submission_time is None:
                submission_time = now
            if engine_arrival_time is None:
                engine_arrival_time = now
        self.submission_time = submission_time
        self.engine_arrival_time = engine_arrival_time
        self.first_scheduled_time = None
        self.first_token_time = None
        self.finish_time = None
        self.first_delivery_time = None
        self.delivery_time = None
        self.token_times = []

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
