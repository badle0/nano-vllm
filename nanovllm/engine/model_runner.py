import gc
import pickle
import struct
from datetime import timedelta
from typing import NamedTuple

import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.speculative_memory import (
    SpeculativeMemoryPlan,
    kv_cache_block_bytes,
    plan_speculative_workspace,
)
from nanovllm.engine.tp_transport import ScheduledSequence, compact_run_args
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.errors import record_cleanup_failure
from nanovllm.utils.loader import load_model


TP_SHM_MAGIC = b"NVTP"
TP_SHM_HEADER = struct.Struct("<4sI")
TP_SHM_MIN_SIZE = 64 * 1024
TP_SHM_MAX_SIZE = 64 * 1024 * 1024
TP_PROCESS_GROUP_TIMEOUT = timedelta(seconds=30)
SPECULATIVE_GRAPH_ALLOCATOR_MARGIN_BYTES = 64 * 1024 * 1024


class TensorParallelTransportError(RuntimeError):
    pass


class SpeculativeKVCacheCapacityError(RuntimeError):
    """The configured dual cache plus reserved workspace cannot fit."""


def _clear_rope_cache():
    """Release the one-engine-per-process global rotary-module owner."""

    get_rope.cache_clear()


class KVCacheBindingError(RuntimeError):
    """A model's attention-layer geometry does not match its HF config."""


class SpeculativeMemoryAudit(NamedTuple):
    """Immutable measured/modelled memory ledger for one V2 runner."""

    workspace_plan: SpeculativeMemoryPlan
    total_memory_bytes: int
    free_before_kv_bytes: int
    used_before_kv_bytes: int
    memory_budget_bytes: int
    allocated_before_kv_bytes: int
    reserved_before_kv_bytes: int
    peak_before_kv_bytes: int
    warmup_transient_bytes: int
    target_warmup_transient_bytes: int
    draft_warmup_transient_bytes: int
    target_weight_bytes: int
    draft_weight_bytes: int
    target_block_bytes: int
    draft_block_bytes: int
    joint_block_bytes: int
    # Graph ownership fields are end-of-capture deltas; graph peak fields are
    # incremental high-water deltas from the same pre-capture baseline.
    profiled_graph_allocated_bytes: int
    profiled_graph_reserved_bytes: int
    profiled_graph_peak_allocated_bytes: int
    profiled_graph_peak_reserved_bytes: int
    profiled_graph_ownership_bytes: int
    profiled_graph_peak_bytes: int
    graph_allocator_margin_bytes: int
    graph_construction_reservation_bytes: int
    # Compatibility alias for graph_construction_reservation_bytes.
    graph_reservation_bytes: int
    runtime_reservation_bytes: int
    sizing_overhead_bytes: int
    selected_num_blocks: int
    target_kv_bytes: int
    draft_kv_bytes: int
    allocated_after_kv_before_graph_bytes: int
    reserved_after_kv_before_graph_bytes: int
    allocated_after_graph_before_pretouch_bytes: int
    reserved_after_graph_before_pretouch_bytes: int
    final_graph_allocated_bytes: int
    final_graph_reserved_bytes: int
    final_graph_peak_allocated_bytes: int
    final_graph_peak_reserved_bytes: int
    post_init_allocated_bytes: int
    post_init_reserved_bytes: int
    post_init_free_bytes: int
    post_init_budget_headroom_bytes: int
    modeled_runtime_headroom_bytes: int
    audit_required_components: tuple[str, ...]
    gpu_certified: bool


def tensor_parallel_shm_size(config: Config) -> int:
    """Return a bounded transport size derived from the configured work limits.

    Pickle uses at most five bytes for the non-negative 32-bit token/block IDs
    expected here.  Sixteen bytes per variable integer, 256 bytes per sequence,
    and 64 KiB of framing slack leave a conservative margin for containers and
    fixed metadata while still putting an explicit 64 MiB ceiling on allocation.
    Every actual frame is checked against the allocated buffer before publication.
    """

    max_blocks_per_seq = (
        config.max_model_len + config.kvcache_block_size - 1
    ) // config.kvcache_block_size
    variable_ints = (
        config.max_num_batched_tokens
        + config.max_num_seqs * max_blocks_per_seq
    )
    estimated = (
        TP_SHM_HEADER.size
        + 64 * 1024
        + 16 * variable_ints
        + 256 * config.max_num_seqs
    )
    size = max(TP_SHM_MIN_SIZE, (estimated + 4095) // 4096 * 4096)
    if size > TP_SHM_MAX_SIZE:
        raise ValueError(
            "tensor-parallel transport requires "
            f"{size} bytes, above the {TP_SHM_MAX_SIZE}-byte safety limit; "
            "reduce max_num_batched_tokens, max_num_seqs, or max_model_len"
        )
    return size


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self._closed = False
        self._owns_process_group = False
        self.shm = None
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.tp_shm_size = (
            tensor_parallel_shm_size(config) if self.world_size > 1 else 0
        )
        self.speculation_enabled = bool(
            getattr(config, "speculation_enabled", False)
        )
        if self.speculation_enabled:
            # Every draft-owned resource starts in a cleanup-safe state before
            # CUDA, NCCL, model construction, compilation, or graph capture can
            # fail. The disabled path creates no draft model/cache/graph object.
            self.draft_model = None
            self.draft_kv_cache = None
            self.draft_graphs = None
            self.draft_graph_vars = None
            self.draft_graph_pool = None
            self.draft_graph_bs = None
            self.speculative_memory_plan = None
            self.speculative_memory_audit = None
            self._speculative_memory_audit_inputs = None
            self._profiled_graph_allocated_bytes = 0
            self._profiled_graph_reserved_bytes = 0
            self._profiled_graph_peak_allocated_bytes = 0
            self._profiled_graph_peak_reserved_bytes = 0
            self._final_graph_allocated_baseline = 0
            self._final_graph_reserved_baseline = 0
            self._allocated_after_graph_before_pretouch = 0
            self._reserved_after_graph_before_pretouch = 0
            self._final_graph_peak_allocated_bytes = 0
            self._final_graph_peak_reserved_bytes = 0
            self._target_warmup_transient_bytes = 0
            self._draft_warmup_transient_bytes = 0
            self._warmup_transient_bytes = 0

        default_device = torch.get_default_device()
        default_dtype = torch.get_default_dtype()
        try:
            torch.cuda.set_device(rank)
            dist.init_process_group(
                "nccl",
                "tcp://localhost:2333",
                world_size=self.world_size,
                rank=rank,
                timeout=TP_PROCESS_GROUP_TIMEOUT,
            )
            self._owns_process_group = True
            try:
                torch.set_default_dtype(hf_config.dtype)
                torch.set_default_device("cuda")
                self.model = Qwen3ForCausalLM(hf_config)
                load_model(self.model, config.model)
                self.sampler = Sampler()
                if self.speculation_enabled:
                    self._run_draft_phase(
                        "draft model construction",
                        self._construct_draft_model,
                    )
                    self._run_draft_phase(
                        "draft weight loading",
                        self._load_draft_model,
                    )
                self.warmup_model()
                if self.speculation_enabled:
                    self._target_warmup_transient_bytes = (
                        self._profiled_transient_bytes()
                    )
                    self._run_draft_phase(
                        "draft warmup",
                        self.warmup_draft_model,
                    )
                    self._draft_warmup_transient_bytes = (
                        self._profiled_transient_bytes()
                    )
                    self._warmup_transient_bytes = max(
                        self._target_warmup_transient_bytes,
                        self._draft_warmup_transient_bytes,
                    )
                    if not self.enforce_eager:
                        self._run_draft_phase(
                            "target/draft graph-memory profile",
                            self._profile_speculative_graph_memory,
                        )
                self.allocate_kv_cache()
                if not self.enforce_eager:
                    self.capture_cudagraph()
                    self.capture_varlen_graphs()
                    if self.speculation_enabled:
                        self._run_draft_phase(
                            "draft CUDA-graph capture",
                            self.capture_draft_cudagraph,
                        )
                        self._record_final_graph_memory_peaks()
            except BaseException as error:
                restore_error = self._restore_torch_defaults(
                    default_device, default_dtype
                )
                if restore_error is not None:
                    record_cleanup_failure(
                        error,
                        "restoring Torch defaults",
                        restore_error,
                    )
                raise
            else:
                restore_error = self._restore_torch_defaults(
                    default_device, default_dtype
                )
                if restore_error is not None:
                    raise restore_error

            if not self.enforce_eager:
                self._pretouch_eager_prefill()
                if self.speculation_enabled:
                    self._run_draft_phase(
                        "draft eager-prefill pretouch",
                        self._pretouch_draft_eager_prefill,
                    )

            if self.speculation_enabled:
                self._finalize_speculative_memory_audit()

            if self.world_size > 1:
                if rank == 0:
                    self.shm = SharedMemory(
                        name="nanovllm", create=True, size=self.tp_shm_size
                    )
                    dist.barrier()
                else:
                    dist.barrier()
                    self.shm = SharedMemory(name="nanovllm")
                    self.loop()
        except BaseException as error:
            cleanup_error = self._close(abort=True)
            if cleanup_error is not None:
                record_cleanup_failure(
                    error,
                    "ModelRunner cleanup",
                    cleanup_error,
                )
            raise

    @staticmethod
    def _restore_torch_defaults(default_device, default_dtype):
        first_error = None
        try:
            torch.set_default_device(default_device)
        except BaseException as error:
            first_error = error
        try:
            torch.set_default_dtype(default_dtype)
        except BaseException as error:
            if first_error is None:
                first_error = error
        return first_error

    @staticmethod
    def _restore_rng_states(cpu_state, cuda_state):
        first_error = None
        try:
            torch.random.set_rng_state(cpu_state)
        except BaseException as error:
            first_error = error
        try:
            torch.cuda.set_rng_state(cuda_state)
        except BaseException as error:
            if first_error is None:
                first_error = error
        return first_error

    def _run_draft_phase(self, phase_name, operation):
        """Run one fallible draft phase without perturbing caller RNG state."""

        cpu_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state()
        try:
            result = operation()
        except BaseException as error:
            restore_error = self._restore_rng_states(cpu_state, cuda_state)
            if restore_error is not None:
                record_cleanup_failure(
                    error,
                    f"restoring RNG after {phase_name}",
                    restore_error,
                )
            raise
        restore_error = self._restore_rng_states(cpu_state, cuda_state)
        if restore_error is not None:
            raise restore_error
        return result

    def _construct_draft_model(self):
        previous_dtype = torch.get_default_dtype()
        primary_error = None
        try:
            torch.set_default_dtype(self.config.draft_hf_config.dtype)
            self.draft_model = Qwen3ForCausalLM(
                self.config.draft_hf_config
            )
        except BaseException as error:
            primary_error = error
            raise
        finally:
            try:
                torch.set_default_dtype(previous_dtype)
            except BaseException as restore_error:
                if primary_error is None:
                    raise
                record_cleanup_failure(
                    primary_error,
                    "restoring target dtype after draft construction",
                    restore_error,
                )

    def _load_draft_model(self):
        load_model(self.draft_model, self.config.draft_model)

    @staticmethod
    def _profiled_transient_bytes():
        stats = torch.cuda.memory_stats()
        peak = stats["allocated_bytes.all.peak"]
        current = stats["allocated_bytes.all.current"]
        return max(peak - current, 0)

    def exit(self):
        error = self._close(abort=False)
        if error is not None:
            raise error

    def _close(self, *, abort: bool) -> BaseException | None:
        """Release all owned resources; return the first cleanup error."""
        if self._closed:
            return None
        self._closed = True
        first_error = None

        def attempt(operation):
            nonlocal first_error
            try:
                operation()
            except BaseException as error:
                if first_error is None:
                    first_error = error

        def probe(operation) -> bool:
            nonlocal first_error
            try:
                return bool(operation())
            except BaseException as error:
                if first_error is None:
                    first_error = error
                return False

        shm, self.shm = self.shm, None
        if shm is not None:
            attempt(shm.close)
            if not abort and self._owns_process_group:
                attempt(dist.barrier)
            if self.rank == 0:
                attempt(shm.unlink)

        # Failed graph capture or eager pretouch can leave global tensor
        # references even when the owning attribute was never completed.
        attempt(reset_context)
        if probe(torch.cuda.is_initialized):
            attempt(torch.cuda.synchronize)

        # Graphs own references into their pool/static buffers, so release
        # graph execs explicitly before those tensors and before either model
        # itself.  Merely deleting CUDAGraph wrappers can leave private-pool
        # allocations active across a failed constructor in this PyTorch build.
        for collection_name in (
            "draft_graphs",
            "varlen_graphs",
            "graphs",
        ):
            collection = getattr(self, collection_name, None)
            if isinstance(collection, dict):
                for graph in collection.values():
                    reset = getattr(graph, "reset", None)
                    if callable(reset):
                        attempt(reset)

        for name in (
            "draft_graphs",
            "varlen_graphs",
            "graphs",
            "draft_graph_pool",
            "graph_pool",
            "draft_graph_vars",
            "varlen_vars",
            "graph_vars",
            "draft_graph_bs",
        ):
            if hasattr(self, name):
                attempt(lambda name=name: delattr(self, name))

        for model_name in ("draft_model", "model"):
            attempt(
                lambda model_name=model_name: self._clear_model_cache_views(
                    getattr(self, model_name, None)
                )
            )

        for name in (
            "draft_kv_cache",
            "kv_cache",
            "sampler",
            "draft_model",
            "model",
            "speculative_memory_plan",
            "speculative_memory_audit",
            "_speculative_memory_audit_inputs",
        ):
            if hasattr(self, name):
                attempt(lambda name=name: delattr(self, name))

        # get_rope() caches the most recently constructed RotaryEmbedding,
        # including its device buffer.  One engine per process means no healthy
        # peer can own that entry when this runner closes; retaining it would
        # leak model CUDA state across failure/restart boundaries.
        attempt(_clear_rope_cache)

        if self._owns_process_group:
            try:
                if probe(dist.is_initialized):
                    attempt(dist.destroy_process_group)
            finally:
                self._owns_process_group = False

        attempt(gc.collect)
        if probe(torch.cuda.is_initialized):
            attempt(torch.cuda.empty_cache)
        return first_error

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        try:
            buffer_size = len(self.shm.buf)
            if buffer_size < TP_SHM_HEADER.size:
                raise TensorParallelTransportError(
                    "tensor-parallel shared-memory buffer is smaller than its header"
                )
            magic, n = TP_SHM_HEADER.unpack(
                bytes(self.shm.buf[:TP_SHM_HEADER.size])
            )
            capacity = buffer_size - TP_SHM_HEADER.size
            if magic != TP_SHM_MAGIC:
                raise TensorParallelTransportError(
                    "invalid tensor-parallel shared-memory header"
                )
            if n == 0 or n > capacity:
                raise TensorParallelTransportError(
                    "invalid tensor-parallel payload length: "
                    f"{n} bytes for {capacity}-byte capacity"
                )
            try:
                payload = pickle.loads(
                    bytes(self.shm.buf[TP_SHM_HEADER.size:TP_SHM_HEADER.size + n])
                )
            except Exception as exc:
                raise TensorParallelTransportError(
                    "could not deserialize tensor-parallel payload"
                ) from exc
            if (
                not isinstance(payload, list)
                or not payload
                or not isinstance(payload[0], str)
            ):
                raise TensorParallelTransportError(
                    "invalid tensor-parallel call payload"
                )
            method_name, *args = payload
            return method_name, args
        finally:
            self.event.clear()

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        worker_args = compact_run_args(args) if method_name == "run" else args
        data = pickle.dumps([method_name, *worker_args], protocol=pickle.HIGHEST_PROTOCOL)
        n = len(data)
        capacity = len(self.shm.buf) - TP_SHM_HEADER.size
        if n == 0 or n > capacity:
            raise TensorParallelTransportError(
                "tensor-parallel payload exceeds shared-memory capacity: "
                f"{n} bytes for {capacity}-byte capacity"
            )
        self.shm.buf[:TP_SHM_HEADER.size] = TP_SHM_HEADER.pack(
            TP_SHM_MAGIC, n
        )
        self.shm.buf[TP_SHM_HEADER.size:TP_SHM_HEADER.size + n] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        if self.rank == 0:
            sampler_warmup_batches = (1,) if self.config.max_num_seqs == 1 else (1, 2)
            vocab_size = self.config.hf_config.vocab_size
            cuda_rng_state = torch.cuda.get_rng_state()
            for batch_size in sampler_warmup_batches:
                warmup_logits = torch.zeros(batch_size, vocab_size)
                warmup_temperatures = torch.ones(batch_size, dtype=torch.float32)
                self.sampler(warmup_logits, warmup_temperatures)
                self.sampler.greedy(warmup_logits)
            torch.cuda.set_rng_state(cuda_rng_state)
            for top_k in sorted({min(1, vocab_size), min(50, vocab_size)}):
                warmup_logits = torch.zeros(2, vocab_size)
                row_indices = torch.zeros(1, dtype=torch.int64)
                self.sampler.filter_top_k(warmup_logits, row_indices, top_k)
                warmup_logits = torch.zeros(2, vocab_size)
                self.sampler.filter_top_k(warmup_logits, None, top_k)
            warmup_temperatures = torch.ones(2, dtype=torch.float32)
            if self.config.top_p_backend == "exact":
                warmup_probability_cutoffs = torch.full(
                    (1,), 1.0 - 0.9, dtype=torch.float32
                )
                warmup_logits = torch.zeros(2, vocab_size)
                row_indices = torch.zeros(1, dtype=torch.int64)
                self.sampler.filter_top_p(
                    warmup_logits,
                    warmup_temperatures,
                    row_indices,
                    warmup_probability_cutoffs,
                )
                warmup_logits = torch.zeros(2, vocab_size)
                warmup_probability_cutoffs = torch.full(
                    (2,), 1.0 - 0.9, dtype=torch.float32
                )
                self.sampler.filter_top_p(
                    warmup_logits,
                    warmup_temperatures,
                    None,
                    warmup_probability_cutoffs,
                )
            else:
                warmup_logits = torch.zeros(2, vocab_size)
                warmup_top_ps = torch.full((2,), 0.9, dtype=torch.float32)
                cuda_rng_state = torch.cuda.get_rng_state()
                self.sampler.sample_top_p_flashinfer(
                    warmup_logits,
                    warmup_temperatures,
                    warmup_top_ps,
                )
                torch.cuda.set_rng_state(cuda_rng_state)
            del warmup_logits
        torch.cuda.empty_cache()

    def warmup_draft_model(self):
        """Compile/profile the inert draft prefill path without committing state."""

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens = self.config.max_num_batched_tokens
        max_model_len = self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(
            max_num_batched_tokens // seq_len,
            self.config.max_num_seqs,
        )
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        input_ids, positions = self.prepare_prefill(seqs)
        try:
            with torch.inference_mode():
                hidden_states = self.draft_model(input_ids, positions)
                self.draft_model.compute_logits(hidden_states)
        finally:
            reset_context()
        del hidden_states, input_ids, positions
        torch.cuda.empty_cache()

    def _kv_cache_shape(self, hf_config, num_blocks):
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", None)
        if head_dim is None:
            head_dim = (
                hf_config.hidden_size // hf_config.num_attention_heads
            )
        return (
            2,
            hf_config.num_hidden_layers,
            num_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
        )

    @staticmethod
    def _clear_model_cache_views(model):
        if model is None:
            return
        for module in model.modules():
            if hasattr(module, "k_cache"):
                module.k_cache = None
            if hasattr(module, "v_cache"):
                module.v_cache = None

    @staticmethod
    def _bind_kv_cache(model, kv_cache, hf_config, *, model_name):
        cache_modules = [
            module
            for module in model.modules()
            if hasattr(module, "k_cache") and hasattr(module, "v_cache")
        ]
        expected_layers = hf_config.num_hidden_layers
        if len(cache_modules) != expected_layers:
            raise KVCacheBindingError(
                f"{model_name} exposes {len(cache_modules)} KV-cache layers, "
                f"but its config declares {expected_layers}"
            )
        for layer_id, module in enumerate(cache_modules):
            module.k_cache = kv_cache[0, layer_id]
            module.v_cache = kv_cache[1, layer_id]

    def _allocate_and_bind_dual_kv_caches(self, num_blocks):
        """Allocate parallel caches transactionally over one logical block space."""

        target_config = self.config.hf_config
        draft_config = self.config.draft_hf_config
        self.kv_cache = None
        self.draft_kv_cache = None
        try:
            self.kv_cache = torch.empty(
                *self._kv_cache_shape(target_config, num_blocks),
                dtype=target_config.dtype,
                device="cuda",
            )
            self._bind_kv_cache(
                self.model,
                self.kv_cache,
                target_config,
                model_name="target",
            )
            self.draft_kv_cache = torch.empty(
                *self._kv_cache_shape(draft_config, num_blocks),
                dtype=draft_config.dtype,
                device="cuda",
            )
            self._bind_kv_cache(
                self.draft_model,
                self.draft_kv_cache,
                draft_config,
                model_name="draft",
            )
        except BaseException:
            self._clear_model_cache_views(self.draft_model)
            self._clear_model_cache_views(self.model)
            self.draft_kv_cache = None
            self.kv_cache = None
            raise

    def _release_profile_graph_owners(self):
        """Release graph-profile owners before replacing provisional caches."""

        for collection_name in (
            "draft_graphs",
            "varlen_graphs",
            "graphs",
        ):
            collection = getattr(self, collection_name, None)
            if isinstance(collection, dict):
                for graph in collection.values():
                    reset = getattr(graph, "reset", None)
                    if callable(reset):
                        reset()
        for name in (
            "draft_graphs",
            "varlen_graphs",
            "graphs",
            "draft_graph_pool",
            "graph_pool",
            "draft_graph_vars",
            "varlen_vars",
            "graph_vars",
        ):
            if hasattr(self, name):
                delattr(self, name)
        self.draft_graphs = None
        self.draft_graph_pool = None
        self.draft_graph_vars = None

    def _profile_speculative_graph_memory(self):
        """Measure target+draft graph ownership with one provisional KV block."""

        self._allocate_and_bind_dual_kv_caches(1)
        allocated_before = torch.cuda.memory_allocated()
        reserved_before = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        self.capture_cudagraph()
        self.capture_varlen_graphs()
        self.capture_draft_cudagraph()
        torch.cuda.synchronize()
        stats = torch.cuda.memory_stats()
        self._profiled_graph_allocated_bytes = max(
            torch.cuda.memory_allocated() - allocated_before,
            0,
        )
        self._profiled_graph_reserved_bytes = max(
            torch.cuda.memory_reserved() - reserved_before,
            0,
        )
        self._profiled_graph_peak_allocated_bytes = max(
            stats["allocated_bytes.all.peak"] - allocated_before,
            0,
        )
        self._profiled_graph_peak_reserved_bytes = max(
            stats["reserved_bytes.all.peak"] - reserved_before,
            0,
        )

        self._release_profile_graph_owners()
        self._clear_model_cache_views(self.draft_model)
        self._clear_model_cache_views(self.model)
        self.draft_kv_cache = None
        self.kv_cache = None
        gc.collect()
        torch.cuda.empty_cache()
        # The profiling transaction is not a runtime transient. Reset the peak
        # baseline so joint sizing uses the recorded target/draft warmup costs
        # rather than double-counting the disposable first graph capture.
        torch.cuda.reset_peak_memory_stats()

    def _record_final_graph_memory_peaks(self):
        """Record incremental graph-capture high-water deltas before pretouch."""

        torch.cuda.synchronize()
        self._allocated_after_graph_before_pretouch = (
            torch.cuda.memory_allocated()
        )
        self._reserved_after_graph_before_pretouch = (
            torch.cuda.memory_reserved()
        )
        stats = torch.cuda.memory_stats()
        self._final_graph_peak_allocated_bytes = max(
            stats["allocated_bytes.all.peak"]
            - self._final_graph_allocated_baseline,
            0,
        )
        self._final_graph_peak_reserved_bytes = max(
            stats["reserved_bytes.all.peak"]
            - self._final_graph_reserved_baseline,
            0,
        )

    def _plan_speculative_memory(self):
        config = self.config
        return plan_speculative_workspace(
            vocab_size=config.hf_config.vocab_size,
            configured_k=config.configured_k,
            max_num_seqs=config.max_num_seqs,
            max_num_batched_tokens=config.max_num_batched_tokens,
            max_model_len=config.max_model_len,
            target_logits_dtype=config.hf_config.dtype,
            draft_logits_dtype=config.draft_hf_config.dtype,
        )

    @staticmethod
    def _model_parameter_bytes(model):
        # ``numel * itemsize`` double-counts tied parameters and distinct views
        # into the same allocation.  The audit is a physical-ownership ledger,
        # so count each backing storage exactly once per device.
        unique_storages = {}
        for parameter in model.parameters():
            storage = parameter.untyped_storage()
            key = (
                storage.device,
                storage.data_ptr(),
                storage.nbytes(),
            )
            unique_storages[key] = storage.nbytes()
        return sum(unique_storages.values())

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        if getattr(self, "speculation_enabled", False):
            return self._allocate_speculative_kv_cache()

        # Keep the disabled path byte-for-byte equivalent to the V1 target-only
        # allocation and its established diagnostics.
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        memory_budget = int(total * config.gpu_memory_utilization)
        transient_peak = max(peak - current, 0)
        usable = memory_budget - used - transient_peak
        auto_blocks = usable // block_bytes
        requested_blocks = config.num_kvcache_blocks
        if auto_blocks <= 0:
            raise RuntimeError(
                "cannot allocate any KV-cache blocks: "
                f"total={total}, free={free}, budget={memory_budget}, "
                f"used={used}, current_allocated={current}, peak_allocated={peak}, "
                f"transient_peak={transient_peak}, usable={usable}, "
                f"block_bytes={block_bytes}; raise gpu_memory_utilization or "
                "free GPU memory held by this process or another process"
            )
        if requested_blocks == -1:
            config.num_kvcache_blocks = auto_blocks
        elif requested_blocks > auto_blocks:
            raise RuntimeError(
                f"requested num_kvcache_blocks={requested_blocks} requires "
                f"{requested_blocks * block_bytes} bytes, but only {usable} "
                f"profiled bytes ({auto_blocks} blocks) are usable"
            )
        # A positive explicit value is an exact, validated override. This is
        # useful for reproducible tests and intentionally does not auto-expand.
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def _allocate_speculative_kv_cache(self):
        config = self.config
        target_config = config.hf_config
        draft_config = config.draft_hf_config
        plan = self._plan_speculative_memory()
        self.speculative_memory_plan = plan

        free, total = torch.cuda.mem_get_info()
        used = total - free
        stats = torch.cuda.memory_stats()
        peak = stats["allocated_bytes.all.peak"]
        current = stats["allocated_bytes.all.current"]
        reserved_before_kv = torch.cuda.memory_reserved()
        # Target and draft warmups execute sequentially in both eager and graph
        # modes. Persistent graph ownership is measured and reserved separately.
        transient_peak = max(
            self._warmup_transient_bytes,
            peak - current,
            0,
        )
        target_block_bytes = kv_cache_block_bytes(
            target_config,
            block_size=self.block_size,
            tensor_parallel_size=self.world_size,
        )
        draft_block_bytes = kv_cache_block_bytes(
            draft_config,
            block_size=self.block_size,
            tensor_parallel_size=self.world_size,
        )
        joint_block_bytes = target_block_bytes + draft_block_bytes
        profiled_graph_ownership = max(
            self._profiled_graph_allocated_bytes,
            self._profiled_graph_reserved_bytes,
        )
        profiled_graph_peak = max(
            profiled_graph_ownership,
            self._profiled_graph_peak_allocated_bytes,
            self._profiled_graph_peak_reserved_bytes,
        )
        # Provisional policy until the route-specific A100 certificate replaces
        # it: retain at least 64 MiB and at least one full joint logical block so
        # larger model geometries cannot receive a sub-block graph cushion.
        graph_allocator_margin = (
            max(
                SPECULATIVE_GRAPH_ALLOCATOR_MARGIN_BYTES,
                joint_block_bytes,
            )
            if not self.enforce_eager
            else 0
        )
        graph_construction_reservation = (
            profiled_graph_peak + graph_allocator_margin
        )
        # Final graph capture and speculative runtime are mutually exclusive.
        # Runtime keeps graph ownership resident while model activation and Wspec
        # overlap; graph construction uses the measured capture high-water plus
        # its provisional allocator margin. Price only the larger envelope.
        runtime_reservation = (
            profiled_graph_ownership
            + transient_peak
            + plan.reservation_bytes
        )
        sizing_overhead = max(
            graph_construction_reservation,
            runtime_reservation,
        )
        memory_budget = int(total * config.gpu_memory_utilization)
        usable = memory_budget - used - sizing_overhead
        auto_blocks = usable // joint_block_bytes
        requested_blocks = config.num_kvcache_blocks
        if auto_blocks <= 0:
            raise SpeculativeKVCacheCapacityError(
                "cannot allocate any joint target/draft KV-cache blocks: "
                f"total={total}, free={free}, budget={memory_budget}, "
                f"used={used}, current_allocated={current}, "
                f"peak_allocated={peak}, transient_peak={transient_peak}, "
                f"profiled_graph_ownership={profiled_graph_ownership}, "
                f"profiled_graph_peak={profiled_graph_peak}, "
                f"graph_allocator_margin={graph_allocator_margin}, "
                f"graph_construction_reservation="
                f"{graph_construction_reservation}, "
                f"runtime_reservation={runtime_reservation}, "
                f"sizing_overhead={sizing_overhead}, "
                f"modeled_workspace_reservation={plan.reservation_bytes}, "
                f"usable={usable}, target_block_bytes={target_block_bytes}, "
                f"draft_block_bytes={draft_block_bytes}, "
                f"joint_block_bytes={joint_block_bytes}; lower configured K or "
                "work limits, raise gpu_memory_utilization, or free GPU memory"
            )
        if requested_blocks == -1:
            config.num_kvcache_blocks = auto_blocks
        elif requested_blocks > auto_blocks:
            raise SpeculativeKVCacheCapacityError(
                f"requested num_kvcache_blocks={requested_blocks} requires "
                f"{requested_blocks * joint_block_bytes} joint KV bytes, but "
                f"only {usable} modeled bytes ({auto_blocks} blocks) are usable "
                f"after sizing_overhead={sizing_overhead} "
                f"(graph_construction={graph_construction_reservation}, "
                f"runtime={runtime_reservation}) reservation"
            )

        self._allocate_and_bind_dual_kv_caches(
            config.num_kvcache_blocks
        )
        self._final_graph_allocated_baseline = (
            torch.cuda.memory_allocated()
        )
        self._final_graph_reserved_baseline = torch.cuda.memory_reserved()
        # Eager mode performs no capture, so its graph-phase endpoint is the
        # post-KV baseline. Graph mode overwrites both values immediately after
        # capture and before eager-prefill pretouch can retain allocator state.
        self._allocated_after_graph_before_pretouch = (
            self._final_graph_allocated_baseline
        )
        self._reserved_after_graph_before_pretouch = (
            self._final_graph_reserved_baseline
        )
        if not self.enforce_eager:
            torch.cuda.reset_peak_memory_stats()
        self._speculative_memory_audit_inputs = dict(
            total_memory_bytes=total,
            free_before_kv_bytes=free,
            used_before_kv_bytes=used,
            memory_budget_bytes=memory_budget,
            allocated_before_kv_bytes=current,
            reserved_before_kv_bytes=reserved_before_kv,
            peak_before_kv_bytes=peak,
            warmup_transient_bytes=transient_peak,
            target_warmup_transient_bytes=(
                self._target_warmup_transient_bytes
            ),
            draft_warmup_transient_bytes=(
                self._draft_warmup_transient_bytes
            ),
            target_weight_bytes=self._model_parameter_bytes(self.model),
            draft_weight_bytes=self._model_parameter_bytes(
                self.draft_model
            ),
            profiled_graph_ownership_bytes=profiled_graph_ownership,
            profiled_graph_peak_bytes=profiled_graph_peak,
            graph_allocator_margin_bytes=graph_allocator_margin,
            graph_construction_reservation_bytes=(
                graph_construction_reservation
            ),
            graph_reservation_bytes=graph_construction_reservation,
            runtime_reservation_bytes=runtime_reservation,
            sizing_overhead_bytes=sizing_overhead,
            target_block_bytes=target_block_bytes,
            draft_block_bytes=draft_block_bytes,
            joint_block_bytes=joint_block_bytes,
            selected_num_blocks=config.num_kvcache_blocks,
            allocated_after_kv_before_graph_bytes=(
                self._final_graph_allocated_baseline
            ),
            reserved_after_kv_before_graph_bytes=(
                self._final_graph_reserved_baseline
            ),
        )

    def _finalize_speculative_memory_audit(self):
        """Freeze measured ownership without claiming unmeasured certification."""

        inputs = self._speculative_memory_audit_inputs
        if inputs is None or self.speculative_memory_plan is None:
            raise RuntimeError(
                "speculative memory audit cannot finalize before joint KV allocation"
            )
        # Both eager-prefill pretouches synchronize before returning, so their
        # input/activation tensors are dead here.  Release only the allocator's
        # reusable cache before asking the driver for physical free memory.
        # Otherwise those dead cached segments are counted as permanent
        # post-init ownership and the future activation/workspace reservation is
        # subtracted again, which can falsely reject the exact auto/explicit KV
        # boundary.  Live weights, KV tensors, and CUDA-graph private pools remain
        # owned and therefore remain visible to the audit.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        post_free, post_total = torch.cuda.mem_get_info()
        if post_total != inputs["total_memory_bytes"]:
            raise RuntimeError(
                "CUDA total memory changed during speculative runner construction"
            )
        post_allocated = torch.cuda.memory_allocated()
        post_reserved = torch.cuda.memory_reserved()
        final_graph_allocated = max(
            self._allocated_after_graph_before_pretouch
            - self._final_graph_allocated_baseline,
            0,
        )
        final_graph_reserved = max(
            self._reserved_after_graph_before_pretouch
            - self._final_graph_reserved_baseline,
            0,
        )
        final_graph_construction_peak = max(
            self._final_graph_peak_allocated_bytes,
            self._final_graph_peak_reserved_bytes,
        )
        if (
            final_graph_construction_peak
            > inputs["graph_construction_reservation_bytes"]
        ):
            raise SpeculativeKVCacheCapacityError(
                "final target/draft graph construction exceeded its profiled "
                "reservation: "
                f"observed_peak={final_graph_construction_peak}, "
                "profiled_peak="
                f"{inputs['profiled_graph_peak_bytes']}, "
                f"allocator_margin={inputs['graph_allocator_margin_bytes']}, "
                "reservation="
                f"{inputs['graph_construction_reservation_bytes']}; reduce "
                "num_kvcache_blocks/work limits or raise the registered "
                "allocator margin before certifying this route"
            )
        budget_headroom = (
            inputs["memory_budget_bytes"] - (post_total - post_free)
        )
        plan = self.speculative_memory_plan
        # Target and draft model activations are sequential at runtime, exactly
        # as in sizing. Captured graph ownership is already resident in post_free.
        runtime_transient_reservation = inputs["warmup_transient_bytes"]
        modeled_runtime_headroom = (
            budget_headroom
            - runtime_transient_reservation
            - plan.reservation_bytes
        )
        if modeled_runtime_headroom < 0:
            raise SpeculativeKVCacheCapacityError(
                "post-init target/draft runner does not retain its modeled "
                "runtime headroom: "
                f"budget_headroom={budget_headroom}, "
                f"runtime_transient={runtime_transient_reservation}, "
                f"workspace_reservation={plan.reservation_bytes}, "
                f"shortfall={-modeled_runtime_headroom}; reduce "
                "num_kvcache_blocks, configured K, or work limits, or raise "
                "gpu_memory_utilization"
            )

        self.speculative_memory_audit = SpeculativeMemoryAudit(
            workspace_plan=plan,
            profiled_graph_allocated_bytes=(
                self._profiled_graph_allocated_bytes
            ),
            profiled_graph_reserved_bytes=(
                self._profiled_graph_reserved_bytes
            ),
            profiled_graph_peak_allocated_bytes=(
                self._profiled_graph_peak_allocated_bytes
            ),
            profiled_graph_peak_reserved_bytes=(
                self._profiled_graph_peak_reserved_bytes
            ),
            target_kv_bytes=(
                inputs["selected_num_blocks"] * inputs["target_block_bytes"]
            ),
            draft_kv_bytes=(
                inputs["selected_num_blocks"] * inputs["draft_block_bytes"]
            ),
            allocated_after_graph_before_pretouch_bytes=(
                self._allocated_after_graph_before_pretouch
            ),
            reserved_after_graph_before_pretouch_bytes=(
                self._reserved_after_graph_before_pretouch
            ),
            final_graph_allocated_bytes=final_graph_allocated,
            final_graph_reserved_bytes=final_graph_reserved,
            final_graph_peak_allocated_bytes=(
                self._final_graph_peak_allocated_bytes
            ),
            final_graph_peak_reserved_bytes=(
                self._final_graph_peak_reserved_bytes
            ),
            post_init_allocated_bytes=post_allocated,
            post_init_reserved_bytes=post_reserved,
            post_init_free_bytes=post_free,
            post_init_budget_headroom_bytes=budget_headroom,
            modeled_runtime_headroom_bytes=modeled_runtime_headroom,
            audit_required_components=plan.audit_required_components,
            gpu_certified=False,
            **inputs,
        )
        self._speculative_memory_audit_inputs = None

    def prepare_block_tables(self, seqs: list[Sequence | ScheduledSequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            list(seq.block_table) + [-1] * (max_len - len(seq.block_table))
            for seq in seqs
        ]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_ragged(self, seqs: list[Sequence | ScheduledSequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            if seq.is_prefill:
                if isinstance(seq, ScheduledSequence):
                    input_ids.extend(seq.scheduled_token_ids)
                else:
                    input_ids.extend(seq[start:end])
            else:
                # decode-mode row: identical indices via the num_cached == len-1
                # invariant; extract the explicit last-token field on worker DTOs
                input_ids.append(seq.last_token)
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if seqs[0].block_table:    # any real step; warmup has no blocks
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    prepare_prefill = prepare_ragged

    def prepare_decode(self, seqs: list[Sequence | ScheduledSequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            last_block_num_tokens = len(seq) - (
                len(seq.block_table) - 1
            ) * self.block_size
            slot_mapping.append(
                seq.block_table[-1] * self.block_size + last_block_num_tokens - 1
            )
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    @staticmethod
    def _prepare_sample_metadata(seqs: list[Sequence], vocab_size: int):
        temperatures = tuple(seq.temperature for seq in seqs)
        all_greedy = all(temperature == 0.0 for temperature in temperatures)
        if all_greedy:
            return temperatures, (), None, True

        rows_by_top_k = {}
        for row, (seq, temperature) in enumerate(zip(seqs, temperatures)):
            if temperature == 0.0 or seq.top_k == -1:
                continue
            effective_top_k = min(seq.top_k, vocab_size)
            if effective_top_k == vocab_size:
                continue
            rows_by_top_k.setdefault(effective_top_k, []).append(row)

        top_k_buckets = tuple(
            (
                top_k,
                None if len(rows_by_top_k[top_k]) == len(seqs)
                else tuple(rows_by_top_k[top_k]),
            )
            for top_k in sorted(rows_by_top_k)
        )
        top_p_rows = []
        top_ps = []
        for row, (seq, temperature) in enumerate(zip(seqs, temperatures)):
            if temperature != 0.0 and seq.top_p != 1.0:
                top_p_rows.append(row)
                top_ps.append(seq.top_p)
        top_p_plan = None
        if top_p_rows:
            top_p_plan = (
                None if len(top_p_rows) == len(seqs) else tuple(top_p_rows),
                tuple(top_ps),
            )
        return temperatures, top_k_buckets, top_p_plan, False

    @staticmethod
    def _expand_top_ps(batch_size: int, top_p_plan):
        """Expand active-row metadata to FlashInfer's full-batch p vector."""

        rows, active_top_ps = top_p_plan
        if rows is None:
            if len(active_top_ps) != batch_size:
                raise ValueError("homogeneous top-p metadata must cover the batch")
            return active_top_ps
        if len(rows) != len(active_top_ps):
            raise ValueError("top-p rows and values must have the same length")
        top_ps = [1.0] * batch_size
        for row, top_p in zip(rows, active_top_ps):
            top_ps[row] = top_p
        return tuple(top_ps)

    def prepare_sample(self, seqs: list[Sequence]):
        (
            temperatures,
            host_top_k_buckets,
            host_top_p_plan,
            all_greedy,
        ) = self._prepare_sample_metadata(
            seqs, self.config.hf_config.vocab_size
        )
        if all_greedy:
            return None, (), None, True

        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        top_k_buckets = tuple(
            (
                top_k,
                (
                    None
                    if rows is None
                    else torch.tensor(rows, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
                ),
            )
            for top_k, rows in host_top_k_buckets
        )
        top_p_plan = None
        if host_top_p_plan is not None:
            if self.config.top_p_backend == "flashinfer":
                top_ps = self._expand_top_ps(
                    len(temperatures), host_top_p_plan
                )
                top_ps = torch.tensor(
                    top_ps,
                    dtype=torch.float32,
                    pin_memory=True,
                ).cuda(non_blocking=True)
                top_p_plan = (None, top_ps)
            else:
                rows, top_ps = host_top_p_plan
                row_indices = (
                    None
                    if rows is None
                    else torch.tensor(rows, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
                )
                probability_cutoffs = torch.tensor(
                    tuple(1.0 - top_p for top_p in top_ps),
                    dtype=torch.float32,
                    pin_memory=True,
                ).cuda(non_blocking=True)
                top_p_plan = (row_indices, probability_cutoffs)
        return temperatures, top_k_buckets, top_p_plan, False

    def _select_varlen_graph_key(self, num_tokens: int, num_seqs: int):
        """Return the smallest captured graph that can hold a ragged step.

        Capture keys can be sparse when a slot tier cannot represent a legal
        dummy layout for the configured ``max_model_len``.  Selecting from the
        graphs that actually exist avoids indexing an empty bucket list and
        avoids assuming that every token/slot cross-product was captured.
        """
        if num_tokens <= 0 or num_seqs <= 0:
            return None
        return min(
            (
                key
                for key in getattr(self, "varlen_graphs", {})
                if key[0] >= num_tokens and key[1] >= num_seqs
            ),
            default=None,
        )

    def _varlen_context_fits_graph(
        self,
        num_tokens: int,
        num_seqs: int,
        ctx,
        graph_key: tuple[int, int],
    ):
        """Whether live ragged metadata fits the persistent capture buffers."""
        buffers = getattr(self, "varlen_vars", None)
        block_tables = ctx.block_tables
        capture_max_q = min(graph_key[0], self.config.max_model_len)
        return bool(
            buffers is not None
            and ctx.slot_mapping is not None
            and ctx.slot_mapping.numel() == num_tokens
            and ctx.cu_seqlens_k is not None
            and ctx.cu_seqlens_k.numel() == num_seqs + 1
            and ctx.max_seqlen_q <= capture_max_q
            and ctx.max_seqlen_k <= self.config.max_model_len
            and block_tables is not None
            and block_tables.ndim == 2
            and block_tables.size(0) >= num_seqs
            and block_tables.size(1) <= buffers["block_tables"].size(1)
        )

    @staticmethod
    def _varlen_capture_boundaries(
        num_tokens: int,
        num_slots: int,
        max_model_len: int,
    ) -> tuple[int, ...] | None:
        """Build a legal padded ``cu_seqlens`` layout for graph capture.

        Every non-empty dummy sequence is capped at ``max_model_len`` and the
        remaining slots are zero-length.  ``None`` means that this graph key is
        structurally impossible for the requested slot tier.
        """
        if num_tokens <= 0 or num_slots <= 0 or max_model_len <= 0:
            return None
        num_nonempty = (num_tokens + max_model_len - 1) // max_model_len
        if num_nonempty > num_slots:
            return None
        boundaries = [0]
        for index in range(num_nonempty):
            boundaries.append(min((index + 1) * max_model_len, num_tokens))
        boundaries.extend([num_tokens] * (num_slots - num_nonempty))
        return tuple(boundaries)

    def _fill_varlen(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx,
        graph_key: tuple[int, int],
    ):
        """Copy a real ragged step into the persistent varlen graph buffers.
        Single fill path shared by replay and the per-bucket bitwise test —
        the certified code IS the production code."""
        v = self.varlen_vars
        t = input_ids.size(0)
        ns = ctx.cu_seqlens_q.numel() - 1
        tp, sl = graph_key
        if t > tp or ns > sl:
            raise ValueError(
                f"ragged step ({t} tokens, {ns} sequences) exceeds graph "
                f"capacity ({tp} tokens, {sl} sequences)"
            )
        v["input_ids"][:t] = input_ids
        v["positions"][:t] = positions
        v["slot_mapping"].fill_(-1)
        v["slot_mapping"][:t] = ctx.slot_mapping
        v["cu_q"][:ns + 1] = ctx.cu_seqlens_q
        v["cu_q"][ns + 1:] = ctx.cu_seqlens_q[-1]      # zero-length tail segments (P1)
        v["cu_k"][:ns + 1] = ctx.cu_seqlens_k
        v["cu_k"][ns + 1:] = ctx.cu_seqlens_k[-1]
        v["block_tables"].fill_(-1)
        v["block_tables"][:ns, :ctx.block_tables.size(1)] = ctx.block_tables

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if not is_prefill:
            if self.enforce_eager or input_ids.size(0) > 512:
                return self.model.compute_logits(self.model(input_ids, positions))
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])
        t = input_ids.size(0)
        ctx = get_context()
        ns = ctx.cu_seqlens_q.numel() - 1
        # block_tables=None is the allocation warmup, which deliberately uses
        # the eager non-paged path and is not a graph miss.
        graph_key = (
            self._select_varlen_graph_key(t, ns)
            if ctx.block_tables is not None
            else None
        )
        if graph_key is not None and not self._varlen_context_fits_graph(
            t, ns, ctx, graph_key
        ):
            graph_key = None
        if graph_key is None:
            if hasattr(self, "varlen_graphs") and ctx.block_tables is not None:
                self.varlen_miss += 1
            return self.model.compute_logits(self.model(input_ids, positions))
        self._fill_varlen(input_ids, positions, ctx, graph_key)
        self.varlen_graphs[graph_key].replay()
        return self.model.compute_logits(self.varlen_vars["outputs"][:t])   # gather runs on the LIVE real context

    def run(
        self,
        seqs: list[Sequence | ScheduledSequence],
        is_prefill: bool,
    ) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures, top_k_buckets, top_p_plan, all_greedy = (
            self.prepare_sample(seqs)
            if self.rank == 0
            else (None, (), None, False)
        )
        logits = self.run_model(input_ids, positions, is_prefill)
        if self.rank == 0:
            if all_greedy:
                tokens = self.sampler.greedy(logits)
            else:
                for top_k, row_indices in top_k_buckets:
                    logits = self.sampler.filter_top_k(logits, row_indices, top_k)
                if (
                    top_p_plan is not None
                    and self.config.top_p_backend == "flashinfer"
                ):
                    _, top_ps = top_p_plan
                    tokens = self.sampler.sample_top_p_flashinfer(
                        logits, temperatures, top_ps
                    )
                else:
                    if top_p_plan is not None:
                        row_indices, probability_cutoffs = top_p_plan
                        logits = self.sampler.filter_top_p(
                            logits, temperatures, row_indices, probability_cutoffs
                        )
                    tokens = self.sampler(logits, temperatures)
            token_ids = tokens.tolist()
        else:
            token_ids = None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

    @torch.inference_mode()
    def capture_draft_cudagraph(self):
        """Capture independently owned draft decode graphs for later V3 use."""

        config = self.config
        hf_config = config.draft_hf_config
        max_bs = min(config.max_num_seqs, 512)
        max_num_blocks = (
            config.max_model_len + self.block_size - 1
        ) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(
            max_bs, max_num_blocks, dtype=torch.int32
        )
        outputs = torch.zeros(
            max_bs,
            hf_config.hidden_size,
            dtype=hf_config.dtype,
        )
        self.draft_graph_bs = list(self.graph_bs)
        self.draft_graphs = {}
        self.draft_graph_pool = None

        for bs in reversed(self.draft_graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )
            try:
                outputs[:bs] = self.draft_model(
                    input_ids[:bs], positions[:bs]
                )
                with torch.cuda.graph(graph, self.draft_graph_pool):
                    outputs[:bs] = self.draft_model(
                        input_ids[:bs], positions[:bs]
                    )
                if self.draft_graph_pool is None:
                    self.draft_graph_pool = graph.pool()
                self.draft_graphs[bs] = graph
                torch.cuda.synchronize()
            finally:
                reset_context()

        self.draft_graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

    @torch.inference_mode()
    def capture_varlen_graphs(self):
        config = self.config
        S1 = config.max_num_seqs + 1    # F3: +1 segment slot for C3's single partial chunk
        max_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        # Initialize the full state before the first possible early return.  A
        # sub-128 token budget intentionally has no graph buckets and all real
        # ragged steps are counted eager misses.
        self.varlen_ts = []
        self.varlen_slots = []
        self.varlen_graphs = {}
        self.varlen_vars = None
        self.varlen_miss = 0
        # bucket top-end 2048, NOT the F3-drafted 4096: P12 measured the 4096 replay at
        # 32 ms best-case ~= paged-eager (host1 is past the E2 dispatch/GPU crossover at
        # that T), so the graph buys nothing there; C3's chunking bounds mixed steps to
        # the token budget anyway. Zero-length slot tax ~0.04 ms/slot at T=4096 (P12).
        self.varlen_ts = [
            t
            for t in (128, 256, 512, 1024, 2048)
            if t <= config.max_num_batched_tokens
        ]
        if not self.varlen_ts:
            return
        T = self.varlen_ts[-1]
        v = dict(
            input_ids=torch.zeros(T, dtype=torch.int64),
            positions=torch.zeros(T, dtype=torch.int64),
            cu_q=torch.zeros(S1 + 1, dtype=torch.int32),
            cu_k=torch.zeros(S1 + 1, dtype=torch.int32),
            slot_mapping=torch.full((T,), -1, dtype=torch.int32),
            block_tables=torch.zeros(S1, max_blocks, dtype=torch.int32),
            outputs=torch.zeros(T, config.hf_config.hidden_size),
        )
        # Two slot tiers per bucket (P13): zero-length padding slots cost real replay
        # time (~0.006-0.011 ms/slot at T=512/1024), so the common few-segment step
        # replays a lean capture while high-ns mixed steps keep a full-slot graph
        # instead of falling back to eager (which would break the ITL bound exactly
        # in the many-decoder regime). Tiers are prefix-slices of the SAME buffers —
        # the baked grid comes from the slice length; _fill_varlen's full-size
        # padding serves every tier.
        self.varlen_slots = sorted({min(64, S1), S1})
        for t in reversed(self.varlen_ts):
            for sl in reversed(self.varlen_slots):
                boundaries = self._varlen_capture_boundaries(
                    t, sl, config.max_model_len
                )
                if boundaries is None:
                    continue
                v["cu_q"].fill_(t)
                v["cu_q"][:sl + 1].copy_(torch.tensor(
                    boundaries,
                    dtype=torch.int32,
                    device=v["cu_q"].device,
                ))
                v["cu_k"].copy_(v["cu_q"])
                set_context(True, v["cu_q"][:sl + 1], v["cu_k"][:sl + 1],
                            min(t, config.max_model_len),
                            config.max_model_len, v["slot_mapping"][:t], None, v["block_tables"][:sl])
                graph = torch.cuda.CUDAGraph()
                v["outputs"][:t] = self.model(v["input_ids"][:t], v["positions"][:t])      # warmup
                with torch.cuda.graph(graph, self.graph_pool):                              # shared pool (P3b)
                    v["outputs"][:t] = self.model(v["input_ids"][:t], v["positions"][:t])
                self.varlen_graphs[(t, sl)] = graph
                torch.cuda.synchronize()
                reset_context()
        if self.varlen_graphs:
            self.varlen_vars = v

    def _pretouch_eager_prefill(self):
        # Must run AFTER __init__ restores the default dtype/device: init-time Dynamo
        # compiles guard on default_dtype (GLOBAL_STATE), and with varlen graphs the
        # small warmup prefills replay buckets, so the compiled modules' first eager
        # call would otherwise be the first bucket-MISS step — paying the full
        # recompile storm there (P10: ~750 ms on the 16k step-1; P11 named the guard).
        # Inputs are created OUTSIDE inference_mode, mirroring run()'s prepare/model
        # split — an inference-tensor input compiles a dispatch-keyset flavor that
        # production never replays (P11 residual: rotary's ADInplaceOrView guard).
        T = self.config.max_num_batched_tokens
        L = min(self.config.max_model_len, T)
        ns = (T + L - 1) // L
        cu = torch.arange(0, ns + 1, dtype=torch.int32, device="cuda") * L
        cu[-1] = T
        input_ids = torch.zeros(T, dtype=torch.int64, device="cuda")
        positions = torch.arange(T, dtype=torch.int64, device="cuda") % L
        set_context(True, cu, cu.clone(), L, L,
                    torch.full((T,), -1, dtype=torch.int32, device="cuda"), None, None)
        with torch.inference_mode():
            self.model(input_ids, positions)
        torch.cuda.synchronize()
        reset_context()

    def _pretouch_draft_eager_prefill(self):
        """Pretouch the draft's eager prefill specialization after defaults restore."""

        T = self.config.max_num_batched_tokens
        L = min(self.config.max_model_len, T)
        ns = (T + L - 1) // L
        cu = torch.arange(
            0, ns + 1, dtype=torch.int32, device="cuda"
        ) * L
        cu[-1] = T
        input_ids = torch.zeros(T, dtype=torch.int64, device="cuda")
        positions = torch.arange(
            T, dtype=torch.int64, device="cuda"
        ) % L
        set_context(
            True,
            cu,
            cu.clone(),
            L,
            L,
            torch.full(
                (T,), -1, dtype=torch.int32, device="cuda"
            ),
            None,
            None,
        )
        try:
            with torch.inference_mode():
                self.draft_model(input_ids, positions)
            torch.cuda.synchronize()
        finally:
            reset_context()
