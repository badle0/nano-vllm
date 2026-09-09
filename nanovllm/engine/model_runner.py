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
from nanovllm.engine.speculative_plan import (
    SpecPlanRow, SpecStepPlan, build_speculative_step_plan,
)
from nanovllm.engine.scheduler import as_draft_discard_plan
from nanovllm.engine.speculative_memory import (
    SpeculativeMemoryPlan,
    kv_cache_block_bytes,
    plan_speculative_workspace,
    speculative_route_fits_plan,
)
from nanovllm.engine.speculative_routes import (
    DraftCatchupFamily,
    DraftExecutionMode,
    DraftRouteAdmission,
    DraftRouteKey,
    DraftRouteRegistry,
    DraftWarmComponentKey,
    MAX_CUDA_GRAPH_BATCH_SIZE,
    build_draft_route_registry,
    draft_graph_batch_buckets,
    max_eligible_draft_catchup,
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


class SpeculativeDraftPlanError(RuntimeError):
    """A V3 discard plan no longer describes safe live runner state."""


class SpeculativeDraftExecutionError(RuntimeError):
    """Host-only boundary for a failed tensor-bearing V3 draft phase."""


class DraftCycleRow(NamedTuple):
    """Immutable runner-owned snapshot of one discard-plan row.

    The scheduler owns the physical reservation and the live ``Sequence``.  The
    runner deliberately copies only the values needed by draft execution so no
    model or sampler call can mutate scheduler/public state through this view.
    """

    seq_id: int
    token_ids: tuple[int, ...]
    committed_len: int
    target_cached_tokens: int
    draft_cached_tokens: int
    block_table: tuple[int, ...]
    temperature: float
    top_k: int
    top_p: float

    def __len__(self) -> int:
        return self.committed_len

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def last_token(self) -> int:
        return self.token_ids[-1]


class DraftCatchupView(NamedTuple):
    """Cycle-local ragged-prefill DTO over an unprocessed committed suffix."""

    scheduled_token_ids: tuple[int, ...]
    is_prefill: bool
    num_cached_tokens: int
    num_scheduled_tokens: int
    num_tokens: int
    last_token: int
    block_table: tuple[int, ...]

    def __len__(self) -> int:
        return self.num_tokens


class DraftProposalExecution(NamedTuple):
    """Tensor-bearing V3 test seam; production consumes it within one cycle."""

    rows: tuple[DraftCycleRow, ...]
    proposal_token_ids: torch.Tensor
    q_storage_kbv: torch.Tensor
    q_probabilities: torch.Tensor
    catchup_positions: int
    graph_decode_steps: int
    eager_decode_steps: int


class DraftDiscardRowResult(NamedTuple):
    """Small host-only result for one compute-then-discard row."""

    seq_id: int
    coverage_after_commit: int
    proposed_token_ids: tuple[int, ...]
    proposal_count: int


class DraftDiscardResult(NamedTuple):
    """Host-only diagnostics returned by the production V3 discard path."""

    rows: tuple[DraftDiscardRowResult, ...]
    route_key: DraftRouteKey
    effective_k: int
    catchup_positions: int
    draft_positions: int
    graph_decode_steps: int
    eager_decode_steps: int
    q_shape: tuple[int, int, int]
    q_stride: tuple[int, int, int]
    q_dtype: str
    q_storage_contiguous: bool
    q_view_zero_copy: bool


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
    persistent_workspace_reservation_bytes: int = 0


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
        if bool(getattr(config, "speculation_enabled", False)):
            # Two model instances, capture/default-dtype variants, and all-query
            # verifier shapes need more than the default eight specializations.
            # Scope this finite construction budget; do not globally disable
            # compilation or suppress failures, and leave speculation-off alone.
            limit_name = ("recompile_limit" if hasattr(torch._dynamo.config, "recompile_limit")
                          else "cache_size_limit")
            with torch._dynamo.config.patch(**{limit_name: 32}):
                self._initialize(config, rank, event)
        else:
            self._initialize(config, rank, event)

    def _initialize(self, config: Config, rank: int, event: Event | list[Event]):
        self._closed = False
        self._owns_process_group = False
        self.shm = None
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.numerical_mode = getattr(config, "numerical_mode", "fast")
        self.enforce_eager = config.enforce_eager or self.numerical_mode == "invariant"
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
            self.draft_route_registry = None
            self.speculative_verifier_ready = False
            self.speculative_verifier_shapes = frozenset()
            self.speculative_verifier_fingerprint = None
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
            self._draft_route_pretouch_peak_bytes = 0
            self._spec_q_rows = None
            self._spec_proposal_ids = None
            self._spec_target_probability_rows = None
            self._spec_bonus_noise = None
            self._spec_result_rows = None

        default_device = torch.get_default_device()
        default_dtype = torch.get_default_dtype()
        try:
            torch.cuda.set_device(rank)
            if self.numerical_mode == "invariant":
                capability = torch.cuda.get_device_capability(rank)
                if capability < (8, 0):
                    raise RuntimeError("numerical_mode='invariant' requires CUDA compute capability >= 8.0")
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
                for module in self.model.modules():
                    if hasattr(module, "numerical_mode"):
                        module.numerical_mode = self.numerical_mode
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
                    self._initialize_speculative_route_registry()
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
                self._run_draft_phase(
                    "V3 draft route pretouch",
                    self._pretouch_draft_routes,
                )
                self._run_draft_phase("target verifier pretouch", self._pretouch_speculative_verifier)
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
            modules = getattr(self.draft_model, "modules", None)
            for module in modules() if callable(modules) else ():
                if hasattr(module, "numerical_mode"):
                    module.numerical_mode = getattr(
                        self, "numerical_mode", "fast"
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
            "draft_route_registry",
            "speculative_verifier_fingerprint",
            "speculative_verifier_ready",
            "speculative_rejection_sampler",
            "speculative_verifier_shapes",
            "_speculative_memory_audit_inputs",
            "_spec_q_rows",
            "_spec_proposal_ids",
            "_spec_result_rows",
            "_spec_target_probability_rows",
            "_spec_bonus_noise",
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
        # Constructor pretouch must not consume a public Sequence ID.  The
        # compact worker DTO already carries exactly the fields needed by the
        # ragged prefill path and has no process-global allocator side effect.
        warmup_tokens = (0,) * seq_len
        seqs = [
            ScheduledSequence(
                scheduled_token_ids=warmup_tokens,
                is_prefill=True,
                num_cached_tokens=0,
                num_scheduled_tokens=seq_len,
                num_tokens=seq_len,
                last_token=0,
                block_table=(),
            )
            for _ in range(num_seqs)
        ]
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

    def _initialize_speculative_route_registry(self):
        """Build the host-only plan/route ledger before any route capture."""

        plan = self._plan_speculative_memory()
        self.speculative_memory_plan = plan
        self.draft_route_registry = build_draft_route_registry(
            plan,
            enforce_eager=self.enforce_eager,
            numerical_backend=getattr(self, "numerical_mode", "fast"),
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
        plan = self.speculative_memory_plan
        if not isinstance(plan, SpeculativeMemoryPlan) or not isinstance(
            getattr(self, "draft_route_registry", None), DraftRouteRegistry
        ):
            # ``allocate_kv_cache`` is also a focused test/embedding seam.  The
            # full constructor initializes earlier so graph profiling and final
            # capture share one immutable registry; direct callers may safely
            # request the same host-only initialization lazily.
            self._initialize_speculative_route_registry()
            plan = self.speculative_memory_plan

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
        # Reusable buffers are allocated after graph capture and remain live
        # across all runtime phases. The legacy plan assumes sequential buffer
        # lifetimes, so retain it as additional scratch rather than subtracting
        # buffers from a phase that may not actually include them. Price their
        # future ownership explicitly; allocator cache reuse cannot supply this
        # information through the pre-KV driver snapshot.
        persistent_workspace_bytes = self._speculative_workspace_reservation_bytes()
        # KV capacity is discrete. Reserve whole joint blocks so an allocation
        # that fits within the old remainder still relinquishes capacity for
        # permanent buffers and allocator segment variation. This is deliberately
        # conservative, like the graph allocator's one-joint-block minimum.
        persistent_workspace_reservation = (
            (persistent_workspace_bytes + joint_block_bytes - 1)
            // joint_block_bytes * joint_block_bytes
        )
        runtime_reservation = (
            persistent_workspace_reservation
            + profiled_graph_ownership
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
            persistent_workspace_reservation_bytes=persistent_workspace_reservation,
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
        query_sequence_ids = []
        query_context_lengths = []
        emission_query_indices = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for sequence_index, seq in enumerate(seqs):
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
            query_sequence_ids.extend([sequence_index] * seqlen_q)
            query_context_lengths.extend(range(start + 1, end + 1))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            if end == seq.num_tokens:
                emission_query_indices.append(cu_seqlens_q[-1] - 1)
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
        emission_query_indices = torch.tensor(
            emission_query_indices, dtype=torch.int64, pin_memory=True
        ).cuda(non_blocking=True)
        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
            tuple(query_sequence_ids),
            tuple(query_context_lengths),
            emission_query_indices,
        )
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
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables, query_sequence_ids=tuple(range(len(seqs))), query_context_lengths=tuple(len(seq) for seq in seqs))
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

    @staticmethod
    def _require_plan_int(name: str, value: int, *, minimum: int = 0) -> int:
        if type(value) is not int:
            raise SpeculativeDraftPlanError(f"{name} must be an integer")
        if value < minimum:
            raise SpeculativeDraftPlanError(
                f"{name} must be at least {minimum}, got {value}"
            )
        return value

    @staticmethod
    def _plan_committed_len(row) -> int:
        value = getattr(row, "committed_tokens", None)
        if value is None:
            value = getattr(row, "committed_len", None)
        return ModelRunner._require_plan_int(
            "row.committed_tokens", value, minimum=1
        )

    def _validate_speculative_step_plan(self, plan: SpecStepPlan, seqs):
        """Recompute V4 bytes and full-cycle geometry before draft allocation."""
        try:
            SpecStepPlan.__post_init__(plan)
            for row in plan.rows:
                SpecPlanRow.__post_init__(row)
            expected = build_speculative_step_plan(
                cycle_id=plan.cycle_id, rows=plan.rows,
                configured_k=self.config.configured_k,
                workspace_route_cap=plan.workspace_route_cap,
                effective_k=plan.effective_k,
                max_num_batched_tokens=self.config.max_num_batched_tokens,
                configured_workspace=self.speculative_memory_plan,
                route_key=plan.route_key,
                bypass_reason=plan.bypass_reason,
            )
            if plan != expected:
                raise ValueError("speculative plan geometry or workspace certificate drifted")
            if not plan.uses_speculation:
                raise ValueError("fallback speculative plans cannot execute")
            cache = getattr(self, "kv_cache", None)
            if not isinstance(cache, torch.Tensor) or cache.ndim < 3:
                raise ValueError("target KV cache is unavailable")
            for row in plan.rows:
                highest = row.highest_target_write_position
                if highest >= self.config.max_model_len:
                    raise ValueError("planned target write exceeds model position limit")
                if len(row.block_table) < highest // self.block_size + 1:
                    raise ValueError("planned target write lacks reserved blocks")
                if len(set(row.block_table)) != len(row.block_table):
                    raise ValueError("speculative row repeats a physical block")
                if any(block_id >= cache.size(2) for block_id in row.block_table):
                    raise ValueError("planned target block exceeds the physical pool")
        except (TypeError, ValueError) as error:
            raise SpeculativeDraftPlanError(str(error)) from error
        # Reuse V3's independent checks of live row order, request bounds,
        # draft readiness, physical draft IDs, tokens, and cache coverage.
        return self._validate_draft_discard_plan(as_draft_discard_plan(plan), seqs)

    def _validate_draft_discard_plan(
        self,
        plan,
        seqs: list[Sequence],
    ) -> tuple[tuple[DraftCycleRow, ...], int, DraftRouteKey]:
        """Fail closed on stale plan snapshots before CUDA work or RNG use."""

        if isinstance(plan, SpecStepPlan):
            return self._validate_speculative_step_plan(plan, seqs)

        if not self.speculation_enabled:
            raise SpeculativeDraftPlanError(
                "draft discard execution requires speculation to be enabled"
            )
        if getattr(plan, "fallback_reason", None) is not None:
            raise SpeculativeDraftPlanError(
                "a fallback discard plan cannot execute the draft model"
            )
        effective_k = self._require_plan_int(
            "plan.effective_k",
            getattr(plan, "effective_k", None),
            minimum=1,
        )
        configured_k = self._require_plan_int(
            "plan.configured_k",
            getattr(plan, "configured_k", None),
            minimum=1,
        )
        if configured_k != self.config.configured_k:
            raise SpeculativeDraftPlanError(
                "discard plan configured_k does not match the runner"
            )
        route_cap = self._require_plan_int(
            "plan.workspace_route_cap",
            getattr(plan, "workspace_route_cap", None),
            minimum=1,
        )
        if effective_k > min(configured_k, route_cap):
            raise SpeculativeDraftPlanError(
                "discard plan effective_k exceeds its configured/workspace cap"
            )

        plan_rows = getattr(plan, "rows", None)
        if not isinstance(plan_rows, tuple) or not plan_rows:
            raise SpeculativeDraftPlanError(
                "discard plan rows must be a non-empty tuple"
            )
        if not isinstance(seqs, list) or len(seqs) != len(plan_rows):
            raise SpeculativeDraftPlanError(
                "live sequence batch must match the discard-plan row count"
            )
        batch_size = len(plan_rows)
        if not speculative_route_fits_plan(
            self.speculative_memory_plan,
            batch_size=batch_size,
            effective_k=effective_k,
        ):
            raise SpeculativeDraftPlanError(
                "discard route exceeds the runner's reserved workspace plan"
            )
        if route_cap > self.speculative_memory_plan.max_effective_k:
            raise SpeculativeDraftPlanError(
                "scheduler workspace cap exceeds the runner's reserved plan"
            )
        draft_catchup_tokens = self._require_plan_int(
            "plan.draft_catchup_tokens",
            getattr(plan, "draft_catchup_tokens", None),
        )
        route_key = getattr(plan, "route_key", None)
        registry = getattr(self, "draft_route_registry", None)
        if not isinstance(route_key, DraftRouteKey) or not isinstance(
            registry, DraftRouteRegistry
        ):
            raise SpeculativeDraftPlanError(
                "discard plan does not carry a registered route key"
            )
        if not registry.validate_runtime_key(
            route_key,
            batch_size=batch_size,
            effective_k=effective_k,
            catchup_tokens=draft_catchup_tokens,
        ):
            raise SpeculativeDraftPlanError(
                "discard plan route is unregistered, uncertified, or unwarmed"
            )
        expected_mode = (
            DraftExecutionMode.EAGER_DYNAMIC
            if self.enforce_eager
            else DraftExecutionMode.CUDA_GRAPH
        )
        if route_key.execution_mode is not expected_mode:
            raise SpeculativeDraftPlanError(
                "discard plan execution mode does not match the runner"
            )
        total_scheduled_tokens = self._require_plan_int(
            "plan.total_scheduled_tokens",
            getattr(plan, "total_scheduled_tokens", None),
            minimum=1,
        )
        expected_total = draft_catchup_tokens + batch_size * (effective_k + 1)
        if total_scheduled_tokens != expected_total:
            raise SpeculativeDraftPlanError(
                "total_scheduled_tokens does not match catch-up plus discard work"
            )
        if total_scheduled_tokens > self.config.max_num_batched_tokens:
            raise SpeculativeDraftPlanError(
                "discard plan exceeds max_num_batched_tokens"
            )

        draft_counts = getattr(plan, "draft_step_token_counts", None)
        if (
            not isinstance(draft_counts, tuple)
            or any(type(count) is not int for count in draft_counts)
            or draft_counts != (batch_size,) * effective_k
        ):
            raise SpeculativeDraftPlanError(
                "draft_step_token_counts must contain one full-batch count per step"
            )
        target_query_tokens = self._require_plan_int(
            "plan.target_query_tokens",
            getattr(plan, "target_query_tokens", None),
            minimum=1,
        )
        if target_query_tokens != batch_size:
            raise SpeculativeDraftPlanError(
                "target_query_tokens does not match the ordinary target batch"
            )

        draft_cache = getattr(self, "draft_kv_cache", None)
        if not isinstance(draft_cache, torch.Tensor) or draft_cache.ndim < 3:
            raise SpeculativeDraftPlanError(
                "draft KV cache is unavailable or has invalid geometry"
            )
        physical_blocks = draft_cache.size(2)
        if physical_blocks < 1:
            raise SpeculativeDraftPlanError("draft KV cache contains no blocks")

        by_id = {}
        for seq in seqs:
            seq_id = getattr(seq, "seq_id", None)
            if type(seq_id) is not int or seq_id in by_id:
                raise SpeculativeDraftPlanError(
                    "live sequences must have unique integer seq_id values"
                )
            by_id[seq_id] = seq
        if len(by_id) != batch_size:
            raise SpeculativeDraftPlanError(
                "live sequence IDs do not match the discard batch"
            )

        views = []
        seen_ids = set()
        vocab_size = self.config.draft_hf_config.vocab_size
        for index, row in enumerate(plan_rows):
            seq_id = self._require_plan_int(
                f"rows[{index}].seq_id",
                getattr(row, "seq_id", None),
            )
            if seq_id in seen_ids or seq_id not in by_id:
                raise SpeculativeDraftPlanError(
                    "discard rows must name each live sequence exactly once"
                )
            seen_ids.add(seq_id)
            seq = seqs[index]
            if seq_id != seq.seq_id:
                raise SpeculativeDraftPlanError(
                    "discard-plan row order must match the live decode batch"
                )
            committed_len = self._plan_committed_len(row)
            if committed_len != len(seq) or committed_len != seq.num_tokens:
                raise SpeculativeDraftPlanError(
                    f"discard row {seq_id} has a stale committed-token snapshot"
                )
            if seq.is_prefill or seq.num_scheduled_tokens != 1:
                raise SpeculativeDraftPlanError(
                    f"discard row {seq_id} is not a one-token decode row"
                )

            target_cached = self._require_plan_int(
                f"rows[{index}].target_cached_tokens",
                getattr(row, "target_cached_tokens", None),
            )
            if (
                target_cached != seq.num_cached_tokens
                or target_cached != committed_len - 1
            ):
                raise SpeculativeDraftPlanError(
                    f"discard row {seq_id} has stale target-cache coverage"
                )
            draft_cached = self._require_plan_int(
                f"rows[{index}].draft_cached_tokens",
                getattr(row, "draft_cached_tokens", None),
            )
            if draft_cached != getattr(seq, "num_draft_cached_tokens", None):
                raise SpeculativeDraftPlanError(
                    f"discard row {seq_id} has stale draft-cache coverage"
                )
            if draft_cached > committed_len - 1:
                raise SpeculativeDraftPlanError(
                    f"discard row {seq_id} draft coverage passes the decode boundary"
                )

            remaining = self._require_plan_int(
                f"rows[{index}].remaining_completion_tokens",
                getattr(row, "remaining_completion_tokens", None),
                minimum=1,
            )
            live_remaining = seq.max_tokens - seq.num_completion_tokens
            if remaining != live_remaining or remaining < effective_k + 1:
                raise SpeculativeDraftPlanError(
                    f"discard row {seq_id} lacks completion-token headroom"
                )
            position_headroom = self._require_plan_int(
                f"rows[{index}].model_position_headroom",
                getattr(row, "model_position_headroom", None),
                minimum=1,
            )
            if (
                position_headroom != self.config.max_model_len - committed_len
                or position_headroom < effective_k
            ):
                raise SpeculativeDraftPlanError(
                    f"discard row {seq_id} has stale model-position headroom"
                )
            highest_position = committed_len + effective_k - 2
            if getattr(row, "highest_proposal_input_position", None) != highest_position:
                raise SpeculativeDraftPlanError(
                    f"discard row {seq_id} has an invalid highest proposal input"
                )
            if highest_position >= self.config.max_model_len:
                raise SpeculativeDraftPlanError(
                    f"discard row {seq_id} exceeds the model position limit"
                )

            block_table = getattr(row, "block_table", None)
            if not isinstance(block_table, tuple) or block_table != tuple(
                seq.block_table
            ):
                raise SpeculativeDraftPlanError(
                    f"discard row {seq_id} has a stale block-table snapshot"
                )
            required_blocks = highest_position // self.block_size + 1
            if len(block_table) < required_blocks:
                raise SpeculativeDraftPlanError(
                    f"discard row {seq_id} lacks reserved proposal blocks"
                )
            if any(
                type(block_id) is not int
                or block_id < 0
                or block_id >= physical_blocks
                for block_id in block_table
            ):
                raise SpeculativeDraftPlanError(
                    f"discard row {seq_id} contains an invalid physical block ID"
                )

            token_ids = tuple(seq.token_ids)
            if (
                len(token_ids) != committed_len
                or any(
                    type(token_id) is not int
                    or token_id < 0
                    or token_id >= vocab_size
                    for token_id in token_ids
                )
            ):
                raise SpeculativeDraftPlanError(
                    f"discard row {seq_id} has invalid committed token IDs"
                )
            views.append(
                DraftCycleRow(
                    seq_id=seq_id,
                    token_ids=token_ids,
                    committed_len=committed_len,
                    target_cached_tokens=target_cached,
                    draft_cached_tokens=draft_cached,
                    block_table=block_table,
                    temperature=seq.temperature,
                    top_k=seq.top_k,
                    top_p=seq.top_p,
                )
            )
        live_catchup_tokens = sum(
            row.committed_len - 1 - row.draft_cached_tokens for row in views
        )
        if live_catchup_tokens != draft_catchup_tokens:
            raise SpeculativeDraftPlanError(
                "draft_catchup_tokens does not match live draft-cache coverage"
            )
        return tuple(views), effective_k, route_key

    @staticmethod
    def _draft_device_tensor(values, *, dtype, device):
        if device.type == "cuda":
            return torch.tensor(values, dtype=dtype, pin_memory=True).to(
                device=device,
                non_blocking=True,
            )
        return torch.tensor(values, dtype=dtype, device=device)

    def _prepare_draft_block_tables(
        self,
        rows: tuple[DraftCycleRow | DraftCatchupView, ...],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        max_len = max(len(row.block_table) for row in rows)
        values = [
            row.block_table + (-1,) * (max_len - len(row.block_table))
            for row in rows
        ]
        return self._draft_device_tensor(
            values,
            dtype=torch.int32,
            device=device,
        )

    def _prepare_draft_catchup(
        self,
        rows: tuple[DraftCycleRow, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        catchup = tuple(
            DraftCatchupView(
                scheduled_token_ids=row.token_ids[
                    row.draft_cached_tokens : row.committed_len - 1
                ],
                is_prefill=True,
                num_cached_tokens=row.draft_cached_tokens,
                num_scheduled_tokens=(
                    row.committed_len - 1 - row.draft_cached_tokens
                ),
                num_tokens=row.committed_len - 1,
                last_token=row.token_ids[row.committed_len - 2],
                block_table=row.block_table,
            )
            for row in rows
            if row.draft_cached_tokens < row.committed_len - 1
        )
        if not catchup:
            device = self.draft_kv_cache.device
            empty = torch.empty(0, dtype=torch.int64, device=device)
            return empty, empty, 0

        device = self.draft_kv_cache.device
        input_ids = []
        positions = []
        cu_q = [0]
        cu_k = [0]
        slot_mapping = []
        query_sequence_ids = []
        query_context_lengths = []
        max_q = 0
        max_k = 0
        for sequence_index, row in enumerate(catchup):
            start = row.num_cached_tokens
            end = row.num_tokens
            input_ids.extend(row.scheduled_token_ids)
            positions.extend(range(start, end))
            query_len = end - start
            cu_q.append(cu_q[-1] + query_len)
            cu_k.append(cu_k[-1] + end)
            max_q = max(max_q, query_len)
            max_k = max(max_k, end)
            query_sequence_ids.extend([sequence_index] * query_len)
            query_context_lengths.extend(range(start + 1, end + 1))
            for position in range(start, end):
                block_id = row.block_table[position // self.block_size]
                slot_mapping.append(
                    block_id * self.block_size + position % self.block_size
                )
        input_ids_tensor = self._draft_device_tensor(
            input_ids, dtype=torch.int64, device=device
        )
        positions_tensor = self._draft_device_tensor(
            positions, dtype=torch.int64, device=device
        )
        set_context(
            True,
            cu_seqlens_q=self._draft_device_tensor(
                cu_q, dtype=torch.int32, device=device
            ),
            cu_seqlens_k=self._draft_device_tensor(
                cu_k, dtype=torch.int32, device=device
            ),
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            slot_mapping=self._draft_device_tensor(
                slot_mapping, dtype=torch.int32, device=device
            ),
            block_tables=self._prepare_draft_block_tables(
                catchup, device=device
            ),
            query_sequence_ids=tuple(query_sequence_ids),
            query_context_lengths=tuple(query_context_lengths),
        )
        return input_ids_tensor, positions_tensor, len(input_ids)

    @torch.inference_mode()
    def _run_draft_catchup(
        self,
        rows: tuple[DraftCycleRow, ...],
        route_key: DraftRouteKey | None = None,
    ) -> int:
        """Populate missing committed-prefix draft KV without changing coverage."""

        missing = tuple(
            row.committed_len - 1 - row.draft_cached_tokens for row in rows
        )
        one_token_rows = tuple(
            row for row, count in zip(rows, missing, strict=True) if count == 1
        )
        if (
            route_key is not None
            and one_token_rows
            and all(count <= 1 for count in missing)
        ):
            # A one-token lag has decode geometry. Reuse the registered draft
            # graph (including its LM head) and discard the logits. Rows already
            # caught up are excluded so their next KV slot is never overwritten.
            device = self.draft_kv_cache.device
            input_ids = self._draft_device_tensor(
                tuple(row.token_ids[row.draft_cached_tokens] for row in one_token_rows),
                dtype=torch.int64,
                device=device,
            )
            try:
                input_ids, positions = self._prepare_draft_decode(
                    one_token_rows, input_ids, -1
                )
                self._run_draft_decode_model(input_ids, positions, route_key)
                return len(one_token_rows)
            finally:
                reset_context()

        try:
            input_ids, positions, count = self._prepare_draft_catchup(rows)
            if count:
                # Longer and genuinely ragged catch-up remains a bounded eager
                # route until an independently measured graph family earns its
                # persistent-memory cost.
                self.draft_model(input_ids, positions)
            return count
        finally:
            reset_context()

    def _prepare_draft_decode(
        self,
        rows: tuple[DraftCycleRow, ...],
        input_token_ids: torch.Tensor,
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.draft_kv_cache.device
        positions = tuple(row.committed_len - 1 + step for row in rows)
        slots = tuple(
            row.block_table[position // self.block_size] * self.block_size
            + position % self.block_size
            for row, position in zip(rows, positions, strict=True)
        )
        positions_tensor = self._draft_device_tensor(
            positions, dtype=torch.int64, device=device
        )
        set_context(
            False,
            slot_mapping=self._draft_device_tensor(
                slots, dtype=torch.int32, device=device
            ),
            context_lens=self._draft_device_tensor(
                tuple(position + 1 for position in positions),
                dtype=torch.int32,
                device=device,
            ),
            block_tables=self._prepare_draft_block_tables(rows, device=device),
            query_sequence_ids=tuple(range(len(rows))),
            query_context_lengths=tuple(position + 1 for position in positions),
        )
        return input_token_ids, positions_tensor

    def _draft_graph_key(
        self,
        batch_size: int,
        context,
        route_key: DraftRouteKey,
    ) -> int | None:
        if route_key.execution_mode is DraftExecutionMode.EAGER_DYNAMIC:
            if not self.enforce_eager:
                raise SpeculativeDraftPlanError(
                    "an eager draft route is not registered for graph mode"
                )
            return None
        if route_key.execution_mode is not DraftExecutionMode.CUDA_GRAPH \
                or self.enforce_eager:
            raise SpeculativeDraftPlanError(
                "draft route execution mode does not match graph availability"
            )
        graph_bs = getattr(self, "draft_graph_bs", ())
        graphs = getattr(self, "draft_graphs", None)
        variables = getattr(self, "draft_graph_vars", None)
        if not graph_bs or not isinstance(graphs, dict) or variables is None:
            raise SpeculativeDraftPlanError(
                "registered draft graph resources are unavailable"
            )
        key = route_key.batch_bucket
        if (
            key not in graph_bs
            or key not in graphs
            or not 0 < batch_size <= key
        ):
            raise SpeculativeDraftPlanError(
                "registered draft graph bucket is unavailable"
            )
        block_tables = context.block_tables
        if (
            context.slot_mapping is None
            or context.context_lens is None
            or block_tables is None
            or block_tables.ndim != 2
            or block_tables.size(0) != batch_size
            or block_tables.size(1) > variables["block_tables"].size(1)
        ):
            raise SpeculativeDraftPlanError(
                "live draft decode metadata does not fit its registered graph"
            )
        return key

    @torch.inference_mode()
    def _run_draft_decode_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        route_key: DraftRouteKey,
    ) -> tuple[torch.Tensor, bool]:
        batch_size = input_ids.size(0)
        context = get_context()
        graph_key = self._draft_graph_key(batch_size, context, route_key)
        if graph_key is None:
            hidden_states = self.draft_model(input_ids, positions)
            return self.draft_model.compute_logits(hidden_states), False

        variables = self.draft_graph_vars
        variables["input_ids"][:batch_size] = input_ids
        variables["positions"][:batch_size] = positions
        variables["slot_mapping"].fill_(-1)
        variables["slot_mapping"][:batch_size] = context.slot_mapping
        variables["context_lens"].zero_()
        variables["context_lens"][:batch_size] = context.context_lens
        variables["block_tables"].fill_(-1)
        variables["block_tables"][
            :batch_size, : context.block_tables.size(1)
        ] = context.block_tables
        self.draft_graphs[graph_key].replay()
        logits = variables.get("logits")
        if isinstance(logits, torch.Tensor):
            return logits[:batch_size], True
        # Compatibility for test doubles and old in-memory registries. New
        # runtime captures always own graph-resident logits.
        return self.draft_model.compute_logits(
            variables["outputs"][:batch_size]
        ), True

    def _prepare_draft_sample_metadata(
        self,
        rows: tuple[DraftCycleRow, ...],
    ):
        temperatures, host_top_k, host_top_p, _ = self._prepare_sample_metadata(
            rows, self.config.draft_hf_config.vocab_size
        )
        device = self.draft_kv_cache.device
        temperatures = self._draft_device_tensor(
            temperatures, dtype=torch.float32, device=device
        )
        top_k_buckets = tuple(
            (
                top_k,
                None
                if active_rows is None
                else self._draft_device_tensor(
                    active_rows, dtype=torch.int64, device=device
                ),
            )
            for top_k, active_rows in host_top_k
        )
        top_p_plan = None
        if host_top_p is not None:
            active_rows, top_ps = host_top_p
            top_p_plan = (
                None
                if active_rows is None
                else self._draft_device_tensor(
                    active_rows, dtype=torch.int64, device=device
                ),
                self._draft_device_tensor(
                    tuple(1.0 - top_p for top_p in top_ps),
                    dtype=torch.float32,
                    device=device,
                ),
            )
        return temperatures, top_k_buckets, top_p_plan

    def _run_draft_proposal_step(
        self,
        rows: tuple[DraftCycleRow, ...],
        input_token_ids: torch.Tensor,
        step: int,
        temperatures: torch.Tensor,
        top_k_buckets,
        top_p_plan,
        probabilities_out: torch.Tensor,
        route_key: DraftRouteKey,
    ) -> tuple[torch.Tensor, bool]:
        try:
            input_ids, positions = self._prepare_draft_decode(
                rows, input_token_ids, step
            )
            logits, used_graph = self._run_draft_decode_model(
                input_ids, positions, route_key
            )
        finally:
            # Attention context must never survive a forward failure or become
            # visible to the sampler/next draft step.
            reset_context()
        trusted_sample = getattr(
            self.sampler, "sample_exact_with_probabilities_trusted", None
        )
        if trusted_sample is None:
            sample = self.sampler.sample_exact_with_probabilities(
                logits,
                temperatures,
                top_k_buckets=top_k_buckets,
                top_p_plan=top_p_plan,
                probabilities_out=probabilities_out,
            )
        else:
            sample = trusted_sample(
                logits,
                temperatures,
                top_k_buckets=top_k_buckets,
                top_p_plan=top_p_plan,
                probabilities_out=probabilities_out,
                all_greedy=all(row.temperature == 0.0 for row in rows),
            )
        if (
            sample.probabilities.shape != probabilities_out.shape
            or sample.probabilities.dtype != torch.float32
            or sample.probabilities.device != probabilities_out.device
            or sample.probabilities.data_ptr() != probabilities_out.data_ptr()
            or sample.probabilities.stride() != probabilities_out.stride()
        ):
            raise SpeculativeDraftPlanError(
                "sampler did not retain probabilities in the reserved q row"
            )
        token_ids = getattr(sample, "token_ids", None)
        vocab_size = self.config.draft_hf_config.vocab_size
        if (
            not isinstance(token_ids, torch.Tensor)
            or token_ids.shape != (len(rows),)
            or token_ids.dtype != torch.int64
            or token_ids.device != probabilities_out.device
        ):
            raise SpeculativeDraftPlanError(
                "draft sampler returned invalid token IDs"
            )
        return token_ids, used_graph

    def _execute_draft_proposals_validated(
        self,
        rows: tuple[DraftCycleRow, ...],
        effective_k: int,
        route_key: DraftRouteKey,
    ) -> DraftProposalExecution:
        """Tensor-bearing proposal seam; callers must keep it cycle-local."""

        batch_size = len(rows)
        vocab_size = self.config.draft_hf_config.vocab_size
        device = self.draft_kv_cache.device
        q_rows = getattr(self, "_spec_q_rows", None)
        required_q_rows = effective_k * batch_size
        if (
            isinstance(q_rows, torch.Tensor)
            and q_rows.device == device
            and q_rows.dtype == torch.float32
            and q_rows.shape[0] >= required_q_rows
            and q_rows.shape[1] == vocab_size
        ):
            q_storage = q_rows[:required_q_rows].view(
                effective_k, batch_size, vocab_size
            )
        else:
            q_storage = torch.empty(
                (effective_k, batch_size, vocab_size),
                dtype=torch.float32,
                device=device,
            )
        if not q_storage.is_contiguous():
            raise SpeculativeDraftPlanError("K-major q storage must be contiguous")
        proposal_storage = getattr(self, "_spec_proposal_ids", None)
        if (
            isinstance(proposal_storage, torch.Tensor)
            and proposal_storage.device == device
            and proposal_storage.dtype == torch.int64
            and proposal_storage.numel() >= batch_size * effective_k
        ):
            proposal_ids = proposal_storage[: batch_size * effective_k].view(
                batch_size, effective_k
            )
        else:
            proposal_ids = torch.empty(
                (batch_size, effective_k),
                dtype=torch.int64,
                device=device,
            )
        temperatures, top_k_buckets, top_p_plan = (
            self._prepare_draft_sample_metadata(rows)
        )
        catchup_positions = self._run_draft_catchup(rows, route_key)
        input_token_ids = self._draft_device_tensor(
            tuple(row.last_token for row in rows),
            dtype=torch.int64,
            device=device,
        )
        graph_steps = 0
        eager_steps = 0
        invalid_token_ids = torch.zeros((), dtype=torch.bool, device=device)
        try:
            for step in range(effective_k):
                q_step = q_storage[step]
                if not q_step.is_contiguous():
                    raise SpeculativeDraftPlanError(
                        "each reserved K-major q destination must be contiguous"
                    )
                sampled_ids, used_graph = self._run_draft_proposal_step(
                    rows,
                    input_token_ids,
                    step,
                    temperatures,
                    top_k_buckets,
                    top_p_plan,
                    q_step,
                    route_key,
                )
                if sampled_ids.shape != (batch_size,):
                    raise SpeculativeDraftPlanError(
                        "draft sampler returned an invalid token shape"
                    )
                invalid_token_ids.logical_or_(
                    ((sampled_ids < 0) | (sampled_ids >= vocab_size)).any()
                )
                proposal_ids[:, step].copy_(sampled_ids)
                # Keep invalid injected values away from the next embedding;
                # the accumulated device flag is checked once before any
                # proposal can reach target verification or commit.
                input_token_ids = sampled_ids.clamp(0, vocab_size - 1)
                graph_steps += int(used_graph)
                eager_steps += int(not used_graph)
            if bool(invalid_token_ids.item()):
                raise SpeculativeDraftPlanError(
                    "draft sampler returned invalid token IDs"
                )
            q_bkv = q_storage.permute(1, 0, 2)
            if q_bkv.untyped_storage().data_ptr() != q_storage.untyped_storage().data_ptr():
                raise SpeculativeDraftPlanError(
                    "B-major q diagnostics must be a zero-copy view"
                )
            return DraftProposalExecution(
                rows=rows,
                proposal_token_ids=proposal_ids,
                q_storage_kbv=q_storage,
                q_probabilities=q_bkv,
                catchup_positions=catchup_positions,
                graph_decode_steps=graph_steps,
                eager_decode_steps=eager_steps,
            )
        finally:
            reset_context()

    def _execute_draft_proposals(
        self,
        plan,
        seqs: list[Sequence],
    ) -> DraftProposalExecution:
        """Lower-level tensor seam for V3 tests; production uses the host wrapper."""

        try:
            rows, effective_k, route_key = self._validate_draft_discard_plan(
                plan, seqs
            )
            return self._execute_draft_proposals_validated(
                rows, effective_k, route_key
            )
        finally:
            reset_context()

    @staticmethod
    def _host_draft_discard_result(
        execution: DraftProposalExecution,
        effective_k: int,
        route_key: DraftRouteKey,
    ) -> DraftDiscardResult:
        proposal_rows = execution.proposal_token_ids.tolist()
        rows = tuple(
            DraftDiscardRowResult(
                seq_id=row.seq_id,
                # Proposal-position KV is deliberately discarded. Step zero did
                # process the prior committed tail, so committed coverage is C.
                coverage_after_commit=row.committed_len,
                proposed_token_ids=tuple(token_ids),
                proposal_count=effective_k,
            )
            for row, token_ids in zip(
                execution.rows, proposal_rows, strict=True
            )
        )
        q = execution.q_probabilities
        q_storage = execution.q_storage_kbv
        return DraftDiscardResult(
            rows=rows,
            route_key=route_key,
            effective_k=effective_k,
            catchup_positions=execution.catchup_positions,
            draft_positions=len(rows) * effective_k,
            graph_decode_steps=execution.graph_decode_steps,
            eager_decode_steps=execution.eager_decode_steps,
            q_shape=tuple(q.shape),
            q_stride=tuple(q.stride()),
            q_dtype=str(q.dtype),
            q_storage_contiguous=q_storage.is_contiguous(),
            q_view_zero_copy=(
                q.untyped_storage().data_ptr()
                == q_storage.untyped_storage().data_ptr()
            ),
        )

    def _speculative_workspace_reservation_bytes(self):
        """Reserve permanent buffers independently of the legacy phase plan.

        Round each allocation to a 2 MiB segment allowance, including small
        metadata buffers. This deliberately allows separate allocator segments
        instead of relying on cache reuse or packing to satisfy the KV budget.
        The final audit measures resident ownership and does not charge this
        future-allocation allowance a second time.
        """
        plan = self.speculative_memory_plan
        batch = min(4, plan.batch_size)
        k = min(4, plan.max_effective_k)
        vocab = self.config.hf_config.vocab_size
        sizes = (
            batch * k * vocab * 4,       # FP32 draft probability rows
            batch * k * 8,              # int64 proposal IDs
            batch * (k + 1) * vocab * 4, # FP32 target probability rows
            batch * vocab * 4,           # FP32 bonus noise
            batch * (k + 3) * 8,         # int64 result rows
        )
        alignment = 2 * 1024 * 1024
        return sum((size + alignment - 1) // alignment * alignment for size in sizes)

    def _initialize_speculative_workspaces(self):
        """Allocate one reusable buffer set for every admitted B/K route."""

        plan = self.speculative_memory_plan
        max_batch = min(4, plan.batch_size)
        max_k = min(4, plan.max_effective_k)
        vocab_size = self.config.hf_config.vocab_size
        device = self.kv_cache.device
        self._spec_q_rows = torch.empty(
            (max_batch * max_k, vocab_size),
            dtype=torch.float32,
            device=device,
        )
        self._spec_proposal_ids = torch.empty(
            max_batch * max_k, dtype=torch.int64, device=device
        )
        self._spec_target_probability_rows = torch.empty(
            (max_batch * (max_k + 1), vocab_size),
            dtype=torch.float32,
            device=device,
        )
        self._spec_bonus_noise = torch.empty(
            (max_batch, vocab_size), dtype=torch.float32, device=device
        )
        self._spec_result_rows = torch.empty(
            (max_batch, max_k + 3), dtype=torch.int64, device=device
        )

    def _pretouch_speculative_verifier(self):
        from nanovllm.engine.speculative_execution import warm_verifier
        from nanovllm.layers.sampler import ModifiedRejectionSampler
        self._initialize_speculative_workspaces()
        self.speculative_rejection_sampler = ModifiedRejectionSampler()
        warm_verifier(self)

    def snapshot_speculative_rng(self):
        device = self.kv_cache.device
        return (torch.get_rng_state(),
                torch.cuda.get_rng_state(device) if device.type == "cuda" else None)

    def restore_speculative_rng(self, snapshot):
        torch.set_rng_state(snapshot[0])
        if snapshot[1] is not None:
            torch.cuda.set_rng_state(snapshot[1], self.kv_cache.device)

    def run_speculative(self, plan, seqs):
        from nanovllm.engine.speculative_execution import execute
        failure = None
        try:
            return execute(self, plan, seqs)
        except Exception as error:
            failure = (type(error).__name__, str(error), tuple(getattr(error, "__notes__", ())))
            error.__traceback__ = None
        finally:
            reset_context()
        name, message, notes = failure
        error = SpeculativeDraftExecutionError(f"speculative verification failed ({name}): {message}")
        add_note = getattr(error, "add_note", None)
        if callable(add_note):
            for note in notes:
                add_note(note)
        raise error from None

    def run_speculative_discard(
        self,
        plan,
        seqs: list[Sequence],
    ) -> DraftDiscardResult:
        """Compute real draft proposals, discard them, and preserve target RNG.

        Plan/state validation deliberately occurs before the RNG snapshot seam,
        any proposal workspace allocation, and every draft kernel. The returned
        object contains no CUDA tensor, so proposal laws cannot escape the cycle.
        """

        # Never validate against a context left by unrelated or failed caller
        # code. Plan errors are host-only and retain their precise public type.
        reset_context()
        rows, effective_k, route_key = self._validate_draft_discard_plan(
            plan, seqs
        )

        def execute_and_discard():
            try:
                execution = self._execute_draft_proposals_validated(
                    rows, effective_k, route_key
                )
                return self._host_draft_discard_result(
                    execution, effective_k, route_key
                )
            finally:
                reset_context()

        failure = None
        try:
            result = self._run_draft_phase(
                "V3 draft compute-then-discard",
                execute_and_discard,
            )
        except Exception as error:
            # Never propagate a tensor-bearing traceback to the engine.  A
            # retained caller exception would otherwise retain q/logits and a
            # substantial CUDA allocation indefinitely.  Preserve only stable
            # host diagnostics and any cleanup notes, then sever the traceback.
            failure = (
                type(error).__name__,
                str(error),
                tuple(getattr(error, "__notes__", ())),
            )
            error.__traceback__ = None
        finally:
            reset_context()
        if failure is not None:
            error_type, message, notes = failure
            wrapped = SpeculativeDraftExecutionError(
                f"V3 draft execution failed ({error_type}): {message}"
            )
            add_note = getattr(wrapped, "add_note", None)
            if callable(add_note):
                for note in notes:
                    add_note(note)
            raise wrapped from None
        return result

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
            if (
                self.enforce_eager
                or input_ids.size(0) > MAX_CUDA_GRAPH_BATCH_SIZE
            ):
                return self.model.compute_logits(self.model(input_ids, positions))
            bs = input_ids.size(0)
            graph_bs = next(
                (x for x in self.graph_bs if x >= bs and x in self.graphs),
                None,
            )
            if graph_bs is None:
                # Graph availability is an optimization, not an execution
                # prerequisite. Keep the live context for ordinary eager
                # decode when no compatible capture exists.
                return self.model.compute_logits(self.model(input_ids, positions))
            context = get_context()
            graph = self.graphs[graph_bs]
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

    @torch.inference_mode()
    def run_model_all_positions(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Run a ragged target pass and retain every live query logit row."""

        t = input_ids.size(0)
        context = get_context()
        if not context.is_prefill or context.cu_seqlens_q is None:
            raise RuntimeError("all-position target verification requires ragged context")
        ns = context.cu_seqlens_q.numel() - 1
        graph_key = (
            self._select_varlen_graph_key(t, ns)
            if context.block_tables is not None
            else None
        )
        if graph_key is not None and not self._varlen_context_fits_graph(
            t, ns, context, graph_key
        ):
            graph_key = None
        if graph_key is None:
            if hasattr(self, "varlen_graphs") and context.block_tables is not None:
                self.varlen_miss += 1
            hidden_states = self.model(input_ids, positions)
        else:
            self._fill_varlen(input_ids, positions, context, graph_key)
            self.varlen_graphs[graph_key].replay()
            hidden_states = self.varlen_vars["outputs"][:t]
        return self.model.compute_logits_all(hidden_states)

    def run(
        self,
        seqs: list[Sequence | ScheduledSequence],
        is_prefill: bool,
    ) -> list[int]:
        try:
            input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
            emitting_indices = (
                [
                    index for index, seq in enumerate(seqs)
                    if seq.num_cached_tokens + seq.num_scheduled_tokens
                    == seq.num_tokens
                ]
                if getattr(self, "numerical_mode", "fast") == "invariant"
                and is_prefill
                else list(range(len(seqs)))
            )
            sample_seqs = [seqs[index] for index in emitting_indices]
            temperatures, top_k_buckets, top_p_plan, all_greedy = (
                self.prepare_sample(sample_seqs)
                if self.rank == 0
                else (None, (), None, False)
            )
            logits = self.run_model(input_ids, positions, is_prefill)
            if self.rank == 0:
                if not sample_seqs:
                    tokens = torch.empty(0, dtype=torch.int64, device=logits.device)
                elif all_greedy:
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
                sampled_token_ids = tokens.tolist()
                token_ids = [0] * len(seqs)
                for index, token_id in zip(
                    emitting_indices, sampled_token_ids, strict=True
                ):
                    token_ids[index] = token_id
            else:
                token_ids = None
            return token_ids
        finally:
            reset_context()

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(
            self.config.max_num_seqs,
            MAX_CUDA_GRAPH_BATCH_SIZE,
        )
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        # Include a nonmultiple endpoint (for example max_num_seqs=17).  Draft
        # admission uses the same bucket policy, so a successful draft cycle can
        # never hand an otherwise valid batch to an absent target graph.
        self.graph_bs = list(draft_graph_batch_buckets(max_bs))
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
        registry = getattr(self, "draft_route_registry", None)
        if not isinstance(registry, DraftRouteRegistry):
            raise RuntimeError(
                "draft route registry must exist before draft graph capture"
            )
        graph_buckets = registry.graph_buckets
        if not graph_buckets:
            raise RuntimeError("draft graph route registry contains no batch buckets")
        max_bs = graph_buckets[-1]
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
        logits = torch.zeros(
            max_bs,
            hf_config.vocab_size,
            dtype=hf_config.dtype,
        )
        self.draft_graph_bs = list(graph_buckets)
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
                logits[:bs] = self.draft_model.compute_logits(outputs[:bs])
                with torch.cuda.graph(graph, self.draft_graph_pool):
                    outputs[:bs] = self.draft_model(
                        input_ids[:bs], positions[:bs]
                    )
                    logits[:bs] = self.draft_model.compute_logits(outputs[:bs])
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
            logits=logits,
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
        # instead of falling back to eager in the many-decoder regime. This reduces
        # dispatch overhead; it does not guarantee a wall-time ITL bound.
        # Tiers are prefix-slices of the SAME buffers —
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

    def resolve_draft_route_admission(
        self,
        seqs: list[Sequence] | tuple[Sequence, ...],
    ) -> DraftRouteAdmission | None:
        """Resolve all ready K values before scheduler reservation or CUDA work."""

        registry = getattr(self, "draft_route_registry", None)
        if not isinstance(registry, DraftRouteRegistry):
            return None
        if not self._draft_route_runtime_resources_ready(registry):
            return None
        selected = tuple(seqs)
        if not selected:
            return None
        catchup_tokens = 0
        for seq in selected:
            committed_tokens = len(seq)
            coverage = getattr(seq, "num_draft_cached_tokens", None)
            if (
                type(coverage) is not int
                or committed_tokens < 1
                or coverage < 0
                or coverage > committed_tokens - 1
            ):
                return None
            catchup_tokens += committed_tokens - 1 - coverage
        return registry.resolve(
            batch_size=len(selected),
            catchup_tokens=catchup_tokens,
        )

    def _draft_route_runtime_resources_ready(
        self,
        registry: DraftRouteRegistry,
    ) -> bool:
        """Host-only preflight for every persistent resource named by a route."""

        if (
            registry.ready_keys != registry.router_admitted_keys
            or getattr(self, "draft_model", None) is None
            or getattr(self, "sampler", None) is None
            or not isinstance(getattr(self, "draft_kv_cache", None), torch.Tensor)
        ):
            return False
        if self.enforce_eager:
            return all(
                key.execution_mode is DraftExecutionMode.EAGER_DYNAMIC
                for key in registry.router_admitted_keys
            )

        buckets = registry.graph_buckets
        graphs = getattr(self, "draft_graphs", None)
        graph_bs = getattr(self, "draft_graph_bs", None)
        variables = getattr(self, "draft_graph_vars", None)
        required_variables = {
            "input_ids",
            "positions",
            "slot_mapping",
            "context_lens",
            "block_tables",
            "outputs",
            "logits",
        }
        if (
            tuple(graph_bs or ()) != buckets
            or not isinstance(graphs, dict)
            or tuple(sorted(graphs)) != buckets
            or not isinstance(variables, dict)
            or not required_variables.issubset(variables)
            or any(
                not isinstance(variables[name], torch.Tensor)
                for name in required_variables
            )
        ):
            return False
        max_bucket = buckets[-1]
        max_blocks = (
            self.config.max_model_len + self.block_size - 1
        ) // self.block_size
        return bool(
            variables["input_ids"].size(0) >= max_bucket
            and variables["positions"].size(0) >= max_bucket
            and variables["slot_mapping"].size(0) >= max_bucket
            and variables["context_lens"].size(0) >= max_bucket
            and variables["block_tables"].ndim == 2
            and variables["block_tables"].size(0) >= max_bucket
            and variables["block_tables"].size(1) >= max_blocks
            and variables["outputs"].size(0) >= max_bucket
        )

    def _pretouch_draft_decode_witness(self, batch_size: int) -> None:
        device = self.draft_kv_cache.device
        physical_blocks = self.draft_kv_cache.size(2)
        max_blocks = (
            self.config.max_model_len + self.block_size - 1
        ) // self.block_size
        input_ids = torch.zeros(batch_size, dtype=torch.int64, device=device)
        positions = torch.zeros(batch_size, dtype=torch.int64, device=device)
        slot_mapping = torch.arange(
            batch_size, dtype=torch.int32, device=device
        ).remainder_(physical_blocks * self.block_size)
        context_lens = torch.ones(
            batch_size, dtype=torch.int32, device=device
        )
        physical_table = torch.arange(
            max_blocks, dtype=torch.int32, device=device
        ).remainder_(physical_blocks)
        block_tables = physical_table.unsqueeze(0).expand(
            batch_size, -1
        ).contiguous()
        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            query_sequence_ids=tuple(range(batch_size)),
            query_context_lengths=(1,) * batch_size,
        )
        try:
            with torch.inference_mode():
                hidden_states = self.draft_model(input_ids, positions)
                self.draft_model.compute_logits(hidden_states)
        finally:
            reset_context()

    @torch.inference_mode()
    def _pretouch_draft_graph_decode_witness(
        self,
        batch_size: int,
        route_key: DraftRouteKey,
    ) -> None:
        """Replay one registered draft graph and execute its live output head."""

        device = self.draft_kv_cache.device
        physical_blocks = self.draft_kv_cache.size(2)
        max_blocks = (
            self.config.max_model_len + self.block_size - 1
        ) // self.block_size
        input_ids = torch.zeros(batch_size, dtype=torch.int64, device=device)
        positions = torch.zeros(batch_size, dtype=torch.int64, device=device)
        slot_mapping = torch.arange(
            batch_size, dtype=torch.int32, device=device
        ).remainder_(physical_blocks * self.block_size)
        context_lens = torch.ones(
            batch_size, dtype=torch.int32, device=device
        )
        physical_table = torch.arange(
            max_blocks, dtype=torch.int32, device=device
        ).remainder_(physical_blocks)
        block_tables = physical_table.unsqueeze(0).expand(
            batch_size, -1
        ).contiguous()
        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        try:
            logits, used_graph = self._run_draft_decode_model(
                input_ids,
                positions,
                route_key,
            )
            if not used_graph or logits.shape != (
                batch_size,
                self.config.draft_hf_config.vocab_size,
            ):
                raise RuntimeError(
                    "draft graph pretouch did not execute its registered route"
                )
        finally:
            reset_context()

    @torch.inference_mode()
    def _pretouch_draft_catchup_witness(
        self,
        *,
        total_tokens: int,
        num_seqs: int,
        high_context: bool,
    ) -> None:
        """Execute one paged ragged witness without publishing logical coverage."""

        if total_tokens < num_seqs or num_seqs < 1:
            raise ValueError("catch-up witness requires at least one token per row")
        device = self.draft_kv_cache.device
        physical_blocks = self.draft_kv_cache.size(2)
        max_blocks = (
            self.config.max_model_len + self.block_size - 1
        ) // self.block_size
        base, remainder = divmod(total_tokens, num_seqs)
        query_lengths = tuple(
            base + int(row < remainder) for row in range(num_seqs)
        )
        starts = tuple(
            (
                max(self.config.max_model_len - query_len - 1, 0)
                if high_context and (num_seqs == 1 or row % 2)
                else 0
            )
            for row, query_len in enumerate(query_lengths)
        )
        cu_q = [0]
        cu_k = [0]
        positions = []
        slot_mapping = []
        query_sequence_ids = []
        query_context_lengths = []
        physical_table = tuple(
            logical_block % physical_blocks
            for logical_block in range(max_blocks)
        )
        for start, query_len in zip(starts, query_lengths, strict=True):
            end = start + query_len
            if end > self.config.max_model_len:
                raise RuntimeError("catch-up witness exceeds max_model_len")
            cu_q.append(cu_q[-1] + query_len)
            cu_k.append(cu_k[-1] + end)
            positions.extend(range(start, end))
            query_sequence_ids.extend([len(cu_q) - 2] * query_len)
            query_context_lengths.extend(range(start + 1, end + 1))
            slot_mapping.extend(
                physical_table[position // self.block_size] * self.block_size
                + position % self.block_size
                for position in range(start, end)
            )
        input_ids = torch.zeros(
            total_tokens, dtype=torch.int64, device=device
        )
        positions_tensor = torch.tensor(
            positions, dtype=torch.int64, device=device
        )
        set_context(
            True,
            cu_seqlens_q=torch.tensor(cu_q, dtype=torch.int32, device=device),
            cu_seqlens_k=torch.tensor(cu_k, dtype=torch.int32, device=device),
            max_seqlen_q=max(query_lengths),
            max_seqlen_k=max(
                start + query_len
                for start, query_len in zip(starts, query_lengths, strict=True)
            ),
            slot_mapping=torch.tensor(
                slot_mapping, dtype=torch.int32, device=device
            ),
            block_tables=torch.tensor(
                (physical_table,) * num_seqs,
                dtype=torch.int32,
                device=device,
            ),
            query_sequence_ids=tuple(query_sequence_ids),
            query_context_lengths=tuple(query_context_lengths),
        )
        try:
            self.draft_model(input_ids, positions_tensor)
        finally:
            reset_context()

    def _pretouch_exact_sampler_envelope(self, batch_size: int) -> None:
        """Exercise the dense worst-case exact sampler and sparse metadata seam."""

        device = self.draft_kv_cache.device
        vocab_size = self.config.draft_hf_config.vocab_size
        logits = torch.zeros(
            batch_size,
            vocab_size,
            dtype=self.config.draft_hf_config.dtype,
            device=device,
        )
        temperatures = torch.ones(
            batch_size, dtype=torch.float32, device=device
        )
        if batch_size > 1:
            temperatures[0] = 0.0
        probabilities = torch.empty(
            batch_size, vocab_size, dtype=torch.float32, device=device
        )
        top_k_buckets = ()
        if vocab_size > 1:
            # The runtime accepts every effective top-k in [1, V-1].  Exercise
            # the maximum returned values/indices payload before publishing the
            # single all-compositions sampler envelope as ready.
            top_k_buckets = ((vocab_size - 1, None),)
        probability_cutoffs = torch.full(
            (batch_size,), 0.1, dtype=torch.float32, device=device
        )
        self.sampler.sample_exact_with_probabilities(
            logits,
            temperatures,
            top_k_buckets=top_k_buckets,
            top_p_plan=(None, probability_cutoffs),
            probabilities_out=probabilities,
        )

        if batch_size > 1 and vocab_size > 2:
            # Worst legal heterogeneous filter shape: all but one row active.
            # Passing explicit row indices exercises the B-1 gather/copy that
            # the dense homogeneous call above deliberately avoids.
            near_dense_rows = torch.arange(
                0, batch_size - 1, dtype=torch.int64, device=device
            )
            near_dense_temperatures = temperatures.clone().fill_(1.0)
            near_dense_cutoffs = torch.full(
                (near_dense_rows.numel(),),
                0.1,
                dtype=torch.float32,
                device=device,
            )
            self.sampler.sample_exact_with_probabilities(
                logits,
                near_dense_temperatures,
                top_k_buckets=((vocab_size - 1, near_dense_rows),),
                top_p_plan=(near_dense_rows, near_dense_cutoffs),
                race_noise=torch.ones(
                    batch_size,
                    vocab_size,
                    dtype=torch.float32,
                    device=device,
                ),
                probabilities_out=torch.empty(
                    batch_size,
                    vocab_size,
                    dtype=torch.float32,
                    device=device,
                ),
            )

            # Sparse row indices and multiple sequential top-k buckets add no
            # larger live tensor than the dense call, but they are a distinct
            # metadata/control route and must be exercised once.
            sparse_batch = min(batch_size, 4)
            sparse_logits = logits[:sparse_batch]
            sparse_temperatures = temperatures[:sparse_batch].clone()
            sparse_temperatures.fill_(1.0)
            split = max(sparse_batch // 2, 1)
            first_rows = torch.arange(
                0, split, dtype=torch.int64, device=device
            )
            second_rows = torch.arange(
                split, sparse_batch, dtype=torch.int64, device=device
            )
            sparse_top_k = [(1, first_rows)]
            if second_rows.numel():
                sparse_top_k.append(
                    (min(2, vocab_size - 1), second_rows)
                )
            top_p_rows = torch.arange(
                0, sparse_batch, 2, dtype=torch.int64, device=device
            )
            self.sampler.sample_exact_with_probabilities(
                sparse_logits,
                sparse_temperatures,
                top_k_buckets=tuple(sparse_top_k),
                top_p_plan=(
                    top_p_rows,
                    torch.full(
                        (top_p_rows.numel(),),
                        0.1,
                        dtype=torch.float32,
                        device=device,
                    ),
                ),
                race_noise=torch.ones(
                    sparse_batch,
                    vocab_size,
                    dtype=torch.float32,
                    device=device,
                ),
                probabilities_out=torch.empty(
                    sparse_batch,
                    vocab_size,
                    dtype=torch.float32,
                    device=device,
                ),
            )

    def _pretouch_draft_routes(self) -> None:
        """Warm every declared V3 component, then atomically publish readiness."""

        registry = getattr(self, "draft_route_registry", None)
        plan = getattr(self, "speculative_memory_plan", None)
        if not isinstance(registry, DraftRouteRegistry) or not isinstance(
            plan, SpeculativeMemoryPlan
        ):
            raise RuntimeError("draft route pretouch requires its memory registry")
        if not registry.entries:
            raise RuntimeError("speculative route registry is unexpectedly empty")

        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        warmed: set[DraftWarmComponentKey] = set()
        try:
            batch_cap = max(
                entry.key.batch_bucket for entry in registry.entries
            )
            if self.enforce_eager:
                witnesses = sorted({1, min(2, batch_cap), batch_cap})
                for batch_size in witnesses:
                    self._pretouch_draft_decode_witness(batch_size)
                warmed.add(
                    DraftWarmComponentKey(
                        "draft_decode_eager_dynamic", batch_cap
                    )
                )
            else:
                actual_graph_buckets = tuple(sorted(self.draft_graphs))
                if actual_graph_buckets != registry.graph_buckets:
                    raise RuntimeError(
                        "captured draft graph buckets do not match the route registry"
                    )
                graph_witnesses = set(actual_graph_buckets)
                if batch_cap >= 3:
                    # One live interior shape proves bucket slicing/output-head
                    # integration rather than only exact capture endpoints.
                    graph_witnesses.add(3)
                for live_batch_size in sorted(graph_witnesses):
                    batch_bucket = min(
                        bucket
                        for bucket in actual_graph_buckets
                        if bucket >= live_batch_size
                    )
                    route_key = next(
                        entry.key
                        for entry in registry.entries
                        if entry.key.batch_bucket == batch_bucket
                        and entry.key.effective_k == 1
                        and entry.key.catchup_family
                        is DraftCatchupFamily.NONE
                    )
                    self._pretouch_draft_graph_decode_witness(
                        live_batch_size,
                        route_key,
                    )
                warmed.update(
                    DraftWarmComponentKey("draft_decode_graph", batch_bucket)
                    for batch_bucket in actual_graph_buckets
                )

            catchup_component = DraftWarmComponentKey(
                "draft_catchup_paged_dynamic", batch_cap
            )
            if catchup_component in registry.desired_warm_components:
                max_catchup, max_catchup_batch = max_eligible_draft_catchup(
                    max_num_batched_tokens=self.config.max_num_batched_tokens,
                    max_model_len=self.config.max_model_len,
                    batch_cap=batch_cap,
                )
                if max_catchup < 1 or max_catchup_batch < 1:
                    raise RuntimeError(
                        "registered catch-up routes have no legal runtime witness"
                    )
                max_single_row = min(
                    self.config.max_num_batched_tokens - 2,
                    self.config.max_model_len - 2,
                )
                catchup_witnesses = {
                    (1, 1, False),
                    (max_single_row, 1, True),
                    (max_catchup, max_catchup_batch, True),
                }
                if max_catchup >= 257:
                    catchup_witnesses.add(
                        (257, min(batch_cap, 257), True)
                    )
                for total_tokens, num_seqs, high_context in sorted(
                    catchup_witnesses
                ):
                    self._pretouch_draft_catchup_witness(
                        total_tokens=total_tokens,
                        num_seqs=num_seqs,
                        high_context=high_context,
                    )
                warmed.add(catchup_component)

            self._pretouch_exact_sampler_envelope(batch_cap)
            warmed.add(
                DraftWarmComponentKey("exact_sampler_envelope", batch_cap)
            )
            torch.cuda.synchronize()
        finally:
            reset_context()

        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        self._draft_route_pretouch_peak_bytes = max(peak - baseline, 0)
        allowed_peak = (
            self._warmup_transient_bytes + plan.reservation_bytes
        )
        if self._draft_route_pretouch_peak_bytes > allowed_peak:
            raise SpeculativeKVCacheCapacityError(
                "V3 route pretouch exceeded the modeled runtime envelope: "
                f"observed={self._draft_route_pretouch_peak_bytes}, "
                f"allowed={allowed_peak}"
            )

        ready_registry = registry.with_warmed_components(warmed)
        if ready_registry.ready_keys != ready_registry.router_admitted_keys:
            missing = ready_registry.router_admitted_keys - ready_registry.ready_keys
            raise RuntimeError(
                f"draft route pretouch left {len(missing)} route(s) unready"
            )
        self.draft_route_registry = ready_registry
