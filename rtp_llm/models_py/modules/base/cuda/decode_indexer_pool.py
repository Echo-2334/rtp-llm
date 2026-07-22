"""Decode-only candidate-pool reuse for the GLM/DSA indexer.

The feature excludes speculative target verify and keeps per-request, per-layer
pools because this mode is used when indexcache cross-layer reuse is disabled.
With CUDA Graph enabled, APPEND batches use per-row masks so new or
discontinuous requests bootstrap from an exact score while stable requests in
the same graph replay continue to use their sparse pools.

Enable it with ``RTP_LLM_DECODE_INDEXER_POOL_PROFILE=A``, ``B``, or ``APPEND``.
Pool reuse starts strictly above
``RTP_LLM_DECODE_INDEXER_POOL_MIN_KV_LENGTH`` (64K by default). APPEND requires
an initial 8K pool, a 16K maximum pool, and 16 source chunks. Profiles A and B
default to a 16K fixed-size pool.

Set ``RTP_LLM_DECODE_INDEXER_PACKED_POOL=1`` with APPEND to keep a fixed,
materialized 8K KV pool. Its steady path scores the pool and current contiguous
source chunk through paged MQA, then updates the pool only from the final Top2K.
Set ``RTP_LLM_DECODE_INDEXER_PACKED_POOL_UPDATE=APPEND`` to retain V1 append
semantics with a ring-backed 16K materialized K pool. Graph buckets below
``RTP_LLM_DECODE_INDEXER_PACKED_POOL_MIN_BATCH`` keep the fused sparse path.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch


ExactScoreFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
]
CandidateScoreFn = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        str,
    ],
    torch.Tensor,
]
PrepareCandidateFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, int],
]
ScoreMaterializedFn = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        torch.Tensor,
        torch.Tensor,
        str,
    ],
    torch.Tensor,
]
SelectTopkFn = Callable[
    [torch.Tensor, torch.Tensor, int, str],
    torch.Tensor,
]
PoolChunkScoreFn = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    torch.Tensor,
]
PoolChunkTopkFn = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        int,
        str,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    torch.Tensor,
]
ScheduledPoolChunkScoreFn = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        int,
        int,
        int,
        Optional[torch.Tensor],
    ],
    Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
]
PackedPoolStepFn = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        int,
        int,
        int,
        Optional[torch.Tensor],
    ],
    torch.Tensor,
]
PackedAppendPoolStepFn = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        int,
        int,
        int,
        Optional[torch.Tensor],
    ],
    torch.Tensor,
]


_REFRESH_STREAMS: Dict[torch.device, torch.cuda.Stream] = {}
_LOGGED_MESSAGES: set[str] = set()


def _log_once(key: str, level: int, message: str, *args: Any) -> None:
    if key in _LOGGED_MESSAGES:
        return
    _LOGGED_MESSAGES.add(key)
    logging.log(level, message, *args)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in ("1", "true", "on", "yes"):
        return True
    if normalized in ("0", "false", "off", "no"):
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}")


def _copy_kernel_k_for_dump(
    kv_cache_fp8: torch.Tensor,
    block_table_row: torch.Tensor,
    length: int,
    head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Copy one request's K pages into a self-contained kernel-ready cache."""
    if kv_cache_fp8.dtype != torch.uint8 or kv_cache_fp8.ndim != 4:
        raise ValueError("indexer K dump expects uint8 [pages, page_size, 1, stride]")
    if kv_cache_fp8.shape[2] != 1:
        raise ValueError("indexer K dump expects one KV head")

    page_size = int(kv_cache_fp8.shape[1])
    cache_stride = int(kv_cache_fp8.shape[3])
    if length <= 0 or head_dim <= 0 or cache_stride != head_dim + 4:
        raise ValueError(
            f"invalid indexer K dump shape: length={length} head_dim={head_dim} "
            f"cache_stride={cache_stride}"
        )

    page_count = (length + page_size - 1) // page_size
    if page_count > block_table_row.numel():
        raise ValueError(
            f"indexer K dump needs {page_count} pages, block table has "
            f"{block_table_row.numel()}"
        )
    physical_pages = block_table_row[:page_count].to(dtype=torch.long)
    if bool((physical_pages < 0).any()) or bool(
        (physical_pages >= kv_cache_fp8.shape[0]).any()
    ):
        raise ValueError("indexer K dump block table contains an invalid physical page")

    # Preserve DeepGEMM's physical page layout: each page contains all token K
    # bytes followed by all token fp32 scale bytes. Renumber pages locally so
    # the dumped cache and block table can be passed to the kernel as-is.
    kernel_cache = kv_cache_fp8.index_select(0, physical_pages).contiguous()
    kernel_block_table = torch.arange(
        page_count, dtype=torch.int32, device=kv_cache_fp8.device
    ).reshape(1, page_count)
    return kernel_cache, kernel_block_table


def _get_refresh_stream(device: torch.device) -> torch.cuda.Stream:
    stream = _REFRESH_STREAMS.get(device)
    if stream is None:
        with torch.cuda.device(device):
            stream = torch.cuda.Stream(device=device)
        _REFRESH_STREAMS[device] = stream
    return stream


@dataclass(frozen=True)
class DecodeIndexerPoolConfig:
    profile: str = "OFF"
    min_kv_length: int = 64 * 1024
    interval: int = 8
    pool_size: int = 16 * 1024
    max_pool_size: int = 16 * 1024
    chunks: int = 8
    chunks_per_step: int = 0
    refresh_lead: int = 0
    q_mode: str = "rolling"
    anchor_phase: int = 0
    async_refresh: bool = True
    state_ttl_steps: int = 64
    packed_pool: bool = False
    packed_pool_update: str = "REPLACE"
    packed_pool_min_batch: int = 32

    @property
    def enabled(self) -> bool:
        return self.profile != "OFF"

    @property
    def full_chunk_mask(self) -> int:
        return (1 << self.chunks) - 1

    @property
    def max_recent_tokens(self) -> int:
        # A pool is built at an anchor and becomes active refresh_lead steps
        # later. It then serves one full interval before the next swap.
        return self.interval + self.refresh_lead

    def refresh_chunks(self, decode_step: int) -> Tuple[int, ...]:
        phase = decode_step % self.interval
        if self.profile == "A":
            return (phase,)
        if self.profile == "B" and phase >= self.anchor_phase:
            first = (phase - self.anchor_phase) * self.chunks_per_step
            return tuple(range(first, min(first + self.chunks_per_step, self.chunks)))
        return ()

    @classmethod
    def from_env(cls) -> "DecodeIndexerPoolConfig":
        env_name = "RTP_LLM_DECODE_INDEXER_POOL_PROFILE"
        profile = os.environ.get(env_name, "OFF").strip().upper()
        if profile in ("", "0", "OFF", "NONE"):
            return cls()
        pool_size = int(
            os.environ.get("RTP_LLM_DECODE_INDEXER_POOL_SIZE", str(16 * 1024))
        )
        chunk_topk = 2048
        if pool_size <= 0 or pool_size % chunk_topk != 0:
            raise ValueError(
                "RTP_LLM_DECODE_INDEXER_POOL_SIZE must be a positive multiple of 2048"
            )
        if profile == "APPEND":
            packed_pool = _env_bool(
                "RTP_LLM_DECODE_INDEXER_PACKED_POOL", False
            )
            packed_pool_update = os.environ.get(
                "RTP_LLM_DECODE_INDEXER_PACKED_POOL_UPDATE", "REPLACE"
            ).strip().upper()
            if packed_pool_update not in ("REPLACE", "APPEND"):
                raise ValueError(
                    "RTP_LLM_DECODE_INDEXER_PACKED_POOL_UPDATE must be REPLACE or APPEND"
                )
            source_chunks = int(
                os.environ.get("RTP_LLM_DECODE_INDEXER_SOURCE_CHUNKS", "16")
            )
            max_pool_size = int(
                os.environ.get(
                    "RTP_LLM_DECODE_INDEXER_POOL_MAX_SIZE",
                    str(
                        pool_size
                        if packed_pool and packed_pool_update == "REPLACE"
                        else 16 * 1024
                    ),
                )
            )
            config = cls(
                profile=profile,
                min_kv_length=int(
                    os.environ.get(
                        "RTP_LLM_DECODE_INDEXER_POOL_MIN_KV_LENGTH",
                        str(64 * 1024),
                    )
                ),
                interval=source_chunks,
                pool_size=pool_size,
                max_pool_size=max_pool_size,
                chunks=source_chunks,
                q_mode="rolling",
                async_refresh=False,
                state_ttl_steps=int(
                    os.environ.get("RTP_LLM_DECODE_INDEXER_POOL_STATE_TTL", "64")
                ),
                packed_pool=packed_pool,
                packed_pool_update=packed_pool_update,
                packed_pool_min_batch=int(
                    os.environ.get(
                        "RTP_LLM_DECODE_INDEXER_PACKED_POOL_MIN_BATCH", "32"
                    )
                ),
            )
            if config.pool_size != 8 * 1024:
                raise ValueError("APPEND profile requires an initial 8192-entry pool")
            expected_max_pool_size = (
                config.pool_size
                if config.packed_pool and config.packed_pool_update == "REPLACE"
                else 16 * 1024
            )
            if config.max_pool_size != expected_max_pool_size:
                raise ValueError(
                    "APPEND profile requires max pool size 8192 for packed REPLACE "
                    "or 16384 otherwise"
                )
            if config.packed_pool_min_batch <= 0:
                raise ValueError(
                    "RTP_LLM_DECODE_INDEXER_PACKED_POOL_MIN_BATCH must be positive"
                )
            if config.chunks != 16:
                raise ValueError("APPEND profile currently requires 16 source chunks")
            if config.state_ttl_steps < config.interval:
                raise ValueError(
                    "RTP_LLM_DECODE_INDEXER_POOL_STATE_TTL must be at least one interval"
                )
            _log_once(
                "decode-indexer-pool-enabled",
                logging.INFO,
                "decode indexer append pool enabled: N=%d initial=%d max=%d min_kv=%d",
                config.interval,
                config.pool_size,
                config.max_pool_size,
                config.min_kv_length,
            )
            if config.packed_pool:
                _log_once(
                    "decode-indexer-packed-pool-enabled",
                    logging.INFO,
                    "decode indexer materialized pool enabled: update=%s pool=%d max=%d "
                    "min_batch=%d chunks=%d",
                    config.packed_pool_update,
                    config.pool_size,
                    config.max_pool_size,
                    config.packed_pool_min_batch,
                    config.chunks,
                )
            return config

        chunks = pool_size // chunk_topk
        interval = chunks
        if profile == "A":
            defaults = dict(
                chunks_per_step=1,
                refresh_lead=chunks,
                q_mode="rolling",
                anchor_phase=0,
            )
        elif profile == "B":
            if chunks % 2 != 0:
                raise ValueError("profile B requires an even number of pool chunks")
            defaults = dict(
                chunks_per_step=2,
                refresh_lead=chunks // 2,
                q_mode="fixed",
                anchor_phase=chunks // 2,
            )
        else:
            raise ValueError(
                f"{env_name} must be OFF, A, B, or APPEND, got {profile!r}"
            )

        q_mode = os.environ.get(
            "RTP_LLM_DECODE_INDEXER_POOL_Q_MODE", defaults["q_mode"]
        ).strip().lower()
        if q_mode not in ("rolling", "fixed"):
            raise ValueError(
                "RTP_LLM_DECODE_INDEXER_POOL_Q_MODE must be rolling or fixed"
            )
        config = cls(
            profile=profile,
            min_kv_length=int(
                os.environ.get(
                    "RTP_LLM_DECODE_INDEXER_POOL_MIN_KV_LENGTH", str(64 * 1024)
                )
            ),
            interval=interval,
            pool_size=pool_size,
            max_pool_size=pool_size,
            chunks=chunks,
            chunks_per_step=defaults["chunks_per_step"],
            refresh_lead=defaults["refresh_lead"],
            q_mode=q_mode,
            anchor_phase=defaults["anchor_phase"],
            async_refresh=_env_bool(
                "RTP_LLM_DECODE_INDEXER_POOL_ASYNC_REFRESH", True
            ),
            state_ttl_steps=int(
                os.environ.get("RTP_LLM_DECODE_INDEXER_POOL_STATE_TTL", "64")
            ),
        )
        if config.pool_size % config.chunks != 0:
            raise ValueError("decode indexer pool size must divide evenly into chunks")
        if config.min_kv_length < config.pool_size:
            raise ValueError(
                "RTP_LLM_DECODE_INDEXER_POOL_MIN_KV_LENGTH must be at least pool size"
            )
        if config.refresh_lead * config.chunks_per_step != config.chunks:
            raise ValueError("decode indexer refresh schedule must cover every chunk")
        if config.anchor_phase != config.interval - config.refresh_lead:
            raise ValueError(
                "decode indexer anchor phase must equal interval - refresh_lead"
            )
        if config.state_ttl_steps < config.interval:
            raise ValueError(
                "RTP_LLM_DECODE_INDEXER_POOL_STATE_TTL must be at least one interval"
            )
        _log_once(
            "decode-indexer-pool-enabled",
            logging.INFO,
            "decode indexer pool enabled: profile=%s q_mode=%s async_refresh=%s "
            "N=%d P=%d C=%d min_kv=%d",
            config.profile,
            config.q_mode,
            config.async_refresh,
            config.interval,
            config.pool_size,
            config.chunks,
            config.min_kv_length,
        )
        return config


@dataclass
class _RequestState:
    request_id: int
    slot: int
    last_step: int = -1
    last_kv_length: int = -1
    last_seen_tick: int = 0
    active_parity: int = 0
    coverage: List[int] = field(default_factory=lambda: [0, 0])
    build_started: bool = False
    build_mask: int = 0
    pending_event: Optional[torch.cuda.Event] = None


class DecodeIndexerPool:
    """Per-indexer-layer runtime for the two decode pool profiles."""

    def __init__(
        self,
        config: DecodeIndexerPoolConfig,
        index_topk: int,
        index_n_heads: int,
        index_head_dim: int,
        layer_idx: int = -1,
    ) -> None:
        if config.profile != "APPEND" and config.pool_size != config.chunks * index_topk:
            raise ValueError(
                "decode indexer pool requires pool_size == chunks * index_topk; "
                f"got {config.pool_size} != {config.chunks} * {index_topk}"
            )
        if config.profile == "APPEND" and config.pool_size != 4 * index_topk:
            raise ValueError("decode indexer APPEND profile requires initial pool == 4 * TopK")
        if config.packed_pool and config.profile != "APPEND":
            raise ValueError("materialized indexer pool requires the APPEND profile")
        if config.packed_pool and config.pool_size != 8 * 1024:
            raise ValueError("materialized indexer pool requires a fixed 8192-entry pool")
        if (
            config.packed_pool
            and config.packed_pool_update == "REPLACE"
            and config.max_pool_size != config.pool_size
        ):
            raise ValueError("materialized REPLACE pool max size must remain 8192")
        if (
            config.packed_pool
            and config.packed_pool_update == "APPEND"
            and config.max_pool_size != 16 * 1024
        ):
            raise ValueError("materialized APPEND pool max size must be 16384")
        if config.packed_pool and (index_head_dim != 128 or index_n_heads != 32):
            raise ValueError("materialized indexer pool currently requires H32 D128")
        self.config = config
        self.index_topk = index_topk
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.layer_idx = layer_idx
        self._states: Dict[int, _RequestState] = {}
        self._free_slots: List[int] = []
        self._capacity = 0
        self._pools: Optional[torch.Tensor] = None
        self._anchor_q: Optional[torch.Tensor] = None
        self._anchor_weights: Optional[torch.Tensor] = None
        self._graph_active_parity: Optional[torch.Tensor] = None
        self._graph_coverage: Optional[torch.Tensor] = None
        self._append_pools: Optional[torch.Tensor] = None
        self._append_pool_lengths: Optional[torch.Tensor] = None
        self._append_inverse_map: Optional[torch.Tensor] = None
        self._append_graph_max_seq_len = 0
        self._packed_pool_kv: Optional[torch.Tensor] = None
        self._packed_ready: Optional[torch.Tensor] = None
        self._packed_protected: Optional[torch.Tensor] = None
        self._packed_cursor: Optional[torch.Tensor] = None
        self._packed_base_offsets: Optional[torch.Tensor] = None
        self._graph_slot_base: Optional[int] = None
        self._device: Optional[torch.device] = None
        self._tick = 0

    def _supported(
        self,
        q_fp8: torch.Tensor,
        attention_inputs: Any,
        batch_size: int,
    ) -> bool:
        if not self.config.enabled or not q_fp8.is_cuda:
            return False
        if bool(getattr(attention_inputs, "is_speculative", False)):
            _log_once(
                "decode-indexer-pool-speculative",
                logging.WARNING,
                "decode indexer pool is disabled for speculative decoding",
            )
            return False
        if bool(getattr(attention_inputs, "is_target_verify", False)):
            _log_once(
                "decode-indexer-pool-target-verify",
                logging.WARNING,
                "decode indexer pool is disabled for target verify",
            )
            return False
        request_ids = getattr(attention_inputs, "decode_request_id", None)
        decode_steps = getattr(attention_inputs, "decode_step", None)
        kv_lengths = getattr(attention_inputs, "decode_kv_length", None)
        metadata = (request_ids, decode_steps, kv_lengths)
        if any(not isinstance(tensor, torch.Tensor) for tensor in metadata):
            return False
        if any(tensor.is_cuda or tensor.numel() != batch_size for tensor in metadata):
            _log_once(
                "decode-indexer-pool-invalid-metadata",
                logging.WARNING,
                "decode indexer pool requires CPU request/step/length metadata for every row",
            )
            return False
        if q_fp8.shape[0] != batch_size:
            return False
        return True

    def _refresh_stream(self, device: torch.device) -> torch.cuda.Stream:
        if not self.config.async_refresh:
            return torch.cuda.current_stream(device)
        return _get_refresh_stream(device)

    def _wait_pending(self, state: _RequestState, device: torch.device) -> None:
        if state.pending_event is not None:
            torch.cuda.current_stream(device).wait_event(state.pending_event)
            state.pending_event = None

    def _reset_runtime(self, device: torch.device) -> None:
        if self._device is not None and self._device != device:
            for state in self._states.values():
                self._wait_pending(state, self._device)
        self._states.clear()
        self._free_slots.clear()
        self._capacity = 0
        self._pools = None
        self._anchor_q = None
        self._anchor_weights = None
        self._graph_active_parity = None
        self._graph_coverage = None
        self._append_pools = None
        self._append_pool_lengths = None
        self._append_inverse_map = None
        self._append_graph_max_seq_len = 0
        self._packed_pool_kv = None
        self._packed_ready = None
        self._packed_protected = None
        self._packed_cursor = None
        self._packed_base_offsets = None
        self._graph_slot_base = None
        self._device = device

    def _ensure_capacity(
        self,
        required: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        device = q_fp8.device
        if self._device != device:
            self._reset_runtime(device)
        if required <= self._capacity:
            return

        new_capacity = max(1, self._capacity)
        while new_capacity < required:
            new_capacity *= 2

        main_stream = torch.cuda.current_stream(device)
        refresh_stream = self._refresh_stream(device)
        if refresh_stream != main_stream:
            main_stream.wait_stream(refresh_stream)

        new_pools: Optional[torch.Tensor] = None
        new_append_pools: Optional[torch.Tensor] = None
        new_append_pool_lengths: Optional[torch.Tensor] = None
        new_append_inverse_map: Optional[torch.Tensor] = None
        new_packed_pool_kv: Optional[torch.Tensor] = None
        new_packed_ready: Optional[torch.Tensor] = None
        new_packed_protected: Optional[torch.Tensor] = None
        new_packed_cursor: Optional[torch.Tensor] = None
        new_packed_base_offsets: Optional[torch.Tensor] = None
        if self.config.profile == "APPEND":
            new_append_pools = torch.zeros(
                (new_capacity, self.config.max_pool_size),
                dtype=torch.int32,
                device=device,
            )
            new_append_pool_lengths = torch.zeros(
                new_capacity, dtype=torch.int32, device=device
            )
            if self._append_pools is not None:
                new_append_pools[: self._capacity].copy_(self._append_pools)
                assert self._append_pool_lengths is not None
                new_append_pool_lengths[: self._capacity].copy_(
                    self._append_pool_lengths
                )
            if self._append_inverse_map is not None:
                new_append_inverse_map = torch.zeros(
                    (new_capacity, self._append_graph_max_seq_len),
                    dtype=torch.int32,
                    device=device,
                )
                new_append_inverse_map[: self._capacity].copy_(
                    self._append_inverse_map
                )
            if self.config.packed_pool:
                packed_pool_capacity = (
                    self.config.max_pool_size
                    if self.config.packed_pool_update == "APPEND"
                    else self.config.pool_size
                )
                pages_per_pool = packed_pool_capacity // 64
                entry_bytes = self.index_head_dim + 4
                new_packed_pool_kv = torch.zeros(
                    (new_capacity * pages_per_pool, 64, 1, entry_bytes),
                    dtype=torch.uint8,
                    device=device,
                )
                new_packed_ready = torch.zeros(
                    new_capacity, dtype=torch.int32, device=device
                )
                new_packed_protected = torch.zeros(
                    (new_capacity, self.config.pool_size),
                    dtype=torch.int32,
                    device=device,
                )
                new_packed_cursor = torch.zeros(
                    new_capacity, dtype=torch.int32, device=device
                )
                new_packed_base_offsets = torch.zeros(
                    new_capacity, dtype=torch.int32, device=device
                )
                if self._packed_pool_kv is not None:
                    new_packed_pool_kv.view(
                        new_capacity, pages_per_pool, 64, 1, entry_bytes
                    )[: self._capacity].copy_(
                        self._packed_pool_kv.view(
                            self._capacity, pages_per_pool, 64, 1, entry_bytes
                        )
                    )
                    assert self._packed_ready is not None
                    assert self._packed_protected is not None
                    assert self._packed_cursor is not None
                    assert self._packed_base_offsets is not None
                    new_packed_ready[: self._capacity].copy_(self._packed_ready)
                    new_packed_protected[: self._capacity].copy_(
                        self._packed_protected
                    )
                    new_packed_cursor[: self._capacity].copy_(self._packed_cursor)
                    new_packed_base_offsets[: self._capacity].copy_(
                        self._packed_base_offsets
                    )
        else:
            new_pools = torch.zeros(
                (
                    new_capacity,
                    2,
                    self.config.chunks,
                    self.index_topk,
                ),
                dtype=torch.int32,
                device=device,
            )
            if self._pools is not None:
                new_pools[: self._capacity].copy_(self._pools)

        new_graph_active_parity = torch.zeros(
            new_capacity, dtype=torch.long, device=device
        )
        new_graph_coverage = torch.full(
            (new_capacity, 2),
            self.config.pool_size,
            dtype=torch.long,
            device=device,
        )
        if self._graph_active_parity is not None:
            new_graph_active_parity[: self._capacity].copy_(
                self._graph_active_parity
            )
            assert self._graph_coverage is not None
            new_graph_coverage[: self._capacity].copy_(self._graph_coverage)

        new_anchor_q: Optional[torch.Tensor] = None
        new_anchor_weights: Optional[torch.Tensor] = None
        if self.config.q_mode == "fixed":
            new_anchor_q = torch.zeros(
                (
                    new_capacity,
                    self.index_n_heads,
                    self.index_head_dim,
                ),
                dtype=q_fp8.dtype,
                device=device,
            )
            new_anchor_weights = torch.zeros(
                (new_capacity, self.index_n_heads),
                dtype=weights.dtype,
                device=device,
            )
            if self._anchor_q is not None:
                new_anchor_q[: self._capacity].copy_(self._anchor_q)
                assert self._anchor_weights is not None
                new_anchor_weights[: self._capacity].copy_(self._anchor_weights)

        self._free_slots.extend(range(new_capacity - 1, self._capacity - 1, -1))
        self._pools = new_pools
        self._append_pools = new_append_pools
        self._append_pool_lengths = new_append_pool_lengths
        self._append_inverse_map = new_append_inverse_map
        self._packed_pool_kv = new_packed_pool_kv
        self._packed_ready = new_packed_ready
        self._packed_protected = new_packed_protected
        self._packed_cursor = new_packed_cursor
        self._packed_base_offsets = new_packed_base_offsets
        self._anchor_q = new_anchor_q
        self._anchor_weights = new_anchor_weights
        self._graph_active_parity = new_graph_active_parity
        self._graph_coverage = new_graph_coverage
        self._capacity = new_capacity

    def _ensure_append_graph_state(self, graph_max_seq_len: int) -> None:
        if self.config.profile != "APPEND":
            return
        if self._append_inverse_map is not None:
            if self._append_inverse_map.shape != (
                self._capacity,
                graph_max_seq_len,
            ):
                raise RuntimeError(
                    "decode indexer APPEND graph max sequence length changed after capture"
                )
            return
        assert self._device is not None
        self._append_inverse_map = torch.zeros(
            (self._capacity, graph_max_seq_len),
            dtype=torch.int32,
            device=self._device,
        )
        self._append_graph_max_seq_len = graph_max_seq_len

    def _evict_stale(self, device: torch.device) -> None:
        stale_ids = [
            request_id
            for request_id, state in self._states.items()
            if self._tick - state.last_seen_tick > self.config.state_ttl_steps
        ]
        for request_id in stale_ids:
            state = self._states.pop(request_id)
            self._wait_pending(state, device)
            self._free_slots.append(state.slot)

    def _claim_slot(
        self,
        request_id: int,
        device: torch.device,
        preferred_slot: Optional[int],
    ) -> int:
        if preferred_slot is None:
            if not self._free_slots:
                raise RuntimeError("decode indexer pool slot allocator is empty")
            return self._free_slots.pop()
        if preferred_slot < 0 or preferred_slot >= self._capacity:
            raise RuntimeError(
                f"decode indexer pool slot {preferred_slot} is outside capacity {self._capacity}"
            )

        occupant_id = next(
            (
                existing_id
                for existing_id, existing in self._states.items()
                if existing.slot == preferred_slot and existing_id != request_id
            ),
            None,
        )
        if occupant_id is not None:
            occupant = self._states.pop(occupant_id)
            self._wait_pending(occupant, device)
        if preferred_slot in self._free_slots:
            self._free_slots.remove(preferred_slot)
        return preferred_slot

    def _new_state(
        self,
        request_id: int,
        device: torch.device,
        preferred_slot: Optional[int] = None,
    ) -> _RequestState:
        slot = self._claim_slot(request_id, device, preferred_slot)
        state = _RequestState(request_id=request_id, slot=slot)
        self._states[request_id] = state
        return state

    def _reset_state(self, state: _RequestState, device: torch.device) -> None:
        self._wait_pending(state, device)
        state.last_step = -1
        state.last_kv_length = -1
        state.active_parity = 0
        state.coverage[0] = 0
        state.coverage[1] = 0
        state.build_started = False
        state.build_mask = 0
        if self._graph_active_parity is not None:
            self._graph_active_parity[state.slot] = 0
            assert self._graph_coverage is not None
            self._graph_coverage[state.slot].zero_()

    def _swap_or_reset(
        self,
        state: _RequestState,
        decode_step: int,
        device: torch.device,
    ) -> bool:
        if decode_step == 0 or decode_step % self.config.interval != 0:
            return True
        if not state.build_started or state.build_mask != self.config.full_chunk_mask:
            self._reset_state(state, device)
            return False
        self._wait_pending(state, device)
        state.active_parity = 1 - state.active_parity
        if self._graph_active_parity is not None:
            self._graph_active_parity[state.slot] = state.active_parity
        state.build_started = False
        state.build_mask = 0
        return True

    def _bootstrap(
        self,
        logits: torch.Tensor,
        states: Sequence[_RequestState],
        kv_lengths: Sequence[int],
        select_topk: SelectTopkFn,
    ) -> None:
        if not states:
            return
        assert self._pools is not None
        device = logits.device
        chunks: List[torch.Tensor] = []

        for chunk in range(self.config.chunks):
            starts = [length * chunk // self.config.chunks for length in kv_lengths]
            ends = [
                length * (chunk + 1) // self.config.chunks for length in kv_lengths
            ]
            local_lengths_host = [end - start for start, end in zip(starts, ends)]
            max_local_length = max(local_lengths_host)
            offsets = torch.arange(max_local_length, device=device, dtype=torch.long)
            starts_d = torch.tensor(starts, device=device, dtype=torch.long)
            gather_indices = starts_d.unsqueeze(1) + offsets.unsqueeze(0)
            local_lengths = torch.tensor(
                local_lengths_host, device=device, dtype=torch.int32
            )
            valid = offsets.unsqueeze(0) < local_lengths.long().unsqueeze(1)
            gather_indices = torch.where(valid, gather_indices, starts_d.unsqueeze(1))
            chunk_logits = torch.gather(logits, 1, gather_indices)
            local_topk = select_topk(
                chunk_logits, local_lengths, max_local_length, "main"
            )
            chunks.append(local_topk.to(torch.long) + starts_d.unsqueeze(1))

        pool = torch.stack(chunks, dim=1).to(torch.int32)
        slots = torch.tensor(
            [state.slot for state in states], device=device, dtype=torch.long
        )
        self._pools[slots, 0] = pool
        self._pools[slots, 1] = pool
        if self._graph_active_parity is not None:
            self._graph_active_parity.index_fill_(0, slots, 0)
            assert self._graph_coverage is not None
            coverage_d = torch.tensor(
                kv_lengths, device=device, dtype=torch.long
            ).unsqueeze(1)
            self._graph_coverage.index_copy_(
                0, slots, coverage_d.expand(-1, 2)
            )
        for state, kv_length in zip(states, kv_lengths):
            state.active_parity = 0
            state.coverage[0] = kv_length
            state.coverage[1] = kv_length
            state.build_started = False
            state.build_mask = self.config.full_chunk_mask

    def _begin_build(
        self,
        states: Sequence[_RequestState],
        rows: Sequence[int],
        kv_lengths: Sequence[int],
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        keep_bootstrap_chunks: bool,
    ) -> None:
        if not states:
            return
        target_parities = [1 - state.active_parity for state in states]
        for state, parity, kv_length in zip(states, target_parities, kv_lengths):
            state.coverage[parity] = kv_length
            state.build_started = True
            state.build_mask = (
                self.config.full_chunk_mask if keep_bootstrap_chunks else 0
            )

        if self._graph_coverage is not None:
            device = q_fp8.device
            slots = torch.tensor(
                [state.slot for state in states], device=device, dtype=torch.long
            )
            parities = torch.tensor(target_parities, device=device, dtype=torch.long)
            coverage = torch.tensor(kv_lengths, device=device, dtype=torch.long)
            self._graph_coverage.index_put_((slots, parities), coverage)

        if self.config.q_mode == "fixed":
            assert self._anchor_q is not None
            assert self._anchor_weights is not None
            device = q_fp8.device
            slots = torch.tensor(
                [state.slot for state in states], device=device, dtype=torch.long
            )
            row_indices = torch.tensor(rows, device=device, dtype=torch.long)
            self._anchor_q[slots] = q_fp8.index_select(0, row_indices)
            self._anchor_weights[slots] = weights.index_select(0, row_indices)

    def _schedule_refresh(
        self,
        jobs: Sequence[Tuple[int, _RequestState, int]],
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        block_table: torch.Tensor,
        prepare_candidates: PrepareCandidateFn,
        score_materialized: ScoreMaterializedFn,
    ) -> None:
        if not jobs:
            return
        assert self._pools is not None
        device = q_fp8.device
        main_stream = torch.cuda.current_stream(device)
        refresh_stream = self._refresh_stream(device)
        rows = [row for row, _, _ in jobs]
        states = [state for _, state, _ in jobs]
        chunks = [chunk for _, _, chunk in jobs]
        starts = [
            state.coverage[1 - state.active_parity] * chunk // self.config.chunks
            for state, chunk in zip(states, chunks)
        ]
        ends = [
            state.coverage[1 - state.active_parity]
            * (chunk + 1)
            // self.config.chunks
            for state, chunk in zip(states, chunks)
        ]
        candidate_lengths_host = [
            end - start for start, end in zip(starts, ends)
        ]
        max_candidates = max(candidate_lengths_host)

        # Read logical KV pages on the main stream. A request can finish after
        # this forward and immediately release its pages; the background stream
        # therefore only owns this temporary packed cache, never allocator pages.
        row_indices = torch.tensor(rows, device=device, dtype=torch.long)
        slots = torch.tensor(
            [state.slot for state in states], device=device, dtype=torch.long
        )
        parities = torch.tensor(
            [1 - state.active_parity for state in states],
            device=device,
            dtype=torch.long,
        )
        chunks_d = torch.tensor(chunks, device=device, dtype=torch.long)
        starts_d = torch.tensor(starts, device=device, dtype=torch.long)
        lengths_d = torch.tensor(
            candidate_lengths_host, device=device, dtype=torch.int32
        )
        offsets = torch.arange(max_candidates, device=device, dtype=torch.long)
        candidate_indices = starts_d.unsqueeze(1) + offsets.unsqueeze(0)
        block_jobs = block_table.index_select(0, row_indices)
        (
            candidate_cache,
            candidate_block_table,
            padded_width,
        ) = prepare_candidates(block_jobs, candidate_indices, lengths_d)

        if refresh_stream != main_stream:
            refresh_stream.wait_stream(main_stream)
            for tensor in (
                q_fp8,
                weights,
                row_indices,
                slots,
                parities,
                chunks_d,
                candidate_indices,
                lengths_d,
                candidate_cache,
                candidate_block_table,
                self._pools,
            ):
                tensor.record_stream(refresh_stream)
            if self._anchor_q is not None:
                self._anchor_q.record_stream(refresh_stream)
                assert self._anchor_weights is not None
                self._anchor_weights.record_stream(refresh_stream)

        with torch.cuda.stream(refresh_stream):
            if self.config.q_mode == "fixed":
                assert self._anchor_q is not None
                assert self._anchor_weights is not None
                q_jobs = self._anchor_q.index_select(0, slots)
                weight_jobs = self._anchor_weights.index_select(0, slots)
            else:
                q_jobs = q_fp8.index_select(0, row_indices)
                weight_jobs = weights.index_select(0, row_indices)
            local_topk = score_materialized(
                q_jobs,
                weight_jobs,
                candidate_cache,
                candidate_block_table,
                padded_width,
                candidate_indices,
                lengths_d,
                "refresh",
            )
            self._pools.index_put_(
                (slots, parities, chunks_d),
                local_topk,
                accumulate=False,
            )
            event = torch.cuda.Event()
            event.record(refresh_stream)

        for state, chunk in zip(states, chunks):
            state.build_mask |= 1 << chunk
            state.pending_event = event

    def bootstrap_cuda_graph_exact(
        self,
        logits: torch.Tensor,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        attention_inputs: Any,
        select_topk: SelectTopkFn,
        graph_max_seq_len: int,
        exact_topk: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialize eligible pool rows inside the exact CUDA Graph variant."""
        if bool(getattr(attention_inputs, "indexer_pool_graph_mode", False)) or not bool(
            getattr(attention_inputs, "indexer_pool_bootstrap_graph_mode", False)
        ):
            return
        slots_input = getattr(attention_inputs, "decode_indexer_pool_slot", None)
        kv_lengths = getattr(attention_inputs, "decode_kv_length", None)
        bootstrap_mask = getattr(
            attention_inputs, "decode_indexer_pool_bootstrap_mask", None
        )
        batch_size = logits.shape[0]
        metadata = (slots_input, kv_lengths)
        if any(not isinstance(tensor, torch.Tensor) for tensor in metadata):
            return
        if any(not tensor.is_cuda or tensor.numel() != batch_size for tensor in metadata):
            return

        device = logits.device
        if self._device != device:
            self._reset_runtime(device)
        if self._graph_slot_base is None:
            self._graph_slot_base = batch_size
        if batch_size > self._graph_slot_base:
            raise RuntimeError(
                f"decode indexer exact graph batch {batch_size} exceeds slot base "
                f"{self._graph_slot_base}"
            )
        self._ensure_capacity(
            self._graph_slot_base + batch_size, q_fp8, weights
        )
        if self.config.profile == "APPEND":
            if not isinstance(bootstrap_mask, torch.Tensor):
                bootstrap_mask = torch.ones_like(kv_lengths, dtype=torch.int32)
            self._bootstrap_append_cuda_graph(
                logits,
                exact_topk,
                slots_input,
                kv_lengths,
                bootstrap_mask,
                select_topk,
                graph_max_seq_len,
            )
            return
        assert self._pools is not None
        assert self._graph_active_parity is not None
        assert self._graph_coverage is not None

        rows = torch.arange(batch_size, device=device, dtype=torch.long)
        kv_lengths_l = kv_lengths.to(torch.long)
        slots_l = slots_input.to(torch.long)
        valid_rows = (kv_lengths_l > self.config.min_kv_length) & slots_l.ge(0)
        dummy_slots = rows + self._graph_slot_base
        slots = torch.where(valid_rows, slots_l, dummy_slots).clamp(
            min=0, max=self._capacity - 1
        )

        # Invalid and padded rows still execute the captured fixed-shape TopK,
        # but write only to disjoint dummy slots.
        safe_kv_lengths = torch.where(
            valid_rows,
            kv_lengths_l,
            torch.full_like(kv_lengths_l, self.config.min_kv_length + 1),
        ).clamp(max=graph_max_seq_len)
        chunks = torch.arange(self.config.chunks, device=device, dtype=torch.long)
        job_rows = rows.repeat_interleave(self.config.chunks)
        job_chunks = chunks.repeat(batch_size)
        job_kv_lengths = safe_kv_lengths.repeat_interleave(self.config.chunks)
        starts = torch.div(
            job_kv_lengths * job_chunks,
            self.config.chunks,
            rounding_mode="floor",
        )
        ends = torch.div(
            job_kv_lengths * (job_chunks + 1),
            self.config.chunks,
            rounding_mode="floor",
        )
        local_lengths = (ends - starts).to(torch.int32)
        max_chunk_width = (
            graph_max_seq_len + self.config.chunks - 1
        ) // self.config.chunks
        offsets = torch.arange(max_chunk_width, device=device, dtype=torch.long)
        gather_indices = starts.unsqueeze(1) + offsets.unsqueeze(0)
        gather_indices = gather_indices.clamp(max=logits.shape[1] - 1)
        chunk_logits = torch.gather(
            logits.index_select(0, job_rows), 1, gather_indices
        )
        local_topk = select_topk(
            chunk_logits, local_lengths, max_chunk_width, "main"
        )
        pool = (
            local_topk.to(torch.long) + starts.unsqueeze(1)
        ).reshape(batch_size, self.config.chunks, self.index_topk).to(torch.int32)
        pool_pair = pool.unsqueeze(1).expand(-1, 2, -1, -1)
        self._pools.index_copy_(0, slots, pool_pair)

        coverage = torch.where(valid_rows, kv_lengths_l, torch.zeros_like(kv_lengths_l))
        self._graph_coverage.index_copy_(
            0, slots, coverage.unsqueeze(1).expand(-1, 2)
        )
        self._graph_active_parity.index_copy_(0, slots, torch.zeros_like(slots))
        if self._anchor_q is not None:
            self._anchor_q.index_copy_(0, slots, q_fp8)
            assert self._anchor_weights is not None
            self._anchor_weights.index_copy_(0, slots, weights)

    def _bootstrap_append_cuda_graph(
        self,
        logits: torch.Tensor,
        exact_topk: Optional[torch.Tensor],
        slots_input: torch.Tensor,
        kv_lengths: torch.Tensor,
        bootstrap_mask: torch.Tensor,
        select_topk: SelectTopkFn,
        graph_max_seq_len: int,
    ) -> None:
        """Build a true global Top8K pool from one exact score tensor."""
        from rtp_llm.models_py.triton_kernels.sparse_mla.sparse_indexer_pool_update import (
            initialize_global_pool_rows_inverse_map,
        )

        self._ensure_append_graph_state(graph_max_seq_len)
        assert self._append_pools is not None
        assert self._append_pool_lengths is not None
        assert self._append_inverse_map is not None

        batch_size = logits.shape[0]
        device = logits.device
        rows = torch.arange(batch_size, device=device, dtype=torch.long)
        kv_lengths_l = kv_lengths.to(torch.long)
        slots_l = slots_input.to(torch.long)
        valid_rows = (
            (kv_lengths_l > self.config.min_kv_length)
            & slots_l.ge(0)
            & bootstrap_mask.ne(0)
        )
        dummy_slots = rows + self._graph_slot_base
        slots = torch.where(valid_rows, slots_l, dummy_slots).clamp(
            min=0, max=self._capacity - 1
        )
        safe_lengths = torch.where(
            valid_rows,
            kv_lengths_l,
            torch.ones_like(kv_lengths_l),
        ).clamp(min=1, max=graph_max_seq_len).to(torch.int32)

        first_topk_raw = (
            exact_topk
            if exact_topk is not None
            else select_topk(logits, safe_lengths, graph_max_seq_len, "bootstrap")
        )
        first_topk = torch.where(
            valid_rows.unsqueeze(1), first_topk_raw, torch.zeros_like(first_topk_raw)
        )
        topk_parts = [first_topk.to(torch.int32)]
        logits.scatter_(1, first_topk.to(torch.long), -float("inf"))
        for _ in range(3):
            part_raw = select_topk(
                logits,
                safe_lengths,
                graph_max_seq_len,
                "bootstrap",
            )
            part = torch.where(
                valid_rows.unsqueeze(1), part_raw, torch.zeros_like(part_raw)
            )
            topk_parts.append(part.to(torch.int32))
            logits.scatter_(1, part.to(torch.long), -float("inf"))
        initial_pool = torch.cat(topk_parts, dim=1)

        self._append_pools[:, : self.config.pool_size].index_copy_(
            0, slots, initial_pool
        )
        initial_lengths = torch.where(
            valid_rows,
            torch.full_like(slots, self.config.pool_size, dtype=torch.int32),
            torch.zeros_like(slots, dtype=torch.int32),
        )
        self._append_pool_lengths.index_copy_(0, slots, initial_lengths)
        self._append_inverse_map.index_fill_(0, slots, 0)
        initialize_global_pool_rows_inverse_map(
            self._append_pools,
            self._append_pool_lengths,
            self._append_inverse_map,
            slots,
        )
        if self.config.packed_pool:
            assert self._packed_ready is not None
            assert self._packed_protected is not None
            assert self._packed_cursor is not None
            assert self._packed_base_offsets is not None
            self._packed_ready.index_fill_(0, slots, 0)
            self._packed_protected.index_fill_(0, slots, 0)
            self._packed_cursor.index_fill_(0, slots, 0)
            self._packed_base_offsets.index_fill_(0, slots, 0)

    def _try_compute_append_cuda_graph(
        self,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        block_table: torch.Tensor,
        attention_inputs: Any,
        pool_chunk_score: PoolChunkScoreFn,
        select_topk: SelectTopkFn,
        graph_max_seq_len: int,
        active_mask: Optional[torch.Tensor] = None,
        pool_chunk_topk: Optional[PoolChunkTopkFn] = None,
        scheduled_pool_chunk_score: Optional[ScheduledPoolChunkScoreFn] = None,
        packed_pool_step: Optional[PackedPoolStepFn] = None,
        packed_append_pool_step: Optional[PackedAppendPoolStepFn] = None,
    ) -> torch.Tensor:
        from rtp_llm.models_py.triton_kernels.sparse_mla.sparse_indexer_pool_update import (
            append_global_pool_from_pool_chunk_topk,
            compact_append_pool_if_full,
        )

        batch_size = block_table.shape[0]
        slots_input = getattr(attention_inputs, "decode_indexer_pool_slot", None)
        decode_steps = getattr(attention_inputs, "decode_step", None)
        kv_lengths = getattr(attention_inputs, "decode_kv_length", None)
        metadata = (slots_input, decode_steps, kv_lengths)
        if any(not isinstance(tensor, torch.Tensor) for tensor in metadata):
            raise RuntimeError("decode indexer APPEND graph metadata is missing")
        if any(not tensor.is_cuda or tensor.numel() != batch_size for tensor in metadata):
            raise RuntimeError(
                "decode indexer APPEND graph metadata must be fixed CUDA tensors"
            )

        device = q_fp8.device
        if self._device != device:
            self._reset_runtime(device)
        if self._graph_slot_base is None:
            self._graph_slot_base = batch_size
        if batch_size > self._graph_slot_base:
            raise RuntimeError(
                f"decode indexer APPEND graph batch {batch_size} exceeds slot base "
                f"{self._graph_slot_base}"
            )
        self._ensure_capacity(
            self._graph_slot_base + batch_size, q_fp8, weights
        )
        self._ensure_append_graph_state(graph_max_seq_len)
        assert self._append_pools is not None
        assert self._append_pool_lengths is not None
        assert self._append_inverse_map is not None

        use_materialized_pool = self.config.packed_pool and (
            self.config.packed_pool_update == "REPLACE"
            or batch_size >= self.config.packed_pool_min_batch
        )
        if use_materialized_pool:
            assert self._packed_pool_kv is not None
            assert self._packed_ready is not None
            assert self._packed_protected is not None
            assert self._packed_cursor is not None
            if self.config.packed_pool_update == "APPEND":
                if packed_append_pool_step is None:
                    raise RuntimeError(
                        "materialized APPEND indexer pool callback is missing"
                    )
                assert self._packed_base_offsets is not None
                return packed_append_pool_step(
                    q_fp8,
                    weights,
                    block_table,
                    self._append_pools,
                    self._append_pool_lengths,
                    self._append_inverse_map,
                    self._packed_pool_kv,
                    self._packed_ready,
                    self._packed_base_offsets,
                    slots_input,
                    decode_steps,
                    kv_lengths,
                    self._graph_slot_base,
                    self.config.min_kv_length,
                    self.config.chunks,
                    graph_max_seq_len,
                    active_mask,
                )
            if packed_pool_step is None:
                raise RuntimeError("materialized indexer pool callback is missing")
            return packed_pool_step(
                q_fp8,
                weights,
                block_table,
                self._append_pools,
                self._append_inverse_map,
                self._packed_pool_kv,
                self._packed_ready,
                self._packed_protected,
                self._packed_cursor,
                slots_input,
                decode_steps,
                kv_lengths,
                self._graph_slot_base,
                self.config.min_kv_length,
                self.config.chunks,
                graph_max_seq_len,
                active_mask,
            )

        if pool_chunk_topk is not None and scheduled_pool_chunk_score is not None:
            (
                logits,
                topk_lengths,
                chunk_starts,
                chunk_lengths,
                slots,
                active_mask_i32,
            ) = scheduled_pool_chunk_score(
                q_fp8,
                weights,
                block_table,
                self._append_pools,
                self._append_pool_lengths,
                slots_input,
                decode_steps,
                kv_lengths,
                self._graph_slot_base,
                self.config.min_kv_length,
                self.config.chunks,
                graph_max_seq_len,
                active_mask,
            )
            candidate_width = self.config.max_pool_size + (
                graph_max_seq_len + self.config.chunks - 1
            ) // self.config.chunks
            result = pool_chunk_topk(
                logits,
                topk_lengths,
                candidate_width,
                "main",
                self._append_pools,
                self._append_pool_lengths,
                chunk_starts,
                chunk_lengths,
                self._append_inverse_map,
                slots,
                active_mask_i32,
            )
            if (
                self.config.packed_pool
                and self.config.packed_pool_update == "APPEND"
                and not use_materialized_pool
            ):
                from rtp_llm.models_py.triton_kernels.sparse_mla.packed_indexer_pool import (
                    invalidate_packed_pool_slots,
                )

                assert self._packed_ready is not None
                invalidate_packed_pool_slots(
                    slots, active_mask_i32, self._packed_ready
                )
            return result

        rows = torch.arange(batch_size, device=device, dtype=torch.int32)
        kv_lengths_l = kv_lengths.to(torch.long)
        valid_rows = (kv_lengths_l > self.config.min_kv_length) & slots_input.ge(0)
        if active_mask is not None:
            valid_rows = valid_rows & active_mask.ne(0)
        active_mask_i32 = valid_rows.to(torch.int32)
        dummy_slots = rows + self._graph_slot_base
        slots = torch.where(valid_rows, slots_input, dummy_slots).clamp(
            min=0, max=self._capacity - 1
        )

        # The fused TopK path compacts a full pool after selecting from it. The
        # fallback path retains the original pre-score two-kernel compaction.
        if pool_chunk_topk is None:
            compact_append_pool_if_full(
                self._append_pools,
                self._append_pool_lengths,
                self._append_inverse_map,
                slots,
                active_mask_i32,
            )

        safe_kv_lengths = torch.where(
            valid_rows,
            kv_lengths_l,
            torch.ones_like(kv_lengths_l),
        ).clamp(min=1, max=graph_max_seq_len)
        phase = torch.remainder(
            decode_steps.to(torch.long) - 1,
            self.config.interval,
        )
        starts = torch.div(
            safe_kv_lengths * phase,
            self.config.chunks,
            rounding_mode="floor",
        )
        ends = torch.div(
            safe_kv_lengths * (phase + 1),
            self.config.chunks,
            rounding_mode="floor",
        )
        chunk_lengths = torch.where(
            valid_rows, ends - starts, torch.zeros_like(ends)
        ).to(torch.int32)
        max_chunk_width = (
            graph_max_seq_len + self.config.chunks - 1
        ) // self.config.chunks
        offsets = torch.arange(max_chunk_width, device=device, dtype=torch.long)
        chunk_indices = starts.unsqueeze(1) + offsets.unsqueeze(0)
        chunk_indices = chunk_indices.clamp(max=graph_max_seq_len - 1).to(torch.int32)

        topk_lengths = torch.empty_like(chunk_lengths)
        logits = pool_chunk_score(
            q_fp8,
            weights,
            block_table,
            self._append_pools,
            self._append_pool_lengths,
            chunk_indices,
            chunk_lengths,
            slots,
            topk_lengths,
        )
        candidate_width = self.config.max_pool_size + max_chunk_width
        if pool_chunk_topk is not None:
            return pool_chunk_topk(
                logits,
                topk_lengths,
                candidate_width,
                "main",
                self._append_pools,
                self._append_pool_lengths,
                chunk_indices,
                chunk_lengths,
                self._append_inverse_map,
                slots,
                active_mask_i32,
            )
        local_topk = select_topk(
            logits,
            topk_lengths,
            candidate_width,
            "main",
        )
        return append_global_pool_from_pool_chunk_topk(
            local_topk,
            self._append_pools,
            self._append_pool_lengths,
            chunk_indices,
            chunk_lengths,
            self._append_inverse_map,
            slots,
            active_mask_i32,
        )

    def _try_compute_append_hybrid_cuda_graph(
        self,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        block_table: torch.Tensor,
        attention_inputs: Any,
        exact_score: ExactScoreFn,
        pool_chunk_score: PoolChunkScoreFn,
        select_topk: SelectTopkFn,
        graph_max_seq_len: int,
        pool_chunk_topk: Optional[PoolChunkTopkFn] = None,
        build_bootstrap_pool: bool = True,
        scheduled_pool_chunk_score: Optional[ScheduledPoolChunkScoreFn] = None,
        packed_pool_step: Optional[PackedPoolStepFn] = None,
        packed_append_pool_step: Optional[PackedAppendPoolStepFn] = None,
    ) -> torch.Tensor:
        """Run per-row exact/bootstrap and sparse pool work in one fixed graph."""
        batch_size = block_table.shape[0]
        slots_input = getattr(attention_inputs, "decode_indexer_pool_slot", None)
        kv_lengths = getattr(attention_inputs, "decode_kv_length", None)
        bootstrap_mask = getattr(
            attention_inputs, "decode_indexer_pool_bootstrap_mask", None
        )
        metadata = (slots_input, kv_lengths, bootstrap_mask)
        if any(not isinstance(tensor, torch.Tensor) for tensor in metadata):
            raise RuntimeError("decode indexer APPEND hybrid graph metadata is missing")
        if any(
            not tensor.is_cuda or tensor.numel() != batch_size for tensor in metadata
        ):
            raise RuntimeError(
                "decode indexer APPEND hybrid metadata must be fixed CUDA tensors"
            )

        from rtp_llm.models_py.triton_kernels.sparse_mla.sparse_indexer_score import (
            prepare_append_hybrid_metadata,
        )

        (
            exact_lengths,
            bootstrap_rows_i32,
            sparse_rows_i32,
            sparse_rows_bool,
        ) = prepare_append_hybrid_metadata(
            kv_lengths,
            slots_input,
            bootstrap_mask,
            min_kv_length=self.config.min_kv_length,
            graph_max_seq_len=graph_max_seq_len,
        )

        exact_logits, exact_topk = exact_score(
            q_fp8, weights, block_table, exact_lengths
        )
        if build_bootstrap_pool:
            self._bootstrap_append_cuda_graph(
                exact_logits,
                exact_topk,
                slots_input,
                kv_lengths,
                bootstrap_rows_i32,
                select_topk,
                graph_max_seq_len,
            )
        sparse_topk = self._try_compute_append_cuda_graph(
            q_fp8,
            weights,
            block_table,
            attention_inputs,
            pool_chunk_score,
            select_topk,
            graph_max_seq_len,
            sparse_rows_i32,
            pool_chunk_topk,
            scheduled_pool_chunk_score,
            packed_pool_step,
            packed_append_pool_step,
        )
        return torch.where(sparse_rows_bool.unsqueeze(1), sparse_topk, exact_topk)

    def _try_compute_cuda_graph(
        self,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        block_table: torch.Tensor,
        lengths: torch.Tensor,
        attention_inputs: Any,
        candidate_score: CandidateScoreFn,
        prepare_candidates: PrepareCandidateFn,
        score_materialized: ScoreMaterializedFn,
        graph_max_seq_len: Optional[int],
        pool_chunk_score: Optional[PoolChunkScoreFn] = None,
        select_topk: Optional[SelectTopkFn] = None,
        pool_chunk_topk: Optional[PoolChunkTopkFn] = None,
        scheduled_pool_chunk_score: Optional[ScheduledPoolChunkScoreFn] = None,
        packed_pool_step: Optional[PackedPoolStepFn] = None,
        packed_append_pool_step: Optional[PackedAppendPoolStepFn] = None,
    ) -> torch.Tensor:
        """Emit the fixed-shape steady-state data path during graph capture.

        Request lifecycle and slot assignment stay on the host. The captured
        graph only reads fixed input buffers and mutates per-slot parity,
        coverage, anchors, and double-buffered pools in place.
        """
        batch_size = block_table.shape[0]
        if graph_max_seq_len is None or graph_max_seq_len <= 0:
            raise RuntimeError("decode indexer pool graph requires max sequence length")
        if self.config.profile == "APPEND":
            if pool_chunk_score is None or select_topk is None:
                raise RuntimeError("decode indexer APPEND graph callbacks are missing")
            return self._try_compute_append_cuda_graph(
                q_fp8,
                weights,
                block_table,
                attention_inputs,
                pool_chunk_score,
                select_topk,
                graph_max_seq_len,
                pool_chunk_topk=pool_chunk_topk,
                scheduled_pool_chunk_score=scheduled_pool_chunk_score,
                packed_pool_step=packed_pool_step,
                packed_append_pool_step=packed_append_pool_step,
            )
        slots_input = getattr(attention_inputs, "decode_indexer_pool_slot", None)
        decode_steps = getattr(attention_inputs, "decode_step", None)
        kv_lengths = getattr(attention_inputs, "decode_kv_length", None)
        metadata = (slots_input, decode_steps, kv_lengths)
        if any(not isinstance(tensor, torch.Tensor) for tensor in metadata):
            raise RuntimeError("decode indexer pool graph metadata is missing")
        if any(not tensor.is_cuda or tensor.numel() != batch_size for tensor in metadata):
            raise RuntimeError(
                "decode indexer pool graph metadata must be fixed CUDA tensors"
            )

        device = q_fp8.device
        if self._device != device:
            self._reset_runtime(device)

        if self._graph_slot_base is None:
            # initCapture first runs with max_bs. Reserve a disjoint dummy range
            # for padded rows selected by smaller captured graph capacities.
            self._graph_slot_base = batch_size
        if batch_size > self._graph_slot_base:
            raise RuntimeError(
                f"decode indexer graph batch {batch_size} exceeds slot base {self._graph_slot_base}"
            )
        self._ensure_capacity(
            self._graph_slot_base + batch_size, q_fp8, weights
        )
        assert self._pools is not None
        assert self._graph_active_parity is not None
        assert self._graph_coverage is not None

        rows = torch.arange(batch_size, device=device, dtype=torch.long)
        kv_lengths_l = kv_lengths.to(torch.long)
        valid_rows = kv_lengths_l > self.config.min_kv_length
        dummy_slots = rows + self._graph_slot_base
        slots = torch.where(valid_rows, slots_input.to(torch.long), dummy_slots)
        slots = slots.clamp(min=0, max=self._capacity - 1)
        phase = torch.remainder(decode_steps.to(torch.long), self.config.interval)

        active_parity = self._graph_active_parity.index_select(0, slots)
        should_swap = valid_rows & phase.eq(0)
        active_parity = torch.where(
            should_swap, 1 - active_parity, active_parity
        )
        self._graph_active_parity.index_copy_(0, slots, active_parity)

        active_pool = self._pools[slots, active_parity].reshape(
            batch_size, self.config.pool_size
        )
        active_coverage = self._graph_coverage[slots, active_parity]
        recent_lengths = torch.clamp(
            kv_lengths_l - active_coverage,
            min=0,
            max=self.config.max_recent_tokens,
        )
        recent_offsets = torch.arange(
            self.config.max_recent_tokens, device=device, dtype=torch.long
        )
        recent_indices = active_coverage.unsqueeze(1) + recent_offsets.unsqueeze(0)
        candidates = torch.cat((active_pool.to(torch.long), recent_indices), dim=1)
        candidate_lengths = (
            recent_lengths + self.config.pool_size
        ).to(torch.int32)
        result = candidate_score(
            q_fp8,
            weights,
            block_table,
            candidates,
            candidate_lengths,
            "main",
        )

        target_parity = 1 - active_parity
        old_target_coverage = self._graph_coverage[slots, target_parity]
        is_anchor = valid_rows & phase.eq(self.config.anchor_phase)
        target_coverage = torch.where(
            is_anchor, kv_lengths_l, old_target_coverage
        )
        self._graph_coverage.index_put_(
            (slots, target_parity), target_coverage
        )

        if self.config.q_mode == "fixed":
            assert self._anchor_q is not None
            assert self._anchor_weights is not None
            anchor_mask_q = is_anchor.reshape(batch_size, 1, 1)
            anchor_mask_w = is_anchor.reshape(batch_size, 1)
            old_q = self._anchor_q.index_select(0, slots)
            old_weights = self._anchor_weights.index_select(0, slots)
            self._anchor_q.index_copy_(
                0, slots, torch.where(anchor_mask_q, q_fp8, old_q)
            )
            self._anchor_weights.index_copy_(
                0, slots, torch.where(anchor_mask_w, weights, old_weights)
            )

        jobs_per_row = self.config.chunks_per_step
        job_rows = rows.repeat_interleave(jobs_per_row)
        job_slots = slots.repeat_interleave(jobs_per_row)
        job_parities = target_parity.repeat_interleave(jobs_per_row)
        job_coverage = target_coverage.repeat_interleave(jobs_per_row)
        chunk_offsets = torch.arange(
            jobs_per_row, device=device, dtype=torch.long
        ).repeat(batch_size)
        if self.config.profile == "A":
            job_chunks = phase.repeat_interleave(jobs_per_row)
            refresh_enabled = valid_rows.repeat_interleave(jobs_per_row)
        else:
            first_chunk = (
                phase - self.config.anchor_phase
            ) * self.config.chunks_per_step
            job_chunks = (
                first_chunk.repeat_interleave(jobs_per_row) + chunk_offsets
            ).clamp(min=0, max=self.config.chunks - 1)
            refresh_enabled = (
                valid_rows & phase.ge(self.config.anchor_phase)
            ).repeat_interleave(jobs_per_row)

        starts = torch.div(
            job_coverage * job_chunks,
            self.config.chunks,
            rounding_mode="floor",
        )
        ends = torch.div(
            job_coverage * (job_chunks + 1),
            self.config.chunks,
            rounding_mode="floor",
        )
        refresh_lengths = (ends - starts).to(torch.int32)
        max_chunk_width = (
            graph_max_seq_len + self.config.chunks - 1
        ) // self.config.chunks
        refresh_offsets = torch.arange(
            max_chunk_width, device=device, dtype=torch.long
        )
        refresh_candidates = starts.unsqueeze(1) + refresh_offsets.unsqueeze(0)
        refresh_tables = block_table.index_select(0, job_rows)
        (
            candidate_cache,
            candidate_block_table,
            padded_width,
        ) = prepare_candidates(
            refresh_tables, refresh_candidates, refresh_lengths
        )

        if self.config.q_mode == "fixed":
            assert self._anchor_q is not None
            assert self._anchor_weights is not None
            refresh_q = self._anchor_q.index_select(0, job_slots)
            refresh_weights = self._anchor_weights.index_select(0, job_slots)
        else:
            refresh_q = q_fp8.index_select(0, job_rows)
            refresh_weights = weights.index_select(0, job_rows)
        refreshed_topk = score_materialized(
            refresh_q,
            refresh_weights,
            candidate_cache,
            candidate_block_table,
            padded_width,
            refresh_candidates,
            refresh_lengths,
            "refresh",
        )
        old_chunks = self._pools[job_slots, job_parities, job_chunks]
        refreshed_topk = torch.where(
            refresh_enabled.unsqueeze(1), refreshed_topk, old_chunks
        )
        self._pools.index_put_(
            (job_slots, job_parities, job_chunks),
            refreshed_topk,
            accumulate=False,
        )
        return result

    def try_compute(
        self,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        block_table: torch.Tensor,
        lengths: torch.Tensor,
        attention_inputs: Any,
        exact_score: ExactScoreFn,
        candidate_score: CandidateScoreFn,
        prepare_candidates: PrepareCandidateFn,
        score_materialized: ScoreMaterializedFn,
        select_topk: SelectTopkFn,
        pool_chunk_score: Optional[PoolChunkScoreFn] = None,
        graph_max_seq_len: Optional[int] = None,
        pool_chunk_topk: Optional[PoolChunkTopkFn] = None,
        scheduled_pool_chunk_score: Optional[ScheduledPoolChunkScoreFn] = None,
        packed_pool_step: Optional[PackedPoolStepFn] = None,
        packed_append_pool_step: Optional[PackedAppendPoolStepFn] = None,
        dump_kv_cache_fp8: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        batch_size = block_table.shape[0]
        if bool(getattr(attention_inputs, "is_cuda_graph", False)):
            if bool(getattr(attention_inputs, "is_speculative", False)) or bool(
                getattr(attention_inputs, "is_target_verify", False)
            ):
                return None
            pool_graph_mode = bool(
                getattr(attention_inputs, "indexer_pool_graph_mode", False)
            )
            hybrid_graph_mode = bool(
                getattr(
                    attention_inputs,
                    "indexer_pool_bootstrap_graph_mode",
                    False,
                )
            )
            mixed_graph_mode = bool(
                getattr(attention_inputs, "indexer_pool_mixed_graph_mode", False)
            )
            if self.config.profile == "APPEND" and (
                hybrid_graph_mode or mixed_graph_mode
            ):
                if (
                    graph_max_seq_len is None
                    or graph_max_seq_len <= 0
                    or pool_chunk_score is None
                ):
                    raise RuntimeError(
                        "decode indexer APPEND hybrid graph callbacks are missing"
                    )
                return self._try_compute_append_hybrid_cuda_graph(
                    q_fp8,
                    weights,
                    block_table,
                    attention_inputs,
                    exact_score,
                    pool_chunk_score,
                    select_topk,
                    graph_max_seq_len,
                    pool_chunk_topk,
                    build_bootstrap_pool=hybrid_graph_mode,
                    scheduled_pool_chunk_score=scheduled_pool_chunk_score,
                    packed_pool_step=packed_pool_step,
                    packed_append_pool_step=packed_append_pool_step,
                )
            if not pool_graph_mode:
                return None
            return self._try_compute_cuda_graph(
                q_fp8,
                weights,
                block_table,
                lengths,
                attention_inputs,
                candidate_score,
                prepare_candidates,
                score_materialized,
                graph_max_seq_len,
                pool_chunk_score,
                select_topk,
                pool_chunk_topk,
                scheduled_pool_chunk_score,
                packed_pool_step,
                packed_append_pool_step,
            )
        if self.config.profile == "APPEND":
            return None
        if not self._supported(q_fp8, attention_inputs, batch_size):
            return None

        request_ids = attention_inputs.decode_request_id.tolist()
        decode_steps = attention_inputs.decode_step.tolist()
        kv_lengths = attention_inputs.decode_kv_length.tolist()
        pool_slots_tensor = getattr(
            attention_inputs, "decode_indexer_pool_slot", None
        )
        pool_slots: List[Optional[int]] = [None] * batch_size
        if isinstance(pool_slots_tensor, torch.Tensor):
            if pool_slots_tensor.is_cuda or pool_slots_tensor.numel() != batch_size:
                return None
            pool_slots = [int(slot) for slot in pool_slots_tensor.tolist()]
            if any(slot < 0 for slot in pool_slots):
                return None
        if len(set(request_ids)) != len(request_ids):
            _log_once(
                "decode-indexer-pool-multibeam",
                logging.WARNING,
                "decode indexer pool does not support multiple rows with the same request id",
            )
            return None

        eligible_rows = [
            row
            for row, kv_length in enumerate(kv_lengths)
            if kv_length > self.config.min_kv_length
        ]
        if not eligible_rows:
            return None

        self._tick += 1
        self._evict_stale(q_fp8.device)
        new_request_count = sum(
            1
            for row in eligible_rows
            if int(request_ids[row]) not in self._states
        )
        required_capacity = len(self._states) + new_request_count
        preferred_slots = [
            pool_slots[row]
            for row in eligible_rows
            if pool_slots[row] is not None
        ]
        if preferred_slots:
            required_capacity = max(required_capacity, max(preferred_slots) + 1)
        self._ensure_capacity(required_capacity, q_fp8, weights)

        exact_rows: List[int] = []
        pool_rows: List[int] = []
        row_states: Dict[int, _RequestState] = {}
        bootstrap_rows: set[int] = set()

        for row in range(batch_size):
            request_id = int(request_ids[row])
            decode_step = int(decode_steps[row])
            kv_length = int(kv_lengths[row])
            preferred_slot = pool_slots[row]
            state = self._states.get(request_id)
            if kv_length <= self.config.min_kv_length:
                if state is not None:
                    self._wait_pending(state, q_fp8.device)
                    self._states.pop(request_id)
                    self._free_slots.append(state.slot)
                exact_rows.append(row)
                continue

            if state is None:
                state = self._new_state(
                    request_id, q_fp8.device, preferred_slot
                )
                needs_bootstrap = True
            else:
                slot_changed = (
                    preferred_slot is not None and state.slot != preferred_slot
                )
                if slot_changed:
                    self._wait_pending(state, q_fp8.device)
                    old_slot = state.slot
                    state.slot = self._claim_slot(
                        request_id, q_fp8.device, preferred_slot
                    )
                    if old_slot != state.slot:
                        self._free_slots.append(old_slot)
                    self._reset_state(state, q_fp8.device)
                needs_bootstrap = slot_changed or (
                    decode_step != state.last_step + 1
                    or kv_length != state.last_kv_length + 1
                )
                if needs_bootstrap:
                    self._reset_state(state, q_fp8.device)

            if not needs_bootstrap and not self._swap_or_reset(
                state, decode_step, q_fp8.device
            ):
                needs_bootstrap = True

            if not needs_bootstrap:
                coverage = state.coverage[state.active_parity]
                recent = kv_length - coverage
                if recent < 0 or recent > self.config.max_recent_tokens:
                    self._reset_state(state, q_fp8.device)
                    needs_bootstrap = True

            state.last_seen_tick = self._tick
            row_states[row] = state
            if needs_bootstrap:
                exact_rows.append(row)
                bootstrap_rows.add(row)
            else:
                pool_rows.append(row)

        result = torch.empty(
            (batch_size, self.index_topk),
            dtype=torch.int32,
            device=q_fp8.device,
        )

        dump_dir = os.environ.get("RTP_LLM_DECODE_INDEXER_POOL_DUMP_DIR", "")
        dump_layer = int(os.environ.get("RTP_LLM_DECODE_INDEXER_POOL_DUMP_LAYER", "-1"))
        dump_steps = int(os.environ.get("RTP_LLM_DECODE_INDEXER_POOL_DUMP_STEPS", "0"))
        dump_active = (
            bool(dump_dir)
            and (dump_layer < 0 or self.layer_idx == dump_layer)
            and dump_steps > 0
        )

        def dump_step(
            row: int,
            exact_logits_row: torch.Tensor,
            exact_topk_row: torch.Tensor,
            pool_topk_row: Optional[torch.Tensor] = None,
            candidates_row: Optional[torch.Tensor] = None,
            candidate_length: int = 0,
        ) -> None:
            step = int(decode_steps[row])
            if not dump_active or step < 0 or step >= dump_steps:
                return
            os.makedirs(dump_dir, exist_ok=True)
            request_id = int(request_ids[row])
            path = os.path.join(
                dump_dir,
                f"layer_{self.layer_idx:02d}_request_{request_id}_step_{step:03d}.pt",
            )
            if os.path.exists(path):
                return
            score_length = int(lengths[row])
            payload = {
                "layer_idx": self.layer_idx,
                "request_id": request_id,
                "decode_step": step,
                "kv_length": int(kv_lengths[row]),
                "score_length": score_length,
                "q_fp8": q_fp8[row].detach().cpu(),
                # The per-head Q scale is already folded into these weights.
                "weights": weights[row].detach().to(torch.float32).cpu(),
                "weights_are_q_scale_folded": True,
                "exact_logits": exact_logits_row[:score_length]
                .detach()
                .to(torch.float16)
                .cpu(),
                "exact_topk": exact_topk_row.detach().to(torch.int32).cpu(),
                "pool_topk": (
                    pool_topk_row.detach().to(torch.int32).cpu()
                    if pool_topk_row is not None
                    else torch.empty(0, dtype=torch.int32)
                ),
                "candidates": (
                    candidates_row[:candidate_length].detach().to(torch.int32).cpu()
                    if candidates_row is not None
                    else torch.empty(0, dtype=torch.int32)
                ),
            }
            if step == dump_steps - 1:
                if dump_kv_cache_fp8 is None:
                    raise RuntimeError("indexer K cache is unavailable for the final dump step")
                k_cache, k_block_table = _copy_kernel_k_for_dump(
                    dump_kv_cache_fp8,
                    block_table[row],
                    score_length,
                    self.index_head_dim,
                )
                payload["k_cache"] = k_cache.detach().cpu()
                payload["k_block_table"] = k_block_table.detach().cpu()
                payload["k_page_size"] = int(k_cache.shape[1])
                payload["k_head_dim"] = self.index_head_dim
                payload["k_scale_format"] = "ue8m0_fp32"
                payload["k_layout"] = "deep_gemm_paged_k_then_scale"
            torch.save(payload, path)

        if pool_rows:
            assert self._pools is not None
            rows_d = torch.tensor(pool_rows, device=q_fp8.device, dtype=torch.long)
            pool_states = [row_states[row] for row in pool_rows]
            slots = torch.tensor(
                [state.slot for state in pool_states],
                device=q_fp8.device,
                dtype=torch.long,
            )
            parities = torch.tensor(
                [state.active_parity for state in pool_states],
                device=q_fp8.device,
                dtype=torch.long,
            )
            active_pool = self._pools[slots, parities].reshape(
                len(pool_rows), self.config.pool_size
            )
            coverage = [
                state.coverage[state.active_parity] for state in pool_states
            ]
            recent_lengths = [
                int(kv_lengths[row]) - start
                for row, start in zip(pool_rows, coverage)
            ]
            max_recent = max(recent_lengths)
            if max_recent:
                starts_d = torch.tensor(
                    coverage, device=q_fp8.device, dtype=torch.long
                )
                offsets = torch.arange(
                    max_recent, device=q_fp8.device, dtype=torch.long
                )
                recent_indices = starts_d.unsqueeze(1) + offsets.unsqueeze(0)
                candidates = torch.cat((active_pool.to(torch.long), recent_indices), 1)
            else:
                candidates = active_pool.to(torch.long)
            candidate_lengths = torch.tensor(
                [self.config.pool_size + length for length in recent_lengths],
                device=q_fp8.device,
                dtype=torch.int32,
            )
            pool_q = q_fp8.index_select(0, rows_d)
            pool_weights = weights.index_select(0, rows_d)
            pool_block_table = block_table.index_select(0, rows_d)
            pool_topk = candidate_score(
                pool_q,
                pool_weights,
                pool_block_table,
                candidates,
                candidate_lengths,
                "main",
            )
            if dump_active:
                exact_logits_pool, exact_topk_pool = exact_score(
                    pool_q,
                    pool_weights,
                    pool_block_table,
                    lengths.index_select(0, rows_d),
                )
                for position, row in enumerate(pool_rows):
                    dump_step(
                        row,
                        exact_logits_pool[position],
                        exact_topk_pool[position],
                        pool_topk[position],
                        candidates[position],
                        int(candidate_lengths[position]),
                    )
                # Keep generation on the exact trajectory while pool state and
                # refresh logic continue to run for all 128 diagnostic steps.
                pool_topk = exact_topk_pool
            result.index_copy_(0, rows_d, pool_topk)

        if exact_rows:
            rows_d = torch.tensor(exact_rows, device=q_fp8.device, dtype=torch.long)
            exact_logits, exact_topk = exact_score(
                q_fp8.index_select(0, rows_d),
                weights.index_select(0, rows_d),
                block_table.index_select(0, rows_d),
                lengths.index_select(0, rows_d),
            )
            result.index_copy_(0, rows_d, exact_topk)
            if dump_active:
                for position, row in enumerate(exact_rows):
                    dump_step(
                        row,
                        exact_logits[position],
                        exact_topk[position],
                    )
            bootstrap_positions = [
                position
                for position, row in enumerate(exact_rows)
                if row in bootstrap_rows
            ]
            if bootstrap_positions:
                positions_d = torch.tensor(
                    bootstrap_positions, device=q_fp8.device, dtype=torch.long
                )
                states = [
                    row_states[exact_rows[position]]
                    for position in bootstrap_positions
                ]
                bootstrap_lengths = [
                    int(kv_lengths[exact_rows[position]])
                    for position in bootstrap_positions
                ]
                self._bootstrap(
                    exact_logits.index_select(0, positions_d),
                    states,
                    bootstrap_lengths,
                    select_topk,
                )

        begin_states: List[_RequestState] = []
        begin_rows: List[int] = []
        begin_lengths: List[int] = []
        begin_keep_chunks: List[bool] = []
        for row in eligible_rows:
            state = row_states[row]
            phase = int(decode_steps[row]) % self.config.interval
            if state.build_started:
                continue
            normal_anchor = phase == self.config.anchor_phase
            bootstrap_fallback = (
                row in bootstrap_rows and phase > self.config.anchor_phase
            )
            if normal_anchor or bootstrap_fallback:
                begin_states.append(state)
                begin_rows.append(row)
                begin_lengths.append(int(kv_lengths[row]))
                begin_keep_chunks.append(bootstrap_fallback)

        # Normal anchors all clear the inactive pool. Bootstrap fallbacks retain
        # untouched bootstrap chunks so a request joining mid-cycle stays valid.
        for keep_chunks in (False, True):
            indices = [
                idx
                for idx, keep in enumerate(begin_keep_chunks)
                if keep == keep_chunks
            ]
            self._begin_build(
                [begin_states[idx] for idx in indices],
                [begin_rows[idx] for idx in indices],
                [begin_lengths[idx] for idx in indices],
                q_fp8,
                weights,
                keep_chunks,
            )

        jobs: List[Tuple[int, _RequestState, int]] = []
        for row in eligible_rows:
            state = row_states[row]
            if not state.build_started:
                continue
            for chunk in self.config.refresh_chunks(int(decode_steps[row])):
                jobs.append((row, state, chunk))
        self._schedule_refresh(
            jobs,
            q_fp8,
            weights,
            block_table,
            prepare_candidates,
            score_materialized,
        )

        for row in eligible_rows:
            state = row_states[row]
            state.last_step = int(decode_steps[row])
            state.last_kv_length = int(kv_lengths[row])

        return result
