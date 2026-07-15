"""Decode-only candidate-pool reuse for the GLM/DSA indexer.

The feature is intentionally eager-only and excludes speculative target verify.
It keeps per-request, per-layer pools because this mode is used when indexcache
cross-layer reuse is disabled.

Enable it with ``RTP_LLM_DECODE_INDEXER_POOL_PROFILE=A`` or ``B``. The optional
``RTP_LLM_DECODE_INDEXER_POOL_Q_MODE`` overrides rolling/fixed anchor mode, and
``RTP_LLM_DECODE_INDEXER_POOL_ASYNC_REFRESH=0`` keeps refresh work on the main
stream for debugging.
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
    interval: int = 8
    pool_size: int = 16 * 1024
    chunks: int = 8
    chunks_per_step: int = 0
    refresh_lead: int = 0
    q_mode: str = "rolling"
    anchor_phase: int = 0
    async_refresh: bool = True
    state_ttl_steps: int = 64

    @property
    def enabled(self) -> bool:
        return self.profile != "OFF"

    @property
    def full_chunk_mask(self) -> int:
        return (1 << self.chunks) - 1

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
        if profile == "A":
            defaults = dict(
                chunks_per_step=1,
                refresh_lead=8,
                q_mode="rolling",
                anchor_phase=0,
            )
        elif profile == "B":
            defaults = dict(
                chunks_per_step=2,
                refresh_lead=4,
                q_mode="fixed",
                anchor_phase=4,
            )
        else:
            raise ValueError(f"{env_name} must be OFF, A, or B, got {profile!r}")

        q_mode = os.environ.get(
            "RTP_LLM_DECODE_INDEXER_POOL_Q_MODE", defaults["q_mode"]
        ).strip().lower()
        if q_mode not in ("rolling", "fixed"):
            raise ValueError(
                "RTP_LLM_DECODE_INDEXER_POOL_Q_MODE must be rolling or fixed"
            )
        config = cls(
            profile=profile,
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
            "N=%d P=%d C=%d",
            config.profile,
            config.q_mode,
            config.async_refresh,
            config.interval,
            config.pool_size,
            config.chunks,
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
    ) -> None:
        if config.pool_size != config.chunks * index_topk:
            raise ValueError(
                "decode indexer pool requires pool_size == chunks * index_topk; "
                f"got {config.pool_size} != {config.chunks} * {index_topk}"
            )
        self.config = config
        self.index_topk = index_topk
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self._states: Dict[int, _RequestState] = {}
        self._free_slots: List[int] = []
        self._capacity = 0
        self._pools: Optional[torch.Tensor] = None
        self._anchor_q: Optional[torch.Tensor] = None
        self._anchor_weights: Optional[torch.Tensor] = None
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
        if bool(getattr(attention_inputs, "is_cuda_graph", False)):
            _log_once(
                "decode-indexer-pool-cuda-graph",
                logging.WARNING,
                "decode indexer pool is eager-only; CUDA graph uses exact indexer topk",
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

        new_pools = torch.empty(
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

        new_anchor_q: Optional[torch.Tensor] = None
        new_anchor_weights: Optional[torch.Tensor] = None
        if self.config.q_mode == "fixed":
            new_anchor_q = torch.empty(
                (
                    new_capacity,
                    self.index_n_heads,
                    self.index_head_dim,
                ),
                dtype=q_fp8.dtype,
                device=device,
            )
            new_anchor_weights = torch.empty(
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
        self._anchor_q = new_anchor_q
        self._anchor_weights = new_anchor_weights
        self._capacity = new_capacity

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

    def _new_state(self, request_id: int) -> _RequestState:
        if not self._free_slots:
            raise RuntimeError("decode indexer pool slot allocator is empty")
        state = _RequestState(request_id=request_id, slot=self._free_slots.pop())
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
    ) -> Optional[torch.Tensor]:
        batch_size = block_table.shape[0]
        if not self._supported(q_fp8, attention_inputs, batch_size):
            return None

        request_ids = attention_inputs.decode_request_id.tolist()
        decode_steps = attention_inputs.decode_step.tolist()
        kv_lengths = attention_inputs.decode_kv_length.tolist()
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
            if kv_length >= self.config.pool_size
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
        self._ensure_capacity(
            len(self._states) + new_request_count, q_fp8, weights
        )

        exact_rows: List[int] = []
        pool_rows: List[int] = []
        row_states: Dict[int, _RequestState] = {}
        bootstrap_rows: set[int] = set()

        for row in range(batch_size):
            request_id = int(request_ids[row])
            decode_step = int(decode_steps[row])
            kv_length = int(kv_lengths[row])
            state = self._states.get(request_id)
            if kv_length < self.config.pool_size:
                if state is not None:
                    self._wait_pending(state, q_fp8.device)
                    self._states.pop(request_id)
                    self._free_slots.append(state.slot)
                exact_rows.append(row)
                continue

            if state is None:
                state = self._new_state(request_id)
                needs_bootstrap = True
            else:
                needs_bootstrap = (
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
                if recent < 0 or recent > self.config.interval:
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
            pool_topk = candidate_score(
                q_fp8.index_select(0, rows_d),
                weights.index_select(0, rows_d),
                block_table.index_select(0, rows_d),
                candidates,
                candidate_lengths,
                "main",
            )
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
