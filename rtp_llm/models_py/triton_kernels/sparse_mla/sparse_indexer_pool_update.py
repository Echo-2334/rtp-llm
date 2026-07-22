"""CUDA Graph-safe append-only indexer pool kernels."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


INITIAL_POOL_SIZE = 8 * 1024
MAX_POOL_SIZE = 16 * 1024


@triton.jit
def _append_pool_from_pool_chunk_topk_kernel(
    local_topk_ptr,
    pool_slots_ptr,
    active_mask_ptr,
    pool_ptr,
    pool_lengths_ptr,
    chunk_ptr,
    chunk_lengths_ptr,
    inverse_map_ptr,
    result_ptr,
    pool_stride_b,
    chunk_stride_b,
    inverse_map_stride_b,
    POOL_CAPACITY: tl.constexpr,
    TOPK: tl.constexpr,
    USE_POOL_SLOTS: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
):
    batch = tl.program_id(0)
    offsets = tl.arange(0, TOPK)
    if USE_ACTIVE_MASK:
        active = tl.load(active_mask_ptr + batch) != 0
        if not active:
            tl.store(result_ptr + batch * TOPK + offsets, -1)
            return
    if USE_POOL_SLOTS:
        pool_batch = tl.load(pool_slots_ptr + batch).to(tl.int64)
    else:
        pool_batch = batch
    local_positions = tl.load(
        local_topk_ptr + batch * TOPK + offsets
    ).to(tl.int64)
    pool_length = tl.load(pool_lengths_ptr + pool_batch).to(tl.int32)
    from_pool = local_positions < pool_length
    pool_ids = tl.load(
        pool_ptr + pool_batch * pool_stride_b + local_positions,
        mask=from_pool,
        other=0,
    ).to(tl.int32)
    chunk_positions = local_positions - pool_length
    chunk_length = tl.load(chunk_lengths_ptr + batch).to(tl.int32)
    chunk_ids = tl.load(
        chunk_ptr + batch * chunk_stride_b + chunk_positions,
        mask=(chunk_positions >= 0) & (chunk_positions < chunk_length),
        other=0,
    ).to(tl.int32)
    selected_ids = tl.where(from_pool, pool_ids, chunk_ids)
    tl.store(result_ptr + batch * TOPK + offsets, selected_ids)

    mapped_slots = tl.load(
        inverse_map_ptr
        + pool_batch * inverse_map_stride_b
        + selected_ids.to(tl.int64)
    ).to(tl.int32)
    is_new = mapped_slots == 0
    new_ranks = tl.cumsum(is_new.to(tl.int32), axis=0) - 1
    append_slots = pool_length + new_ranks
    append = is_new & (append_slots < POOL_CAPACITY)
    tl.store(
        pool_ptr + pool_batch * pool_stride_b + append_slots,
        selected_ids,
        mask=append,
    )
    tl.store(
        inverse_map_ptr
        + pool_batch * inverse_map_stride_b
        + selected_ids.to(tl.int64),
        append_slots + 1,
        mask=append,
    )
    appended = tl.minimum(
        tl.sum(is_new.to(tl.int32), axis=0),
        POOL_CAPACITY - pool_length,
    )
    tl.store(pool_lengths_ptr + pool_batch, pool_length + appended)


@triton.jit
def _compact_append_pool_kernel(
    pool_slots_ptr,
    active_mask_ptr,
    pool_ptr,
    pool_lengths_ptr,
    inverse_map_ptr,
    pool_stride_b,
    inverse_map_stride_b,
    KEEP_SIZE: tl.constexpr,
    POOL_CAPACITY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_POOL_SLOTS: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
):
    batch = tl.program_id(0)
    block = tl.program_id(1)
    if USE_ACTIVE_MASK:
        active = tl.load(active_mask_ptr + batch) != 0
        if not active:
            return
    if USE_POOL_SLOTS:
        pool_batch = tl.load(pool_slots_ptr + batch).to(tl.int64)
    else:
        pool_batch = batch
    pool_length = tl.load(pool_lengths_ptr + pool_batch).to(tl.int32)
    if pool_length < POOL_CAPACITY:
        return

    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < KEEP_SIZE
    evicted_ids = tl.load(
        pool_ptr + pool_batch * pool_stride_b + offsets,
        mask=valid,
        other=0,
    ).to(tl.int64)
    kept_ids = tl.load(
        pool_ptr + pool_batch * pool_stride_b + KEEP_SIZE + offsets,
        mask=valid,
        other=0,
    ).to(tl.int64)
    tl.store(
        inverse_map_ptr + pool_batch * inverse_map_stride_b + evicted_ids,
        0,
        mask=valid,
    )
    tl.store(
        pool_ptr + pool_batch * pool_stride_b + offsets,
        kept_ids.to(tl.int32),
        mask=valid,
    )
    tl.store(
        inverse_map_ptr + pool_batch * inverse_map_stride_b + kept_ids,
        offsets + 1,
        mask=valid,
    )


@triton.jit
def _finish_pool_compaction_kernel(
    pool_slots_ptr,
    active_mask_ptr,
    pool_lengths_ptr,
    KEEP_SIZE: tl.constexpr,
    POOL_CAPACITY: tl.constexpr,
    USE_POOL_SLOTS: tl.constexpr,
    USE_ACTIVE_MASK: tl.constexpr,
):
    batch = tl.program_id(0)
    if USE_ACTIVE_MASK:
        active = tl.load(active_mask_ptr + batch) != 0
        if not active:
            return
    if USE_POOL_SLOTS:
        pool_batch = tl.load(pool_slots_ptr + batch).to(tl.int64)
    else:
        pool_batch = batch
    pool_length = tl.load(pool_lengths_ptr + pool_batch).to(tl.int32)
    tl.store(
        pool_lengths_ptr + pool_batch,
        tl.where(pool_length >= POOL_CAPACITY, KEEP_SIZE, pool_length),
    )


@triton.jit
def _scatter_pool_inverse_map_kernel(
    pool_slots_ptr,
    pool_ptr,
    inverse_map_ptr,
    pool_lengths_ptr,
    pool_stride_b,
    inverse_map_stride_b,
    POOL_CAPACITY: tl.constexpr,
    USE_POOL_SLOTS: tl.constexpr,
):
    batch = tl.program_id(0)
    if USE_POOL_SLOTS:
        pool_batch = tl.load(pool_slots_ptr + batch).to(tl.int64)
    else:
        pool_batch = batch
    offsets = tl.arange(0, POOL_CAPACITY)
    pool_length = tl.load(pool_lengths_ptr + pool_batch).to(tl.int32)
    valid = offsets < pool_length
    pool_ids = tl.load(
        pool_ptr + pool_batch * pool_stride_b + offsets,
        mask=valid,
        other=0,
    ).to(tl.int64)
    tl.store(
        inverse_map_ptr + pool_batch * inverse_map_stride_b + pool_ids,
        offsets + 1,
        mask=valid,
    )


def initialize_global_pool_inverse_map(
    pool: torch.Tensor,
    max_seq_len: int,
    pool_lengths: torch.Tensor,
) -> torch.Tensor:
    """Build logical-token to pool-slot mapping for an initialized pool."""
    _validate_pool(pool, pool_lengths)
    inverse_map = torch.zeros(
        (pool.shape[0], max_seq_len),
        dtype=torch.int32,
        device=pool.device,
    )
    _scatter_pool_inverse_map_kernel[(pool.shape[0],)](
        pool_lengths,
        pool,
        inverse_map,
        pool_lengths,
        pool.stride(0),
        inverse_map.stride(0),
        POOL_CAPACITY=pool.shape[1],
        USE_POOL_SLOTS=False,
        num_warps=8,
        num_stages=1,
    )
    return inverse_map


def initialize_global_pool_rows_inverse_map(
    pool: torch.Tensor,
    pool_lengths: torch.Tensor,
    inverse_map: torch.Tensor,
    pool_slots: torch.Tensor,
) -> None:
    """Populate membership rows for graph-selected global pool slots."""
    _validate_pool_state(pool, pool_lengths, inverse_map)
    if pool_slots.dtype not in (torch.int32, torch.int64) or pool_slots.ndim != 1:
        raise ValueError("pool_slots must be a rank-1 int32 or int64 tensor")
    _scatter_pool_inverse_map_kernel[(pool_slots.numel(),)](
        pool_slots,
        pool,
        inverse_map,
        pool_lengths,
        pool.stride(0),
        inverse_map.stride(0),
        POOL_CAPACITY=pool.shape[1],
        USE_POOL_SLOTS=True,
        num_warps=8,
        num_stages=1,
    )


def compact_append_pool_if_full(
    pool: torch.Tensor,
    pool_lengths: torch.Tensor,
    inverse_map: torch.Tensor,
    pool_slots: torch.Tensor | None = None,
    active_mask: torch.Tensor | None = None,
) -> None:
    """Retain the newest 8K entries when an append-only pool reaches 16K."""
    _validate_pool_state(pool, pool_lengths, inverse_map)
    block_size = 128
    batch = pool.shape[0] if pool_slots is None else pool_slots.numel()
    if pool_slots is not None and (
        pool_slots.dtype not in (torch.int32, torch.int64) or pool_slots.ndim != 1
    ):
        raise ValueError("pool_slots must be a rank-1 int32 or int64 tensor")
    if active_mask is not None and (
        active_mask.dtype != torch.int32 or active_mask.shape != (batch,)
    ):
        raise ValueError("active_mask must be int32 [B]")
    _compact_append_pool_kernel[
        (batch, triton.cdiv(INITIAL_POOL_SIZE, block_size))
    ](
        pool_slots if pool_slots is not None else pool_lengths,
        active_mask if active_mask is not None else pool_lengths,
        pool,
        pool_lengths,
        inverse_map,
        pool.stride(0),
        inverse_map.stride(0),
        KEEP_SIZE=INITIAL_POOL_SIZE,
        POOL_CAPACITY=pool.shape[1],
        BLOCK_SIZE=block_size,
        USE_POOL_SLOTS=pool_slots is not None,
        USE_ACTIVE_MASK=active_mask is not None,
        num_warps=4,
        num_stages=1,
    )
    # This separate launch is the global synchronization between the compact
    # CTAs and the new length becoming visible to the following score kernel.
    _finish_pool_compaction_kernel[(batch,)](
        pool_slots if pool_slots is not None else pool_lengths,
        active_mask if active_mask is not None else pool_lengths,
        pool_lengths,
        KEEP_SIZE=INITIAL_POOL_SIZE,
        POOL_CAPACITY=pool.shape[1],
        USE_POOL_SLOTS=pool_slots is not None,
        USE_ACTIVE_MASK=active_mask is not None,
        num_warps=1,
        num_stages=1,
    )


def append_global_pool_from_pool_chunk_topk(
    local_topk_indices: torch.Tensor,
    pool: torch.Tensor,
    pool_lengths: torch.Tensor,
    chunk_indices: torch.Tensor,
    chunk_lengths: torch.Tensor,
    inverse_map: torch.Tensor,
    pool_slots: torch.Tensor | None = None,
    active_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map pool+chunk TopK positions and append newly selected chunk tokens."""
    _validate_pool_state(pool, pool_lengths, inverse_map)
    batch, topk = local_topk_indices.shape
    if local_topk_indices.dtype != torch.int32 or topk != 2048:
        raise ValueError("local_topk_indices must be int32 [B,2048]")
    if chunk_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("chunk_indices must be int32 or int64")
    if chunk_indices.ndim != 2 or chunk_indices.shape[0] != batch:
        raise ValueError("chunk_indices must have matching batch size")
    if chunk_lengths.dtype != torch.int32 or chunk_lengths.shape != (batch,):
        raise ValueError("chunk_lengths must be int32 [B]")
    if pool_slots is not None and (
        pool_slots.dtype not in (torch.int32, torch.int64)
        or pool_slots.shape != (batch,)
    ):
        raise ValueError("pool_slots must be int32 or int64 [B]")
    if active_mask is not None and (
        active_mask.dtype != torch.int32 or active_mask.shape != (batch,)
    ):
        raise ValueError("active_mask must be int32 [B]")

    result = torch.empty_like(local_topk_indices)
    _append_pool_from_pool_chunk_topk_kernel[(batch,)](
        local_topk_indices,
        pool_slots if pool_slots is not None else pool_lengths,
        active_mask if active_mask is not None else pool_lengths,
        pool,
        pool_lengths,
        chunk_indices,
        chunk_lengths,
        inverse_map,
        result,
        pool.stride(0),
        chunk_indices.stride(0),
        inverse_map.stride(0),
        POOL_CAPACITY=pool.shape[1],
        TOPK=topk,
        USE_POOL_SLOTS=pool_slots is not None,
        USE_ACTIVE_MASK=active_mask is not None,
        num_warps=8,
        num_stages=1,
    )
    return result


def _validate_pool(pool: torch.Tensor, pool_lengths: torch.Tensor) -> None:
    if pool.dtype != torch.int32 or pool.ndim != 2:
        raise ValueError("pool must be a rank-2 int32 tensor")
    if pool.shape[1] not in (INITIAL_POOL_SIZE, MAX_POOL_SIZE):
        raise ValueError(
            f"pool capacity must be {INITIAL_POOL_SIZE} or {MAX_POOL_SIZE}"
        )
    if pool_lengths.dtype != torch.int32 or pool_lengths.shape != (pool.shape[0],):
        raise ValueError("pool_lengths must be int32 [B]")


def _validate_pool_state(
    pool: torch.Tensor,
    pool_lengths: torch.Tensor,
    inverse_map: torch.Tensor,
) -> None:
    _validate_pool(pool, pool_lengths)
    if inverse_map.dtype != torch.int32 or inverse_map.shape[0] != pool.shape[0]:
        raise ValueError("inverse_map must be int32 with matching batch size")
