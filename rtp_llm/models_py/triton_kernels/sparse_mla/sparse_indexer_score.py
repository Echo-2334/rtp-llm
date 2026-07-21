"""Fused sparse FP8 indexer score for decode candidate pools."""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_fp8_mqa_logits_kernel(
    q_ptr,
    weights_ptr,
    kv_ptr,
    block_table_ptr,
    candidates_ptr,
    candidate_lengths_ptr,
    candidate_slots_ptr,
    tail_candidates_ptr,
    tail_lengths_ptr,
    exclude_starts_ptr,
    exclude_ends_ptr,
    logits_ptr,
    q_stride_b,
    q_stride_h,
    weights_stride_b,
    block_table_stride_b,
    candidates_stride_b,
    tail_candidates_stride_b,
    logits_stride_b,
    candidate_count,
    BLOCK_N: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    ENTRY_BYTES: tl.constexpr,
    EXCLUDE_RANGE: tl.constexpr,
    DYNAMIC_TAIL: tl.constexpr,
    USE_CANDIDATE_SLOTS: tl.constexpr,
):
    batch = tl.program_id(0)
    candidate_block = tl.program_id(1)
    if USE_CANDIDATE_SLOTS:
        candidate_batch = tl.load(candidate_slots_ptr + batch).to(tl.int64)
    else:
        candidate_batch = batch

    candidate_offsets = candidate_block * BLOCK_N + tl.arange(0, BLOCK_N)
    pool_length = tl.load(candidate_lengths_ptr + candidate_batch).to(tl.int32)
    if DYNAMIC_TAIL:
        tail_length = tl.load(tail_lengths_ptr + batch).to(tl.int32)
        candidate_length = pool_length + tail_length
    else:
        candidate_length = pool_length
    output_mask = candidate_offsets < candidate_count
    if candidate_block * BLOCK_N >= candidate_length:
        tl.store(
            logits_ptr + batch * logits_stride_b + candidate_offsets,
            -float("inf"),
            mask=output_mask,
        )
        return
    candidate_mask = (candidate_offsets < candidate_count) & (
        candidate_offsets < candidate_length
    )
    pool_slots = tl.load(
        candidates_ptr
        + candidate_batch * candidates_stride_b
        + candidate_offsets,
        mask=candidate_mask & (candidate_offsets < pool_length),
        other=0,
    ).to(tl.int32)
    if DYNAMIC_TAIL:
        tail_offsets = candidate_offsets - pool_length
        tail_slots = tl.load(
            tail_candidates_ptr
            + batch * tail_candidates_stride_b
            + tail_offsets,
            mask=candidate_mask & (candidate_offsets >= pool_length),
            other=0,
        ).to(tl.int32)
        logical_slots = tl.where(candidate_offsets < pool_length, pool_slots, tail_slots)
    else:
        logical_slots = pool_slots
    logical_blocks = logical_slots // PAGE_SIZE
    page_offsets = logical_slots % PAGE_SIZE
    physical_blocks = tl.load(
        block_table_ptr
        + batch * block_table_stride_b
        + logical_blocks,
        mask=candidate_mask,
        other=0,
    ).to(tl.int64)

    dim_offsets = tl.arange(0, HEAD_DIM)
    page_bases = physical_blocks * (PAGE_SIZE * ENTRY_BYTES)
    k_byte_ptrs = (
        kv_ptr
        + page_bases[:, None]
        + page_offsets[:, None] * HEAD_DIM
        + dim_offsets[None, :]
    )
    k_bytes = tl.load(k_byte_ptrs, mask=candidate_mask[:, None], other=0)
    k_fp8 = k_bytes.to(tl.float8e4nv, bitcast=True)

    head_offsets = tl.arange(0, HEADS)
    q_ptrs = (
        q_ptr
        + batch * q_stride_b
        + head_offsets[:, None] * q_stride_h
        + dim_offsets[None, :]
    )
    q_fp8 = tl.load(q_ptrs)
    scores = tl.dot(q_fp8, tl.trans(k_fp8), out_dtype=tl.float32)
    scores = tl.maximum(scores, 0.0)
    weights = tl.load(
        weights_ptr + batch * weights_stride_b + head_offsets
    ).to(tl.float32)
    logits = tl.sum(scores * weights[:, None], axis=0)

    scale_byte_offsets = tl.arange(0, 4)
    scale_byte_ptrs = (
        kv_ptr
        + page_bases[:, None]
        + PAGE_SIZE * HEAD_DIM
        + page_offsets[:, None] * 4
        + scale_byte_offsets[None, :]
    )
    scale_bytes = tl.load(
        scale_byte_ptrs, mask=candidate_mask[:, None], other=0
    ).to(tl.int32)
    scale_bits = tl.sum(
        scale_bytes << (scale_byte_offsets[None, :] * 8), axis=1
    )
    scales = scale_bits.to(tl.float32, bitcast=True)
    logits *= scales
    if EXCLUDE_RANGE:
        if DYNAMIC_TAIL:
            exclude_start = tl.load(
                tail_candidates_ptr + batch * tail_candidates_stride_b
            ).to(tl.int32)
            exclude_end = exclude_start + tail_length
        else:
            exclude_start = tl.load(exclude_starts_ptr + batch).to(tl.int32)
            exclude_end = tl.load(exclude_ends_ptr + batch).to(tl.int32)
        excluded = (
            (candidate_offsets < pool_length)
            & (logical_slots >= exclude_start)
            & (logical_slots < exclude_end)
        )
        logits = tl.where(excluded, -float("inf"), logits)
    logits = tl.where(candidate_mask, logits, -float("inf"))

    tl.store(
        logits_ptr + batch * logits_stride_b + candidate_offsets,
        logits,
        mask=output_mask,
    )


def sparse_fp8_mqa_logits(
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    kv_cache_u8: torch.Tensor,
    block_table: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_lengths: torch.Tensor,
    *,
    block_n: int = 64,
    num_warps: int = 8,
    exclude_starts: Optional[torch.Tensor] = None,
    exclude_ends: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Score sparse paged-KV candidates without materializing candidate KV."""
    if q_fp8.dtype != torch.float8_e4m3fn:
        raise ValueError(f"q_fp8 must be float8_e4m3fn, got {q_fp8.dtype}")
    if weights.dtype != torch.float32:
        raise ValueError(f"weights must be float32, got {weights.dtype}")
    if kv_cache_u8.dtype != torch.uint8:
        raise ValueError(f"kv_cache_u8 must be uint8, got {kv_cache_u8.dtype}")
    if block_table.dtype != torch.int32:
        raise ValueError(f"block_table must be int32, got {block_table.dtype}")
    if candidate_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"candidate_indices must be int32 or int64, got {candidate_indices.dtype}"
        )
    if candidate_lengths.dtype != torch.int32:
        raise ValueError(
            f"candidate_lengths must be int32, got {candidate_lengths.dtype}"
        )
    if q_fp8.ndim != 3 or candidate_indices.ndim != 2:
        raise ValueError("expected q_fp8 [B,H,D] and candidate_indices [B,N]")

    batch, heads, head_dim = q_fp8.shape
    if heads != 32 or head_dim != 128:
        raise ValueError(
            f"sparse FP8 score currently supports H=32,D=128, got H={heads},D={head_dim}"
        )
    if weights.shape != (batch, heads):
        raise ValueError(
            f"weights shape must be {(batch, heads)}, got {tuple(weights.shape)}"
        )
    if block_table.shape[0] != batch or candidate_indices.shape[0] != batch:
        raise ValueError("batch size mismatch in sparse FP8 score inputs")
    if candidate_lengths.numel() != batch:
        raise ValueError("candidate_lengths must contain one value per batch row")
    if (exclude_starts is None) != (exclude_ends is None):
        raise ValueError("exclude_starts and exclude_ends must be provided together")
    if exclude_starts is not None:
        if exclude_starts.dtype != torch.int32 or exclude_ends.dtype != torch.int32:
            raise ValueError("exclude bounds must be int32")
        if exclude_starts.numel() != batch or exclude_ends.numel() != batch:
            raise ValueError("exclude bounds must contain one value per batch row")
    if block_n not in (16, 32, 64, 128):
        raise ValueError(f"block_n must be one of 16,32,64,128, got {block_n}")

    candidate_count = candidate_indices.shape[1]
    logits = torch.empty(
        (batch, candidate_count), dtype=torch.float32, device=q_fp8.device
    )
    if batch == 0 or candidate_count == 0:
        return logits

    page_size = kv_cache_u8.shape[1]
    entry_bytes = kv_cache_u8.shape[-1]
    if page_size != 64 or entry_bytes != 132:
        raise ValueError(
            f"expected paged KV [*,64,*,132], got {tuple(kv_cache_u8.shape)}"
        )
    grid = (batch, triton.cdiv(candidate_count, block_n))
    _sparse_fp8_mqa_logits_kernel[grid](
        q_fp8,
        weights,
        kv_cache_u8,
        block_table,
        candidate_indices,
        candidate_lengths,
        candidate_lengths,
        candidate_indices,
        candidate_lengths,
        exclude_starts if exclude_starts is not None else candidate_lengths,
        exclude_ends if exclude_ends is not None else candidate_lengths,
        logits,
        q_fp8.stride(0),
        q_fp8.stride(1),
        weights.stride(0),
        block_table.stride(0),
        candidate_indices.stride(0),
        candidate_indices.stride(0),
        logits.stride(0),
        candidate_count,
        BLOCK_N=block_n,
        HEADS=heads,
        HEAD_DIM=head_dim,
        PAGE_SIZE=page_size,
        ENTRY_BYTES=entry_bytes,
        EXCLUDE_RANGE=exclude_starts is not None,
        DYNAMIC_TAIL=False,
        USE_CANDIDATE_SLOTS=False,
        num_warps=num_warps,
        num_stages=2,
    )
    return logits


def sparse_fp8_mqa_pool_chunk_logits(
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    kv_cache_u8: torch.Tensor,
    block_table: torch.Tensor,
    pool_indices: torch.Tensor,
    pool_lengths: torch.Tensor,
    chunk_indices: torch.Tensor,
    chunk_lengths: torch.Tensor,
    *,
    pool_slots: Optional[torch.Tensor] = None,
    block_n: int = 64,
    num_warps: int = 8,
) -> torch.Tensor:
    """Score a growing pool plus one source chunk in a single sparse launch."""
    batch, heads, head_dim = q_fp8.shape
    if q_fp8.dtype != torch.float8_e4m3fn or weights.dtype != torch.float32:
        raise ValueError("pool+chunk score requires FP8 query and FP32 weights")
    if kv_cache_u8.dtype != torch.uint8 or block_table.dtype != torch.int32:
        raise ValueError("pool+chunk score requires uint8 KV and int32 block table")
    if pool_indices.dtype not in (torch.int32, torch.int64) or chunk_indices.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("pool and chunk indices must be int32 or int64")
    if pool_lengths.dtype != torch.int32 or chunk_lengths.dtype != torch.int32:
        raise ValueError("pool and chunk lengths must be int32")
    if chunk_indices.shape[0] != batch or chunk_lengths.numel() != batch:
        raise ValueError("chunk batch size must match query")
    if pool_slots is None:
        if pool_indices.shape[0] != batch or pool_lengths.numel() != batch:
            raise ValueError("pool batch size must match query when slots are omitted")
    elif pool_slots.dtype not in (torch.int32, torch.int64) or pool_slots.shape != (
        batch,
    ):
        raise ValueError("pool_slots must be int32 or int64 [B]")
    if heads != 32 or head_dim != 128:
        raise ValueError("pool+chunk sparse score currently requires H=32,D=128")
    if block_n not in (16, 32, 64, 128):
        raise ValueError("block_n must be one of 16,32,64,128")

    pool_capacity = pool_indices.shape[1]
    chunk_capacity = chunk_indices.shape[1]
    candidate_count = pool_capacity + chunk_capacity
    logits = torch.empty((batch, candidate_count), dtype=torch.float32, device=q_fp8.device)
    if batch == 0:
        return logits

    page_size = kv_cache_u8.shape[1]
    entry_bytes = kv_cache_u8.shape[-1]
    _sparse_fp8_mqa_logits_kernel[
        (batch, triton.cdiv(candidate_count, block_n))
    ](
        q_fp8,
        weights,
        kv_cache_u8,
        block_table,
        pool_indices,
        pool_lengths,
        pool_slots if pool_slots is not None else pool_lengths,
        chunk_indices,
        chunk_lengths,
        chunk_indices,
        chunk_lengths,
        logits,
        q_fp8.stride(0),
        q_fp8.stride(1),
        weights.stride(0),
        block_table.stride(0),
        pool_indices.stride(0),
        chunk_indices.stride(0),
        logits.stride(0),
        candidate_count,
        BLOCK_N=block_n,
        HEADS=heads,
        HEAD_DIM=head_dim,
        PAGE_SIZE=page_size,
        ENTRY_BYTES=entry_bytes,
        EXCLUDE_RANGE=True,
        DYNAMIC_TAIL=True,
        USE_CANDIDATE_SLOTS=pool_slots is not None,
        num_warps=num_warps,
        num_stages=2,
    )
    return logits
