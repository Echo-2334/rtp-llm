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
    candidate_lengths_out_ptr,
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
    WRITE_CANDIDATE_LENGTHS: tl.constexpr,
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
    if WRITE_CANDIDATE_LENGTHS:
        if candidate_block == 0:
            tl.store(
                candidate_lengths_out_ptr + batch,
                tl.maximum(candidate_length, 1),
            )
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


@triton.jit
def _append_fp8_mqa_logits_kernel(
    q_ptr,
    weights_ptr,
    kv_ptr,
    block_table_ptr,
    pool_ptr,
    pool_lengths_ptr,
    pool_slots_ptr,
    decode_steps_ptr,
    kv_lengths_ptr,
    external_active_ptr,
    logits_ptr,
    candidate_lengths_out_ptr,
    chunk_starts_out_ptr,
    chunk_lengths_out_ptr,
    safe_slots_out_ptr,
    active_mask_out_ptr,
    q_stride_b,
    q_stride_h,
    weights_stride_b,
    block_table_stride_b,
    pool_stride_b,
    logits_stride_b,
    candidate_count,
    dummy_slot_base,
    BLOCK_N: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    ENTRY_BYTES: tl.constexpr,
    MIN_KV_LENGTH: tl.constexpr,
    SOURCE_CHUNKS: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,
    USE_EXTERNAL_ACTIVE: tl.constexpr,
):
    batch = tl.program_id(0)
    candidate_block = tl.program_id(1)
    raw_slot = tl.load(pool_slots_ptr + batch).to(tl.int32)
    kv_length = tl.load(kv_lengths_ptr + batch).to(tl.int32)
    active = (kv_length > MIN_KV_LENGTH) & (raw_slot >= 0)
    if USE_EXTERNAL_ACTIVE:
        active &= tl.load(external_active_ptr + batch) != 0
    safe_slot = tl.where(active, raw_slot, dummy_slot_base + batch).to(tl.int64)

    safe_kv_length = tl.minimum(tl.maximum(kv_length, 1), MAX_SEQ_LEN)
    decode_step = tl.load(decode_steps_ptr + batch).to(tl.int32)
    phase = tl.maximum(decode_step - 1, 0) % SOURCE_CHUNKS
    chunk_start = (safe_kv_length * phase) // SOURCE_CHUNKS
    chunk_end = (safe_kv_length * (phase + 1)) // SOURCE_CHUNKS
    chunk_length = tl.where(active, chunk_end - chunk_start, 0).to(tl.int32)
    pool_length = tl.load(pool_lengths_ptr + safe_slot).to(tl.int32)
    # Keep one canonical candidate for the K generated at this decode step.
    # It may sit outside the rotating source chunk, and is retained by the
    # fused TopK postprocess regardless of whether TopK selects it.
    candidate_length = tl.where(active, pool_length + chunk_length + 1, 0)

    if candidate_block == 0:
        tl.store(candidate_lengths_out_ptr + batch, tl.maximum(candidate_length, 1))
        tl.store(chunk_starts_out_ptr + batch, chunk_start)
        tl.store(chunk_lengths_out_ptr + batch, chunk_length)
        tl.store(safe_slots_out_ptr + batch, safe_slot.to(tl.int32))
        tl.store(active_mask_out_ptr + batch, active.to(tl.int32))

    candidate_offsets = candidate_block * BLOCK_N + tl.arange(0, BLOCK_N)
    output_mask = candidate_offsets < candidate_count
    if candidate_block * BLOCK_N >= candidate_length:
        tl.store(
            logits_ptr + batch * logits_stride_b + candidate_offsets,
            -float("inf"),
            mask=output_mask,
        )
        return

    candidate_mask = output_mask & (candidate_offsets < candidate_length)
    from_pool = candidate_offsets < pool_length
    pool_ids = tl.load(
        pool_ptr + safe_slot * pool_stride_b + candidate_offsets,
        mask=candidate_mask & from_pool,
        other=0,
    ).to(tl.int32)
    tail_offsets = candidate_offsets - pool_length
    from_chunk = (~from_pool) & (tail_offsets < chunk_length)
    chunk_ids = chunk_start + tail_offsets
    latest_id = safe_kv_length - 1
    logical_slots = tl.where(
        from_pool,
        pool_ids,
        tl.where(from_chunk, chunk_ids, latest_id),
    )
    logical_blocks = logical_slots // PAGE_SIZE
    page_offsets = logical_slots % PAGE_SIZE
    physical_blocks = tl.load(
        block_table_ptr + batch * block_table_stride_b + logical_blocks,
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
    logits *= scale_bits.to(tl.float32, bitcast=True)

    # The source range is already present in the candidate tail. Mask the
    # same logical IDs in the pool so TopK output stays duplicate-free.
    duplicate_pool = from_pool & (
        ((logical_slots >= chunk_start) & (logical_slots < chunk_end))
        | (logical_slots == latest_id)
    )
    duplicate_chunk = from_chunk & (logical_slots == latest_id)
    duplicate = duplicate_pool | duplicate_chunk
    logits = tl.where(candidate_mask & ~duplicate, logits, -float("inf"))
    tl.store(
        logits_ptr + batch * logits_stride_b + candidate_offsets,
        logits,
        mask=output_mask,
    )


@triton.jit
def _append_hybrid_metadata_kernel(
    kv_lengths_ptr,
    pool_slots_ptr,
    bootstrap_mask_ptr,
    exact_lengths_out_ptr,
    bootstrap_out_ptr,
    sparse_out_ptr,
    sparse_bool_out_ptr,
    batch_size,
    MIN_KV_LENGTH: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    rows = tl.arange(0, BLOCK)
    mask = rows < batch_size
    kv_length = tl.load(kv_lengths_ptr + rows, mask=mask, other=1).to(tl.int32)
    pool_slot = tl.load(pool_slots_ptr + rows, mask=mask, other=-1).to(tl.int32)
    bootstrap_requested = (
        tl.load(bootstrap_mask_ptr + rows, mask=mask, other=0) != 0
    )
    eligible = (kv_length > MIN_KV_LENGTH) & (pool_slot >= 0)
    bootstrap = eligible & bootstrap_requested
    sparse = eligible & ~bootstrap
    safe_exact_length = tl.minimum(tl.maximum(kv_length, 1), MAX_SEQ_LEN)
    exact_length = tl.where(sparse, 1, safe_exact_length)
    tl.store(exact_lengths_out_ptr + rows, exact_length, mask=mask)
    tl.store(bootstrap_out_ptr + rows, bootstrap.to(tl.int32), mask=mask)
    tl.store(sparse_out_ptr + rows, sparse.to(tl.int32), mask=mask)
    tl.store(sparse_bool_out_ptr + rows, sparse, mask=mask)


def prepare_append_hybrid_metadata(
    kv_lengths: torch.Tensor,
    pool_slots: torch.Tensor,
    bootstrap_mask: torch.Tensor,
    *,
    min_kv_length: int,
    graph_max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build fixed-graph exact/bootstrap/sparse row metadata in one launch."""
    batch = kv_lengths.numel()
    metadata = (kv_lengths, pool_slots, bootstrap_mask)
    if any(
        tensor.dtype != torch.int32
        or tensor.shape != (batch,)
        or not tensor.is_cuda
        for tensor in metadata
    ):
        raise ValueError("APPEND hybrid metadata must be CUDA int32 [B]")
    if min_kv_length <= 0 or graph_max_seq_len <= 0:
        raise ValueError("invalid APPEND hybrid length configuration")
    exact_lengths = torch.empty_like(kv_lengths)
    bootstrap_rows = torch.empty_like(kv_lengths)
    sparse_rows = torch.empty_like(kv_lengths)
    sparse_rows_bool = torch.empty(batch, dtype=torch.bool, device=kv_lengths.device)
    if batch == 0:
        return exact_lengths, bootstrap_rows, sparse_rows, sparse_rows_bool
    block = max(16, triton.next_power_of_2(batch))
    _append_hybrid_metadata_kernel[(1,)](
        kv_lengths,
        pool_slots,
        bootstrap_mask,
        exact_lengths,
        bootstrap_rows,
        sparse_rows,
        sparse_rows_bool,
        batch,
        MIN_KV_LENGTH=min_kv_length,
        MAX_SEQ_LEN=graph_max_seq_len,
        BLOCK=block,
        num_warps=1,
    )
    return exact_lengths, bootstrap_rows, sparse_rows, sparse_rows_bool


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
        candidate_lengths,
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
        WRITE_CANDIDATE_LENGTHS=False,
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
    candidate_lengths_out: Optional[torch.Tensor] = None,
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
    if candidate_lengths_out is not None and (
        candidate_lengths_out.dtype != torch.int32
        or candidate_lengths_out.shape != (batch,)
        or candidate_lengths_out.device != q_fp8.device
    ):
        raise ValueError("candidate_lengths_out must be int32 [B] on the query device")
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
        candidate_lengths_out
        if candidate_lengths_out is not None
        else chunk_lengths,
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
        WRITE_CANDIDATE_LENGTHS=candidate_lengths_out is not None,
        num_warps=num_warps,
        num_stages=2,
    )
    return logits


def sparse_fp8_mqa_append_logits(
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    kv_cache_u8: torch.Tensor,
    block_table: torch.Tensor,
    pool_indices: torch.Tensor,
    pool_lengths: torch.Tensor,
    pool_slots: torch.Tensor,
    decode_steps: torch.Tensor,
    kv_lengths: torch.Tensor,
    *,
    min_kv_length: int,
    source_chunks: int,
    graph_max_seq_len: int,
    dummy_slot_base: int,
    external_active: Optional[torch.Tensor] = None,
    block_n: int = 64,
    num_warps: int = 8,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Score APPEND pool+range+latest-K candidates and build graph metadata."""
    batch, heads, head_dim = q_fp8.shape
    if q_fp8.dtype != torch.float8_e4m3fn or weights.dtype != torch.float32:
        raise ValueError("APPEND score requires FP8 query and FP32 weights")
    if kv_cache_u8.dtype != torch.uint8 or block_table.dtype != torch.int32:
        raise ValueError("APPEND score requires uint8 KV and int32 block table")
    if pool_indices.dtype != torch.int32 or pool_lengths.dtype != torch.int32:
        raise ValueError("APPEND pool state must be int32")
    metadata = (pool_slots, decode_steps, kv_lengths)
    if any(tensor.dtype != torch.int32 or tensor.shape != (batch,) for tensor in metadata):
        raise ValueError("APPEND graph metadata must be int32 [B]")
    if external_active is not None and (
        external_active.dtype != torch.int32
        or external_active.shape != (batch,)
        or external_active.device != q_fp8.device
    ):
        raise ValueError("APPEND external_active must be int32 [B] on the query device")
    if block_table.shape[0] != batch or weights.shape != (batch, heads):
        raise ValueError("APPEND score batch size mismatch")
    if heads != 32 or head_dim != 128:
        raise ValueError("APPEND score currently requires H=32,D=128")
    if source_chunks <= 0 or graph_max_seq_len <= 0 or min_kv_length <= 0:
        raise ValueError("invalid APPEND schedule configuration")
    if dummy_slot_base < batch or dummy_slot_base + batch > pool_indices.shape[0]:
        raise ValueError("APPEND dummy slots exceed pool capacity")
    if block_n not in (16, 32, 64, 128):
        raise ValueError("block_n must be one of 16,32,64,128")

    chunk_capacity = (graph_max_seq_len + source_chunks - 1) // source_chunks
    candidate_count = pool_indices.shape[1] + chunk_capacity + 1
    logits = torch.empty(
        (batch, candidate_count), dtype=torch.float32, device=q_fp8.device
    )
    candidate_lengths = torch.empty(batch, dtype=torch.int32, device=q_fp8.device)
    chunk_starts = torch.empty_like(candidate_lengths)
    chunk_lengths = torch.empty_like(candidate_lengths)
    safe_slots = torch.empty_like(candidate_lengths)
    active_mask = torch.empty_like(candidate_lengths)
    if batch == 0:
        return (
            logits,
            candidate_lengths,
            chunk_starts,
            chunk_lengths,
            safe_slots,
            active_mask,
        )

    page_size = kv_cache_u8.shape[1]
    entry_bytes = kv_cache_u8.shape[-1]
    if page_size != 64 or entry_bytes != 132:
        raise ValueError(
            f"expected paged KV [*,64,*,132], got {tuple(kv_cache_u8.shape)}"
        )
    _append_fp8_mqa_logits_kernel[
        (batch, triton.cdiv(candidate_count, block_n))
    ](
        q_fp8,
        weights,
        kv_cache_u8,
        block_table,
        pool_indices,
        pool_lengths,
        pool_slots,
        decode_steps,
        kv_lengths,
        external_active if external_active is not None else kv_lengths,
        logits,
        candidate_lengths,
        chunk_starts,
        chunk_lengths,
        safe_slots,
        active_mask,
        q_fp8.stride(0),
        q_fp8.stride(1),
        weights.stride(0),
        block_table.stride(0),
        pool_indices.stride(0),
        logits.stride(0),
        candidate_count,
        dummy_slot_base,
        BLOCK_N=block_n,
        HEADS=heads,
        HEAD_DIM=head_dim,
        PAGE_SIZE=page_size,
        ENTRY_BYTES=entry_bytes,
        MIN_KV_LENGTH=min_kv_length,
        SOURCE_CHUNKS=source_chunks,
        MAX_SEQ_LEN=graph_max_seq_len,
        USE_EXTERNAL_ACTIVE=external_active is not None,
        num_warps=num_warps,
        num_stages=2,
    )
    return (
        logits,
        candidate_lengths,
        chunk_starts,
        chunk_lengths,
        safe_slots,
        active_mask,
    )
