"""CUDA Graph-safe fixed-size materialized indexer pool kernels."""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


PACKED_POOL_SIZE = 8 * 1024
PACKED_APPEND_POOL_SIZE = 16 * 1024


@triton.jit
def _prepare_packed_metadata_kernel(
    block_table_ptr,
    pool_slots_input_ptr,
    decode_steps_ptr,
    kv_lengths_ptr,
    external_active_ptr,
    ready_ptr,
    rows_out_ptr,
    slots_out_ptr,
    active_out_ptr,
    init_out_ptr,
    pool_lengths_out_ptr,
    packed_block_table_out_ptr,
    chunk_starts_out_ptr,
    chunk_lengths_out_ptr,
    chunk_offsets_out_ptr,
    window_lengths_out_ptr,
    window_block_table_out_ptr,
    block_table_stride_b,
    packed_block_table_stride_b,
    window_block_table_stride_b,
    CAPACITY: tl.constexpr,
    DUMMY_SLOT_BASE: tl.constexpr,
    MIN_KV_LENGTH: tl.constexpr,
    SOURCE_CHUNKS: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    POOL_PAGES: tl.constexpr,
    WINDOW_PAGES: tl.constexpr,
    WINDOW_WIDTH: tl.constexpr,
    USE_EXTERNAL_ACTIVE: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
):
    batch = tl.program_id(0)
    page_offsets = tl.arange(0, BLOCK_PAGES)
    raw_slot = tl.load(pool_slots_input_ptr + batch).to(tl.int32)
    kv_length = tl.load(kv_lengths_ptr + batch).to(tl.int32)
    active = (kv_length > MIN_KV_LENGTH) & (raw_slot >= 0)
    if USE_EXTERNAL_ACTIVE:
        active &= tl.load(external_active_ptr + batch) != 0
    safe_slot = tl.where(active, raw_slot, DUMMY_SLOT_BASE + batch)
    safe_slot = tl.minimum(tl.maximum(safe_slot, 0), CAPACITY - 1)
    initialize = active & (tl.load(ready_ptr + safe_slot) == 0)

    safe_kv_length = tl.where(
        active,
        tl.minimum(tl.maximum(kv_length, 1), MAX_SEQ_LEN),
        1,
    )
    decode_step = tl.load(decode_steps_ptr + batch).to(tl.int32)
    phase = tl.where(active, tl.maximum(decode_step - 1, 0) % SOURCE_CHUNKS, 0)
    chunk_start = (safe_kv_length * phase) // SOURCE_CHUNKS
    chunk_end = (safe_kv_length * (phase + 1)) // SOURCE_CHUNKS
    chunk_length = tl.where(active, chunk_end - chunk_start, 0)
    chunk_offset = chunk_start % PAGE_SIZE
    first_page = chunk_start // PAGE_SIZE

    tl.store(rows_out_ptr + batch, batch)
    tl.store(slots_out_ptr + batch, safe_slot)
    tl.store(active_out_ptr + batch, active.to(tl.int32))
    tl.store(init_out_ptr + batch, initialize.to(tl.int32))
    tl.store(
        pool_lengths_out_ptr + batch,
        tl.where(active, POOL_PAGES * PAGE_SIZE, 1),
    )
    tl.store(chunk_starts_out_ptr + batch, chunk_start)
    tl.store(chunk_lengths_out_ptr + batch, chunk_length)
    tl.store(chunk_offsets_out_ptr + batch, chunk_offset)
    tl.store(
        window_lengths_out_ptr + batch,
        tl.where(active, chunk_length + chunk_offset, 1),
    )

    tl.store(
        packed_block_table_out_ptr
        + batch * packed_block_table_stride_b
        + page_offsets,
        safe_slot * POOL_PAGES + page_offsets,
        mask=page_offsets < POOL_PAGES,
    )
    logical_pages = tl.minimum(
        first_page + page_offsets,
        MAX_BLOCKS - 1,
    )
    physical_pages = tl.load(
        block_table_ptr + batch * block_table_stride_b + logical_pages,
        mask=page_offsets < WINDOW_PAGES,
        other=0,
    )
    tl.store(
        window_block_table_out_ptr
        + batch * window_block_table_stride_b
        + page_offsets,
        physical_pages,
        mask=page_offsets < WINDOW_PAGES,
    )


@triton.jit
def _gather_packed_kv_kernel(
    kv_ptr,
    block_table_ptr,
    source_ids_ptr,
    source_rows_ptr,
    destination_positions_ptr,
    item_counts_ptr,
    pool_slots_ptr,
    active_mask_ptr,
    destination_base_offsets_ptr,
    packed_kv_ptr,
    block_table_stride_b,
    source_ids_stride_b,
    destination_positions_stride_b,
    packed_slot_stride,
    MAX_ITEMS: tl.constexpr,
    BLOCK_ITEMS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    ENTRY_BYTES: tl.constexpr,
    POOL_CAPACITY: tl.constexpr,
    IDENTITY_DESTINATIONS: tl.constexpr,
    USE_DESTINATION_BASE: tl.constexpr,
):
    batch = tl.program_id(0)
    item_block = tl.program_id(1)
    active = tl.load(active_mask_ptr + batch) != 0
    item_offsets = item_block * BLOCK_ITEMS + tl.arange(0, BLOCK_ITEMS)
    item_count = tl.load(item_counts_ptr + batch).to(tl.int32)
    valid = active & (item_offsets < item_count) & (item_offsets < MAX_ITEMS)

    source_row = tl.load(source_rows_ptr + batch).to(tl.int64)
    source_ids = tl.load(
        source_ids_ptr + source_row * source_ids_stride_b + item_offsets,
        mask=valid,
        other=0,
    ).to(tl.int64)
    pool_slot = tl.load(pool_slots_ptr + batch).to(tl.int64)
    if IDENTITY_DESTINATIONS:
        destination_positions = item_offsets
    else:
        destination_positions = tl.load(
            destination_positions_ptr
            + batch * destination_positions_stride_b
            + item_offsets,
            mask=valid,
            other=0,
        ).to(tl.int64)
    if USE_DESTINATION_BASE:
        destination_base = tl.load(
            destination_base_offsets_ptr + pool_slot
        ).to(tl.int64)
        destination_positions = (
            destination_positions + destination_base
        ) % POOL_CAPACITY

    logical_blocks = source_ids // PAGE_SIZE
    page_offsets = source_ids % PAGE_SIZE
    physical_blocks = tl.load(
        block_table_ptr + batch * block_table_stride_b + logical_blocks,
        mask=valid,
        other=0,
    ).to(tl.int64)

    dim_offsets = tl.arange(0, HEAD_DIM)
    source_page_bases = physical_blocks * (PAGE_SIZE * ENTRY_BYTES)
    source_k_ptrs = (
        kv_ptr
        + source_page_bases[:, None]
        + page_offsets[:, None] * HEAD_DIM
        + dim_offsets[None, :]
    )

    destination_page_bases = (
        pool_slot * packed_slot_stride
        + (destination_positions // PAGE_SIZE) * (PAGE_SIZE * ENTRY_BYTES)
    )
    destination_page_offsets = destination_positions % PAGE_SIZE
    destination_k_ptrs = (
        packed_kv_ptr
        + destination_page_bases[:, None]
        + destination_page_offsets[:, None] * HEAD_DIM
        + dim_offsets[None, :]
    )
    k_bytes = tl.load(source_k_ptrs, mask=valid[:, None], other=0)
    tl.store(destination_k_ptrs, k_bytes, mask=valid[:, None])

    scale_offsets = tl.arange(0, 4)
    source_scale_ptrs = (
        kv_ptr
        + source_page_bases[:, None]
        + PAGE_SIZE * HEAD_DIM
        + page_offsets[:, None] * 4
        + scale_offsets[None, :]
    )
    destination_scale_ptrs = (
        packed_kv_ptr
        + destination_page_bases[:, None]
        + PAGE_SIZE * HEAD_DIM
        + destination_page_offsets[:, None] * 4
        + scale_offsets[None, :]
    )
    scale_bytes = tl.load(source_scale_ptrs, mask=valid[:, None], other=0)
    tl.store(destination_scale_ptrs, scale_bytes, mask=valid[:, None])


@triton.jit
def _finish_packed_initialization_kernel(
    pool_slots_ptr,
    active_mask_ptr,
    ready_ptr,
):
    batch = tl.program_id(0)
    if tl.load(active_mask_ptr + batch) != 0:
        slot = tl.load(pool_slots_ptr + batch).to(tl.int64)
        tl.store(ready_ptr + slot, 1)


@triton.jit
def _invalidate_packed_ready_kernel(
    pool_slots_ptr,
    active_mask_ptr,
    ready_ptr,
):
    batch = tl.program_id(0)
    if tl.load(active_mask_ptr + batch) != 0:
        pool_slot = tl.load(pool_slots_ptr + batch).to(tl.int64)
        tl.store(ready_ptr + pool_slot, 0)


@triton.jit
def _initialize_packed_pool_kv_kernel(
    kv_ptr,
    block_table_ptr,
    pool_ids_ptr,
    pool_slots_ptr,
    initialize_mask_ptr,
    pool_lengths_ptr,
    base_offsets_ptr,
    ready_ptr,
    packed_kv_ptr,
    block_table_stride_b,
    pool_ids_stride_b,
    packed_slot_stride,
    MAX_ITEMS: tl.constexpr,
    BLOCK_ITEMS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    ENTRY_BYTES: tl.constexpr,
    POOL_CAPACITY: tl.constexpr,
    USE_DESTINATION_BASE: tl.constexpr,
):
    batch = tl.program_id(0)
    initialize = tl.load(initialize_mask_ptr + batch) != 0
    if not initialize:
        return
    pool_slot = tl.load(pool_slots_ptr + batch).to(tl.int64)
    item_count = tl.load(pool_lengths_ptr + batch).to(tl.int32)
    destination_base = 0
    if USE_DESTINATION_BASE:
        destination_base = tl.load(base_offsets_ptr + pool_slot).to(tl.int64)
    dim_offsets = tl.arange(0, HEAD_DIM)
    scale_offsets = tl.arange(0, 4)
    for item_start in tl.range(0, MAX_ITEMS, BLOCK_ITEMS):
        item_offsets = item_start + tl.arange(0, BLOCK_ITEMS)
        valid = item_offsets < item_count
        source_ids = tl.load(
            pool_ids_ptr
            + pool_slot * pool_ids_stride_b
            + item_offsets,
            mask=valid,
            other=0,
        ).to(tl.int64)
        logical_blocks = source_ids // PAGE_SIZE
        page_offsets = source_ids % PAGE_SIZE
        physical_blocks = tl.load(
            block_table_ptr
            + batch * block_table_stride_b
            + logical_blocks,
            mask=valid,
            other=0,
        ).to(tl.int64)
        destination_positions = (
            item_offsets + destination_base
        ) % POOL_CAPACITY
        source_page_bases = physical_blocks * (PAGE_SIZE * ENTRY_BYTES)
        destination_page_bases = (
            pool_slot * packed_slot_stride
            + (destination_positions // PAGE_SIZE) * (PAGE_SIZE * ENTRY_BYTES)
        )
        destination_page_offsets = destination_positions % PAGE_SIZE
        source_k_ptrs = (
            kv_ptr
            + source_page_bases[:, None]
            + page_offsets[:, None] * HEAD_DIM
            + dim_offsets[None, :]
        )
        destination_k_ptrs = (
            packed_kv_ptr
            + destination_page_bases[:, None]
            + destination_page_offsets[:, None] * HEAD_DIM
            + dim_offsets[None, :]
        )
        k_bytes = tl.load(source_k_ptrs, mask=valid[:, None], other=0)
        tl.store(destination_k_ptrs, k_bytes, mask=valid[:, None])

        source_scale_ptrs = (
            kv_ptr
            + source_page_bases[:, None]
            + PAGE_SIZE * HEAD_DIM
            + page_offsets[:, None] * 4
            + scale_offsets[None, :]
        )
        destination_scale_ptrs = (
            packed_kv_ptr
            + destination_page_bases[:, None]
            + PAGE_SIZE * HEAD_DIM
            + destination_page_offsets[:, None] * 4
            + scale_offsets[None, :]
        )
        scale_bytes = tl.load(
            source_scale_ptrs, mask=valid[:, None], other=0
        )
        tl.store(destination_scale_ptrs, scale_bytes, mask=valid[:, None])
    tl.store(ready_ptr + pool_slot, 1)


@triton.jit
def _gather_packed_kv_persistent_kernel(
    kv_ptr,
    block_table_ptr,
    source_ids_ptr,
    source_rows_ptr,
    destination_positions_ptr,
    item_counts_ptr,
    pool_slots_ptr,
    active_mask_ptr,
    destination_base_offsets_ptr,
    packed_kv_ptr,
    block_table_stride_b,
    source_ids_stride_b,
    destination_positions_stride_b,
    packed_slot_stride,
    MAX_ITEMS: tl.constexpr,
    BLOCK_ITEMS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    ENTRY_BYTES: tl.constexpr,
    POOL_CAPACITY: tl.constexpr,
    CTAS_PER_ROW: tl.constexpr,
    USE_DESTINATION_BASE: tl.constexpr,
):
    batch = tl.program_id(0)
    cta = tl.program_id(1)
    active = tl.load(active_mask_ptr + batch) != 0
    item_count = tl.load(item_counts_ptr + batch).to(tl.int32)
    if not active or item_count <= 0:
        return
    source_row = tl.load(source_rows_ptr + batch).to(tl.int64)
    pool_slot = tl.load(pool_slots_ptr + batch).to(tl.int64)
    destination_base = 0
    if USE_DESTINATION_BASE:
        destination_base = tl.load(
            destination_base_offsets_ptr + pool_slot
        ).to(tl.int64)
    dim_offsets = tl.arange(0, HEAD_DIM)
    scale_offsets = tl.arange(0, 4)
    for item_start in tl.range(
        cta * BLOCK_ITEMS,
        MAX_ITEMS,
        CTAS_PER_ROW * BLOCK_ITEMS,
    ):
        item_offsets = item_start + tl.arange(0, BLOCK_ITEMS)
        valid = item_offsets < item_count
        source_ids = tl.load(
            source_ids_ptr
            + source_row * source_ids_stride_b
            + item_offsets,
            mask=valid,
            other=0,
        ).to(tl.int64)
        destination_positions = tl.load(
            destination_positions_ptr
            + batch * destination_positions_stride_b
            + item_offsets,
            mask=valid,
            other=0,
        ).to(tl.int64)
        destination_positions = (
            destination_positions + destination_base
        ) % POOL_CAPACITY
        logical_blocks = source_ids // PAGE_SIZE
        page_offsets = source_ids % PAGE_SIZE
        physical_blocks = tl.load(
            block_table_ptr
            + batch * block_table_stride_b
            + logical_blocks,
            mask=valid,
            other=0,
        ).to(tl.int64)
        source_page_bases = physical_blocks * (PAGE_SIZE * ENTRY_BYTES)
        destination_page_bases = (
            pool_slot * packed_slot_stride
            + (destination_positions // PAGE_SIZE) * (PAGE_SIZE * ENTRY_BYTES)
        )
        destination_page_offsets = destination_positions % PAGE_SIZE
        source_k_ptrs = (
            kv_ptr
            + source_page_bases[:, None]
            + page_offsets[:, None] * HEAD_DIM
            + dim_offsets[None, :]
        )
        destination_k_ptrs = (
            packed_kv_ptr
            + destination_page_bases[:, None]
            + destination_page_offsets[:, None] * HEAD_DIM
            + dim_offsets[None, :]
        )
        k_bytes = tl.load(source_k_ptrs, mask=valid[:, None], other=0)
        tl.store(destination_k_ptrs, k_bytes, mask=valid[:, None])
        source_scale_ptrs = (
            kv_ptr
            + source_page_bases[:, None]
            + PAGE_SIZE * HEAD_DIM
            + page_offsets[:, None] * 4
            + scale_offsets[None, :]
        )
        destination_scale_ptrs = (
            packed_kv_ptr
            + destination_page_bases[:, None]
            + PAGE_SIZE * HEAD_DIM
            + destination_page_offsets[:, None] * 4
            + scale_offsets[None, :]
        )
        scale_bytes = tl.load(
            source_scale_ptrs, mask=valid[:, None], other=0
        )
        tl.store(destination_scale_ptrs, scale_bytes, mask=valid[:, None])


@triton.jit
def _prepare_packed_append_metadata_kernel(
    block_table_ptr,
    pool_slots_input_ptr,
    decode_steps_ptr,
    kv_lengths_ptr,
    external_active_ptr,
    state_pool_lengths_ptr,
    ready_ptr,
    base_offsets_ptr,
    rows_out_ptr,
    slots_out_ptr,
    active_out_ptr,
    init_out_ptr,
    pool_lengths_out_ptr,
    packed_block_table_out_ptr,
    chunk_starts_out_ptr,
    chunk_lengths_out_ptr,
    chunk_offsets_out_ptr,
    window_lengths_out_ptr,
    window_block_table_out_ptr,
    block_table_stride_b,
    packed_block_table_stride_b,
    window_block_table_stride_b,
    CAPACITY: tl.constexpr,
    DUMMY_SLOT_BASE: tl.constexpr,
    MIN_KV_LENGTH: tl.constexpr,
    SOURCE_CHUNKS: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    POOL_PAGES: tl.constexpr,
    WINDOW_PAGES: tl.constexpr,
    WINDOW_WIDTH: tl.constexpr,
    USE_EXTERNAL_ACTIVE: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
):
    batch = tl.program_id(0)
    page_offsets = tl.arange(0, BLOCK_PAGES)
    raw_slot = tl.load(pool_slots_input_ptr + batch).to(tl.int32)
    kv_length = tl.load(kv_lengths_ptr + batch).to(tl.int32)
    active = (kv_length > MIN_KV_LENGTH) & (raw_slot >= 0)
    if USE_EXTERNAL_ACTIVE:
        active &= tl.load(external_active_ptr + batch) != 0
    safe_slot = tl.where(active, raw_slot, DUMMY_SLOT_BASE + batch)
    safe_slot = tl.minimum(tl.maximum(safe_slot, 0), CAPACITY - 1)
    initialize = active & (tl.load(ready_ptr + safe_slot) == 0)
    state_pool_length = tl.load(state_pool_lengths_ptr + safe_slot).to(tl.int32)
    pool_length = tl.where(
        active,
        tl.minimum(tl.maximum(state_pool_length, 1), POOL_PAGES * PAGE_SIZE),
        1,
    )

    safe_kv_length = tl.where(
        active,
        tl.minimum(tl.maximum(kv_length, 1), MAX_SEQ_LEN),
        1,
    )
    decode_step = tl.load(decode_steps_ptr + batch).to(tl.int32)
    phase = tl.where(active, tl.maximum(decode_step - 1, 0) % SOURCE_CHUNKS, 0)
    chunk_start = (safe_kv_length * phase) // SOURCE_CHUNKS
    chunk_end = (safe_kv_length * (phase + 1)) // SOURCE_CHUNKS
    chunk_length = tl.where(active, chunk_end - chunk_start, 0)
    chunk_offset = chunk_start % PAGE_SIZE
    first_page = chunk_start // PAGE_SIZE

    tl.store(rows_out_ptr + batch, batch)
    tl.store(slots_out_ptr + batch, safe_slot)
    tl.store(active_out_ptr + batch, active.to(tl.int32))
    tl.store(init_out_ptr + batch, initialize.to(tl.int32))
    tl.store(pool_lengths_out_ptr + batch, pool_length)
    tl.store(chunk_starts_out_ptr + batch, chunk_start)
    tl.store(chunk_lengths_out_ptr + batch, chunk_length)
    tl.store(chunk_offsets_out_ptr + batch, chunk_offset)
    tl.store(
        window_lengths_out_ptr + batch,
        tl.where(active, chunk_length + chunk_offset, 1),
    )

    base_offset = tl.load(base_offsets_ptr + safe_slot).to(tl.int32)
    base_page = base_offset // PAGE_SIZE
    physical_pool_pages = (
        base_page + page_offsets
    ) % POOL_PAGES
    tl.store(
        packed_block_table_out_ptr
        + batch * packed_block_table_stride_b
        + page_offsets,
        safe_slot * POOL_PAGES + physical_pool_pages,
        mask=page_offsets < POOL_PAGES,
    )
    logical_pages = tl.minimum(first_page + page_offsets, MAX_BLOCKS - 1)
    physical_pages = tl.load(
        block_table_ptr + batch * block_table_stride_b + logical_pages,
        mask=page_offsets < WINDOW_PAGES,
        other=0,
    )
    tl.store(
        window_block_table_out_ptr
        + batch * window_block_table_stride_b
        + page_offsets,
        physical_pages,
        mask=page_offsets < WINDOW_PAGES,
    )


@triton.jit
def _combine_append_pool_chunk_logits_kernel(
    pool_logits_ptr,
    chunk_logits_ptr,
    pool_ids_ptr,
    pool_slots_ptr,
    pool_lengths_ptr,
    chunk_starts_ptr,
    chunk_lengths_ptr,
    chunk_offsets_ptr,
    active_mask_ptr,
    output_ptr,
    candidate_lengths_ptr,
    pool_logits_stride_b,
    chunk_logits_stride_b,
    pool_ids_stride_b,
    output_stride_b,
    POOL_CAPACITY: tl.constexpr,
    CHUNK_CAPACITY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total_width = POOL_CAPACITY + CHUNK_CAPACITY
    output_mask = offsets < total_width
    active = tl.load(active_mask_ptr + batch) != 0
    pool_slot = tl.load(pool_slots_ptr + batch).to(tl.int64)
    pool_length = tl.load(pool_lengths_ptr + batch).to(tl.int32)
    chunk_start = tl.load(chunk_starts_ptr + batch).to(tl.int64)
    chunk_length = tl.load(chunk_lengths_ptr + batch).to(tl.int32)
    chunk_offset = tl.load(chunk_offsets_ptr + batch).to(tl.int64)
    candidate_length = tl.where(active, pool_length + chunk_length, 1)
    if block == 0:
        tl.store(candidate_lengths_ptr + batch, candidate_length)
    if block * BLOCK_SIZE >= candidate_length:
        return

    from_pool = offsets < pool_length
    pool_positions = tl.minimum(offsets, POOL_CAPACITY - 1)
    pool_ids = tl.load(
        pool_ids_ptr + pool_slot * pool_ids_stride_b + pool_positions,
        mask=output_mask & from_pool & active,
        other=0,
    ).to(tl.int64)
    pool_values = tl.load(
        pool_logits_ptr + batch * pool_logits_stride_b + pool_positions,
        mask=output_mask & from_pool & active,
        other=-float("inf"),
    )
    duplicate = (pool_ids >= chunk_start) & (
        pool_ids < chunk_start + chunk_length
    )

    chunk_positions = offsets - pool_length
    valid_chunk = (
        output_mask
        & ~from_pool
        & active
        & (chunk_positions >= 0)
        & (chunk_positions < chunk_length)
    )
    chunk_values = tl.load(
        chunk_logits_ptr
        + batch * chunk_logits_stride_b
        + chunk_offset
        + chunk_positions,
        mask=valid_chunk,
        other=-float("inf"),
    )
    values = tl.where(
        from_pool,
        tl.where(duplicate, -float("inf"), pool_values),
        chunk_values,
    )
    values = tl.where(
        active, values, tl.where(offsets == 0, 0.0, -float("inf"))
    )
    tl.store(output_ptr + batch * output_stride_b + offsets, values, mask=output_mask)


@triton.jit
def _append_materialized_pool_topk_kernel(
    local_topk_ptr,
    pool_ids_ptr,
    pool_lengths_ptr,
    inverse_map_ptr,
    pool_slots_ptr,
    chunk_starts_ptr,
    active_mask_ptr,
    base_offsets_ptr,
    result_ptr,
    changed_ids_ptr,
    changed_positions_ptr,
    changed_counts_ptr,
    pool_ids_stride_b,
    inverse_map_stride_b,
    POOL_CAPACITY: tl.constexpr,
    KEEP_SIZE: tl.constexpr,
    TOPK: tl.constexpr,
    COMPACT_BLOCK: tl.constexpr,
):
    batch = tl.program_id(0)
    topk_offsets = tl.arange(0, TOPK)
    active = tl.load(active_mask_ptr + batch) != 0
    if not active:
        tl.store(result_ptr + batch * TOPK + topk_offsets, -1)
        tl.store(changed_counts_ptr + batch, 0)
        return

    pool_slot = tl.load(pool_slots_ptr + batch).to(tl.int64)
    old_pool_length = tl.load(pool_lengths_ptr + pool_slot).to(tl.int32)
    local_positions = tl.load(
        local_topk_ptr + batch * TOPK + topk_offsets
    ).to(tl.int32)
    from_pool = (local_positions >= 0) & (local_positions < old_pool_length)
    pool_positions = tl.minimum(tl.maximum(local_positions, 0), POOL_CAPACITY - 1)
    selected_pool_ids = tl.load(
        pool_ids_ptr + pool_slot * pool_ids_stride_b + pool_positions,
        mask=from_pool,
        other=-1,
    ).to(tl.int32)
    chunk_start = tl.load(chunk_starts_ptr + batch).to(tl.int32)
    selected_chunk_ids = chunk_start + local_positions - old_pool_length
    selected_ids = tl.where(from_pool, selected_pool_ids, selected_chunk_ids)
    tl.store(result_ptr + batch * TOPK + topk_offsets, selected_ids)

    full = old_pool_length >= POOL_CAPACITY
    if full:
        for start in tl.static_range(0, KEEP_SIZE, COMPACT_BLOCK):
            offsets = start + tl.arange(0, COMPACT_BLOCK)
            evicted_ids = tl.load(
                pool_ids_ptr + pool_slot * pool_ids_stride_b + offsets
            ).to(tl.int64)
            kept_ids = tl.load(
                pool_ids_ptr
                + pool_slot * pool_ids_stride_b
                + KEEP_SIZE
                + offsets
            ).to(tl.int64)
            tl.store(
                inverse_map_ptr
                + pool_slot * inverse_map_stride_b
                + evicted_ids,
                0,
            )
            tl.store(
                pool_ids_ptr + pool_slot * pool_ids_stride_b + offsets,
                kept_ids.to(tl.int32),
            )
        old_base = tl.load(base_offsets_ptr + pool_slot).to(tl.int32)
        tl.store(
            base_offsets_ptr + pool_slot,
            (old_base + KEEP_SIZE) % POOL_CAPACITY,
        )
    tl.debug_barrier()

    pool_length = tl.where(full, KEEP_SIZE, old_pool_length)
    selected_ids = tl.load(result_ptr + batch * TOPK + topk_offsets).to(tl.int32)
    mapped_slots = tl.load(
        inverse_map_ptr
        + pool_slot * inverse_map_stride_b
        + selected_ids.to(tl.int64)
    ).to(tl.int32)
    is_new = mapped_slots == 0
    new_ranks = tl.cumsum(is_new.to(tl.int32), axis=0) - 1
    append_positions = pool_length + new_ranks
    append = is_new & (append_positions < POOL_CAPACITY)
    tl.store(
        pool_ids_ptr
        + pool_slot * pool_ids_stride_b
        + append_positions,
        selected_ids,
        mask=append,
    )
    tl.store(
        inverse_map_ptr
        + pool_slot * inverse_map_stride_b
        + selected_ids.to(tl.int64),
        1,
        mask=append,
    )
    tl.store(
        changed_ids_ptr + batch * TOPK + new_ranks,
        selected_ids,
        mask=append,
    )
    tl.store(
        changed_positions_ptr + batch * TOPK + new_ranks,
        append_positions,
        mask=append,
    )
    appended = tl.minimum(
        tl.sum(is_new.to(tl.int32), axis=0),
        POOL_CAPACITY - pool_length,
    )
    tl.store(changed_counts_ptr + batch, appended)
    tl.store(pool_lengths_ptr + pool_slot, pool_length + appended)


@triton.jit
def _combine_pool_chunk_logits_kernel(
    pool_logits_ptr,
    chunk_logits_ptr,
    pool_ids_ptr,
    pool_slots_ptr,
    chunk_starts_ptr,
    chunk_lengths_ptr,
    chunk_offsets_ptr,
    active_mask_ptr,
    output_ptr,
    candidate_lengths_ptr,
    pool_logits_stride_b,
    chunk_logits_stride_b,
    pool_ids_stride_b,
    output_stride_b,
    POOL_SIZE: tl.constexpr,
    CHUNK_CAPACITY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total_width = POOL_SIZE + CHUNK_CAPACITY
    output_mask = offsets < total_width
    active = tl.load(active_mask_ptr + batch) != 0
    pool_slot = tl.load(pool_slots_ptr + batch).to(tl.int64)
    chunk_start = tl.load(chunk_starts_ptr + batch).to(tl.int64)
    chunk_length = tl.load(chunk_lengths_ptr + batch).to(tl.int32)
    chunk_offset = tl.load(chunk_offsets_ptr + batch).to(tl.int64)

    from_pool = offsets < POOL_SIZE
    pool_offsets = offsets
    pool_ids = tl.load(
        pool_ids_ptr + pool_slot * pool_ids_stride_b + pool_offsets,
        mask=output_mask & from_pool & active,
        other=0,
    ).to(tl.int64)
    pool_logits = tl.load(
        pool_logits_ptr + batch * pool_logits_stride_b + pool_offsets,
        mask=output_mask & from_pool & active,
        other=-float("inf"),
    )
    duplicate = (pool_ids >= chunk_start) & (
        pool_ids < chunk_start + chunk_length
    )

    chunk_positions = offsets - POOL_SIZE
    valid_chunk = (
        output_mask
        & ~from_pool
        & active
        & (chunk_positions >= 0)
        & (chunk_positions < chunk_length)
    )
    chunk_values = tl.load(
        chunk_logits_ptr
        + batch * chunk_logits_stride_b
        + chunk_offset
        + chunk_positions,
        mask=valid_chunk,
        other=-float("inf"),
    )
    values = tl.where(from_pool, tl.where(duplicate, -float("inf"), pool_logits), chunk_values)
    values = tl.where(active, values, tl.where(offsets == 0, 0.0, -float("inf")))
    tl.store(output_ptr + batch * output_stride_b + offsets, values, mask=output_mask)
    if block == 0:
        tl.store(
            candidate_lengths_ptr + batch,
            tl.where(active, POOL_SIZE + chunk_length, 1),
        )


@triton.jit
def _map_topk_collect_new_kernel(
    local_topk_ptr,
    pool_ids_ptr,
    inverse_map_ptr,
    pool_slots_ptr,
    chunk_starts_ptr,
    epochs_ptr,
    active_mask_ptr,
    protected_ptr,
    new_ids_ptr,
    new_counts_ptr,
    result_ptr,
    pool_ids_stride_b,
    inverse_map_stride_b,
    protected_stride_b,
    POOL_SIZE: tl.constexpr,
    TOPK: tl.constexpr,
):
    batch = tl.program_id(0)
    offsets = tl.arange(0, TOPK)
    active = tl.load(active_mask_ptr + batch) != 0
    pool_slot = tl.load(pool_slots_ptr + batch).to(tl.int64)
    local_positions = tl.load(local_topk_ptr + batch * TOPK + offsets).to(tl.int64)
    from_pool = local_positions < POOL_SIZE
    pool_positions = tl.minimum(local_positions, POOL_SIZE - 1)
    pool_ids = tl.load(
        pool_ids_ptr + pool_slot * pool_ids_stride_b + pool_positions,
        mask=active & from_pool,
        other=0,
    ).to(tl.int32)
    chunk_start = tl.load(chunk_starts_ptr + batch).to(tl.int32)
    epoch = tl.load(epochs_ptr + batch).to(tl.int32) + 1
    chunk_ids = chunk_start + (local_positions - POOL_SIZE).to(tl.int32)
    selected_ids = tl.where(from_pool, pool_ids, chunk_ids)
    mapped_slots = tl.load(
        inverse_map_ptr
        + pool_slot * inverse_map_stride_b
        + selected_ids.to(tl.int64),
        mask=active & ~from_pool,
        other=0,
    ).to(tl.int32)
    protected_positions = tl.where(from_pool, pool_positions, mapped_slots - 1)
    protect = active & (from_pool | (mapped_slots > 0))
    tl.store(
        protected_ptr + pool_slot * protected_stride_b + protected_positions,
        epoch,
        mask=protect,
    )

    is_new = active & ~from_pool & (mapped_slots == 0)
    new_ranks = tl.cumsum(is_new.to(tl.int32), axis=0) - 1
    tl.store(
        new_ids_ptr + batch * TOPK + new_ranks,
        selected_ids,
        mask=is_new,
    )
    tl.store(
        result_ptr + batch * TOPK + offsets,
        tl.where(active, selected_ids, -1),
    )
    tl.store(new_counts_ptr + batch, tl.sum(is_new.to(tl.int32), axis=0))


@triton.jit
def _replace_fixed_pool_kernel(
    pool_ids_ptr,
    inverse_map_ptr,
    protected_ptr,
    new_ids_ptr,
    new_counts_ptr,
    changed_ids_ptr,
    changed_positions_ptr,
    pool_slots_ptr,
    active_mask_ptr,
    epochs_ptr,
    cursor_ptr,
    pool_ids_stride_b,
    inverse_map_stride_b,
    protected_stride_b,
    POOL_SIZE: tl.constexpr,
    TOPK: tl.constexpr,
    VICTIM_WINDOW: tl.constexpr,
):
    batch = tl.program_id(0)
    offsets = tl.arange(0, VICTIM_WINDOW)
    active = tl.load(active_mask_ptr + batch) != 0
    pool_slot = tl.load(pool_slots_ptr + batch).to(tl.int64)
    new_count = tl.load(new_counts_ptr + batch).to(tl.int32)
    cursor = tl.load(cursor_ptr + pool_slot).to(tl.int32)
    epoch = tl.load(epochs_ptr + batch).to(tl.int32) + 1
    positions = (cursor + offsets) % POOL_SIZE
    protected = tl.load(
        protected_ptr + pool_slot * protected_stride_b + positions,
        mask=active,
        other=1,
    ) == epoch
    victim_rank = tl.cumsum((~protected).to(tl.int32), axis=0) - 1
    replace = active & ~protected & (victim_rank < new_count)
    new_ids = tl.load(
        new_ids_ptr + batch * TOPK + victim_rank,
        mask=replace,
        other=0,
    ).to(tl.int32)
    old_ids = tl.load(
        pool_ids_ptr + pool_slot * pool_ids_stride_b + positions,
        mask=replace,
        other=0,
    ).to(tl.int64)
    tl.store(
        inverse_map_ptr + pool_slot * inverse_map_stride_b + old_ids,
        0,
        mask=replace,
    )
    tl.store(
        pool_ids_ptr + pool_slot * pool_ids_stride_b + positions,
        new_ids,
        mask=replace,
    )
    tl.store(
        inverse_map_ptr
        + pool_slot * inverse_map_stride_b
        + new_ids.to(tl.int64),
        positions + 1,
        mask=replace,
    )
    tl.store(
        changed_ids_ptr + batch * TOPK + victim_rank,
        new_ids,
        mask=replace,
    )
    tl.store(
        changed_positions_ptr + batch * TOPK + victim_rank,
        positions,
        mask=replace,
    )
    tl.store(cursor_ptr + pool_slot, (cursor + new_count) % POOL_SIZE)


def gather_packed_pool_kv(
    kv_cache_u8: torch.Tensor,
    block_table: torch.Tensor,
    source_ids: torch.Tensor,
    source_rows: torch.Tensor,
    item_counts: torch.Tensor,
    pool_slots: torch.Tensor,
    active_mask: torch.Tensor,
    packed_kv: torch.Tensor,
    *,
    destination_positions: Optional[torch.Tensor] = None,
    destination_base_offsets: Optional[torch.Tensor] = None,
    pool_capacity: int = PACKED_POOL_SIZE,
) -> None:
    """Copy selected paged FP8 K/scale entries into packed-pool slots."""
    batch = block_table.shape[0]
    metadata = (source_rows, item_counts, pool_slots, active_mask)
    if any(tensor.dtype != torch.int32 or tensor.shape != (batch,) for tensor in metadata):
        raise ValueError("packed gather metadata must be int32 [B]")
    if source_ids.dtype != torch.int32 or source_ids.ndim != 2:
        raise ValueError("source_ids must be rank-2 int32")
    if destination_positions is not None and (
        destination_positions.dtype != torch.int32
        or destination_positions.shape != (batch, source_ids.shape[1])
    ):
        raise ValueError("destination_positions must match source_ids as int32")
    if destination_base_offsets is not None and (
        destination_base_offsets.dtype != torch.int32
        or destination_base_offsets.ndim != 1
    ):
        raise ValueError("destination_base_offsets must be rank-1 int32")
    if kv_cache_u8.dtype != torch.uint8 or packed_kv.dtype != torch.uint8:
        raise ValueError("packed gather requires uint8 KV tensors")
    if block_table.dtype != torch.int32:
        raise ValueError("block_table must be int32")
    page_size = kv_cache_u8.shape[1]
    entry_bytes = kv_cache_u8.shape[-1]
    if (
        page_size != 64
        or entry_bytes != 132
        or packed_kv.shape[-2:] != (1, 132)
        or not packed_kv.is_contiguous()
    ):
        raise ValueError("packed gather currently requires FP8 PAGE64 H128")
    if pool_capacity not in (PACKED_POOL_SIZE, PACKED_APPEND_POOL_SIZE):
        raise ValueError("packed gather pool capacity must be 8192 or 16384")
    if packed_kv.numel() % (pool_capacity * entry_bytes) != 0:
        raise ValueError("packed KV storage does not match its pool capacity")

    max_items = source_ids.shape[1]
    block_items = 16
    packed_slot_stride = pool_capacity * entry_bytes
    _gather_packed_kv_kernel[(batch, triton.cdiv(max_items, block_items))](
        kv_cache_u8,
        block_table,
        source_ids,
        source_rows,
        destination_positions if destination_positions is not None else source_ids,
        item_counts,
        pool_slots,
        active_mask,
        destination_base_offsets
        if destination_base_offsets is not None
        else active_mask,
        packed_kv,
        block_table.stride(0),
        source_ids.stride(0),
        destination_positions.stride(0) if destination_positions is not None else 0,
        packed_slot_stride,
        MAX_ITEMS=max_items,
        BLOCK_ITEMS=block_items,
        HEAD_DIM=128,
        PAGE_SIZE=page_size,
        ENTRY_BYTES=entry_bytes,
        POOL_CAPACITY=pool_capacity,
        IDENTITY_DESTINATIONS=destination_positions is None,
        USE_DESTINATION_BASE=destination_base_offsets is not None,
        num_warps=8,
        num_stages=2,
    )


def invalidate_packed_pool_slots(
    pool_slots: torch.Tensor,
    active_mask: torch.Tensor,
    ready: torch.Tensor,
) -> None:
    """Mark logical pools changed by the sparse route as needing rematerialization."""
    if (
        pool_slots.dtype != torch.int32
        or active_mask.dtype != torch.int32
        or pool_slots.shape != active_mask.shape
        or pool_slots.ndim != 1
        or ready.dtype != torch.int32
        or ready.ndim != 1
    ):
        raise ValueError("packed invalidation requires int32 vector metadata")
    _invalidate_packed_ready_kernel[(pool_slots.numel(),)](
        pool_slots,
        active_mask,
        ready,
        num_warps=1,
        num_stages=1,
    )


def gather_packed_pool_kv_persistent(
    kv_cache_u8: torch.Tensor,
    block_table: torch.Tensor,
    source_ids: torch.Tensor,
    source_rows: torch.Tensor,
    destination_positions: torch.Tensor,
    item_counts: torch.Tensor,
    pool_slots: torch.Tensor,
    active_mask: torch.Tensor,
    packed_kv: torch.Tensor,
    *,
    destination_base_offsets: Optional[torch.Tensor] = None,
    pool_capacity: int = PACKED_POOL_SIZE,
) -> None:
    """Gather a small dynamic delta with one persistent CTA per request."""
    batch = block_table.shape[0]
    metadata = (source_rows, item_counts, pool_slots, active_mask)
    if any(tensor.dtype != torch.int32 or tensor.shape != (batch,) for tensor in metadata):
        raise ValueError("persistent packed gather metadata must be int32 [B]")
    if (
        source_ids.dtype != torch.int32
        or destination_positions.dtype != torch.int32
        or source_ids.shape != destination_positions.shape
        or source_ids.ndim != 2
        or source_ids.shape[0] != batch
    ):
        raise ValueError("persistent packed gather items must be matching int32 matrices")
    if destination_base_offsets is not None and (
        destination_base_offsets.dtype != torch.int32
        or destination_base_offsets.ndim != 1
    ):
        raise ValueError("destination_base_offsets must be rank-1 int32")
    if pool_capacity not in (PACKED_POOL_SIZE, PACKED_APPEND_POOL_SIZE):
        raise ValueError("packed gather pool capacity must be 8192 or 16384")
    ctas_per_row = 16
    _gather_packed_kv_persistent_kernel[(batch, ctas_per_row)](
        kv_cache_u8,
        block_table,
        source_ids,
        source_rows,
        destination_positions,
        item_counts,
        pool_slots,
        active_mask,
        destination_base_offsets
        if destination_base_offsets is not None
        else active_mask,
        packed_kv,
        block_table.stride(0),
        source_ids.stride(0),
        destination_positions.stride(0),
        pool_capacity * kv_cache_u8.shape[-1],
        MAX_ITEMS=source_ids.shape[1],
        BLOCK_ITEMS=16,
        HEAD_DIM=128,
        PAGE_SIZE=64,
        ENTRY_BYTES=132,
        POOL_CAPACITY=pool_capacity,
        CTAS_PER_ROW=ctas_per_row,
        USE_DESTINATION_BASE=destination_base_offsets is not None,
        num_warps=8,
        num_stages=2,
    )


def prepare_packed_pool_metadata(
    block_table: torch.Tensor,
    pool_slots_input: torch.Tensor,
    decode_steps: torch.Tensor,
    kv_lengths: torch.Tensor,
    ready: torch.Tensor,
    *,
    dummy_slot_base: int,
    min_kv_length: int,
    source_chunks: int,
    graph_max_seq_len: int,
    external_active: Optional[torch.Tensor] = None,
) -> tuple[
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
]:
    """Build all fixed-pool and contiguous-window metadata in one launch."""
    batch = block_table.shape[0]
    metadata = (pool_slots_input, decode_steps, kv_lengths)
    if any(tensor.dtype != torch.int32 or tensor.shape != (batch,) for tensor in metadata):
        raise ValueError("packed pool metadata inputs must be int32 [B]")
    if external_active is not None and (
        external_active.dtype != torch.int32
        or external_active.shape != (batch,)
    ):
        raise ValueError("external_active must be int32 [B]")
    if block_table.dtype != torch.int32 or block_table.ndim != 2:
        raise ValueError("block_table must be rank-2 int32")
    if ready.dtype != torch.int32 or ready.ndim != 1:
        raise ValueError("ready must be rank-1 int32")
    pool_pages = PACKED_POOL_SIZE // 64
    chunk_capacity = (graph_max_seq_len + source_chunks - 1) // source_chunks
    window_width = ((chunk_capacity + 126) // 64) * 64
    window_pages = window_width // 64

    rows = torch.empty(batch, dtype=torch.int32, device=block_table.device)
    slots = torch.empty_like(rows)
    active = torch.empty_like(rows)
    initialize = torch.empty_like(rows)
    pool_lengths = torch.empty_like(rows)
    chunk_starts = torch.empty_like(rows)
    chunk_lengths = torch.empty_like(rows)
    chunk_offsets = torch.empty_like(rows)
    window_lengths = torch.empty_like(rows)
    packed_block_table = torch.empty(
        (batch, pool_pages), dtype=torch.int32, device=block_table.device
    )
    window_block_table = torch.empty(
        (batch, window_pages), dtype=torch.int32, device=block_table.device
    )
    block_pages = triton.next_power_of_2(max(pool_pages, window_pages))
    _prepare_packed_metadata_kernel[(batch,)](
        block_table,
        pool_slots_input,
        decode_steps,
        kv_lengths,
        external_active if external_active is not None else kv_lengths,
        ready,
        rows,
        slots,
        active,
        initialize,
        pool_lengths,
        packed_block_table,
        chunk_starts,
        chunk_lengths,
        chunk_offsets,
        window_lengths,
        window_block_table,
        block_table.stride(0),
        packed_block_table.stride(0),
        window_block_table.stride(0),
        CAPACITY=ready.numel(),
        DUMMY_SLOT_BASE=dummy_slot_base,
        MIN_KV_LENGTH=min_kv_length,
        SOURCE_CHUNKS=source_chunks,
        MAX_SEQ_LEN=graph_max_seq_len,
        MAX_BLOCKS=block_table.shape[1],
        PAGE_SIZE=64,
        POOL_PAGES=pool_pages,
        WINDOW_PAGES=window_pages,
        WINDOW_WIDTH=window_width,
        USE_EXTERNAL_ACTIVE=external_active is not None,
        BLOCK_PAGES=block_pages,
        num_warps=4,
        num_stages=1,
    )
    return (
        rows,
        slots,
        active,
        initialize,
        pool_lengths,
        packed_block_table,
        chunk_starts,
        chunk_lengths,
        chunk_offsets,
        window_lengths,
        window_block_table,
    )


def initialize_missing_packed_pool(
    kv_cache_u8: torch.Tensor,
    block_table: torch.Tensor,
    pool_ids: torch.Tensor,
    pool_slots: torch.Tensor,
    initialize_mask: torch.Tensor,
    pool_lengths: torch.Tensor,
    ready: torch.Tensor,
    packed_kv: torch.Tensor,
) -> None:
    _initialize_packed_pool_kv_kernel[(block_table.shape[0],)](
        kv_cache_u8,
        block_table,
        pool_ids,
        pool_slots,
        initialize_mask,
        pool_lengths,
        ready,
        ready,
        packed_kv,
        block_table.stride(0),
        pool_ids.stride(0),
        PACKED_POOL_SIZE * kv_cache_u8.shape[-1],
        MAX_ITEMS=PACKED_POOL_SIZE,
        BLOCK_ITEMS=16,
        HEAD_DIM=128,
        PAGE_SIZE=64,
        ENTRY_BYTES=132,
        POOL_CAPACITY=PACKED_POOL_SIZE,
        USE_DESTINATION_BASE=False,
        num_warps=8,
        num_stages=2,
    )


def combine_pool_chunk_logits(
    pool_logits: torch.Tensor,
    chunk_logits: torch.Tensor,
    pool_ids: torch.Tensor,
    pool_slots: torch.Tensor,
    chunk_starts: torch.Tensor,
    chunk_lengths: torch.Tensor,
    chunk_offsets: torch.Tensor,
    active_mask: torch.Tensor,
    chunk_capacity: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = pool_logits.shape[0]
    output = torch.empty(
        (batch, PACKED_POOL_SIZE + chunk_capacity),
        dtype=torch.float32,
        device=pool_logits.device,
    )
    candidate_lengths = torch.empty(batch, dtype=torch.int32, device=pool_logits.device)
    block_size = 128
    _combine_pool_chunk_logits_kernel[
        (batch, triton.cdiv(output.shape[1], block_size))
    ](
        pool_logits,
        chunk_logits,
        pool_ids,
        pool_slots,
        chunk_starts,
        chunk_lengths,
        chunk_offsets,
        active_mask,
        output,
        candidate_lengths,
        pool_logits.stride(0),
        chunk_logits.stride(0),
        pool_ids.stride(0),
        output.stride(0),
        POOL_SIZE=PACKED_POOL_SIZE,
        CHUNK_CAPACITY=chunk_capacity,
        BLOCK_SIZE=block_size,
        num_warps=4,
        num_stages=1,
    )
    return output, candidate_lengths


def update_fixed_packed_pool(
    local_topk: torch.Tensor,
    pool_ids: torch.Tensor,
    inverse_map: torch.Tensor,
    pool_slots: torch.Tensor,
    chunk_starts: torch.Tensor,
    epochs: torch.Tensor,
    active_mask: torch.Tensor,
    protected: torch.Tensor,
    cursor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map final Top2K and replace cold pool slots with newly selected chunk IDs."""
    batch, topk = local_topk.shape
    if topk != 2048 or local_topk.dtype != torch.int32:
        raise ValueError("local_topk must be int32 [B,2048]")
    metadata = (pool_slots, chunk_starts, epochs, active_mask)
    if any(tensor.dtype != torch.int32 or tensor.shape != (batch,) for tensor in metadata):
        raise ValueError("fixed pool update metadata must be int32 [B]")
    if (
        pool_ids.dtype != torch.int32
        or pool_ids.ndim != 2
        or pool_ids.shape[1] != PACKED_POOL_SIZE
    ):
        raise ValueError("pool_ids must be int32 [slots,8192]")
    if protected.dtype != torch.int32 or protected.shape != pool_ids.shape:
        raise ValueError("protected must be int32 with the same shape as pool_ids")
    if cursor.dtype != torch.int32 or cursor.shape != (pool_ids.shape[0],):
        raise ValueError("cursor must be int32 [slots]")
    result = torch.empty_like(local_topk)
    new_ids = torch.empty_like(local_topk)
    new_counts = torch.empty(batch, dtype=torch.int32, device=local_topk.device)
    changed_ids = torch.empty_like(local_topk)
    changed_positions = torch.empty_like(local_topk)
    _map_topk_collect_new_kernel[(batch,)](
        local_topk,
        pool_ids,
        inverse_map,
        pool_slots,
        chunk_starts,
        epochs,
        active_mask,
        protected,
        new_ids,
        new_counts,
        result,
        pool_ids.stride(0),
        inverse_map.stride(0),
        protected.stride(0),
        POOL_SIZE=PACKED_POOL_SIZE,
        TOPK=topk,
        num_warps=8,
        num_stages=1,
    )
    _replace_fixed_pool_kernel[(batch,)](
        pool_ids,
        inverse_map,
        protected,
        new_ids,
        new_counts,
        changed_ids,
        changed_positions,
        pool_slots,
        active_mask,
        epochs,
        cursor,
        pool_ids.stride(0),
        inverse_map.stride(0),
        protected.stride(0),
        POOL_SIZE=PACKED_POOL_SIZE,
        TOPK=topk,
        VICTIM_WINDOW=4096,
        num_warps=8,
        num_stages=1,
    )
    return result, changed_ids, changed_positions, new_counts


def prepare_packed_append_pool_metadata(
    block_table: torch.Tensor,
    pool_slots_input: torch.Tensor,
    decode_steps: torch.Tensor,
    kv_lengths: torch.Tensor,
    state_pool_lengths: torch.Tensor,
    ready: torch.Tensor,
    base_offsets: torch.Tensor,
    *,
    dummy_slot_base: int,
    min_kv_length: int,
    source_chunks: int,
    graph_max_seq_len: int,
    external_active: Optional[torch.Tensor] = None,
) -> tuple[
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
]:
    """Build dynamic append-pool and contiguous-window metadata."""
    batch = block_table.shape[0]
    metadata = (pool_slots_input, decode_steps, kv_lengths)
    if any(tensor.dtype != torch.int32 or tensor.shape != (batch,) for tensor in metadata):
        raise ValueError("packed append metadata inputs must be int32 [B]")
    state = (state_pool_lengths, ready, base_offsets)
    if any(tensor.dtype != torch.int32 or tensor.ndim != 1 for tensor in state):
        raise ValueError("packed append state must be rank-1 int32")
    if not (state_pool_lengths.shape == ready.shape == base_offsets.shape):
        raise ValueError("packed append state tensors must have matching shapes")
    if external_active is not None and (
        external_active.dtype != torch.int32
        or external_active.shape != (batch,)
    ):
        raise ValueError("external_active must be int32 [B]")
    if block_table.dtype != torch.int32 or block_table.ndim != 2:
        raise ValueError("block_table must be rank-2 int32")

    page_size = 64
    pool_pages = PACKED_APPEND_POOL_SIZE // page_size
    chunk_capacity = (graph_max_seq_len + source_chunks - 1) // source_chunks
    window_width = ((chunk_capacity + 2 * page_size - 2) // page_size) * page_size
    window_pages = window_width // page_size
    rows = torch.empty(batch, dtype=torch.int32, device=block_table.device)
    slots = torch.empty_like(rows)
    active = torch.empty_like(rows)
    initialize = torch.empty_like(rows)
    pool_lengths = torch.empty_like(rows)
    chunk_starts = torch.empty_like(rows)
    chunk_lengths = torch.empty_like(rows)
    chunk_offsets = torch.empty_like(rows)
    window_lengths = torch.empty_like(rows)
    packed_block_table = torch.empty(
        (batch, pool_pages), dtype=torch.int32, device=block_table.device
    )
    window_block_table = torch.empty(
        (batch, window_pages), dtype=torch.int32, device=block_table.device
    )
    block_pages = triton.next_power_of_2(max(pool_pages, window_pages))
    _prepare_packed_append_metadata_kernel[(batch,)](
        block_table,
        pool_slots_input,
        decode_steps,
        kv_lengths,
        external_active if external_active is not None else kv_lengths,
        state_pool_lengths,
        ready,
        base_offsets,
        rows,
        slots,
        active,
        initialize,
        pool_lengths,
        packed_block_table,
        chunk_starts,
        chunk_lengths,
        chunk_offsets,
        window_lengths,
        window_block_table,
        block_table.stride(0),
        packed_block_table.stride(0),
        window_block_table.stride(0),
        CAPACITY=ready.numel(),
        DUMMY_SLOT_BASE=dummy_slot_base,
        MIN_KV_LENGTH=min_kv_length,
        SOURCE_CHUNKS=source_chunks,
        MAX_SEQ_LEN=graph_max_seq_len,
        MAX_BLOCKS=block_table.shape[1],
        PAGE_SIZE=page_size,
        POOL_PAGES=pool_pages,
        WINDOW_PAGES=window_pages,
        WINDOW_WIDTH=window_width,
        USE_EXTERNAL_ACTIVE=external_active is not None,
        BLOCK_PAGES=block_pages,
        num_warps=4,
        num_stages=1,
    )
    return (
        rows,
        slots,
        active,
        initialize,
        pool_lengths,
        packed_block_table,
        chunk_starts,
        chunk_lengths,
        chunk_offsets,
        window_lengths,
        window_block_table,
    )


def initialize_missing_packed_append_pool(
    kv_cache_u8: torch.Tensor,
    block_table: torch.Tensor,
    pool_ids: torch.Tensor,
    pool_slots: torch.Tensor,
    initialize_mask: torch.Tensor,
    pool_lengths: torch.Tensor,
    ready: torch.Tensor,
    base_offsets: torch.Tensor,
    packed_kv: torch.Tensor,
) -> None:
    _initialize_packed_pool_kv_kernel[(block_table.shape[0],)](
        kv_cache_u8,
        block_table,
        pool_ids,
        pool_slots,
        initialize_mask,
        pool_lengths,
        base_offsets,
        ready,
        packed_kv,
        block_table.stride(0),
        pool_ids.stride(0),
        PACKED_APPEND_POOL_SIZE * kv_cache_u8.shape[-1],
        MAX_ITEMS=PACKED_APPEND_POOL_SIZE,
        BLOCK_ITEMS=16,
        HEAD_DIM=128,
        PAGE_SIZE=64,
        ENTRY_BYTES=132,
        POOL_CAPACITY=PACKED_APPEND_POOL_SIZE,
        USE_DESTINATION_BASE=True,
        num_warps=8,
        num_stages=2,
    )


def combine_append_pool_chunk_logits(
    pool_logits: torch.Tensor,
    chunk_logits: torch.Tensor,
    pool_ids: torch.Tensor,
    pool_slots: torch.Tensor,
    pool_lengths: torch.Tensor,
    chunk_starts: torch.Tensor,
    chunk_lengths: torch.Tensor,
    chunk_offsets: torch.Tensor,
    active_mask: torch.Tensor,
    chunk_capacity: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = pool_logits.shape[0]
    output = torch.empty(
        (batch, PACKED_APPEND_POOL_SIZE + chunk_capacity),
        dtype=torch.float32,
        device=pool_logits.device,
    )
    candidate_lengths = torch.empty(batch, dtype=torch.int32, device=pool_logits.device)
    block_size = 128
    _combine_append_pool_chunk_logits_kernel[
        (batch, triton.cdiv(output.shape[1], block_size))
    ](
        pool_logits,
        chunk_logits,
        pool_ids,
        pool_slots,
        pool_lengths,
        chunk_starts,
        chunk_lengths,
        chunk_offsets,
        active_mask,
        output,
        candidate_lengths,
        pool_logits.stride(0),
        chunk_logits.stride(0),
        pool_ids.stride(0),
        output.stride(0),
        POOL_CAPACITY=PACKED_APPEND_POOL_SIZE,
        CHUNK_CAPACITY=chunk_capacity,
        BLOCK_SIZE=block_size,
        num_warps=4,
        num_stages=1,
    )
    return output, candidate_lengths


def update_materialized_append_pool(
    local_topk: torch.Tensor,
    pool_ids: torch.Tensor,
    pool_lengths: torch.Tensor,
    inverse_map: torch.Tensor,
    pool_slots: torch.Tensor,
    chunk_starts: torch.Tensor,
    active_mask: torch.Tensor,
    base_offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply V1 append/compact semantics and return incremental KV writes."""
    batch, topk = local_topk.shape
    if local_topk.dtype != torch.int32 or topk != 2048:
        raise ValueError("local_topk must be int32 [B,2048]")
    metadata = (pool_slots, chunk_starts, active_mask)
    if any(tensor.dtype != torch.int32 or tensor.shape != (batch,) for tensor in metadata):
        raise ValueError("packed append update metadata must be int32 [B]")
    if (
        pool_ids.dtype != torch.int32
        or pool_ids.ndim != 2
        or pool_ids.shape[1] != PACKED_APPEND_POOL_SIZE
    ):
        raise ValueError("pool_ids must be int32 [slots,16384]")
    if inverse_map.dtype != torch.uint8 or inverse_map.ndim != 2:
        raise ValueError("APPEND inverse map must be uint8 [slots,max_seq_len]")
    state = (pool_lengths, base_offsets)
    if any(
        tensor.dtype != torch.int32 or tensor.shape != (pool_ids.shape[0],)
        for tensor in state
    ):
        raise ValueError("packed append state must be int32 [slots]")
    result = torch.empty_like(local_topk)
    changed_ids = torch.empty_like(local_topk)
    changed_positions = torch.empty_like(local_topk)
    changed_counts = torch.empty(batch, dtype=torch.int32, device=local_topk.device)
    _append_materialized_pool_topk_kernel[(batch,)](
        local_topk,
        pool_ids,
        pool_lengths,
        inverse_map,
        pool_slots,
        chunk_starts,
        active_mask,
        base_offsets,
        result,
        changed_ids,
        changed_positions,
        changed_counts,
        pool_ids.stride(0),
        inverse_map.stride(0),
        POOL_CAPACITY=PACKED_APPEND_POOL_SIZE,
        KEEP_SIZE=PACKED_POOL_SIZE,
        TOPK=topk,
        COMPACT_BLOCK=256,
        num_warps=8,
        num_stages=1,
    )
    return result, changed_ids, changed_positions, changed_counts
