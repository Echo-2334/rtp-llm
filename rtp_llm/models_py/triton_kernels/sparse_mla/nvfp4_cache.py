"""MLA NVFP4: 256B E2M1, 32B E4M3, 4B FP32, 128B BF16, 12B metadata.

The FP32 scale is per token; E4M3 scales cover 16 latent elements each.
Clustered layouts append stripes of the block center table to each row.
All metadata travels with the complete physical block during PD transfer.
"""

import torch
import triton
import triton.language as tl
from rtp_llm.models_py.triton_kernels.sparse_mla.nvfp4_cluster import (
    _center_offsets,
    initialize_block_centers,
)

CACHE_BYTES = 432


def is_supported(latent: torch.Tensor, rope: torch.Tensor) -> bool:
    return (
        latent.is_cuda
        and rope.is_cuda
        and latent.dtype == rope.dtype == torch.bfloat16
        and latent.shape[-1] == 512
        and rope.shape[-1] == 64
        and latent.device == rope.device
        and torch.cuda.get_device_capability(latent.device)[0] >= 9
    )


@triton.jit
def _encode_fp4(values):
    magnitude = tl.abs(values)
    code = tl.where(magnitude > 0.25, 1, 0)
    code = tl.where(magnitude >= 0.75, 2, code)
    code = tl.where(magnitude > 1.25, 3, code)
    code = tl.where(magnitude >= 1.75, 4, code)
    code = tl.where(magnitude > 2.5, 5, code)
    code = tl.where(magnitude >= 3.5, 6, code)
    code = tl.where(magnitude > 5.0, 7, code)
    return (code | tl.where(values < 0, 8, 0)).to(tl.uint8)


@triton.jit
def _decode_fp4(code):
    magnitude = code & 7
    value = tl.where(
        magnitude < 4,
        magnitude.to(tl.float32) * 0.5,
        tl.where(
            magnitude == 4,
            2.0,
            tl.where(magnitude == 5, 3.0, tl.where(magnitude == 6, 4.0, 6.0)),
        ),
    )
    return tl.where((code & 8) != 0, -value, value)


@triton.jit
def _store_latent(
    Cache, slot, values, ROW_BYTES: tl.constexpr, CENTERS: tl.constexpr, CLUSTER, tag
):
    groups = tl.arange(0, 32)
    lanes = tl.arange(0, 16)
    values = tl.reshape(values, (32, 16)).to(tl.float32)
    center_id = tl.full((), 255, tl.int32)
    if CENTERS:
        if CLUSTER:
            best_distance = tl.full((), float("inf"), tl.float32)
            best_center = tl.full((32, 16), 0.0, tl.float32)
            features = groups[:, None] * 16 + lanes[None, :]
            for cluster in tl.static_range(CENTERS):
                offsets = _center_offsets(
                    slot // 64, cluster, features, ROW_BYTES, CENTERS
                )
                center = tl.load((Cache + offsets).to(tl.pointer_type(tl.bfloat16))).to(
                    tl.float32
                )
                difference = values - center
                distance = tl.sum(tl.sum(difference * difference, 1), 0)
                better = distance < best_distance
                best_center = tl.where(better, center, best_center)
                center_id = tl.where(better, cluster, center_id)
                best_distance = tl.minimum(best_distance, distance)
            values -= best_center
    maxima = tl.max(tl.abs(values), 1)
    global_scale = tl.maximum(tl.max(maxima, 0) / (6.0 * 448.0), 1.0e-30)
    scales = tl.maximum(maxima / (6.0 * global_scale), 0.001953125).to(tl.float8e4nv)
    codes = _encode_fp4(values / (scales.to(tl.float32)[:, None] * global_scale))
    pairs = tl.reshape(codes, (256, 2))
    packed = tl.sum(
        tl.where(
            tl.arange(0, 2)[None, :] == 0, pairs.to(tl.int32), pairs.to(tl.int32) << 4
        ),
        1,
    )
    base = Cache + slot * ROW_BYTES
    tl.store(base + tl.arange(0, 256), packed.to(tl.uint8))
    tl.store((base + 256).to(tl.pointer_type(tl.float8e4nv)) + groups, scales)
    tl.store((base + 288).to(tl.pointer_type(tl.float32)), global_scale)
    padding = tl.arange(0, 16)
    tl.store(
        base + 420 + padding,
        tl.where(padding == 0, center_id, 0).to(tl.uint8),
        padding < 4,
    )
    tl.store((base + 424).to(tl.pointer_type(tl.int64)), tag)


@triton.jit
def _write_kernel(
    Latent,
    Rope,
    Cache,
    Slots,
    LATENT_STRIDE: tl.constexpr,
    ROPE_STRIDE: tl.constexpr,
    CAPACITY: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    CENTERS: tl.constexpr,
):
    token = tl.program_id(0)
    slot = tl.load(Slots + token).to(tl.int64)
    if slot >= 0 and slot < CAPACITY:
        values = tl.load(Latent + token * LATENT_STRIDE + tl.arange(0, 512)).to(
            tl.float32
        )
        _store_latent(Cache, slot, values, ROW_BYTES, CENTERS, True, 0)
        rope_values = tl.load(Rope + token * ROPE_STRIDE + tl.arange(0, 64))
        tl.store(
            (Cache + slot * ROW_BYTES + 292).to(tl.pointer_type(tl.bfloat16))
            + tl.arange(0, 64),
            rope_values,
        )


def write_nvfp4_cache(latent, rope, cache, slots, *, deferred_state=None):
    if not is_supported(latent, rope):
        raise ValueError("NVFP4 MLA requires CUDA BF16 latent[*,512] and RoPE[*,64]")
    centers = _validate_cache(cache)
    if (
        latent.device != cache.device
        or slots.device != cache.device
        or not slots.is_contiguous()
    ):
        raise ValueError(
            "NVFP4 MLA slots and inputs must be contiguous/on the cache device"
        )
    if slots.dtype not in (torch.int32, torch.int64):
        raise ValueError("NVFP4 MLA slots must be int32 or int64")
    latent = latent.reshape(-1, 512)
    rope = rope.reshape(-1, 64)
    if latent.stride(-1) != 1 or rope.stride(-1) != 1:
        raise ValueError("NVFP4 MLA input rows must have contiguous features")
    if latent.shape[0] != rope.shape[0] or slots.numel() != latent.shape[0]:
        raise ValueError("NVFP4 MLA latent, RoPE and slot counts must match")
    if deferred_state is not None:
        deferred_state.write(latent, rope, cache, slots)
        return
    if latent.shape[0]:
        if centers:
            initialize_block_centers(latent, cache, slots, centers)
        _write_kernel[(latent.shape[0],)](
            latent,
            rope,
            cache,
            slots,
            latent.stride(0),
            rope.stride(0),
            cache.numel() // cache.shape[-1],
            cache.shape[-1],
            centers,
            num_warps=4,
        )


def _validate_cache(cache):
    if (
        not cache.is_cuda
        or cache.dtype != torch.uint8
        or not cache.is_contiguous()
        or cache.ndim != 3
        or cache.shape[-1] not in (432, 448, 464, 496)
    ):
        raise ValueError(
            "NVFP4 MLA cache must be contiguous CUDA uint8 [blocks,page,432/448/464/496]"
        )
    centers = (cache.shape[-1] - CACHE_BYTES) // 16
    if centers and cache.shape[1] != 64:
        raise ValueError("Clustered NVFP4 MLA requires 64-token blocks")
    return centers


@triton.jit
def _load_latent(
    Cache, slots, valid, features, ROW_BYTES: tl.constexpr, CENTERS: tl.constexpr
):
    base = Cache + slots[:, None].to(tl.int64) * ROW_BYTES
    packed = tl.load(base + features[None, :] // 2, valid[:, None], 0)
    code = (packed >> ((features[None, :] % 2) * 4)) & 15
    block_scale = tl.load(
        (base + 256).to(tl.pointer_type(tl.float8e4nv)) + features[None, :] // 16,
        valid[:, None],
        0.0,
    ).to(tl.float32)
    global_scale = tl.load(
        (Cache + slots.to(tl.int64) * ROW_BYTES + 288).to(tl.pointer_type(tl.float32)),
        valid,
        0,
    )
    values = _decode_fp4(code) * block_scale * global_scale[:, None]
    if CENTERS:
        center_ids = tl.load(Cache + slots.to(tl.int64) * ROW_BYTES + 420, valid, 0).to(
            tl.int32
        )
        has_center = valid & (center_ids < CENTERS)
        offsets = _center_offsets(
            slots[:, None].to(tl.int64) // 64,
            tl.minimum(center_ids, CENTERS - 1)[:, None],
            features[None, :],
            ROW_BYTES,
            CENTERS,
        )
        centers = tl.load(
            (Cache + offsets).to(tl.pointer_type(tl.bfloat16)), has_center[:, None], 0
        ).to(tl.float32)
        values += centers
    return values.to(tl.bfloat16)


@triton.jit
def _load_mixed_latent(
    Cache,
    slots,
    valid,
    features,
    ROW_BYTES: tl.constexpr,
    CENTERS: tl.constexpr,
    Tail,
    Mapping,
    Owners,
    Tags,
    Counts,
    USE_TAIL: tl.constexpr,
):
    values = _load_latent(Cache, slots, valid, features, ROW_BYTES, CENTERS)
    if USE_TAIL:
        blocks = slots.to(tl.int64) // 64
        mapping = tl.load(Mapping + blocks, valid, -1)
        safe = tl.maximum(mapping, 0)
        owner = tl.load(Owners + safe, valid & (mapping >= 0), -1)
        stored_tag = tl.load(
            (Cache + slots.to(tl.int64) * ROW_BYTES + 424).to(
                tl.pointer_type(tl.int64)
            ),
            valid,
            0,
        )
        pool_tag = tl.load(Tags + safe, valid & (mapping >= 0), -1)
        count = tl.load(Counts + safe, valid & (mapping >= 0), 0)
        marker = tl.load(Cache + slots.to(tl.int64) * ROW_BYTES + 420, valid, 0)
        raw_valid = (
            valid
            & (mapping >= 0)
            & (owner == blocks)
            & (stored_tag == pool_tag)
            & (slots % 64 < count)
            & (marker == 255)
        )
        raw = tl.load(
            Tail
            + safe[:, None].to(tl.int64) * 64 * 512
            + (slots % 64)[:, None] * 512
            + features[None, :],
            raw_valid[:, None],
            0,
        )
        values = tl.where(raw_valid[:, None], raw, values)
    return values


def _tail_args(cache):
    from rtp_llm.models_py.triton_kernels.sparse_mla.nvfp4_deferred import find_state

    state = find_state(cache)
    if state is None:
        return (cache, cache, cache, cache, cache, False)
    return (state.data, state.mapping, state.owners, state.tags, state.counts, True)


@triton.jit
def _gather_kernel(
    Tail,
    Mapping,
    Owners,
    Tags,
    Counts,
    USE_TAIL: tl.constexpr,
    Cache,
    Output,
    Table,
    Lengths,
    Starts,
    TABLE_STRIDE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    CAPACITY: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    CENTERS: tl.constexpr,
):
    position = tl.program_id(0)
    request = tl.program_id(1)
    length = tl.load(Lengths + request)
    if position < length:
        block = tl.load(Table + request * TABLE_STRIDE + position // PAGE_SIZE).to(
            tl.int64
        )
        slot = block * PAGE_SIZE + position % PAGE_SIZE
        valid = (slot >= 0) & (slot < CAPACITY)
        features = tl.arange(0, 512)
        latent = _load_mixed_latent(
            Cache,
            tl.full((1,), slot, tl.int64),
            tl.full((1,), valid, tl.int1),
            features,
            ROW_BYTES,
            CENTERS,
            Tail,
            Mapping,
            Owners,
            Tags,
            Counts,
            USE_TAIL,
        )
        destination = (tl.load(Starts + request).to(tl.int64) + position) * 576
        tl.store(Output + destination + features, tl.reshape(latent, (512,)))
        rope_features = tl.arange(0, 64)
        rope = tl.load(
            (Cache + slot * ROW_BYTES + 292).to(tl.pointer_type(tl.bfloat16))
            + rope_features,
            valid,
            0,
        )
        tl.store(Output + destination + 512 + rope_features, rope)


def gather_nvfp4_cache(
    cache, output, block_table, seq_lens, starts, batch_size, total_kv_len
):
    centers = _validate_cache(cache)
    if (
        output.dtype != torch.bfloat16
        or output.shape != (total_kv_len, 576)
        or not output.is_contiguous()
    ):
        raise ValueError(
            "NVFP4 gather destination must be contiguous BF16 [total_kv_len,576]"
        )
    if total_kv_len:
        _gather_kernel[(total_kv_len, batch_size)](
            *_tail_args(cache),
            cache,
            output,
            block_table,
            seq_lens,
            starts,
            block_table.stride(0),
            cache.shape[1],
            cache.numel() // cache.shape[-1],
            cache.shape[-1],
            centers,
            num_warps=4,
        )


@triton.jit
def _attention_kernel(
    Tail,
    Mapping,
    Owners,
    Tags,
    Counts,
    USE_TAIL: tl.constexpr,
    Query,
    Cache,
    Indices,
    Partial,
    Lse,
    QUERY_STRIDE: tl.constexpr,
    HEAD_STRIDE: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
    HEADS: tl.constexpr,
    TOPK: tl.constexpr,
    CAPACITY: tl.constexpr,
    SCALE: tl.constexpr,
    SPLITS: tl.constexpr,
    SPLIT_SIZE: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    CENTERS: tl.constexpr,
):
    token = tl.program_id(0)
    head_block = tl.program_id(1)
    split = tl.program_id(2)
    heads = head_block * BLOCK_HEADS + tl.arange(0, BLOCK_HEADS)
    features = tl.arange(0, 512)
    rope_features = tl.arange(0, 64)
    query = tl.load(
        Query + token * QUERY_STRIDE + heads[:, None] * HEAD_STRIDE + features[None, :],
        heads[:, None] < HEADS,
        0,
    )
    query_rope = tl.load(
        Query
        + token * QUERY_STRIDE
        + heads[:, None] * HEAD_STRIDE
        + 512
        + rope_features[None, :],
        heads[:, None] < HEADS,
        0,
    )
    maximum = tl.full((BLOCK_HEADS,), -1.0e30, tl.float32)
    normalizer = tl.zeros((BLOCK_HEADS,), tl.float32)
    accumulator = tl.zeros((BLOCK_HEADS, 512), tl.float32)
    for start in range(split * SPLIT_SIZE, (split + 1) * SPLIT_SIZE, 32):
        offsets = start + tl.arange(0, 32)
        slots = tl.load(
            Indices + token * INDEX_STRIDE + offsets, offsets < TOPK, -1
        ).to(tl.int64)
        valid = (slots >= 0) & (slots < CAPACITY) & (offsets < TOPK)
        latent = _load_mixed_latent(
            Cache,
            slots,
            valid,
            features,
            ROW_BYTES,
            CENTERS,
            Tail,
            Mapping,
            Owners,
            Tags,
            Counts,
            USE_TAIL,
        )
        rope = tl.load(
            (Cache + slots[:, None] * ROW_BYTES + 292).to(tl.pointer_type(tl.bfloat16))
            + rope_features[None, :],
            valid[:, None],
            0,
        )
        scores = (
            tl.dot(query, tl.trans(latent)) + tl.dot(query_rope, tl.trans(rope))
        ) * SCALE
        scores = tl.where(valid[None, :], scores, -1.0e30)
        next_maximum = tl.maximum(maximum, tl.max(scores, 1))
        correction = tl.exp(maximum - next_maximum)
        probabilities = tl.where(
            valid[None, :], tl.exp(scores - next_maximum[:, None]), 0.0
        )
        accumulator = accumulator * correction[:, None]
        accumulator += tl.dot(probabilities.to(tl.bfloat16), latent)
        normalizer = normalizer * correction + tl.sum(probabilities, 1)
        maximum = next_maximum
    output_base = (token * HEADS + heads) * SPLITS + split
    normalized = accumulator / tl.where(normalizer > 0, normalizer, 1.0)[:, None]
    tl.store(
        Partial + output_base[:, None] * 512 + features[None, :],
        normalized,
        heads[:, None] < HEADS,
    )
    tl.store(
        Lse + output_base,
        tl.where(normalizer > 0, maximum + tl.log(normalizer), -float("inf")),
        heads < HEADS,
    )


@triton.jit
def _merge_kernel(Partial, Lse, Output, SPLITS: tl.constexpr):
    row = tl.program_id(0)
    splits = tl.arange(0, SPLITS)
    features = tl.arange(0, 512)
    lse = tl.load(Lse + row * SPLITS + splits)
    maximum = tl.max(lse, 0)
    maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    weights = tl.exp(lse - maximum)
    denominator = tl.sum(weights, 0)
    partial = tl.load(
        Partial + (row * SPLITS + splits[:, None]) * 512 + features[None, :]
    )
    output = tl.sum(partial * weights[:, None], 0) / tl.where(
        denominator > 0, denominator, 1.0
    )
    tl.store(Output + row * 512 + features, output)


def sparse_nvfp4_attention(query, cache, physical_indices, scale):
    """Direct paged-cache attention with shape-stable allocations for CUDA graphs."""
    centers = _validate_cache(cache)
    if query.device != cache.device or physical_indices.device != cache.device:
        raise ValueError("NVFP4 MLA query, cache and indices must share a CUDA device")
    if query.dtype != torch.bfloat16 or query.shape[-1] != 576 or query.stride(-1) != 1:
        raise ValueError("NVFP4 sparse MLA requires BF16 queries [tokens,heads,576]")
    tokens, heads, _ = query.shape
    if tokens == 0:
        return torch.empty((0, heads, 512), dtype=query.dtype, device=query.device)
    indices = physical_indices.reshape(tokens, -1)
    if indices.stride(-1) != 1:
        raise ValueError("NVFP4 sparse MLA indices must have contiguous rows")
    topk = indices.shape[1]
    splits = 8 if tokens <= 64 and topk >= 256 else 1
    split_size = triton.cdiv(topk, splits * 32) * 32
    partial = torch.empty(
        (tokens, heads, splits, 512), dtype=torch.float32, device=query.device
    )
    lse = torch.empty((tokens, heads, splits), dtype=torch.float32, device=query.device)
    output = torch.empty((tokens, heads, 512), dtype=query.dtype, device=query.device)
    if tokens:
        _attention_kernel[(tokens, triton.cdiv(heads, 16), splits)](
            *_tail_args(cache),
            query,
            cache,
            indices,
            partial,
            lse,
            query.stride(0),
            query.stride(1),
            indices.stride(0),
            heads,
            topk,
            cache.numel() // cache.shape[-1],
            scale,
            splits,
            split_size,
            16,
            cache.shape[-1],
            centers,
            num_warps=8,
            num_stages=1,
        )
        _merge_kernel[(tokens * heads,)](partial, lse, output, splits, num_warps=4)
    return output
