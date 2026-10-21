"""Frozen per-layer, per-64-token-block BF16 centers for residual NVFP4."""

import torch
import triton
import triton.language as tl


@triton.jit
def _center_offsets(
    blocks, center_ids, features, ROW_BYTES: tl.constexpr, CENTERS: tl.constexpr
):
    linear = center_ids * 512 + features
    stripe_elements: tl.constexpr = CENTERS * 8
    return (
        blocks * 64 * ROW_BYTES
        + linear // stripe_elements * ROW_BYTES
        + 432
        + linear % stripe_elements * 2
    )


@triton.jit
def _fit_centers(values, valid, CENTERS: tl.constexpr):
    rows = tl.arange(0, 64)
    clusters = tl.arange(0, CENTERS)
    centers = tl.full((CENTERS, 512), 0.0, tl.float32)
    nearest = tl.full((64,), float("inf"), tl.float32)
    selected = tl.full((), 0, tl.int32)
    for cluster in tl.static_range(CENTERS):
        center = tl.sum(tl.where(rows[:, None] == selected, values, 0.0), 0)
        centers = tl.where(clusters[:, None] == cluster, center[None, :], centers)
        difference = values - center[None, :]
        distance = tl.sum(difference * difference, 1)
        nearest = tl.minimum(nearest, distance)
        selected = tl.argmax(tl.where(valid, nearest, -1.0), 0, tie_break_left=True)
    for iteration in range(4):
        nearest = tl.full((64,), float("inf"), tl.float32)
        assignment = tl.full((64,), 0, tl.int32)
        for cluster in tl.static_range(CENTERS):
            center = tl.sum(tl.where(clusters[:, None] == cluster, centers, 0.0), 0)
            difference = values - center[None, :]
            distance = tl.sum(difference * difference, 1)
            assignment = tl.where(distance < nearest, cluster, assignment)
            nearest = tl.minimum(nearest, distance)
        for cluster in tl.static_range(CENTERS):
            members = valid & (assignment == cluster)
            count = tl.sum(members.to(tl.int32), 0)
            mean = tl.sum(tl.where(members[:, None], values, 0.0), 0) / tl.maximum(
                count, 1
            )
            centers = tl.where(
                (clusters[:, None] == cluster) & (count > 0), mean[None, :], centers
            )
    return centers.to(tl.bfloat16)


@triton.jit
def _initialize_centers_kernel(
    Latent,
    Cache,
    SortedSlots,
    Order,
    TOKENS: tl.constexpr,
    LATENT_STRIDE: tl.constexpr,
    CAPACITY: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    CENTERS: tl.constexpr,
    FULL_ONLY: tl.constexpr = False,
):
    first = tl.program_id(0)
    slot = tl.load(SortedSlots + first).to(tl.int64)
    if slot >= 0 and slot < CAPACITY and slot % 64 == 0:
        rows = tl.arange(0, 64)
        features = tl.arange(0, 512)
        clusters = tl.arange(0, CENTERS)
        positions = first + rows
        slots = tl.load(SortedSlots + positions, positions < TOKENS, -1).to(tl.int64)
        valid = (positions < TOKENS) & (slots >= slot) & (slots < slot + 64)
        tokens = tl.load(Order + positions, valid, 0).to(tl.int64)
        values = tl.load(
            Latent + tokens[:, None] * LATENT_STRIDE + features[None, :],
            valid[:, None],
            0,
        ).to(tl.float32)
        if FULL_ONLY:
            valid_full = tl.sum(valid.to(tl.int32), 0) == 64
        else:
            valid_full = True
        centers = _fit_centers(values, valid, CENTERS)
        offsets = _center_offsets(
            slot // 64, clusters[:, None], features[None, :], ROW_BYTES, CENTERS
        )
        tl.store(
            (Cache + offsets).to(tl.pointer_type(tl.bfloat16)),
            centers.to(tl.bfloat16),
            valid_full,
        )


def initialize_block_centers(latent, cache, slots, centers, full_only=False):
    """Initialize on slot zero only; later appends keep the stored center table."""
    sorted_slots, order = torch.sort(slots.reshape(-1))
    _initialize_centers_kernel[(slots.numel(),)](
        latent,
        cache,
        sorted_slots,
        order,
        slots.numel(),
        latent.stride(0),
        cache.shape[0] * 64,
        cache.shape[-1],
        centers,
        FULL_ONLY=full_only,
        num_warps=8,
    )
