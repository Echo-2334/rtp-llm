"""Frozen per-layer, per-64-token-block BF16 subspace centers for residual NVFP4."""

import torch
import triton
import triton.language as tl


# The 512 latent features are quantized as 32 independent 16-feature NVFP4
# groups.  Clustered pages keep one center id for each such group.
# Read inside @triton.jit functions, so these must be constexpr globals.
SUBSPACE_SIZE = tl.constexpr(16)
NUM_SUBSPACES = tl.constexpr(512 // 16)
CENTER_DATA_OFFSET = tl.constexpr(448)


@triton.jit
def _center_offsets(
    blocks, center_ids, features, ROW_BYTES: tl.constexpr, CENTERS: tl.constexpr
):
    linear = center_ids * 512 + features
    stripe_elements: tl.constexpr = CENTERS * 8
    return (
        blocks * 64 * ROW_BYTES
        + linear // stripe_elements * ROW_BYTES
        + CENTER_DATA_OFFSET
        + linear % stripe_elements * 2
    )


@triton.jit
def _fit_centers(values, valid, CENTERS: tl.constexpr):
    """Fit C centers independently in each 16-channel latent subspace.

    The C stored centers are fitted independently for every subspace.  The
    origin is a separate virtual candidate during residual encoding, so the
    learned center budget remains C.
    """
    rows = tl.arange(0, 64)
    groups = tl.arange(0, NUM_SUBSPACES)
    lanes = tl.arange(0, SUBSPACE_SIZE)
    clusters = tl.arange(0, CENTERS)
    values = tl.reshape(values, (64, NUM_SUBSPACES, SUBSPACE_SIZE))
    centers = tl.full(
        (CENTERS, NUM_SUBSPACES, SUBSPACE_SIZE), 0.0, tl.float32
    )

    # Farthest-point initialization is performed independently per subspace.
    nearest = tl.full((64, NUM_SUBSPACES), float("inf"), tl.float32)
    # Every subspace starts from the first valid row.
    first = tl.argmax(valid.to(tl.int32), 0, tie_break_left=True)
    selected = tl.full((NUM_SUBSPACES,), 0, tl.int32) + first
    for cluster in tl.static_range(CENTERS):
        center = tl.sum(
            tl.where(
                rows[:, None, None] == selected[None, :, None],
                values,
                0.0,
            ),
            0,
        )
        centers = tl.where(
            clusters[:, None, None] == cluster,
            center[None, :, :],
            centers,
        )
        difference = values - center[None, :, :]
        distance = tl.sum(difference * difference, 2)
        nearest = tl.minimum(nearest, distance)
        selected = tl.argmax(
            tl.where(valid[:, None], nearest, -1.0), 0, tie_break_left=True
        )

    # A few iterations are enough for the 64-token block and keep the kernel
    # shape fixed for CUDA graph capture.
    for _ in range(4):
        nearest = tl.full((64, NUM_SUBSPACES), float("inf"), tl.float32)
        assignment = tl.full((64, NUM_SUBSPACES), 0, tl.int32)
        for cluster in tl.static_range(CENTERS):
            center = tl.sum(
                tl.where(
                    clusters[:, None, None] == cluster,
                    centers,
                    0.0,
                ),
                0,
            )
            difference = values - center[None, :, :]
            distance = tl.sum(difference * difference, 2)
            assignment = tl.where(distance < nearest, cluster, assignment)
            nearest = tl.minimum(nearest, distance)
        for cluster in tl.static_range(CENTERS):
            members = valid[:, None] & (assignment == cluster)
            count = tl.sum(members.to(tl.int32), 0)
            mean = tl.sum(
                tl.where(members[:, :, None], values, 0.0), 0
            ) / tl.maximum(count[:, None], 1)
            centers = tl.where(
                (clusters[:, None, None] == cluster) & (count[None, :, None] > 0),
                mean[None, :, :],
                centers,
            )
    return tl.reshape(centers, (CENTERS, 512)).to(tl.bfloat16)


# TOKENS varies with every prefill batch; specializing on it recompiles per length.
@triton.jit(do_not_specialize=["TOKENS"])
def _initialize_centers_kernel(
    Latent,
    Cache,
    SortedSlots,
    Order,
    TOKENS,
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
