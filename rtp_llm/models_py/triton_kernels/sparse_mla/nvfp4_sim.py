"""Simulated NVFP4 MLA KV quantization on an ordinary BF16 or FP8 cache.

The cache keeps its normal layout, writer and attention path. Only the 512
latent features are replaced by the values the real NVFP4 cache would return
when read, so accuracy runs measure the quantization itself without the packed
layout, deferred tail pool, center stripes or PD byte format.

RTP_MLA_NVFP4_SIM selects the mode:
  plain    every written row is NVFP4-quantized (the real C=0 cache).
  cluster  the real deferred clustered v2 cache: rows stay unquantized until a
           write fills row 63 of their 64-token block, then the whole block is
           read back, fitted per 16-channel subspace and residual-quantized.
           RTP_MLA_NVFP4_SIM_CENTERS picks 1, 2 or 4 centers (default 2).

On a PD prefill role, prefill writes store clustered rows whose block is still
partial plain-quantized. The real prefill ships those rows as the portable
plain mirror, and decode later seals the block from that mirror.
RTP_MLA_NVFP4_SIM_PREFILL_MIRROR=auto|0|1 overrides the role check; 1 also
reproduces that PD boundary inside a PDFUSION process. Decode writes always
keep partial rows unquantized, like the real BF16 tail pool.

With a BF16 cache, sealed rows are bit-identical to what the real kernels
decode. With an FP8 cache, both unquantized rows and simulated values pass
through the normal fp8_ds_mla writer, which adds FP8 rounding on top.

Launch grids depend only on the token count and need no host sync, so every
captured batch size can replay under CUDA Graph. Large eager writes (prefill)
first compact the tokens that seal a block.
"""

import os
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import triton
import triton.language as tl

# Assignment of rows that stay on the origin instead of a learned center.
NO_CENTER = tl.constexpr(15)
# Compacting sealing tokens costs a host sync. Below this many tokens, and
# always while a CUDA graph is capturing, every token proposes its own block.
COMPACT_MIN_TOKENS = 256


@dataclass(frozen=True)
class SimConfig:
    mode: str
    centers: int = 0
    prefill_mirror: bool = False

    @staticmethod
    def from_env(prefill_role: bool = False) -> Optional["SimConfig"]:
        mode = os.environ.get("RTP_MLA_NVFP4_SIM", "")
        if not mode:
            return None
        if mode not in ("plain", "cluster"):
            raise ValueError("RTP_MLA_NVFP4_SIM must be plain or cluster")
        if mode == "plain":
            return SimConfig(mode)
        centers = int(os.environ.get("RTP_MLA_NVFP4_SIM_CENTERS", "2"))
        if centers not in (1, 2, 4):
            raise ValueError("RTP_MLA_NVFP4_SIM_CENTERS must be 1, 2 or 4")
        mirror = os.environ.get("RTP_MLA_NVFP4_SIM_PREFILL_MIRROR", "auto")
        if mirror not in ("auto", "0", "1"):
            raise ValueError("RTP_MLA_NVFP4_SIM_PREFILL_MIRROR must be auto, 0 or 1")
        return SimConfig(mode, centers, prefill_role if mirror == "auto" else mirror == "1")


@triton.jit
def _round_fp4(magnitude):
    """Nearest E2M1 magnitude with ties to even, matching the real encoder."""
    q = tl.where(magnitude > 0.25, 0.5, 0.0)
    q = tl.where(magnitude >= 0.75, 1.0, q)
    q = tl.where(magnitude > 1.25, 1.5, q)
    q = tl.where(magnitude >= 1.75, 2.0, q)
    q = tl.where(magnitude > 2.5, 3.0, q)
    q = tl.where(magnitude >= 3.5, 4.0, q)
    return tl.where(magnitude > 5.0, 6.0, q)


@triton.jit
def _fake_quant(values):
    """(32, 16) fp32 -> dequantized NVFP4: E4M3 scale per 16, FP32 scale per row."""
    maxima = tl.max(tl.abs(values), 1)
    row_scale = tl.maximum(tl.max(maxima, 0) / (6.0 * 448.0), 1.0e-30)
    scales = tl.maximum(maxima / (6.0 * row_scale), 0.001953125)
    scales = scales.to(tl.float8e4nv).to(tl.float32)
    scaled = values / (scales[:, None] * row_scale)
    q = _round_fp4(tl.abs(scaled))
    q = tl.where(scaled < 0, -q, q)
    return q * scales[:, None] * row_scale


@triton.jit
def _load_latent(Cache, offsets, features, FP8: tl.constexpr):
    """Latent features of cache rows as fp32; offsets/features broadcast."""
    if FP8:
        # fp8_ds_mla: 512 E4M3, 4 FP32 scales (one per 128), 64 BF16 RoPE.
        values = tl.load(Cache + offsets + features)
        values = values.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        scales = tl.load(
            (Cache + offsets + 512).to(tl.pointer_type(tl.float32)) + features // 128
        )
        return values * scales
    return tl.load(Cache + offsets + features).to(tl.float32)


# TOKENS varies with batch size; specializing on it recompiles per batch.
@triton.jit(do_not_specialize=["TOKENS"])
def _plain_kernel(Input, Output, Keep, TOKENS, INPUT_STRIDE, HAS_KEEP: tl.constexpr):
    token = tl.program_id(0)
    groups = tl.arange(0, 32)
    lanes = tl.arange(0, 16)
    features = groups[:, None] * 16 + lanes[None, :]
    values = tl.load(Input + token * INPUT_STRIDE + features).to(tl.float32)
    quantized = _fake_quant(values)
    if HAS_KEEP:
        quantized = tl.where(tl.load(Keep + token) != 0, values, quantized)
    tl.store(Output + token * 512 + features, quantized.to(tl.bfloat16))


@triton.jit
def _sealing_slot(Slots, Candidates, CAPACITY, COMPACT: tl.constexpr):
    """Slot of this program's token, or -1 unless it fills row 63 of a block."""
    if COMPACT:
        token = tl.load(Candidates + tl.program_id(0))
    else:
        token = tl.program_id(0)
    slot = tl.load(Slots + token).to(tl.int64)
    sealing = (slot >= 0) & (slot < CAPACITY) & (slot % 64 == 63)
    return tl.where(sealing, slot, -1)


@triton.jit(do_not_specialize=["TOKENS"])
def _fit_kernel(
    Cache,
    Slots,
    Candidates,
    Centers,
    Assign,
    TOKENS,
    CAPACITY,
    BLOCK_STRIDE: tl.constexpr,
    ENTRY_STRIDE: tl.constexpr,
    CENTERS: tl.constexpr,
    FP8: tl.constexpr,
    COMPACT: tl.constexpr,
):
    """One 16-channel subspace of one sealing block: fit centers, pick per row."""
    program = tl.program_id(0)
    group = tl.program_id(1)
    slot = _sealing_slot(Slots, Candidates, CAPACITY, COMPACT)
    if slot >= 0:
        rows = tl.arange(0, 64)
        lanes = tl.arange(0, 16)
        clusters = tl.arange(0, CENTERS)
        offsets = (slot // 64) * BLOCK_STRIDE + rows * ENTRY_STRIDE
        x = _load_latent(Cache, offsets[:, None], group * 16 + lanes[None, :], FP8)
        # Farthest-point initialization from row 0, then 4 Lloyd iterations.
        centers = tl.zeros((CENTERS, 16), tl.float32)
        nearest = tl.full((64,), float("inf"), tl.float32)
        selected = tl.full((), 0, tl.int32)
        for c in tl.static_range(CENTERS):
            center = tl.sum(tl.where(rows[:, None] == selected, x, 0.0), 0)
            centers = tl.where(clusters[:, None] == c, center[None, :], centers)
            difference = x - center[None, :]
            nearest = tl.minimum(nearest, tl.sum(difference * difference, 1))
            selected = tl.argmax(nearest, 0, tie_break_left=True)
        for _ in range(4):
            nearest = tl.full((64,), float("inf"), tl.float32)
            assignment = tl.zeros((64,), tl.int32)
            for c in tl.static_range(CENTERS):
                center = tl.sum(tl.where(clusters[:, None] == c, centers, 0.0), 0)
                difference = x - center[None, :]
                distance = tl.sum(difference * difference, 1)
                assignment = tl.where(distance < nearest, c, assignment)
                nearest = tl.minimum(nearest, distance)
            for c in tl.static_range(CENTERS):
                members = assignment == c
                count = tl.sum(members.to(tl.int32), 0)
                mean = tl.sum(tl.where(members[:, None], x, 0.0), 0) / tl.maximum(count, 1)
                centers = tl.where(
                    (clusters[:, None] == c) & (count > 0), mean[None, :], centers
                )
        # Centers are stored in BF16; encode against the stored values.
        centers = centers.to(tl.bfloat16).to(tl.float32)
        # The origin is the first candidate; strict < keeps the earlier choice.
        best = tl.sum(x * x, 1)
        choice = tl.full((64,), NO_CENTER, tl.int32)
        for c in tl.static_range(CENTERS):
            center = tl.sum(tl.where(clusters[:, None] == c, centers, 0.0), 0)
            difference = x - center[None, :]
            distance = tl.sum(difference * difference, 1)
            choice = tl.where(distance < best, c, choice)
            best = tl.minimum(best, distance)
        tl.store(
            Centers
            + ((program * 32 + group) * CENTERS + clusters[:, None]) * 16
            + lanes[None, :],
            centers,
        )
        tl.store(Assign + (program * 64 + rows) * 32 + group, choice.to(tl.int8))


@triton.jit(do_not_specialize=["TOKENS"])
def _seal_kernel(
    Cache,
    Slots,
    Candidates,
    Centers,
    Assign,
    Latent,
    Rope,
    WriteSlots,
    TOKENS,
    CAPACITY,
    BLOCK_STRIDE: tl.constexpr,
    ENTRY_STRIDE: tl.constexpr,
    CENTERS: tl.constexpr,
    FP8: tl.constexpr,
    COMPACT: tl.constexpr,
):
    """One row of one sealing block: residual-quantize against its centers."""
    program = tl.program_id(0)
    row = tl.program_id(1)
    out = program * 64 + row
    slot = _sealing_slot(Slots, Candidates, CAPACITY, COMPACT)
    if slot >= 0:
        groups = tl.arange(0, 32)
        lanes = tl.arange(0, 16)
        features = groups[:, None] * 16 + lanes[None, :]
        offset = (slot // 64) * BLOCK_STRIDE + row * ENTRY_STRIDE
        x = _load_latent(Cache, offset, features, FP8)
        choice = tl.load(Assign + out * 32 + groups).to(tl.int32)
        has_center = choice < CENTERS
        center = tl.load(
            Centers
            + ((program * 32 + groups[:, None]) * CENTERS + tl.minimum(choice, CENTERS - 1)[:, None])
            * 16
            + lanes[None, :],
            has_center[:, None],
            0.0,
        )
        values = _fake_quant(x - center) + center
        tl.store(Latent + out * 512 + features, values.to(tl.bfloat16))
        rope_lanes = tl.arange(0, 64)
        if FP8:
            rope = tl.load((Cache + offset + 528).to(tl.pointer_type(tl.bfloat16)) + rope_lanes)
        else:
            rope = tl.load(Cache + offset + 512 + rope_lanes)
        tl.store(Rope + out * 64 + rope_lanes, rope)
        tl.store(WriteSlots + out, slot - 63 + row)
    else:
        tl.store(WriteSlots + out, -1)


def plain_fake_quant(latent: torch.Tensor, keep: Optional[torch.Tensor] = None) -> torch.Tensor:
    """NVFP4-quantize BF16 rows [N, 512]; rows with keep != 0 are copied as is."""
    output = torch.empty((latent.shape[0], 512), dtype=torch.bfloat16, device=latent.device)
    if latent.shape[0]:
        _plain_kernel[(latent.shape[0],)](
            latent,
            output,
            latent if keep is None else keep,
            latent.shape[0],
            latent.stride(0),
            HAS_KEEP=keep is not None,
            num_warps=4,
        )
    return output


def completed_rows(slots: torch.Tensor) -> torch.Tensor:
    """Rows whose 64-token block gets its last row in this same write."""
    slots = slots.reshape(-1).to(torch.int64)
    valid = slots >= 0
    last = torch.where(valid, slots - slots % 64 + 63, torch.full_like(slots, -2))
    present = torch.sort(torch.where(valid, slots, torch.full_like(slots, -1))).values
    index = torch.searchsorted(present, last).clamp(max=present.numel() - 1)
    return valid & (present[index] == last)


def seal_blocks(config: SimConfig, cache: torch.Tensor, slots: torch.Tensor, writer) -> None:
    """Cluster-quantize every block whose row 63 is in `slots`, in place."""
    fp8 = cache.dtype != torch.bfloat16
    view = cache.view(torch.uint8) if fp8 else cache
    if view.ndim != 3 or view.shape[1] != 64 or view.shape[2] != (656 if fp8 else 576):
        raise ValueError(
            "NVFP4 cluster simulation needs a [blocks, 64, 576] BF16 or "
            "[blocks, 64, 656B] fp8_ds_mla MLA cache"
        )
    tokens = slots.numel()
    compact = tokens > COMPACT_MIN_TOKENS and not torch.cuda.is_current_stream_capturing()
    if compact:
        candidates = torch.nonzero((slots >= 0) & (slots % 64 == 63)).flatten()
        programs = candidates.numel()
        if programs == 0:
            return
    else:
        candidates, programs = slots, tokens
    device = cache.device
    centers = torch.empty((programs, 32, config.centers, 16), dtype=torch.float32, device=device)
    assign = torch.empty((programs, 64, 32), dtype=torch.int8, device=device)
    latent = torch.empty((programs * 64, 512), dtype=torch.bfloat16, device=device)
    rope = torch.empty((programs * 64, 64), dtype=torch.bfloat16, device=device)
    write_slots = torch.empty((programs * 64,), dtype=torch.int64, device=device)
    common = dict(
        TOKENS=tokens,
        CAPACITY=view.shape[0] * 64,
        BLOCK_STRIDE=view.stride(0),
        ENTRY_STRIDE=view.stride(1),
        CENTERS=config.centers,
        FP8=fp8,
        COMPACT=compact,
    )
    _fit_kernel[(programs, 32)](view, slots, candidates, centers, assign, **common, num_warps=2)
    _seal_kernel[(programs, 64)](
        view, slots, candidates, centers, assign, latent, rope, write_slots, **common, num_warps=4
    )
    writer(latent, rope, cache, write_slots)


def write_simulated(
    config: SimConfig,
    latent: torch.Tensor,
    rope: torch.Tensor,
    cache: torch.Tensor,
    slots: torch.Tensor,
    writer: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None],
    prefill: bool = True,
) -> None:
    """Write MLA KV through `writer` with NVFP4 quantization error simulated.

    writer(latent [N, 512], rope [N, 64], cache, slots [N] int64) is the
    ordinary cache writer; slot -1 rows must be skipped. `prefill` marks a
    prefill forward, the only writes the prefill mirror applies to.
    """
    latent = latent.reshape(-1, latent.shape[-1])
    if latent.shape[-1] != 512 or latent.dtype != torch.bfloat16 or latent.stride(-1) != 1:
        raise ValueError("NVFP4 simulation needs BF16 latent rows [N, 512] with unit stride")
    if slots.numel() != latent.shape[0]:
        raise ValueError("NVFP4 simulation latent and slot counts must match")
    if not latent.shape[0]:
        return
    stored = latent
    if config.mode == "plain":
        stored = plain_fake_quant(latent)
    elif config.prefill_mirror and prefill:
        stored = plain_fake_quant(latent, completed_rows(slots))
    writer(stored, rope, cache, slots)
    if config.mode == "cluster":
        seal_blocks(config, cache, slots, writer)
