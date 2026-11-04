"""Length-aware unsorted TopK for non-NaN FP32 indexer logits.

Four radix passes locate the exact K-th ordered float key. Per-tile counts and
prefix offsets then emit strictly greater keys, followed by threshold ties in
column order. Short rows are -1 padded. Bounds stay on the GPU during capture
and replay; no score-matrix padding is read. Signed zeros use IEEE bit order.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _hist(
    X,
    LO,
    HI,
    PREFIX,
    HIST,
    N: tl.constexpr,
    S: tl.constexpr,
    TILES: tl.constexpr,
    SHIFT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b, t = tl.program_id(0), tl.program_id(1)
    idx = t * BLOCK + tl.arange(0, BLOCK)
    lo, hi = tl.load(LO + b), tl.load(HI + b)
    bins = tl.arange(0, 256)
    if t * BLOCK < hi and (t + 1) * BLOCK > lo:
        live = (idx < N) & (idx >= lo) & (idx < hi)
        v = tl.load(X + b.to(tl.int64) * S + idx, live, 0.0)
        bits = v.to(tl.uint32, bitcast=True)
        key = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)
        if SHIFT < 24:
            prefix = tl.load(PREFIX + b)
            live &= (key >> (SHIFT + 8)) == (prefix >> (SHIFT + 8))
        bucket = tl.where(live, ((key >> SHIFT) & 255).to(tl.int32), 256)
        hist = tl.histogram(bucket, 256)
    else:
        hist = tl.full((256,), 0, tl.int32)
    tl.store(HIST + (b * TILES + t) * 256 + bins, hist)


@triton.jit
def _reduce(HIST, SMALL, TILES: tl.constexpr, TB: tl.constexpr):
    b, g = tl.program_id(0), tl.program_id(1)
    t = tl.arange(0, TB)
    col = g * 16 + tl.arange(0, 16)
    vals = tl.load(
        HIST + (b * TILES + t[:, None]) * 256 + col[None, :], t[:, None] < TILES, 0
    )
    tl.store(SMALL + b * 256 + col, tl.sum(vals, 0))


@triton.jit
def _choose(SMALL, PREFIX, RANK, K: tl.constexpr, SHIFT: tl.constexpr):
    b = tl.program_id(0)
    idx = tl.arange(0, 256)
    hist = tl.load(SMALL + b * 256 + idx)
    total = tl.sum(hist, 0)
    rank = tl.minimum(K, total) if SHIFT == 24 else tl.load(RANK + b)
    above = total - tl.cumsum(hist, 0)
    chosen = tl.max(tl.where((above < rank) & (above + hist >= rank), idx, 0), 0)
    n_above = tl.sum(tl.where(idx == chosen, above, 0), 0)
    prefix = tl.full((), 0, tl.uint32) if SHIFT == 24 else tl.load(PREFIX + b)
    tl.store(PREFIX + b, prefix | (chosen.to(tl.uint32) << SHIFT))
    tl.store(RANK + b, rank - n_above)


@triton.jit
def _count(
    X,
    LO,
    HI,
    PREFIX,
    COUNTS,
    N: tl.constexpr,
    S: tl.constexpr,
    TILES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b, t = tl.program_id(0), tl.program_id(1)
    idx = t * BLOCK + tl.arange(0, BLOCK)
    lo, hi = tl.load(LO + b), tl.load(HI + b)
    live = (idx < N) & (idx >= lo) & (idx < hi)
    v = tl.load(X + b.to(tl.int64) * S + idx, live, 0.0)
    bits = v.to(tl.uint32, bitcast=True)
    key = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)
    threshold = tl.load(PREFIX + b)
    tl.store(
        COUNTS + (b * TILES + t) * 2, tl.sum((live & (key > threshold)).to(tl.int32), 0)
    )
    tl.store(
        COUNTS + (b * TILES + t) * 2 + 1,
        tl.sum((live & (key == threshold)).to(tl.int32), 0),
    )


@triton.jit
def _emit(
    X,
    LO,
    HI,
    PREFIX,
    COUNTS,
    OUT,
    N: tl.constexpr,
    S: tl.constexpr,
    K: tl.constexpr,
    TILES: tl.constexpr,
    TB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b, t = tl.program_id(0), tl.program_id(1)
    idx = t * BLOCK + tl.arange(0, BLOCK)
    lo, hi = tl.load(LO + b), tl.load(HI + b)
    if t * BLOCK < hi and (t + 1) * BLOCK > lo:
        live = (idx < N) & (idx >= lo) & (idx < hi)
        v = tl.load(X + b.to(tl.int64) * S + idx, live, 0.0)
        bits = v.to(tl.uint32, bitcast=True)
        key = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)
        threshold = tl.load(PREFIX + b)
        gt = live & (key > threshold)
        eq = live & (key == threshold)
        j = tl.arange(0, TB)
        ng = tl.load(COUNTS + (b * TILES + j) * 2, j < TILES, 0)
        ne = tl.load(COUNTS + (b * TILES + j) * 2 + 1, j < TILES, 0)
        beforeg = tl.sum(tl.where(j < t, ng, 0), 0)
        beforee = tl.sum(tl.where(j < t, ne, 0), 0)
        totalg = tl.sum(ng, 0)
        pos = tl.where(
            gt,
            beforeg + tl.cumsum(gt.to(tl.int32), 0) - 1,
            totalg + beforee + tl.cumsum(eq.to(tl.int32), 0) - 1,
        )
        tl.store(OUT + b * K + pos, idx, (gt | eq) & (pos < K))


def radix_topk(x, lo, hi, k):
    """Select absolute columns from [lo, hi), with unit column stride.

    The caller supplies valid bounds and nonempty rows/column capacity. The
    candidate budget is fixed at graph capture; only scores and bounds vary.
    """
    b, n = x.shape
    # Shorter tiles reduce shared-histogram contention on short/medium rows.
    block = (512 if b == 1 or n <= 65536 else 1024) if n <= 262144 else 4096
    warps = 8 if n > 262144 else 4
    tiles = triton.cdiv(n, block)
    tb = triton.next_power_of_2(tiles)
    prefix = torch.empty(b, device=x.device, dtype=torch.uint32)
    rank = torch.empty(b, device=x.device, dtype=torch.int32)
    hist = torch.empty((b, tiles, 256), device=x.device, dtype=torch.int32)
    small = torch.empty((b, 256), device=x.device, dtype=torch.int32)
    for shift in [24, 16, 8, 0]:
        _hist[(b, tiles)](
            x,
            lo,
            hi,
            prefix,
            hist,
            n,
            x.stride(0),
            tiles,
            shift,
            block,
            num_warps=warps,
        )
        _reduce[(b, 16)](hist, small, tiles, tb)
        _choose[(b,)](small, prefix, rank, k, shift)
    counts = torch.empty((b, tiles, 2), device=x.device, dtype=torch.int32)
    out = torch.full((b, k), -1, device=x.device, dtype=torch.int32)
    _count[(b, tiles)](
        x, lo, hi, prefix, counts, n, x.stride(0), tiles, block, num_warps=warps
    )
    _emit[(b, tiles)](
        x,
        lo,
        hi,
        prefix,
        counts,
        out,
        n,
        x.stride(0),
        k,
        tiles,
        tb,
        block,
        num_warps=warps,
    )
    return out
