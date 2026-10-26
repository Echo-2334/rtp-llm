"""Reuse a per-request top-8192 Indexer pool across GLM-5.3 decode steps.

A refresh step scores the full context with the original Indexer logits and
keeps the request's top-8192 token indices as its pool. That step and the next
`reuse_steps` decode steps then score only the pool plus the tokens appended
since the refresh, and select top-2048 from those candidates; on the refresh
step this is the full-context top-2048 up to near-ties. Rows shorter than
`min_length` refresh every step, with all their tokens in the pool when they
have at most 8192.

Rows find their pool slot through KV pages rather than request ids: the first
physical page, and the page that held the refresh step's last token. That page
was private to the request when the pool was built, so a reordered batch finds
its pools again and a new request never inherits one. All decisions are made on
the device, so one CUDA graph per batch size serves refresh and reuse steps.
"""

from typing import Callable

import torch
import triton
import triton.language as tl

# Default pool size; IndexerPool(pool_size=...) overrides it.
POOL = 8192
# Candidates after the pool for tokens appended since the refresh step.
TAIL = 64
CANDIDATES = POOL + TAIL
TOPK = 2048
PAGE = 64
HEAD_DIM = 128
PAGE_BYTES = PAGE * (HEAD_DIM + 4)
CANDIDATE_TILE = 64
# Pool building: logits above a sampled floor (about twice the pool size) are
# compacted per 4096-token segment into a row scratch of four pool sizes, then
# selected exactly.
SEGMENT = 4096


@triton.jit
def _plan_kernel(
    block_table,
    lengths,
    slot_first_page,
    slot_anchor,
    slot_anchor_page,
    row_slot,
    pool_lengths,
    candidate_lengths,
    rows,
    TABLE_STRIDE,
    TABLE_COLS,
    REUSE_STEPS,
    MIN_LENGTH,
    PAGE: tl.constexpr,
    CANDIDATES: tl.constexpr,
    ROWS: tl.constexpr,
    SLOTS: tl.constexpr,
):
    r = tl.arange(0, ROWS)
    s = tl.arange(0, SLOTS)
    live = r < rows
    length = tl.load(lengths + r, live, other=0)
    first = tl.load(block_table + r.to(tl.int64) * TABLE_STRIDE, live, other=-1)
    s_first = tl.load(slot_first_page + s)
    s_anchor = tl.load(slot_anchor + s)
    s_page = tl.load(slot_anchor_page + s)

    # A slot matches a row that still owns both pages and is inside the window.
    in_window = (
        live[:, None]
        & (s_anchor[None, :] >= MIN_LENGTH)
        & (length[:, None] > s_anchor[None, :])
        & (length[:, None] <= s_anchor[None, :] + REUSE_STEPS)
    )
    anchor_col = (s_anchor - 1) // PAGE
    page_at = tl.load(
        block_table + r[:, None].to(tl.int64) * TABLE_STRIDE + anchor_col[None, :],
        in_window & (anchor_col[None, :] < TABLE_COLS),
        other=-2,
    )
    match = (
        in_window
        & (first[:, None] == s_first[None, :])
        & (page_at == s_page[None, :])
    )
    found = tl.min(tl.where(match, s[None, :], SLOTS), axis=1)
    matched = found < SLOTS
    taken = tl.max(match.to(tl.int32), axis=0) > 0

    # Every other row rebuilds its pool in a free slot, taken in order.
    need = live & ~matched & (length > 0)
    free = ~taken
    need_rank = tl.cumsum(need.to(tl.int32), 0) - 1
    free_rank = tl.cumsum(free.to(tl.int32), 0) - 1
    assign = need[:, None] & free[None, :] & (need_rank[:, None] == free_rank[None, :])
    fresh = tl.min(tl.where(assign, s[None, :], SLOTS), axis=1)
    slot = tl.where(matched, found, tl.where(fresh < SLOTS, fresh, -1))
    building = ~matched & (slot >= 0)

    tl.store(row_slot + r, slot, live)
    tl.store(pool_lengths + r, tl.where(building, length, 0), live)
    tl.store(candidate_lengths + r, tl.where(slot >= 0, CANDIDATES, 0), live)

    last_page = tl.load(
        block_table + r.to(tl.int64) * TABLE_STRIDE + (length - 1) // PAGE,
        live & building,
        other=-1,
    )
    # Every slot read above has completed before any slot is rewritten.
    tl.debug_barrier()
    target = tl.where(building & live, slot, 0)
    write = building & live
    tl.store(slot_first_page + target, first, write)
    tl.store(slot_anchor + target, length, write)
    tl.store(slot_anchor_page + target, last_page, write)


@triton.jit
def _count_at_least(base, n, lo, span, CHUNK: tl.constexpr, SPLITS: tl.constexpr):
    """Thresholds lo + span * j / (SPLITS + 1) for j = 1..SPLITS and how many
    of the row's first n logits reach each."""
    offs = tl.arange(0, CHUNK)
    j = tl.arange(0, SPLITS) + 1
    thresholds = lo + span * (j.to(tl.float32) / (SPLITS + 1))
    # Reduce once after the loop; a per-chunk reduction costs more than the compares.
    counts = tl.zeros((SPLITS, CHUNK), tl.int32)
    for start in range(0, n, CHUNK):
        x = tl.load(base + start + offs, start + offs < n, other=float("-inf"))
        counts += (x[None, :] >= thresholds[:, None]).to(tl.int32)
    return thresholds, tl.sum(counts, axis=1)


@triton.jit
def _select_into(
    base,
    tokens,
    n,
    out,
    POOL: tl.constexpr,
    CHUNK: tl.constexpr,
    SPLITS: tl.constexpr,
    PASSES: tl.constexpr,
    INDIRECT: tl.constexpr,
):
    """Top-POOL of n logits at base into out, by multiway bisection that keeps
    [lo, hi) around the POOL-th largest with count(>= lo) >= POOL >
    count(>= hi), then takes every logit >= hi and the first ones in
    [lo, hi). The bracket ends within (max - min) / (SPLITS + 1) ** PASSES, so
    only near-ties can be swapped. Positions are written, or tokens[position]
    when INDIRECT."""
    offs = tl.arange(0, CHUNK)
    low = tl.full((CHUNK,), float("inf"), tl.float32)
    high = tl.full((CHUNK,), float("-inf"), tl.float32)
    for start in range(0, n, CHUNK):
        valid = start + offs < n
        x = tl.load(base + start + offs, valid, other=0.0)
        low = tl.minimum(low, tl.where(valid, x, float("inf")))
        high = tl.maximum(high, tl.where(valid, x, float("-inf")))
    lo = tl.min(low)
    top = tl.max(high)
    hi = float("inf")
    above = 0  # count(>= hi)
    for _ in tl.static_range(PASSES):
        thresholds, counts = _count_at_least(base, n, lo, top - lo, CHUNK, SPLITS)
        j = tl.arange(0, SPLITS)
        # Highest threshold still reaching POOL, then the one after it.
        last = tl.max(tl.where(counts >= POOL, j, -1))
        lo = tl.maximum(lo, tl.max(tl.where(j == last, thresholds, float("-inf"))))
        nxt = last + 1
        has_next = nxt < SPLITS
        hi = tl.where(has_next, tl.sum(tl.where(j == nxt, thresholds, 0.0)), hi)
        above = tl.where(has_next, tl.sum(tl.where(j == nxt, counts, 0)), above)
        top = tl.minimum(top, hi)
    above_seen = tl.full((), 0, tl.int32)
    inside_seen = tl.full((), 0, tl.int32)
    for start in range(0, n, CHUNK):
        valid = start + offs < n
        x = tl.load(base + start + offs, valid, other=float("-inf"))
        take = valid & (x >= hi)
        inside = valid & (x >= lo) & (x < hi)
        if INDIRECT:
            token = tl.load(tokens + start + offs, valid, other=-1)
        else:
            token = start + offs
        take_at = above_seen + tl.cumsum(take.to(tl.int32), 0) - 1
        inside_at = above + inside_seen + tl.cumsum(inside.to(tl.int32), 0) - 1
        tl.store(out + take_at, token, take)
        tl.store(out + inside_at, token, inside & (inside_at < POOL))
        above_seen += tl.sum(take.to(tl.int32))
        inside_seen += tl.sum(inside.to(tl.int32))


@triton.jit
def _sample_floor(base, n, TARGET: tl.constexpr, SCRATCH: tl.constexpr, SPLITS: tl.constexpr):
    """A logit floor that about TARGET of the row's n logits reach, from 16
    evenly spaced 128-logit blocks; -inf for rows of at most TARGET."""
    b = tl.arange(0, 16)
    e = tl.arange(0, 128)
    long_row = n > TARGET
    first = (b.to(tl.int64) * (n - 128).to(tl.int64)) // 15
    sample = tl.load(base + first[:, None] + e[None, :], long_row, other=0.0)
    rank = tl.maximum(TARGET * 2048 // tl.maximum(n, 1), 1)
    lo = tl.min(tl.min(sample, axis=1))
    top = tl.max(tl.max(sample, axis=1))
    j = tl.arange(0, SPLITS)
    for _ in tl.static_range(3):
        thresholds = lo + (top - lo) * ((j + 1).to(tl.float32) / (SPLITS + 1))
        hits = sample[None, :, :] >= thresholds[:, None, None]
        counts = tl.sum(tl.sum(hits.to(tl.int32), axis=2), axis=1)
        last = tl.max(tl.where(counts >= rank, j, -1))
        lo = tl.maximum(lo, tl.max(tl.where(j == last, thresholds, float("-inf"))))
        top = tl.where(last + 1 < SPLITS, tl.sum(tl.where(j == last + 1, thresholds, 0.0)), top)
    return tl.where(long_row, lo, float("-inf"))


@triton.jit
def _count_kernel(
    logits,
    LOGITS_STRIDE,
    pool_lengths,
    floor,
    segment_counts,
    SEGMENTS: tl.constexpr,
    SEGMENT: tl.constexpr,
    TARGET: tl.constexpr,
    SCRATCH: tl.constexpr,
    POOL: tl.constexpr,
    SPLITS: tl.constexpr,
):
    row = tl.program_id(0)
    seg = tl.program_id(1)
    n = tl.load(pool_lengths + row)
    if n > POOL:
        base = logits + row.to(tl.int64) * LOGITS_STRIDE
        lo = _sample_floor(base, n, TARGET, SCRATCH, SPLITS)
        if seg == 0:
            tl.store(floor + row, lo)
        offs = seg * SEGMENT + tl.arange(0, SEGMENT)
        x = tl.load(base + offs, offs < n, other=float("-inf"))
        count = tl.sum(((offs < n) & (x >= lo)).to(tl.int32))
        tl.store(segment_counts + row * SEGMENTS + seg, count)


@triton.jit
def _compact_kernel(
    logits,
    LOGITS_STRIDE,
    pool_lengths,
    floor,
    segment_counts,
    scratch_values,
    scratch_tokens,
    SEGMENTS: tl.constexpr,
    SEGMENTS_P2: tl.constexpr,
    SEGMENT: tl.constexpr,
    SCRATCH: tl.constexpr,
    POOL: tl.constexpr,
):
    """Logits >= floor, in token order, into the row's scratch."""
    row = tl.program_id(0)
    seg = tl.program_id(1)
    n = tl.load(pool_lengths + row)
    if (n > POOL) & (seg * SEGMENT < n):
        s = tl.arange(0, SEGMENTS_P2)
        before = tl.sum(tl.load(segment_counts + row * SEGMENTS + s, s < seg, other=0))
        if before < SCRATCH:
            lo = tl.load(floor + row)
            offs = seg * SEGMENT + tl.arange(0, SEGMENT)
            x = tl.load(logits + row.to(tl.int64) * LOGITS_STRIDE + offs, offs < n, other=float("-inf"))
            keep = (offs < n) & (x >= lo)
            at = before + tl.cumsum(keep.to(tl.int32), 0) - 1
            keep = keep & (at < SCRATCH)
            tl.store(scratch_values + row.to(tl.int64) * SCRATCH + at, x, keep)
            tl.store(scratch_tokens + row.to(tl.int64) * SCRATCH + at, offs, keep)


@triton.jit
def _pool_select_kernel(
    logits,
    LOGITS_STRIDE,
    pool_lengths,
    segment_counts,
    scratch_values,
    scratch_tokens,
    row_slot,
    pool,
    SEGMENTS: tl.constexpr,
    SEGMENTS_P2: tl.constexpr,
    SCRATCH: tl.constexpr,
    POOL: tl.constexpr,
    CHUNK: tl.constexpr,
    SPLITS: tl.constexpr,
    PASSES: tl.constexpr,
):
    """Exact top-POOL of each building row's scratch into its slot. A row
    whose sampled floor kept too few or too many logits selects from the
    whole row instead (correct, slower). Rows of at most POOL tokens keep
    them all."""
    row = tl.program_id(0)
    n = tl.load(pool_lengths + row)
    if (n > 0) & (n <= POOL):
        out = pool + tl.load(row_slot + row).to(tl.int64) * POOL
        for start in range(0, POOL, CHUNK):
            token = start + tl.arange(0, CHUNK)
            tl.store(out + token, tl.where(token < n, token, -1))
    if n > POOL:
        s = tl.arange(0, SEGMENTS_P2)
        kept = tl.sum(tl.load(segment_counts + row * SEGMENTS + s, s < SEGMENTS, other=0))
        out = pool + tl.load(row_slot + row).to(tl.int64) * POOL
        if (kept >= POOL) & (kept <= SCRATCH):
            _select_into(
                scratch_values + row.to(tl.int64) * SCRATCH,
                scratch_tokens + row.to(tl.int64) * SCRATCH,
                kept, out, POOL, CHUNK, SPLITS, PASSES, True,
            )
        else:
            _select_into(
                logits + row.to(tl.int64) * LOGITS_STRIDE,
                scratch_tokens, n, out, POOL, CHUNK, SPLITS, PASSES, False,
            )


@triton.jit
def _candidate_token(c, pool, slot, anchor, length, POOL: tl.constexpr):
    from_pool = c < POOL
    token = tl.where(
        from_pool,
        tl.load(pool + slot * POOL + c, from_pool & (c >= 0), other=-1),
        anchor + c - POOL,
    )
    valid = (c >= 0) & tl.where(from_pool, token >= 0, token < length)
    return token, valid


@triton.jit
def _candidate_logits_kernel(
    q,
    weights,
    cache,
    block_table,
    lengths,
    row_slot,
    candidate_lengths,
    pool,
    slot_anchor,
    out,
    TABLE_STRIDE,
    TABLE_COLS,
    WEIGHTS_STRIDE,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE: tl.constexpr,
    POOL: tl.constexpr,
    CANDIDATES: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    if tl.load(candidate_lengths + row) > 0:
        slot = tl.load(row_slot + row).to(tl.int64)
        anchor = tl.load(slot_anchor + slot)
        length = tl.load(lengths + row)
        c = tl.program_id(1) * TILE + tl.arange(0, TILE)
        token, valid = _candidate_token(c, pool, slot, anchor, length, POOL)
        valid = valid & (token // PAGE < TABLE_COLS)
        page = tl.load(
            block_table + row.to(tl.int64) * TABLE_STRIDE + token // PAGE,
            valid,
            other=0,
        ).to(tl.int64)
        page_base = cache + page * (PAGE * (HEAD_DIM + 4))
        in_page = token % PAGE
        d = tl.arange(0, HEAD_DIM)
        k = tl.load(
            page_base[:, None] + in_page[:, None] * HEAD_DIM + d[None, :],
            valid[:, None],
            other=0,
        ).to(tl.float8e4nv, bitcast=True)
        scale = tl.load(
            (page_base + PAGE * HEAD_DIM + in_page * 4).to(tl.pointer_type(tl.float32)),
            valid,
            other=0.0,
        )
        h = tl.arange(0, HEADS)
        query = tl.load(q + row.to(tl.int64) * HEADS * HEAD_DIM + h[:, None] * HEAD_DIM + d[None, :])
        # E4M3 values and their products are exact in fp16/fp32.
        score = tl.dot(
            query.to(tl.float16), tl.trans(k.to(tl.float16)), out_dtype=tl.float32
        )
        w = tl.load(weights + row.to(tl.int64) * WEIGHTS_STRIDE + h).to(tl.float32)
        logit = tl.sum(tl.maximum(score, 0.0) * w[:, None], axis=0) * scale
        tl.store(
            out + row.to(tl.int64) * CANDIDATES + c,
            tl.where(valid, logit, float("-inf")),
        )


@triton.jit
def _choose_kernel(
    candidate_topk,
    lengths,
    row_slot,
    pool,
    slot_anchor,
    out,
    TOPK: tl.constexpr,
    POOL: tl.constexpr,
):
    """Candidate positions back to token indices; -1 for unused entries."""
    row = tl.program_id(0)
    k = tl.arange(0, TOPK)
    slot = tl.load(row_slot + row)
    c = tl.load(candidate_topk + row * TOPK + k)
    safe = tl.maximum(slot, 0).to(tl.int64)
    token, valid = _candidate_token(
        c, pool, safe, tl.load(slot_anchor + safe), tl.load(lengths + row), POOL
    )
    tl.store(out + row * TOPK + k, tl.where(valid & (slot >= 0), token, -1))


class IndexerPool:
    """Per-layer pool slots, allocated before CUDA graph capture."""

    def __init__(self, slots: int, reuse_steps: int, min_length: int, device, pool_size: int = POOL):
        if not 1 <= reuse_steps < TAIL:
            raise ValueError(f"reuse_steps must be in [1, {TAIL - 1}]")
        if slots <= 0 or slots & (slots - 1):
            raise ValueError("slots must be a positive power of two")
        if pool_size < TOPK or pool_size & (pool_size - 1):
            raise ValueError(f"pool_size must be a power of two of at least {TOPK}")
        self.slots = slots
        self.pool_size = pool_size
        self.candidates = pool_size + TAIL
        self.scratch = 4 * pool_size
        self.reuse_steps = reuse_steps
        self.min_length = max(min_length, 1)
        i32 = dict(dtype=torch.int32, device=device)
        self.first_page = torch.full((slots,), -1, **i32)
        self.anchor = torch.zeros((slots,), **i32)
        self.anchor_page = torch.full((slots,), -1, **i32)
        self.pool = torch.full((slots, pool_size), -1, **i32)

    def reset(self) -> None:
        self.anchor.zero_()

    def supports(self, rows: int, block_table: torch.Tensor) -> bool:
        return 0 < rows <= self.slots and block_table.shape[0] == rows

    def plan(self, block_table: torch.Tensor, lengths: torch.Tensor):
        rows = lengths.numel()
        i32 = dict(dtype=torch.int32, device=lengths.device)
        row_slot = torch.empty((rows,), **i32)
        pool_lengths = torch.empty((rows,), **i32)
        candidate_lengths = torch.empty((rows,), **i32)
        _plan_kernel[(1,)](
            block_table,
            lengths,
            self.first_page,
            self.anchor,
            self.anchor_page,
            row_slot,
            pool_lengths,
            candidate_lengths,
            rows,
            block_table.stride(0),
            block_table.shape[1],
            self.reuse_steps,
            self.min_length,
            PAGE=PAGE,
            CANDIDATES=self.candidates,
            ROWS=max(triton.next_power_of_2(rows), 16),
            SLOTS=self.slots,
            num_warps=8,
        )
        return row_slot, pool_lengths, candidate_lengths

    def _build_pools(self, logits, pool_lengths, row_slot) -> None:
        rows, width = logits.shape
        segments = triton.cdiv(width, SEGMENT)
        floor = torch.empty((rows,), dtype=torch.float32, device=logits.device)
        segment_counts = torch.empty((rows, segments), dtype=torch.int32, device=logits.device)
        scratch_values = torch.empty((rows, self.scratch), dtype=torch.float32, device=logits.device)
        scratch_tokens = torch.empty((rows, self.scratch), dtype=torch.int32, device=logits.device)
        common = dict(SEGMENTS=segments, SEGMENT=SEGMENT, SCRATCH=self.scratch, POOL=self.pool_size)
        _count_kernel[(rows, segments)](
            logits, logits.stride(0), pool_lengths, floor, segment_counts,
            TARGET=2 * self.pool_size, SPLITS=16, num_warps=4, **common,
        )
        _compact_kernel[(rows, segments)](
            logits, logits.stride(0), pool_lengths, floor, segment_counts,
            scratch_values, scratch_tokens,
            SEGMENTS_P2=triton.next_power_of_2(segments), num_warps=4, **common,
        )
        _pool_select_kernel[(rows,)](
            logits, logits.stride(0), pool_lengths, segment_counts,
            scratch_values, scratch_tokens, row_slot, self.pool,
            SEGMENTS=segments, SEGMENTS_P2=triton.next_power_of_2(segments),
            SCRATCH=self.scratch, POOL=self.pool_size, CHUNK=2048, SPLITS=8, PASSES=5, num_warps=8,
        )

    def score_candidates(
        self, q_fp8, weights, cache, block_table, lengths, row_slot, candidate_lengths
    ) -> torch.Tensor:
        """[rows, candidates] logits of each row's pool, then its appended
        tokens; -inf for unused candidates. Rows with zero candidate length
        are left unwritten."""
        rows, heads, _ = q_fp8.shape
        logits = torch.empty((rows, self.candidates), dtype=torch.float32, device=q_fp8.device)
        _candidate_logits_kernel[(rows, self.candidates // CANDIDATE_TILE)](
            q_fp8,
            weights,
            cache,
            block_table,
            lengths,
            row_slot,
            candidate_lengths,
            self.pool,
            self.anchor,
            logits,
            block_table.stride(0),
            block_table.shape[1],
            weights.stride(0),
            HEADS=heads,
            HEAD_DIM=HEAD_DIM,
            PAGE=PAGE,
            POOL=self.pool_size,
            CANDIDATES=self.candidates,
            TILE=CANDIDATE_TILE,
            num_warps=4,
        )
        return logits

    def topk(
        self,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        cache: torch.Tensor,
        block_table: torch.Tensor,
        lengths: torch.Tensor,
        full_logits: Callable[[torch.Tensor], torch.Tensor],
        select_topk: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], None],
    ) -> torch.Tensor:
        """Top-2048 token indices per decode row.

        q_fp8: [rows, heads, 128] E4M3; weights: [rows, heads] fp32 (Indexer
        head weights with the query scale folded in, as for DeepGEMM).
        cache: the paged FP8 Indexer cache, any uint8 view of whole pages.
        full_logits(lengths) returns [rows, max_len] fp32 logits for rows
        with nonzero lengths; select_topk(logits, lengths, out, max_len)
        writes a top-2048 into out.
        """
        rows, heads, head_dim = q_fp8.shape
        assert head_dim == HEAD_DIM and weights.shape == (rows, heads)
        row_slot, pool_lengths, candidate_lengths = self.plan(block_table, lengths)
        # Rows rebuilding a pool score their full context; others have zero
        # length here and cost nothing.
        self._build_pools(full_logits(pool_lengths), pool_lengths, row_slot)
        candidate_logits = self.score_candidates(
            q_fp8, weights, cache, block_table, lengths, row_slot, candidate_lengths
        )
        candidate_topk = torch.empty((rows, TOPK), dtype=torch.int32, device=q_fp8.device)
        select_topk(candidate_logits, candidate_lengths, candidate_topk, self.candidates)
        result = torch.empty_like(candidate_topk)
        _choose_kernel[(rows,)](
            candidate_topk,
            lengths,
            row_slot,
            self.pool,
            self.anchor,
            result,
            TOPK=TOPK,
            POOL=self.pool_size,
        )
        return result
