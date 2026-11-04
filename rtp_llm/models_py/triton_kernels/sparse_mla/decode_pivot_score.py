"""Pack paged candidate keys for native four-query FP8 MQA scoring."""

import torch
import triton
import deep_gemm
import triton.language as tl


@triton.jit
def _pack(
    CACHE,
    TABLE,
    IDS,
    GOOD,
    PACK,
    TNEW,
    LNEW,
    C: tl.constexpr,
    PAGE: tl.constexpr,
    TC: tl.constexpr,
    TS: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b, t = tl.program_id(0), tl.program_id(1)
    c = t * BLOCK + tl.arange(0, BLOCK)
    d = tl.arange(0, D)
    ids = tl.load(IDS + b * C + c, c < C, -1)
    live = (c < C) & (ids >= 0) & (ids < TC * PAGE) & (tl.load(GOOD + b) != 0)
    page = tl.load(TABLE + b * TS + ids // PAGE, live, 0).to(tl.int64)
    pos = ids % PAGE
    key = tl.load(
        CACHE + page[:, None] * PAGE * (D + 4) + pos[:, None] * D + d[None, :],
        live[:, None],
        0,
    )
    sf = tl.load(
        (CACHE + page * PAGE * (D + 4) + PAGE * D + pos * 4).to(
            tl.pointer_type(tl.float32)
        ),
        live,
        0.0,
    )
    destpage = b * (C // PAGE) + c // PAGE
    destpos = c % PAGE
    tl.store(
        PACK + destpage[:, None] * PAGE * (D + 4) + destpos[:, None] * D + d[None, :],
        key,
        (c < C)[:, None],
    )
    tl.store(
        (PACK + destpage * PAGE * (D + 4) + PAGE * D + destpos * 4).to(
            tl.pointer_type(tl.float32)
        ),
        sf,
        c < C,
    )
    if t == 0:
        j = tl.arange(0, 4)
        tl.store(LNEW + b * 4 + j, C)
    for j in range(BLOCK // PAGE):
        p = t * (BLOCK // PAGE) + j
        if p < C // PAGE:
            tl.store(TNEW + b * (C // PAGE) + p, b * (C // PAGE) + p)


@triton.jit
def _mask(SCORE, IDS, L, GOOD, C: tl.constexpr, S: tl.constexpr, BLOCK: tl.constexpr):
    row, t = tl.program_id(0), tl.program_id(1)
    i = t * BLOCK + tl.arange(0, BLOCK)
    token = tl.load(IDS + row // 4 * C + i, i < C, -1)
    bad = (token < 0) | (token >= tl.load(L + row)) | (tl.load(GOOD + row // 4) == 0)
    tl.store(SCORE + row * S + i, -float("inf"), (i < C) & bad)


def score_packed_candidates(q, w, cache, table, lens, ids, good):
    """Score an aligned candidate pool using independent scratch pages per request.

    The caller requires 64-token pages, 32 heads, dim128 and C divisible by64.
    Invalid IDs pack zero keys/scales; masking afterwards restores -inf for
    invalid, future and ineligible positions before the existing final TopK.
    FP8 dot/reduction follows DeepGEMM and may round differently from Triton.
    """
    b, c = ids.shape
    p = 64
    d = 128
    packed = torch.empty((b * c // p, p, 1, d + 4), device=q.device, dtype=torch.uint8)
    newtable = torch.empty((b, c // p), device=q.device, dtype=torch.int32)
    newlens = torch.empty((b, 4), device=q.device, dtype=torch.int32)
    _pack[(b, triton.cdiv(c, 64))](
        cache,
        table,
        ids,
        good,
        packed,
        newtable,
        newlens,
        c,
        p,
        table.shape[1],
        table.stride(0),
        d,
        64,
        num_warps=4,
    )
    meta = deep_gemm.get_paged_mqa_logits_metadata(newlens, p, deep_gemm.get_num_sms())
    out = deep_gemm.fp8_paged_mqa_logits(
        q.view(b, 4, 32, 128), packed, w, newlens, newtable, meta, c, clean_logits=False
    )
    _mask[(b * 4, triton.cdiv(c, 256))](out, ids, lens, good, c, out.stride(0), 256)
    return out
