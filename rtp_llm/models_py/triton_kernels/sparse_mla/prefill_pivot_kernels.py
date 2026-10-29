"""GPU proxy construction and grouped candidate scoring for prefill PIVOT."""

import torch
import triton
import triton.language as tl


@triton.jit
def _proxy(Q, QS, W, PQ, PW, H: tl.constexpr, D: tl.constexpr, G: tl.constexpr):
    group, head = tl.program_id(0), tl.program_id(1)
    r = group * G + tl.arange(0, G)
    d = tl.arange(0, D)
    scale = tl.load(QS + r * H + head)
    q = tl.load(Q + (r[:, None] * H + head) * D + d[None, :]).to(tl.float32)
    w = tl.load(W + r * H + head)
    mean_q = tl.sum(q * scale[:, None], axis=0) / G
    mean_w = tl.sum(w / scale, axis=0) / G
    # Power-of-two FP8 scale, matching the deployed UE8M0 Q representation.
    proxy_scale = tl.exp2(
        tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(mean_q)), 1.0e-4) / 448.0))
    )
    tl.store(PQ + (group * H + head) * D + d, mean_q / proxy_scale)
    tl.store(PW + group * H + head, mean_w * proxy_scale)


def build_proxy(q, q_scale, weights, group_size):
    rows, heads, dim = q.shape
    proxy_q = torch.empty(
        (rows // group_size, heads, dim), dtype=q.dtype, device=q.device
    )
    proxy_w = torch.empty(
        (rows // group_size, heads), dtype=torch.float32, device=q.device
    )
    _proxy[(rows // group_size, heads)](
        q, q_scale, weights, proxy_q, proxy_w, heads, dim, group_size
    )
    return proxy_q, proxy_w


@triton.jit
def _mask(
    Logits, KS, KE, N: tl.constexpr, S0: tl.constexpr, S1: tl.constexpr, B: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.program_id(1) * B + tl.arange(0, B)
    start, end = tl.load(KS + row), tl.load(KE + row)
    tl.store(
        Logits + row.to(tl.int64) * S0 + col * S1,
        float("-inf"),
        (col < N) & ((col < start) | (col >= end)),
    )


def mask_logits_(logits, ks, ke):
    _mask[(logits.shape[0], triton.cdiv(logits.shape[1], 1024))](
        logits, ks, ke, logits.shape[1], logits.stride(0), logits.stride(1), 1024
    )


@triton.jit
def _score(
    Q,
    W,
    K,
    SF,
    C,
    KS,
    KE,
    OUT,
    H: tl.constexpr,
    D: tl.constexpr,
    G: tl.constexpr,
    NC: tl.constexpr,
    NK: tl.constexpr,
    TILE: tl.constexpr,
):
    group = tl.program_id(0)
    c = tl.program_id(1) * TILE + tl.arange(0, TILE)
    token = tl.load(C + group * NC + c, c < NC, other=-1)
    valid = (c < NC) & (token >= 0) & (token < NK)
    d = tl.arange(0, D)
    key = tl.load(
        K.to(tl.pointer_type(tl.uint8)) + token[:, None].to(tl.int64) * D + d[None, :],
        valid[:, None],
        other=0,
    ).to(tl.float8e4nv, bitcast=True)
    scale = tl.load(SF + token, valid, other=0)
    gh = tl.arange(0, G * H)
    query = tl.load(Q + (group.to(tl.int64) * G * H + gh[:, None]) * D + d[None, :])
    dot = tl.dot(
        query.to(tl.float16), tl.trans(key.to(tl.float16)), out_dtype=tl.float32
    )
    weight = tl.load(W + group * G * H + gh)
    per_head = (tl.maximum(dot, 0.0) * weight[:, None]).reshape(G, H, TILE)
    logits = tl.sum(per_head, axis=1) * scale[None, :]
    row = group * G + tl.arange(0, G)
    start, end = tl.load(KS + row), tl.load(KE + row)
    causal = (
        valid[None, :]
        & (token[None, :] >= start[:, None])
        & (token[None, :] < end[:, None])
    )
    tl.store(
        OUT + row[:, None].to(tl.int64) * NC + c[None, :],
        tl.where(causal, logits, float("-inf")),
        c[None, :] < NC,
    )


def _score_candidates_reference(q, weights, kv, candidates, ks, ke, group_size):
    rows, heads, dim = q.shape
    k, scale = kv
    scores = torch.empty(
        (rows, candidates.shape[1]), dtype=torch.float32, device=q.device
    )
    _score[(rows // group_size, triton.cdiv(candidates.shape[1], 64))](
        q,
        weights,
        k,
        scale,
        candidates,
        ks,
        ke,
        scores,
        heads,
        dim,
        group_size,
        candidates.shape[1],
        k.shape[0],
        64,
        num_warps=8,
    )
    return scores


@triton.jit
def _plan(
    KS,
    KE,
    OFF,
    GROUPS,
    FALLBACK,
    COUNTS,
    N: tl.constexpr,
    G: tl.constexpr,
    MIN_KV: tl.constexpr,
    B: tl.constexpr,
):
    group = tl.arange(0, B // G)
    row = group[:, None] * G + tl.arange(0, G)[None, :]
    ks = tl.load(KS + row, row < N, 0)
    ke = tl.load(KE + row, row < N, 0)
    off = tl.load(OFF + row, row < N, 0)
    first_ks = tl.load(KS + group * G, group * G < N, 0)
    first_ke = tl.load(KE + group * G, group * G < N, 0)
    first_off = tl.load(OFF + group * G, group * G < N, 0)
    good = (
        (row < N)
        & (ks == first_ks[:, None])
        & (ke == first_ke[:, None] + tl.arange(0, G)[None, :])
        & (off == first_off[:, None])
    )
    valid = (tl.sum(good.to(tl.int32), 1) == G) & (first_ke - first_ks >= MIN_KV)
    pos = tl.cumsum(valid.to(tl.int32)) - 1
    tl.store(GROUPS + pos[:, None] * G + tl.arange(0, G)[None, :], row, valid[:, None])
    fallback = ((~valid[:, None]) & (row < N)).reshape(B)
    fpos = tl.cumsum(fallback.to(tl.int32)) - 1
    tl.store(FALLBACK + fpos, row.reshape(B), fallback)
    tl.store(COUNTS, tl.sum(valid.to(tl.int32), 0))
    tl.store(COUNTS + 1, tl.sum(fallback.to(tl.int32), 0))


def plan_groups_cuda(ks, ke, offsets, config):
    n, g = ks.numel(), config.group_size
    groups = torch.empty((n // g, g), dtype=torch.int64, device=ks.device)
    fallback = torch.empty(n, dtype=torch.int64, device=ks.device)
    counts = torch.empty(2, dtype=torch.int32, device=ks.device)
    if n == 0:
        return groups, fallback
    _plan[(1,)](
        ks,
        ke,
        offsets,
        groups,
        fallback,
        counts,
        n,
        g,
        config.min_kv_len,
        max(g, triton.next_power_of_2(n)),
    )
    # One explicit synchronization; no GPU nonzero or boolean-index compaction.
    ng, nf = counts.cpu().tolist()
    return groups[:ng], fallback[:nf]


@triton.jit
def _prepare_rows(
    Q,
    QS,
    W,
    ROWS,
    KS,
    KE,
    OQ,
    OQS,
    OW,
    RKS,
    RKE,
    PKS,
    PKE,
    ZERO,
    LENGTH,
    H: tl.constexpr,
    D: tl.constexpr,
    G: tl.constexpr,
    C: tl.constexpr,
    WINDOW: tl.constexpr,
):
    r, h = tl.program_id(0), tl.program_id(1)
    src = tl.load(ROWS + r)
    d = tl.arange(0, D)
    q = tl.load(Q + (src * H + h) * D + d)
    tl.store(OQ + (r * H + h) * D + d, q)
    tl.store(OQS + r * H + h, tl.load(QS + src * H + h))
    tl.store(OW + r * H + h, tl.load(W + src * H + h))
    if h == 0:
        lo, hi = tl.load(KS + src), tl.load(KE + src)
        tl.store(RKS + r, lo)
        tl.store(RKE + r, hi)
        tl.store(ZERO + r, 0)
        tl.store(LENGTH + r, C)
        if r % G == 0:
            tl.store(PKS + r // G, lo)
            tl.store(PKE + r // G, tl.maximum(lo, hi - WINDOW))


def prepare_rows(q, qs, w, groups, ks, ke, config):
    g, h, d = config.group_size, q.shape[1], q.shape[2]
    ng, nr = groups.shape[0], groups.numel()
    oq = torch.empty((nr, h, d), dtype=q.dtype, device=q.device)
    oqs = torch.empty((nr, h), dtype=qs.dtype, device=q.device)
    ow = torch.empty((nr, h), dtype=w.dtype, device=q.device)
    meta = torch.empty(4 * nr + 2 * ng, dtype=torch.int32, device=q.device)
    rks, rke, zero, length = [meta[i * nr : (i + 1) * nr] for i in range(4)]
    pks, pke = meta[4 * nr : 4 * nr + ng], meta[4 * nr + ng :]
    _prepare_rows[(nr, h)](
        q,
        qs,
        w,
        groups,
        ks,
        ke,
        oq,
        oqs,
        ow,
        rks,
        rke,
        pks,
        pke,
        zero,
        length,
        h,
        d,
        g,
        config.candidates,
        config.local_window,
    )
    return oq, oqs, ow, rks, rke, pks, pke, zero, length


@triton.jit
def _candidates(
    IDS,
    LOGITS,
    PKS,
    PKE,
    RKE,
    OUT,
    NR: tl.constexpr,
    C: tl.constexpr,
    G: tl.constexpr,
    WINDOW: tl.constexpr,
    LS: tl.constexpr,
    NK: tl.constexpr,
    RELATIVE: tl.constexpr,
    B: tl.constexpr,
):
    group = tl.program_id(0)
    c = tl.program_id(1) * B + tl.arange(0, B)
    start, end = tl.load(PKS + group), tl.load(PKE + group)
    raw = tl.load(IDS + group * NR + c, c < NR, -1)
    token = raw + start if RELATIVE else raw
    valid = (raw >= 0) & (token >= start) & (token < end) & (token < NK)
    score = tl.load(
        LOGITS + group.to(tl.int64) * LS + token, (c < NR) & valid, -float("inf")
    )
    valid = valid & (score == score) & (tl.abs(score) != float("inf"))
    window_start = tl.load(RKE + group * G) - WINDOW
    local = window_start + c - NR
    value = tl.where(
        c < NR,
        tl.where(valid, token, -1),
        tl.where((local >= start) & (local < NK), local, -1),
    )
    tl.store(OUT + group * C + c, value, c < C)


def recall_candidates(logits, pks, pke, row_ke, config):
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    ng = logits.shape[0]
    nr = config.candidates - config.local_window - config.group_size + 1
    # Keep the previous general path for larger candidate budgets. The native
    # op uses dynamic shared memory proportional to K; 4089 is validated here.
    relative = nr <= 4096
    if relative:
        ids = torch.empty((ng, nr), dtype=torch.int32, device=logits.device)
        rtp_llm_ops.dsv4_top_k_per_row_prefill(
            logits, pks, pke, ids, ng, logits.stride(0), logits.stride(1), nr, True
        )
    else:
        mask_logits_(logits, pks, pke)
        _, ids = torch.topk(logits, nr, dim=1, sorted=False)
    out = torch.empty((ng, config.candidates), dtype=torch.int32, device=logits.device)
    _candidates[(ng, triton.cdiv(config.candidates, 256))](
        ids,
        logits,
        pks,
        pke,
        row_ke,
        out,
        nr,
        config.candidates,
        config.group_size,
        config.local_window,
        logits.stride(0),
        logits.shape[1],
        relative,
        256,
    )
    return out


@triton.jit
def _pack_keys(
    K,
    SF,
    IDS,
    OUT,
    OSF,
    STARTS,
    ENDS,
    C: tl.constexpr,
    PAD: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    NK: tl.constexpr,
    B: tl.constexpr,
):
    group = tl.program_id(0)
    c = tl.program_id(1) * B + tl.arange(0, B)
    token = tl.load(IDS + group * C + c, c < C, -1)
    valid = (c < C) & (token >= 0) & (token < NK)
    d = tl.arange(0, D)
    key = tl.load(K + token[:, None].to(tl.int64) * D + d[None, :], valid[:, None], 0)
    sf = tl.load(SF + token, valid, 0)
    idx = group * PAD + c
    tl.store(OUT + idx[:, None].to(tl.int64) * D + d[None, :], key, (c < PAD)[:, None])
    tl.store(OSF + idx, sf, c < PAD)
    if tl.program_id(1) == 0:
        r = group * G + tl.arange(0, G)
        tl.store(STARTS + r, group * PAD)
        tl.store(ENDS + r, group * PAD + C)


@triton.jit
def _mask_candidates(
    X,
    IDS,
    KS,
    KE,
    C: tl.constexpr,
    G: tl.constexpr,
    STR: tl.constexpr,
    NK: tl.constexpr,
    B: tl.constexpr,
):
    r = tl.program_id(0)
    c = tl.program_id(1) * B + tl.arange(0, B)
    token = tl.load(IDS + r // G * C + c, c < C, -1)
    lo, hi = tl.load(KS + r), tl.load(KE + r)
    invalid = (token < 0) | (token >= NK) | (token < lo) | (token >= hi)
    # Only invalid slots need writing. Avoid reading/writing the whole matrix.
    tl.store(X + r.to(tl.int64) * STR + c, -float("inf"), (c < C) & invalid)


def score_candidates(q, weights, kv, candidates, ks, ke, group_size):
    import deep_gemm

    # SM100 ragged MQA schedules four queries together for 32 heads. A group
    # of two would mix two different packed K origins in one compressed tile.
    if group_size != 4 or torch.cuda.get_device_capability(q.device)[0] != 10:
        return _score_candidates_reference(
            q, weights, kv, candidates, ks, ke, group_size
        )
    ng, c = candidates.shape
    pad = triton.cdiv(c, 256) * 256
    k, sf = kv
    packed = torch.empty((ng * pad, k.shape[1]), dtype=k.dtype, device=k.device)
    packed_sf = torch.empty(ng * pad, dtype=sf.dtype, device=k.device)
    bounds = torch.empty((2, q.shape[0]), dtype=torch.int32, device=q.device)
    starts, ends = bounds[0], bounds[1]
    _pack_keys[(ng, triton.cdiv(pad, 64))](
        k.view(torch.uint8),
        sf,
        candidates,
        packed.view(torch.uint8),
        packed_sf,
        starts,
        ends,
        c,
        pad,
        group_size,
        k.shape[1],
        k.shape[0],
        64,
    )
    scores = deep_gemm.fp8_mqa_logits(
        q,
        (packed, packed_sf),
        weights,
        starts,
        ends,
        clean_logits=False,
        max_seqlen_k=pad,
    )[:, :c]
    _mask_candidates[(q.shape[0], triton.cdiv(c, 256))](
        scores, candidates, ks, ke, c, group_size, scores.stride(0), k.shape[0], 256
    )
    return scores


@triton.jit
def _remap(
    SELECTED,
    SCORES,
    CANDIDATES,
    ROWS,
    KS,
    KE,
    OFF,
    OUT,
    TOPK: tl.constexpr,
    C: tl.constexpr,
    G: tl.constexpr,
    SS: tl.constexpr,
    APPLY_OFFSET: tl.constexpr,
    B: tl.constexpr,
):
    r = tl.program_id(0)
    j = tl.program_id(1) * B + tl.arange(0, B)
    slot = tl.load(SELECTED + r * TOPK + j, j < TOPK, -1)
    valid = (j < TOPK) & (slot >= 0) & (slot < C)
    token = tl.load(CANDIDATES + r // G * C + slot, valid, -1)
    score = tl.load(SCORES + r.to(tl.int64) * SS + slot, valid, -float("inf"))
    lo, hi = tl.load(KS + r), tl.load(KE + r)
    valid = (
        valid
        & (token >= lo)
        & (token < hi)
        & (score == score)
        & (tl.abs(score) != float("inf"))
    )
    dest = tl.load(ROWS + r)
    if APPLY_OFFSET:
        token += tl.load(OFF + dest)
    tl.store(OUT + dest * TOPK + j, tl.where(valid, token, -1), j < TOPK)


def remap_output(
    selected, scores, candidates, groups, ks, ke, offsets, out, apply_offsets
):
    _remap[(selected.shape[0], triton.cdiv(selected.shape[1], 256))](
        selected,
        scores,
        candidates,
        groups,
        ks,
        ke,
        offsets,
        out,
        selected.shape[1],
        candidates.shape[1],
        groups.shape[1],
        scores.stride(0),
        apply_offsets,
        256,
    )


@triton.jit
def _scatter(
    IN, ROWS, OFF, OUT, TOPK: tl.constexpr, APPLY_OFFSET: tl.constexpr, B: tl.constexpr
):
    r = tl.program_id(0)
    j = tl.program_id(1) * B + tl.arange(0, B)
    dest = tl.load(ROWS + r)
    value = tl.load(IN + r * TOPK + j, j < TOPK, -1)
    if APPLY_OFFSET:
        value = tl.where(value >= 0, value + tl.load(OFF + dest), value)
    tl.store(OUT + dest * TOPK + j, value, j < TOPK)


def scatter_output(indices, rows, offsets, out, apply_offsets):
    _scatter[(indices.shape[0], triton.cdiv(indices.shape[1], 256))](
        indices, rows, offsets, out, indices.shape[1], apply_offsets, 256
    )
