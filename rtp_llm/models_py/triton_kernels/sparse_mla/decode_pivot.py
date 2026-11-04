"""Stateless PIVOT for four-row MTP target verification, disabled by default.

Page tables and output indices remain request-local. Eligibility is evaluated
on the GPU; short, padded or nonconsecutive groups use full-context scoring.
"""

import os
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from .prefill_pivot import PrefillPivotConfig
from .prefill_pivot_kernels import recall_candidates, _candidates
from .decode_pivot_topk import radix_topk


@dataclass(frozen=True)
class DecodePivotConfig(PrefillPivotConfig):
    candidates: int = 16384

    def validate(self, topk):
        super().validate(topk)
        if self.group_size != 4:
            raise ValueError(
                "Decode PIVOT requires MTP3 / four target verification rows"
            )

    @classmethod
    def from_env(cls, topk):
        mode = os.environ.get("GLM5_DECODE_PIVOT_MODE", "off").strip().lower()
        if mode == "off":
            return None
        if mode != "refine":
            raise ValueError("GLM5_DECODE_PIVOT_MODE must be off or refine")
        cfg = cls(
            **{
                key: int(
                    os.environ.get("GLM5_DECODE_PIVOT_" + key.upper(), str(default))
                )
                for key, default in (
                    ("group_size", 4),
                    ("candidates", 16384),
                    ("local_window", 4),
                    ("min_kv_len", 32768),
                )
            }
        )
        cfg.validate(topk)
        return cfg


@triton.jit
def _proxy_plan(
    Q,
    QS,
    W,
    L,
    GOOD,
    PL,
    FL,
    Z,
    PQ,
    PW,
    H: tl.constexpr,
    D: tl.constexpr,
    MIN: tl.constexpr,
    WINDOW: tl.constexpr,
    MAX: tl.constexpr,
):
    b, head = tl.program_id(0), tl.program_id(1)
    j = tl.arange(0, 4)
    lens = tl.load(L + b * 4 + j)
    first = tl.load(L + b * 4)
    good = (
        (first >= MIN)
        & (first + 3 <= MAX)
        & (tl.sum((lens == first + j).to(tl.int32), 0) == 4)
    )
    if head == 0:
        tl.store(GOOD + b, good)
        tl.store(PL + b, tl.where(good, first - WINDOW, 0))
        tl.store(FL + b * 4 + j, tl.where(good, 0, lens))
        tl.store(Z + b, 0)
    d = tl.arange(0, D)
    scale = tl.maximum(tl.load(QS + (b * 4 + j) * H + head), 1.0e-30)
    q = tl.load(Q + ((b * 4 + j[:, None]) * H + head) * D + d[None, :]).to(tl.float32)
    w = tl.load(W + (b * 4 + j) * H + head)
    mean_q = tl.sum(q * scale[:, None], 0) / 4
    mean_w = tl.sum(w / scale, 0) / 4
    proxy_scale = tl.exp2(
        tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(mean_q)), 1.0e-4) / 448.0))
    )
    tl.store(PQ + (b * H + head) * D + d, mean_q / proxy_scale)
    tl.store(PW + b * H + head, mean_w * proxy_scale)


def _recall(logits, starts, ends, lengths, config):
    nr = config.candidates - config.local_window - config.group_size + 1
    # Large batches with a full 1M score matrix favor torch's parallel selector.
    # This routing uses only captured shapes, never host reads of GPU lengths.
    if nr <= 4096 or (logits.shape[0] >= 4 and logits.shape[1] > 262144):
        return recall_candidates(logits, starts, ends, lengths, config)
    ids = radix_topk(logits, starts, ends, nr)
    out = torch.empty(
        (logits.shape[0], config.candidates), dtype=torch.int32, device=logits.device
    )
    _candidates[(logits.shape[0], triton.cdiv(config.candidates, 256))](
        ids,
        logits,
        starts,
        ends,
        lengths,
        out,
        nr,
        config.candidates,
        config.group_size,
        config.local_window,
        logits.stride(0),
        logits.shape[1],
        False,
        256,
    )
    return out


@triton.jit
def _score(
    Q,
    W,
    CACHE,
    TABLE,
    L,
    IDS,
    GOOD,
    OUT,
    H: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    TS: tl.constexpr,
    TC: tl.constexpr,
    C: tl.constexpr,
    TILE: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1) * TILE + tl.arange(0, TILE)
    token = tl.load(IDS + b * C + c, c < C, -1)
    valid = (c < C) & (token >= 0) & (token < TC * PAGE) & (tl.load(GOOD + b) != 0)
    page = tl.load(TABLE + b * TS + token // PAGE, valid, 0).to(tl.int64)
    base = CACHE + page * PAGE * (D + 4)
    pos = token % PAGE
    d = tl.arange(0, D)
    key = tl.load(base[:, None] + pos[:, None] * D + d[None, :], valid[:, None], 0).to(
        tl.float8e4nv, bitcast=True
    )
    sf = tl.load((base + PAGE * D + pos * 4).to(tl.pointer_type(tl.float32)), valid, 0)
    gh = tl.arange(0, 4 * H)
    q = tl.load(Q + (b.to(tl.int64) * 4 * H + gh[:, None]) * D + d[None, :])
    dot = tl.dot(q.to(tl.float16), tl.trans(key.to(tl.float16)), out_dtype=tl.float32)
    w = tl.load(W + b * 4 * H + gh)
    score = (
        tl.sum((tl.maximum(dot, 0.0) * w[:, None]).reshape(4, H, TILE), 1) * sf[None, :]
    )
    r = b * 4 + tl.arange(0, 4)
    lens = tl.load(L + r)
    causal = valid[None, :] & (token[None, :] < lens[:, None])
    tl.store(
        OUT + r[:, None].to(tl.int64) * C + c[None, :],
        tl.where(causal, score, -float("inf")),
        (c < C)[None, :],
    )


@triton.jit
def _merge(
    SELECT,
    SCORES,
    IDS,
    FULL,
    GOOD,
    L,
    OUT,
    K: tl.constexpr,
    C: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    j = tl.program_id(1) * B + tl.arange(0, B)
    slot = tl.load(SELECT + row * K + j, j < K, -1)
    valid = (slot >= 0) & (slot < C) & (j < K)
    token = tl.load(IDS + row // 4 * C + slot, valid, -1)
    score = tl.load(SCORES + row.to(tl.int64) * C + slot, valid, -float("inf"))
    valid &= (token >= 0) & (token < tl.load(L + row)) & (tl.abs(score) < float("inf"))
    fallback = tl.load(FULL + row * K + j, j < K, -1)
    value = tl.where(
        tl.load(GOOD + row // 4) != 0, tl.where(valid, token, -1), fallback
    )
    tl.store(OUT + row * K + j, value, j < K)


def decode_pivot_topk(
    q,
    q_scale,
    weights,
    cache,
    table,
    lengths,
    page_size,
    topk,
    config,
    full_logits,
    select_topk,
):
    """Callbacks use fresh paged scheduler metadata and the native TopK backend.

    full_logits(q, weights, lengths_2d, table) returns flattened query logits.
    No persistent reuse state or host-side length reads are involved.
    """
    n, heads, dim = q.shape
    if n != table.shape[0] * 4 or dim != 128:
        raise ValueError("Expected exactly four queries per request and head dim 128")
    config.validate(topk)
    q = q.contiguous()
    device = q.device
    c, batches = config.candidates, table.shape[0]
    lengths = lengths.reshape(-1).contiguous()
    good = torch.empty(batches, dtype=torch.int32, device=device)
    proxy_len = torch.empty((batches, 1), dtype=torch.int32, device=device)
    full_len = torch.empty_like(lengths)
    max_len = table.shape[1] * page_size
    scales = q_scale.reshape(n, heads).contiguous()
    w = weights.reshape(n, heads).contiguous()
    pq = torch.empty((batches, heads, dim), dtype=q.dtype, device=device)
    pw = torch.empty((batches, heads), dtype=torch.float32, device=device)
    zeros = torch.empty(batches, dtype=torch.int32, device=device)
    _proxy_plan[(batches, heads)](
        q,
        scales,
        w,
        lengths,
        good,
        proxy_len,
        full_len,
        zeros,
        pq,
        pw,
        heads,
        dim,
        config.min_kv_len,
        config.local_window,
        max_len,
    )
    logits = full_logits(pq, pw, proxy_len, table)
    candidates = _recall(logits, zeros, proxy_len.reshape(-1), lengths, config)
    del logits
    if (
        batches >= 4
        and page_size == 64
        and os.environ.get("RTP_LLM_INDEXER_NATIVE_MTP", "1") != "0"
        and c % page_size == 0
        and torch.cuda.get_device_capability(device)[0] >= 10
    ):
        from .decode_pivot_score import score_packed_candidates

        scores = score_packed_candidates(q, w, cache, table, lengths, candidates, good)
    else:
        scores = torch.empty((n, c), dtype=torch.float32, device=device)
        _score[(batches, triton.cdiv(c, 128))](
            q,
            w,
            cache,
            table,
            lengths,
            candidates,
            good,
            scores,
            heads,
            dim,
            page_size,
            table.stride(0),
            table.shape[1],
            c,
            128,
            num_warps=4,
        )
    selected = torch.empty((n, topk), dtype=torch.int32, device=device)
    select_topk(scores, torch.full_like(lengths, c), selected, c)
    # Zero lengths skip full-K scoring for eligible requests, including mixed batches.
    fallback_logits = full_logits(q, w, full_len.reshape(batches, 4), table)
    fallback = torch.empty_like(selected)
    select_topk(fallback_logits, full_len, fallback, max_len)
    out = torch.empty_like(selected)
    _merge[(n, triton.cdiv(topk, 256))](
        selected, scores, candidates, fallback, good, lengths, out, topk, c, 256
    )
    return out
