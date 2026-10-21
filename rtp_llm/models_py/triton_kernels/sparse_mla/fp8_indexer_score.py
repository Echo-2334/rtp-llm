"""FP8 DSA score compatibility for DeepGEMM releases without legacy APIs.

The cache remains RTP's existing block-packed FP8 + FP32-scale format. FP8
values convert exactly to BF16 for tensor-core products, with FP32 reduction
and the existing FP32 head weights. No whole-cache repacking or format change
is needed. This path is selected only when the legacy DeepGEMM APIs are absent.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _score(
    Q,
    K,
    Scale,
    W,
    Table,
    Length,
    Start,
    End,
    Out,
    Q_ROW: tl.constexpr,
    W_ROW: tl.constexpr,
    K_ROW: tl.constexpr,
    TABLE_ROW: tl.constexpr,
    N: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    PAGE: tl.constexpr,
    PAGED: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    tokens = tl.program_id(1) * TILE + tl.arange(0, TILE)
    h = tl.arange(0, HEADS)
    d = tl.arange(0, DIM)
    q = tl.load(
        Q + row * Q_ROW + h[:, None] * DIM + d[None, :],
        (h[:, None] < HEADS) & (d[None, :] < DIM),
        0.0,
    ).to(tl.bfloat16)
    if PAGED:
        length = tl.load(Length + row)
        valid = (tokens < N) & (tokens < length)
        block = tl.load(Table + row * TABLE_ROW + tokens // PAGE, valid, 0)
        offset = tokens % PAGE
        valid = valid & (block >= 0)
        k = tl.load(
            (
                K
                + block[None, :].to(tl.int64) * K_ROW
                + offset[None, :] * DIM
                + d[:, None]
            ).to(tl.pointer_type(tl.float8e4nv)),
            valid[None, :] & (d[:, None] < DIM),
            0.0,
        ).to(tl.bfloat16)
        scale = tl.load(
            (K + block.to(tl.int64) * K_ROW + PAGE * DIM + offset * 4).to(
                tl.pointer_type(tl.float32)
            ),
            valid,
            0,
        )
    else:
        start = tl.load(Start + row)
        end = tl.load(End + row)
        valid = (tokens < N) & (tokens >= start) & (tokens < end)
        k = tl.load(
            K + tokens[None, :] * K_ROW + d[:, None],
            valid[None, :] & (d[:, None] < DIM),
            0.0,
        ).to(tl.bfloat16)
        scale = tl.load(Scale + tokens, valid, 0)
    dots = tl.dot(q, k, out_dtype=tl.float32)
    weights = tl.load(W + row * W_ROW + h, h < HEADS, 0).to(tl.float32)
    score = tl.sum(tl.maximum(dots, 0) * weights[:, None], 0) * scale
    tl.store(Out + row * N + tokens, tl.where(valid, score, -float("inf")), tokens < N)


def _validate(q, weights):
    if q.ndim != 3 or q.dtype != torch.float8_e4m3fn or not q.is_contiguous():
        raise ValueError("DSA Q must be contiguous FP8 [tokens,heads,dim]")
    if q.shape[1] not in (16, 32, 64) or q.shape[2] not in (64, 128):
        raise ValueError("DSA score supports 16/32/64 heads and 64/128 dimensions")
    if weights.shape != q.shape[:2] or weights.stride(-1) != 1:
        raise ValueError("DSA weights must have shape [tokens,heads]")


def fp8_paged_indexer_logits(q, cache, weights, lengths, block_table, max_context_len):
    _validate(q, weights)
    if (
        cache.ndim != 4
        or cache.dtype != torch.uint8
        or cache.shape[2:] != (1, q.shape[-1] + 4)
    ):
        raise ValueError("DSA cache must be block-packed uint8 [blocks,page,1,dim+4]")
    if not cache.is_contiguous():
        raise ValueError("DSA paged cache must be contiguous")
    if (
        block_table.shape[0] != q.shape[0]
        or max_context_len > block_table.shape[1] * cache.shape[1]
    ):
        raise ValueError("DSA block table does not cover requested context")
    out = torch.empty(
        (q.shape[0], max_context_len), dtype=torch.float32, device=q.device
    )
    if out.numel():
        _score[(q.shape[0], triton.cdiv(max_context_len, 64))](
            q,
            cache,
            cache,
            weights,
            block_table,
            lengths,
            lengths,
            lengths,
            out,
            Q_ROW=q.stride(0),
            W_ROW=weights.stride(0),
            K_ROW=cache.stride(0),
            TABLE_ROW=block_table.stride(0),
            N=max_context_len,
            HEADS=q.shape[1],
            DIM=q.shape[2],
            PAGE=cache.shape[1],
            PAGED=True,
            TILE=64,
            num_warps=4,
        )
    return out


def fp8_ragged_indexer_logits(q, kv, weights, starts, ends):
    _validate(q, weights)
    k, scale = kv
    if (
        k.ndim != 2
        or k.dtype != q.dtype
        or k.shape[1] != q.shape[2]
        or k.stride(-1) != 1
    ):
        raise ValueError("DSA K must be FP8 [tokens,dim]")
    if (
        scale.dtype != torch.float32
        or scale.numel() != k.shape[0]
        or not scale.is_contiguous()
    ):
        raise ValueError("DSA K scale must be contiguous FP32 [tokens]")
    out = torch.empty((q.shape[0], k.shape[0]), dtype=torch.float32, device=q.device)
    if out.numel():
        _score[(q.shape[0], triton.cdiv(k.shape[0], 64))](
            q,
            k,
            scale,
            weights,
            starts,
            starts,
            starts,
            ends,
            out,
            Q_ROW=q.stride(0),
            W_ROW=weights.stride(0),
            K_ROW=k.stride(0),
            TABLE_ROW=0,
            N=k.shape[0],
            HEADS=q.shape[1],
            DIM=q.shape[2],
            PAGE=1,
            PAGED=False,
            TILE=64,
            num_warps=4,
        )
    return out
