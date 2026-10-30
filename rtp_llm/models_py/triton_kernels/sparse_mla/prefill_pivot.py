"""Opt-in, stateless PIVOT-Refine for eager prefill (including CP).

Candidate indices address the assembled ragged K tensor. Output indices are
relative to each row's K start, matching native ragged TopK. The output writer
applies the per-row offset exactly once, optionally fused with remap/scatter.
Boundary/short rows use the caller's original full-context Indexer kernels.
"""

import os
from dataclasses import dataclass
from typing import Callable, Optional

import torch


@dataclass(frozen=True)
class PrefillPivotConfig:
    group_size: int = 4
    candidates: int = 4096
    local_window: int = 4
    min_kv_len: int = 32768

    def validate(self, topk: int) -> None:
        if self.group_size not in (2, 4):
            raise ValueError("PIVOT prefill group size must be 2 or 4")
        if self.local_window < self.group_size:
            raise ValueError("PIVOT local window must cover the entire query group")
        if self.candidates <= max(topk, self.local_window + self.group_size - 1):
            raise ValueError("PIVOT candidate budget must exceed TopK and window union")
        if self.candidates > 16384:
            raise ValueError("PIVOT prefill candidate budget must be <= 16384")
        if self.min_kv_len < self.candidates:
            raise ValueError("PIVOT minimum KV length must be >= candidate budget")

    @classmethod
    def from_env(cls, topk: int) -> Optional["PrefillPivotConfig"]:
        mode = os.environ.get("GLM5_PREFILL_PIVOT_MODE", "off").strip().lower()
        if mode == "off":
            return None
        if mode != "refine":
            raise ValueError("GLM5_PREFILL_PIVOT_MODE must be 'off' or 'refine'")
        config = cls(
            group_size=int(os.environ.get("GLM5_PREFILL_PIVOT_GROUP_SIZE", "4")),
            candidates=int(os.environ.get("GLM5_PREFILL_PIVOT_CANDIDATES", "4096")),
            local_window=int(os.environ.get("GLM5_PREFILL_PIVOT_LOCAL_WINDOW", "4")),
            min_kv_len=int(os.environ.get("GLM5_PREFILL_PIVOT_MIN_KV_LEN", "32768")),
        )
        config.validate(topk)
        return config


def plan_groups(ks, ke, offsets, config):
    """Compact complete contiguous groups; return other rows for native scoring.

    Equal ks/offset identifies a request in the assembled ragged K tensor;
    consecutive exclusive ke identifies adjacent logical positions. In
    particular, a CP zigzag join cannot become a group. GPU compaction returns
    two counts in one host synchronization; this path remains eager-only.
    """
    if ks.is_cuda:
        from .prefill_pivot_kernels import plan_groups_cuda

        return plan_groups_cuda(ks, ke, offsets, config)
    g = config.group_size
    n = ks.numel()
    complete = n // g * g
    rows = torch.arange(complete, device=ks.device).reshape(-1, g)
    starts = ks[:complete].reshape(-1, g)
    ends = ke[:complete].reshape(-1, g)
    off = offsets[:complete].reshape(-1, g)
    consecutive = torch.arange(g, device=ks.device)
    valid = (
        (starts == starts[:, :1]).all(1)
        & (off == off[:, :1]).all(1)
        & (ends == ends[:, :1] + consecutive).all(1)
        & ((ends[:, 0] - starts[:, 0]) >= config.min_kv_len)
    )
    groups = rows[valid]
    fallback_mask = torch.ones(n, dtype=torch.bool, device=ks.device)
    fallback_mask[groups.flatten()] = False
    return groups, fallback_mask.nonzero().flatten()


def pivot_topk(
    q: torch.Tensor,
    q_scale: torch.Tensor,
    weights: torch.Tensor,
    kv: tuple,
    ks: torch.Tensor,
    ke: torch.Tensor,
    offsets: torch.Tensor,
    topk: int,
    config: PrefillPivotConfig,
    full_logits: Callable,
    select_topk: Callable,
    output: Optional[torch.Tensor] = None,
    apply_offsets: bool = False,
) -> torch.Tensor:
    """One bounded chunk, returning K indices with -1 padding.

    full_logits follows deep_gemm.fp8_mqa_logits's tensor arguments.
    select_topk follows IndexerOp._run_prefill_topk, preserving the configured
    native backend both for fallback rows and the final Top-2048.
    With apply_offsets=False, the caller owns final offset/canonicalization.
    Otherwise remap/scatter writes directly to output with offsets applied.
    """
    from .prefill_pivot_kernels import (
        build_proxy,
        prepare_rows,
        recall_candidates,
        remap_output,
        scatter_output,
        score_candidates,
    )

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("PIVOT prefill currently requires ENABLE_CUDA_GRAPH=0")
    config.validate(topk)
    n = q.shape[0]
    result = (
        output
        if output is not None
        else torch.empty((n, topk), dtype=torch.int32, device=q.device)
    )
    if n == 0:
        return result
    groups, fallback = plan_groups(ks, ke, offsets, config)
    weights = weights.reshape(n, q.shape[1]).contiguous()

    if fallback.numel():
        fks, fke = ks[fallback].contiguous(), ke[fallback].contiguous()
        logits = full_logits(q[fallback].contiguous(), kv, weights[fallback], fks, fke)
        out = torch.empty((fallback.numel(), topk), dtype=torch.int32, device=q.device)
        select_topk(logits, fks, fke, out, kv[0].shape[0])
        scatter_output(out, fallback, offsets, result, apply_offsets)
        del logits, out

    if not groups.numel():
        return result
    g = config.group_size
    (grouped_q, grouped_scale, grouped_w, row_ks, row_ke, pks, pke, zeros, lengths) = (
        prepare_rows(q, q_scale.reshape(n, q.shape[1]), weights, groups, ks, ke, config)
    )
    proxy_q, proxy_w = build_proxy(grouped_q, grouped_scale, grouped_w, g)
    logits = full_logits(proxy_q, kv, proxy_w, pks, pke)
    candidates = recall_candidates(logits, pks, pke, row_ke, config)
    del logits, proxy_q, proxy_w, grouped_scale
    scores = score_candidates(grouped_q, grouped_w, kv, candidates, row_ks, row_ke, g)
    selected = torch.empty((groups.numel(), topk), dtype=torch.int32, device=q.device)
    select_topk(scores, zeros, lengths, selected, config.candidates)
    remap_output(
        selected,
        scores,
        candidates,
        groups,
        row_ks,
        row_ke,
        offsets,
        result,
        apply_offsets,
    )
    return result
