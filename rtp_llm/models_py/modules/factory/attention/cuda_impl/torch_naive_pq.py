"""Torch Naive PQ Attention Backend.

PQ (Product Quantization) 加速 decode attention：
- Prefill：将 head_dim 切成 num_subspaces 个子空间，每个子空间独立 K-Means 聚类
- Decode：q_sub @ centroids_sub 得到每个子空间的簇分数，按 token 的 cids 累加；
  每个 q-head 选 top_k_tokens，同 KV group 内取 union，再对 union + 新 decode token
  做 full attention。

默认配置：num_subspaces=16, num_clusters=256, top_k_tokens=2000.
"""

import logging
import math
import os
from typing import Optional

import torch
import triton
import triton.language as tl
from flashinfer import xqa as flashinfer_xqa

try:
    from flashinfer import xqa_continuous
except ImportError:
    xqa_continuous = (
        None  # falls back to SDPA in _xqa_full_attention / _packed_attention
    )

from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.cuda_impl.pq_kmeans_triton import (
    batch_kmeans_Euclid,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.torch_naive import (
    TorchNaiveDecodeImpl,
    TorchNaivePrefillImpl,
)
from rtp_llm.ops import AttentionConfigs, ParallelismConfig
from rtp_llm.ops.compute_ops import KVCache, PyAttentionInputs

# ============================================================================
# PD separation helpers: store / load PQ buffers via PyAttentionInputs
# ============================================================================


def store_pq_to_attn_inputs(
    attn_inputs,
    layer_id: int,
    cids: torch.Tensor,  # [num_kv_heads, S, seq_len]
    cents: torch.Tensor,  # [num_kv_heads, S, K, sub_dim]
) -> None:
    """Store PQ buffers for *layer_id* onto attn_inputs for C++ PD transfer.

    Uses flat format: per_layer_cids[layer_id] = single Tensor per layer.
    """
    try:
        raw = attn_inputs.per_layer_cids
        current_cids = list(raw) if raw is not None else []
    except (TypeError, AttributeError):
        current_cids = []

    try:
        raw = attn_inputs.per_layer_cents
        current_cents = list(raw) if raw is not None else []
    except (TypeError, AttributeError):
        current_cents = []

    while len(current_cids) <= layer_id:
        current_cids.append(torch.empty(0))
        current_cents.append(torch.empty(0))

    current_cids[layer_id] = cids.to(torch.int32) if cids.dtype == torch.int64 else cids
    current_cents[layer_id] = (
        cents.float()
        if cents.dtype not in (torch.float32, torch.float16, torch.bfloat16)
        else cents
    )

    try:
        attn_inputs.per_layer_cids = current_cids
        attn_inputs.per_layer_cents = current_cents
    except (TypeError, AttributeError):
        pass


def load_pq_batch_from_attn_inputs(
    attn_inputs,
    layer_id: int,
    kv_head_idx: int,
    batch_size: int,
) -> Optional[dict]:
    """Load PQ data for ALL batch items at a given (layer, kv_head).

    Handles two formats from C++:
      - 3D: [H, S, prefill_len]               (single stream / prefill-side)
      - 4D: [batch, H, S, max_prefill_len]    (merged from NormalModelInputGatherer)

    Returns dict with:
        cids  : [batch, S, prefill_len]  int64, CUDA
        cents : [batch, S, K, sub_dim]   CUDA
        prefill_len : int  (max across batches when padded)
    or None if unavailable.
    """
    per_layer_cids = getattr(attn_inputs, "per_layer_cids", None)
    if per_layer_cids is None or layer_id >= len(per_layer_cids):
        return None

    layer_cids = per_layer_cids[layer_id]
    if not isinstance(layer_cids, torch.Tensor) or layer_cids.numel() == 0:
        return None

    per_layer_cents = getattr(attn_inputs, "per_layer_cents", None)
    layer_cents = per_layer_cents[layer_id]

    device = torch.device("cuda")

    if layer_cids.dim() == 4:
        # 4D: [batch, H, S, prefill_len] from merged decode streams
        if layer_cids.shape[0] < batch_size:
            return None
        cids = layer_cids[:batch_size, kv_head_idx]  # [batch, S, prefill_len]
        cents = layer_cents[:batch_size, kv_head_idx]  # [batch, S, K, sub_dim]
    elif layer_cids.dim() == 3:
        # 3D: [H, S, prefill_len] single stream — expand to batch
        cids = layer_cids[kv_head_idx].unsqueeze(0).expand(batch_size, -1, -1)
        cents = layer_cents[kv_head_idx].unsqueeze(0).expand(batch_size, -1, -1, -1)
    else:
        return None

    cids = cids.contiguous()
    cents = cents.contiguous()

    if cids.dtype != torch.int64:
        cids = cids.to(torch.int64)
    if not cids.is_cuda:
        cids = cids.to(device)
    if not cents.is_cuda:
        cents = cents.to(device)

    return {
        "cids": cids,
        "cents": cents,
        "prefill_len": int(cids.shape[-1]),
    }


# ============================================================================
# Triton kernel: fused gather + sum over subspaces
# ============================================================================
@triton.jit
def _pq_aggregate_kernel(
    cluster_scores_ptr,  # [bs, S, K, num_groups]
    cids_ptr,  # [bs, S, prefill_len] int64
    token_scores_ptr,  # [bs, num_groups, prefill_len]
    cs_stride_b,
    cs_stride_s,
    cs_stride_k,
    cid_stride_b,
    cid_stride_s,
    ts_stride_b,
    ts_stride_q,
    PREFILL_LEN,
    NUM_GROUPS: tl.constexpr,
    S: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)

    t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_offsets < PREFILL_LEN
    q_offsets = tl.arange(0, NUM_GROUPS)

    acc = tl.zeros((BLOCK_T, NUM_GROUPS), dtype=tl.float32)

    cid_base = cids_ptr + pid_b * cid_stride_b
    cs_base = cluster_scores_ptr + pid_b * cs_stride_b

    for s in tl.static_range(S):
        k_idx = tl.load(
            cid_base + s * cid_stride_s + t_offsets,
            mask=t_mask,
            other=0,
        )
        cs_ptrs = (
            cs_base
            + s * cs_stride_s
            + k_idx[:, None] * cs_stride_k
            + q_offsets[None, :]
        )
        cs_vals = tl.load(cs_ptrs, mask=t_mask[:, None], other=0.0)
        acc += cs_vals.to(tl.float32)

    acc_t = tl.trans(acc)
    out_ptrs = (
        token_scores_ptr
        + pid_b * ts_stride_b
        + q_offsets[:, None] * ts_stride_q
        + t_offsets[None, :]
    )
    tl.store(
        out_ptrs,
        acc_t.to(token_scores_ptr.dtype.element_ty),
        mask=t_mask[None, :],
    )


def _next_po2(n: int) -> int:
    return 1 << (n - 1).bit_length() if n > 0 else 1


def _pq_aggregate_triton(
    cluster_scores: torch.Tensor,  # [bs, S, K, num_groups]
    cids: torch.Tensor,  # [bs, S, prefill_len] int64
    num_groups: int,
) -> torch.Tensor:
    bs, S, K, _ = cluster_scores.shape
    _, _, prefill_len = cids.shape

    assert cluster_scores.is_contiguous() and cids.is_contiguous()
    assert cids.dtype == torch.int64

    NUM_GROUPS_PO2 = _next_po2(num_groups)
    if NUM_GROUPS_PO2 != num_groups:
        cluster_scores = torch.nn.functional.pad(
            cluster_scores, (0, NUM_GROUPS_PO2 - num_groups)
        ).contiguous()

    token_scores = torch.empty(
        bs,
        NUM_GROUPS_PO2,
        prefill_len,
        dtype=cluster_scores.dtype,
        device=cluster_scores.device,
    )

    BLOCK_T = 128
    grid = (bs, triton.cdiv(prefill_len, BLOCK_T))

    _pq_aggregate_kernel[grid](
        cluster_scores,
        cids,
        token_scores,
        cluster_scores.stride(0),
        cluster_scores.stride(1),
        cluster_scores.stride(2),
        cids.stride(0),
        cids.stride(1),
        token_scores.stride(0),
        token_scores.stride(1),
        prefill_len,
        NUM_GROUPS=NUM_GROUPS_PO2,
        S=S,
        BLOCK_T=BLOCK_T,
    )
    return token_scores[:, :num_groups, :]


# ============================================================================
# Page-sparse 路径:融合 page-score kernel(ADC 查 cids 累加子空间 + 页内 max)
# 每 program 处理 (b, q-head h, 连续 PPB 页);输出 per-head page max,免物化 token_score。
# ============================================================================
@triton.jit
def _pq_page_score_kernel(
    LUT,          # [B, H, S, K] (bf16/f32)  per-head 子空间-簇 LUT
    CIDS,         # [B, HKV, S, Tpad] i32     (尾部 pad 到整页, pad=0)
    OUT,          # [B, H, n_pages] f32       per-head page max
    lut_b, lut_h, lut_s,
    cid_b, cid_kh, cid_s,
    out_b, out_h,
    G_: tl.constexpr, S_: tl.constexpr, PAGE_: tl.constexpr, PPB_: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    pb = tl.program_id(2)
    kh = h // G_
    p0 = pb * PPB_
    offs_t = p0 * PAGE_ + tl.arange(0, PPB_ * PAGE_)
    lut_h_base = LUT + b * lut_b + h * lut_h
    cid_base = CIDS + b * cid_b + kh * cid_kh
    acc = tl.zeros((PPB_ * PAGE_,), dtype=tl.float32)
    for s in tl.static_range(S_):
        cid = tl.load(cid_base + s * cid_s + offs_t)
        lut = tl.load(lut_h_base + s * lut_s + cid).to(tl.float32)
        acc += lut
    acc2 = tl.reshape(acc, (PPB_, PAGE_))
    pmax = tl.max(acc2, axis=1)
    tl.store(OUT + b * out_b + h * out_h + p0 + tl.arange(0, PPB_), pmax)


# ============================================================================
# Prefill: 子空间 PQ 聚类
# ============================================================================


class TorchNaivePQPrefillImpl(TorchNaivePrefillImpl):
    """带 PQ 子空间聚类的 Prefill 实现."""

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        super().__init__(attn_configs, attn_inputs, parallelism_config)

        self.num_subspaces = 16
        self.num_clusters = 256
        self.kmeans_iters = 20

        assert (
            self.head_dim % self.num_subspaces == 0
        ), f"head_dim {self.head_dim} must be divisible by num_subspaces {self.num_subspaces}"

        logging.info(
            f"TorchNaivePQPrefillImpl: S={self.num_subspaces}, K={self.num_clusters}, "
            f"sub_dim={self.head_dim // self.num_subspaces}"
        )

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        if self.need_rope_kv_cache:
            qkv = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)

        q, k, v = self._split_qkv(qkv)
        self._perform_pq_clustering(k, kv_cache)

        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        output = self._run_attention_extend(q, k, v)
        return output.reshape(output.shape[0], -1)

    def _perform_pq_clustering(
        self,
        k: torch.Tensor,  # [total_tokens, num_kv_heads, head_dim]
        kv_cache: Optional[KVCache],
    ) -> None:
        if kv_cache is None:
            return

        layer_id = kv_cache.layer_id
        batch_size = self.attn_inputs.input_lengths.size(0)
        cu_seqlens = self.attn_inputs.cu_seqlens[: batch_size + 1]
        num_kv_heads = k.shape[1]
        S = self.num_subspaces
        sub_dim = self.head_dim // S

        # PD separation currently only supports batch_size=1 through the C++ path.
        # For multi-batch, cids/cents per layer are stored for seq_idx=0 only.
        all_cids = []
        all_cents = []

        # 方向1:只对长序列做 PQ 聚类;短序列(<PQ_MIN_SEQ_LEN)存 sentinel(cids 第0位=-1),
        # decode 端据此 per-row 走 full attention(无损)。阈值默认 16k。
        pq_min_seq_len = int(os.getenv("PQ_MIN_SEQ_LEN", "16384"))
        for seq_idx in range(batch_size):
            start_idx = cu_seqlens[seq_idx].item()
            end_idx = cu_seqlens[seq_idx + 1].item()
            seq_len = end_idx - start_idx

            if seq_len < pq_min_seq_len:
                # sentinel:不聚类。cids 第0位=-1 标记 no_pq;cents 占位 zeros(对齐 stack shape)
                cids_b = torch.full(
                    (num_kv_heads, S, 1), -1, dtype=torch.int64, device=k.device
                )
                cents_b = torch.zeros(
                    (num_kv_heads, S, self.num_clusters, sub_dim),
                    dtype=torch.float32,
                    device=k.device,
                )
                all_cids.append(cids_b)
                all_cents.append(cents_b)
                continue

            k_seq = k[start_idx:end_idx]  # [seq_len, H, head_dim]
            k_subs = (
                k_seq.reshape(seq_len, num_kv_heads, S, sub_dim)
                .permute(1, 2, 0, 3)  # [H, S, seq_len, sub_dim]
                .reshape(num_kv_heads * S, seq_len, sub_dim)
                .contiguous()
            )

            cids_b, cents_b, _ = batch_kmeans_Euclid(
                k_subs,
                self.num_clusters,
                max_iters=self.kmeans_iters,
                tol=1e-4,
            )
            cids_b = cids_b.to(torch.int64).reshape(num_kv_heads, S, seq_len)
            cents_b = cents_b.reshape(num_kv_heads, S, self.num_clusters, sub_dim)

            all_cids.append(cids_b)
            all_cents.append(cents_b)

        # Store for PD transfer (only seq_idx=0 in current C++ path)
        store_pq_to_attn_inputs(
            self.attn_inputs,
            layer_id,
            all_cids[0],
            all_cents[0],
        )
        _stored_cids = getattr(self.attn_inputs, "per_layer_cids", None)
        _stored_len = len(_stored_cids) if _stored_cids is not None else -1
        _this_shape = tuple(all_cids[0].shape)
        logging.info(
            "[PQ-PREFILL] store done layer=%d bs=%d H=%d S=%d K=%d "
            "cids_shape=%s per_layer_cids_len=%d is_set_on_attn_inputs=%s",
            layer_id,
            batch_size,
            num_kv_heads,
            S,
            self.num_clusters,
            _this_shape,
            _stored_len,
            _stored_cids is not None,
        )


# ============================================================================
# Decode: PQ 打分 + per-q-head top_k union + 全 attention
# ============================================================================


class TorchNaivePQDecodeImpl(TorchNaiveDecodeImpl):
    """带 PQ 加速的 Decode 实现.

    每个 KV head：
      1. q_sub @ centroids_sub 得到 [Q, S, K] 簇分数
      2. 按 cids 查表得到 [Q, N] token 分数
      3. 每个 q-head 选 top_k_tokens，取 union
      4. 加上 prefill_len 之后的所有新 decode token
      5. 用 union + 新 token 做 full attention
    """

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        super().__init__(attn_configs, attn_inputs, parallelism_config)

        self.num_subspaces = int(os.getenv("PQ_NUM_SUBSPACES", "16"))
        self.top_k_tokens = int(os.getenv("PQ_TOP_K_TOKENS", "2000"))
        # 动态 topk(方案A):PQ_TOP_K_RATIO>0 时 P 按 ratio × (MAX_SEQ_LEN/tpb) 算,
        # 即"选页数 = 部署最大序列的固定比例"。静态(部署期定),cuda-graph-safe。
        # 解决固定 token 数在长序列下读太稀的问题(32k够/128k不够)。
        self.top_k_ratio = float(os.getenv("PQ_TOP_K_RATIO", "0"))
        self._max_seq_len_cfg = int(os.getenv("MAX_SEQ_LEN", "40960"))

        self._sm_count = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        self._xqa_scratch: Optional[torch.Tensor] = None
        self._xqa_semaphores: Optional[torch.Tensor] = None

        logging.info(
            f"TorchNaivePQDecodeImpl.__init__: S={self.num_subspaces}, top_k={self.top_k_tokens}"
        )

    @classmethod
    def support(cls, attn_configs, attn_inputs) -> bool:
        _ok = super().support(attn_configs, attn_inputs)
        logging.info(
            "[PQ-DECODE] support() classmethod: result=%s use_mla=%s is_prefill=%s",
            _ok,
            attn_configs.use_mla,
            attn_inputs.is_prefill,
        )
        return _ok

    def support_cuda_graph(self) -> bool:
        _ok = (
            super().support_cuda_graph()
            if hasattr(super(), "support_cuda_graph")
            else True
        )
        logging.info("[PQ-DECODE] support_cuda_graph() -> %s", _ok)
        return _ok

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        logging.info("[PQ-DECODE] prepare_cuda_graph() called")
        super().prepare_cuda_graph(attn_inputs)
        self._cuda_graph_mode = True
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        batch_size = attn_inputs.input_lengths.size(0)
        if self._xqa_scratch is None:
            self._xqa_scratch = torch.zeros(256 << 20, dtype=torch.uint8, device=device)
        nb_seq = self.num_kv_heads * batch_size
        nb_sem = ((nb_seq + 1) // 2 * 2) + 2 + nb_seq + 2
        if self._xqa_semaphores is None or self._xqa_semaphores.shape[0] < nb_sem:
            self._xqa_semaphores = torch.zeros(
                max(nb_sem, 256), dtype=torch.uint32, device=device
            )

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        _layer_id_entry = kv_cache.layer_id if kv_cache is not None else layer_idx
        logging.info(
            "[PQ-DECODE] forward() entry: layer=%d qkv.shape=%s qkv.dtype=%s cuda_graph=%s",
            _layer_id_entry,
            tuple(qkv.shape),
            qkv.dtype,
            getattr(self, "_cuda_graph_mode", False),
        )

        if self.need_rope_kv_cache:
            q = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
            q = q.reshape(q.shape[0], self.num_heads, self.head_dim)
        else:
            q, _, _ = self._split_qkv(qkv)

        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        batch_size = q.shape[0]
        layer_id = kv_cache.layer_id if kv_cache is not None else 0

        has_pq = any(
            load_pq_batch_from_attn_inputs(self.attn_inputs, layer_id, h, batch_size)
            is not None
            for h in range(self.num_kv_heads)
        )

        if not has_pq:
            _plc = getattr(self.attn_inputs, "per_layer_cids", None)
            if _plc is None:
                _reason = (
                    "per_layer_cids=None (prefill PQ data not propagated to decode)"
                )
            elif layer_id >= len(_plc):
                _reason = f"layer_id={layer_id} >= len(per_layer_cids)={len(_plc)}"
            else:
                _t = _plc[layer_id]
                if isinstance(_t, torch.Tensor):
                    _reason = f"layer_cids[{layer_id}] empty: numel={_t.numel()} dim={_t.dim()} shape={tuple(_t.shape)}"
                else:
                    _reason = (
                        f"layer_cids[{layer_id}] not a Tensor: type={type(_t).__name__}"
                    )
            logging.info(
                "[PQ-DECODE] branch=FULL_XQA layer=%d bs=%d num_kv_heads=%d reason=%s",
                layer_id,
                batch_size,
                self.num_kv_heads,
                _reason,
            )
            output = self._xqa_paged_attention(q, kv_cache)
        else:
            _plc = getattr(self.attn_inputs, "per_layer_cids", None)
            _pq_heads = sum(
                1
                for h in range(self.num_kv_heads)
                if load_pq_batch_from_attn_inputs(
                    self.attn_inputs, layer_id, h, batch_size
                )
                is not None
            )
            _layer_cids_shape = (
                tuple(_plc[layer_id].shape)
                if _plc is not None
                and layer_id < len(_plc)
                and isinstance(_plc[layer_id], torch.Tensor)
                else None
            )
            logging.info(
                "[PQ-DECODE] branch=PQ_SPARSE layer=%d bs=%d pq_heads=%d/%d layer_cids_shape=%s",
                layer_id,
                batch_size,
                _pq_heads,
                self.num_kv_heads,
                _layer_cids_shape,
            )
            output = None
            if os.getenv("PQ_SPARSE_TRTLLM", "0") == "1" and kv_cache is not None:
                try:
                    output = self._run_pq_sparse_trtllm(q, kv_cache)
                except Exception as e:  # 出错回退到 dense 路径,保住精度
                    logging.warning(
                        "[PQ-SPARSE-TRT] fallback to dense path: %s", repr(e)
                    )
                    output = None
            if output is None:
                k_full, v_full = self._read_kv_from_cache(kv_cache)
                output = self._run_pq_attention_decode(q, k_full, v_full, kv_cache)
        return output.reshape(output.shape[0], -1)

    def _xqa_paged_attention(
        self,
        q: torch.Tensor,  # [bs, num_q_heads, head_dim]
        kv_cache: KVCache,
    ) -> torch.Tensor:
        batch_size = q.shape[0]
        device = q.device
        layer_id = kv_cache.layer_id if kv_cache is not None else 0

        kv_base = kv_cache.kv_cache_base
        tpb = self.tokens_per_block

        if kv_base.ndim == 2:
            block_num = kv_base.shape[0]
            expected = 2 * self.num_kv_heads * tpb * self.head_dim
            kv_tensor = kv_base[:, :expected].reshape(
                block_num, 2, self.num_kv_heads, tpb, self.head_dim
            )
        else:
            kv_tensor = kv_base

        k_cache = kv_tensor[:, 0]  # [num_pages, num_kv_heads, tpb, head_dim]
        v_cache = kv_tensor[:, 1]

        if hasattr(self, "_block_indices_gpu"):
            page_table = self._block_indices_gpu.to(torch.int32)
        elif (
            getattr(self.attn_inputs, "kv_cache_kernel_block_id_device", None)
            is not None
        ):
            page_table = self.attn_inputs.kv_cache_kernel_block_id_device[
                :batch_size, :
            ].to(torch.int32)
        elif self.attn_inputs.kv_cache_block_id_host is not None:
            page_table = self.attn_inputs.kv_cache_block_id_host[layer_id][
                :batch_size, :
            ].to(device=device, dtype=torch.int32)
        else:
            return torch.zeros(
                batch_size,
                self.num_heads,
                self.head_dim,
                dtype=q.dtype,
                device=device,
            )

        # CUDA-graph safe path: reuse device-resident seq_lens (sequence_lengths+1 in decode mode).
        # Fallback to fill_mla_params for the rare case where sequence_lengths_plus_1_d is absent.
        _seq_lens_d = getattr(self.attn_inputs, "sequence_lengths_plus_1_d", None)
        if _seq_lens_d is not None:
            seq_lens = _seq_lens_d[:batch_size].to(torch.int32)
        else:
            seq_lens = self._compute_kv_seq_lens(batch_size).to(
                device=device, dtype=torch.int32
            )

        # XQA: q must be fp16/bf16 even when kv cache is FP8. Keep q in its original (hi-prec) dtype.
        q_xqa = q.unsqueeze(1)
        if q_xqa.dtype not in (torch.float16, torch.bfloat16):
            q_xqa = q_xqa.to(torch.bfloat16)
        output = torch.empty_like(q_xqa)

        self._ensure_xqa_buffers(device, batch_size)
        self._xqa_semaphores.zero_()

        xqa_kwargs = dict(
            num_kv_heads=self.num_kv_heads,
            page_size=tpb,
            kv_layout="HND",
            sm_count=self._sm_count,
        )
        if getattr(self, "_cuda_graph_mode", False):
            xqa_kwargs["nb_sub_seq_per_seq"] = 16

        # When KV cache is FP8, XQA needs kv_scale to dequantize internally.
        _is_fp8_cache = k_cache.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
        _kv_scale_src = "none"
        if _is_fp8_cache:
            scale_base = getattr(kv_cache, "kv_scale_base", None)
            if scale_base is not None and scale_base.numel() > 0:
                xqa_kwargs["kv_scale"] = scale_base
                _kv_scale_src = f"kv_scale_base(shape={tuple(scale_base.shape)})"
            else:
                xqa_kwargs["kv_scale"] = torch.tensor(1.0, device=device)
                _kv_scale_src = "unit(1.0)"

        logging.info(
            "[XQA-CALL] layer=%d bs=%d q.dtype=%s kv.dtype=%s fp8=%s cuda_graph=%s "
            "page_table.shape=%s seq_lens_src=%s kv_scale=%s",
            layer_id,
            batch_size,
            q_xqa.dtype,
            k_cache.dtype,
            _is_fp8_cache,
            getattr(self, "_cuda_graph_mode", False),
            tuple(page_table.shape),
            (
                "sequence_lengths_plus_1_d"
                if _seq_lens_d is not None
                else "fill_mla_params"
            ),
            _kv_scale_src,
        )

        flashinfer_xqa(
            q_xqa,
            k_cache,
            v_cache,
            page_table,
            seq_lens,
            output,
            self._xqa_scratch,
            self._xqa_semaphores,
            **xqa_kwargs,
        )
        return output.squeeze(1).to(q.dtype)

    def _compute_kv_seq_lens(self, batch_size: int) -> torch.Tensor:
        from rtp_llm.ops.compute_ops import fill_mla_params

        params = fill_mla_params(
            (
                self.attn_inputs.prefix_lengths
                if getattr(self.attn_inputs, "prefix_lengths", None) is not None
                else torch.tensor([], dtype=torch.int32)
            ),
            self.attn_inputs.sequence_lengths,
            self.attn_inputs.input_lengths,
            (
                self.attn_inputs.kv_cache_block_id_host
                if self.attn_inputs.kv_cache_block_id_host is not None
                else torch.tensor([], dtype=torch.int32)
            ),
            self.tokens_per_block,
        )
        return params.kvlen_h[:batch_size]

    def _ensure_xqa_buffers(self, device: torch.device, batch_size: int):
        if self._xqa_scratch is None:
            self._xqa_scratch = torch.zeros(256 << 20, dtype=torch.uint8, device=device)
        nb_seq = self.num_kv_heads * batch_size
        nb_sem = ((nb_seq + 1) // 2 * 2) + 2 + nb_seq + 2
        if self._xqa_semaphores is None or self._xqa_semaphores.shape[0] < nb_sem:
            self._xqa_semaphores = torch.zeros(
                max(nb_sem, 256), dtype=torch.uint32, device=device
            )

    def _xqa_full_attention(
        self,
        q: torch.Tensor,  # [bs, num_q_heads, head_dim]
        k_full: torch.Tensor,  # [bs, max_seq_len, num_kv_heads, head_dim]
        v_full: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
    ) -> torch.Tensor:
        # SDPA-based fallback (replaces flashinfer xqa_continuous)
        bs, num_q, head_dim = q.shape
        device = q.device

        q_sdpa = q.unsqueeze(2).to(q.dtype)  # [bs, num_q_heads, 1, head_dim]
        k_sdpa = k_full.permute(0, 2, 1, 3).to(
            q.dtype
        )  # [bs, num_kv, max_seq_len, head_dim]
        v_sdpa = v_full.permute(0, 2, 1, 3).to(q.dtype)

        if num_q != self.num_kv_heads:
            num_groups = num_q // self.num_kv_heads
            k_sdpa = k_sdpa.repeat_interleave(num_groups, dim=1)
            v_sdpa = v_sdpa.repeat_interleave(num_groups, dim=1)

        pos = torch.arange(max_seq_len, device=device)
        valid = pos[None, :] < seq_lens.to(device=device, dtype=torch.int32)[:, None]
        attn_mask = valid[:, None, None, :]  # [bs, 1, 1, max_seq_len]

        out = torch.nn.functional.scaled_dot_product_attention(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            attn_mask=attn_mask,
            scale=self.scaling,
        )
        return out.squeeze(2)  # [bs, num_q_heads, head_dim]

    def _run_pq_sparse_trtllm(self, q: torch.Tensor, kv_cache: KVCache) -> torch.Tensor:
        """Page-sparse decode(全 GPU / 静态 shape / 零 host 同步,cuda-graph-safe):
        PQ 选固定 top-P prefill 页 + 强制纳入最近 R 页(含 current),喂压缩 block_table 给
        trtllm_batch_decode_with_kv_cache,直接在 paged KV 上稀疏。选页/掩码全用 device 上的
        kvlen,不再依赖 cids sentinel(顺带修掉多 batch padding bug)。bf16 KV + sm100。"""
        import flashinfer
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.trtllm_gen import (
            get_trt_workspace_buffer,
        )

        B = q.shape[0]
        layer_id = kv_cache.layer_id
        device = q.device
        Hkv, Hq, Dh = self.num_kv_heads, self.num_heads, self.head_dim
        Gg = Hq // Hkv
        Sd = self.num_subspaces
        SUBd = Dh // Sd
        tpb = self.tokens_per_block
        PPB = 4
        R = int(os.getenv("PQ_TAIL_PAGES", "8"))  # 强制纳入最近 R 页(含 current 部分页)

        pqs = [
            load_pq_batch_from_attn_inputs(self.attn_inputs, layer_id, h, B)
            for h in range(Hkv)
        ]
        cids = torch.stack([p["cids"] for p in pqs], dim=1).to(torch.int32).contiguous()
        cents = torch.stack([p["cents"] for p in pqs], dim=1)  # [B,Hkv,Sd,K,SUB]
        plen = cids.shape[-1]  # shape 元数据,非 host sync
        # per-row 聚类标记:prefill 对 <PQ_MIN_SEQ_LEN 的样本不聚类、存 sentinel(cids 第0位=-1)。
        # 必须在 clamp 之前取(clamp 会把 -1 抹成 0)。cuda-graph-safe:cids 每步随 copy-in 刷新。
        has_pq_row = cids[:, 0, 0, 0] >= 0  # [B] bool(cids=[B,Hkv,S,plen]);True=有聚类→PQ,False→读全 KV
        cids = cids.clamp(min=0)  # 防 sentinel/负值在 page-score kernel 索引越界

        seq_lens_d = getattr(self.attn_inputs, "sequence_lengths_plus_1_d", None)
        if seq_lens_d is not None:
            kvlen = seq_lens_d[:B].to(torch.int32)
        else:
            kvlen = self._compute_kv_seq_lens(B).to(device=device, dtype=torch.int32)
        full_bt = self.attn_inputs.kv_cache_kernel_block_id_device[:B].to(torch.int32)

        # ---- predict:LUT + 融合 page-score kernel ----
        n_pp = (plen + tpb - 1) // tpb
        n_pp_pad = ((n_pp + PPB - 1) // PPB) * PPB
        if n_pp_pad * tpb != plen:
            cids = torch.nn.functional.pad(cids, (0, n_pp_pad * tpb - plen), value=0)
        cents_h = cents.repeat_interleave(Gg, dim=1).contiguous()  # [B,Hq,Sd,K,SUB]
        q_r = q.view(B, Hq, Sd, SUBd)
        lut = torch.einsum("bhsd,bhskd->bhsk", q_r, cents_h.to(q_r.dtype)).contiguous()
        ph = torch.empty(B, Hq, n_pp_pad, device=device, dtype=torch.float32)
        _pq_page_score_kernel[(B, Hq, n_pp_pad // PPB)](
            lut, cids, ph,
            lut.stride(0), lut.stride(1), lut.stride(2),
            cids.stride(0), cids.stride(1), cids.stride(2),
            ph.stride(0), ph.stride(1),
            G_=Gg, S_=Sd, PAGE_=tpb, PPB_=PPB,
        )
        page_score = ph.amax(dim=1)  # [B, n_pp_pad] union over heads

        # ---- 选页(全 device,无 host sync)----
        n_total = (kvlen + tpb - 1) // tpb            # [B] 含 decode 的总页数
        score_hi = (n_total - R).clamp(min=0)          # [B] 评分上界(避开最近 R 页 + pad 页)
        cols = torch.arange(n_pp_pad, device=device, dtype=torch.int32)
        page_score = page_score.masked_fill(cols[None, :] >= score_hi[:, None], float("-inf"))

        if self.top_k_ratio > 0:
            # P = ratio × 部署最大页数(静态,与 MAX_SEQ_LEN 相关,不随每请求 kvlen 变)
            max_pages_cfg = -(-self._max_seq_len_cfg // tpb)
            P = max(1, int(self.top_k_ratio * max_pages_cfg))
        else:
            P = max(1, -(-self.top_k_tokens // tpb))
        P = min(P, n_pp_pad)                            # shape 级 int
        top_prefill = page_score.topk(P, dim=-1, sorted=False).indices.to(torch.int32)  # [B,P]

        # 最近 R 页:[n_total-R, n_total),current 页在最后(device 算术)
        rr = torch.arange(R, device=device, dtype=torch.int32)
        recent = (n_total[:, None] - R + rr[None, :]).clamp(min=0)  # [B,R]

        W = P + R
        # 路由按"有无聚类结果"(per-row):有聚类 → PQ 稀疏选页(前 W 页);无聚类 → 读全部页(=full
        # attention),一起喂同一个 trtllm decode kernel。非聚类行用完整顺序页表(full_bt)读全 KV,
        # 因此即便短输入长输出、decode 把页数撑过 W 也能读全(修了 n_total≤W 路由在长输出下误判进 PQ
        # 读 sentinel 垃圾页的 bug)。cuda-graph-safe:has_pq_row 来自每步 copy-in 的 cids、
        # block table 宽度=部署最大页数(静态)、max_seq_len 静态。
        pq_pages = torch.cat([top_prefill, recent], dim=1)          # [B, W] 选中页(prefill top-P + recent R)
        max_blocks = full_bt.shape[1]                               # 静态:部署最大页数 = MAX_SEQ_LEN/tpb
        pq_pages = pq_pages.clamp(min=0, max=max_blocks - 1)
        pq_bt_W = torch.gather(full_bt, 1, pq_pages.to(torch.int64)).to(torch.int32)  # [B, W] 选中页 block id

        # 默认整表=完整顺序页表(非聚类行读全部页);聚类行把前 W 列覆盖为选中页(其余列被 seqlen 截断、读不到)
        sparse_bt = full_bt.clone()                                 # [B, max_blocks]
        sparse_bt[:, :W] = torch.where(has_pq_row[:, None], pq_bt_W, sparse_bt[:, :W])

        recent_valid = (kvlen - score_hi * tpb).clamp(min=0, max=R * tpb)  # [B]
        pq_seqlen = (P * tpb + recent_valid).to(torch.int32)              # [B] 聚类行:读 W 页
        sparse_seqlen = torch.where(has_pq_row, pq_seqlen, kvlen.to(torch.int32))  # 非聚类行:读全 KV
        max_seq_len = max_blocks * tpb                              # =MAX,支持非聚类长 decode 读全

        # ---- trtllm sparse decode ----
        kv_base = kv_cache.kv_cache_base
        if kv_base.ndim == 2:
            block_num = kv_base.shape[0]
            expected = 2 * Hkv * tpb * Dh
            kv = kv_base[:, :expected].reshape(block_num, 2, Hkv, tpb, Dh)
        else:
            kv = kv_base.view(kv_base.shape[0], 2, Hkv, tpb, Dh)
        ws = get_trt_workspace_buffer()
        q_in = q.contiguous().view(B, Hq, Dh).to(kv.dtype)
        o = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=q_in, kv_cache=kv, workspace_buffer=ws,
            block_tables=sparse_bt, seq_lens=sparse_seqlen, max_seq_len=max_seq_len,
            bmm1_scale=self.scaling, bmm2_scale=1.0, window_left=-1,
            out_dtype=q.dtype, q_len_per_req=1,
        )
        return o.view(B, Hq, Dh).to(q.dtype)

    def _run_pq_attention_decode(
        self,
        q: torch.Tensor,  # [batch, num_heads, head_dim]
        k_full: torch.Tensor,  # [batch, max_seq_len, num_kv_heads, head_dim]
        v_full: torch.Tensor,
        kv_cache: Optional[KVCache],
    ) -> torch.Tensor:
        batch_size = q.shape[0]
        layer_id = kv_cache.layer_id if kv_cache is not None else 0
        num_groups = self.num_heads // self.num_kv_heads if self.enable_gqa else 1
        seq_lens = self._compute_kv_seq_lens(batch_size)
        device = q.device
        max_seq_len = k_full.shape[1]

        self._ensure_xqa_buffers(device, batch_size)

        has_pq = False
        for kv_head_idx in range(self.num_kv_heads):
            if (
                load_pq_batch_from_attn_inputs(
                    self.attn_inputs,
                    layer_id,
                    kv_head_idx,
                    batch_size,
                )
                is not None
            ):
                has_pq = True
                break

        if not has_pq:
            logging.info(
                "[PQ-DECODE] inner=XQA_FULL layer=%d bs=%d (no PQ for any kv_head; running XQA over dequanted k_full/v_full)",
                layer_id,
                batch_size,
            )
            return self._xqa_full_attention(q, k_full, v_full, seq_lens, max_seq_len)

        output = torch.empty_like(q)
        _pq_count = 0
        _fb_count = 0
        for kv_head_idx in range(self.num_kv_heads):
            start_h = kv_head_idx * num_groups
            end_h = start_h + num_groups
            q_kv = q[:, start_h:end_h, :].contiguous()

            pq = load_pq_batch_from_attn_inputs(
                self.attn_inputs,
                layer_id,
                kv_head_idx,
                batch_size,
            )
            if pq is not None:
                _pq_count += 1
                output[:, start_h:end_h, :] = self._batched_pq_decode_one_kv(
                    q_kv,
                    pq["cids"],  # [batch, S, prefill_len]
                    pq["cents"],  # [batch, S, K, sub_dim]
                    pq["prefill_len"],
                    k_full,
                    v_full,
                    kv_head_idx,
                    seq_lens,
                    max_seq_len,
                )
            else:
                _fb_count += 1
                logging.info(
                    "[PQ-DECODE] inner=FALLBACK_FULL layer=%d kv_head=%d (this head has no PQ data, using full attention)",
                    layer_id,
                    kv_head_idx,
                )
                output[:, start_h:end_h, :] = self._fallback_full_attn(
                    q_kv,
                    k_full,
                    v_full,
                    kv_head_idx,
                    seq_lens,
                    max_seq_len,
                )
        logging.info(
            "[PQ-DECODE] layer=%d bs=%d summary: %d heads used PQ, %d heads used fallback",
            layer_id,
            batch_size,
            _pq_count,
            _fb_count,
        )
        return output

    def _batched_pq_decode_one_kv(
        self,
        q_kv: torch.Tensor,  # [bs, num_groups, head_dim]
        cids: torch.Tensor,  # [bs, S, max_prefill_len] (padded)
        cents: torch.Tensor,  # [bs, S, K, sub_dim]
        max_prefill_len: int,
        k_full: torch.Tensor,
        v_full: torch.Tensor,
        kv_head_idx: int,
        seq_lens: torch.Tensor,
        max_seq_len: int,
    ) -> torch.Tensor:
        bs, num_groups, head_dim = q_kv.shape
        device = q_kv.device
        S = cids.shape[1]
        sub_dim = head_dim // S

        # Per-batch actual prefill length = number of valid (non-padding) cids
        # positions. cids 在 prefill 端就是按真实 prefill 长度建的（无 padding）;
        # NormalModelInputGatherer 合并多 batch 时用 -1 padding 到 max_prefill_len。
        # 注意: decode 阶段 attn_inputs.input_lengths 不是 prefill prompt 长度，
        # 不能用它当边界（会把绝大多数 prefill token 误 mask、新 decode token 也漏选）。
        input_lens = (cids[:, 0, :] >= 0).sum(dim=1).to(
            device=device, dtype=torch.int32
        )  # [bs]

        q_subs = q_kv.reshape(bs, num_groups, S, sub_dim)
        cluster_scores = torch.einsum(
            "bqsd,bskd->bskq", q_subs, cents.to(q_subs.dtype)
        ).contiguous()
        token_scores = _pq_aggregate_triton(cluster_scores, cids, num_groups)
        # token_scores: [bs, num_groups, max_prefill_len]

        # Mask padded positions to -inf so topk never selects them.
        # For batch b, positions >= input_lens[b] are padding.
        pq_pos = torch.arange(max_prefill_len, device=device, dtype=torch.int32)
        valid_pq = pq_pos[None, :] < input_lens[:, None]  # [bs, max_prefill_len]
        token_scores.masked_fill_(~valid_pq.unsqueeze(1), float("-inf"))

        per_q_k = min(self.top_k_tokens, max_prefill_len)
        topk_ids = token_scores.topk(per_q_k, dim=-1).indices  # [bs, G, per_q_k]

        # Build per-sequence mask over full KV length
        mask = torch.zeros(bs, max_seq_len, dtype=torch.bool, device=device)
        topk_flat = topk_ids.reshape(bs, -1).clamp(max=max_seq_len - 1)
        mask.scatter_(1, topk_flat, True)

        pos = torch.arange(max_seq_len, device=device, dtype=torch.int32)
        seq_lens_gpu = seq_lens.to(device=device, dtype=torch.int32)
        within_seq = pos[None, :] < seq_lens_gpu[:, None]

        # Add new decode tokens: positions >= per-batch prefill boundary
        prefill_boundary = input_lens[:, None]
        new_decode_mask = (pos[None, :] >= prefill_boundary) & within_seq
        mask = (mask | new_decode_mask) & within_seq

        logging.info(
            "[PQ-DBG] kv_head=%d max_seq_len=%d max_prefill_len=%d per_q_k=%d "
            "input_lens=%s topk_uniq=%s new_decode=%s within_seq=%s seq_lens=%s",
            kv_head_idx, max_seq_len, max_prefill_len, per_q_k,
            input_lens.tolist(),
            mask.sum(dim=1).tolist() if False else (topk_ids.reshape(bs,-1).clamp(max=max_seq_len-1).unique().numel()),
            new_decode_mask.sum(dim=1).tolist(),
            within_seq.sum(dim=1).tolist(),
            seq_lens_gpu.tolist(),
        )
        _sel = mask.sum(dim=1).tolist()
        _tot = seq_lens_gpu.tolist()
        _ratio = [
            f"{s}/{t}({s * 100.0 / t:.1f}%)" if t > 0 else f"{s}/0"
            for s, t in zip(_sel, _tot)
        ]
        logging.info(
            "[PQ-SPARSE] kv_head=%d per_q_k=%d max_prefill_len=%d bs=%d selected/total=%s",
            kv_head_idx,
            per_q_k,
            max_prefill_len,
            len(_sel),
            _ratio,
        )

        return self._packed_attention(
            q_kv,
            mask,
            k_full,
            v_full,
            kv_head_idx,
            max_seq_len,
            max_seq_len,
        )

    def _fallback_full_attn(
        self,
        q_kv: torch.Tensor,
        k_full: torch.Tensor,
        v_full: torch.Tensor,
        kv_head_idx: int,
        seq_lens: torch.Tensor,
        max_seq_len: int,
    ) -> torch.Tensor:
        """No PQ cache: use all tokens within each sequence."""
        bs = q_kv.shape[0]
        device = q_kv.device
        pos = torch.arange(max_seq_len, device=device, dtype=torch.int32)
        seq_lens_gpu = seq_lens.to(device=device, dtype=torch.int32)
        mask = pos[None, :] < seq_lens_gpu[:, None]
        return self._packed_attention(
            q_kv,
            mask,
            k_full,
            v_full,
            kv_head_idx,
            max_seq_len,
            max_seq_len,
        )

    def _packed_attention(
        self,
        q_kv: torch.Tensor,
        mask: torch.Tensor,  # [bs, max_seq_len]
        k_full: torch.Tensor,
        v_full: torch.Tensor,
        kv_head_idx: int,
        max_seq_len: int,
        max_seqlen_k_bound: int,
    ) -> torch.Tensor:
        bs, num_groups, head_dim = q_kv.shape
        device = q_kv.device

        kv_lens = mask.sum(dim=1).to(torch.int32)
        max_kv_len = int(kv_lens.max().item())

        if max_kv_len == 0:
            return torch.zeros_like(q_kv)

        # SDPA-based fallback (replaces flashinfer xqa_continuous):
        # gather selected KV per batch into a packed [bs, max_kv_len, head_dim] then run SDPA.
        target_dtype = q_kv.dtype
        k_packed = torch.zeros(
            bs, max_kv_len, head_dim, dtype=target_dtype, device=device
        )
        v_packed = torch.zeros_like(k_packed)
        for b in range(bs):
            sel = mask[b].nonzero(as_tuple=True)[0]
            n = sel.shape[0]
            if n > 0:
                k_packed[b, :n] = k_full[b, sel, kv_head_idx].to(target_dtype)
                v_packed[b, :n] = v_full[b, sel, kv_head_idx].to(target_dtype)

        q_sdpa = q_kv.unsqueeze(2)  # [bs, num_groups, 1, head_dim]
        # k/v: [bs, 1, max_kv_len, head_dim] → expand to num_groups for SDPA (GQA)
        k_sdpa = k_packed.unsqueeze(1).expand(-1, num_groups, -1, -1).contiguous()
        v_sdpa = v_packed.unsqueeze(1).expand(-1, num_groups, -1, -1).contiguous()

        pos = torch.arange(max_kv_len, device=device)
        valid = pos[None, :] < kv_lens[:, None]
        attn_mask = valid[:, None, None, :]  # [bs, 1, 1, max_kv_len]

        out = torch.nn.functional.scaled_dot_product_attention(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            attn_mask=attn_mask,
            scale=self.scaling,
        )
        return out.squeeze(2)  # [bs, num_groups, head_dim]
