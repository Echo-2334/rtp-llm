"""Torch Naive PQ Attention Backend.

PQ (Product Quantization) 加速 decode attention：
- Prefill：将 head_dim 切成 num_subspaces 个子空间，每个子空间独立 K-Means 聚类
- Decode：q_sub @ centroids_sub 得到每个子空间的簇分数，按 token 的 cids 累加；
  每个 q-head 选 top_k_tokens，同 KV group 内取 union，再对 union + 新 decode token
  做 full attention。

默认配置：num_subspaces=16, num_clusters=256, top_k_tokens=2000.
"""

import logging
import os
from typing import Optional

import torch
from torch.nn.functional import scaled_dot_product_attention

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
# Global PQ cache: key -> {"cids": [S, N], "cents": [S, K, sub_dim], "prefill_len": int}
# ============================================================================
_PQ_CACHE: dict = {}


def _pq_key(layer_id: int, seq_idx: int, kv_head_idx: int) -> str:
    return f"pq_layer_{layer_id}_seq_{seq_idx}_kv_head_{kv_head_idx}"


def _pq_score_and_select(
    q_group: torch.Tensor,  # [num_q, head_dim]
    cids: torch.Tensor,  # [S, clustered_len]
    cents: torch.Tensor,  # [S, K, sub_dim]
    top_k: int,
) -> torch.Tensor:
    """对该 KV head 的 q_group 计算 PQ 分数，每个 q-head 选 top_k，取 union.

    Returns:
        selected: [num_selected] (long, sorted unique token indices)
    """
    num_q, head_dim = q_group.shape
    num_subspaces, clustered_len = cids.shape
    sub_dim = head_dim // num_subspaces

    q_subs = q_group.reshape(num_q, num_subspaces, sub_dim)  # [Q, S, sub_dim]
    # 每子空间每簇得分: [Q, S, K]
    cluster_scores = torch.einsum("qsd,skd->qsk", q_subs, cents)
    # 查表展开到每 token: [Q, S, N]
    cids_exp = cids.unsqueeze(0).expand(num_q, -1, -1)  # [Q, S, N]
    token_subspace = cluster_scores.gather(2, cids_exp)  # [Q, S, N]
    token_scores = token_subspace.sum(dim=1)  # [Q, N]

    per_head_k = min(top_k, clustered_len)
    topk_ids = token_scores.topk(per_head_k, dim=-1).indices  # [Q, per_head_k]
    return torch.unique(topk_ids.flatten())


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

        logging.debug(
            f"TorchNaivePQPrefillImpl: S={self.num_subspaces}, K={self.num_clusters}, "
            f"sub_dim={self.head_dim // self.num_subspaces}"
        )

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        # 1. RoPE
        if self.need_rope_kv_cache:
            qkv = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)

        # 2. Split QKV
        q, k, v = self._split_qkv(qkv)

        # 3. PQ 聚类（在写 cache 前对 K 做）
        self._perform_pq_clustering(k, kv_cache)

        # 4. Write cache
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # 5. 全 K 走标准 flash attention
        output = self._run_attention_extend(q, k, v)
        output = output.reshape(output.shape[0], -1)
        return output

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

        for seq_idx in range(batch_size):
            start_idx = cu_seqlens[seq_idx].item()
            end_idx = cu_seqlens[seq_idx + 1].item()
            seq_len = end_idx - start_idx

            # 把 num_kv_heads × S 合成 batch 维：[H*S, seq_len, sub_dim]
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
                init_centroids=None,
                verbose=False,
            )
            cids_b = cids_b.to(torch.int64).reshape(num_kv_heads, S, seq_len)
            cents_b = cents_b.reshape(num_kv_heads, S, self.num_clusters, sub_dim)

            for h in range(num_kv_heads):
                _PQ_CACHE[_pq_key(layer_id, seq_idx, h)] = {
                    "cids": cids_b[h],  # [S, seq_len]
                    "cents": cents_b[h],  # [S, K, sub_dim]
                    "prefill_len": seq_len,
                }

            logging.debug(
                f"[PQ Prefill] layer={layer_id} seq={seq_idx} "
                f"H={num_kv_heads} S={S} K={self.num_clusters} seq_len={seq_len}"
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

        logging.debug(
            f"TorchNaivePQDecodeImpl: S={self.num_subspaces}, top_k={self.top_k_tokens}"
        )

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        # 1. RoPE
        if self.need_rope_kv_cache:
            q = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
            q = q.reshape(q.shape[0], self.num_heads, self.head_dim)
        else:
            q, _, _ = self._split_qkv(qkv)

        # 2. Write cache（含本步新 K, V）
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # 3. Read full K, V
        k_full, v_full = self._read_kv_from_cache(kv_cache)

        # 4. PQ-guided sparse attention
        output = self._run_pq_attention_decode(q, k_full, v_full, kv_cache)
        output = output.reshape(output.shape[0], -1)
        return output

    def _compute_kv_seq_lens(self, batch_size: int) -> torch.Tensor:
        """每条序列的实际 KV 长度（CPU [batch_size]）— k_full 是 padded 的。"""
        from rtp_llm.ops.compute_ops import fill_mla_params

        ai = self.attn_inputs
        params = fill_mla_params(
            (
                ai.prefix_lengths
                if getattr(ai, "prefix_lengths", None) is not None
                else torch.tensor([], dtype=torch.int32)
            ),
            ai.sequence_lengths,
            ai.input_lengths,
            (
                ai.kv_cache_block_id_host
                if ai.kv_cache_block_id_host is not None
                else torch.tensor([], dtype=torch.int32)
            ),
            self.tokens_per_block,
        )
        return params.kvlen_h[:batch_size]

    def _run_pq_attention_decode(
        self,
        q: torch.Tensor,  # [batch, num_heads, head_dim]
        k_full: torch.Tensor,  # [batch, max_seq_len, num_kv_heads, head_dim] (padded)
        v_full: torch.Tensor,
        kv_cache: Optional[KVCache],
    ) -> torch.Tensor:
        batch_size = q.shape[0]
        layer_id = kv_cache.layer_id if kv_cache is not None else 0
        num_groups = self.num_heads // self.num_kv_heads if self.enable_gqa else 1
        seq_lens = self._compute_kv_seq_lens(batch_size)

        output = torch.empty_like(q)

        for batch_idx in range(batch_size):
            total_seq_len = int(seq_lens[batch_idx])
            for kv_head_idx in range(self.num_kv_heads):
                key = _pq_key(layer_id, batch_idx, kv_head_idx)

                start_h = kv_head_idx * num_groups
                end_h = start_h + num_groups
                q_group = q[batch_idx, start_h:end_h, :]  # [num_groups, head_dim]

                # 没聚类信息 → 退回 full attention（用实际长度，避开 padding）
                if key not in _PQ_CACHE:
                    kv_k = k_full[batch_idx, :total_seq_len, kv_head_idx, :]
                    kv_v = v_full[batch_idx, :total_seq_len, kv_head_idx, :]
                    output[batch_idx, start_h:end_h, :] = (
                        self._full_attention_gqa_group(q_group, kv_k, kv_v)
                    )
                    continue

                pq = _PQ_CACHE[key]
                cids = pq["cids"]  # [S, prefill_len]
                cents = pq["cents"]  # [S, K, sub_dim]
                prefill_len = pq["prefill_len"]

                # PQ 打分 + top-k union（仅在 prefill 部分内做）
                selected = _pq_score_and_select(
                    q_group, cids, cents, self.top_k_tokens
                )  # [num_selected]

                # 加上 prefill 之后的新 decode token（用本 batch 实际长度，避免误选 padding）
                if total_seq_len > prefill_len:
                    new_tokens = torch.arange(
                        prefill_len,
                        total_seq_len,
                        device=selected.device,
                        dtype=selected.dtype,
                    )
                    selected = torch.cat([selected, new_tokens])

                kv_k = k_full[batch_idx, selected, kv_head_idx, :]
                kv_v = v_full[batch_idx, selected, kv_head_idx, :]
                output[batch_idx, start_h:end_h, :] = self._full_attention_gqa_group(
                    q_group, kv_k, kv_v
                )

                logging.info(
                    f"[PQ Decode] layer={layer_id} seq={batch_idx} kv_head={kv_head_idx} "
                    f"selected={selected.shape[0]}/{total_seq_len}"
                )

        return output

    def _full_attention_gqa_group(
        self,
        q_group: torch.Tensor,  # [num_q, head_dim]
        k: torch.Tensor,  # [num_selected, head_dim]
        v: torch.Tensor,  # [num_selected, head_dim]
    ) -> torch.Tensor:
        """同 KV head 内多 Q heads 共享 K/V 的 attention."""
        q = q_group.unsqueeze(0).unsqueeze(2)  # [1, num_q, 1, head_dim]
        k = k.unsqueeze(0).unsqueeze(0)  # [1, 1, num_selected, head_dim]
        v = v.unsqueeze(0).unsqueeze(0)

        if not (q.dtype == k.dtype == v.dtype):
            k = k.to(q.dtype)
            v = v.to(q.dtype)

        out = scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scaling,
        )
        return out.squeeze(0).squeeze(1)  # [num_q, head_dim]
