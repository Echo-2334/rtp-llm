"""Torch Naive Attention Backend - Fallback implementation using PyTorch's scaled_dot_product_attention.

This module provides a fallback attention implementation that uses PyTorch's native
scaled_dot_product_attention function. It serves as a lowest-priority backend that
works in any environment, useful for debugging and development.

Reference: SGLang's torch_native_backend.py
"""

import logging
import os
import time
from typing import Optional

import torch
from torch.nn.functional import scaled_dot_product_attention

from rtp_llm.models_py.modules.factory.attention.cuda_impl.local_kmeans import (
    batch_kmeans_Euclid,
)

# Try to import optimized kernels for centroid scoring (optional)
try:
    from rtp_llm.models_py.modules.factory.attention.cuda_impl.triton_kernels import (
        fused_centroid_scoring_topp,
        fused_centroid_scoring_topp_vectorized,
    )

    OPTIMIZED_KERNEL_AVAILABLE = True
except ImportError:
    OPTIMIZED_KERNEL_AVAILABLE = False
    logging.debug("Optimized kernels not available, using naive PyTorch fallback")

# Import cluster utilities for CSR token gathering
try:
    from rtp_llm.models_py.modules.factory.attention.cuda_impl.cluster_utils import (
        build_attention_mask_vectorized,
        gather_tokens_from_clusters_batch_csr,
        gather_tokens_from_clusters_csr,
        precompute_csr_cache,
    )

    CSR_UTILS_AVAILABLE = True
except ImportError:
    CSR_UTILS_AVAILABLE = False
    logging.debug("CSR cluster utilities not available")

# Try to import fused Triton kernels for optimized cluster gathering
try:
    from rtp_llm.models_py.modules.factory.attention.cuda_impl.triton_kernels.fused_cluster_gather import (
        TRITON_AVAILABLE as FUSED_KERNEL_AVAILABLE,
    )
    from rtp_llm.models_py.modules.factory.attention.cuda_impl.triton_kernels.fused_cluster_gather import (
        fused_cluster_union_gather_kv,
    )

    FUSED_CLUSTER_GATHER_AVAILABLE = FUSED_KERNEL_AVAILABLE
except ImportError:
    FUSED_CLUSTER_GATHER_AVAILABLE = False
    logging.debug("Fused cluster gather kernel not available")

from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.cuda_impl.kv_cache_write_op import (
    KVCacheWriteOp,
)
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import AttentionConfigs, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCacheDecodeOp,
    FusedRopeKVCachePrefillOpQKVOut,
    KVCache,
    PyAttentionInputs,
    rtp_llm_ops,
)

# ============================================================================
# Dummy FMHA Params for Interface Compatibility
# ============================================================================


class DummyFMHAParams:
    """Dummy FMHA params for TorchNaive implementation.

    This class provides interface compatibility with PyModelOutputs which expects
    an fmha_params object. Since TorchNaive doesn't use FlashInfer's FMHA operations,
    we provide a minimal dummy implementation.
    """

    def fill_params(
        self,
        sequence_lengths,
        input_lengths,
        kv_cache_block_id_host,
        batch_size,
        seq_size_per_block,
    ):
        """Dummy implementation for CUDA graph compatibility.

        This method is required by the ParamsBase interface but is not used
        by TorchNaive since it doesn't participate in CUDA graph execution.
        """
        pass


# ============================================================================
# FP4 E2M1 Simulated Quantization for K/V
# ============================================================================

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
FP4_BLOCK_SIZE = 16


def _round_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    """将 float 值就近舍入到 FP4 E2M1 的 16 个离散值。"""
    sign = torch.sign(x)
    a = torch.abs(x)
    out = torch.zeros_like(a)
    out[a > 0.25] = 0.5
    out[a >= 0.75] = 1.0
    out[a > 1.25] = 1.5
    out[a >= 1.75] = 2.0
    out[a > 2.5] = 3.0
    out[a >= 3.5] = 4.0
    out[a > 5.0] = 6.0
    # return sign
    return out * sign


def _fp4_simulate_quant(x: torch.Tensor) -> torch.Tensor:
    """FP4 E2M1 模拟量化：quantize → dequantize，返回 float tensor。

    基于 NVFP4 block-wise 量化：
    1. 按 block_size=16 分组，计算 per-block scale
    2. 缩放到 FP4 范围 [-6, 6]
    3. 就近舍入到 E2M1 离散值
    4. 反量化还原

    Args:
        x: 输入 tensor，最后一维需要能被 16 整除
    Returns:
        模拟量化后的 float tensor，shape 与输入相同
    """
    orig_shape = x.shape
    orig_dtype = x.dtype
    # 展平为 2D: [*, head_dim]
    x_2d = x.reshape(-1, orig_shape[-1]).float()
    m, n = x_2d.shape

    # 计算 global_scale
    tensor_amax = torch.abs(x_2d).max()
    global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax.clamp(min=1e-12)

    # Block-wise 量化
    x_blocked = x_2d.reshape(m, n // FP4_BLOCK_SIZE, FP4_BLOCK_SIZE)
    vec_max = torch.max(torch.abs(x_blocked), dim=-1, keepdim=True)[0].float()

    # per-block scale (模拟 FP8 精度)
    scale = global_scale * (vec_max / FLOAT4_E2M1_MAX)
    scale = scale.to(torch.float8_e4m3fn).float()

    # 缩放 + clamp
    output_scale = 1.0 / (scale / global_scale).clamp(min=1e-12)
    scaled_x = x_blocked.float() * output_scale
    clipped_x = torch.clamp(scaled_x, -6.0, 6.0)

    # 就近舍入到 E2M1 离散值
    quantized = _round_to_e2m1(clipped_x)

    # 反量化：还原 scale
    dequantized = quantized / output_scale
    logging.info(f"Quant For KV Cache")

    return dequantized.reshape(orig_shape).to(orig_dtype)


def _clustered_residual_fp4_quant(
    x: torch.Tensor,  # [seq_len, head_dim]
    cluster_ratio: int,
    max_iters: int = 20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """聚类残差 FP4 伪量化（单 head）。

    Returns:
        reconstructed: [seq_len, head_dim]
        centroids: [num_clusters, head_dim] 质心
        labels: [seq_len] 簇分配
    """
    seq_len = x.shape[0]
    num_clusters = max(1, seq_len // cluster_ratio)

    centroids, labels, _ = _kmeans_clustering(
        x, num_clusters, max_iters=max_iters, use_kmeanspp=False, build_indices=False
    )
    diff = x - centroids[labels]
    diff_q = _fp4_simulate_quant(diff)
    reconstructed = centroids[labels] + diff_q

    return reconstructed, centroids, labels


def _apply_residual_fp4_decode(
    k_full: torch.Tensor,  # [batch, total_seq_len, num_kv_heads, head_dim]
    v_full: torch.Tensor,
    kv_cache: Optional[KVCache],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode 阶段复用 prefill 聚类结果。

    历史 token: centroid[label] + fp4_quant(diff)
    新增 token: fp8 伪量化
    """
    layer_id = kv_cache.layer_id if kv_cache is not None else 0
    k_info = _CLUSTER_CACHE.get(f"residual_k_{layer_id}")
    v_info = _CLUSTER_CACHE.get(f"residual_v_{layer_id}")

    if k_info is None:
        return k_full, v_full

    orig_dtype = k_full.dtype
    prefill_len = k_info["labels"].shape[0]

    for b in range(k_full.shape[0]):
        # --- K ---
        k_hist = k_full[b, :prefill_len, 0, :]
        centroids = k_info["centroids"]
        labels = k_info["labels"]
        diff = k_hist - centroids[labels]
        k_full[b, :prefill_len, 0, :] = centroids[labels] + _fp4_simulate_quant(diff)

        k_new = k_full[b, prefill_len:, 0, :]
        k_full[b, prefill_len:, 0, :] = k_new.to(torch.float8_e4m3fn).to(orig_dtype)

        # --- V ---
        v_hist = v_full[b, :prefill_len, 0, :]
        v_centroids = v_info["centroids"]
        v_labels = v_info["labels"]
        v_diff = v_hist - v_centroids[v_labels]
        v_full[b, :prefill_len, 0, :] = v_centroids[v_labels] + _fp4_simulate_quant(
            v_diff
        )

        v_new = v_full[b, prefill_len:, 0, :]
        v_full[b, prefill_len:, 0, :] = v_new.to(torch.float8_e4m3fn).to(orig_dtype)

    return k_full, v_full


# ============================================================================
# K-Clustering Utilities for Attention Acceleration
# ============================================================================

# 全局聚类信息缓存
_CLUSTER_CACHE = {}  # key: "layer_{id}_seq_{idx}_head_{idx}" -> cluster_info


def _build_cluster_indices_gpu(
    labels: torch.Tensor,  # [seq_len]
    num_clusters: int,
) -> list:
    """Build cluster_indices on GPU to avoid CPU sync.

    Args:
        labels: Cluster assignment for each token [seq_len]
        num_clusters: Number of clusters

    Returns:
        cluster_indices: list[list[int]] tokens grouped by cluster
    """
    # Use GPU operations to build indices
    cluster_indices = []
    for k in range(num_clusters):
        mask = labels == k
        cluster_tokens = torch.nonzero(mask).squeeze(1)  # [num_tokens_in_cluster]
        # Convert to Python list only once at the end
        cluster_indices.append(cluster_tokens.tolist())

    return cluster_indices


def _kmeans_clustering(
    k: torch.Tensor,  # [seq_len, head_dim]
    num_clusters: int,
    max_iters: int = 10,
    use_optimized: bool = True,
    use_kmeanspp: bool = True,
    build_indices: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, list]:
    """K-Means clustering using optimized batch K-means.

    Args:
        k: Key tensor for clustering [seq_len, head_dim]
        num_clusters: Number of clusters
        max_iters: Maximum iterations
        use_optimized: Whether to use optimized scatter-based K-means
        use_kmeanspp: Whether to use K-means++ initialization (slower but better)
        build_indices: Whether to build cluster_indices (expensive, skip if not needed)

    Returns:
        centroids: [num_clusters, head_dim] cluster centers
        labels: [seq_len] cluster assignment per token
        cluster_indices: list[list[int]] tokens grouped by cluster (empty if build_indices=False)
    """
    # Add batch dimension: [seq_len, head_dim] -> [1, seq_len, head_dim]
    k_batched = k.unsqueeze(0)

    # Call optimized K-means
    if use_optimized:
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.local_kmeans import (
            batch_kmeans_Euclid_optimized,
        )

        cluster_ids, centroids, n_iters = batch_kmeans_Euclid_optimized(
            k_batched,
            num_clusters,
            max_iters=max_iters,
            tol=1e-4,
            init_centroids=None,
            verbose=False,
            use_kmeanspp=use_kmeanspp,
        )
    else:
        cluster_ids, centroids, n_iters = batch_kmeans_Euclid(
            k_batched,
            num_clusters,
            max_iters=max_iters,
            tol=1e-4,
            init_centroids=None,
            verbose=False,
            use_kmeanspp=use_kmeanspp,
        )

    # Remove batch dimension
    labels = cluster_ids.squeeze(0)  # [seq_len]
    centroids = centroids.squeeze(0)  # [num_clusters, head_dim]

    # Build cluster_indices (expensive due to GPU sync, skip if not needed)
    cluster_indices = (
        _build_cluster_indices_gpu(labels, num_clusters) if build_indices else []
    )

    return centroids, labels, cluster_indices


def _update_local_window(cluster_info: dict) -> bool:
    """更新 Local Window 计数，检查是否需要触发合并.

    Args:
        cluster_info: 聚类信息字典

    Returns:
        bool: 如果 Local Window 满了需要触发合并，返回 True
    """
    local_window = cluster_info["local_window"]
    cluster_info["seq_len"] += 1
    local_window["count"] += 1

    # 检查是否满了
    if local_window["count"] >= local_window["window_size"]:
        return True  # 需要触发合并
    return False


def _merge_local_window_to_global(
    cluster_info: dict,
    kv_cache,
    impl_obj,
    layer_idx: int,
    head_idx: int,
) -> None:
    """对 Local Window 独立聚类并合并到全局.

    这个函数会：
    1. 从 KV Cache 提取 Local Window 的 K 向量
    2. 对这些 K 向量独立聚类（不考虑全局质心）
    3. 将新簇的信息追加到全局聚类状态中
    4. 重置 Local Window

    Args:
        cluster_info: 聚类信息字典
        kv_cache: KV cache 对象
        impl_obj: 实现对象（用于调用 _read_kv_from_cache）
        layer_idx: 层索引
        head_idx: 头索引
    """
    local_window = cluster_info["local_window"]
    start_idx = local_window["start_idx"]
    count = local_window["count"]

    if count == 0:
        return  # 没有需要合并的数据

    # 1. 从 KV Cache 提取 Local Window 的 K 向量
    k_full_temp, _ = impl_obj._read_kv_from_cache(kv_cache)
    # k_full_temp: [B, seq_len, num_kv_heads, head_dim]

    # 提取对应 head 和 Local Window 范围的 K
    k_local = k_full_temp[
        :, start_idx : start_idx + count, head_idx, :
    ]  # [B, count, head_dim]
    k_local = k_local.squeeze(0)  # [count, head_dim], 假设 batch_size=1

    # 2. 对 Local Window 独立聚类（类似 Prefill 阶段）
    # 计算簇数量：例如每 64 个 token 一个簇
    cluster_ratio = int(os.getenv("CLUSTER_RATIO", "64"))
    num_clusters_local = max(1, count // cluster_ratio)

    # 调用 K-means 聚类（复用 Prefill 阶段的逻辑）
    centroids_local, labels_local, cluster_indices_local_relative = _kmeans_clustering(
        k_local,
        num_clusters=num_clusters_local,
        max_iters=int(os.getenv("KMEANS_ITERS", "20")),
    )

    # 3. 将相对索引转换为全局索引
    cluster_indices_local = []
    for cluster_relative_indices in cluster_indices_local_relative:
        global_indices = [start_idx + idx for idx in cluster_relative_indices]
        cluster_indices_local.append(global_indices)

    # 4. 计算新簇的 cluster_sizes
    labels_local = labels_local.reshape(-1).to(torch.int64)
    cluster_sizes_local = torch.bincount(labels_local, minlength=num_clusters_local)

    # 5. 将新簇追加到全局信息中
    # 5a. 追加 centroids
    cluster_info["centroids"] = torch.cat(
        [cluster_info["centroids"], centroids_local], dim=0
    )  # [K_old + K_local, D]

    # 5b. 追加 cluster_indices
    cluster_info["cluster_indices"].extend(cluster_indices_local)

    # 5c. 追加 cluster_sizes
    cluster_info["cluster_sizes"] = torch.cat(
        [cluster_info["cluster_sizes"], cluster_sizes_local], dim=0
    )  # [K_old + K_local]

    # 6. 重置 Local Window
    local_window["start_idx"] = cluster_info["seq_len"]
    local_window["count"] = 0

    logging.info(
        f"Merged Local Window to global: layer_{layer_idx}_head_{head_idx}, "
        f"added {num_clusters_local} clusters from {count} tokens, "
        f"total clusters: {len(cluster_info['cluster_indices'])}"
    )


class TorchNaivePrefillImpl(FMHAImplBase):
    """Torch Naive Prefill Attention Implementation.

    Uses PyTorch's scaled_dot_product_attention as a fallback when optimized
    kernels are not available. Processes sequences one at a time in a loop.
    """

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        """Initialize Torch Naive Prefill implementation.

        Args:
            attn_configs: Attention configuration
            attn_inputs: Attention inputs
            parallelism_config: Parallelism configuration (optional)
        """
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs

        # Extract configuration
        self.num_heads = attn_configs.head_num
        self.num_kv_heads = attn_configs.kv_head_num
        self.head_dim = attn_configs.size_per_head
        self.scaling = 1.0 / (self.head_dim**0.5)
        self.enable_gqa = self.num_heads != self.num_kv_heads

        # Create RoPE and KV Cache write operations
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.rope_kvcache_impl = FusedRopeKVCachePrefillOpQKVOut(attn_configs)

        # Create write cache store implementation
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)

        # Create dummy fmha_params for interface compatibility (PyModelOutputs expects it)
        self.fmha_params = DummyFMHAParams()

        logging.debug(
            f"TorchNaivePrefillImpl initialized: heads={self.num_heads}, "
            f"kv_heads={self.num_kv_heads}, head_dim={self.head_dim}, gqa={self.enable_gqa}"
        )

    @classmethod
    def support(
        cls,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
    ) -> bool:
        """Check if this implementation supports the given configuration.

        Always returns True as a fallback, except for unsupported cases.

        Args:
            attn_configs: Attention configuration
            attn_inputs: Attention inputs

        Returns:
            True if supported, False otherwise
        """
        # Don't support MLA
        if attn_configs.use_mla:
            return False

        # Always support as fallback
        return True

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
    ) -> torch.Tensor:
        """Forward pass for prefill attention.

        Args:
            qkv: Input QKV tensor [total_tokens, (num_heads + 2*num_kv_heads) * head_dim]
            kv_cache: KV cache object (optional)

        Returns:
            Attention output [total_tokens, num_heads * head_dim]
        """
        # 1. Apply RoPE if needed
        if self.need_rope_kv_cache:
            qkv = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)

        # 2. Split QKV
        q, k, v = self._split_qkv(qkv)

        # 4. Apply write cache store (for prefill with prefix)
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        k = _fp4_simulate_quant(k)
        v = _fp4_simulate_quant(v)

        # 5. Execute attention (K, V are already complete for prefill)
        output = self._run_attention_extend(q, k, v)

        # 6. Reshape output to [total_tokens, num_heads * head_dim]
        output = output.reshape(output.shape[0], -1)

        return output

    def _perform_k_clustering_if_available(
        self,
        k: torch.Tensor,  # [total_tokens, num_kv_heads, head_dim]
        kv_cache: Optional[KVCache],
    ) -> None:
        """对 K 进行聚类（可选）用于 Decode 阶段加速.

        此方法在原始 Prefill 中调用，为 Decode 的聚类做准备。
        """
        import os

        cluster_ratio = int(os.getenv("CLUSTER_RATIO", "64"))
        kmeans_iters = int(os.getenv("KMEANS_ITERS", "20"))

        if kv_cache is None:
            return

        layer_id = kv_cache.layer_id
        batch_size = self.attn_inputs.input_lengths.size(0)
        cu_seqlens = self.attn_inputs.cu_seqlens[: batch_size + 1]

        # 按序列和 head 聚类
        for seq_idx in range(batch_size):
            start_idx = cu_seqlens[seq_idx].item()
            end_idx = cu_seqlens[seq_idx + 1].item()
            seq_len = end_idx - start_idx

            per_seq_k = k[start_idx:end_idx, :, :]  # [seq_len, num_kv_heads, head_dim]

            # 对每个 KV head 独立聚类
            for head_idx in range(per_seq_k.shape[1]):
                k_head = per_seq_k[:, head_idx, :]  # [seq_len, head_dim]

                # 计算簇数量
                num_clusters = max(1, seq_len // cluster_ratio)

                # K-Means 聚类
                centroids, labels, cluster_indices = _kmeans_clustering(
                    k_head, num_clusters, max_iters=kmeans_iters
                )

                # 计算每个簇的大小
                cluster_sizes = torch.bincount(labels, minlength=num_clusters)

                # 存储到全局缓存
                key = f"layer_{layer_id}_seq_{seq_idx}_head_{head_idx}"
                _CLUSTER_CACHE[key] = {
                    "centroids": centroids,
                    "cluster_sizes": cluster_sizes,
                    "cluster_indices": cluster_indices,
                    "seq_len": seq_len,
                }

                logging.debug(
                    f"K-Clustering (from Prefill): {key}, seq_len={seq_len}, "
                    f"num_clusters={num_clusters}, sizes={cluster_sizes.tolist()}"
                )

    def _split_qkv(
        self, qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split QKV tensor into query, key, value.

        Args:
            qkv: QKV tensor [total_tokens, (num_heads + 2*num_kv_heads) * head_dim]

        Returns:
            Tuple of (query, key, value) tensors
            - query: [total_tokens, num_heads, head_dim]
            - key: [total_tokens, num_kv_heads, head_dim]
            - value: [total_tokens, num_kv_heads, head_dim]
        """
        qkv = qkv.reshape(qkv.shape[0], -1)

        q, k, v = torch.split(
            qkv,
            [
                self.head_dim * self.num_heads,
                self.head_dim * self.num_kv_heads,
                self.head_dim * self.num_kv_heads,
            ],
            dim=-1,
        )

        q = q.reshape(q.shape[0], self.num_heads, self.head_dim)
        k = k.reshape(k.shape[0], self.num_kv_heads, self.head_dim)
        v = v.reshape(v.shape[0], self.num_kv_heads, self.head_dim)

        return q, k, v

    def _run_attention_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Execute prefill attention with causal masking.

        Processes each sequence separately due to varying sequence lengths.
        This is inefficient but works as a fallback.

        Args:
            q: Query tensor [total_tokens, num_heads, head_dim]
            k: Key tensor [total_tokens, num_kv_heads, head_dim]
            v: Value tensor [total_tokens, num_kv_heads, head_dim]

        Returns:
            Attention output [total_tokens, num_heads, head_dim]
        """
        # Get sequence information
        batch_size = self.attn_inputs.input_lengths.size(0)
        cu_seqlens = self.attn_inputs.cu_seqlens[: batch_size + 1]

        # Prepare output tensor
        output = torch.empty_like(q)

        # Process each sequence separately
        for seq_idx in range(batch_size):
            start_idx = cu_seqlens[seq_idx].item()
            end_idx = cu_seqlens[seq_idx + 1].item()
            seq_len = end_idx - start_idx

            # Extract per-sequence tensors
            per_seq_q = q[start_idx:end_idx, :, :]  # [seq_len, num_heads, head_dim]
            per_seq_k = k[start_idx:end_idx, :, :]  # [seq_len, num_kv_heads, head_dim]
            per_seq_v = v[start_idx:end_idx, :, :]  # [seq_len, num_kv_heads, head_dim]

            # Handle GQA: expand K, V heads to match Q heads
            if self.enable_gqa:
                # Repeat K, V heads: [seq_len, num_kv_heads, head_dim] -> [seq_len, num_heads, head_dim]
                num_groups = self.num_heads // self.num_kv_heads
                per_seq_k = per_seq_k.repeat_interleave(num_groups, dim=1)
                per_seq_v = per_seq_v.repeat_interleave(num_groups, dim=1)

            # Transpose for SDPA: [num_heads, seq_len, head_dim]
            per_seq_q = per_seq_q.movedim(0, 1)
            per_seq_k = per_seq_k.movedim(0, 1)
            per_seq_v = per_seq_v.movedim(0, 1)

            # Handle dtype mismatch (SDPA requires same dtype)
            if not (per_seq_q.dtype == per_seq_k.dtype == per_seq_v.dtype):
                per_seq_k = per_seq_k.to(per_seq_q.dtype)
                per_seq_v = per_seq_v.to(per_seq_q.dtype)

            # Execute scaled_dot_product_attention
            # Add batch dimension: [1, num_heads, seq_len, head_dim]
            per_seq_out = scaled_dot_product_attention(
                per_seq_q.unsqueeze(0),
                per_seq_k.unsqueeze(0),
                per_seq_v.unsqueeze(0),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
                scale=self.scaling,
            ).squeeze(0)

            # Transpose back: [seq_len, num_heads, head_dim]
            per_seq_out = per_seq_out.movedim(1, 0)

            # Store result
            output[start_idx:end_idx, :, :] = per_seq_out

        return output


class TorchNaiveDecodeImpl(FMHAImplBase):
    """Torch Naive Decode Attention Implementation.

    Uses PyTorch's scaled_dot_product_attention for decode phase.
    Currently a placeholder - will be implemented in Phase 2.
    """

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        """Initialize Torch Naive Decode implementation.

        Args:
            attn_configs: Attention configuration
            attn_inputs: Attention inputs
            parallelism_config: Parallelism configuration (optional)
        """
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs

        # Extract configuration
        self.num_heads = attn_configs.head_num
        self.num_kv_heads = attn_configs.kv_head_num
        self.head_dim = attn_configs.size_per_head
        self.scaling = 1.0 / (self.head_dim**0.5)
        self.enable_gqa = self.num_heads != self.num_kv_heads
        self.tokens_per_block = attn_configs.tokens_per_block

        # Create RoPE and KV Cache operations
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.rope_kvcache_impl = FusedRopeKVCacheDecodeOp(attn_configs)

        # Create write cache store implementation
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)

        # Create dummy fmha_params for interface compatibility (PyModelOutputs expects it)
        self.fmha_params = DummyFMHAParams()

        logging.debug(
            f"TorchNaiveDecodeImpl initialized: heads={self.num_heads}, "
            f"kv_heads={self.num_kv_heads}, head_dim={self.head_dim}, gqa={self.enable_gqa}"
        )

    @classmethod
    def support(
        cls,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
    ) -> bool:
        """Check if this implementation supports the given configuration.

        Always returns True as a fallback, except for unsupported cases.

        Args:
            attn_configs: Attention configuration
            attn_inputs: Attention inputs

        Returns:
            True if supported, False otherwise
        """
        # Don't support MLA
        if attn_configs.use_mla:
            return False

        # Always support as fallback
        return True

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
    ) -> torch.Tensor:
        """Forward pass for decode attention.

        Args:
            qkv: Input QKV tensor [batch_size, (num_heads + 2*num_kv_heads) * head_dim]
            kv_cache: KV cache object (required for decode)

        Returns:
            Attention output [batch_size, num_heads * head_dim]
        """
        # 1. Apply RoPE if needed
        # NOTE: Decode RoPE writes K,V to cache directly and only returns Q
        if self.need_rope_kv_cache:
            q = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)

            # RoPE may return Q in different shapes, normalize to [batch, num_heads, head_dim]
            if q.ndim == 2:
                # 2D: [batch, num_heads * head_dim] -> reshape to 3D
                q = q.reshape(q.shape[0], self.num_heads, self.head_dim)
            elif q.ndim == 3:
                # Already 3D [batch, num_heads, head_dim] - no change needed
                pass
            else:
                raise ValueError(f"Unexpected Q shape from RoPE: {q.shape}")
        else:
            # No RoPE: split QKV manually (though this path is unlikely for decode)
            q, k, v = self._split_qkv(qkv)

        # 4. Apply write cache store (for decode with new tokens)
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # 5. Read complete K, V from cache (including history)
        k_full, v_full = self._read_kv_from_cache(kv_cache)

        k_full = _fp4_simulate_quant(k_full)
        v_full = _fp4_simulate_quant(v_full)

        # 6. Execute decode attention (no causal mask needed - single query token)
        output = self._run_attention_decode(q, k_full, v_full)
        logging.info(f"[Decode] output shape: {output.shape}")

        # 7. Reshape output to [batch_size, num_heads * head_dim]
        output = output.reshape(output.shape[0], -1)

        return output

    def _split_qkv(
        self, qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split QKV tensor into query, key, value.

        Args:
            qkv: QKV tensor [batch_size, (num_heads + 2*num_kv_heads) * head_dim]

        Returns:
            Tuple of (query, key, value) tensors
            - query: [batch_size, num_heads, head_dim]
            - key: [batch_size, num_kv_heads, head_dim]
            - value: [batch_size, num_kv_heads, head_dim]
        """
        qkv = qkv.reshape(qkv.shape[0], -1)

        # Debug: check dimensions
        expected_size = self.head_dim * (self.num_heads + 2 * self.num_kv_heads)
        actual_size = qkv.shape[-1]
        if expected_size != actual_size:
            logging.error(
                f"QKV size mismatch in {self.__class__.__name__}: "
                f"expected {expected_size} (heads={self.num_heads}, kv_heads={self.num_kv_heads}, "
                f"head_dim={self.head_dim}), got {actual_size}"
            )
            # Adjust num_heads based on actual size
            # This might happen if RoPE changed the format
            actual_qkv_heads = actual_size // self.head_dim
            if actual_qkv_heads < 2 * self.num_kv_heads:
                logging.error(f"Cannot split: not enough space for K and V")
                raise ValueError(
                    f"QKV size {actual_size} too small for kv_heads={self.num_kv_heads}"
                )

        q, k, v = torch.split(
            qkv,
            [
                self.head_dim * self.num_heads,
                self.head_dim * self.num_kv_heads,
                self.head_dim * self.num_kv_heads,
            ],
            dim=-1,
        )

        q = q.reshape(q.shape[0], self.num_heads, self.head_dim)
        k = k.reshape(k.shape[0], self.num_kv_heads, self.head_dim)
        v = v.reshape(v.shape[0], self.num_kv_heads, self.head_dim)

        return q, k, v

    def _read_kv_from_cache(
        self, kv_cache: KVCache
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read complete K, V from paged KV cache (including history).

        Args:
            kv_cache: KV cache object containing paged cache data

        Returns:
            Tuple of (k_full, v_full) tensors
            - k_full: [batch_size, total_seq_len, num_kv_heads, head_dim]
            - v_full: [batch_size, total_seq_len, num_kv_heads, head_dim]
        """
        # Get batch size
        batch_size = self.attn_inputs.input_lengths.size(0)

        # Get real sequence lengths using FlashInferMlaAttnParams
        # This correctly computes kvlen from sequence_lengths + 1 in decode mode
        from rtp_llm.ops.compute_ops import fill_mla_params

        params = fill_mla_params(
            (
                self.attn_inputs.prefix_lengths
                if hasattr(self.attn_inputs, "prefix_lengths")
                else torch.tensor([], dtype=torch.int32)
            ),
            self.attn_inputs.sequence_lengths,
            self.attn_inputs.input_lengths,
            self.attn_inputs.kv_cache_block_id_host,
            self.tokens_per_block,
        )

        # kvlen contains the REAL sequence lengths (including current token in decode mode)
        sequence_lengths = params.kvlen_h[:batch_size]
        max_seq_len = sequence_lengths.max().item()

        # logging.info(
        #     f"[_read_kv_from_cache] batch_size={batch_size}, real_seq_lengths={sequence_lengths.tolist()}, max_seq_len={max_seq_len}"
        # )

        # Get KV cache tensor and reshape if needed
        # kv_cache_base may be 2D [num_blocks, kv_block_stride_elems] and needs reshaping to 5D
        kv_cache_base = kv_cache.kv_cache_base
        layer_id = kv_cache.layer_id

        # Reshape to 5D: [num_blocks, 2, num_kv_heads, tokens_per_block, head_dim]
        if kv_cache_base.ndim == 2:
            block_num = kv_cache_base.shape[0]
            expected_elems = (
                2 * self.num_kv_heads * self.tokens_per_block * self.head_dim
            )
            kv_cache_tensor = kv_cache_base[:, :expected_elems].reshape(
                block_num,
                2,
                self.num_kv_heads,
                self.tokens_per_block,
                self.head_dim,
            )
        else:
            kv_cache_tensor = kv_cache_base

        # Get block indices for each sequence
        # Shape: [batch_size, max_blocks_per_seq]
        block_indices = self.attn_inputs.kv_cache_block_id_host[:batch_size, :]

        # Prepare output tensors
        k_full = torch.zeros(
            batch_size,
            max_seq_len,
            self.num_kv_heads,
            self.head_dim,
            dtype=kv_cache_tensor.dtype,
            device=kv_cache_tensor.device,
        )
        v_full = torch.zeros(
            batch_size,
            max_seq_len,
            self.num_kv_heads,
            self.head_dim,
            dtype=kv_cache_tensor.dtype,
            device=kv_cache_tensor.device,
        )

        # Read K, V for each sequence
        for batch_idx in range(batch_size):
            seq_len = sequence_lengths[batch_idx].item()
            num_blocks = (seq_len + self.tokens_per_block - 1) // self.tokens_per_block

            # Collect K, V from blocks
            for block_idx in range(num_blocks):
                block_id = block_indices[batch_idx, block_idx].item()

                # Calculate token range for this block
                start_token = block_idx * self.tokens_per_block
                end_token = min(start_token + self.tokens_per_block, seq_len)
                block_token_count = end_token - start_token

                # Read K, V from cache
                # kv_cache_tensor shape: [num_blocks, 2, num_kv_heads, tokens_per_block, head_dim]
                k_block = kv_cache_tensor[
                    block_id, 0, :, :block_token_count, :
                ]  # [kv_heads, block_tokens, head_dim]
                v_block = kv_cache_tensor[
                    block_id, 1, :, :block_token_count, :
                ]  # [kv_heads, block_tokens, head_dim]

                # Store in output tensors
                # Transpose to [block_tokens, kv_heads, head_dim]
                k_full[batch_idx, start_token:end_token, :, :] = k_block.transpose(0, 1)
                v_full[batch_idx, start_token:end_token, :, :] = v_block.transpose(0, 1)

        return k_full, v_full

    def _run_attention_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Execute decode attention.

        In decode phase, each query is a single token attending to all previous tokens.

        Args:
            q: Query tensor [batch_size, num_heads, head_dim]
            k: Key tensor [batch_size, total_seq_len, num_kv_heads, head_dim]
            v: Value tensor [batch_size, total_seq_len, num_kv_heads, head_dim]

        Returns:
            Attention output [batch_size, num_heads, head_dim]
        """
        batch_size = q.shape[0]

        # Handle GQA: expand K, V heads to match Q heads
        if self.enable_gqa:
            # k, v: [batch_size, seq_len, num_kv_heads, head_dim]
            # Need to expand to [batch_size, seq_len, num_heads, head_dim]
            num_groups = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(num_groups, dim=2)
            v = v.repeat_interleave(num_groups, dim=2)

        # Reshape for SDPA
        # q: [batch_size, num_heads, head_dim] -> [batch_size, num_heads, 1, head_dim]
        # k, v: [batch_size, seq_len, num_heads, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
        q = q.unsqueeze(2)  # Add seq_len dimension
        k = k.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        v = v.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]

        # Handle dtype mismatch (SDPA requires same dtype)
        if not (q.dtype == k.dtype == v.dtype):
            k = k.to(q.dtype)
            v = v.to(q.dtype)

        # Execute scaled_dot_product_attention
        # No causal mask needed - single query attends to all past tokens
        output = scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,  # No causal mask for decode
            scale=self.scaling,
        )

        # output shape: [batch_size, num_heads, 1, head_dim]
        # Squeeze to remove seq_len dimension: [batch_size, num_heads, head_dim]
        output = output.squeeze(2)

        return output


# ============================================================================
# K-Clustering Enhanced Implementations
# ============================================================================


class TorchNaiveClusteredPrefillImpl(TorchNaivePrefillImpl):
    """带 K 聚类的 Prefill Attention 实现.

    继承自 TorchNaivePrefillImpl，在 Prefill 阶段对 K 进行聚类，
    存储质心、簇大小和 token 索引，供 Decode 阶段使用。
    """

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        """初始化带聚类的 Prefill 实现."""
        # 调用父类初始化
        super().__init__(attn_configs, attn_inputs, parallelism_config)

        # 聚类配置（从环境变量读取）
        import os

        self.cluster_ratio = int(os.getenv("CLUSTER_RATIO", "64"))
        self.kmeans_iters = int(os.getenv("KMEANS_ITERS", "20"))
        # 是否使用批量并行聚类（默认开启以提升性能）
        self.use_batched_clustering = os.getenv("USE_BATCHED_CLUSTERING", "1") == "1"

        logging.debug(
            f"TorchNaiveClusteredPrefillImpl initialized: "
            f"ratio={self.cluster_ratio}, kmeans_iters={self.kmeans_iters}, "
            f"batched={self.use_batched_clustering}"
        )

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        """支持检查：同父类."""
        return super(TorchNaiveClusteredPrefillImpl, cls).support(
            attn_configs, attn_inputs
        )

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
    ) -> torch.Tensor:
        """Forward pass: 先做聚类，再执行 attention."""

        # 1. Apply RoPE if needed
        if self.need_rope_kv_cache:
            qkv = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)

        # 2. Split QKV
        q, k, v = self._split_qkv(qkv)

        # 4. K Clustering - 在写入 cache 之前做聚类
        if self.use_batched_clustering:
            self._perform_k_clustering_batched(k, kv_cache)
        else:
            self._perform_k_clustering(k, kv_cache)

        # 5. Apply write cache store
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # 6. Execute attention (同父类，使用完整的 K)
        output = self._run_attention_extend(q, k, v)

        # 7. Reshape output
        output = output.reshape(output.shape[0], -1)

        return output

    def _perform_k_clustering(
        self,
        k: torch.Tensor,  # [total_tokens, num_kv_heads, head_dim]
        kv_cache: Optional[KVCache],
    ) -> None:
        """对 K 进行聚类并存储结果.

        Args:
            k: Key tensor
            kv_cache: KV cache object (用于获取 layer_id)
        """
        layer_id = kv_cache.layer_id if kv_cache is not None else 0
        batch_size = self.attn_inputs.input_lengths.size(0)
        cu_seqlens = self.attn_inputs.cu_seqlens[: batch_size + 1]

        # 按序列和 head 聚类
        for seq_idx in range(batch_size):
            start_idx = cu_seqlens[seq_idx].item()
            end_idx = cu_seqlens[seq_idx + 1].item()
            seq_len = end_idx - start_idx

            per_seq_k = k[start_idx:end_idx, :, :]  # [seq_len, num_kv_heads, head_dim]

            # 对每个 KV head 独立聚类
            for head_idx in range(per_seq_k.shape[1]):
                k_head = per_seq_k[:, head_idx, :]  # [seq_len, head_dim]

                # 计算簇数量
                num_clusters = max(1, seq_len // self.cluster_ratio)

                # 使用原有的批量聚类初始化质心
                centroids, labels, cluster_indices = _kmeans_clustering(
                    k_head, num_clusters, max_iters=self.kmeans_iters
                )
                # Ensure labels is 1-d and int64 for bincount
                labels = labels.reshape(-1).to(torch.int64)

                # Verify no negative values (sanity check)
                if labels.min() < 0:
                    raise RuntimeError(
                        f"Invalid labels: contains negative values. "
                        f"min={labels.min()}, max={labels.max()}, shape={labels.shape}"
                    )

                cluster_sizes = torch.bincount(labels, minlength=num_clusters)

                # 存储聚类信息到 _CLUSTER_CACHE
                key = f"layer_{layer_id}_seq_{seq_idx}_head_{head_idx}"
                _CLUSTER_CACHE[key] = {
                    "centroids": centroids,
                    "cluster_sizes": cluster_sizes,
                    "cluster_indices": cluster_indices,
                    "seq_len": seq_len,
                    "prefill_len": seq_len,  # 记录 Prefill 阶段的长度
                    # Local Window 状态（简化版：只存储计数，不存储 tensor）
                    "local_window": {
                        "start_idx": seq_len,  # Local Window 从 Prefill 结束位置开始
                        "count": 0,  # 当前 Local Window 中的 token 数
                        "window_size": int(os.getenv("LOCAL_WINDOW_SIZE", "4096")),
                    },
                }

                logging.debug(
                    f"K-Clustering: {key}, "
                    f"seq_len={seq_len}, num_clusters={num_clusters}"
                )

    def _perform_k_clustering_batched(
        self,
        k: torch.Tensor,  # [total_tokens, num_kv_heads, head_dim]
        kv_cache: Optional[KVCache],
    ) -> None:
        """对 K 进行聚类并存储结果 - 批量并行版本（优化性能）.

        此版本将所有 (seq, head) 对批量处理，避免嵌套循环，大幅提升性能。

        Args:
            k: Key tensor
            kv_cache: KV cache object (用于获取 layer_id)
        """
        layer_id = kv_cache.layer_id if kv_cache is not None else 0
        batch_size = self.attn_inputs.input_lengths.size(0)
        cu_seqlens = self.attn_inputs.cu_seqlens[: batch_size + 1]

        # Step 1: Collect all k_head tensors with metadata
        k_heads_list = []
        metadata_list = []  # (layer_id, seq_idx, head_idx, seq_len)

        for seq_idx in range(batch_size):
            start_idx = cu_seqlens[seq_idx].item()
            end_idx = cu_seqlens[seq_idx + 1].item()
            seq_len = end_idx - start_idx

            per_seq_k = k[start_idx:end_idx, :, :]  # [seq_len, num_kv_heads, head_dim]

            for head_idx in range(per_seq_k.shape[1]):
                k_head = per_seq_k[:, head_idx, :]  # [seq_len, head_dim]
                k_heads_list.append(k_head)
                metadata_list.append((layer_id, seq_idx, head_idx, seq_len))

        if not k_heads_list:
            return

        # Step 2: Handle variable sequence lengths with padding
        max_seq_len = max(k.shape[0] for k in k_heads_list)
        num_heads_total = len(k_heads_list)

        k_padded = torch.zeros(
            num_heads_total, max_seq_len, self.head_dim, device=k.device, dtype=k.dtype
        )

        # Also track actual sequence lengths for each head
        seq_lens = torch.zeros(num_heads_total, dtype=torch.int32, device=k.device)

        for i, k_head in enumerate(k_heads_list):
            seq_len = k_head.shape[0]
            k_padded[i, :seq_len] = k_head
            seq_lens[i] = seq_len

        # Step 3: Compute num_clusters for each head
        num_clusters_list = [
            max(1, metadata[3] // self.cluster_ratio) for metadata in metadata_list
        ]
        max_clusters = max(num_clusters_list)

        # Step 4: Single batched K-means call (MAJOR OPTIMIZATION!)
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.local_kmeans import (
            batch_kmeans_Euclid_optimized,
        )

        # K-means returns (labels, centroids, n_iters) - labels first!
        labels_batch, centroids_batch, n_iters = batch_kmeans_Euclid_optimized(
            k_padded,  # [num_heads_total, max_seq_len, head_dim]
            max_clusters,
            max_iters=self.kmeans_iters,
            tol=1e-4,
            init_centroids=None,
            verbose=False,
            use_kmeanspp=True,
        )

        # Step 5: Process results for each head
        for i, metadata in enumerate(metadata_list):
            layer_id, seq_idx, head_idx, seq_len = metadata
            num_clusters = num_clusters_list[i]

            # Extract results for this head (remove padding)
            centroids = centroids_batch[i, :num_clusters]  # [num_clusters, head_dim]
            labels = labels_batch[i, :seq_len]  # [seq_len]

            # Ensure labels is 1-d and int64 for bincount
            labels = labels.reshape(-1).to(torch.int64)

            # Verify no negative values (sanity check)
            if labels.min() < 0:
                raise RuntimeError(
                    f"Invalid labels: contains negative values. "
                    f"min={labels.min()}, max={labels.max()}, shape={labels.shape}"
                )

            # Build cluster_indices using GPU operations
            cluster_indices = _build_cluster_indices_gpu(labels, num_clusters)
            cluster_sizes = torch.bincount(labels, minlength=num_clusters)

            # Store results
            key = f"layer_{layer_id}_seq_{seq_idx}_head_{head_idx}"
            _CLUSTER_CACHE[key] = {
                "centroids": centroids,
                "cluster_sizes": cluster_sizes,
                "cluster_indices": cluster_indices,
                "seq_len": seq_len,
                "prefill_len": seq_len,  # 记录 Prefill 阶段的长度
                # Local Window 状态（简化版：只存储计数，不存储 tensor）
                "local_window": {
                    "start_idx": seq_len,  # Local Window 从 Prefill 结束位置开始
                    "count": 0,  # 当前 Local Window 中的 token 数
                    "window_size": int(os.getenv("LOCAL_WINDOW_SIZE", "4096")),
                },
            }

            logging.debug(
                f"K-Clustering (Batched): {key}, "
                f"seq_len={seq_len}, num_clusters={num_clusters}"
            )


class TorchNaiveClusteredDecodeImpl(TorchNaiveDecodeImpl):
    """带 K 聚类加速的 Decode Attention 实现.

    继承自 TorchNaiveDecodeImpl，使用 Prefill 阶段的聚类信息，
    通过 Q @ centroids + top_p 选择 + Full Attention 来加速计算。
    """

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        """初始化带聚类的 Decode 实现."""
        super().__init__(attn_configs, attn_inputs, parallelism_config)

        # 聚类配置
        import os

        self.top_p = float(os.getenv("CLUSTER_TOP_P", "0.9"))

        # Triton 优化配置
        self.use_triton_fusion = os.getenv("USE_TRITON_FUSION", "1") == "1"

        if self.use_triton_fusion and not FUSED_CLUSTER_GATHER_AVAILABLE:
            logging.warning(
                "USE_TRITON_FUSION=1 but fused kernel not available, falling back to PyTorch"
            )
            self.use_triton_fusion = False

        logging.debug(
            f"TorchNaiveClusteredDecodeImpl initialized: "
            f"top_p={self.top_p}, "
            f"use_triton_fusion={self.use_triton_fusion}"
        )

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        """支持检查：同父类."""
        return super(TorchNaiveClusteredDecodeImpl, cls).support(
            attn_configs, attn_inputs
        )

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
    ) -> torch.Tensor:
        """Forward pass: 使用聚类加速 attention."""
        # logging.info(
        #     f"[ClusteredDecode] forward: input qkv shape={qkv.shape}, need_rope={self.need_rope_kv_cache}"
        # )

        # 1. Apply RoPE if needed
        # NOTE: Decode RoPE may write K,V to cache directly and only return Q
        if self.need_rope_kv_cache:
            q = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)

        # 4. Apply write cache store
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # 5. Update clustering: 将新 K 分配到簇并更新质心
        k_full, v_full = self._read_kv_from_cache(kv_cache)
        k = k_full[:, -1:, :, :]  # Get only the last token (new K)
        k = k.squeeze(1)  # Remove seq dimension: [batch, kv_heads, head_dim]

        self._update_clustering(k, kv_cache)

        # 7. Execute clustered decode attention (NEW)
        output = self._run_clustered_attention_decode(q, k_full, v_full, kv_cache)

        # 8. Reshape output
        output = output.reshape(output.shape[0], -1)

        return output

    def _update_clustering(
        self,
        k_new: torch.Tensor,  # [batch_size, num_kv_heads, head_dim]
        kv_cache: Optional[KVCache],
    ) -> None:
        """更新 Local Window 并在需要时触发批量聚类合并（NEW 实现）.

        这个函数不再使用 IncrementalKMeans 进行增量更新，而是：
        1. 更新 Local Window 计数
        2. 检查 Local Window 是否满了
        3. 如果满了，触发批量聚类并合并到全局

        Args:
            k_new: 新生成的 Key tensor
            kv_cache: KV cache object
        """

        if kv_cache is None:
            return

        layer_id = kv_cache.layer_id
        batch_size = k_new.shape[0]

        for batch_idx in range(batch_size):
            for head_idx in range(k_new.shape[1]):
                key = f"layer_{layer_id}_seq_{batch_idx}_head_{head_idx}"

                if key not in _CLUSTER_CACHE:
                    # 没有聚类信息，跳过（可能是新序列）
                    logging.warning(f"No cluster info found for {key}, skipping")
                    continue

                cluster_info = _CLUSTER_CACHE[key]

                # NEW: 更新 Local Window 计数
                need_merge = _update_local_window(cluster_info)

                # 检查是否需要触发批量聚类合并
                if need_merge:
                    logging.info(f"Local Window full for {key}, triggering merge")
                    _merge_local_window_to_global(
                        cluster_info, kv_cache, self, layer_id, head_idx
                    )

                logging.debug(
                    f"Update clustering (Local Window): {key}, "
                    f"seq_len={cluster_info['seq_len']}, "
                    f"local_window_count={cluster_info['local_window']['count']}"
                )

    def _full_attention_single(
        self,
        q: torch.Tensor,  # [head_dim]
        k: torch.Tensor,  # [seq_len, head_dim]
        v: torch.Tensor,  # [seq_len, head_dim]
    ) -> torch.Tensor:
        """单个 query 的 Full Attention.

        Returns:
            output: [head_dim]
        """
        # Reshape for SDPA
        q = q.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, head_dim]
        k = k.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
        v = v.unsqueeze(0).unsqueeze(0)

        # Handle dtype mismatch
        if not (q.dtype == k.dtype == v.dtype):
            k = k.to(q.dtype)
            v = v.to(q.dtype)

        output = scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scaling,
        ).squeeze()

        return output

    def _full_attention_gqa_group(
        self,
        q_group: torch.Tensor,  # [num_q_heads, head_dim]
        k: torch.Tensor,  # [num_selected_tokens, head_dim]
        v: torch.Tensor,  # [num_selected_tokens, head_dim]
    ) -> torch.Tensor:
        """批量执行 GQA group 的 attention，多个 Q heads 共享同一组 K, V.

        用于 GQA 场景：同一个 KV head 对应的多个 Q heads 共享相同的 K, V。

        Args:
            q_group: [num_q_heads, head_dim] - 该 KV head 对应的所有 Q heads
            k: [num_selected_tokens, head_dim] - 共享的 K
            v: [num_selected_tokens, head_dim] - 共享的 V

        Returns:
            output: [num_q_heads, head_dim]
        """
        # Reshape for SDPA
        # q: [1, num_q_heads, 1, head_dim]
        # k, v: [1, 1, num_selected_tokens, head_dim] - 会被广播到每个 Q head
        q = q_group.unsqueeze(0).unsqueeze(2)  # [1, num_q_heads, 1, head_dim]
        k = k.unsqueeze(0).unsqueeze(0)  # [1, 1, num_selected_tokens, head_dim]
        v = v.unsqueeze(0).unsqueeze(0)

        # Handle dtype mismatch
        if not (q.dtype == k.dtype == v.dtype):
            k = k.to(q.dtype)
            v = v.to(q.dtype)

        # Execute attention - k, v 会被广播到每个 Q head
        output = scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scaling,
        )

        # output: [1, num_q_heads, 1, head_dim] -> [num_q_heads, head_dim]
        # 使用指定维度的 squeeze 避免当 num_q_heads=1 时出错
        return output.squeeze(0).squeeze(1)  # 先去掉 batch dim，再去掉 query_len dim

    def _full_attention_batch(
        self,
        q: torch.Tensor,  # [num_heads, head_dim]
        k: torch.Tensor,  # [num_heads, num_selected, head_dim]
        v: torch.Tensor,  # [num_heads, num_selected, head_dim]
        attn_mask: torch.Tensor,  # [num_heads, num_selected], bool
    ) -> torch.Tensor:
        """批量执行多个 head 的 attention，使用 mask 控制每个 head 关注的 tokens.

        Args:
            q: [num_heads, head_dim]
            k: [num_heads, num_selected, head_dim]
            v: [num_heads, num_selected, head_dim]
            attn_mask: [num_heads, num_selected], True 表示该 head 可以 attend 该 token

        Returns:
            output: [num_heads, head_dim]
        """
        # Reshape for SDPA: add batch and query_len dimensions
        # q: [1, num_heads, 1, head_dim]
        # k, v: [1, num_heads, num_selected, head_dim]
        q = q.unsqueeze(0).unsqueeze(2)  # [1, num_heads, 1, head_dim]
        k = k.unsqueeze(0)  # [1, num_heads, num_selected, head_dim]
        v = v.unsqueeze(0)

        # attn_mask: [num_heads, num_selected] -> [1, num_heads, 1, num_selected]
        # PyTorch SDPA 要求 mask 为 bool 或 float
        # True/False -> 转为 float，True=0.0（可见），False=-inf（屏蔽）
        mask = torch.zeros_like(attn_mask, dtype=q.dtype)
        mask[~attn_mask] = float("-inf")  # 屏蔽未选中的 token
        mask = mask.unsqueeze(0).unsqueeze(2)  # [1, num_heads, 1, num_selected]

        # Handle dtype mismatch
        if not (q.dtype == k.dtype == v.dtype):
            k = k.to(q.dtype)
            v = v.to(q.dtype)

        # Execute attention
        output = scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scaling,
        )

        # output: [1, num_heads, 1, head_dim] -> [num_heads, head_dim]
        return output.squeeze(0).squeeze(1)

    def _run_clustered_attention_decode(
        self,
        q: torch.Tensor,  # [batch_size, num_heads, head_dim]
        k_full: torch.Tensor,  # [batch_size, total_seq_len, num_kv_heads, head_dim]
        v_full: torch.Tensor,  # [batch_size, total_seq_len, num_kv_heads, head_dim]
        kv_cache: Optional[KVCache],
    ) -> torch.Tensor:
        """使用聚类加速的 Decode Attention (向量化优化版本).

        优化策略: Union-based Batching + Vectorized Operations
        1. 使用融合的质心计算 + top-p 选择内核
        2. 使用 CSR 格式加速 token 收集
        3. 向量化构建 attention mask
        4. 批量执行所有 head 的 attention

        Returns:
            output: [batch_size, num_heads, head_dim]
        """
        batch_size = q.shape[0]
        num_heads = q.shape[1]
        layer_id = kv_cache.layer_id if kv_cache is not None else 0
        total_tokens = k_full.shape[1]

        # GQA handling
        if self.enable_gqa:
            num_groups = self.num_heads // self.num_kv_heads
            # k_full = k_full.repeat_interleave(num_groups, dim=2)
            # v_full = v_full.repeat_interleave(num_groups, dim=2)
        else:
            num_groups = 1

        output = torch.empty_like(q)

        # 按 KV head 分组批量计算
        num_kv_heads_to_process = self.num_kv_heads if self.enable_gqa else num_heads

        # 外层循环：只按 batch
        for batch_idx in range(batch_size):
            total_tokens_selected = 0  # 用于统计
            total_q_heads_processed = 0

            # 按 KV head 循环，每个 KV head 独立处理
            for kv_head_idx in range(num_kv_heads_to_process):
                # Step 1: 获取 cluster_info
                key = f"layer_{layer_id}_seq_{batch_idx}_head_{kv_head_idx}"

                if key not in _CLUSTER_CACHE:
                    logging.error(f"Missing cluster info for {key}")
                    raise RuntimeError(f"Cluster info not found: {key}")

                cluster_info = _CLUSTER_CACHE[key]

                # Precompute CSR if not already done
                if "flat_indices" not in cluster_info:
                    cluster_info = precompute_csr_cache(cluster_info)
                    _CLUSTER_CACHE[key] = cluster_info

                # 获取质心和 CSR 数据
                if "model" in cluster_info:
                    centroids = cluster_info["model"].get_centroids()
                else:
                    centroids = cluster_info["centroids"]

                cluster_sizes = cluster_info["cluster_sizes"]
                flat_indices = cluster_info["flat_indices"]
                offsets = cluster_info["offsets"]
                local_window = cluster_info["local_window"]

                # Step 2: 获取该 KV head 对应的 Q heads
                if self.enable_gqa:
                    start_head_idx = kv_head_idx * num_groups
                    end_head_idx = start_head_idx + num_groups
                    q_group = q[
                        batch_idx, start_head_idx:end_head_idx, :
                    ]  # [num_groups, head_dim]
                else:
                    # 非 GQA：每个 head 独立
                    start_head_idx = kv_head_idx
                    end_head_idx = kv_head_idx + 1
                    q_group = q[
                        batch_idx, kv_head_idx : kv_head_idx + 1, :
                    ]  # [1, head_dim]

                # Step 3: 使用融合内核选择 clusters
                num_clusters = centroids.shape[0]
                selected_cluster_ids_batch, num_selected_batch, _ = (
                    fused_centroid_scoring_topp_vectorized(
                        q_group,  # [num_groups, head_dim]
                        centroids,  # [num_clusters, head_dim]
                        cluster_sizes,  # [num_clusters]
                        top_p=self.top_p,
                        max_selected=num_clusters,
                        scaling=self.scaling,
                    )
                )
                # selected_cluster_ids_batch: [num_groups, num_clusters] (padded with -1)
                # num_selected_batch: [num_groups]

                # Steps 4-7: 使用 Triton 融合 kernel 或 PyTorch fallback
                if self.use_triton_fusion and FUSED_CLUSTER_GATHER_AVAILABLE:
                    # 使用 Triton 融合 kernel：cluster union + token gather + KV extraction
                    try:
                        kv_k, kv_v, selected_tokens, num_selected_tokens = (
                            fused_cluster_union_gather_kv(
                                selected_cluster_ids_batch=selected_cluster_ids_batch,
                                num_selected_batch=num_selected_batch,
                                flat_indices=flat_indices,
                                offsets=offsets,
                                local_window_start=local_window["start_idx"],
                                local_window_count=local_window["count"],
                                k_cache=k_full[batch_idx, :, kv_head_idx, :],
                                v_cache=v_full[batch_idx, :, kv_head_idx, :],
                                use_triton=True,
                            )
                        )
                        total_tokens_selected += num_selected_tokens

                    except Exception as e:
                        logging.warning(
                            f"Triton fused kernel failed: {e}, falling back to PyTorch"
                        )
                        self.use_triton_fusion = False  # Disable for future iterations
                        # Fall through to PyTorch implementation below

                if not (self.use_triton_fusion and FUSED_CLUSTER_GATHER_AVAILABLE):
                    # PyTorch fallback implementation
                    # Step 4: 收集该 KV head 对应的所有 Q heads 选中的 clusters 的并集
                    kv_head_all_clusters = []
                    for head_offset in range(q_group.shape[0]):
                        n_selected = num_selected_batch[head_offset].item()
                        selected_cluster_ids = selected_cluster_ids_batch[
                            head_offset, :n_selected
                        ]
                        if n_selected > 0:
                            kv_head_all_clusters.append(selected_cluster_ids)

                    # 计算该 KV head 的 cluster 并集
                    if len(kv_head_all_clusters) > 0:
                        kv_head_union_clusters = torch.unique(
                            torch.cat(kv_head_all_clusters)
                        )
                    else:
                        kv_head_union_clusters = torch.tensor(
                            [], dtype=torch.int32, device=q.device
                        )

                    # Step 5: 从选中的 clusters 收集 tokens
                    if kv_head_union_clusters.numel() > 0:
                        selected_tokens = gather_tokens_from_clusters_csr(
                            kv_head_union_clusters, flat_indices, offsets
                        )
                    else:
                        selected_tokens = torch.tensor(
                            [], dtype=torch.int32, device=q.device
                        )

                    # Step 6: 添加 Local Window tokens
                    # 注意: local_window tokens 和 cluster tokens 不会重合，所以直接 cat 不需要 unique
                    if local_window["count"] > 0:
                        local_window_start = local_window["start_idx"]
                        local_window_end = local_window_start + local_window["count"]

                        # 创建 Local Window tensor 并合并
                        local_window_tensor = torch.arange(
                            local_window_start,
                            local_window_end,
                            dtype=selected_tokens.dtype,
                            device=selected_tokens.device,
                        )
                        # 直接 cat，不需要 unique（local window 和 clusters 不会重合）
                        selected_tokens = torch.cat(
                            [selected_tokens, local_window_tensor]
                        )

                    num_selected_tokens = selected_tokens.shape[0]
                    total_tokens_selected += num_selected_tokens

                    # Step 7: 提取该 KV head 的 K, V（只提取 selected tokens）
                    # k_full: [batch_size, total_seq_len, num_heads, head_dim]
                    # -> kv_k: [num_selected_tokens, head_dim]
                    kv_k = k_full[
                        batch_idx, selected_tokens, kv_head_idx, :
                    ]  # [num_selected_tokens, head_dim]
                    kv_v = v_full[
                        batch_idx, selected_tokens, kv_head_idx, :
                    ]  # [num_selected_tokens, head_dim]

                # Check if we have any tokens
                if num_selected_tokens == 0:
                    # 没有选中任何 token，输出 0
                    output[batch_idx, start_head_idx:end_head_idx, :] = 0.0
                    logging.warning(
                        f"No tokens selected for kv_head={kv_head_idx}, batch={batch_idx}, using zero output"
                    )
                    continue

                # Step 8: 批量执行该 Q group 的 attention
                # q_group 中的所有 Q heads 共享同一组 K, V
                output[batch_idx, start_head_idx:end_head_idx, :] = (
                    self._full_attention_gqa_group(
                        q_group,  # [num_groups, head_dim]
                        kv_k,  # [num_selected_tokens, head_dim]
                        kv_v,  # [num_selected_tokens, head_dim]
                    )
                )
                total_q_heads_processed += q_group.shape[0]

            # 统计信息
            avg_tokens_per_head = (
                total_tokens_selected / total_q_heads_processed
                if total_q_heads_processed > 0
                else 0
            )

            logging.info(
                f"[Clustering Optimized] layer={layer_id}, batch={batch_idx}: "
                f"avg_tokens_per_head={avg_tokens_per_head:.1f}/{total_tokens} ({avg_tokens_per_head/total_tokens*100:.1f}%), "
                f"num_kv_heads={num_kv_heads_to_process}, "
                f"num_q_heads={total_q_heads_processed}"
            )

        return output


# ============================================================================
# Clustered Residual FP4 Quantization Implementations
# ============================================================================


class TorchNaiveResidualFP4PrefillImpl(TorchNaivePrefillImpl):
    """Prefill with clustered residual FP4 quantization on K/V.

    继承 TorchNaivePrefillImpl，在 attention 前对 K/V 做聚类残差 FP4 伪量化，
    并将聚类结果存入 _CLUSTER_CACHE 供 decode 复用。
    """

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
    ) -> torch.Tensor:
        # 1. RoPE
        if self.need_rope_kv_cache:
            qkv = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)

        # 2. Split QKV
        q, k, v = self._split_qkv(qkv)

        # 3. Write cache store
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # 4. 聚类残差 FP4 伪量化
        cluster_ratio = int(os.getenv("CLUSTER_RATIO", "64"))
        kmeans_iters = int(os.getenv("KMEANS_ITERS", "20"))
        seq_len = k.shape[0]

        k_res, k_centroids, k_labels = _clustered_residual_fp4_quant(
            k.squeeze(1), cluster_ratio, max_iters=kmeans_iters
        )
        k = k_res.unsqueeze(1)

        v_res, v_centroids, v_labels = _clustered_residual_fp4_quant(
            v.squeeze(1), cluster_ratio, max_iters=kmeans_iters
        )
        v = v_res.unsqueeze(1)

        # 5. 存聚类结果供 decode 复用
        layer_id = kv_cache.layer_id if kv_cache is not None else 0
        _CLUSTER_CACHE[f"residual_k_{layer_id}"] = {
            "centroids": k_centroids,
            "labels": k_labels,
        }
        _CLUSTER_CACHE[f"residual_v_{layer_id}"] = {
            "centroids": v_centroids,
            "labels": v_labels,
        }

        logging.info(
            f"[ResidualFP4 Prefill] seq_len={seq_len}, "
            f"num_clusters={k_centroids.shape[0]}"
        )

        # 6. Attention
        output = self._run_attention_extend(q, k, v)
        output = output.reshape(output.shape[0], -1)
        return output


class TorchNaiveResidualFP4DecodeImpl(TorchNaiveDecodeImpl):
    """Decode with clustered residual FP4 quantization on K/V.

    继承 TorchNaiveDecodeImpl，复用 prefill 的聚类结果：
    - 历史 token: centroid[label] + fp4_quant(diff)
    - 新增 token: FP8 伪量化
    """

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
    ) -> torch.Tensor:
        # 1. RoPE
        if self.need_rope_kv_cache:
            q = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
            if q.ndim == 2:
                q = q.reshape(q.shape[0], self.num_heads, self.head_dim)
        else:
            q, k, v = self._split_qkv(qkv)

        # 2. Write cache store
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # 3. Read K/V from cache
        k_full, v_full = self._read_kv_from_cache(kv_cache)

        # 4. 复用 prefill 聚类结果做残差 FP4 + 新 token FP8
        k_full, v_full = _apply_residual_fp4_decode(k_full, v_full, kv_cache)

        # 5. Attention
        output = self._run_attention_decode(q, k_full, v_full)
        output = output.reshape(output.shape[0], -1)
        return output
