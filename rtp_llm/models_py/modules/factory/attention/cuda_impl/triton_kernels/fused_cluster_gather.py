"""Fused Triton kernel for cluster union, token gathering, and KV extraction.

Phase 1 Implementation (Practical Approach):
1. Fused CSR token gathering kernel (replaces Python loop)
2. Optimized KV gathering kernel (coalesced memory access)
3. Use PyTorch for cluster union and unique (fast enough on small tensors)

This gives us most of the performance benefit with lower implementation complexity.
"""

import logging
from typing import Tuple

import torch

# Check if Triton is available
try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    logging.warning("Triton not available, will use PyTorch fallback")


if TRITON_AVAILABLE:

    @triton.jit
    def _gather_tokens_from_csr_kernel(
        # Inputs
        selected_cluster_ids_ptr,  # [num_selected_clusters]
        flat_indices_ptr,  # [total_tokens]
        offsets_ptr,  # [num_clusters + 1]
        # Outputs
        output_tokens_ptr,  # [max_output_tokens]
        output_offsets_ptr,  # [num_selected_clusters + 1] - where each cluster's tokens start
        # Dimensions
        num_selected_clusters,
        max_output_tokens,
        # Block size
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Gather tokens from selected clusters in parallel.

        Each thread block processes one cluster.
        """
        pid = tl.program_id(0)

        if pid >= num_selected_clusters:
            return

        # Get the cluster ID for this block
        cluster_id = tl.load(selected_cluster_ids_ptr + pid)

        # Get CSR range for this cluster
        start_idx = tl.load(offsets_ptr + cluster_id)
        end_idx = tl.load(offsets_ptr + cluster_id + 1)
        cluster_size = end_idx - start_idx

        # Get output offset for this cluster
        output_offset = tl.load(output_offsets_ptr + pid)

        # Copy tokens in blocks
        for i in range(0, cluster_size, BLOCK_SIZE):
            offset = i + tl.arange(0, BLOCK_SIZE)
            mask = offset < cluster_size

            # Load tokens from flat_indices
            token_ids = tl.load(
                flat_indices_ptr + start_idx + offset, mask=mask, other=0
            )

            # Store to output
            tl.store(output_tokens_ptr + output_offset + offset, token_ids, mask=mask)

    @triton.jit
    def _gather_kv_kernel(
        # Inputs
        token_ids_ptr,  # [num_tokens] - token IDs to gather
        k_cache_ptr,  # [seq_len, head_dim]
        v_cache_ptr,  # [seq_len, head_dim]
        # Outputs
        output_k_ptr,  # [num_tokens, head_dim]
        output_v_ptr,  # [num_tokens, head_dim]
        # Dimensions
        num_tokens,
        head_dim,
        seq_len,
        # Strides
        k_stride_seq,
        k_stride_dim,
        v_stride_seq,
        v_stride_dim,
        out_stride_token,
        out_stride_dim,
        # Block sizes
        BLOCK_SIZE_TOKEN: tl.constexpr,
        BLOCK_SIZE_DIM: tl.constexpr,
    ):
        """
        Gather K, V from cache for selected tokens with coalesced memory access.

        Grid: (num_tokens, triton.cdiv(head_dim, BLOCK_SIZE_DIM))
        Each program processes one token and a block of dimensions.
        """
        pid_token = tl.program_id(0)
        pid_dim = tl.program_id(1)

        # Check if this token is valid
        if pid_token >= num_tokens:
            return

        # Calculate dimension range for this block
        dim_start = pid_dim * BLOCK_SIZE_DIM
        dim_offsets = dim_start + tl.arange(0, BLOCK_SIZE_DIM)
        dim_mask = dim_offsets < head_dim

        # Load token ID for this program
        token_id = tl.load(token_ids_ptr + pid_token)

        # Calculate source addresses for K, V cache
        k_src_base = token_id * k_stride_seq + dim_offsets * k_stride_dim
        v_src_base = token_id * v_stride_seq + dim_offsets * v_stride_dim

        # Calculate destination addresses
        out_base = pid_token * out_stride_token + dim_offsets * out_stride_dim

        # Load from K, V cache
        k_vals = tl.load(k_cache_ptr + k_src_base, mask=dim_mask, other=0.0)
        v_vals = tl.load(v_cache_ptr + v_src_base, mask=dim_mask, other=0.0)

        # Store to output
        tl.store(output_k_ptr + out_base, k_vals, mask=dim_mask)
        tl.store(output_v_ptr + out_base, v_vals, mask=dim_mask)


def fused_cluster_union_gather_kv(
    selected_cluster_ids_batch: torch.Tensor,  # [num_q_heads, max_clusters]
    num_selected_batch: torch.Tensor,  # [num_q_heads]
    flat_indices: torch.Tensor,  # [total_tokens]
    offsets: torch.Tensor,  # [num_clusters + 1]
    local_window_start: int,
    local_window_count: int,
    k_cache: torch.Tensor,  # [seq_len, head_dim]
    v_cache: torch.Tensor,  # [seq_len, head_dim]
    use_triton: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Fused operation: cluster union + token gather + KV extraction.

    Phase 1 implementation uses Triton for:
    - CSR token gathering (parallel)
    - KV gathering (coalesced access)

    Uses PyTorch for:
    - Cluster union (small tensor, fast enough)
    - Unique operation (well-optimized in PyTorch)

    Returns:
        output_k: [num_selected_tokens, head_dim]
        output_v: [num_selected_tokens, head_dim]
        token_ids: [num_selected_tokens]
        num_selected_tokens: int
    """

    if not TRITON_AVAILABLE or not use_triton:
        return fused_cluster_union_gather_kv_pytorch(
            selected_cluster_ids_batch,
            num_selected_batch,
            flat_indices,
            offsets,
            local_window_start,
            local_window_count,
            k_cache,
            v_cache,
        )

    device = selected_cluster_ids_batch.device
    num_q_heads = selected_cluster_ids_batch.shape[0]
    seq_len, head_dim = k_cache.shape

    # Step 1: Compute cluster union (PyTorch - fast enough for small tensors)
    all_clusters = []
    for head_idx in range(num_q_heads):
        n_selected = num_selected_batch[head_idx].item()
        if n_selected > 0:
            clusters = selected_cluster_ids_batch[head_idx, :n_selected]
            all_clusters.append(clusters)

    if len(all_clusters) == 0:
        # No clusters selected
        empty = torch.tensor([], dtype=torch.int32, device=device)
        empty_kv = torch.empty(0, head_dim, dtype=k_cache.dtype, device=device)
        return empty_kv, empty_kv, empty, 0

    union_clusters = torch.unique(torch.cat(all_clusters))
    num_union_clusters = union_clusters.shape[0]

    # Step 2: Compute output offsets for each cluster (prefix sum of cluster sizes)
    cluster_sizes = offsets[union_clusters + 1] - offsets[union_clusters]
    output_offsets = torch.cat(
        [
            torch.tensor([0], dtype=torch.int32, device=device),
            torch.cumsum(cluster_sizes, dim=0),
        ]
    )
    total_tokens = output_offsets[-1].item()

    # Step 3: Gather tokens from CSR using Triton kernel
    token_buffer = torch.empty(total_tokens, dtype=torch.int32, device=device)

    BLOCK_SIZE = 128
    grid = (num_union_clusters,)

    _gather_tokens_from_csr_kernel[grid](
        union_clusters,
        flat_indices,
        offsets,
        token_buffer,
        output_offsets,
        num_union_clusters,
        total_tokens,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Step 4: Add local window tokens (direct cat, no unique needed)
    if local_window_count > 0:
        local_tokens = torch.arange(
            local_window_start,
            local_window_start + local_window_count,
            dtype=torch.int32,
            device=device,
        )
        token_buffer = torch.cat([token_buffer, local_tokens])

    # Step 5: Unique (PyTorch - well-optimized)
    unique_tokens = torch.unique(token_buffer)
    num_unique = unique_tokens.shape[0]

    if num_unique == 0:
        empty_kv = torch.empty(0, head_dim, dtype=k_cache.dtype, device=device)
        return empty_kv, empty_kv, unique_tokens, 0

    # Step 6: Gather K, V using Triton kernel
    output_k = torch.empty(num_unique, head_dim, dtype=k_cache.dtype, device=device)
    output_v = torch.empty(num_unique, head_dim, dtype=v_cache.dtype, device=device)

    BLOCK_SIZE_DIM = 64

    # Grid: one program per token, multiple programs per dimension
    grid = (
        num_unique,
        triton.cdiv(head_dim, BLOCK_SIZE_DIM),
    )

    _gather_kv_kernel[grid](
        unique_tokens,
        k_cache,
        v_cache,
        output_k,
        output_v,
        num_unique,
        head_dim,
        seq_len,
        k_cache.stride(0),
        k_cache.stride(1),
        v_cache.stride(0),
        v_cache.stride(1),
        output_k.stride(0),
        output_k.stride(1),
        BLOCK_SIZE_TOKEN=1,  # Not used anymore, but kept for interface compatibility
        BLOCK_SIZE_DIM=BLOCK_SIZE_DIM,
    )

    return output_k, output_v, unique_tokens, num_unique


# ============================================================================
# Fallback PyTorch implementation (for comparison/debugging)
# ============================================================================


def fused_cluster_union_gather_kv_pytorch(
    selected_cluster_ids_batch: torch.Tensor,
    num_selected_batch: torch.Tensor,
    flat_indices: torch.Tensor,
    offsets: torch.Tensor,
    local_window_start: int,
    local_window_count: int,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """PyTorch reference implementation (matches current torch_naive.py logic)."""

    device = k_cache.device
    num_q_heads = selected_cluster_ids_batch.shape[0]
    head_dim = k_cache.shape[1]

    # Step 1: Compute cluster union
    all_clusters = []
    for head_idx in range(num_q_heads):
        n_selected = num_selected_batch[head_idx].item()
        if n_selected > 0:
            clusters = selected_cluster_ids_batch[head_idx, :n_selected]
            all_clusters.append(clusters)

    if len(all_clusters) == 0:
        empty = torch.tensor([], dtype=torch.int32, device=device)
        empty_kv = torch.empty(0, head_dim, dtype=k_cache.dtype, device=device)
        return empty_kv, empty_kv, empty, 0

    union_clusters = torch.unique(torch.cat(all_clusters))

    # Step 2: Gather tokens from clusters (CSR)
    all_tokens = []
    for cluster_id in union_clusters:
        start = offsets[cluster_id].item()
        end = offsets[cluster_id + 1].item()
        cluster_tokens = flat_indices[start:end]
        all_tokens.append(cluster_tokens)

    if len(all_tokens) > 0:
        selected_tokens = torch.cat(all_tokens)
    else:
        selected_tokens = torch.tensor([], dtype=torch.int32, device=device)

    # Step 3: Add local window (direct cat, no unique needed)
    if local_window_count > 0:
        local_tokens = torch.arange(
            local_window_start,
            local_window_start + local_window_count,
            dtype=torch.int32,
            device=device,
        )
        selected_tokens = torch.cat([selected_tokens, local_tokens])

    # Step 4: Unique
    unique_tokens = torch.unique(selected_tokens)
    num_tokens = unique_tokens.shape[0]

    if num_tokens == 0:
        empty_kv = torch.empty(0, head_dim, dtype=k_cache.dtype, device=device)
        return empty_kv, empty_kv, unique_tokens, 0

    # Step 5: Gather K, V
    output_k = k_cache[unique_tokens]
    output_v = v_cache[unique_tokens]

    return output_k, output_v, unique_tokens, num_tokens
