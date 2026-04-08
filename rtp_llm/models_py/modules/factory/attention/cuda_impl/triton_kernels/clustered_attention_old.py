"""Triton kernel for clustered decode attention optimization.

This module provides a fused kernel for centroid scoring and top-p selection
in the clustered attention mechanism.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def fused_centroid_scoring_topp_kernel(
    Q_ptr,  # [num_heads, head_dim]
    centroids_ptr,  # [num_clusters, head_dim]
    cluster_sizes_ptr,  # [num_clusters]
    output_ids_ptr,  # [num_heads, max_selected]
    output_scores_ptr,  # [num_heads, max_selected]
    num_selected_ptr,  # [num_heads]
    num_heads: tl.constexpr,
    num_clusters: tl.constexpr,
    head_dim: tl.constexpr,
    max_selected: tl.constexpr,
    top_p,  # float
    scaling,  # float
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for centroid scoring + top-p selection.

    Each program handles one query head:
    1. Compute Q @ centroids.T for all clusters
    2. Add log(cluster_sizes) weighting
    3. Apply softmax
    4. Perform top-p (nucleus) selection using iterative max-finding

    Args:
        Q_ptr: Query vectors [num_heads, head_dim]
        centroids_ptr: Cluster centroids [num_clusters, head_dim]
        cluster_sizes_ptr: Tokens per cluster [num_clusters]
        output_ids_ptr: Selected cluster IDs [num_heads, max_selected]
        output_scores_ptr: Scores for selected clusters [num_heads, max_selected]
        num_selected_ptr: Actual number of selected clusters [num_heads]
        num_heads: Number of query heads
        num_clusters: Number of clusters
        head_dim: Dimension of each head
        max_selected: Maximum clusters to select (buffer size)
        top_p: Cumulative probability threshold (e.g., 0.9)
        scaling: Attention scale factor (1/sqrt(head_dim))
        BLOCK_SIZE: Block size for vectorized loads (must be >= head_dim)
    """
    # Each program handles one head
    pid = tl.program_id(0)

    if pid >= num_heads:
        return

    # Load Q vector for this head
    q_offset = pid * head_dim
    q_mask = tl.arange(0, BLOCK_SIZE) < head_dim
    q_block = tl.load(
        Q_ptr + q_offset + tl.arange(0, BLOCK_SIZE), mask=q_mask, other=0.0
    )

    # Allocate scores array in registers/local memory
    # We'll compute one cluster at a time to save memory
    scores = tl.zeros([num_clusters], dtype=tl.float32)

    # Compute scores for all clusters
    for c_id in range(num_clusters):
        # Load centroid vector
        c_offset = c_id * head_dim
        centroid = tl.load(
            centroids_ptr + c_offset + tl.arange(0, BLOCK_SIZE), mask=q_mask, other=0.0
        )

        # Dot product: Q @ centroid
        dot = tl.sum(q_block * centroid) * scaling

        # Add log cluster size
        cluster_size = tl.load(cluster_sizes_ptr + c_id)
        log_size = tl.log(cluster_size.to(tl.float32) + 1e-8)

        scores = tl.where(tl.arange(0, num_clusters) == c_id, dot + log_size, scores)

    # Softmax
    max_score = tl.max(scores, axis=0)
    scores_exp = tl.exp(scores - max_score)
    sum_exp = tl.sum(scores_exp, axis=0)
    scores_softmax = scores_exp / sum_exp

    # Top-p selection using iterative max-finding
    # Make a copy for modification
    temp_scores = scores_softmax
    cumulative_prob = 0.0
    selected_count = 0

    # Iteratively find max, accumulate probability
    for k in range(max_selected):
        if k >= num_clusters:
            break

        # Find cluster with maximum score
        max_prob = tl.max(temp_scores, axis=0)

        # Find the index of max (simple linear search)
        max_idx = 0
        for i in range(num_clusters):
            if temp_scores[i] == max_prob:
                max_idx = i
                break

        # Add to cumulative probability
        cumulative_prob += max_prob

        # Store result
        out_offset = pid * max_selected + selected_count
        tl.store(output_ids_ptr + out_offset, max_idx)
        tl.store(output_scores_ptr + out_offset, max_prob)

        selected_count += 1

        # Check if we've exceeded top_p threshold
        if cumulative_prob > top_p:
            break

        # Mask out selected cluster for next iteration
        temp_scores = tl.where(tl.arange(0, num_clusters) == max_idx, -1e9, temp_scores)

    # Store actual number of selected clusters
    tl.store(num_selected_ptr + pid, selected_count)

    # Pad remaining slots with -1
    for k in range(selected_count, max_selected):
        out_offset = pid * max_selected + k
        tl.store(output_ids_ptr + out_offset, -1)
        tl.store(output_scores_ptr + out_offset, 0.0)


def fused_centroid_scoring_topp(
    q: torch.Tensor,  # [num_heads, head_dim]
    centroids: torch.Tensor,  # [num_clusters, head_dim]
    cluster_sizes: torch.Tensor,  # [num_clusters]
    top_p: float = 0.9,
    max_selected: int = None,
    scaling: float = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Python wrapper for fused centroid scoring + top-p selection.

    Args:
        q: Query vectors [num_heads, head_dim]
        centroids: Cluster centroids [num_clusters, head_dim]
        cluster_sizes: Number of tokens in each cluster [num_clusters]
        top_p: Cumulative probability threshold (default: 0.9)
        max_selected: Maximum number of clusters to select (default: num_clusters)
        scaling: Attention scale factor (default: 1/sqrt(head_dim))

    Returns:
        selected_cluster_ids: [num_heads, max_selected] - Selected cluster indices (padded with -1)
        num_selected: [num_heads] - Actual number of selected clusters per head
        cluster_scores: [num_heads, max_selected] - Scores for selected clusters
    """
    # Input validation
    assert q.is_cuda, "Q must be on CUDA device"
    assert centroids.is_cuda, "Centroids must be on CUDA device"
    assert cluster_sizes.is_cuda, "Cluster sizes must be on CUDA device"
    assert q.dim() == 2, f"Q must be 2D [num_heads, head_dim], got {q.shape}"
    assert (
        centroids.dim() == 2
    ), f"Centroids must be 2D [num_clusters, head_dim], got {centroids.shape}"
    assert (
        cluster_sizes.dim() == 1
    ), f"Cluster sizes must be 1D [num_clusters], got {cluster_sizes.shape}"

    num_heads, head_dim = q.shape
    num_clusters = centroids.shape[0]

    assert (
        centroids.shape[1] == head_dim
    ), f"Centroid head_dim {centroids.shape[1]} must match Q head_dim {head_dim}"
    assert (
        cluster_sizes.shape[0] == num_clusters
    ), f"Cluster sizes length {cluster_sizes.shape[0]} must match num_clusters {num_clusters}"

    # Set defaults
    if max_selected is None:
        max_selected = num_clusters
    if scaling is None:
        scaling = 1.0 / (head_dim**0.5)

    # Allocate output tensors
    selected_cluster_ids = torch.full(
        (num_heads, max_selected), fill_value=-1, dtype=torch.int32, device=q.device
    )
    num_selected = torch.zeros(num_heads, dtype=torch.int32, device=q.device)
    cluster_scores = torch.zeros(
        (num_heads, max_selected), dtype=torch.float32, device=q.device
    )

    # Determine block size (must be power of 2 and >= head_dim)
    BLOCK_SIZE = triton.next_power_of_2(head_dim)
    BLOCK_SIZE = max(BLOCK_SIZE, 128)  # Minimum 128 for efficiency

    # Launch kernel: one program per head
    grid = (num_heads,)

    fused_centroid_scoring_topp_kernel[grid](
        q,
        centroids,
        cluster_sizes,
        selected_cluster_ids,
        cluster_scores,
        num_selected,
        num_heads=num_heads,
        num_clusters=num_clusters,
        head_dim=head_dim,
        max_selected=max_selected,
        top_p=top_p,
        scaling=scaling,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return selected_cluster_ids, num_selected, cluster_scores


# Benchmark helper function
def benchmark_triton_vs_pytorch(
    num_heads: int = 32,
    num_clusters: int = 500,
    head_dim: int = 128,
    top_p: float = 0.9,
    num_iterations: int = 100,
    device: str = "cuda",
):
    """Benchmark Triton kernel vs PyTorch implementation.

    Args:
        num_heads: Number of query heads
        num_clusters: Number of clusters
        head_dim: Dimension of each head
        top_p: Top-p threshold
        num_iterations: Number of iterations for timing
        device: Device to run on

    Returns:
        dict with timing results and speedup
    """
    import math
    import time

    # Generate random test data
    q = torch.randn(num_heads, head_dim, device=device)
    centroids = torch.randn(num_clusters, head_dim, device=device)
    cluster_sizes = torch.randint(10, 100, (num_clusters,), device=device)
    scaling = 1.0 / math.sqrt(head_dim)

    # Warmup
    for _ in range(10):
        _ = fused_centroid_scoring_topp(
            q, centroids, cluster_sizes, top_p, scaling=scaling
        )
    torch.cuda.synchronize()

    # Benchmark Triton
    start = time.time()
    for _ in range(num_iterations):
        _ = fused_centroid_scoring_topp(
            q, centroids, cluster_sizes, top_p, scaling=scaling
        )
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / num_iterations * 1000  # ms

    # Benchmark PyTorch (reference implementation)
    def pytorch_reference():
        results = []
        for h in range(num_heads):
            # Centroid scoring
            score = torch.matmul(q[h], centroids.T) * scaling
            score = score + torch.log(cluster_sizes.float() + 1e-8)
            score = torch.softmax(score, dim=0)

            # Top-p selection
            sorted_scores, sorted_indices = torch.sort(score, descending=True)
            cumsum_scores = torch.cumsum(sorted_scores, dim=0)
            mask = cumsum_scores <= top_p
            if mask.sum() == 0:
                mask[0] = True
            else:
                # Include first element that exceeds threshold
                first_exceed = (cumsum_scores > top_p).nonzero(as_tuple=True)[0]
                if len(first_exceed) > 0:
                    mask[first_exceed[0]] = True
            selected = sorted_indices[mask]
            results.append(selected)
        return results

    # Warmup PyTorch
    for _ in range(10):
        _ = pytorch_reference()
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_iterations):
        _ = pytorch_reference()
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / num_iterations * 1000  # ms

    speedup = pytorch_time / triton_time

    return {
        "triton_time_ms": triton_time,
        "pytorch_time_ms": pytorch_time,
        "speedup": speedup,
        "num_heads": num_heads,
        "num_clusters": num_clusters,
        "head_dim": head_dim,
        "top_p": top_p,
    }


if __name__ == "__main__":
    # Quick test
    print("Testing Triton kernel...")

    num_heads = 16
    num_clusters = 500
    head_dim = 128
    top_p = 0.9

    q = torch.randn(num_heads, head_dim, device="cuda")
    centroids = torch.randn(num_clusters, head_dim, device="cuda")
    cluster_sizes = torch.randint(10, 100, (num_clusters,), device="cuda")

    selected_ids, num_selected, scores = fused_centroid_scoring_topp(
        q, centroids, cluster_sizes, top_p=top_p
    )

    print(f"Selected cluster IDs shape: {selected_ids.shape}")
    print(f"Number selected per head: {num_selected}")
    print(f"First head selected {num_selected[0].item()} clusters")
    print(f"First head cluster IDs: {selected_ids[0, :num_selected[0].item()]}")

    # Run benchmark
    print("\nRunning benchmark...")
    results = benchmark_triton_vs_pytorch()
    print(f"Triton time: {results['triton_time_ms']:.3f} ms")
    print(f"PyTorch time: {results['pytorch_time_ms']:.3f} ms")
    print(f"Speedup: {results['speedup']:.2f}x")
