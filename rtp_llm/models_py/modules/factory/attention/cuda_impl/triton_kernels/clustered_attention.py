"""Simplified Triton kernel for clustered decode attention.

This version simplifies the implementation to work within Triton's constraints:
- No break statements
- Use masking instead of early termination
- Pre-allocated power-of-2 sized buffers
"""

import math
from typing import Tuple

import torch


def fused_centroid_scoring_topp_pytorch(
    q: torch.Tensor,  # [num_heads, head_dim]
    centroids: torch.Tensor,  # [num_clusters, head_dim]
    cluster_sizes: torch.Tensor,  # [num_clusters]
    top_p: float = 0.9,
    max_selected: int = None,
    scaling: float = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched PyTorch implementation (optimized fallback when Triton not available).

    This uses batched matrix operations to process all heads in parallel.
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
        scaling = 1.0 / math.sqrt(head_dim)

    # Allocate output tensors
    selected_cluster_ids = torch.full(
        (num_heads, max_selected), fill_value=-1, dtype=torch.int32, device=q.device
    )
    num_selected = torch.zeros(num_heads, dtype=torch.int32, device=q.device)
    cluster_scores = torch.zeros(
        (num_heads, max_selected), dtype=torch.float32, device=q.device
    )

    # Batched centroid scoring: [num_heads, head_dim] @ [head_dim, num_clusters] = [num_heads, num_clusters]
    scores = torch.matmul(q, centroids.T) * scaling  # [num_heads, num_clusters]

    # Add log cluster sizes (broadcasting)
    log_sizes = torch.log(cluster_sizes.float() + 1e-8)  # [num_clusters]
    scores = scores + log_sizes  # [num_heads, num_clusters]

    # Softmax
    scores = torch.softmax(scores, dim=1)  # [num_heads, num_clusters]

    # Top-p selection per head (this part still needs to be sequential)
    for head_idx in range(num_heads):
        head_scores = scores[head_idx, :]  # [num_clusters]

        # Sort in descending order
        sorted_scores, sorted_indices = torch.sort(head_scores, descending=True)
        cumsum_scores = torch.cumsum(sorted_scores, dim=0)

        # Find clusters within top_p
        mask = cumsum_scores <= top_p

        # Always include at least one cluster
        if mask.sum() == 0:
            mask[0] = True
        else:
            # Include the first cluster that exceeds threshold
            first_exceed = (cumsum_scores > top_p).nonzero(as_tuple=True)[0]
            if len(first_exceed) > 0:
                mask[first_exceed[0]] = True

        selected = sorted_indices[mask]
        n = min(len(selected), max_selected)

        # Store results
        selected_cluster_ids[head_idx, :n] = selected[:n]
        cluster_scores[head_idx, :n] = sorted_scores[mask][:n]
        num_selected[head_idx] = n

    return selected_cluster_ids, num_selected, cluster_scores


def fused_centroid_scoring_topp_vectorized(
    q: torch.Tensor,  # [num_heads, head_dim]
    centroids: torch.Tensor,  # [num_clusters, head_dim]
    cluster_sizes: torch.Tensor,  # [num_clusters]
    top_p: float = 0.9,
    max_selected: int = None,
    scaling: float = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized PyTorch implementation with batched top-p selection.

    This version vectorizes the top-p selection loop to process all heads in parallel.
    Uses padding to handle variable-length outputs.
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
        scaling = 1.0 / math.sqrt(head_dim)

    # Batched centroid scoring: [num_heads, head_dim] @ [head_dim, num_clusters] = [num_heads, num_clusters]
    scores = torch.matmul(q, centroids.T) * scaling  # [num_heads, num_clusters]

    # Add log cluster sizes (broadcasting)
    log_sizes = torch.log(cluster_sizes.float() + 1e-8)  # [num_clusters]
    scores = scores + log_sizes  # [num_heads, num_clusters]

    # Softmax
    scores = torch.softmax(scores, dim=1)  # [num_heads, num_clusters]

    # Vectorized top-p selection (all heads in parallel)
    # Step 1: Parallel sort across all heads
    sorted_scores, sorted_indices = torch.sort(
        scores, dim=1, descending=True
    )  # [num_heads, num_clusters]

    # Step 2: Parallel cumsum
    cumsum_scores = torch.cumsum(sorted_scores, dim=1)  # [num_heads, num_clusters]

    # Step 3: Vectorized masking
    # Mask for clusters within top_p
    within_threshold = cumsum_scores <= top_p  # [num_heads, num_clusters]

    # Find first cluster that exceeds threshold
    exceed_threshold = cumsum_scores > top_p  # [num_heads, num_clusters]

    # For each head, find the index of the first exceeding cluster
    # Use cumsum trick: first True becomes 1, rest become > 1
    first_exceed_mask = exceed_threshold & (
        torch.cumsum(exceed_threshold.long(), dim=1) == 1
    )

    # Combine: include within_threshold OR first_exceed
    final_mask = within_threshold | first_exceed_mask  # [num_heads, num_clusters]

    # Edge case: ensure at least one cluster per head
    # If a head has all False, set the first position to True
    no_selection = ~final_mask.any(dim=1)  # [num_heads], True if head has no selection
    final_mask[no_selection, 0] = True

    # Step 4: Gather selected clusters
    # Count how many selected per head
    num_selected = final_mask.sum(dim=1).int()  # [num_heads]

    # Clip to max_selected
    num_selected = torch.minimum(
        num_selected, torch.tensor(max_selected, device=q.device, dtype=torch.int32)
    )

    # Build output tensors with padding
    selected_cluster_ids = torch.full(
        (num_heads, max_selected), fill_value=-1, dtype=torch.int32, device=q.device
    )
    cluster_scores_out = torch.zeros(
        (num_heads, max_selected), dtype=torch.float32, device=q.device
    )

    # For each head, gather the selected indices
    # We need to handle variable-length selections, so use a loop (but it's just assignment, not heavy compute)
    for head_idx in range(num_heads):
        mask = final_mask[head_idx]  # [num_clusters]
        n = num_selected[head_idx].item()

        # Get selected cluster IDs and scores
        selected_ids = sorted_indices[head_idx, mask][:n]
        selected_scores = sorted_scores[head_idx, mask][:n]

        # Store results
        selected_cluster_ids[head_idx, :n] = selected_ids
        cluster_scores_out[head_idx, :n] = selected_scores

    return selected_cluster_ids, num_selected, cluster_scores_out


# Use the PyTorch implementation as the main one for now
# (Triton has too many constraints for this use case)
fused_centroid_scoring_topp = fused_centroid_scoring_topp_pytorch

# Export both versions
__all__ = [
    "fused_centroid_scoring_topp",
    "fused_centroid_scoring_topp_pytorch",
    "fused_centroid_scoring_topp_vectorized",
    "benchmark_triton_vs_pytorch",
]


def benchmark_triton_vs_pytorch(
    num_heads: int = 32,
    num_clusters: int = 500,
    head_dim: int = 128,
    top_p: float = 0.9,
    num_iterations: int = 100,
    device: str = "cuda",
):
    """Benchmark optimized PyTorch vs naive PyTorch implementation."""
    import time

    # Generate random test data
    q = torch.randn(num_heads, head_dim, device=device)
    centroids = torch.randn(num_clusters, head_dim, device=device)
    cluster_sizes = torch.randint(10, 100, (num_clusters,), device=device)
    scaling = 1.0 / math.sqrt(head_dim)

    # Warmup optimized
    for _ in range(10):
        _ = fused_centroid_scoring_topp(
            q, centroids, cluster_sizes, top_p, scaling=scaling
        )
    torch.cuda.synchronize()

    # Benchmark optimized PyTorch
    start = time.time()
    for _ in range(num_iterations):
        _ = fused_centroid_scoring_topp(
            q, centroids, cluster_sizes, top_p, scaling=scaling
        )
    torch.cuda.synchronize()
    optimized_time = (time.time() - start) / num_iterations * 1000  # ms

    # Benchmark naive PyTorch (reference implementation)
    def pytorch_naive():
        results = []
        for h in range(num_heads):
            # Centroid scoring (per-head, not batched)
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
                first_exceed = (cumsum_scores > top_p).nonzero(as_tuple=True)[0]
                if len(first_exceed) > 0:
                    mask[first_exceed[0]] = True
            selected = sorted_indices[mask]
            results.append(selected)
        return results

    # Warmup naive
    for _ in range(10):
        _ = pytorch_naive()
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_iterations):
        _ = pytorch_naive()
    torch.cuda.synchronize()
    naive_time = (time.time() - start) / num_iterations * 1000  # ms

    speedup = naive_time / optimized_time

    return {
        "triton_time_ms": optimized_time,  # Actually optimized PyTorch
        "pytorch_time_ms": naive_time,  # Naive PyTorch
        "speedup": speedup,
        "num_heads": num_heads,
        "num_clusters": num_clusters,
        "head_dim": head_dim,
        "top_p": top_p,
    }


if __name__ == "__main__":
    # Quick test
    print("Testing optimized PyTorch implementation...")

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
    print(f"Optimized PyTorch time: {results['triton_time_ms']:.3f} ms")
    print(f"Naive PyTorch time: {results['pytorch_time_ms']:.3f} ms")
    print(f"Speedup: {results['speedup']:.2f}x")
