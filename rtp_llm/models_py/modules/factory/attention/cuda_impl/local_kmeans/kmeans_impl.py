"""
PyTorch native K-Means implementation.

This module provides a pure PyTorch implementation of K-Means clustering
using Euclidean distance, ported from flash-kmeans to avoid external dependencies.

Original implementation: flash-kmeans (https://github.com/your-repo/flash-kmeans)
"""

from typing import Optional, Tuple

import torch


def kmeans_plusplus_init(x: torch.Tensor, n_clusters: int) -> torch.Tensor:
    """
    K-means++ initialization for better convergence.

    Args:
        x: Tensor of shape (B, N, D) - batch_size B, N points per batch, D dimensions
        n_clusters: Number of clusters (K)

    Returns:
        centroids: (B, n_clusters, D) - initialized centroids
    """
    B, N, D = x.shape
    centroids = torch.zeros(B, n_clusters, D, device=x.device, dtype=x.dtype)

    # Select first centroid randomly
    first_idx = torch.randint(0, N, (B,), device=x.device)
    centroids[:, 0] = x[torch.arange(B, device=x.device), first_idx]

    # Select remaining centroids
    for k in range(1, n_clusters):
        # Compute distances to nearest existing centroid
        x_for_dist = x.to(torch.float32)
        centroids_for_dist = centroids[:, :k].to(torch.float32)
        distances = torch.cdist(x_for_dist, centroids_for_dist, p=2).min(dim=-1)[
            0
        ]  # (B, N)

        # Select next centroid with probability proportional to distance^2
        probs = distances**2
        probs = probs / (
            probs.sum(dim=-1, keepdim=True) + 1e-8
        )  # Add epsilon for numerical stability

        # Sample from categorical distribution
        next_idx = torch.multinomial(probs, 1).squeeze(-1)  # (B,)
        centroids[:, k] = x[torch.arange(B, device=x.device), next_idx]

    return centroids


def batch_kmeans_Euclid(
    x: torch.Tensor,
    n_clusters: int,
    max_iters: int = 100,
    tol: float = 0.0,
    init_centroids: Optional[torch.Tensor] = None,
    verbose: bool = False,
    use_kmeanspp: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Batched K-Means clustering using Euclidean distance.

    Args:
        x: Tensor of shape (B, N, D) - batch_size B, N points per batch, D dimensions
        n_clusters: Number of clusters (K)
        max_iters: Maximum number of iterations
        tol: Tolerance for convergence (centroid movement threshold)
        init_centroids: Optional initial centroids of shape (B, n_clusters, D)
        verbose: Whether to print convergence information
        use_kmeanspp: Whether to use K-means++ initialization (default: True)

    Returns:
        cluster_ids: (B, N) LongTensor - cluster assignment for each point
        centroids: (B, n_clusters, D) - final cluster centers
        n_iters: Number of iterations performed

    Examples:
        >>> x = torch.randn(2, 1000, 128, device='cuda')
        >>> cluster_ids, centroids, n_iters = batch_kmeans_Euclid(x, n_clusters=10)
        >>> print(cluster_ids.shape)  # (2, 1000)
        >>> print(centroids.shape)    # (2, 10, 128)
    """
    B, N, D = x.shape

    # Initialize centroids
    if init_centroids is None:
        if use_kmeanspp:
            # Use K-means++ initialization for better convergence
            centroids = kmeans_plusplus_init(x, n_clusters)
        else:
            # Randomly select initial centers from x
            indices = torch.randint(0, N, (B, n_clusters), device=x.device)
            centroids = torch.gather(
                x, dim=1, index=indices[..., None].expand(-1, -1, D)
            )  # (B, n_clusters, D)
    else:
        centroids = init_centroids.view(B, n_clusters, D)

    cluster_ids = None

    # K-Means iterations
    for it in range(max_iters):
        # Step 1: Assign points to nearest centroid
        # Compute pairwise Euclidean distances: (B, N, K)
        # Note: torch.cdist doesn't support bfloat16, convert to float32
        x_for_dist = x.to(torch.float32)
        centroids_for_dist = centroids.to(torch.float32)
        distances = torch.cdist(x_for_dist, centroids_for_dist, p=2)

        # Assign to nearest cluster
        cluster_ids = distances.argmin(dim=-1)  # (B, N)

        # Step 2: Update centroids
        centroids_new = torch.zeros_like(centroids)  # (B, n_clusters, D)

        # For each batch
        for b in range(B):
            for k in range(n_clusters):
                # Find points assigned to cluster k
                mask = cluster_ids[b] == k
                if mask.any():
                    # Compute mean of assigned points (ensure dtype compatibility)
                    centroids_new[b, k] = x[b, mask].mean(dim=0).to(centroids_new.dtype)
                else:
                    # Keep old centroid if no points assigned
                    centroids_new[b, k] = centroids[b, k]

        # Step 3: Check for convergence
        center_shift = (centroids_new - centroids).norm(dim=-1).max()

        if verbose:
            print(f"Iter {it}, center shift: {center_shift.item():.6f}")

        if center_shift < tol:
            centroids = centroids_new
            break

        centroids = centroids_new

    # Ensure cluster_ids is int64 (required for bincount and other operations)
    return cluster_ids.to(torch.int64), centroids, it + 1


def batch_kmeans_Euclid_optimized(
    x: torch.Tensor,
    n_clusters: int,
    max_iters: int = 100,
    tol: float = 0.0,
    init_centroids: Optional[torch.Tensor] = None,
    verbose: bool = False,
    use_kmeanspp: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Optimized batched K-Means using scatter operations.

    This version uses scatter_add for faster centroid updates,
    avoiding explicit loops over clusters.
    """
    B, N, D = x.shape

    # Initialize centroids
    if init_centroids is None:
        if use_kmeanspp:
            # Use K-means++ initialization for better convergence
            centroids = kmeans_plusplus_init(x, n_clusters)
        else:
            indices = torch.randint(0, N, (B, n_clusters), device=x.device)
            centroids = torch.gather(
                x, dim=1, index=indices[..., None].expand(-1, -1, D)
            )
    else:
        centroids = init_centroids.view(B, n_clusters, D)

    cluster_ids = None

    # K-Means iterations
    for it in range(max_iters):
        # Assign points to nearest centroid
        # Note: torch.cdist doesn't support bfloat16, convert to float32
        x_for_dist = x.to(torch.float32)
        centroids_for_dist = centroids.to(torch.float32)
        distances = torch.cdist(x_for_dist, centroids_for_dist, p=2)
        cluster_ids = distances.argmin(dim=-1)  # (B, N)

        # Update centroids using scatter operations
        centroids_new = torch.zeros_like(centroids)
        counts = torch.zeros(B, n_clusters, device=x.device, dtype=torch.float32)

        # Accumulate sums and counts for each cluster
        for b in range(B):
            # Expand labels for scatter_add: (N,) -> (N, D)
            labels_expanded = cluster_ids[b].unsqueeze(-1).expand(-1, D)

            # Sum points for each cluster (ensure dtype compatibility)
            centroids_new[b].scatter_add_(
                0, labels_expanded, x[b].to(centroids_new.dtype)
            )

            # Count points in each cluster
            counts[b].scatter_add_(
                0, cluster_ids[b], torch.ones(N, device=x.device, dtype=counts.dtype)
            )

        # Compute means (avoid division by zero)
        counts_safe = counts.unsqueeze(-1).clamp(min=1.0)
        centroids_new = (centroids_new / counts_safe).to(centroids.dtype)

        # Keep old centroids for empty clusters
        empty_mask = (counts == 0).unsqueeze(-1)  # (B, K, 1)
        centroids_new = torch.where(empty_mask, centroids, centroids_new)

        # Check convergence
        center_shift = (centroids_new - centroids).norm(dim=-1).max()

        if verbose:
            print(f"Iter {it}, center shift: {center_shift.item():.6f}")

        if center_shift < tol:
            centroids = centroids_new
            break

        centroids = centroids_new

    # Ensure cluster_ids is int64 (required for bincount and other operations)
    return cluster_ids.to(torch.int64), centroids, it + 1


# Export functions
__all__ = [
    "batch_kmeans_Euclid",
    "batch_kmeans_Euclid_optimized",
    "kmeans_plusplus_init",
]
