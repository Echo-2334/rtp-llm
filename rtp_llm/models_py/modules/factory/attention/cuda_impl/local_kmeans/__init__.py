"""
Local K-Means implementation for RTP-LLM attention clustering.

This package provides PyTorch native implementations of K-Means clustering
algorithms, replacing the external flash-kmeans dependency.

Main Components:
- batch_kmeans_Euclid: Batched K-Means clustering with Euclidean distance
- IncrementalKMeans: Online K-Means with incremental updates

These implementations use pure PyTorch operations instead of Triton kernels,
trading some performance for simplified dependencies and better maintainability.

Usage:
    >>> from local_kmeans import batch_kmeans_Euclid, IncrementalKMeans
    >>>
    >>> # Initial clustering
    >>> x = torch.randn(1, 1000, 128, device='cuda')
    >>> cluster_ids, centroids, n_iters = batch_kmeans_Euclid(x, n_clusters=10)
    >>>
    >>> # Incremental updates
    >>> model = IncrementalKMeans(n_clusters=10, dim=128, device='cuda')
    >>> model.init_centroids(centroids.squeeze(0))
    >>> new_point = torch.randn(1, 128, device='cuda')
    >>> label = model.add_points(new_point, update_centroids=True)
"""

from .incremental_kmeans import IncrementalKMeans
from .kmeans_impl import (
    batch_kmeans_Euclid,
    batch_kmeans_Euclid_optimized,
    kmeans_plusplus_init,
)

__all__ = [
    "batch_kmeans_Euclid",
    "batch_kmeans_Euclid_optimized",
    "kmeans_plusplus_init",
    "IncrementalKMeans",
]

__version__ = "1.0.0"
