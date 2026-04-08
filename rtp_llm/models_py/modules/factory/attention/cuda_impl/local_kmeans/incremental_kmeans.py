"""
Incremental K-Means implementation using PyTorch native operations.

This module provides online/incremental K-Means clustering by maintaining
running statistics (centroid sums and counts) and updating them as new
points arrive.

Ported from flash-kmeans to use pure PyTorch operations instead of Triton,
removing external dependencies while maintaining the same API.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


class IncrementalKMeans:
    """
    Incremental K-Means clustering using PyTorch native operations.

    This class maintains running statistics (sums and counts) for each cluster,
    allowing efficient updates when new points are added without recomputing
    from scratch.

    Parameters
    ----------
    n_clusters : int
        Number of clusters (K)
    dim : int
        Feature dimensionality (D)
    device : torch.device or str, optional
        Device to use (default: cuda if available)
    dtype : torch.dtype, optional
        Data type for centroids (default: torch.float32)

    Attributes
    ----------
    centroids : torch.Tensor (K, D)
        Current cluster centroids
    centroid_sums : torch.Tensor (K, D)
        Sum of all points assigned to each cluster
    centroid_counts : torch.Tensor (K,)
        Number of points assigned to each cluster
    is_fitted : bool
        Whether the model has been initialized

    Examples
    --------
    >>> # Initialize with existing centroids
    >>> model = IncrementalKMeans(n_clusters=100, dim=128)
    >>> initial_centroids = torch.randn(100, 128, device='cuda')
    >>> model.init_centroids(initial_centroids)
    >>>
    >>> # Add new points incrementally
    >>> new_points = torch.randn(500, 128, device='cuda')
    >>> labels = model.add_points(new_points)
    >>>
    >>> # Get current centroids
    >>> centroids = model.get_centroids()
    """

    def __init__(
        self,
        n_clusters: int,
        dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.K = n_clusters
        self.D = dim
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.dtype = dtype or torch.float32

        # State variables
        self.centroids: Optional[torch.Tensor] = None
        self.centroid_sums: Optional[torch.Tensor] = None
        self.centroid_counts: Optional[torch.Tensor] = None
        self.is_fitted = False

    def init_centroids(self, centroids: torch.Tensor) -> "IncrementalKMeans":
        """
        Initialize centroids from existing data.

        Parameters
        ----------
        centroids : torch.Tensor (K, D)
            Initial centroids

        Returns
        -------
        self : IncrementalKMeans
        """
        if centroids.ndim != 2:
            raise ValueError(
                f"centroids must be 2D (K, D), got shape {centroids.shape}"
            )

        K, D = centroids.shape
        if K != self.K or D != self.D:
            raise ValueError(
                f"centroids shape {centroids.shape} doesn't match (K={self.K}, D={self.D})"
            )

        self.centroids = centroids.to(device=self.device, dtype=self.dtype).clone()

        # Initialize empty statistics
        self.centroid_sums = torch.zeros(
            (self.K, self.D), device=self.device, dtype=torch.float32
        )
        self.centroid_counts = torch.zeros(
            self.K, device=self.device, dtype=torch.int32
        )

        self.is_fitted = True
        return self

    def init_with_data(
        self, data: torch.Tensor, method: str = "kmeans++"
    ) -> "IncrementalKMeans":
        """
        Initialize centroids from data using specified method.

        Parameters
        ----------
        data : torch.Tensor (N, D)
            Data points to initialize from
        method : str
            Initialization method: 'random' or 'kmeans++'

        Returns
        -------
        self : IncrementalKMeans
        """
        if data.ndim != 2:
            raise ValueError(f"data must be 2D (N, D), got shape {data.shape}")

        N, D = data.shape
        if D != self.D:
            raise ValueError(
                f"data dimension {D} doesn't match model dimension {self.D}"
            )

        if N < self.K:
            raise ValueError(
                f"data has {N} points but need at least {self.K} for initialization"
            )

        data = data.to(device=self.device, dtype=self.dtype)

        if method == "random":
            # Random selection
            indices = torch.randperm(N, device=self.device)[: self.K]
            centroids = data[indices].clone()
        elif method == "kmeans++":
            # K-means++ initialization
            centroids = self._kmeans_plus_plus_init(data)
        else:
            raise ValueError(f"Unknown initialization method: {method}")

        return self.init_centroids(centroids)

    def _kmeans_plus_plus_init(self, data: torch.Tensor) -> torch.Tensor:
        """K-means++ initialization algorithm."""
        N, D = data.shape
        centroids = torch.zeros((self.K, D), device=self.device, dtype=self.dtype)

        # Choose first centroid randomly
        first_idx = torch.randint(0, N, (1,), device=self.device)
        centroids[0] = data[first_idx]

        # Choose remaining centroids
        for k in range(1, self.K):
            # Compute distances to nearest centroid
            dists = torch.cdist(
                data.unsqueeze(0), centroids[:k].unsqueeze(0)
            )  # (1, N, k)
            min_dists = dists[0].min(dim=1)[0]  # (N,)

            # Choose next centroid with probability proportional to distance squared
            probs = min_dists**2
            probs = probs / probs.sum()

            next_idx = torch.multinomial(probs, 1)
            centroids[k] = data[next_idx]

        return centroids

    def add_points(
        self, points: torch.Tensor, update_centroids: bool = True
    ) -> torch.Tensor:
        """
        Add new points to the clustering and optionally update centroids.

        Parameters
        ----------
        points : torch.Tensor (N, D) or (B, N, D)
            New points to add. If 3D, treats first dimension as batch.
        update_centroids : bool, default=True
            Whether to update centroids after adding points.
            Set to False if you want to batch multiple additions.

        Returns
        -------
        labels : torch.Tensor (N,) or (B, N)
            Cluster assignments for the new points

        Examples
        --------
        >>> # Single batch
        >>> labels = model.add_points(new_points)
        >>>
        >>> # Multiple batches without intermediate updates
        >>> labels1 = model.add_points(batch1, update_centroids=False)
        >>> labels2 = model.add_points(batch2, update_centroids=False)
        >>> model.recompute_centroids()
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Model not initialized. Call init_centroids() or init_with_data() first."
            )

        # Handle 2D and 3D inputs
        if points.ndim == 2:
            N, D = points.shape
            points_2d = points
            is_batched = False
        elif points.ndim == 3:
            B, N, D = points.shape
            # Flatten batch dimension for processing
            points_2d = points.view(B * N, D)
            is_batched = True
        else:
            raise ValueError(f"points must be 2D or 3D, got shape {points.shape}")

        if D != self.D:
            raise ValueError(
                f"points dimension {D} doesn't match model dimension {self.D}"
            )

        points_2d = points_2d.to(device=self.device, dtype=self.dtype)

        # Step 1: Assign points to nearest centroid using Euclidean distance
        # Compute distances: (N, K)
        # Note: torch.cdist doesn't support bfloat16, so convert to float32
        points_for_dist = points_2d.to(torch.float32)
        centroids_for_dist = self.centroids.to(torch.float32)
        distances = torch.cdist(
            points_for_dist.unsqueeze(0), centroids_for_dist.unsqueeze(0), p=2
        )[0]
        labels = distances.argmin(dim=-1)  # (N,) or (B*N,)

        # Step 2: Update statistics using scatter_add
        # Expand labels for scatter_add: (N,) -> (N, D)
        labels_expanded = labels.unsqueeze(-1).expand(-1, self.D)

        # Sum points for each cluster
        self.centroid_sums.scatter_add_(0, labels_expanded, points_2d.to(torch.float32))

        # Count points in each cluster
        ones = torch.ones(labels.shape[0], device=self.device, dtype=torch.int32)
        self.centroid_counts.scatter_add_(0, labels, ones)

        # Step 3: Recompute centroids if requested
        if update_centroids:
            self.recompute_centroids()

        # Return labels in original shape
        if is_batched:
            return labels.view(B, N).to(torch.int64)
        else:
            return labels.to(torch.int64)

    def recompute_centroids(self) -> torch.Tensor:
        """
        Recompute centroids from accumulated statistics.

        Returns
        -------
        centroids : torch.Tensor (K, D)
            Updated centroids
        """
        if not self.is_fitted:
            raise RuntimeError("Model not initialized.")

        # Compute new centroids: mean = sum / count
        counts_safe = self.centroid_counts.to(torch.float32).clamp(
            min=1.0
        )  # Avoid division by zero
        new_centroids = self.centroid_sums / counts_safe.unsqueeze(-1)

        # Keep old centroids for empty clusters
        empty_mask = self.centroid_counts == 0
        new_centroids[empty_mask] = self.centroids[empty_mask].to(torch.float32)

        self.centroids = new_centroids.to(self.dtype)
        return self.centroids.clone()

    def reset_statistics(self):
        """
        Reset accumulation statistics while keeping centroids.

        Useful if you want to start fresh accumulation from current centroids.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not initialized.")

        self.centroid_sums.zero_()
        self.centroid_counts.zero_()

    def get_centroids(self) -> torch.Tensor:
        """
        Get current centroids.

        Returns
        -------
        centroids : torch.Tensor (K, D)
            Current cluster centroids
        """
        if not self.is_fitted:
            raise RuntimeError("Model not initialized.")
        return self.centroids.clone()

    def get_statistics(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get current accumulation statistics.

        Returns
        -------
        sums : torch.Tensor (K, D)
            Sum of points in each cluster
        counts : torch.Tensor (K,)
            Number of points in each cluster
        """
        if not self.is_fitted:
            raise RuntimeError("Model not initialized.")
        return self.centroid_sums.clone(), self.centroid_counts.clone()

    def predict(self, points: torch.Tensor) -> torch.Tensor:
        """
        Predict cluster assignments without updating centroids.

        Parameters
        ----------
        points : torch.Tensor (N, D) or (B, N, D)
            Points to assign

        Returns
        -------
        labels : torch.Tensor (N,) or (B, N)
            Cluster assignments
        """
        if not self.is_fitted:
            raise RuntimeError("Model not initialized.")

        # Handle 2D and 3D inputs
        if points.ndim == 2:
            points_2d = points
            is_batched = False
        elif points.ndim == 3:
            B, N, D = points.shape
            points_2d = points.view(B * N, D)
            is_batched = True
        else:
            raise ValueError(f"points must be 2D or 3D, got shape {points.shape}")

        points_2d = points_2d.to(device=self.device, dtype=self.dtype)

        # Compute distances and assign to nearest centroid
        # Note: torch.cdist doesn't support bfloat16, so convert to float32
        points_for_dist = points_2d.to(torch.float32)
        centroids_for_dist = self.centroids.to(torch.float32)
        distances = torch.cdist(
            points_for_dist.unsqueeze(0), centroids_for_dist.unsqueeze(0), p=2
        )[0]
        labels = distances.argmin(dim=-1)

        if is_batched:
            return labels.view(B, N).to(torch.int64)
        else:
            return labels.to(torch.int64)

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return (
            f"IncrementalKMeans(n_clusters={self.K}, dim={self.D}, "
            f"device={self.device}, dtype={self.dtype}, status={status})"
        )
