"""
Unit tests for local K-Means implementation.

This test file verifies that the PyTorch native implementation
maintains API compatibility and produces reasonable results.
"""

import torch

try:
    # Try relative import (when run as module)
    from .incremental_kmeans import IncrementalKMeans
    from .kmeans_impl import batch_kmeans_Euclid
except ImportError:
    # Try absolute import (when run directly)
    from incremental_kmeans import IncrementalKMeans
    from kmeans_impl import batch_kmeans_Euclid


def test_batch_kmeans_euclid_basic():
    """Test basic functionality of batch_kmeans_Euclid."""
    # Create simple test data
    x = torch.randn(2, 100, 128, device="cuda", dtype=torch.float32)
    n_clusters = 10

    cluster_ids, centroids, n_iters = batch_kmeans_Euclid(
        x, n_clusters=n_clusters, max_iters=20, tol=1e-4
    )

    # Check shapes
    assert cluster_ids.shape == (2, 100), f"Expected (2, 100), got {cluster_ids.shape}"
    assert centroids.shape == (
        2,
        n_clusters,
        128,
    ), f"Expected (2, {n_clusters}, 128), got {centroids.shape}"
    assert 0 < n_iters <= 20, f"Expected n_iters in (0, 20], got {n_iters}"

    # Check that all points are assigned to valid clusters
    assert cluster_ids.min() >= 0
    assert cluster_ids.max() < n_clusters

    print(f"✓ batch_kmeans_Euclid basic test passed (n_iters={n_iters})")


def test_batch_kmeans_euclid_single_batch():
    """Test batch_kmeans_Euclid with single batch (common use case)."""
    x = torch.randn(1, 500, 64, device="cuda", dtype=torch.bfloat16)
    n_clusters = 20

    cluster_ids, centroids, n_iters = batch_kmeans_Euclid(
        x, n_clusters=n_clusters, max_iters=10
    )

    assert cluster_ids.shape == (1, 500)
    assert centroids.shape == (1, n_clusters, 64)

    # Check that centroids are reasonable (not NaN or inf)
    assert not torch.isnan(centroids).any()
    assert not torch.isinf(centroids).any()

    print(f"✓ batch_kmeans_Euclid single batch test passed")


def test_incremental_kmeans_init_and_add():
    """Test IncrementalKMeans initialization and adding points."""
    n_clusters = 10
    dim = 128
    device = "cuda"
    dtype = torch.bfloat16

    # Initialize model
    model = IncrementalKMeans(
        n_clusters=n_clusters, dim=dim, device=device, dtype=dtype
    )

    # Initialize with random centroids
    init_centroids = torch.randn(n_clusters, dim, device=device, dtype=dtype)
    model.init_centroids(init_centroids)

    assert model.is_fitted
    assert model.centroids.shape == (n_clusters, dim)

    # Add some points
    points = torch.randn(50, dim, device=device, dtype=dtype)
    labels = model.add_points(points, update_centroids=True)

    assert labels.shape == (50,)
    assert labels.min() >= 0
    assert labels.max() < n_clusters

    # Get centroids and statistics
    centroids = model.get_centroids()
    sums, counts = model.get_statistics()

    assert centroids.shape == (n_clusters, dim)
    assert sums.shape == (n_clusters, dim)
    assert counts.shape == (n_clusters,)
    assert counts.sum() == 50  # Total points added

    print(f"✓ IncrementalKMeans init and add test passed")


def test_incremental_kmeans_single_point():
    """Test adding single points (decode use case)."""
    n_clusters = 5
    dim = 64
    device = "cuda"
    dtype = torch.float32

    # Initialize
    model = IncrementalKMeans(
        n_clusters=n_clusters, dim=dim, device=device, dtype=dtype
    )
    init_centroids = torch.randn(n_clusters, dim, device=device, dtype=dtype)
    model.init_centroids(init_centroids)

    # Add single point multiple times
    for _ in range(10):
        single_point = torch.randn(1, dim, device=device, dtype=dtype)
        label = model.add_points(single_point, update_centroids=True)
        assert label.shape == (1,)

    _, counts = model.get_statistics()
    assert counts.sum() == 10

    print(f"✓ IncrementalKMeans single point test passed")


def test_incremental_kmeans_compatibility():
    """Test that IncrementalKMeans works with batch_kmeans_Euclid output."""
    # Simulate the torch_naive.py usage pattern
    n_clusters = 10
    dim = 128
    seq_len = 500

    # Step 1: Initial clustering with batch_kmeans_Euclid
    k = torch.randn(seq_len, dim, device="cuda", dtype=torch.bfloat16)
    k_batched = k.unsqueeze(0)  # [1, seq_len, dim]

    cluster_ids, centroids, n_iters = batch_kmeans_Euclid(
        k_batched,
        n_clusters,
        max_iters=10,
        tol=1e-4,
    )

    # Remove batch dimension
    cluster_ids = cluster_ids.squeeze(0)  # [seq_len]
    centroids = centroids.squeeze(0)  # [n_clusters, dim]

    # Step 2: Create IncrementalKMeans and initialize
    model = IncrementalKMeans(
        n_clusters=n_clusters,
        dim=dim,
        device=k.device,
        dtype=k.dtype,
    )
    model.init_centroids(centroids)

    # Add initial data
    model.add_points(k, update_centroids=False)

    # Step 3: Add new points (simulating decode)
    for _ in range(5):
        k_new = torch.randn(1, dim, device="cuda", dtype=torch.bfloat16)
        label = model.add_points(k_new, update_centroids=True)
        assert label.shape == (1,)

    # Verify statistics
    new_centroids = model.get_centroids()
    _, counts = model.get_statistics()

    assert new_centroids.shape == (n_clusters, dim)
    assert counts.sum() == seq_len + 5  # Initial + 5 new points

    print(f"✓ IncrementalKMeans compatibility test passed")


def test_error_handling():
    """Test error handling for invalid inputs."""
    model = IncrementalKMeans(n_clusters=10, dim=128)

    # Should raise error before initialization
    try:
        model.add_points(torch.randn(10, 128))
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass

    # Initialize
    model.init_centroids(torch.randn(10, 128))

    # Should raise error for wrong dimension
    try:
        model.add_points(torch.randn(10, 64))  # Wrong dim
        assert False, "Expected ValueError"
    except ValueError:
        pass

    print(f"✓ Error handling test passed")


if __name__ == "__main__":
    print("Running local K-Means tests...\n")

    test_batch_kmeans_euclid_basic()
    test_batch_kmeans_euclid_single_batch()
    test_incremental_kmeans_init_and_add()
    test_incremental_kmeans_single_point()
    test_incremental_kmeans_compatibility()
    test_error_handling()

    print("\n✅ All tests passed!")
