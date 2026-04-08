"""Unit tests for Triton clustered attention kernels."""

import math

import pytest
import torch

from .clustered_attention import fused_centroid_scoring_topp


def _top_p_selection_reference(scores: torch.Tensor, top_p: float) -> torch.Tensor:
    """Reference PyTorch implementation of top-p selection.

    Args:
        scores: Probability distribution [num_clusters]
        top_p: Cumulative probability threshold

    Returns:
        selected_indices: Indices of selected clusters
    """
    sorted_scores, sorted_indices = torch.sort(scores, descending=True)
    cumsum_scores = torch.cumsum(sorted_scores, dim=0)

    # Find clusters within top_p threshold
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
    return selected


def pytorch_centroid_scoring_topp_reference(
    q: torch.Tensor,  # [num_heads, head_dim]
    centroids: torch.Tensor,  # [num_clusters, head_dim]
    cluster_sizes: torch.Tensor,  # [num_clusters]
    top_p: float,
    scaling: float,
):
    """Reference PyTorch implementation for comparison.

    Returns:
        List of sets of selected cluster IDs per head
    """
    num_heads = q.shape[0]
    results = []

    for h in range(num_heads):
        # Centroid scoring
        score = torch.matmul(q[h], centroids.T) * scaling
        score = score + torch.log(cluster_sizes.float() + 1e-8)
        score = torch.softmax(score, dim=0)

        # Top-p selection
        selected = _top_p_selection_reference(score, top_p)
        results.append(set(selected.tolist()))

    return results


class TestFusedCentroidScoringTopp:
    """Test suite for fused_centroid_scoring_topp kernel."""

    @pytest.mark.parametrize("num_heads", [1, 8, 16, 32])
    @pytest.mark.parametrize("num_clusters", [100, 500, 1000])
    @pytest.mark.parametrize("head_dim", [64, 128, 256])
    @pytest.mark.parametrize("top_p", [0.8, 0.9, 0.95])
    def test_correctness(self, num_heads, num_clusters, head_dim, top_p):
        """Test that Triton kernel produces same results as PyTorch reference."""
        device = "cuda"
        scaling = 1.0 / math.sqrt(head_dim)

        # Generate random test data
        q = torch.randn(num_heads, head_dim, device=device)
        centroids = torch.randn(num_clusters, head_dim, device=device)
        cluster_sizes = torch.randint(10, 100, (num_clusters,), device=device)

        # PyTorch reference
        expected = pytorch_centroid_scoring_topp_reference(
            q, centroids, cluster_sizes, top_p, scaling
        )

        # Triton kernel
        selected_ids, num_selected, _ = fused_centroid_scoring_topp(
            q, centroids, cluster_sizes, top_p=top_p, scaling=scaling
        )

        # Compare results
        for h in range(num_heads):
            n = num_selected[h].item()
            triton_set = set(selected_ids[h, :n].tolist())

            assert triton_set == expected[h], (
                f"Head {h}: Mismatch in selected clusters.\n"
                f"Expected: {sorted(expected[h])}\n"
                f"Got: {sorted(triton_set)}"
            )

    def test_edge_case_single_cluster(self):
        """Test edge case with only one cluster."""
        num_heads = 4
        num_clusters = 1
        head_dim = 128
        top_p = 0.9

        q = torch.randn(num_heads, head_dim, device="cuda")
        centroids = torch.randn(num_clusters, head_dim, device="cuda")
        cluster_sizes = torch.tensor([50], device="cuda")

        selected_ids, num_selected, _ = fused_centroid_scoring_topp(
            q, centroids, cluster_sizes, top_p=top_p
        )

        # Should select the only cluster for all heads
        assert (num_selected == 1).all(), "Should select the single cluster"
        assert (selected_ids[:, 0] == 0).all(), "Should select cluster 0"

    def test_edge_case_all_clusters(self):
        """Test edge case with top_p = 1.0 (select all clusters)."""
        num_heads = 4
        num_clusters = 10
        head_dim = 128
        top_p = 1.0

        q = torch.randn(num_heads, head_dim, device="cuda")
        centroids = torch.randn(num_clusters, head_dim, device="cuda")
        cluster_sizes = torch.randint(10, 100, (num_clusters,), device="cuda")

        selected_ids, num_selected, _ = fused_centroid_scoring_topp(
            q, centroids, cluster_sizes, top_p=top_p
        )

        # Should select all clusters for all heads
        assert (num_selected == num_clusters).all(), (
            f"Should select all {num_clusters} clusters with top_p=1.0, "
            f"got {num_selected}"
        )

    def test_edge_case_uniform_distribution(self):
        """Test with uniform cluster sizes and similar centroids."""
        num_heads = 4
        num_clusters = 100
        head_dim = 128
        top_p = 0.9

        # Create nearly identical centroids (small random noise)
        base_centroid = torch.randn(1, head_dim, device="cuda")
        centroids = base_centroid + 0.01 * torch.randn(
            num_clusters, head_dim, device="cuda"
        )

        q = torch.randn(num_heads, head_dim, device="cuda")
        cluster_sizes = torch.full((num_clusters,), 50, device="cuda")  # Uniform sizes

        selected_ids, num_selected, scores = fused_centroid_scoring_topp(
            q, centroids, cluster_sizes, top_p=top_p
        )

        # With uniform distribution, should select many clusters to reach top_p
        for h in range(num_heads):
            n = num_selected[h].item()
            # Should select at least 80% of clusters for top_p=0.9 with uniform dist
            assert n >= num_clusters * 0.8, (
                f"Head {h}: Expected to select many clusters with uniform dist, "
                f"got {n}/{num_clusters}"
            )

    def test_edge_case_single_dominant_cluster(self):
        """Test with one very large cluster dominating."""
        num_heads = 4
        num_clusters = 100
        head_dim = 128
        top_p = 0.9

        q = torch.randn(num_heads, head_dim, device="cuda")
        centroids = torch.randn(num_clusters, head_dim, device="cuda")

        # Make cluster 0 very large
        cluster_sizes = torch.randint(1, 10, (num_clusters,), device="cuda")
        cluster_sizes[0] = 10000  # Dominant cluster

        selected_ids, num_selected, scores = fused_centroid_scoring_topp(
            q, centroids, cluster_sizes, top_p=top_p
        )

        # Cluster 0 should be selected for all heads due to log weighting
        for h in range(num_heads):
            n = num_selected[h].item()
            assert (
                0 in selected_ids[h, :n].tolist()
            ), f"Head {h}: Dominant cluster 0 should be selected"

    def test_padding_correctness(self):
        """Test that padding values are correctly set to -1."""
        num_heads = 4
        num_clusters = 100
        head_dim = 128
        top_p = 0.5  # Low threshold to select fewer clusters

        q = torch.randn(num_heads, head_dim, device="cuda")
        centroids = torch.randn(num_clusters, head_dim, device="cuda")
        cluster_sizes = torch.randint(10, 100, (num_clusters,), device="cuda")

        selected_ids, num_selected, scores = fused_centroid_scoring_topp(
            q, centroids, cluster_sizes, top_p=top_p, max_selected=num_clusters
        )

        # Check padding
        for h in range(num_heads):
            n = num_selected[h].item()
            # Valid IDs should be in range [0, num_clusters)
            assert (selected_ids[h, :n] >= 0).all(), "Valid IDs should be non-negative"
            assert (
                selected_ids[h, :n] < num_clusters
            ).all(), "Valid IDs should be < num_clusters"
            # Padding should be -1
            if n < num_clusters:
                assert (selected_ids[h, n:] == -1).all(), "Padding should be -1"
            # Scores for padding should be 0
            if n < num_clusters:
                assert (scores[h, n:] == 0.0).all(), "Padding scores should be 0"

    def test_determinism(self):
        """Test that kernel produces deterministic results."""
        num_heads = 16
        num_clusters = 500
        head_dim = 128
        top_p = 0.9

        q = torch.randn(num_heads, head_dim, device="cuda")
        centroids = torch.randn(num_clusters, head_dim, device="cuda")
        cluster_sizes = torch.randint(10, 100, (num_clusters,), device="cuda")

        # Run twice
        result1 = fused_centroid_scoring_topp(q, centroids, cluster_sizes, top_p=top_p)
        result2 = fused_centroid_scoring_topp(q, centroids, cluster_sizes, top_p=top_p)

        # Should get identical results
        assert torch.equal(result1[0], result2[0]), "Selected IDs should be identical"
        assert torch.equal(result1[1], result2[1]), "Num selected should be identical"
        assert torch.allclose(result1[2], result2[2]), "Scores should be identical"

    def test_input_validation(self):
        """Test input validation and error handling."""
        num_heads = 4
        num_clusters = 100
        head_dim = 128

        q = torch.randn(num_heads, head_dim, device="cuda")
        centroids = torch.randn(num_clusters, head_dim, device="cuda")
        cluster_sizes = torch.randint(10, 100, (num_clusters,), device="cuda")

        # Test CPU tensor (should fail)
        with pytest.raises(AssertionError):
            fused_centroid_scoring_topp(q.cpu(), centroids, cluster_sizes, top_p=0.9)

        # Test shape mismatch (should fail)
        with pytest.raises(AssertionError):
            fused_centroid_scoring_topp(q, centroids[:50, :], cluster_sizes, top_p=0.9)

        # Test wrong dimensions (should fail)
        with pytest.raises(AssertionError):
            fused_centroid_scoring_topp(
                q.unsqueeze(0), centroids, cluster_sizes, top_p=0.9
            )


class TestBenchmark:
    """Benchmark tests for performance comparison."""

    def test_benchmark_smoke(self):
        """Smoke test for benchmark function."""
        from .clustered_attention import benchmark_triton_vs_pytorch

        results = benchmark_triton_vs_pytorch(
            num_heads=16,
            num_clusters=500,
            head_dim=128,
            top_p=0.9,
            num_iterations=10,  # Small number for testing
        )

        assert "triton_time_ms" in results
        assert "pytorch_time_ms" in results
        assert "speedup" in results
        assert results["triton_time_ms"] > 0
        assert results["pytorch_time_ms"] > 0
        assert results["speedup"] > 0

    @pytest.mark.benchmark
    def test_benchmark_detailed(self):
        """Detailed benchmark across various configurations."""
        from .clustered_attention import benchmark_triton_vs_pytorch

        configs = [
            {"num_heads": 16, "num_clusters": 100},
            {"num_heads": 16, "num_clusters": 500},
            {"num_heads": 32, "num_clusters": 500},
            {"num_heads": 32, "num_clusters": 1000},
        ]

        print("\n" + "=" * 80)
        print("Triton Kernel Performance Benchmark")
        print("=" * 80)
        print(
            f"{'Config':<30} {'Triton (ms)':<15} {'PyTorch (ms)':<15} {'Speedup':<10}"
        )
        print("-" * 80)

        for config in configs:
            results = benchmark_triton_vs_pytorch(
                num_heads=config["num_heads"],
                num_clusters=config["num_clusters"],
                head_dim=128,
                top_p=0.9,
                num_iterations=100,
            )

            config_str = f"H={config['num_heads']}, C={config['num_clusters']}"
            print(
                f"{config_str:<30} "
                f"{results['triton_time_ms']:<15.3f} "
                f"{results['pytorch_time_ms']:<15.3f} "
                f"{results['speedup']:<10.2f}x"
            )

        print("=" * 80)


if __name__ == "__main__":
    # Run basic tests
    print("Running basic correctness test...")
    test = TestFusedCentroidScoringTopp()

    try:
        test.test_correctness(num_heads=16, num_clusters=500, head_dim=128, top_p=0.9)
        print("✓ Correctness test passed")
    except AssertionError as e:
        print(f"✗ Correctness test failed: {e}")

    try:
        test.test_edge_case_single_cluster()
        print("✓ Single cluster edge case passed")
    except AssertionError as e:
        print(f"✗ Single cluster edge case failed: {e}")

    try:
        test.test_padding_correctness()
        print("✓ Padding correctness passed")
    except AssertionError as e:
        print(f"✗ Padding correctness failed: {e}")

    try:
        test.test_determinism()
        print("✓ Determinism test passed")
    except AssertionError as e:
        print(f"✗ Determinism test failed: {e}")

    # Run benchmark
    print("\nRunning benchmark...")
    benchmark_test = TestBenchmark()
    benchmark_test.test_benchmark_detailed()
