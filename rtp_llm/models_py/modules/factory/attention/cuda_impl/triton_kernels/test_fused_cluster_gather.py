"""Tests for fused_cluster_gather.py

Usage:
    python test_fused_cluster_gather.py
"""

import time

import numpy as np
import torch
from fused_cluster_gather import (
    TRITON_AVAILABLE,
    fused_cluster_union_gather_kv,
    fused_cluster_union_gather_kv_pytorch,
)


def create_test_data(
    num_q_heads=4,
    num_clusters=100,
    avg_tokens_per_cluster=50,
    seq_len=4096,
    head_dim=128,
    device="cuda",
):
    """Create synthetic test data."""

    # Create CSR cluster structure
    cluster_sizes = torch.randint(20, 100, (num_clusters,), device=device)

    # Build flat_indices and offsets
    offsets = torch.cat(
        [torch.tensor([0], device=device), torch.cumsum(cluster_sizes, dim=0)]
    )
    total_tokens = offsets[-1].item()

    # Random token indices (ensure within seq_len)
    flat_indices = torch.randint(
        0, seq_len, (total_tokens,), dtype=torch.int32, device=device
    )

    # Selected clusters (each Q head selects 10-20 clusters)
    max_selected = 30
    selected_cluster_ids_batch = torch.full(
        (num_q_heads, max_selected), fill_value=-1, dtype=torch.int32, device=device
    )
    num_selected_batch = torch.zeros(num_q_heads, dtype=torch.int32, device=device)

    for head_idx in range(num_q_heads):
        n_select = torch.randint(10, 21, (1,)).item()
        selected = torch.randperm(num_clusters, device=device)[:n_select]
        selected_cluster_ids_batch[head_idx, :n_select] = selected
        num_selected_batch[head_idx] = n_select

    # Local window
    local_window_start = max(0, seq_len - 128)
    local_window_count = 128

    # K, V cache
    k_cache = torch.randn(seq_len, head_dim, dtype=torch.float16, device=device)
    v_cache = torch.randn(seq_len, head_dim, dtype=torch.float16, device=device)

    return {
        "selected_cluster_ids_batch": selected_cluster_ids_batch,
        "num_selected_batch": num_selected_batch,
        "flat_indices": flat_indices,
        "offsets": offsets,
        "local_window_start": local_window_start,
        "local_window_count": local_window_count,
        "k_cache": k_cache,
        "v_cache": v_cache,
    }


def test_correctness():
    """Test that Triton kernel matches PyTorch implementation."""
    print("=" * 60)
    print("Testing Correctness")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    device = "cuda"
    test_data = create_test_data(device=device)

    # Run PyTorch version
    k_pytorch, v_pytorch, tokens_pytorch, n_pytorch = (
        fused_cluster_union_gather_kv_pytorch(**test_data)
    )

    print(f"PyTorch: num_tokens={n_pytorch}, k.shape={k_pytorch.shape}")

    if not TRITON_AVAILABLE:
        print("Triton not available, skipping Triton comparison")
        return

    # Run Triton version
    k_triton, v_triton, tokens_triton, n_triton = fused_cluster_union_gather_kv(
        **test_data,
        use_triton=True,
    )

    print(f"Triton: num_tokens={n_triton}, k.shape={k_triton.shape}")

    # Compare
    assert n_pytorch == n_triton, f"Token count mismatch: {n_pytorch} vs {n_triton}"
    assert torch.equal(tokens_pytorch, tokens_triton), "Token IDs mismatch"
    assert torch.allclose(
        k_pytorch, k_triton, rtol=1e-3, atol=1e-3
    ), "K values mismatch"
    assert torch.allclose(
        v_pytorch, v_triton, rtol=1e-3, atol=1e-3
    ), "V values mismatch"

    print("✓ Correctness test PASSED")
    print()


def benchmark(num_runs=100, warmup=10):
    """Benchmark Triton vs PyTorch."""
    print("=" * 60)
    print("Benchmarking Performance")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    if not TRITON_AVAILABLE:
        print("Triton not available, skipping benchmark")
        return

    device = "cuda"
    test_data = create_test_data(device=device)

    # Warmup
    for _ in range(warmup):
        fused_cluster_union_gather_kv_pytorch(**test_data)
        fused_cluster_union_gather_kv(**test_data, use_triton=True)
    torch.cuda.synchronize()

    # Benchmark PyTorch
    times_pytorch = []
    for _ in range(num_runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        fused_cluster_union_gather_kv_pytorch(**test_data)
        end.record()

        torch.cuda.synchronize()
        times_pytorch.append(start.elapsed_time(end))

    # Benchmark Triton
    times_triton = []
    for _ in range(num_runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        fused_cluster_union_gather_kv(**test_data, use_triton=True)
        end.record()

        torch.cuda.synchronize()
        times_triton.append(start.elapsed_time(end))

    # Report
    pytorch_mean = np.mean(times_pytorch)
    pytorch_std = np.std(times_pytorch)
    triton_mean = np.mean(times_triton)
    triton_std = np.std(times_triton)
    speedup = pytorch_mean / triton_mean

    print(f"PyTorch: {pytorch_mean:.3f} ± {pytorch_std:.3f} ms")
    print(f"Triton:  {triton_mean:.3f} ± {triton_std:.3f} ms")
    print(f"Speedup: {speedup:.2f}x")
    print()


def test_edge_cases():
    """Test edge cases."""
    print("=" * 60)
    print("Testing Edge Cases")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    device = "cuda"

    # Test 1: Empty selection
    print("Test 1: Empty selection...")
    test_data = create_test_data(device=device)
    test_data["num_selected_batch"].fill_(0)

    k, v, tokens, n = fused_cluster_union_gather_kv_pytorch(**test_data)
    assert (
        n == 0 or n == test_data["local_window_count"]
    ), "Should only have local window tokens"
    print("✓ Passed")

    # Test 2: Single cluster
    print("Test 2: Single cluster...")
    test_data = create_test_data(num_clusters=1, device=device)
    k_pytorch, v_pytorch, tokens_pytorch, n_pytorch = (
        fused_cluster_union_gather_kv_pytorch(**test_data)
    )
    if TRITON_AVAILABLE:
        k_triton, v_triton, tokens_triton, n_triton = fused_cluster_union_gather_kv(
            **test_data, use_triton=True
        )
        assert n_pytorch == n_triton
        assert torch.equal(tokens_pytorch, tokens_triton)
    print("✓ Passed")

    # Test 3: Large head_dim
    print("Test 3: Large head_dim (256)...")
    test_data = create_test_data(head_dim=256, device=device)
    k_pytorch, v_pytorch, tokens_pytorch, n_pytorch = (
        fused_cluster_union_gather_kv_pytorch(**test_data)
    )
    if TRITON_AVAILABLE:
        k_triton, v_triton, tokens_triton, n_triton = fused_cluster_union_gather_kv(
            **test_data, use_triton=True
        )
        assert n_pytorch == n_triton
        assert torch.allclose(k_pytorch, k_triton, rtol=1e-3, atol=1e-3)
    print("✓ Passed")

    # Test 4: No local window
    print("Test 4: No local window...")
    test_data = create_test_data(device=device)
    test_data["local_window_count"] = 0
    k_pytorch, v_pytorch, tokens_pytorch, n_pytorch = (
        fused_cluster_union_gather_kv_pytorch(**test_data)
    )
    if TRITON_AVAILABLE:
        k_triton, v_triton, tokens_triton, n_triton = fused_cluster_union_gather_kv(
            **test_data, use_triton=True
        )
        assert n_pytorch == n_triton
    print("✓ Passed")

    print("✓ All edge case tests PASSED")
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Fused Cluster Gather - Test Suite")
    print("=" * 60 + "\n")

    # Test availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Triton available: {TRITON_AVAILABLE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Run tests
    test_correctness()
    test_edge_cases()
    benchmark()

    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
