"""Optimized kernels for clustered attention.

Note: Originally planned as Triton kernels, but due to Triton's limitations
(no break statements, fixed power-of-2 array sizes), we use an optimized
batched PyTorch implementation instead, which still provides speedup by:
- Batching centroid scoring across all heads
- Eliminating Python loop overhead for scoring
- Using efficient PyTorch operations

This provides similar benefits to a Triton kernel for this use case.
"""

try:
    from .clustered_attention import (
        fused_centroid_scoring_topp,
        fused_centroid_scoring_topp_vectorized,
    )

    __all__ = [
        "fused_centroid_scoring_topp",
        "fused_centroid_scoring_topp_vectorized",
    ]
except ImportError as e:
    # Graceful degradation if dependencies not available
    import logging

    logging.warning(
        f"Failed to import optimized kernels: {e}. Optimization will not be available."
    )
    __all__ = []
