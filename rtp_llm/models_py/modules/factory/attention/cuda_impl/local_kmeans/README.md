# Local K-Means Implementation

This package provides PyTorch native implementations of K-Means clustering algorithms for RTP-LLM's attention clustering feature, replacing the external `flash-kmeans` dependency.

## Overview

The implementation consists of three main components:

1. **`kmeans_impl.py`** - Batched K-Means clustering with Euclidean distance
2. **`incremental_kmeans.py`** - Online/incremental K-Means for efficient updates
3. **`__init__.py`** - Package interface exposing `batch_kmeans_Euclid` and `IncrementalKMeans`

## Design Decisions

### Why Pure PyTorch?

The original `flash-kmeans` library uses Triton kernels for optimal GPU performance. We replaced these with pure PyTorch operations for:

- **Dependency Simplification**: No external package dependencies
- **Maintainability**: Standard PyTorch operations are easier to debug and modify
- **Compatibility**: Works with any PyTorch version without Triton compilation

### Performance Trade-offs

- **Expected Overhead**: 1.5-2x slower than Triton implementation
- **Acceptable for Use Case**: K-Means is a small fraction of total inference time
- **Optimization**: Uses `torch.cdist` and `scatter_add` for vectorized operations

### BFloat16 Handling

PyTorch's `torch.cdist` doesn't support BFloat16 on CUDA, so we:
- Convert to float32 for distance computation
- Keep centroids in original dtype for storage
- Minimal overhead as distance computation is memory-bound

## API Compatibility

The implementation maintains 100% API compatibility with `flash-kmeans`:

### batch_kmeans_Euclid

```python
from rtp_llm.models_py.modules.factory.attention.cuda_impl.local_kmeans import batch_kmeans_Euclid

# Input: [B, N, D] tensor
cluster_ids, centroids, n_iters = batch_kmeans_Euclid(
    x,                        # [B, N, D]
    n_clusters=10,
    max_iters=20,
    tol=1e-4,
    init_centroids=None,      # Optional [B, K, D]
    verbose=False
)

# Returns:
# - cluster_ids: [B, N] int64 - cluster assignment per point
# - centroids: [B, K, D] - final cluster centers
# - n_iters: int - number of iterations performed
```

### IncrementalKMeans

```python
from rtp_llm.models_py.modules.factory.attention.cuda_impl.local_kmeans import IncrementalKMeans

# Initialize
model = IncrementalKMeans(
    n_clusters=100,
    dim=128,
    device='cuda',
    dtype=torch.bfloat16
)

# Set initial centroids
model.init_centroids(centroids)  # [K, D]

# Add points incrementally
labels = model.add_points(
    new_points,              # [N, D] or [B, N, D]
    update_centroids=True    # Update centroids immediately
)

# Query state
centroids = model.get_centroids()           # [K, D]
sums, counts = model.get_statistics()       # [K, D], [K]
```

## Usage in torch_naive.py

The implementation is used in two places:

### 1. Prefill Phase - Initial Clustering

```python
# Line 92 in torch_naive.py
cluster_ids, centroids, n_iters = batch_kmeans_Euclid(
    k_batched,           # [1, seq_len, head_dim]
    num_clusters,
    max_iters=max_iters,
    tol=1e-4,
)
```

### 2. Decode Phase - Incremental Updates

```python
# Line 898-907 in torch_naive.py
model = IncrementalKMeans(
    n_clusters=num_clusters,
    dim=self.head_dim,
    device=k_head.device,
    dtype=k_head.dtype,
)
model.init_centroids(centroids)
model.add_points(k_head, update_centroids=False)

# Line 1035 in decode
label = model.add_points(k_single, update_centroids=True)
```

## Testing

Run the test suite:

```bash
cd rtp_llm/models_py/modules/factory/attention/cuda_impl/local_kmeans
PYTHONPATH=/path/to/RTP-LLM/github-opensource:$PYTHONPATH python test_local_kmeans.py
```

Test coverage:
- ✅ Basic batched K-Means
- ✅ Single batch (typical use case)
- ✅ Incremental updates
- ✅ Single point additions (decode scenario)
- ✅ End-to-end compatibility with torch_naive.py
- ✅ Error handling

## Implementation Details

### Batched K-Means Algorithm

1. **Initialization**: Randomly select K points or use provided centroids
2. **Assignment**: Compute distances `torch.cdist(x, centroids)` and assign to nearest
3. **Update**: Use `scatter_add` to accumulate point sums and counts per cluster
4. **Convergence**: Check if centroid movement < tolerance

### Incremental K-Means Algorithm

Maintains running statistics:
- `centroid_sums`: [K, D] - accumulated sum of points per cluster
- `centroid_counts`: [K] - number of points per cluster
- `centroids`: [K, D] - current cluster centers

When adding new points:
1. Assign to nearest existing centroid
2. Update statistics with `scatter_add`
3. Recompute centroids: `centroids = sums / counts`

## Performance Characteristics

Measured on standalone_decode_test.py (32K sequence, 500 clusters, 16 heads):

- **Centroid Scoring (batched)**: 0.079 ms
- **Update Clustering**: 0.127 ms per decode step
- **Total overhead**: < 0.5ms per decode iteration

These numbers are acceptable as they represent < 5% of total decode time.

## Future Improvements

Potential optimizations if performance becomes critical:

1. **JIT Compilation**: Use `torch.jit.script` for inner loops
2. **Fused Operations**: Custom CUDA kernels for assign + update
3. **Quantized Distance**: Use int8 distances for faster computation
4. **Adaptive Clustering**: Only update centroids every N steps

For now, the pure PyTorch implementation provides good balance of simplicity and performance.

## License

This implementation is derived from the flash-kmeans project and maintains compatibility
with its API while using pure PyTorch operations.

## Acknowledgments

- Original flash-kmeans library for the algorithm design
- SGLang project for the attention clustering approach
- RTP-LLM team for integration and testing
