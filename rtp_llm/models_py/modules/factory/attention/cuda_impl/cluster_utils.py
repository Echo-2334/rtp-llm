"""Utility functions for clustered attention optimization."""

from typing import Dict, List, Tuple

import torch


def convert_cluster_indices_to_csr(
    cluster_indices: List[List[int]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert cluster_indices from list-of-lists to CSR (Compressed Sparse Row) format.

    Args:
        cluster_indices: List of lists, where cluster_indices[i] contains token indices for cluster i
        device: Target device for tensors

    Returns:
        flat_indices: Flattened tensor containing all token indices [total_tokens]
        offsets: CSR offset array [num_clusters + 1], where offsets[i]:offsets[i+1] gives range for cluster i
    """
    num_clusters = len(cluster_indices)

    # Build offsets and flatten indices
    offsets = [0]
    flat_indices_list = []

    for cluster_tokens in cluster_indices:
        flat_indices_list.extend(cluster_tokens)
        offsets.append(offsets[-1] + len(cluster_tokens))

    # Convert to tensors
    flat_indices = torch.tensor(flat_indices_list, dtype=torch.int32, device=device)
    offsets_tensor = torch.tensor(offsets, dtype=torch.int32, device=device)

    return flat_indices, offsets_tensor


def gather_tokens_from_clusters_csr(
    selected_cluster_ids: torch.Tensor,  # [num_selected_clusters]
    flat_indices: torch.Tensor,  # [total_tokens]
    offsets: torch.Tensor,  # [num_clusters + 1]
) -> torch.Tensor:
    """Gather token indices from selected clusters using CSR format.

    Args:
        selected_cluster_ids: Tensor of selected cluster IDs [num_selected_clusters]
        flat_indices: Flattened token indices from all clusters [total_tokens]
        offsets: CSR offsets [num_clusters + 1]

    Returns:
        selected_tokens: Tensor of unique token indices (sorted)
    """
    if selected_cluster_ids.numel() == 0:
        return torch.tensor([], dtype=torch.int32, device=flat_indices.device)

    # Get start and end indices for each selected cluster
    starts = offsets[selected_cluster_ids]  # [num_selected_clusters]
    ends = offsets[selected_cluster_ids + 1]  # [num_selected_clusters]

    # Gather tokens for each cluster (still need small loop, but GPU operations)
    all_tokens = []
    for i in range(selected_cluster_ids.numel()):
        cluster_tokens = flat_indices[starts[i] : ends[i]]
        all_tokens.append(cluster_tokens)

    # Concatenate and get unique tokens
    if len(all_tokens) > 0:
        all_tokens_cat = torch.cat(all_tokens)
        # Use torch.unique to get sorted unique token indices
        unique_tokens = torch.unique(all_tokens_cat)
        return unique_tokens
    else:
        return torch.tensor([], dtype=torch.int32, device=flat_indices.device)


def gather_tokens_from_clusters_batch_csr(
    selected_cluster_ids_batch: torch.Tensor,  # [num_heads, max_selected]
    num_selected_batch: torch.Tensor,  # [num_heads]
    flat_indices: torch.Tensor,  # [total_tokens]
    offsets: torch.Tensor,  # [num_clusters + 1]
) -> Tuple[List[torch.Tensor], torch.Tensor, Dict[int, int]]:
    """Gather tokens from multiple heads and compute union set using CSR format.

    Args:
        selected_cluster_ids_batch: Selected cluster IDs for each head [num_heads, max_selected]
        num_selected_batch: Number of selected clusters per head [num_heads]
        flat_indices: Flattened token indices [total_tokens]
        offsets: CSR offsets [num_clusters + 1]

    Returns:
        all_selected_tokens: List of tensors, one per head containing selected token IDs
        union_token_list: Sorted tensor of unique tokens across all heads
        token_to_idx: Dict mapping token ID to index in union_token_list
    """
    num_heads = selected_cluster_ids_batch.shape[0]
    all_selected_tokens = []

    # Gather tokens for each head
    for head_idx in range(num_heads):
        n_selected = num_selected_batch[head_idx].item()
        selected_cluster_ids = selected_cluster_ids_batch[head_idx, :n_selected]

        # Gather tokens for this head using CSR
        head_tokens = gather_tokens_from_clusters_csr(
            selected_cluster_ids, flat_indices, offsets
        )
        all_selected_tokens.append(head_tokens)

    # Compute union using torch.unique on concatenated tensor
    if len(all_selected_tokens) > 0:
        all_tokens_cat = torch.cat(all_selected_tokens)
        union_token_list = torch.unique(all_tokens_cat)  # Automatically sorted
    else:
        union_token_list = torch.tensor(
            [], dtype=torch.int32, device=flat_indices.device
        )

    # Build token_to_idx mapping (on CPU for dict creation)
    union_list_cpu = union_token_list.cpu().tolist()
    token_to_idx = {token: idx for idx, token in enumerate(union_list_cpu)}

    return all_selected_tokens, union_token_list, token_to_idx


def build_attention_mask_vectorized(
    all_selected_tokens: List[
        torch.Tensor
    ],  # List of [num_selected_i] tensors per head
    union_token_list: torch.Tensor,  # [num_union_tokens]
    device: torch.device,
) -> torch.Tensor:
    """Build attention mask using vectorized operations.

    Args:
        all_selected_tokens: List of tensors, one per head with selected token IDs
        union_token_list: Sorted tensor of unique tokens
        device: Target device

    Returns:
        attn_mask: Boolean mask [num_heads, num_union_tokens]
    """
    num_heads = len(all_selected_tokens)
    num_union_tokens = union_token_list.shape[0]

    # Initialize mask as False
    attn_mask = torch.zeros(
        num_heads, num_union_tokens, dtype=torch.bool, device=device
    )

    # For each head, mark which union tokens are selected
    for head_idx in range(num_heads):
        head_tokens = all_selected_tokens[head_idx]  # [num_selected_i]

        if head_tokens.numel() == 0:
            continue

        # Find indices of head_tokens in union_token_list
        # Use searchsorted since union_token_list is sorted
        indices = torch.searchsorted(union_token_list, head_tokens)

        # Set mask to True for these indices
        attn_mask[head_idx, indices] = True

    return attn_mask


def precompute_csr_cache(cluster_info: Dict) -> Dict:
    """Precompute CSR representation for a cluster_info dictionary.

    Modifies cluster_info in-place to add 'flat_indices' and 'offsets' keys.

    Args:
        cluster_info: Dictionary containing 'cluster_indices' key

    Returns:
        cluster_info: Same dictionary with added CSR fields
    """
    if "flat_indices" in cluster_info and "offsets" in cluster_info:
        # Already precomputed
        return cluster_info

    cluster_indices = cluster_info["cluster_indices"]

    # Get device from centroids or model
    if "centroids" in cluster_info:
        device = cluster_info["centroids"].device
    elif "model" in cluster_info:
        device = cluster_info["model"].get_centroids().device
    else:
        device = torch.device("cuda")  # Default to CUDA

    # Convert to CSR
    flat_indices, offsets = convert_cluster_indices_to_csr(cluster_indices, device)

    # Store in cluster_info
    cluster_info["flat_indices"] = flat_indices
    cluster_info["offsets"] = offsets

    return cluster_info
