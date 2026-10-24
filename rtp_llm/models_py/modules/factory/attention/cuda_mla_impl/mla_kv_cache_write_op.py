"""KV cache write operation for Multi-Latent Attention (MLA).

This module provides the KV cache writing operation specifically for MLA architecture,
which uses a compressed KV cache layout.
"""

from typing import Optional

import torch
from rtp_llm.models_py.triton_kernels.sparse_mla.nvfp4_cache import write_nvfp4_cache
from rtp_llm.models_py.triton_kernels.sparse_mla.nvfp4_sim import (
    SimConfig,
    write_simulated,
)
from rtp_llm.ops import KvCacheDataType, compute_ops
from rtp_llm.ops.compute_ops import LayerKVCache, rtp_llm_ops


class MlaKVCacheWriteOp:
    """Write compressed KV cache for Multi-Latent Attention."""

    def __init__(
        self,
        kv_cache_dtype: KvCacheDataType,
        defer_full_blocks: bool = False,
        tail_capacity: int = 256,
        prefill_role: bool = False,
        is_prefill: bool = True,
    ) -> None:
        self.tail_capacity = tail_capacity
        self.defer_full_blocks = defer_full_blocks
        self._deferred_states = {}
        self.is_nvfp4 = kv_cache_dtype == KvCacheDataType.NVFP4
        self.kv_cache_type = (
            "fp8_ds_mla" if kv_cache_dtype == KvCacheDataType.FP8 else "auto"
        )
        # Scale tensor is required for concat_and_cache_mla even in non-FP8 mode
        self.scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
        # RTP_MLA_NVFP4_SIM simulates NVFP4 error on an ordinary BF16/FP8 cache.
        self.is_prefill = is_prefill
        self.sim = (
            SimConfig.from_env(prefill_role)
            if kv_cache_dtype in (KvCacheDataType.BASE, KvCacheDataType.FP8)
            else None
        )

    def _write(self, latent, rope, cache, slots) -> None:
        compute_ops.concat_and_cache_mla(
            latent, rope, cache, slots, self.kv_cache_type, self.scale
        )

    def forward(
        self,
        append_ckv_t: torch.Tensor,
        key_pe: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        fmha_params: rtp_llm_ops.SparseMlaParams,
        total_global_ids: torch.Tensor = None,
        slot_mapping_override: torch.Tensor = None,
    ) -> None:
        """Write compressed KV and position-encoded key to MLA cache.

        Args:
            append_ckv_t: Compressed KV tensor to append [num_tokens, kv_lora_rank]
            key_pe: Position-encoded key tensor [num_tokens, rope_head_dim]
            kv_cache: MLA KV cache with compressed layout
        """
        if kv_cache is not None:
            if slot_mapping_override is not None:
                slot_mapping = slot_mapping_override
            elif total_global_ids is None:
                slot_mapping = fmha_params.slot_mapping
            else:
                slot_mapping = fmha_params.slot_mapping[total_global_ids]
            if self.is_nvfp4:
                state = None
                if self.defer_full_blocks:
                    from rtp_llm.models_py.triton_kernels.sparse_mla.nvfp4_deferred import (
                        DeferredNvfp4State,
                        find_state,
                        poll_overflow,
                    )

                    base = kv_cache.kv_cache_base
                    state = find_state(base)
                    if state is None:
                        state = DeferredNvfp4State(base, self.tail_capacity)
                    self._deferred_states[base.data_ptr()] = state
                write_nvfp4_cache(
                    append_ckv_t,
                    key_pe,
                    kv_cache.kv_cache_base,
                    slot_mapping,
                    deferred_state=state,
                )
                if state is not None:
                    poll_overflow()
                return
            if self.sim is not None:
                write_simulated(
                    self.sim,
                    append_ckv_t,
                    key_pe,
                    kv_cache.kv_cache_base,
                    slot_mapping,
                    self._write,
                    prefill=self.is_prefill,
                )
                return
            compute_ops.concat_and_cache_mla(
                append_ckv_t,
                key_pe,
                kv_cache.kv_cache_base,
                slot_mapping,
                self.kv_cache_type,
                self.scale,
            )
