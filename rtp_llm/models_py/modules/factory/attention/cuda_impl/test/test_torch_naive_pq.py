"""Unit tests for TorchNaivePQPrefillImpl 与 TorchNaivePQDecodeImpl.

Prefill: bs=1, qkv_len=65536, q_head_num=16, kv_head_num=1, head_dim=256
Decode:  bs=16, kv_len=32768, q_len=1, q_head_num=16, kv_head_num=1, head_dim=256
两边均使用默认 PQ 配置 (num_subspaces=16, num_clusters=256, kmeans_iters=20).

为避免拉起 RoPE / cache_store 等 C++ 依赖，
用子类绕过父 __init__ 并直接 mock _read_kv_from_cache / _compute_kv_seq_lens.
"""

import os
import unittest
from types import SimpleNamespace
from typing import List, Optional

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_impl.torch_naive_pq import (
    _PQ_CACHE,
    TorchNaivePQDecodeImpl,
    TorchNaivePQPrefillImpl,
    _pq_key,
)

# ============================================================================
# Test doubles：bypass parent __init__ 中的 RoPE / WriteCacheStore 构造
# ============================================================================


class _TestablePQPrefill(TorchNaivePQPrefillImpl):
    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        attn_inputs,
    ) -> None:
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scaling = 1.0 / (head_dim**0.5)
        self.enable_gqa = num_heads != num_kv_heads
        self.attn_inputs = attn_inputs

        self.need_rope_kv_cache = False
        self.rope_kvcache_impl = None
        self.write_cache_store_impl = None
        self.rope_params = None
        self.fmha_params = None

        # 与 TorchNaivePQPrefillImpl 默认配置保持一致
        self.num_subspaces = 16
        self.num_clusters = 256
        self.kmeans_iters = 20
        assert head_dim % self.num_subspaces == 0


class _TestablePQDecode(TorchNaivePQDecodeImpl):
    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        tokens_per_block: int,
        attn_inputs,
        fake_seq_lens: torch.Tensor,
        fake_k_full: torch.Tensor,
        fake_v_full: torch.Tensor,
    ) -> None:
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scaling = 1.0 / (head_dim**0.5)
        self.enable_gqa = num_heads != num_kv_heads
        self.tokens_per_block = tokens_per_block
        self.attn_inputs = attn_inputs

        self.need_rope_kv_cache = False
        self.rope_kvcache_impl = None
        self.write_cache_store_impl = None
        self.rope_params = None
        self.fmha_params = None

        # 与 TorchNaivePQDecodeImpl 默认配置保持一致（top_k_tokens=2000）
        self.num_subspaces = int(os.getenv("PQ_NUM_SUBSPACES", "16"))
        self.top_k_tokens = int(os.getenv("PQ_TOP_K_TOKENS", "2000"))

        # 测试桩：避免触碰 fill_mla_params / paged KV 读取
        self._fake_seq_lens = fake_seq_lens
        self._fake_k_full = fake_k_full
        self._fake_v_full = fake_v_full

    def _compute_kv_seq_lens(self, batch_size: int) -> torch.Tensor:
        return self._fake_seq_lens[:batch_size]

    def _read_kv_from_cache(self, kv_cache):
        return self._fake_k_full, self._fake_v_full


def _make_attn_inputs(
    input_lengths: List[int], device: torch.device
) -> SimpleNamespace:
    """构造最小化的 attn_inputs；cache_store_inputs=None 让 write 路径短路。"""
    input_lens = torch.tensor(input_lengths, dtype=torch.int32, device="cpu")
    cu = torch.zeros(len(input_lengths) + 1, dtype=torch.int32, device=device)
    cu[1:] = input_lens.to(device).cumsum(0).to(torch.int32)
    return SimpleNamespace(
        input_lengths=input_lens,
        cu_seqlens=cu,
        cache_store_inputs=None,
        is_prefill=True,
    )


# ============================================================================
# Tests
# ============================================================================


class TestTorchNaivePQPrefillImpl(unittest.TestCase):
    """Prefill: bs=1, qkv_len=65536, q_head=16, kv_head=1, head_dim=256."""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        self.device = torch.device("cuda")
        torch.manual_seed(42)
        _PQ_CACHE.clear()

    def tearDown(self) -> None:
        _PQ_CACHE.clear()

    def test_prefill_forward(self) -> None:
        bs = 1
        qkv_len = 65536
        q_head_num = 16
        kv_head_num = 1
        head_dim = 256
        dtype = torch.bfloat16
        layer_id = 0

        attn_inputs = _make_attn_inputs([qkv_len] * bs, self.device)
        impl = _TestablePQPrefill(q_head_num, kv_head_num, head_dim, attn_inputs)

        total = bs * qkv_len
        qkv_dim = (q_head_num + 2 * kv_head_num) * head_dim
        qkv = torch.randn(total, qkv_dim, device=self.device, dtype=dtype) * 0.1

        # KVCache 仅需提供 layer_id（_perform_pq_clustering 用它来生成 cache key）
        kv_cache = SimpleNamespace(layer_id=layer_id)

        # Warmup 一次（让 Triton kmeans / flash_attn 的首次 JIT 不计入 nsys）
        impl.forward(qkv, kv_cache)
        _PQ_CACHE.clear()
        torch.cuda.synchronize()

        # 用 cudaProfilerApi 限定 capture 区间 + NVTX 标记 forward
        import torch.cuda.profiler as cuprof

        cuprof.start()
        torch.cuda.nvtx.range_push("pq_prefill_forward")
        output = impl.forward(qkv, kv_cache)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        cuprof.stop()

        # 输出形状 / dtype / 数值健康度
        self.assertEqual(output.shape, (total, q_head_num * head_dim))
        self.assertEqual(output.dtype, dtype)
        self.assertFalse(torch.isnan(output).any().item())
        self.assertFalse(torch.isinf(output).any().item())

        # _PQ_CACHE 已为每个 (seq, kv_head) 写入聚类结果
        sub_dim = head_dim // impl.num_subspaces
        for seq_idx in range(bs):
            for h in range(kv_head_num):
                key = _pq_key(layer_id, seq_idx, h)
                self.assertIn(key, _PQ_CACHE)
                entry = _PQ_CACHE[key]
                self.assertEqual(entry["prefill_len"], qkv_len)
                self.assertEqual(entry["cids"].shape, (impl.num_subspaces, qkv_len))
                self.assertEqual(
                    entry["cents"].shape,
                    (impl.num_subspaces, impl.num_clusters, sub_dim),
                )
                # cids 取值在 [0, num_clusters)
                self.assertGreaterEqual(int(entry["cids"].min().item()), 0)
                self.assertLess(int(entry["cids"].max().item()), impl.num_clusters)


class TestTorchNaivePQDecodeImpl(unittest.TestCase):
    """Decode: bs=16, kv_len=32768, q_len=1, q_head=16, kv_head=1, head_dim=256."""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        self.device = torch.device("cuda")
        torch.manual_seed(42)
        _PQ_CACHE.clear()

    def tearDown(self) -> None:
        _PQ_CACHE.clear()

    def _populate_pq_cache(
        self,
        bs: int,
        kv_head_num: int,
        prefill_len: int,
        head_dim: int,
        num_subspaces: int,
        num_clusters: int,
        layer_id: int,
        dtype: torch.dtype,
    ) -> None:
        """模拟 prefill 已经跑过 → 为每个 (seq, kv_head) 写入聚类条目。

        cents dtype 跟随 K（bf16），与 batch_kmeans_Euclid 在 bf16 输入下的输出一致。
        """
        sub_dim = head_dim // num_subspaces
        for b in range(bs):
            for h in range(kv_head_num):
                key = _pq_key(layer_id, b, h)
                _PQ_CACHE[key] = {
                    "cids": torch.randint(
                        0,
                        num_clusters,
                        (num_subspaces, prefill_len),
                        dtype=torch.int64,
                        device=self.device,
                    ),
                    "cents": torch.randn(
                        num_subspaces,
                        num_clusters,
                        sub_dim,
                        dtype=dtype,
                        device=self.device,
                    )
                    * 0.1,
                    "prefill_len": prefill_len,
                }

    def test_decode_forward(self) -> None:
        bs = 16
        kv_len = 32768
        q_len = 1
        q_head_num = 16
        kv_head_num = 1
        head_dim = 256
        tokens_per_block = 64
        dtype = torch.bfloat16
        layer_id = 0

        # 默认配置（与类内默认一致）
        S = 16
        K = 256
        top_k = 2000

        # decode 时 input_lengths 每条都是 q_len=1
        attn_inputs = _make_attn_inputs([q_len] * bs, self.device)
        attn_inputs.is_prefill = False

        # 预灌 PQ cache：相当于 prefill 已跑完 prefill_len = kv_len 的上下文
        self._populate_pq_cache(
            bs=bs,
            kv_head_num=kv_head_num,
            prefill_len=kv_len,
            head_dim=head_dim,
            num_subspaces=S,
            num_clusters=K,
            layer_id=layer_id,
            dtype=dtype,
        )

        # 提供 padded K/V（_read_kv_from_cache 被桩替换）
        fake_seq_lens = torch.tensor([kv_len] * bs, dtype=torch.int32, device="cpu")
        k_full = (
            torch.randn(
                bs, kv_len, kv_head_num, head_dim, device=self.device, dtype=dtype
            )
            * 0.1
        )
        v_full = (
            torch.randn(
                bs, kv_len, kv_head_num, head_dim, device=self.device, dtype=dtype
            )
            * 0.1
        )

        impl = _TestablePQDecode(
            q_head_num,
            kv_head_num,
            head_dim,
            tokens_per_block,
            attn_inputs,
            fake_seq_lens,
            k_full,
            v_full,
        )

        # decode 输入 qkv: [bs, (q_h + 2*kv_h) * head_dim]
        qkv_dim = (q_head_num + 2 * kv_head_num) * head_dim
        qkv = torch.randn(bs, qkv_dim, device=self.device, dtype=dtype) * 0.1

        kv_cache = SimpleNamespace(layer_id=layer_id)

        # 多次 warmup，让 Triton / flash_attn 的 JIT / autotune 稳定下来
        n_warmup = 10
        n_iters = 30
        for _ in range(n_warmup):
            impl.forward(qkv, kv_cache)
        torch.cuda.synchronize()

        # 用 cudaProfilerApi 限定 capture 区间；每次 forward 一对 NVTX range，
        # nsys stats 会给出 mean/median/std，靠中位数判断稳定性能。
        import torch.cuda.profiler as cuprof

        cuprof.start()
        for _ in range(n_iters):
            torch.cuda.nvtx.range_push("pq_decode_forward")
            output = impl.forward(qkv, kv_cache)
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
        cuprof.stop()

        # 形状 / dtype / 数值健康度
        self.assertEqual(output.shape, (bs, q_head_num * head_dim))
        self.assertEqual(output.dtype, dtype)
        self.assertFalse(torch.isnan(output).any().item())
        self.assertFalse(torch.isinf(output).any().item())

        # 配置侧 sanity
        self.assertEqual(impl.num_subspaces, S)
        self.assertEqual(impl.top_k_tokens, top_k)


if __name__ == "__main__":
    unittest.main()
