from unittest import SkipTest, TestCase, main

import torch

from rtp_llm.models_py.triton_kernels.sparse_mla.sparse_indexer_pool_update import (
    append_global_pool_from_pool_chunk_topk,
    compact_append_pool_if_full,
    initialize_global_pool_inverse_map,
)
from rtp_llm.models_py.triton_kernels.sparse_mla.sparse_indexer_score import (
    sparse_fp8_mqa_logits,
    sparse_fp8_mqa_pool_chunk_logits,
)


class SparseIndexerPoolKernelsTest(TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is required")
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        torch.manual_seed(20260721)

    def test_pool_chunk_score_matches_explicit_candidates_in_graph(self):
        batch = 2
        seq_len = 16 * 1024
        page_size = 64
        heads = 32
        head_dim = 128
        entry_bytes = head_dim + 4
        blocks_per_row = seq_len // page_size
        total_blocks = batch * blocks_per_row

        q = (
            torch.randn(batch, heads, head_dim, device=self.device)
            .clamp_(-3, 3)
            .to(torch.float8_e4m3fn)
        )
        weights = torch.rand(batch, heads, device=self.device)
        raw_cache = torch.empty(
            total_blocks,
            page_size * entry_bytes,
            device=self.device,
            dtype=torch.uint8,
        )
        raw_cache[:, : page_size * head_dim].copy_(
            torch.randint(
                0,
                120,
                (total_blocks, page_size * head_dim),
                device=self.device,
                dtype=torch.uint8,
            )
        )
        scales = torch.ones(
            total_blocks, page_size, device=self.device, dtype=torch.float32
        )
        raw_cache[:, page_size * head_dim :].copy_(
            scales.view(torch.uint8).reshape(total_blocks, -1)
        )
        kv_cache = raw_cache.view(
            total_blocks, page_size, 1, entry_bytes
        )
        block_table = torch.arange(
            total_blocks, device=self.device, dtype=torch.int32
        ).view(batch, blocks_per_row)

        pool = torch.zeros(
            batch, 16 * 1024, device=self.device, dtype=torch.int32
        )
        pool[0, :4096] = torch.arange(
            0, 4096, device=self.device, dtype=torch.int32
        )
        pool[1, :6144] = torch.arange(
            8192, 14336, device=self.device, dtype=torch.int32
        )
        pool_lengths = torch.tensor(
            [4096, 6144], device=self.device, dtype=torch.int32
        )
        chunk = torch.stack(
            (
                torch.arange(4096, 8192, device=self.device, dtype=torch.int32),
                torch.arange(0, 4096, device=self.device, dtype=torch.int32),
            )
        )
        chunk_lengths = torch.full(
            (batch,), 4096, device=self.device, dtype=torch.int32
        )

        packed = torch.zeros_like(pool)
        for row, pool_length in enumerate(pool_lengths.tolist()):
            packed[row, :pool_length] = pool[row, :pool_length]
            packed[row, pool_length : pool_length + 4096] = chunk[row]
        combined_lengths = pool_lengths + chunk_lengths
        expected = sparse_fp8_mqa_logits(
            q,
            weights,
            kv_cache,
            block_table,
            packed,
            combined_lengths,
        )

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = sparse_fp8_mqa_pool_chunk_logits(
                q,
                weights,
                kv_cache,
                block_table,
                pool,
                pool_lengths,
                chunk,
                chunk_lengths,
            )
        graph.replay()
        torch.cuda.synchronize()

        global_pool = torch.zeros(
            3, 16 * 1024, device=self.device, dtype=torch.int32
        )
        global_pool[2].copy_(pool[0])
        global_pool[0].copy_(pool[1])
        global_lengths = torch.tensor(
            [6144, 0, 4096], device=self.device, dtype=torch.int32
        )
        pool_slots = torch.tensor(
            [2, 0], device=self.device, dtype=torch.int32
        )
        slotted_actual = sparse_fp8_mqa_pool_chunk_logits(
            q,
            weights,
            kv_cache,
            block_table,
            global_pool,
            global_lengths,
            chunk,
            chunk_lengths,
            pool_slots=pool_slots,
        )
        torch.cuda.synchronize()

        for row, combined_length in enumerate(combined_lengths.tolist()):
            torch.testing.assert_close(
                actual[row, :combined_length],
                expected[row, :combined_length],
                rtol=0,
                atol=0,
            )
            self.assertTrue(torch.isneginf(actual[row, combined_length:]).all())
            torch.testing.assert_close(
                slotted_actual[row], actual[row], rtol=0, atol=0
            )

    def test_append_and_compact_preserve_inverse_map(self):
        seq_len = 32 * 1024
        pool = torch.zeros(
            1, 16 * 1024, device=self.device, dtype=torch.int32
        )
        pool[0, :8192] = torch.arange(
            8192, device=self.device, dtype=torch.int32
        )
        pool_lengths = torch.tensor(
            [8192], device=self.device, dtype=torch.int32
        )
        inverse_map = initialize_global_pool_inverse_map(
            pool, seq_len, pool_lengths
        )
        chunk = torch.arange(
            8192, 16 * 1024, device=self.device, dtype=torch.int32
        ).view(1, -1)
        chunk_lengths = torch.tensor(
            [8192], device=self.device, dtype=torch.int32
        )
        local_topk = (
            pool_lengths[:, None]
            + torch.arange(2048, device=self.device, dtype=torch.int32)
        ).contiguous()

        selected = append_global_pool_from_pool_chunk_topk(
            local_topk,
            pool,
            pool_lengths,
            chunk,
            chunk_lengths,
            inverse_map,
        )
        torch.cuda.synchronize()
        self.assertEqual(pool_lengths.item(), 10 * 1024)
        torch.testing.assert_close(selected, chunk[:, :2048])
        torch.testing.assert_close(pool[0, 8192 : 10 * 1024], chunk[0, :2048])
        torch.testing.assert_close(
            inverse_map[0, chunk[0, :2048]],
            torch.arange(8193, 10 * 1024 + 1, device=self.device, dtype=torch.int32),
        )

        full_pool = torch.arange(
            16 * 1024, device=self.device, dtype=torch.int32
        ).view(1, -1)
        full_lengths = torch.tensor(
            [16 * 1024], device=self.device, dtype=torch.int32
        )
        full_inverse = initialize_global_pool_inverse_map(
            full_pool, seq_len, full_lengths
        )
        compact_append_pool_if_full(full_pool, full_lengths, full_inverse)
        torch.cuda.synchronize()

        self.assertEqual(full_lengths.item(), 8192)
        torch.testing.assert_close(
            full_pool[0, :8192],
            torch.arange(8192, 16 * 1024, device=self.device, dtype=torch.int32),
        )
        self.assertFalse(full_inverse[0, :8192].any())
        torch.testing.assert_close(
            full_inverse[0, 8192 : 16 * 1024],
            torch.arange(1, 8193, device=self.device, dtype=torch.int32),
        )


if __name__ == "__main__":
    main()
