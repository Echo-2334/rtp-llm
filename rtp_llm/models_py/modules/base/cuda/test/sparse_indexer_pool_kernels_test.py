from unittest import SkipTest, TestCase, main

import torch

from rtp_llm.models_py.modules.base.cuda.indexer_op import IndexerOp
from rtp_llm.models_py.triton_kernels.sparse_mla.packed_indexer_pool import (
    PACKED_APPEND_POOL_SIZE,
    PACKED_POOL_SIZE,
    gather_packed_pool_kv,
    gather_packed_pool_kv_persistent,
    initialize_missing_packed_pool,
    update_materialized_append_pool,
    update_fixed_packed_pool,
)
from rtp_llm.models_py.triton_kernels.sparse_mla.sparse_indexer_pool_update import (
    append_global_pool_from_pool_chunk_topk,
    compact_append_pool_if_full,
    initialize_global_pool_inverse_map,
)
from rtp_llm.models_py.triton_kernels.sparse_mla.sparse_indexer_score import (
    prepare_append_hybrid_metadata,
    sparse_fp8_mqa_append_logits,
    sparse_fp8_mqa_logits,
    sparse_fp8_mqa_pool_chunk_logits,
)
from rtp_llm.ops.compute_ops import rtp_llm_ops


class SparseIndexerPoolKernelsTest(TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is required")
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        torch.manual_seed(20260721)

    def test_append_hybrid_metadata_matches_row_routing(self):
        kv_lengths = torch.tensor(
            [1, 4096, 4097, 8192, 10000, -5],
            device=self.device,
            dtype=torch.int32,
        )
        slots = torch.tensor(
            [-1, 0, 1, 2, 3, 4], device=self.device, dtype=torch.int32
        )
        bootstrap_mask = torch.tensor(
            [0, 0, 1, 0, 1, 0], device=self.device, dtype=torch.int32
        )
        exact, bootstrap, sparse, sparse_bool = prepare_append_hybrid_metadata(
            kv_lengths,
            slots,
            bootstrap_mask,
            min_kv_length=4096,
            graph_max_seq_len=8192,
        )
        torch.cuda.synchronize()
        self.assertEqual(exact.tolist(), [1, 4096, 4097, 1, 8192, 1])
        self.assertEqual(bootstrap.tolist(), [0, 0, 1, 0, 1, 0])
        self.assertEqual(sparse.tolist(), [0, 0, 0, 1, 0, 0])
        self.assertEqual(
            sparse_bool.tolist(), [False, False, False, True, False, False]
        )

    def test_fixed_packed_pool_initializes_and_replaces_only_new_topk(self):
        batch = 1
        seq_len = 16 * 1024
        page_size = 64
        head_dim = 128
        entry_bytes = head_dim + 4
        pages_per_row = seq_len // page_size
        raw_cache = torch.empty(
            pages_per_row,
            page_size * entry_bytes,
            dtype=torch.uint8,
            device=self.device,
        )
        source_k = torch.arange(
            pages_per_row * page_size * head_dim,
            dtype=torch.int64,
            device=self.device,
        ).remainder(251).to(torch.uint8).view(pages_per_row, page_size, head_dim)
        source_scales = torch.arange(
            pages_per_row * page_size,
            dtype=torch.float32,
            device=self.device,
        ).view(pages_per_row, page_size)
        raw_cache[:, : page_size * head_dim].copy_(source_k.reshape(pages_per_row, -1))
        raw_cache[:, page_size * head_dim :].copy_(
            source_scales.view(torch.uint8).reshape(pages_per_row, -1)
        )
        kv_cache = raw_cache.view(pages_per_row, page_size, 1, entry_bytes)
        block_table = torch.arange(
            pages_per_row, dtype=torch.int32, device=self.device
        ).view(batch, -1)

        state_capacity = 2
        pool = torch.zeros(
            state_capacity, 16 * 1024, dtype=torch.int32, device=self.device
        )
        pool[0, :PACKED_POOL_SIZE] = torch.arange(
            PACKED_POOL_SIZE, dtype=torch.int32, device=self.device
        )
        pool_lengths = torch.tensor(
            [PACKED_POOL_SIZE, 0], dtype=torch.int32, device=self.device
        )
        inverse_map = initialize_global_pool_inverse_map(
            pool, seq_len, pool_lengths
        )
        packed = torch.zeros(
            state_capacity * (PACKED_POOL_SIZE // page_size),
            page_size,
            1,
            entry_bytes,
            dtype=torch.uint8,
            device=self.device,
        )
        slots = torch.tensor([0], dtype=torch.int32, device=self.device)
        active = torch.ones(batch, dtype=torch.int32, device=self.device)
        ready = torch.zeros(state_capacity, dtype=torch.int32, device=self.device)
        initialize_missing_packed_pool(
            kv_cache,
            block_table,
            pool[:, :PACKED_POOL_SIZE],
            slots,
            active,
            torch.full_like(active, PACKED_POOL_SIZE),
            ready,
            packed,
        )
        torch.cuda.synchronize()
        self.assertEqual(ready[0].item(), 1)
        torch.testing.assert_close(
            packed[: PACKED_POOL_SIZE // page_size],
            kv_cache[: PACKED_POOL_SIZE // page_size],
            rtol=0,
            atol=0,
        )

        local_topk = torch.arange(2048, dtype=torch.int32, device=self.device).view(1, -1)
        local_topk[0, -4:] = PACKED_POOL_SIZE + torch.arange(
            4, dtype=torch.int32, device=self.device
        )
        protected = torch.zeros(
            state_capacity, PACKED_POOL_SIZE, dtype=torch.int32, device=self.device
        )
        cursor = torch.zeros(state_capacity, dtype=torch.int32, device=self.device)
        chunk_starts = torch.tensor(
            [PACKED_POOL_SIZE], dtype=torch.int32, device=self.device
        )
        result, changed_ids, changed_positions, changed_counts = update_fixed_packed_pool(
            local_topk,
            pool[:, :PACKED_POOL_SIZE],
            inverse_map,
            slots,
            chunk_starts,
            torch.ones(batch, dtype=torch.int32, device=self.device),
            active,
            protected,
            cursor,
        )
        gather_packed_pool_kv(
            kv_cache,
            block_table,
            changed_ids,
            torch.zeros(batch, dtype=torch.int32, device=self.device),
            changed_counts,
            slots,
            active,
            packed,
            destination_positions=changed_positions,
        )
        torch.cuda.synchronize()

        self.assertEqual(changed_counts.item(), 4)
        torch.testing.assert_close(
            result[0, -4:],
            torch.arange(
                PACKED_POOL_SIZE,
                PACKED_POOL_SIZE + 4,
                dtype=torch.int32,
                device=self.device,
            ),
        )
        changed = changed_positions[0, :4].long()
        new_ids = changed_ids[0, :4].long()
        torch.testing.assert_close(pool[0, changed], new_ids.to(torch.int32))
        self.assertTrue((inverse_map[0, new_ids] == changed.to(torch.int32) + 1).all())

        packed_raw = packed.view(state_capacity, PACKED_POOL_SIZE // page_size, page_size * entry_bytes)
        packed_k = packed_raw[:, :, : page_size * head_dim].reshape(
            state_capacity, PACKED_POOL_SIZE, head_dim
        )
        packed_s = packed_raw[:, :, page_size * head_dim :].reshape(
            state_capacity, PACKED_POOL_SIZE, 4
        )
        expected_blocks = torch.div(new_ids, page_size, rounding_mode="floor")
        expected_offsets = torch.remainder(new_ids, page_size)
        torch.testing.assert_close(
            packed_k[0, changed], source_k[expected_blocks, expected_offsets]
        )
        torch.testing.assert_close(
            packed_s[0, changed],
            source_scales[expected_blocks, expected_offsets]
            .view(torch.uint8)
            .reshape(-1, 4),
        )

    def test_materialized_append_pool_compacts_with_ring_base(self):
        batch = 1
        page_size = 64
        head_dim = 128
        entry_bytes = head_dim + 4
        seq_len = 32 * 1024
        pages = seq_len // page_size
        raw_cache = torch.empty(
            pages, page_size * entry_bytes, dtype=torch.uint8, device=self.device
        )
        source_k = torch.arange(
            pages * page_size * head_dim,
            dtype=torch.int64,
            device=self.device,
        ).remainder(251).to(torch.uint8).view(pages, page_size, head_dim)
        source_scales = torch.arange(
            pages * page_size, dtype=torch.float32, device=self.device
        ).view(pages, page_size)
        raw_cache[:, : page_size * head_dim].copy_(source_k.reshape(pages, -1))
        raw_cache[:, page_size * head_dim :].copy_(
            source_scales.view(torch.uint8).reshape(pages, -1)
        )
        kv_cache = raw_cache.view(pages, page_size, 1, entry_bytes)
        block_table = torch.arange(
            pages, dtype=torch.int32, device=self.device
        ).view(batch, -1)

        pool = torch.arange(
            PACKED_APPEND_POOL_SIZE, dtype=torch.int32, device=self.device
        ).view(1, -1)
        pool_lengths = torch.tensor(
            [PACKED_APPEND_POOL_SIZE], dtype=torch.int32, device=self.device
        )
        inverse_map = initialize_global_pool_inverse_map(
            pool, seq_len, pool_lengths
        )
        packed = kv_cache[: PACKED_APPEND_POOL_SIZE // page_size].clone()
        slots = torch.zeros(batch, dtype=torch.int32, device=self.device)
        active = torch.ones(batch, dtype=torch.int32, device=self.device)
        base_offsets = torch.zeros(batch, dtype=torch.int32, device=self.device)
        local_topk = (
            PACKED_POOL_SIZE
            + torch.arange(2048, dtype=torch.int32, device=self.device)
        ).view(1, -1)
        local_topk[0, -4:] = PACKED_APPEND_POOL_SIZE + torch.arange(
            4, dtype=torch.int32, device=self.device
        )
        chunk_start = torch.tensor(
            [PACKED_APPEND_POOL_SIZE], dtype=torch.int32, device=self.device
        )
        result, changed_ids, changed_positions, changed_counts = (
            update_materialized_append_pool(
                local_topk,
                pool,
                pool_lengths,
                inverse_map,
                slots,
                chunk_start,
                active,
                base_offsets,
            )
        )
        gather_packed_pool_kv_persistent(
            kv_cache,
            block_table,
            changed_ids,
            torch.zeros_like(slots),
            changed_positions,
            changed_counts,
            slots,
            active,
            packed,
            destination_base_offsets=base_offsets,
            pool_capacity=PACKED_APPEND_POOL_SIZE,
        )
        torch.cuda.synchronize()

        self.assertEqual(base_offsets.item(), PACKED_POOL_SIZE)
        self.assertEqual(pool_lengths.item(), PACKED_POOL_SIZE + 4)
        self.assertEqual(changed_counts.item(), 4)
        torch.testing.assert_close(
            pool[0, :PACKED_POOL_SIZE],
            torch.arange(
                PACKED_POOL_SIZE,
                PACKED_APPEND_POOL_SIZE,
                dtype=torch.int32,
                device=self.device,
            ),
        )
        torch.testing.assert_close(
            pool[0, PACKED_POOL_SIZE : PACKED_POOL_SIZE + 4],
            torch.arange(
                PACKED_APPEND_POOL_SIZE,
                PACKED_APPEND_POOL_SIZE + 4,
                dtype=torch.int32,
                device=self.device,
            ),
        )
        torch.testing.assert_close(
            result[0, -4:],
            torch.arange(
                PACKED_APPEND_POOL_SIZE,
                PACKED_APPEND_POOL_SIZE + 4,
                dtype=torch.int32,
                device=self.device,
            ),
        )

        packed_raw = packed.view(
            PACKED_APPEND_POOL_SIZE // page_size, page_size * entry_bytes
        )
        packed_k = packed_raw[:, : page_size * head_dim].reshape(
            PACKED_APPEND_POOL_SIZE, head_dim
        )
        logical_positions = torch.arange(
            PACKED_POOL_SIZE + 4, dtype=torch.long, device=self.device
        )
        physical_positions = torch.remainder(
            logical_positions + base_offsets.item(), PACKED_APPEND_POOL_SIZE
        )
        logical_ids = pool[0, : PACKED_POOL_SIZE + 4].long()
        expected_blocks = torch.div(logical_ids, page_size, rounding_mode="floor")
        expected_offsets = torch.remainder(logical_ids, page_size)
        torch.testing.assert_close(
            packed_k[physical_positions],
            source_k[expected_blocks, expected_offsets],
        )

    def test_fixed_packed_dual_paged_matches_unaligned_sparse_candidates(self):
        seq_len = 70003
        graph_max_seq_len = ((seq_len + 63) // 64) * 64
        pages = graph_max_seq_len // 64
        raw_cache = torch.empty(
            pages, 64 * 132, dtype=torch.uint8, device=self.device
        )
        raw_cache[:, : 64 * 128].copy_(
            torch.randint(
                0,
                120,
                (pages, 64 * 128),
                dtype=torch.uint8,
                device=self.device,
            )
        )
        scales = torch.ones(pages, 64, dtype=torch.float32, device=self.device)
        raw_cache[:, 64 * 128 :].copy_(
            scales.view(torch.uint8).reshape(pages, -1)
        )
        kv_cache = raw_cache.view(pages, 64, 1, 132)
        block_table = torch.arange(
            pages, dtype=torch.int32, device=self.device
        ).view(1, -1)
        q = torch.randn(1, 32, 128, device=self.device).clamp_(-3, 3).to(
            torch.float8_e4m3fn
        )
        weights = torch.randn(1, 32, dtype=torch.float32, device=self.device)
        pool = torch.zeros(2, 16 * 1024, dtype=torch.int32, device=self.device)
        pool[0, :PACKED_POOL_SIZE] = torch.randperm(
            seq_len, dtype=torch.int32, device=self.device
        )[:PACKED_POOL_SIZE]
        pool_lengths = torch.tensor(
            [PACKED_POOL_SIZE, 0], dtype=torch.int32, device=self.device
        )
        inverse_map = initialize_global_pool_inverse_map(
            pool, graph_max_seq_len, pool_lengths
        )
        packed = torch.zeros(
            2 * (PACKED_POOL_SIZE // 64),
            64,
            1,
            132,
            dtype=torch.uint8,
            device=self.device,
        )
        ready = torch.zeros(2, dtype=torch.int32, device=self.device)
        protected = torch.zeros(
            2, PACKED_POOL_SIZE, dtype=torch.int32, device=self.device
        )
        cursor = torch.zeros(2, dtype=torch.int32, device=self.device)
        slots = torch.zeros(1, dtype=torch.int32, device=self.device)
        decode_step = torch.tensor([7], dtype=torch.int32, device=self.device)
        kv_lengths = torch.tensor([seq_len], dtype=torch.int32, device=self.device)

        phase = decode_step.item() - 1
        chunk_start = seq_len * phase // 16
        chunk_end = seq_len * (phase + 1) // 16
        chunk_capacity = (graph_max_seq_len + 15) // 16
        chunk = (
            chunk_start
            + torch.arange(chunk_capacity, dtype=torch.int32, device=self.device)
        ).view(1, -1)
        chunk_lengths = torch.tensor(
            [chunk_end - chunk_start], dtype=torch.int32, device=self.device
        )
        old_pool = pool[0, :PACKED_POOL_SIZE].clone().view(1, -1)
        reference_logits = sparse_fp8_mqa_pool_chunk_logits(
            q,
            weights,
            kv_cache,
            block_table,
            pool,
            pool_lengths,
            chunk,
            chunk_lengths,
            pool_slots=slots,
        )
        op = IndexerOp(32, 128, 2048, rope_head_dim=0)
        candidate_lengths = torch.tensor(
            [PACKED_POOL_SIZE + chunk_lengths.item()],
            dtype=torch.int32,
            device=self.device,
        )
        reference_local = op._select_persistent_topk(
            reference_logits,
            candidate_lengths,
            16 * 1024 + chunk_capacity,
            "reference",
        )
        candidate_ids = torch.cat((old_pool, chunk), dim=1)
        reference_ids = torch.gather(
            candidate_ids, 1, reference_local.long()
        ).to(torch.int32)

        actual = op._score_paged_packed_pool_step(
            q,
            weights,
            kv_cache,
            block_table,
            pool,
            inverse_map,
            packed,
            ready,
            protected,
            cursor,
            slots,
            decode_step,
            kv_lengths,
            1,
            64 * 1024,
            16,
            graph_max_seq_len,
        )
        torch.cuda.synchronize()
        overlap = len(
            set(actual[0].cpu().tolist())
            & set(reference_ids[0].cpu().tolist())
        )
        self.assertEqual(overlap, 2048)

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
        slotted_lengths = torch.empty(
            batch, device=self.device, dtype=torch.int32
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
            candidate_lengths_out=slotted_lengths,
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
        torch.testing.assert_close(slotted_lengths, combined_lengths)

        scheduled_pool = torch.zeros(
            4, 16 * 1024, device=self.device, dtype=torch.int32
        )
        scheduled_pool[:batch].copy_(pool)
        scheduled_lengths = torch.tensor(
            [4096, 6144, 0, 0], device=self.device, dtype=torch.int32
        )
        scheduled_slots = torch.tensor(
            [0, 1], device=self.device, dtype=torch.int32
        )
        decode_steps = torch.tensor(
            [1, 5], device=self.device, dtype=torch.int32
        )
        kv_lengths = torch.full(
            (batch,), seq_len, device=self.device, dtype=torch.int32
        )
        (
            scheduled_logits,
            scheduled_topk_lengths,
            scheduled_starts,
            scheduled_chunk_lengths,
            safe_slots,
            scheduled_active,
        ) = sparse_fp8_mqa_append_logits(
            q,
            weights,
            kv_cache,
            block_table,
            scheduled_pool,
            scheduled_lengths,
            scheduled_slots,
            decode_steps,
            kv_lengths,
            min_kv_length=4096,
            source_chunks=16,
            graph_max_seq_len=seq_len,
            dummy_slot_base=2,
        )
        expected_starts = torch.tensor(
            [0, 4096], device=self.device, dtype=torch.int32
        )
        expected_chunk_lengths = torch.full(
            (batch,), 1024, device=self.device, dtype=torch.int32
        )
        scheduled_chunks = expected_starts.unsqueeze(1) + torch.arange(
            1024, device=self.device, dtype=torch.int32
        ).unsqueeze(0)
        scheduled_reference = sparse_fp8_mqa_pool_chunk_logits(
            q,
            weights,
            kv_cache,
            block_table,
            scheduled_pool,
            scheduled_lengths,
            scheduled_chunks,
            expected_chunk_lengths,
            pool_slots=scheduled_slots,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(scheduled_logits, scheduled_reference, rtol=0, atol=0)
        torch.testing.assert_close(scheduled_starts, expected_starts, rtol=0, atol=0)
        torch.testing.assert_close(
            scheduled_chunk_lengths, expected_chunk_lengths, rtol=0, atol=0
        )
        torch.testing.assert_close(safe_slots, scheduled_slots, rtol=0, atol=0)
        torch.testing.assert_close(
            scheduled_active,
            torch.ones_like(scheduled_active),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            scheduled_topk_lengths,
            scheduled_lengths[:batch] + expected_chunk_lengths,
            rtol=0,
            atol=0,
        )

        external_active = torch.tensor(
            [1, 0], device=self.device, dtype=torch.int32
        )
        (
            masked_logits,
            masked_topk_lengths,
            _,
            masked_chunk_lengths,
            masked_slots,
            masked_active,
        ) = sparse_fp8_mqa_append_logits(
            q,
            weights,
            kv_cache,
            block_table,
            scheduled_pool,
            scheduled_lengths,
            scheduled_slots,
            decode_steps,
            kv_lengths,
            min_kv_length=4096,
            source_chunks=16,
            graph_max_seq_len=seq_len,
            dummy_slot_base=2,
            external_active=external_active,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(masked_logits[0], scheduled_logits[0])
        self.assertTrue(torch.isneginf(masked_logits[1]).all())
        self.assertEqual(masked_topk_lengths.tolist(), [5120, 1])
        self.assertEqual(masked_chunk_lengths.tolist(), [1024, 0])
        self.assertEqual(masked_slots.tolist(), [0, 3])
        self.assertEqual(masked_active.tolist(), [1, 0])

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
        pool_slots = torch.zeros(1, device=self.device, dtype=torch.int32)
        active_mask = torch.ones(1, device=self.device, dtype=torch.int32)

        selected = append_global_pool_from_pool_chunk_topk(
            local_topk,
            pool,
            pool_lengths,
            chunk,
            chunk_lengths,
            inverse_map,
            pool_slots,
            active_mask,
        )
        torch.cuda.synchronize()
        self.assertEqual(pool_lengths.item(), 10 * 1024)
        torch.testing.assert_close(selected, chunk[:, :2048])
        self.assertEqual(
            set(pool[0, 8192 : 10 * 1024].cpu().tolist()),
            set(chunk[0, :2048].cpu().tolist()),
        )
        appended_slots = inverse_map[0, chunk[0, :2048]]
        self.assertTrue((appended_slots >= 8193).all())
        self.assertTrue((appended_slots <= 10 * 1024).all())
        self.assertEqual(torch.unique(appended_slots).numel(), 2048)

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

    def test_fused_topk_pool_matches_separate_postprocess(self):
        batch = 2
        topk = 2048
        candidate_width = 24 * 1024
        pool_capacity = 16 * 1024
        initial_pool_size = 8 * 1024
        max_seq_len = 32 * 1024

        initial_pool = torch.zeros(
            batch, pool_capacity, device=self.device, dtype=torch.int32
        )
        initial_pool[0, :initial_pool_size] = torch.arange(
            0, initial_pool_size, device=self.device, dtype=torch.int32
        )
        initial_pool[1, :initial_pool_size] = torch.arange(
            16 * 1024,
            24 * 1024,
            device=self.device,
            dtype=torch.int32,
        )
        initial_lengths = torch.full(
            (batch,), initial_pool_size, device=self.device, dtype=torch.int32
        )
        chunk = torch.stack(
            (
                torch.arange(
                    8 * 1024, 16 * 1024, device=self.device, dtype=torch.int32
                ),
                torch.arange(
                    24 * 1024, 32 * 1024, device=self.device, dtype=torch.int32
                ),
            )
        )
        chunk_lengths = torch.full(
            (batch,), 8 * 1024, device=self.device, dtype=torch.int32
        )
        pool_slots = torch.tensor([1, 0], device=self.device, dtype=torch.int32)
        active_mask = torch.tensor([1, 0], device=self.device, dtype=torch.int32)
        candidate_lengths = torch.full(
            (batch,), 16 * 1024, device=self.device, dtype=torch.int32
        )
        logits = torch.randn(
            batch, candidate_width, device=self.device, dtype=torch.float32
        )
        logits[:, 16 * 1024 :].fill_(-float("inf"))

        reference_pool = initial_pool.clone()
        reference_lengths = initial_lengths.clone()
        reference_inverse = initialize_global_pool_inverse_map(
            reference_pool, max_seq_len, reference_lengths
        )
        local_topk = torch.empty(
            batch, topk, device=self.device, dtype=torch.int32
        )
        reference_workspace = torch.empty(
            1 << 20, device=self.device, dtype=torch.uint8
        )
        rtp_llm_ops.dsv4_persistent_topk(
            logits,
            candidate_lengths,
            local_topk,
            reference_workspace,
            topk,
            candidate_width,
        )
        reference_output = append_global_pool_from_pool_chunk_topk(
            local_topk,
            reference_pool,
            reference_lengths,
            chunk,
            chunk_lengths,
            reference_inverse,
            pool_slots,
            active_mask,
        )

        fused_pool = initial_pool.clone()
        fused_lengths = initial_lengths.clone()
        fused_inverse = initialize_global_pool_inverse_map(
            fused_pool, max_seq_len, fused_lengths
        )
        fused_output = torch.empty_like(local_topk)
        fused_workspace = torch.empty_like(reference_workspace)
        rtp_llm_ops.dsv4_persistent_topk_pool(
            logits,
            candidate_lengths,
            fused_output,
            fused_workspace,
            candidate_width,
            fused_pool,
            fused_lengths,
            chunk,
            chunk_lengths,
            fused_inverse,
            pool_slots,
            active_mask,
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(fused_lengths, reference_lengths, rtol=0, atol=0)
        for row in range(batch):
            self.assertEqual(
                set(fused_output[row].cpu().tolist()),
                set(reference_output[row].cpu().tolist()),
            )
        for pool_slot in range(batch):
            pool_length = int(fused_lengths[pool_slot].item())
            self.assertEqual(
                set(fused_pool[pool_slot, :pool_length].cpu().tolist()),
                set(reference_pool[pool_slot, :pool_length].cpu().tolist()),
            )
            ids = fused_pool[pool_slot, :pool_length].to(torch.long)
            positions = fused_inverse[pool_slot, ids].to(torch.long) - 1
            torch.testing.assert_close(
                fused_pool[pool_slot, positions], ids.to(torch.int32), rtol=0, atol=0
            )

    def test_fused_topk_compacts_full_pool_after_selection(self):
        topk = 2048
        pool_capacity = 16 * 1024
        keep_size = 8 * 1024
        chunk_size = 8 * 1024
        candidate_width = pool_capacity + chunk_size
        max_seq_len = candidate_width

        initial_pool = torch.arange(
            pool_capacity, device=self.device, dtype=torch.int32
        ).unsqueeze(0)
        pool = initial_pool.clone()
        pool_lengths = torch.tensor(
            [pool_capacity], device=self.device, dtype=torch.int32
        )
        inverse_map = initialize_global_pool_inverse_map(
            pool, max_seq_len, pool_lengths
        )
        chunk = torch.arange(
            pool_capacity,
            candidate_width,
            device=self.device,
            dtype=torch.int32,
        ).unsqueeze(0)
        chunk_lengths = torch.tensor(
            [chunk_size], device=self.device, dtype=torch.int32
        )
        candidate_lengths = torch.tensor(
            [candidate_width], device=self.device, dtype=torch.int32
        )
        logits = torch.randn(
            1, candidate_width, device=self.device, dtype=torch.float32
        )
        expected_local = torch.topk(
            logits, topk, dim=1, sorted=False
        ).indices.to(torch.int32)
        expected_ids = torch.gather(
            torch.cat((initial_pool, chunk), dim=1),
            1,
            expected_local.to(torch.long),
        )

        output = torch.empty(
            1, topk, device=self.device, dtype=torch.int32
        )
        workspace = torch.empty(
            1 << 20, device=self.device, dtype=torch.uint8
        )
        rtp_llm_ops.dsv4_persistent_topk_pool(
            logits,
            candidate_lengths,
            output,
            workspace,
            candidate_width,
            pool,
            pool_lengths,
            chunk,
            chunk_lengths,
            inverse_map,
            torch.tensor([0], device=self.device, dtype=torch.int32),
            torch.tensor([1], device=self.device, dtype=torch.int32),
        )
        torch.cuda.synchronize()

        self.assertEqual(
            set(output[0].cpu().tolist()), set(expected_ids[0].cpu().tolist())
        )
        retained = set(range(keep_size, pool_capacity))
        expected_pool = retained | set(expected_ids[0].cpu().tolist())
        pool_length = int(pool_lengths[0].item())
        self.assertEqual(pool_length, len(expected_pool))
        self.assertEqual(
            set(pool[0, :pool_length].cpu().tolist()), expected_pool
        )
        ids = pool[0, :pool_length].to(torch.long)
        positions = inverse_map[0, ids].to(torch.long) - 1
        torch.testing.assert_close(
            pool[0, positions], ids.to(torch.int32), rtol=0, atol=0
        )


if __name__ == "__main__":
    main()
