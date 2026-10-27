"""Request lifecycle coverage for the decode candidate pool."""

import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.base.cuda.indexer_op import IndexerOp
from rtp_llm.models_py.modules.hybrid.indexer import Indexer
from rtp_llm.models_py.triton_kernels.sparse_mla.indexer_pool_reuse import IndexerPool


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class IndexerPoolLifecycleTest(unittest.TestCase):
    def make_indexer(self):
        # Exercise the real forward boundary and real GPU planner. Projections
        # are irrelevant to request identity and do not require model weights.
        indexer = Indexer.__new__(Indexer)
        torch.nn.Module.__init__(indexer)
        pool = IndexerPool(slots=4, reuse_steps=15, min_length=16384, device="cuda")
        indexer.indexer_op = SimpleNamespace(
            _pool=pool, quant_k_only=lambda key, cache, slots: None
        )
        indexer._get_k_bf16 = lambda hidden, params: hidden
        table = torch.arange(400, dtype=torch.int32, device="cuda").view(1, -1)
        lengths = torch.tensor([20000], dtype=torch.int32, device="cuda")
        pool.plan(table, lengths)
        return indexer, pool, table

    def forward_boundary(self, indexer, **flags):
        hidden = torch.empty((0,), device="cuda")
        indexer.forward(
            hidden,
            hidden,
            None,
            SimpleNamespace(slot_mapping=None),
            SimpleNamespace(**flags),
            use_fast_path=True,
        )

    def test_prefill_invalidates_recycled_pages(self):
        indexer, pool, table = self.make_indexer()
        # Request A ended. Request B reclaims A's pages; its initial decode
        # length is inside A's old reuse window. It must build its own pool.
        self.forward_boundary(indexer, is_prefill=True)
        lengths = torch.tensor([20001], dtype=torch.int32, device="cuda")
        _, full_lengths, _ = pool.plan(table, lengths)
        self.assertEqual(full_lengths.tolist(), [20001])

    def test_decode_retains_live_pool(self):
        indexer, pool, table = self.make_indexer()
        self.forward_boundary(indexer, is_prefill=False)
        lengths = torch.tensor([20001], dtype=torch.int32, device="cuda")
        _, full_lengths, _ = pool.plan(table, lengths)
        self.assertEqual(full_lengths.tolist(), [0])

    def test_mtp_fast_path_invalidates_recycled_pages(self):
        for flags in (
            dict(is_prefill=True, is_target_verify=True),
            dict(is_prefill=True, is_draft_extend=True),
            dict(is_prefill=False, is_target_verify=True),
            dict(is_prefill=False, is_draft_extend=True),
        ):
            with self.subTest(**flags):
                indexer, pool, table = self.make_indexer()
                # A new request can begin in a path that skips pool planning.
                # Its later ordinary decode may already be past prompt+1.
                self.forward_boundary(indexer, **flags)
                lengths = torch.tensor([20002], dtype=torch.int32, device="cuda")
                _, full_lengths, _ = pool.plan(table, lengths)
                self.assertEqual(full_lengths.tolist(), [20002])

    def call_decode(self, pool, table, rows, **flags):
        op = IndexerOp(32, 128, 2048, 64)
        op._pool = pool
        lengths = torch.full((rows,), 20001, dtype=torch.int32, device="cuda")
        inputs = SimpleNamespace(
            is_prefill=False,
            kv_cache_kernel_block_id_device=table.expand(rows, -1).contiguous(),
            input_lengths=lengths - 1,
            **flags,
        )
        cache = SimpleNamespace(
            kv_scale_base=torch.zeros((400, 64, 132), dtype=torch.uint8, device="cuda")
        )
        return op._get_topk_paged(
            torch.zeros((rows, 32, 128), dtype=torch.float8_e4m3fn, device="cuda"),
            torch.ones((rows, 32), device="cuda"),
            cache,
            SimpleNamespace(expanded_seq_lens=lengths, kvlen_d=lengths),
            inputs,
        )

    def test_decode_entry_passes_request_boundary_to_pool(self):
        _, pool, table = self.make_indexer()
        out = self.call_decode(pool, table, 1)
        self.assertEqual(out.shape, (1, 2048))
        self.assertEqual(pool.anchor[0].item(), 20001)

    def test_oversized_batch_invalidates_bypassed_pool(self):
        _, pool, table = self.make_indexer()
        out = self.call_decode(pool, table, 5)
        self.assertEqual(out.shape, (5, 2048))
        self.assertTrue(torch.equal(pool.anchor, torch.zeros_like(pool.anchor)))

    def test_mtp_backend_invalidates_bypassed_pool(self):
        for flag in ("is_target_verify", "is_draft_extend"):
            with self.subTest(flag=flag):
                _, pool, table = self.make_indexer()
                self.call_decode(pool, table, 1, **{flag: True})
                self.assertTrue(torch.equal(pool.anchor, torch.zeros_like(pool.anchor)))


if __name__ == "__main__":
    unittest.main()
