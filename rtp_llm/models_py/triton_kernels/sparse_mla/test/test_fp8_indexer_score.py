import unittest
from types import SimpleNamespace
from unittest.mock import patch

import librtp_compute_ops
import libth_transformer
import libth_transformer_config
import torch
from rtp_llm.models_py.modules.base.cuda.indexer_op import IndexerOp
from rtp_llm.models_py.triton_kernels.sparse_mla.fp8_indexer_score import (
    fp8_paged_indexer_logits,
    fp8_ragged_indexer_logits,
)
from rtp_llm.ops.compute_ops import LayerKVCache, rtp_llm_ops


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class Fp8IndexerScoreTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.device = "cuda"
        self.lengths = torch.tensor([65, 131], device="cuda", dtype=torch.int32)
        self.table = torch.tensor(
            [[2, 5, 0], [1, 3, 6]], device="cuda", dtype=torch.int32
        )
        slots = torch.cat(
            [
                self.table[b, torch.arange(n, device="cuda") // 64].long() * 64
                + torch.arange(n, device="cuda") % 64
                for b, n in enumerate([65, 131])
            ]
        )
        self.cache = torch.zeros((8, 64, 132), dtype=torch.uint8, device="cuda")
        key = torch.randn((196, 128), dtype=torch.bfloat16, device="cuda")
        rtp_llm_ops.indexer_k_quant_and_cache(key, self.cache, slots, 128, "ue8m0")
        pages = self.cache.view(8, -1)
        k = pages[:, : 64 * 128].contiguous().view(torch.float8_e4m3fn).view(8, 64, 128)
        sf = pages[:, 64 * 128 :].contiguous().view(torch.float32).view(8, 64)
        self.keys = torch.stack(
            [
                k[
                    self.table[b, torch.arange(192, device="cuda") // 64].long(),
                    torch.arange(192, device="cuda") % 64,
                ]
                for b in range(2)
            ]
        )
        self.scales = torch.stack(
            [
                sf[
                    self.table[b, torch.arange(192, device="cuda") // 64].long(),
                    torch.arange(192, device="cuda") % 64,
                ]
                for b in range(2)
            ]
        )
        self.q = torch.randn((2, 32, 128), device="cuda").to(torch.float8_e4m3fn)
        self.w = torch.randn((2, 32), device="cuda")

    def reference(self):
        dot = torch.einsum("bhd,btd->bht", self.q.float(), self.keys.float())
        scores = (dot.relu() * self.w[:, :, None]).sum(1) * self.scales
        return scores.masked_fill(
            torch.arange(192, device="cuda")[None, :] >= self.lengths[:, None],
            -torch.inf,
        )

    def test_native_cache_scores_and_dynamic_graph(self):
        def run():
            return fp8_paged_indexer_logits(
                self.q, self.cache[:, :, None, :], self.w, self.lengths, self.table, 192
            )

        got = run()
        torch.testing.assert_close(got, self.reference(), atol=2e-4, rtol=3e-5)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        self.lengths.copy_(torch.tensor([31, 80], device="cuda", dtype=torch.int32))
        self.q.copy_(torch.randn_like(self.q.float()).to(self.q.dtype))
        graph.replay()
        torch.testing.assert_close(captured, self.reference(), atol=2e-4, rtol=3e-5)

    def test_ragged_scores_match_reference(self):
        k = torch.cat([self.keys[0, :65], self.keys[1, :131]])
        sf = torch.cat([self.scales[0, :65], self.scales[1, :131]])
        starts = torch.tensor([0, 65], dtype=torch.int32, device="cuda")
        ends = torch.tensor([65, 196], dtype=torch.int32, device="cuda")
        got = fp8_ragged_indexer_logits(self.q, (k, sf), self.w, starts, ends)
        ref = (
            torch.einsum("bhd,td->bht", self.q.float(), k.float()).relu()
            * self.w[:, :, None]
        ).sum(1) * sf[None, :]
        cols = torch.arange(196, device="cuda")[None, :]
        ref.masked_fill_((cols < starts[:, None]) | (cols >= ends[:, None]), -torch.inf)
        torch.testing.assert_close(got, ref, atol=2e-4, rtol=3e-5)

    def test_actual_indexer_decode_and_prefill_topk(self):
        with patch.dict(
            "os.environ",
            {
                "GLM5_INDEXER_TOPK_BACKEND": "topk_v3",
                "GLM5_PREFILL_INDEXER_TOPK_BACKEND": "topk_v3_tie_break",
            },
        ):
            op = IndexerOp(32, 128, 512, 64)
        kv = LayerKVCache()
        kv.kv_scale_base = self.cache
        params = SimpleNamespace(kvlen_d=self.lengths, expanded_seq_lens=self.lengths)
        inputs = SimpleNamespace(
            is_target_verify=False,
            is_draft_extend=False,
            kv_cache_kernel_block_id_device=self.table,
            decode_cu_seqlens_d=torch.arange(3, dtype=torch.int32, device="cuda"),
        )
        topk = op._get_topk_paged(self.q, self.w[:, :, None], kv, params, inputs)
        for b, length in enumerate([65, 131]):
            self.assertEqual(set(topk[b, :length].tolist()), set(range(length)))
            self.assertTrue(torch.all(topk[b, length:] == -1))
        params.prefill_total_kv_tokens = 196
        params.prefill_max_kv_len = 131
        params.ks = torch.tensor([0, 65], dtype=torch.int32, device="cuda")
        params.ke = torch.tensor([65, 196], dtype=torch.int32, device="cuda")
        params.topk_indices_offset = torch.tensor(
            [0, 0], dtype=torch.int32, device="cuda"
        )
        inputs.cu_kv_seqlens = torch.tensor(
            [0, 65, 196], dtype=torch.int32, device="cuda"
        )
        topk = op._get_topk_ragged(self.q, self.w[:, :, None], kv, params, inputs)
        for b, length in enumerate([65, 131]):
            self.assertEqual(set(topk[b, :length].tolist()), set(range(length)))
            self.assertTrue(torch.all(topk[b, length:] == -1))


if __name__ == "__main__":
    unittest.main()
