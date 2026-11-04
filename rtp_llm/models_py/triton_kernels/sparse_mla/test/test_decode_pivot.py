import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
import deep_gemm
from rtp_llm.models_py.triton_kernels.sparse_mla.decode_pivot import (
    DecodePivotConfig,
    decode_pivot_topk,
    _score,
)
from rtp_llm.models_py.triton_kernels.sparse_mla.prefill_pivot_kernels import (
    recall_candidates,
)
from rtp_llm.models_py.modules.base.cuda.indexer_op import (
    IndexerOp,
    _get_topk_workspace,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DecodePivotGpuTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261102)
        self.b, self.n, self.h, self.d, self.page, self.cols = 2, 8, 32, 128, 64, 128
        self.q = torch.randn((8, 32, 128), device="cuda").to(torch.float8_e4m3fn)
        self.qs = torch.exp2(torch.randint(-3, 3, (8, 32), device="cuda").float())
        self.w = torch.randn((8, 32), device="cuda") * self.qs / 32
        self.k = torch.randn((256, 64, 128), device="cuda").to(torch.float8_e4m3fn)
        self.sf = torch.rand((256, 64), device="cuda") + 0.5
        self.cache = (
            torch.cat(
                [
                    self.k.view(torch.uint8).reshape(256, -1),
                    self.sf.view(torch.uint8).reshape(256, -1),
                ],
                1,
            )
            .contiguous()
            .view(256, 64, 1, 132)
        )
        self.table = torch.randperm(256, device="cuda").to(torch.int32).reshape(2, 128)
        self.lens = torch.tensor(
            [6000, 6001, 6002, 6003, 1500, 1501, 1502, 1503],
            device="cuda",
            dtype=torch.int32,
        )
        with patch.dict(
            os.environ,
            {
                "GLM5_DECODE_PIVOT_MODE": "off",
                "GLM5_PREFILL_PIVOT_MODE": "off",
                "GLM5_INDEXER_POOL_REUSE_STEPS": "0",
            },
        ):
            self.op = IndexerOp(32, 128, 2048, 64)
        self.config = DecodePivotConfig(candidates=4096, min_kv_len=4096)

    def logits(self, q, w, lens, table):
        meta = deep_gemm.get_paged_mqa_logits_metadata(
            lens, 64, deep_gemm.get_num_sms()
        )
        return deep_gemm.fp8_paged_mqa_logits(
            q.view(*lens.shape, 32, 128),
            self.cache,
            w,
            lens,
            table,
            meta,
            8192,
            clean_logits=False,
        )

    def select(self, x, l, out, m):
        self.op._paged_topk_op(
            x, l, out, _get_topk_workspace(x.device), out.shape[1], m
        )

    def run_pivot(self):
        return decode_pivot_topk(
            self.q,
            self.qs,
            self.w,
            self.cache,
            self.table,
            self.lens,
            64,
            2048,
            self.config,
            self.logits,
            self.select,
        )

    def test_candidate_scores_match_dense_and_causality(self):
        ids = torch.randint(0, 7000, (2, 256), device="cuda", dtype=torch.int32)
        ids[:, 0] = -1
        good = torch.ones(2, device="cuda", dtype=torch.int32)
        out = torch.empty((8, 256), device="cuda")
        _score[(2, 4)](
            self.q,
            self.w,
            self.cache,
            self.table,
            self.lens,
            ids,
            good,
            out,
            32,
            128,
            64,
            128,
            128,
            256,
            64,
            num_warps=8,
        )
        for row in range(8):
            tokens = ids[row // 4].long()
            pages = self.table[row // 4, tokens.clamp_min(0) // 64].long()
            keys = self.k.float()[pages, tokens.clamp_min(0) % 64]
            sf = self.sf[pages, tokens.clamp_min(0) % 64]
            expect = (
                torch.relu(self.q[row].float() @ keys.T) * self.w[row, :, None]
            ).sum(0) * sf
            expect.masked_fill_(
                (tokens < 0) | (tokens >= self.lens[row]), -float("inf")
            )
            torch.testing.assert_close(out[row], expect, rtol=2e-4, atol=2e-4)

    def test_fused_proxy_preserves_reference(self):
        from rtp_llm.models_py.triton_kernels.sparse_mla.decode_pivot import _proxy_plan
        from rtp_llm.models_py.triton_kernels.sparse_mla.prefill_pivot_kernels import (
            build_proxy,
        )

        expected_q, expected_w = build_proxy(self.q, self.qs, self.w, 4)
        pq, pw = torch.empty_like(expected_q), torch.empty_like(expected_w)
        good = torch.empty(2, device="cuda", dtype=torch.int32)
        pl, zeros = torch.empty_like(good), torch.empty_like(good)
        fl = torch.empty_like(self.lens)
        _proxy_plan[(2, 32)](
            self.q,
            self.qs,
            self.w,
            self.lens,
            good,
            pl,
            fl,
            zeros,
            pq,
            pw,
            32,
            128,
            4096,
            4,
            8192,
        )
        torch.testing.assert_close(pq.float(), expected_q.float(), rtol=0, atol=0)
        torch.testing.assert_close(pw, expected_w, rtol=0, atol=0)
        self.assertEqual(good.tolist(), [1, 0])
        self.assertEqual(pl.tolist(), [5996, 0])
        torch.testing.assert_close(fl[:4], torch.zeros_like(fl[:4]))
        torch.testing.assert_close(fl[4:], self.lens[4:])
        self.assertEqual(zeros.tolist(), [0, 0])

    def test_length_aware_topk_graph_ties_and_padding(self):
        from rtp_llm.models_py.triton_kernels.sparse_mla.decode_pivot_topk import (
            radix_topk,
        )

        x = torch.randn(8, 32768, device="cuda")
        lo = torch.tensor([0, 17, 3, 0, 0, 9, 100, 0], device="cuda", dtype=torch.int32)
        hi = torch.tensor(
            [32768, 32760, 20000, 24000, 0, 100, 100, 32768],
            device="cuda",
            dtype=torch.int32,
        )
        k = 16377
        radix_topk(x, lo, hi, k)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = radix_topk(x, lo, hi, k)
        for iteration in range(2):
            if iteration:
                x.normal_()
                x[0].fill_(1)
                x[1].fill_(-2)
                x[2].zero_()
                x[3].fill_(-float("inf"))
                hi[4] = 16384
                hi[7] = 0
            graph.replay()
            for b in range(8):
                start, end = int(lo[b]), int(hi[b])
                n = min(k, end - start)
                ids = out[b, :n].long()
                self.assertEqual(ids.unique().numel(), n)
                self.assertTrue(((ids >= start) & (ids < end)).all().item())
                expected = torch.topk(x[b, start:end], n).values.sort().values
                torch.testing.assert_close(
                    x[b, ids].sort().values, expected, rtol=0, atol=0
                )
                self.assertTrue((out[b, n:] == -1).all().item())

    def test_packed_candidate_scores_graph_and_causality(self):
        from rtp_llm.models_py.triton_kernels.sparse_mla.decode_pivot_score import (
            score_packed_candidates,
        )

        ids = torch.randint(0, 8192, (2, 4096), device="cuda", dtype=torch.int32)
        ids[:, 0] = -1
        good = torch.ones(2, device="cuda", dtype=torch.int32)

        def run():
            return score_packed_candidates(
                self.q, self.w, self.cache, self.table, self.lens, ids, good
            )

        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = run()
        for turn in range(2):
            if turn:
                self.table.copy_(self.table.flip(0))
                ids.random_(0, 8192)
                good[1] = 0
            graph.replay()
            ref = torch.empty_like(out)
            _score[(2, 64)](
                self.q,
                self.w,
                self.cache,
                self.table,
                self.lens,
                ids,
                good,
                ref,
                32,
                128,
                64,
                128,
                128,
                4096,
                64,
                num_warps=8,
            )
            torch.testing.assert_close(out, ref, rtol=2e-4, atol=2e-4)

    def test_native_mtp_matches_flat_logits(self):
        native = self.logits(self.q, self.w, self.lens.reshape(2, 4), self.table)
        flat = self.logits(
            self.q, self.w, self.lens.reshape(8, 1), self.table.repeat_interleave(4, 0)
        )
        for row in range(8):
            torch.testing.assert_close(
                native[row, : self.lens[row]],
                flat[row, : self.lens[row]],
                rtol=2e-4,
                atol=2e-4,
            )

    def test_mixed_fallback_and_padded_rows(self):
        out = self.run_pivot()
        baseline = torch.empty_like(out)
        self.select(
            self.logits(self.q, self.w, self.lens.reshape(2, 4), self.table),
            self.lens,
            baseline,
            8192,
        )
        torch.testing.assert_close(
            out[4:].sort().values, baseline[4:].sort().values, rtol=0, atol=0
        )
        for row in range(4):
            self.assertTrue(
                ((out[row] >= 0) & (out[row] < self.lens[row])).all().item()
            )
            self.assertEqual(out[row].unique().numel(), 2048)
        self.lens.zero_()
        out = self.run_pivot()
        self.assertTrue((out == -1).all().item())

    def test_group_new_tokens_and_no_duplicate_candidates(self):
        x = torch.randn(2, 8192, device="cuda")
        ends = self.lens.view(2, 4)[:, 0].contiguous() - 4
        ids = recall_candidates(x, torch.zeros_like(ends), ends, self.lens, self.config)
        for b in range(2):
            expected = torch.arange(
                self.lens[b * 4] - 4, self.lens[b * 4] + 3, device="cuda"
            )
            torch.testing.assert_close(ids[b, -7:].long(), expected.long())
            v = ids[b][ids[b] >= 0]
            self.assertEqual(v.unique().numel(), v.numel())

    def test_graph_replay_changes_eligibility(self):
        self.run_pivot()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = self.run_pivot()
        g.replay()
        torch.testing.assert_close(out.sort().values, self.run_pivot().sort().values)

        self.lens.copy_(
            torch.tensor(
                [100, 101, 102, 103, 6000, 6001, 6002, 6003],
                device="cuda",
                dtype=torch.int32,
            )
        )
        g.replay()
        torch.testing.assert_close(out.sort().values, self.run_pivot().sort().values)

    def test_production_graph_multiple_batch_sizes_and_slot_reuse(self):
        # Capture the production dispatcher, not just the standalone kernel.
        # Reuse graph input addresses while requests enter/leave/change slots.
        for batch in (1, 2, 3, 4, 5, 8, 16, 32, 64, 128):
            with self.subTest(batch=batch):
                n, capacity = batch * 4, 32768
                q = torch.randn(n, 32, 128, device="cuda").to(torch.float8_e4m3fn)
                scales = torch.ones(n, 32, device="cuda")
                weights = torch.randn(n, 32, device="cuda") / 32
                pages = capacity // 64
                # No repeated physical pages inside a request: artificial score
                # ties otherwise allow the native TopK to choose different IDs.
                table = torch.stack(
                    [torch.randperm(pages, device="cuda") for _ in range(batch)]
                ).to(torch.int32)
                keys = torch.randn(pages, 64, 128, device="cuda").to(
                    torch.float8_e4m3fn
                )
                sf = torch.rand(pages, 64, device="cuda") + 0.5
                packed = torch.cat(
                    [
                        keys.view(torch.uint8).reshape(pages, -1),
                        sf.view(torch.uint8).reshape(pages, -1),
                    ],
                    1,
                )
                lengths = torch.empty(n, device="cuda", dtype=torch.int32)
                self.op._decode_pivot = DecodePivotConfig(
                    candidates=16384, min_kv_len=16384
                )
                cache = SimpleNamespace(kv_scale_base=packed.view(pages, 64, 1, 132))
                fmha = SimpleNamespace(
                    expanded_seq_lens=lengths, schedule_metadata=None
                )
                attn = SimpleNamespace(
                    is_prefill=True,
                    is_target_verify=True,
                    is_draft_extend=False,
                    kv_cache_kernel_block_id_device=table,
                )

                def run():
                    return self.op._get_topk_paged(
                        q, weights, cache, fmha, attn, q_scale=scales
                    )

                def update(phase):
                    # Long, short, padded and nonconsecutive groups share a graph.
                    base = torch.tensor(
                        [24000, 100, 0, 24000], device="cuda", dtype=torch.int32
                    )
                    slot = (torch.arange(batch, device="cuda") + phase) % 4
                    lens = base[slot, None] + torch.arange(4, device="cuda")
                    lens[slot == 2] = 0
                    lens[slot == 3, 2] -= 1
                    lengths.copy_(lens.flatten())
                    table.copy_(table.roll(1, dims=0).roll(7, dims=1))
                    padded = (slot == 2).repeat_interleave(4)
                    random_q = torch.randn_like(q.float())
                    random_q[padded] = 0
                    q.copy_(random_q.to(q.dtype))
                    scales.copy_(
                        torch.exp2(
                            torch.randint(-2, 3, scales.shape, device="cuda").float()
                        )
                    )
                    scales[padded] = 0
                    weights.copy_(torch.randn_like(weights) * scales / 32)

                update(0)
                run()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = run()
                for phase in (0, 1, 2, 3, 0):
                    update(phase)
                    graph.replay()
                    actual = captured.clone()
                    eager = run()
                    torch.testing.assert_close(
                        actual.sort().values, eager.sort().values, rtol=0, atol=0
                    )
                    # A request's candidates must not depend on its batch slot
                    # or on other requests' Q, scales, lengths and page tables.
                    for slot_index in sorted({0, batch - 1}):
                        rows = slice(slot_index * 4, (slot_index + 1) * 4)
                        single_fmha = SimpleNamespace(
                            expanded_seq_lens=lengths[rows], schedule_metadata=None
                        )
                        single_attn = SimpleNamespace(
                            is_prefill=True,
                            is_target_verify=True,
                            is_draft_extend=False,
                            kv_cache_kernel_block_id_device=table[
                                slot_index : slot_index + 1
                            ],
                        )
                        single = self.op._get_topk_paged(
                            q[rows],
                            weights[rows],
                            cache,
                            single_fmha,
                            single_attn,
                            q_scale=scales[rows],
                        )
                        torch.testing.assert_close(
                            actual[rows].sort().values,
                            single.sort().values,
                            rtol=0,
                            atol=0,
                        )
                    saved = self.op._decode_pivot
                    self.op._decode_pivot = None
                    native = run()
                    self.op._decode_pivot = saved
                    ll = lengths.view(batch, 4)
                    eligible = (ll[:, 0] >= 16384) & (
                        ll == ll[:, :1] + torch.arange(4, device="cuda")
                    ).all(1)
                    fallback = (~eligible).repeat_interleave(4)
                    torch.testing.assert_close(
                        actual[fallback].sort().values,
                        native[fallback].sort().values,
                        rtol=0,
                        atol=0,
                    )
                    self.assertTrue(
                        ((actual == -1) | ((actual >= 0) & (actual < lengths[:, None])))
                        .all()
                        .item()
                    )
                del graph, captured

    def test_production_dispatch_only_target_verify(self):
        self.op._decode_pivot = self.config
        cache = SimpleNamespace(kv_scale_base=self.cache)
        fmha = SimpleNamespace(expanded_seq_lens=self.lens, schedule_metadata=None)
        for target, draft in [(True, False), (False, True)]:
            attn = SimpleNamespace(
                is_prefill=True,
                is_target_verify=target,
                is_draft_extend=draft,
                kv_cache_kernel_block_id_device=self.table,
            )
            with patch(
                "rtp_llm.models_py.triton_kernels.sparse_mla.decode_pivot.decode_pivot_topk",
                wraps=decode_pivot_topk,
            ) as pivot:
                result = self.op._get_topk_paged(
                    self.q, self.w, cache, fmha, attn, q_scale=self.qs
                )
                self.assertEqual(pivot.call_count, int(target))
                self.assertEqual(tuple(result.shape), (8, 2048))
        # Single-token decode uses the original full-context implementation.
        attn.is_prefill = False
        attn.is_draft_extend = False
        fmha.expanded_seq_lens = self.lens[::4].contiguous()
        fmha.kvlen_d = fmha.expanded_seq_lens
        with patch(
            "rtp_llm.models_py.triton_kernels.sparse_mla.decode_pivot.decode_pivot_topk",
            side_effect=AssertionError("ordinary decode entered PIVOT"),
        ):
            result = self.op._get_topk_paged(
                self.q[::4].contiguous(), self.w[::4].contiguous(), cache, fmha, attn
            )
        self.assertEqual(tuple(result.shape), (2, 2048))

    def test_default_16k_candidates_with_identical_group_queries(self):
        # Exact proxy when the four Q/gate rows match. Repeated physical pages
        # intentionally create ties; compare selected values rather than tie IDs.
        self.q = self.q[::4].repeat_interleave(4, dim=0).contiguous()
        self.qs.fill_(1)
        self.w = self.w[::4].repeat_interleave(4, dim=0).contiguous()
        self.table = self.table.repeat(1, 8)
        self.lens = torch.arange(40000, 40004, dtype=torch.int32, device="cuda").repeat(
            2
        )
        self.op._decode_pivot = DecodePivotConfig()
        cache = SimpleNamespace(kv_scale_base=self.cache)
        fmha = SimpleNamespace(expanded_seq_lens=self.lens, schedule_metadata=None)
        attn = SimpleNamespace(
            is_prefill=True,
            is_target_verify=True,
            is_draft_extend=False,
            kv_cache_kernel_block_id_device=self.table,
        )
        out = self.op._get_topk_paged(
            self.q, self.w, cache, fmha, attn, q_scale=self.qs
        )
        self.op._decode_pivot = None
        native = self.op._get_topk_paged(self.q, self.w, cache, fmha, attn)
        for row in range(8):
            self.assertEqual(out[row].unique().numel(), 2048)
            self.assertTrue(
                ((out[row] >= 0) & (out[row] < self.lens[row])).all().item()
            )
            values = []
            for ids in (out[row], native[row]):
                pages = self.table[row // 4, ids.long() // 64].long()
                keys = self.k.float()[pages, ids.long() % 64]
                scales = self.sf[pages, ids.long() % 64]
                scores = (
                    torch.relu(self.q[row].float() @ keys.T) * self.w[row, :, None]
                ).sum(0) * scales
                values.append(scores.sort().values)
            torch.testing.assert_close(values[0], values[1], rtol=2e-4, atol=2e-4)


class DecodePivotConfigTest(unittest.TestCase):
    def test_opt_in_and_invalid_configuration(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(DecodePivotConfig.from_env(2048))
        with patch.dict(os.environ, {"GLM5_DECODE_PIVOT_MODE": "refine"}, clear=True):
            self.assertEqual(DecodePivotConfig.from_env(2048), DecodePivotConfig())
        for config in [
            DecodePivotConfig(group_size=2),
            DecodePivotConfig(local_window=2),
            DecodePivotConfig(candidates=2048),
            DecodePivotConfig(min_kv_len=8192),
        ]:
            with self.assertRaises(ValueError):
                config.validate(2048)


if __name__ == "__main__":
    unittest.main(verbosity=2)
