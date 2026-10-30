"""PIVOT prefill boundaries, FP8 numerics and production dispatch coverage."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.triton_kernels.sparse_mla.prefill_pivot import (
    PrefillPivotConfig,
    pivot_topk,
    plan_groups,
)


class PrefillPivotPlanTest(unittest.TestCase):
    def test_disabled_by_default_and_validates_enabled_config(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(PrefillPivotConfig.from_env(2048))
        with patch.dict(os.environ, {"GLM5_PREFILL_PIVOT_MODE": "refine"}, clear=True):
            self.assertEqual(PrefillPivotConfig.from_env(2048), PrefillPivotConfig())
        for config in (
            PrefillPivotConfig(group_size=8),
            PrefillPivotConfig(local_window=2),
            PrefillPivotConfig(candidates=2048),
            PrefillPivotConfig(min_kv_len=1024),
        ):
            with self.assertRaises(ValueError):
                config.validate(2048)

    def test_request_cp_jump_short_prefix_offset_and_tail(self):
        ks = torch.tensor([0] * 8 + [100000] * 4 + [0] * 11)
        ke = torch.tensor(
            [40000, 40001, 40002, 40003, 40004, 40005, 60000, 60001]
            + [140000, 140001, 140002, 140003]
            + [100, 101, 102, 103, 70000, 70001, 70002, 70003, 70004, 70005, 70006]
        )
        offsets = -ks.clone()
        offsets[18] = 42
        groups, fallback = plan_groups(ks, ke, offsets, PrefillPivotConfig())
        self.assertEqual(groups.tolist(), [[0, 1, 2, 3], [8, 9, 10, 11]])
        self.assertEqual(fallback.tolist(), list(range(4, 8)) + list(range(12, 23)))

    def test_empty_plan(self):
        empty = torch.empty(0, dtype=torch.int32)
        groups, fallback = plan_groups(empty, empty, empty, PrefillPivotConfig())
        self.assertEqual(tuple(groups.shape), (0, 4))
        self.assertEqual(fallback.numel(), 0)

    def test_decode_and_mtp_never_enter_pivot(self):
        from rtp_llm.models_py.modules.hybrid.indexer import Indexer

        for flags in (
            dict(is_prefill=False),
            dict(is_prefill=True, is_target_verify=True),
            dict(is_prefill=True, is_draft_extend=True),
        ):
            op = SimpleNamespace(_get_topk_paged=Mock(return_value="native"))
            obj = SimpleNamespace(
                indexer_op=op, _prefill_cp_enabled=Mock(side_effect=AssertionError)
            )
            self.assertEqual(
                Indexer._compute_topk(
                    obj, None, None, None, None, SimpleNamespace(**flags), None
                ),
                "native",
            )
            op._get_topk_paged.assert_called_once()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class PrefillPivotGpuTest(unittest.TestCase):
    def setUp(self):
        import deep_gemm
        from rtp_llm.models_py.modules.base.cuda.indexer_op import IndexerOp

        self.deep_gemm = deep_gemm
        with patch.dict(
            os.environ,
            {"GLM5_PREFILL_PIVOT_MODE": "off", "GLM5_INDEXER_POOL_REUSE_STEPS": "0"},
        ):
            self.op = IndexerOp(32, 128, 2048, 64)
        torch.manual_seed(29)

    def inputs(self, rows=12, length=12288):
        q = torch.randn((rows, 32, 128), device="cuda").to(torch.float8_e4m3fn)
        qs = torch.exp2(torch.randint(-4, 3, (rows, 32, 1), device="cuda").float())
        weights = torch.randn((rows, 32), device="cuda") * qs.squeeze(-1) / 32
        k = torch.randn((length, 128), device="cuda").to(torch.float8_e4m3fn)
        sf = torch.rand(length, device="cuda") + 0.5
        ks = torch.zeros(rows, dtype=torch.int32, device="cuda")
        ke = torch.arange(
            length - rows + 1, length + 1, device="cuda", dtype=torch.int32
        )
        return q, qs, weights, (k, sf), ks, ke

    def logits(self, q, kv, w, ks, ke):
        return self.deep_gemm.fp8_mqa_logits(q, kv, w, ks, ke, clean_logits=False)

    def run_pivot(self, inputs, config, offsets=None):
        q, qs, w, kv, ks, ke = inputs
        return pivot_topk(
            q,
            qs,
            w,
            kv,
            ks,
            ke,
            torch.zeros_like(ks) if offsets is None else offsets,
            2048,
            config,
            self.logits,
            self.op._run_prefill_topk,
        )

    def test_proxy_uses_unscaled_gate_and_real_query(self):
        from rtp_llm.models_py.triton_kernels.sparse_mla.prefill_pivot_kernels import (
            build_proxy,
        )

        q, qs, w, *_ = self.inputs(rows=8)
        for g in (2, 4):
            pq, pw = build_proxy(q, qs.squeeze(-1).contiguous(), w.contiguous(), g)
            mean_q = (q.float() * qs).reshape(-1, g, 32, 128).mean(1)
            mean_w = (w / qs.squeeze(-1)).reshape(-1, g, 32).mean(1)
            scale = torch.exp2(
                torch.ceil(torch.log2(mean_q.abs().amax(-1).clamp_min(1.0e-4) / 448))
            )
            expected_q = (mean_q / scale[..., None]).to(torch.float8_e4m3fn)
            torch.testing.assert_close(pq.float(), expected_q.float(), rtol=0, atol=0)
            torch.testing.assert_close(pw, mean_w * scale, rtol=1.0e-5, atol=1.0e-7)
        zero_q = torch.zeros_like(q)
        pq, pw = build_proxy(zero_q, qs.squeeze(-1).contiguous(), w, 4)
        self.assertTrue(torch.isfinite(pw).all().item())
        self.assertEqual(pq.float().abs().max().item(), 0)

    def test_fused_gpu_plan_matches_boundary_reference(self):
        for g in (2, 4):
            config = PrefillPivotConfig(group_size=g)
            for n in (0, 1, 3, 7, 23, 2048):
                ks = torch.zeros(n, dtype=torch.int32)
                ke = torch.arange(40000, 40000 + n, dtype=torch.int32)
                offsets = -ks.clone()
                if n > 12:
                    ks[4:8] = 100
                    ke[9:] += 200
                    offsets[11] = 17
                    ke[:g] = torch.arange(100, 100 + g, dtype=torch.int32)
                expected = plan_groups(ks, ke, offsets, config)
                actual = plan_groups(ks.cuda(), ke.cuda(), offsets.cuda(), config)
                for a, e in zip(actual, expected):
                    torch.testing.assert_close(a.cpu(), e)

    def test_fused_recall_nonzero_origin_dirty_logits_and_padding(self):
        from rtp_llm.models_py.triton_kernels.sparse_mla.prefill_pivot_kernels import (
            recall_candidates,
        )

        config = PrefillPivotConfig()
        logits = torch.full((2, 12000), 1.0e9, device="cuda")
        starts = torch.tensor([2000, 6000], device="cuda", dtype=torch.int32)
        ends = torch.tensor([7000, 6003], device="cuda", dtype=torch.int32)
        # Strictly increasing values give an unambiguous reference TopK.
        logits[0, 2000:7000] = torch.arange(5000, device="cuda")
        logits[1, 6000:6003] = torch.tensor([1.0, 3.0, 2.0], device="cuda")
        row_ends = ends[:, None] + 4 + torch.arange(4, device="cuda")
        ids = recall_candidates(logits, starts, ends, row_ends.flatten(), config)
        torch.testing.assert_close(
            ids[0, :4089].sort().values,
            torch.arange(7000 - 4089, 7000, device="cuda", dtype=torch.int32),
        )
        self.assertEqual((ids[1, :4089] >= 0).sum().item(), 3)
        torch.testing.assert_close(
            ids[1, :3].sort().values,
            torch.arange(6000, 6003, device="cuda", dtype=torch.int32),
        )
        for i in range(2):
            torch.testing.assert_close(
                ids[i, 4089:],
                ends[i] + torch.arange(7, device="cuda", dtype=torch.int32),
            )

    def test_fused_remap_invalid_slots_nonfinite_scores_and_offsets(self):
        from rtp_llm.models_py.triton_kernels.sparse_mla.prefill_pivot_kernels import (
            remap_output,
        )

        selected = torch.tensor(
            [[0, 1, 2, 3, -1, 5]] * 4, device="cuda", dtype=torch.int32
        )
        candidates = torch.tensor(
            [[10, 11, 12, 20, -1]], device="cuda", dtype=torch.int32
        )
        backing = torch.ones((4, 8), device="cuda")
        scores = backing[:, :5]  # Compressed logits need not have stride == width.
        scores[0, 1] = float("nan")
        scores[1, 2] = float("inf")
        scores[2, 0] = -float("inf")
        groups = torch.tensor([[3, 1, 4, 0]], device="cuda")
        ks = torch.full((4,), 10, device="cuda", dtype=torch.int32)
        ke = torch.full((4,), 14, device="cuda", dtype=torch.int32)
        offsets = torch.tensor([5, 10, 0, 100, 8], device="cuda", dtype=torch.int32)
        out = torch.full((5, 6), -777, device="cuda", dtype=torch.int32)
        remap_output(selected, scores, candidates, groups, ks, ke, offsets, out, True)
        expected = torch.tensor(
            [
                [5, 6, 7, -1, -1, -1],
                [10, 11, -1, -1, -1, -1],
                [-777] * 6,
                [100, -1, 102, -1, -1, -1],
                [-1, 9, 10, -1, -1, -1],
            ],
            device="cuda",
            dtype=torch.int32,
        )
        torch.testing.assert_close(out, expected)

    def test_large_budget_and_canonicalized_output(self):
        q, qs, w, kv, ks, ke = self.inputs(rows=9, length=24000)
        config = PrefillPivotConfig(candidates=8192, min_kv_len=8192)
        self.op._prefill_pivot = config
        offsets = torch.full_like(ks, 17)
        with patch.dict(os.environ, {"DSV4_INDEXER_TOPK_CANONICALIZE": "0"}):
            plain = self.op._get_prefill_pivot_topk(q, qs, w, kv, ks, ke, offsets)
        with patch.dict(os.environ, {"DSV4_INDEXER_TOPK_CANONICALIZE": "1"}):
            canonical = self.op._get_prefill_pivot_topk(q, qs, w, kv, ks, ke, offsets)
        torch.testing.assert_close(canonical, plain.sort(1).values)
        self.assertTrue(((plain >= 17) & (plain < ke[:, None] + 17)).all().item())

    def test_grouped_candidate_scores_match_production_deepgemm(self):
        from rtp_llm.models_py.triton_kernels.sparse_mla.prefill_pivot_kernels import (
            score_candidates,
        )

        for g in (2, 4):
            q, qs, w, kv, ks, ke = self.inputs(rows=8)
            ks[4:] = 3000
            candidates = torch.randint(
                0, kv[0].shape[0], (8 // g, 129), device="cuda", dtype=torch.int32
            )
            candidates[:, :3] = torch.tensor(
                [-1, kv[0].shape[0] - 1, kv[0].shape[0]], device="cuda"
            )
            actual = score_candidates(q, w, kv, candidates, ks, ke, g)
            dense = self.logits(q, kv, w, ks, ke)
            ids = candidates.repeat_interleave(g, 0).long()
            valid = (ids >= ks[:, None]) & (ids < ke[:, None])
            expected = dense.gather(1, ids.clamp(0, kv[0].shape[0] - 1))
            torch.testing.assert_close(
                actual[valid], expected[valid], rtol=3.0e-5, atol=3.0e-5
            )
            self.assertTrue(torch.isneginf(actual[~valid]).all().item())

    def test_mask_cleans_dirty_ragged_logits(self):
        from rtp_llm.models_py.triton_kernels.sparse_mla.prefill_pivot_kernels import (
            mask_logits_,
        )

        logits = torch.full((3, 2049), 123.0, device="cuda")
        ks = torch.tensor([0, 73, 2000], device="cuda", dtype=torch.int32)
        ke = torch.tensor([1, 1025, 2000], device="cuda", dtype=torch.int32)
        mask_logits_(logits, ks, ke)
        valid = (torch.arange(2049, device="cuda")[None, :] >= ks[:, None]) & (
            torch.arange(2049, device="cuda")[None, :] < ke[:, None]
        )
        self.assertTrue((logits[valid] == 123).all().item())
        self.assertTrue(torch.isneginf(logits[~valid]).all().item())

    def test_native_fallback_and_zero_rows(self):
        inputs = self.inputs(rows=7)
        seen = []

        def native(logits, ks, ke, out, max_len):
            self.op._run_prefill_topk(logits, ks, ke, out, max_len)
            seen.append(out.clone())

        q, qs, w, kv, ks, ke = inputs
        out = pivot_topk(
            q, qs, w, kv, ks, ke, ks, 2048, PrefillPivotConfig(), self.logits, native
        )
        self.assertEqual(len(seen), 1)
        self.assertTrue(torch.equal(out, seen[0]))
        out = pivot_topk(
            q[:0],
            qs[:0],
            w[:0],
            kv,
            ks[:0],
            ke[:0],
            ks[:0],
            2048,
            PrefillPivotConfig(),
            self.logits,
            native,
        )
        self.assertEqual(tuple(out.shape), (0, 2048))

    def test_fused_fallback_write_preserves_padding(self):
        q, qs, w, kv, ks, ke = self.inputs(rows=7, length=2048)
        config = PrefillPivotConfig()
        offsets = torch.full_like(ks, 123)
        expected = self.run_pivot((q, qs, w, kv, ks, ke), config, offsets)
        expected = torch.where(expected >= 0, expected + 123, expected)
        self.op._prefill_pivot = config
        with patch.dict(os.environ, {"DSV4_INDEXER_TOPK_CANONICALIZE": "0"}):
            actual = self.op._get_prefill_pivot_topk(q, qs, w, kv, ks, ke, offsets)
        torch.testing.assert_close(actual, expected)
        self.assertTrue((actual == -1).any().item())

    def test_ragged_offsets_causal_unique_and_window(self):
        from rtp_llm.models_py.triton_kernels.sparse_mla import (
            prefill_pivot_kernels as kernels,
        )

        q, qs, w, kv, ks, ke = self.inputs(rows=11, length=24000)
        ks[4:8] = 12000
        ke[:4] = torch.arange(11000, 11004, device="cuda", dtype=torch.int32)
        ke[4:8] = torch.arange(23000, 23004, device="cuda", dtype=torch.int32)
        config = PrefillPivotConfig(min_kv_len=8192)
        real_score = kernels.score_candidates
        seen = []

        def inspect(q, w, kv, candidates, starts, ends, g):
            seen.append(candidates.clone())
            for group in range(candidates.shape[0]):
                ids = candidates[group]
                self.assertEqual(ids.unique().numel(), config.candidates)
                self.assertTrue((ids >= starts[group * g]).all().item())
                for row in range(g):
                    window = torch.arange(
                        ends[group * g + row] - config.local_window,
                        ends[group * g + row],
                        device="cuda",
                    )
                    self.assertTrue(torch.isin(window, ids).all().item())
            return real_score(q, w, kv, candidates, starts, ends, g)

        with patch.object(kernels, "score_candidates", side_effect=inspect):
            out = self.run_pivot((q, qs, w, kv, ks, ke), config, -ks)
        self.assertEqual(len(seen), 1)
        self.assertTrue(((out >= 0) & (out < (ke - ks)[:, None])).all().item())
        for row in out:
            self.assertEqual(row.unique().numel(), 2048)
        self.op._prefill_pivot = config
        shifted = self.op._get_prefill_pivot_topk(q, qs, w, kv, ks, ke, ks)
        self.assertTrue(
            ((shifted >= ks[:, None]) & (shifted < ke[:, None])).all().item()
        )

    def test_nonzero_ragged_origin_matches_native_output_coordinates(self):
        q, qs, w, kv, ks, ke = self.inputs(rows=9, length=80000)
        q = q[:1].expand_as(q).contiguous()
        qs = torch.ones_like(qs)
        w = w[:1].abs().expand_as(w).contiguous()
        ks[4:] = 40000
        ke[:4] = torch.arange(39997, 40001, device="cuda", dtype=torch.int32)
        ke[4:] = torch.arange(79996, 80001, device="cuda", dtype=torch.int32)
        dense = self.logits(q, kv, w, ks, ke)
        for backend in ("dsv4_per_row", "topk_v3_tie_break"):
            self.op._prefill_topk_backend = backend
            native = torch.empty((9, 2048), device="cuda", dtype=torch.int32)
            self.op._run_prefill_topk(dense, ks, ke, native, 40000)
            threshold = dense.gather(1, native.long() + ks[:, None]).min(1).values
            for budget in (4096, 16384):
                self.op._prefill_pivot = PrefillPivotConfig(candidates=budget)
                for canonicalize in ("0", "1"):
                    with patch.dict(
                        os.environ, {"DSV4_INDEXER_TOPK_CANONICALIZE": canonicalize}
                    ):
                        output = self.op._get_prefill_pivot_topk(
                            q, qs, w, kv, ks, ke, torch.zeros_like(ks)
                        )
                    # Includes two refined request groups and a native tail row.
                    self.assertTrue(
                        ((output >= 0) & (output < (ke - ks)[:, None])).all().item()
                    )
                    scores = dense.gather(1, output.long() + ks[:, None])
                    self.assertTrue((scores >= threshold[:, None] - 1e-4).all().item())

    def test_identical_queries_retain_full_context_topk(self):
        q, qs, w, kv, ks, ke = self.inputs(rows=8)
        q = q[:1].expand_as(q).contiguous()
        qs = torch.ones_like(qs)
        w = w[:1].abs().expand_as(w).contiguous()
        dense = self.logits(q, kv, w, ks, ke)
        columns = torch.arange(dense.shape[1], device="cuda")
        dense.masked_fill_(columns[None, :] >= ke[:, None], -float("inf"))
        cutoff = torch.topk(dense, 2048, dim=1).values[:, -1]
        for backend in ("dsv4_per_row", "topk_v3_tie_break"):
            self.op._prefill_topk_backend = backend
            out = self.run_pivot(
                (q, qs, w, kv, ks, ke), PrefillPivotConfig(min_kv_len=8192)
            )
            selected_scores = dense.gather(1, out.long())
            self.assertTrue((selected_scores >= cutoff[:, None] - 1.0e-4).all().item())

    def test_capture_is_explicitly_rejected(self):
        with patch("torch.cuda.is_current_stream_capturing", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "ENABLE_CUDA_GRAPH=0"):
                self.run_pivot(self.inputs(rows=4), PrefillPivotConfig())

    def test_normal_and_cp_entry_preserve_query_scale_alignment(self):
        from rtp_llm.models_py.modules.base.cuda import indexer_op as module

        q, qs, w, kv, ks, ke = self.inputs(rows=12)
        self.op._prefill_pivot = PrefillPivotConfig(min_kv_len=8192)
        inputs = SimpleNamespace(is_prefill=True, cu_kv_seqlens=None)
        params = SimpleNamespace(
            prefill_total_kv_tokens=kv[0].shape[0], ks=ks, ke=ke, topk_indices_offset=ks
        )

        def gather(cache, key, scale, table, cu):
            key.copy_(kv[0])
            scale.view(torch.float32).copy_(kv[1][:, None])

        with patch.object(
            module, "_prefill_physical_block_table", return_value=None
        ), patch.object(self.op, "_kv_cache_blocks", return_value=None), patch.object(
            module.rtp_llm_ops, "cp_gather_indexer_k_quant_cache", side_effect=gather
        ), patch.object(
            self.op, "_get_prefill_pivot_topk", wraps=self.op._get_prefill_pivot_topk
        ) as entry:
            normal = self.op._get_topk_ragged(
                q, w[..., None], None, params, inputs, q_scale=qs
            )
            torch.testing.assert_close(entry.call_args.args[1], qs)
            for identity in (False, True):
                ids = (
                    torch.arange(8, device="cuda")
                    if identity
                    else torch.tensor([8, 9, 10, 11, 0, 1, 2, 3], device="cuda")
                )
                cp = self.op._get_topk_ragged_cp(
                    q,
                    w[..., None],
                    None,
                    params,
                    inputs,
                    ids,
                    None,
                    kv[0].shape[0],
                    ks[ids],
                    ke[ids],
                    ke[ids] - ks[ids],
                    ks[ids],
                    total_local_ids_is_identity=identity,
                    q_scale=qs,
                )
                torch.testing.assert_close(entry.call_args.args[1], qs[ids])
                # Native TopK ordering can vary; the selected IDs must agree.
                torch.testing.assert_close(
                    cp.sort(1).values, normal[ids].sort(1).values
                )

    def test_cp_nonzero_request_origin_matches_native(self):
        from rtp_llm.models_py.modules.base.cuda import indexer_op as module

        q, qs, w, kv, ks, ke = self.inputs(rows=12, length=80000)
        q = q[:1].expand_as(q).contiguous()
        qs = torch.ones_like(qs)
        w = w[:1].abs().expand_as(w).contiguous()
        ks[4:] = 40000
        ke[:4] = torch.arange(39997, 40001, device="cuda", dtype=torch.int32)
        ke[4:] = torch.arange(79993, 80001, device="cuda", dtype=torch.int32)
        inputs = SimpleNamespace(is_prefill=True, cu_kv_seqlens=None)
        params = SimpleNamespace(prefill_max_kv_len=40000)

        def gather(cache, key, scale, table, cu):
            key.copy_(kv[0])
            scale.view(torch.float32).copy_(kv[1][:, None])

        with patch.object(
            module, "_prefill_physical_block_table", return_value=None
        ), patch.object(self.op, "_kv_cache_blocks", return_value=None), patch.object(
            module.rtp_llm_ops, "cp_gather_indexer_k_quant_cache", side_effect=gather
        ):
            for identity in (False, True):
                ids = (
                    torch.arange(12, device="cuda")
                    if identity
                    else torch.tensor(
                        [8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7], device="cuda"
                    )
                )
                args = (
                    q,
                    w[..., None],
                    None,
                    params,
                    inputs,
                    ids,
                    None,
                    kv[0].shape[0],
                    ks[ids],
                    ke[ids],
                    ke[ids] - ks[ids],
                    torch.zeros_like(ks)[ids],
                )
                self.op._prefill_pivot = None
                native = self.op._get_topk_ragged_cp(
                    *args, total_local_ids_is_identity=identity, q_scale=qs
                )
                self.op._prefill_pivot = PrefillPivotConfig(candidates=16384)
                actual = self.op._get_topk_ragged_cp(
                    *args, total_local_ids_is_identity=identity, q_scale=qs
                )
                torch.testing.assert_close(actual.sort(1).values, native.sort(1).values)

    def test_chunked_long_context_and_tail(self):
        q, qs, w, kv, ks, ke = self.inputs(rows=13, length=131072)
        self.op._prefill_pivot = PrefillPivotConfig()
        with patch.dict(os.environ, {"DSV4_FP8_INDEXER_SCORE_CHUNK_ROWS": "8"}):
            out = self.op._get_prefill_pivot_topk(q, qs, w, kv, ks, ke, ks)
        self.assertEqual(tuple(out.shape), (13, 2048))
        self.assertTrue(((out >= 0) & (out < ke[:, None])).all().item())

    def test_zero_local_cp_still_assembles_sharded_k(self):
        from rtp_llm.models_py.modules.base.cuda import indexer_op as module
        from rtp_llm.models_py.triton_kernels.sparse_mla import (
            fused_indexer_cp_assemble as assemble,
        )

        self.op._prefill_pivot = PrefillPivotConfig()
        q, qs, w, kv, ks, ke = self.inputs(rows=0, length=8192)
        empty = torch.empty(0, dtype=torch.long, device="cuda")
        with patch.object(
            module, "_prefill_physical_block_table", return_value=None
        ), patch.object(assemble, "fused_copy_and_assemble_indexer_k") as collective:
            out = self.op._get_topk_ragged_cp(
                q,
                w[..., None],
                None,
                None,
                SimpleNamespace(),
                empty,
                None,
                kv[0].shape[0],
                ks,
                ke,
                ke,
                ks,
                kv_cache_sharded=True,
                cp_size=8,
                indexer_cp_plan=SimpleNamespace(total_actual_local_T=0),
                indexer_cp_local_cu=empty,
                q_scale=qs,
            )
            collective.assert_called_once()
        self.assertEqual(tuple(out.shape), (0, 2048))


if __name__ == "__main__":
    unittest.main()
