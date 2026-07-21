import os
from types import SimpleNamespace
from unittest import SkipTest, TestCase, main
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.base.cuda.decode_indexer_pool import (
    DecodeIndexerPool,
    DecodeIndexerPoolConfig,
)
from rtp_llm.models_py.modules.base.cuda.indexer_op import IndexerOp
from rtp_llm.ops.compute_ops import rtp_llm_ops


POOL_ENV = {
    "RTP_LLM_DECODE_INDEXER_POOL_PROFILE": None,
    "RTP_LLM_DECODE_INDEXER_POOL_Q_MODE": None,
    "RTP_LLM_DECODE_INDEXER_POOL_ASYNC_REFRESH": None,
    "RTP_LLM_DECODE_INDEXER_POOL_STATE_TTL": None,
    "RTP_LLM_DECODE_INDEXER_POOL_SIZE": None,
    "RTP_LLM_DECODE_INDEXER_POOL_MAX_SIZE": None,
    "RTP_LLM_DECODE_INDEXER_SOURCE_CHUNKS": None,
    "RTP_LLM_DECODE_INDEXER_POOL_MIN_KV_LENGTH": None,
}


def _pool_env(**values):
    env = {key: value for key, value in POOL_ENV.items() if value is not None}
    env.update(values)
    return patch.dict(os.environ, env, clear=False)


class DecodeIndexerPoolConfigTest(TestCase):
    def setUp(self) -> None:
        self._old_env = {key: os.environ.pop(key, None) for key in POOL_ENV}

    def tearDown(self) -> None:
        for key in POOL_ENV:
            os.environ.pop(key, None)
        for key, value in self._old_env.items():
            if value is not None:
                os.environ[key] = value

    def test_default_is_disabled(self):
        config = DecodeIndexerPoolConfig.from_env()
        self.assertFalse(config.enabled)

    def test_profile_a_refreshes_one_rolling_chunk_each_step(self):
        with _pool_env(RTP_LLM_DECODE_INDEXER_POOL_PROFILE="A"):
            config = DecodeIndexerPoolConfig.from_env()

        self.assertTrue(config.enabled)
        self.assertEqual(config.q_mode, "rolling")
        self.assertEqual(config.refresh_lead, 8)
        self.assertEqual(config.max_recent_tokens, 16)
        self.assertEqual(
            [config.refresh_chunks(step) for step in range(8)],
            [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,)],
        )

    def test_profile_a_8k_pool_uses_four_step_interval(self):
        with _pool_env(
            RTP_LLM_DECODE_INDEXER_POOL_PROFILE="A",
            RTP_LLM_DECODE_INDEXER_POOL_SIZE="8192",
        ):
            config = DecodeIndexerPoolConfig.from_env()

        self.assertEqual(config.min_kv_length, 64 * 1024)
        self.assertEqual(config.pool_size, 8192)
        self.assertEqual(config.chunks, 4)
        self.assertEqual(config.interval, 4)
        self.assertEqual(config.refresh_lead, 4)
        self.assertEqual(
            [config.refresh_chunks(step) for step in range(4)],
            [(0,), (1,), (2,), (3,)],
        )

    def test_profile_b_refreshes_two_fixed_anchor_chunks_from_phase_four(self):
        with _pool_env(RTP_LLM_DECODE_INDEXER_POOL_PROFILE="B"):
            config = DecodeIndexerPoolConfig.from_env()

        self.assertEqual(config.q_mode, "fixed")
        self.assertEqual(config.anchor_phase, 4)
        self.assertEqual(config.refresh_lead, 4)
        self.assertEqual(config.max_recent_tokens, 12)
        self.assertEqual(
            [config.refresh_chunks(step) for step in range(8)],
            [(), (), (), (), (0, 1), (2, 3), (4, 5), (6, 7)],
        )

    def test_append_profile_uses_8k_initial_16k_max_and_16_chunks(self):
        with _pool_env(
            RTP_LLM_DECODE_INDEXER_POOL_PROFILE="APPEND",
            RTP_LLM_DECODE_INDEXER_POOL_SIZE="8192",
            RTP_LLM_DECODE_INDEXER_POOL_MAX_SIZE="16384",
            RTP_LLM_DECODE_INDEXER_SOURCE_CHUNKS="16",
        ):
            config = DecodeIndexerPoolConfig.from_env()

        self.assertEqual(config.profile, "APPEND")
        self.assertEqual(config.pool_size, 8192)
        self.assertEqual(config.max_pool_size, 16384)
        self.assertEqual(config.interval, 16)
        self.assertEqual(config.chunks, 16)
        self.assertFalse(config.async_refresh)

    def test_q_mode_can_override_a_profile(self):
        with _pool_env(
            RTP_LLM_DECODE_INDEXER_POOL_PROFILE="A",
            RTP_LLM_DECODE_INDEXER_POOL_Q_MODE="fixed",
        ):
            config = DecodeIndexerPoolConfig.from_env()

        self.assertEqual(config.q_mode, "fixed")
        self.assertEqual(config.anchor_phase, 0)

    def test_invalid_profile_is_rejected(self):
        with _pool_env(RTP_LLM_DECODE_INDEXER_POOL_PROFILE="C"):
            with self.assertRaisesRegex(ValueError, "OFF, A, B, or APPEND"):
                DecodeIndexerPoolConfig.from_env()


def _cuda_deep_gemm_available() -> bool:
    try:
        if not torch.cuda.is_available():
            return False
        import deep_gemm  # noqa: F401

        return True
    except ImportError:
        return False


class DecodeIndexerCandidateScoreTest(TestCase):
    def setUp(self) -> None:
        if not _cuda_deep_gemm_available():
            raise SkipTest("CUDA and deep_gemm are required")
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        torch.manual_seed(2026)

    def test_full_candidate_set_matches_exact_paged_topk(self):
        topk = 2048
        seq_len = 4096
        page_size = 64
        n_heads = 32
        head_dim = 128
        op = IndexerOp(
            index_n_heads=n_heads,
            index_head_dim=head_dim,
            index_topk=topk,
            rope_head_dim=64,
            blocksize=page_size,
            block_size=128,
        )

        page_count = seq_len // page_size
        packed_stride = head_dim + 4
        cache = torch.zeros(
            page_count,
            page_size,
            1,
            packed_stride,
            dtype=torch.uint8,
            device=self.device,
        )
        keys = torch.randn(
            seq_len, head_dim, dtype=torch.bfloat16, device=self.device
        )
        slots = torch.arange(seq_len, dtype=torch.long, device=self.device)
        rtp_llm_ops.indexer_k_quant_and_cache(
            keys,
            cache.view(page_count, page_size, packed_stride),
            slots,
            128,
            "ue8m0",
        )

        q_fp8 = torch.randn(
            1, n_heads, head_dim, dtype=torch.bfloat16, device=self.device
        ).to(torch.float8_e4m3fn)
        weights = torch.rand(1, n_heads, dtype=torch.float32, device=self.device)
        block_table = torch.arange(
            page_count, dtype=torch.int32, device=self.device
        ).view(1, page_count)
        lengths = torch.tensor([seq_len], dtype=torch.int32, device=self.device)

        _, exact_topk = op._score_paged_exact(
            q_fp8, weights, cache, block_table, lengths
        )
        # A monotonic full candidate set preserves the raw page bytes even when
        # the implementation incorrectly treats partitioned [all K][all scale]
        # storage as token-major [K|scale]. Permute it to exercise real gathers.
        candidates = torch.randperm(
            seq_len, dtype=torch.long, device=self.device
        ).view(1, seq_len)
        with patch.dict(
            os.environ,
            {"RTP_LLM_DECODE_INDEXER_FUSED_SPARSE_SCORE": "1"},
        ), patch.object(
            op,
            "_materialize_paged_candidates",
            side_effect=AssertionError("fused score unexpectedly materialized KV"),
        ):
            candidate_topk = op._score_paged_candidates(
                q_fp8,
                weights,
                cache,
                block_table,
                candidates,
                lengths,
                "main",
            )
        torch.testing.assert_close(
            torch.sort(candidate_topk, dim=1).values,
            torch.sort(exact_topk, dim=1).values,
            rtol=0,
            atol=0,
        )

    def test_page_window_refresh_matches_materialized_candidates_in_graph(self):
        topk = 2048
        seq_len = 4096
        page_size = 64
        n_heads = 32
        head_dim = 128
        op = IndexerOp(
            index_n_heads=n_heads,
            index_head_dim=head_dim,
            index_topk=topk,
            rope_head_dim=64,
            blocksize=page_size,
            block_size=128,
        )

        page_count = seq_len // page_size
        packed_stride = head_dim + 4
        cache = torch.zeros(
            page_count,
            page_size,
            1,
            packed_stride,
            dtype=torch.uint8,
            device=self.device,
        )
        keys = torch.randn(
            seq_len, head_dim, dtype=torch.bfloat16, device=self.device
        )
        slots = torch.arange(seq_len, dtype=torch.long, device=self.device)
        rtp_llm_ops.indexer_k_quant_and_cache(
            keys, cache.view(page_count, page_size, packed_stride), slots, 128, "ue8m0"
        )

        q_fp8 = torch.randn(
            1, n_heads, head_dim, dtype=torch.bfloat16, device=self.device
        ).to(torch.float8_e4m3fn)
        weights = torch.rand(1, n_heads, dtype=torch.float32, device=self.device)
        block_table = torch.randperm(
            page_count, dtype=torch.int32, device=self.device
        ).view(1, page_count)
        start = 13
        candidate_width = 3000
        candidates = torch.arange(
            start,
            start + candidate_width,
            dtype=torch.long,
            device=self.device,
        ).view(1, candidate_width)
        lengths = torch.tensor(
            [candidate_width], dtype=torch.int32, device=self.device
        )

        candidate_cache, candidate_table, padded_width = (
            op._materialize_paged_candidates(
                cache, block_table, candidates, lengths
            )
        )
        expected = op._score_materialized_candidates(
            q_fp8,
            weights,
            candidate_cache,
            candidate_table,
            padded_width,
            candidates,
            lengths,
            "test_ref",
        )

        def score_page_window():
            return op._score_paged_candidate_range(
                q_fp8,
                weights,
                cache,
                block_table,
                candidates,
                lengths,
                "refresh",
            )

        for _ in range(3):
            score_page_window()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = score_page_window()
        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(
            torch.sort(actual, dim=1).values,
            torch.sort(expected, dim=1).values,
            rtol=0,
            atol=0,
        )


class DecodeIndexerCudaGraphTest(TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is required")
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        self.topk_workspace = torch.empty(
            1 << 20, dtype=torch.uint8, device=self.device
        )

    def _fused_pool_topk(
        self,
        logits,
        lengths,
        max_seq_len,
        lane,
        pool,
        pool_lengths,
        chunk,
        chunk_lengths,
        inverse_map,
        pool_slots,
        active_mask,
    ):
        del lane
        output = torch.empty(
            (logits.shape[0], 2048), dtype=torch.int32, device=logits.device
        )
        rtp_llm_ops.dsv4_persistent_topk_pool(
            logits,
            lengths,
            output,
            self.topk_workspace,
            max_seq_len,
            pool,
            pool_lengths,
            chunk,
            chunk_lengths,
            inverse_map,
            pool_slots,
            active_mask,
        )
        return output

    def test_first_graph_call_initializes_slots_and_replays(self):
        config = DecodeIndexerPoolConfig(
            profile="A",
            min_kv_length=16 * 1024,
            chunks_per_step=1,
            refresh_lead=8,
            q_mode="rolling",
            anchor_phase=0,
            async_refresh=False,
        )
        pool = DecodeIndexerPool(
            config,
            index_topk=2048,
            index_n_heads=1,
            index_head_dim=4,
        )
        q_fp8 = torch.zeros((1, 1, 4), dtype=torch.float16, device=self.device)
        weights = torch.ones((1, 1), dtype=torch.float32, device=self.device)
        block_table = torch.zeros((1, 512), dtype=torch.int32, device=self.device)
        lengths = torch.tensor([16385], dtype=torch.int32, device=self.device)
        attention_inputs = SimpleNamespace(
            is_cuda_graph=True,
            indexer_pool_graph_mode=True,
            is_speculative=False,
            is_target_verify=False,
            decode_indexer_pool_slot=torch.tensor(
                [0], dtype=torch.int32, device=self.device
            ),
            decode_step=torch.tensor([1], dtype=torch.int32, device=self.device),
            decode_kv_length=torch.tensor(
                [16385], dtype=torch.int32, device=self.device
            ),
        )

        def candidate_score(q, w, table, candidates, lens, lane):
            return candidates[:, :2048].to(torch.int32)

        def prepare_candidates(table, candidates, lens):
            cache = torch.empty(
                (candidates.shape[0], 1), dtype=torch.uint8, device=self.device
            )
            return cache, table[:, :1], candidates.shape[1]

        def score_materialized(q, w, cache, table, width, candidates, lens, lane):
            return candidates[:, :2048].to(torch.int32)

        def run_graph_path():
            return pool._try_compute_cuda_graph(
                q_fp8,
                weights,
                block_table,
                lengths,
                attention_inputs,
                candidate_score,
                prepare_candidates,
                score_materialized,
                32768,
            )

        for _ in range(3):
            run_graph_path()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run_graph_path()
        attention_inputs.decode_step.fill_(2)
        attention_inputs.decode_kv_length.fill_(16386)
        graph.replay()
        torch.cuda.synchronize()

        self.assertEqual(captured.shape, (1, 2048))
        self.assertEqual(captured.dtype, torch.int32)

    def test_exact_graph_bootstraps_8k_pool_above_64k(self):
        config = DecodeIndexerPoolConfig(
            profile="A",
            min_kv_length=64 * 1024,
            interval=4,
            pool_size=8 * 1024,
            chunks=4,
            chunks_per_step=1,
            refresh_lead=4,
            q_mode="rolling",
            anchor_phase=0,
            async_refresh=False,
        )
        pool = DecodeIndexerPool(
            config,
            index_topk=2048,
            index_n_heads=1,
            index_head_dim=4,
        )
        graph_max_seq_len = 64 * 1024 + 64
        logits = torch.randn(
            (1, graph_max_seq_len), dtype=torch.float32, device=self.device
        )
        q_fp8 = torch.zeros((1, 1, 4), dtype=torch.float16, device=self.device)
        weights = torch.ones((1, 1), dtype=torch.float32, device=self.device)
        attention_inputs = SimpleNamespace(
            indexer_pool_graph_mode=False,
            indexer_pool_bootstrap_graph_mode=True,
            decode_indexer_pool_slot=torch.tensor(
                [0], dtype=torch.int32, device=self.device
            ),
            decode_kv_length=torch.tensor(
                [64 * 1024 + 1], dtype=torch.int32, device=self.device
            ),
        )

        def select_topk(values, lengths, max_seq_len, lane):
            return torch.topk(values, 2048, dim=1).indices.to(torch.int32)

        def run_exact_bootstrap():
            pool.bootstrap_cuda_graph_exact(
                logits,
                q_fp8,
                weights,
                attention_inputs,
                select_topk,
                graph_max_seq_len,
            )

        for _ in range(2):
            run_exact_bootstrap()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_exact_bootstrap()
        attention_inputs.decode_kv_length.fill_(64 * 1024 + 2)
        graph.replay()
        torch.cuda.synchronize()

        assert pool._pools is not None
        assert pool._graph_coverage is not None
        self.assertEqual(tuple(pool._pools.shape), (2, 2, 4, 2048))
        self.assertEqual(pool._graph_coverage[0].tolist(), [64 * 1024 + 2] * 2)

    def test_normal_exact_graph_does_not_run_bootstrap(self):
        config = DecodeIndexerPoolConfig(
            profile="A",
            min_kv_length=64 * 1024,
            interval=4,
            pool_size=8 * 1024,
            chunks=4,
            chunks_per_step=1,
            refresh_lead=4,
            q_mode="rolling",
            anchor_phase=0,
            async_refresh=False,
        )
        pool = DecodeIndexerPool(
            config,
            index_topk=2048,
            index_n_heads=1,
            index_head_dim=4,
        )
        logits = torch.zeros((1, 128), dtype=torch.float32, device=self.device)
        q_fp8 = torch.zeros((1, 1, 4), dtype=torch.float16, device=self.device)
        weights = torch.ones((1, 1), dtype=torch.float32, device=self.device)
        attention_inputs = SimpleNamespace(
            indexer_pool_graph_mode=False,
            indexer_pool_bootstrap_graph_mode=False,
        )

        pool.bootstrap_cuda_graph_exact(
            logits,
            q_fp8,
            weights,
            attention_inputs,
            lambda *args: (_ for _ in ()).throw(AssertionError("unexpected TopK")),
            128,
        )

        self.assertIsNone(pool._pools)

    def test_append_profile_bootstrap_and_steady_graph_replay(self):
        config = DecodeIndexerPoolConfig(
            profile="APPEND",
            min_kv_length=16 * 1024,
            interval=16,
            pool_size=8 * 1024,
            max_pool_size=16 * 1024,
            chunks=16,
            async_refresh=False,
        )
        pool = DecodeIndexerPool(
            config,
            index_topk=2048,
            index_n_heads=1,
            index_head_dim=4,
        )
        graph_max_seq_len = 32 * 1024
        base_logits = torch.randn(
            (1, graph_max_seq_len), dtype=torch.float32, device=self.device
        )
        logits = torch.empty_like(base_logits)
        q_fp8 = torch.zeros((1, 1, 4), dtype=torch.float16, device=self.device)
        weights = torch.ones((1, 1), dtype=torch.float32, device=self.device)
        block_table = torch.zeros((1, 512), dtype=torch.int32, device=self.device)
        lengths = torch.tensor(
            [graph_max_seq_len], dtype=torch.int32, device=self.device
        )
        attention_inputs = SimpleNamespace(
            is_cuda_graph=True,
            indexer_pool_graph_mode=False,
            indexer_pool_bootstrap_graph_mode=True,
            is_speculative=False,
            is_target_verify=False,
            decode_indexer_pool_slot=torch.tensor(
                [0], dtype=torch.int32, device=self.device
            ),
            decode_step=torch.tensor([0], dtype=torch.int32, device=self.device),
            decode_kv_length=torch.tensor(
                [graph_max_seq_len], dtype=torch.int32, device=self.device
            ),
        )

        def select_topk(values, topk_lengths, max_seq_len, lane):
            return torch.topk(values, 2048, dim=1).indices.to(torch.int32)

        def bootstrap():
            logits.copy_(base_logits)
            exact_topk = select_topk(
                logits, lengths, graph_max_seq_len, "main"
            )
            pool.bootstrap_cuda_graph_exact(
                logits,
                q_fp8,
                weights,
                attention_inputs,
                select_topk,
                graph_max_seq_len,
                exact_topk,
            )

        bootstrap()
        torch.cuda.synchronize()
        assert pool._append_pools is not None
        assert pool._append_pool_lengths is not None
        expected_top8k = torch.topk(
            base_logits, 8 * 1024, dim=1, sorted=False
        ).indices
        self.assertEqual(pool._append_pool_lengths[0].item(), 8 * 1024)
        self.assertEqual(
            set(pool._append_pools[0, : 8 * 1024].cpu().tolist()),
            set(expected_top8k[0].cpu().tolist()),
        )

        attention_inputs.indexer_pool_graph_mode = True
        attention_inputs.indexer_pool_bootstrap_graph_mode = False
        attention_inputs.decode_step.fill_(1)

        def pool_chunk_score(
            q,
            w,
            table,
            global_pool,
            global_lengths,
            chunk,
            chunk_lengths,
            slots,
            candidate_lengths,
        ):
            width = global_pool.shape[1] + chunk.shape[1]
            positions = torch.arange(width, device=self.device).unsqueeze(0)
            current_lengths = global_lengths.index_select(0, slots).unsqueeze(1)
            candidate_lengths.copy_(
                global_lengths.index_select(0, slots) + chunk_lengths
            )
            return -(positions - current_lengths).abs().to(torch.float32)

        def unused(*args):
            raise AssertionError("legacy APPEND callback was used")

        def steady():
            return pool._try_compute_cuda_graph(
                q_fp8,
                weights,
                block_table,
                lengths,
                attention_inputs,
                unused,
                unused,
                unused,
                graph_max_seq_len,
                pool_chunk_score,
                select_topk,
                self._fused_pool_topk,
            )

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = steady()
        attention_inputs.decode_step.fill_(2)
        graph.replay()
        torch.cuda.synchronize()

        self.assertEqual(result.shape, (1, 2048))
        self.assertGreater(pool._append_pool_lengths[0].item(), 8 * 1024)

    def test_append_hybrid_graph_routes_bootstrap_and_sparse_per_row(self):
        config = DecodeIndexerPoolConfig(
            profile="APPEND",
            min_kv_length=4 * 1024,
            interval=16,
            pool_size=8 * 1024,
            max_pool_size=16 * 1024,
            chunks=16,
            async_refresh=False,
        )
        pool = DecodeIndexerPool(
            config,
            index_topk=2048,
            index_n_heads=1,
            index_head_dim=4,
        )
        batch_size = 2
        graph_max_seq_len = 16 * 1024
        base_logits = torch.randn(
            (batch_size, graph_max_seq_len),
            dtype=torch.float32,
            device=self.device,
        )
        q_fp8 = torch.zeros(
            (batch_size, 1, 4), dtype=torch.float16, device=self.device
        )
        weights = torch.ones(
            (batch_size, 1), dtype=torch.float32, device=self.device
        )
        block_table = torch.zeros(
            (batch_size, graph_max_seq_len // 64),
            dtype=torch.int32,
            device=self.device,
        )
        lengths = torch.full(
            (batch_size,),
            graph_max_seq_len,
            dtype=torch.int32,
            device=self.device,
        )
        attention_inputs = SimpleNamespace(
            is_cuda_graph=True,
            indexer_pool_graph_mode=False,
            indexer_pool_bootstrap_graph_mode=True,
            is_speculative=False,
            is_target_verify=False,
            decode_indexer_pool_slot=torch.tensor(
                [0, 1], dtype=torch.int32, device=self.device
            ),
            decode_indexer_pool_bootstrap_mask=torch.tensor(
                [1, 1], dtype=torch.int32, device=self.device
            ),
            decode_step=torch.tensor(
                [0, 1], dtype=torch.int32, device=self.device
            ),
            decode_kv_length=lengths.clone(),
        )

        def select_topk(values, topk_lengths, max_seq_len, lane):
            return torch.topk(values, 2048, dim=1).indices.to(torch.int32)

        initial_logits = base_logits.clone()
        initial_topk = select_topk(
            initial_logits, lengths, graph_max_seq_len, "main"
        )
        pool.bootstrap_cuda_graph_exact(
            initial_logits,
            q_fp8,
            weights,
            attention_inputs,
            select_topk,
            graph_max_seq_len,
            initial_topk,
        )
        assert pool._append_pool_lengths is not None
        self.assertEqual(pool._append_pool_lengths[:2].tolist(), [8192, 8192])

        observed_exact_lengths = []
        observed_chunk_lengths = []

        def exact_score(q, w, table, exact_lengths):
            observed_exact_lengths.append(exact_lengths)
            logits = base_logits.clone()
            return logits, select_topk(
                logits, exact_lengths, graph_max_seq_len, "main"
            )

        def pool_chunk_score(
            q,
            w,
            table,
            global_pool,
            global_lengths,
            chunk,
            chunk_lengths,
            slots,
            candidate_lengths,
        ):
            observed_chunk_lengths.append(chunk_lengths)
            width = global_pool.shape[1] + chunk.shape[1]
            positions = torch.arange(width, device=self.device).unsqueeze(0)
            current_lengths = global_lengths.index_select(0, slots).unsqueeze(1)
            candidate_lengths.copy_(
                torch.clamp(
                    global_lengths.index_select(0, slots) + chunk_lengths,
                    min=1,
                )
            )
            return -(positions - current_lengths).abs().to(torch.float32)

        def unused(*args):
            raise AssertionError("legacy APPEND callback was used")

        attention_inputs.decode_indexer_pool_bootstrap_mask.copy_(
            torch.tensor([1, 0], dtype=torch.int32, device=self.device)
        )

        def hybrid():
            result = pool.try_compute(
                q_fp8,
                weights,
                block_table,
                lengths,
                attention_inputs,
                exact_score,
                unused,
                unused,
                unused,
                select_topk,
                pool_chunk_score,
                graph_max_seq_len,
                self._fused_pool_topk,
            )
            assert result is not None
            return result

        hybrid()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = hybrid()

        attention_inputs.decode_indexer_pool_bootstrap_mask.copy_(
            torch.tensor([0, 1], dtype=torch.int32, device=self.device)
        )
        attention_inputs.decode_step.add_(1)
        graph.replay()
        torch.cuda.synchronize()

        expected_exact = torch.topk(
            base_logits[1], 2048, sorted=True
        ).indices.to(torch.int32)
        torch.testing.assert_close(result[1], expected_exact, rtol=0, atol=0)
        self.assertEqual(observed_exact_lengths[-1].tolist(), [1, graph_max_seq_len])
        self.assertEqual(observed_chunk_lengths[-1].tolist(), [1024, 0])
        self.assertGreater(pool._append_pool_lengths[0].item(), 8192)
        self.assertEqual(pool._append_pool_lengths[1].item(), 8192)


if __name__ == "__main__":
    main()
