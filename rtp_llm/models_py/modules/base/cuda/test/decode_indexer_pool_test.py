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
            with self.assertRaisesRegex(ValueError, "OFF, A, or B"):
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
            keys, cache.view(page_count, page_size, packed_stride), slots, 128, "ue8m0"
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


class DecodeIndexerCudaGraphTest(TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is required")
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)

    def test_first_graph_call_initializes_slots_and_replays(self):
        config = DecodeIndexerPoolConfig(
            profile="A",
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


if __name__ == "__main__":
    main()
