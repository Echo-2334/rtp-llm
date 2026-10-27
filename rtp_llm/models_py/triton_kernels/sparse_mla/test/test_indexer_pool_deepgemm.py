"""Indexer pool reuse against the production DeepGEMM logits and TopK kernels."""

import unittest

import torch

from rtp_llm.models_py.triton_kernels.sparse_mla.indexer_pool_reuse import (
    CANDIDATES,
    PAGE,
    POOL,
    TOPK,
    IndexerPool,
)

try:
    import deep_gemm

    from rtp_llm.ops.compute_ops import rtp_llm_ops
except Exception:  # pragma: no cover - CPU or partial installs
    deep_gemm = None

HEADS = 32
WORKSPACE = None


def deepgemm_logits(q, w, cache, table, lengths):
    context = lengths.view(-1, 1)
    schedule = deep_gemm.get_paged_mqa_logits_metadata(
        context, PAGE, deep_gemm.get_num_sms()
    )
    return deep_gemm.fp8_paged_mqa_logits(
        q.view(q.shape[0], 1, HEADS, 128),
        cache.view(-1, PAGE, 1, 132),
        w,
        context,
        table,
        schedule,
        table.shape[1] * PAGE,
        clean_logits=False,
    )


def persistent_topk(logits, lengths, out, max_len):
    rtp_llm_ops.dsv4_persistent_topk(logits, lengths, out, WORKSPACE, TOPK, max_len)


def original(q, w, cache, table, lengths):
    """The Indexer decode path with the pool disabled."""
    out = torch.empty((q.shape[0], TOPK), dtype=torch.int32, device="cuda")
    persistent_topk(
        deepgemm_logits(q, w, cache, table, lengths),
        lengths,
        out,
        table.shape[1] * PAGE,
    )
    return out


def make_inputs(rows, table_cols, pages, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    keys = (torch.randn((pages, PAGE * 128), device="cuda", generator=g) * 2).to(
        torch.float8_e4m3fn
    )
    scales = torch.rand((pages, PAGE), device="cuda", generator=g) + 0.5
    cache = torch.cat(
        [keys.view(torch.uint8), scales.view(torch.uint8)], dim=1
    ).contiguous()
    perm = torch.randperm(pages, device="cuda", generator=g)
    table = perm[: rows * table_cols].view(rows, table_cols).int().contiguous()
    return cache, table


def make_query(rows, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn((rows, HEADS, 128), device="cuda", generator=g).to(
        torch.float8_e4m3fn
    )
    w = torch.randn((rows, HEADS), device="cuda", generator=g) / HEADS
    return q, w


def as_sets(t):
    return [set(r[r >= 0].tolist()) for r in t.cpu()]


@unittest.skipIf(
    deep_gemm is None or not hasattr(deep_gemm, "fp8_paged_mqa_logits"),
    "DeepGEMM fp8_paged_mqa_logits unavailable",
)
class IndexerPoolDeepGemmTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global WORKSPACE
        WORKSPACE = torch.empty(1 << 20, dtype=torch.uint8, device="cuda")

    def test_candidate_logits_match_deepgemm(self):
        cache, table = make_inputs(4, 520, 4096, seed=0)
        lengths = torch.tensor(
            [33000, 17000, 20000, 25000], dtype=torch.int32, device="cuda"
        )
        pool = IndexerPool(slots=8, reuse_steps=15, min_length=16384, device="cuda")
        q, w = make_query(4, seed=1)
        pool.topk(
            q,
            w,
            cache,
            table,
            lengths,
            lambda n: deepgemm_logits(q, w, cache, table, n),
            persistent_topk,
        )
        lengths += 11
        q, w = make_query(4, seed=2)
        row_slot, rebuild, candidate_lengths = pool.plan(table, lengths)
        self.assertEqual(rebuild.tolist(), [0, 0, 0, 0])
        ours = pool.score_candidates(
            q, w, cache, table, lengths, row_slot, candidate_lengths
        )
        dense = deepgemm_logits(q, w, cache, table, lengths)
        for row in range(4):
            slot = row_slot[row].item()
            anchor = pool.anchor[slot].item()
            tokens = torch.cat(
                [
                    pool.pool[slot],
                    torch.arange(anchor, anchor + 11, device="cuda", dtype=torch.int32),
                ]
            ).long()
            torch.testing.assert_close(
                ours[row, : POOL + 11], dense[row, tokens], rtol=2e-5, atol=1e-5
            )

    def test_refresh_matches_original_and_reuse_tracks_it(self):
        rows = 16
        cache, table = make_inputs(rows, 640, 16384, seed=3)
        lengths = torch.randint(17000, 40000, (rows,), dtype=torch.int32, device="cuda")
        pool = IndexerPool(slots=32, reuse_steps=15, min_length=16384, device="cuda")
        recall = []
        for step in range(17):
            q, w = make_query(rows, seed=10 + step)
            out = pool.topk(
                q,
                w,
                cache,
                table,
                lengths,
                lambda n: deepgemm_logits(q, w, cache, table, n),
                persistent_topk,
            )
            exact = original(q, w, cache, table, lengths)
            if step in (0, 16):
                # Triton and DeepGEMM sum heads in different orders: near-ties may swap.
                for a, b in zip(as_sets(out), as_sets(exact)):
                    self.assertLessEqual(len(a ^ b), 8, f"refresh step {step}")
            else:
                hits = [len(a & b) / TOPK for a, b in zip(as_sets(out), as_sets(exact))]
                recall.append(min(hits))
            lengths += 1
        # Random queries share nothing across steps; this only checks that
        # reuse returns valid, in-range, mostly-overlapping selections.
        self.assertGreater(min(recall), 0.0)

    def test_cuda_graph_batch_sizes_match_eager(self):
        cache, table_all = make_inputs(16, 400, 8192, seed=5)
        lengths_all = torch.randint(
            17000, 25000, (16,), dtype=torch.int32, device="cuda"
        )
        graph_pool = IndexerPool(
            slots=16, reuse_steps=3, min_length=16384, device="cuda"
        )
        eager_pool = IndexerPool(
            slots=16, reuse_steps=3, min_length=16384, device="cuda"
        )
        graphs = {}
        for rows in (1, 4, 16):
            s = dict(
                q=torch.zeros((rows, HEADS, 128), device="cuda").to(
                    torch.float8_e4m3fn
                ),
                w=torch.zeros((rows, HEADS), device="cuda"),
                table=torch.zeros((rows, 400), dtype=torch.int32, device="cuda"),
                lengths=torch.zeros((rows,), dtype=torch.int32, device="cuda"),
            )

            def step(s=s):
                return graph_pool.topk(
                    s["q"],
                    s["w"],
                    cache,
                    s["table"],
                    s["lengths"],
                    lambda n: deepgemm_logits(s["q"], s["w"], cache, s["table"], n),
                    persistent_topk,
                )

            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                step()
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = step()
            graphs[rows] = (graph, s, out)
        graph_pool.reset()

        orders = [
            list(range(16)),
            [5, 0, 9, 3],
            [9],
            list(range(15, -1, -1)),
            [3, 5, 9, 0],
        ]
        for i in range(20):
            order = torch.tensor(orders[i % len(orders)], device="cuda")
            graph, s, out = graphs[order.numel()]
            q, w = make_query(order.numel(), seed=50 + i)
            table, lengths = (
                table_all[order].contiguous(),
                lengths_all[order].contiguous(),
            )
            for name, value in dict(q=q, w=w, table=table, lengths=lengths).items():
                s[name].copy_(value)
            graph.replay()
            expected = eager_pool.topk(
                q,
                w,
                cache,
                table,
                lengths,
                lambda n: deepgemm_logits(q, w, cache, table, n),
                persistent_topk,
            )
            self.assertEqual(as_sets(out), as_sets(expected), f"replay {i}")
            lengths_all += 1


if __name__ == "__main__":
    unittest.main()
