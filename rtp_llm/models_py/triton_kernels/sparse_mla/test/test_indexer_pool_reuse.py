import unittest

import torch

from rtp_llm.models_py.triton_kernels.sparse_mla.indexer_pool_reuse import (
    CANDIDATES,
    PAGE,
    PAGE_BYTES,
    POOL,
    TOPK,
    IndexerPool,
)

HEADS = 32


def make_cache(pages, seed=0):
    """Paged FP8 Indexer cache: per page 64 x 128 E4M3 keys, then 64 fp32 scales."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    keys = (torch.randn((pages, PAGE * 128), device="cuda", generator=g) * 2).to(
        torch.float8_e4m3fn
    )
    scales = torch.rand((pages, PAGE), device="cuda", generator=g) + 0.5
    return torch.cat([keys.view(torch.uint8), scales.view(torch.uint8)], dim=1)


def make_query(rows, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = (torch.randn((rows, HEADS, 128), device="cuda", generator=g)).to(
        torch.float8_e4m3fn
    )
    w = torch.randn((rows, HEADS), device="cuda", generator=g) / HEADS
    return q, w


def reference_logits(q, w, cache, table, lengths, max_len):
    """Graph-capturable dense logits: sum_h w_h * relu(q_h . k_t) * scale_t."""
    token = torch.arange(max_len, device="cuda")
    page = table[:, token // PAGE].long()
    base = page * PAGE_BYTES
    flat = cache.view(-1)
    key_offsets = (
        base[..., None]
        + ((token % PAGE) * 128)[None, :, None]
        + torch.arange(128, device="cuda")
    )
    keys = flat[key_offsets].view(torch.float8_e4m3fn).float()
    scale_offsets = (
        base[..., None]
        + (PAGE * 128 + (token % PAGE) * 4)[None, :, None]
        + torch.arange(4, device="cuda")
    )
    scales = flat[scale_offsets].contiguous().view(torch.float32)[..., 0]
    score = torch.einsum("rhd,rld->rhl", q.float(), keys)
    logits = (score.clamp_min(0) * w[:, :, None]).sum(1) * scales
    return torch.where(token[None, :] < lengths[:, None], logits, float("-inf"))


def select_topk(logits, lengths, out, max_len):
    masked = torch.where(
        torch.arange(logits.shape[1], device="cuda")[None, :] < lengths[:, None],
        logits,
        float("-inf"),
    )
    values, indices = masked[:, :max_len].topk(TOPK, dim=1)
    out.copy_(torch.where(values == float("-inf"), -1, indices).int())


class Batch:
    """Requests with private page tables over one cache."""

    def __init__(self, lengths, table_cols, pages, seed=0):
        self.cache = make_cache(pages, seed)
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(pages, generator=g)
        self.tables = (
            perm[: len(lengths) * table_cols]
            .view(len(lengths), table_cols)
            .int()
            .cuda()
        )
        self.lengths = list(lengths)
        self.max_len = table_cols * PAGE

    def inputs(self, order):
        table = self.tables[order].contiguous()
        lengths = torch.tensor(
            [self.lengths[i] for i in order], dtype=torch.int32, device="cuda"
        )
        return table, lengths


def run(pool, batch, order, seed):
    q, w = make_query(len(order), seed)
    table, lengths = batch.inputs(order)
    calls = []

    def full_logits(full_lengths):
        calls.append(full_lengths.clone())
        return reference_logits(q, w, batch.cache, table, full_lengths, batch.max_len)

    out = pool.topk(q, w, batch.cache, table, lengths, full_logits, select_topk)
    exact = torch.empty_like(out)
    select_topk(
        reference_logits(q, w, batch.cache, table, lengths, batch.max_len),
        lengths,
        exact,
        batch.max_len,
    )
    return out, exact, calls[0], (q, w, table, lengths)


def as_sets(t):
    return [set(r[r >= 0].tolist()) for r in t.cpu()]


def near_equal(test, out, exact, slack=4):
    """Same selections up to a few near-tie swaps (summation order differs)."""
    for a, b in zip(as_sets(out), as_sets(exact)):
        test.assertLessEqual(len(a ^ b), 2 * slack)


class IndexerPoolTest(unittest.TestCase):
    def test_refresh_is_exact_then_reuse_selects_from_pool(self):
        batch = Batch([20000, 30000, 17000], table_cols=480, pages=2048)
        pool = IndexerPool(slots=8, reuse_steps=3, min_length=16384, device="cuda")
        out, exact, full, (q, w, table, lengths) = run(pool, batch, [0, 1, 2], seed=10)
        self.assertEqual(full.tolist(), batch.lengths)
        near_equal(self, out, exact)
        # Each pool is that step's top-8192 up to near-ties, so it holds the top-2048.
        logits = reference_logits(q, w, batch.cache, table, lengths, batch.max_len)
        top8k = logits.topk(POOL, dim=1).indices
        for row in range(3):
            slot = (pool.anchor == batch.lengths[row]).nonzero().item()
            members = set(pool.pool[slot].tolist())
            self.assertEqual(len(members), POOL)
            self.assertGreaterEqual(len(members & set(top8k[row].tolist())), POOL - 8)
            self.assertTrue(set(exact[row].tolist()) <= members)

        for step in range(1, 4):
            batch.lengths = [n + 1 for n in batch.lengths]
            out, exact, full, (q, w, table, lengths) = run(
                pool, batch, [0, 1, 2], seed=10 + step
            )
            self.assertEqual(full.tolist(), [0, 0, 0])
            logits = reference_logits(q, w, batch.cache, table, lengths, batch.max_len)
            for row in range(3):
                slot = pool.anchor.tolist().index(batch.lengths[row] - step)
                candidates = pool.pool[slot].tolist() + list(
                    range(batch.lengths[row] - step, batch.lengths[row])
                )
                candidates = torch.tensor(candidates, device="cuda")
                expected = candidates[logits[row, candidates].topk(TOPK).indices]
                # Summation order differs from the reference; only near-ties may swap.
                self.assertLessEqual(
                    len(set(out[row].tolist()) ^ set(expected.tolist())), 4
                )
                # Appended tokens are candidates on reuse steps.
                self.assertTrue(set(out[row].tolist()) <= set(candidates.tolist()))

        batch.lengths = [n + 1 for n in batch.lengths]
        out, exact, full, _ = run(pool, batch, [0, 1, 2], seed=20)
        self.assertEqual(full.tolist(), batch.lengths)
        near_equal(self, out, exact)

    def test_other_pool_sizes(self):
        for pool_size in (4096, 16384):
            with self.subTest(pool_size=pool_size):
                batch = Batch([40000, 23000], table_cols=700, pages=2048, seed=12)
                pool = IndexerPool(
                    slots=4,
                    reuse_steps=2,
                    min_length=16384,
                    device="cuda",
                    pool_size=pool_size,
                )
                out, exact, _, (q, w, table, lengths) = run(
                    pool, batch, [0, 1], seed=40
                )
                near_equal(self, out, exact)
                top = (
                    reference_logits(q, w, batch.cache, table, lengths, batch.max_len)
                    .topk(pool_size, dim=1)
                    .indices
                )
                for row in range(2):
                    slot = pool.anchor.tolist().index(batch.lengths[row])
                    members = set(pool.pool[slot].tolist())
                    self.assertGreaterEqual(
                        len(members & set(top[row].tolist())), pool_size - 8
                    )
                batch.lengths = [n + 1 for n in batch.lengths]
                out, _, full, _ = run(pool, batch, [0, 1], seed=41)
                self.assertEqual(full.tolist(), [0, 0])
                for row in range(2):
                    slot = pool.anchor.tolist().index(batch.lengths[row] - 1)
                    allowed = set(pool.pool[slot].tolist()) | {batch.lengths[row] - 1}
                    self.assertTrue(set(out[row].tolist()) <= allowed)

    def test_candidate_logits_match_reference(self):
        batch = Batch([18000, 25000], table_cols=400, pages=1024, seed=3)
        pool = IndexerPool(slots=4, reuse_steps=15, min_length=16384, device="cuda")
        run(pool, batch, [0, 1], seed=4)
        batch.lengths = [n + 9 for n in batch.lengths]
        q, w = make_query(2, seed=5)
        table, lengths = batch.inputs([1, 0])
        row_slot, rebuild, candidate_lengths = pool.plan(table, lengths)
        self.assertEqual(rebuild.tolist(), [0, 0])
        logits = pool.score_candidates(
            q, w, batch.cache, table, lengths, row_slot, candidate_lengths
        )
        dense = reference_logits(q, w, batch.cache, table, lengths, batch.max_len)
        for row in range(2):
            slot = row_slot[row].item()
            anchor = pool.anchor[slot].item()
            tokens = torch.cat(
                [
                    pool.pool[slot],
                    torch.arange(anchor, anchor + 9, device="cuda", dtype=torch.int32),
                ]
            )
            torch.testing.assert_close(
                logits[row, : POOL + 9], dense[row, tokens.long()], rtol=1e-5, atol=1e-5
            )
            self.assertTrue(torch.isneginf(logits[row, POOL + 9 :]).all())

    def test_reordered_new_and_short_rows(self):
        batch = Batch(
            [20000, 21000, 22000, 9000, 23000], table_cols=400, pages=4096, seed=7
        )
        pool = IndexerPool(slots=8, reuse_steps=15, min_length=16384, device="cuda")
        table, lengths = batch.inputs([0, 1, 2, 3])
        pool.plan(table, lengths)
        batch.lengths = [n + 1 for n in batch.lengths]
        # Row order changes, request 1 leaves, request 4 is new; the short
        # request 3 rebuilds its pool every step.
        table, lengths = batch.inputs([2, 0, 4, 3])
        row_slot, rebuild, candidates = pool.plan(table, lengths)
        self.assertEqual(rebuild.tolist(), [0, 0, batch.lengths[4], batch.lengths[3]])
        self.assertEqual(candidates.tolist(), [CANDIDATES] * 4)
        self.assertEqual(len(set(row_slot.tolist())), 4)
        # A row whose pages changed is a different request.
        batch.lengths = [n + 1 for n in batch.lengths]
        table, lengths = batch.inputs([2, 0, 4])
        table[1, 0] = batch.tables[1, 0]
        _, rebuild, _ = pool.plan(table, lengths)
        self.assertEqual(rebuild.tolist(), [0, batch.lengths[0], 0])

    def test_short_rows_are_exact(self):
        batch = Batch([1500, 5000, 8192, 12000], table_cols=200, pages=1024, seed=8)
        pool = IndexerPool(slots=4, reuse_steps=15, min_length=16384, device="cuda")
        for step in range(3):
            out, exact, full, _ = run(pool, batch, [0, 1, 2, 3], seed=30 + step)
            self.assertEqual(full.tolist(), batch.lengths)
            near_equal(self, out, exact)
            self.assertEqual((out[0] >= 0).sum().item(), batch.lengths[0])
            batch.lengths = [n + 1 for n in batch.lengths]

    def test_slots_recycle_without_running_out(self):
        pool = IndexerPool(slots=4, reuse_steps=2, min_length=16384, device="cuda")
        batch = Batch(
            [17000 + 100 * i for i in range(16)], table_cols=300, pages=4800, seed=9
        )
        for start in range(0, 16, 4):
            table, lengths = batch.inputs(list(range(start, start + 4)))
            row_slot, rebuild, _ = pool.plan(table, lengths)
            self.assertEqual(rebuild.tolist(), lengths.tolist())
            self.assertEqual(sorted(row_slot.tolist()), [0, 1, 2, 3])

    def test_cuda_graph_multi_batch_matches_eager(self):
        # Lengths stay within the 360-page tables for all 12 steps.
        batch = Batch(
            [17000 + 697 * i for i in range(8)], table_cols=360, pages=4096, seed=11
        )
        graph_pool = IndexerPool(
            slots=8, reuse_steps=4, min_length=16384, device="cuda"
        )
        eager_pool = IndexerPool(
            slots=8, reuse_steps=4, min_length=16384, device="cuda"
        )
        graphs = {}
        for rows in (4, 8):
            static = dict(
                q=torch.zeros((rows, HEADS, 128), device="cuda").to(
                    torch.float8_e4m3fn
                ),
                w=torch.zeros((rows, HEADS), device="cuda"),
                table=torch.zeros((rows, 360), dtype=torch.int32, device="cuda"),
                lengths=torch.zeros((rows,), dtype=torch.int32, device="cuda"),
            )

            def step(s=static):
                return graph_pool.topk(
                    s["q"],
                    s["w"],
                    batch.cache,
                    s["table"],
                    s["lengths"],
                    lambda n: reference_logits(
                        s["q"], s["w"], batch.cache, s["table"], n, batch.max_len
                    ),
                    select_topk,
                )

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                step()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = step()
            graphs[rows] = (graph, static, out)
        graph_pool.reset()

        orders = [
            list(range(8)),
            [3, 1, 0, 2],
            list(range(8)),
            [7, 6, 5, 4, 3, 2, 1, 0],
            [0, 2, 4, 6],
        ]
        for step_idx in range(12):
            order = orders[step_idx % len(orders)]
            graph, static, out = graphs[len(order)]
            q, w = make_query(len(order), seed=100 + step_idx)
            table, lengths = batch.inputs(order)
            static["q"].copy_(q)
            static["w"].copy_(w)
            static["table"].copy_(table)
            static["lengths"].copy_(lengths)
            graph.replay()
            expected = eager_pool.topk(
                q,
                w,
                batch.cache,
                table,
                lengths,
                lambda n: reference_logits(q, w, batch.cache, table, n, batch.max_len),
                select_topk,
            )
            self.assertEqual(as_sets(out), as_sets(expected), f"step {step_idx}")
            batch.lengths = [n + 1 for n in batch.lengths]


if __name__ == "__main__":
    unittest.main()
