import unittest

import torch
from rtp_llm.models_py.triton_kernels.sparse_mla.nvfp4_cache import (
    gather_nvfp4_cache,
    sparse_nvfp4_attention,
    write_nvfp4_cache,
)
from rtp_llm.models_py.triton_kernels.sparse_mla.nvfp4_deferred import (
    DeferredNvfp4State,
)


def decode(cache):
    rows = cache.reshape(-1, cache.shape[-1])
    packed = rows[:, :256].long()
    codes = torch.stack((packed & 15, packed >> 4), -1).reshape(-1, 512)
    lookup = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=cache.device)
    values = lookup[codes & 7] * torch.where(codes & 8 != 0, -1, 1)
    scales = (
        rows[:, 256:288]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .float()
        .repeat_interleave(16, -1)
    )
    outer = rows[:, 288:292].contiguous().view(torch.float32)
    values = values * scales * outer
    n = (cache.shape[-1] - 432) // 16
    centers = cache[:, :, 432:].contiguous().view(torch.bfloat16).reshape(-1, n, 512)
    ids = rows[:, 420].long()
    blocks = torch.arange(len(rows), device=cache.device) // 64
    values += centers[blocks, ids.clamp(max=n - 1)].float() * (ids < n)[:, None]
    return torch.cat(
        (values.bfloat16(), rows[:, 292:420].contiguous().view(torch.bfloat16)), -1
    )


def gather(cache, block, count):
    out = torch.empty((count, 576), device="cuda", dtype=torch.bfloat16)
    gather_nvfp4_cache(
        cache,
        out,
        torch.tensor([[block]], device="cuda", dtype=torch.int32),
        torch.tensor([count], device="cuda", dtype=torch.int32),
        torch.tensor([0], device="cuda", dtype=torch.int32),
        1,
        count,
    )
    return out


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DeferredNvfp4Test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261021)

    def make(self, centers=4, capacity=8):
        cache = torch.zeros(
            (12, 64, 432 + centers * 16), device="cuda", dtype=torch.uint8
        )
        return cache, DeferredNvfp4State(cache, capacity)

    def write(self, cache, state, latent, rope, slots):
        slots = torch.as_tensor(slots, device="cuda", dtype=torch.int64)
        write_nvfp4_cache(latent, rope, cache, slots, deferred_state=state)

    def test_mixed_batch_only_full_blocks_seal(self):
        cache, state = self.make()
        blocks, lengths = [2, 5, 7], [63, 20, 63]
        x = torch.randn((3, 64, 512), device="cuda", dtype=torch.bfloat16)
        rope = torch.randn((3, 64, 64), device="cuda", dtype=torch.bfloat16)
        slots = torch.cat(
            [
                torch.arange(b * 64, b * 64 + n, device="cuda")
                for b, n in zip(blocks, lengths)
            ]
        )
        initial = torch.cat([x[i, :n] for i, n in enumerate(lengths)])
        initial_rope = torch.cat([rope[i, :n] for i, n in enumerate(lengths)])
        perm = torch.randperm(len(slots), device="cuda")
        self.write(cache, state, initial[perm], initial_rope[perm], slots[perm])
        for i, (b, n) in enumerate(zip(blocks, lengths)):
            self.assertTrue((cache[b, :n, 420] == 255).all().item())
            torch.testing.assert_close(
                gather(cache, b, n)[:, :512], x[i, :n], atol=0, rtol=0
            )
        self.write(
            cache,
            state,
            torch.stack([x[i, n] for i, n in enumerate(lengths)]),
            torch.stack([rope[i, n] for i, n in enumerate(lengths)]),
            [b * 64 + n for b, n in zip(blocks, lengths)],
        )
        gold = torch.zeros_like(cache)
        for i in (0, 2):
            b = blocks[i]
            write_nvfp4_cache(
                x[i], rope[i], gold, torch.arange(b * 64, (b + 1) * 64, device="cuda")
            )
            self.assertTrue((cache[b, :, 420] < 4).all().item())
            torch.testing.assert_close(
                gather(cache, b, 64), gather(gold, b, 64), atol=0, rtol=0
            )
            torch.testing.assert_close(
                cache[b, :, :424], gold[b, :, :424], atol=0, rtol=0
            )
            torch.testing.assert_close(
                cache[b, :, 432:], gold[b, :, 432:], atol=0, rtol=0
            )
        self.assertTrue((cache[5, :21, 420] == 255).all().item())
        torch.testing.assert_close(
            gather(cache, 5, 21)[:, :512], x[1, :21], atol=0, rtol=0
        )
        query = torch.randn((3, 16, 576), device="cuda", dtype=torch.bfloat16)
        indices = torch.stack(
            [
                torch.arange(256, device="cuda") % (n + 1) + b * 64
                for b, n in zip(blocks, lengths)
            ]
        )
        actual = sparse_nvfp4_attention(query, cache, indices, 1 / 16)
        dense = decode(cache).float()
        dense[5 * 64 : 5 * 64 + 21, :512] = x[1, :21].float()
        selected = dense[indices]
        probs = (
            torch.einsum("thd,tkd->thk", query.float(), selected)
            .mul(1 / 16)
            .softmax(-1)
        )
        expected = torch.einsum("thk,tkd->thd", probs, selected[..., :512]).bfloat16()
        torch.testing.assert_close(actual, expected, atol=0.04, rtol=0.025)

    def test_full_prefill_bypasses_tail_capacity(self):
        for centers in (1, 2, 4):
            with self.subTest(centers=centers):
                cache, state = self.make(centers, capacity=1)
                x = torch.randn((192, 512), device="cuda", dtype=torch.bfloat16)
                rope = torch.randn((192, 64), device="cuda", dtype=torch.bfloat16)
                slots = torch.arange(192, device="cuda")
                self.write(cache, state, x, rope, slots)
                gold = torch.zeros_like(cache)
                write_nvfp4_cache(x, rope, gold, slots)
                torch.testing.assert_close(cache, gold, atol=0, rtol=0)
                self.assertEqual(state.overflow_count.item(), 0)

    def test_partial_then_seal_center_counts(self):
        for centers in (1, 2, 4):
            with self.subTest(centers=centers):
                cache, state = self.make(centers)
                x = torch.randn((64, 512), device="cuda", dtype=torch.bfloat16)
                rope = torch.randn((64, 64), device="cuda", dtype=torch.bfloat16)
                self.write(
                    cache, state, x[:33], rope[:33], torch.arange(33, device="cuda")
                )
                torch.testing.assert_close(
                    gather(cache, 0, 33)[:, :512], x[:33], atol=0, rtol=0
                )
                self.write(
                    cache, state, x[33:], rope[33:], torch.arange(33, 64, device="cuda")
                )
                gold = torch.zeros_like(cache)
                write_nvfp4_cache(x, rope, gold, torch.arange(64, device="cuda"))
                torch.testing.assert_close(
                    gather(cache, 0, 64), gather(gold, 0, 64), atol=0, rtol=0
                )

    def test_pd_copy_and_append_restore_portable_tail(self):
        source, state = self.make()
        x = torch.randn((64, 512), device="cuda", dtype=torch.bfloat16)
        rope = torch.randn((64, 64), device="cuda", dtype=torch.bfloat16)
        self.write(source, state, x[:19], rope[:19], torch.arange(19, device="cuda"))
        dest, dstate = self.make()
        # Reuse a physical block with stale local BF16 state before receiving PD.
        self.write(
            dest, dstate, x[:19] + 10, rope[:19], torch.arange(128, 147, device="cuda")
        )
        dest[2].copy_(source[0])
        portable = gather(dest, 2, 19).clone()
        torch.testing.assert_close(
            portable, decode(dest)[128:147], atol=1e-6, rtol=0.008
        )
        self.write(
            dest, dstate, x[19:], rope[19:], torch.arange(147, 192, device="cuda")
        )
        gold = torch.zeros_like(dest)
        baseline = torch.cat((portable[:, :512], x[19:]))
        write_nvfp4_cache(baseline, rope, gold, torch.arange(128, 192, device="cuda"))
        torch.testing.assert_close(
            gather(dest, 2, 64), gather(gold, 2, 64), atol=0, rtol=0
        )

    def test_same_cache_page_copy_does_not_alias_another_raw_tail(self):
        cache, state = self.make(capacity=256)
        blocks, count = [2, 5], 17
        latent = torch.randn((2, count, 512), device="cuda", dtype=torch.bfloat16)
        latent[1] += 20
        rope = torch.randn((2, count, 64), device="cuda", dtype=torch.bfloat16)
        slots = torch.cat(
            [torch.arange(b * 64, b * 64 + count, device="cuda") for b in blocks]
        )
        self.write(cache, state, latent.flatten(0, 1), rope.flatten(0, 1), slots)
        portable = cache.clone()
        portable[5].copy_(portable[2])
        cache[5].copy_(cache[2])
        torch.testing.assert_close(
            gather(cache, 5, count), gather(portable, 5, count), atol=0, rtol=0
        )
        torch.testing.assert_close(
            gather(cache, 2, count)[:, :512], latent[0], atol=0, rtol=0
        )
        suffix = torch.randn((64 - count, 512), device="cuda", dtype=torch.bfloat16)
        suffix_rope = torch.randn((64 - count, 64), device="cuda", dtype=torch.bfloat16)
        prefix = gather(portable, 5, count)
        self.write(
            cache,
            state,
            suffix,
            suffix_rope,
            torch.arange(5 * 64 + count, 6 * 64, device="cuda"),
        )
        golden = torch.zeros_like(cache)
        write_nvfp4_cache(
            torch.cat([prefix[:, :512], suffix]),
            torch.cat([prefix[:, 512:], suffix_rope]),
            golden,
            torch.arange(5 * 64, 6 * 64, device="cuda"),
        )
        # The generation tag is intentionally local and differs from legacy.
        torch.testing.assert_close(
            cache[5, :, :424], golden[5, :, :424], atol=0, rtol=0
        )
        torch.testing.assert_close(
            cache[5, :, 432:], golden[5, :, 432:], atol=0, rtol=0
        )
        torch.testing.assert_close(
            gather(cache, 5, 64), gather(golden, 5, 64), atol=0, rtol=0
        )

    def test_rewrite_full_block_uses_original_shadow(self):
        cache, state = self.make()
        x = torch.randn((64, 512), device="cuda", dtype=torch.bfloat16)
        rope = torch.randn((64, 64), device="cuda", dtype=torch.bfloat16)
        self.write(cache, state, x[:63], rope[:63], torch.arange(63, device="cuda"))
        self.write(cache, state, x[63:], rope[63:], [63])
        x[63].normal_()
        self.write(cache, state, x[63:], rope[63:], [63])
        gold = torch.zeros_like(cache)
        write_nvfp4_cache(x, rope, gold, torch.arange(64, device="cuda"))
        torch.testing.assert_close(
            gather(cache, 0, 64), gather(gold, 0, 64), atol=0, rtol=0
        )
        # Recycle the same physical page for a new request.
        self.write(cache, state, x[:1] + 5, rope[:1], [0])
        torch.testing.assert_close(
            gather(cache, 0, 1)[:, :512], x[:1] + 5, atol=0, rtol=0
        )

    def test_bounded_pool_overflow_and_lru(self):
        cache, state = self.make(capacity=1)
        x = torch.randn((2, 512), device="cuda", dtype=torch.bfloat16)
        rope = torch.randn((2, 64), device="cuda", dtype=torch.bfloat16)
        self.write(cache, state, x, rope, [0, 64])
        self.assertEqual(state.overflow_count.item(), 1)
        torch.testing.assert_close(
            gather(cache, 1, 1), decode(cache)[64:65], atol=0, rtol=0
        )
        more = torch.randn((63, 512), device="cuda", dtype=torch.bfloat16)
        more_rope = torch.randn((63, 64), device="cuda", dtype=torch.bfloat16)
        self.write(cache, state, more, more_rope, torch.arange(65, 128, device="cuda"))
        self.assertTrue((cache[1, :, 420] < 4).all().item())
        self.assertEqual(state.data.shape, (1, 64, 512))
        self.assertTrue(torch.isfinite(gather(cache, 1, 64)).all().item())
        before = cache.clone()
        self.write(cache, state, x, rope, [-1, cache.shape[0] * 64])
        torch.testing.assert_close(cache, before, atol=0, rtol=0)

    def test_overflow_full_block_still_seals_safely(self):
        cache, state = self.make(capacity=1)
        x = torch.randn((2, 64, 512), device="cuda", dtype=torch.bfloat16)
        rope = torch.randn((2, 64, 64), device="cuda", dtype=torch.bfloat16)
        self.write(
            cache,
            state,
            torch.cat((x[0, :63], x[1, :63])),
            torch.cat((rope[0, :63], rope[1, :63])),
            torch.cat(
                (torch.arange(63, device="cuda"), torch.arange(64, 127, device="cuda"))
            ),
        )
        portable = gather(cache, 1, 63)[:, :512].clone()
        self.write(cache, state, x[:, 63], rope[:, 63], [63, 127])
        self.assertTrue((cache[:2, :, 420] < 4).all().item())
        gold = torch.zeros_like(cache)
        ordinary = torch.zeros((1, 64, 432), device="cuda", dtype=torch.uint8)
        write_nvfp4_cache(
            x[1, 63:], rope[1, 63:], ordinary, torch.tensor([0], device="cuda")
        )
        last = gather(ordinary, 0, 1)[:, :512]
        write_nvfp4_cache(
            torch.cat((portable, last)),
            rope[1],
            gold,
            torch.arange(64, 128, device="cuda"),
        )
        torch.testing.assert_close(
            gather(cache, 1, 64), gather(gold, 1, 64), atol=0, rtol=0
        )

    def test_cuda_graph_mixed_sealing_and_dynamic_slots(self):
        cache, state = self.make()
        truth = torch.randn((12, 64, 512), device="cuda", dtype=torch.bfloat16)
        ropes = torch.randn((12, 64, 64), device="cuda", dtype=torch.bfloat16)
        initial = [(0, 61), (2, 20), (4, 63)]
        for b, n in initial:
            self.write(
                cache,
                state,
                truth[b, :n],
                ropes[b, :n],
                torch.arange(b * 64, b * 64 + n, device="cuda"),
            )
        slots = torch.tensor([61, 148, 319], device="cuda")
        latent = torch.stack([truth[s // 64, s % 64] for s in [61, 148, 319]])
        rope = torch.stack([ropes[s // 64, s % 64] for s in [61, 148, 319]])
        query = torch.randn((3, 16, 576), device="cuda", dtype=torch.bfloat16)
        indices = slots[:, None].expand(-1, 256).contiguous()
        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            for _ in range(3):
                self.write(cache, state, latent, rope, slots)
                sparse_nvfp4_attention(query, cache, indices, 1 / 16)
        torch.cuda.current_stream().wait_stream(warm)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            write_nvfp4_cache(latent, rope, cache, slots, deferred_state=state)
            result = sparse_nvfp4_attention(query, cache, indices, 1 / 16)
        for values in ([62, 149, 320], [63, 150, 321], [64, 151, 322]):
            latent.copy_(torch.stack([truth[s // 64, s % 64] for s in values]))
            rope.copy_(torch.stack([ropes[s // 64, s % 64] for s in values]))
            slots.copy_(torch.tensor(values, device="cuda"))
            indices.copy_(slots[:, None].expand_as(indices))
            graph.replay()
            torch.cuda.synchronize()
            expected = []
            for s in values:
                if cache[s // 64, s % 64, 420].item() == 255:
                    expected.append(truth[s // 64, s % 64])
                else:
                    expected.append(decode(cache)[s, :512])
            expected = torch.stack(expected)[:, None, :].expand_as(result)
            torch.testing.assert_close(result, expected, atol=0.025, rtol=0.025)
        self.assertEqual(state.overflow_count.item(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
