import unittest

import torch

from rtp_llm.models_py.triton_kernels.sparse_mla.nvfp4_cache import (
    CACHE_BYTES,
    gather_nvfp4_cache,
    sparse_nvfp4_attention,
    write_nvfp4_cache,
)


def reference_decode(cache):
    rows = cache.reshape(-1, cache.shape[-1])
    packed = rows[:, :256].to(torch.long)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(-1, 512)
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
    rope = rows[:, 292:420].contiguous().view(torch.bfloat16)
    latent = values * scales * outer
    centers = (cache.shape[-1] - CACHE_BYTES) // 16
    if centers:
        table = (
            cache[:, :, 432:]
            .contiguous()
            .view(torch.bfloat16)
            .reshape(-1, centers, 512)
        )
        blocks = torch.arange(rows.shape[0], device=cache.device) // 64
        ids = rows[:, 420].long()
        latent += torch.where(
            (ids < centers)[:, None],
            table[blocks, ids.clamp(max=centers - 1)].float(),
            0.0,
        )
    return torch.cat((latent.bfloat16(), rope), -1)


def reference_attention(query, cache, indices, scale):
    decoded = reference_decode(cache).float()
    indices = indices.reshape(query.shape[0], -1)
    valid = (indices >= 0) & (indices < decoded.shape[0])
    selected = decoded[indices.clamp(0, decoded.shape[0] - 1)]
    scores = torch.einsum("thd,tkd->thk", query.float(), selected) * scale
    scores.masked_fill_(~valid[:, None, :], -float("inf"))
    probabilities = scores.softmax(-1).nan_to_num()
    return torch.einsum("thk,tkd->thd", probabilities, selected[..., :512]).bfloat16()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class Nvfp4CacheTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1729)
        self.cache = torch.zeros((8, 64, CACHE_BYTES), dtype=torch.uint8, device="cuda")
        self.latent = torch.randn((300, 512), device="cuda", dtype=torch.bfloat16)
        self.rope = torch.randn((300, 64), device="cuda", dtype=torch.bfloat16)
        self.slots = torch.randperm(512, device="cuda")[:300]

    def test_pack_gather_and_pd_byte_copy(self):
        joined = torch.cat((self.latent, self.rope), dim=-1)
        self.latent = joined[:, :512]
        self.rope = joined[:, 512:]
        self.latent[0].zero_()
        self.latent[1].mul_(1e-5)
        self.latent[2].mul_(1000)
        write_nvfp4_cache(self.latent, self.rope, self.cache, self.slots)
        decoded = reference_decode(self.cache)[self.slots]
        torch.testing.assert_close(decoded[:, 512:], self.rope, rtol=0, atol=0)
        relative_rmse = (decoded[:, :512].float() - self.latent.float()).square().mean(
            -1
        ).sqrt() / self.latent.float().square().mean(-1).sqrt().clamp_min(1e-10)
        self.assertLess(relative_rmse.max().item(), 0.15)
        self.assertEqual(decoded[0, :512].count_nonzero().item(), 0)
        peer_cache = self.cache.clone()
        table = torch.tensor([[3, 0, 5], [6, 2, 1]], device="cuda", dtype=torch.int32)
        lengths = torch.tensor([130, 71], device="cuda", dtype=torch.int32)
        starts = torch.tensor([0, 130], device="cuda", dtype=torch.int32)
        output = torch.empty((201, 576), device="cuda", dtype=torch.bfloat16)
        gather_nvfp4_cache(peer_cache, output, table, lengths, starts, 2, 201)
        physical = torch.cat(
            (
                table[0].repeat_interleave(64)[:130] * 64
                + torch.arange(130, device="cuda") % 64,
                table[1].repeat_interleave(64)[:71] * 64
                + torch.arange(71, device="cuda") % 64,
            )
        )
        torch.testing.assert_close(
            output, reference_decode(self.cache)[physical.long()], rtol=0, atol=0
        )

    def test_invalid_write_and_attention(self):
        slots = self.slots.clone()
        slots[0] = -1
        slots[1] = 512
        write_nvfp4_cache(self.latent, self.rope, self.cache, slots)
        self.assertEqual(
            self.cache.reshape(-1, CACHE_BYTES)[self.slots[:2]].count_nonzero().item(),
            0,
        )
        for heads, tokens, topk in ((16, 3, 73), (64, 4, 2048), (16, 65, 64)):
            with self.subTest(heads=heads, tokens=tokens, topk=topk):
                query = torch.randn(
                    (tokens, heads, 576), device="cuda", dtype=torch.bfloat16
                )
                indices = self.slots[
                    torch.randint(0, 300, (tokens, topk), device="cuda")
                ]
                indices[0].fill_(-1)
                indices[1, ::7] = -1
                actual = sparse_nvfp4_attention(query, self.cache, indices, 1 / 16)
                expected = reference_attention(query, self.cache, indices, 1 / 16)
                torch.testing.assert_close(actual, expected, rtol=0.025, atol=0.012)

    def test_cuda_graph_replay_updates(self):
        query = torch.randn((4, 64, 576), device="cuda", dtype=torch.bfloat16)
        indices = self.slots[torch.randint(0, 300, (4, 256), device="cuda")]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                write_nvfp4_cache(self.latent, self.rope, self.cache, self.slots)
                sparse_nvfp4_attention(query, self.cache, indices, 1 / 16)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            write_nvfp4_cache(self.latent, self.rope, self.cache, self.slots)
            output = sparse_nvfp4_attention(query, self.cache, indices, 1 / 16)
        for step in range(3):
            query.normal_()
            self.latent.normal_()
            self.rope.normal_()
            self.slots.copy_(torch.randperm(512, device="cuda")[:300])
            indices.copy_(
                self.slots[torch.randint(0, 300, indices.shape, device="cuda")]
            )
            indices[step].fill_(-1)
            graph.replay()
            expected = reference_attention(query, self.cache, indices, 1 / 16)
            torch.testing.assert_close(output, expected, rtol=0.025, atol=0.012)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class ClusteredNvfp4CacheTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1729)

    def test_clustering_quality_and_block_transfer(self):
        for centers in (1, 2, 4):
            with self.subTest(centers=centers):
                cache = torch.zeros(
                    (4, 64, 432 + 16 * centers), device="cuda", dtype=torch.uint8
                )
                source_centers = torch.randn((2, centers, 512), device="cuda") * 4
                assignments = torch.arange(64, device="cuda") % centers
                latent = (
                    (
                        source_centers[:, assignments]
                        + torch.randn((2, 64, 512), device="cuda") * 0.1
                    )
                    .reshape(-1, 512)
                    .bfloat16()
                )
                rope = torch.randn((128, 64), device="cuda", dtype=torch.bfloat16)
                permutation = torch.randperm(128, device="cuda")
                joined = torch.cat((latent, rope), -1)[permutation]
                write_nvfp4_cache(joined[:, :512], joined[:, 512:], cache, permutation)
                decoded = reference_decode(cache)[:128]
                torch.testing.assert_close(decoded[:, 512:], rope, atol=0, rtol=0)
                direct = torch.zeros((2, 64, 432), device="cuda", dtype=torch.uint8)
                write_nvfp4_cache(
                    latent, rope, direct, torch.arange(128, device="cuda")
                )
                direct_error = (
                    (reference_decode(direct)[:, :512].float() - latent.float())
                    .square()
                    .mean()
                )
                residual_error = (
                    (decoded[:, :512].float() - latent.float()).square().mean()
                )
                self.assertLess(residual_error.item(), direct_error.item() * 0.1)
                if centers == 1:
                    actual_centers = (
                        cache[:2, :, 432:]
                        .contiguous()
                        .view(torch.bfloat16)
                        .reshape(2, 512)
                    )
                    torch.testing.assert_close(
                        actual_centers,
                        latent.reshape(2, 64, 512).float().mean(1).bfloat16(),
                        atol=0,
                        rtol=0,
                    )
                peer = torch.zeros_like(cache)
                peer[3].copy_(cache[0])
                peer[1].copy_(cache[1])
                output = torch.empty((128, 576), device="cuda", dtype=torch.bfloat16)
                table = torch.tensor([[3, 1]], device="cuda", dtype=torch.int32)
                lengths = torch.tensor([128], device="cuda", dtype=torch.int32)
                starts = torch.tensor([0], device="cuda", dtype=torch.int32)
                gather_nvfp4_cache(peer, output, table, lengths, starts, 1, 128)
                torch.testing.assert_close(
                    output[:, :512], decoded[:, :512], atol=1e-6, rtol=0.008
                )
                torch.testing.assert_close(output[:, 512:], rope, atol=0, rtol=0)
                source_output = torch.empty_like(output)
                gather_nvfp4_cache(
                    cache,
                    source_output,
                    torch.tensor([[0, 1]], device="cuda", dtype=torch.int32),
                    lengths,
                    starts,
                    1,
                    128,
                )
                torch.testing.assert_close(output, source_output, atol=0, rtol=0)
                query = torch.randn((3, 16, 576), device="cuda", dtype=torch.bfloat16)
                indices = torch.randint(0, 128, (3, 256), device="cuda")
                indices = torch.where(indices < 64, indices + 192, indices)
                indices[0].fill_(-1)
                actual = sparse_nvfp4_attention(query, peer, indices, 1 / 32)
                expected = reference_attention(query, peer, indices, 1 / 32)
                torch.testing.assert_close(actual, expected, rtol=0.025, atol=0.04)

    def test_zero_center_fallback_preserves_sink_and_pd_copy(self):
        latent = torch.randn((64, 512), device="cuda", dtype=torch.bfloat16)
        latent[1].mul_(0.003)
        latent[2].zero_()
        rope = torch.randn((64, 64), device="cuda", dtype=torch.bfloat16)
        direct = torch.zeros((1, 64, CACHE_BYTES), device="cuda", dtype=torch.uint8)
        write_nvfp4_cache(latent, rope, direct, torch.arange(64, device="cuda"))
        for centers in (1, 2, 4):
            with self.subTest(centers=centers):
                cache = torch.zeros(
                    (3, 64, CACHE_BYTES + 16 * centers),
                    device="cuda",
                    dtype=torch.uint8,
                )
                write_nvfp4_cache(
                    latent, rope, cache, torch.arange(64, 128, device="cuda")
                )
                self.assertTrue((cache[1, 1:3, 420] == 255).all().item())
                torch.testing.assert_close(
                    cache[1, 1:3, :292], direct[0, 1:3, :292], rtol=0, atol=0
                )
                decoded = reference_decode(cache)[64:128]
                torch.testing.assert_close(
                    decoded[1:3], reference_decode(direct)[1:3], rtol=0, atol=0
                )
                # The zero-center marker must survive relocation without a tail pool.
                peer = torch.zeros_like(cache)
                peer[2].copy_(cache[1])
                output = torch.empty((64, 576), device="cuda", dtype=torch.bfloat16)
                gather_nvfp4_cache(
                    peer,
                    output,
                    torch.tensor([[2]], device="cuda", dtype=torch.int32),
                    torch.tensor([64], device="cuda", dtype=torch.int32),
                    torch.tensor([0], device="cuda", dtype=torch.int32),
                    1,
                    64,
                )
                torch.testing.assert_close(output, decoded, rtol=0.008, atol=1e-6)
                source_output = torch.empty_like(output)
                gather_nvfp4_cache(
                    cache,
                    source_output,
                    torch.tensor([[1]], device="cuda", dtype=torch.int32),
                    torch.tensor([64], device="cuda", dtype=torch.int32),
                    torch.tensor([0], device="cuda", dtype=torch.int32),
                    1,
                    64,
                )
                torch.testing.assert_close(output, source_output, rtol=0, atol=0)
                torch.testing.assert_close(
                    output[1:3], reference_decode(direct)[1:3], rtol=0, atol=0
                )
                query = torch.zeros((2, 16, 576), device="cuda", dtype=torch.bfloat16)
                indices = (
                    torch.tensor([129, 130], device="cuda")[:, None]
                    .expand(2, 64)
                    .contiguous()
                )
                actual = sparse_nvfp4_attention(query, peer, indices, 1 / 32)
                expected = decoded[1:3, :512][:, None, :].expand_as(actual)
                torch.testing.assert_close(actual, expected, rtol=0.025, atol=1e-6)

    def test_partial_block_append_and_recycle(self):
        cache = torch.zeros((3, 64, 496), device="cuda", dtype=torch.uint8)
        cache[:, :, 432:].fill_(255)
        latent = torch.randn((64, 512), device="cuda", dtype=torch.bfloat16)
        rope = torch.randn((64, 64), device="cuda", dtype=torch.bfloat16)
        write_nvfp4_cache(
            latent[:19], rope[:19], cache, torch.arange(64, 83, device="cuda")
        )
        saved_centers = cache[1, :, 432:].clone()
        prefix = reference_decode(cache[1:2])[:19].clone()
        peer = cache.clone()
        write_nvfp4_cache(
            latent[19:], rope[19:], peer, torch.arange(83, 128, device="cuda")
        )
        torch.testing.assert_close(peer[1, :, 432:], saved_centers, atol=0, rtol=0)
        decoded = reference_decode(peer[1:2])
        torch.testing.assert_close(decoded[:19], prefix, atol=0, rtol=0)
        torch.testing.assert_close(decoded[:, 512:], rope, atol=0, rtol=0)
        self.assertLess(
            (decoded[:, :512].float() - latent.float()).square().mean().sqrt().item(),
            0.2,
        )
        latent.add_(10)
        write_nvfp4_cache(latent[:1], rope[:1], peer, torch.tensor([64], device="cuda"))
        centers = peer[1, :, 432:].contiguous().view(torch.bfloat16).reshape(4, 512)
        torch.testing.assert_close(centers, latent[:1].expand(4, -1), atol=0, rtol=0)
        torch.testing.assert_close(peer[0], cache[0], atol=0, rtol=0)
        torch.testing.assert_close(peer[2], cache[2], atol=0, rtol=0)
        before = peer.clone()
        write_nvfp4_cache(
            latent[:2], rope[:2], peer, torch.tensor([-1, 192], device="cuda")
        )
        torch.testing.assert_close(peer, before, atol=0, rtol=0)

    def test_clustered_cuda_graph_replay(self):
        cache = torch.zeros((8, 64, 496), device="cuda", dtype=torch.uint8)
        latent = torch.randn((4, 512), device="cuda", dtype=torch.bfloat16)
        rope = torch.randn((4, 64), device="cuda", dtype=torch.bfloat16)
        slots = torch.arange(4, device="cuda") * 64
        query = torch.randn((4, 64, 576), device="cuda", dtype=torch.bfloat16)
        indices = slots[:, None].expand(4, 2048).contiguous()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                write_nvfp4_cache(latent, rope, cache, slots)
                sparse_nvfp4_attention(query, cache, indices, 1 / 16)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            write_nvfp4_cache(latent, rope, cache, slots)
            output = sparse_nvfp4_attention(query, cache, indices, 1 / 16)
        for offset in (0, 1, 256, 257):
            latent.normal_()
            rope.normal_()
            query.normal_()
            slots.copy_(torch.arange(4, device="cuda") * 64 + offset)
            indices.copy_(slots[:, None].expand_as(indices))
            graph.replay()
            expected = reference_attention(query, cache, indices, 1 / 16)
            torch.testing.assert_close(output, expected, rtol=0.025, atol=0.012)
            decoded = reference_decode(cache)[slots]
            torch.testing.assert_close(decoded[:, 512:], rope, atol=0, rtol=0)
            self.assertLess(
                (decoded[:, :512].float() - latent.float())
                .square()
                .mean()
                .sqrt()
                .item(),
                0.2,
            )

    def test_unsupported_page_size(self):
        cache = torch.zeros((2, 32, 496), device="cuda", dtype=torch.uint8)
        latent = torch.zeros((1, 512), device="cuda", dtype=torch.bfloat16)
        rope = torch.zeros((1, 64), device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "64-token"):
            write_nvfp4_cache(
                latent, rope, cache, torch.zeros(1, device="cuda", dtype=torch.int64)
            )


if __name__ == "__main__":
    unittest.main()
