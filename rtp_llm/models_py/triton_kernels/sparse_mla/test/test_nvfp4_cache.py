import unittest

import torch
from rtp_llm.models_py.triton_kernels.sparse_mla.nvfp4_cache import (
    CACHE_BYTES,
    gather_nvfp4_cache,
    sparse_nvfp4_attention,
    write_nvfp4_cache,
)


def reference_decode(cache):
    rows = cache.reshape(-1, CACHE_BYTES)
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
    return torch.cat(((values * scales * outer).bfloat16(), rope), -1)


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


if __name__ == "__main__":
    unittest.main()
