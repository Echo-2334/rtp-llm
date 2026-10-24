import unittest
from unittest import mock

import torch

from rtp_llm.models_py.triton_kernels.sparse_mla import nvfp4_sim
from rtp_llm.models_py.triton_kernels.sparse_mla.nvfp4_cache import (
    gather_nvfp4_cache,
    write_nvfp4_cache,
)
from rtp_llm.models_py.triton_kernels.sparse_mla.nvfp4_deferred import (
    DeferredNvfp4State,
)
from rtp_llm.models_py.triton_kernels.sparse_mla.nvfp4_sim import (
    SimConfig,
    completed_rows,
    plain_fake_quant,
    write_simulated,
)
from rtp_llm.ops import compute_ops

SCALE = None


def writer(kind):
    def write(latent, rope, cache, slots):
        compute_ops.concat_and_cache_mla(latent, rope, cache, slots, kind, SCALE)

    return write


BF16 = "auto"
FP8 = "fp8_ds_mla"


def latent_rows(n, seed=0):
    """Gaussian rows with the outliers that matter for NVFP4: hot channels,
    large tokens, attention-sink-like tiny rows and exact zeros."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((n, 512), device="cuda", generator=g) * 0.5
    x[:, 7] += 6.0
    x[:, 300:316] *= 4.0
    x[::17] *= 20.0
    x[1::23] *= 0.003
    x[5::97] = 0.0
    return x.bfloat16()


def rope_rows(n, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn((n, 64), device="cuda", generator=g).bfloat16()


def bf16_cache(blocks):
    return torch.zeros((blocks, 64, 576), device="cuda", dtype=torch.bfloat16)


def fp8_cache(blocks):
    return torch.zeros((blocks, 64, 656), device="cuda", dtype=torch.uint8).view(
        torch.float8_e4m3fn
    )


def nvfp4_cache(blocks, centers):
    width = 432 if centers == 0 else 448 + 16 * centers
    return torch.zeros((blocks, 64, width), device="cuda", dtype=torch.uint8)


def real_decode(cache):
    """Every row of a real NVFP4 cache as the attention kernels read it."""
    blocks = cache.shape[0]
    out = torch.empty((blocks * 64, 576), device="cuda", dtype=torch.bfloat16)
    gather_nvfp4_cache(
        cache,
        out,
        torch.arange(blocks, device="cuda", dtype=torch.int32)[None],
        torch.tensor([blocks * 64], device="cuda", dtype=torch.int32),
        torch.zeros(1, device="cuda", dtype=torch.int32),
        1,
        blocks * 64,
    )
    return out


def fp8_decode(cache):
    rows = cache.view(torch.uint8).reshape(-1, 656)
    values = rows[:, :512].contiguous().view(torch.float8_e4m3fn).float()
    scales = rows[:, 512:528].contiguous().view(torch.float32)
    return values * scales.repeat_interleave(128, -1)


def slots(start, count):
    return torch.arange(start, start + count, device="cuda", dtype=torch.int64)


class Nvfp4SimTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global SCALE
        SCALE = torch.tensor(1.0, device="cuda")

    def test_plain_matches_real_c0_bitwise(self):
        x, rope = latent_rows(300), rope_rows(300)
        cache = nvfp4_cache(7, 0)
        write_nvfp4_cache(x, rope, cache, slots(64, 300))
        expected = real_decode(cache)[64:364, :512]
        torch.testing.assert_close(plain_fake_quant(x), expected, atol=0, rtol=0)
        # The simulated write stores exactly those values next to the RoPE.
        sim = bf16_cache(7)
        write_simulated(SimConfig("plain"), x, rope, sim, slots(64, 300), writer(BF16))
        rows = sim.reshape(-1, 576)[64:364]
        torch.testing.assert_close(rows[:, :512], expected, atol=0, rtol=0)
        torch.testing.assert_close(rows[:, 512:], rope, atol=0, rtol=0)

    def test_full_block_seal_matches_real_bitwise(self):
        x, rope = latent_rows(4 * 64, seed=3), rope_rows(4 * 64)
        for centers in (1, 2, 4):
            with self.subTest(centers=centers):
                real = nvfp4_cache(6, centers)
                write_nvfp4_cache(x, rope, real, slots(64, 256))
                sim = bf16_cache(6)
                config = SimConfig("cluster", centers)
                write_simulated(config, x, rope, sim, slots(64, 256), writer(BF16))
                torch.testing.assert_close(
                    sim.reshape(-1, 576)[64:320, :512],
                    real_decode(real)[64:320, :512],
                    atol=0,
                    rtol=0,
                )
                # Untouched blocks stay untouched.
                self.assertEqual(sim[0].abs().sum().item(), 0)
                self.assertEqual(sim[5].abs().sum().item(), 0)

    def test_decode_appends_match_real_deferred_bitwise(self):
        x, rope = latent_rows(64, seed=5), rope_rows(64)
        real = nvfp4_cache(4, 2)
        state = DeferredNvfp4State(real, 8)
        sim = bf16_cache(4)
        config = SimConfig("cluster", 2)
        for row in range(64):
            s = slots(128 + row, 1)
            write_nvfp4_cache(x[row : row + 1], rope[row : row + 1], real, s, deferred_state=state)
            write_simulated(config, x[row : row + 1], rope[row : row + 1], sim, s, writer(BF16))
            if row < 63:
                # Unsealed rows stay exact, like the real BF16 tail.
                torch.testing.assert_close(
                    sim[2, : row + 1, :512], x[: row + 1], atol=0, rtol=0
                )
        torch.testing.assert_close(
            sim[2, :, :512], real_decode(real)[128:192, :512], atol=0, rtol=0
        )

    def test_pd_boundary_block_matches_real_bitwise(self):
        """Prefill ships a partial block as the plain mirror; decode seals from it."""
        x, rope = latent_rows(64, seed=7), rope_rows(64)
        prompt = 20
        # Real: prefill state -> page copy -> fresh decode state.
        real_p = nvfp4_cache(4, 2)
        p_state = DeferredNvfp4State(real_p, 8)
        write_nvfp4_cache(x[:prompt], rope[:prompt], real_p, slots(64, prompt), deferred_state=p_state)
        real_d = nvfp4_cache(4, 2)
        real_d[1].copy_(real_p[1])
        d_state = DeferredNvfp4State(real_d, 8)
        # Simulation: prefill role mirrors partial rows, decode keeps new rows exact.
        sim_p = bf16_cache(4)
        write_simulated(
            SimConfig("cluster", 2, prefill_mirror=True),
            x[:prompt],
            rope[:prompt],
            sim_p,
            slots(64, prompt),
            writer(BF16),
        )
        torch.testing.assert_close(
            sim_p[1, :prompt, :512], plain_fake_quant(x[:prompt]), atol=0, rtol=0
        )
        sim_d = bf16_cache(4)
        sim_d[1].copy_(sim_p[1])
        decode = SimConfig("cluster", 2)
        for row in range(prompt, 64):
            s = slots(64 + row, 1)
            write_nvfp4_cache(x[row : row + 1], rope[row : row + 1], real_d, s, deferred_state=d_state)
            write_simulated(decode, x[row : row + 1], rope[row : row + 1], sim_d, s, writer(BF16))
            if row == 40:
                # Before sealing: restored mirror rows plus exact decode rows.
                torch.testing.assert_close(
                    sim_d[1, : row + 1, :512],
                    real_decode(real_d)[64 : 65 + row, :512],
                    atol=0,
                    rtol=0,
                )
        torch.testing.assert_close(
            sim_d[1, :, :512], real_decode(real_d)[64:128, :512], atol=0, rtol=0
        )

    def test_prefill_mirror_only_touches_partial_blocks(self):
        x, rope = latent_rows(64 + 30, seed=9), rope_rows(94)
        sim = bf16_cache(4)
        write_simulated(
            SimConfig("cluster", 2, prefill_mirror=True), x, rope, sim, slots(64, 94), writer(BF16)
        )
        full = bf16_cache(4)
        write_simulated(SimConfig("cluster", 2), x[:64], rope[:64], full, slots(64, 64), writer(BF16))
        torch.testing.assert_close(sim[1], full[1], atol=0, rtol=0)
        torch.testing.assert_close(sim[2, :30, :512], plain_fake_quant(x[64:]), atol=0, rtol=0)
        # Decode writes in the same process never mirror (PDFUSION emulation).
        extra, extra_rope = latent_rows(1, seed=10), rope_rows(1, seed=11)
        write_simulated(
            SimConfig("cluster", 2, prefill_mirror=True),
            extra,
            extra_rope,
            sim,
            slots(64 + 94, 1),
            writer(BF16),
            prefill=False,
        )
        torch.testing.assert_close(sim[2, 30, :512], extra[0], atol=0, rtol=0)

    def test_fp8_storage(self):
        x, rope = latent_rows(2 * 64 + 10, seed=11), rope_rows(138)
        s = slots(64, 138)
        # Plain: exactly the FP8 bytes of the simulated values.
        sim, ref = fp8_cache(5), fp8_cache(5)
        write_simulated(SimConfig("plain"), x, rope, sim, s, writer(FP8))
        writer(FP8)(plain_fake_quant(x), rope, ref, s)
        self.assertTrue(torch.equal(sim.view(torch.uint8), ref.view(torch.uint8)))
        # Cluster: unsealed decode rows are the ordinary FP8 write; sealed rows
        # are the BF16 simulation of the FP8-rounded inputs, FP8-rounded again.
        sim, ref = fp8_cache(5), fp8_cache(5)
        write_simulated(SimConfig("cluster", 2), x, rope, sim, s, writer(FP8))
        writer(FP8)(x, rope, ref, s)
        sim_u8, ref_u8 = sim.view(torch.uint8), ref.view(torch.uint8)
        self.assertTrue(torch.equal(sim_u8[3, :10], ref_u8[3, :10]))
        self.assertTrue(torch.equal(sim_u8[1:3, :, 528:], ref_u8[1:3, :, 528:]))
        through_bf16 = bf16_cache(5)
        stored = fp8_decode(ref)[64:192].bfloat16()
        write_simulated(SimConfig("cluster", 2), stored, rope[:128], through_bf16, slots(64, 128), writer(BF16))
        expected = through_bf16.reshape(-1, 576)[64:192, :512].float()
        actual = fp8_decode(sim)[64:192]
        error = (actual - expected).norm() / expected.norm()
        self.assertLess(error.item(), 0.03)
        self.assertFalse(torch.equal(sim_u8[1:3, :, :512], ref_u8[1:3, :, :512]))

    def test_completed_rows(self):
        # Blocks 1 and 3 get row 63 (slots 127, 255); block 4 (300) and block 6
        # (400) do not, and padding never counts.
        s = torch.tensor([64, 65, 127, 200, -1, 255, 300, 400], device="cuda")
        expected = [True, True, True, True, False, True, False, False]
        self.assertEqual(completed_rows(s).tolist(), expected)

    def test_compacted_prefill_matches_per_token_path(self):
        # Three requests with gaps and a partial block; > COMPACT_MIN_TOKENS.
        parts = [slots(64, 200), slots(64 * 8, 64 * 3 + 5), slots(64 * 20 + 10, 54)]
        s = torch.cat(parts)
        x, rope = latent_rows(s.numel(), seed=13), rope_rows(s.numel())
        compact, per_token = bf16_cache(24), bf16_cache(24)
        config = SimConfig("cluster", 4, prefill_mirror=True)
        write_simulated(config, x, rope, compact, s, writer(BF16))
        with mock.patch.object(nvfp4_sim, "COMPACT_MIN_TOKENS", 1 << 30):
            write_simulated(config, x, rope, per_token, s, writer(BF16))
        torch.testing.assert_close(compact, per_token, atol=0, rtol=0)

    def test_cuda_graph_multi_batch_matches_eager(self):
        config = SimConfig("cluster", 2)
        for batch in (1, 8):
            with self.subTest(batch=batch):
                graph_cache, eager_cache = bf16_cache(batch * 4 + 1), bf16_cache(batch * 4 + 1)
                x_static = torch.zeros((batch, 512), device="cuda", dtype=torch.bfloat16)
                rope_static = torch.zeros((batch, 64), device="cuda", dtype=torch.bfloat16)
                slot_static = torch.full((batch,), -1, device="cuda", dtype=torch.int64)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        write_simulated(config, x_static, rope_static, graph_cache, slot_static, writer(BF16))
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    write_simulated(config, x_static, rope_static, graph_cache, slot_static, writer(BF16))
                # Sequence i starts at a different row so seals land on different steps.
                starts = torch.arange(batch, device="cuda") * 64 * 4 + 64 + torch.arange(batch, device="cuda") * 7
                for step in range(140):
                    x = latent_rows(batch, seed=100 + step)
                    rope = rope_rows(batch, seed=200 + step)
                    s = starts + step
                    s = torch.where(torch.arange(batch, device="cuda") == batch - 1, torch.full_like(s, -1), s) if batch > 1 and step % 5 == 0 else s
                    x_static.copy_(x)
                    rope_static.copy_(rope)
                    slot_static.copy_(s)
                    graph.replay()
                    write_simulated(config, x, rope, eager_cache, s.clone(), writer(BF16))
                torch.testing.assert_close(graph_cache, eager_cache, atol=0, rtol=0)

    def test_batch_sizes_share_compiled_kernels(self):
        kernels = (nvfp4_sim._plain_kernel, nvfp4_sim._fit_kernel, nvfp4_sim._seal_kernel)
        if not all(hasattr(k, "device_caches") for k in kernels):
            self.skipTest("Triton JIT cache introspection unavailable")

        def compiled():
            return [sum(len(e[0]) for e in k.device_caches.values()) for k in kernels]

        config = SimConfig("cluster", 2, prefill_mirror=True)
        counts = []
        for n in (1, 3, 8, 37, 64, 129):
            cache = bf16_cache(n // 64 + 4)
            write_simulated(config, latent_rows(n), rope_rows(n), cache, slots(64 + 63 - (n - 1) % 64, n), writer(BF16))
            counts.append(compiled())
        self.assertEqual(counts[1], counts[-1])


if __name__ == "__main__":
    unittest.main()
