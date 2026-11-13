import json
import os
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from rtp_llm.models_py.triton_kernels.sparse_mla.prefill_pivot import PrefillPivotConfig
from rtp_llm.models_py.triton_kernels.sparse_mla.prefill_pivot_policy import (
    resolve_prefill_pivot_layers,
)

FULL_LAYERS = (0, 1, 2, *range(6, 78, 4))


def model_config():
    return SimpleNamespace(
        num_layers=78,
        is_mtp=False,
        indexer_types=["full" if i in FULL_LAYERS else "shared" for i in range(78)],
        attn_config=SimpleNamespace(
            use_mla=True,
            is_sparse=True,
            indexer_head_num=32,
            indexer_head_dim=128,
            indexer_topk=2048,
        ),
    )


class PrefillPivotPolicyTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "layers.json"
        self.path.write_text(
            json.dumps({"default": "native", "layers": {"2": "G4P4K"}})
        )
        env = patch.dict(
            os.environ,
            {
                "GLM5_PREFILL_PIVOT_MODE": "layerwise",
                "GLM5_PREFILL_PIVOT_LAYER_CONFIG": str(self.path),
                "GLM5_DECODE_PIVOT_MODE": "off",
                "GLM5_INDEXER_POOL_REUSE_STEPS": "0",
                "ENABLE_FUSE_KERNELS": "0",
            },
        )
        env.start()
        self.addCleanup(env.stop)

    def test_physical_layer_ids_defaults_and_immutable_snapshot(self):
        settings = resolve_prefill_pivot_layers(model_config())
        self.assertIsInstance(settings, tuple)
        self.assertEqual(len(settings), 78)
        self.assertEqual(
            [i for i, v in enumerate(settings) if v is not None], list(FULL_LAYERS)
        )
        self.assertEqual(settings[2], "G4P4K")
        self.assertEqual(settings[6], "native")
        self.assertIsNone(settings[3])
        self.path.write_text('{"default":"G4P4K","layers":{"6":"native"}}')
        updated = resolve_prefill_pivot_layers(model_config())
        self.assertEqual(settings[0], "native")
        self.assertEqual(updated[0], "G4P4K")
        self.assertEqual(updated[6], "native")

    def test_invalid_configuration_fails_instead_of_silent_fallback(self):
        invalid = [
            "[]",
            '{"default":"G8P4K"}',
            '{"defaults":"native"}',
            '{"layers":[]}',
            '{"layers":{"3":"G4P4K"}}',
            '{"layers":{"78":"native"}}',
            '{"layers":{"-1":"native"}}',
            '{"layers":{"02":"G4P4K"}}',
            '{"layers":{"x":"native"}}',
            '{"layers":{"2":true}}',
            '{"layers":{"2":{"mode":"native"}}}',
            '{"layers":{"2":"native","2":"G4P4K"}}',
            '{"default":"native","default":"G4P4K"}',
            "{bad json",
        ]
        for raw in invalid:
            with self.subTest(raw=raw):
                self.path.write_text(raw)
                with self.assertRaises(ValueError):
                    resolve_prefill_pivot_layers(model_config())
        self.path.unlink()
        with self.assertRaisesRegex(ValueError, "Cannot read"):
            resolve_prefill_pivot_layers(model_config())
        with patch.dict(os.environ, {"GLM5_PREFILL_PIVOT_LAYER_CONFIG": ""}):
            with self.assertRaisesRegex(ValueError, "is required"):
                resolve_prefill_pivot_layers(model_config())

    def test_wrong_architecture_rejected_and_mtp_ignores_target_layer_ids(self):
        config = model_config()
        config.attn_config.indexer_head_num = 64
        with self.assertRaisesRegex(ValueError, "requires MLA"):
            resolve_prefill_pivot_layers(config)
        config = model_config()
        config.is_mtp, config.num_layers = True, 1
        self.path.unlink()
        self.assertEqual(resolve_prefill_pivot_layers(config), ("native",))

    def test_legacy_modes_ignore_layer_file_and_keep_global_settings(self):
        self.path.write_text("invalid; legacy modes must not read this")
        with patch.dict(os.environ, {"GLM5_PREFILL_PIVOT_MODE": "off"}):
            self.assertIsNone(resolve_prefill_pivot_layers(model_config()))
            self.assertIsNone(PrefillPivotConfig.resolve(2048))
        with patch.dict(
            os.environ,
            {
                "GLM5_PREFILL_PIVOT_MODE": "refine",
                "GLM5_PREFILL_PIVOT_GROUP_SIZE": "2",
                "GLM5_PREFILL_PIVOT_CANDIDATES": "8192",
                "GLM5_PREFILL_PIVOT_LOCAL_WINDOW": "8",
                "GLM5_PREFILL_PIVOT_MIN_KV_LEN": "40000",
            },
        ):
            self.assertIsNone(resolve_prefill_pivot_layers(model_config()))
            self.assertEqual(
                PrefillPivotConfig.resolve(2048), PrefillPivotConfig(2, 8192, 8, 40000)
            )
            self.assertEqual(
                PrefillPivotConfig.resolve(2048, "G4P4K"), PrefillPivotConfig()
            )
            self.assertIsNone(PrefillPivotConfig.resolve(2048, "native"))
        with self.assertRaisesRegex(ValueError, "model initializer"):
            PrefillPivotConfig.resolve(2048)

    def test_real_model_construction_and_graph_clones_keep_each_layer_choice(self):
        # Tiny CPU weights and stand-ins for unrelated linears/MLPs avoid loading
        # a model. The model -> decoder -> MLA -> Indexer -> IndexerOp chain is real.
        from rtp_llm.models_py.model_desc import generic_moe as module
        from rtp_llm.models_py.model_desc.generic_moe_mtp import GenericMoeMTPModel

        config = model_config()
        config.quant_config = None
        config.layernorm_eps = 1e-6
        config.moe_layer_index = []
        config.activation_type = None
        attn = config.attn_config
        for key, value in dict(
            head_num=64,
            nope_head_dim=128,
            rope_head_dim=64,
            kv_lora_rank=512,
            v_head_dim=128,
            q_lora_rank=0,
            kernel_tokens_per_block=64,
            tokens_per_block=64,
        ).items():
            setattr(attn, key, value)
        attn.rope_config = SimpleNamespace(indexer_is_neox_style=True)
        weights = SimpleNamespace(
            weights=[defaultdict(lambda: torch.ones(128)) for _ in range(78)],
            global_weights=defaultdict(lambda: torch.ones(128)),
            get_global_weight=lambda key: torch.ones(128),
        )
        parallel = SimpleNamespace(get_attn_tp_size=lambda: 8)
        moe = SimpleNamespace(moe_strategy="")

        def init_base(obj, model_config, *args, **kwargs):
            nn.Module.__init__(obj)
            obj.layer_num = model_config.num_layers

        with (
            patch.object(module.GptModelBase, "__init__", init_base),
            patch.object(
                module.LinearFactory,
                "create_linear_from_weights",
                side_effect=lambda *a, **k: nn.Identity(),
            ),
            patch.object(
                module, "Embedding", side_effect=lambda *a, **k: nn.Identity()
            ),
            patch(
                "rtp_llm.models_py.model_desc.generic_moe_mtp.Embedding",
                side_effect=lambda *a, **k: nn.Identity(),
            ),
            patch.object(module, "DenseMLP", side_effect=lambda *a, **k: nn.Identity()),
            patch.object(module, "resolve_glm5_cmp_enabled", return_value=False),
        ):
            for default, overrides in [
                ("native", {}),
                ("G4P4K", {}),
                ("native", {"2": "G4P4K", "74": "G4P4K"}),
            ]:
                with self.subTest(default=default, overrides=overrides):
                    self.path.write_text(
                        json.dumps({"default": default, "layers": overrides})
                    )
                    model = module.GenericMoeModel(config, parallel, weights, moe, 4)
                    for i, layer in enumerate(model.layers):
                        indexer = layer.self_attn.indexer
                        if i not in FULL_LAYERS:
                            self.assertIsNone(indexer)
                            continue
                        expected = (
                            PrefillPivotConfig()
                            if overrides.get(str(i), default) == "G4P4K"
                            else None
                        )
                        self.assertEqual(indexer.indexer_op._prefill_pivot, expected)
                        clone = layer.clone_for_cuda_graph()
                        self.assertIs(clone.self_attn.indexer, indexer)
            config.is_mtp, config.num_layers = True, 1
            mtp = GenericMoeMTPModel(config, parallel, weights, moe, 4)
            self.assertIsNone(mtp.layers[0].self_attn.indexer.indexer_op._prefill_pivot)

    def test_decode_and_mtp_dispatch_never_calls_prefill(self):
        from rtp_llm.models_py.modules.hybrid.indexer import Indexer

        for setting in ("native", "G4P4K"):
            for flags in (
                dict(is_prefill=False),
                dict(is_prefill=True, is_target_verify=True),
                dict(is_prefill=True, is_draft_extend=True),
            ):
                op = SimpleNamespace(
                    _prefill_pivot=PrefillPivotConfig.resolve(2048, setting),
                    _get_topk_paged=Mock(return_value="decode"),
                )
                obj = SimpleNamespace(
                    indexer_op=op, _prefill_cp_enabled=Mock(side_effect=AssertionError)
                )
                self.assertEqual(
                    Indexer._compute_topk(
                        obj, None, None, None, None, SimpleNamespace(**flags), None
                    ),
                    "decode",
                )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class PrefillPivotPolicyGpuTest(unittest.TestCase):
    def setUp(self):
        from rtp_llm.models_py.modules.base.cuda.indexer_op import IndexerOp
        from rtp_llm.models_py.modules.hybrid.indexer import Indexer

        torch.manual_seed(20261113)
        self.Indexer = Indexer
        env = patch.dict(
            os.environ,
            {
                "GLM5_DECODE_PIVOT_MODE": "off",
                "GLM5_INDEXER_POOL_REUSE_STEPS": "0",
                "GLM5_PREFILL_PIVOT_GROUP_SIZE": "4",
                "GLM5_PREFILL_PIVOT_CANDIDATES": "4096",
                "GLM5_PREFILL_PIVOT_LOCAL_WINDOW": "4",
                "GLM5_PREFILL_PIVOT_MIN_KV_LEN": "32768",
            },
        )
        env.start()
        self.addCleanup(env.stop)
        with patch.dict(os.environ, {"GLM5_PREFILL_PIVOT_MODE": "off"}):
            self.legacy_native = IndexerOp(32, 128, 2048, 64)
        with patch.dict(os.environ, {"GLM5_PREFILL_PIVOT_MODE": "refine"}):
            self.legacy_pivot = IndexerOp(32, 128, 2048, 64)
        with patch.dict(os.environ, {"GLM5_PREFILL_PIVOT_MODE": "layerwise"}):
            self.native = IndexerOp(32, 128, 2048, 64, prefill_pivot_setting="native")
            self.pivot = IndexerOp(32, 128, 2048, 64, prefill_pivot_setting="G4P4K")

    def compute(self, op, q, w, cache, params, inputs, qs=None):
        obj = SimpleNamespace(indexer_op=op, _prefill_cp_enabled=lambda: False)
        return self.Indexer._compute_topk(
            obj, q, w, cache, params, inputs, None, q_scale=qs
        )

    def assert_same_indices(self, actual, expected):
        # Native TopK does not promise an ordering. Require the exact selected
        # set and padding, including when comparing eager and graph replay.
        actual_sorted = actual.sort(dim=-1).values
        expected_sorted = expected.sort(dim=-1).values
        torch.testing.assert_close(actual_sorted, expected_sorted, rtol=0, atol=0)

    def test_multirequest_prefill_matches_legacy_with_mixed_lengths_and_tails(self):
        # Mock only cache transport; run production dispatch, proxy, recall,
        # refine, native fallback and TopK kernels on the actual GPU.
        from rtp_llm.models_py.modules.base.cuda import indexer_op as module

        for batch in (1, 2, 4):
            with self.subTest(batch=batch):
                lengths = [40000, 33000, 50000, 200][:batch]
                rows = [9, 7, 8, 5][:batch]
                n, total = sum(rows), sum(lengths)
                q = torch.randn(n, 32, 128, device="cuda").to(torch.float8_e4m3fn)
                qs = torch.exp2(torch.randint(-3, 3, (n, 32), device="cuda").float())
                w = torch.randn(n, 32, device="cuda") * qs / 32
                k = torch.randn(total, 128, device="cuda").to(torch.float8_e4m3fn)
                sf = torch.rand(total, device="cuda") + 0.5
                starts, ends, origin = [], [], 0
                for length, count in zip(lengths, rows):
                    starts.extend([origin] * count)
                    ends.extend(range(origin + length - count + 1, origin + length + 1))
                    origin += length
                ks = torch.tensor(starts, dtype=torch.int32, device="cuda")
                ke = torch.tensor(ends, dtype=torch.int32, device="cuda")
                params = SimpleNamespace(
                    ks=ks,
                    ke=ke,
                    topk_indices_offset=torch.zeros_like(ks),
                    prefill_total_kv_tokens=total,
                    prefill_max_kv_len=max(lengths),
                )
                inputs = SimpleNamespace(is_prefill=True, cu_kv_seqlens=None)
                cache = SimpleNamespace(
                    kv_scale_base=torch.empty(0, dtype=torch.uint8, device="cuda")
                )

                def gather(cache, key, scale, table, cu):
                    key.copy_(k)
                    scale.view(torch.float32).copy_(sf[:, None])

                with (
                    patch.object(
                        module, "_prefill_physical_block_table", return_value=None
                    ),
                    patch.object(
                        module.rtp_llm_ops,
                        "cp_gather_indexer_k_quant_cache",
                        side_effect=gather,
                    ),
                ):
                    for old, new in (
                        (self.legacy_native, self.native),
                        (self.legacy_pivot, self.pivot),
                    ):
                        expected = self.compute(old, q, w, cache, params, inputs, qs)
                        actual = self.compute(new, q, w, cache, params, inputs, qs)
                        self.assert_same_indices(actual, expected)
                        valid = actual >= 0
                        self.assertTrue(
                            ((~valid) | (actual < (ke - ks)[:, None])).all().item()
                        )
                        for i, count in enumerate((ke - ks).tolist()):
                            ids = actual[i][valid[i]]
                            self.assertEqual(ids.numel(), min(2048, count))
                            self.assertEqual(ids.unique().numel(), ids.numel())

    def test_decode_graph_replay_multiple_batch_sizes_and_changing_inputs(self):
        for batch in (1, 2, 4, 8):
            with self.subTest(batch=batch):
                pages, page_size = 128, 64
                blocks = batch * pages
                k = torch.randn(blocks, page_size, 128, device="cuda").to(
                    torch.float8_e4m3fn
                )
                sf = torch.rand(blocks, page_size, device="cuda") + 0.5
                packed = (
                    torch.cat(
                        [
                            k.view(torch.uint8).reshape(blocks, -1),
                            sf.view(torch.uint8).reshape(blocks, -1),
                        ],
                        dim=1,
                    )
                    .contiguous()
                    .view(blocks, page_size, 132)
                )
                cache = SimpleNamespace(kv_scale_base=packed)
                table = (
                    torch.randperm(blocks, device="cuda")
                    .to(torch.int32)
                    .reshape(batch, pages)
                )
                lengths = torch.full((batch,), 6000, dtype=torch.int32, device="cuda")
                q = torch.randn(batch, 32, 128, device="cuda").to(torch.float8_e4m3fn)
                w = torch.randn(batch, 32, device="cuda") / 32
                inputs = SimpleNamespace(
                    is_prefill=False, kv_cache_kernel_block_id_device=table
                )
                params = SimpleNamespace(expanded_seq_lens=lengths, kvlen_d=lengths)

                def run():
                    return tuple(
                        self.compute(op, q, w, cache, params, inputs)
                        for op in (self.native, self.pivot)
                    )

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        run()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    outputs = run()
                for step in range(3):
                    q.copy_(torch.randn(batch, 32, 128, device="cuda").to(q.dtype))
                    w.normal_()
                    lengths.copy_(
                        torch.arange(batch, device="cuda", dtype=torch.int32) * 7
                        + 1500
                        + step * 2000
                    )
                    table.copy_(
                        torch.randperm(blocks, device="cuda")
                        .to(torch.int32)
                        .reshape(batch, pages)
                    )
                    expected = self.compute(
                        self.legacy_native, q, w, cache, params, inputs
                    )
                    graph.replay()
                    for actual in outputs:
                        self.assert_same_indices(actual, expected)


if __name__ == "__main__":
    unittest.main()
