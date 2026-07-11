"""Phase 1 decode indexer top-k reuse — algorithm-level validation.

These tests do NOT need the FP8 model / GPU. They validate:
  1. ``_indexer_reuse_cfg`` env parsing.
  2. The two-tier selection *logic* the code implements (fine top-K within a
     grouped coarse pool), including the hard invariant that with a coarse pool
     large enough to cover the whole context the reuse output is byte-identical
     to the per-step baseline top-K, and a soft recall check under query drift
     that mirrors the real coarse(group-middle) vs fine(per-step) mismatch.

Run: python -m pytest rtp_llm/models_py/modules/dsv4/test/test_indexer_decode_reuse.py -q
"""

import os
import unittest

import torch


# ---- pure-torch reference of the selection the code implements ----------
def _baseline_topk(fine_scores: torch.Tensor, length: int, K: int) -> torch.Tensor:
    """Per-step top-K over the full context (what baseline does)."""
    s = fine_scores.clone()
    s[length:] = float("-inf")
    K_eff = min(K, length)
    return torch.sort(s.topk(K_eff, dim=-1)[1]).values


def _reuse_topk(
    fine_scores: torch.Tensor,
    coarse_scores: torch.Tensor,
    length: int,
    K: int,
    C: int,
) -> torch.Tensor:
    """Coarse top-C (from coarse_scores) then fine top-K within it (mirrors
    ``IndexerFP8._topk_reuse_select`` fine path)."""
    cs = coarse_scores.clone()
    cs[length:] = float("-inf")
    C_eff = min(C, length)
    cand = cs.topk(C_eff, dim=-1)[1]  # [C_eff] global idx
    fs = fine_scores.gather(0, cand)
    K_eff = min(K, C_eff)
    local = fs.topk(K_eff, dim=-1)[1]
    return torch.sort(cand.gather(0, local)).values


class TestIndexerReuseCfg(unittest.TestCase):
    def _cfg(self, **env):
        from rtp_llm.models_py.modules.dsv4.fp8.indexer import _indexer_reuse_cfg

        old = {k: os.environ.get(k) for k in env}
        try:
            for k, v in env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            return _indexer_reuse_cfg()
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_disabled_by_default(self):
        self.assertIsNone(self._cfg(DSV4_INDEXER_REUSE="0"))

    def test_defaults(self):
        cfg = self._cfg(
            DSV4_INDEXER_REUSE="1",
            DSV4_INDEXER_REUSE_GROUP=None,
            DSV4_INDEXER_REUSE_COARSE=None,
            DSV4_INDEXER_REUSE_ROPE_OFFSET=None,
        )
        self.assertEqual(cfg, (16, 8192, 8))  # offset -1 -> G//2 == 8

    def test_overrides(self):
        cfg = self._cfg(
            DSV4_INDEXER_REUSE="on",
            DSV4_INDEXER_REUSE_GROUP="32",
            DSV4_INDEXER_REUSE_COARSE="4096",
            DSV4_INDEXER_REUSE_ROPE_OFFSET="12",
        )
        self.assertEqual(cfg, (32, 4096, 12))


class TestReuseSelectionLogic(unittest.TestCase):
    def test_coarse_covers_all_is_exact(self):
        """C >= length: reuse must equal baseline top-K exactly, for any
        coarse/fine mismatch (fine re-scores within the full candidate set)."""
        torch.manual_seed(0)
        T, K = 4000, 2048
        for _ in range(20):
            fine = torch.randn(T)
            coarse = torch.randn(T)  # arbitrary / unrelated coarse ranking
            length = int(torch.randint(K + 1, T + 1, (1,)))
            b = _baseline_topk(fine, length, K)
            r = _reuse_topk(fine, coarse, length, K, C=T)
            self.assertTrue(torch.equal(b, r))

    def test_short_context_padding(self):
        """length < K: both return all valid indices."""
        T, K, C = 5000, 2048, 8192
        fine = torch.randn(T)
        coarse = torch.randn(T)
        length = 1500
        b = _baseline_topk(fine, length, K)
        r = _reuse_topk(fine, coarse, length, K, C)
        self.assertTrue(torch.equal(b, r))

    def test_recall_under_group_drift(self):
        """Simulate a 16-step group: coarse built once at the group center,
        fine changes per step (query + position drift). Report worst-step
        recall of the reused top-2048 vs the per-step baseline top-2048.

        Models each key's per-step score as a stable relevance plus a
        small step-dependent perturbation (~ query/RoPE drift across 16
        steps). With C=8192 (4x headroom) recall should stay very high.
        """
        torch.manual_seed(1)
        T, K, C, G = 131072, 2048, 8192, 16
        relevance = torch.randn(T)  # stable per-key affinity
        # coarse query = group center (step 8); fine = each step 0..15.
        # drift grows with |step - center| (positional) plus small content noise.
        center = G // 2
        drift_scale = 0.15  # relative to unit-variance relevance
        length = T
        coarse = relevance + drift_scale * (center / G) * torch.randn(T)
        worst = 1.0
        for step in range(G):
            fine = relevance + drift_scale * (abs(step - center) / G + 0.3) * torch.randn(T)
            b = set(_baseline_topk(fine, length, K).tolist())
            r = set(_reuse_topk(fine, coarse, length, K, C).tolist())
            recall = len(b & r) / len(b)
            worst = min(worst, recall)
        print(f"\n[drift recall] worst-step recall over G={G}, C={C}: {worst:.4f}")
        # Soft gate: 4x headroom should keep recall high under modeled drift.
        self.assertGreater(worst, 0.95)


class TestCompactSlotMapping(unittest.TestCase):
    def test_compact_gather_matches_paged_lookup(self):
        """`_reuse_compact_slot_index` + index_select must reproduce the exact
        132B slots the paged kernel would address for each candidate:
        abs = bt[b, t//eb]*eb + t%eb."""
        from rtp_llm.models_py.modules.dsv4.fp8.indexer import (
            _reuse_compact_slot_index,
        )

        torch.manual_seed(0)
        B, eb, D = 3, 64, 132
        num_blocks, max_blocks = 40, 20
        pool = torch.randn(num_blocks * eb, D)
        # random physical block table per row (distinct blocks)
        bt = torch.stack(
            [torch.randperm(num_blocks)[:max_blocks] for _ in range(B)]
        ).to(torch.int32)
        C = 500
        coarse_idx = torch.randint(0, max_blocks * eb, (B, C), dtype=torch.int32)
        # sprinkle some -1 pads
        coarse_idx[:, -10:] = -1

        abs_slot = _reuse_compact_slot_index(coarse_idx, bt, eb)
        got = pool.index_select(0, abs_slot.reshape(-1)).view(B, C, D)

        # reference: element-wise paged lookup
        for b in range(B):
            for c in range(C):
                t = int(coarse_idx[b, c])
                tt = max(t, 0)
                phys = int(bt[b, tt // eb])
                ref = pool[phys * eb + (tt % eb)]
                self.assertTrue(torch.equal(got[b, c], ref))


if __name__ == "__main__":
    unittest.main(verbosity=2)
