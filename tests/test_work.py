"""Work accounting must be exact, conservative, and CP-invariant."""

import unittest

from grcost.presets import mtgr_layout, resolve_model
from grcost.spec import HstuModel, Layout, LongerModel
from grcost.work import flops_per_example, imbalance, rank_work, total_flops


class TestFlops(unittest.TestCase):
    def setUp(self):
        self.m = resolve_model("mtgr-large")

    def test_causal_dense_only_limit(self):
        """As T -> small, FLOPs -> n_layers * (10 d^2 T + 4 d pairs)."""
        m = HstuModel("x", d=64, n_layers=3)
        got = flops_per_example(m, Layout.causal(8), 8)
        want = 3 * (10 * 64 * 64 * 8) + 3 * (4 * 64 * (8 * 9 // 2))
        self.assertAlmostEqual(got, want, delta=1e-6 * want)

    def test_attention_pairs_are_exactly_quadratic(self):
        """Dense is linear, attention is quadratic; test the quadratic part alone."""
        from grcost.work import rank_work

        m = HstuModel("x", d=64, n_layers=1)
        a = rank_work(m, Layout.causal(1024), 1024, B=1, cp=1)[0]
        b = rank_work(m, Layout.causal(2048), 2048, B=1, cp=1)[0]
        self.assertAlmostEqual(b.flops_attn / a.flops_attn, 4.0, delta=0.01)
        self.assertAlmostEqual(b.flops_dense / a.flops_dense, 2.0, places=9)
        # and the total must land between the two
        self.assertGreater(b.flops / a.flops, 2.0)
        self.assertLess(b.flops / a.flops, 4.0)

    def test_linear_in_batch(self):
        lay = Layout.causal(1024)
        a = total_flops(self.m, lay, 1024, B=8)
        b = total_flops(self.m, lay, 1024, B=16)
        self.assertAlmostEqual(b / a, 2.0, places=9)

    def test_linear_in_layers_and_width(self):
        lay = Layout.causal(512)
        m1 = HstuModel("a", d=256, n_layers=4)
        m2 = HstuModel("b", d=256, n_layers=8)
        self.assertAlmostEqual(
            flops_per_example(m2, lay, 512) / flops_per_example(m1, lay, 512), 2.0
        )

    def test_ffn_moves_the_crossover(self):
        """Adding an FFN of ratio 4 pushes the crossover from 5d to 13d.

        HSTU's fused 4-wide UVQK costs 10 d^2; an FFN of ratio 4 adds 4*4 d^2 =
        16 d^2, so dense goes 10 -> 26 d^2 and T* goes 5d -> 13d.
        (A conventional decoder is a further step away, because its QKV is
        3-wide: 8 + 16 = 24 d^2 -> T* = 12d. ``HstuModel`` models HSTU, so use
        the 5d/13d pair for GR comparisons.)
        """
        hstu = HstuModel("h", d=768, n_layers=1, ffn_ratio=0.0)
        with_ffn = HstuModel("d", d=768, n_layers=1, ffn_ratio=4.0)
        self.assertAlmostEqual(hstu.crossover_len(), 5 * 768, delta=1)
        self.assertAlmostEqual(with_ffn.crossover_len(), 13 * 768, delta=1)
        self.assertAlmostEqual(with_ffn.crossover_len() / hstu.crossover_len(), 2.6, delta=0.01)

    def test_pair_width_matches_the_verified_operator_table(self):
        """The official HSTU layer: H=4, linear_dim=32, attention_dim=16, d=64.

        qk^T MACs = B*H*L^2*attention_dim, p.v MACs = B*H*L^2*linear_dim, so the
        per-pair (summed over heads) FLOPs coefficient is 2*H*(a+v) = 384.
        """
        official = HstuModel("official", d=64, n_layers=1, n_heads=4,
                             attn_dim=16, value_dim=32)
        self.assertEqual(official.pair_width(), 4 * (16 + 32))
        self.assertEqual(official.attn_flops_per_pair(), 384.0)
        # and the dense side: 2*d*H*(3v+2a)
        self.assertEqual(official.dense_flops_per_token(), 2 * 64 * 4 * (96 + 32))

    def test_pair_width_is_4d_for_ordinary_attention(self):
        """MTGR-style: head_dim = d/H for both contractions."""
        for d, h in ((768, 3), (512, 2), (64, 4)):
            m = HstuModel("x", d=d, n_layers=1, n_heads=h)
            self.assertEqual(m.pair_width(), 2 * d)
            self.assertEqual(m.attn_flops_per_pair(), 4.0 * d)
            self.assertAlmostEqual(m.dense_flops_per_token(), 10.0 * d * d, delta=1e-6)

    def test_unequal_widths_break_the_10d2_assumption(self):
        """A concrete reason the FLOPs table needs value_dim, not just d."""
        narrow_v = HstuModel("x", d=64, n_layers=1, n_heads=4, attn_dim=16, value_dim=32)
        self.assertGreater(narrow_v.dense_flops_per_token(), 10 * 64 * 64)

    def test_static_block_costs_more_than_causal(self):
        """A rectangular static block is ~2x the pairs of a triangle."""
        t = 2048
        cand = mtgr_layout(t)
        caus = Layout.causal(t)
        self.assertGreater(cand.pairs_total(), 1.5 * caus.pairs_total())
        self.assertLess(cand.pairs_total(), 2.0 * caus.pairs_total())

    def test_cp_invariance_of_total_work(self):
        lay = mtgr_layout(2048)
        base = total_flops(self.m, lay, 2048, B=4)
        for cp in (2, 4, 8):
            works = rank_work(self.m, lay, 2048, B=4, cp=cp)
            self.assertAlmostEqual(sum(w.flops for w in works) / base, 1.0, places=9)


class TestImbalance(unittest.TestCase):
    def test_hstu_causal_is_the_worst_case(self):
        m = resolve_model("hstu-base")
        lay = Layout.causal(4096)
        w = rank_work(m, lay, 4096, B=8, cp=8, scheme="block")
        self.assertGreater(imbalance(w), 1.8)

    def test_mtgr_is_better_balanced_than_hstu(self):
        m = resolve_model("mtgr-large")
        t = 4096
        h = imbalance(rank_work(m, Layout.causal(t), t, 8, cp=8, scheme="block"))
        g = imbalance(rank_work(m, mtgr_layout(t), t, 8, cp=8, scheme="block"))
        self.assertLess(g, h)

    def test_balancer_helps_only_when_imbalanced(self):
        m = resolve_model("mtgr-large")
        t = 4096
        for lay in (Layout.causal(t), mtgr_layout(t)):
            blk = imbalance(rank_work(m, lay, t, 8, cp=8, scheme="block"))
            bal = imbalance(rank_work(m, lay, t, 8, cp=8, scheme="balanced"))
            self.assertLessEqual(bal, blk + 1e-9)


class TestLonger(unittest.TestCase):
    def test_merge_projection_shards_but_short_layers_do_not(self):
        """LONGER is the Amdahl limit case: only the merge projection shards."""
        m = LongerModel("l", d=256, n_layers=12, merge=4, n_global=2, n_recent=100)
        t = 8192
        w1 = rank_work(m, Layout.causal(t), t, B=8, cp=1)
        w8 = rank_work(m, Layout.causal(t), t, B=8, cp=8)
        # dense shrinks, but nowhere near 8x: the 11 short layers run on m+k
        # tokens and cost more than the merge projection does, so CP on LONGER
        # buys only ~1.5x of dense work.
        ratio = w1[0].flops_dense / w8[0].flops_dense
        self.assertGreater(ratio, 1.3)
        self.assertLess(ratio, 1.7)
        # attention does not shrink at all: the long layer is query-limited and
        # the short layers are replicated
        self.assertAlmostEqual(w8[0].pairs / w1[0].pairs, 1.0, delta=0.02)

    def test_shrinkage_is_8x_when_no_short_layers(self):
        m = LongerModel("l", d=256, n_layers=12, merge=4, n_global=0, n_recent=0)
        t = 8192
        w1 = rank_work(m, Layout.causal(t), t, B=8, cp=1)
        w8 = rank_work(m, Layout.causal(t), t, B=8, cp=8)
        self.assertAlmostEqual(w1[0].flops_dense / w8[0].flops_dense, 8.0, delta=0.01)

    def test_phases(self):
        m = LongerModel("l", d=256, n_layers=12, merge=4, n_global=2, n_recent=100)
        ph = m.phases(4096)
        self.assertEqual(len(ph), 2)
        self.assertEqual(ph[0].n_layers, 1)
        self.assertEqual(ph[0].kv_len, 2 + 4096 // 4)
        self.assertFalse(ph[1].sharded)
        self.assertTrue(ph[0].q_concentrated)


if __name__ == "__main__":
    unittest.main()
