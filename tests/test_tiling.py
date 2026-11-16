"""Tile quantisation and the CP-scheme tension.

The claim these tests pin: a CP degree has a *discrete* cost that has nothing to
do with efficiency or communication, and it becomes visible exactly when a
constraint forces an odd ``cp``.

Two constraints collide on HSTU:

* head-parallel CP (Ulysses) needs ``cp | H``, and HSTU uses ``H = 2..4`` heads;
* ring CP allows any ``cp`` but pays ``cp-1`` sequential steps.

At ``H = 3`` the only multi-card head-parallel choice is ``cp = 3``, whose chunk
is not a tile multiple.  That is the tension, and it is answerable from geometry.
"""

import math
import unittest

from grcost.tiling import (
    DEFAULT_BM, alignment_factor, alignment_table, best_aligned, key_tiles,
    launch_tiles, ring_admissible, scheme_tension, tiles, ulysses_admissible,
    useful_tiles, usefulness, waste,
)


class TestTiles(unittest.TestCase):
    def test_exact_multiples_have_unit_usefulness(self):
        for bm in (64, 128, 256):
            for k in (1, 2, 4, 8, 32):
                self.assertAlmostEqual(usefulness(k * bm, bm), 1.0, places=12)

    def test_partial_tile_wastes(self):
        self.assertAlmostEqual(usefulness(1365.333, 128), 1365.333 / 1408, places=9)
        self.assertLess(usefulness(1365.333, 128), 1.0)

    def test_waste_complements_usefulness(self):
        for c in (100.0, 128.0, 200.0, 1365.333):
            self.assertAlmostEqual(usefulness(c) + waste(c), 1.0, places=12)

    def test_more_tiles_means_less_waste(self):
        """A finer tile wastes less of the tail -- the reason BM matters."""
        self.assertGreater(usefulness(1365.333, 128), usefulness(1365.333, 512))

    def test_tiles_at_least_one(self):
        self.assertEqual(tiles(1.0), 1)
        self.assertEqual(tiles(0.5), 1)

    def test_launch_tiles_is_a_product(self):
        c, T = 1365.333, 4096
        self.assertEqual(launch_tiles(c, T), tiles(c) * key_tiles(T))

    def test_useful_tiles_is_fractional(self):
        self.assertAlmostEqual(useful_tiles(1365.333, 4096), 1365.333 / 128 * 32)


class TestAlignmentFactor(unittest.TestCase):
    def test_one_when_aligned(self):
        for cp in (1, 2, 4, 8, 16):
            self.assertAlmostEqual(alignment_factor(4096 / cp, 4096), 1.0, places=12)

    def test_above_one_when_not(self):
        for cp in (3, 5, 6, 12):
            self.assertGreater(alignment_factor(4096 / cp, 4096), 1.0, msg=f"cp={cp}")

    def test_matches_the_closed_form(self):
        T, BM = 4096, 128
        for cp in (3, 5, 6):
            c = T / cp
            want = (math.ceil(c / BM) * math.ceil(T / BM)) / ((c / BM) * (T / BM))
            self.assertAlmostEqual(alignment_factor(c, T), want, places=12)

    def test_cp3_costs_about_three_percent(self):
        self.assertAlmostEqual(alignment_factor(4096 / 3, 4096), 1.0312, delta=5e-4)

    def test_coarser_tiles_cost_more(self):
        c = 4096 / 6
        self.assertLess(alignment_factor(c, 4096, 128),
                        alignment_factor(c, 4096, 512))


class TestAdmissibility(unittest.TestCase):
    def test_ulysses_requires_divisibility(self):
        self.assertEqual(ulysses_admissible(2), [1, 2])
        self.assertEqual(ulysses_admissible(3), [1, 3])
        self.assertEqual(ulysses_admissible(4), [1, 2, 4])

    def test_ring_allows_anything(self):
        self.assertEqual(ring_admissible(6), [1, 2, 3, 4, 5, 6])

    def test_head_three_is_the_tension_case(self):
        """Ulysses at H=3 offers only cp=3, and cp=3 is the worst-aligned."""
        u = ulysses_admissible(3)
        self.assertEqual([c for c in u if c > 1], [3])
        c3 = usefulness(4096 / 3)
        c4 = usefulness(4096 / 4)
        self.assertLess(c3, c4)

    def test_best_aligned_picks_a_power_of_two_when_available(self):
        self.assertIn(best_aligned(4096, [3, 4, 5, 6, 8]), (4, 8))

    def test_alignment_table_shape(self):
        rows = alignment_table(4096, [1, 2, 3, 4])
        self.assertEqual(len(rows), 4)
        cp, chunk, use, mult = rows[2]
        self.assertEqual(cp, 3)
        self.assertAlmostEqual(chunk, 4096 / 3, places=9)
        self.assertLess(use, 1.0)
        self.assertGreater(mult, 1.0)

    def test_tension_report_names_both_schemes(self):
        text = scheme_tension(3, 4096)
        self.assertIn("Ulysses", text)
        self.assertIn("ring", text)
        self.assertIn("张力", text)

    def test_no_tension_when_heads_is_a_power_of_two(self):
        text = scheme_tension(4, 4096)
        self.assertNotIn("张力", text)


class TestModelIntegration(unittest.TestCase):
    def test_work_carries_the_alignment(self):
        from grcost.presets import mtgr_layout, resolve_model
        from grcost.work import rank_work

        m = resolve_model("mtgr-large")
        lay = mtgr_layout(4096)
        self.assertGreater(rank_work(m, lay, 4096, 32, cp=3)[0].align, 1.0)
        self.assertAlmostEqual(rank_work(m, lay, 4096, 32, cp=4)[0].align, 1.0)

    def test_alignment_raises_attention_flops(self):
        from grcost.presets import mtgr_layout, resolve_model
        from grcost.work import rank_work

        m = resolve_model("mtgr-large")
        lay = mtgr_layout(4096)
        on = rank_work(m, lay, 4096, 32, cp=3, quantise=True)[0]
        off = rank_work(m, lay, 4096, 32, cp=3, quantise=False)[0]
        self.assertGreater(on.flops_attn, off.flops_attn)
        self.assertAlmostEqual(on.flops_dense, off.flops_dense, places=6)

    def test_prediction_can_be_switched(self):
        from grcost.cost import Params, predict
        from grcost.hardware import ASCEND_910C
        from grcost.presets import mtgr_layout, resolve_model

        m = resolve_model("mtgr-large")
        lay = mtgr_layout(4096)
        p = Params(eff_inf=0.42, c_star=1800, gamma=1.1, alpha=15e-6,
                   bw_comm=149e9, overlap=0.0, bw_util=0.55, recompute=1.0)
        a = predict(m, lay, 4096, 32, 3, ASCEND_910C, p.copy(quantise_tiles=True))
        b = predict(m, lay, 4096, 32, 3, ASCEND_910C, p.copy(quantise_tiles=False))
        self.assertGreater(a, b)


if __name__ == "__main__":
    unittest.main()
