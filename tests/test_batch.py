"""Batch is not a free axis: the efficiency curve is a slice at one batch.

These tests pin the reason. ``eff`` is a function of tiles per device, which is
``B*T/CP``. A curve fitted at one batch describes one slice of that surface, so
moving the batch by 32x is not a small extrapolation -- it is a factor-of-32 move
along the very coordinate the curve was fitted in.

What the tests do *not* do is pretend the curve can be corrected. It cannot be,
for free: ``eff`` saturates, so extrapolating it far below the knee drives it
toward zero, which no real kernel does (the head and KV dimensions still fill
part of the machine). The honest response is a warning plus a gate, and that is
what is tested here.
"""

import unittest

from grcost.cost import (
    Params, batch_warning, eff_coordinate, predict,
)
from grcost.hardware import ASCEND_910C
from grcost.presets import mtgr_layout, resolve_model

M = resolve_model("mtgr-large")
LAY = mtgr_layout(4096)
P = Params(eff_inf=0.42, c_star=1800.0, gamma=1.1, alpha=15e-6,
           bw_comm=149e9, overlap=0.0, bw_util=0.55, recompute=1.0,
           t_emb_ms=3.0)


class TestLinearTerms(unittest.TestCase):
    """Compute, elementwise and communication all scale with B; fixed does not."""

    def test_compute_and_bytes_are_linear_in_batch(self):
        """Assert on the breakdown: the two byte-proportional terms are linear.

        The *total* is not, because the collective latency and the embedding
        lookup sit outside the proportionality -- which is precisely the point of
        the next test.
        """
        _, a = predict(M, LAY, 4096, 8, 4, ASCEND_910C, P, t_fixed_ms=0.0,
                       detail=True)
        _, b = predict(M, LAY, 4096, 16, 4, ASCEND_910C, P, t_fixed_ms=0.0,
                       detail=True)
        self.assertAlmostEqual(b["compute_ms"] / a["compute_ms"], 2.0, places=6)
        self.assertAlmostEqual(b["elementwise_ms"] / a["elementwise_ms"], 2.0,
                               places=6)

    def test_total_growth_deficit_equals_the_fixed_share(self):
        """Doubling B multiplies the total by ``2 - fixed_share``, exactly.

        Since only the fixed term stays put,
            t(2B) = 2*(t(B) - f) + f   =>   t(2B)/t(B) = 2 - f/t(B)
        so the shortfall from 2x *is* the fixed share of the smaller step. This
        is the cleanest statement of why small batches make CP look useless.
        """
        a, bd = predict(M, LAY, 4096, 8, 4, ASCEND_910C, P, t_fixed_ms=4.0,
                        detail=True)
        b = predict(M, LAY, 4096, 16, 4, ASCEND_910C, P, t_fixed_ms=4.0)
        share = bd["fixed_ms"] / bd["step_ms"]
        self.assertAlmostEqual(2.0 - b / a, share, delta=0.02)
        self.assertLess(b / a, 2.0)

    def test_fixed_part_is_identical_at_both_batches(self):
        _, a = predict(M, LAY, 4096, 8, 4, ASCEND_910C, P, t_fixed_ms=4.0,
                       detail=True)
        _, b = predict(M, LAY, 4096, 16, 4, ASCEND_910C, P, t_fixed_ms=4.0,
                       detail=True)
        self.assertAlmostEqual(a["fixed_ms"], b["fixed_ms"], places=9)
        self.assertAlmostEqual(a["t_emb_ms"], b["t_emb_ms"], places=9)

    def test_fixed_share_explodes_as_batch_shrinks(self):
        small, bd1 = predict(M, LAY, 4096, 1, 4, ASCEND_910C, P, 4.0, detail=True)
        big, bd32 = predict(M, LAY, 4096, 32, 4, ASCEND_910C, P, 4.0, detail=True)
        self.assertGreater(bd1["fixed_ms"] / bd1["step_ms"],
                           5 * bd32["fixed_ms"] / bd32["step_ms"])


class TestCoordinate(unittest.TestCase):
    def test_coordinate_is_B_times_chunk(self):
        self.assertEqual(eff_coordinate(1, 1024), 1024)
        self.assertEqual(eff_coordinate(32, 1024), 32768)

    def test_moving_batch_moves_the_coordinate(self):
        """Same chunk, different batch: the curve is evaluated in a different place."""
        self.assertEqual(eff_coordinate(1, 1024) * 32, eff_coordinate(32, 1024))

    def test_detail_reports_the_coordinate(self):
        _, bd = predict(M, LAY, 4096, 4, 4, ASCEND_910C, P, detail=True)
        self.assertAlmostEqual(bd["tokens_per_device"], 4 * 4096 / 4)


class TestWarning(unittest.TestCase):
    def test_silent_within_a_factor_of_two(self):
        self.assertIsNone(batch_warning(P, 16))
        self.assertIsNone(batch_warning(P, 32))
        self.assertIsNone(batch_warning(P, 64))

    def test_warns_below(self):
        msg = batch_warning(P, 1)
        self.assertIsNotNone(msg)
        self.assertIn("Re-run the L1 sweep", msg)

    def test_warns_above(self):
        self.assertIsNotNone(batch_warning(P, 128))

    def test_warning_names_the_ratio(self):
        self.assertIn("32x", batch_warning(P, 1))

    def test_guard_follows_the_calibration_batch(self):
        p8 = P.copy(b_cal=8.0)
        self.assertIsNone(batch_warning(p8, 8))
        self.assertIsNotNone(batch_warning(p8, 1))

    def test_no_warning_when_uncalibrated(self):
        self.assertIsNone(batch_warning(P.copy(b_cal=0.0), 1))

    def test_detail_carries_the_warning(self):
        _, bd = predict(M, LAY, 4096, 1, 4, ASCEND_910C, P, detail=True)
        self.assertIsNotNone(bd["batch_warning"])


class TestSensitivity(unittest.TestCase):
    """How much a wrong eff costs, and why that is the reassuring part.

    At B=1 the fixed term is over half the step, so a 2x error in eff moves the
    total by roughly a quarter. At B=32 the compute term dominates and the same
    error moves it by about 60%. Counter-intuitive consequence: predicting at
    B=1 is *more* robust to an uncalibrated eff -- and correspondingly, a B=1
    experiment carries almost no information about eff.
    """

    def _ratio(self, B):
        a = predict(M, LAY, 4096, B, 4, ASCEND_910C, P, t_fixed_ms=4.0)
        b = predict(M, LAY, 4096, B, 4, ASCEND_910C,
                    P.copy(eff_inf=P.eff_inf / 2), t_fixed_ms=4.0)
        return b / a

    def test_small_batch_is_less_sensitive(self):
        self.assertLess(self._ratio(1), self._ratio(32))

    def test_small_batch_error_stays_under_a_third(self):
        self.assertLess(self._ratio(1), 1.36)

    def test_large_batch_error_is_large(self):
        self.assertGreater(self._ratio(32), 1.5)


class TestOrdering(unittest.TestCase):
    def test_cp_ordering_survives_the_batch_change(self):
        """The ranking is unchanged; only its spread compresses."""
        def speeds(B):
            t1 = predict(M, LAY, 4096, B, 1, ASCEND_910C, P, 4.0)
            return [t1 / predict(M, LAY, 4096, B, cp, ASCEND_910C, P, 4.0)
                    for cp in (2, 4, 8)]

        s1, s32 = speeds(1), speeds(32)
        self.assertEqual([v > 1 for v in s1], [v > 1 for v in s32])
        self.assertEqual(s1, sorted(s1))
        self.assertEqual(s32, sorted(s32))
        # and the gain is smaller at B=1, because fixed cost dominates
        self.assertLess(s1[-1], s32[-1])


if __name__ == "__main__":
    unittest.main()
