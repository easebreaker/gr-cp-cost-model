"""The collective model must explain size dependence, not fit it piecewise.

The tests here pin the three claims the module makes:

1. apparent bandwidth is a saturating function of message size, with the same
   shape as ``eff(c)`` -- so one alpha and one bw explain a whole curve;
2. the break-even message size V* = S*alpha*BW scales with the step count, hence
   with the rank count -- which is why a piecewise fit cannot transfer;
3. rounds (how many collectives) and steps (how many hops inside one) are
   independent, and conflating them inflates the latency term by cp-1.
"""

import unittest

from grcost.comm import (
    CommModel, allgather_peak_buffer, allgather_volume_per_rank,
    diagnose_size_dependence, steps_recursive_doubling, steps_ring,
)


class TestSteps(unittest.TestCase):
    def test_recursive_doubling_is_log2(self):
        self.assertEqual(steps_recursive_doubling(1), 0)
        self.assertEqual(steps_recursive_doubling(2), 1)
        self.assertEqual(steps_recursive_doubling(4), 2)
        self.assertEqual(steps_recursive_doubling(8), 3)
        self.assertEqual(steps_recursive_doubling(3), 2)   # rounds up

    def test_ring_is_n_minus_one(self):
        self.assertEqual(steps_ring(8), 7)

    def test_ring_is_worse_than_doubling_in_step_count(self):
        """Equal at n=2 (both one hop); ring pulls away from n=4."""
        self.assertEqual(steps_ring(2), steps_recursive_doubling(2))
        for n in (4, 8, 16):
            self.assertGreater(steps_ring(n), steps_recursive_doubling(n))


class TestModel(unittest.TestCase):
    def setUp(self):
        self.m = CommModel(alpha=15e-6, bw=149e9, policy="auto", overlap=0.0)

    def test_zero_for_single_rank(self):
        self.assertEqual(self.m.time(1, 1e9), 0.0)

    def test_monotone_in_bytes(self):
        ts = [self.m.time(8, v) for v in (1e5, 1e6, 1e7, 1e8)]
        self.assertEqual(ts, sorted(ts))

    def test_asymptotic_bandwidth(self):
        """Large messages must approach bw."""
        big = 1e12
        got = self.m.apparent_bw(8, big)
        self.assertGreater(got, 0.95 * self.m.bw)

    def test_small_messages_look_much_slower(self):
        """This is the whole 'bandwidth depends on size' phenomenon."""
        small = self.m.apparent_bw(8, 65536)
        big = self.m.apparent_bw(8, 1e10)
        self.assertLess(small, big / 50)

    def test_apparent_bandwidth_is_saturating(self):
        """BW_app = BW / (1 + V*/V).

        Saturation here means the *relative* gain shrinks, not the absolute one:
        the derivative of V/(V+V*) peaks at V = V* and falls on both sides, so the
        curve has an inflection exactly at the break-even size.  That inflection
        is why people reach for a piecewise fit -- it looks like two regimes.
        """
        vs = [1e6, 4e6, 1.6e7, 6.4e7, 2.56e8]
        bws = [self.m.apparent_bw(8, v) for v in vs]
        self.assertEqual(bws, sorted(bws))
        rel = [bws[i + 1] / bws[i] - 1 for i in range(len(bws) - 1)]
        self.assertEqual(rel, sorted(rel, reverse=True))

    def test_inflection_sits_at_the_break_even_size(self):
        """The steepest rise is across V*, not off in either tail.

        ``V/(V+V*)`` is symmetric in log V about V*, so the gain from V*/2 to 2V*
        exceeds the gain over the same ratio far from the knee.
        """
        v_star = self.m.break_even_bytes(8)
        across = self.m.apparent_bw(8, v_star * 2) - self.m.apparent_bw(8, v_star / 2)
        far_below = (self.m.apparent_bw(8, v_star / 16)
                     - self.m.apparent_bw(8, v_star / 64))
        far_above = (self.m.apparent_bw(8, v_star * 64)
                     - self.m.apparent_bw(8, v_star * 16))
        self.assertGreater(across, far_below)
        self.assertGreater(across, far_above)

    def test_break_even_scales_with_step_count(self):
        v2 = self.m.break_even_bytes(2)
        v8 = self.m.break_even_bytes(8)
        self.assertAlmostEqual(v8 / v2, steps_recursive_doubling(8)
                               / steps_recursive_doubling(2), places=6)

    def test_break_even_is_where_the_two_terms_are_equal(self):
        for n in (2, 4, 8):
            v = self.m.break_even_bytes(n)
            lat, xfer = self.m.time_split(n, v)
            self.assertAlmostEqual(lat, xfer, delta=1e-12 + 1e-3 * xfer)

    def test_below_break_even_latency_dominates(self):
        v = self.m.break_even_bytes(8) / 10
        lat, xfer = self.m.time_split(8, v)
        self.assertGreater(lat, 5 * xfer)

    def test_above_break_even_bandwidth_dominates(self):
        v = self.m.break_even_bytes(8) * 20
        lat, xfer = self.m.time_split(8, v)
        self.assertGreater(xfer, 5 * lat)

    def test_rounds_multiply_only_the_latency(self):
        v = 1e9
        t1 = self.m.time(8, v, rounds=1)
        t15 = self.m.time(8, v, rounds=15)
        l1, x1 = self.m.time_split(8, v, rounds=1)
        l15, x15 = self.m.time_split(8, v, rounds=15)
        self.assertAlmostEqual(x1, x15, places=12)          # bytes unchanged
        self.assertAlmostEqual(l15 / l1, 15.0, places=6)    # latency x15
        self.assertGreater(t15, t1)

    def test_overlap_removes_latency_not_bytes(self):
        no = CommModel(alpha=15e-6, bw=149e9, overlap=0.0)
        yes = CommModel(alpha=15e-6, bw=149e9, overlap=1.0)
        v = 1e8
        self.assertAlmostEqual(no.time_split(8, v)[1], yes.time_split(8, v)[1])
        self.assertGreater(no.time(8, v), yes.time(8, v))

    def test_v_switch_changes_step_count_only_above_it(self):
        m = CommModel(alpha=15e-6, bw=149e9, v_switch=1e8)
        self.assertEqual(m.steps_for(8, 1e7), steps_recursive_doubling(8))
        self.assertEqual(m.steps_for(8, 1e9), steps_ring(8))


class TestVolume(unittest.TestCase):
    def test_allgather_volume_matches_ring(self):
        """Ring and all-gather move the same volume; they differ in steps."""
        for n in (2, 4, 8):
            self.assertAlmostEqual(allgather_volume_per_rank(1e9, n),
                                   1e9 * (n - 1) / n)

    def test_allgather_needs_the_whole_buffer(self):
        """Peak memory is where all-gather differs from ring, not bandwidth."""
        self.assertEqual(allgather_peak_buffer(1e9, 8), 1e9)


class TestDiagnose(unittest.TestCase):
    def _curve(self, m, n, sizes):
        return [(v, m.time(n, v)) for v in sizes]

    def test_explains_a_clean_curve(self):
        m = CommModel(alpha=20e-6, bw=150e9)
        text = diagnose_size_dependence(self._curve(m, 8, (1e5, 1e6, 1e7, 1e8, 1e9)), 8)
        self.assertIn("break-even", text)
        self.assertIn("amortisation", text)
        self.assertNotIn("does not explain", text)

    def test_refuses_a_curve_with_two_bandwidths(self):
        """What a real algorithm switch looks like: the slope changes.

        A step-count change alone is *not* detectable from a curve whose regimes
        are far apart -- in the bandwidth regime the intercept stops mattering.
        So the diagnosable signature is two different effective bandwidths, which
        is what an algorithm change usually produces.
        """
        slow = CommModel(alpha=20e-6, bw=60e9)
        fast = CommModel(alpha=20e-6, bw=150e9)
        pts = ([(v, slow.time(8, v)) for v in (1e7, 3e7, 1e8)]
               + [(v, fast.time(8, v)) for v in (3e8, 1e9, 3e9)])
        text = diagnose_size_dependence(pts, 8)
        self.assertIn("does not explain", text)
        self.assertIn("Do NOT fit piecewise breakpoints", text)
        self.assertIn("slope", text)

    def test_held_out_check_is_the_discriminator(self):
        """In-sample residual is a weak test; extrapolating down in size is not.

        A free intercept absorbs a surprising amount of structure, so the two
        cases here both fit well in-sample yet differ by an order of magnitude
        when the fit is asked to predict the small messages.
        """
        one = CommModel(alpha=15e-6, bw=149e9)
        clean = [(v, one.time(8, v))
                 for v in (1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9)]
        text = diagnose_size_dependence(clean, 8)
        self.assertIn("held-out", text)
        self.assertNotIn("does not explain", text)

    def test_clean_curve_is_recovered_exactly(self):
        one = CommModel(alpha=15e-6, bw=149e9)
        clean = [(v, one.time(8, v))
                 for v in (1e5, 1e6, 1e7, 1e8, 1e9)]
        text = diagnose_size_dependence(clean, 8)
        self.assertIn("alpha=15.0us", text)
        self.assertIn("149", text)

    def test_reports_the_regime_per_point(self):
        m = CommModel(alpha=15e-6, bw=149e9)
        text = diagnose_size_dependence(
            self._curve(m, 8, (1e4, 1e5, 1e6, 1e8, 1e10)), 8)
        self.assertIn("latency", text)
        self.assertIn("bandwidth", text)

    def test_too_few_points(self):
        self.assertIn("need >= 3", diagnose_size_dependence([(1e6, 1e-3)], 8))


if __name__ == "__main__":
    unittest.main()
