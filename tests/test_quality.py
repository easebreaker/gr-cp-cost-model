"""The quality model must be monotone, invertible, and honest about identifiability."""

import unittest

from grcost.quality import QualityLaw, QualityPoint, demo_law, fit_quality


class TestShape(unittest.TestCase):
    def setUp(self):
        self.law = QualityLaw(q_inf=0.75, a_len=0.03, alpha=0.4,
                              a_size=0.02, beta=0.25,
                              a_data=0.04, gamma=0.3)

    def test_monotone_in_every_lever(self):
        q = self.law.quality
        self.assertLess(q(1000, 1e8, 1e11), q(2000, 1e8, 1e11))
        self.assertLess(q(1000, 5e7, 1e11), q(1000, 1e8, 1e11))
        self.assertLess(q(1000, 1e8, 5e10), q(1000, 1e8, 1e11))

    def test_bounded_above_by_asymptote(self):
        q = self.law.quality(10 ** 9, 10 ** 14, 10 ** 18)
        self.assertLess(q, self.law.q_inf)          # approaches from below
        self.assertGreater(q, self.law.q_inf - 0.01)

    def test_saturating_returns(self):
        """Equal multiplicative steps must give shrinking absolute gains."""
        q = self.law.quality
        d1 = q(2000, 1e8, 1e11) - q(1000, 1e8, 1e11)
        d2 = q(4000, 1e8, 1e11) - q(2000, 1e8, 1e11)
        d3 = q(8000, 1e8, 1e11) - q(4000, 1e8, 1e11)
        self.assertGreater(d1, d2)
        self.assertGreater(d2, d3)

    def test_marginal_length_decreases(self):
        self.assertGreater(self.law.marginal_length(1000),
                           self.law.marginal_length(10000))

    def test_length_for_deficit_inverts_the_model(self):
        target = 0.010
        L = self.law.length_for_deficit(target)
        got = self.law.a_len * (self.law.l_ref / L) ** self.law.alpha
        self.assertAlmostEqual(got, target, places=9)

    def test_length_gain_is_not_monotone_in_alpha(self):
        """The reason ``breakeven_alpha`` scans instead of bisecting.

        Doubling from 4096 to 8192 with ``l_ref = 1000``: a tiny ``alpha`` means
        no return to length, a huge ``alpha`` means the return already happened
        below 4096. The gain peaks in between, so a bisection on a bracket that
        straddles the peak silently fails.
        """
        law = QualityLaw(l_ref=1000.0, a_len=0.03)

        def gain(a):
            return (law.copy(alpha=a).quality(8192, 1e9, 1e15)
                    - law.copy(alpha=a).quality(4096, 1e9, 1e15))

        vals = [gain(a) for a in (0.05, 0.2, 0.5, 1.0, 2.0, 4.0)]
        peak = max(range(len(vals)), key=lambda i: vals[i])
        self.assertTrue(0 < peak < len(vals) - 1, f"no interior peak: {vals}")
        self.assertLess(vals[-1], vals[peak])

    def test_demo_law_is_marked_unsourced(self):
        self.assertFalse(demo_law().sourced)
        self.assertIn("UNSOURCED", demo_law().describe())


class TestFitting(unittest.TestCase):
    def setUp(self):
        self.truth = QualityLaw(q_inf=0.74, a_len=0.032, alpha=0.42,
                                a_size=0.018, beta=0.24,
                                a_data=0.045, gamma=0.33)

    def _grid(self):
        pts = []
        for L in (1024, 4096, 16384):
            for m in (5e7, 4e8):
                for D in (2e11, 2e12):
                    pts.append(QualityPoint(L, m, D,
                                            self.truth.quality(L, m, D), "grid"))
        return pts

    def test_recovers_parameters_from_a_three_lever_grid(self):
        law, rmse, identifiable = fit_quality(self._grid(), self.truth)
        self.assertTrue(identifiable)
        self.assertLess(rmse, 0.005)
        self.assertAlmostEqual(law.q_inf, 0.74, delta=0.02)
        self.assertAlmostEqual(law.alpha / 0.42, 1.0, delta=0.25)
        self.assertAlmostEqual(law.beta / 0.24, 1.0, delta=0.25)

    def test_single_lever_grid_is_flagged_not_guessed(self):
        """Only L varies: `a_size` and `a_data` trade off against `q_inf`."""
        pts = [QualityPoint(L, 1e8, 1e11, self.truth.quality(L, 1e8, 1e11))
               for L in (1024, 2048, 4096, 8192, 16384)]
        _, _, identifiable = fit_quality(pts, self.truth)
        self.assertFalse(identifiable)

    def test_too_few_points_raises(self):
        with self.assertRaises(ValueError):
            fit_quality([QualityPoint(1024, 1e8, 1e11, 0.7)])


if __name__ == "__main__":
    unittest.main()
