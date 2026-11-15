"""Verify the CP=1 sweep really can be inverted into the efficiency curve.

``extract_eff`` solves for ``eff`` algebraically, so if the forward model is
self-consistent the round-trip must be exact: generate times from a known
parameter set, extract, refit, and recover the parameters.
"""

import unittest

from grcost.compose import extract_eff, fit_curve, report
from grcost.cost import Params, predict
from grcost.hardware import H100_SXM
from grcost.presets import mtgr_layout, resolve_model

TRUE = Params(eff_inf=0.42, c_star=1800.0, gamma=1.1, alpha=0.0, bw_comm=1e12,
              overlap=0.0, bw_util=0.55, recompute=1.0)
M = resolve_model("mtgr-large")
LENGTHS = [512, 1024, 2048, 4096, 8192, 16384]


def fake_measurements(p=TRUE, B=32, t_fixed=0.0):
    out = []
    for T in LENGTHS:
        lay = mtgr_layout(T)
        ms = predict(M, lay, T, B, 1, H100_SXM, p, t_fixed_ms=t_fixed)
        out.append((T, ms))
    return out


class TestExtraction(unittest.TestCase):
    def test_round_trip_recovers_parameters(self):
        pts = extract_eff(M, mtgr_layout, fake_measurements(), H100_SXM, B=32,
                          bw_util=TRUE.bw_util)
        for pt in pts:
            self.assertAlmostEqual(
                pt.eff, 0.42 / (1 + (1800.0 / pt.chunk) ** 1.1), places=6,
                msg=f"c={pt.chunk}")
        p, rmse = fit_curve(pts)
        self.assertAlmostEqual(p.eff_inf, 0.42, delta=0.02)
        self.assertAlmostEqual(p.c_star / 1800.0, 1.0, delta=0.15)
        self.assertAlmostEqual(p.gamma / 1.1, 1.0, delta=0.15)
        self.assertLess(rmse, 0.02)

    def test_works_with_a_fixed_cost_present(self):
        pts = extract_eff(M, mtgr_layout, fake_measurements(t_fixed=8.0),
                          H100_SXM, B=32, t_fixed_ms=8.0, bw_util=TRUE.bw_util)
        p, rmse = fit_curve(pts)
        self.assertAlmostEqual(p.eff_inf, 0.42, delta=0.03)
        self.assertLess(rmse, 0.05)

    def test_wrong_fixed_cost_is_visible(self):
        """Understating t_fixed biases the *small-chunk* points low.

        The bias is not a uniform scale factor: an 8 ms offset eats a large
        fraction of a short run and almost nothing of a long one, so the curve
        gets bent and the fit compensates by moving c_star. The tell-tale is a
        worse residual plus a distorted c_star, not a lower eff_inf.
        """
        good = extract_eff(M, mtgr_layout, fake_measurements(t_fixed=8.0),
                           H100_SXM, B=32, t_fixed_ms=8.0, bw_util=TRUE.bw_util)
        bad = extract_eff(M, mtgr_layout, fake_measurements(t_fixed=8.0),
                          H100_SXM, B=32, t_fixed_ms=0.0, bw_util=TRUE.bw_util)
        self.assertLess(bad[0].eff, good[0].eff)         # small c biased low
        self.assertAlmostEqual(bad[-1].eff, good[-1].eff, delta=0.02)  # large c ok
        _, rmse_good = fit_curve(good)
        _, rmse_bad = fit_curve(bad)
        self.assertGreater(rmse_bad, rmse_good)

    def test_unusable_points_are_flagged_not_silently_dropped(self):
        pts = extract_eff(M, mtgr_layout, fake_measurements(), H100_SXM, B=32,
                          t_fixed_ms=1e9, bw_util=TRUE.bw_util)
        self.assertTrue(all("unusable" in pt.note for pt in pts))

    def test_report_is_readable(self):
        pts = extract_eff(M, mtgr_layout, fake_measurements(), H100_SXM, B=32,
                          bw_util=TRUE.bw_util)
        p, rmse = fit_curve(pts)
        text = report(pts, p, rmse)
        self.assertIn("eff_inf", text)
        self.assertIn("predicted eff", text)


if __name__ == "__main__":
    unittest.main()
