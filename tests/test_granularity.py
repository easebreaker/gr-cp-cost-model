"""Regression tests for the two reported deviations.

1. **Absolute time too low.** The sweep timed one HSTU block, but ``extract_eff``
   divided its time by a 15-layer FLOP count. ``eff`` comes out exactly
   ``n_layers`` too large, and the prediction then lands at the per-block scale.
   The fit is internally consistent, so nothing looks wrong -- which is why this
   needs a test rather than a careful reader.

2. **Speedup too low.** Two independent causes: a missing ``t_fixed`` (everything
   that does not shard) and a communication term taken from defaults instead of
   from a measured collective. ``cost.attribute`` makes the two separable, and
   these tests pin its arithmetic.
"""

import unittest
from dataclasses import replace

from grcost.compose import extract_eff, fit_curve
from grcost.cost import Params, attribute, attribution_report, predict
from grcost.hardware import ASCEND_910B, H100_SXM
from grcost.presets import mtgr_layout, resolve_model

TRUE = Params(eff_inf=0.42, c_star=1800.0, gamma=1.1, alpha=25e-6,
              bw_comm=120e9, overlap=0.5, bw_util=0.55, recompute=1.0)
FULL = resolve_model("mtgr-large")      # 15 layers
LENGTHS = [512, 1024, 2048, 4096, 8192]


def block_measurements(B=32, t_fixed=0.0, device=H100_SXM):
    """Times of ONE block, as ``sweep.py --layers 1`` would report them."""
    out = []
    for T in LENGTHS:
        ms = predict(FULL, mtgr_layout(T), T, B, 1, device, TRUE, t_fixed_ms=t_fixed)
        out.append((T, ms / FULL.n_layers))
    return out


class TestLayerGranularity(unittest.TestCase):
    def test_true_efficiency_is_recovered_when_depth_is_declared(self):
        pts = extract_eff(FULL, mtgr_layout, block_measurements(), H100_SXM,
                          layers_measured=1, bw_util=TRUE.bw_util)
        for pt in pts:
            want = 0.42 / (1 + (1800.0 / pt.chunk) ** 1.1)
            self.assertAlmostEqual(pt.eff, want, places=6, msg=f"c={pt.chunk}")
            self.assertEqual(pt.n_layers, 1)

    def test_undeclared_depth_inflates_eff_by_at_least_n_layers(self):
        """No ``layers_measured`` means 'the timed module is the whole model'.

        Two things go wrong at once, and the second is worse than a scale error:
        the FLOPs are ``n_layers`` too large (inflating ``eff``) *and* the
        elementwise term subtracted from the measurement is ``n_layers`` too
        large. Once that term exceeds the measured time the point becomes
        unusable rather than merely wrong -- so the failure is loud.
        """
        trust = extract_eff(FULL, mtgr_layout, block_measurements(), H100_SXM,
                            layers_measured=1, bw_util=TRUE.bw_util)
        wrong = extract_eff(FULL, mtgr_layout, block_measurements(), H100_SXM,
                            bw_util=TRUE.bw_util)
        self.assertGreater(wrong[0].eff / trust[0].eff, FULL.n_layers)
        usable = [p for p in wrong if p.eff == p.eff]
        self.assertLess(len(usable), len(wrong),
                        "a 15x memory over-subtraction should invalidate points")

    def test_wrong_depth_is_refused_rather_than_silently_fitted(self):
        """The mismatch must not produce a plausible-looking fit.

        With most points invalidated, ``fit_curve`` raises. That is the desired
        failure mode: a wrong granularity should be loud, not a curve that fits
        the bad points and extrapolates confidently.
        """
        wrong = extract_eff(FULL, mtgr_layout, block_measurements(), H100_SXM,
                            bw_util=TRUE.bw_util)
        with self.assertRaises(ValueError):
            fit_curve(wrong)

    def test_declaring_the_right_depth_is_what_silences_it(self):
        pts = extract_eff(FULL, mtgr_layout, block_measurements(), H100_SXM,
                          layers_measured=1, bw_util=TRUE.bw_util)
        p, rmse = fit_curve(pts)          # must not raise
        self.assertLess(rmse, 0.03)

    def test_correct_depth_restores_the_full_model_time(self):
        pts = extract_eff(FULL, mtgr_layout, block_measurements(), H100_SXM,
                          layers_measured=1, bw_util=TRUE.bw_util)
        p, rmse = fit_curve(pts)
        lay = mtgr_layout(4096)
        got = predict(FULL, lay, 4096, 32, 1, H100_SXM, p)
        want = predict(FULL, lay, 4096, 32, 1, H100_SXM, TRUE)
        self.assertAlmostEqual(got / want, 1.0, delta=0.03)
        self.assertLess(rmse, 0.03)

    def test_full_stack_measurement_needs_no_scaling(self):
        """``sweep.py --layers 15``: declare 15 (or nothing) and it is exact."""
        meas = [(T, predict(FULL, mtgr_layout(T), T, 32, 1, H100_SXM, TRUE))
                for T in LENGTHS]
        pts = extract_eff(FULL, mtgr_layout, meas, H100_SXM,
                          layers_measured=15, bw_util=TRUE.bw_util)
        p, rmse = fit_curve(pts)
        self.assertAlmostEqual(p.eff_inf, 0.42, delta=0.01)
        self.assertLess(rmse, 0.01)

    def test_note_warns_when_depth_differs(self):
        pts = extract_eff(FULL, mtgr_layout, block_measurements(), H100_SXM,
                          layers_measured=1, bw_util=TRUE.bw_util)
        self.assertIn("per-1-layer", pts[0].note)


class TestAttribution(unittest.TestCase):
    def setUp(self):
        self.lay = mtgr_layout(4096)

    def test_increments_sum_to_the_prediction(self):
        for cp in (2, 4, 8):
            a = attribute(FULL, self.lay, 4096, 32, cp, H100_SXM, TRUE, 6.0)
            total = (a["fixed_ms"] + a["ideal_shard_ms"] + a["eff_loss_ms"]
                     + a["imbalance_loss_ms"] + a["comm_ms"])
            self.assertAlmostEqual(total, a["step_ms"], places=6)
            self.assertAlmostEqual(
                a["step_ms"],
                predict(FULL, self.lay, 4096, 32, cp, H100_SXM, TRUE, 6.0),
                places=6)

    def test_each_increment_is_non_negative(self):
        for cp in (2, 4, 8):
            a = attribute(FULL, self.lay, 4096, 32, cp, H100_SXM, TRUE, 6.0)
            for k in ("eff_loss_ms", "imbalance_loss_ms", "comm_ms"):
                self.assertGreaterEqual(a[k], 0.0, f"{k} at cp={cp}")

    def test_shares_sum_to_one(self):
        a = attribute(FULL, self.lay, 4096, 32, 8, H100_SXM, TRUE, 6.0)
        total = (a["fixed_share"] + a["ideal_share"] + a["eff_share"]
                 + a["imbalance_share"] + a["comm_share"])
        self.assertAlmostEqual(total, 1.0, places=9)

    def test_cp1_has_no_losses(self):
        a = attribute(FULL, self.lay, 4096, 32, 1, H100_SXM, TRUE, 6.0)
        self.assertAlmostEqual(a["eff_loss_ms"], 0.0, places=6)
        self.assertAlmostEqual(a["imbalance_loss_ms"], 0.0, places=6)
        self.assertAlmostEqual(a["comm_ms"], 0.0, places=6)
        self.assertAlmostEqual(a["speedup"], 1.0, places=9)

    def test_fixed_cost_limits_the_speedup(self):
        """Adding a non-shardable cost must lower the speedup."""
        lay = mtgr_layout(4096)
        no_fixed = attribute(FULL, lay, 4096, 32, 8, H100_SXM, TRUE, 0.0)["speedup"]
        with_fixed = attribute(FULL, lay, 4096, 32, 8, H100_SXM, TRUE, 50.0)["speedup"]
        self.assertGreater(no_fixed, with_fixed)

    def test_bigger_comm_lowers_the_speedup(self):
        """Isolates the second reported cause: over-stated communication."""
        lay = mtgr_layout(4096)
        cheap = attribute(FULL, lay, 4096, 32, 8, H100_SXM,
                          TRUE.copy(bw_comm=800e9))["speedup"]
        dear = attribute(FULL, lay, 4096, 32, 8, H100_SXM,
                         TRUE.copy(bw_comm=50e9))["speedup"]
        self.assertGreater(cheap, dear)

    def test_more_overlap_raises_the_speedup(self):
        lay = mtgr_layout(4096)
        lo = attribute(FULL, lay, 4096, 32, 8, H100_SXM,
                       TRUE.copy(overlap=0.1))["speedup"]
        hi = attribute(FULL, lay, 4096, 32, 8, H100_SXM,
                       TRUE.copy(overlap=0.9))["speedup"]
        self.assertGreater(hi, lo)

    def test_efficiency_dominates_communication_on_a_910b(self):
        """The headline diagnostic: it is the curve, not the collective."""
        a = attribute(FULL, self.lay, 4096, 32, 8, ASCEND_910B, TRUE, 6.0)
        self.assertGreater(a["eff_share"], a["comm_share"])

    def test_report_lists_every_cp(self):
        text = attribution_report(FULL, self.lay, 4096, 32, [1, 2, 4, 8],
                                  H100_SXM, TRUE, 6.0)
        for cp in (1, 2, 4, 8):
            self.assertIn(f"{cp:>3}", text)
        self.assertEqual(len(text.strip().splitlines()), 6)


class TestSpeedupDiagnosis(unittest.TestCase):
    def test_ratio_near_one_over_n_layers_is_the_signature(self):
        """What ``e2e.py --check-csv`` looks for.

        Simulate a *partial* mismatch -- a correctly extracted curve applied at
        cp=1 -- and confirm the ratio signature is ``1/n_layers``, which is what
        the check reports.
        """
        pts = extract_eff(FULL, mtgr_layout, block_measurements(), H100_SXM,
                          layers_measured=1, bw_util=TRUE.bw_util)
        p, _ = fit_curve(pts)
        lay = mtgr_layout(4096)
        one_layer = replace(FULL, n_layers=1)
        pred_block = predict(one_layer, lay, 4096, 32, 1, H100_SXM, p)
        meas_full = predict(FULL, lay, 4096, 32, 1, H100_SXM, TRUE)
        self.assertAlmostEqual(pred_block / meas_full, 1.0 / FULL.n_layers,
                               delta=0.02)


if __name__ == "__main__":
    unittest.main()
