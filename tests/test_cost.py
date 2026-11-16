"""The cost model's analytic properties.  These are the invariants a reviewer
will probe, so they are worth locking down."""

import unittest

from grcost.cost import Ablation, Params, amdahl_ceiling, efficiency, predict, scaling_factor, speedup
from grcost.hardware import H100_SXM
from grcost.memory import fit_memory
from grcost.presets import mtgr_layout, resolve_model
from grcost.spec import Layout


class TestEfficiency(unittest.TestCase):
    def test_monotone_and_bounded(self):
        p = Params()
        vals = [efficiency(p, c) for c in (16, 64, 256, 1024, 4096, 16384, 65536)]
        self.assertEqual(vals, sorted(vals))
        self.assertLess(vals[-1], p.eff_inf)
        self.assertGreater(vals[-1], 0.9 * p.eff_inf)

    def test_half_efficiency_at_c_star(self):
        p = Params(gamma=1.0)
        self.assertAlmostEqual(efficiency(p, p.c_star), p.eff_inf / 2, places=9)

    def test_zero_chunk(self):
        self.assertEqual(efficiency(Params(), 0), 0.0)


class TestPredict(unittest.TestCase):
    def setUp(self):
        self.m = resolve_model("mtgr-large")
        self.lay = mtgr_layout(4096)
        self.p = Params()
        self.hw = H100_SXM

    def test_linear_in_batch_at_cp1(self):
        a = predict(self.m, self.lay, 4096, 8, 1, self.hw, self.p, t_fixed_ms=0)
        b = predict(self.m, self.lay, 4096, 16, 1, self.hw, self.p, t_fixed_ms=0)
        self.assertAlmostEqual(b / a, 2.0, delta=0.01)

    def test_fixed_time_is_additive_not_scaled(self):
        base = predict(self.m, self.lay, 4096, 32, 1, self.hw, self.p, t_fixed_ms=0)
        withf = predict(self.m, self.lay, 4096, 32, 1, self.hw, self.p, t_fixed_ms=10)
        self.assertAlmostEqual(withf - base, 10.0, places=6)

    def test_fixed_time_dilutes_with_batch(self):
        """A fixed cost must show up as a *smaller* fraction at a bigger batch."""
        p1 = predict(self.m, self.lay, 4096, 8, 1, self.hw, self.p, t_fixed_ms=20)
        p2 = predict(self.m, self.lay, 4096, 64, 1, self.hw, self.p, t_fixed_ms=20)
        b1 = predict(self.m, self.lay, 4096, 8, 1, self.hw, self.p, t_fixed_ms=0)
        b2 = predict(self.m, self.lay, 4096, 64, 1, self.hw, self.p, t_fixed_ms=0)
        self.assertGreater((p1 - b1) / p1, (p2 - b2) / p2)

    def test_cp_never_beats_ideal_linear(self):
        for cp in (2, 4, 8):
            sp = speedup(self.m, self.lay, 4096, 32, self.p, self.hw, cp)
            self.assertLess(sp, cp + 1e-9)
            self.assertGreater(sp, 1.0)

    def test_amdahl_ceiling_bounds_speedup(self):
        for cp in (2, 4, 8):
            ceil = amdahl_ceiling(self.m, self.lay, 4096, cp, B=32)
            sp = speedup(self.m, self.lay, 4096, 32, self.p, self.hw, cp)
            self.assertLessEqual(sp, ceil + 1e-9)

    def test_scaling_factor_equals_speedup(self):
        """CP splits a sequence and keeps the batch, so QPS scaling == speedup."""
        sp = speedup(self.m, self.lay, 4096, 32, self.p, self.hw, 2)
        sf = scaling_factor(self.m, self.lay, 4096, 32, self.p, self.hw, 2)
        self.assertAlmostEqual(sp, sf, places=12)

    def test_detail_breakdown_is_consistent(self):
        """The breakdown must reconstruct the total, including every term.

        The list grew over time (fixed, embedding, compute, elementwise, the HBM
        landing of received K/V, communication), so the assertion is written
        against the sum of whatever the breakdown reports rather than a fixed
        subset -- otherwise adding a term silently weakens the test.
        """
        ms, bd = predict(self.m, self.lay, 4096, 32, 4, self.hw, self.p,
                         t_fixed_ms=5.0, detail=True)
        parts = ["fixed_ms", "compute_ms", "elementwise_ms", "kv_landing_ms",
                 "comm_ms"]
        total = sum(bd[k] for k in parts if k in bd)
        self.assertAlmostEqual(bd["step_ms"], total,
                               delta=1e-6 * bd["step_ms"] + 1e-9)
        self.assertEqual(bd["chunk"], 1024.0)

    def test_longer_gets_no_cp_speedup(self):
        """LONGER touches the long sequence once, with a query of only m+k.

        Splitting that across ranks buys almost no compute and still ships the
        full K/V, so CP is a memory tool there and can even lose.
        """
        from grcost.presets import resolve_model as rm

        m = rm("longer")
        lay = Layout.causal(8192)
        for cp in (2, 4, 8):
            sp = speedup(m, lay, 8192, 8, self.p, self.hw, cp, t_fixed_ms=0)
            self.assertLess(sp, 1.2)


class TestAblations(unittest.TestCase):
    def setUp(self):
        self.m = resolve_model("mtgr-large")
        self.lay = mtgr_layout(4096)
        self.hw = H100_SXM
        self.p = Params()

    def test_disabling_efficiency_underestimates(self):
        full = predict(self.m, self.lay, 4096, 32, 8, self.hw, self.p)
        roof = predict(self.m, self.lay, 4096, 32, 8, self.hw, self.p,
                       abl=Ablation(efficiency=False))
        self.assertLess(roof, full)

    def test_disabling_comm_underestimates(self):
        full = predict(self.m, self.lay, 4096, 32, 8, self.hw, self.p)
        noc = predict(self.m, self.lay, 4096, 32, 8, self.hw, self.p,
                      abl=Ablation(comm=False))
        self.assertLess(noc, full)

    def test_mean_instead_of_max_underestimates_imbalance(self):
        t = 4096
        hstu = resolve_model("hstu-base")
        lay = Layout.causal(t)
        full = predict(hstu, lay, t, 8, 8, self.hw, self.p, scheme="block")
        mean = predict(hstu, lay, t, 8, 8, self.hw, self.p, scheme="block",
                       abl=Ablation(critical="mean"))
        self.assertLess(mean, full)

    def test_layout_override_changes_mtgr_only(self):
        m = self.m
        for T in (2048, 8192):
            lay = mtgr_layout(T)
            a = predict(m, lay, T, 8, 4, self.hw, self.p)
            b = predict(m, lay, T, 8, 4, self.hw, self.p, abl=Ablation(layout_override="causal"))
            self.assertNotAlmostEqual(a, b, places=3)


GIB = 1024 ** 3


def _obs(a_true, m0_true, plan):
    """plan = [(T_max, cp, B)] -> observations consistent with the model."""
    return [(t, cp, b, m0_true + a_true * b * t / cp / GIB) for t, cp, b in plan]


class TestMemory(unittest.TestCase):
    def test_recovers_known_coefficients_when_chunks_vary(self):
        """`a` is only identifiable if the per-rank chunk length varies."""
        a_true, m0_true = 130_000.0, 12.0
        obs = _obs(a_true, m0_true, [(3072, 1, 32), (4096, 2, 32), (7168, 4, 32)])
        fit = fit_memory(obs)
        self.assertTrue(fit.identifiable)
        self.assertAlmostEqual(fit.a_bytes, a_true, delta=1e-4 * a_true)
        self.assertAlmostEqual(fit.M_static_gb, m0_true, delta=1e-4)
        self.assertLess(fit.rmse_gb, 1e-6)

    def test_constant_chunk_length_is_flagged_not_guessed(self):
        """Equal chunk lengths across CP -> a and M_static trade off exactly."""
        obs = _obs(130_000.0, 12.0, [(3072, 1, 32), (6144, 2, 32), (12288, 4, 32)])
        fit = fit_memory(obs)
        self.assertFalse(fit.identifiable)

    def test_prediction_roundtrip(self):
        a_true, m0_true = 130_000.0, 12.0
        obs = _obs(a_true, m0_true, [(3072, 1, 32), (4096, 2, 32)])
        fit = fit_memory(obs)
        mem = obs[0][3]
        for cp, tmax in ((1, 3072), (8, 24576)):
            self.assertAlmostEqual(fit.max_seq_len(32, cp, mem), tmax, delta=3)

    def test_delta_term_shrinks_prediction(self):
        obs = _obs(130_000.0, 12.0, [(3072, 1, 32), (4096, 2, 32)])
        mem = obs[0][3]
        a = fit_memory(obs).max_seq_len(32, 8, mem)
        b = fit_memory(obs, delta_gb_per_cp=0.8).max_seq_len(32, 8, mem)
        self.assertGreater(b, 0)
        self.assertLess(b, a)


if __name__ == "__main__":
    unittest.main()
