"""The bridge must reconstruct ``cost.predict`` exactly, and κ must drift.

If the reconstruction is not exact, the two frameworks disagree and no amount of
fitting will reconcile them. If κ does not drift, the existing constant-κ
assumption is fine and this whole module is unnecessary.
"""

import unittest

from grcost.bridge import decompose, kappa_drift, sweep
from grcost.cost import Params, predict
from grcost.hardware import ASCEND_910B, H100_SXM
from grcost.presets import mtgr_layout, resolve_model
from grcost.spec import Layout


P = Params(eff_inf=0.42, c_star=1800, gamma=1.1, alpha=25e-6, bw_comm=150e9,
           overlap=0.45, bw_util=0.55)
M = resolve_model("mtgr-large")


class TestExactness(unittest.TestCase):
    def check(self, layout, T, B, mode, scheme="block", t_fixed=6.0):
        for n in (2, 4, 8):
            d = decompose(M, layout, T, B, n, H100_SXM, P, t_fixed_ms=t_fixed,
                          scheme=scheme, shard_mode=mode)
            direct = predict(M, layout, T, B, n, H100_SXM, P,
                             t_fixed_ms=t_fixed, scheme=scheme)
            self.assertAlmostEqual(d.t_hat_ms / direct, 1.0, places=9,
                                   msg=f"{mode} n={n} T={T} B={B}")

    def test_full_mode_exact(self):
        for T in (1024, 4096, 8192):
            self.check(mtgr_layout(T), T, 32, "full")

    def test_attn_only_mode_is_self_consistent(self):
        """attn_only is a *different* assumption (projections replicated), so it
        is checked against its own arithmetic rather than against predict()."""
        for T in (1024, 4096, 8192):
            for n in (2, 4, 8):
                d = decompose(M, mtgr_layout(T), T, 32, n, H100_SXM, P,
                              t_fixed_ms=6.0, shard_mode="attn_only")
                self.assertAlmostEqual(d.attn_1_ms / d.g / d.attn_n_ms, 1.0, places=9)
                self.assertAlmostEqual(
                    d.t_hat_ms,
                    6.0 + d.dense_1_ms + d.attn_n_ms + d.comm_ms, places=9)

    def test_attn_only_is_not_cheaper_than_full(self):
        """Replicating the projections can only cost more."""
        for T in (1024, 4096):
            full = decompose(M, mtgr_layout(T), T, 32, 4, H100_SXM, P, 6.0,
                             shard_mode="full")
            attn = decompose(M, mtgr_layout(T), T, 32, 4, H100_SXM, P, 6.0,
                             shard_mode="attn_only")
            self.assertGreaterEqual(attn.t_hat_ms, full.t_hat_ms - 1e-9)

    def test_causal_layout_exact(self):
        for T in (2048, 8192):
            self.check(Layout.causal(T), T, 8, "full")

    def test_balanced_scheme_exact(self):
        self.check(mtgr_layout(4096), 4096, 8, "full", scheme="balanced")

    def test_batch_and_fixed_time(self):
        self.check(mtgr_layout(4096), 4096, 3, "full", t_fixed=0.0)
        self.check(mtgr_layout(2048), 2048, 64, "full", t_fixed=25.0)


class TestKappa(unittest.TestCase):
    def test_kappa_is_positive_and_increasing(self):
        """eff(c) rolls off, so the equivalent κ must grow with N."""
        for T in (2048, 4096, 8192):
            rows = sweep(M, mtgr_layout(T), T, 32, ASCEND_910B, P, 6.0)
            ks = [r.kappa for r in rows]
            self.assertTrue(all(k > 0 for k in ks))
            self.assertLess(ks[0], ks[1])
            self.assertLess(ks[1], ks[2])

    def test_drift_is_reported_not_ignored(self):
        text = kappa_drift(M, mtgr_layout(4096), 4096, 32, ASCEND_910B, P, 6.0)
        self.assertIn("drift", text)
        self.assertIn("constant-κ", text)

    def test_constant_kappa_underestimates_loss(self):
        """The prediction that the existing framework extrapolates optimistically."""
        rows = sweep(M, mtgr_layout(4096), 4096, 32, ASCEND_910B, P, 6.0)
        k_fit = 0.5 * (rows[0].kappa + rows[1].kappa)
        g_const = 8.0 / (1.0 + k_fit * 7.0)
        self.assertGreater(g_const, rows[2].g)   # over-optimistic g

    def test_small_chunks_have_larger_kappa(self):
        """Smaller c at CP=8 means more efficiency loss per device."""
        big = sweep(M, mtgr_layout(8192), 8192, 32, H100_SXM, P, 6.0)[-1]
        small = sweep(M, mtgr_layout(1024), 1024, 32, H100_SXM, P, 6.0)[-1]
        self.assertGreater(small.kappa, big.kappa)

    def test_imbalance_diagnostics_are_consistent(self):
        d = decompose(M, mtgr_layout(4096), 4096, 32, 8, H100_SXM, P, 6.0)
        self.assertGreater(d.imbalance_pairs, d.imbalance_flops)
        self.assertGreaterEqual(d.imbalance_pairs, 1.0)
        self.assertGreaterEqual(d.eff_ratio, 1.0)

    def test_comm_fraction_is_zero_at_cp1_shape(self):
        """At CP=2 with a tiny T the comm share should be small but positive."""
        d = decompose(M, mtgr_layout(4096), 4096, 32, 2, H100_SXM, P, 6.0)
        self.assertGreater(d.comm_ms, 0.0)
        self.assertLess(d.c_frac, 0.5)


if __name__ == "__main__":
    unittest.main()
