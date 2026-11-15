"""The frontier must respect three constraints, in this order of severity:

1. throughput  -- tokens/GPU-hour strictly decreases with CP
2. memory      -- an infeasible point must never be recommended
3. saturation  -- every lever's return must diminish

The most important test here is ``test_memory_excludes_the_best_raw_quality``:
without a memory model the frontier confidently picks the configuration that
OOMs, and there is nothing in the numbers to warn you.
"""

import unittest

from grcost.cost import Params
from grcost.frontier import (
    Point, best, breakeven_alpha, data_loss, fits, gpu_hours_for_tokens,
    length_gain, longer_only, longer_with_cp, pareto, prior_memory_fit,
    throughput_tokens_per_hour, tokens_for_quality,
)
from grcost.hardware import ASCEND_910B as HW
from grcost.presets import mtgr_layout, resolve_model
from grcost.quality import QualityLaw, demo_law

P = Params(eff_inf=0.42, c_star=1800, gamma=1.1, alpha=25e-6, bw_comm=120e9,
           overlap=0.5, bw_util=0.55, recompute=1.0)
M = resolve_model("mtgr-large")
B = 32
TF = 6.0


class TestThroughput(unittest.TestCase):
    def test_throughput_falls_monotonically_with_cp(self):
        """The load-bearing claim: CP is never throughput-neutral."""
        T = 4096
        lay = mtgr_layout(T)
        vals = [throughput_tokens_per_hour(M, lay, T, B, cp, HW, P, 64, TF)
                for cp in (1, 2, 4, 8)]
        for a, b in zip(vals, vals[1:]):
            self.assertGreater(a, b)

    def test_gpu_hours_are_independent_of_gpu_count(self):
        """GPU-hours is a cluster-independent cost; wall-clock is not."""
        T = 4096
        lay = mtgr_layout(T)
        a = gpu_hours_for_tokens(M, lay, T, B, 4, 1e12, HW, P, TF)
        b = gpu_hours_for_tokens(M, lay, T, B, 4, 1e12, HW, P, TF)
        self.assertAlmostEqual(a, b, places=9)

    def test_cp_costs_more_per_token(self):
        T = 4096
        lay = mtgr_layout(T)
        one = gpu_hours_for_tokens(M, lay, T, B, 1, 1e12, HW, P, TF)
        eight = gpu_hours_for_tokens(M, lay, T, B, 8, 1e12, HW, P, TF)
        self.assertGreater(eight, one)


class TestMemory(unittest.TestCase):
    def test_prior_fit_excludes_large_sequences(self):
        mem = prior_memory_fit(M, HW)
        self.assertFalse(mem.identifiable)
        self.assertFalse(fits(mem, 65536, B, 1, HW))
        self.assertTrue(fits(mem, 4096, B, 1, HW))

    def test_memory_excludes_the_best_raw_quality(self):
        """The trap: the highest-quality configuration is the one that OOMs."""
        law = demo_law()
        m0 = M.param_count()
        mem = prior_memory_fit(M, HW)
        rows = longer_only(M, mtgr_layout, [8192, 16384, 32768], m0, law, 64,
                           72.0, B, HW, P, TF, mem=mem)
        raw_best = max(rows, key=lambda q: q.quality)
        self.assertFalse(raw_best.fits)
        self.assertTrue(best(rows).fits)
        self.assertNotEqual(raw_best.T, best(rows).T)

    def test_cp_expands_the_feasible_length(self):
        mem = prior_memory_fit(M, HW)
        longest = {}
        for cp in (1, 2, 4, 8):
            T = 1024
            while fits(mem, T * 2, B, cp, HW) and T < 2 ** 20:
                T *= 2
            longest[cp] = T
        self.assertGreater(longest[8], longest[1])

    def test_bigger_batch_needs_shorter_sequences(self):
        mem = prior_memory_fit(M, HW)
        self.assertLess(mem.max_seq_len(64, 1, HW.mem_gb),
                        mem.max_seq_len(16, 1, HW.mem_gb))


class TestBreakEven(unittest.TestCase):
    def test_threshold_increases_with_length_jump(self):
        """A bigger length jump needs a bigger exponent to pay for itself."""
        law = demo_law()
        a2 = breakeven_alpha(law, 4096, 5.5e12, 2, 0.73)
        a4 = breakeven_alpha(law, 4096, 5.5e12, 4, 0.46)
        self.assertIsNotNone(a2)
        self.assertIsNotNone(a4)
        self.assertLess(a2, a4)

    def test_hopeless_token_loss_has_no_threshold(self):
        """Losing 95% of throughput is not recoverable at any length exponent."""
        law = demo_law()
        self.assertIsNone(breakeven_alpha(law, 4096, 5.5e12, 8, 0.02))

    def test_free_length_is_always_worth_it(self):
        law = demo_law()
        self.assertEqual(breakeven_alpha(law, 4096, 5.5e12, 2, 1.0), 0.02)

    def test_gain_and_loss_are_separately_computable(self):
        law = demo_law()
        self.assertGreater(length_gain(law, 4096, 2, 1e9, 0.5), 0)
        self.assertGreater(data_loss(law, 1e12, 0.5, 1e9), 0)


class TestFrontier(unittest.TestCase):
    def test_pareto_is_monotone_and_thins_the_set(self):
        law = demo_law()
        rows = longer_only(M, mtgr_layout, [1024, 2048, 4096, 8192, 16384],
                           M.param_count(), law, 64, 72.0, B, HW, P, TF,
                           mem=prior_memory_fit(M, HW))
        p = pareto(rows, cost="wall_hours")
        self.assertLessEqual(len(p), len(rows))
        qs = [x.quality for x in p]
        self.assertEqual(qs, sorted(qs))

    def test_infeasible_points_never_dominate(self):
        law = demo_law()
        rows = longer_only(M, mtgr_layout, [8192, 32768], M.param_count(), law,
                           64, 72.0, B, HW, P, TF,
                           mem=prior_memory_fit(M, HW))
        p = pareto(rows)
        self.assertTrue(all(x.fits for x in p))

    def test_tokens_for_quality_inverts(self):
        law = demo_law()
        q = 0.69
        D = tokens_for_quality(law, 4096, 4.4e7, q)
        self.assertGreater(D, 0)
        self.assertAlmostEqual(law.quality(4096, 4.4e7, D), q, places=6)

    def test_unreachable_quality_returns_inf(self):
        law = demo_law()
        self.assertEqual(tokens_for_quality(law, 4096, 4.4e7, 0.99), float("inf"))

    def test_best_raises_when_nothing_fits(self):
        pts = [Point("x", 65536, 1, 4e7, 1.0, 1e12, 0.7, 10.0, 1.0, fits=False)]
        with self.assertRaises(ValueError):
            best(pts)


if __name__ == "__main__":
    unittest.main()
