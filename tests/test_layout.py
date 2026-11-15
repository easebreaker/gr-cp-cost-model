"""The mask topology closed form must match brute force, exactly.

This is the load-bearing test of the whole package: every FLOPs number derives
from ``Layout.cum_pairs``.  If it is wrong, everything downstream is wrong.
"""

import unittest

from grcost.spec import Layout, assign, work_per_rank


def _base_keys(lay: Layout) -> int:
    p = lay.prefix_len
    base = (p + 1) if lay.cand_sees_prefix else 1
    if lay.cand_block:
        base += lay.n_cand - 1
    return base


def brute_pairs(layout: Layout, a: int, b: int) -> int:
    """Reference implementation: count visible keys by iterating."""
    lay = layout
    s, p = lay.n_static, lay.prefix_len
    total = 0
    for q in range(a, b):
        if q < s:
            total += s if lay.static_full else q + 1
        elif q < p:
            total += q + 1
        else:
            total += _base_keys(lay)
    return total


def brute_naive(layout: Layout, a: int, b: int) -> int:
    """Even more naive: build the full boolean mask, then sum the slice."""
    lay = layout
    n, s, p = lay.total, lay.n_static, lay.prefix_len
    rows = []
    for q in range(n):
        row = [False] * n
        if q < s:
            upto = s if lay.static_full else q + 1
            for k in range(upto):
                row[k] = True
        elif q < p:
            for k in range(q + 1):
                row[k] = True
        else:
            if lay.cand_sees_prefix:
                for k in range(p + 1):
                    row[k] = True
            else:
                row[q] = True
            if lay.cand_block:
                for k in range(p, n):
                    row[k] = True
        rows.append(row)
    return sum(sum(r) for r in rows[a:b])


ALL_LAYOUTS = [
    Layout.causal(1),
    Layout.causal(7),
    Layout(n_static=0, n_causal=13),
    Layout(n_static=6, n_causal=9),
    Layout(n_static=6, n_causal=9, static_full=False),
    Layout(n_static=0, n_causal=13, n_cand=4),
    Layout(n_static=9, n_causal=6, n_cand=5),
    Layout(n_static=9, n_causal=6, n_cand=5, cand_block=True),
    Layout(n_static=9, n_causal=6, n_cand=5, cand_sees_prefix=False),
    Layout(n_static=4, n_causal=0, n_cand=3),
    Layout(n_static=4, n_causal=0, n_cand=3, static_full=False),
]


class TestLayoutClosedForm(unittest.TestCase):
    def test_matches_bruteforce_every_range(self):
        for lay in ALL_LAYOUTS:
            for a in range(lay.total + 1):
                for b in range(a, lay.total + 1):
                    self.assertEqual(
                        lay.cum_pairs(a, b), brute_pairs(lay, a, b),
                        f"{lay.describe()} [{a},{b})",
                    )

    def test_matches_full_mask(self):
        for lay in ALL_LAYOUTS:
            self.assertEqual(lay.pairs_total(), brute_naive(lay, 0, lay.total),
                             lay.describe())

    def test_no_static_is_triangular_number(self):
        for t in (1, 2, 10, 1024, 4096):
            self.assertEqual(Layout.causal(t).pairs_total(), t * (t + 1) // 2)

    def test_static_block_is_a_square(self):
        """A rectangular static block of length S costs exactly S^2 pairs."""
        for s in (1, 2, 16):
            lay = Layout(n_static=s)
            self.assertEqual(lay.pairs_total(), s * s)

    def test_static_block_is_two_parts_of_quadratic(self):
        """The derivation the paper needs:

            pairs = S^2  +  [P(P+1) - S(S+1)]/2  +  K*(P+1)

        where P = S + C.  The middle term is *not* C(C+1)/2, because a causal
        query at global position q also sees the whole static block, so it
        carries q+1 keys rather than (q-S+1).
        """
        s, c = 300, 700
        p = s + c
        lay = Layout(n_static=s, n_causal=c)
        self.assertEqual(lay.pairs_total(), s * s + (p * (p + 1) - s * (s + 1)) // 2)

    def test_causal_part_is_shifted_triangle(self):
        s, c = 6, 9
        p = s + c
        lay = Layout(n_static=s, n_causal=c)
        causal_only = sum(range(s + 1, p + 1))       # 7 + 8 + ... + p
        self.assertEqual(lay.pairs_total(), s * s + causal_only)

    def test_static_triangular_degenerates_to_causal(self):
        """With static_full=False the block folds into one triangle."""
        for s in (0, 1, 5, 17):
            for c in (0, 1, 4, 23):
                lay = Layout(n_static=s, n_causal=c, static_full=False)
                self.assertEqual(lay.pairs_total(), Layout.causal(s + c).pairs_total())

    def test_candidate_region_is_linear(self):
        """Adding candidates adds exactly (prefix+1) pairs each."""
        base = Layout(n_static=100, n_causal=50)
        more = Layout(n_static=100, n_causal=50, n_cand=10)
        self.assertEqual(more.pairs_total() - base.pairs_total(), 10 * (150 + 1))


class TestAssignment(unittest.TestCase):
    def test_each_scheme_partitions_the_sequence(self):
        for total, cp in [(8, 1), (8, 2), (8, 4), (10, 3), (100, 7)]:
            for scheme in ("block", "balanced", "striped"):
                seen = [q for rs in assign(total, cp, scheme) for a, b in rs
                        for q in range(a, b)]
                self.assertEqual(sorted(seen), list(range(total)), scheme)

    def test_pairs_are_conserved(self):
        """CP must not change total work -- it is a parallelization, not an algorithm."""
        lay = Layout(n_static=200, n_causal=800, n_cand=20)
        for cp in (1, 2, 4, 8):
            for scheme in ("block", "balanced", "striped"):
                per = work_per_rank(lay, cp, scheme)
                self.assertEqual(sum(per), lay.pairs_total(), f"cp={cp} {scheme}")

    def test_block_imbalance_closed_form(self):
        """For a purely causal mask, block assignment gives max/mean = 2 - 1/CP."""
        lay = Layout.causal(8192)
        for cp in (2, 4, 8, 16):
            w = work_per_rank(lay, cp, "block")
            self.assertAlmostEqual(max(w) / (sum(w) / cp), 2.0 - 1.0 / cp, delta=0.02)

    def test_balanced_removes_imbalance(self):
        lay = Layout.causal(8192)
        for cp in (2, 4, 8):
            w = work_per_rank(lay, cp, "balanced")
            self.assertLess(max(w) / (sum(w) / cp), 1.02, f"cp={cp}")

    def test_striped_removes_imbalance(self):
        for cp in (2, 4, 8):
            w = work_per_rank(Layout.causal(8192), cp, "striped")
            self.assertLess(max(w) / (sum(w) / cp), 1.02, f"cp={cp}")

    def test_static_prefix_damps_imbalance(self):
        """The headline structural prediction: a GR with a big static block is
        far more CP-balanced than a causal-only decoder of the same length."""
        t = 4096
        ratios = {}
        for share in (0.0, 0.5, 0.85, 0.95):
            n_static = int(t * share)
            lay = Layout(n_static=n_static, n_causal=t - n_static - 20, n_cand=20)
            w = work_per_rank(lay, 8, "block")
            ratios[share] = max(w) / (sum(w) / 8)
        self.assertGreater(ratios[0.0], 1.8)
        self.assertGreater(ratios[0.0], ratios[0.85])
        self.assertGreater(ratios[0.5], ratios[0.95])
        self.assertLess(ratios[0.95], 1.10)

    def test_imbalance_upper_bounds_benefit_of_balancer(self):
        """Speedup from rebalancing cannot exceed the block imbalance."""
        lay = Layout.causal(8192)
        for cp in (2, 4, 8):
            blk = max(work_per_rank(lay, cp, "block"))
            bal = max(work_per_rank(lay, cp, "balanced"))
            self.assertLess(blk / bal, 2.0 - 1.0 / cp + 0.02)


if __name__ == "__main__":
    unittest.main()
