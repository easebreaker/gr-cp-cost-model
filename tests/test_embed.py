"""The embedding term must be driven by distinct rows, and sharding it is a trade.

Two failure modes are pinned here:

1. **Modelling on token count instead of distinct rows.**  Production traffic
   repeats ids heavily; a model that charges ``B*T`` row reads over-predicts by
   the dedup factor, which is routinely 2-10x.

2. **Claiming a CP win from sharded lookup without paying for the re-gather.**
   Sharding the lookup divides the table traffic by ``cp``, then adds a dense
   all-gather of ``B*T*d`` elements.  For ``row_bytes = d*dtype_bytes`` the two
   are the same order, so the win is not automatic -- it depends on dedup and on
   whether the collective is latency- or bandwidth-bound.
"""

import unittest

from grcost.comm import CommModel
from grcost.embed import EmbedModel
from grcost.hardware import ASCEND_910C


class TestLookup(unittest.TestCase):
    def test_row_reads_scale_with_dedup(self):
        m = EmbedModel(dedup=0.25)
        self.assertAlmostEqual(m.row_reads(1e6), 2.5e5)

    def test_sharding_divides_row_reads(self):
        m = EmbedModel(dedup=0.3, shard_by_cp=True)
        self.assertAlmostEqual(m.row_reads(1e6, cp=4), 7.5e4)

    def test_no_shard_keeps_full_reads(self):
        m = EmbedModel(dedup=0.3, shard_by_cp=False)
        self.assertAlmostEqual(m.row_reads(1e6, cp=4), 3e5)

    def test_more_dedup_is_faster(self):
        a = EmbedModel(dedup=1.0)
        b = EmbedModel(dedup=0.2)
        ta = a.time(1e6, 768, 2, ASCEND_910C, 0.55)
        tb = b.time(1e6, 768, 2, ASCEND_910C, 0.55)
        self.assertGreater(ta, tb)

    def test_cache_hits_reduce_cost(self):
        a = EmbedModel(cache_hit=0.0)
        b = EmbedModel(cache_hit=0.8)
        self.assertGreater(a.time(1e6, 768, 2, ASCEND_910C, 0.55),
                           b.time(1e6, 768, 2, ASCEND_910C, 0.55))

    def test_lookup_is_bandwidth_bound_not_flop_bound(self):
        """Byte-driven: doubling rows doubles the time."""
        m = EmbedModel(dedup=1.0)
        t1 = m.time(1e6, 768, 2, ASCEND_910C, 0.55)
        t2 = m.time(2e6, 768, 2, ASCEND_910C, 0.55)
        self.assertAlmostEqual(t2 / t1, 2.0, places=6)


class TestShardTrade(unittest.TestCase):
    def setUp(self):
        self.cm = CommModel(alpha=15e-6, bw=149e9)

    def test_regather_bytes_grow_with_cp(self):
        m = EmbedModel(shard_by_cp=True)
        a = m.regather_bytes(1e6, 768, 2, cp=2)
        b = m.regather_bytes(1e6, 768, 2, cp=8)
        self.assertGreater(b, a)

    def test_no_regather_without_sharding(self):
        m = EmbedModel(shard_by_cp=False)
        self.assertEqual(m.regather_bytes(1e6, 768, 2, cp=8), 0.0)

    def test_net_effect_names_both_sides(self):
        m = EmbedModel(dedup=0.5, shard_by_cp=True)
        text = m.net_effect(1e6, 768, 2, ASCEND_910C, 0.55, self.cm, cp=4)
        self.assertIn("saved", text)
        self.assertIn("re-gather", text)

    def test_naive_sharding_does_not_pay(self):
        """The blunt result: a dense re-gather dwarfs the saved table reads.

        ``row_bytes == d*dtype_bytes``, so the gather moves ``1/dedup`` times what
        the table would have read -- and it is charged at collective bandwidth.
        """
        for dedup in (1.0, 0.3, 0.05):
            m = EmbedModel(dedup=dedup, shard_by_cp=True)
            text = m.net_effect(1e6, 768, 2, ASCEND_910C, 0.55, self.cm, cp=8)
            self.assertIn("DOES NOT pay", text, f"dedup={dedup}")

    def test_suggests_gathering_ids_instead(self):
        m = EmbedModel(dedup=0.3, shard_by_cp=True)
        text = m.net_effect(1e6, 768, 2, ASCEND_910C, 0.55, self.cm, cp=8)
        self.assertIn("gathering ids", text)

    def test_credit_for_a_fused_gather_makes_it_pay(self):
        m = EmbedModel(dedup=1.0, shard_by_cp=True, credit_regather=True)
        text = m.net_effect(1e6, 768, 2, ASCEND_910C, 0.55, self.cm, cp=8)
        self.assertIn("pays", text)
        self.assertNotIn("DOES NOT", text)

    def test_cp1_has_nothing_to_shard(self):
        m = EmbedModel()
        self.assertIn("nothing to shard",
                      m.net_effect(1e6, 768, 2, ASCEND_910C, 0.55, self.cm, 1))


if __name__ == "__main__":
    unittest.main()
