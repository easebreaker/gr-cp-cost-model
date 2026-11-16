"""Embedding lookup: the term that is neither compute nor collective.

Why it needs its own model
--------------------------
On a GR the sparse side is not a rounding error.  A lookup reads one row of
``d`` elements per id, so its arithmetic intensity is ``1`` FLOP per byte -- it is
permanently bandwidth-bound, and it competes with the attention kernel for the
same HBM.  Reported experience on DLRM-class models: the embedding stage can be
the primary bottleneck (3.2x slowdown versus a fully cached case), and the access
pattern matters more than the byte count (random-lookup bandwidth is a fraction
of streaming bandwidth).

Two things therefore have to be modelled separately from ``t_fixed``:

* **deduplication.**  ``B*T`` tokens do not imply ``B*T`` row reads.  Popular ids
  repeat, and dedup turns the lookup into ``unique`` reads plus a gather.  A
  model driven by token count over-predicts by the dedup factor, which on
  production traffic is routinely 2-10x.
* **sharding with CP.**  If CP splits the sequence, each rank only looks up its
  own shard's tokens, so the per-rank lookup cost falls as ``1/cp``.  That is the
  user's requirement 7 -- but it is not free: the *activations* must then be
  re-gathered before attention, and that all-gather carries ``B*T*d`` elements,
  which is a dense collective, not a few ids.  Whether sharding the lookup pays
  depends on which of the two is larger:

      saved   : (1 - 1/cp) * B*T*row_bytes            (HBM reads, bandwidth-bound)
      added   : (cp-1)/cp * B*T*d*dtype_bytes         (all-gather, bandwidth-bound)

  For ``row_bytes = d*dtype_bytes`` the two are the *same order* -- so sharded
  lookup only wins if dedup makes the saved side much smaller than the added
  side, or if the re-gather is overlapped.  Ignoring this is how a model claims a
  CP speedup that the implementation cannot deliver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .comm import CommModel


@dataclass
class EmbedModel:
    """Lookup cost, with dedup and an optional CP shard.

    ``dedup``      unique ids / total tokens.  Measure it on real traffic; a
                   value of 1.0 means every token reads its own row.
    ``cache_hit``  fraction of reads served from cache.  Cached reads cost
                   roughly an order of magnitude less bandwidth; default 0 means
                   "assume nothing is cached", which is the safe direction.
    ``shard_by_cp`` whether the lookup runs on the CP shard of the sequence
                   (requirement 7) rather than on the full batch.
    ``credit_regather`` the re-gather can be credited against a transfer CP was
                   already going to make, so only the difference is charged.
                   Leave False unless that fusion is actually implemented.
    """

    dedup: float = 1.0
    cache_hit: float = 0.0
    shard_by_cp: bool = True
    cached_cost: float = 0.1        # relative cost of a cache hit
    credit_regather: bool = False

    def row_reads(self, tokens: float, cp: int = 1) -> float:
        """Distinct rows read per device."""
        shard = max(cp, 1) if self.shard_by_cp else 1
        return tokens * self.dedup / shard

    def time(self, tokens: float, dim: int, dtype_bytes: int, hw,
             bw_util: float, cp: int = 1) -> float:
        """Seconds of lookup traffic on the critical device."""
        rows = self.row_reads(tokens, cp)
        nbytes = rows * dim * dtype_bytes
        eff_bw = hw.hbm_bw * bw_util
        penalty = (1.0 - self.cache_hit) + self.cache_hit * self.cached_cost
        # Random row access does not reach streaming bandwidth; the shortfall is
        # not a free parameter -- it is the same "many small transactions"
        # problem as a small collective, so it is reported separately.
        return nbytes * penalty / eff_bw

    def regather_bytes(self, tokens: float, dim: int, dtype_bytes: int,
                       cp: int = 1) -> float:
        """Dense all-gather needed if the lookup was sharded (requirement 7)."""
        if not self.shard_by_cp or cp <= 1:
            return 0.0
        return tokens * dim * dtype_bytes * (cp - 1) / cp

    def net_effect(self, tokens: float, dim: int, dtype_bytes: int, hw,
                   bw_util: float, comm: CommModel, cp: int) -> str:
        """Does sharding the lookup actually pay?  Answered in seconds.

        The arithmetic is blunt, and it usually comes out negative:

            saved     (1 - 1/cp) * dedup * tokens * row_bytes     table reads
            added     (1 - 1/cp) * tokens * d * dtype_bytes        dense gather

        With ``row_bytes = d*dtype_bytes`` the two differ by exactly ``dedup`` --
        so sharding the lookup only wins when ``dedup < 1`` and the gather is
        cheaper than the reads it replaces.  On repeated-id traffic ``dedup`` is
        small, which makes the saving small, while the gather stays proportional
        to the token count.

        The usual production resolution is not to shard the lookup at all:
        gather the *ids* (4 bytes) and look up on every rank, because every rank
        needs the full sequence's K/V anyway.  ``credit_regather=True`` models the
        case where the embedding gather replaces a transfer CP was already paying.
        """
        if cp <= 1:
            return "  cp=1: nothing to shard"
        full = self.time(tokens, dim, dtype_bytes, hw, bw_util, cp=1)
        shard = self.time(tokens, dim, dtype_bytes, hw, bw_util, cp=cp)
        saved = full - shard
        added_b = self.regather_bytes(tokens, dim, dtype_bytes, cp)
        if self.credit_regather:
            added_b = 0.0
        added = comm.time(cp, added_b, rounds=1)
        verdict = "pays" if saved > added else "DOES NOT pay"
        alt = ""
        if added_b and not self.credit_regather:
            alt = ("\n            (gathering ids instead of embeddings moves "
                   f"{tokens * 4 / 1e6:.2f} MB rather than {added_b / 1e6:.2f} MB)")
        return (f"  cp={cp}: lookup saved {saved * 1e3:7.3f} ms, "
                f"re-gather {added_b / 1e6:7.2f} MB costs {added * 1e3:7.3f} ms "
                f"-> {verdict}{alt}")
