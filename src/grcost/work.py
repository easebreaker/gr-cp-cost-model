"""Per-rank work accounting: FLOPs, pairs, elementwise bytes, collective bytes.

Nothing in this module is fitted.  Everything is derived from
``(model, layout, T, B, cp, scheme)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .spec import Layout, Phase, assign

FP16 = 2  # bytes per element


@dataclass
class RankWork:
    rank: int
    ntok: float          # logical tokens of dense work on this rank
    pairs: float         # visible attention (query, key) pairs
    flops_dense: float
    flops_attn: float
    bytes_ew: float      # elementwise traffic over the attention matrix
    bytes_kv: float      # what this rank must ship for CP
    notes: str = ""

    @property
    def flops(self) -> float:
        return self.flops_dense + self.flops_attn


def rank_work(
    model,
    layout: Layout,
    T: int,
    B: int,
    cp: int = 1,
    scheme: str = "block",
    ew_ops_per_pair: float = 3.0,
    dtype_bytes: int = FP16,
    kv_bytes_per_token: int = 0,
) -> List[RankWork]:
    """Work on every CP rank.  ``model`` is an HstuModel or LongerModel."""
    d = model.d
    phases: List[Phase] = model.phases(T, layout)

    chunks = assign(T, cp, "block")
    ntok_per_rank = [B * (b - a) for (a, b) in (c[0] for c in chunks)]

    # Dense projections are not mask-dependent: they split evenly even when the
    # mask does not.  This is a real effect -- it damps CP imbalance -- so the
    # two terms must be kept separate rather than lumped into one "work".
    dense_total = model.dense_flops_per_rank(T, B, cp)
    dense_flops = [dense_total] * cp

    pairs = [0.0] * cp
    for ph in phases:
        per = ph.pairs_per_rank(cp, scheme)
        for r in range(cp):
            pairs[r] += B * ph.n_layers * per[r]

    if kv_bytes_per_token <= 0:
        # ring attention: each rank ships its K/V shard to the other cp-1 ranks
        kv_bytes_per_token = 2 * dtype_bytes * d  # K and V, one token each
    # K/V are exchanged per *attention layer*, not once per step -- forgetting
    # this under-counts CP communication by a factor of n_layers and makes it
    # look like communication never matters.
    comm_layers = sum(ph.n_layers for ph in phases if ph.sharded)
    kv_total = B * T * kv_bytes_per_token * comm_layers
    bytes_kv = [kv_total * (cp - 1) / cp for _ in range(cp)]

    per_pair = model.attn_flops_per_pair() if hasattr(model, "attn_flops_per_pair") else 4.0 * d
    out = []
    for r in range(cp):
        flops_attn = per_pair * pairs[r]  # QK^T + AV summed over heads
        out.append(
            RankWork(
                rank=r,
                ntok=ntok_per_rank[r],
                pairs=pairs[r],
                flops_dense=dense_flops[r],
                flops_attn=flops_attn,
                bytes_ew=ew_ops_per_pair * dtype_bytes * pairs[r],
                bytes_kv=bytes_kv[r],
            )
        )
    return out


def imbalance(works: List[RankWork]) -> float:
    """max/mean of the attention term -- the part CP can unbalance."""
    vals = [w.pairs for w in works]
    mean = sum(vals) / len(vals)
    return (max(vals) / mean) if mean else 1.0


def total_flops(model, layout: Layout, T: int, B: int) -> float:
    """Unsharded FLOPs for one step (CP-invariant, so evaluate at cp=1).

    Summing over ranks equals the sharded total, because the number of visible
    pairs is conserved -- that conservation is exactly what makes CP a pure
    parallelization and not a change of algorithm.
    """
    return sum(w.flops for w in rank_work(model, layout, T, B, cp=1))


def flops_per_example(model, layout: Layout, T: int) -> float:
    return total_flops(model, layout, T, B=1)
