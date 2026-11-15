"""The cost model: 8 parameters, of which 5 are usually fitted.

Structure
---------
    t_step(CP) = t_fixed
               + max_r [ r * t_comp(rank r)  +  (1 - eta) * t_comm(cp) ]

    t_comp(rank) = flops / (P_peak * eff(c))  +  bytes_ew / (BW_hbm * bw_util)
    t_comm(cp)   = alpha + bytes_kv / bw_comm

    eff(c)       = eff_inf / (1 + (c_star / c) ** gamma),      c = T / CP

The point of the split is that ``t_fixed`` and the kernel quality live in
different places: ``t_fixed`` is *measured* per setup (embedding lookup,
all2all, optimizer, dataloader) and never fitted, so that it cannot silently
absorb CP-independent variance and flatter the fit.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields, replace
from typing import Dict, List, Optional

from .spec import Layout
from .work import rank_work

GIB = 1024.0 ** 3


@dataclass
class Params:
    eff_inf: float = 0.45      # asymptotic kernel efficiency (fraction of peak)
    c_star: float = 2000.0     # half-efficiency chunk length, in tokens
    gamma: float = 1.0         # sharpness of the efficiency roll-off
    alpha: float = 12e-6       # *exposed* per-collective latency, seconds
    bw_comm: float = 300e9     # *exposed* effective collective bandwidth, bytes/s
    overlap: float = 0.0       # fraction of comm hidden; PIN THIS, do not fit it
    bw_util: float = 0.6       # HBM efficiency for elementwise traffic; PIN THIS
    recompute: float = 4.0 / 3.0  # activation-recompute multiplier on fwd work

    # Which parameters are fitted by default.
    #
    # ``overlap`` is deliberately *not* here.  In the additive cost
    #     t = t_comp + (1 - overlap) * (alpha + V / bw_comm)
    # only the products ``(1-overlap)*alpha`` and ``(1-overlap)/bw_comm`` enter,
    # so fitting all three gives a one-parameter family of equally good fits and
    # the optimizer will happily drive ``overlap -> 1, bw_comm -> 0``.  Measure
    # the overlap from a communication-only microbenchmark and pin it; then the
    # two exposed terms become identifiable.  Same reasoning for ``bw_util``
    # and ``recompute``, which can be measured directly.
    FREE: tuple = ("eff_inf", "c_star", "gamma", "alpha", "bw_comm")

    def copy(self, **kw) -> "Params":
        return replace(self, **kw)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "Params":
        return cls(**json.loads(s))

    def describe(self, names: Optional[List[str]] = None) -> str:
        names = names or [f.name for f in fields(self)]
        return ", ".join(f"{n}={getattr(self, n):.4g}" for n in names)


def efficiency(p: Params, chunk: float) -> float:
    if chunk <= 0:
        return 0.0
    return p.eff_inf / (1.0 + (p.c_star / chunk) ** p.gamma)


@dataclass
class Ablation:
    """Deliberately cripple one structural ingredient.

    Used to show that each ingredient earns its place.  If turning off
    ``exact_pairs`` does not hurt, the mask-topology derivation is decoration
    and the paper has no thesis.
    """

    layout_override: Optional[str] = None   # "causal": pretend the mask is a triangle
    critical: str = "max"                   # "max" | "mean"
    efficiency: bool = True                 # False: pure roofline, eff = 1
    comm: bool = True                       # False: no collective cost
    elementwise: bool = True                # False: drop the L^2 byte term

    def label(self) -> str:
        bits = []
        if self.layout_override:
            bits.append(f"layout={self.layout_override}")
        if self.critical != "max":
            bits.append(f"critical={self.critical}")
        if not self.efficiency:
            bits.append("no-eff")
        if not self.comm:
            bits.append("no-comm")
        if not self.elementwise:
            bits.append("no-ew")
        return "+".join(bits) or "full"


def predict(
    model,
    layout: Layout,
    T: int,
    B: int,
    cp: int,
    hw,
    p: Params,
    t_fixed_ms: float = 0.0,
    scheme: str = "block",
    detail: bool = False,
    abl: Optional[Ablation] = None,
):
    """Predicted step time in milliseconds.

    Returns a float, or ``(ms, breakdown)`` when ``detail`` is True.
    """
    if abl is not None and abl.layout_override == "causal":
        layout = Layout.causal(T)

    works = rank_work(model, layout, T, B, cp=cp, scheme=scheme)
    chunk = T / cp
    eff = efficiency(p, chunk) if (abl is None or abl.efficiency) else 1.0

    t_rank = []
    for w in works:
        t_comp = w.flops / (hw.peak_flops * eff)
        t_mem = w.bytes_ew / (hw.hbm_bw * p.bw_util) if (abl is None or abl.elementwise) else 0.0
        t_comm = (p.alpha + w.bytes_kv / p.bw_comm) if (abl is None or abl.comm) else 0.0
        t_rank.append(p.recompute * (t_comp + t_mem) + (1.0 - p.overlap) * t_comm)

    crit = max(t_rank) if (abl is None or abl.critical == "max") else (sum(t_rank) / cp)
    total = t_fixed_ms / 1e3 + crit

    if not detail:
        return total * 1e3

    crit_rank = max(range(cp), key=lambda r: t_rank[r])
    w = works[crit_rank]
    bd = {
        "step_ms": total * 1e3,
        "fixed_ms": t_fixed_ms,
        "compute_ms": p.recompute * w.flops / (hw.peak_flops * eff) * 1e3,
        "elementwise_ms": p.recompute * w.bytes_ew / (hw.hbm_bw * p.bw_util) * 1e3,
        "comm_ms": (1.0 - p.overlap) * (p.alpha + w.bytes_kv / p.bw_comm) * 1e3,
        "eff": eff,
        "chunk": chunk,
        "crit_rank": crit_rank,
        "imbalance": max(x.pairs for x in works) / (sum(x.pairs for x in works) / cp),
        "flops_per_rank": w.flops,
        "mfus": w.flops / (hw.peak_flops * crit) if crit else 0.0,
        "amdahl_ceiling": amdahl_ceiling(model, layout, T, cp),
    }
    return total * 1e3, bd


def amdahl_ceiling(model, layout: Layout, T: int, cp: int, B: int = 1) -> float:
    """Best possible speedup from CP if comm, imbalance and kernel inefficiency vanish."""
    works = rank_work(model, layout, T, B, cp=1)
    attn = sum(w.flops_attn for w in works)
    dense = sum(w.flops_dense for w in works)
    a = attn / (attn + dense) if (attn + dense) else 0.0
    return 1.0 / ((1.0 - a) + a / cp)


def speedup(model, layout, T, B, p: Params, hw, cp: int, t_fixed_ms: float = 0.0,
            scheme: str = "block") -> float:
    base = predict(model, layout, T, B, 1, hw, p, t_fixed_ms, scheme)
    return base / predict(model, layout, T, B, cp, hw, p, t_fixed_ms, scheme)


def scaling_factor(model, layout, T, B, p: Params, hw, cp: int,
                   t_fixed_ms: float = 0.0, scheme: str = "block") -> float:
    """QPS scaling from 1 GPU to ``cp`` GPUs at fixed batch -- what CP papers report.

    Because CP splits one sequence and leaves the batch alone, throughput
    scaling and per-step speedup are the *same number*:

        QPS(cp GPUs) / QPS(1 GPU) = (B / t_cp) / (B / t_1) = t_1 / t_cp

    The ideal value is ``cp``, so a reported 1.33x at CP=2 is 67% parallel
    efficiency: per-rank time fell to 0.75 of the single-GPU time rather than
    0.5.  It is not a 1.33x speedup over CP=1 in the Amdahl sense.
    """
    return speedup(model, layout, T, B, p, hw, cp, t_fixed_ms, scheme)


def roofline_flops_per_sec(hw, p: Params, chunk: float) -> float:
    return hw.peak_flops * efficiency(p, chunk)
