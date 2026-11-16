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

from .comm import CommModel
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
    overlap: float = 0.0       # fraction of latency hidden. PIN 0 on 910C: HCCL
                               # is not pipelined against compute there.
    bw_util: float = 0.6       # HBM efficiency for elementwise traffic; PIN THIS
    recompute: float = 4.0 / 3.0  # activation-recompute multiplier on fwd work
    v_switch: float = 0.0      # message size where the collective algorithm
                               # switches to ring; 0 = same policy at all sizes
    step_policy: str = "auto"  # "auto"/"rd": ceil(log2 n); "ring": n-1
    t_emb_ms: float = 0.0      # measured single-device lookup time at cp=1
    emb_shard_by_cp: bool = True   # does the lookup run on the CP shard?
    emb_unexplained: float = 1.0   # d_stage - sum(known stages), for reconciliation

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

    def comm_model(self) -> CommModel:
        """The latency/bandwidth pair plus its step count.

        ``alpha`` and ``bw_comm`` stay the fitted numbers; the step count is
        derived from ``step_policy`` and the rank count, so no per-size bucket is
        ever fitted.
        """
        return CommModel(alpha=self.alpha, bw=self.bw_comm,
                         policy=self.step_policy, v_switch=self.v_switch,
                         overlap=self.overlap)

    def t_emb_for(self, cp: int) -> float:
        """Per-device lookup time in ms (requirement 7: sharded by CP)."""
        if self.t_emb_ms <= 0:
            return 0.0
        return self.t_emb_ms / max(cp, 1) if self.emb_shard_by_cp else self.t_emb_ms

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

    cm = p.comm_model()
    t_rank = []
    for w in works:
        t_comp = w.flops / (hw.peak_flops * eff)
        t_mem = w.bytes_ew / (hw.hbm_bw * p.bw_util) if (abl is None or abl.elementwise) else 0.0
        t_comm = cm.time(cp, w.bytes_kv, rounds=w.rounds) \
            if (abl is None or abl.comm) else 0.0
        t_rank.append(p.recompute * (t_comp + t_mem) + t_comm)

    crit = max(t_rank) if (abl is None or abl.critical == "max") else (sum(t_rank) / cp)
    total = t_fixed_ms / 1e3 + p.t_emb_for(cp) / 1e3 + crit

    if not detail:
        return total * 1e3

    crit_rank = max(range(cp), key=lambda r: t_rank[r])
    w = works[crit_rank]
    bd = {
        "step_ms": total * 1e3,
        "fixed_ms": t_fixed_ms + p.t_emb_for(cp),
        "compute_ms": p.recompute * w.flops / (hw.peak_flops * eff) * 1e3,
        "elementwise_ms": p.recompute * w.bytes_ew / (hw.hbm_bw * p.bw_util) * 1e3,
        "comm_ms": cm.time(cp, w.bytes_kv, rounds=w.rounds) * 1e3,
        "comm_latency_ms": cm.time_split(cp, w.bytes_kv, rounds=w.rounds)[0] * 1e3,
        "comm_bytes_ms": cm.time_split(cp, w.bytes_kv, rounds=w.rounds)[1] * 1e3,
        "comm_rounds": w.rounds,
        "comm_steps": cm.steps_for(cp, w.bytes_kv / max(w.rounds, 1)),
        "comm_v_star_mb": cm.break_even_bytes(cp) / 1e6,
        "t_emb_ms": p.t_emb_for(cp),
        "eff": eff,
        "chunk": chunk,
        "crit_rank": crit_rank,
        "imbalance": max(x.pairs for x in works) / (sum(x.pairs for x in works) / cp),
        "flops_per_rank": w.flops,
        "mfus": w.flops / (hw.peak_flops * crit) if crit else 0.0,
        "amdahl_ceiling": amdahl_ceiling(model, layout, T, cp),
    }
    return total * 1e3, bd


def attribute(
    model,
    layout: Layout,
    T: int,
    B: int,
    cp: int,
    hw,
    p: Params,
    t_fixed_ms: float = 0.0,
    scheme: str = "block",
) -> dict:
    """Split the gap between perfect scaling and reality into four named parts.

    Built by turning one mechanism on at a time, so the increments sum to the
    prediction exactly and each has a physical name:

        stage 0   t_fixed + shard_1 / cp          perfect efficiency, balance, no comm
        stage 1   + efficiency loss               eff(c) instead of eff(T), mean over ranks
        stage 2   + load imbalance                max over ranks instead of mean
        stage 3   + exposed communication         == cost.predict

    This is the diagnostic for "the predicted speedup is too low".  Instead of
    guessing whether the culprit is communication or a missing fixed cost, read
    the three increments: they are in milliseconds and in percent of the step.
    """
    if cp < 1:
        raise ValueError("cp must be >= 1")

    works1 = rank_work(model, layout, T, B, cp=1, scheme=scheme)
    works_n = rank_work(model, layout, T, B, cp=cp, scheme=scheme)
    eff1 = efficiency(p, float(T))
    effc = efficiency(p, T / cp)

    def comp(w, eff):
        return p.recompute * (
            w.flops / (hw.peak_flops * eff)
            + w.bytes_ew / (hw.hbm_bw * p.bw_util)
        )

    cm = p.comm_model()

    def exposed_comm(w):
        return cm.time(cp, w.bytes_kv, rounds=w.rounds)

    shard_1 = sum(comp(w, eff1) for w in works1)
    mean_n = sum(comp(w, effc) for w in works_n) / cp
    worst = max(works_n, key=lambda w: comp(w, effc) + exposed_comm(w))
    max_n = comp(worst, effc)
    comm = exposed_comm(worst)

    emb = p.t_emb_for(cp) / 1e3
    s0 = t_fixed_ms / 1e3 + emb + shard_1 / cp
    s1 = t_fixed_ms / 1e3 + emb + mean_n
    s2 = t_fixed_ms / 1e3 + emb + max_n
    s3 = t_fixed_ms / 1e3 + emb + max_n + comm

    total = s3
    return {
        "cp": cp,
        "step_ms": total * 1e3,
        "fixed_ms": (t_fixed_ms / 1e3 + emb) * 1e3,
        "emb_ms": emb * 1e3,
        "ideal_shard_ms": (shard_1 / cp) * 1e3,
        "eff_loss_ms": (s1 - s0) * 1e3,
        "imbalance_loss_ms": (s2 - s1) * 1e3,
        "comm_ms": (s3 - s2) * 1e3,
        "speedup": (t_fixed_ms / 1e3 + shard_1) / total,
        "speedup_ideal": float(cp),
        # shares of the *step*, which is what a reader can act on
        "fixed_share": (t_fixed_ms / 1e3) / total,
        "ideal_share": (shard_1 / cp) / total,
        "eff_share": (s1 - s0) / total,
        "imbalance_share": (s2 - s1) / total,
        "comm_share": (s3 - s2) / total,
        "eff": effc,
        "eff_at_cp1": eff1,
    }


def attribution_report(
    model, layout: Layout, T: int, B: int, cps, hw, p: Params,
    t_fixed_ms: float = 0.0, scheme: str = "block",
) -> str:
    """Table of ``attribute`` over several CP degrees."""
    head = (f"  {'cp':>3} {'step ms':>9} {'speedup':>8} {'ideal':>6} "
            f"{'fixed':>7} {'eff':>7} {'imbal':>7} {'comm':>7}")
    lines = [head]
    for cp in cps:
        a = attribute(model, layout, T, B, cp, hw, p, t_fixed_ms, scheme)
        lines.append(
            f"  {cp:>3} {a['step_ms']:>9.2f} {a['speedup']:>8.2f} "
            f"{a['speedup_ideal']:>6.0f} "
            f"{a['fixed_share']:>6.1%} {a['eff_share']:>6.1%} "
            f"{a['imbalance_share']:>6.1%} {a['comm_share']:>6.1%}")
    lines.append("  (columns after 'ideal' are shares of the step; they sum to 100%)")
    return "\n".join(lines)


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
