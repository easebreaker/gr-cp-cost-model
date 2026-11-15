"""Metrics that match the actual use of a cost model: *ranking*, not absolutes.

If the model is used to choose a CP degree, then a 20% absolute error that
preserves ordering is harmless, while a 5% error that flips the best
configuration is fatal.  So ranking metrics are primary and MAPE is secondary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


def mape(pred: Sequence[float], meas: Sequence[float]) -> float:
    return 100.0 * sum(abs(p - m) / m for p, m in zip(pred, meas)) / len(pred)


def mdape(pred: Sequence[float], meas: Sequence[float]) -> float:
    e = sorted(abs(p - m) / m for p, m in zip(pred, meas))
    n = len(e)
    return 100.0 * (e[n // 2] if n % 2 else 0.5 * (e[n // 2 - 1] + e[n // 2]))


def bias(pred: Sequence[float], meas: Sequence[float]) -> float:
    """Signed mean percentage error; +10 means the model over-predicts."""
    return 100.0 * sum((p - m) / m for p, m in zip(pred, meas)) / len(pred)


def p90_ape(pred: Sequence[float], meas: Sequence[float]) -> float:
    e = sorted(abs(p - m) / m for p, m in zip(pred, meas))
    k = min(len(e) - 1, int(math.ceil(0.9 * len(e))) - 1)
    return 100.0 * e[k]


def kendall_tau(a: Sequence[float], b: Sequence[float]) -> float:
    """Kendall's tau-a with the O(n^2) definition (small n, no scipy)."""
    n = len(a)
    if n < 2:
        return 1.0
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            sa = (a[i] - a[j])
            sb = (b[i] - b[j])
            prod = sa * sb
            if prod > 0:
                conc += 1
            elif prod < 0:
                disc += 1
    tot = n * (n - 1) / 2
    return (conc - disc) / tot


def best_config_accuracy(pred: Sequence[float], meas: Sequence[float]) -> float:
    """Strict: 1.0 iff the model's argmin *is* the measured argmin.

    This metric punishes the model for a coin flip between two configurations
    that are 1% apart -- which is below the noise of any real measurement, and
    below the noise of the model itself.  Use ``selection_accuracy`` for the
    number you would report.
    """
    return float(min(range(len(pred)), key=lambda i: pred[i]) ==
                 min(range(len(meas)), key=lambda i: meas[i]))


def selection_accuracy(pred: Sequence[float], meas: Sequence[float],
                       tol: float = 0.05) -> float:
    """Tie-aware: 1.0 if the config the model picks is within ``tol`` of the best.

    Why a band and not strict equality: choosing between two configs means
    comparing two noisy measurements, so the resolution of the comparison is
    ~sqrt(2) times the per-measurement noise.  With 3% measurement noise the
    honest tie band is ~4-5%.  Claims of "our model picks the right CP degree"
    are only meaningful if that band is stated.

    When the speedup curve is flat -- small T, where CP buys little -- every
    reasonable model produces a near-tie, and reporting strict top-1 accuracy
    measures luck rather than skill.
    """
    i = min(range(len(pred)), key=lambda k: pred[k])
    best = min(meas)
    return float(meas[i] <= best * (1.0 + tol))


@dataclass
class Report:
    n: int
    mape: float
    mdape: float
    p90: float
    bias: float
    tau: Optional[float] = None
    top1: Optional[float] = None
    sel: Optional[float] = None

    def row(self, label: str = "") -> str:
        rank = ""
        if self.tau is not None:
            rank = f"  tau={self.tau:+.3f}  top1={self.top1:.0%}  sel={self.sel:.0%}"
        return (
            f"{label:<26} n={self.n:<4} MAPE={self.mape:6.2f}%  "
            f"MdAPE={self.mdape:6.2f}%  P90={self.p90:6.2f}%  "
            f"bias={self.bias:+6.2f}%{rank}"
        )

    def ranking_row(self, label: str = "") -> str:
        if self.tau is None:
            return f"{label:<26} n={self.n:<4} (no CP sweep -- ranking undefined)"
        return (f"{label:<26} tau={self.tau:+.3f}  top1={self.top1:.0%}  "
                f"sel={self.sel:.0%}  n={self.n}")


def report(pred: Sequence[float], meas: Sequence[float],
           groups: Optional[Sequence[str]] = None,
           tol: float = 0.05) -> Report:
    """Ranking metrics are only defined *within* a CP sweep.

    Ranking across different T or different models is meaningless (the short
    sequence is always faster), so pass ``groups`` only when the members of a
    group differ in a knob you are choosing -- normally CP.
    """
    tau: Optional[float] = None
    top1: Optional[float] = None
    sel: Optional[float] = None
    if groups is not None:
        taus, hits, sels = [], [], []
        for g in sorted(set(groups)):
            idx = [i for i, gg in enumerate(groups) if gg == g]
            if len(idx) < 2:
                continue
            taus.append(kendall_tau([pred[i] for i in idx], [meas[i] for i in idx]))
            hits.append(best_config_accuracy([pred[i] for i in idx], [meas[i] for i in idx]))
            sels.append(selection_accuracy([pred[i] for i in idx], [meas[i] for i in idx], tol))
        if taus:
            tau = sum(taus) / len(taus)
            top1 = sum(hits) / len(hits)
            sel = sum(sels) / len(sels)
    return Report(
        n=len(pred),
        mape=mape(pred, meas),
        mdape=mdape(pred, meas),
        p90=p90_ape(pred, meas),
        bias=bias(pred, meas),
        tau=tau,
        top1=top1,
        sel=sel,
    )


def by_group(pred: Sequence[float], meas: Sequence[float],
             groups: Sequence[str], rank: bool = False) -> Dict[str, Report]:
    """Per-group report.

    ``rank=True`` treats each group as one CP sweep and reports tau/top1 within
    it.  Leave it False when the group varies T or model, where "best config"
    has no meaning.
    """
    out = {}
    for g in sorted(set(groups)):
        idx = [i for i, gg in enumerate(groups) if gg == g]
        sub = [g] * len(idx) if rank else None
        out[g] = report([pred[i] for i in idx], [meas[i] for i in idx], sub)
    return out


def leave_one_out(groups: Sequence[str]) -> List[List[int]]:
    """Index sets for leave-one-group-out validation."""
    uniq = sorted(set(groups))
    return [[i for i, g in enumerate(groups) if g != u] for u in uniq]
