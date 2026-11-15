"""Quality side of the composition: power-law returns to length, size, data.

Reserved extension point for the "cost model x scaling law" topic.  Nothing here
is a result -- it is the *interface* the cost model needs, plus the guards that
keep a fit from producing a frontier it cannot support.

The form
--------
    Q(L, m, D) = Q_inf - A_L (L_ref/L)^alpha
                        - A_m (m_ref/m)^beta
                        - A_D (D_ref/D)^gamma

Chinchilla-style additive deficits over three levers.  Additive-in-deficit
rather than multiplicative because the published GR curves (LONGER Fig. 4,
MTGR's three sizes) both show saturating returns, and a deficit form keeps Q
bounded above by something interpretable.

This is an *assumption*, not a law.  It says the three levers substitute for each
other one-for-one in deficit space, which is the weakest part of the composition
and the first thing to test (docs/10-scaling-law.md §4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

from .fit import nelder_mead


@dataclass
class QualityLaw:
    """Saturating additive-deficit quality model.

    The defaults are deliberately *plausible but unsourced* and are labelled as
    such: they exist so the frontier code runs, not so anyone can quote a number
    out of them.
    """

    q_inf: float = 0.75            # asymptote (AUC-like, in [0,1])
    a_len: float = 0.030           # length deficit at L = l_ref
    alpha: float = 0.35            # length exponent
    a_size: float = 0.020
    beta: float = 0.25             # capacity exponent
    a_data: float = 0.040
    gamma: float = 0.30            # token-budget exponent
    l_ref: float = 1000.0
    m_ref: float = 1.0e8
    d_ref: float = 1.0e11
    sourced: bool = False          # True only once fitted to real points

    def copy(self, **kw) -> "QualityLaw":
        return replace(self, **kw)

    # -- the model -------------------------------------------------------
    def quality(self, L: float, m: float, D: float) -> float:
        """Predicted quality; monotone increasing in all three arguments."""
        q = self.q_inf
        if self.a_len and L > 0:
            q -= self.a_len * (self.l_ref / L) ** self.alpha
        if self.a_size and m > 0:
            q -= self.a_size * (self.m_ref / m) ** self.beta
        if self.a_data and D > 0:
            q -= self.a_data * (self.d_ref / D) ** self.gamma
        return q

    def marginal_length(self, L: float) -> float:
        """dQ/dL -- what one more token of context is worth."""
        if not self.a_len or L <= 0:
            return 0.0
        return (self.a_len * self.alpha * self.l_ref ** self.alpha
                * L ** (-self.alpha - 1.0))

    def length_for_deficit(self, target_deficit: float) -> float:
        """The L that achieves a given length-deficit (budget inversion)."""
        if target_deficit <= 0:
            return float("inf")
        return self.l_ref * (self.a_len / target_deficit) ** (1.0 / self.alpha)

    def describe(self) -> str:
        tag = "" if self.sourced else "  [UNSOURCED: placeholder defaults]"
        return (f"Q_inf={self.q_inf:.4f}  A_L={self.a_len:.4g} alpha={self.alpha:.3f}  "
                f"A_m={self.a_size:.4g} beta={self.beta:.3f}  "
                f"A_D={self.a_data:.4g} gamma={self.gamma:.3f}{tag}")


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------
@dataclass
class QualityPoint:
    L: float
    m: float
    D: float
    q: float
    source: str = ""

    def row(self) -> str:
        return (f"  L={self.L:<8.0f} m={self.m / 1e6:<8.1f}M "
                f"D={self.D / 1e9:<8.2f}B q={self.q:.4f}  {self.source}")


def _spread(vals: Sequence[float]) -> float:
    lo, hi = min(vals), max(vals)
    return hi / lo if lo > 0 else float("inf")


def fit_quality(points: Sequence[QualityPoint],
                p0: Optional[QualityLaw] = None,
                restarts: int = 8) -> Tuple[QualityLaw, float, bool]:
    """Fit the seven free numbers. Returns ``(law, rmse, identifiable)``.

    ``identifiable`` is False unless all three levers actually vary.  A grid that
    only varies ``L`` will happily "converge" with ``a_size`` and ``a_data``
    pinned at their initial values, and the resulting frontier is then driven by
    arbitrary constants -- the same trap as ``memory.fit_memory``.
    """
    if len(points) < 4:
        raise ValueError(f"need >= 4 points, got {len(points)}")
    spread = {
        "L": _spread([p.L for p in points]),
        "m": _spread([p.m for p in points]),
        "D": _spread([p.D for p in points]),
    }
    identifiable = all(v > 1.2 for v in spread.values())

    base = p0 or QualityLaw()
    data = [(p.L, p.m, p.D, p.q) for p in points]

    def unpack(x) -> QualityLaw:
        return base.copy(
            q_inf=1.0 / (1.0 + math.exp(-x[0])),
            a_len=math.exp(x[1]), alpha=math.exp(x[2]),
            a_size=math.exp(x[3]), beta=math.exp(x[4]),
            a_data=math.exp(x[5]), gamma=math.exp(x[6]),
            sourced=True,
        )

    def obj(x) -> float:
        law = unpack(x)
        return sum((law.quality(L, m, D) - q) ** 2 for L, m, D, q in data)

    x0 = [math.log(max(base.q_inf, 1e-6) / max(1 - base.q_inf, 1e-6)),
          math.log(max(base.a_len, 1e-9)), math.log(max(base.alpha, 1e-6)),
          math.log(max(base.a_size, 1e-9)), math.log(max(base.beta, 1e-6)),
          math.log(max(base.a_data, 1e-9)), math.log(max(base.gamma, 1e-6))]

    best, best_f = None, float("inf")
    for k in range(restarts):
        jitter = 0.0 if k == 0 else 0.5
        start = [x0[i] + jitter * ((k * 7 + i * 3) % 5 - 2) / 3.0
                 for i in range(len(x0))]
        x, f, _ = nelder_mead(obj, start, step=0.25)
        x, f, _ = nelder_mead(obj, x, step=0.05)
        if f < best_f:
            best, best_f = x, f
    return unpack(best), math.sqrt(best_f / len(data)), identifiable


# --------------------------------------------------------------------------
# A named placeholder used by the demo
# --------------------------------------------------------------------------
ANCHOR_NOTE = "placeholder: shape only, not a measurement"


def demo_law() -> QualityLaw:
    """Numbers chosen to give a *visible* knee and a plausible frontier.

    Not derived from any table.  ``sourced`` stays False so every report that
    uses it prints the warning.
    """
    return QualityLaw(q_inf=0.752, a_len=0.0285, alpha=0.38,
                      a_size=0.0175, beta=0.22,
                      a_data=0.041, gamma=0.31, sourced=False)
