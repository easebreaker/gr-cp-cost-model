"""Collective cost: why "bandwidth depends on message size", and how to model it.

The observation
---------------
Measured collective throughput rises with message size.  Reporting that as
"``bw_comm`` depends on size" and fitting a piecewise curve works on the data you
have, then fails on any other card count -- because the breakpoints move with
``N``.

The mechanism
-------------
An all-gather is not one transfer.  It is a series of dependent steps:

    t(V) = S(N, V) * alpha   +   V / BW_inf

with ``S`` the number of *sequential* steps (recursive doubling gives
``ceil(log2 N)``; a ring gives ``N-1``; a chunked ring also pipelines, which the
``V/BW`` term already implies).  Only the first term is serial per step.

Divide V by t and the *apparent* bandwidth is

    BW_app(V) = V / t = BW_inf / (1 + V* / V),
    V* = S * alpha * BW_inf        <-- break-even message size

That is a saturating curve of exactly the same shape as ``eff(c)``.  So:

* the "size dependence" is not a property of the link -- it is the latency term
  being amortised over a growing payload;
* there is one number that decides the regime, ``V*``, and it is computable from
  a latency and a bandwidth;
* ``V*`` scales with ``S``, hence with ``N``: 8 cards need ~7x the message size
  that 2 cards need before they look bandwidth-limited.

Why this matters for CP on jagged input
--------------------------------------
``V*`` is compared against the *per-collective* payload, not the step total.  A
jagged implementation that issues one collective per sample or per bucket
produces many small messages, so ``rounds`` rises and each one sits in the
latency-dominated regime.  That is the same failure the earlier revision of this
model hit from a different direction (forgetting ``n_layers``): the byte total is
right but the *number of times you pay latency* is wrong.

So the model carries two independent counts:

    rounds        how many collectives per step   (scheme x layers x buckets)
    S(N, V)       how many dependent steps inside each one

and the cost is

    t = rounds * ( S(N,V) * alpha ) + V_total / BW_inf

with ``overlap`` multiplying the latency part only where the hardware can hide it
(see ``Hardware.pipeline_overlap``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

GIB = 1024.0 ** 3


# --------------------------------------------------------------------------
# Step-count policies
# --------------------------------------------------------------------------
def steps_recursive_doubling(n: int) -> int:
    """log2(N) dependent steps: the small-message algorithm."""
    return max(0, int(math.ceil(math.log2(max(n, 1)))))


def steps_ring(n: int) -> int:
    """N-1 dependent steps: the large-message algorithm."""
    return max(0, n - 1)


def steps(n: int, policy: str = "auto") -> int:
    if policy == "ring":
        return steps_ring(n)
    if policy in ("recursive_doubling", "rd", "small"):
        return steps_recursive_doubling(n)
    if policy == "auto":
        return steps_recursive_doubling(n)
    raise ValueError(f"unknown step policy {policy!r}")


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------
@dataclass
class CommModel:
    """Latency + bandwidth, with an explicit step count.

    ``alpha``  per-step latency, seconds.  This is the only latency parameter;
               do not fit one alpha per size bucket.
    ``bw``     asymptotic bandwidth, bytes/s.
    ``policy`` step-count policy: ``auto``/``rd`` (log N) or ``ring`` (N-1).
    ``v_switch`` message size above which the ring algorithm takes over.  Set to
               0 to disable switching (one algorithm for all sizes).  This is the
               *only* place the size dependence is allowed to appear as a
               discrete change, and it is one integer, not a set of breakpoints.
    ``overlap`` fraction of the latency term hidden behind compute.
    """

    alpha: float = 15e-6
    bw: float = 149e9
    policy: str = "auto"
    v_switch: float = 0.0
    overlap: float = 0.0

    def steps_for(self, n: int, v_bytes: float) -> int:
        if self.policy == "ring":
            return steps_ring(n)
        if self.v_switch > 0 and v_bytes >= self.v_switch:
            return steps_ring(n)
        return steps_recursive_doubling(n)

    def break_even_bytes(self, n: int) -> float:
        """V* = S * alpha * BW: below this, latency dominates."""
        s = self.steps_for(n, 0.0)
        return s * self.alpha * self.bw

    def apparent_bw(self, n: int, v_bytes: float) -> float:
        """What a throughput measurement would report at this message size."""
        if v_bytes <= 0:
            return 0.0
        return v_bytes / self.time(n, v_bytes)

    def time(self, n: int, v_bytes: float, rounds: int = 1) -> float:
        """Seconds for ``rounds`` collectives carrying ``v_bytes`` in total."""
        if n <= 1 or v_bytes <= 0:
            return 0.0
        per_round = v_bytes / max(rounds, 1)
        s = self.steps_for(n, per_round)
        latency = rounds * s * self.alpha
        transfer = v_bytes / self.bw
        return (1.0 - self.overlap) * latency + transfer

    def time_split(self, n: int, v_bytes: float, rounds: int = 1) -> Tuple[float, float]:
        """(latency part, transfer part) -- for attribution and for tests."""
        if n <= 1 or v_bytes <= 0:
            return 0.0, 0.0
        per_round = v_bytes / max(rounds, 1)
        s = self.steps_for(n, per_round)
        return ((1.0 - self.overlap) * rounds * s * self.alpha,
                v_bytes / self.bw)


def diagnose_size_dependence(
    points,
    n: int,
    policy: str = "auto",
) -> str:
    """Explain a measured (bytes, seconds) curve, or say that it cannot be.

    Fits only ``alpha`` and ``bw`` -- two numbers -- and then reports whether the
    residual is consistent with one step count.  If it is not, the curve is not a
    latency/bandwidth curve and piecewise fitting it would only hide that.
    """
    pts = [(float(v), float(t)) for v, t in points if v > 0 and t > 0]
    if len(pts) < 3:
        return "  need >= 3 points"

    s = steps(n, policy)
    # Least squares for t = s*alpha + V/bw, solved as a + b*V with a = s*alpha.
    xs = [v for v, _ in pts]
    ys = [t for _, t in pts]
    m = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    det = sxx * m - sx * sx
    if abs(det) < 1e-30:
        return "  degenerate byte range"
    slope = (sxy * m - sx * sy) / det          # = 1/bw
    intercept = (sy - slope * sx) / m          # = s*alpha
    bw = 1.0 / slope if slope > 0 else float("inf")
    alpha = intercept / s if s else 0.0
    resid = [intercept + slope * x - y for x, y in zip(xs, ys)]
    rmse = math.sqrt(sum(r * r for r in resid) / m)
    mean_t = sum(ys) / m
    rel = rmse / mean_t if mean_t else 0.0

    v_star = s * alpha * bw

    # Held-out check: fit on the larger messages, predict the smaller ones.  A
    # free intercept can absorb a lot of structure, so an in-sample residual is a
    # weak test; extrapolating down in size is a strong one.
    order = sorted(range(m), key=lambda i: xs[i])
    hold = max(1, m // 3)
    train = order[hold:]
    test = order[:hold]
    hx = [xs[i] for i in train]
    hy = [ys[i] for i in train]
    mt = len(hx)
    tx, ty = sum(hx), sum(hy)
    txx = sum(x * x for x in hx)
    txy = sum(x * y for x, y in zip(hx, hy))
    det2 = txx * mt - tx * tx
    holdout_rel = None
    if abs(det2) > 1e-30 and test:
        sl2 = (txy * mt - tx * ty) / det2
        ic2 = (ty - sl2 * tx) / mt
        errs = [abs(ic2 + sl2 * xs[i] - ys[i]) / ys[i] for i in test]
        holdout_rel = sum(errs) / len(errs)

    lines = [
        f"  n={n}  steps={s}  alpha={alpha * 1e6:.1f}us  bw={bw / 1e9:.1f} GB/s",
        f"  break-even message size V* = {v_star / 1e6:.2f} MB"
        f"   (below this, latency dominates)",
        f"  fit residual: {rel:.1%} of the mean time",
    ]
    lines.append(f"  {'bytes':>10} {'measured':>10} {'model':>10} "
                 f"{'BW_app GB/s':>12} {'regime':>10}")
    for (v, t), r in zip(pts, resid):
        reg = "latency" if v < v_star else "bandwidth"
        lines.append(f"  {v / 1e6:>9.2f}M {t * 1e6:>9.1f}us "
                     f"{(t + r) * 1e6:>9.1f}us "
                     f"{v / t / 1e9:>12.1f} {reg:>10}")
    if holdout_rel is not None:
        lines.insert(3, f"  held-out (fit on the large third out, predict the "
                        f"smallest {hold}): {holdout_rel:.0%} error")
    if (holdout_rel is not None and holdout_rel > 0.5) or rel > 0.25:
        lines.append("  >> the single-step-count model does not explain this curve.")
        lines.append("     Do NOT fit piecewise breakpoints; check whether the")
        lines.append("     implementation changes algorithm mid-range (v_switch),")
        lines.append("     or whether the number of collectives changes with size.")
        lines.append("     Note: a pure step-count change is nearly invisible when the")
        lines.append("     two regimes are far apart -- in the bandwidth regime the")
        lines.append("     intercept no longer matters. A residual this large means")
        lines.append("     the *slope* changed, i.e. the effective bandwidth did.")
    else:
        lines.append("  >> one alpha and one bw explain the curve. The apparent")
        lines.append("     size dependence is amortisation of V* / V, not a")
        lines.append("     property of the link.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Volume for the schemes we care about
# --------------------------------------------------------------------------
def allgather_volume_per_rank(v_bytes: float, n: int) -> float:
    """Bytes each rank must receive for an all-gather of ``v_bytes`` total.

    Ring and recursive doubling move the same volume: ``(n-1)/n * V`` per rank.
    They differ in step count and peak buffer, not in bandwidth consumed.  That
    distinction is the reason "all-gather instead of ring" is a *latency and
    memory* change, not a bandwidth change -- and it is easy to mis-model.
    """
    if n <= 1:
        return 0.0
    return v_bytes * (n - 1) / n


def allgather_peak_buffer(v_bytes: float, n: int) -> float:
    """Peak per-rank buffer: all-gather needs the whole tensor, ring needs 1/n."""
    return v_bytes
