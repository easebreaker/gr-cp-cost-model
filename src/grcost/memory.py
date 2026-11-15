"""Activation memory: the *first* reason CP exists, and the easier thing to predict.

Time is a secondary objective.  CP is adopted because it is the only way to fit
longer sequences, and the memory question ("how long a sequence can I run?") is
deterministic to within a few percent:

    M_rank(cp) ~= M_static + a * B * (T / cp) + delta(cp)

``a`` (bytes of activation per token) can be *inferred from one observation*
of the maximum sequence length, which makes the whole memory model essentially
free.  ``delta(cp)`` captures per-rank CP overhead (chunk buffers, padding to
tile granularity, mini-chunk duplication); leaving it out is the classic way to
over-predict long sequences by 1.5x.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, List, Sequence, Tuple

GIB = 1024.0 ** 3

# A back-of-envelope ratio used as a *prior* only.  HSTU stores, per layer per
# token: the fused UVQK projection (4d), the pooled attention output (d), the
# gated output (d).  Measured values land near 6d elements per layer.
DEFAULT_ACT_RATIO = 6.0


def estimate_act_bytes_per_token(model, dtype_bytes: int = 2, ratio: float = DEFAULT_ACT_RATIO) -> float:
    """Prior estimate of checkpointed activation bytes per token."""
    return model.n_layers * ratio * model.d * dtype_bytes


@dataclass
class MemoryFit:
    a_bytes: float
    M_static_gb: float
    delta_gb_per_cp: float = 0.0
    rmse_gb: float = 0.0
    identifiable: bool = True

    def report(self) -> str:
        flag = "" if self.identifiable else "  [NOT identifiable: chunk lengths do not vary]"
        return (f"a={self.a_bytes / 1024:.0f} KiB/token  M_static={self.M_static_gb:.2f} GB  "
                f"delta={self.delta_gb_per_cp:.3f} GB/CP  rmse={self.rmse_gb:.2f} GB{flag}")

    def max_seq_len(self, B: float, cp: int, mem_gb: float) -> int:
        budget = (mem_gb - self.M_static_gb - self.delta_gb_per_cp * cp) * GIB
        if budget <= 0:
            return 0
        return int(cp * budget / (self.a_bytes * B))


def fit_memory(
    observations: Sequence[Tuple[int, int, int, float]],
    delta_gb_per_cp: float = 0.0,
) -> MemoryFit:
    """Fit ``a`` and ``M_static`` from ``(T_max, cp, B, mem_gb)`` observations.

    Each observation says: at CP=cp with batch B on a mem_gb device, the largest
    sequence that fit was T_max.  Two unknowns, so two observations suffice;
    with more, solve the 2x2 normal equations in closed form.

    The model is ``a * B * T_max/cp + M_static = mem - delta*cp``.
    """
    xs, ys = [], []
    for t_max, cp, b, mem in observations:
        xs.append(b * t_max / cp)
        ys.append(mem - delta_gb_per_cp * cp)

    n = len(xs)
    if n == 0:
        raise ValueError("no observations")
    sxx = sum(x * x for x in xs)
    sx = sum(xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sy = sum(ys)
    det = sxx * n - sx * sx
    identifiable = abs(det) > 1e-9 * max(1.0, sxx * n)
    if not identifiable:
        # Every observation shares the same per-rank chunk length, so `a` and
        # M_static trade off exactly and cannot be separated.  Returning a
        # number here would be fabrication, so flag it and return the
        # least-committal solution.
        a_scaled = sxy / sxx if sxx else 0.0
        m0 = (sy - a_scaled * sx) / n
    else:
        a_scaled = (sxy * n - sx * sy) / det
        m0 = (sy - a_scaled * sx) / n

    resid = [a_scaled * x + m0 - y for x, y in zip(xs, ys)]
    return MemoryFit(
        a_bytes=a_scaled * GIB,
        M_static_gb=m0,
        delta_gb_per_cp=delta_gb_per_cp,
        rmse_gb=sqrt(sum(r * r for r in resid) / n),
        identifiable=identifiable,
    )


def check_table(t_max_by_cp: dict) -> str:
    """Is a published (CP -> T_max) table consistent with a pure 1/cp model?

    Two signatures to look for:

    * ``T_max * cp`` roughly constant  -> activation memory is the binding
      constraint and the 1/cp model holds.
    * ``T_max`` itself roughly constant -> something else is binding (kernel
      workspace, tile granularity, or the sweep simply stopped early).  A 1/cp
      model fitted on any single point will then over-predict.
    """
    cps = sorted(t_max_by_cp)
    lines = ["  CP    T_max    T_max*cp   chunk=T/CP   pred@1/cp"]
    base = t_max_by_cp[cps[0]]
    for cp in cps:
        t = t_max_by_cp[cp]
        lines.append(
            f"  {cp:<4} {t:<8} {t * cp:<10} {t / cp:<11.0f} {base * cp:<8}"
        )
    prod = [t_max_by_cp[c] * c for c in cps]
    spread_prod = max(prod) / min(prod)
    ts = [t_max_by_cp[c] for c in cps]
    spread_t = max(ts) / min(ts)
    if spread_prod < 1.15:
        verdict = "consistent with 1/CP (memory-bound)"
    elif spread_t < 1.3:
        verdict = "NOT 1/CP: chunks saturate -> a second constraint is binding"
    else:
        verdict = "partially 1/CP: needs an explicit delta(cp) term"
    lines.append(f"  spread(T*cp)={spread_prod:.2f}x  spread(T)={spread_t:.2f}x")
    lines.append(f"  verdict: {verdict}")
    return "\n".join(lines)
