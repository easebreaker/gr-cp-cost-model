"""Turn a measured single-layer sweep into the efficiency curve -- without fitting.

The trick: at ``CP=1`` every term in the cost model except ``eff`` is known from
structure, so ``eff(c)`` can be *solved* point by point rather than fitted
jointly with everything else:

    t = t_fixed + recomp * ( flops / (P * eff) + bytes / (BW * bw_util) )

    =>  eff(c) = flops / ( P * ( (t - t_fixed)/recomp - bytes/(BW * bw_util) ) )

Only the three-parameter shape ``eff_inf / (1 + (c_star/c)^gamma)`` is then fitted
to the extracted points, which is a 1-D problem with a visible knee -- far
better conditioned than fitting eight parameters to a step time.

This is what makes the "no CP implementation needed" claim concrete: the sweep
is measured at CP=1, ``c = T``, and CP is then predicted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

from .cost import Params, efficiency
from .fit import nelder_mead
from .spec import Layout
from .work import rank_work


@dataclass
class EffPoint:
    chunk: float
    eff: float
    t_ms: float
    flops: float
    bytes_ew: float
    note: str = ""
    n_layers: int = 0
    bw_util: float = 0.6          # the constant used to solve this point
    recompute: float = 1.0

    def row(self) -> str:
        return (f"  c={self.chunk:<8.0f} eff={self.eff:6.3f}  "
                f"t={self.t_ms:9.3f}ms  flops={self.flops / 1e12:7.2f}T  "
                f"bytes={self.bytes_ew / 1e9:7.2f}GB  L={self.n_layers:<3} "
                f"{self.note}")


def extract_eff(
    model,
    layout_of,
    measurements: Sequence[Tuple[int, float]],
    hw,
    B: int = 32,
    t_fixed_ms: float = 0.0,
    bw_util: float = 0.6,
    recompute: float = 1.0,
    layers_measured: Optional[int] = None,
) -> List[EffPoint]:
    """Solve ``eff`` for each measured ``(T, t_ms)`` at CP=1.

    ``layout_of(T) -> Layout`` because the mask family may depend on length
    (MTGR's candidate tail does).

    ``layers_measured`` is **the number of layers the timed module contained**,
    which is not necessarily ``model.n_layers``.  Measuring one block and then
    dividing its time by a 15-layer FLOP count inflates ``eff`` by exactly
    ``model.n_layers / layers_measured``, and the resulting prediction then
    comes out at the *per-layer* scale -- a factor-of-``n_layers`` error that
    looks like "the model predicts too fast" and is very hard to spot, because
    the fit itself is internally consistent.

    Pass ``layers_measured=1`` for a single-block benchmark, or leave it
    ``None`` when the timed module really is the whole model.
    """
    n_meas = model.n_layers if layers_measured is None else int(layers_measured)
    if n_meas < 1:
        raise ValueError("layers_measured must be >= 1")
    meas_model = replace(model, n_layers=n_meas) if n_meas != model.n_layers else model

    out: List[EffPoint] = []
    for T, t_ms in measurements:
        layout: Layout = layout_of(T)
        w = rank_work(meas_model, layout, T, B, cp=1)[0]
        t_s = t_ms / 1e3 - t_fixed_ms / 1e3
        if t_s <= 0:
            out.append(EffPoint(T, float("nan"), t_ms, w.flops, w.bytes_ew,
                                "t_fixed >= measured: unusable", n_meas,
                                bw_util, recompute))
            continue
        compute_s = t_s / recompute
        mem_s = w.bytes_ew / (hw.hbm_bw * bw_util)
        denom = hw.peak_flops * (compute_s - mem_s)
        if denom <= 0:
            out.append(EffPoint(T, float("nan"), t_ms, w.flops, w.bytes_ew,
                                "elementwise term exceeds total: check bw_util",
                                n_meas, bw_util, recompute))
            continue
        note = "" if n_meas == model.n_layers else \
            f"per-{n_meas}-layer measurement, scaled to L={model.n_layers}"
        out.append(EffPoint(T, w.flops / denom, t_ms, w.flops, w.bytes_ew,
                            note, n_meas, bw_util, recompute))
    return out


def fit_curve(points: Sequence[EffPoint], p0: Optional[Params] = None,
              restarts: int = 8) -> Tuple[Params, float]:
    """Fit ``(eff_inf, c_star, gamma)`` to the extracted points.

    The physical constants used to *solve* the points (``bw_util``,
    ``recompute``) are carried on the points themselves and inherited here.
    Without that, the returned ``Params`` would silently revert to defaults and
    the curve would then be evaluated under different physics than it was
    measured under -- a systematic error that looks like a bad fit.

    Robust in log space, and constrained so ``eff_inf`` stays a plausible
    fraction of peak -- an unconstrained fit will happily return ``eff_inf > 1``
    when the sweep is too narrow.
    """
    pts = [pt for pt in points if pt.eff == pt.eff and 0 < pt.eff < 20]
    if len(pts) < 3:
        raise ValueError(f"need >= 3 usable points, got {len(pts)}")
    base = p0 or Params(bw_util=pts[0].bw_util, recompute=pts[0].recompute)

    def obj(x):
        eff_inf = 1.0 / (1.0 + math.exp(-x[0])) * 1.5     # cap at 150% of peak
        c_star = math.exp(x[1])
        gamma = math.exp(x[2])
        total = 0.0
        for pt in pts:
            pred = eff_inf / (1.0 + (c_star / pt.chunk) ** gamma)
            r = math.log(max(pred, 1e-12)) - math.log(pt.eff)
            total += r * r
        return total

    x0 = [math.log(base.eff_inf / (1.5 - base.eff_inf)),
          math.log(base.c_star), math.log(base.gamma)]
    best, best_f = None, float("inf")
    for k in range(restarts):
        start = [x0[0] + (0.0 if k == 0 else 0.6 * ((k % 3) - 1)),
                 x0[1] + (0.0 if k == 0 else 0.5 * ((k % 5) - 2)),
                 x0[2] + (0.0 if k == 0 else 0.3 * ((k % 3) - 1))]
        x, f, _ = nelder_mead(obj, start, step=0.3)
        x, f, _ = nelder_mead(obj, x, step=0.05)
        if f < best_f:
            best, best_f = x, f
    eff_inf = 1.0 / (1.0 + math.exp(-best[0])) * 1.5
    return base.copy(eff_inf=eff_inf, c_star=math.exp(best[1]),
                     gamma=math.exp(best[2])), math.sqrt(best_f / len(pts))


def report(points: Sequence[EffPoint], p: Params, rmse_log: float) -> str:
    lines = ["  extracted points (no joint fitting):"]
    for pt in points:
        lines.append(pt.row())
    lines.append(f"  fitted: eff_inf={p.eff_inf:.3f} c_star={p.c_star:.0f} "
                 f"gamma={p.gamma:.2f}   rmse(log)={rmse_log * 100:.2f}%")
    if points and points[0].n_layers:
        lines.append(f"  measured module: {points[0].n_layers} layer(s); "
                     f"predictions below use each model's own depth")
    lines.append("  predicted eff at the CP=1..8 chunks of T=8192: " + ", ".join(
        f"c={8192 // n}:{efficiency(p, 8192 / n):.3f}" for n in (1, 2, 4, 8)))
    return "\n".join(lines)
