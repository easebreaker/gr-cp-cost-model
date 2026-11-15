"""Parameter fitting with no third-party dependency at all.

Why fit in log space: step times span 1-2 orders of magnitude across T and CP,
so a plain least-squares on milliseconds is dominated by the largest configs.
Why a robust loss: the interesting residuals (a kernel that behaves differently
at small chunks) should not drag the whole fit.

The optimizer is Nelder-Mead on transformed parameters, restarted from a few
seeds.  Five free parameters is small enough that this is both faster and more
predictable than anything fancier -- and it keeps the package importable on a
bare Python install, which matters when the thing you are predicting runs on a
cluster you only have a shell on.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .cost import Params, predict
from .presets import resolve_layout, resolve_model

# --------------------------------------------------------------------------
# Parameter transforms
# --------------------------------------------------------------------------
TRANSFORM = {
    "eff_inf": "logit",
    "overlap": "logit",
    "c_star": "log",
    "gamma": "log",
    "alpha": "log",
    "bw_comm": "log",
    "bw_util": "log",
    "recompute": "log",
}


def _to_raw(v: float, kind: str) -> float:
    if kind == "log":
        return math.log(max(v, 1e-30))
    if kind == "logit":
        v = min(max(v, 1e-6), 1 - 1e-6)
        return math.log(v / (1 - v))
    return v


def _from_raw(x: float, kind: str) -> float:
    if kind == "log":
        return math.exp(x)
    if kind == "logit":
        return 1.0 / (1.0 + math.exp(-x))
    return x


def pack(p: Params, names: Sequence[str]) -> List[float]:
    return [_to_raw(getattr(p, n), TRANSFORM.get(n, "log")) for n in names]


def unpack(x: Sequence[float], names: Sequence[str], base: Params) -> Params:
    kw = {n: _from_raw(float(v), TRANSFORM.get(n, "log")) for n, v in zip(names, x)}
    return base.copy(**kw)


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------
@dataclass
class Record:
    model: str
    layout: str
    T: int
    B: int
    cp: int
    scheme: str
    measured_ms: float
    t_fixed_ms: float = 0.0
    kernel: str = "default"
    tag: str = ""

    @property
    def group(self) -> str:
        """Unit for ranking metrics: one (model, kernel, T, B) sweep over CP."""
        return f"{self.model}|{self.kernel}|{self.T}|{self.B}"

    def describe(self) -> str:
        return (f"{self.model} {self.layout} T={self.T} B={self.B} cp={self.cp} "
                f"{self.scheme} -> {self.measured_ms:.1f} ms")


def _f(x, default=0.0):
    return default if x in (None, "", "nan") else float(x)


def load_csv(path: str) -> List[Record]:
    out: List[Record] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if not row.get("model") or row["model"].startswith("#"):
                continue
            if not row.get("step_ms"):        # unfilled template row
                continue
            out.append(
                Record(
                    model=row["model"].strip(),
                    layout=(row.get("layout") or "causal").strip(),
                    T=int(float(row["T"])),
                    B=int(float(row["B"])),
                    cp=int(float(row["cp"])),
                    scheme=(row.get("scheme") or "block").strip(),
                    measured_ms=float(row["step_ms"]),
                    t_fixed_ms=_f(row.get("t_fixed_ms")),
                    kernel=(row.get("kernel") or "default").strip(),
                    tag=(row.get("tag") or "").strip(),
                )
            )
    return out


def save_csv(records: Sequence[Record], path: str) -> str:
    cols = ["model", "layout", "T", "B", "cp", "scheme", "step_ms",
            "t_fixed_ms", "kernel", "tag"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in records:
            w.writerow([r.model, r.layout, r.T, r.B, r.cp, r.scheme,
                        f"{r.measured_ms:.4f}", f"{r.t_fixed_ms:.4f}",
                        r.kernel, r.tag])
    return path


def predict_record(rec: Record, p: Params, hw, detail: bool = False):
    model = resolve_model(rec.model)
    layout = resolve_layout(rec.layout, rec.T)
    return predict(model, layout, rec.T, rec.B, rec.cp, hw, p,
                   t_fixed_ms=rec.t_fixed_ms, scheme=rec.scheme, detail=detail)


# --------------------------------------------------------------------------
# Nelder-Mead
# --------------------------------------------------------------------------
def nelder_mead(
    f: Callable[[Sequence[float]], float],
    x0: Sequence[float],
    step: float = 0.3,
    max_iter: int = 4000,
    tol: float = 1e-10,
) -> Tuple[List[float], float, int]:
    n = len(x0)
    simplex: List[List[float]] = [list(x0)]
    for i in range(n):
        v = list(x0)
        v[i] += step
        simplex.append(v)
    vals = [f(v) for v in simplex]

    it = 0
    for it in range(max_iter):
        order = sorted(range(n + 1), key=lambda i: vals[i])
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) <= tol * (abs(vals[0]) + tol):
            break
        centroid = [sum(simplex[i][k] for i in range(n)) / n for k in range(n)]

        xr = [centroid[k] + (centroid[k] - simplex[-1][k]) for k in range(n)]
        fr = f(xr)
        if fr < vals[0]:
            xe = [centroid[k] + 2.0 * (centroid[k] - simplex[-1][k]) for k in range(n)]
            fe = f(xe)
            simplex[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            simplex[-1], vals[-1] = xr, fr
        else:
            xc = [centroid[k] + 0.5 * (simplex[-1][k] - centroid[k]) for k in range(n)]
            fc = f(xc)
            if fc < vals[-1]:
                simplex[-1], vals[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = [simplex[0][k] + 0.5 * (simplex[i][k] - simplex[0][k])
                                  for k in range(n)]
                    vals[i] = f(simplex[i])
    best = min(range(n + 1), key=lambda i: vals[i])
    return simplex[best], vals[best], it


def robust_loss(resid: Sequence[float], f_scale: float = 0.05) -> float:
    """Pseudo-Huber on log residuals: quadratic near zero, linear in the tail."""
    total = 0.0
    for r in resid:
        u = r / f_scale
        total += 2.0 * f_scale * f_scale * (math.sqrt(1.0 + u * u) - 1.0)
    return total


# --------------------------------------------------------------------------
# Fitting driver
# --------------------------------------------------------------------------
@dataclass
class FitResult:
    params: Params
    names: List[str]
    loss: float
    rmse_log: float
    n: int
    restarts: int = 0
    residual: Optional[List[float]] = None

    def report(self, label: str = "fit") -> str:
        return (f"{label}: n={self.n} params={len(self.names)} "
                f"rmse(log)={self.rmse_log * 100:.2f}%  "
                f"{self.params.describe(self.names)}")


def residuals(records: Sequence[Record], p: Params, hw) -> List[float]:
    out = []
    for rec in records:
        pred = predict_record(rec, p, hw)
        out.append(math.log(max(pred, 1e-9)) - math.log(rec.measured_ms))
    return out


def fit(
    records: Sequence[Record],
    hw,
    params0: Optional[Params] = None,
    free: Sequence[str] = Params.FREE,
    restarts: int = 4,
    f_scale: float = 0.05,
    seed: int = 0,
) -> FitResult:
    base = params0 or Params()
    names = list(free)
    rnd = random.Random(seed)

    def obj(x):
        return robust_loss(residuals(records, unpack(x, names, base), hw), f_scale)

    x0 = pack(base, names)
    starts = [x0]
    for _ in range(restarts):
        starts.append([v + rnd.gauss(0.0, 0.5) for v in x0])

    best_x, best_f = None, float("inf")
    for x0 in starts:
        x, fv, _ = nelder_mead(obj, x0)
        x, fv, _ = nelder_mead(obj, x, step=0.05)
        if fv < best_f:
            best_x, best_f = x, fv

    p = unpack(best_x, names, base)
    r = residuals(records, p, hw)
    rmse = math.sqrt(sum(v * v for v in r) / len(r))
    return FitResult(params=p, names=names, loss=best_f, rmse_log=rmse,
                     n=len(records), restarts=restarts, residual=r)


# --------------------------------------------------------------------------
# Identifiability: which parameters can this data set even determine?
# --------------------------------------------------------------------------
def sensitivity(records: Sequence[Record], p: Params, hw,
                names: Optional[Sequence[str]] = None,
                delta: float = 0.01) -> Dict[str, float]:
    """Percent change in mean predicted time per 1% change in each parameter.

    A parameter whose sensitivity is far below the measurement noise is not
    identifiable: no amount of fitting will pin it down and it will soak up
    whatever residual is left.  Print this *before* trusting a fit.
    """
    names = list(names or p.FREE)
    base = [predict_record(r, p, hw) for r in records]
    mean0 = sum(base) / len(base)
    out = {}
    for n in names:
        kw = {n: getattr(p, n) * (1.0 + delta)}
        pert = [predict_record(r, p.copy(**kw), hw) for r in records]
        mean1 = sum(pert) / len(pert)
        out[n] = (mean1 / mean0 - 1.0) / delta * 100.0
    return out
