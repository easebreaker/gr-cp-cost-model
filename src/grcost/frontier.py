"""The cost x quality composition: which lever buys quality per GPU-hour.

Reserved extension point for the "cost model x scaling law" topic.  The pieces
are real; the quality numbers are placeholders (``quality.demo_law()``).

The one structural fact this module is built to expose
-----------------------------------------------------
**CP trades throughput for sequence length.  It is not throughput-neutral, and it
is never free.**

For ``P`` GPUs split into ``P/cp`` replicas of a sequence of length ``T``:

    tokens/hour        = (P/cp) * B * T / t_step(cp, T)
    tokens/GPU-hour    =      B * T / (cp * t_step(cp, T))
    GPU-hours for D    = D * cp * t_step(cp, T) / (B * T * 3600)      [P cancels]

The ``cp * t_step`` product is what matters.  Even if ``t_step(cp) = t_step(1)/cp``
exactly -- a perfect linear speedup -- the product is unchanged, so a token costs
the same as before and CP has bought *nothing* except the ability to hold a long
`T` in memory.  In reality ``t_step(cp) > t_step(1)/cp``, so CP makes each token
*more* expensive.

Therefore the only question CP can answer is: **is the quality return to a longer
sequence worth the throughput you give up to get it?**  That is a question about
``alpha`` (the length exponent), and this module makes the answer numeric.

Two budget framings, both of which matter and which give different answers:

* **Fixed deadline** (``P`` GPUs, ``H`` hours): every configuration costs the same
  GPU-hours, so quality is the only axis.  This is the production framing.
* **Fixed GPU-hours** (cost ceiling): ``P`` and ``H`` both float, so the frontier
  is ``(GPU-hours, quality)``.  This is the planning framing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .cost import Params, predict
from .memory import MemoryFit, estimate_act_bytes_per_token
from .quality import QualityLaw
from .spec import HstuModel, Layout
from .work import rank_work


# --------------------------------------------------------------------------
# One evaluated configuration
# --------------------------------------------------------------------------
@dataclass
class Point:
    label: str
    T: int
    cp: int
    params_m: float          # dense parameter count
    t_step_ms: float
    tokens: float            # token budget actually processed
    quality: float
    gpu_hours: float
    wall_hours: float
    fits: bool = True        # does this sequence fit in device memory?
    mem_gb: float = 0.0

    @property
    def tokens_per_gpu_hour(self) -> float:
        return self.tokens / self.gpu_hours if self.gpu_hours else 0.0

    def row(self) -> str:
        flag = "" if self.fits else "  OOM"
        return (f"  {self.label:<18} T={self.T:<7} cp={self.cp:<3} "
                f"m={self.params_m / 1e6:<7.1f}M  t={self.t_step_ms:8.2f}ms  "
                f"D={self.tokens / 1e9:7.2f}B  Q={self.quality:.4f}  "
                f"{self.gpu_hours:9.1f} GPUh{flag}")


# --------------------------------------------------------------------------
# Feasibility -- without this the frontier recommends configurations that OOM
# --------------------------------------------------------------------------
def prior_memory_fit(model, hw, ratio: float = 6.0, static_gb: float = 8.0,
                     dtype_bytes: int = 2) -> MemoryFit:
    """A memory model from the structural prior, not from measurements.

    Good enough to *exclude* impossible configurations, which is all feasibility
    needs. Replace with ``memory.fit_memory`` once you have two ``T_max``
    observations at different per-rank chunk lengths.
    """
    a = estimate_act_bytes_per_token(model, dtype_bytes=dtype_bytes, ratio=ratio)
    return MemoryFit(a_bytes=a, M_static_gb=static_gb, identifiable=False)


def fits(mem: MemoryFit, T: int, B: int, cp: int, hw) -> bool:
    return mem.max_seq_len(B, cp, hw.mem_gb) >= T


def activation_gb(mem: MemoryFit, T: int, B: int, cp: int) -> float:
    return mem.a_bytes * B * T / cp / (1024 ** 3)


# --------------------------------------------------------------------------
# Throughput and cost
# --------------------------------------------------------------------------
def throughput_tokens_per_hour(model, layout: Layout, T: int, B: int, cp: int,
                              hw, p: Params, ngpu: int,
                              t_fixed_ms: float = 0.0,
                              scheme: str = "block") -> float:
    """Tokens processed per wall-clock hour by the whole cluster."""
    ms = predict(model, layout, T, B, cp, hw, p, t_fixed_ms=t_fixed_ms, scheme=scheme)
    replicas = max(1, ngpu // cp)
    return replicas * B * T * 3600.0 / (ms / 1e3)


def gpu_hours_for_tokens(model, layout: Layout, T: int, B: int, cp: int,
                         tokens: float, hw, p: Params,
                         t_fixed_ms: float = 0.0,
                         scheme: str = "block") -> float:
    """GPU-hours to process ``tokens``.  Independent of the GPU count."""
    ms = predict(model, layout, T, B, cp, hw, p, t_fixed_ms=t_fixed_ms, scheme=scheme)
    return tokens * cp * (ms / 1e3) / (B * T * 3600.0)


def tokens_for_quality(law: QualityLaw, T: int, params_m: float,
                       target_q: float) -> float:
    """Invert the quality law for the token budget.  ``inf`` if unreachable."""
    deficit = law.q_inf - target_q
    if T > 0 and law.a_len:
        deficit -= law.a_len * (law.l_ref / T) ** law.alpha
    if params_m > 0 and law.a_size:
        deficit -= law.a_size * (law.m_ref / params_m) ** law.beta
    if deficit <= 0 or not law.a_data:
        return float("inf")
    return law.d_ref * (law.a_data / deficit) ** (1.0 / law.gamma)


# --------------------------------------------------------------------------
# The three levers
# --------------------------------------------------------------------------
def longer_only(model, layout_of, lengths: Sequence[int], m_fixed: float,
                law: QualityLaw, ngpu: int, wall_hours: float, B: int,
                hw, p: Params, t_fixed_ms: float = 0.0,
                scheme: str = "block", label: str = "longer only",
                mem: Optional[MemoryFit] = None) -> List[Point]:
    """Spend everything on sequence length, no CP."""
    out = []
    for T in lengths:
        lay = layout_of(T)
        ok = True if mem is None else fits(mem, T, B, 1, hw)
        th = throughput_tokens_per_hour(model, lay, T, B, 1, hw, p, ngpu,
                                        t_fixed_ms, scheme)
        D = th * wall_hours
        out.append(Point(label, T, 1, m_fixed,
                         predict(model, lay, T, B, 1, hw, p, t_fixed_ms, scheme),
                         D, law.quality(T, m_fixed, D),
                         wall_hours * ngpu, wall_hours,
                         fits=ok,
                         mem_gb=0.0 if mem is None else activation_gb(mem, T, B, 1)))
    return out


def bigger_only(model_of, m_sizes: Sequence[float], T_fixed: int, layout: Layout,
                law: QualityLaw, ngpu: int, wall_hours: float, B: int,
                hw, p: Params, t_fixed_ms: float = 0.0,
                scheme: str = "block", label: str = "bigger only") -> List[Point]:
    """Spend everything on capacity at a fixed length, no CP."""
    out = []
    for m in m_sizes:
        m_model = model_of(m)
        th = throughput_tokens_per_hour(m_model, layout, T_fixed, B, 1, hw, p,
                                        ngpu, t_fixed_ms, scheme)
        D = th * wall_hours
        out.append(Point(label, T_fixed, 1, m,
                         predict(m_model, layout, T_fixed, B, 1, hw, p,
                                 t_fixed_ms, scheme),
                         D, law.quality(T_fixed, m, D),
                         wall_hours * ngpu, wall_hours))
    return out


def longer_with_cp(model, layout_of, pairs: Sequence[Tuple[int, int]],
                   m_fixed: float, law: QualityLaw, ngpu: int,
                   wall_hours: float, B: int, hw, p: Params,
                   t_fixed_ms: float = 0.0, scheme: str = "block",
                   label: str = "longer + CP",
                   mem: Optional[MemoryFit] = None) -> List[Point]:
    """Spend the deadline on a longer sequence, paid for with CP."""
    out = []
    for T, cp in pairs:
        cp = min(cp, ngpu)
        lay = layout_of(T)
        ok = True if mem is None else fits(mem, T, B, cp, hw)
        th = throughput_tokens_per_hour(model, lay, T, B, cp, hw, p, ngpu,
                                        t_fixed_ms, scheme)
        D = th * wall_hours
        out.append(Point(f"{label} cp={cp}", T, cp, m_fixed,
                         predict(model, lay, T, B, cp, hw, p, t_fixed_ms, scheme),
                         D, law.quality(T, m_fixed, D),
                         wall_hours * ngpu, wall_hours,
                         fits=ok,
                         mem_gb=0.0 if mem is None else activation_gb(mem, T, B, cp)))
    return out


# --------------------------------------------------------------------------
# The break-even exponent: when does buying length with CP actually pay?
# --------------------------------------------------------------------------
def length_gain(law: QualityLaw, T_short: float, length_mult: float,
                params_m: float, alpha: float) -> float:
    """Quality gained by lengthening from ``T`` to ``T*length_mult`` at fixed data."""
    lw = law.copy(alpha=alpha)
    return (lw.quality(T_short * length_mult, params_m, 1e15)
            - lw.quality(T_short, params_m, 1e15))


def data_loss(law: QualityLaw, D_short: float, token_ratio: float,
              params_m: float) -> float:
    """Quality lost by having ``token_ratio < 1`` of the training tokens."""
    return (law.quality(1e9, params_m, D_short)
            - law.quality(1e9, params_m, D_short * token_ratio))


def breakeven_alpha(law: QualityLaw, T_short: float, D_short: float,
                    length_mult: float, token_ratio: float,
                    params_m: float = 1.0e9,
                    lo: float = 0.02, hi: float = 2.0,
                    samples: int = 400) -> Optional[float]:
    """The length exponent above which buying context with CP pays for itself.

    Buying ``length_mult`` more context costs a factor ``token_ratio`` of the
    token throughput, so

        length_gain(alpha)   >   data_loss

    Note the left side is **not monotone in ``alpha``**: it vanishes as
    ``alpha -> 0`` (no return to length) *and* as ``alpha -> infinity`` (returns
    already saturated at the current ``T``), peaking near
    ``alpha = -1/ln(l_ref/T)``.  So the threshold is the *first* crossing, and a
    plain bisection on ``[lo, hi]`` silently returns ``None`` when ``hi`` sits
    past the peak -- which is exactly the mistake this scan avoids.

    Interpretation: a threshold on an exponent you can estimate from a length
    sweep, i.e. the same shape of result as the break-even chunk ``c* = F/BW``.
    """
    loss = data_loss(law, D_short, token_ratio, params_m)
    prev_a, prev_g = lo, length_gain(law, T_short, length_mult, params_m, lo) - loss
    if prev_g >= 0:
        return lo
    step = (hi - lo) / samples
    for i in range(1, samples + 1):
        a = lo + step * i
        g = length_gain(law, T_short, length_mult, params_m, a) - loss
        if g >= 0:
            x0, x1 = prev_a, a
            for _ in range(60):
                mid = 0.5 * (x0 + x1)
                if (length_gain(law, T_short, length_mult, params_m, mid)
                        - loss) >= 0:
                    x1 = mid
                else:
                    x0 = mid
            return 0.5 * (x0 + x1)
        prev_a, prev_g = a, g
    return None


def required_alpha(model, layout_of, law: QualityLaw, T: int, B: int,
                   params_m: float, ngpu: int, wall_hours: float,
                   cps: Sequence[int], hw, p: Params,
                   t_fixed_ms: float = 0.0,
                   scheme: str = "block") -> List[str]:
    """For each CP degree: the throughput cost, and the ``alpha`` it demands."""
    base_th = throughput_tokens_per_hour(model, layout_of(T), T, B, 1, hw, p,
                                         ngpu, t_fixed_ms, scheme)
    lines = [f"  T={T}  B={B}  {ngpu} NPUs  {wall_hours:.0f} h"]
    lines.append(f"  {'cp':>3} {'tokens(B)':>10} {'tok ratio':>10} {'T_long':>8} "
                 f"{'alpha*':>8}")
    for cp in cps:
        T_long = T * cp
        th = throughput_tokens_per_hour(model, layout_of(T_long), T_long, B, cp,
                                        hw, p, ngpu, t_fixed_ms, scheme)
        ratio = th / base_th
        a_star = breakeven_alpha(law, T, base_th * wall_hours, cp, ratio, params_m)
        shown = "never" if a_star is None else f"{a_star:.3f}"
        lines.append(f"  {cp:>3} {th * wall_hours / 1e9:>9.2f} {ratio:>10.2f} "
                     f"{T_long:>8} {shown:>8}")
    return lines


# --------------------------------------------------------------------------
# Pareto
# --------------------------------------------------------------------------
def pareto(points: Sequence[Point], cost: str = "gpu_hours",
           only_feasible: bool = True) -> List[Point]:
    """Non-dominated points: no other point is both cheaper and better."""
    if only_feasible:
        points = [q for q in points if q.fits]
    key = (lambda q: q.gpu_hours) if cost == "gpu_hours" else (lambda q: q.wall_hours)
    ordered = sorted(points, key=lambda q: (key(q), -q.quality))
    out: List[Point] = []
    best = -math.inf
    for q in ordered:
        if q.quality > best + 1e-12:
            out.append(q)
            best = q.quality
    return out


def best(points: Sequence[Point], only_feasible: bool = True) -> Point:
    pool = [q for q in points if q.fits] if only_feasible else list(points)
    if not pool:
        raise ValueError("no feasible configuration")
    return max(pool, key=lambda q: q.quality)


def better_than(points: Sequence[Point], baseline: Point) -> List[Point]:
    return [q for q in points if q.quality > baseline.quality + 1e-9]


def quality_per_gpuh(points: Sequence[Point]) -> List[Tuple[str, float]]:
    """Marginal quality per GPU-hour relative to the cheapest configuration.

    The number that decides whether a lever is worth pulling: the deficit closed
    divided by the extra GPU-hours paid.
    """
    if not points:
        return []
    base = min(points, key=lambda q: q.gpu_hours)
    out = []
    for q in points:
        extra = q.gpu_hours - base.gpu_hours
        gain = q.quality - base.quality
        if extra > 1e-9:
            out.append((q.label + f" T={q.T}", gain / extra * 1e3))
        elif gain > 0:
            out.append((q.label + f" T={q.T}", math.inf))
    return out


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def ascii_curve(points: Sequence[Point], width: int = 56, height: int = 14,
                x: str = "gpu_hours") -> str:
    """Cheap terminal plot; the figures go through plotting.py."""
    if len(points) < 2:
        return "  (need >= 2 points)"
    xs = [getattr(q, x) for q in points]
    ys = [q.quality for q in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if x1 <= x0:
        x0, x1 = x0, x0 + 1.0
    if y1 <= y0:
        y0, y1 = y0 - 1e-3, y0 + 1e-3
    grid = [[" "] * width for _ in range(height)]
    for px, py in zip(xs, ys):
        cx = int((px - x0) / (x1 - x0) * (width - 1))
        cy = height - 1 - int((py - y0) / (y1 - y0) * (height - 1))
        grid[cy][cx] = "*"
    lines = [f"  quality ^ {y1:.4f}"]
    for r, row in enumerate(grid):
        lines.append("           | " + "".join(row))
    lines.append(f"           +" + "-" * width)
    lines.append(f"             {x0:.0f}{' ' * max(1, width - 12)}{x1:.0f}  ->  {x}")
    return "\n".join(lines)


def summary(points: Sequence[Point], law: QualityLaw) -> str:
    lines = [f"  quality law: {law.describe()}"]
    if not law.sourced:
        lines.append("  WARNING: the quality law is a placeholder. Every number "
                     "below is a shape, not a result.")
    lines.append("")
    lines.append(f"  {'config':<18} {'T':>7} {'cp':>3} {'m(M)':>7} "
                 f"{'step ms':>9} {'D(B)':>8} {'Q':>7} {'GPUh':>9} "
                 f"{'tok/GPUh':>9} {'act GB':>7}")
    for q in sorted(points, key=lambda q: (not q.fits, -q.quality)):
        lines.append(f"  {q.label:<18} {q.T:>7} {q.cp:>3} "
                     f"{q.params_m / 1e6:>7.1f} {q.t_step_ms:>9.2f} "
                     f"{q.tokens / 1e9:>8.2f} {q.quality:>7.4f} "
                     f"{q.gpu_hours:>9.1f} {q.tokens_per_gpu_hour / 1e9:>9.3f} "
                     f"{q.mem_gb:>7.1f}" + ("" if q.fits else "  <= OOM"))
    return "\n".join(lines)
