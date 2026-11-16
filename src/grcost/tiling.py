"""Tile quantisation: the discrete part of CP that a continuous model cannot see.

Why this is a separate module
-----------------------------
``eff(c)`` describes how efficiency falls as the per-rank chunk shrinks.  That is
a *continuous* effect.  But a CP degree also has to satisfy a *discrete* one: the
chunk has to be cut into whole tiles, and the last tile is usually partial.

    tiles(c)      = ceil(c / BM)
    useful        = c / (tiles(c) * BM)  <= 1

For ``c`` a multiple of ``BM`` this is exactly 1.  For ``c = T/cp`` with ``T`` a
power of two and ``cp`` a power of two it is also 1 -- which is why the effect is
invisible in most LLM work, where every CP degree anyone tries divides the
sequence evenly.  It stops being invisible the moment a constraint forces an
awkward ``cp``.

The constraint that does that here
----------------------------------
Head-parallel context parallelism (Ulysses) splits attention across ranks by
head, so it requires ``cp | H``.  HSTU uses ``H = 2..4`` heads, **not** powers of
two.  At ``H = 3`` the only multi-card choice is ``cp = 3``, and

    c = 4096 / 3 = 1365.33   ->  ceil(1365.33 / 128) = 11 tiles
                             ->  usefulness = 1365.33 / 1408 = 97.0%

So the scheme with the *cheapest* communication can be the one with the *worst*
tile alignment, and there is no CP degree that avoids both.  That tension is the
design question, and it is answerable from geometry alone.

Ring-style CP has no head constraint, so it can use any ``cp`` -- but then the
ring's ``cp-1`` sequential steps make every odd degree expensive.  The two
constraints therefore push in opposite directions:

    ring : any cp, but latency ~ (cp-1) steps
    ulysses : cp must divide H, but latency ~ O(1) collective
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

DEFAULT_BM = 128     # query-direction tile height (a kernel property, not a model one)


def tiles(chunk: float, bm: int = DEFAULT_BM) -> int:
    return max(1, int(math.ceil(chunk / bm)))


def usefulness(chunk: float, bm: int = DEFAULT_BM) -> float:
    """Fraction of the launched query tiles that carries real work."""
    if chunk <= 0:
        return 0.0
    return chunk / (tiles(chunk, bm) * bm)


def waste(chunk: float, bm: int = DEFAULT_BM) -> float:
    return 1.0 - usefulness(chunk, bm)


def key_tiles(total: float, bn: int = DEFAULT_BM) -> int:
    return max(1, int(math.ceil(total / bn)))


def launch_tiles(chunk: float, total: float,
                 bm: int = DEFAULT_BM, bn: int = DEFAULT_BM) -> int:
    """Tiles the kernel actually launches for one rank."""
    return tiles(chunk, bm) * key_tiles(total, bn)


def useful_tiles(chunk: float, total: float,
                 bm: int = DEFAULT_BM, bn: int = DEFAULT_BM) -> float:
    """Tiles that would be needed with perfect packing (fractional)."""
    return (chunk / bm) * (total / bn)


def alignment_factor(chunk: float, total: float,
                     bm: int = DEFAULT_BM, bn: int = DEFAULT_BM) -> float:
    """Work multiplier: 1.0 when aligned, >1 when the last tile is partial.

    This multiplies the *attention* work, not the dense projections -- a partial
    tile wastes matrix-multiply lanes, not elementwise traffic.
    """
    n_launch = launch_tiles(chunk, total, bm, bn)
    n_useful = useful_tiles(chunk, total, bm, bn)
    if n_launch <= 0 or n_useful <= 0:
        return 1.0
    return n_launch / n_useful


# --------------------------------------------------------------------------
# Which CP degrees are even admissible
# --------------------------------------------------------------------------
def ulysses_admissible(heads: int, cp_max: int = 16) -> List[int]:
    """Head-parallel CP requires cp to divide the head count."""
    if heads < 1:
        return [1]
    return [cp for cp in range(1, min(cp_max, heads) + 1) if heads % cp == 0]


def ring_admissible(cp_max: int = 16) -> List[int]:
    """Ring CP has no head constraint, but pays cp-1 sequential steps."""
    return list(range(1, cp_max + 1))


def alignment_table(total: int, cps: Sequence[int],
                    bm: int = DEFAULT_BM) -> List[Tuple[int, float, float, float]]:
    """(cp, chunk, usefulness, work multiplier) for a set of degrees."""
    out = []
    for cp in cps:
        c = total / cp
        out.append((cp, c, usefulness(c, bm), alignment_factor(c, total, bm)))
    return out


def best_aligned(total: int, cps: Sequence[int], bm: int = DEFAULT_BM) -> int:
    """The admissible degree with the least quantisation waste."""
    return max(cps, key=lambda cp: usefulness(total / cp, bm))


def scheme_tension(heads: int, total: int, cp_max: int = 8,
                   bm: int = DEFAULT_BM) -> str:
    """For a given head count, which degrees each scheme allows and how they align.

    The point of the table is that the two admissible sets need not intersect in
    anything good: at ``H = 3`` the only head-parallel degree above one is 3, and
    it is the worst-aligned degree in the range.
    """
    u = ulysses_admissible(heads, cp_max)
    r = ring_admissible(cp_max)
    lines = [f"  heads = {heads}", ""]
    lines.append(f"  {'cp':>3} {'chunk':>9} {'useful':>8} {'x work':>8} "
                 f"{'Ulysses':>9} {'ring':>6}")
    for cp, c, use, mult in alignment_table(total, r, bm):
        lines.append(
            f"  {cp:>3} {c:>9.1f} {use:>7.1%} {mult:>8.3f} "
            f"{'yes' if cp in u else '-':>9} {'yes' if cp in r else '-':>6}")
    bu, br = (max(u), max(r))
    lines.append("")
    lines.append(f"  Ulysses 可用: {u}   -> 最大 {bu}")
    lines.append(f"  ring    可用: 任意    -> 但延迟 ~ (cp-1) 步")
    if len(u) > 1:
        uw = usefulness(total / max(u), bm)
        best_r = best_aligned(total, [c for c in r if c > 1] or [1], bm)
        bw_ = usefulness(total / best_r, bm)
        if uw < bw_ - 1e-9:
            lines.append(
                f"  >> 张力: Ulysses 最大的 {max(u)} 只有 {uw:.1%} 的 tile 利用率，"
                f"而 ring 可以选 {best_r}（{bw_:.1%}）。")
            lines.append("     通信更省的方案，恰是对齐更差的方案。")
    return "\n".join(lines)
