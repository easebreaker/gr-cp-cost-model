"""The CP design question: alignment vs communication, and what the two cost.

Run:  PYTHONPATH=src python3 scripts/tiling_analysis.py

The output is the argument, in three parts:

1. tile quantisation is a *discrete* cost that only appears at odd CP degrees;
2. head-parallel CP forbids odd degrees unless the head count allows them, and
   HSTU's head counts are 2..4 -- so at H=3 the only two-card option is the
   worst-aligned one;
3. measure which effect dominates, and note that the fixpipe path is what makes
   the score matrix cost up to a handful of HBM passes rather than one.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grcost.cost import Params, predict
from grcost.hardware import ASCEND_910C, H100_SXM
from grcost.presets import mtgr_layout, resolve_model
from grcost.tiling import (
    DEFAULT_BM, alignment_factor, alignment_table, scheme_tension, usefulness,
)


def rule(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


def main() -> int:
    T, B = 4096, 32
    model = resolve_model("mtgr-large")
    p = Params(eff_inf=0.42, c_star=1800, gamma=1.1, alpha=15e-6, bw_comm=149e9,
               overlap=0.0, bw_util=0.55, recompute=1.0, t_emb_ms=3.0)

    rule("1. A DISCRETE COST THAT ONLY APPEARS AT ODD CP DEGREES")
    print(f"  T = {T}, BM = {DEFAULT_BM}; c = T/cp")
    print(f"  {'cp':>3} {'c':>9} {'ceil(c/BM)':>11} {'useful':>8} {'work x':>8}")
    for cp, c, use, mult in alignment_table(T, [1, 2, 3, 4, 5, 6, 8, 12, 16]):
        flag = "" if abs(use - 1.0) < 1e-9 else "   <- partial last tile"
        print(f"  {cp:>3} {c:>9.1f} {int(-(-c // DEFAULT_BM)):>11} "
              f"{use:>7.1%} {mult:>8.3f}{flag}")
    print()
    print("  Every power-of-two CP degree is exactly aligned, because T is a power")
    print("  of two. That is why this cost is invisible in most CP work: everyone")
    print("  tries degrees that divide evenly. It only bites when something else")
    print("  forces an odd degree.")

    rule("2. THE CONSTRAINT: HEAD-PARALLEL CP NEEDS cp | H, AND H IS 2..4")
    for heads in (2, 3, 4):
        print(scheme_tension(heads, T))
        print()
    print("  H=4 behaves like an LLM (powers of two available). H=3 does not, and")
    print("  H=2 offers nothing above cp=2. So the awkward case is not exotic --")
    print("  it is one of the three head counts HSTU actually uses.")

    rule("3. WHICH EFFECT DOMINATES: ALIGNMENT, OR THE RING'S EXTRA STEPS?")
    print("  Compare, at equal cp, the alignment multiplier with the ring step")
    print("  count. Both are closed forms; neither needs a measurement.")
    print()
    print(f"  {'cp':>3} {'align x':>8} {'ring steps':>11} {'ring/rd':>8} "
          f"{'Ulysses?':>9}")
    from grcost.comm import steps_recursive_doubling, steps_ring
    for cp in (2, 3, 4, 5, 6, 8):
        al = alignment_factor(T / cp, T)
        sr = steps_ring(cp)
        sd = steps_recursive_doubling(cp)
        print(f"  {cp:>3} {al:>8.3f} {sr:>11} {sr / max(sd, 1):>8.1f}x "
              f"{'yes' if 3 % cp == 0 or cp == 1 else '-':>9}")
    print()
    print("  For H=3 the choice is between")
    print("    cp=3 via Ulysses : shorter collective, 3.1% wasted tiles")
    print("    cp=4 via ring    : no waste, but 3 ring steps instead of 2")
    print("  Which wins depends on alpha and the message size -- both are")
    print("  measurable, and the model says exactly where the crossover sits.")

    rule("4. THE SCORE MATRIX IS NOT ONE HBM PASS (fixpipe)")
    print("  The L^2 score matrix is produced by the Cube unit into L0C, then has")
    print("  to reach the Vector unit for silu/mask/gating.")
    print()
    for name, hops, hw in (("910B/910C  (DAV_2201, no L0C->UB)", 3.0, ASCEND_910C),
                           ("A5/950     (DAV_3510, L0C->UB direct)", 2.0, ASCEND_910C),
                           ("H100       (fused kernel, score stays on chip)", 0.0, H100_SXM)):
        a = predict(model, mtgr_layout(T), T, B, 4, hw, p.copy(fixpipe_hops=max(hops, 0.01)),
                    t_fixed_ms=4.0, detail=True)
        print(f"  {name:<46} ew = {a[1]['elementwise_ms']:6.2f} ms")
    print()
    print("  A note on the earlier claim that CP *amplifies* the fixpipe share:")
    print("  the elementwise term scales with pairs, the compute term with")
    print("  pairs * eff, so the ratio is 0.708 * eff(c) -- it FALLS with CP.")
    print("  What CP does amplify is the *contention*: the received K/V also")
    print("  lands in HBM, at the same bandwidth the fixpipe write-back uses.")

    rule("5. WHAT TO MEASURE, AND THE ORDER")
    print("  S1.3  the same kernel at cp = 3, 5, 6 vs 4, 8. If the gap matches")
    print("        1/useful(c), tile quantisation is real and the closed form is")
    print("        right. If not, eff(c) already contains it -- do not double count.")
    print("  M3.3  the score matrix written out and read back, alone. This pins")
    print("        fixpipe_hops; 3 is a prior, not a fact.")
    print("  M5.*  the collective sweep at two rank counts already decides the")
    print("        alpha vs step-count trade-off that part 3 depends on.")
    print()
    print("  All three are single-card or fine-grained jobs. None needs a working")
    print("  CP implementation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
