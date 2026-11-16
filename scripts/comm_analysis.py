"""Why "bw_comm depends on message size" is a symptom, and what to model instead.

Run:  PYTHONPATH=src python3 scripts/comm_analysis.py

Three sections:

1. What a piecewise fit would have to learn, and why it does not transfer.
2. The step-count model, and the break-even message size it produces.
3. What this means for the two collectives in the plan (per-layer at full size
   vs per-bucket), which is where the latency actually bites.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grcost.comm import (CommModel, allgather_volume_per_rank,
                         diagnose_size_dependence, steps_recursive_doubling,
                         steps_ring)


def rule(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


def main() -> int:
    alpha, bw = 15e-6, 149e9
    m = CommModel(alpha=alpha, bw=bw)

    rule("1. THE APPARENT BANDWIDTH IS A CURVE, NOT A CONSTANT")
    print(f"  alpha = {alpha * 1e6:.0f} us/step, BW_inf = {bw / 1e9:.0f} GB/s, "
          f"n = 8 (steps = {steps_recursive_doubling(8)})")
    print()
    print(f"  {'message':>10} {'time':>10} {'BW_app':>10} {'fraction of peak':>18}")
    for v in (1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24, 1 << 26, 1 << 28, 1 << 30):
        t = m.time(8, v)
        bwa = m.apparent_bw(8, v)
        print(f"  {v / 1e6:>9.2f}M {t * 1e6:>9.1f}us {bwa / 1e9:>9.1f}G "
              f"{bwa / bw:>17.0%}")
    print()
    print("  A piecewise fit would put a breakpoint wherever the eye sees a bend.")
    print("  But the bend is at V* = S*alpha*BW, and S changes with the rank count:")
    for n in (2, 4, 8, 16):
        vstar = m.break_even_bytes(n)
        print(f"    n={n:<3} steps={steps_recursive_doubling(n):<2} "
              f"V* = {vstar / 1e6:>7.2f} MB    ring算法 steps={steps_ring(n):<2} "
              f"V* = {m.__class__(alpha=alpha, bw=bw, policy='ring').break_even_bytes(n) / 1e6:>7.2f} MB")
    print()
    print("  So any breakpoint fitted at n=8 is in the wrong place at n=2 or n=16.")
    print("  That is the concrete reason not to fit segments.")

    rule("2. ONE ALPHA AND ONE BW, RECOVERED FROM A WHOLE CURVE")
    pts = [(v, m.time(8, v)) for v in (1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9)]
    print(diagnose_size_dependence(pts, 8))

    rule("3. WHERE THE LATENCY ACTUALLY BITES")
    print("  Per-collective payload decides the regime. Two implementations of the")
    print("  same CP: one collective per sharded layer, versus one per bucket.")
    print()
    B, T, d, layers, dtype = 32, 4096, 768, 15, 2
    kv_per_token = 2 * dtype * d
    total = B * T * kv_per_token * layers          # bytes shipped over the step
    print(f"  step total: B*T*2*d*L = {B}*{T}*2*{d}*{layers} = "
          f"{total / 1e9:.2f} GB")
    print()
    print(f"  {'case':<34} {'per collective':>16} {'rounds':>7} "
          f"{'regime':>10} {'t (ms)':>9}")
    for label, rounds, n_buckets in (
            ("one collective per layer", layers, 1),
            ("one per layer per bucket", layers * B, B),
            ("one per layer, 8 buckets", layers * 8, 8)):
        per = total / rounds
        cm = m
        t = cm.time(8, total, rounds=rounds)
        reg = "latency" if per < cm.break_even_bytes(8) else "bandwidth"
        print(f"  {label:<34} {per / 1e6:>15.2f}M {rounds:>7} {reg:>10} "
              f"{t * 1e3:>9.2f}")
    print()
    print("  Read the last column: splitting the same byte total into more")
    print("  collectives raises the cost, because latency is paid per collective.")
    print("  This is the term a jagged implementation multiplies without meaning to.")
    print()
    print("  For reference, per-rank volume under all-gather:")
    print(f"    (n-1)/n * V = {allgather_volume_per_rank(total, 8) / 1e9:.2f} GB "
          f"per rank at n=8")
    print("  Ring and all-gather move the same bytes; they differ in step count")
    print("  and peak buffer, not in bandwidth. Modelling them as different")
    print("  volumes is a common error.")

    rule("4. WHAT TO MEASURE")
    print("  1. Sweep message size across V* on both sides (roughly 64 KB .. 64 MB")
    print("     for a node-local link). Without points below V*, alpha is not")
    print("     identifiable and the fit will trade it against BW.")
    print("  2. Repeat at two rank counts. The shift in V* verifies the step-count")
    print("     model instead of assuming it.")
    print("  3. Record the number of collectives per step separately from the")
    print("     payload. If the implementation is jagged, that count is the term")
    print("     that will be wrong.")
    print("  4. Run diagnose_size_dependence() on the curve. If the held-out error")
    print("     is large, the implementation changes algorithm mid-range -- model")
    print("     that as v_switch (one integer), not as breakpoints.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
