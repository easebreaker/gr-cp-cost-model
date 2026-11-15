"""The composition question, end to end.

Question: with a fixed cluster and a fixed deadline, is it better to buy quality
with sequence length, with model capacity, or with more data -- and does CP ever
pay for itself?

The answer is constrained three ways, and each constraint changes the answer:

  1. throughput      tokens/GPU-hour falls with CP (it is never free)
  2. memory          at a given cp only some T fit at all
  3. quality         returns are power-law, so every lever saturates

Run:  PYTHONPATH=src python3 scripts/frontier_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grcost.cost import Params, predict  # noqa: E402
from grcost.frontier import (  # noqa: E402
    Point, activation_gb, ascii_curve, best, breakeven_alpha, fits,
    gpu_hours_for_tokens, length_gain, data_loss, longer_only, longer_with_cp,
    pareto, prior_memory_fit, quality_per_gpuh, required_alpha, summary,
    throughput_tokens_per_hour, tokens_for_quality,
)
from grcost.hardware import ASCEND_910B as HW  # noqa: E402
from grcost.presets import mtgr_layout, resolve_model  # noqa: E402
from grcost.quality import demo_law  # noqa: E402

P = Params(eff_inf=0.42, c_star=1800, gamma=1.1, alpha=25e-6, bw_comm=120e9,
           overlap=0.5, bw_util=0.55, recompute=1.0)
MODEL = resolve_model("mtgr-large")
B, NGpu = 32, 64
T_FIXED = 6.0


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    rule("1. IS CP THROUGHPUT-NEUTRAL?   (it is not -- this is the whole point)")
    T = 4096
    lay = mtgr_layout(T)
    print(f"  {MODEL.name}, T={T}, B={B}, {NGpu} NPUs")
    print(f"  {'cp':>3} {'step ms':>9} {'speedup':>8} {'tok/GPUh':>11} {'vs cp=1':>8}")
    base_th = None
    t1 = predict(MODEL, lay, T, B, 1, HW, P, t_fixed_ms=T_FIXED)
    for cp in (1, 2, 4, 8):
        ms = predict(MODEL, lay, T, B, cp, HW, P, t_fixed_ms=T_FIXED)
        th = throughput_tokens_per_hour(MODEL, lay, T, B, cp, HW, P, NGpu, T_FIXED)
        base_th = base_th or th
        print(f"  {cp:>3} {ms:>9.2f} {t1 / ms:>8.2f} {th / NGpu / 1e6:>10.3f}M "
              f"{th / base_th:>8.2f}x")
    print()
    print("  GPU-hours per token = cp * t_step / (B*T*3600). The product cannot")
    print("  drop below cp=1 even with perfect speedup, so CP always spends")
    print("  throughput to buy length. It is a feasibility lever, not a speed one.")
    print()
    print("  For a fixed deadline this is not a loss in *time* -- it is a loss in")
    print("  *tokens*, hence in quality, and that is what the next section costs.")

    rule("2. WHAT ACTUALLY FITS   (memory is what forces CP in the first place)")
    mem = prior_memory_fit(MODEL, HW)
    print(f"  prior: {mem.a_bytes / 1024:.0f} KiB/token on {HW.name} "
          f"({HW.mem_gb:.0f} GB, minus {mem.M_static_gb:.0f} GB static)")
    print(f"  {'T':>7} " + "".join(f"{'cp=' + str(c):>12}" for c in (1, 2, 4, 8)))
    for T in (2048, 4096, 8192, 16384, 32768, 65536):
        row = f"  {T:>7} "
        for cp in (1, 2, 4, 8):
            gb = activation_gb(mem, T, B, cp)
            row += f"{gb:>9.1f}GB" + ("  " if fits(mem, T, B, cp, HW) else " x")
        print(row)
    print("  'x' = does not fit. Note the shape: T_max grows roughly with cp,")
    print("  but the achievable cp is capped by the cluster, not by memory.")

    rule("3. FIXED DEADLINE: same GPU-hours, only quality differs")
    law = demo_law()
    HOURS = 72.0
    print(f"  {NGpu} NPUs x {HOURS:.0f} h = {NGpu * HOURS:.0f} GPU-hours for every row")
    print()
    print(f"  quality law: {law.describe()}")
    if not law.sourced:
        print("  WARNING: placeholder quality law -- every Q below is a shape, "
              "not a result.")
    print()

    m0 = MODEL.param_count()
    lengths = [1024, 2048, 4096, 8192, 16384, 32768]
    rows = longer_only(MODEL, mtgr_layout, lengths, m0, law, NGpu, HOURS, B, HW,
                       P, T_FIXED, mem=mem)
    rows += longer_with_cp(
        MODEL, mtgr_layout,
        [(4096, 2), (4096, 4), (8192, 2), (8192, 4), (8192, 8),
         (16384, 4), (16384, 8), (32768, 8), (32768, 16)],
        m0, law, NGpu, HOURS, B, HW, P, T_FIXED, mem=mem)
    print(summary(rows, law))

    b = best(rows)
    print()
    print(f"  best feasible: {b.label} T={b.T} cp={b.cp}  Q={b.quality:.4f} "
          f"(D={b.tokens / 1e9:.2f}B, {b.mem_gb:.1f} GB activations)")
    oom = [q for q in rows if not q.fits]
    if oom:
        top = max(oom, key=lambda q: q.quality)
        print(f"  best infeasible: {top.label} T={top.T} cp={top.cp} "
              f"Q={top.quality:.4f} -- needs {top.mem_gb:.0f} GB, OOM. "
              f"This is exactly the trap without a memory model.")

    rule("4. TWO DIFFERENT QUESTIONS, EASY TO CONFUSE")
    print("  (a) At the SAME T, is cp>1 better than cp=1?")
    print("      Never. Same length, fewer tokens -> strictly worse.")
    same = {}
    for q in rows:
        same.setdefault(q.T, {})[q.cp] = q
    print(f"      {'T':>7} " + "".join(f"{'cp=' + str(c):>10}" for c in (1, 2, 4, 8)))
    for T in sorted(same):
        row = f"      {T:>7} "
        for cp in (1, 2, 4, 8):
            q = same[T].get(cp)
            row += f"{q.quality:>10.4f}" if q and q.fits else f"{'-':>10}"
        print(row)
    print()
    print("  (b) At PROPORTIONALLY LONGER T (cp=2 at 2T vs cp=1 at T), is it")
    print("      better? That depends on the length exponent alpha, and has a")
    print("      threshold:")
    print()
    th_lines = required_alpha(MODEL, mtgr_layout, law, 4096, B, m0, NGpu, HOURS,
                              [2, 4, 8], HW, P, T_FIXED)
    for line in th_lines:
        print(line)
    print(f"      measured alpha of the placeholder law: {law.alpha:.3f}")
    print()
    print("  So CP beats no-CP *at equal length-scaled quality* as soon as")
    print("  alpha > ~0.16 for an 8x length jump. But that is not the same as")
    print("  being globally optimal: a moderate T at cp=1 can still win, because")
    print("  the length lever saturates before the memory wall is reached.")
    print()
    print("  Three regimes, and CP only matters in the middle one:")
    print("    - short deadline    : tokens are scare -> cp=1 wins outright")
    print("    - memory wall hit   : only cp>1 fits at all -> CP is mandatory,")
    print("                          and the question is which cp, not whether")
    print("    - length saturates  : buying more context with CP loses, because")
    print("                          alpha has already delivered its return")
    print()
    print("  Which regime you are in is decided by two measurable numbers: the")
    print("  memory wall (T_max per cp) and alpha. That is what makes this a")
    print("  research question with an answer rather than a preference.")

    rule("5. COST FRONTIER: GPU-hours to reach a target quality")
    print(f"  {'target Q':>9}  {'config':<22} {'T':>7} {'cp':>3} {'D(B)':>9} "
          f"{'GPUh':>9} {'fit':>4}")
    frontier: list[Point] = []
    for target in (0.660, 0.680, 0.700, 0.706, 0.710, 0.714):
        cands = []
        for T in lengths:
            for cp in (1, 2, 4, 8):
                if not fits(mem, T, B, cp, HW):
                    continue
                D = tokens_for_quality(law, T, m0, target)
                if D != D or D == float("inf") or D > 1e14:
                    continue
                lay = mtgr_layout(T)
                gh = gpu_hours_for_tokens(MODEL, lay, T, B, cp, D, HW, P, T_FIXED)
                cands.append((gh, T, cp, D))
        if not cands:
            print(f"  {target:>9.3f}  unreachable within the feasible set")
            continue
        gh, T, cp, D = min(cands)
        name = "longer" + (f" + cp{cp}" if cp > 1 else " only")
        print(f"  {target:>9.3f}  {name:<22} {T:>7} {cp:>3} {D / 1e9:>9.2f} "
              f"{gh:>9.1f} {'yes':>4}")
        frontier.append(Point(name, T, cp, m0, 0.0, D, target, gh, 0.0))

    if len(frontier) >= 2:
        print()
        print("  frontier (quality vs GPU-hours):")
        print(ascii_curve(frontier))
        print()
        print("  The knee is where the length lever stops paying: past it, more")
        print("  GPU-hours buy almost nothing, and the only way forward is a")
        print("  bigger model or more data -- which this cost model does not price.")

    rule("6. WHAT WOULD MAKE THIS A RESULT RATHER THAN A SHAPE")
    print("  1. Fit the quality law on real points that vary L, m AND D.")
    print("     A grid that varies only L is not identifiable (fit_quality says so),")
    print("     and the frontier is then driven by arbitrary constants.")
    print("  2. Replace the prior memory model with two measured T_max points at")
    print("     different per-rank chunk lengths (memory.fit_memory).")
    print("  3. Test the additive-deficit assumption: it says the three levers")
    print("     substitute one-for-one in deficit space. Fit with and without the")
    print("     cross terms; if the frontier ordering flips, the composition is")
    print("     not identified and the question needs a different design.")
    print("  4. Report the frontier as a *ranking* of strategies, not as a")
    print("     predicted quality. The absolute Q is the part most likely wrong.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
