"""End-to-end demonstration: calibrate on 12 points, validate on 32 unseen ones.

This script is the executable version of the argument in ``docs/``:

1. generate measurements from a known "truth" (stand-in for real profiling),
2. fit only on a deliberately narrow calibration set,
3. validate on held-out sequence lengths, a held-out CP degree, and two
   other models,
4. run the ablation study that shows each structural ingredient matters,
5. write figures.

Everything downstream of step 1 uses only what you would have on real hardware
if you ran the 12 calibration jobs.

Usage:  PYTHONPATH=src python3 scripts/demo_synthetic.py [--out out/]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import random
from math import exp

from grcost.cost import Ablation, Params, amdahl_ceiling, predict, scaling_factor, speedup
from grcost.fit import Record, fit, sensitivity
from grcost.hardware import H100_SXM as HW
from grcost.memory import check_table, fit_memory
from grcost.metrics import by_group, report
from grcost.presets import mtgr_layout, resolve_layout, resolve_model

rng = random.Random(7)

# ---------------------------------------------------------------- ground truth
TRUE = Params(
    eff_inf=0.42,
    c_star=1800.0,
    gamma=1.1,
    alpha=25e-6,        # exposed, i.e. after overlap
    bw_comm=250e9,      # exposed
    overlap=0.45,       # pinned from a comm-only microbenchmark, never fitted
    bw_util=0.55,       # pinned from a bandwidth microbenchmark
    recompute=4.0 / 3.0,
)
T_FIXED_MS = 6.0          # embedding + all2all + optimizer + dataloader, profiled once
NOISE = 0.03              # 3% multiplicative measurement noise


def measure(model_name, layout_name, T, B, cp, scheme="block", kernel="cutlass"):
    m = resolve_model(model_name)
    lay = resolve_layout(layout_name, T)
    t = predict(m, lay, T, B, cp, HW, TRUE, t_fixed_ms=T_FIXED_MS, scheme=scheme)
    return Record(
        model=model_name, layout=layout_name, T=T, B=B, cp=cp, scheme=scheme,
        measured_ms=float(t * exp(rng.gauss(0.0, NOISE))),
        t_fixed_ms=T_FIXED_MS, kernel=kernel,
    )


def calibration_set():
    """12 jobs: one model, one kernel, two lengths, four CP degrees.

    Deliberately narrow.  Two extra jobs swap in a second kernel and a second
    batch size, which is what lets the fit separate the efficiency curve from
    the collective cost.
    """
    recs = []
    for T in (2048, 4096):
        for cp in (1, 2, 4, 8):
            recs.append(measure("mtgr-large", "mtgr", T, 32, cp))
    recs.append(measure("mtgr-large", "mtgr", 4096, 32, 4, kernel="triton"))
    for B in (16, 64):
        recs.append(measure("mtgr-large", "mtgr", 4096, B, 4))
    recs.append(measure("mtgr-large", "mtgr", 2048, 32, 2, scheme="balanced"))
    return recs


def validation_set():
    """40 jobs that the fit never sees: other lengths, other shapes, other models."""
    recs = []
    for model, layout in (("mtgr-large", "mtgr"), ("mtgr-medium", "mtgr"),
                          ("hstu-base", "causal")):
        for T in (1024, 3072, 6144, 8192):
            for cp in ((1, 4) if model == "hstu-base" else (1, 2, 4, 8)):
                recs.append(measure(model, layout, T, 32, cp))
    return recs


# ---------------------------------------------------------------------- report
def fmt_params(p: Params, names) -> str:
    rows = ["  parameter      truth        fitted     ratio"]
    for n in names:
        t, f = getattr(TRUE, n), getattr(p, n)
        rows.append(f"  {n:<12} {t:>10.4g} {f:>12.4g} {f / t:>9.2f}x")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("=" * 78)
    print("1. CALIBRATE  (12 jobs, ~2 GPU-hours)")
    print("=" * 78)
    cal = calibration_set()
    print(f"  {len(cal)} records, T in {{2048,4096}}, cp in {{1,2,4,8}}")
    res = fit(cal, HW, restarts=6, seed=1)
    print(fmt_params(res.params, res.names))
    print(f"  in-sample: {res.report()}")

    print()
    print("=" * 78)
    print("2. VALIDATE  (held-out jobs)")
    print("=" * 78)
    val = validation_set()
    pred = [predict(resolve_model(r.model), resolve_layout(r.layout, r.T), r.T, r.B,
                    r.cp, HW, res.params, t_fixed_ms=r.t_fixed_ms, scheme=r.scheme)
            for r in val]
    meas = [r.measured_ms for r in val]
    groups = [r.group for r in val]
    overall = report(pred, meas, groups)
    print("  " + overall.row("held-out (all)"))

    by_model = [r.model for r in val]
    for name, rep in by_group(pred, meas, by_model).items():
        print("  " + rep.row(f"  {name}"))

    print("  -- by CP degree --")
    by_cp = [f"cp={r.cp}" for r in val]
    for name, rep in by_group(pred, meas, by_cp).items():
        print("  " + rep.row(f"  {name}"))

    print("  -- by sequence length --")
    by_t = [f"T={r.T}" for r in val]
    for name, rep in by_group(pred, meas, by_t).items():
        print("  " + rep.row(f"  {name}"))

    print("  -- ranking within each CP sweep (the metric that matters) --")
    sweep = [f"{r.model}|T={r.T}" for r in val]
    for name, rep in by_group(pred, meas, sweep, rank=True).items():
        print("  " + rep.ranking_row(f"  {name}"))

    print("  -- all validation lengths are unseen (none is 2048 or 4096) --")
    print(f"     T in {{1024, 3072, 6144, 8192}}; "
          f"cross-model transfer without re-calibration is the mtgr-medium and "
          f"hstu-base rows above")

    print()
    print("=" * 78)
    print("3. IDENTIFIABILITY  (which parameters this data set can even determine)")
    print("=" * 78)
    print("  % change in mean predicted time per +1% in each parameter.")
    print("  Compare against the 3% measurement noise: anything under ~0.3%")
    print("  is not identifiable from these jobs, whatever the optimizer says.")
    sens = sensitivity(cal, res.params, HW)
    for k, v in sens.items():
        flag = "  <- NOT identifiable" if abs(v) < 0.3 else ""
        print(f"    {k:<10}{v:+7.2f}%{flag}")

    print()
    print("=" * 78)
    print("4. ABLATION  (each structural ingredient switched off in turn)")
    print("=" * 78)
    variants = [
        ("full model", None),
        ("no mask topology", Ablation(layout_override="causal")),
        ("mean instead of max", Ablation(critical="mean")),
        ("no efficiency curve", Ablation(efficiency=False)),
        ("no communication", Ablation(comm=False)),
        ("no L^2 byte term", Ablation(elementwise=False)),
    ]
    print(f"  {'variant':<22}{'MAPE':>8}{'P90':>8}{'bias':>8}")
    for label, abl in variants:
        p2 = [predict(resolve_model(r.model), resolve_layout(r.layout, r.T), r.T, r.B,
                      r.cp, HW, res.params, t_fixed_ms=r.t_fixed_ms,
                      scheme=r.scheme, abl=abl) for r in val]
        rep = report(p2, meas, groups)
        print(f"  {label:<22}{rep.mape:>7.1f}%{rep.p90:>7.1f}%{rep.bias:>7.1f}%")

    print()
    print("=" * 78)
    print("5. WHAT THE MODEL SAYS  (the part that is not a number)")
    print("=" * 78)
    m = resolve_model("mtgr-large")
    for T in (1024, 4096, 8192):
        lay = mtgr_layout(T)
        ceil = amdahl_ceiling(m, lay, T, 8, B=32)
        line = f"  T={T:<6} amdahl ceiling @CP=8 = {ceil:.2f}x | predicted speedup:"
        for cp in (2, 4, 8):
            sp = speedup(m, lay, T, 32, res.params, HW, cp, T_FIXED_MS)
            line += f"  CP{cp}={sp:.2f}x"
        print(line)
    print("  (speedup = t(CP=1)/t(CP) = QPS(cp GPUs)/QPS(1 GPU). A published")
    print("   'scaling factor' of 1.33 at CP=2 means the per-rank step fell only")
    print("   to 0.75 of the single-GPU time, i.e. 67% parallel efficiency.)")

    print()
    print("  a larger batch restores CP efficiency (the fixed alpha amortises):")
    lay = mtgr_layout(4096)
    for B in (8, 16, 32, 64):
        sf = scaling_factor(m, lay, 4096, B, res.params, HW, 2, T_FIXED_MS)
        print(f"    B={B:<4} scaling(CP=2) = {sf:.2f}x  (ideal 2.00x, "
              f"efficiency {sf / 2:.0%})")

    print()
    print("  memory table from the RecSys'25 CP-on-HSTU paper:")
    print(check_table({1: 3072, 2: 4096, 4: 7168, 8: 16384}))

    if not args.no_figures:
        make_figures(res, val, pred, meas, args.out)


def make_figures(res, val, pred, meas, out):
    from grcost import plotting as pl

    try:
        pl.pred_vs_measured(pred, meas, os.path.join(out, "pred_vs_measured.png"),
                            title=f"held-out ({len(val)} configs), MAPE={report(pred, meas).mape:.1f}%")

        curves = {}
        m = resolve_model("mtgr-large")
        for T in (1024, 4096, 8192):
            lay = mtgr_layout(T)
            cps = [1, 2, 4, 8]
            curves[f"T={T}"] = (cps, [speedup(m, lay, T, 32, res.params, HW, cp, T_FIXED_MS)
                                       for cp in cps])
        pl.cp_scaling(curves, os.path.join(out, "cp_speedup.png"))

        pl.efficiency_curve(res.params, os.path.join(out, "efficiency.png"),
                            T=8192, title="fitted kernel efficiency vs chunk length")
        print(f"\n  figures written to {out}/")
    except Exception as e:  # pragma: no cover - figures are a nicety
        print(f"  (skipped figures: {e})")


if __name__ == "__main__":
    main()
