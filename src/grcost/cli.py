"""Command line entry point: ``python -m grcost <command>``.

Commands
--------
  info                     what the package knows about models and hardware
  break-even               the ring-attention c* rule, no measurement needed
  predict                  one configuration
  sweep                    a CP sweep with a per-config breakdown
  fit                      fit parameters from a CSV of measurements
  validate                 score a fitted parameter set on another CSV
  sql                      print a ready-to-fill calibration table
  manifest                 the NPU measurement plan, with gates and cost
  bridge                   derive the timepred framework's kappa per CP degree
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from .cost import Ablation, Params, amdahl_ceiling, predict, scaling_factor, speedup
from .fit import load_csv, predict_record, save_csv, sensitivity, Record
from .hardware import PRESETS, get as get_hw
from .memory import check_table, fit_memory
from .metrics import by_group, report
from .presets import MODELS, mtgr_layout, resolve_layout, resolve_model
from .spec import crossover_report
from .work import flops_per_example


def _params(args) -> Params:
    p = Params()
    for name in ("eff_inf", "c_star", "gamma", "alpha", "bw_comm", "overlap",
                 "bw_util", "recompute"):
        v = getattr(args, name, None)
        if v is not None:
            p = p.copy(**{name: v})
    return p


def cmd_info(args):
    print("models")
    for name in sorted(MODELS):
        m = MODELS[name]
        print(f"  {name:<13} d={m.d:<5} layers={m.n_layers:<3} "
              f"dense={m.dense_flops_per_token():.3e} FLOPs/token")
        print(f"                {crossover_report(m)}")
    print("\nhardware")
    for name in sorted(PRESETS):
        print("  " + PRESETS[name].report().replace("\n", "\n  "))
    print("\nlayouts: causal, mtgr, mtgr_70, mtgr_95, mtgr_block, mtgr:<cand>:<static_share>")


def cmd_break_even(args):
    hw = get_hw(args.hardware)
    print(hw.report())
    print()
    for T in (2048, 8192, 32768, 131072):
        c = hw.break_even_chunk()
        print(f"  T={T:<7} -> CP beyond ~{max(1, int(T / c))} has a chunk below "
              f"the break-even and cannot hide its transfer")


def cmd_predict(args):
    m = resolve_model(args.model)
    lay = resolve_layout(args.layout, args.T)
    hw = get_hw(args.hardware)
    ms, bd = predict(m, lay, args.T, args.B, args.cp, hw, _params(args),
                     t_fixed_ms=args.t_fixed, scheme=args.scheme, detail=True)
    print(f"{args.model} layout={args.layout} T={args.T} B={args.B} cp={args.cp} "
          f"scheme={args.scheme} on {hw.name}")
    print(f"  step            {bd['step_ms']:9.2f} ms")
    print(f"    fixed         {bd['fixed_ms']:9.2f} ms")
    print(f"    compute       {bd['compute_ms']:9.2f} ms  (eff={bd['eff']:.3f}, "
          f"chunk={bd['chunk']:.0f})")
    print(f"    elementwise   {bd['elementwise_ms']:9.2f} ms")
    print(f"    exposed comm  {bd['comm_ms']:9.2f} ms")
    print(f"    critical rank {bd['crit_rank']}  imbalance={bd['imbalance']:.3f}")
    print(f"    MFU           {bd['mfus']:.1%}")
    print(f"    amdahl ceiling@CP{args.cp} = {bd['amdahl_ceiling']:.2f}x")


def cmd_sweep(args):
    m = resolve_model(args.model)
    hw = get_hw(args.hardware)
    p = _params(args)
    print(f"{args.model} T={args.T} B={args.B} on {hw.name}")
    print(f"  {'cp':>3} {'chunk':>7} {'step_ms':>9} {'speedup':>8} {'scaling':>8} "
          f"{'eff':>6} {'imbal':>6} {'comm_ms':>8}")
    base = None
    for cp in [int(x) for x in args.cps.split(",")]:
        lay = resolve_layout(args.layout, args.T)
        ms, bd = predict(m, lay, args.T, args.B, cp, hw, p,
                         t_fixed_ms=args.t_fixed, scheme=args.scheme, detail=True)
        if cp == 1:
            base = ms
        print(f"  {cp:>3} {bd['chunk']:>7.0f} {ms:>9.2f} {base / ms:>8.2f} "
              f"{scaling_factor(m, lay, args.T, args.B, p, hw, cp, args.t_fixed):>8.2f} "
              f"{bd['eff']:>6.3f} {bd['imbalance']:>6.2f} {bd['comm_ms']:>8.2f}")


def cmd_fit(args):
    recs = load_csv(args.csv)
    hw = get_hw(args.hardware)
    from .fit import fit as do_fit

    res = do_fit(recs, hw, restarts=args.restarts, seed=args.seed)
    print(res.report("fit"))
    print()
    print("  identifiability (% per +1% parameter), vs measurement noise "
          f"~{args.noise:.1f}%:")
    for k, v in sensitivity(recs, res.params, hw).items():
        print(f"    {k:<10}{v:+7.2f}%")
    if args.save:
        with open(args.save, "w", encoding="utf-8") as fh:
            fh.write(res.params.to_json())
        print(f"\n  params written to {args.save}")


def cmd_validate(args):
    recs = load_csv(args.csv)
    hw = get_hw(args.hardware)
    p = Params.from_json(open(args.params, encoding="utf-8").read()) if args.params else _params(args)
    pred = [predict_record(r, p, hw) for r in recs]
    meas = [r.measured_ms for r in recs]
    print("  " + report(pred, meas, [r.group for r in recs]).row("all"))
    for key in ("model", "T", "cp"):
        groups = [str(getattr(r, key)) for r in recs]
        for g, rep in by_group(pred, meas, groups).items():
            print("  " + rep.row(f"  {key}={g}"))
    worst = sorted(zip(pred, recs), key=lambda t: -abs(t[0] / t[1].measured_ms - 1))[:5]
    print("\n  worst residuals:")
    for pr, r in worst:
        print(f"    {r.describe():<62} pred={pr:8.2f} err={(pr / r.measured_ms - 1) * 100:+6.1f}%")


def cmd_sql(args):
    """Print a ready-to-fill calibration table (the 12-job design)."""
    recs: List[Record] = []
    for T in (2048, 4096):
        for cp in (1, 2, 4, 8):
            recs.append(Record("mtgr-large", "mtgr", T, 32, cp, "block", 0.0))
    recs.append(Record("mtgr-large", "mtgr", 4096, 32, 4, "block", 0.0, kernel="triton"))
    for B in (16, 64):
        recs.append(Record("mtgr-large", "mtgr", 4096, B, 4, "block", 0.0))
    recs.append(Record("mtgr-large", "mtgr", 2048, 32, 2, "balanced", 0.0))

    out = args.out or "data/calibration.csv"
    import csv

    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "layout", "T", "B", "cp", "scheme", "step_ms",
                    "t_fixed_ms", "kernel", "tag"])
        for r in recs:
            # step_ms and t_fixed_ms are left blank on purpose: step_ms is what
            # you measure, t_fixed_ms is what you profile separately.
            w.writerow([r.model, r.layout, r.T, r.B, r.cp, r.scheme, "",
                        "", r.kernel, "calibration"])
    print(f"  wrote {len(recs)} rows to {out}")
    print("  fill in step_ms (and t_fixed_ms, which must be profiled separately,")
    print("  not fitted) then: python -m grcost fit --csv " + out)


def cmd_manifest(args):
    from .manifest import build_plan, summary, write_manifest

    jobs = build_plan(args.model, args.T, args.B, args.cp_max)
    print(summary(jobs))
    if args.out:
        write_manifest(jobs, args.out)
        print(f"\nmanifest written to {args.out}")


def cmd_bridge(args):
    from .bridge import decompose, kappa_drift
    from .presets import resolve_layout

    m = resolve_model(args.model)
    lay = resolve_layout(args.layout, args.T)
    hw = get_hw(args.hardware)
    p = _params(args)
    if args.drift:
        for T in (2048, 4096, 8192):
            print(kappa_drift(m, resolve_layout(args.layout, T), T, args.B, hw, p,
                               args.t_fixed))
            print()
        return
    print(f"{args.model} layout={args.layout} T={args.T} B={args.B} on {hw.name}")
    for n in [int(x) for x in args.cps.split(",")]:
        d = decompose(m, lay, args.T, args.B, n, hw, p,
                      t_fixed_ms=args.t_fixed, scheme=args.scheme,
                      shard_mode=args.shard_mode)
        print("  " + d.row())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="grcost", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--model", default="mtgr-large")
        sp.add_argument("--layout", default="mtgr")
        sp.add_argument("--T", type=int, default=4096)
        sp.add_argument("--B", type=int, default=32)
        sp.add_argument("--cp", type=int, default=2)
        sp.add_argument("--scheme", default="block", choices=["block", "balanced", "striped"])
        sp.add_argument("--hardware", default="h100")
        sp.add_argument("--t-fixed", dest="t_fixed", type=float, default=0.0)
        for name in ("eff_inf", "c_star", "gamma", "alpha", "bw_comm", "overlap",
                     "bw_util", "recompute"):
            sp.add_argument(f"--{name.replace('_', '-')}", dest=name, type=float, default=None)

    sp = sub.add_parser("info", help="models, hardware, layouts")
    sp.set_defaults(func=cmd_info)

    sp = sub.add_parser("break-even", help="ring-attention c* rule")
    sp.add_argument("--hardware", default="h100")
    sp.set_defaults(func=cmd_break_even)

    sp = sub.add_parser("predict", help="one configuration")
    common(sp)
    sp.set_defaults(func=cmd_predict)

    sp = sub.add_parser("sweep", help="a CP sweep")
    common(sp)
    sp.add_argument("--cps", default="1,2,4,8")
    sp.set_defaults(func=cmd_sweep)

    for name, fn in (("fit", cmd_fit), ("validate", cmd_validate)):
        sp = sub.add_parser(name)
        sp.add_argument("--csv", required=True)
        sp.add_argument("--hardware", default="h100")
        if name == "fit":
            sp.add_argument("--restarts", type=int, default=4)
            sp.add_argument("--seed", type=int, default=0)
            sp.add_argument("--noise", type=float, default=3.0,
                            help="per-measurement noise in %%, for the identifiability table")
            sp.add_argument("--save", default=None)
        else:
            sp.add_argument("--params", default=None)
            for nm in ("eff_inf", "c_star", "gamma", "alpha", "bw_comm", "overlap",
                       "bw_util", "recompute"):
                sp.add_argument(f"--{nm.replace('_', '-')}", dest=nm, type=float, default=None)
        sp.set_defaults(func=fn)

    sp = sub.add_parser("sql", help="print a fillable calibration table")
    sp.add_argument("--out", default=None)
    sp.set_defaults(func=cmd_sql)

    sp = sub.add_parser("manifest", help="the NPU measurement plan")
    sp.add_argument("--out", default=None)
    sp.add_argument("--model", default="mtgr-large")
    sp.add_argument("--T", type=int, default=4096)
    sp.add_argument("--B", type=int, default=32)
    sp.add_argument("--cp-max", dest="cp_max", type=int, default=8)
    sp.set_defaults(func=cmd_manifest)

    sp = sub.add_parser("bridge", help="derive timepred's kappa")
    sp.add_argument("--model", default="mtgr-large")
    sp.add_argument("--layout", default="mtgr")
    sp.add_argument("--T", type=int, default=4096)
    sp.add_argument("--B", type=int, default=32)
    sp.add_argument("--cps", default="2,4,8")
    sp.add_argument("--scheme", default="block")
    sp.add_argument("--shard-mode", dest="shard_mode", default="full",
                    choices=["full", "attn_only"])
    sp.add_argument("--hardware", default="h100")
    sp.add_argument("--t-fixed", dest="t_fixed", type=float, default=0.0)
    sp.add_argument("--drift", action="store_true")
    for nm in ("eff_inf", "c_star", "gamma", "alpha", "bw_comm", "overlap",
               "bw_util", "recompute"):
        sp.add_argument(f"--{nm.replace('_', '-')}", dest=nm, type=float, default=None)
    sp.set_defaults(func=cmd_bridge)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
