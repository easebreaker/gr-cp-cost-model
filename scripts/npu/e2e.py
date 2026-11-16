"""L2: end-to-end step time, and the compose path that skips CP entirely.

Two modes
---------
``--measure``  runs the real step at each (T, CP) and writes both schemas:
               the ``grcost`` calibration CSV *and* the ``timepred`` sample CSV.
               Needs a working CP implementation.

``--compose``  reads what ``micro.py`` and ``sweep.py`` already produced and
               predicts the whole CP curve without ever running CP. This is the
               path to take first: it costs no NPU time, and if it reproduces
               the measured points it is a much stronger result than a fit.

Usage
-----
    python scripts/npu/e2e.py --compose --sweep data/npu_sweep.csv \\
        --micro data/npu_micro.csv --out out/
    torchrun --nproc_per_node=4 scripts/npu/e2e.py --measure --cp 4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from harness import (  # noqa: E402
    Device, Timing, measure, print_environment, write_calibration,
    write_timepred_sample,
)


# --------------------------------------------------------------------------
# CSV reading
# --------------------------------------------------------------------------
def read_timings(path: str, job: Optional[str] = None) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [r for r in rows if job is None or r.get("job") == job]


def eff_measurements(sweep_csv: str) -> List[Tuple[int, float]]:
    """``(T, ms)`` pairs from the ``eff`` job of ``sweep.py``."""
    out = []
    for r in read_timings(sweep_csv, "eff"):
        label = r.get("label", "")
        if not label.startswith("layer T="):
            continue
        T = int(label.split("=")[1].split(" ")[0])
        out.append((T, float(r["median_ms"])))
    return sorted(out)


def layers_measured(sweep_csv: str) -> Optional[int]:
    """Depth of the timed module, from the ``layers`` column.

    Returns ``None`` when the column is absent (an older CSV), which the caller
    must treat as "unknown" rather than "1": guessing wrong here is a
    factor-of-N error in the extracted efficiency.
    """
    vals = {int(float(r["layers"])) for r in read_timings(sweep_csv, "eff")
            if r.get("layers") not in (None, "")}
    if not vals:
        return None
    if len(vals) > 1:
        raise SystemExit(f"mixed layer counts in {sweep_csv}: {sorted(vals)}")
    return vals.pop()


def collective_points(micro_csv: str) -> List[Tuple[str, int, float]]:
    """``(collective, bytes, ms)`` from the ``collective`` job."""
    out = []
    for r in read_timings(micro_csv, "collective"):
        label = r.get("label", "")
        if "KB" not in label:
            continue
        kind, rest = label.split(" ", 1)
        kb = int(rest.split("KB")[0])
        out.append((kind, kb * 1024, float(r["median_ms"])))
    return out


def fit_comm(points: List[Tuple[str, int, float]]) -> Tuple[float, float, str]:
    """Least-squares fit of ``t = alpha + bytes / bw`` per collective kind.

    Two unknowns, so this is a closed-form 2x2 solve. Reports the kind that was
    measured, because ``alpha`` and ``bw`` differ between all-gather, all-to-all
    and ring -- and the ring is what CP actually uses.
    """
    if len(points) < 2:
        return 0.0, 0.0, "no collective data"
    xs = [b for _, b, _ in points]
    ys = [t / 1e3 for _, _, t in points]      # seconds
    n = len(xs)
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    det = sxx * n - sx * sx
    if abs(det) < 1e-9:
        return 0.0, 0.0, "degenerate byte range"
    slope = (sxy * n - sx * sy) / det        # seconds per byte
    alpha = (sy - slope * sx) / n
    bw = 1.0 / slope if slope > 0 else 0.0
    kinds = sorted({k for k, _, _ in points})
    return alpha, bw, f"kinds={'+'.join(kinds)}"


# --------------------------------------------------------------------------
# compose
# --------------------------------------------------------------------------
def cmd_compose(args) -> int:
    from grcost.compose import extract_eff, fit_curve, report
    from grcost.cost import Params, attribution_report, predict
    from grcost.hardware import get as get_hw
    from grcost.presets import resolve_model, resolve_layout

    hw = get_hw(args.hardware)
    model = resolve_model(args.model)
    layout_name = args.layout

    meas = eff_measurements(args.sweep)
    if len(meas) < 3:
        print(f"need >=3 eff points in {args.sweep}; found {len(meas)}")
        print("run: python scripts/npu/sweep.py --only eff")
        return 2

    # --- layer granularity: the factor-of-N trap -------------------------
    n_meas = args.layers_measured or layers_measured(args.sweep)
    if n_meas is None:
        print("WARNING: the sweep CSV has no `layers` column. Assuming the timed")
        print("         module had model.n_layers layers. If it was a single")
        print("         block, pass --layers-measured 1 or the prediction will")
        print("         come out n_layers times too fast.")
        n_meas = model.n_layers
    if n_meas != model.n_layers:
        print(f"  timed module depth: {n_meas} layer(s); prediction target: "
              f"{model.n_layers} layer(s)")
        print(f"  -> eff is solved on {n_meas} layer(s) and applied to the full "
              f"model (a kernel property, so it transfers)")

    # --- t_fixed: required, not optional --------------------------------
    if args.t_fixed <= 0:
        print()
        print("WARNING: --t-fixed is 0. Everything that does not shard")
        print("         (embedding lookup, sparse all2all, optimizer, dataloader)")
        print("         is then missing from the baseline. Consequences:")
        print("           * absolute step time too low")
        print("           * speedup wrong in either direction, depending on which")
        print("             of the two errors dominates")
        print("         Measure it with `micro.py --only dispatch` (two independent")
        print("         routes: layer-count slope and empty-kernel slope).")

    base = Params(bw_util=args.bw_util, recompute=args.recompute,
                  overlap=args.overlap)
    pts = extract_eff(model, lambda T: resolve_layout(layout_name, T), meas, hw,
                      B=args.B, t_fixed_ms=args.t_fixed,
                      bw_util=args.bw_util, recompute=args.recompute,
                      layers_measured=n_meas)
    params, rmse = fit_curve(pts, p0=base)
    print(report(pts, params, rmse))

    alpha, bw, note = fit_comm(collective_points(args.micro))
    if bw:
        params = params.copy(alpha=alpha, bw_comm=bw)
        print(f"\n  collective fit: alpha={alpha * 1e6:.1f}us  bw={bw / 1e9:.1f} GB/s "
              f"({note})")
    else:
        print(f"\n  collective: {note} -- falling back to nominal link bandwidth")
        params = params.copy(alpha=params.alpha, bw_comm=hw.link_bw)

    cps = [1] + [int(x) for x in args.cps.split(",")]
    print("\n  composed CP prediction (no CP run used as input):")
    print(f"  {'CP':>3} {'chunk':>7} {'ms':>9} {'speedup':>8} {'eff':>6} "
          f"{'imbal':>6} {'comm_ms':>8}")
    base_ms = None
    for cp in cps:
        lay = resolve_layout(layout_name, args.T)
        ms, bd = predict(model, lay, args.T, args.B, cp, hw, params,
                         t_fixed_ms=args.t_fixed, scheme=args.scheme, detail=True)
        if cp == 1:
            base_ms = ms
        print(f"  {cp:>3} {bd['chunk']:>7.0f} {ms:>9.2f} {base_ms / ms:>8.2f} "
              f"{bd['eff']:>6.3f} {bd['imbalance']:>6.2f} {bd['comm_ms']:>8.2f}")

    # --- why the speedup is what it is ---------------------------------
    print("\n  where the speedup goes (each column is a share of the step):")
    print(attribution_report(model, resolve_layout(layout_name, args.T),
                             args.T, args.B, cps, hw, params,
                             t_fixed_ms=args.t_fixed, scheme=args.scheme))

    if args.check_csv:
        check_against(args.check_csv, model, hw, params, layout_name)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "params_composed.json")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(params.to_json())
    print(f"\n  params written to {path}")
    if args.write_calibration:
        for cp in [1] + [int(x) for x in args.cps.split(",")]:
            lay = resolve_layout(layout_name, args.T)
            ms = predict(model, lay, args.T, args.B, cp, hw, params,
                         t_fixed_ms=args.t_fixed, scheme=args.scheme)
            write_calibration(args.write_calibration, args.model, layout_name,
                              args.T, args.B, cp, ms, args.t_fixed,
                              args.scheme, kernel=args.kernel, tag="composed")
        print(f"  composed rows appended to {args.write_calibration}")
    return 0


def check_against(csv_path: str, model, hw, params, layout_name: str) -> None:
    """Score a composed parameter set against measured end-to-end rows.

    This is the check that catches a granularity mistake: if the parameters were
    composed from single-block timings but the CSV is full-model steps, the
    ratio comes out near ``1 / model.n_layers`` and is impossible to miss.
    """
    from grcost.fit import load_csv, predict_record
    from grcost.metrics import report

    recs = load_csv(csv_path)
    if not recs:
        print(f"\n  check: {csv_path} has no rows")
        return
    pred = [predict_record(r, params, hw) for r in recs]
    meas = [r.measured_ms for r in recs]
    print(f"\n  check against {len(recs)} measured rows in {csv_path}:")
    print("  " + report(pred, meas, [r.group for r in recs]).row("all"))
    ratios = sorted(p / m for p, m in zip(pred, meas))
    med = ratios[len(ratios) // 2]
    print(f"  median pred/measured = {med:.3f}")
    if 0.05 < med < 0.2:
        print(f"  >>> looks like a granularity error: ratio near "
              f"1/{model.n_layers} = {1 / model.n_layers:.3f}.")
        print(f"      Pass --layers-measured 1 if the sweep timed a single block,")
        print(f"      or re-run sweep.py --layers {model.n_layers}.")
    elif med > 3:
        print("  >>> prediction is far too slow: is t_fixed double counted, or")
        print("      was the sweep taken with a much smaller batch/model?")
    for r, p in sorted(zip(recs, pred), key=lambda t: -abs(t[1] / t[0].measured_ms - 1))[:5]:
        print(f"    {r.describe():<58} pred={p:8.2f} "
              f"err={(p / r.measured_ms - 1) * 100:+6.1f}%")


# --------------------------------------------------------------------------
# measure
# --------------------------------------------------------------------------
def cmd_measure(args) -> int:
    """Run the real step. Needs the caller's model; we do not ship one."""
    dev = Device(args.device)
    print_environment(dev)
    print()
    print("  This mode expects a step function you provide. Two options:")
    print("    1. set --step-module to a python file exposing `build_step(cfg)`")
    print("    2. drop your own loop in here -- the two writers are the contract")
    print()
    print(f"  would measure T={args.T} B={args.B} cp in "
          f"{[1] + [int(x) for x in args.cps.split(',')]}")
    print(f"  writing: {args.calibration} (grcost) and "
          f"{args.timepred} (timepred)")
    if not args.step_module:
        print("\n  no --step-module given: nothing executed")
        return 0

    sys.path.insert(0, os.path.dirname(os.path.abspath(args.step_module)))
    import importlib.util

    spec = importlib.util.spec_from_file_location("user_step", args.step_module)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    for cp in [1] + [int(x) for x in args.cps.split(",")]:
        cfg = {"T": args.T, "B": args.B, "cp": cp, "device": dev.kind,
               "dtype": args.dtype}
        step = mod.build_step(cfg)
        t = measure(step, dev, f"step T={args.T} cp={cp}", repeats=args.repeats)
        print(t.line())
        write_calibration(args.calibration, args.model, args.layout, args.T,
                          args.B, cp, t.median_ms, args.t_fixed, args.scheme,
                          kernel=args.kernel, tag="measured")
        write_timepred_sample(args.timepred, args.B, t.median_ms / 1e3,
                              t.peak_mem_bytes, cp, args.T)
    print(f"\n  written: {args.calibration}, {args.timepred}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["compose", "measure"])
    ap.add_argument("--sweep", default="data/npu_sweep.csv")
    ap.add_argument("--micro", default="data/npu_micro.csv")
    ap.add_argument("--out", default="out")
    ap.add_argument("--calibration", default="data/npu_calibration.csv")
    ap.add_argument("--timepred", default="data/timepred_samples.csv")
    ap.add_argument("--write-calibration", dest="write_calibration", default=None)
    ap.add_argument("--step-module", dest="step_module", default=None)
    ap.add_argument("--model", default="mtgr-large")
    ap.add_argument("--layout", default="mtgr")
    ap.add_argument("--hardware", default="ascend-910b")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--T", type=int, default=4096)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--cps", default="2,4,8")
    ap.add_argument("--scheme", default="block")
    ap.add_argument("--kernel", default="default")
    ap.add_argument("--t-fixed", dest="t_fixed", type=float, default=0.0)
    ap.add_argument("--bw-util", dest="bw_util", type=float, default=0.6)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--recompute", type=float, default=1.0)
    ap.add_argument("--layers-measured", dest="layers_measured", type=int,
                    default=None,
                    help="depth of the timed module (1 = single block). "
                         "Defaults to the `layers` column of --sweep.")
    ap.add_argument("--check-csv", dest="check_csv", default=None,
                    help="measured end-to-end rows to score the composition against")
    ap.add_argument("--repeats", type=int, default=15)
    args = ap.parse_args(argv)
    return cmd_compose(args) if args.mode == "compose" else cmd_measure(args)


if __name__ == "__main__":
    raise SystemExit(main())
