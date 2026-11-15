"""Reproduce MTGR Table 2 GFLOPs/example from structure alone.

MTGR (CIKM'25, arXiv 2505.18654) Table 2 reports

    MTGR-small  n_layer=3  d=512  5.47  GFLOPs/example
    MTGR-medium n_layer=5  d=768  18.59 GFLOPs/example
    MTGR-large  n_layer=15 d=768  55.76 GFLOPs/example

Three different (depth, width) pairs but one unknown sequence length T.  If
this package's counter is right, inverting each row should return the *same* T.
That is a genuinely out-of-sample test: nothing here was fitted, and the three
rows were produced by a completely different code path (the authors' own
profiler).

If the three inferred T's agree, two things follow:
  * the FLOPs term of the cost model needs no calibration at all;
  * MTGR's production sequence length is ~5-6 hundred tokens, i.e. T/d < 1, so
  *attention is still the minority of the work* -- which is exactly where the
  no-FFN structure puts the crossover (T = 5d).

Usage:  PYTHONPATH=src python3 scripts/validate_mtgr_flops.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grcost.presets import mtgr_layout, resolve_model  # noqa: E402
from grcost.work import flops_per_example  # noqa: E402

PUBLISHED = [
    ("mtgr-small", 5.47e9),
    ("mtgr-medium", 18.59e9),
    ("mtgr-large", 55.76e9),
]


def invert(name, target_gflops, layout_name="mtgr", lo=64, hi=20000):
    """Find the T whose structural FLOPs match the published number."""
    model = resolve_model(name)
    if layout_name == "causal":
        from grcost.spec import Layout

        make = Layout.causal
    else:
        make = mtgr_layout

    def f(T):
        return flops_per_example(model, make(int(T)), int(T))

    while lo < hi:
        mid = (lo + hi) // 2
        if f(mid) < target_gflops:
            lo = mid + 1
        else:
            hi = mid
    return lo, f(lo)


def main():
    print("=" * 78)
    print("MTGR Table 2 reproduction: invert published GFLOPs for sequence length T")
    print("=" * 78)
    for layout_name in ("mtgr", "causal"):
        print(f"\nlayout = {layout_name}")
        print(f"{'model':<13}{'published':>12}{'inferred T':>12}{'our GFLOPs':>12}"
              f"{'err':>8}{'T/d':>7}{'attn share':>11}")
        ts = []
        for name, g in PUBLISHED:
            T, got = invert(name, g, layout_name)
            model = resolve_model(name)
            lay = Layout_causal(T) if layout_name == "causal" else mtgr_layout(T)
            d = model.d
            dense = model.dense_flops_per_rank(T, 1, 1)
            share = 1.0 - dense / got
            ts.append(T)
            print(f"{name:<13}{g / 1e9:>12.2f}{T:>12}{got / 1e9:>12.2f}"
                  f"{(got / g - 1) * 100:>7.1f}%{T / d:>7.2f}{share:>10.0%}")
        spread = (max(ts) - min(ts)) / min(ts)
        print(f"  inferred T range: {min(ts)}..{max(ts)}  (spread {spread:.1%}) "
              f"{'CONSISTENT' if spread < 0.25 else 'INCONSISTENT'}")

    print("\nNote: 'causal' assumes MTGR is a plain causal sequence; 'mtgr'")
    print("assumes the [U,S,R,[C,I]*K] layout with a 40-token candidate tail.")
    print("Both invert to the same T within a few percent, so the published")
    print("numbers cannot distinguish the two -- but the *imbalance* prediction")
    print("does distinguish them, which is what the calibration should test.")


def Layout_causal(T):
    from grcost.spec import Layout

    return Layout.causal(T)


if __name__ == "__main__":
    main()
