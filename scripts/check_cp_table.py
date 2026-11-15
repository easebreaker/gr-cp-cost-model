"""Read the only two public CP-for-GR data points and say what they imply.

Sources
-------
* Meta, "Scaling Generative Recommendations with Context Parallelism on
  Hierarchical Sequential Transducers" (RecSys'25, arXiv 2508.04711)
    Table 1: CP -> max sequence length {1:3072, 2:4096, 4:7168, 8:16384}
    Table 2: scaling factor {ideal 2.00, DDP only 1.6-1.7, DDP+CP(2) 1.33,
             DDP+CP(2, 2x batch) 1.55}
    and the headroom study: fusing the segmented attention kernels is expected
    to lift 1.33 -> ~1.5-1.6.

* NVIDIA recsys-examples HSTU benchmark (T = 2S + C, S=2K -> T=4099 at B=32).

Nothing below requires our own measurements; the point is to derive, from
numbers other people published, what the model must absorb.

Usage:  PYTHONPATH=src python3 scripts/check_cp_table.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grcost.cost import Params, amdahl_ceiling
from grcost.hardware import H100_SXM, PRESETS
from grcost.memory import check_table, fit_memory
from grcost.presets import hstu_layout, mtgr_layout, resolve_model
from grcost.work import flops_per_example

T_MAX = {1: 3072, 2: 4096, 4: 7168, 8: 16384}
SCALING = {"ideal": 2.00, "DDP only": 1.65, "DDP+CP(cp=2)": 1.33,
           "DDP+CP(cp=2, 2x batch)": 1.55}
HEADROOM = (1.50, 1.60)


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main():
    rule("1. THE RING-ATTENTION BREAK-EVEN CHUNK  (zero measurement required)")
    for name in ("h100-sxm", "a100-80g", "ascend-910b"):
        print(PRESETS[name].report())
        print()

    rule("2. IS THE PUBLISHED MEMORY TABLE CONSISTENT WITH 1/CP?")
    print(check_table(T_MAX))
    print()
    print("  Fit a 1/CP model to CP=1 only, then predict the rest:")
    fit = fit_memory([(T_MAX[1], 1, 32, 80.0)])
    for cp in sorted(T_MAX):
        pred = fit.max_seq_len(32, cp, 80.0)
        act = T_MAX[cp]
        print(f"    CP={cp}: predicted {pred:>6}  actual {act:>6}  "
              f"over-prediction {pred / act:>5.2f}x")
    print()
    print("  Interpretation: chunks saturate near 2-3K tokens no matter how many")
    print("  GPUs you add. A pure 1/CP activation model over-predicts CP=8 by")
    print("  ~1.5x, so the memory model needs an explicit delta(cp) term -- or")
    print("  the sweep was bounded by something other than activation memory.")
    print("  Either way this is the *first* thing to re-measure on your hardware.")

    rule("3. WHERE THE 1.33x COMES FROM  (and where it cannot come from)")
    print("  A 'scaling factor' in these tables is QPS(cp GPUs)/QPS(1 GPU) at")
    print("  fixed batch, ideal = cp. So 1.33x at CP=2 means the per-rank step")
    print("  time fell only to 1/1.33 = 75% of the single-GPU time, not 50%.")
    for label, sf in SCALING.items():
        print(f"    {label:<24} scaling={sf:.2f}x  =>  t_cp/t_1 = {1 / sf:.2f}  "
              f"(ideal {1 / 2:.2f})")
    print()
    print("  Amdahl ceiling: even with zero communication, CP=2 can only speed")
    print("  up the attention fraction. For MTGR at its published operating point")
    print("  (T inverted from GFLOPs, see validate_mtgr_flops.py):")
    m = resolve_model("mtgr-large")
    for T in (520, 2048, 4096):
        lay = mtgr_layout(T)
        tot = flops_per_example(m, lay, T)
        dense = m.dense_flops_per_rank(T, 1, 1)
        print(f"    T={T:<5} attention share={1 - dense / tot:5.1%}  "
              f"=> Amdahl ceiling @CP=2 = {amdahl_ceiling(m, lay, T, 2):.2f}x, "
              f"@CP=8 = {amdahl_ceiling(m, lay, T, 8):.2f}x")
    print()
    print("  Cross-check these two published numbers against each other:")
    print("    - GFLOPs inversion says T ~ 520  ->  attention share ~ 18%")
    print("      -> Amdahl ceiling at CP=2 is 1.10x")
    print("    - the scaling table says 1.33x at CP=2")
    print("  1.33 > 1.10, so the two cannot both be true at face value. Solving")
    ceil = 1.33
    a_needed = 2 * (1 - 1 / ceil)
    print(f"  for the needed share: a = {a_needed:.0%} of work in attention, i.e.")
    m2 = resolve_model("mtgr-large")
    T_needed = None
    for T in range(256, 8192, 16):
        lay = mtgr_layout(T)
        tot = flops_per_example(m2, lay, T)
        if 1 - m2.dense_flops_per_rank(T, 1, 1) / tot >= a_needed:
            T_needed = T
            break
    print(f"  T ~ {T_needed} tokens -- about 4x longer than the inversion implies.")
    print()
    print("  Three ways to resolve it, in increasing cost:")
    print("    a) the GFLOPs table counts something else (per candidate? B>1?),")
    print("    b) the 1.33x was measured with a different batch/global-batch rule,")
    print("    c) the mask really does put ~50% of the work in attention at T~520")
    print("       (i.e. the static block is bigger than our 85% assumption).")
    print("  The cheap discriminator: measure the attention fraction directly by")
    print("  timing one layer with attention replaced by a no-op, or read it off")
    print("  NCU. One job settles it, and it also pins c_star.")

    rule("4. WHAT THE MODEL MUST ABSORB  (the calibration target)")
    p = Params()
    print("  Two knobs carry all of the unexplained variance:")
    print(f"    eff(c): a {p.eff_inf:.2f} asymptote and a half-efficiency chunk")
    print(f"            c_star ~ {p.c_star:.0f} tokens, which sits right between the")
    print(f"            per-rank chunks of CP=1 and CP=8 in the table above.")
    print("    comm  : exposed alpha + V/bw, with V ~ n_layers * B * T * 2 * d * 2")
    print("            bytes -- the n_layers factor is easy to forget and is")
    print("            worth a factor of 15 in the estimate.")
    print()
    print("  Everything else (FLOPs, pairs, imbalance) is derived, not fitted.")
    print("  A 5-parameter fit on 12 jobs therefore has real degrees of freedom")
    print("  left for validation -- see scripts/demo_synthetic.py.")


if __name__ == "__main__":
    main()
