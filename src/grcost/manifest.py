"""The measurement manifest: what to run on the NPU, in what order, and why.

This module is pure Python and hardware-free, so the plan itself can be
reviewed, diffed and tested before anyone books machine time.

Design notes
------------
The single most useful property of the plan: **most of the calibration does not
require CP to be implemented at all.**

* ``eff(c)`` is a function of the per-rank chunk length ``c``. At ``CP=1``,
  ``c = T``, so sweeping ``T`` on one card *directly* traces the curve.
* Imbalance is a *work distribution* question. Run only rank ``i``'s query chunk
  and time it; the ratio of the slowest chunk to the average is the imbalance,
  no collectives involved.
* Collective cost is ``α + V/BW``, measurable with any collective primitive.

The model is a product of these three, so they can be measured independently and
composed. Only the final end-to-end block needs real CP, and that block is a
*validation*, not a source of calibration data.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

# --------------------------------------------------------------------------
# Job
# --------------------------------------------------------------------------
LAYER_NAMES = {
    "L0": "microbenchmark (constants)",
    "L1": "single-layer sweep (the curve)",
    "L2": "end-to-end (validation)",
}


@dataclass(frozen=True)
class Job:
    """One measurement the harness can execute."""

    id: str
    layer: str                 # L0 | L1 | L2
    purpose: str               # short tag, groups the analysis
    knobs: Dict[str, object]   # what the harness varies
    measures: str              # what to record
    decision: str              # which model parameter/conclusion it changes
    depends_on: str = ""       # gate: job that must pass first
    repeats: int = 5
    est_minutes: float = 1.5
    needs_cp: bool = False     # requires a working CP implementation

    @property
    def cost(self) -> float:
        return self.repeats * self.est_minutes

    def row(self) -> dict:
        d = {"id": self.id, "layer": self.layer, "purpose": self.purpose,
             "measures": self.measures, "decision": self.decision,
             "depends_on": self.depends_on, "repeats": self.repeats,
             "est_minutes": round(self.cost, 1), "needs_cp": int(self.needs_cp)}
        for k, v in self.knobs.items():
            d[f"knob_{k}"] = v
        return d


# --------------------------------------------------------------------------
# The plan
# --------------------------------------------------------------------------
def build_plan(model: str = "mtgr-large", T_ref: int = 4096,
               B: int = 32, cp_max: int = 8,
               lengths: Optional[Sequence[int]] = None) -> List[Job]:
    """The full measurement plan for one model on one node.

    ``lengths`` is the sweep grid for the efficiency curve.  It must span
    ``c*`` (the break-even chunk) on both sides -- otherwise ``c_star`` is not
    identifiable, which is the single most common way this calibration fails.
    """
    L = list(lengths or (512, 1024, 2048, 4096, 8192, 16384))
    cp_grid = [n for n in (2, 4, 8) if n <= cp_max]
    jobs: List[Job] = []

    # ---------------- L0: constants -------------------------------------
    jobs += [
        Job("M1.1", "L0", "noise_floor", {"op": "single_layer", "T": T_ref, "B": B},
            "10 consecutive repeats of one identical config",
            "the noise floor sigma that every later verdict is judged against",
            repeats=1, est_minutes=6.0),
        Job("M1.2", "L0", "noise_floor", {"op": "single_layer", "T": L[0], "B": B},
            "same, at the smallest length",
            "whether small-T verdicts are measurable at all",
            repeats=1, est_minutes=4.0),
        Job("M2.1", "L0", "dispatch", {"op": "empty_kernel", "n_kernels": [1, 8, 64, 256]},
            "time vs kernel count with zero work",
            "c_dispatch (us/kernel) -> t_fixed, and b decomposition",
            repeats=10, est_minutes=1.0),
        Job("M2.2", "L0", "dispatch", {"op": "same_kernel_repeat", "n": [1, 2, 4, 8]},
            "one kernel launched n times vs once",
            "separates per-launch cost from per-byte cost",
            repeats=10, est_minutes=1.0),
        Job("M2.3", "L0", "fixed_cost", {"op": "embedding_and_sparse_only",
                                         "T": T_ref, "B": B},
            "the step with the dense backbone stubbed out",
            "t_fixed by an independent route: embedding + sparse all2all + "
            "optimizer. Omitting it biases the CP=1 baseline, and therefore "
            "every speedup computed against that baseline",
            repeats=10, est_minutes=2.0),
        Job("M3.1", "L0", "hbm_bw", {"op": "stream_copy", "bytes": [1 << 26, 1 << 28]},
            "GB/s of a pure copy/scale at two sizes",
            "bw_util (pin it; do not fit it)",
            repeats=10, est_minutes=1.0),
        Job("M3.2", "L0", "compute_peak", {"op": "matmul", "shapes": [[4096, 4096, 4096]]},
            "achievable TFLOPS at the working dtype",
            "eff_inf ceiling; sanity-checks the nominal peak",
            repeats=10, est_minutes=1.0),
        Job("M4.1", "L0", "sync_cost", {"op": "sync_only", "n": [1, 10, 100]},
            "cost of torch.npu.synchronize() alone",
            "t_sync, and whether the timing protocol is sound",
            repeats=20, est_minutes=0.5),
    ]

    # collective cost curve: alpha + V/BW, per collective kind
    # The size sweep has to straddle V* = S*alpha*BW, otherwise alpha and bw are
    # not separately identifiable and the fit trades them against each other.
    # V* runs from ~2 MB (2 ranks, log N steps) to ~30 MB (16 ranks, ring), so
    # the grid spans 64 KB .. 256 MB and is repeated at two rank counts.
    for kind in ("all_gather", "all_to_all"):
        jobs.append(
            Job(f"M5.{kind}", "L0", "collective", {
                "op": kind, "npus": sorted({2, max(cp_grid)}),
                "bytes": [1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24,
                          1 << 26, 1 << 28],
            },
                "time vs message size, at two rank counts",
                "one alpha and one bw per collective; the shift of the knee "
                "with N tests the step-count model instead of assuming it",
                repeats=10, est_minutes=4.0)
        )

    jobs.append(
        Job("M6.1", "L0", "memory", {"op": "max_seq_len",
                                     "npus": [1] + cp_grid, "B": B},
            "largest T that fits, per NPU count, before OOM",
            "a (bytes/token) and M_static; the T_max(CP) model",
            repeats=1, est_minutes=8.0)
    )

    # ---------------- L1: the curve -------------------------------------
    # eff(c) via CP=1: c == T, so no CP implementation needed.
    jobs.append(
        Job("S1.1", "L1", "eff_curve", {"op": "single_layer_attn", "cp": 1, "T": L},
            "attention kernel time at each T on one card",
            "eff(c): the only fitted curve in the model",
            depends_on="M3.1", repeats=9, est_minutes=1.5)
    )
    jobs.append(
        Job("S1.2", "L1", "eff_curve", {"op": "block_forward", "cp": 1, "T": L,
                                        "layers": "must equal the model depth"},
            "same sweep through the FULL stack (all layers + projections)",
            "cross-checks S1.1 and fixes the layer granularity: a 1-layer time "
            "divided by an N-layer FLOP count is an N-fold error",
            depends_on="S1.1", repeats=9, est_minutes=1.5)
    )
    # machine-derived chunk: emulate CP by sharding the sequence by hand
    jobs.append(
        Job("S2.1", "L1", "imbalance", {"op": "shard_queries", "cp": cp_grid,
                                        "scheme": ["block", "balanced", "striped"],
                                        "T": T_ref},
            "time of each query chunk run in isolation, no collectives",
            "validates max/mean and the 2 - 1/CP closed form; decides balancer value",
            depends_on="S1.1", repeats=9, est_minutes=2.0)
    )
    jobs.append(
        Job("S3.1", "L1", "mask_topology", {"op": "static_share",
                                            "share": [0.0, 0.5, 0.85, 0.95],
                                            "T": T_ref},
            "layer time while the static/rectangular fraction is swept",
            "pairs ~ S^2 vs triangle; confirms the layout coefficient",
            depends_on="S1.1", repeats=9, est_minutes=2.0)
    )
    jobs.append(
        Job("S4.1", "L1", "ragged", {"op": "pad_vs_packed",
                                     "dist": "zipf", "median": T_ref},
            "time with padded dense input vs packed/jagged input",
            "the padding waste; decides whether it belongs in eff or t_fixed",
            depends_on="S1.1", repeats=9, est_minutes=3.0)
    )

    # ---------------- L2: validation -------------------------------------
    jobs.append(
        Job("E1.1", "L2", "end2end", {"op": "step", "T": [2048, 4096], "B": B,
                                      "cp": [1] + cp_grid, "scheme": "block"},
            "full training step time at each (T, CP)",
            "the 12-job calibration table; the headline validation",
            depends_on="S1.1,S2.1,M5.all_gather", repeats=7, est_minutes=2.0,
            needs_cp=True)
    )
    jobs.append(
        Job("E1.2", "L2", "end2end", {"op": "step", "T": T_ref, "B": [16, 64],
                                      "cp": [2, 4], "scheme": "block"},
            "batch sweep at fixed T and CP",
            "the alpha term: batch must scale the step, not the latency",
            depends_on="E1.1", repeats=7, est_minutes=2.0, needs_cp=True)
    )
    jobs.append(
        Job("E2.1", "L2", "end2end", {"op": "step", "T": T_ref, "B": B,
                                      "cp": 4, "scheme": "balanced"},
            "same as E1.1 but with the balancer on",
            "measured gain of rebalancing vs the predicted 2-1/CP bound",
            depends_on="S2.1,E1.1", repeats=7, est_minutes=2.0, needs_cp=True)
    )
    jobs.append(
        Job("E3.1", "L2", "cross_model", {"op": "step", "model": ["hstu-base", "longer"],
                                          "T": T_ref, "B": B, "cp": [1, 4]},
            "two other models, no re-calibration",
            "transfer claim: does one curve cover three models?",
            depends_on="E1.1", repeats=7, est_minutes=3.0, needs_cp=True)
    )
    jobs.append(
        Job("E4.1", "L2", "kernel_swap", {"op": "step", "T": T_ref, "B": B,
                                          "cp": [1, 4],
                                          "kernel": ["triton", "fused"]},
            "same config, two kernel implementations",
            "whether eff(c) is a kernel property (decides the 'calibrate once' claim)",
            depends_on="E1.1", repeats=7, est_minutes=3.0, needs_cp=True)
    )
    return jobs


# --------------------------------------------------------------------------
# Output, cost, and the gate
# --------------------------------------------------------------------------
def write_manifest(jobs: Sequence[Job], path: str) -> str:
    rows = [j.row() for j in jobs]
    cols: List[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def cost_report(jobs: Sequence[Job]) -> str:
    total = 0.0
    lines = []
    for layer in ("L0", "L1", "L2"):
        sub = [j for j in jobs if j.layer == layer]
        minutes = sum(j.cost for j in sub)
        total += minutes
        cp = sum(j.cost for j in sub if j.needs_cp)
        lines.append(f"  {layer} {LAYER_NAMES[layer]:<30} {len(sub):>2} jobs "
                     f"{minutes:>7.1f} min  (needs CP: {cp:.1f} min)")
    lines.append(f"  {'total':<36} {len(jobs):>2} jobs {total:>7.1f} min "
                 f"= {total / 60:.1f} h")
    no_cp = sum(j.cost for j in jobs if not j.needs_cp)
    lines.append(f"  without any CP implementation:  {no_cp:>7.1f} min "
                 f"= {no_cp / 60:.1f} h "
                 f"({no_cp / total:.0%} of the plan)")
    return "\n".join(lines)


def gates(jobs: Sequence[Job]) -> str:
    """The stop/go rules, in execution order.

    Written down before the data exists, so that a bad result cannot be
    re-narrated afterwards.
    """
    return "\n".join([
        "  G0  M1.1  noise floor",
        "        sigma > 5%  -> abandon the staircase/ceil tests, report only",
        "                       the coarse curve; do not fit gamma",
        "  G1  M1.1+M3.1  c_star from F/BW must be within 10x of the fitted c_star",
        "        else the curve is absorbing something else; stop and profile",
        "  G2  S1.1  the fitted curve must be monotone with a visible knee",
        "        no knee in the swept range -> extend the grid, not the model",
        "  G3  S2.1  block-vs-balanced gain must be <= 2 - 1/CP",
        "        exceeding it -> the closed form is wrong, revisit the mask",
        "  G4  E1.1  cp=1 MAPE < 5%",
        "        else the work counter is wrong: stop, do not calibrate further",
        "  G5  E4.1  eff curve refit on the second kernel: if c_star moves > 3x,",
        "        drop the 'calibrate once' claim; it is a kernel-specific model",
        "  G6  E3.1  cross-model error < 20% with no re-calibration",
        "        else report per-model calibration and reduce the claim",
    ])


def unresolved(jobs: Sequence[Job]) -> str:
    """Jobs that the NPU stack may not support, with the fallback for each."""
    return "\n".join([
        "  CP not implemented on the NPU stack",
        "    -> do not block. S1/S2/M5 give eff(c), imbalance and collective cost",
        "       independently; compose them. E1/E2 then become validation only.",
        "  jagged/packed inputs unsupported by the kernel",
        "    -> measure pad_vs_packed (S4.1) and charge the padding to t_fixed;",
        "       report the waste explicitly, never silently divide by it.",
        "  no per-kernel timing available (profiler restricted)",
        "    -> fall back to differential timing: same config with one phase",
        "       removed, difference = that phase. Two jobs per phase.",
        "  single-card only",
        "    -> run the collective curve on 2 cards if you can borrow them;",
        "       alpha and BW are node properties, not model properties.",
    ])


def summary(jobs: Sequence[Job]) -> str:
    return "\n".join([
        "=" * 78,
        "measurement manifest",
        "=" * 78,
        cost_report(jobs),
        "",
        "gates",
        gates(jobs),
        "",
        "if something is unsupported",
        unresolved(jobs),
    ])


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/npu_manifest.csv")
    ap.add_argument("--model", default="mtgr-large")
    ap.add_argument("--T", type=int, default=4096)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--cp-max", type=int, default=8)
    args = ap.parse_args(argv)

    jobs = build_plan(args.model, args.T, args.B, args.cp_max)
    import os
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    write_manifest(jobs, args.out)
    print(summary(jobs))
    print(f"\nmanifest written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
