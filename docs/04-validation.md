# 04 · Validation: how to avoid fooling yourself

The failure mode of every performance model is: a good in-sample fit, an
excellent story, and a useless extrapolation. This document is the checklist
that makes that hard.

---

## 1. Split by structure, not by row

Random row splits leak. Split on the axes you intend to extrapolate along:

| held out | why | command |
|---|---|---|
| sequence lengths `{3072, 6144}` | `T` never seen; tests the analytic `T` dependence | filter the CSV |
| one CP degree (typically 8) | tests extrapolation past the calibrated `c` range | leave-one-CP-out |
| one model (`mtgr-medium`, `hstu-base`) | tests "calibrate once, predict elsewhere" | filter by `model` |
| the second kernel | tests whether `eff(c)` is a kernel property | filter by `kernel` |
| a length distribution (Zipf) | tests the max-over-ranks term | generate jagged batches |

The demo (`scripts/demo_synthetic.py`) does lengths + two models + a balancer;
it never fits on `T ∈ {1024,3072,6144,8192}` or on `cp=8`-heavy shapes.

## 2. Report in this order

1. **Ranking** — Kendall τ and tie-aware selection accuracy per CP sweep. This
   is the metric that matches the use case (choose a configuration).
2. **Ratio error** — MAPE of `t̂(CP)/t̂(1)`. Fixed costs and kernel quality cancel.
3. **Absolute error** — MAPE, MdAPE, P90, signed bias. Should be dominated by the
   `cp=1` rows, which are the easiest.
4. **Memory** — error in `T_max`, not in GB.
5. **Attribution** — the four-way split of the CP penalty (below).

Never lead with absolute MAPE. A model that is 10% off but always picks the right
CP degree is more useful than one that is 4% off and picks wrong.

## 3. Tie-aware selection accuracy, and why the strict version lies

Choosing between two configurations compares two noisy numbers, so its
resolution is ~`sqrt(2)·σ`. With 3% measurement noise the honest tie band is
~4–5%. `metrics.selection_accuracy(pred, meas, tol=0.05)` scores a hit if the
model's pick lands within the band of the true best.

This matters because **the speedup curve is flat exactly where the question is
interesting**. At small `T`, `CP=2` and `CP=4` can differ by 2%, i.e. inside the
noise. Strict top-1 accuracy then measures luck:

```
demo_synthetic.py, held-out:
  tau=+0.750  top1=75%  sel=92%
  mtgr-large|T=1024   tau=+0.333  top1=  0%  sel=100%
  mtgr-large|T=8192   tau=+1.000  top1=100%  sel=100%
```

The `T=1024` sweep has an interior optimum that is flat; the model gets the sign
of a 2% difference wrong and is correct in the only sense that matters.

Report both numbers, state the band, and do not claim "predicts the optimal CP
degree" without it.

## 4. The ablation is the contribution

If a structural ingredient can be removed without hurting, it is decoration. The
table to produce:

| variant | MAPE | P90 | signed bias |
|---|---:|---:|---:|
| full model | 3.0% | 6.1% | −0.3% |
| no mask topology (force causal) | **8.5%** | 23.2% | −6.8% |
| mean instead of max over ranks | 5.2% | 9.2% | −4.1% |
| no efficiency curve (pure roofline) | **59.5%** | 70.8% | −59.5% |
| no communication term | 7.8% | 13.2% | −7.2% |
| no `L²` elementwise byte term | 9.3% | 20.6% | −9.3% |

Read it as an ordering of what matters: **efficiency curve ≫ mask topology >
elementwise bytes ≈ communication > imbalance > nothing**. That ordering is the
paper's thesis in one table, and it also tells you where the next unit of
modelling effort belongs.

(The numbers above are from the synthetic demo; on real hardware the ordering
should be checked, and the *elementwise* row is the one most likely to move,
because it depends on whether the attention kernel is fused.)

## 5. The ragged-batch check

A synchronous step is bounded by the slowest rank, not the average. With a
heavy-tailed length distribution the two differ a lot:

```
t_step = max_r [ ... ]      not    mean_r [ ... ]
```

Test protocol:

1. take the real length distribution (not uniform);
2. form batches, compute per-rank predicted pairs and per-rank predicted time;
3. compare `max_r t_r` against `mean_r t_r` and against the measurement;
4. repeat with the balancer on and off.

If using the mean gives you a better fit than the max, your balancer is doing
its job better than the model believes — check the `scheme` column. If neither
matches, the bottleneck is not attention and you should look at `t_fixed`.

## 6. Sanity checks that must pass before any headline number

| check | expectation |
|---|---|
| `cp=1` only | MAPE < 5% — otherwise the work counter is wrong, stop |
| `CP=2` scaling | in the 1.3–1.6 range if your stack resembles the published one |
| LONGER | CP barely helps; if you predict a 3× speedup, the comm term is wrong |
| `c_star` from the fit vs `F/BW` | within one order of magnitude |
| batch ×2 | predicted step time ×2 ± 2% at fixed `CP` |

## 7. Attribution: the useful deliverable

Decompose the CP penalty into four pieces and report them side by side:

```
speedup(CP) / CP   =   1 / (1 + loss)
loss = kernel-efficiency loss   (from eff(c) vs eff_inf)
     + imbalance loss           (from max vs mean over ranks)
     + exposed-communication    (from (1-overlap)(alpha + V/bw))
     + Amdahl floor             (from the attention share)
```

For MTGR at `T≈520` this predicts: Amdahl is not binding (1.10 ceiling vs 1.33
observed), so the loss is efficiency + communication. That is a mechanism-level
claim, and it makes a prediction: **fusing the segmented attention kernels
should recover most of it**, which is exactly what the CP-on-HSTU paper's
headroom study reports (1.33 → 1.5–1.6). Two independent arguments landing on
the same mechanism is the strongest evidence available here.

## 8. Reporting template

```
Hardware:            H100-SXM x8 (NVLink), 80 GB, clocks pinned at ...
Model:               mtgr-large (d=768, L=15), layout [U,S,R,[C,I]*K], K=40, S/T=0.85
Workload:            T ~ Zipf(1.2), median 4096, P99 12288, B=32
Kernel:              cutlass fbgemm hstu attention, fp16, activation recompute on
Calibration:         12 jobs (data/calibration.csv), t_fixed profiled = 6.1 ms
Fitted:              eff_inf=0.41, c_star=1712, gamma=1.11, alpha=1.4e-4, bw_comm=394 GB/s
Pinned:              overlap=0.45, bw_util=0.55, recompute=1.33

Held-out (40 jobs, T∉{2048,4096}, cp=8 included, 2 other models):
  MAPE 2.99%   MdAPE 2.83%   P90 6.12%   bias -0.34%
  tau +0.75    sel 92%  (tol 5%)
Ablations:           see table in §4
Attribution @CP=8:   efficiency 62% | imbalance 6% | comm 19% | amdahl floor 13%
Cost of calibration: 12 jobs, 2.1 GPU-hours  (vs 96 jobs for the full grid)
```
