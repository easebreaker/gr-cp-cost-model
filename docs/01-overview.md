# 01 · The argument in one page

**Question.** Can we predict how long a GR training step takes after CP-sharding
(HSTU / MTGR / LONGER), cheaply enough that the model is useful for choosing a
configuration?

**Answer.** Yes, if you split the problem correctly.

## The split

```
                    what you derive                    what you fit
    ┌──────────────────────────────────────┐   ┌───────────────────────┐
    │ mask topology  → pairs (closed form) │   │ eff(c)                │
    │ layout         → imbalance (exact)   │   │ alpha, bw_comm        │
    │ depth/width    → dense FLOPs         │   └───────────────────────┘
    │ layers         → collective volume   │         3 + 2 params
    │ dtype          → elementwise bytes   │      one 1-D curve in
    │ memory         → T_max               │      c = T / CP
    └──────────────────────────────────────┘
```

The left block is exact integer arithmetic, unit-tested against brute-force
masks. The right block is one smooth function of a single variable. Because
`c` does not care whether you are running HSTU or MTGR, or how many GPUs, the
calibration transfers — which is the whole reason this is cheap.

## Five facts that come out of the structure

1. **The crossover is `T = 5d`, not `12d`.** No FFN. At any given length a GR
   puts ~2.4× more of its budget into attention than a decoder does, so LLM cost
   coefficients do not transfer.

2. **A static rectangular block costs `S²`, and that changes the balance story.**
   MTGR's `[U,S,R,[C,C],…]` layout has ~1.7× the pairs of a causal sequence of
   the same length, *and* is far more CP-balanced — the block imbalance
   `2 − 1/CP` collapses toward 1 as the static share grows. So the 37% gain from
   rebalancing reported for causal HSTU should shrink on MTGR. Testable.

3. **At MTGR's own operating point, attention is a minority of the work.** Their
   published GFLOPs invert to `T ≈ 520` (consistent across three model sizes to
   1.2%). Attention share there is ~18%, so the Amdahl ceiling for CP=2 is
   1.10×. Their reported 1.33× therefore *cannot* come from the work split; it
   has to come from kernel efficiency at small chunks and from exposed
   communication — which is what their own kernel-fusion headroom study implies.

4. **Communication scales with depth, not with the number of collectives.**
   `V ∝ n_layers · B · T · d`. Forgetting `n_layers` makes comm look like 1% of
   the step instead of ~20%.

5. **Most of the remaining error is one curve.** Ablating `eff(c)` takes held-out
   error from 3.0% to 59.5%; ablating the mask topology takes it to 8.5%. So the
   modelling effort belongs in the mask algebra *and* the efficiency curve, and
   nowhere else.

## The calibration consequence

12 jobs, not a grid:

- 2 lengths × 4 CP degrees on one model, one kernel → the curve and the
  collective terms;
- 1 kernel swap, 2 batch sizes, 1 balancer swap → separate the curve from the
  fixed costs and check the imbalance prediction.

Three parameters are *pinned*, not fitted, because step time cannot identify
them (`overlap` in particular: only `(1−overlap)·α` enters, so leaving it free
gives a degenerate family of fits — see `03-calibration.md` §4).

## What would falsify it

- If the efficiency curve does not transfer across kernels even after re-fitting
  its 3 parameters, the "calibrate once" claim dies. Then the honest claim
  becomes "kernel-aware cost model, recalibrate per kernel" — still useful, less
  exciting.
- If ranking accuracy is bad in the region where the speedup curve is *not*
  flat, the model is not ready to choose configurations.
- If the mask topology turns out not to matter (ablation shows no damage), the
  thesis is decoration and the contribution collapses to "profile + fit".

Each of those is checked by a specific script in this repository and a specific
row in `04-validation.md`.

## Read next

- `02-cost-model.md` — the derivations, including the two places where the naive
  formula is wrong.
- `03-calibration.md` — the 12-job design and what must be pinned.
- `04-validation.md` — protocol, leakage controls, and what to report.
- `05-related-work.md` — where this sits, and the gap it fills.
- `06-from-model-to-topic.md` — turning the cost model into a research claim.
