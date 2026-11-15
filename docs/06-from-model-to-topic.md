# 06 · From cost model to research claim

A cost model alone is engineering. This document is the ladder that turns it
into a contribution — and the honest ranking of how far up it is worth climbing.

---

## 1. Why "predict the step time" is not enough

Three objections, and all of them are fair:

1. *"That's profiling plus a fit."* True, if the only deliverable is a number.
2. *"Why not just sweep?"* Because a grid is 96 jobs and the point is to need
   12 — but that argument only lands if the sweep is genuinely unaffordable.
3. *"What decision does it change?"* This is the real one. A prediction that
   does not change a decision is a curiosity.

So the deliverable has to be **a decision**, not a number: *which CP degree, at
which sequence length, at which model size, for the lowest cost at a given
quality.*

## 2. The composition worth trying

Both LONGER and MTGR report quality-vs-length as approximately power-law, and
MTGR reports three model sizes. So you can write

```
minimise   H(L, CP, m) = N_steps(m) · t_step(L, CP, m) / 3600      [GPU-hours]
subject to Q(L, m) ≥ Q*
```

with `t_step` from this package and `Q(·)` taken from a published (or locally
fitted) scaling law. The output is a Pareto front in `(GPU-hours, quality)` with
three curves on it:

- **longer only** — grow `L`, keep the model;
- **bigger only** — grow `m`, keep `L`;
- **longer with CP** — grow `L` using `CP` extra devices.

**The decisive figure**: if the CP curve lies below the other two in the
mid-to-high budget region, the whole line of work has a motivation that needs no
explanation. If it does not, you have learned something important early and
cheaply.

```
quality (AUC / NE)
   ^
   |        ___------  longer + CP
   |    ___/
   |   / ________------  bigger model
   |  /_/
   | / _______----------  longer only
   |/
   +-----------------------> GPU-hours (log)
```

## 3. Three tiers of contribution

### Tier 1 — measurement & model  (safe, publishable today)
A cross-model, CP-aware, analytically-grounded predictor plus a four-way
attribution of the CP penalty. Contribution is *understanding*: which of
efficiency / imbalance / communication / Amdahl dominates, per model.
Venue shape: MLSys / ASPLOS / NSDI measurement track, or the industrial track at
RecSys / CIKM. This tier does not require the quality side at all.

### Tier 2 — model + autotuner  (medium)
Add a search over `(CP, kernel, scheme, batch)` at fixed memory budget. The
narrative writes itself because Vidur-Search established it (42K GPU·h → 1
CPU·h); the novelty here is that CP and a GR mask are in the search space.
Deliverable: the search picks a configuration within X% of the best grid point
using 12 calibration jobs.

### Tier 3 — model + scaling law  (ambitious)
The Pareto composition of §2. Highest ceiling, highest risk: the quality side is
noisy, business-specific, and reviewers will ask whether the two halves are
really coupled or just multiplied together.

**Recommendation:** build Tier 1 first. It is a strict subset of Tier 2, needs no
quality data, and the attribution result is the part a reviewer is most likely
to remember. Escalate only if the ablation ordering comes out as predicted.

## 4. The three claims that are actually novel

Ranked by how defensible they are:

1. **The quadratic coefficient of a GR is derivable, not fittable.** For MTGR's
   `[static ‖ causal ‖ candidate]` layout the pair count is
   `S² + [P(P+1) − S(S+1)]/2 + K(P+1)` — closed form, exactly testable against a
   brute-force mask. Every prior cost model uses `L²` or a fitted exponent
   (`AdaptiveLoad`: "R=0.35 with tokens, 0.92 with B·S^p"). This is a small,
   clean, verifiable step past that.
2. **A static block makes CP more balanced, and by a computable amount.**
   `2 − 1/CP` for a causal block, →1 for a static prefix. This predicts that the
   published +37% from 2·CP mini-chunks should shrink on MTGR, and gives a
   diagnostic if it does not.
3. **The CP penalty decomposes, and at MTGR's operating point it is not Amdahl.**
   The published GFLOPs invert to `T ≈ 520`, giving an Amdahl ceiling of 1.10× at
   CP=2 against a reported 1.33× — so the loss must be efficiency +
   communication. That is consistent with the authors' own kernel-fusion
   headroom, from an independent direction.

Claim 1 is the foundation; claim 3 is the one that will be argued about, which is
good — it means it is a real claim.

## 5. Go / no-go, decided before you start

Run the calibration and the held-out validation. Then:

| outcome | verdict |
|---|---|
| held-out MAPE < 15%, selection accuracy > 90%, cross-model < 20% | all three tiers viable |
| MAPE < 15% but selection accuracy poor where the curve is *not* flat | Tier 1 only; the model is descriptive, not decisive |
| cross-model transfer fails badly | drop "calibrate once"; sell kernel-aware modelling + attribution (Tier 1, weaker) |
| ablation shows mask topology does not matter | stop. The thesis is gone; write up the negative result as a note on `L²` sufficiency |
| `c_star` fit lands far from `F/BW` and the comm ablation does no damage | the model is fitting noise; revisit the measurement, not the model |

Write this table into the experiment plan *before* seeing results. It is the
cheapest defence against the usual failure mode of performance work: after the
fact, every residual has a story.

## 6. What to build next, in order

1. `eff(c)` for two kernels on your own hardware — settles whether "calibrate
   once" is true (`scripts/demo_synthetic.py --no-figures` shows the shape).
2. The ragged-batch experiment (§5 of `04-validation.md`) — settles the
   max-vs-mean question, which is where GR differs most from LLM training.
3. The balanced-vs-block measurement on an MTGR layout — tests claim 2, one job.
4. Only then: the scaling-law composition.

## 7. What would make this a *bad* paper

- Reporting MAPE without ranking accuracy.
- Fitting `overlap` and reporting a 1% in-sample error.
- Comparing to no baseline. The baseline is "predict with a pure roofline" and
  "predict with `L²`" — both are in the ablation table and both lose badly, which
  is the point.
- Claiming CP "scales" without decomposing, which is what the existing paper
  already does and does not need repeating.
- Quoting the synthetic demo numbers as results. They are a pipeline test.
