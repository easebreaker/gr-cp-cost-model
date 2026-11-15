# KnapFormer — the closest existing formula

**arXiv 2508.06001** · Hu et al. · 2025

## One line
An online load balancer for diffusion-transformer training that solves a global
knapsack over tokens, using a **semi-empirical latency model**. That model is
almost exactly the one in this package — except its quadratic term is fitted,
while here it is derived from the mask.

## The latency model

```
t = k · ( 24·l·d²  +  γ · 4·l²·d )
```

- `24³l·d²` — the dense part, linear in sequence length `l`;
- `4·l²·d` — the attention part, quadratic;
- `γ` — a correction factor **fitted per GPU architecture**, because attention
  turns out to be more memory-bandwidth bound than its FLOP count suggests.

They state the reason plainly: "the attention term tends to be more
memory-bandwidth bound and does not scale linearly with FLOPs count".

## Results
| item | value |
|---|---|
| workload discrepancy across ranks | <1% |
| sequence lengths covered | hundreds → tens of thousands |
| speedup on FLUX-scale diffusion training | 2–3× |

## What to take
- **The two-term form is validated by someone else.** Linear dense + quadratic
  attention + one fitted coefficient is a working, published, load-balancing-grade
  model. This package keeps the shape and replaces the quadratic term with an
  exact pair count.
- **The `γ` correction justifies `eff(c)`.** They found they *needed* an empirical
  correction to the quadratic term. Here the correction is (a) made a function of
  `c` rather than a constant, and (b) kept separate from the derived pair count so
  it cannot absorb mask-structure error.
- **The knapsack formulation** (minimise variance of per-GPU work subject to
  memory) is the right way to state the assignment problem; `spec.assign`'s
  `balanced` scheme is the cheap analytic version of the same idea.
- **Discrepancy <1% is achievable** for *load*, as opposed to *time*. That is a
  useful separation: predicting the *distribution* of work across ranks is easier
  than predicting absolute time, and it is what a balancer needs.

## Where it breaks for GR/CP
- **No mask topology**: `4·l²·d` implicitly assumes a full or halved-square
  attention. MTGR's static rectangle makes the true count ~1.7× a causal one at
  the same `l`, so a fitted `γ` would be absorbing a *structural* factor and would
  not transfer to another layout.
- **DP load balancing, not CP**: no per-layer collective, no `CP` dimension, no
  sequence-sharded memory limit.
- **DiT, mixed-resolution video**: its heterogeneity is spatial/temporal, not
  user-behaviour-tail.

## The precise delta of this repository
| KnapFormer | here |
|---|---|
| `4·l²·d` with fitted `γ` | exact pair count over a 3-region mask |
| γ constant per GPU | `eff(c)` curve in chunk length |
| DP knapsack | CP assignment schemes + analytic imbalance |
| no memory model | explicit `T_max(CP)` model |
| no collective | per-layer comm volume, collective as a dimension |

Cite it as the direct precedent and state the delta in exactly these terms.
