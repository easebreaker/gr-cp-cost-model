# Variable-length load modelling in distributed training

**AdaptiveLoad** arXiv 2605.17923 · **Zeppelin** arXiv 2509.21841 · **SlimPack**
arXiv 2509.26246

## One line
Three recent papers that independently establish the same empirical fact: **step
time correlates with token count only weakly, and with `B·S^p` strongly.** The
quadratic term is not optional.

## The evidence

| paper | claim | number |
|---|---|---|
| AdaptiveLoad | correlation of step latency with total tokens | **R ≈ 0.35** |
| AdaptiveLoad | ... with `B·S^p` | **R ≈ 0.92** |
| AdaptiveLoad | imbalance before → after | 39% → 18.9% |
| AdaptiveLoad | throughput gain | +27.2% |
| Zeppelin | compute/comm ratio varies with length, so the quadratic and linear parts want *different* partitions | — |
| SlimPack | sample-level packing wastes work; needs fine-grained asymmetric packing | — |

AdaptiveLoad's dual constraint — `B·S ≤ M_mem` **and** `B·S^p ≤ M_comp` — is the
operational form of "memory is linear, compute is quadratic, so long-sequence
buckets must run at a smaller batch".

## What to take
- **`R = 0.35` vs `R = 0.92` is the single best citation against any model that
  scales with token count.** If you were tempted to predict step time from
  `tokens processed`, this kills it. Use it in the introduction.
- **The exponent `p` must be measured, not assumed 2.** For a causal HSTU it is
  2; for LONGER's one-long-layer design it is effectively 1; for MTGR's static
  block the coefficient is ~1.7× the causal one. This package's contribution is
  that `p` and its coefficient are derived per layout rather than fitted per
  dataset.
- **Zeppelin's point maps exactly onto CP**: the part that shards well (dense,
  linear) and the part that does not (attention with cross-rank K/V) want
  different splits. In `work.py` these are deliberately kept as two separate
  terms (`flops_dense` vs `flops_attn`) so their response to `CP` differs.
- **Packing matters**: equal-token bucketing is the wrong invariant, which is the
  same conclusion as the `B·S^p` constraint.

## Where they break for GR/CP
- Video-DiT / LLM workloads: full or causal attention, no static block, no
  candidate tail.
- Load balancing, not parallel-degree modelling: no per-layer collective, no
  memory-vs-`CP` curve.
- The exponent is fitted to the data they have; nothing says it transfers to a
  different mask topology — which is precisely the failure this package avoids by
  deriving it.

## Used here as
- The justification for the quadratic/pair-count term
  (`docs/02-cost-model.md` §1).
- The design of the ablation: if a pure-`L²` model is good enough, the mask
  algebra is decoration (`docs/04-validation.md` §4 shows it is not: 3.0% → 8.5%).
- The `p`-exponent table in `presets.py` (`crossover_len` per model) is the
  derived version of their fitted `p`.
