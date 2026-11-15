# Striped Attention — the closed-form load imbalance

**arXiv 2311.09431** · Brandon et al. · 2023

## One line
Ring attention with a causal mask wastes up to half the machine, because in most
rounds some device has a fully-masked block and another has a fully-unmasked one.
Permuting tokens to device `i mod N` fixes it — and in doing so gives the exact
work formula this package uses.

## The observation

With contiguous chunks, in ring step `t`, device `i` holds K/V block
`(i − t) mod N`. If that block is later in the sequence than device `i`'s own
queries, **every pair is masked** — the device idles. If it is earlier, every
pair is live. So per round the slowest device does a full unmasked block, and the
ring "runs as fast as a workload with no attention masking, despite in principle
needing to compute only half the operations."

## The fix and the formula

Give device `i` the tokens with residue `i mod N`. Then every device sees a
near-triangular mask on every round, and per-device work is

```
Work(i, j) = c(c+1)/2   if i ≥ j      (query block i vs K/V block j)
           = c(c-1)/2   if i < j
```

The two differ only by the diagonal, so the theoretical speedup approaches **2×**.

| setting | result |
|---|---|
| 8×A100, 256K tokens | 1.41–1.45× end-to-end |
| 16×TPUv4, 786K tokens | 1.65× |
| 16K tokens, small blocks | **slower than ring** (0.94×) — tile granularity |

## What to take
- **The work formula is exact and free** → this is the basis of
  `spec.work_per_rank` and the `2 − 1/CP` closed form for contiguous blocks
  (`test_block_imbalance_closed_form`).
- **Small blocks lose.** At 16K tokens the striping was *slower*, because a
  device's block held only two attention tiles so nothing could be skipped. This
  is a concrete instance of the `eff(c)` roll-off, and an argument that the
  efficiency curve is not a fudge factor but the dominant kernel behaviour at
  small chunks.
- **Equivalence is exact** — the permutation is a relabelling, so quality is
  untouched. Any balancing scheme should be evaluated on the same basis: same
  total work, better distribution.
- **The upper bound on any balancer is the imbalance.** `blk/bal` can never
  exceed `2 − 1/CP`, which caps what a rebalancing result can claim
  (`test_imbalance_upper_bounds_benefit_of_balancer`).

## Where it breaks for GR
- Requires permuting tokens; with jagged, multi-region masks (and a static block
  that sits *before* all queries), residue striping does not map cleanly onto the
  dependency structure.
- On an MTGR layout the imbalance is already small (`static_share` raises the
  floor), so the potential gain of any balancer is small — a prediction the model
  makes and a single measurement can test.
- The 2× headroom is a causal-mask property. Quote it only for causal layouts.

## Used here as
`spec.assign(total, cp, "striped")`, the `2 − 1/CP` identity, and the argument
that kernel efficiency at small `c` — not the mask — is the binding constraint in
the small-`T` regime.
