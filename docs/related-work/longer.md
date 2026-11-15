# LONGER — Long-sequence Optimized transformer for GPU-Efficient Recommenders

**arXiv 2505.04421** · Deng et al., ByteDance · 2025

## One line
Don't make 12 layers attend to 4000 behaviours. Make **one** layer do it with a
deliberately small query, and let the other eleven work on a short sequence. That
choice makes LONGER the Amdahl limit case for CP — and a perfect unit test for a
communication term.

## Structure

```
1. global tokens     m anchors: candidate item, UID embedding, CLS
2. token merge       L items → L/K merged tokens of width K·d, projected to d
                     (+ InnerTrans: a 1-layer transformer *inside* each group)
3. hybrid attention
   layer 1:  Q = m + k   (k most recent behaviours, k≈100)
             KV = m + L/K (full merged sequence)              ← cross, cheap Q
   layers 2..N: causal self-attention on the m+k sequence     ← short
4. engineering       mixed precision, activation recompute, KV-cache serving
```

## Numbers

| item | value |
|---|---|
| query sampling | k=100 recent keeps >95% of full-query quality |
| FLOPs | ~half the attention FLOPs of full-query at that length |
| attention FLOPs reduction (L=2048, d=32, K=4) | ~43% |
| length-vs-quality | power law; deeper models benefit more from length, with diminishing returns |

## What to take
- **The two-phase cost structure** → `spec.py::LongerModel.phases`. Phase 1 is
  `(m+k) × (m + T/K)` cross-attention with **queries concentrated on one rank**;
  phase 2 is `(m+k)²` causal, replicated on every rank.
- **`q_concentrated=True`** is the interesting bit: only the last `k` behaviours
  are queries, and they live on one rank. So CP cannot split that work — the
  slowest rank does all of it, and it still has to receive every other rank's
  K/V. CP on LONGER is therefore a *memory* operation, not a speed one.
- **Communication is almost length-independent** (one layer, not `L`). That makes
  it the cleanest available test of a comm term: if a model predicts a big CP
  speedup for LONGER, the comm term is wrong (`test_longer_gets_no_cp_speedup`).
- **Token merge does not reduce dense FLOPs**, only attention. Merging `K` tokens
  of width `d` into one of width `Kd` and projecting back costs `2·K·d²` per
  merged token = `2·d²` per original token — identical to a `d→d` projection.
  Worth remembering before assuming "compression = cheaper".

## Where it breaks for CP
- Only one layer is long, so the fraction of work CP can shard is small.
- The query concentration is a load-balancing pathology that no 2·CP mini-chunk
  scheme fixes.
- Any cost model that treats all `L` layers as quadratic over-predicts CP's
  benefit by roughly `L`.

## Predictive consequence
For a LONGER-shaped model, the model predicts the speedup saturates almost
immediately and can go **below 1** (CP makes the step slower) once the
communication of the merged sequence exceeds the compute it saves. That is not a
bug in the model; it is the correct reading of a one-layer-long-query design.

## Used here as
`presets.py::longer` (`d=256, L=12, merge=4, m=2, k=100`) and the
LONGER-is-a-comm-unit-test argument in `docs/04-validation.md` §6.
