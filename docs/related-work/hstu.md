# HSTU — Actions Speak Louder than Words

**arXiv 2402.17152** · Zhai et al., Meta · 2024

## One line
A generative-recommendation encoder whose block is *one fused projection + a
softmax-free gated attention*, with no FFN — which is why a GR's cost profile is
nothing like a transformer decoder's.

## Structure

```
X ──► [U,V,Q,K] = Split(SiLU(W1·X + b1))            # one linear, d → 4d
      A = SiLU(QKᵀ + rab^{p,t})                      # no softmax, no rescale
      Y = f2( LayerNorm(A·V) ⊙ U )                   # gating, d → d
      X = X + Y                                      # residual outside the block
```

`rab^{p,t}` is a learnable bias indexed by (relative position bucket, relative
timestamp bucket), **shared across heads**, added to the scores before `SiLU`.

## Why softmax is dropped
The number of prior events that bear on the target is itself a feature (engagement
intensity), and softmax normalisation destroys it. Under a Dirichlet-process
synthetic stream the gap between softmax and pointwise attention reaches 44.7%.

## Numbers worth quoting
| item | value |
|---|---|
| dense cost per token per layer | `10 d²` FLOPs (4d² UVQK + d² out) |
| attention cost per visible pair | `4 d` FLOPs |
| FFN | **none** |
| attention/dense crossover | `T* = 5d` (vs `12d` for a decoder) |
| fused-kernel speedup reported | large, from fusing UVQK + gating |

## What to take
- **`10 d²` and no FFN** → `spec.py::HstuModel.dense_flops_per_token`, and the
  `T = 5d` crossover (`test_ffn_moves_the_crossover`).
- **`4d` per pair** → the attention term. Nothing else in the block is quadratic,
  so the pair count fully determines attention cost.
- **`rab` is shared across heads**: it is `O(T²)` elementwise work computed once,
  not `H` times. If your implementation materialises it per head you have an
  `H×` elementwise tax — see the `bytes_ew` term and its 9.3% ablation.
- The residual is *outside* the block, so the activation footprint per layer is
  ~6 stored tensors of width `d`, not the ~10 a decoder keeps.

## Where it breaks for CP
Nothing structural — which is exactly why it is the clean baseline. But it hides
two things that matter for cost modelling:
- the fused kernel makes `eff(c)` strongly implementation-dependent (a Triton
  port and a CUTLASS kernel differ by 4×; see `gr-systems.md`);
- the paper's own sequence lengths are large, so it does not exercise the
  small-`T` regime where MTGR actually operates.

## Used here as
`presets.py::hstu-base` (`d=768, L=15`), `Layout.causal(T)`, and the
`no-FFN → crossover at 5d` argument in `docs/02-cost-model.md` §2.
