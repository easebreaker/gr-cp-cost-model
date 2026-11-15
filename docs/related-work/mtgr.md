# MTGR — Industrial-Scale Generative Recommendation at Meituan

**arXiv 2505.18654** · Han et al. · CIKM'25

## One line
HSTU as the backbone, but keep every DLRM cross feature. Achieved by packing a
user's context, history, real-time events and **all K candidates** into one
sequence with a three-region dynamic mask — and that layout is the entire cost
model.

## The layout (this is the point)

```
D = [ U , S , R , [C,I]_1 , [C,I]_2 , ... , [C,I]_K ]
    └──────┬──────┘ └┬┘ └──────────┬───────────┘
        static     causal        candidate
      (rectangular) (triangle)   (prefix + self)
```

- **static** `U, S`: user profile + long-term history, visible to all later tokens;
- **causal** `R`: real-time events, causal to avoid leakage;
- **candidate**: each `[C, I]` sees the whole prefix, but only itself — this is
  the "dynamic masking" that prevents cross-candidate leakage.

`Group-LayerNorm` normalises each semantic region separately, because they live
in different spaces.

## Why the sequence is short and the batch is long
"User-level compression": one sequence serves all K candidates, so the user
representation is computed once and the incremental cost per candidate is
sub-linear. That is why their `T` is a few hundred tokens while `d = 768` —
`T/d < 1`, the opposite of an LLM.

## Numbers

| | small | medium | large |
|---|---|---|---|
| layers | 3 | 5 | 15 |
| `d_model` | 512 | 768 | 768 |
| heads | 2 | 3 | 3 |
| GFLOPs / example | 5.47 | 18.59 | 55.76 |

Other reported figures: 65× the FLOPs of their DLRM baseline for one-sample
forward inference; 1.6–2.4× training throughput over TorchRec; 10–100× model
compute "without significant cost increases"; +2.88pp CTCVR GAUC; +1.22% orders;
train cost parity with the baseline; −12% online inference resources.

## What to take
- **The layout, verbatim**: `presets.py::mtgr_layout(T, cand_tokens=40,
  static_share=0.85)`.
- **The three GFLOPs numbers are a free correctness test.** Invert them with this
  package's counter and all three land on the same `T`:
  `T = 524 / 518 / 518`, spread 1.2% (`scripts/validate_mtgr_flops.py`). One
  unknown, three equations, consistent — that is a strong check that the FLOPs
  term needs no calibration.
- **`head_dim = d/H = 256`** at every size. Not 32–64 as in LLM inference. That
  matters for the elementwise-bytes term and for whether a kernel is fused.
- **The candidate tail is part of every query's cost**: candidates each pay
  `P+1` keys, so `K` candidates add `K(P+1)` pairs — linear in `K`, not free.

## Two derived facts this repository adds
1. **Pairs are ~1.7× a causal sequence of the same length.** A rectangle costs
   twice a triangle per token, so MTGR's attention share exceeds what a
   `T = 5d` causal crossover suggests. At `T = 4096` it is 65%, not 52%.
2. **Its CP imbalance is far lower.** The static block gives every row the same
   key count, raising the floor on early queries. Block-assignment imbalance
   `2 − 1/CP` collapses toward 1 as `static_share → 1`
   (`test_static_prefix_damps_imbalance`). Prediction: the +37% reported for
   rebalancing causal HSTU should be much smaller here.

## Where the paper is silent (the gap)
No cost model, no CP, and no decomposition of why scaling behaves as it does.
The GFLOPs table is the only quantitative cost statement in it.

## Caveat when using the GFLOPs
"Per example" may mean per user (K candidates) or per candidate. The inversion
assumes per user-sequence, i.e. `T` is the concatenated length. If it is per
candidate, `T` doubles and all the shares shift — worth checking before quoting
the 18% attention share.
