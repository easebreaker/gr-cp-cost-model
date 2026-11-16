# 02 · The cost model, derived

This document derives every term in the model from the mask topology, so that
the only thing left to measure is a one-dimensional efficiency curve.

Notation: `T` sequence length, `B` batch, `d` model width, `L` layers, `CP`
context-parallel degree, `c = T/CP` per-rank chunk, `H` heads
(`d_h = d/H`).

---

## 1. Attention pairs: the object everything else is built on

The cost of attention is proportional to the number of visible
`(query, key)` pairs. Write `n_keys(q)` for the number of keys visible to query
position `q`, and `P(a,b) = Σ_{q∈[a,b)} n_keys(q)`.

Three shapes cover every model in scope:

| region | rule | `n_keys(q)` | total pairs |
|---|---|---|---|
| static `[0,S)` | rectangle (all-to-all) | `S` | `S²` |
| causal `[S,S+C)` | lower triangle | `q+1` | `C(2S+C+1)/2` |
| candidate `[S+C,P+K)` | prefix + self | `P+1` | `K(P+1)` |

where `P = S + C`. Note the middle row: the causal block's cost is **not**
`C(C+1)/2`, because a causal query at global position `q` also sees the whole
static block, so it carries `q+1` keys rather than `q-S+1`. Getting this wrong
under-counts by `S·C`, which for MTGR's layout is comparable to the whole
triangular term. (`tests/test_layout.py::test_causal_part_is_shifted_triangle`.)

Concretely, for a pure causal mask of length `T`, `P(0,T) = T(T+1)/2`; for MTGR's
`[U,S,R,[C,I]×K]` with `S = 0.85(T-K)` and `K = 40`, the pair count is about
**1.7×** the causal count at the same `T` — a rectangle costs twice what a
triangle does for the same length. That is the entire reason MTGR's attention
share is higher than a causal model of the same `T`.

`Layout.cum_pairs` evaluates `P(a,b)` in O(1) for any range, which is what makes
CP assignment cheap to enumerate: a rank owning ranges `{(a₁,b₁), …}` does
`Σ P(aᵢ,bᵢ)` pairs.

## 2. FLOPs

Per token, per layer:

| model | projection | +FFN | total `d²` units |
|---|---|---|---|
| HSTU / MTGR (fused UVQK + `W_o`) | `4d²` + `d²` | none | **10** |
| decoder transformer (QKV + `W_o`) | `3d²` + `d²` | `4f d²` | **8 + 4f** (24 at `f=4`) |

Attention adds `4·d` FLOPs per visible pair (`2d` MACs for `QKᵀ`, `2d` for `AV`).

So

```
FLOPs = L · 10 d² · B T       (dense, HSTU)
      + L · 4 d · B · pairs   (attention)
```

**Validation against a table we did not produce.** MTGR (CIKM'25) Table 2
reports 5.47 / 18.59 / 55.76 GFLOPs per example for 3/512, 5/768 and 15/768.
Inverting each row for `T` with the formula above gives `T = 524 / 518 / 518`
— a 1.2% spread across three different (depth, width) pairs. One unknown, three
independent equations, consistent. `scripts/validate_mtgr_flops.py` prints it.

Two structural consequences fall straight out:

- **The crossover moves from `12d` to `5d`.** Attention per token is `2dT`,
  dense is `(proj + 4f)d²`; they are equal at `T* = (proj+4f)d/2`. Dropping the
  FFN cuts `T*` by 2.4×, so at any given length a GR spends far more of its
  budget in attention than a decoder does. LLM cost models are therefore not
  transferable.
- **At the length MTGR actually reports running (`T ≈ 520`), attention is only
  ~18% of FLOPs.** So CP cannot possibly be a big compute win there (Amdahl
  ceiling 1.10× at CP=2). Anything larger must come from somewhere else — see
  §6 and `scripts/check_cp_table.py`.

## 3. CP assignment: work, conservation, imbalance

CP shards the **queries**. Three assignments matter:

| scheme | who owns what | per-rank work |
|---|---|---|
| `block` | rank `i` owns one contiguous chunk | `P(ic,(i+1)c)` |
| `balanced` | rank `i` owns mini-chunks `i` and `2CP-1-i` | sums to ≈ mean |
| `striped` | rank `i` owns positions `≡ i (mod CP)` | near-triangular for all |

Two invariant facts:

1. **Conservation.** `Σ_r P_r = P(0,T)` for every scheme. CP does not change the
   algorithm, so it must not change total work — this is asserted for all
   schemes and degrees in `test_pairs_are_conserved`.
2. **Block assignment under a pure causal mask has a closed-form imbalance**

   ```
   max/mean = 2 - 1/CP
   ```

   because the last chunk carries `c(T - c/2)` pairs while the mean is `cT/2`.
   At `CP=8` that is 1.875, and it is why the causal triangular mask wastes up
   to half the machine (cf. Striped Attention, `docs/related-work/striped-attention.md`).

**The structural prediction that separates GR from LLMs.** A static rectangular
block gives every one of its rows the *same* number of keys, which raises the
floor on the early queries and flattens the load. With `static_share = 0.95` the
block imbalance collapses from 1.875 to below 1.10 at the same `T` and `CP`. So:

> CP load balancing (2×CP mini-chunks, striping, work stealing) should buy much
> less on MTGR-style layouts than the 37% reported for causal HSTU. If you
> measure 30%+, the static region is smaller than assumed.

That is a falsifiable claim with a specific diagnostic attached, which is the
kind of thing worth taking to a group meeting.

## 4. Memory: the actual reason CP is used

CP is adopted to fit longer sequences, not to go faster. The memory model is
easier and more robust than the time model:

```
M_rank(CP) ≈ M_static + a · B · (T/CP) + Δ(CP)
T_max(CP)  ≈ CP · (M_device − M_static − Δ(CP)) / (a B)
```

`a` (activation bytes per token) is inferable from a *single* observation of
`T_max`, which makes this almost free. Two cautions, both learned the hard way:

- **`a` is only identifiable if the per-rank chunk length varies** across your
  observations. If `B·T/CP` is the same in every row, `a` and `M_static` trade
  off exactly and any fitted value is fiction. `fit_memory` returns
  `identifiable=False` in that case rather than inventing a number.
- **The published table is not consistent with pure `1/CP`.** Fitting on CP=1
  and extrapolating over-predicts `T_max` by 1.5–1.7× at CP=8, because the
  per-rank chunk saturates near 2–3K tokens regardless of CP. Either a second
  constraint binds (kernel workspace, tile granularity) or the sweep stopped
  early. Either way: re-measure before trusting a `1/CP` extrapolation.

## 5. Communication

> Full derivation, including why apparent bandwidth depends on message size:
> `11-comm-model.md`. The short version is that a collective is `S(N)` *serial*
> steps, so `BW_app = BW_inf/(1 + V*/V)` with `V* = S·alpha·BW_inf` -- one curve,
> two parameters, and a knee that moves with the rank count.

Per layer, each rank must ship its K/V shard to the other `CP-1` ranks:

```
V = n_layers · B · T · 2 (K,V) · dtype_bytes · d · (CP-1)/CP
```

The `n_layers` factor is easy to forget and is worth a factor of 15 at
`L=15` — forgetting it makes communication look negligible when it is ~20% of
the step. (`work.py` multiplies by the number of *sharded* phases, which is 1
for LONGER and `L` for HSTU.)

Whether you can hide `V` is decided by one dimensionless comparison:

```
c* ≈ F_peak / BW_link        —  H100 NVLink ~2.2K tokens
                                A100 NVLink ~1.0K
                                Ascend 910B HCCS ~0.8K
                                100 Gb/s IB   ~25–80K
```

Below `c*`, a ring step's compute cannot cover its transfer. This single number
frames the whole search space: it says the interesting CP values are those with
`c ≳ c*`, i.e. `CP ≲ T/c*`, and that cross-node CP is not worth it below ~20K
tokens. No profiling required.

## 6. The efficiency curve, and why it is the only thing you fit

```
eff(c) = eff_inf / (1 + (c_star/c)^gamma)
```

Three parameters. It absorbs, in one shot:

- limited parallelism when the chunk is short (few rows per GEMM, wave
  quantisation);
- the segmented-kernel penalty: a chunked attention call has worse
  register/tile behaviour than a single call;
- anything else that makes a small chunk slower per FLOP than a big one.

This is where the ablated model collapses: removing `eff(c)` and using a pure
roofline raises held-out MAPE from 3.0% to **59.5%** (see `docs/04-validation.md`).
Nothing else in the model is remotely as important, which is exactly the
argument for spending the calibration budget on this curve and deriving the
rest.

`c_star` is not a free knob: it should land near the break-even chunk of §5.
If a fit returns `c_star` two orders of magnitude away from `F/BW`, the fit is
compensating for something else and the extrapolation will not hold.

## 7. Identifiability: what the data cannot tell you

With

```
t_comm_exposed = (1 − overlap) · (alpha + V/bw_comm)
```

only the products `(1−overlap)·alpha` and `(1−overlap)/bw_comm` enter. Fitting
all three gives a one-parameter family of equally good fits and the optimizer
will drive `overlap → 1, bw_comm → 0` — which is exactly what happened on the
first version of this model (it returned `bw_comm = 1.5e5 bytes/s`, physically
absurd, with a perfect in-sample fit).

Fix: pin `overlap` from a communication-only microbenchmark, then fit the two
exposed terms. Same for `bw_util` and `recompute`, which are directly
measurable.

Beyond that, `alpha` (per-collective latency) is often unidentifiable *at all*
when the step is ~100 ms: 25 µs × 15 layers is 0.4 ms, i.e. 0.4% — below any
realistic noise floor. `fit.sensitivity()` reports this before you trust a fit:
a parameter whose sensitivity is under the noise level will soak up residual and
make the in-sample number look better while the extrapolation gets worse.

## 8. What the model deliberately does not include

- **Host-side jitter, dataloader stalls, embedding-table traffic.** These live in
  `t_fixed` and are measured, not modelled. In production they can be 5–30% of
  the step, which is why `t_fixed` must never be fitted.
- **Kernel-specific behaviour across implementations.** Triton vs CUTLASS is a
  4× difference; the model treats them as different calibrations (the
  `kernel` column), not as one curve.
- **Decode / serving.** Decode is weight+KV-byte bound; CP saves KV but not
  weights, so Amdahl is much worse and the formula differs. This package is
  training/forward-prefill only.
- **Numerical effects** (which attention variant reorders reductions) and
  **scheduling** (whether the collective actually overlaps in your framework).
  Overlap enters only as the pinned scalar `overlap`.
