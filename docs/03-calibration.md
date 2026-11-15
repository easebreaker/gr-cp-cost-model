# 03 · Calibration: 12 jobs, and what not to fit

Goal: pin the five identifiable parameters with as few GPU-hours as possible,
and refuse to fit anything the data cannot determine.

---

## 1. The design

Fixed for the whole set: one model (`mtgr-large`, `d=768`, `L=15`), one machine,
one kernel, one batch unless stated.

| # | T | B | CP | scheme | kernel | purpose |
|---|---|---|---|---|---|---|
| 1–4 | 2048 | 32 | 1,2,4,8 | block | cutlass | curve + comm, `c` from 2048→256 |
| 5–8 | 4096 | 32 | 1,2,4,8 | block | cutlass | curve at a second `T`, same `c` range |
| 9 | 4096 | 32 | 4 | block | **triton** | is the curve a property of the kernel? |
| 10 | 4096 | 16 | 4 | block | cutlass | batch linearity + fixed-cost dilution |
| 11 | 4096 | 64 | 4 | block | cutlass | same, other direction |
| 12 | 2048 | 32 | 2 | **balanced** | cutlass | does the imbalance prediction hold? |

`python -m grcost sql` writes exactly this table as a fillable CSV.

Two `T` values are enough because `T` enters the *work* analytically; the only
thing that changes with `T` through the unknown part is `c = T/CP`. The four CP
values span `c` over a 16× range, which is what locates `c_star` and `gamma`.
Job 9 is the single most informative extra row: it tells you whether "calibrate
once" is a claim you can make.

## 2. Measure `t_fixed` separately — never fit it

`t_fixed` is everything that does not shard: embedding lookups, all-to-all for
the sparse part, optimizer, dataloader, host synchronisation. It is often 5–30%
of the step and it is completely CP-independent.

If you leave it as a free parameter, it will absorb CP-independent variance, the
in-sample fit will look excellent, and the extrapolation to other batches will be
wrong. Profiling recipe:

```
run the step with the dense backbone replaced by a no-op (or with L=1)
run the same step with the optimizer step removed
difference = t_fixed  (report both; they bound each other)
```

Write it into the `t_fixed_ms` column per row; `predict` adds it outside the max.

## 3. The three pinned parameters

| parameter | how to pin | typical |
|---|---|---|
| `overlap` | comm-only microbenchmark: time the K/V collective with compute disabled, divide by (comm bytes / link BW) | 0.3–0.7 |
| `bw_util` | streaming copy kernel at the working dtype | 0.5–0.8 |
| `recompute` | 1.0 without activation checkpointing, 4/3 with | 1.0 or 1.33 |

`overlap` is the important one. See §4.

## 4. Why `overlap` must be pinned

The model has

```
t_comm_exposed = (1 − overlap) · (alpha + V / bw_comm)
```

Only the products `(1−overlap)·alpha` and `(1−overlap)/bw_comm` are observable
from step time. Fitting all three is a rank-deficient problem, and the optimizer
solves it in the least useful way: `overlap → 1`, `bw_comm → ~0`. The first
version of this code did exactly that — a perfect in-sample fit with
`bw_comm = 1.5e5 bytes/s`, which is four orders of magnitude below any real
interconnect. The number "fits" and is meaningless.

Pin `overlap`, then fit the two exposed terms. This is the general rule:
**anything you can measure in isolation, measure it.**

## 5. `alpha` is usually unidentifiable — decide in advance

At a ~100 ms step, 25 µs × 15 layers = 0.4 ms = 0.4%. Below the noise. So
`alpha` will fit to ~0 and the model will predict almost no batch effect — while
the CP-on-HSTU paper reports 1.33× → 1.55× when the batch doubles, which *only*
happens if exposed latency is real.

Two options:

- **(a)** measure `alpha` with a tiny-batch, high-CP run where per-rank work is
  small enough for latency to dominate; or
- **(b)** accept it as a two-parameter comm model and note the batch-independence
  as a known limitation.

Do this deliberately. `fit.sensitivity()` prints the number so you cannot
accidentally ship an unfalsifiable parameter.

```
python -m grcost fit --csv data/calibration.csv
# identifiability (% per +1% parameter), vs measurement noise ~3.0%:
#   eff_inf   -75.9%
#   c_star    +36.0%
#   gamma      +7.4%
#   alpha      +0.1%   <- below noise
#   bw_comm    -5.7%
```

## 6. Practical measurement hygiene

- **Warm up, then take the median of ≥5 steps.** Compilation, autotuning and
  allocator behaviour are all in the first few steps.
- **Fix clocks or report them.** GPU boost variance on a 100 ms step is a few
  percent, which is the same order as the quantity you are fitting.
- **Record the kernel implementation.** Triton vs CUTLASS is a 4× gap
  (`docs/related-work/gr-systems.md`). Put it in the `kernel` column.
- **Keep everything else constant across rows** — same node, same
  NCCL/collective settings, same sequence-length distribution. If you use
  jagged/packed inputs, keep the same length distribution across the sweep,
  otherwise `T` is not the only thing changing.
- **If your inputs are jagged, decide what `T` means.** For MTGR,
  `T = |U|+|S|+|R|+K(|C|+|I|)`; report the mean and the P99 separately, because
  the step is bounded by the max across ranks, not the mean
  (`docs/04-validation.md` §5).
- **Two repeats of one config** is worth more than one extra config: it gives you
  the noise floor that all the metrics are judged against.

## 7. Budget

Twelve jobs × ~200 steps × ~100 ms ≈ **7 minutes of GPU time** each once the
model is loaded, so ~2 GPU-hours with startup and evaluation overhead. Even at
10× for a badly-behaved stack this is cheaper than one full grid point of a
96-configuration sweep.

## 8. What to do when the fit is bad

| symptom | likely cause |
|---|---|
| CP=1 residual large, all CP small | work counter wrong (check `flops_per_example` against a known GFLOPs table) |
| residuals grow monotonically with CP | comm term wrong (forgot `n_layers`? wrong collective?) |
| only the small-`c` points are off | `eff` curve shape; try `gamma` free with more `c` coverage |
| balanced vs block rows disagree | imbalance model; check `static_share` |
| one kernel only | refit `eff` for that kernel, keep everything else |
| everything off by a constant | `t_fixed` mis-profiled |
