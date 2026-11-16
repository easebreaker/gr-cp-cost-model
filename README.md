# grcost

Analytic step-time and memory prediction for **context-parallel (CP) generative
recommenders** — HSTU, MTGR, LONGER.

The claim this repository is built to support:

> For a GR backbone, the amount of work is not something you fit. It follows from
> the attention-mask topology and the sequence layout. Only the *efficiency* with
> which that work becomes time is fitted — and that is a one-dimensional curve in
> the per-rank chunk length `c = T / CP`, so you calibrate it once and reuse it
> across models, sequence lengths and CP degrees.

Everything else here — the FLOPs counter, the pair count, the load-imbalance
ratio, the memory limit — is derived and exactly testable. That is what makes the
calibration cheap (12 jobs instead of a grid) and the validation honest.

```
t_step(CP) = t_fixed
           + max_r [ recompute * ( flops_r / (P_peak * eff(c))  +  bytes_r / (BW * bw_util) )
                     + (1 - overlap) * ( alpha + kv_bytes / bw_comm ) ]
```

with

```
eff(c)   = eff_inf / (1 + (c_star / c) ** gamma)      kernel efficiency vs chunk
t_comm   = rounds * S(N) * alpha + V / bw_comm         collective: serial steps
BW_app   = bw_comm / (1 + v_star / V)                  what a measurement reports
v_star   = S(N) * alpha * bw_comm                      break-even message size
```

Two things that look like fitted constants and are not:

* **Apparent bandwidth depends on message size** because a collective is `S(N)`
  serial steps, so the latency is amortised over a growing payload. One `alpha`
  and one `bw_comm` explain the whole curve; a per-size fit puts its breakpoints
  at `v_star`, which moves with the rank count and therefore does not transfer.
  See `docs/11-comm-model.md`.
* **`rounds` and `S(N)` are different counts.** `rounds` = how many collectives
  per step (layers x buckets); `S(N)` = how many hops inside one. Multiplying
  them together inflates the latency term by `cp-1`.

Five fitted parameters. Three more are pinned from microbenchmarks because they
are *not identifiable* from step time alone (see `docs/03-calibration.md`).

## Roadmap (three strictly nested tiers)

| tier | deliverable | status |
|---|---|---|
| **1** | cross-model, CP-aware cost + memory predictor, with a four-way attribution of the CP penalty | code complete; needs real measurements |
| **2** | configuration search on top of it: pick `(T, cp, kernel)` under a memory and deadline budget | scaffolded |
| **3** | compose with a quality law to get a `(GPU-hours, quality)` frontier and rank the levers: longer / bigger / more data | **interface reserved**, not a result |

Tier 3 is deliberately last and deliberately incomplete. `quality.py` and
`frontier.py` fix the *interface*; the quality numbers are placeholders that
print a warning. See `docs/10-scaling-law.md`.

One result in tier 3 already holds without any quality data: **CP is not a
throughput lever.** GPU-hours per token is `D·cp·t_step/(B·T·3600)`, so even a
perfect speedup leaves it unchanged, and in practice CP makes every token more
expensive. It buys feasibility and wall-clock, never cost per token. Measured
with the model: `tok/GPU-hour` falls 1.00 → 0.73 → 0.46 → 0.25 for cp = 1, 2, 4, 8.

## What is here now

Two things beyond the cost model, because they turned out to be the load-bearing
parts:

* **`grcost.manifest`** — the NPU measurement plan as code: 21 jobs, ~5 h, 72% of
  them runnable *without a CP implementation*, plus the stop/go gates written
  down before any data exists. See `docs/08-npu-experiment-design.md`.
* **`grcost.cost.attribute`** — splits the gap from perfect scaling into
  *fixed cost / efficiency / imbalance / communication*, so "the predicted
  speedup is wrong" becomes a number you can read instead of a hypothesis.

* **`grcost.bridge`** — the existing `timepred` framework's `κ` is *derivable*
  from `eff(c)` and the mask, not fittable. The reconstruction is exact, and it
  predicts that `κ` drifts ~16% from CP=2 to CP=8, so a constant-κ fit
  under-predicts the loss by 6-7%. See `docs/09-bridge-to-timepred.md`.

## Install and run

No required dependencies beyond the standard library.

```bash
pip install -e .                     # optional; or just PYTHONPATH=src
PYTHONPATH=src python -m grcost info
PYTHONPATH=src python -m grcost sweep --T 4096 --cps 1,2,4,8 --t-fixed 6
```

Full worked example (12 calibration jobs → 40 held-out):

```bash
PYTHONPATH=src python scripts/demo_synthetic.py
```

The NPU plan, with no hardware and no torch:

```bash
PYTHONPATH=src python -m grcost manifest --out data/npu_manifest.csv
python scripts/npu/micro.py --dry-run
python scripts/npu/sweep.py --dry-run
```

The blank data-collection form (regenerate after changing any parameter):

```bash
PYTHONPATH=src python scripts/make_sheet.py
# -> data/measurement_fill.md      <- type into this one
# -> data/measurement_fill.csv     (same content, for a spreadsheet)
# -> data/measurement_sheet.csv    (machine sheet: provenance + prerequisites)
# -> docs/12-measurement-sheet.md  (printable)
```

Fill `data/measurement_fill.md`, then read it back. The reader converts
human units (us, GB/s) to SI, refuses to compute anything whose prerequisites
are blank, runs the gates it can decide, and writes `out/params.json`:

```bash
PYTHONPATH=src python scripts/read_fill.py data/measurement_fill.md
PYTHONPATH=src python scripts/read_fill.py data/measurement_fill.md --all   # every blank
```

The tier-3 composition (placeholders, prints a warning):

```bash
PYTHONPATH=src python scripts/frontier_demo.py
```

Check the two published numbers this is built around:

```bash
PYTHONPATH=src python scripts/validate_mtgr_flops.py   # reproduces MTGR Table 2
PYTHONPATH=src python scripts/check_cp_table.py        # reads the RecSys'25 CP table
```

Tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v     # 248 tests
```

> If you are on macOS with the system Python and see the interpreter die with
> signal 11 (exit 139) when redirecting output to a file, that is a local
> `numpy`/`interpreter` issue, not this package — `grcost` imports no numpy.
> Run with `python3 -u` or don't redirect.

## What is derived vs fitted

| Quantity | How | Test |
|---|---|---|
| Visible `(query, key)` pairs, any mask | closed form, O(1) | `tests/test_layout.py` vs brute-force mask |
| Static block cost | `S^2` (rectangle) | `test_static_block_is_a_square` |
| Causal block cost | `C(2S+C+1)/2` (shifted triangle) | `test_causal_part_is_shifted_triangle` |
| CP load imbalance | `max/mean`, exact per scheme | `test_block_imbalance_closed_form` (`2 - 1/CP`) |
| Attention/dense crossover | `T* = 5d` (HSTU), `12d` (decoder) | `test_ffn_moves_the_crossover` |
| Collective volume | `n_layers * B * T * 4d * (CP-1)/CP` bytes | `test_pairs_are_conserved` |
| Max sequence length | linear in `T/CP` | `test_prediction_roundtrip` |
| Attention per-pair FLOPs | `2·H·(attn_dim + value_dim)` | `test_pair_width_matches_the_verified_operator_table` |
| `eff(c)` | **fitted** (3 params), or *solved* from a CP=1 sweep | `tests/test_compose.py` |
| Exposed latency / bandwidth | **fitted** (2 params) | — |

## Layout

```
src/grcost/
  spec.py        mask topology, CP assignment, model specs
  work.py        per-rank FLOPs / pairs / bytes
  cost.py        the 5-parameter cost model + ablations
  memory.py      activation-memory model and the T_max fit
  fit.py         Nelder–Mead in log space, identifiability probe
  metrics.py     MAPE, Kendall tau, tie-aware selection accuracy
  compose.py     solve eff(c) from a CP=1 sweep (no joint fitting)
  bridge.py      derive the timepred framework's κ from structure
  manifest.py    the NPU measurement plan + gates, as data
  tiling.py      tile quantisation and the head-parallel CP constraint
  quality.py     [tier 3] power-law quality model + identifiability guard
  frontier.py    [tier 3] throughput/memory/quality composition, alpha* threshold
  presets.py     HSTU / MTGR / LONGER configurations
  hardware.py    H100 / A100 / Ascend 910B, and the break-even chunk rule
  plotting.py    figures with Pillow only (optional)
docs/            the argument: model, protocol, literature, framing, NPU plan
scripts/         the three runs above
scripts/npu/     Ascend measurement drivers (harness, micro, sweep, e2e)
data/            fillable calibration table, NPU manifest, measurement sheet
```

## Status

Research scaffold, not a product. The synthetic demo validates the *pipeline*
and the *identifiability analysis*; the numbers it reports are not measurements.
Real conclusions require the 12 jobs in `data/calibration_template.csv`.

See `docs/01-overview.md` for the argument, `docs/05-related-work.md` for where
this sits relative to the literature (and what it does not do).
