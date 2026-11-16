# docs

The argument, in reading order. Each file is self-contained; `01` is the one-page
version.

| file | what it answers |
|---|---|
| `01-overview.md` | the thesis in one page: what is derived, what is fitted, what would falsify it |
| `02-cost-model.md` | the derivation, incl. the two places the naive formula is wrong |
| `03-calibration.md` | the 12-job design, what must be pinned, and measurement hygiene |
| `04-validation.md` | leakage controls, metrics, ablations, attribution, reporting template |
| `05-related-work.md` | the annotated map and the exact gap this fills |
| `06-from-model-to-topic.md` | composing with a scaling law; three contribution tiers; go/no-go |
| `07-objections.md` | the questions that will be asked, with short answers |
| `08-npu-experiment-design.md` | the NPU measurement plan: what, how, in what order, and the gates |
| `09-bridge-to-timepred.md` | deriving the existing framework's `κ` instead of fitting it |
| `10-scaling-law.md` | **reserved extension**: composing with a quality law; three tiers; go/no-go |
| `11-comm-model.md` | why apparent collective bandwidth depends on message size, and the step-count model that replaces piecewise fits |
| `12-measurement-sheet.md` | the blank data-collection form, by stage and metric (generated) |
| `13-pipeline-and-formulas.md` | **the whole chain on one page**: every formula, its meaning, how it is measured, and which claim it supports |

## related-work/

One file per paper or cluster. Each has the same shape: one line, mechanism,
numbers worth quoting, **what you can take**, **where it breaks for CP-sharded
GR**, and which part of this repository it justifies.

- `hstu.md` — the backbone, `10d²`, no FFN, `T = 5d` crossover
- `mtgr.md` — the `[U,S,R,[C,I]×K]` layout; the GFLOPs table used as validation
- `longer.md` — one long layer; the Amdahl limit case; a comm-term unit test
- `cp-on-hstu.md` — the baseline; jagged CP; the four numbers to explain
- `ring-attention.md` — the break-even chunk `c* = F/BW`
- `striped-attention.md` — the closed-form imbalance `c(c±1)/2`, `2 − 1/CP`
- `vidur.md` — profile-a-little methodology; 42K GPU-h → 1 CPU-h
- `habitat.md` — wave scaling; `γ` as the ancestor of `eff(c)`
- `knapformer.md` — the closest existing formula; the precise delta
- `variable-length-loads.md` — R=0.35 vs R=0.92; why the quadratic term is not optional
- `gr-systems.md` — 4× from kernel choice, 47% imbalance, 90% bubbles

## The upgrade path

`10-scaling-law.md` is deliberately the last document and deliberately
incomplete: it fixes the *interface* for the "cost model x scaling law" topic
(`quality.py`, `frontier.py`) without pretending the quality numbers exist.
The three tiers in it are strictly nested, so tiers 1 and 2 stand on their own.

## Status of the numbers

Numbers quoted from papers are as published. Numbers produced by this repository
are labelled: the synthetic demo is a *pipeline test*, and the MTGR GFLOPs
inversion and the CP-table analysis are *derived from published tables* — those
two are real results, reproducible with the scripts in `scripts/`.
