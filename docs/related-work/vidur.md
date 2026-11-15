# Vidur — the "why bother" argument, and the methodology

**arXiv 2405.05465** · Agrawal et al., Microsoft Research · 2024

## One line
An LLM inference simulator built from per-operator profiles plus a learned
runtime predictor, accurate to <9%, which makes configuration search cheap enough
to be routine. The template for "profile a little, predict a lot" — but it has no
CP.

## Method, in three parts
1. Decompose the model into token-level, sequence-level and communication ops.
2. Profile a minimal set of input sizes; train a predictor for the rest
   (so you do not need a full profile grid).
3. Discrete-event simulation over the deployment config to produce TTFT, TBT,
   latency, throughput, MFU.

## Numbers
| item | value |
|---|---|
| end-to-end latency error | <9% |
| P95 static workload error | 3.33% |
| dynamic workload error at 85% capacity | <5% |
| search cost for LLaMA2-70B | 1 CPU-hour vs 42K GPU-hours (~$218K) |

## What to take
- **The cost comparison is the argument for this whole repository.** 42K GPU-hours
  → 1 CPU-hour is what "a model instead of a sweep" is worth, and it is a number
  a reviewer already accepts.
- **Profile a small set, interpolate the rest.** That is exactly the 12-job
  design: `c = T/CP` spans a 16× range with four points, and `eff(c)` interpolates.
- **Predict at the operator level, compose by summation.** Here the "operators"
  are dense / attention / elementwise / collective.
- **Report MFU and memory utilisation alongside latency.** Both are free from the
  model and both are what practitioners check first.

## Where it breaks for GR/CP
- **Only TP and PP.** The operator set has no sequence-sharded attention with a
  per-layer collective, so there is no `CP` dimension to search.
- **It learns absolute time.** A learned predictor on `(B, L, d, TP)` does not
  transfer to a different mask topology; this package instead derives the `T` and
  mask dependence and learns only `eff(c)`.
- **Its known failure mode is instructive**: public sim-to-real comparisons show
  Vidur-style simulators can be hundreds of percent off out of distribution when
  a feature (prefix caching) is missing from the operator model. The lesson is
  *know exactly which effects are in the model* — here they are listed in
  `docs/02-cost-model.md` §8.
- Inference-only, decode included; the prefill/training regime differs.

## How this package differs
Same philosophy (profile a little, model the structure, search cheaply),
different split: Vidur learns per-op times and simulates; `grcost` derives the
work and learns one scalar curve. The latter transfers across models; the former
does not.

## Used here as
The framing for Tier 2 (`docs/06-from-model-to-topic.md`) and the target accuracy
band (single-digit MAPE on held-out configurations).
