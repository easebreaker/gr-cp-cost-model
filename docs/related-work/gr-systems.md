# GR training systems — where the time actually goes

**NVIDIA recsys-examples** · **TurboGR** arXiv 2605.13433 · **MtgrBoost** arXiv
2505.12663 · **FreeScale** MLSys'26 · **LinkedIn GR training**

## One line
Four independent industrial reports all say the same thing: on generative
recommenders the gap between a naive implementation and an optimised one is
**3–10×**, and it is dominated by *kernel implementation, padding, and
load imbalance* — not by the asymptotic FLOPs. That is the empirical case for a
separate efficiency term in any cost model.

## NVIDIA recsys-examples (HSTU benchmark)

Optimisations applied cumulatively, H100, `T = 2S + C = 4099` at `B=32`:

| step | avg TFLOPS/GPU | MFU | speedup |
|---|---:|---:|---:|
| baseline (Triton attn, no shuffler) | 75.6 | 7.65% | 1.00× |
| + workload-balanced shuffler | 115.8 | 11.70% | 1.53× |
| + CUTLASS attention | **302.6** | 30.6% | **4.00×** |
| + DynamicEmb caching | 296.8 | 30.0% | 3.93× |
| + hash-round-robin sharding | 308.6 | 31.2% | 4.08× |
| + prefetch pipeline | 310.6 | 31.4% | 4.11× |

Two things follow immediately:

- **A kernel swap is 2.6× of the 4× total.** So `eff(c)` is *kernel-family
  specific*; the `kernel` column in the calibration CSV is not optional.
- **A shuffler is 1.53×** — i.e. load imbalance alone can be over a third of the
  achievable gain, which is why `max`-over-ranks (not `mean`) is the right
  composition rule.

## TurboGR (Ascend NPUs)
Baseline MFU **<10%**; variable-length padding redundancy is the culprit.
Reported: jagged fusion → 70% memory reduction and 2.2× latency; dynamic load
balancing → inter-device imbalance **47% → 2.4%**; hierarchical sparse
parallelism → 75.9% less all-to-all latency; 94% NPU utilisation; linear
scalability from 0.38 (tiny) to 0.97 (long).
Take: **imbalance of 47% is the number to beat**, and it is exactly what
`work.imbalance()` predicts analytically.

## MtgrBoost
SRAM tiling of the fused HSTU forward, using causal-mask vectors to skip
unnecessary tiles. Take: the "skip masked tiles" behaviour is `eff(c)`'s
dependence on tile granularity — same phenomenon as Striped Attention's 0.94× at
16K. It is a kernel property, so it belongs in the fitted curve, not the derived
part.

## FreeScale (MLSys'26)
Sequence recommendation models: heterogeneity → stragglers + blocking
communication → compute bubbles. Reports **90.3% reduction in computational
bubbles** on 256 H100s, via load-balanced samples, prioritised/overlapped
embedding communication, and SM-free communication.
Take: bubbles are `t_fixed` + imbalance; they belong *inside* the model (as a
measured offset plus an exact imbalance ratio), not in the residual.

## LinkedIn GR training
End-to-end GPU-hours **−65%**; incremental training with refreshed data cut one
job from 25.5 h to 9 h.
Take: the data lifecycle can dominate. Any model that reports "step time" must be
explicit about whether it includes ingestion; `t_fixed` is where that lives.

## What this means for the cost model

| observation | consequence |
|---|---|
| 4× from kernel choice | `kernel` is a calibration dimension, not a constant |
| 1.5–2× from balancing | `max`-over-ranks, and imbalance is a first-class term |
| 47% measured imbalance | the analytic `2 − 1/CP` (causal) is the same order |
| MFU 7–34% | absolute times are ~3–14× off peak; `eff_inf ≈ 0.4` is realistic |
| bubbles / ingestion | `t_fixed`, measured, never fitted |

## Where they break for CP
None of these vary the context-parallel degree, so none of them produce the
quantity this package predicts. They establish *which terms must exist*; this
package is about how those terms respond to `CP` and `T`.
