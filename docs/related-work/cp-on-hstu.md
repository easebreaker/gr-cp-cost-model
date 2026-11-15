# CP-on-HSTU — the baseline you must beat or complement

**"Scaling Generative Recommendations with Context Parallelism on Hierarchical
Sequential Transducers"** · arXiv 2508.04711 · Dong et al., Meta · RecSys'25

## One line
Makes context parallelism *work* on jagged, variable-length recommendation
sequences, and reports the first public CP numbers for a GR. It is a capability
paper: no predictive model, one model, one GPU type, and no decomposition of
where the scaling goes.

## The four technical contributions

| # | change | reported effect |
|---|---|---|
| 1 | **AllGather → AllToAll** for jagged inputs (send only the relevant chunk per rank) | 2.7× QPS; >60% lower peak memory |
| 2 | **2·CP mini-chunks**, rank `i` gets chunks `(i, 2CP−1−i)` | +37% throughput |
| 3 | custom Triton kernels for the reordering (avoid chunk/concat overhead) | part of (2) |
| 4 | remove `.item()` host sync on offsets (async copy, sync late) | +2% QPS |

## The numbers

**Max sequence length vs CP** (H100 80 GB, `B=32`, no activation checkpointing):

| CP | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| `T_max` | 3072 | 4096 | 7168 | 16384 |

**Scaling factor** (baseline 24K QPS, ideal 2.00×):

| case | scaling |
|---|---|
| ideal | 2.00× |
| DDP only | 1.6–1.7× |
| DDP + CP (cp=2) | 1.33× |
| DDP + CP (cp=2, 2× batch) | 1.55× |

Plus: 5.3× supported sequence length; and a headroom study claiming kernel fusion
would take the cp=2 scaling from 1.33× to ~1.5–1.6×.

## What to take

- **The baseline numbers** — they are the calibration target and the sanity check
  (`scripts/check_cp_table.py`).
- **`T_max` per CP** — an immediate test of the memory model. It is *not* `1/CP`:
  per-rank chunks land at 3072 / 2048 / 1792 / 2048 tokens. See `memory.py` and
  `docs/02-cost-model.md` §4.
- **The 2·CP mini-chunk scheme** → `spec.assign(..., scheme="balanced")`, and the
  imbalance it removes is analytically `2 − 1/CP` for a causal mask.
- **The headroom claim is the strongest external support for `eff(c)`.** The
  authors are saying, in their own words, that the loss is dominated by a kernel
  that was "optimized for shorter sequences", and that fusing it recovers most of
  the gap. That is exactly the `eff(c)` term.

## Three things it does *not* do (the gap)

1. **No model.** No prediction, no extrapolation, no attribution.
2. **One model, one GPU, one kernel.** Nothing about transfer.
3. **No decomposition.** "1.33×" is reported as a number, not as
   efficiency + imbalance + communication + Amdahl.

## The inconsistency worth pointing out

The GFLOPs table of MTGR inverts to `T ≈ 520`, where attention is ~18% of the
work and the Amdahl ceiling at CP=2 is 1.10×. But this paper reports 1.33×. Both
cannot be true at face value; resolving it needs one measurement of the attention
fraction. See `scripts/check_cp_table.py` §3 — the script prints the required
attention share (≈50%) and the `T` (≈2.1K) that would make them consistent.

## Reading it critically
- Convert every "scaling factor" to a per-rank time ratio before comparing to
  anything: `1.33×` at CP=2 means the per-rank step is `0.75×`, i.e. 67%
  parallel efficiency, not a 1.33× speedup over CP=1.
- Jagged `T` is a distribution. A synchronous step is bounded by the slowest
  rank, so `E[T]` is the wrong statistic to report.
