"""Bridge to the ``timepred`` framework: derive its ``κ`` instead of fitting it.

The existing framework models one CP'd block as

    t_hat = t_cut / g(N) + τ + v·Vol,        g(N) = N / (1 + κ (N - 1))

or, in ratio form over a whole step,

    t_hat = t_0 [ (1 - r) + r / g + c ]

with ``r`` the shardable fraction, ``κ`` a fitted "how far from ideal
parallelism" coefficient, and ``c`` the un-overlapped communication share.

All three are derivable here:

    Δ(N) = sharded time at N / sharded time at 1        (exact, from work model)
    g(N) = 1 / Δ(N)                                    ->  κ(N) = (N g - 1)/(N - 1)
    r    = attention share of the sharded time          (``attn_only`` mode)
    c    = (1 - overlap)(alpha + Vol/BW) / t_0

The reconstruction is exact for both shard modes (`tests/test_bridge.py`).
The practical consequence: **κ is not a constant** -- it grows with N, because
``eff(c)`` rolls off as the per-rank chunk shrinks. A κ fitted on N∈{2,4}
under-predicts the loss at N=8 by 7-10% on a 910B-class machine. That error is
large enough to matter when choosing a parallel degree, and it is free to
compute once ``eff`` is calibrated -- no extra NPU time.

Why λ must be defined on *time*, not on pairs: the dense projection splits
evenly across ranks while the attention term carries the mask imbalance, so the
pair-level imbalance ``max/mean`` overstates the time imbalance. Using the pair
ratio makes the bridge wrong by 4-13% (growing with N). This is the same "dense
dampens imbalance" effect that ``docs/02-cost-model.md`` §3 describes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .cost import Params, efficiency, predict
from .spec import Layout
from .work import rank_work


@dataclass
class CPDecomposition:
    """The timepred-compatible view of one CP configuration."""

    n: int
    t0_ms: float                 # full step at CP=1
    t_fixed_ms: float
    shard_1_ms: float            # shardable (compute+elementwise) time at CP=1
    shard_n_ms: float            # same, at CP=n, on the critical rank
    attn_1_ms: float
    attn_n_ms: float
    dense_1_ms: float
    dense_n_ms: float
    eff_ratio: float             # eff(c1)/eff(cn)  -- diagnostic
    imbalance_pairs: float       # max/mean of pairs -- diagnostic
    imbalance_flops: float       # max/mean of flops -- diagnostic
    comm_ms: float               # exposed collective time
    r: float                     # shardable fraction used by the mode
    g: float                     # effective speedup of that fraction
    kappa: float                 # equivalent constant-κ
    c_frac: float                # exposed comm as a fraction of t0
    t_hat_ms: float              # reconstruction from (r, g, c)
    mode: str = "full"

    def row(self) -> str:
        return (f"N={self.n:<3} g={self.g:5.3f}  κ={self.kappa:6.3f}  "
                f"eff×{self.eff_ratio:5.3f}  imb={self.imbalance_pairs:5.3f}  "
                f"comm={self.comm_ms:7.2f}ms  t={self.t_hat_ms:8.2f}ms")


def _per_rank(model, layout: Layout, T: int, B: int, n: int, hw,
              p: Params, scheme: str,
              shard_dense: bool = True) -> Tuple[float, float, float]:
    """Return (total, attention, dense) milliseconds on the critical rank.

    ``shard_dense=False`` models an implementation where the projections are
    replicated on every rank and only attention is split.  That is a *different*
    cost, not a reformulation -- it is slower, and it is the honest fallback if
    the NPU stack's CP prototype does not shard the projection weights.
    """
    works = rank_work(model, layout, T, B, cp=n, scheme=scheme,
                      quantise=p.quantise_tiles,
                      comm_lands_in_hbm=p.comm_lands_in_hbm)
    eff = efficiency(p, T / n)
    eff1 = efficiency(p, float(T))
    ew_scale = p.fixpipe_hops / 3.0

    def ew_and_land(w):
        return p.recompute * (
            w.bytes_ew * ew_scale / (hw.hbm_bw * p.bw_util)
            + w.kv_hbm / (hw.hbm_bw * p.bw_util)
        )

    best = (-1.0, 0.0, 0.0)
    for w in works:
        t_attn = p.recompute * w.flops_attn / (hw.peak_flops * eff) + ew_and_land(w)
        if shard_dense:
            dense_flops = w.flops_dense
            e = eff
        else:
            dense_flops = w.flops_dense * n     # replicated
            e = eff1
        t_dense = p.recompute * dense_flops / (hw.peak_flops * e)
        tot = t_attn + t_dense
        if tot > best[0]:
            best = (tot, t_attn, t_dense)
    return tuple(v * 1e3 for v in best)  # type: ignore[return-value]


def _imbalance(works, key) -> float:
    vals = [key(w) for w in works]
    mean = sum(vals) / len(vals)
    return (max(vals) / mean) if mean else 1.0


def decompose(
    model,
    layout: Layout,
    T: int,
    B: int,
    n: int,
    hw,
    p: Params,
    t_fixed_ms: float = 0.0,
    scheme: str = "block",
    shard_mode: str = "full",
) -> CPDecomposition:
    """Compute the (r, g, κ, c) that reproduce ``cost.predict`` exactly.

    ``shard_mode``:
      ``full``      CP shards the whole block, projections included, so the only
                    unsharded part is ``t_fixed``.  This is what real CP does.
      ``attn_only`` only the attention tensor is split and the projections are
                    replicated, so ``r`` becomes the attention share of the
                    sharded time and ``g`` is defined on the attention part.
    """
    if n < 2:
        raise ValueError("decomposition needs n >= 2")
    if shard_mode not in ("full", "attn_only"):
        raise ValueError(f"unknown shard_mode {shard_mode!r}")

    t0 = predict(model, layout, T, B, 1, hw, p, t_fixed_ms=t_fixed_ms, scheme=scheme)
    tot1, attn1, dense1 = _per_rank(model, layout, T, B, 1, hw, p, scheme)
    totn, attnn, densenn = _per_rank(
        model, layout, T, B, n, hw, p, scheme,
        shard_dense=(shard_mode == "full"),
    )

    works_n = rank_work(model, layout, T, B, cp=n, scheme=scheme)
    w = works_n[0]
    cm = p.comm_model()
    comm_ms = cm.time(n, w.bytes_kv, rounds=w.rounds) * 1e3

    eff_ratio = efficiency(p, float(T)) / max(efficiency(p, float(T / n)), 1e-12)

    if shard_mode == "full":
        r = 1.0
        g = tot1 / totn                     # exact: totn = tot1 / g
        t_hat = t_fixed_ms + tot1 / g + comm_ms
    else:
        # projections replicated: r stays the attention share of the sharded
        # time, and g is defined on the attention part alone.
        r = attn1 / tot1 if tot1 else 0.0
        g = attn1 / attnn if attnn else 1.0
        t_hat = t_fixed_ms + dense1 + attn1 / g + comm_ms
        totn = dense1 + attnn

    lam = n / g
    return CPDecomposition(
        n=n, t0_ms=t0, t_fixed_ms=t_fixed_ms,
        shard_1_ms=tot1, shard_n_ms=totn,
        attn_1_ms=attn1, attn_n_ms=attnn,
        dense_1_ms=dense1, dense_n_ms=densenn,
        eff_ratio=eff_ratio,
        imbalance_pairs=_imbalance(works_n, lambda x: x.pairs),
        imbalance_flops=_imbalance(works_n, lambda x: x.flops),
        comm_ms=comm_ms, r=r, g=g,
        kappa=(lam - 1.0) / (n - 1.0),
        c_frac=comm_ms / t0 if t0 else 0.0,
        t_hat_ms=t_hat, mode=shard_mode,
    )


def sweep(
    model, layout: Layout, T: int, B: int, hw, p: Params,
    t_fixed_ms: float = 0.0, ns: Optional[List[int]] = None,
    scheme: str = "block", shard_mode: str = "full",
) -> List[CPDecomposition]:
    return [decompose(model, layout, T, B, n, hw, p, t_fixed_ms, scheme, shard_mode)
            for n in (ns or [2, 4, 8])]


def kappa_drift(model, layout: Layout, T: int, B: int, hw, p: Params,
                t_fixed_ms: float = 0.0, scheme: str = "block") -> str:
    """How much the 'constant' κ actually moves between N=2 and N=8.

    Fit κ on the low degrees, predict the high one, and report the error.  The
    cheapest possible test of the existing framework's assumption, and it needs
    no NPU time -- only the calibrated ``eff`` curve.
    """
    rows = sweep(model, layout, T, B, hw, p, t_fixed_ms, [2, 4, 8], scheme)
    k2, k4, k8 = (r.kappa for r in rows)
    k_fit = 0.5 * (k2 + k4)
    g8 = 8.0 / (1.0 + k_fit * 7.0)
    last = rows[-1]
    pred8 = t_fixed_ms + last.shard_1_ms / g8 + last.comm_ms
    err = (pred8 / last.t_hat_ms - 1.0) * 100.0
    lines = [f"  T={T:<6} c(cp1)={T:<6} c(cp8)={T // 8:<6}"]
    for r in rows:
        lines.append("  " + r.row())
    lines.append(f"  κ: N=2 {k2:.3f} | N=4 {k4:.3f} | N=8 {k8:.3f}"
                 f"   (drift {k8 / max(k2, 1e-9):.2f}x)")
    lines.append(f"  constant-κ fitted on N∈{{2,4}}, used at N=8: {err:+.1f}% error")
    return "\n".join(lines)
