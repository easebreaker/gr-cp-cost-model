"""L1 sweeps: the only fitted curve, and the imbalance it does not explain.

The four jobs here are the calibration core. Note the third column: **three of
the four do not need a CP implementation**. That is the point of the whole design
-- ``eff(c)`` is a function of the chunk length, and at CP=1 the chunk length *is*
the sequence length; imbalance is a work-distribution property, measurable by
running one rank's query rows alone; the mask coefficient is measurable on one
card. Only the final composition needs real CP, and that is validation.

Usage
-----
    python scripts/npu/sweep.py --dry-run
    python scripts/npu/sweep.py --only eff --out data/npu_sweep.csv
    python scripts/npu/sweep.py --only eff,shard,static,pad
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from harness import Device, Timing, measure, print_environment, write_micro  # noqa: E402


def _layer(dev, args):
    from hstu_layer import HstuLayer

    dt = {"fp16": dev.torch.float16, "bf16": dev.torch.bfloat16,
          "fp32": dev.torch.float32}[args.dtype]
    return HstuLayer(args.d, args.heads, args.attn_dim, args.value_dim,
                     dtype=dt).to(dev.kind).eval()


def _mask(args, T, share=0.85, cand=40, static_full=True, device="cpu", dtype=None):
    from hstu_layer import region_mask

    core = max(0, T - cand)
    n_static = int(round(core * share))
    n_causal = core - n_static
    return region_mask(n_static, n_causal, cand, static_full=static_full,
                       device=device, dtype=dtype)


# --------------------------------------------------------------------------
# S1  eff(c): sweep T at CP=1, because then c == T
# --------------------------------------------------------------------------
def job_eff(dev, args) -> List[Timing]:
    torch = dev.torch
    layer = _layer(dev, args)
    out = []
    for T in args.lengths:
        x = torch.randn(args.B, T, args.d, device=dev.kind, dtype=layer.uvqk.weight.dtype)
        mask = _mask(args, T, device=dev.kind, dtype=layer.uvqk.weight.dtype)

        def fn(x=x, mask=mask):
            with torch.no_grad():
                return layer(x, mask)

        t = measure(fn, dev, f"layer T={T}", repeats=args.repeats)
        flops = layer_flops(args, T, args.B)
        out.append(Timing(f"layer T={T} ({flops / 1e9:.1f} GFLOP)",
                          t.median_ms, t.p25_ms, t.p75_ms, t.cv, t.n,
                          peak_mem_bytes=t.peak_mem_bytes,
                          notes=f"TFLOPS={flops / (t.median_ms / 1e3) / 1e12:.1f}"))
    return out


def layer_flops(args, T, B) -> float:
    """Structural FLOPs of one layer -- from grcost, not from the module."""
    from grcost.presets import hstu_layout
    from grcost.spec import HstuModel

    m = HstuModel("bench", d=args.d, n_layers=1, n_heads=args.heads,
                  attn_dim=args.attn_dim, value_dim=args.value_dim)
    return m.dense_flops_per_rank(T, B, 1) + B * m.attn_flops_per_pair() * \
        hstu_layout(T).pairs_total()


# --------------------------------------------------------------------------
# S2  imbalance: run one rank's query rows, no collectives at all
# --------------------------------------------------------------------------
def job_shard(dev, args) -> List[Timing]:
    """Per-rank compute time, measured by slicing the query rows.

    This is the CP compute of each rank with **communication removed**, so the
    ratio max/mean is the pure load imbalance -- directly comparable to the
    closed form ``2 - 1/CP`` for a causal mask.
    """
    torch = dev.torch
    layer = _layer(dev, args)
    T, B, d = args.T, args.B, args.d
    dtype = layer.uvqk.weight.dtype
    mask_full = _mask(args, T, device=dev.kind, dtype=dtype)
    out = []

    x_full = torch.randn(B, T, d, device=dev.kind, dtype=dtype)
    for scheme in args.schemes:
        for cp in args.cp_grid:
            ranges = _assign(T, cp, scheme)
            times, pairs = [], []
            for i, rs in enumerate(ranges):
                idx = [q for a, b in rs for q in range(a, b)]
                if not idx:
                    continue
                sel = torch.tensor(idx, device=mask_full.device)
                xq = x_full[:, sel]                 # this rank's queries
                rows = mask_full[sel]               # their mask rows, full key range

                def fn(xq=xq, rows=rows):
                    with torch.no_grad():
                        return _attend_subset(layer, xq, x_full, rows)

                t = measure(fn, dev, f"{scheme} cp={cp} rank={i}", repeats=args.repeats)
                times.append(t.median_ms)
                pairs.append(int((rows != 0).sum().item()) * B)
            if times:
                imb_t = max(times) / (sum(times) / len(times))
                imb_p = max(pairs) / (sum(pairs) / len(pairs))
                out.append(Timing(f"imbalance {scheme} cp={cp}", imb_t, imb_t, imb_t,
                                  0.0, len(times),
                                  notes=f"pairs-based={imb_p:.3f}  "
                                        f"closed-form={2 - 1 / cp:.3f}"))
    return out


def _attend_subset(layer, xq, xk, rows):
    """This rank's queries against the *full* K/V, using the layer's weights.

    This is exactly what a CP rank computes locally before any communication:
    its own query rows, all of the key/value rows. Timing it therefore isolates
    the compute imbalance from the collective cost.
    """
    import torch.nn.functional as F

    B, Lq, _ = xq.shape
    h, a, v = layer.h, layer.attn_dim, layer.value_dim
    p = F.silu(layer.uvqk(layer.norm_in(xk)))
    uv, qk = p.split([2 * v * h, 2 * a * h], dim=-1)
    _, val = uv.split([v * h, v * h], dim=-1)
    _, k = qk.split([a * h, a * h], dim=-1)
    pq = F.silu(layer.uvqk(layer.norm_in(xq)))
    uv_q, qk_q = pq.split([2 * v * h, 2 * a * h], dim=-1)
    u_q, _ = uv_q.split([v * h, v * h], dim=-1)
    q, _ = qk_q.split([a * h, a * h], dim=-1)

    def sh(t, w):
        return t.view(t.shape[0], t.shape[1], h, w).transpose(1, 2)

    q, k, val = sh(q, a), sh(k, a), sh(val, v)
    logits = F.silu(torch.einsum("bnhd,bmhd->bhnm", q, k)) * rows
    attn = torch.einsum("bhnm,bmhd->bnhd", logits, val)
    attn = attn.transpose(1, 2).reshape(B, Lq, v * h)
    gate = sh(u_q, v).transpose(1, 2).reshape(B, Lq, v * h)
    return layer.norm_attn(layer.out(attn * gate))


def _assign(total, cp, scheme):
    from grcost.spec import assign

    return assign(total, cp, scheme)


# --------------------------------------------------------------------------
# S3  mask topology: rectangle vs triangle at the same length
# --------------------------------------------------------------------------
def job_static(dev, args) -> List[Timing]:
    torch = dev.torch
    layer = _layer(dev, args)
    dtype = layer.uvqk.weight.dtype
    out = []
    for share in args.shares:
        mask = _mask(args, args.T, share=share, device=dev.kind, dtype=dtype)
        x = torch.randn(args.B, args.T, args.d, device=dev.kind, dtype=dtype)

        def fn(x=x, mask=mask):
            with torch.no_grad():
                return layer(x, mask)

        t = measure(fn, dev, f"static_share={share}", repeats=args.repeats)
        pairs = int((mask != 0).sum().item())
        out.append(Timing(f"static_share={share}", t.median_ms, t.p25_ms, t.p75_ms,
                          t.cv, t.n, notes=f"pairs={pairs:,} "
                                           f"density={pairs / args.T ** 2:.1%}"))
    return out


# --------------------------------------------------------------------------
# S4  ragged: what padding actually costs
# --------------------------------------------------------------------------
def job_pad(dev, args) -> List[Timing]:
    """Padded dense input vs a packed sequence of the same total tokens.

    The difference is the padding waste. It must be *reported* and charged to
    ``t_fixed``; silently dividing by it is how a CP speedup gets invented.
    """
    torch = dev.torch
    layer = _layer(dev, args)
    dtype = layer.uvqk.weight.dtype
    out = []

    lengths = _zipf_lengths(args.B, args.T, args.zipf_s)
    mean = sum(lengths) / len(lengths)
    Tmax = max(lengths)

    out = []
    # Same batch, same kernel, two lengths: the difference *is* the padding
    # waste. Building one block-diagonal mask over the concatenated length would
    # be an O((B*T)^2) tensor -- feasible for B=2 and catastrophic for B=32 --
    # and it would also change the kernel-launch count, confounding the result.
    for T, label in ((int(mean), "at mean length (ideal packing)"),
                     (Tmax, "padded dense (T = max)")):
        x = torch.randn(args.B, T, args.d, device=dev.kind, dtype=dtype)
        mask = _mask(args, T, device=dev.kind, dtype=dtype)

        def fn(x=x, mask=mask):
            with torch.no_grad():
                return layer(x, mask)

        t = measure(fn, dev, f"T={T}", repeats=args.repeats)
        out.append(Timing(f"{label} T={T}", t.median_ms, t.p25_ms, t.p75_ms,
                          t.cv, t.n, notes=f"tokens={args.B * T:,}"))

    n_pad = args.B * Tmax
    n_ideal = int(args.B * mean)
    waste = (n_pad - n_ideal) / max(n_pad, 1)
    out.append(Timing("padding waste (fraction of tokens)", waste * 100, waste * 100,
                      waste * 100, 0.0, 1,
                      notes="charge to t_fixed; report explicitly, never divide by it"))
    return out


def _zipf_lengths(B: int, T: int, s: float) -> List[int]:
    import random

    rnd = random.Random(0)
    w = [1.0 / (i + 1) ** s for i in range(200)]
    tot = sum(w)
    out = []
    for _ in range(B):
        r, acc = rnd.random() * tot, 0.0
        for i, wi in enumerate(w):
            acc += wi
            if acc >= r:
                out.append(max(256, int(T * (0.25 + 1.5 * i / 200))))
                break
        else:
            out.append(T)
    return out


JOBS = {"eff": job_eff, "shard": job_shard, "static": job_static, "pad": job_pad}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/npu_sweep.csv")
    ap.add_argument("--only", default="eff")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--heads", type=int, default=3)
    ap.add_argument("--attn-dim", type=int, default=None)
    ap.add_argument("--value-dim", type=int, default=None)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--T", type=int, default=4096)
    ap.add_argument("--lengths", type=int, nargs="+",
                    default=[512, 1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--cp-grid", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--schemes", nargs="+", default=["block", "balanced", "striped"])
    ap.add_argument("--shares", type=float, nargs="+", default=[0.0, 0.5, 0.85, 0.95])
    ap.add_argument("--zipf-s", type=float, default=1.2)
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args(argv)

    wanted = [j.strip() for j in args.only.split(",") if j.strip()]
    if args.dry_run:
        print("dry run:", ", ".join(wanted))
        print(f"  eff:    T in {args.lengths} at CP=1  (c == T, so no CP needed)")
        print(f"  shard:  cp in {args.cp_grid} x {args.schemes}  (query slices only)")
        print(f"  static: share in {args.shares} at T={args.T}")
        print(f"  pad:    B={args.B} zipf s={args.zipf_s} vs padded dense")
        print("  CP implementation required: none of the above")
        return 0

    dev = Device(args.device)
    print_environment(dev)
    print()
    for j in wanted:
        print(f"[{j}]")
        timings = JOBS[j](dev, args)
        for t in timings:
            print(t.line())
        if timings:
            write_micro(args.out, j, timings)
        print()
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
