"""L0 microbenchmarks: the constants every other number depends on.

Run these first, on the machine you will actually use, and re-run them whenever
the CANN/driver version changes. Everything reported per unit mass here gets
*frozen* in the cost model instead of being fitted from step time.

Usage
-----
    python scripts/npu/micro.py --dry-run           # print the plan, no torch
    python scripts/npu/micro.py --out data/micro.csv
    python scripts/npu/micro.py --only dispatch,hbm
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from harness import (  # noqa: E402
    Device, Timing, measure, noise_floor, print_environment, write_micro,
)

DTYPE_NAME = "bf16"


def _dtype(torch, name: str):
    return {"fp16": torch.float16, "bf16": torch.bfloat16,
            "fp32": torch.float32}[name]


# --------------------------------------------------------------------------
# M1  noise floor
# --------------------------------------------------------------------------
def job_noise(dev, args) -> List[Timing]:
    """Same input, repeated. The CV here is the resolution of everything else."""
    torch = dev.torch
    dt = _dtype(torch, args.dtype)
    out = []
    for T in (args.lengths[0], args.lengths[-1]):
        a = torch.randn(args.B, T, args.d, device=dev.kind, dtype=dt)

        def fn(a=a):
            b = a.reshape(-1, a.shape[-1])
            return b @ b.transpose(0, 1).contiguous()

        # repeat the whole measurement 3 times to get a distribution of CVs
        runs = [measure(fn, dev, f"noise T={T}", repeats=args.repeats) for _ in range(3)]
        out.extend(runs)
    out.append(noise_floor(out))
    return out


# --------------------------------------------------------------------------
# M2  dispatch cost: how many microseconds per kernel launch
# --------------------------------------------------------------------------
def job_dispatch(dev, args) -> List[Timing]:
    torch = dev.torch
    x = torch.zeros(8, 8, device=dev.kind)
    out = []
    for n in (1, 8, 64, 256):
        def fn(n=n):
            for _ in range(n):
                x.add_(0.0)
            return None

        out.append(measure(fn, dev, f"dispatch n={n}", repeats=args.repeats,
                           notes="zero-work kernels; slope = us/launch"))
    # slope from the two extreme points, in microseconds per launch
    if len(out) >= 2 and out[-1].n:
        slope = (out[-1].median_ms - out[0].median_ms) / (256 - 1) * 1e3
        out.append(Timing("c_dispatch (us/launch)", slope, slope, slope, 0.0, 1,
                          notes="pin t_fixed with this"))
    return out


# --------------------------------------------------------------------------
# M3  HBM bandwidth and achievable compute
# --------------------------------------------------------------------------
def job_hbm(dev, args) -> List[Timing]:
    torch = dev.torch
    dt = _dtype(torch, args.dtype)
    out = []
    for mb in (64, 256):
        n = mb * 1024 * 1024 // 2  # elements of 2 bytes
        src = torch.empty(n, device=dev.kind, dtype=dt)
        dst = torch.empty_like(src)

        def fn(src=src, dst=dst):
            dst.copy_(src)

        t = measure(fn, dev, f"copy {mb}MB", repeats=args.repeats)
        gbs = 2 * n * 2 / (t.median_ms / 1e3) / 1e9
        out.append(Timing(f"hbm bw {mb}MB (GB/s)", gbs, gbs, gbs, t.cv, t.n,
                          notes="pin bw_util = measured / nominal"))
    return out


def job_peak(dev, args) -> List[Timing]:
    torch = dev.torch
    dt = _dtype(torch, args.dtype)
    n = args.peak_n
    a = torch.randn(n, n, device=dev.kind, dtype=dt)
    b = torch.randn(n, n, device=dev.kind, dtype=dt)
    t = measure(lambda: a @ b, dev, f"matmul {n}^3", repeats=args.repeats)
    tflops = 2 * n ** 3 / (t.median_ms / 1e3) / 1e12
    return [Timing(f"peak {args.dtype} (TFLOPS)", tflops, tflops, tflops,
                   t.cv, t.n, notes="eff_inf ceiling sanity check")]


# --------------------------------------------------------------------------
# M4  sync cost
# --------------------------------------------------------------------------
def job_sync(dev, args) -> List[Timing]:
    out = []
    for n in (1, 100):
        def fn(n=n):
            for _ in range(n):
                dev.synchronize()

        out.append(measure(fn, dev, f"sync x{n}", repeats=20,
                           notes="if this is large, every timing is polluted"))
    return out


# --------------------------------------------------------------------------
# M5  collective cost: alpha + V/BW
# --------------------------------------------------------------------------
def job_collective(dev, args) -> List[Timing]:
    """Requires a multi-card launcher.

    Run with ``torchrun --nproc_per_node=N``. On one card this job is skipped,
    and the model falls back to the *nominal* link bandwidth, which is only good
    enough to bound ``c*`` -- not to fit ``alpha``.
    """
    try:
        import torch.distributed as dist
    except Exception:
        return []

    if not dist.is_initialized():
        return [Timing("collective SKIPPED", 0, 0, 0, 0, 0,
                       notes="single process: launch with torchrun --nproc_per_node=N")]

    torch = dev.torch
    backend = "hccl" if dev.kind == "npu" else "nccl"
    rank, world = dist.get_rank(), dist.get_world_size()
    out = []
    for kb in (64, 256, 1024, 4096, 16384, 65536):
        n = kb * 1024 // 2
        buf = torch.ones(n, device=dev.kind, dtype=torch.float16)
        for name, fn in (
            ("all_gather", lambda b=buf: dist.all_gather_into_tensor(
                torch.empty(n * world, device=dev.kind, dtype=torch.float16), b)),
            ("all_reduce", lambda b=buf: dist.all_reduce(b)),
        ):
            t = measure(fn, dev, f"{name} {kb}KB", repeats=args.repeats)
            if rank == 0:
                gbs = n * 2 / (t.median_ms / 1e3) / 1e9
                out.append(Timing(f"{name} {kb}KB", t.median_ms, t.p25_ms, t.p75_ms,
                                  t.cv, t.n, notes=f"{gbs:.1f} GB/s effective"))
    return out


# --------------------------------------------------------------------------
# M6  memory: largest sequence that fits
# --------------------------------------------------------------------------
def job_memory(dev, args) -> List[Timing]:
    """Bisect the largest T that fits at this B, per available NPU count.

    On a single card this gives one point; the memory model needs at least two
    *different per-rank chunk lengths* or it reports ``identifiable=False``.
    """
    torch = dev.torch
    dt = _dtype(torch, args.dtype)
    out = []
    lo, hi = 256, 65536
    while lo + 64 < hi:
        mid = (lo + hi) // 2
        try:
            a = torch.randn(args.B, mid, args.d, device=dev.kind, dtype=dt)
            s = (a @ a.transpose(1, 2)).sum()
            dev.synchronize()
            del a, s
            lo = mid
        except Exception:
            hi = mid
            try:
                torch.npu.empty_cache()  # type: ignore[attr-defined]
            except Exception:
                pass
    out.append(Timing("max_seq_len", float(lo), float(lo), float(lo), 0.0, 1,
                      notes=f"B={args.B} d={args.d}; largest T that fits"))
    return out


JOBS = {
    "noise": job_noise,
    "dispatch": job_dispatch,
    "hbm": job_hbm,
    "peak": job_peak,
    "sync": job_sync,
    "collective": job_collective,
    "memory": job_memory,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/npu_micro.csv")
    ap.add_argument("--only", default=",".join(JOBS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default=DTYPE_NAME, choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192])
    ap.add_argument("--peak-n", type=int, default=4096)
    ap.add_argument("--repeats", type=int, default=15)
    args = ap.parse_args(argv)

    wanted = [j.strip() for j in args.only.split(",") if j.strip()]
    unknown = [j for j in wanted if j not in JOBS]
    if unknown:
        ap.error(f"unknown job(s) {unknown}; available {sorted(JOBS)}")

    if args.dry_run:
        print("dry run: would execute", ", ".join(wanted))
        for j in wanted:
            print(f"  {j:<12} {JOBS[j].__doc__.strip().splitlines()[0] if JOBS[j].__doc__ else ''}")
        print("\nprotocol: warmup 5, median of 15, synchronize before/after, no L2 flush")
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
