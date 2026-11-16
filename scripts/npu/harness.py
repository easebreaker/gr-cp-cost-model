"""Timing harness for the NPU: one protocol, one output format.

Protocol (fixed, because otherwise the numbers are not comparable)
------------------------------------------------------------------
1. warm up ``WARMUP`` iterations and throw them away -- first-call compilation,
   autotuning and allocator behaviour all live there.
2. time ``REPEATS`` iterations, each between freshly recorded NPU events.
3. ``torch.npu.synchronize()`` before starting the clock and after stopping it.
   Without the trailing sync you are timing the *CPU launch*, which can be
   10-100x short of the device time.
4. report the **median** and the IQR. Never the mean: one descheduled step
   poisons it.
5. do not flush L2. The production path does not either.

Every number this harness produces is written with its own CV, because the whole
analysis downstream is built on comparing differences against that CV.

Deliberately no torch import at module scope: ``--dry-run`` and the manifest
must work on a laptop with no CANN install.
"""

from __future__ import annotations

import csv
import math
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

WARMUP = 5
REPEATS = 15


# --------------------------------------------------------------------------
# Device abstraction
# --------------------------------------------------------------------------
class Device:
    """Thin wrapper so the same script runs on NPU, CUDA or CPU."""

    def __init__(self, prefer: str = "auto"):
        self.kind = "cpu"
        self.mod = None
        self._torch = None
        order = ["npu", "cuda", "cpu"] if prefer == "auto" else [prefer]
        import importlib

        for kind in order:
            if kind == "npu":
                try:
                    import torch
                    import torch_npu  # noqa: F401

                    if torch.npu.is_available():
                        self.kind, self._torch = "npu", torch
                        self.mod = torch.npu
                        break
                except Exception:
                    continue
            if kind == "cuda":
                try:
                    import torch

                    if torch.cuda.is_available():
                        self.kind, self._torch = "cuda", torch
                        self.mod = torch.cuda
                        break
                except Exception:
                    continue
            if kind == "cpu":
                import torch

                self.kind, self._torch = "cpu", torch
                self.mod = torch
                break
        if self._torch is None:
            raise RuntimeError("no usable torch device; try --dry-run")

    @property
    def torch(self):
        return self._torch

    def synchronize(self) -> None:
        if self.kind in ("npu", "cuda"):
            self.mod.synchronize()

    def event(self):
        """Timing event; falls back to wall clock on CPU."""
        if self.kind in ("npu", "cuda"):
            return self.mod.Event(enable_timing=True)
        return None

    def elapsed_ms(self, start, end, wall: float) -> float:
        if start is not None and end is not None:
            return float(start.elapsed_time(end))
        return wall * 1e3

    def max_mem_bytes(self) -> int:
        try:
            if self.kind == "npu":
                return int(self.mod.max_memory_allocated())
            if self.kind == "cuda":
                return int(self._torch.cuda.max_memory_allocated())
        except Exception:
            pass
        return 0


@dataclass
class Timing:
    label: str
    median_ms: float
    p25_ms: float
    p75_ms: float
    cv: float               # coefficient of variation of the timed samples
    n: int
    peak_mem_bytes: int = 0
    notes: str = ""

    @property
    def iqr_ms(self) -> float:
        return self.p75_ms - self.p25_ms

    def row(self) -> dict:
        return {
            "label": self.label,
            "median_ms": round(self.median_ms, 4),
            "iqr_ms": round(self.iqr_ms, 4),
            "cv_pct": round(self.cv * 100, 2),
            "n": self.n,
            "peak_mem_gb": round(self.peak_mem_bytes / (1024 ** 3), 3),
            "notes": self.notes,
        }

    def line(self) -> str:
        return (f"  {self.label:<38} {self.median_ms:9.3f} ms  "
                f"IQR {self.iqr_ms:7.3f}  CV {self.cv * 100:4.1f}%")


def measure(fn: Callable[[], object], dev: Device, label: str,
            warmup: int = WARMUP, repeats: int = REPEATS,
            notes: str = "") -> Timing:
    """Time ``fn`` under the fixed protocol."""
    for _ in range(warmup):
        fn()
    dev.synchronize()

    samples = []
    for _ in range(repeats):
        start, end = dev.event(), dev.event()
        if start is not None:
            start.record()
        t0 = time.perf_counter()
        fn()
        dev.synchronize()
        if end is not None:
            end.record()
            dev.synchronize()
        samples.append(dev.elapsed_ms(start, end, time.perf_counter() - t0))

    samples.sort()
    med = statistics.median(samples)
    n = len(samples)
    q1 = samples[n // 4] if n >= 4 else samples[0]
    q3 = samples[(3 * n) // 4] if n >= 4 else samples[-1]
    mean = sum(samples) / n
    var = sum((s - mean) ** 2 for s in samples) / max(n - 1, 1)
    cv = math.sqrt(var) / mean if mean else 0.0
    return Timing(label, med, q1, q3, cv, n,
                  peak_mem_bytes=dev.max_mem_bytes(), notes=notes)


def noise_floor(timings: Sequence[Timing]) -> Timing:
    """The single most important number in the plan: same-input repeatability.

    Gate G0. Everything else is judged against this.
    """
    cvs = [t.cv for t in timings if t.cv > 0]
    med = statistics.median([t.median_ms for t in timings])
    if not cvs:
        return Timing("noise floor", med, med, med, 0.0, len(timings))
    cv = statistics.median(cvs)
    return Timing("noise floor (same-input CV)", med, med, med, cv, len(timings),
                  notes="G0: >5% invalidates the fine-grained verdicts")


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
# `layers` is not decoration: a 1-layer measurement divided by an N-layer FLOP
# count inflates the extracted efficiency by exactly N, and the resulting
# prediction silently lands at the per-layer scale. Recording it here is what
# lets the analysis refuse to make that mistake (grcost.compose.extract_eff).
MICRO_HEADER = ["job", "label", "median_ms", "iqr_ms", "cv_pct", "n",
                "peak_mem_gb", "layers", "notes"]


def _append_csv(path: str, header: Sequence[str], rows: Sequence[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(header))
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def write_micro(path: str, job: str, timings: Sequence[Timing],
                layers: int = 0) -> str:
    """Append timings.  ``layers`` is the depth of the timed module (1 = block)."""
    rows = []
    for t in timings:
        r = t.row()
        r["job"] = job
        r["layers"] = layers
        rows.append(r)
    _append_csv(path, MICRO_HEADER, rows)
    return path


CALIB_HEADER = ["model", "layout", "T", "B", "cp", "scheme", "step_ms",
                "t_fixed_ms", "kernel", "tag"]


def write_calibration(path: str, model: str, layout: str, T: int, B: int, cp: int,
                      step_ms: float, t_fixed_ms: float = 0.0,
                      scheme: str = "block", kernel: str = "default",
                      tag: str = "") -> str:
    """One row in the schema ``grcost.fit.load_csv`` reads."""
    _append_csv(path, CALIB_HEADER, [{
        "model": model, "layout": layout, "T": T, "B": B, "cp": cp,
        "scheme": scheme, "step_ms": round(step_ms, 4),
        "t_fixed_ms": round(t_fixed_ms, 4), "kernel": kernel, "tag": tag,
    }])
    return path


TIMEPRED_HEADER = ["batch", "time_s", "peak_mem_bytes", "expert_frac", "plan",
                   "len_embedding", "t_embedding", "len_attention", "t_attention"]


def write_timepred_sample(path: str, batch: int, time_s: float,
                          peak_mem_bytes: int, cp: int,
                          len_attention: int,
                          t_embedding_s: Optional[float] = None,
                          t_attention_s: Optional[float] = None,
                          blocks: Sequence[str] = ("embedding", "attention"),
                          ) -> str:
    """A row for the existing ``timepred`` measurement table.

    Same measurement, second schema -- so one NPU run feeds both the analytic
    model and the fitting framework without a conversion step in between.
    Column names and the ``plan`` grammar follow ``timepred/io.py``.
    """
    row: Dict[str, object] = {
        "batch": batch,
        "time_s": f"{time_s:.6f}",
        "peak_mem_bytes": peak_mem_bytes,
        "expert_frac": 1.0,
        "plan": f"attention=cp:{cp}" if cp > 1 else "none",
        "len_embedding": "",
        "t_embedding": f"{t_embedding_s:.6f}" if t_embedding_s else "",
        "len_attention": len_attention,
        "t_attention": f"{t_attention_s:.6f}" if t_attention_s else "",
    }
    _append_csv(path, TIMEPRED_HEADER, [row])
    return path


# --------------------------------------------------------------------------
# Host-side context worth capturing with every run
# --------------------------------------------------------------------------
def environment(dev: Device) -> Dict[str, str]:
    import platform
    import sys

    env = {
        "device": dev.kind,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": "",
        "torch_npu": "",
        "cann": os.environ.get("ASCEND_HOME_PATH", os.environ.get("ASCEND_HOME", "")),
    }
    try:
        import torch

        env["torch"] = torch.__version__
    except Exception:
        pass
    try:
        import torch_npu

        env["torch_npu"] = getattr(torch_npu, "__version__", "")
    except Exception:
        pass
    return env


def print_environment(dev: Device) -> None:
    print("environment")
    for k, v in environment(dev).items():
        print(f"  {k:<10} {v}")
