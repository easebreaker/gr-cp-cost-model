"""Hardware constants and the one dimensionless rule that decides most of CP.

The only thing you need before running any experiment is the *ring attention
break-even chunk length*:

    c* ~= F_peak / BW_link

Below it, a ring step's compute cannot hide its transfer, so CP stops paying
off.  For H100 NVLink that is ~1-2k tokens; over a 100 Gb/s InfiniBand link it
is ~20k tokens.  Everything else in this package is refinement on top of that.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Hardware:
    name: str
    peak_flops: float        # dense FLOPS at the training dtype (bf16/fp16)
    hbm_bw: float            # bytes/s
    link_bw: float           # bytes/s, per direction, within a scale-up domain
    link_latency: float      # s per collective/message (alpha term)
    mem_gb: float
    scale_out_bw: float = 12.5e9   # bytes/s, 100 Gb/s InfiniBand
    notes: str = ""

    def break_even_chunk(self, bandwidth: float | None = None) -> float:
        """c* ~= F / BW: chunk length above which transfer can be hidden."""
        bw = self.link_bw if bandwidth is None else bandwidth
        return self.peak_flops / bw

    def report(self) -> str:
        c_nv = self.break_even_chunk()
        c_ib = self.break_even_chunk(self.scale_out_bw)
        return (
            f"{self.name}: {self.peak_flops / 1e12:.0f} TFLOPS, "
            f"HBM {self.hbm_bw / 1e12:.2f} TB/s, "
            f"link {self.link_bw / 1e9:.0f} GB/s\n"
            f"  break-even chunk: {c_nv:,.0f} tokens (scale-up), "
            f"{c_ib:,.0f} tokens (scale-out)\n"
            f"  => max CP worth trying: ~T/{c_nv:,.0f} within a node"
        )


H100_SXM = Hardware(
    name="H100-SXM",
    peak_flops=989e12,
    hbm_bw=3.35e12,
    link_bw=450e9,
    link_latency=5e-6,
    mem_gb=80.0,
    notes="NVLink4 900GB/s bidirectional (450 per direction)",
)

A100_80 = Hardware(
    name="A100-80G",
    peak_flops=312e12,
    hbm_bw=2.0e12,
    link_bw=300e9,
    link_latency=5e-6,
    mem_gb=80.0,
)

# Values are for the common 910B class parts; verify against your SKU before
# trusting absolute numbers -- the *ratios* (which is what c* depends on) are
# usually close, but the advertised TFLOPS vary a lot by variant and dtype.
ASCEND_910B = Hardware(
    name="Ascend-910B",
    peak_flops=320e12,
    hbm_bw=1.6e12,
    link_bw=392e9,
    link_latency=8e-6,
    mem_gb=64.0,
    scale_out_bw=25e9,
    notes="HCCS ~392GB/s; 64GB HBM. VERIFY: SKU/dtype dependent.",
)

PRESETS = {h.name.lower(): h for h in (H100_SXM, A100_80, ASCEND_910B)}


def get(name: str) -> Hardware:
    key = name.strip().lower()
    for k, v in PRESETS.items():
        if key == k or key in k:
            return v
    raise KeyError(f"unknown hardware {name!r}; available: {list(PRESETS)}")
