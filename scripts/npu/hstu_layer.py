"""A cost-equivalent HSTU block for timing, plus the mask families.

Why a stand-in and not the real model
-------------------------------------
For a *cost* model, what matters is the **operator shapes**, not the weights.
The op sequence below is the one in the official
``SequentialTransductionUnitJagged`` layer, as verified against shapes in
``执行时间预测/HSTU核对.md``:

    uvqk    mm      (B*L, d)        @ (d, 2H(lin+attn))
    qk      einsum  bnhd,bmhd->bhnm with d = attention_dim
    silu    elementwise on (B,H,L,L)
    mask    elementwise on (B,H,L,L)
    pv      einsum  bhnm,bmhd->bnhd with d = linear_dim
    o       mm      (B*L, lin*H)    @ (lin*H, d)
    ln      layer_norm
    gate    elementwise

So timing this module with the right (d, H, attention_dim, linear_dim) tells you
what the official kernel costs, without needing fbgemm_gpu or a checkpoint.
The one thing it cannot tell you is the cost of jagged gather/scatter -- that is
measured separately by ``job_pad`` and charged to ``t_fixed``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class HstuLayer(nn.Module):
    """One HSTU block, shapes matched to the official implementation."""

    def __init__(self, d: int = 768, n_heads: int = 3,
                 attn_dim: Optional[int] = None, value_dim: Optional[int] = None,
                 dtype=torch.float16):
        super().__init__()
        self.d, self.h = d, n_heads
        self.attn_dim = attn_dim or d // n_heads
        self.value_dim = value_dim or d // n_heads
        w = 2 * self.h * (self.value_dim + self.attn_dim)   # uvqk output width
        self.uvqk = nn.Linear(d, w, bias=False, dtype=dtype)
        self.out = nn.Linear(self.value_dim * self.h, d, bias=False, dtype=dtype)
        self.norm_in = nn.LayerNorm(d, dtype=dtype)
        self.norm_attn = nn.LayerNorm(d, dtype=dtype)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """``x``: (B, L, d). ``mask``: (L, L) multiplicative, or None for causal."""
        B, L, _ = x.shape
        h, a, v = self.h, self.attn_dim, self.value_dim

        h_in = self.norm_in(x)
        p = F.silu(self.uvqk(h_in))                        # (B, L, W)
        uv, qk = p.split([2 * v * h, 2 * a * h], dim=-1)
        u, val = uv.split([v * h, v * h], dim=-1)
        q, k = qk.split([a * h, a * h], dim=-1)

        def shape(t, w):
            return t.view(B, L, h, w).transpose(1, 2)      # (B,H,L,w)

        u, val, q, k = shape(u, v), shape(val, v), shape(q, a), shape(k, a)

        logits = torch.einsum("bnhd,bmhd->bhnm", q, k)     # (B,H,L,L)
        logits = F.silu(logits)
        if mask is None:
            mask = causal_mask(L, x.device, logits.dtype)
        logits = logits * mask.to(logits.dtype)
        attn = torch.einsum("bhnm,bmhd->bnhd", logits, val)
        attn = attn.transpose(1, 2).reshape(B, L, v * h)
        gate = u.transpose(1, 2).reshape(B, L, v * h)
        return x + self.norm_attn(self.out(attn * gate))


class HstuStack(nn.Module):
    """``n_layers`` HSTU blocks, so you can time the depth you will predict.

    The single-block version is cheaper (and its ``eff`` transfers, because
    ``eff`` is a kernel property), but only if the analysis *knows* it was one
    block.  Timing the stack removes that assumption entirely.
    """

    def __init__(self, d: int = 768, n_layers: int = 1, n_heads: int = 3,
                 attn_dim: Optional[int] = None, value_dim: Optional[int] = None,
                 dtype=torch.float16):
        super().__init__()
        self.layers = nn.ModuleList(
            HstuLayer(d, n_heads, attn_dim, value_dim, dtype=dtype)
            for _ in range(n_layers)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


# --------------------------------------------------------------------------
# Mask families -- three regions, built on the host once per shape
# --------------------------------------------------------------------------
def causal_mask(L: int, device="cpu", dtype=torch.float16) -> torch.Tensor:
    return torch.ones(L, L, device=device, dtype=dtype).tril()


def region_mask(n_static: int, n_causal: int, n_cand: int, cand_block: bool = False,
                static_full: bool = True, device="cpu",
                dtype=torch.float16) -> torch.Tensor:
    """``[static | causal | candidate]`` exactly as ``grcost.spec.Layout`` defines it.

    Built explicitly rather than imported so that the mask you benchmark is
    auditable line by line against the paper's description.
    """
    T = n_static + n_causal + n_cand
    m = torch.zeros(T, T, dtype=dtype)
    if n_static:
        block = torch.ones(n_static, n_static, dtype=dtype)
        if not static_full:
            block = block.tril()
        m[:n_static, :n_static] = block
    lo, hi = n_static, n_static + n_causal
    if n_causal:
        m[lo:hi, :hi] = torch.ones(hi, dtype=dtype)
        m[lo:hi, lo:hi] = torch.ones(n_causal, n_causal, dtype=dtype).tril()
    if n_cand:
        m[hi:, :hi] = torch.ones(hi, dtype=dtype)
        m[hi:, hi:] = (torch.ones(n_cand, n_cand, dtype=dtype) if cand_block
                       else torch.eye(n_cand, dtype=dtype))
    return m.to(device)


def query_slice(mask: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """The mask rows a CP rank would own -- used to emulate CP without CP."""
    return mask[lo:hi]


def pair_count(mask: torch.Tensor) -> int:
    return int((mask != 0).sum().item())


def describe_mask(mask: torch.Tensor) -> str:
    n = mask.shape[0]
    pairs = pair_count(mask)
    return f"{n}x{n}, pairs={pairs:,}, density={pairs / (n * n):.1%}"
