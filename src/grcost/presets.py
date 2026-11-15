"""Concrete model presets and layout builders.

Numbers are taken from the papers:

* MTGR  (arXiv 2505.18654, Table 2): small n_layer=3 d=512 heads=2,
  medium 5/768/3, large 15/768/3, reporting 5.47 / 18.59 / 55.76 GFLOPs per
  example.  Those three numbers are reproduced by this package's FLOPs counter
  at a single sequence length -- see ``scripts/validate_mtgr_flops.py``.
* HSTU  (arXiv 2402.17152): same block, no FFN.
* LONGER (arXiv 2505.04421): token merge K=4, one long cross-attention layer,
  query = global tokens + most recent k (=100) behaviours.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .spec import HstuModel, Layout, LongerModel

# --------------------------------------------------------------------------
# Layouts
# --------------------------------------------------------------------------


def hstu_layout(T: int, n_context: int = 0) -> Layout:
    """Pure causal over the whole sequence (contextual tokens included)."""
    return Layout.causal(T)


def mtgr_layout(
    T: int,
    cand_tokens: int = 40,
    static_share: float = 0.85,
    cand_block: bool = False,
) -> Layout:
    """MTGR ``[U, S, R, [C,I]_1 .. [C,I]_K]``.

    ``static_share`` is the fraction of the non-candidate sequence that is
    static user/profile/history (rectangular), the rest being the real-time
    causal part.  This single number is the main knob for how *balanced* CP
    will be, so it is worth measuring in production logs.
    """
    core = max(0, T - cand_tokens)
    n_static = int(round(core * static_share))
    n_causal = core - n_static
    return Layout(n_static=n_static, n_causal=n_causal, n_cand=cand_tokens,
                  cand_block=cand_block)


LAYOUTS = {
    "causal": lambda T: Layout.causal(T),
    "hstu": hstu_layout,
    "mtgr": mtgr_layout,
    "mtgr_block": lambda T: mtgr_layout(T, cand_block=True),
    "mtgr_70": lambda T: mtgr_layout(T, static_share=0.70),
    "mtgr_95": lambda T: mtgr_layout(T, static_share=0.95),
}


def resolve_layout(name: str, T: int) -> Layout:
    key = name.strip().lower()
    if key in LAYOUTS:
        return LAYOUTS[key](T)
    # allow "mtgr:40:0.85" style overrides
    if key.startswith("mtgr:"):
        _, cand, share = key.split(":")
        return mtgr_layout(T, cand_tokens=int(cand), static_share=float(share))
    raise KeyError(f"unknown layout {name!r}; available {sorted(LAYOUTS)}")


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
MODELS: Dict[str, object] = {
    "hstu-base": HstuModel("hstu-base", d=768, n_layers=15),
    "hstu-small": HstuModel("hstu-small", d=512, n_layers=6),
    "mtgr-small": HstuModel("mtgr-small", d=512, n_layers=3),
    "mtgr-medium": HstuModel("mtgr-medium", d=768, n_layers=5),
    "mtgr-large": HstuModel("mtgr-large", d=768, n_layers=15),
    "longer": LongerModel("longer", d=256, n_layers=12, merge=4, n_global=2,
                          n_recent=100),
}


def resolve_model(name: str):
    key = name.strip().lower()
    if key in MODELS:
        return MODELS[key]
    raise KeyError(f"unknown model {name!r}; available {sorted(MODELS)}")


def list_models() -> List[str]:
    return sorted(MODELS)
