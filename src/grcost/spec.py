"""Mask topology, CP chunk assignment, and model specifications.

The central idea of this package: for GR backbones the *amount of work* is not
something you fit, it is something you derive from the attention mask topology
and the sequence layout.  Only the *efficiency* with which that work turns into
time is fitted.

Everything here is integer arithmetic, so it is cheap enough to sweep thousands
of (model, T, B, CP) points in a second.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

Range = Tuple[int, int]


def _split(total: int, n: int) -> List[Range]:
    """Split [0, total) into n contiguous near-equal chunks."""
    base, rem = divmod(total, n)
    out, start = [], 0
    for i in range(n):
        end = start + base + (1 if i < rem else 0)
        out.append((start, end))
        start = end
    return out


# --------------------------------------------------------------------------
# Mask topology
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Layout:
    """Visibility topology of one logical sequence.

    Positions are laid out as ``[ static | causal | candidate ]``:

    * ``static``    a *rectangular* block: each of these tokens sees the whole
                    static block (that is what a precomputed user profile or an
                    offline history summary looks like);
    * ``causal``    a lower-triangular block;
    * ``candidate`` tokens that see the prefix and (optionally) each other.

    Why this covers HSTU / MTGR / LONGER:

    * HSTU        -> ``static=0, causal=T``                  (pure causal)
    * MTGR        -> ``static=U+S, causal=R, candidate=K*(C+I)``
    * LONGER p2   -> ``static=0, causal=m+k``                (causal on short seq)

    Two algebraic facts carry the whole package:

    1. A *causal* block of length L contributes a triangular number ``L(L+1)/2``
       of pairs; a *static* block contributes a perfect square ``S^2``.  Both are
       closed forms, so the quadratic coefficient is derived, not fitted.
    2. A static block is what *flattens* CP load imbalance: its rows all carry
       the same number of keys, which raises the floor on early queries.  With a
       purely causal mask the block assignment is maximally unbalanced
       (``max/mean -> 2 - 1/CP``); with a large static prefix it approaches 1.
       That is a real, testable difference between GR and an LLM decoder, and
       ``static_share`` is the single knob that controls it.

    ``static_full=False`` keeps the block triangular instead (equivalent to
    folding it into the causal triangle).  That degenerate case is what you get
    if the static tokens are re-encoded as a sequence rather than as a block.
    """

    n_static: int = 0
    n_causal: int = 0
    n_cand: int = 0
    cand_block: bool = False       # candidate tokens see each other
    cand_sees_prefix: bool = True  # candidate tokens see static+causal prefix
    static_full: bool = True       # static block is rectangular, not triangular

    @property
    def prefix_len(self) -> int:
        """Everything that a candidate token can see (static + causal)."""
        return self.n_static + self.n_causal

    @property
    def total(self) -> int:
        return self.n_static + self.n_causal + self.n_cand

    # -- exact per-query key counts ---------------------------------------
    def n_keys(self, q: int) -> int:
        """Number of keys visible to query position ``q``."""
        s, p = self.n_static, self.prefix_len
        if q < s:
            return s if self.static_full else q + 1
        if q < p:
            return q + 1
        base = (p + 1) if self.cand_sees_prefix else 1
        if self.cand_block:
            base += self.n_cand - 1
        return base

    def cum_pairs(self, a: int, b: int) -> int:
        """sum_{q in [a,b)} n_keys(q), in O(1)."""
        if b <= a:
            return 0
        s, p = self.n_static, self.prefix_len

        # 1) static block: a full square if rectangular, else a triangle itself
        s_lo, s_hi = a, min(b, s)
        if s_hi > s_lo:
            if self.static_full:
                pairs = (s_hi - s_lo) * s
            else:
                pairs = (s_hi * (s_hi + 1) - s_lo * (s_lo + 1)) // 2
        else:
            pairs = 0

        # 2) causal block
        c_lo, c_hi = max(a, s), min(b, p)
        if c_hi > c_lo:
            pairs += (c_hi * (c_hi + 1) - c_lo * (c_lo + 1)) // 2

        # 3) candidate block
        n_cand_q = max(0, b - max(a, p))
        pairs += n_cand_q * self.n_keys(p)
        return pairs

    def pairs_over(self, ranges: Sequence[Range]) -> int:
        return sum(self.cum_pairs(a, b) for a, b in ranges)

    def pairs_total(self) -> int:
        return self.cum_pairs(0, self.total)

    # -- constructors ------------------------------------------------------
    @classmethod
    def causal(cls, length: int) -> "Layout":
        return cls(n_static=0, n_causal=length)

    @classmethod
    def mtgr(cls, n_static: int, n_causal: int, n_candidates: int) -> "Layout":
        """MTGR layout ``[U, S, R, [C,I]_1 .. [C,I]_K]``.

        Candidate tokens see the full prefix but only themselves (dynamic mask),
        which is what prevents leakage between candidates.
        """
        return cls(n_static=n_static, n_causal=n_causal, n_cand=n_candidates)

    def describe(self) -> str:
        return (
            f"static={self.n_static} causal={self.n_causal} "
            f"cand={self.n_cand} (block={self.cand_block}, "
            f"prefix={self.cand_sees_prefix})"
        )


# --------------------------------------------------------------------------
# CP chunk assignment
# --------------------------------------------------------------------------
def assign(total: int, cp: int, scheme: str = "block") -> List[List[Range]]:
    """Return, for each CP rank, the list of query ranges (positions) it owns.

    ``block``    rank i owns one contiguous chunk (vanilla ring attention);
    ``balanced`` rank i owns chunk i and chunk 2*cp-1-i, so that the
                 triangular work of the two roughly cancels
                 (TransformerEngine / RecSys'25 style 2*CP mini-chunks);
    ``striped``  rank i owns positions i, i+cp, i+2cp, ... (Striped Attention),
                 represented as singleton ranges.
    """
    if cp < 1:
        raise ValueError("cp must be >= 1")
    if scheme == "block":
        return [[c] for c in _split(total, cp)]
    if scheme == "balanced":
        mini = _split(total, 2 * cp)
        return [[mini[i], mini[2 * cp - 1 - i]] for i in range(cp)]
    if scheme == "striped":
        return [[(q, q + 1) for q in range(r, total, cp)] for r in range(cp)]
    raise ValueError(f"unknown scheme: {scheme!r}")


def work_per_rank(layout: Layout, cp: int, scheme: str = "block") -> List[int]:
    """Visible (query, key) pairs handled by each CP rank."""
    return [layout.pairs_over(rs) for rs in assign(layout.total, cp, scheme)]


# --------------------------------------------------------------------------
# Model specifications
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Phase:
    """One attention phase: ``n_layers`` layers of ``q_len x kv_len`` attention."""

    n_layers: int
    q_len: int
    kv_len: int
    layout: Optional[Layout] = None   # None -> dense rectangle
    sharded: bool = True              # are the *queries* split across CP ranks?
    q_concentrated: bool = False      # all queries on one rank (LONGER phase 1)

    def pairs_per_rank(self, cp: int, scheme: str = "block") -> List[int]:
        total_pairs = (
            self.q_len * self.kv_len if self.layout is None else self.layout.pairs_total()
        )
        if not self.sharded:
            # Replicated on every rank (LONGER's short layers): no CP benefit.
            return [total_pairs] * cp
        if cp == 1:
            return [total_pairs]
        if self.q_concentrated:
            # Everything lands on a single rank: this is the row that becomes
            # the critical path, which is why LONGER's long layer cannot be
            # accelerated by CP.
            return [total_pairs] + [0] * (cp - 1)
        if self.layout is None:
            # Each rank's queries still see the full (gathered) KV.
            per = (self.q_len / cp) * self.kv_len
            return [per] * cp
        return work_per_rank(self.layout, cp, scheme)


@dataclass(frozen=True)
class HstuModel:
    """HSTU / MTGR backbone: one fused UVQK projection + gated attention.

    Per token the dense cost is ``10 d^2`` FLOPs (U,V,Q,K = 8d^2 plus the
    output projection 2d^2).  There is **no FFN**, which is what moves the
    attention/dense crossover from ``T = 12d`` (decoder transformer) to
    ``T = 5d``.  That is the single most useful structural fact here.
    """

    name: str
    d: int
    n_layers: int
    ffn_ratio: float = 0.0
    n_heads: int = 1               # 1 keeps the historical 10 d^2 / 4 d form
    attn_dim: Optional[int] = None  # per-head qk contraction width
    value_dim: Optional[int] = None  # per-head p.v contraction width

    # -- widths ---------------------------------------------------------
    @property
    def head_qk(self) -> int:
        """Per-head width of the qk^T contraction."""
        return self.attn_dim if self.attn_dim is not None else self.d // self.n_heads

    @property
    def head_v(self) -> int:
        """Per-head width of the p.v contraction."""
        return self.value_dim if self.value_dim is not None else self.d // self.n_heads

    def pair_width(self) -> int:
        """FLOPs-per-pair coefficient / 2: ``H * (qk + v)``.

        This is the number that makes the L^2 term right.  It is **not**
        derivable from ``d`` alone: the official HSTU layer uses
        ``linear_dim=32``, ``attention_dim=16``, ``H=4``, ``d=64``, so
        ``H(qk+v) = 192`` while ``2d = 128``.  Assuming ``2d`` (i.e. ordinary
        multi-head attention) under-counts the L^2 term by 1.5x on that config.
        For MTGR, ``head_dim = d/H`` and the two coincide.

        Cross-checked against the operator table in the project notes: qk^T
        MACs ``B*H*L^2*attention_dim`` and p.v MACs ``B*H*L^2*linear_dim``.
        """
        return self.n_heads * (self.head_qk + self.head_v)

    def attn_flops_per_pair(self) -> float:
        return 2.0 * self.pair_width()

    def dense_flops_per_token(self) -> float:
        """``2 d H (3*v + 2*a)`` FLOPs, plus ``4 f d^2`` if there is an FFN.

        Derived, not assumed: the fused UVQK projection has output width
        ``2H(a+v)`` and the output projection contracts ``v*H``.  With
        ``a = v = d/H`` this reduces to the familiar ``10 d^2``; for the official
        HSTU config (a=16, v=32, H=4, d=64) it gives 2*64*4*(96+32) = 65,536
        FLOPs/token = 16 d^2, not 10 d^2.
        """
        proj = 2.0 * self.d * self.n_heads * (3.0 * self.head_v + 2.0 * self.head_qk)
        ffn = 4.0 * self.ffn_ratio * self.d * self.d
        return proj + ffn

    def param_count(self) -> float:
        """Dense parameters only. Embedding tables are excluded on purpose.

        ``uvqk`` is ``d -> 2H(v+a)`` and the output projection is ``Hv -> d``, so
        with ``a = v = d/H`` this is ``5 d^2`` per layer.  That is the axis the
        quality law's ``m`` refers to; sparse tables are a different (memory)
        budget and are out of scope for this model.
        """
        uvqk = self.d * 2.0 * self.n_heads * (self.head_v + self.head_qk)
        out = self.head_v * self.n_heads * self.d
        return self.n_layers * (uvqk + out)

    def dense_flops_per_rank(self, T: int, B: int, cp: int) -> float:
        """Every layer owns its own projection, so the per-token cost is * n_layers."""
        return self.n_layers * self.dense_flops_per_token() * B * T / cp

    def crossover_len(self) -> float:
        """T at which causal attention FLOPs equal dense FLOPs (B-independent).

        attention/token = pair_width * T ; dense/token = 2 d H (3v + 2a) + FFN
        => T* = 2 d (3v + 2a) / (v + a)  ->  5d for HSTU's equal widths,
        and 12d when an FFN of ratio 4 is present.
        """
        dense = self.dense_flops_per_token()
        return dense / self.pair_width() if self.pair_width() else float("inf")

    def phases(self, T: int, layout: Optional[Layout] = None) -> List[Phase]:
        lay = layout if layout is not None else Layout.causal(T)
        return [Phase(self.n_layers, T, T, lay)]


@dataclass(frozen=True)
class LongerModel:
    """LONGER: token merge + one long cross-attention layer + short self-attn.

    Layer 1 queries only ``n_global + n_recent`` positions against the merged
    full sequence; the remaining layers run on the short ``n_global+n_recent``
    sequence *replicated on every CP rank*.  Consequence: CP buys memory, not
    time, and communication is almost independent of sequence length.  That is
    why this model makes a good unit test for the communication term.
    """

    name: str
    d: int
    n_layers: int
    merge: int = 4
    n_global: int = 0
    n_recent: int = 100
    ffn_ratio: float = 4.0

    def param_count(self) -> float:
        """Merge projection ``Kd -> d`` plus the short transformer stack."""
        merge = self.merge * self.d * self.d
        per_tok = (8.0 + 4.0 * self.ffn_ratio) * self.d * self.d
        return merge + max(0, self.n_layers - 1) * per_tok

    def pair_width(self) -> int:
        return 2 * self.d

    def attn_flops_per_pair(self) -> float:
        return 4.0 * self.d

    def dense_flops_per_token(self) -> float:
        # Only the Kd -> d merge projection touches the long sequence: 2*d^2
        # FLOPs per logical token (K*d^2 MACs per merged token * T/K tokens).
        return 2.0 * self.d * self.d

    def dense_flops_per_rank(self, T: int, B: int, cp: int) -> float:
        """Merge projection is sharded; the short transformer layers are not."""
        merge = self.dense_flops_per_token() * B * T / cp
        # standard layer: QKV 6d^2 + out 2d^2 + FFN 4*f*d^2
        per_tok = (8.0 + 4.0 * self.ffn_ratio) * self.d * self.d
        short = (self.n_layers - 1) * per_tok * B * self.short_len()
        return merge + max(0.0, short)

    def short_len(self) -> int:
        return self.n_global + self.n_recent

    def crossover_len(self) -> float:
        return float("inf")

    def phases(self, T: int, layout: Optional[Layout] = None) -> List[Phase]:
        s = self.short_len()
        kv_long = self.n_global + max(1, T // self.merge)
        short_layers = max(0, self.n_layers - 1)
        out = [Phase(1, s, kv_long, None, sharded=True, q_concentrated=True)]
        if short_layers:
            out.append(Phase(short_layers, s, s, Layout.causal(s), sharded=False))
        return out


def crossover_report(model) -> str:
    d = model.d
    t = model.crossover_len()
    if t == float("inf"):
        return f"{model.name}: long sequence is cross-attention only -> no crossover"
    return (
        f"{model.name}: dense={model.dense_flops_per_token():.3e} FLOPs/token, "
        f"attention==dense at T={t:.0f} (T/d={t / d:.2f}); "
        f"at T=4096 attention share={4096 / (t + 4096):.0%}"
    )
