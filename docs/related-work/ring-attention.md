# Ring Attention

**arXiv 2310.01889** · Liu, Zaharia, Abbeel · 2023

## One line
Sequence-parallel attention by circulating K/V blocks around a ring, overlapping
each transfer with the block's compute. The source of the single most useful
number in CP planning.

## Mechanism
Each host holds one sequence block, computes local Q, K, V, then in `N−1` steps
sends its K/V block to the next host and receives from the previous one, doing
blockwise attention each round. Memory per host is `O(c)` for the block plus
`O(c²)` for the attention tile, independent of total length.

## The rule
Communication can be hidden behind compute when the per-block compute exceeds the
per-block transfer:

```
c  ≳  c* = F_peak / BW_link
```

| hardware | `c*` |
|---|---|
| A100, NVLink (~300 GB/s) | ~1,000 tokens |
| H100, NVLink (~450 GB/s) | ~2,200 tokens |
| 100 Gb/s InfiniBand (~12.5 GB/s) | ~25,000 tokens |

## What to take
- **`c* = F/BW` is the frame for the whole search space** →
  `hardware.py::Hardware.break_even_chunk`, printed by
  `python -m grcost break-even`. It needs no profiling and immediately bounds
  which CP degrees are worth trying (`CP ≲ T/c*`).
- **Ring and all-gather move the same volume**; they differ in latency, overlap
  and memory. So the choice between them is an α/β/memory argument, not a
  bandwidth one — which is why the collective is a model *dimension*
  (`scheme`/collective columns), not a constant.
- **Consequence for cross-node CP**: `c* ≈ 25K` over IB means cross-node CP only
  pays when `T` is well above 20–50K. For production GR lengths this rules it out,
  and that is a useful negative result to state early.

## Where it breaks for GR
- Only causal and full masks are well supported in practice. MTGR's
  `[static ‖ causal ‖ candidate]` mask has three regions and a block that is
  *ahead* of some of its queries (the static block), which is not causal — ring
  scheduling assumes a monotone dependency.
- The balance argument is about a triangular mask; a static rectangular block
  changes the per-round work profile, which is precisely the effect modelled in
  `spec.py` and measured in `test_static_prefix_damps_imbalance`.
- Jagged inputs: `c` is per-sample, so `c*` must be compared against the
  *distribution*, and the tail decides.

## Used here as
The break-even rule (`hardware.py`) and the statement that communication is
`O(n_layers · (N−1)/N · B·T·d)` bytes, not `O(1)` per step.
