# Habitat — measure once, scale the kernels

**USENIX ATC'21** · Yu et al. · 2021

## One line
Measure one training iteration on the GPU you have, scale each operator to the
GPU you are considering, and predict the iteration time there. Average error
11.8% across six architectures. The ancestor of the `eff(c)` idea.

## Method
1. Run one iteration on an existing GPU; record per-operator time.
2. For a target GPU, scale each operator by either
   - **wave scaling** — a hardware-execution-model argument based on how many
     waves of thread blocks the kernel launches, or
   - **a pretrained MLP** where the kernel differs between GPUs (e.g. conv).
3. Sum to get the iteration time, then extrapolate to throughput and cost.

Wave scaling needs one coefficient per kernel: **`γ`, the "memory-boundedness"**
of the kernel, obtained from the roofline model's arithmetic intensity.
`γ = 0` means compute-bound, `γ = 1` means bandwidth-bound, values in between
blend the two scaling ratios.

## What to take
- **`γ` is `eff`'s ancestor, and the framing is the same**: do not model the
  kernel, model *how much it slows down along one axis*. Here the axis is the
  chunk length `c`; there the axis is the GPU.
- **The measurement protocol** (one iteration, per-operator, then scale) is
  exactly what `t_fixed` requires in `docs/03-calibration.md` §2.
- **11.8% is the realistic target** for a cross-hardware, per-op model. Anything
  in the single digits on held-out GR configurations would be a good result; the
  synthetic demo reaches 3% precisely because it has no hardware mismatch.
- **Wave quantisation** is the mechanism behind small-`c` inefficiency: fewer
  rows per GEMM → fewer blocks → a partially-filled final wave. That is why
  `eff(c)` must roll off, and why `c_star` is on the order of thousands of tokens
  for `d` in the hundreds.

## Where it breaks for GR/CP
- Scales *across GPUs*, not *across parallelism degrees*; `c` is not a variable.
- Per-kernel MLPs require a kernel-identification step, which is awkward for
  fused fused-UVQK/gated-attention kernels.
- It assumes the same source-level operator maps to a comparable kernel on the
  target. With Triton vs CUTLASS HSTU attention that assumption fails (4× gap), so
  `kernel` must be a calibration column, not a lookup.

## Used here as
The two-coefficient ("compute ratio, bandwidth ratio") design of `predict`, the
pinning of `bw_util` from a microbenchmark, and the expectation setting for
accuracy targets.
