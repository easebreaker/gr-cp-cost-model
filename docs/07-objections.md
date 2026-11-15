# 07 · Objections, and short answers

Written for the questions that actually come up in a group meeting.

---

### "Isn't this just profiling plus a fit?"

The fit is five numbers on a one-dimensional curve. Everything else — the pair
counts, the imbalance, the FLOPs, the collective volume, the memory limit — is
derived and unit-tested against brute force. The test of that claim is the
ablation: remove the derived parts and see the error go up. It does, by 3–8×
(`04-validation.md` §4).

If the answer were "profiling plus a fit", the fit would need a free parameter
per configuration. It needs five, total, for all configurations of all three
models.

### "Why not just sweep the grid?"

96 points vs 12. But the real answer is that the sweep does not answer the
question. You want to know what happens at `T = 12288` on a machine you have not
bought yet, or with a kernel you have not written. A fit to the grid tells you
nothing about either; a model with a derived `T`-dependence does.

Vidur's number is the useful precedent: deployment-based exploration of one
LLaMA-70B configuration costs 42K GPU-hours; the model-based search takes an
hour on CPU.

### "GR sequences are only a few hundred tokens. Isn't CP pointless?"

At MTGR's reported operating point (`T ≈ 520`, inverted from their GFLOPs) that
is close to right, and the model says so out loud: attention is 18% of the work
and the Amdahl ceiling at CP=2 is 1.10×.

Two things keep CP relevant anyway:

- **The trend.** LONGER, MTGR and the HSTU line all report quality improving with
  length; the whole point of CP is to make the next length feasible. Predicting
  the *crossover* is the contribution, not predicting today's point.
- **Memory, not time.** CP's first-order benefit is fitting a longer sequence at
  all. That is why the memory model in `memory.py` is treated as a first-class
  deliverable rather than a footnote.

Also this makes the model's job interesting rather than trivial: if CP is
*not* paying off for compute reasons, saying precisely why (efficiency, not
Amdahl) is the useful result.

### "Your `eff(c)` is a fudge factor."

It is a semi-empirical curve, which is standard practice: Habitat calls the same
idea `γ` (memory-boundedness), KnapFormer calls it a correction factor `γ`, and
both fit it per architecture. What distinguishes it here:

- it is a function of one variable, `c = T/CP`, which is the variable CP changes;
- `c_star` has a physical anchor (`F_peak / BW_link`), so a fit that violates it
  is a red flag rather than a result;
- its removal is the single largest ablation effect (3.0% → 59.5%), which is the
  honest way to state how much of the model is empirical.

The claim is not "we model kernels". The claim is "we localise the empirical part
to one dimension, so calibration is cheap and transferable".

### "How is this different from KnapFormer?"

KnapFormer's workload model is
`t = k(24·l·d² + γ·4·l²·d)` with `γ` fitted per GPU, used for a knapsack
redistribution of tokens in *data-parallel* DiT training. This package replaces
the `4l²d` term with an exact pair count over a three-region mask, adds CP as an
axis (assignment scheme, imbalance, per-layer collectives), and separates the
memory question. The overlap is real and should be cited prominently; the
difference is "derived mask algebra + CP" versus "fitted quadratic correction +
DP load balancing".

### "Isn't the 1.33× number just Amdahl? You can't beat that."

Amdahl is an *upper* bound, so 1.33 < 1.10 is not a paradox — it is a
contradiction, and that is the point. Either the operating `T` is ~4× larger than
the GFLOPs table implies, or the attention share is higher than a 5d-crossover
predicts (a bigger static block does that), or the GFLOPs are counted differently.
The model makes the inconsistency visible; one measurement (attention fraction by
layer ablation, or NCU) resolves it. See `scripts/check_cp_table.py` §3.

Stating a contradiction between two published numbers, and naming the one
experiment that resolves it, is a legitimate contribution in itself.

### "Where does the data come from? You have no measurements."

Correct, and that is why the repository's demo is labelled synthetic and why
`docs/04-validation.md` is written as a protocol rather than a results section.
What exists is: exact derived quantities (tested), a fitting procedure with an
identifiability check, a validation protocol with leakage controls, and a 12-job
design that fits in ~2 GPU-hours. The remaining work is running it.

### "What if the static-block layout is different in production?"

Then `static_share` is the wrong number and the imbalance prediction changes —
but nothing else does, because the layout is an input to the pair count, not an
assumption baked into the model. `mtgr_layout()` exposes `static_share`,
`cand_tokens`, `cand_block`, `cand_sees_prefix` and `static_full`; the last one
matters most: if the static region is encoded as a *sequence* rather than a
*block*, the imbalance story flips back to the causal case. That is a
one-line change and a one-job test.

### "Does it handle ragged/jagged inputs?"

Partially, and honestly. `T` is the per-sample concatenated length; raggedness
enters through (a) the memory model (which is per-token), (b) the
max-over-ranks rule for a synchronous step, and (c) `t_fixed` for the jagged
gather/scatter overhead. It does **not** model per-sample load-balancing as a
scheduling problem — that is what FreeScale/KnapFormer/Zeppelin do, and they
should be cited as the complementary layer.

### "Why is `t_fixed` not fitted?"

Because it is CP-independent and large. A free `t_fixed` absorbs exactly the
variance you are trying to explain, and the fitted value will be wrong whenever
the batch changes. It is measured with two ablation runs (backbone no-op;
optimizer off) and inserted per row. This is the mistake that produces beautiful
in-sample fits and useless extrapolations.

### "Is the Ascend preset trustworthy?"

No. It is there because the break-even chunk (`c* ≈ 816` tokens) is the number
that frames the search space, and it is roughly right in ratio terms. The
absolute TFLOPS figures vary a lot by SKU and dtype; replace them with your own
before quoting any absolute time. The package says so at the definition site.

### "What's the single most likely reason this fails?"

That `eff(c)` does not transfer across kernels even after refitting its three
parameters. Then "calibrate once" becomes "calibrate once per kernel", the
calibration cost triples, and the contribution shrinks to a kernel-aware
attribution study. Still worth publishing, much less exciting. Test it with job
9 in `03-calibration.md` before building anything on top.
