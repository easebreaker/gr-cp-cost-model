# 05 · Related work, and the gap this fills

A curated map, not a bibliography. For each entry: what it establishes, the
number worth quoting, **what you can take**, and **where it breaks for CP-sharded
GR**. Per-paper notes live in `docs/related-work/`.

---

## 1. The gap, stated precisely

Two literatures bracket this problem and neither covers it.

| | CP/sequence-parallel literature | Performance-modelling literature |
|---|---|---|
| representative | Ring Attention, Striped, DistFlashAttn, USP, **CP-on-HSTU** | Vidur, Habitat, roofline models, KnapFormer |
| assumes | causal or full masks; dense, fixed-length sequences | operator shapes determined by `(B, L, d)`; kernel fixed |
| breaks on GR because | GR masks are `[static ‖ causal ‖ candidate]` and inputs are jagged; static blocks are the *majority* of the sequence | CP changes the per-rank tile shape, and kernel choice alone is a 4× effect |

Explicitly:

- **The dedicated CP-for-GR paper is a capability paper.** "Scaling Generative
  Recommendations with Context Parallelism on HSTU" (RecSys'25) makes jagged CP
  work and reports 5.3× longer sequences and 1.33–1.55× scaling. It does **not**
  offer a predictive model, does not vary the model, and does not decompose the
  loss. Its own headroom study ("fusing the segmented kernels could take 1.33 to
  ~1.5–1.6") is a *statement that the loss is kernel efficiency*, unquantified.
- **The performance-modelling papers have no CP at all.** Vidur models TP and PP,
  not CP; its operator set does not include a sequence-sharded attention with a
  collective per layer. Habitat scales across GPUs, not across parallelism
  degrees. The roofline/config-search papers (`2410.00273`, `2404.12674`) search
  over TP/PP/DP/ZoD.

So the contribution slot is: **an analytic, structure-aware cost model with CP as
a first-class axis, calibrated once on a small number of jobs, that transfers
across GR models — plus a decomposition of where the CP penalty goes.**

## 2. The models being predicted

### HSTU — `arXiv 2402.17152`
Pointwise (non-softmax) attention with relative position+time bias
`SiLU(QKᵀ + rab^{p,t})V`, gated by `U`, fused UVQK projection, **no FFN**.
Take: the 3-sublayer block; the `10d²` per-token dense cost; the fact that
attention is *not* the only place a denominator lives (layer norm after pointwise
pooling).
Breaks: nothing — but the no-FFN structure is the reason a naive LLM cost model
is off by ~2.4× at the crossover.

### MTGR — `arXiv 2505.18654` (CIKM'25)
HSTU-based, keeps DLRM cross features, user-level compression (all `K`
candidates of one user in one sequence), Group-LayerNorm, dynamic masking
`[U, S, R, [C,I]×K]`, jagged/padding-free.
Numbers: 65× the FLOPs of the DLRM baseline, 1.6–2.4× throughput over TorchRec,
10–100× model compute without proportional cost, 5.47/18.59/55.76 GFLOPs per
example for small/medium/large.
Take: **the layout is the whole cost model** — it sets both the pair count (1.7×
causal) and the imbalance. Also: the three GFLOPs numbers are a free validation
of any FLOPs counter (`scripts/validate_mtgr_flops.py`).
Breaks: for a cost model, `T` is not `|S|`; it is the concatenation, and the
candidate tail is part of every query.

### LONGER — `arXiv 2505.04421` (ByteDance)
Global tokens, token merge (K adjacent → `Kd` → `d`), InnerTrans, then a hybrid
where layer 1 uses `m+k` queries (k = 100 recent) against the merged full
sequence and the remaining layers run on the short `m+k` sequence.
Numbers: k=100 keeps >95% of full-query quality at ~half the FLOPs; 43% attention
FLOPs reduction at L=2048, d=32, K=4.
Take: it is the **Amdahl limit case**, and a near-perfect unit test for a comm
term — CP helps one layer, communication is tiny and length-independent.
Breaks: any model that assumes every layer is quadratic will over-predict CP's
benefit by a lot.

## 3. Context parallelism: mechanisms and their cost structure

| work | mechanism | number | take | break |
|---|---|---|---|---|
| Ring Attention `2310.01889` | ring P2P of K/V blocks, overlapped | hide comm when `c ≳ F/B` (~1K tokens A100 NVLink, ~25K IB) | **the break-even rule** | assumes causal/full masks |
| Striped Attention `2311.09431` | token `i` → device `i mod N` | 1.41–1.45× @256K; 1.65× @786K/16 TPUv4; theory max 2× | **`Work(i,j)=c(c±1)/2`, so imbalance is a closed form** | needs token permutation; tile granularity eats the gain at small `c` |
| DistFlashAttn `2310.03294` | work stealing | — | predict *residual* imbalance after balancing, not ideal balance | schedule-dependent |
| Megatron SP `2105.13120` | all-gather K/V | fwd `2(N−1)BZ(L/N)A` | comm-volume formula, directly reusable | dense fixed-length |
| USP `2405.07719`, MaxText | all-gather / A2A / P2P / hybrid + selection rule | 70% MFU @4M tokens TPU5p | **collective choice is a model dimension, not a constant** | GQA/MLA assumptions; GR has 1–4 heads, `d_h` 32–256 |
| **CP-on-HSTU `2508.04711`** | jagged CP: AllToAll instead of AllGather, 2·CP mini-chunks, async offsets | 5.3× seq len; 1.33–1.55× scaling; AllToAll 2.7× QPS and −60% peak mem; mini-chunks +37%; `.item()` removal +2% | **the baseline, and the four numbers to explain** | no model, no decomposition, single model, one card type |

Two facts worth internalising from this table:

1. **Load imbalance is analytically known** — triangular work, `2 − 1/CP` for
   block assignment — so it should never be a fitted parameter. What *is*
   uncertain is whether the balancer's residual imbalance is small.
2. **Collective choice changes the α–β constants by a lot** (all-gather vs ring
   vs all-to-all), so the model needs the scheme as an input, not a constant.

## 4. Performance modelling: what has already been proven

| work | method | reported accuracy | take |
|---|---|---|---|
| Vidur `2405.05465` | per-op profile + learned predictor + discrete-event sim; **TP/PP only** | <9% e2e, 3.33% P95; search 42K GPU·h → 1 CPU·h | the "why bother" argument, and the value of config search |
| Habitat (ATC'21) | measure once, scale per-op by wave scaling + roofline memory-boundedness `γ` | 11.8% mean | **`γ` is the ancestor of `eff(c)`**; and "calibrate cheaply, transfer" |
| `2410.00273` | FLOPs + bytes + comm, brute-force parallel-config search | — | the full roofline+comm formulation |
| `2404.12674` | universal per-kernel models, multi-GPU | — | kernel-level tables beat pure roofline |
| KnapFormer `2508.06001` | knapsack token redistribution + semi-empirical workload model | <1% load discrepancy; 2–3× | **`t = k(24ld² + γ·4l²d)` is nearly this package's form**; the increment is replacing fitted `γ` with a derived pair count |
| AdaptiveLoad `2605.17923` | dual constraint `B·S^p ≤ M` | latency vs tokens R=0.35, vs `B·S^p` R=0.92; imbalance 39→18.9% | **the empirical case for a quadratic term** |
| Zeppelin `2509.21841` | length-aware partition; quadratic vs linear want different splits | — | compute/comm ratio varies with length |
| TIE scheduler (PMLR v306) | output length ~ log-t, tail-aware metric | 2.31× lower per-token latency | model the **tail**, not the mean, for a synchronous step |

Honest counter-evidence, worth citing against yourself: public sim-to-real
comparisons have found pure roofline and Vidur-style simulators can be *badly*
wrong out of distribution (missing prefix caching → hundreds of percent error),
while a trained cross-model stayed near 12%. The lesson is not "don't model"; it
is **"analytic skeleton + a small learned residual, and never extrapolate a
kernel model outside its family."** That is exactly what `eff(c)` is.

## 5. Systems work on GR training (where the time actually goes)

| work | finding | why it matters here |
|---|---|---|
| NVIDIA recsys-examples HSTU | `T = 2S + C = 4099` at `B=32`; 7.65% MFU baseline → 34.19% after optimisations; shuffler 1.53×, CUTLASS 4× | **kernel choice alone is 4×**; and the reference shape for `T` |
| TurboGR `2605.13433` (Ascend) | baseline MFU <10%; inter-device imbalance 47% → 2.4%; jagged fusion 2.2×; 94% NPU util | the imbalance term is not academic; and Ascend needs its own constants |
| MtgrBoost `2505.12663` | SRAM tiling for the fused HSTU forward | kernel-level detail the model abstracts into `eff` |
| FreeScale (MLSys'26) | 90.3% of compute bubbles removed on 256 H100 | bubbles ≈ `t_fixed` + imbalance; they belong in the model, not the residual |
| LinkedIn GR training | end-to-end GPU-hours −65%; warm-start training 25.5 h → 9 h | the dataloader/lifecycle part dwarfs some CP gains |
| DLRM on GPUs `2410.22249` | embedding stage is the bottleneck, 3.2× slowdown vs cached | why `t_fixed` cannot be ignored; sparse and dense parts scale differently |

## 6. What this repository does **not** claim

- Not a simulator: no event loop, no scheduler, no overlap timeline.
- Not a kernel model: `eff(c)` is a scalar curve, not a roofline per instruction
  mix. It will not tell you *why* a kernel is slow.
- No inference/decode model. Prefill/training only.
- No sparse/embedding model beyond a measured `t_fixed`.
- No claim about model quality. `06-from-model-to-topic.md` discusses composing
  with a scaling law; that composition is a proposal, not a result.

## 7. Reading order

1. `related-work/cp-on-hstu.md` — the baseline you must beat or complement.
2. `related-work/striped-attention.md` — the closed-form imbalance, and the
   reason block assignment is the worst case.
3. `related-work/knapformer.md` — the closest existing formula; see exactly what
   is new here.
4. `related-work/vidur.md` — methodology and the "why bother" number.
5. `related-work/mtgr.md` — the layout that makes this problem different.

## 8. Citation list

```
HSTU            arXiv 2402.17152   Zhai et al., "Actions Speak Louder than Words"
MTGR            arXiv 2505.18654   Han et al., CIKM'25
LONGER          arXiv 2505.04421   Deng et al., ByteDance
CP-on-HSTU      arXiv 2508.04711   Dong et al., RecSys'25
Ring Attention  arXiv 2310.01889   Liu et al.
Striped         arXiv 2311.09431   Brandon et al.
DistFlashAttn   arXiv 2310.03294   Li et al.
Megatron SP     arXiv 2105.13120   Li et al.
USP             arXiv 2405.07719   Fang & Zhao
Vidur           arXiv 2405.05465   Agrawal et al.
Habitat         USENIX ATC'21      Yu et al.
Perf. modeling  arXiv 2410.00273   Subramanian et al.
Universal perf  arXiv 2404.12674   Wang et al.
KnapFormer      arXiv 2508.06001   Hu et al.
AdaptiveLoad    arXiv 2605.17923
Zeppelin        arXiv 2509.21841
SlimPack        arXiv 2509.26246
FreeScale       MLSys'26
TurboGR         arXiv 2605.13433
MtgrBoost       arXiv 2505.12663
DLRM            arXiv 1906.00091   Naumov et al.
DLRM on GPUs    arXiv 2410.22249   Gupta et al.
```
