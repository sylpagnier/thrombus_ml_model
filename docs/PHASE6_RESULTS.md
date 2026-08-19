# PHASE 6 RESULTS — steps 1–3 of `docs/PHASE6_HANDOFF.md` §5

Written 2026-08-15. Read alongside `PHASE6_HANDOFF.md`; §7 below lists the places this
document **corrects** it.

> **§10–13 SUPERSEDE §0–4 ON THE METRIC AND THE VERDICT.** The metric of record changed
> from median-over-time to **mean-over-time** (§10), the two-scalar Damköhler rollout was
> run and is negative (§11), and §12 explains *why* every physics arm nets zero: the score
> pays about equally for onset **order** and onset **spread**, and the AP closure buys
> spread on an axis (shear) whose ordering is wrong, so the two cancel. §13 turns that into
> a go/no-go for the ML step. Read §10–13 first; §0–4 remain as the record of the
> median-metric round.

---

## 0. HEADLINE (median-over-time round — see §10 for the current metric)

The AP closure is **confirmed as a mechanism fix and rejected as a score fix.**

On the 22 vessels that were never selected on (SEALED 8 + FIT 14), paired against the
flash arm that ships today:

| metric | delta | 95% CI | verdict |
|---|---|---|---|
| median-over-time `deploy_clot_score` | **+0.0042** | [−0.0007, +0.0091] | **not distinguishable from zero** |
| `curve_l1` (growth-curve shape) | **−0.0242** | [−0.0410, −0.0075] | **real improvement**, 14/22 better |
| onset `rho` | +0.0581 | [−0.0139, +0.1302] | not significant |
| vessels with **no onset order at all** (`rho` undefined) | **5/22 → 0/22** | — | **the flash is broken** |
| final committed mask | **identical on all 27 vessels** | — | kill criterion §9 satisfied |

`patient043`, the canonical flash vessel — all 84 wall nodes committing in one step —
goes `curve_l1` **0.1403 → 0.0213** and `rho` **undefined → +0.484**. `patient010` goes
0.1855 → 0.1150 and undefined → +0.514. The mechanism works. It just does not show up in
the primary metric, because on SEALED the flash arm already scores 0.9530 against a
perfect-onset ceiling of 0.9984 — the whole prize there is +0.045.

**Do not report this as a score win.** The honest sentence is: *a zero-learned-parameter
wall-AP closure removes the flash artefact and measurably improves the growth curve, at no
cost to the final mask and no measurable gain on the time-resolved deploy score.*

---

## 1. THE PACKS CARRY THE SURFACE SPECIES — patient007 IS NO LONGER NEEDED

Every constant in the handoff was fitted on `outputs/comsol_p007_wall.npz`, and
patient007 is SEALED (§6.1). That is now moot: the packs ship `M_log1p_nd`,
`Mas_log1p_nd`, `Mat_log1p_nd` at all 201 timesteps for every vessel. Verified
node-by-node against the export on patient007 (583/583 wall nodes matched by position):

```
Mas   pearson 0.999978   median(pack/export) 1.0000
Mat   pearson 0.999995   median(pack/export) 1.0000
ap    pearson 1.000000   median(pack/export) 1.0000
```

with `surface_cgs = expm1(nd) * 7e10` [plt/cm²] and `bulk_cgs = expm1(nd) * 2.5e14`
[plt/cm³]. `scripts/build_wall_species_cache.py` writes the wall slice of all 27
full-horizon vessels to `outputs/wall_species_cache/` in ~90 s; every fit below reads that
and **nothing is fitted on patient007**.

Cache validated end-to-end: `scripts/diag_onset_sign_test.py` reproduces off it
bit-for-bit, and the flash arm of `scripts/eval_ap_closure_protocol.py` reproduces
`scripts/diag_time_resolved_ceiling.py` exactly (p043 0.9894, p020 0.8624, train flash
mean 0.8875, SEALED 0.9342).

---

## 2. THE HANDOFF'S "consumption" IS ALGEBRAICALLY STATIC

`Sat = 1 − Mas/Minf` and `k_aa == k_as` in the config, so

```
consumption = gate * (Sat + Mas/Minf) * k_as  ==  gate * k_as
```

**identically**, until `Mas` overshoots `Minf`. Measured: the two kernels differ in the
4th decimal of pooled R² (0.7233 vs 0.7235). With `gate` and `sr` frozen at t=0 the entire
closure is therefore a **static spatial multiplier**, not the depletion feedback the
handoff describes it as.

This is not a defect — a static shear-graded multiplier is exactly what breaks the flash,
because identically-gated nodes stop having identical ODEs — but the description has to
change, because it determines what to build next (§6).

---

## 3. STEP 1 — `C` REFIT ON TRAIN (`scripts/fit_ap_closure.py`)

### 3.1 The exponent replicates, the constant does not

`q = 1` is confirmed on TRAIN and Léveque's 1/3 is decisively worse:

```
q      0.333   0.500   0.667   1.000   1.100   1.500
R2     0.5375  0.6513  0.7099  0.7235  0.7234  0.6581
```

Renewal is **linear** in wall shear — a stirred-replenishment balance, not a diffusive
boundary layer. Keep the handoff's instruction not to call it Léveque.

`C`, however, does not replicate: pooled over TRAIN at the handoff's kernel, **C = 258.7,
R² = 0.7233** — against patient007's C = 68, R² = 0.9041.

### 3.2 …and that disagreement is a weighting artefact, not a cohort disagreement

Refitting `C` on disjoint slices of the horizon (TRAIN, 19 vessels):

| kernel | C[0–25%] | C[25–60%] | C[60–100%] | drift | pooled R² |
|---|---|---|---|---|---|
| `static` (= handoff) | 141.2 | 243.1 | 377.4 | **2.67×** | 0.7235 |
| `mat_linear` a=0.10 | 124.9 | 173.8 | 205.4 | 1.64× | 0.7748 |
| **`mat_linear` a=0.30** | **101.3** | **110.9** | **106.2** | **1.09×** | **0.7842** |
| `mat_linear` a=1.00 | 61.0 | 49.1 | 39.4 | 1.55× | 0.7679 |
| `sat_plus_mat` | 90.2 | 61.5 | 43.8 | 2.06× | 0.7482 |

A static kernel's `C` moves **2.67×** depending on which part of the run you weight. That
is the whole of the 68-vs-258 gap: they are one measurement under two weightings, and
neither is "the" constant.

A correctly specified kernel must recover the same `C` on any window. That criterion —
which never looks at R² — picks

```
consumption = gate * k_as * (1 + 0.3 * Mat/Minf)
```

drift 1.09×, and the pooled R² is at its plateau. Physically: the AP sink grows with the
mature deposit but **sub-linearly**, as an ageing clot buries its own reactive surface.
This kernel is new here; it is not in the handoff.

### 3.3 Kill criterion §9: `C` spread across vessels — borderline, survives

19 TRAIN vessels: median 50.9, IQR [31.3, 72.3], **max/min 16.3×, log₁₀ sd 0.311**. Using
the pooled `C` instead of each vessel's own costs median R² 0.492 → 0.357. Not the clean
pass the handoff hoped for, not the wild variation that kills it. The one conditioning
signal that exists is `spearman(log₁₀ C, median wall shear) = −0.595`; nothing else
(`frac_gated` +0.02, `u_ref` −0.14, `d_bar` +0.14) is informative.

### 3.4 Smoothing the sink does **not** help — which matters for §6

Mesh-averaging the consumption term over 1/2/4/8 hops monotonically *reduces* pooled R²
(0.7235 → 0.7063 → 0.6962 → 0.6878). The non-locality in `ap` is **not** a diffusive smear
of the local sink. A naive smoothing baseline is therefore not the competitor for a graph
model; ridge on features is (§9 of the handoff still stands).

---

## 4. STEP 2 — THE CLEAN PROTOCOL (`scripts/eval_ap_closure_protocol.py`)

FIT 14 / DEV 5 / SEALED 8, disjoint, positional split identical to
`scripts/sweep_temporal_only.py`. `C` fitted on **FIT only**. Kernel, `C` scale and
`da_scale` selected on **DEV only**. Truncated (`T<150`) and zero-GT vessels excluded
(§6.2).

### 4.1 DEV could not separate five configurations

```
DEV TIE at median 0.9389 across 5 configurations:
   sat_plus_mat C=51.14  da=40   curveL1 0.0389   rho +0.775
   mat_linear   C=49.91  da=40   curveL1 0.0399   rho +0.777
   clip1_gate   C=62.70  da=40   curveL1 0.0412   rho +0.788
   static       C=62.42  da=40   curveL1 0.0412   rho +0.788
   sat_plus_mat C=25.57  da=40   curveL1 0.0432   rho +0.790
```

A 5-vessel DEV median is too coarse to choose a kernel. Rather than let python's loop
order decide, the tie is declared, broken by the stated secondary metric (lowest DEV
`curve_l1`), and **every tied configuration is carried to SEALED and reported as a band**:

```
kernel        C       | SEALED score  delta | curveL1  delta
sat_plus_mat  51.14   |     0.9530  +0.0000 |  0.0687  -0.0231
mat_linear    49.91   |     0.9530  +0.0000 |  0.0720  -0.0197
clip1_gate    62.70   |     0.9530  +0.0000 |  0.0752  -0.0166
static        62.42   |     0.9530  +0.0000 |  0.0752  -0.0165
sat_plus_mat  25.57   |     0.9530  +0.0000 |  0.0793  -0.0125
```

The score is **flat to four decimals across the entire tie**, and equal to the flash arm's
0.9530. The curve improves under every one of them. The conclusion does not depend on the
tie-break, which is the point of reporting the band.

Note the DEV-selected deployment `C` (≈50–63) lands near patient007's 68 and ~4× below the
pooled least-squares fit — consistent with §3.2: onset is decided in the early, lightly
depleted regime, which is where the small `C` is right.

### 4.2 Per-vessel, SEALED

| vessel | flash | closure | oracle | cL1 flash → closure | rho flash → closure |
|---|---|---|---|---|---|
| patient001 | 0.9642 | 0.9642 | 1.0000 | 0.0729 → 0.0756 | +0.983 → +0.983 |
| patient007 | 0.9517 | 0.9514 | 0.9887 | 0.0950 → 0.1334 | +0.306 → **+0.580** |
| patient010 | 0.9524 | 0.9524 | 1.0000 | 0.1855 → **0.1150** | **nan → +0.514** |
| patient013 | 0.9536 | 0.9536 | 1.0000 | 0.0563 → **0.0329** | +0.816 → +0.821 |
| patient014 | 0.9894 | 0.9916 | 0.9968 | 0.0335 → 0.0356 | +0.991 → +0.992 |
| patient031 | 0.9304 | 0.9304 | 0.9921 | 0.0471 → 0.0435 | +0.237 → +0.270 |
| patient042 | 0.7426 | **0.7542** | 0.7834 | 0.1034 → 0.0918 | +0.395 → +0.373 |
| patient043 | 0.9894 | 0.9886 | 1.0000 | 0.1403 → **0.0213** | **nan → +0.484** |

patient007 is the one clear regression on curve shape (0.0950 → 0.1334, overshoot) — the
same behaviour the handoff saw at C=68, so it is a property of the closure and not of the
recalibration.

### 4.3 Prize recovered

```
                    flash    closure   oracle    prize   recovered
SEALED  (8)         0.9530   0.9530    0.9984   +0.0454    +0.0000   ( 0%)
FIT     (14)        0.8635   0.8730    0.9946   +0.1311    +0.0095   ( 7%)
DEV     (5)         0.9177   0.9389    0.9878   +0.0701    +0.0212   (30%)
```

DEV was selected on; its 30% is the optimistic number. FIT and SEALED are the honest ones:
**0–7%**.

### 4.4 The mask does not move

Asserted per vessel, all 27: **identical**. This is structural, not luck — the shipped
predictor takes its mask from the two t=0 gates plus shear-admitted graph growth, and the
ODE supplies only *when*. The assertion stays in the script because that architecture
could change.

`da_scale = 40` won on DEV both with and without the closure; every larger value wrecks
`curve_l1` (0.05 → 0.39). The existing value is right and the closure did not need
rate compensation (96–97% of committed nodes still cross).

---

## 5. STEP 3 — THE DAMKÖHLER RATIO (`scripts/diag_damkohler_cohort.py`)

Refit per TRAIN vessel from the packs' own `Mas`/`Mat` (central differences; validated
against COMSOL's analytic derivatives on the export, pearson 0.992 for `d(Mat,t)`).

```
                 patient007 export      TRAIN cohort (19)
A_s / Da              31.9              median  20.7   IQR [15.7, 26.1]   max/min  4.7x
A_a / Da             140.5              median  67.6   IQR [27.6, 83.1]   max/min 14.7x
ratio A_a/A_s          4.40             median   3.07  IQR [1.97, 3.83]   max/min 26.2x
```

**The ratio is real and it is not a patient007 artefact** — median 3.07 across the cohort,
positive in 15/19 vessels. It is smaller than the 4.40 the handoff quotes, and it varies a
lot (26× across vessels).

The ablations dispose of the two obvious explanations: clipping `Sat` and including
`step2t` both move the joint fit by <0.005 R². And §D disposes of the fibrin hypothesis —
adding `d(fi,t)` as a third basis function on the export changes R² by **0.0000**
(0.8795 → 0.8795).

What remains is the cleanest statement of the anomaly, restated from the export:

```
median d(Mas,t) / J0_Mas =  25.75
median d(Mat,t) / J0_Mat = 145.63
```

COMSOL's own **exported fluxes** are consistent with `Da = 1e-4`; its **state
derivatives** are ~26× and ~146× larger — the same two numbers as the fitted constants.
So the discrepancy is a multiplier sitting between the flux and the state equation, it is
**not one number**, and `da_scale = 40` is the model absorbing the smaller of the two.

**This is not resolved.** What is now established: it is cohort-wide, it is ~3× not ~4.4×,
it is not `Sat` clipping, not `step2t`, and not fibrin. The next place to look is the
`.mph` surface-reaction node itself, not the export.

Consequence for timing: one global `da_scale` cannot represent two constants that differ
3×, and the autocatalytic term is the one that decides how long a node idles below `crit`.
A two-scalar `(A_s, A_a)` rollout is a cheap, zero-ML experiment that has not been run.

---

## 6. WHAT THIS MEANS FOR STEP 4 (the GNN) — the budget, measured

`scripts/diag_ap_closure_ordering.py`. With `gate` and `sr` frozen at t=0, `ap_pred` is a
deterministic monotone function of them, so **among gate==1 nodes `rank(ap_pred) ==
rank(sr)` and the closure's onset order can be nothing else.** Measured on TRAIN, on the
gate==1 set (the flash set, where the ODEs are identical):

```
rho(sr,            onset)  = -0.301     <- the closure's CEILING, and what it delivers
rho(gate*ap_early, onset)  = -0.723     <- the transport-set AP field
                                   BUDGET for a graph model: -0.422
```

That is the quantitative case for §4 of the handoff, and it is **larger** than the handoff
estimated. Two refinements to how it should be spent:

1. **The residual is not a diffusive smear** (§3.4) — mesh-smoothing the sink makes the
   fit worse. Whatever carries the non-locality is advective/upstream, so the graph model
   needs directed or flow-aware message passing, not isotropic diffusion.
2. **Predict `ap` at an EARLY time, not at `t_final`.** `ap@t_final` is
   outcome-contaminated — a node that ignited early has been consuming `ap` ever since —
   which is why the sign of `rho(ap, onset)` flips between the gate==1 subset (−0.727) and
   the full gated set (**+0.257**). The handoff quotes only the former. Supervising a GNN
   on `ap@t_final` would teach it the consequence of onset, not its cause.

Also measured, and it cuts against a `C_i` residual specifically: over the **full** gated
set, at the deployed `C`, the closure makes ordering **worse** than the bare gate
(`rho` −0.839 → −0.748), because it suppresses `ap` hardest exactly where the gate is
strongest and so cancels the gate's own signal. On the gate==1 subset the gate carries no
ordering at all and the closure is the only thing supplying one — so the two sets genuinely
disagree, and a `C_i` residual inherits both. Predicting the `ap` **field** directly, and
letting the physics consume it, is the better-posed target.

Residual locality, remeasured on TRAIN: **median neighbour-correlation 0.926** (the
handoff quotes 0.58 from patient007 alone), against `ap`'s own 0.993.

---

## 7. CORRECTIONS TO `PHASE6_HANDOFF.md`

* **§2.1 / §5.1 `C = 68`** — patient007-fitted and window-dependent. TRAIN pooled gives
  258.7 at the same kernel; the deployment value is ~50–63. All three are the same
  measurement under different time weighting (§3.2). There is no single `C` for a static
  kernel.
* **§2.1 `R2 = 0.9041`** — 0.7233 on TRAIN at the same kernel and exponent.
* **§2.1 `consumption = gate*(Sat + Mas/Minf)*k_as`** — algebraically `gate*k_as` (§2).
  Calling it a depletion feedback is wrong; it is a static shear-graded multiplier.
  Use `gate*k_as*(1 + 0.3*Mat/Minf)` instead, which is window-stable.
* **§2.2 table** — reproduced in direction, not in magnitude, and it was `curve_l1`/`rho`
  the whole time; the table has no score column and should not be read as one. p043
  `curve_l1` 0.1403 → **0.0213** here (0.0520 there); p007 still overshoots.
* **§4 "6-neighbour correlation is 0.58"** — 0.926 median on TRAIN (§6).
* **§4 suggested form `C_i = C0*exp(GNN_i)`** — measurably the wrong place to put the
  capacity (§6). Predict the `ap` field.
* **§1 Damköhler ratio 4.40** — cohort median **3.07**, spread 26× (§5). Real, but the
  point estimate was one vessel.
* **§9 "the AP closure moves the final mask → a bug"** — it structurally cannot, since the
  mask is gate-derived and the ODE only supplies timing. Asserted anyway.

---

## 8. PROTOCOL DISCLOSURE

SEALED was read **three times** for this question, all three on disk:

1. handoff kernel set → `outputs/ap_closure/protocol_gt_run1_handoff_kernels.json`
2. after adding `mat_linear` (chosen on FIT window-stability, §3.2) — selected the *same*
   configuration, so the numbers were identical to run 1
3. after replacing the accidental loop-order tie-break with the stated
   lowest-DEV-`curve_l1` rule → `outputs/ap_closure/protocol_gt.json`

No SEALED number informed any of those changes, and the §4.1 band shows the result is
invariant across the whole tie. But the set has been opened three times and that is a
real, if small, selection risk. **Do not open it again for this question.** Arm B
(`--flow pred`, §5.5) has not been run and is still clean.

---

## 9. REPRODUCE

```bash
python scripts/build_wall_species_cache.py          # ~90 s, writes outputs/wall_species_cache/
python scripts/fit_ap_closure.py                    # step 1
python scripts/diag_ap_closure_ordering.py          # the step-4 budget
python scripts/eval_ap_closure_protocol.py --flow gt # step 2, ~4 min
python scripts/diag_damkohler_cohort.py             # step 3
python -m pytest src/tests/test_ap_closure.py -q    # 16 guards
```

| file | what it is |
|---|---|
| `src/core_physics/ap_closure.py` | the closure operator, kernels, smoother, `fit_C` |
| `src/core_physics/physics_wall_model.py` | `integrate_mat_trajectory(ap_closure=...)`; `None` is bit-identical to before |
| `scripts/build_wall_species_cache.py` | wall-slice cache + the pack↔export unit verification |
| `scripts/fit_ap_closure.py` | step 1: kernel/exponent scan, window stability, per-vessel `C`, locality |
| `scripts/diag_ap_closure_ordering.py` | how much onset ORDER the closure can carry, and what is left |
| `scripts/eval_ap_closure_protocol.py` | step 2: FIT/DEV/SEALED, the DEV tie band, mask assertion |
| `scripts/diag_damkohler_cohort.py` | step 3: `A_s`/`A_a` per vessel, ablations, the fibrin test |
| `src/tests/test_ap_closure.py` | guards, incl. "the closure breaks the flash" and the no-op check |

---

## 10. METRIC OF RECORD IS NOW MEAN-OVER-TIME

`scripts/diag_horizon_sensitivity.py` scored the shipped **final mask** against GT at every
horizon. SEALED median:

| t/T | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| score | 0.098 | 0.323 | 0.682 | 0.929 | 0.965 | **0.972** | 0.966 | 0.957 | 0.954 | 0.947 |
| GT complete | 0.00 | 0.13 | 0.56 | 0.81 | 0.88 | 0.91 | 0.93 | 0.96 | 0.98 | 1.00 |

Two things follow. First, **the 30000 s horizon is not flattering the model** — it peaks at
t/T≈0.6 and *decays* to the end, because GT keeps creeping outward after 0.6 and the frozen
mask never follows. Second, with 12 evaluation times the **median lands at t/T≈0.5, inside
the plateau**, so the old primary metric was structurally blind to the only region where
anything is wrong. Split on frozen predictions, the oracle prize is **+0.185 early
(t/T≤0.35) against +0.056 late**.

`arm_metrics` now reports **mean-over-time** as `score`, with `score_median`,
`score_early`, `score_late` alongside. DEV selection aggregates by **mean across vessels**,
not median — the previous median-of-medians produced an exact 5-way tie because a 5-vessel
median can only take five values.

Prize on the 22 never-selected-on vessels under the new metric: **+0.0990**.

---

## 11. THE TWO-SCALAR DAMKÖHLER ROLLOUT — RUN, AND NEGATIVE

`integrate_mat_trajectory(da_scale_auto=...)` splits the rate:

```
d(Mas)/dt = A_s * gate * Sat * (k_rs*rp + k_as*ap)
d(Mat)/dt = A_s * gate * Sat * (...)  +  A_a * gate * (Mas/Minf) * k_aa * ap
```

`None` is bit-identical to the shipped one-scalar model (asserted). DEV grid over
`A_s ∈ {10,20,40,80} × A_a/A_s ∈ {1,2,3,5}`:

```
   A_s=20  ratio=3.0   DEV mean 0.8584      <- the MEASURED physical constants (§5)
   A_s=40  ratio=1.0   DEV mean 0.8829      <- the shipped one-scalar model, and the winner
   A_s=40  ratio=3.0   DEV mean 0.7326      <- measured ratio at the tuned A_s: much worse
```

**DEV selected ratio = 1.0, i.e. the two-scalar model collapses back to the one-scalar
model, and the arm contributes exactly +0.0000 on SEALED.**

Two honest readings, both worth keeping:

* The measured constants are *validated but not useful*. `A_s≈20, A_a≈62` (the step-3
  cohort medians) score 0.8584 against the empirically tuned single scalar's 0.8829 — close
  enough to confirm the measurement, not better. The single `da_scale = 40` was already
  absorbing the right **effective** rate; the ratio it cannot represent turns out not to
  matter for the score.
* **My stated hypothesis was wrong.** I predicted `A_a` controls the idle time below `crit`
  and that correcting it would delay the first commitment. It does control that, but in the
  wrong direction: a faster autocatalytic term makes nodes commit *earlier*, and the early
  score collapses (0.8768 → 0.5629 → 0.5054 → 0.3824 as the ratio goes 1 → 2 → 3 → 5 at
  `A_s=40`). There is a ridge in `(A_s, ratio)` where the product is right; along it,
  ratio 1 wins.

---

## 12. WHY EVERY PHYSICS ARM NETS ZERO — the metric pays for order AND spread, equally

Four arms, **22 never-selected-on vessels**, mean-over-time:

| arm | mean | early | late | curve_l1 | rho | vs 1sc |
|---|---|---|---|---|---|---|
| 1sc physics (ships today) | 0.8515 | 0.8036 | 0.8755 | 0.1126 | +0.587 | — |
| 2sc physics | 0.8515 | 0.8036 | 0.8755 | 0.1126 | +0.587 | +0.0000 |
| AP closure | 0.8514 | 0.8041 | 0.8750 | **0.0947** | +0.583 | −0.0001 |
| closure + 2sc | 0.8514 | 0.8041 | 0.8750 | **0.0947** | +0.583 | −0.0001 |
| perfect onset | 0.9505 | 0.9890 | 0.9313 | 0.0730 | +1.000 | **+0.0990** |

The closure improves the curve substantially and the score not at all. Timing diagnostics
say why — onset `spread_ratio` (model IQR / GT IQR, GT = 1.0):

```
              bias    spread_ratio   curve_l1     rho
1sc physics  -0.007       0.392        0.1075    +0.602
closure      -0.003       0.739        0.0876    +0.539
oracle       +0.000       0.897        0.0670    +1.000
```

**The model is not firing early — bias is −0.007.** It fires a step that is too *steep*,
centred about right. The closure nearly doubles the spread (0.392 → 0.739, most of the way
to the oracle's 0.897) and pays for it with ordering (0.602 → 0.539).

`scripts/diag_curve_vs_order.py` prices those two independently by degrading the **oracle**
one dimension at a time — same committed set, same onset times (TRAIN, n=19):

```
oracle (perfect order + spread)      0.9466      --
  order KEPT, spread x0.39           0.8124   -0.1342    (the model's spread)
  order KEPT, spread x0.60           0.8527   -0.0940
  spread KEPT, order DESTROYED       0.8019   -0.1448    (curve_l1 unchanged: 0.0758)
the shipped model                    0.8475   -0.0991
```

Shuffling leaves the aggregate growth curve **bit-identical** (`curve_l1` 0.0758 → 0.0758)
and still costs −0.145. So:

**Order and spread cost about the same (−0.145 vs −0.134), and the model is deficient in
both. Any mechanism that buys one at the other's expense nets zero — which is exactly what
the AP closure does.** It spreads onsets along the *shear* axis, and shear's rank
correlation with true onset on the flash set is only ≈0.30 (§6), so it injects ordering
error at the same rate it adds spread.

That is the whole explanation for §0's null result, and it is not a property of the
closure's calibration — no value of `C` fixes it, because the axis is wrong.

---

## 13. GO/NO-GO FOR THE ML STEP

The job description is now quantitative. A learned onset model must, **simultaneously**:

1. reach `spread_ratio` ≳ 0.6 (worth about +0.040 over the current 0.39 at perfect order,
   +0.134 at spread 1.0), **and**
2. hold onset `rho` **above the physics model's 0.587–0.602**. Below that it nets zero or
   negative however good the curve looks.

Condition 2 is the kill criterion, and it is the one the AP closure fails. It also rules
out the handoff's suggested `C_i = C0*exp(GNN_i)`: a residual on `C` moves nodes along the
same shear axis and inherits the same trade.

What still looks viable, in order:

* **Supervise a graph model on the early-time `ap` field** and let the physics consume it.
  The measured ordering budget is `rho` 0.30 → 0.72 on the flash set (§6) — comfortably
  above the 0.60 bar, which is the only reason to think this can clear condition 2.
  Use `ap` at ~10% of horizon, never `ap@t_final` (outcome-contaminated; the sign of
  `rho(ap, onset)` flips between subsets — §6).
* **Report `spread_ratio` and `rho` on every run**, not just the score. This round would
  have been read as "the closure does nothing" without them; it in fact does two things
  that cancel, and that is actionable where "nothing" is not.
* **Ridge on the same features first** (handoff §9). The bar it must clear is now explicit:
  `rho` > 0.60 at `spread_ratio` > 0.4.

Also unchanged from §5: the late under-coverage (score decaying 0.972 → 0.947 as GT creeps
outward past t/T 0.6) is a **mask** problem. No onset model addresses it, and it caps the
late-window arm at 0.9313 even for a perfect oracle.

---

## 14. PROTOCOL DISCLOSURE, UPDATED

SEALED has now been read **four** times for this question — three in §8, plus once for the
four-arm mean-over-time run (`outputs/ap_closure/protocol_gt_meanovertime.json`). The
fourth used a changed metric and two new mechanisms, and every selection was made on
FIT/DEV. All four readings are on disk. The set is spent for this question; arm B
(`--flow pred`) remains clean and untouched.

`scripts/diag_horizon_sensitivity.py` and `scripts/diag_curve_vs_order.py` are diagnostics
on already-frozen models and select nothing, so they spend no budget.

---

## 15. THE ML ROUND — ridge, direct onset, and why the metric was the problem

### 15.1 Ridge study: no learned model beats one physics feature

`scripts/fit_onset_direct.py`. Train FIT (14), select DEV (5), **SEALED not opened**.
Per-vessel onset rank correlation:

```
physics: gate alone          FIT +0.624 | DEV +0.850     <- the bar
physics: gate*ap_closure     FIT +0.669 | DEV +0.790
ridge alpha=0.1              FIT +0.684 | DEV +0.779
gbm n=300 d=3                FIT +0.966 | DEV +0.802     <- badly overfit
```

**Best learned − best physics on DEV: −0.048.** `PHASE6_HANDOFF` §9 fires: *the residual
does not beat ridge on the same features → do not ship the network.* Here it is worse —
nothing beats a **single feature**, the gate.

Note also that `gate` alone ranks onset at 0.62–0.85 while **the ODE it feeds produces
0.587**. The readout destroys ordering its own input already has, which is what §12–13
inferred and this measures directly.

### 15.2 Direct onset prediction, end to end: every arm negative

`scripts/eval_onset_direct_endtoend.py`, mean-over-time, mask held at `S`:

```
DEV (5)                    mean    early    curveL1     rho   spread |  vs ODE
physics ODE              0.8829   0.8768     0.0495   +0.829   1.103 |  +0.0000
ridge (absolute)         0.8668   0.8489     0.0437   +0.815   1.052 |  -0.0161
gbm (absolute)           0.8338   0.7523     0.0557   +0.622   1.369 |  -0.0492
gate rank + FIT dist     0.6884   0.3721     0.2310   +0.843   1.014 |  -0.1945
onset oracle             0.9543   1.0000     0.0422   +1.000   1.141 |  +0.0714
```

`gate rank + FIT dist` has the **best ordering of any deployable arm** (`rho` +0.843, above
the ODE's +0.829) and the **worst score by a factor of four**.

### 15.3 Why: the metric is a cliff, not a gradient

Scoring an empty prediction and the full mask against GT at each time:

```
                    GT empty      GT non-empty
predict nothing      1.0000          0.0000
predict full S       0.05-0.24       partial
```

Predicting nothing is **perfect** while GT is empty and **catastrophic** the instant GT has
one node. So the mean-over-time score is dominated by *when you first commit*, and barely
sees ordering or spread at all. Any arm that delays commitment to a physically realistic
time is empty during GT's early phase and scores 0 there — which is exactly what happened
to every oracle in §12 and every direct-onset arm here.

### 15.4 The one thing that works, and it is a single scalar

```
FIT+DEV (19)                 mean    early    curveL1     rho   spread |  vs ODE
physics ODE                0.8475   0.8188     0.1047   +0.650   0.813 |  +0.0000
ODE + GLOBAL SHIFT         0.8617   0.8540     0.1036   +0.650   0.813 |  +0.0141
ODE + shift + spread       0.8075   0.7473     0.0651   +0.652   0.923 |  -0.0401
onset oracle               0.9466   0.9932     0.0758   +1.000   1.040 |  +0.0991
```

Aligning the model's **first commit time** to GT's — one scalar per vessel, ordering and
spread untouched — recovers **+0.0141, 14% of the prize**, more than every other mechanism
tried in this whole phase combined. Additionally matching GT's *spread* costs **−0.0401**,
even though it produces the best `curve_l1` of any non-oracle arm (0.0651).

**`curve_l1` and the score are anti-correlated.** The arm with the best growth curve is the
worst scorer. That is not a modelling failure; it is the metric.

### 15.5 Where this leaves the project

1. **Do not build the GNN.** It fails the repo's own §9 gate against a one-feature baseline,
   and the end-to-end arms are negative regardless of `rho`.
2. **The stated objective and the metric of record are in conflict.** "Make the growth curve
   match GT" (`curve_l1`, `spread_ratio`) and "raise mean-over-time `deploy_clot_score`" pull
   in opposite directions, measurably. This must be resolved before more modelling — it is
   a decision about the deliverable, not an experiment.
3. If the deliverable is the **score**: the target is per-vessel *first-commit time*, a
   single scalar, and the ceiling on that route is ~+0.014 of a +0.099 prize (the rest of
   the prize is per-node ordering the metric cannot reward without also punishing spread).
4. If the deliverable is the **growth curve**: the AP closure (§4) and `ODE + shift + spread`
   already deliver it — `curve_l1` 0.1047 → 0.0651 — and the score should be dropped as the
   metric of record for this question.
5. The metric's empty-prediction cliff (1.0 → 0.0) is worth fixing on its own terms
   regardless; it makes any time-resolved score discontinuous in the model's commit time.

---

## 16. THE METRIC, REDONE — growth-count error, and the verdict on continuing

`src/core_physics/growth_count_metrics.py`, `scripts/eval_growth_count.py`.

```
growth_l1 = mean_t |n_pred(t) - n_gt(t)| / N_gt_final
```

Continuous in every onset time (guarded by a test), blind to node identity, and it measures
the stated objective directly. It replaces the overlap score **for this question only** —
the committed set is scored separately and must always be reported beside it.

### 16.1 Two thirds of the error is timing, and it is recoverable

```
                          growth_l1   worst_t   final_err   vs floor
train (19)
  shipped physics            0.1316    0.4700     -0.058     +0.0892
  + AP closure               0.1224    0.4142     -0.058     +0.0800
  + hop delay                0.1318    0.4354     -0.058     +0.0893
  + global shift (oracle)    0.1165    0.4560     -0.058     +0.0741
  COUNT FLOOR (mask)         0.0424    0.2339     -0.058     +0.0000
  onset oracle               0.0576    0.2629     -0.058     +0.0152
SEALED (8)
  shipped physics            0.1158    0.5498     -0.045     +0.0804
  + AP closure               0.1078    0.4354     -0.045     +0.0723
  COUNT FLOOR (mask)         0.0355    0.1652     -0.045     +0.0000
```

**Prize for any timing model: +0.0892 train / +0.0804 SEALED — 68% / 69% of the total
error.** The remaining ~32% is mask-size error that no onset model can touch.

### 16.2 The AP closure is a real win on this metric

`0.1316 → 0.1224` on train and `0.1158 → 0.1078` on SEALED: **−7% of total error on both
sets, same sign, ~10% of the prize.** Under the overlap score the identical arm read
−0.0001. The closure was never the problem; the metric was.

`hop delay` moves nothing here (0.1316 → 0.1318). Its +0.0040 under the old metric was an
artefact of the empty-prediction cliff, not a mechanism.

### 16.3 The shape of the error, and what to attack next

Mean committed fraction across 19 train vessels (normalised by GT final count):

```
t/T          0.0   0.1   0.2   0.3   0.4   0.5   0.6   0.7   0.8   0.9   1.0
GT          0.00  0.01  0.06  0.22  0.47  0.68  0.80  0.85  0.92  0.97  1.00
shipped     0.00  0.01  0.02  0.38  0.48  0.65  0.86  0.91  0.94  0.94  0.94
closure     0.00  0.01  0.02  0.29  0.46  0.64  0.85  0.91  0.94  0.94  0.94
```

Two defects, both legible:
1. **the front arrives too early** — +0.15 of the final count at t/T = 0.3. The AP closure
   halves this to +0.07, which is exactly the mechanism it was built for.
2. **it saturates at 0.94, not 1.00** — a 6% permanent deficit; the mask never acquires
   GT's late creep.

### 16.4 The mask is NOT as solved as the overlap score says

`final_err` per vessel on train ranges **−0.49 (p028, p012) to +0.70 (p019)** — the model
commits half as many nodes as GT on some vessels and 70% too many on others. These largely
cancel in the cohort mean (−0.058), which is why the aggregate curve above looks close while
per-vessel `growth_l1` is 0.13.

`deploy_clot_score` 0.9093 with 2-hop dilation and F0.5 tolerates that; a count metric does
not. **"The spatial problem is solved and closed" (PHASE6_HANDOFF 0) is metric-dependent**
and false under the count metric — 32% of the growth error is mask-count error.

### 16.5 Verdict: worth continuing, with the target changed

* **Yes.** 68% of the growth error is timing and it is now measurable, monotone, and
  attributable. The three null rounds in §12–15 were the metric, not the physics.
* **Ship the AP closure.** −7% on train and SEALED, consistent sign, zero learned parameters,
  final mask bit-identical.
* **Next lever: delay first commitment.** The t/T = 0.3 overshoot is the largest single
  feature of the error, and the global-shift oracle is worth +0.0151 of the +0.0892 prize
  on top of what the closure already takes.
* **Reopen the mask.** 32% of the error is per-vessel count error of −49% to +70%, currently
  invisible to the scoring of record. This is probably the larger prize and nobody has
  looked at it, because the overlap score said it was finished.

---

## 17. SHIPPED — the AP closure is now in the deployed predictor

`src.core_physics.ap_closure.SHIPPED = ApClosure(C=62.42, q=1.0, kernel="static")`,
`SHIPPED_DA_SCALE = 40.0`, pinned by a test so the constants cannot drift silently.

`scripts/predict_wall_clot.py` gains `predict_wall_onset(data, bio, flow=...)` returning
`(mask, onset, t)`, and a `--temporal` flag. The mask is unchanged and asserted so in the
CLI path. Verified on both arms:

```
patient043 --flow gt    91 nodes, deploy 0.9796 (unchanged);  50% committed by t = 9450 s
patient020 --flow pred  59 nodes, deploy 0.6061 (unchanged);  50% committed by t = 8100 s
```

patient043 previously committed all 84 nodes in a single step at t = 3000 s; it now spreads
8415 → 10920 s against GT's ~7000 → 12000 s.

### 17.1 The final CV verdict on every method tried

Leave-one-out over all 19 train vessels, 3 seeds per fold (`scripts/eval_onset_cv.py`).
This replaces the 5-vessel DEV that 3091 search trials were selected against.

```
arm                              growth_l1   vs physics        95% CI    better   sign-test
physics (backbone)                  0.1316           -              -         -           -
AP closure (shipped C=62.42)        0.1224     -0.0092  [-0.0204,+0.0020]  15/19   p = 0.019
AP closure (CV-best C=100)          0.1181     -0.0135  [-0.0292,+0.0023]  14/19   p = 0.064
node_mlp gate (97 par)              0.1241     -0.0075  [-0.0193,+0.0043]   7/19   p = 0.359
global_affine (2 par)               0.1340     +0.0024  [-0.0057,+0.0104]   9/19   p = 1.000
COUNT FLOOR (mask)                  0.0424     -0.0892           -           -           -
```

* **The AP closure is the only method that survives.** Its mean CI includes zero, but it
  improves 15 of 19 vessels (p = 0.019); the mean is dragged by a few where it hurts
  (p044 +0.035) while helping strongly elsewhere (p006 −0.081).
* **The 97-parameter model does not.** −0.0114 on the 5-vessel DEV becomes −0.0075 under
  LOO, it **loses to the zero-parameter closure**, and wins on only 7/19 vessels. Its mean
  comes from a handful of vessels; on most it hurts. 3091 trials against 5 vessels is what
  that looks like.
* **The 2-parameter affine is worse than physics.** Its DEV −0.0040 was noise.

After a 3091-trial GPU search the best method is still the zero-learned-parameter physics
closure. **No method recovers more than 15% of the +0.0892 prize.**

### 17.2 `C = 100` is left unshipped, deliberately

It scores better on the growth metric (0.1181 vs 0.1224) on a smooth unimodal 1-D sweep,
and the shipped 62.42 was selected under the overlap score since shown to be broken. But
that minimum is in-sample across all 19 vessels and its sign test is *weaker* (14/19,
p = 0.064 vs 15/19, p = 0.019). Consistency was preferred over the ~4%. Revisit with a
larger cohort, not with SEALED.

---

## 18. ROLLOUT vs STATIC Track A — measured unclipped, under growth_l1

`scripts/diag_rollout_trackA.py`, 19 train vessels. The gate is recomputed from GT flow at
every step and **whatever crosses IS the mask** — no clipping to `S`, which is the flaw that
made the lever panel's gate-oracle arm uninterpretable.

```
arm                        growth_l1   final_err   wall F1   n_mask
static Track A (shipped)      0.1224     -0.0581    0.8405     81.5
flow-oracle rollout           0.1262     -0.0211    0.8953     85.6
+ front admission             0.1210     -0.0121    0.9004     86.4
self-blockage (deployable)    0.1615     -0.1474    0.7890     73.1
FLOOR static mask             0.0424     -0.0581    0.8405     81.5
FLOOR oracle-rollout mask     0.0254     -0.0211    0.8953     85.6
```

### 18.1 The rollout mask is genuinely better — and it buys nothing today

Evolving the flow produces a **materially better committed set**: wall F1 0.8405 → 0.9004,
final count error −0.058 → −0.012, and the count **floor drops 0.0424 → 0.0254**, a 40%
reduction in the irreducible error. Per vessel it rescues exactly the under-predicting ones
(p012 51→78 against GT 96; p044 93→145 against 163; p041 70→95 against 113).

And end-to-end `growth_l1` does not move: 0.1224 → 0.1262 (oracle) / 0.1210 (+front).

**The mask ceiling only pays off once timing is fixed.** At current timing quality the model
sits at 0.1224, nowhere near either floor, so a better floor is worth ~0. The decomposition:

```
perfect timing on the STATIC mask    0.1224 -> 0.0424    worth 0.0800
perfect timing on the ROLLOUT mask   0.1224 -> 0.0254    worth 0.0970
the mask improvement alone                               worth ~0.000
```

### 18.2 This corrects §16.5

§16.5 recommended reopening the mask on the grounds that it carried 32% of the error. That
32% is the *floor* — reachable only with perfect timing — and this measurement shows
attacking it now buys nothing. **Timing is the binding constraint; the mask is not.** Order
matters: mask-first is worth zero, timing-first is worth up to 0.080, both is 0.097.

### 18.3 The deployable rollout is worse than the static rule

Algebraic self-blockage — the only flow-evolution mechanism that needs no GT — scores
**0.1615 against static's 0.1224**, with F1 0.789 and a −0.147 count error. It does not
approximate what the true flow does. So the rollout's benefit is entirely gated behind a
genuine evolving-flow predictor, which does not exist in the stack today.

**Verdict: keep the static Track A.** It is not the limitation. Revisit only when a flow
predictor exists AND timing has improved enough for the mask ceiling to bind.

---

## 19. THE TRAINED CORRECTOR CANNOT SUBSTITUTE FOR GT FLOW — it has the wrong sign

`scripts/diag_corrector_rollout.py`. The repo's `LocalKinematicCorrector` (GATv2, trained on
1000 QC-passed COMSOL patch-factory patches) driving the rollout gate, base flow = GT t=0
(the same bandaid arm A ships), 19 train vessels, ~14 corrector calls per rollout.

```
arm                       growth_l1   final_err   wall F1   n_mask     (GT mean n = 97)
static Track A (shipped)     0.1224     -0.0581    0.8405     81.5
corrector rollout            0.1306     -0.1749    0.8243     73.3
FLOOR corrector mask         0.0661     -0.1749    0.8243     73.3
--- GT-flow oracle, for reference ---
flow-oracle rollout          0.1262     -0.0211    0.8953     85.6
FLOOR oracle mask            0.0254     -0.0211    0.8953     85.6
```

**It moves the wrong way on every axis**, and its ceiling gets *worse*: floor 0.0424 → 0.0661
against the oracle's 0.0254.

### 19.1 The mechanism, and it is a sign error not a accuracy problem

GT flow **opens** gates as the clot grows — the low-shear open fraction rises 0.153 → 0.298
over the run (recorded in `shear_redistribution.py`), and the oracle's mask accordingly grows
81.5 → 85.6 toward GT's 97. The corrector **closes** them: 81.5 → 73.3.

That is the physics its training distribution contains. A patch-factory clot is an isolated
high-viscosity obstruction in a Couette channel, so the corrector learned flow *diversion*:
fluid reroutes around the blockage and **accelerates** past it, raising wall shear and shutting
the low-shear gate. What GT actually shows at the wall is the opposite regime — committed
tissue is a no-slip obstacle at 80x viscosity that sheds a **stagnation wake**, lowering the
shear its neighbours see and opening their gates.

This is the same sign disagreement `shear_redistribution.py` already documents between its
`feedback='occlude'` and `feedback='wake'` modes. Both deployable mechanisms tried here shrink
the mask (`self-blockage` n_mask 73.1, the corrector 73.3); only GT flow grows it.

### 19.2 It helps exactly where we over-predict, which is the minority

p025 0.2046 → 0.0592 (mask 87 → 65 against GT 60), p018 0.1349 → 0.0592, p005 0.0506 → 0.0254
— all vessels the static rule over-committed. It badly hurts the under-committing majority:
p044 0.2343 → 0.3231 (93 → 86 against GT 163), p041 0.2075 → 0.2895, p012 0.2010 → 0.2737.
Since the cohort under-predicts on balance (final_err −0.058), the net is negative.

### 19.3 Verdict

The machinery exists and is correctly wired — §18.3's claim that no evolving-flow predictor
exists in the stack was **wrong**. But the trained corrector is out of distribution for the
regime that matters, so the deployable route to the oracle's mask gain is still closed. What
would open it: retrain the corrector on wall-attached clots that shed a stagnation wake, not
isolated micro-clots in a free channel. Until then, keep the static Track A (§18).

---

## 20. WHY t=0 WORKS, AND WHAT THE WALL MASK HAS BEEN HIDING

`scripts/diag_nucleation_and_lumen.py`, 19 train vessels.

### 20.1 Three quarters of the clot nucleates where the flow is ALREADY pathological

Splitting GT's committed wall nodes by whether their gate was open at t=0:

```
   nucleation (gated at t=0)   75%      median onset t/T 0.38
   creep      (gated later)    25%      median onset t/T 0.61   (+0.23 later)
   never gated                  1%
```

**That is why a static snapshot works.** The gates are stagnation and separation — properties
of the *geometry's* flow, not of the clot — and the clot occupies only a few percent of the
lumen, so it barely perturbs what created it. The t=0 field is a marker of a pre-existing
hemodynamic defect, and three quarters of the outcome is already written in it.

### 20.2 Autoregression's role is precisely the other 25%, and it is vessel-specific

The creep fraction ranges 0% to 60%. Six vessels are 100% nucleation (p019, p024, p025,
p036, p018); the worst static failures are exactly the high-creep ones:

```
   p044  creep 60%   mask  93 vs GT 163   growth_l1 0.2343
   p028  creep 58%   mask  28 vs GT  55   growth_l1 0.1393
   p012  creep 55%   mask  51 vs GT  96   growth_l1 0.2010
   p041  creep 50%   mask  70 vs GT 113   growth_l1 0.2075
```

So the architecture that fits the data is **t=0 for nucleation + autoregression for creep**,
which is exactly what the GT-flow oracle rollout delivered (§18: mask 81.5 → 85.6, F1 0.8405
→ 0.8953). There are two distinct failure modes, and they need different fixes: high-creep
vessels **under**-predict, while low-creep vessels with over-open gates **over**-predict
(p019 mask 46 vs GT 27, p025 87 vs 60 — both 100% nucleation, both among the worst scores).

### 20.3 17% of GT clot is OFF-WALL, and every score in this phase excluded it

```
   GT clot off-wall:   mean 17%   median 15%   max 48%
   FULL-MESH deploy score   wall-only 0.7651   wall+lumen 0.7613   (-0.0038)
   WALL-MASKED score this project reports:  ~0.91
```

**The headline 0.9093 is computed on a subset that omits a sixth of the clot.** On the full
mesh the same model scores **0.765**.

The existing lumen arm is directionally right but **ungated**: it helps strongly where
off-wall clot exists (p012 0.5945 → 0.7056, p044 0.6293 → 0.7418, p041 0.6611 → 0.7634) and
hurts where there is none (p036 0.9837 → 0.9096, p024 0.9856 → 0.8710), because it adds false
positives with nothing to find. Net −0.0038, 9/19 vessels improving.

**With an oracle per-vessel on/off gate it is worth +0.0323.** And off-wall fraction
rank-correlates 0.644 with creep fraction — the same vessels, the same mechanism: clot that
propagates away from its nucleation site, whether along the wall or into the lumen.

### 20.4 Consequence for priorities

A per-vessel "does this vessel have off-wall clot" predictor is worth **+0.032 on the score
users actually see**, against **+0.030** left in the entire growth-curve problem. It is a
per-vessel binary from t=0 geometry and flow — a far easier learning problem than onset
timing, on the metric that matters more, and nobody has tried it.

---

## 21. THE LUMEN ARM IS MISCALIBRATED, AND CONTACT-MEDIATED AUTOREGRESSION IS ALREADY DEAD

### 21.1 One scalar, not a classifier

`grow_into_lumen` already predicts off-wall clot exactly as one would guess: take the t=0
wall mask and grow it `LUMEN_HOPS` into lumen nodes whose speed is below `LUMEN_SPEED`. The
*where* needs no new model. Sweeping the two scalars on the full mesh, 19 train vessels:

```
  hops  speed |    score   vs wall-only
     1   0.20 |   0.7799      +0.0148
     2   0.20 |   0.7831      +0.0181     <- best global
     2   0.30 |   0.7613      -0.0038     <- SHIPPED
     3   0.30 |   0.7077      -0.0574
```

**The shipped `LUMEN_SPEED = 0.3` is worse than not running the arm at all.** Moving it to
0.2 is worth **+0.0181** unconditionally, with zero ML — 56% of what the oracle per-vessel
gate (+0.0323) was worth. §20.4's classifier proposal was premature: the residual a
classifier could add is ~+0.014 oracle-valued on 19 vessels, and is not worth building until
the free scalar has been taken. The sweep is unimodal in `speed`, so the 1-D selection is low
risk, but it is in-sample on train and unconfirmed on SEALED.

### 21.2 Nucleation + autoregressive creep was built, and it lost

`src/core_physics/clot_trigger_rollout.py` is literally that architecture --
`E(tau) = wall @ tau=0 OR 1-hop from prior **predicted** commits`. `WALL_MODEL_PLAN.md`
§14.5-14.6 records the verdict: **187k parameters, a 256-dim frozen latent and a 200-step
autoregressive rollout scored F1 0.5403 against a field-only logistic regression's 0.5590**,
and *"a seed-and-grow model has a ceiling near F1 0.58 on this cohort. That ceiling is where
ten fine-tuning legs have been stuck."*

Its diagnosis: 27-58% of commits are **nucleation** (no committed neighbour at commit time),
continuing into the last time quartile. Autoregression strengthens growth-from-existing-clot,
which is the half that already works.

### 21.3 Reconciling that with §20 — two kinds of autoregression, one of them dead

§14.6's "nucleation" (no committed *neighbour*) and §20.1's (gate open at *t=0*) are different
measurements on different cohorts, and they agree: **clot appears where the FIELD is bad, not
where clot already is.** That is why the static field model superseded the autoregressive
stack, and why §20.1 finds 75% of commits were already gated at t=0.

It also sharpens §20.2. The 25% "creep" is **flow-mediated** — the gate opened later because
the flow changed — not **contact-mediated**. That single distinction explains three results at
once:

* the old contact-based trigger rollout failed (wrong mechanism);
* `hop delay` and `thrombin` both scored **exactly 0.0000** in the lever panel (§12) -- both
  are contact-mediated;
* the GT-flow oracle rollout **did** improve the mask (§18) -- flow-mediated, right mechanism.

**Contact-mediated autoregression is dead: tried in two separate phases, failed both times.**
Flow-mediated autoregression is the only live version, and it is blocked on the corrector's
sign error (§19), not on the idea.

---

## 22. THE FLOW-COUPLED CORRECTOR ARM — CLOSED, negative on held-out

Session 2026-08-16. Promoted out of the diagnostics into
`physics_wall_model.predict_phi(mode="corrector")`, calibrated on TRAIN, spent once on
dev-holdout + SEALED. **It does not transfer. Do not reopen without new evidence.**

### 22.1 What the arm is

Every `every=10` rollout steps: current occlusion -> per-node viscosity bump (`delta_mu`)
-> `couple_flow_with_corrector` -> MLS gradients -> `sr`/`dsrx` -> both gates -> keep
integrating. Plus a `seed_ramp` that feeds the model's own t=0 predicted mask in as
occlusion, so the loop does not have to wait for ODE commitment to bootstrap.

Fully isolated: `mode="corrector"` is a separate branch of `predict_phi` requiring an
explicit `CorrectorArm`; the `ode` and `gate` paths are byte-identical, the shipped
`scripts/predict_wall_clot.py` never constructs an arm, and the headline table (0.7866 /
0.7210 / 0.9093 / 0.8567) is unchanged. 650 tests pass.

### 22.2 It reverses sign out of sample

```
                 static (shipped)   corrector arm     delta
TRAIN   (n=26)        0.7489           0.7689        +0.020
dev-hold (n= 2)       0.8554           0.8341        -0.021
SEALED  (n= 8)        0.9093           0.8953        -0.014
```

**Negative on 9 of 10 held-out vessels**, the tenth exactly 0.0000 (`patient031`). Not one
improves. A uniform small harm, not noise around zero.

### 22.3 The ramp was not the problem, and neither was the guess

Swept on TRAIN (26 vessels, GT t=0 flow, `delta_mu` 0.68):

```
ramp   0.00    0.50    1.00    1.50    2.00    3.00
score 0.7659  0.7689  0.7600  0.7568  0.7512  0.7445     (static 0.7489)
```

Two things worth carrying. The guessed 2.0 used in the diagnostics was **nearly the worst
point on the curve** -- so sweeping it was right. But `ramp=0.00`, i.e. no seeding at all,
already scores 0.7659. **The TRAIN gain is the corrector coupling itself, not the
seeding**, and it is that coupling which fails to transfer. Retuning the ramp cannot
rescue this.

### 22.4 A correction to 21.x's reading

The diagnostics reported this arm beating the static rule (wall F1 0.8477 vs 0.8405, count
floor 0.0336 vs 0.0424) and 21.x recorded that as overturning
`diag_corrector_rollout.py`'s "clean negative". **Those were TRAIN-only numbers.** The
underlying mechanism findings stand and are independently verified --

  * the corrector has the right SIGN (lowers wake shear, opens low-shear gates);
  * the patch factory's clots ARE wall-attached (`patch_factory_comsol.py`: "Bottom (y=0):
    no-slip wall (clot attaches here)");
  * `delta_mu` 3.0 over-applies ~4.4x against a measured GT median of 0.68;
  * `diag_corrector_rollout.py`'s "mask shrink 81.5 -> 73.3" compared an ignition-only mask
    against one carrying 6-hop growth; like-for-like it was 73.7 -> 73.3.

-- but the *conclusion* drawn from them does not. `diag_corrector_rollout.py` reached the
right verdict for wrong reasons; this arm reaches a wrong verdict on TRAIN and the right
one on SEALED. **The original recommendation to regenerate the patch factory remains
unjustified** (its premise is factually wrong), but so does deploying this arm.

### 22.5 What is NOT established

`ramp=0.00` was never spent on SEALED -- only `0.50` was, and the sealed set is spent
once (standing constraint 5.1). So "the coupling itself fails to transfer" is inferred
from the TRAIN curve plus the single SEALED point, not measured directly. Re-spending to
chase it would corrupt the only clean held-out evidence this project has. If someone wants
that number, it needs a fresh holdout, not another pass at SEALED.

Reproduce: `python scripts/sweep_corrector_arm.py` then `--spend --ramp 0.50`.
