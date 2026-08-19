# PHASE 7 FINDINGS — the off-wall problem, and what the COMSOL `.mph` actually says

Opened 2026-08-16. Scope: the **full-mesh** `deploy_clot_score` (wall **and** off-wall),
which is the deliverable. The growth curve is a diagnostic for timing only, not a target.

> **Protocol from here on (FIT / DEV / SEALED).** Numbers in this document are TRAIN-mean
> on the 19 eligible full-horizon clot-carrying vessels. That pool **includes DEV**
> (039/040/041/044) and treats `patient020` as if it were a holdout — it is FIT.
> Wall-cohort physics going forward uses the same split as
> `scripts/sweep_ml_clean_protocol.py`: fit on FIT, select on DEV, SEALED closed until a
> choice is frozen (`src/core_physics/wall_cohort_splits.py`,
> `python scripts/eval_wall_protocol.py`). Do not mix this with the wall-gen small cohort
> in AGENTS.md (val = `patient020`).

> **HEADLINE.** Off-wall clot is not a propagation rule and not a mask problem — it is a
> `Mat` **magnitude** problem, and the magnitude is a thing the current model throws away.
> The mechanism is worth **+0.068** full-mesh deploy score with an oracle `Mat`
> (0.7831 → 0.8514 on 19 train vessels). Driven by the model's own `Mat` it is worth
> **−0.017**, i.e. it loses to the shipped speed heuristic. The whole gap is
> `spearman(model Mat, GT Mat) = 0.586` at the wall, and that number is invariant to every
> scalar in the physics model.
>
> **§7 answers the top open question and it is not the answer we wanted.** A perfect
> evolving-flow oracle — the gate recomputed from GT velocity at every step, the ceiling on
> *any* flow model including RGP-DEQ and the corrector — moves the ordering 0.534 → 0.632,
> i.e. **21% of the gap**, and moves the deploy score by **+0.0003**. Ordering is not
> flow-limited. Splitting the interval instead by *calibration* versus *ordering* (§7.2):
> **calibration alone is 53% of the score gap**, and it is a 1-D monotone map rather than a
> network. That reverses the ML priority this document set in §5.
>
> **§8 closes the mesh blocker and finds a worse problem behind it.** The packs are
> **quadratic meshes**: 3/4 of every pack is mid-edge nodes, and the 2.1-edge shell straddled
> a near-wall family of them that carries no species. A purely **topological** shell
> (`first_corner_shell`, no length anywhere) fixes it — off-wall F1 **0.409 → 0.561**,
> precision **0.43 → 0.85** — so §5.1's pre-deploy blocker is closed rather than downgraded.
> But half the *wall* nodes are mid-edge too, and §8.5 shows the headline **0.586 is inflated
> by agreed structural zeros: measured on species-carrying nodes the ordering is 0.193.**
>
> **§9 finds that §5–§8 were all debugging the inputs to an equation that is short a term.**
> The surface ODE only ever **accumulates** — `Mat` is monotone by construction — but in the
> `.mph` `Mat` is a *convected domain field* (`tds2`, convection on, crosswind stabilised), so
> material deposited at the wall is carried away. Handed a **perfect** oracle (GT `RP`, `AP`,
> `M`, `Mas`, `sr`, `∂sr/∂x` at every step) the per-node law still ranks GT `Mat` at only
> **0.310** and is **anti-correlated on 5 of 19 vessels**. That is a ceiling on the equation,
> not on any input model, and it is why every input-side fix has read as "no gain". Adding the
> sink, one global scalar, takes the oracle to **0.447 leave-one-vessel-out** — beating a bare
> lifetime, with saturation buying exactly nothing, and well clear of the `1/sr` null (0.271).
>
> **It does not ship yet, and the reason is the useful part.** On the shipped model the sink
> makes ordering *worse* (`rho_corner` 0.482 → 0.084), because with the gate and `ap`/`rp`
> frozen at t=0 a constant source against a linear sink has one attractor, `J0/(λ·sr)`.
> Accumulation is what let the frozen-input approximation survive. Crossing inputs against
> removal, all against the frozen accumulate-only baseline (0.219): removal alone **−0.123**,
> evolving chemistry alone **−0.245**, evolving flow alone **+0.176**, all three together
> **+0.245**. So these are not three independent knobs, and **§7 measured flow with the other
> two switched off** — its "+0.0003 score" stands, its "ordering is not flow-limited" does not.
>
> **§10 finds the score that *is* sitting on the table, and it is the wall mask.** GT clot
> **is** `{Mat >= 2e7}` in both directions (0.0% of clot below the platelet step, 0.19% of
> high-Mat nodes not clot); fibrin is inert.  Off-wall, the model's own `Mat` cannot beat
> the shipped speed arm at any attenuation (0.766 vs 0.783).  Wall-side, every false
> negative is a t=0 gate that never opened (19.3% of GT), and graph-growth FP is 2 nodes.
> The union of that gate over GT flow is worth **+0.051** deploy score -- §7 never measured
> this because it froze the mask.  The deployable slice of it is longer along-wall growth
> inside the existing admission band: `GROW_HOPS` 6 → 20, unimodal, leave-one-vessel-out
> **+0.018** (predicted t=0 flow **+0.025**).  Filling the whole component overshoots.

---

## 0. HOW TO READ A COMSOL `.mph` — do this before deriving physics again

`.mph` files are **zip archives**. `smodel.json` inside is the complete model tree:
parameters, variables, every physics node, every boundary condition, the material law.

```python
import zipfile, json
sm = json.loads(zipfile.ZipFile("comsol_models/phase2_template_nowound.mph").read("smodel.json"))
```

This is strictly better than reverse-engineering from exports. Two of the repo's standing
open questions were answerable directly from it (§2, §3). Caveat from the model author:
the parameter list can retain stale entries for deleted mechanisms — trust the **physics
node tree** (what has a `Reactions` child, what a BC actually references), not the presence
of a parameter.

---

## 1. THERE ARE TWO GENERATIONS, AND THE PRODUCTION ONE IS THE OLD-LOOKING ONE

| | `phase2_nowound_001/002/003.mph` (dated May 5) | everything else, incl. both templates and `011/012/025/033/036/037/040/044` |
|---|---|---|
| `M_inf` | `3.5e8 plt/cm³` | **`7e6 plt/cm²`** |
| `Reactions_3spec` node | present | **absent** |
| `R_Mat` | `k_vol*AP*(Mat/M_inf)`, `k_vol = 1 [1/s]` | **none** |
| separation gate | `sr_grad_flow` (traction-projected ∇ of capped shear) | **`d(spf.sr,x)`** |
| `mu_max` cap | `min(mu_max, ...)` | none |

**`001/002/003` are an experimental branch and are not the production physics.** An earlier
draft of this document had that backwards; the correction matters, because the
volumetric-aggregation reading suggested a bulk reaction-front model that does not exist.

Consequences:

* **`docs/COMSOL_PHYSICS_VALIDATION.md` is correct as written.** The surface law, `Da=1e-4`,
  the gates, `mu1(Mat)` at `2e7`, fibrin inert — all confirmed against the production tree.
* `src/core_physics/ap_closure.py` and the `PHASE6` derivation are calibrated on the right
  generation. `patient007`'s export matches the production template.
* The `sr_grad_flow` form is **not** production and the repo's `d(spf.sr,x)` gate is right.
  Retract that concern.

## 1.1 But `Mat` is still a DOMAIN field, in every generation

Production `tds2` = *Transport of Diluted Species*, 3 species `M, Mas, Mat`, with

```
D_M = D_Mas = D_Mat = 0            <- zero diffusion
Reacting Flow, Diluted Species 1   <- coupled to spf, so Mat IS advected
wall_surface_reactions_3spec       <- J0_Mat as an INWARD FLUX boundary condition
```

So the governing equation is `∂Mat/∂t + u·∇Mat = 0` with a wall flux source — **not** a
surface ODE. `Mat` lives in the domain, the clot label is a threshold on that domain field
(`mu1(Mat)` steps 1→80 at `2e7`), and off-wall clot is simply the part of that field that
sits off the wall. There is no separate lumen mechanism to model. This is the single fact
the shipped lumen arm does not encode.

---

## 2. `da_scale` IS `1/h_cell`, AND THAT KILLS THE "UNEXPLAINED DAMKÖHLER RATIO"

`PHASE6_HANDOFF` §1 called `d(Mat,t) ≈ 146·J0_Mat` and `d(Mas,t) ≈ 26·J0_Mas` "the biggest
open question in the physics." It is a **discretisation factor**, not a chemistry factor.

`J0_Mat` is a flux `[plt/(cm²·s)]` into a volumetric field `[plt/cm³]`. The nodal balance
for a boundary node is `dMat_i/dt ≈ J0_i / h_i`, where `h_i` is the first-cell thickness.
So the repo's `da_scale` carries units of **1/length** and should equal `1/h_cell`.

Measured on the packs — distance from each wall node to its nearest off-wall node:

```
19 train vessels, first-cell thickness h:  0.03543 - 0.03583 cm   (1/h = 27.9 - 28.2)
cohort median 1/h = 28.1                   shipped da_scale = 40
```

`h` is constant to 1% across the cohort, which is why a single global `da_scale` works at
all. And on the export:

```
d(Mas,t) / J0_Mas = 25.75      vs    1/h = 28.1      <- 8% agreement
d(Mat,t) / J0_Mat = 145.63     vs    1/h = 28.1      <- 5.2x unexplained
```

**The Mas anomaly is fully resolved.** The Mat one is reduced from "146× and unexplained"
to "5.2× and localised to the autocatalytic branch" — `Mas` saturates (`Sat(M) → 0`) so its
own `J0` collapses while `J0_Mat` keeps growing through the `(Mas/M_inf)·k_aa·AP` term, and
`Mat` alone accumulates without bound. Still open, but it is now one term, not a mystery.

### 2.1 The geometric value is worse than the tuned one — keep 40

`scripts/eval_offwall_mat_arm.py --da-sweep`, and a direct growth sweep on 19 train vessels:

```
da_scale     growth_l1   vs 40    onset rho
      20        0.3644  +0.2103       0.667
      28        0.2221  +0.0680       0.674     <- the geometric 1/h
      40        0.1541  +0.0000       0.685     <- shipped, and the winner
      60        0.2371  +0.0830       0.678
     100        0.3263  +0.1722       0.680
```

So this is an **interpretation win, not a score win**: `da_scale` is now a named physical
quantity within 1.4× of its geometric value, rather than a free scalar absorbing an
unexplained 26–146×. Do not change it. A per-node `1/h_i` was also considered and is
pointless on this cohort — `h` varies by 1%.

---

## 3. OFF-WALL, MEASURED

Shipped predictor (`--flow gt --lumen`), 19 train vessels, strict F1 split by domain:

```
              precision   recall      F1
wall            0.877      0.822    0.822
off-wall        0.110      0.286    0.126      <- 25% of GT clot on train
```

`patient032`: 120 off-wall GT nodes, model finds **0**. `patient029`: 14, finds 0.

### 3.1 The geometry is trivial and the wall mask is not the bottleneck

12 vessels carrying off-wall GT clot:

* off-wall GT clot is **one boundary-layer node row**: normal offset 1.66–1.80 median edge
  lengths, `p50 ≈ p90`. Confirmed as a **mesh artefact** by the model author against the
  COMSOL field plots — the clot is a thin band of finite physical thickness and the BL mesh
  resolves it in ~2 rows.
* its nearest wall node is GT-committed **99.9%** of the time.
* and yet a pure thickness rule seeded on **perfect GT wall clot** peaks at off-wall
  **F1 0.275**. Recall is free; precision is the whole problem.

**So the `where` is determined and the wall mask is not the limiter.** `physics_lumen_model`'s
premise ("a propagation rule seeded by the wall arm") and `PHASE6_RESULTS` §21.1's "the
*where* needs no new model" are both refuted by that 0.275.

### 3.2 What separates a clotting shell node from a non-clotting one

```
                        clots     doesn't
speed_nd (t=0)          0.272      0.442      weak
shear    (t=0)          42.8       40.4       none
owner wall node's Mat   1.46e8     6.19e7     2.4x, same sign on 12/12 vessels
```

And the attenuation through the shell is one constant:

```
Mat_offwall / Mat_owner, median per vessel (12 vessels):
0.166 0.149 0.161 0.170 0.172 0.146 0.155 0.159 0.149 0.152 0.163 0.166
```

**0.16 everywhere.** So an off-wall node commits iff `0.16 · Mat_owner ≥ crit`. Independent
confirmation: at off-wall nodes, `Mat > crit` has **precision 1.000, recall 0.811** against
the μ-based GT label. The label *is* the Mat threshold, as §1.1 says it must be.

---

## 4. STEP 1 — THE MAT-MAGNITUDE LUMEN ARM: BUILT, AND IT DOES NOT SHIP

`src/core_physics/physics_lumen_model.grow_into_lumen_by_mat` +
`fill_grown_wall_mat`, wired as `predict_wall_clot(..., lumen="mat", mat_wall=...)`.
The shipped `lumen=True` path is untouched and byte-identical (asserted by the unchanged
wall-only and speed-arm rows below). `scripts/eval_offwall_mat_arm.py --flow gt`:

```
arm                score(FULL) score(wall)   wall F1   off F1   vs shipped
wall-only               0.7651      0.8330    0.8218   0.0000    -0.0181
speed (shipped)         0.7831      0.8330    0.8218   0.1595    +0.0000
mat                     0.7659      0.8330    0.8218   0.0232    -0.0173
mat + fill              0.7659      0.8330    0.8218   0.0232    -0.0173
mat ORACLE  (att 0.16)  0.8304      0.8330    0.8218   0.5614    +0.0473
mat ORACLE  (att 0.25)  0.8514      0.8330    0.8218   0.5829    +0.0683
```

**These are the §8.4 topological-shell numbers, not the ones this document opened with.**
The shell changed under this table for the reason in §8; the original wide shell read
`0.8324 / off F1 0.409` and `0.8588 / 0.372`. Score is ~0.002–0.007 lower, off-wall F1 is
+0.15 to +0.21 higher, and the difference is the empty band of §8.2.

* **The mechanism is real and large: +0.066 on the deliverable metric.** That is ~1.4× the
  entire remaining onset-timing prize Phase 6 was chasing, on the metric that matters more.
* **The deployable version is negative.** `−0.0173` against the shipped speed heuristic.
* `att = 0.25` beats the measured `0.16` because the deploy score is 2-hop relaxed and
  F0.5-weighted; `0.25` requires `Mat_owner > 4·crit` instead of `6.25·crit`, buying recall
  at a precision cost the relaxed metric barely charges. Quote **0.16** as the physics and
  **0.25** as a TRAIN-fitted scoring tune. On the corrected shell the two now agree in sign
  (0.25 wins on both F1 and score); on the wide shell they disagreed, because the relaxation
  was paying for empty-band recall.
* `fill_grown_wall_mat` changes nothing (`mat` == `mat + fill`). Graph-grown wall nodes do
  inherit a magnitude, but the magnitudes they inherit are themselves too small to clear the
  threshold, so the fill is masked by the same bias that breaks the arm.

### 4.1 Why it fails, precisely

```
da_scale   median model Mat   median GT Mat   ratio   spearman(model,GT) at wall
      28          3.15e7          7.19e7      0.439        0.586
      40          4.32e7          7.19e7      0.602        0.586
      80          8.22e7          7.19e7      1.144        0.586
     160          1.60e8          7.19e7      2.229        0.586
     320          3.16e8          7.19e7      4.400        0.586
```

The magnitude bias is fixable by a scalar; **the ordering is not**. `spearman` is invariant
to `da_scale` because `da_scale` is a monotone rescaling, and the sweep over
`(da_scale, attenuation)` confirms the arm saturates at off-wall F1 ≈ 0.14 for every
combination — it degenerates into the thickness rule of §3.1 and inherits its ceiling.

The model's `Mat` ordering is essentially the frozen `gate`'s ordering (the same mechanism
`PHASE6_RESULTS` §6 established for onset: with `gate` and `sr` frozen at t=0, the rollout
is a deterministic monotone function of them). 0.586 is what that buys.

---

## 5. WHAT THIS MEANS — the ML target is now quantified and it is a static field regression

The job description, in one line:

> **Predict wall `Mat` magnitude at t_final well enough to raise
> `spearman(pred Mat, GT Mat)` from 0.586 toward 1.0. Each point of that converts
> mechanically into full-mesh deploy score via the §3.2 rule, and the whole interval is
> worth +0.0757.**

**Revised by §7.2 — read that before acting on this section.** The interval splits roughly
half calibration and half ordering, and the calibration half is a 1-D monotone map rather
than a network. Ordering remains the ML target; it is no longer the *first* thing to build.

Why this is a better ML target than anything Phase 6 tried:

* **No recurrence, no thresholded readout.** Both killed the in-ODE corrector and the
  survival head (`PHASE6_HANDOFF` §4).
* **Dense supervision.** `Mat_log1p_nd` at 201 timesteps on every wall node of every pack.
* **The physics stays structural.** The gate, the ODE and the 0.16 attenuation all remain;
  the network supplies only the magnitude field the rollout currently gets wrong.
* **It is scored on the deliverable**, not on a proxy that fights it (`PHASE6_RESULTS` §15.5).

Gates before anything is shipped, inherited from `PHASE6_HANDOFF` §9:

1. **Beat ridge on the same features.** Nothing in this repo has ever survived that.
2. **The wall mask must not move.** The mask is gate-derived; a Mat-magnitude model changes
   only the lumen arm and onset. Assert it.
3. Report `spearman(Mat)`, off-wall F1 **and** full-mesh score together. §4 shows off-wall
   F1 and the score can disagree in sign.

## 5.1 Open, in priority order

**§7 and §8 close the first and third of these.** Kept for the reasoning, superseded by
§7.3 for what to do next.

* ~~**Why is model `Mat` ordering only 0.586?** Ablate: frozen gate vs GT-flow gate rollout.~~
  **Answered in §7: flow-coupling buys 21% of the ordering gap and ~0 score. Not the fix.**
  **Re-opened and re-answered in §9.4:** flow was ablated with removal and chemistry frozen,
  which is the one configuration in which it cannot pay. It is worth +0.176 of ordering once
  the equation has a sink; it still does not convert to score.
* ~~**The 5.2× residual on `d(Mat,t)/J0_Mat`** (§2) — one term, the autocatalytic branch.~~
  **Explained in §9.5:** a monotone accumulator fitted against a field that reaches a balance
  must show an inflated apparent rate, worst on the branch with no saturation of its own.
* **Why the wall-normal mid-edge nodes are empty while the along-row ones are not** (§8.3).
  Unexplained. It does not block anything — the shell is built on the measurement — but it
  is the one place where we are pattern-matching the export rather than the physics.
  §9.1 removes one candidate: `tds2` is Quadratic, so it is not a discretisation-order gap.
* ~~**Mesh portability.** Re-express the shell as a physical thickness.~~ **§8: re-expressing
  it in cm changes nothing on this cohort (3% edge-length spread); the real defect was a
  structurally-empty node family inside the shell, and `first_corner_shell` removes the
  length entirely. Closed.**
* **SEALED is untouched by all of the above.** Every number here is TRAIN. The oracle arm is
  a mechanism measurement and selects nothing; the `att = 0.25` tune was fitted on TRAIN and
  has not been spent.

---

## 7. STEP 2 — THE FLOW-COUPLING ABLATION: ORDERING IS NOT FLOW-LIMITED

The §5.1 top item, run. If recomputing the gate from *evolving* flow recovers most of
`0.586 → 1.0`, off-wall is a flow-coupling fix and there is no ML problem. It does not.

Five arms, same ODE and same `att = 0.16` shell, differing only in where the gate's
`(sr, ∂sr/∂x)` comes from:

| arm | t=0 flow | gate during rollout | deployable |
|---|---|---|---|
| `frozen t0 / gt` | COMSOL | frozen at t=0 | no (GT at t=0) |
| `frozen t0 / pred` | RGP-DEQ `u0_pred` | frozen at t=0 | **yes** |
| `evolving / GT flow` | COMSOL | **recomputed from GT `[u,v]` every step** | no — this is the ceiling |
| `evolving / corr gt` | COMSOL | local kinematic corrector, every 10 steps | no (GT at t=0) |
| `evolving / corr pred` | RGP-DEQ `u0_pred` | local kinematic corrector, every 10 steps | **yes** |

`evolving / GT flow` is an oracle in the strict sense: it is what a *perfect* flow model
would give, so it upper-bounds RGP-DEQ, the corrector, and anything else. 19 cached train
vessels, `rho = spearman(arm Mat, GT Mat)` over wall nodes:

```
arm                            rho   d_rho   % of rho gap    score   d_score   % of score gap
frozen t0 / gt               0.534       —              —   0.7659         —                —
evolving / corr gt           0.587  +0.052          11.3%   0.7659   +0.0000             0.0%
evolving / GT flow           0.632  +0.098          21.0%   0.7662   +0.0003             0.4%
ORACLE Mat                   1.000  +0.466         100.0%   0.8304   +0.0645           100.0%
```

**A perfect evolving-flow model buys 21% of the ordering gap and 0.4% of the score gap.**
The learned corrector recovers about half of the oracle's ordering gain (0.052 of 0.098) and
none of the score. On the 13 off-wall-carrying vessels the shape is identical (26.6% of rho,
0.4% of score); on the 15 vessels with a cached `u0_pred`, the two deployable arms are
*worse* than their GT-t=0 twins (`frozen t0 / pred` 0.439 vs `frozen t0 / gt` 0.525), so the
t=0 flow error currently costs more ordering than evolving the gate recovers.

`rho` here is over all wall nodes, i.e. §4.1's definition, so the ablation is anchored to the
documented 0.586. **§8.5 shows that number is inflated** — on species-carrying nodes the same
arms read 0.193 / 0.251 / 0.309 — but the *proportions* in the table above survive it: the
oracle still buys 14% of the corner-node gap and the same 0.4% of the score.

So `PHASE6_RESULTS` §18's result does **not** generalise from the mask to the magnitude.
Evolving flow reopens gates in the right *places* — that is a mask edit — but the magnitude
ordering is set by the surface ODE's response to `AP`, which the gate only switches on and
off.

### 7.1 The corrector is doing real work, and it is not enough

Not a plumbing failure: `n_ODE` (mean wall nodes whose surface ODE ever integrates) goes
73.7 → 85.6 under the GT-flow oracle and 73.7 → 78.5 under the corrector, so the corrector
recovers ~40% of the gates the oracle opens. Per-vessel it is largest exactly where flow
rerouting should matter most — `patient041` 0.572 → 0.684 (oracle 0.846), `patient020`
0.465 → 0.594 (oracle 0.677), `patient012` 0.520 → 0.571 (oracle 0.761). The mechanism is
right and the headroom above it is real; it is the *total* that is small.

### 7.2 The decomposition that actually matters: calibration is 53% of the score gap

The ablation also runs every arm through a quantile match — `pred Mat` remapped onto the GT
`Mat` distribution, which **destroys calibration error while preserving rank order exactly**.
The gap between an arm and its `+qmatch` twin is therefore pure calibration, and what
survives is pure ordering.

```
arm                            rho    off F1    score
frozen t0 / gt               0.534    0.0232   0.7659
frozen t0 / gt  +qmatch      0.534    0.3409   0.8003     <- +0.0344 from calibration alone
evolving / GT flow  +qmatch  0.632    0.3865   0.8194
ORACLE Mat                   1.000    0.5614   0.8304
```

Reading the three numbers `0.7659 / 0.8003 / 0.8304`:

* **Calibration alone, at today's 0.534 ordering, is +0.0344 — 53% of the score gap.**
* **Ordering alone, from 0.534 to perfect, is +0.0301 — the remaining 47%**, and 21% of that
  is reachable with flow.
* Combined with a perfect flow model, calibration + its ordering gain reaches 0.8194, or
  **83% of the interval**, with ordering `rho = 0.632` still far from 1.

`n_trig` (off-wall nodes admitted per vessel) is the mechanism: **1.4 raw, 18.1 after
qmatch, 19.3 for the oracle.** The arm is not mis-*ranking* the shell, it is failing to
reach the threshold at all — `0.16 · Mat_owner ≥ crit` needs `Mat_owner ≥ 6.25·crit`, and
the model's `Mat` distribution barely reaches it (§4.1: ratio 0.602 at the shipped
`da_scale`). The arm is threshold-starved before it is order-limited.

### 7.3 What this changes about the ML target

§5 said "raise `spearman` from 0.586 toward 1.0, worth +0.0757." The prize is real but it is
now **the harder half of a smaller one**, and §8.5 makes the ordering half harder still:

1. **Fix the distribution first.** A monotone calibration map from model `Mat` to GT `Mat` —
   one 1-D transform, fit on TRAIN wall nodes — is worth ~+0.034 and does not need a GNN.
   §4.1 shows a *scalar* cannot do it (the ratio moves with `da_scale`, `off F1` saturates at
   0.14), so the map must be shape-changing, not a rescale. This is the cheapest remaining
   win in Phase 7 and it should be attempted before any network.
2. **Then ordering, and it is genuinely an ML problem — a bigger one than §5 thought.**
   Flow gives 21% of it, and §8.5 shows the honest starting point is `rho_corner = 0.193`,
   not 0.586. Score it on corner nodes; `rho_all` can be raised by learning the mesh.
3. **Do not spend effort on evolving-flow plumbing for the off-wall arm.** It is +0.0003.
   The corrector and RGP-DEQ remain right for the *mask* and for onset (`PHASE6_RESULTS` §18);
   this section says only that they do not fix `Mat` magnitude.

---

## 8. THE MESH ARTEFACT IS NOT A BLOCKER — IT WAS A BUG

§5.1 recorded the shell bound (median-edge-length units) as a pre-deploy blocker, with the
proposed fix "re-express as a physical thickness." Two findings, in order of importance.

### 8.1 Re-expressing it in cm is a no-op on this cohort

```
h_cm      (median edge length)   0.03473 - 0.03577    spread  2.9%
d_bar_cm  (vessel diameter)      0.82617 - 1.99825    spread 76.4%
```

The median edge length is constant to 3% across all 19 vessels while the vessels themselves
vary by 76%, so a mesh-unit bound and a cm bound select **the same nodes** — formulations A
and B in the table below are numerically identical. This cohort cannot distinguish them, and
that is precisely why §5.1 could not tell whether either was right.

### 8.2 The real defect: the shell straddled a node family that carries no species

Walking normal offsets in fixed bands, one node per wall node per band:

```
band [edges]     s / h    n/vessel   Mat/owner   off-wall GT clot
0.50 - 1.35       1.01         555      0.0000                 83     <- EMPTY
1.35 - 2.20       1.72         554      0.1537                493     <- species band
2.20 - 3.00       2.63         551      0.0000                  7     <- EMPTY
3.00 - 3.80       3.43         556      0.0217                  0
```

**The nodes at ~1.0 h carry identically zero `M/Mas/Mat` while carrying normal velocity and
pressure**, and so do the ones at ~2.6 h — the species field is present on every *other*
band. The original shell was `0 - 2.1` edges, which **included the whole empty family at
1.0 h**: ~23 nodes per vessel that are structural false positives, admitted purely as a
function of their owner.

Excluding it, GT-Mat oracle, 19 vessels:

```
formulation                      off F1   prec    rec  n_pred | coarser BL: F1   prec
A  0 - 2.1 edges (was shipped)   0.4091  0.428  0.484    44.9 |          0.5811  0.812
B  physical cm (5.1 proposal)    0.4091  0.428  0.484    44.9 |          0.5811  0.812
C  species band 1.35-2.20        0.5303  0.809  0.446    22.4 |          0.5788  0.809
D  empty band alone              0.0364  0.036  0.038    23.0 |          0.0000  0.000
E  topological (no length)       0.5614  0.849  0.470    21.1 |          0.6019  0.850
```

* **Off-wall F1 0.409 → 0.530, precision 0.428 → 0.809, at a recall cost of 0.038.** Half
  the shell's predictions were nodes that cannot clot.
* Row D is the control: the empty family alone scores **0.036**, i.e. it was contributing
  essentially nothing but volume.
* The `coarser BL` column deletes that family and **re-measures** the median edge length on
  the surviving subgraph — the same vessel meshed without the interleaved band. A and C
  converge there (0.581 / 0.579), which is the point: the wide shell only looked competitive
  *because* of the family it should never have contained.
* **Row E is the fix, and §8.4 is why.**

### 8.3 What the empty family is — a quadratic mesh, and one wrong inference

The packs are **quadratic (P2) triangle meshes**. The mid-edge nodes are detectable from
topology alone — degree 2, sitting at the midpoint of their two neighbours — and they are
**0.742–0.746 of all nodes on all 19 vessels**, i.e. exactly the 3/4 a triangulation gives
(three mid-edge nodes per corner node). The empty 1.0 h band is **99.8% mid-edge**, and each
of its nodes is precisely `midpoint(its owner wall node, a species-band node)`.

**But mid-side does not imply empty, and an earlier draft of this section said it did.**
Splitting the species band itself by node type:

```
                     n/vessel   frac Mat == 0   Mat/owner   off-wall GT clot (cohort)
band 1.35-2.20, corner    ~273           0.000      0.176                        323
band 1.35-2.20, mid-side  ~273           0.000      0.147                        170
```

The mid-side nodes *lying along* the species row carry `Mat` perfectly normally and hold
**170 of the 493 off-wall GT clot nodes**. So "drop all mid-side nodes" is wrong and costs a
third of the recall — measured, it scores 0.429 against 0.561. What is empty is specifically
**the mid-edge node of an edge crossing outward from the wall**, which is a much narrower
set. Why those in particular are unpopulated is *not* established; a P2-velocity /
P1-species story predicts the along-row nodes would be empty too, and they are not. The
measurement is stable on all 19 vessels and is what §8.4 is built on; the mechanism is open.

### 8.4 The fix: a shell with no length in it at all

`first_corner_shell` navigates the mesh's own layering instead of measuring a distance:

1. `wall_normal_midside` — the empty family bridging the wall outward;
2. the corner nodes on its far side: the first species-carrying row;
3. the mid-side nodes along that row (both neighbours in it), which §8.3 says must be kept.

Every step is a statement about element order and connectivity, so **there is no constant to
recalibrate on a customer mesh**. It reproduces the calibrated 1.35–2.20 band with Jaccard
**1.000 on 12 of 19 vessels and ≥ 0.84 on all of them**, and it is the best-scoring
formulation in §8.2 both natively (0.5614) and under the coarser-BL perturbation (0.6019).
It is now the default in `grow_into_lumen_by_mat`.

One safety property matters: the rule needs a quadratic mesh, and on a linear one it would
select nothing — a silent no-op for the whole arm. `resolve_offwall_shell` detects the
absent mid-edge family and falls back to the calibrated band, and there is a test for it.

**§5.1's pre-deploy blocker is therefore closed**, not merely downgraded. The residual is
much smaller: the fallback path is still in mesh units, and the emptiness mechanism in §8.3
is unexplained.

### 8.5 The consequence that matters most: the 0.586 is inflated, and the real ordering is 0.193

Half the **wall** nodes are mid-edge nodes too, and GT `Mat` is zero on far more of them:

```
of WALL nodes, mid-side                0.496   (min 0.487 max 0.498)
GT Mat == 0 at mid-side wall nodes     0.446   (min 0.056 max 0.762)
GT Mat == 0 at corner  wall nodes      0.176   (min 0.000 max 0.711)
```

§4.1's headline `spearman(model Mat, GT Mat)` is taken over **all** wall nodes, so half its
sample is a family on which GT `Mat` is structurally zero 45% of the time. Recomputing it on
corner wall nodes only (`rho_corner` in the ablation):

```
arm                    rho_all   rho_corner   both zero, mid-side   both zero, corner
frozen t0 / gt           0.534        0.193                 0.445               0.176
evolving / corr gt       0.587        0.251                 0.445               0.176
evolving / GT flow       0.632        0.309                 0.444               0.176
```

**`rho_corner` is 0.193, not 0.534** — lower on 16 of 19 vessels, not higher. The last two
columns are why: on 44.5% of mid-side wall nodes the model and GT are *both* exactly zero,
against 17.6% on corner nodes, and the `ORACLE Mat` row confirms 0.446 is precisely GT's own
zero fraction there. The model reproduces that zero block almost perfectly — not by ordering
anything, but because neither field deposits there — and a rank correlation scores a shared
block of tied zeros as perfect agreement.

So roughly **two thirds of the reported 0.586 is agreement about structural zeros**, and on
the nodes that actually carry species the model's `Mat` ordering is nearly uncorrelated with
GT. Two consequences, both bad for §5 as written:

* **The ML problem is harder than advertised** — `0.193 → 1.0`, not `0.586 → 1.0`.
* **`rho_all` is the wrong metric to optimise.** A model could raise it by learning which
  nodes are mid-edge, which is worth nothing. Report `rho_corner`.

This does not change the *score* interval (+0.066); the deploy score never saw the mid-side
zeros as clot either. It changes what has to be true of a model to collect it, and it
reinforces §7.2's ordering: fix calibration first, and treat ordering as the long project.

---

## 9. THE EQUATION IS SHORT A TERM — and that is why every input fix has read as "no gain"

§5 through §8 all assumed the surface law was right and the *inputs* were wrong. That
assumption was never tested. Testing it changes the diagnosis.

`integrate_mat_trajectory` integrates every wall node **independently**, and it only ever
**accumulates**:

```
dMat/dt = da * gate * (Sat*(k_rs*RP + k_as*AP) + (Mas/Minf)*k_aa*AP)
```

Term for term that is exactly COMSOL's `J0_Mat` (§9.1 checks this against the `.mph`, and it
matches, including the fact that the gate's two branches are *added* and not chosen between).
The problem is not the source. It is that there is no sink, so `Mat` is monotone by
construction and has no steady state.

### 9.1 What the `.mph` says, read directly

`scripts/diag_mph_surface_law.py` pulls the law out of the node tree instead of re-deriving it.
Four things it settles:

| question | answer |
|---|---|
| `step2t(t)`, the ramp on every `J0` | the function named `step2t` is the one **tagged `step4`** — location 12 s, transition width 2.5 s. So `surface_time_gate_s = 12.0` is **correct**, and against a 150 s sampling interval it is 1 at every stored step. Not a timing lever. |
| `Sat` | `an4` = `1 - M/M_inf`. In `srf1`, `J0_M` and `J0_Mas` are the *same expression*, so `M ≡ Mas` — which is why the repo evaluating `Sat` on `mas` is right, not a bug. |
| species discretisation | `tds` (the 9-species cascade) is **Linear**; `tds2` (`M`/`Mas`/`Mat`) is **Quadratic**. This corrects §8.3's guess. |
| does `Mat` move? | **Yes.** `tds2` has convection enabled, nonconservative form, Do Carmo and Galeão crosswind stabilisation. `Mat` is an advected domain field with a wall flux source — not a surface coverage. |

**The ungated `srf2` node is not a missing source, and this is worth recording because it looked
like one.** The tree has *two* active surface-reaction nodes: `srf1`
(`wall_surface_reactions_3spec`, gated) and `srf2` (`SfcRxn_3spec`), whose `J0_Mat` has **no
shear gate at all**. But `srf2` appears only in `phase2_template_wound.mph`; the actual
production patient models (`phase2_nowound_011.mph`, and the cohort is all `nowound`) carry
`srf1` **only**. The repo's gated law is complete for this cohort. Forward-looking risk: if a
wound case ever enters the cohort, `srf2` deposits regardless of shear and a gate-based model
cannot produce clot there at all.

### 9.2 The per-node ODE cannot order GT `Mat` even with perfect inputs

`scripts/diag_local_ode_closure.py` hands the local law a total oracle — GT `RP`, `AP`, `M`,
`Mas`, `sr` and `∂sr/∂x` at **every** timestep, i.e. every input any model could ever supply —
and integrates COMSOL's own `J0_Mat`. On the nodes where both the flux and GT `Mat` are live,
19 train vessels:

| | |
|---|---|
| rank vs GT `Mat` | **0.310** |
| vessels **anti**-correlated | **5 of 19** |
| log-R² | **negative** |
| within-vessel spread of the rate scalar needed to close the local balance | **IQR/median 0.67** |

No input model and no choice of `da_scale` can cross this. It is a ceiling on the *equation*.
Two obvious non-local rescues are dead on measurement:

* **`da_scale` should be per-node `1/h_i`** (the natural reading of §2). `spearman(k_i, 1/h_i)`
  = **−0.018**. The rate scalar is not the local cell size.
* **Tangential advection along the wall** — upwind-accumulating the flux along the wall in the
  flow direction buys **exactly 0.000**, which is what no-slip predicts: the tangential
  velocity *at* the wall is zero.

### 9.3 A removal term is the missing structure — and it does not ship

Add the sink the domain field implies, `− λ·sr·Mat`, with a single global dimensionless `λ`
(`scripts/diag_mat_washout.py`). Against the two cheaper stories, on the same oracle:

| mechanism | best λ | rank vs GT `Mat` | anti-corr vessels |
|---|---|---|---|
| none (shipped form) | — | 0.310 | 5 |
| **washout** `−λ·sr·Mat` | 1.54e-6 | **0.464** | 3 |
| lifetime `−λ·Mat` | 5.6e-5 | 0.430 | 4 |
| saturation `J0·(1−Mat/Msat)` | — | 0.310 | 5 |

Saturation buys **nothing**. Leave-one-vessel-out, `λ` fitted on 18 and scored on the held-out
one: **0.310 → 0.447**, and 16 of 19 vessels pick the same `λ`, so almost none of the gain is
the fit reading its own answer. It is also not a shear correlate in disguise — the nulls `1/sr`,
`−sr` and `J0/sr` reach only 0.271, 0.271 and 0.287.

**And on the shipped model it makes things worse.** `scripts/eval_washout_arm.py`, model's own
`Mat`, everything else held: `rho_corner` **0.482 → 0.084**, off-wall F1 **0.023 → 0.000**,
score −0.0008. Spearman is invariant to rescaling, so this is not a `da_scale` recalibration
artifact — the *ordering* is destroyed. The per-vessel signature is exact sign flips
(p020 +0.924 → −0.924, p021 +0.907 → −0.907), the mark of the solution having reached its
steady state `J0/(λ·sr)`, whose ordering is precisely the `1/sr` null.

### 9.4 Why, and the interaction that reframes §7

The shipped model freezes the gate **and** `ap`/`rp` at t=0, so its source is constant in time,
and a constant source against a linear sink has exactly one attractor. **Accumulation is what
let the frozen-input approximation survive**: integrating a constant source still yields a
growing, informative field. Splitting `J0 = gate(flow) · chem` and freezing each factor
independently, on oracle inputs:

| inputs | accumulate-only | with washout | Δ |
|---|---|---|---|
| frozen both | 0.219 | 0.097 | −0.123 |
| evolving **flow** only | **0.395** | 0.356 | −0.039 |
| evolving **chemistry** only | −0.026 | −0.078 | −0.052 |
| evolving **both** | 0.310 | **0.464** | **+0.153** |

Two things fall out, and both correct earlier sections:

1. **The removal term needs flow *and* chemistry to evolve.** With either frozen it costs. This
   is a genuine three-way interaction, so removal, flow coupling and chemistry cannot be
   evaluated one at a time — and §7 evaluated flow with the other two switched off.
2. **Evolving flow alone is the best accumulate-only cell (0.219 → 0.395).** §7's "ordering is
   not flow-limited" was a statement about the *deploy score* (+0.0003), and it stands as that.
   As a statement about ordering it is too strong: flow is worth a lot of ordering. What it does
   *not* do is convert, and §7.2 already says why — the score is a threshold crossing at `crit`,
   not a ranking, so calibration gates whether any ordering gain is collectable.

Run on the real model path (`scripts/eval_flow_washout_2x2.py`, GT-flow gate — an **oracle**,
never a generalization claim), the same cross confirms nothing collectable yet: every cell sits
at score 0.765 ± 0.001, because only the gate evolves there and `ap`/`rp` are still t=0
constants.

### 9.5 Where this leaves the priorities

The term is implemented and **defaults to off** (`washout=0.0` reproduces every Phase 3–8
number bit-for-bit, and there is a test pinning that). It should not be switched on until the
inputs it depends on evolve.

* **Ranked first for score this week:** evolving flow through the **wall mask** (§10.4),
  ceiling **+0.051**.  The shipped slice is `GROW_HOPS` 20 (§10.5, +0.018 / +0.025 pred).
  Next is a deployable evolving gate (corrector / RGP-DEQ) that opens the nodes the t=0
  law never does, not a t=0 halo around `lss` (graded gate loses).
* **Then the coupled ODE change from §9.** Removal + evolving gate + evolving `ap`/`rp`.
  That is what off-wall `Mat` ordering needs; it does not ship on frozen inputs.
* **Closed by this section:** the 5.2× `d(Mat,t)/J0_Mat` residual of §2 no longer needs a
  bespoke explanation. A monotone accumulator fitted against a field that actually reaches a
  balance *must* show an inflated apparent rate, and the inflation is worst on the
  autocatalytic branch because that is the branch with no saturation of its own.
* **Dead, with numbers:** per-node `1/h_i` (−0.018), wall-tangential advection (0.000),
  saturation (0.000), `step2t` timing (correct already, and a no-op at 150 s sampling).

---

## 10. THE SCORE ON THE TABLE IS THE WALL MASK — and evolving flow is worth +0.05 there

§5–§9 chased off-wall `Mat` magnitude.  Three measurements this round say that is the
right *field* and the wrong *knob for a score this week*.

### 10.1 GT clot is `{Mat >= 2e7}`, wall and off-wall

`mu = mu_b * (mu2(FI) + mu1(Mat))`.  `mu1` steps 1→80 at `Mat = 2e7`; `mu2` steps 0→80 at
`FI = 0.6`.  Fibrin could have been a second clot route.  It is not
(`scripts/diag_fibrin_clot_route.py`):

| | |
|---|---|
| GT clot nodes with `Mat` below the platelet step | **0.0%** wall and **0.0%** off-wall |
| nodes with `Mat >= 2.35e7` that are *not* GT clot | **0.19%** |
| `FI_nd` max over the cohort | 8.8e-4 against `mu2` at 0.6 |

So the entire deploy score, wall and off-wall, is how well `{Mat >= crit}` is reproduced.
The 583 off-wall GT clot nodes carry real `Mat >= crit` of their own — they are not a
0.16-attenuated echo of the wall.

### 10.2 The attenuation form is near its ceiling; the model's `Mat` is not

`scripts/diag_offwall_owner.py`, GT wall `Mat`, topological shell:

| rule | off F1 | deploy score |
|---|---|---|
| `shell & (Mat_self >= crit)` — the field itself | 0.778 | **0.856** |
| `att * Mat_nearest_wall >= crit`, att=0.198 | 0.801 | — |
| same, topological owner (P2 bridge) | 0.667 | — |
| model's own `Mat`, any att | — | **0.766** |
| shipped speed arm | 0.160 | **0.783** |
| model-`Mat` OR speed | — | 0.784 |

A topological owner is *worse* than Euclidean nearest.  Restricting the speed arm to the
species shell raises strict off F1 0.160 → 0.267 and *lowers* deploy score 0.783 → 0.774
— the same relaxed-metric artefact as §8.  No scalar on the model's `Mat` beats speed.
Off-wall is blocked on wall-`Mat` ordering (`rho_corner = 0.193`), which §9 already
showed is an equation-level ceiling until chemistry evolves.

### 10.3 Wall error is the t=0 gate

`scripts/diag_wall_error.py`, 1753 GT wall clot nodes:

| | count | |
|---|---|---|
| FN, t=0 gate closed | **339** | 19.3% of GT; 335 of them *are* `Mat >= crit` |
| FN, gated but never committed | **0** | the ODE is not the wall-mask problem |
| FN inside the admission band | 181 | reachable by longer growth |
| FN outside admission | 158 | need the gate to open later |
| FP on a t=0 gate (over-ignition) | 133 | 8.6% of pred |
| FP by graph growth | **2** | growth is not the FP problem |

### 10.4 Evolving flow for the MASK, which §7 never measured

`scripts/eval_wall_gate_ceiling.py`, shipped speed lumen held:

| wall mask | score | wall F1 | Δ vs shipped |
|---|---|---|---|
| t=0 gate + 6-hop growth (shipped) | 0.7831 | 0.822 | — |
| **OR of the gate over GT flow, no growth** | **0.8338** | 0.917 | **+0.051** |
| wall `{GT Mat >= crit}` | 0.8772 | 0.947 | +0.094 |
| graded-gate t=0 halo (any tau/cut) | ≤ 0.766 | — | loses |

§7's "+0.0003" was evolving flow inside an accumulate-only *ODE* with a *frozen mask*.
The mask is where flow coupling actually pays.  A t=0 halo around `lss` (graded gate) is
the wrong surrogate: it opens the neighbours of the t=0 gate, not the nodes the gate
migrates to.  That is why the corrector / RGP-DEQ investment now has a **+0.05 ceiling
on the number of record**, acting through which wall nodes are allowed to ignite, not
through `Mat` ordering.

### 10.5 What ships: `GROW_HOPS` 6 → 20

The 181 in-band false negatives are a front-speed miss.  Swept on TRAIN, hard gate,
`relax = 2.0` (`scripts/eval_graded_gate_arm.py`):

| hops | score | Δ |
|---|---|---|
| 6 (was shipped) | 0.7831 | — |
| 12 | 0.7956 | +0.013 |
| **20** | **0.8016** | **+0.018** |
| 40 | 0.7939 | +0.011 |
| saturation of the admission component | 0.7894 | +0.006 |

Unimodal.  Leave-one-vessel-out picks 20 on every fold **of that TRAIN pool, which
includes DEV**.  Re-check under FIT/DEV before treating 20 as locked
(`scripts/eval_wall_protocol.py`).  Filling the whole component
destroys patient020 (0.725 → 0.596 at 66 hops) — the cap is a crude front speed, not a
bug.  Under deployable predicted t=0 flow, 15 vessels with `u0_pred`: **0.708 → 0.733
(+0.025)**.  Shipped in `scripts/predict_wall_clot.py`.

### 10.6 Protocol check (FIT / DEV, SEALED closed)

`scripts/eval_wall_protocol.py`, eligible full-horizon clot-carrying: FIT n=16, DEV n=3
(040/041/044; 039 is T=92 so dropped). SEALED not opened.

| | FIT | DEV |
|---|---|---|
| hops=6 | 0.7870 | 0.7624 |
| hops=20 (shipped) | **0.8006** | 0.8065 |
| hops=40 | 0.7901 | **0.8141** |
| extra seed hop<=4 & sr<2.5 lss | +0.0035 | +0.0141 |
| wake=1.0 re-grow | +0.0026 | +0.0030 |

DEV n=3 prefers hops=40 and the extra-seed rule because 041/044 sit in DEV -- that is the
leak the TRAIN-mean was hiding. FIT still wants hops=20. Same-sign FIT+DEV gains exist
(wake, extra-seed) but are not frozen and SEALED is not spent.

### 10.7 Domain targets: wall deploy > 0.9, off-wall deploy > 0.7

The blended full-mesh number (FIT 0.801 / DEV 0.807) hides two different remaining
problems. Domain-restricted scores (`docs/VIZ_STANDARD.md`: zero pred and GT outside the
domain, then `clot_score_from_deploy_dict`) under the FIT/DEV protocol, t=0 GT flow,
shipped hops=20 + speed lumen (`scripts/eval_domain_targets.py --flow gt`):

| | FIT | DEV | target |
|---|---|---|---|
| wall deploy | **0.858** | **0.890** | > 0.9 |
| off-wall deploy (vessels with off-wall GT) | **0.365** | **0.505** | > 0.7 |
| full mesh | 0.801 | 0.807 | — |

DEV wall is 0.01 short; FIT wall is 0.04 short, dragged by a handful of vessels, not a
uniform gap. Off-wall is not close.

**Wall error is two opposite failure modes**, so one global gate scalar cannot close 0.9
(`scripts/eval_gate_scalars.py`, `eval_sep_rank_cut.py`, `eval_persist_gate.py`):

| vessel | wall deploy | error |
|---|---|---|
| 018, 019, 025 | 0.58 / 0.65 / 0.81 | recall 1.0; all error is t=0-gate **FP** (weak separation) |
| 012, 028 | 0.75 / 0.58 | all FN are **ungated** (gate never opened) |
| 020, 021, 024, 036 | >= 0.97 | already over 0.9 |

Tightening `sgt` to -8.5e4 takes **FIT wall to 0.891** (full-mesh +0.026) and **DEV wall
-0.009**. Loosening `lss` to 30 takes DEV wall to 0.918 and crashes FIT -0.095. A
within-vessel sep-rank cut (drop weakest 10%) is the same disagreement. AND of the t=0
gate with the final-time gate sends 018 to **1.000** (those FP do close later) and
destroys 024/036 (0.986 -> 0.135 / 0.984 -> 0.250) because their clot is a *migrated*
gate, not a persistent core. Union-over-time can only add seeds; it cannot fix
over-ignition. Do not ship a new `sgt`/`lss`. SEALED stays closed.

**Off-wall > 0.7 is `{Mat_self >= crit}` on the species shell, not a wall-to-lumen map.**

| off-wall rule (wall mask held at shipped hops=20) | FIT | DEV |
|---|---|---|
| shipped speed | 0.365 | 0.505 |
| `att *` model wall Mat | 0.031 | 0.121 |
| flux / `u_n` residence on model Mat | 0.468 | 0.511 |
| speed OR flux (model Mat) | 0.477 | 0.575 |
| `att *` **GT** wall Mat | 0.577 | 0.835 |
| D=0 upwind of **GT** wall Mat | 0.394 | 0.619 |
| **`shell & (GT Mat_self >= crit)`** | **0.819** | **0.901** |

The Mat_self oracle clears 0.7 on both splits and lifts full-mesh FIT 0.801 -> 0.862.
Nearest-owner attenuation, even with **perfect** wall Mat, does **not** clear FIT 0.7
(0.577). Flux/residence on model Mat raises the off-wall *domain* score (same-sign
FIT+DEV) and **destroys full-mesh FIT 0.801 -> 0.701** by painting shell FP on wall-only
vessels (024/036 -0.32). Coupled ODE (`mat_linear` AP + washout + algebraic-wake `sr`)
is worse off-wall than shipped, same failure as §9 on frozen-quality inputs. Do not
ship flux, convection, or the coupled ODE.

**Deployable RGP-DEQ (`--flow pred`).** FIT+DEV packs now have `u0_pred` / `v0_pred` /
`sr0_pred` (`python scripts/precache_rgp_deq.py --cohort fitdev`). Rel L2 vs GT t=0 is
0.18 on 005 and 0.39-0.60 on the rest of the eligible set (040/041/044: 0.54 / 0.48 /
0.45). Shipped hops=20 + speed, MLS-on-`u0` for `sr`/`dsrx`:

| | FIT | DEV | target |
|---|---|---|---|
| wall deploy | **0.515** | **0.505** | > 0.9 |
| off-wall deploy | **0.175** | **0.265** | > 0.7 |
| full mesh | 0.461 | 0.443 | — |

The kinematics shear head is **not** a substitute for MLS-on-velocity. On patient005,
GT-MLS wall `sr` median is 193 1/s (2.5% below `lss=25`); the cached head is 54 1/s,
wall corr **0.17**, and `dsrx` from that field never trips `sgt` (wall gate mean 0.002).
MLS-on-`u0` keeps wall corr 0.82. Consuming `sr0_pred` dropped FIT wall to 0.240; do not
wire it. `t0_flow_fields(..., flow_source='pred')` always MLS-differentiates `u0_pred`.

DEV wall 0.505 vs GT-flow 0.890 is kinematics quality on 040/041/044, not the lumen arm
(GT-flow those three: 0.958 / 0.872 / 0.840). Do not retune `sgt`/`lss` on this gap.

**`tds2` discretisation (D=0, unique-upstream copy, no-slip wall BC) and the P2 first
cell** (`characteristic_origin`, `tds2_mat_field`, `grow_into_lumen_by_tds2`,
`grow_into_lumen_by_first_cell` in `physics_lumen_model.py`) were measured, not shipped:

| off-wall rule (GT t=0 flow, wall mask held) | FIT | DEV | full-mesh FIT |
|---|---|---|---|
| shipped speed | 0.365 | 0.505 | 0.801 |
| `tds2` (blend = owner-att) | 0.038 | 0.166 | 0.796 |
| `tds2` characteristics only | 0.026 | 0.121 | 0.795 |
| speed OR `tds2` | 0.372 | 0.526 | 0.802 |
| P2 first-cell (`u_n` at the wall-normal mid-edge) | 0.293 | 0.376 | 0.782 |
| first-cell committed (owner in wall mask) | 0.293 | 0.376 | 0.782 |
| speed AND flux | 0.297 | 0.353 | 0.797 |
| `oracle first-cell * GT` | 0.483 | 0.580 | 0.809 |
| `oracle tds2 * GT` (= owner-att) | 0.577 | 0.835 | 0.841 |
| **`shell & (GT Mat_self >= crit)`** | **0.819** | **0.901** | 0.862 |

Bulk-streamline `tds2` equilibrates the shell to the wall or misses it (CFL ≫ 1 in one
150 s step at pack `u_ref/d_bar`). Blended `tds2` is owner-att. Speed OR `tds2` is
same-sign FIT+DEV off-wall (+0.007 / +0.021) and does not drop full-mesh; it is noise
against 0.7. The P2-bridge first cell is the right face for `J/u_n` and still sits
below speed, with a GT-Mat ceiling of 0.483 FIT -- below owner-att's 0.577. Speed AND
flux protects full-mesh (0.797 vs flux's 0.701) by throwing away the recall flux was
buying. Do not ship `tds2`, first-cell, or flux.

What would actually hit the targets, given these ceilings:

* **Wall 0.9** -- under GT t=0 flow: a flow update that can *shut* 018-style weak sep
  gates without shutting the 024 migrated front. Global `sgt`/`lss` and AND-over-time
  are the wrong operators. Under deployable `u0_pred`: the Stage-A Rel L2 on this
  cohort (0.4-0.6) is the gate. MLS-on-`u0` is the shear operator; the shear head is not.
* **Off-wall 0.7** -- predict the shell's own `Mat`, not a lookup of wall `Mat`. Owner-att,
  D=0 characteristics, and the P2 first-cell balance are all below the FIT ceiling of
  0.577 with perfect wall Mat. `Mat_self` is 0.819 / 0.901.

---

## 11. REPRODUCE

```bash
python scripts/eval_offwall_mat_arm.py --flow gt              # the table in 4
python scripts/eval_offwall_mat_arm.py --flow gt --da-sweep   # 2.1
python scripts/diag_mat_ordering_flow_ablation.py --corrector # 7, needs CUDA for the corrector
python scripts/diag_offwall_mesh_portability.py              # 8
python scripts/diag_mph_surface_law.py                       # 9.1 the law, from the node tree
python scripts/diag_mph_time_units.py                        # 9.1 step2t is step4, not step2
python scripts/diag_local_ode_closure.py                     # 9.2 the oracle ceiling
python scripts/diag_local_ode_residual.py                    # 9.2 the dead rescues
python scripts/diag_mat_washout.py                           # 9.3 + 9.4 mechanisms, LOO, nulls
python scripts/eval_washout_arm.py --flow gt --lam-sweep     # 9.3 on the shipped model
python scripts/eval_flow_washout_2x2.py                      # 9.4 on the deploy score
python scripts/diag_fibrin_clot_route.py                     # 10.1 GT clot is {Mat >= crit}
python scripts/diag_offwall_owner.py                         # 10.2 field ceiling vs attenuation
python scripts/eval_speed_shell.py                           # 10.2 speed arm on the species shell
python scripts/diag_wall_error.py                            # 10.3 FN/FP split
python scripts/eval_wall_gate_ceiling.py                     # 10.4 evolving-gate mask oracle
python scripts/eval_graded_gate_arm.py                       # 10.5 hops sweep + LOO
python scripts/eval_hops20_pred_flow.py                      # 10.5 under u0_pred
python scripts/eval_wall_protocol.py                         # FIT/DEV/SEALED; SEALED stays closed
python scripts/eval_domain_targets.py --flow gt              # 10.7 wall/off domain scores vs 0.9 / 0.7
python scripts/eval_domain_targets.py --flow pred            # 10.7 deployable RGP-DEQ; needs u0_pred
python scripts/precache_rgp_deq.py --cohort fitdev           # bake u0_pred/sr0_pred on FIT+DEV
python scripts/eval_gate_scalars.py                          # 10.7 sgt/lss; FIT/DEV disagree
python scripts/eval_persist_gate.py                          # 10.7 AND-over-time precision oracle
python scripts/eval_sep_rank_cut.py                          # 10.7 within-vessel sep-rank cut
python -m pytest src/tests/test_ap_closure.py src/tests/test_temporal_wall_model.py \
                 src/tests/test_offwall_mat_arm.py src/tests/test_shipped_lumen_arm.py \
                 src/tests/test_wall_cohort_splits.py -q
```

`diag_mat_ordering_flow_ablation.py` without `--corrector` skips the two corrector arms and
needs no GPU. Each of its summary tables is restricted to vessels where **every** arm in
that table ran, so rows are like-for-like; the `u0_pred subset` table is the only place the
deployable `pred` arms appear.

| file | what it is |
|---|---|
| `src/core_physics/physics_lumen_model.py` | `grow_into_lumen_by_mat`, `fill_grown_wall_mat`, `MAT_ATTENUATION`; §8 shell helpers; §10.7 `grow_into_lumen_by_flux` / `grow_into_lumen_by_convection` / `grow_into_lumen_by_tds2` / `grow_into_lumen_by_first_cell` (measured, not shipped) |
| `src/core_physics/wall_cohort_splits.py` | FIT / DEV / SEALED for wall-cohort physics; `patient020` is FIT |
| `scripts/eval_wall_protocol.py` | §10.6 — hops / extra-seed / wake under that protocol, SEALED closed |
| `scripts/eval_domain_targets.py` | §10.7 — wall / off-wall domain scores vs 0.9 / 0.7 |
| `scripts/eval_offwall_mat_arm.py` | the arm comparison, full-mesh + wall-split |
| `scripts/diag_mat_ordering_flow_ablation.py` | §7 — the flow ablation and the qmatch decomposition |
| `scripts/diag_offwall_mesh_portability.py` | §8 — node families and shell formulations |
| `scripts/diag_mph_surface_law.py`, `diag_mph_time_units.py` | §9.1 — the law and the ramp, read out of the node tree |
| `scripts/diag_local_ode_closure.py`, `diag_local_ode_residual.py` | §9.2 — the oracle ceiling on a per-node ODE, and the rescues that failed |
| `scripts/diag_mat_washout.py` | §9.3/§9.4 — mechanism comparison, LOO, nulls, inputs × removal |
| `scripts/eval_washout_arm.py`, `eval_flow_washout_2x2.py` | §9.3/§9.4 — the same on the shipped model and the deploy score |
| `scripts/diag_fibrin_clot_route.py` | §10.1 — GT clot is `{Mat >= crit}` |
| `scripts/diag_offwall_owner.py`, `eval_speed_shell.py` | §10.2 — field ceiling, owner, speed ∩ shell |
| `scripts/diag_wall_error.py` | §10.3 — wall FN/FP split |
| `scripts/eval_wall_gate_ceiling.py`, `eval_graded_gate_arm.py` | §10.4/§10.5 — mask oracle and hops sweep |
| `scripts/predict_wall_clot.py` | `GROW_HOPS = 20`, `lumen="mat"` branch, `wall_mat_field` |
| `outputs/phase7_offwall_mat_arm.json` | per-vessel results |
| `outputs/phase7_mat_ordering_flow_ablation.json` | per-vessel §7 results |
| `outputs/phase7_offwall_mesh_portability.json` | per-vessel §8 results |

---

## 12. CORRECTION TO §9 — the equation is NOT short a term; the OPERATOR fails after t=0

Independent verification round, 2026-08-16. §9's baseline reproduces exactly (0.303 here
against 0.310 there, `scripts/diag_wall_mat_closure_terms.py`), but its **conclusion does
not survive contact with COMSOL's own fields.** Three measurements, in the order that
settles it.

### 12.1 COMSOL's own flux, integrated with no sink and no transport, ranks its own Mat at 0.855

`patient007` is the one vessel with a raw export, so it is the only place the law can be
evaluated on COMSOL's *own* `spf.sr`, `d(spf.sr,x)`, `Sat`, `rp`, `ap`, `Mas` rather than on
reconstructions. **SEALED vessel, used to validate an operator against COMSOL's own fields —
the use `PHASE6_HANDOFF` §6.1 explicitly permits. Nothing is fitted here.** 349 live wall
nodes, `spearman(., final GT Mat)`:

```
COMSOL's own J0_Mat column, integrated in time      0.855
law rebuilt from COMSOL's sr / dsrx  (A + B)        0.857
   ... low-shear branch B only                      0.866
   ... separation branch A only                    -0.108
COMSOL's own gate_low / gate_sep columns            0.857
integral of COMSOL's own d(Mat,t)  vs final Mat     0.999
```

The last line is the decisive one: **wall `Mat` is the exact time-integral of its own nodal
derivative**, so at the wall it is a pure accumulator and there is no sink to find. And the
first line says COMSOL's own deposition flux — integrated locally, no removal, no transport,
no neighbours — ranks COMSOL's own final wall `Mat` at **0.855**.

**§9.2's "0.310 is a ceiling on the equation, not on any input model" is wrong.** It is a
ceiling on the *inputs that diagnostic supplied*. Consequently §9.3's washout term is
compensating an input error, and §9.5's plan to "switch removal on once the inputs evolve"
should be dropped rather than deferred. Keeping `washout=0.0` as the default was right.

### 12.2 What actually breaks: MLS `d(sr,x)` collapses once a clot exists

`sr_t` / `dsrx_t` in `outputs/wall_species_cache/` are MLS derivatives of GT velocity on the
**downsampled pack graph**, not COMSOL's columns. On `patient007`, 583/583 wall nodes matched
to the export:

```
                     rho(sr)   ratio   rho(dsrx)   ratio   sep-gate Jaccard   low-gate Jaccard
MLS @ t=0              0.998   0.983       0.992   1.026              0.911              1.000
MLS @ t=final          0.994   0.973       0.346   0.901              0.000              0.956
```

At `t=0` the operator is excellent and the repo's validation (rank 0.990) holds. At `t_final`
`d(sr,x)` degrades to rank **0.346** and the separation gate has **Jaccard 0.000** — MLS
fires on 18 nodes, COMSOL on 62, with **zero overlap**. `sr` itself stays fine (0.994); it is
specifically the *derivative* that fails, because a clot puts an 80x viscosity jump into the
velocity field and a 3-hop graph stencil on a coarsened mesh cannot resolve its gradient.

A wall-chain (boundary-layer two-point) alternative is **not** the fix and is worse: `sr`
rank 0.996 but magnitude ratio 0.73, and `dsrx` rank **-0.199**. The reason is structural —
`d(spf.sr,x)` is a derivative with respect to *global x* evaluated at the wall, so on a
vertical wall segment it is the wall-**normal** derivative. A derivative along the wall chain
cannot represent it. MLS has the right structure; it needs accuracy, not replacement.

### 12.3 The separation branch's MLS magnitude is anti-informative, and capping it is free

`scripts/diag_gate_branch_magnitude.py`. The gate is `A + B` with
`A = (L/gamma_m)*|d(sr,x)|` (a **magnitude**) and `B = 1` (an **indicator**), and
`L/gamma_m = 0.05` against `|d(sr,x)| ~ 1e3`, so **A outweighs B by ~50x wherever it fires**.
Sweeping how much of `A` the *rate* may see (the **mask** is untouched — §10.3 shows
graph-growth FP is 2 nodes, so mask and rate are separable), oracle inputs, 19 train vessels:

```
A cap        all (19)   low-shear (15)   sep-only (4)   anti-corr
inf (ships)     0.303            0.492         -0.406        5/19
1.0             0.302            0.495         -0.422        5/19
0.3             0.399            0.537         -0.121        2/19
0.1             0.501            0.667         -0.121        2/19
0.0             0.703            0.703            n/a        0/19
```

On the 15 vessels where the low-shear branch fires at all, dropping `A` from the **rate**
takes ordering **0.492 -> 0.703 and removes every anti-correlation**. On the 4 vessels where
`A` is the only source it is **-0.406** — actively backwards. With COMSOL's own `dsrx` (§12.1)
branch `A` is merely uninformative alone (-0.108) and harmless in the sum (0.857 vs 0.866),
so this is an operator defect, not a defect in COMSOL's law.

Neighbour mixing — `(I + kappa*L)^-1` on the local integral, testing whether COMSOL's
consistent mass matrix explains the residual — moves the baseline 0.303 -> 0.404 at
kappa=16 but adds **nothing** on top of the cap (0.703 -> 0.704). Not the mechanism.

### 12.4 Frozen inputs, not a missing sink, are what cost the ordering

Same law, same vessel, COMSOL's own fields, `spearman(., final GT Mat)` on live nodes:

```
patient007, COMSOL operators        chem frozen    chem evolving
flow frozen at t=0                       -0.161           -0.233
flow evolving                             0.651            0.857
```

Repeated on all 19 train vessels with the MLS operators that actually exist today:

```
19 train vessels, MLS operators     chem frozen    chem evolving
flow frozen at t=0                        0.234           -0.027
flow evolving (GT)                        0.404            0.303
```

Two things follow:

* **Evolving the gate is the dominant lever on wall `Mat` ordering**, worth +0.17 cohort-wide
  with today's operators and +0.81 on `patient007` with COMSOL's. It improves 13/19 vessels.
* **The "evolving chemistry makes it worse" inversion in §9.4 is an operator artefact.** With
  COMSOL's fields evolving both is *best* (0.857 > 0.651); with MLS it is worse
  (0.303 < 0.404), because the late-time `dsrx` noise compounds over 201 steps.

§7's "ordering is not flow-limited" was also measured through the collapsed operator and on
`rho_all` (which §8.5 shows is inflated by structural zeros). On live nodes with a working
operator, ordering is **mostly** flow-limited. §7's separate finding — that the evolving-flow
gain did not convert to deploy score with the mask frozen — is unaffected, and §10.4 already
showed the mask is where it converts (+0.051).

### 12.5 Revised priorities

1. **Stage-A flow quality is the largest single number on the board and is not a physics
   problem.** Wall deploy goes **0.858 -> 0.515 (FIT)** and **0.890 -> 0.505 (DEV)** moving
   from GT `t=0` flow to RGP-DEQ `u0_pred` (§10.7). That -0.34 dwarfs every physics arm in
   Phases 7-8 combined (all +-0.02-0.05), and the target is 0.9 *deployable*. AGENTS.md quotes
   the production kinematics at Rel L2 **~0.087**; this cohort is at **0.39-0.60**. A 5x
   degradation is a checkpoint/preprocessing discrepancy to hunt, not a research programme,
   and nothing else can matter until it is closed.
2. **Cap branch `A` in the rate** (`A_cap ~ 0` on today's operator, keeping `A > 0` in the
   mask). Zero parameters, +0.21 oracle ordering, removes all anti-correlations. Must be
   checked on the model path and under FIT/DEV before shipping.
3. **Evolving gate, for the mask *and* the rate.** §10.4's +0.051 mask ceiling and §12.4's
   ordering gain are the same mechanism, and the corrector already recovers ~40% of the gates
   the GT-flow oracle opens (§7.1). This is the one physics arm with two independent payoffs.
4. **Do not ship the washout term** (§12.1), and do not spend more on removal, saturation, or
   per-node `1/h`.
5. **Off-wall: the residual is the attenuation's *variance*, not its form.** `att*Mat_owner`
   with perfect wall Mat is 0.577 FIT against `Mat_self`'s 0.819. Within a vessel
   `Mat_off/Mat_owner` spans ~0.12-0.19 while its median is 0.16 everywhere, and near a
   threshold that spread is the whole gap. If the 0.16 is the FEM consistent-mass/first-cell
   coupling, `att_i` is computable from the mesh — the packs already carry `Laplacian`, `W`,
   `V`, `M_inv`. That is the cheapest untried route to off-wall 0.7 and needs no ML.

### 12.6 Reproduce

```bash
python scripts/diag_wall_mat_closure_terms.py     # 12.1/12.3 gate shape, mixing, time-weighting
python scripts/diag_gate_branch_magnitude.py      # 12.3 the A-cap sweep
```
