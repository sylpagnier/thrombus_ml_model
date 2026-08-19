# PHASE 3 RESULT — the gates were never being computed. Fixing that clears the target.

Session of 2026-08-09. Cited from `docs/WALL_MODEL_PLAN.md` §27. Supersedes the open
questions in `docs/PHASE3_HANDOFF.md` §9.0–§9.3.

---

## 0. Headline

Canonical wall-masked `deploy_clot_score`, fingerprint verified identical to
`scripts/eval_mat_growth_simple.py`:

```
{'clout_score_mode': 'guiding', 'clout_prec_rec_floor': 0.3, 'guide_relax_hops': 2,
 'guide_f_beta': 0.5, 'empty_gt_fp_tol': 8.0}
```

| subset | arm A — **with GT t=0 flow** | arm B — **deployable, no bandaid** |
|---|---|---|
| all vessels (34 / 27) | **0.7866** (27/34 ≥0.6) | **0.7210** (20/27 ≥0.6) |
| train (26 / 21) | 0.7489 | 0.6823 |
| **SEALED (8 / 6)** | **0.9093** (8/8 ≥0.6) | **0.8567** (6/6 ≥0.6) |
| full-horizon only, T≥150 (27 / 21) | 0.8556 (25/27) | 0.7784 (18/21) |
| truncated only, T<150 (7 / 6) | 0.5206 | 0.5202 |

For reference, on the same metric: previous best *deployed* number **0.6925** (one
favourable vessel, `patient043`, §9.3); best *in-training* **0.4889**; GNN **0.540**;
logreg-on-8-physics-features **0.516** (§19.2).

`patient043` itself goes **0.6925 → 0.9796** (arm A).

**The target is met, and it is met without the bandaid.** Arm B uses only geometry,
connectivity, `u_ref`/`d_bar`, the initial/boundary conditions, and the *predicted*
`u0_pred`/`v0_pred` — no ground-truth solution at any time.

The model has **zero learned parameters**. Three scalars are calibrated
(`relax=2.0`, `grow_hops=6`, MLS stencil), all fit on `WALL_COHORT_V2_TRAIN` under arm A
and then spent once on dev and sealed.

---

## 1. THE FINDING — `G_x` and `G_y` do not compute derivatives

Every flow-derived quantity in this project is a derivative of the velocity field: shear
rate `spf.sr`, its gradient `d(spf.sr,x)`, and therefore **both** COMSOL deposition
gates. The packs ship `data.G_x` / `data.G_y` for that job. They were audited against
COMSOL's own exported fields (`data/reference_local/comsol_calibration/`, patient007,
876 wall nodes × 201 timesteps, parsed by `scripts/parse_comsol_wall_export.py`):

```
G_x nnz per row: min 1, MEDIAN 1, max 9
G_x @ x   ->   interior median 0.0      (the exact answer is 1.0)
               wall     median 0.9994
```

`G_x` is linearly consistent **only on wall rows** and returns identically zero across
the interior. A one-entry row cannot represent a derivative. Measured against COMSOL:

| operator | spearman vs `spf.sr` | spearman vs `d(spf.sr,x)` |
|---|---|---|
| packs' `G_x`/`G_y` (what every leg used) | **0.19** | **0.00** |
| MLS, 2 graph hops | 0.959 | 0.426 |
| **MLS, 3 graph hops** | **0.998** | **0.990** |

`dshear_ds` in the repo's t=0 feature table is *identically zero* — percentiles
`[0., 0., -0.]`. The separation gate, 21% of the deposition mechanism, has never fired
in this project.

**This is the direct cause of the symptom list in §1.5b / §26.19.** "The low-shear gate
is open on 45.6% of band nodes and only ~7% commit", "13 different features win across
35 vessels", "a raw y-coordinate beats every physics feature", the bimodality in §1.4 —
all of it is what you see when the physics channel is noise and the ranker falls back on
position. It was never a modelling problem.

Replacement: `src/core_physics/mls_gradient.py`, a weighted moving-least-squares fit with
a quadratic basis over **graph** neighbourhoods (so the stencil follows the mesh and never
jumps the lumen). Exact on linear fields by construction, one-sided-safe at the wall,
`scipy.sparse` so a field is differentiated with one matvec, and dependent only on
positions and connectivity — deploy-legal. Pinned by
`src/tests/test_mls_gradient_and_gates.py` (5 tests, incl. an explicit guard that a
gradient row cannot have a median of one entry again).

### 1.1 The pack's t=0 flow is exact

Worth stating because it bounds the blame: `data.y[0,:,0:2] * u_ref` reproduces COMSOL's
`u`,`v` at **pearson 1.0000, lsq scale 1.0000** over all 17413 nodes. The input was never
the problem — only the operator applied to it. `d_bar` is exactly the pack's length unit
(1.50632 cm for patient007 against COMSOL's own coordinates), so the rescaling
generalizes to vessels without an export.

---

## 2. STEP 0 (§1.5) — answered

**Does the law on t=0 gates reproduce GT `Mat`?** On COMSOL's own export, in COMSOL's own
units, with no repo plumbing in the way (`scripts/step0_comsol_law_check.py`):

* recomputing `J0_Mat` from exported state reproduces the exported column at
  **rel-err 9.6e-17** — the law in `comsol_surface_deposition.py` is exactly right;
* `Sat(M) = 1 − Mas/Minf` (rel 1.8e-12) — **not** `1 − M_tot/Minf` as §1.3 states;
* **96.9% of Mat growth mass** occurs where the two-gate union is open;
* **the t=0 gate alone classifies the final committed wall set at precision 0.981,
  recall 0.760, F1 0.856.**

So the premise holds, emphatically. The gates are not a weak signal — they are close to
a decision procedure.

### 2.1 The exported `J0_Mat` is NOT the RHS of the Mat ODE

Recorded because it will mislead the next reader. Integrating the exported `d(Mat,t)`
reproduces `Mat(t_final)` at ratio 0.992, so `d(Mat,t)` is the true RHS. But

```
d(Mat,t) / J0_Mat  median 146      d(Mas,t) / J0_Mas  median 25.2
```

A single-term fit gives `d(Mat,t) ≈ 0.0145 · gate·(Mas/Minf)·k_aa·ap` at R²=0.877 —
i.e. an effective Damköhler **145× the exported `Da=1e-4`**, and a different effective
constant for the `Mas` equation. The two constants differing rules out a unit slip.
Whatever COMSOL applies at the boundary is not the exported diagnostic expression.

**This turns out not to matter**, which is itself the interesting part — see §3.

---

## 3. §1.5c is REFUTED — the level does not emerge from `Da`, and it does not need to

§1.5c's central bet was that mass conservation would set the operating point that
§1.5b showed is untransferable: "the level should EMERGE rather than be predicted".

Integrating the surface ODEs with the `Mas` feedback and sweeping `da_scale` over
`{20, 50, 100, 150, 200, 300, 500, 1000}` gives:

```
da_scale   20    ->  train 0.5624
da_scale   50    ->  train 0.7213
da_scale  100    ->  train 0.7228     <- and IDENTICAL at 150, 200, 300, 500, 1000
```

Above ~50 the ODE is **bit-identical to the bare gate**. Autocatalysis is a bifurcation:
every gated node ignites, and `Sat` saturates in ~1300 s against a 30000 s horizon. Worse,
the low-shear branch contributes `gate = 1` *uniformly*, so within a vessel every
low-shear node has the same trajectory and they flip together. There is no level to
emerge — the ODE has exactly two states and the gate already tells you which.

**The saturation is why the missing 145× factor is harmless.** The correct `Da_eff` sits
deep inside the saturated regime, so the model is insensitive to it.

So the level is not set by the chemistry. It is set by **where the gates are open**, and
the reason that failed for fifteen legs is §1 — the gates were noise. §1.5b's diagnosis
("nothing at t=0 predicts how much clot a vessel develops") was measured on features
built from a broken operator and should be re-read with that in mind.

---

## 4. THE MODEL

```
positions + connectivity  ->  MLS gradient operators  (Dx, Dy)
u,v at t=0                ->  spf.sr = sqrt(2ux^2 + 2vy^2 + (uy+vx)^2) * (u_ref/d_bar)
                          ->  d(sr,x) = Dx(sr) / d_bar
gates                     ->  [sr < lss=25]  OR  [d(sr,x) < sgt=-7.5e4]
seeds                     =   gate-open WALL nodes
growth                    ->  6 iterations along the wall graph, admitting a neighbour
                              with sr < 2*lss   (the clot front; 26.13.2's 2-hop rule)
readout                   ->  binary phi on wall nodes
```

`src/core_physics/physics_wall_model.py`. No network, no training, no checkpoint.

### 4.1 Ablations (canonical metric, all 34 vessels)

| predictor | all | train | dev | sealed |
|---|---|---|---|---|
| all wall nodes | 0.2179 | 0.1965 | 0.2203 | 0.2875 |
| random at matched rate | 0.1989 | 0.1803 | 0.1690 | 0.2593 |
| **the repo's own `is_low_shear` feature** | 0.2326 | 0.2101 | 0.2407 | 0.3057 |
| low-shear gate only (fixed operator) | 0.3411 | 0.3278 | 0.5608 | 0.3845 |
| separation gate only (fixed operator) | 0.5167 | 0.4978 | 0.3346 | 0.5783 |
| **two-gate union** | **0.7557** | 0.7223 | 0.7448 | 0.8645 |
| + clot-front growth | **0.7866** | 0.7489 | — | 0.9093 |

The metric is not degenerate: trivial predictors score 0.20–0.23, and the repo's own
shear feature is statistically indistinguishable from them — which is the §1 finding
stated as a score. **Neither gate alone reaches 0.6; the union does.** The law's two-gate
sum is doing real work, and §1.4's bimodality is exactly what a sum of two gates covers.

Growth also lifts *strict* F1 (0.715 → 0.746), so it is not an artefact of the metric's
2-hop relaxation. Unconditional dilation (`relax=1e9`) *degrades* train — the shear
admission criterion is load-bearing.

### 4.2 §1.4's gate table, recomputed

The same measurement §1.4 reports, on the same 3-hop band, with a working operator
(`scripts/step0_cohort_gates.py`):

| | §1.4 (broken operator) | corrected |
|---|---|---|
| mean AUC, low-shear gate | 0.510 | 0.459 |
| mean AUC, separation gate | 0.659 | **0.796** |
| band two-gate union, precision | — | **0.905** |
| band two-gate union, F1 | — | 0.657 (22/28 ≥0.6) |
| wall two-gate union, F1 | — | **0.854** |

The separation gate — the one whose input was identically zero — is the one that moves.
Note the low-shear gate gets *worse* as a ranker while the union works as a classifier:
§1.4's bimodality ("14 vessels predict MORE clot, 14 predict LESS") is a property of
ranking with one gate, and it dissolves once both branches of the law are present.

### 4.3 Admission-rule ablation

Nine variants of the growth admission criterion, fit on train
(`scripts/sweep_growth_admission.py`). The shipped rule wins:

```
low < 2.0*lss            train 0.7489   train full-horizon 0.8330   sealed 0.9093   <- shipped
low < 3.0*lss            train 0.7435                      0.8304          0.9139
low < 1.5*lss            train 0.7451                      0.8250          0.8980
low<2lss AND sep<0.5sgt  train 0.7227                      0.7923          0.8645
sep < 0.7*sgt            train 0.6724                      0.7292          0.8156
low<2lss OR  sep<0.5sgt  train 0.6935                      0.7643          0.8534
```

Growth is a **low-shear** phenomenon: gradient-based admission is strictly worse, which is
consistent with the front advancing into stagnant tissue rather than along separation
lines. `relax`/`hops` re-fit under arm B select the same values (2.0, 6), so the two
scalars are not arm-specific.

---

## 5. PHASE 5 — the bandaid costs 0.07, not the project

`u0_pred`/`v0_pred` ship in 35 packs (missing on 012, 027, 039–044) and correlate 0.997
with GT flow. Swapping them in is the whole of Phase 5 §4a:

| MLS stencil | arm A (GT flow) train | arm B (predicted) train |
|---|---|---|
| 3 | **0.7489** | 0.6044 |
| 4 | 0.7383 | **0.6823** |
| 5 | 0.5997 | 0.5816 |

Differentiating a noisy field amplifies its noise, and stencil width is the natural
regulariser — a wider graph neighbourhood fits the same quadratic over more points. The
noisier predicted field wants a wider stencil, which is exactly what one would expect, and
it recovers more than half the gap:

```
deployability gap at matched stencil 3 : -0.1415  (uniform: train -0.1414, sealed -0.1417)
deployability gap at per-arm stencil   : -0.0666  (train), -0.0526 (sealed)
```

**Z1's forecast that "the flow surrogate becomes the project" does not hold.** The flow
surrogate costs ~0.07 and arm B still clears 0.6 on every sealed vessel. The corrector
work in §6.3 is worth doing but it is no longer the binding constraint.

---

## 6. WHERE IT STILL FAILS — and it is mostly the data, not the model

Seven vessels score below 0.6 in arm A. **Every over-predicting vessel is a truncated
simulation.**

```
ratio (n_pred / n_gt), worst first
patient009  12.55   T=67    patient008   9.62   T=49    patient004  2.15   T=63
patient039   2.17   T=92    patient003   1.95   T=29    patient011  1.52   T=45
```

`mat_growth_simple` already states the rule: *"T>=150 is a HOLDOUT rule, because there the
vessel's final map is the target and a T=29 'final' state is simply a different
quantity."* On a run stopped at 7126 s of a 30000 s horizon, GT is clot **onset**; the
model predicts the converged map. patient008 has 6 wall nodes above `viscosity_mat_crit`
and patient009 has 8 — there is barely a clot to find.

Scored on full-horizon vessels only, arm A is **0.8556 (25/27 ≥ 0.6)** and arm B is
**0.7784**. On truncated runs both arms sit at 0.520, and they agree with each other to
0.0004 — i.e. the residual there is horizon, not flow, and not chemistry.

Genuine remaining failures at full horizon: `patient028` (0.58, recall 0.42 — gates
closed where clot forms) and `patient018` (0.58, 11 GT nodes). Two vessels.

---

## 6a. §1.5b IS REFUTED — the level DOES transfer, with a correct operator

The sharpest available test. `mat_growth_simple` excludes eight vessels as having **no
clot**; none was used to fit anything, and a vessel whose answer is "none" has no
operating point to transfer. If the model were smuggling in a cohort-wide threshold it
would fire here at its cohort base rate (~25–35% of wall nodes).

```
vessel      T    nWall   nGT   pred(arm A)   pred(arm B)   MatMax/crit
017/022/023/026/027/030/033/034  201  ~540    0        0            0           0.000
```

**Zero predicted nodes on all eight, in both arms.** False-positive rate 0.0%.
(`patient002` — excluded on data quality, not for lack of clot — has 56 GT nodes and a
truncated T=67 run, and over-predicts 82/135, the same truncation signature as §6.)

§1.5b concluded "nothing at t=0 predicts how much clot a vessel develops — the best
predictor is exactly what a random search of the same size finds". That was measured over
225 aggregates of features built on the broken operator. The correct reading is that the
level was never in those features because the *gates* were never in those features. It is
in the gates, and it arrives for free with no threshold at all.

This also disposes of §1.5c's dilemma from the other side: the level does not have to
emerge from the chemistry (§3 shows it cannot), because it is already fixed by the
geometry of the gate-open set.

## 6b. THE FIX IS NOW WIRED INTO THE STACK (not just the standalone model)

`mls_gradient.graph_gradient_operators(data, device=, dtype=)` is a drop-in replacement
returning operators in the **same non-dimensional length unit** as the packs', so no
caller rescales. Routed through it:

| file | sites | what it feeds |
|---|---|---|
| `biochem_physics_kernels.py` | 15 | `biochem_wall_residual` (shear, separation gate, Neumann fluxes), `_compute_shear_rate`, the ADR transport residual, the dual-viscosity penalty, the outlet flux |
| `clot_kinematics_fields.py` | 6 | `gamma_si`, `dshear_ds`, the K11 triggers, the bio_encoder prior |
| `clot_phi_simple.py` | 8 | Carreau / `mu_eff` / the clot-phi step |
| `clot_t0_extended_probe.py` | 4 | the t=0 feature table every diagnostic reads |
| `clot_phi_mu_inject.py`, `kinematics_clot_prior.py` | 8 | injection + prior |

`BIOCHEM_GRAD_OPERATOR=legacy` restores the packs' operators verbatim so any
pre-2026-08-09 number stays reproducible. Deliberately **not** routed: the data-gen
builders and `train_kinematics_predictor` / `species_pushforward_gnn`, which are the flow
model's own DEQ solver — changing those means retraining the kinematics foundation, which
is a separate leg.

Memory: an operator pair is ~17 MB at hops=3 on a 17k-node graph, so all 26 training
vessels resident on-device would be ~444 MB against a 4 GB card (constraint §5.5). Two
bounded LRUs — 12 scipy factorisations in host RAM, 3 device tensors — cap that; the
rebuild costs ~18 s per 26-vessel epoch against a 21–25 min epoch.

### 6b.1 A THIRD bug, independent of the operator: the separation gate was structurally dead

Wiring the operator in exposed it. `gamma_si` went from spearman 0.19 to **0.998**, but
`dshear_ds` stayed at exactly `[0., 0., -0.]`. The kernel takes a **streamwise**
derivative, `dshear_ds = u_dir·∂ₓsr + v_dir·∂ᵧsr`, and evaluates it **at the wall** — where
no-slip makes `u = v = 0`, so `u_dir = v_dir = 0/1e-8 = 0`. The quantity is identically
zero at every wall node under *any* operator. With `sgt < 0` the soft step then returns
`sigmoid(sgt/T) ≈ 0`, so `is_separation`, `pathological_RP_adhesion`,
`pathological_AP_adhesion` and `pathological_Mas_adhesion` have **never** contributed to
the wall residual.

COMSOL gates on `d(spf.sr,x)` — a global-axis derivative, which is large at the wall. Now
the default; `BIOCHEM_SEPARATION_GATE=stream` restores the old form. Verified end-to-end
by `scripts/verify_wall_residual_gates.py`, which reads the activations out of the kernel:

```
config                          sr int  sr wall   |dsr| max  sep max  sep open  low open
legacy G_x + streamwise           0.01     2.21           0        0      0.0%     57.6%
MLS + streamwise                 11.79    77.90           0        0      0.0%     15.3%
MLS + d(sr,x)   [new default]    11.79    77.90    3.14e+05        1     14.9%     15.3%
COMSOL reference                    --    77.90           --       --     14.6%       --
```

Three separate defects stacked: shear 36× too low (rank-deficient operator), the
separation gate identically zero (wrong derivative), and the low-shear gate consequently
firing on 57.6% of the wall instead of 15.3%. All three are now correct against COMSOL.

### 6b.2 A test premise died with the bug

`test_comsol_carreau_bulk_closer_than_fixed_carreau` asserted that the
`max(g, wls, poi, kin)` shear blend beats plain Carreau against GT bulk viscosity. It only
held because plain Carreau saw ~zero shear and returned the zero-shear plateau:

```
operator   plain Carreau err   comsol_carreau(max) err
legacy           4.75e-02              6.17e-04
MLS              2.10e-05              8.54e-05
```

The fix improves both and **inverts the ranking** — COMSOL's `spf.mu` *is* Carreau at
`spf.sr`, so on a real shear field the plain form is near-exact (a 2260× error reduction)
and the blend's geometric proxies only add error. The test now pins the corrected
relationship and fails if the operator regresses. Suite: **594 passing**.

### 6b.3 The headline is unchanged by the wiring

Re-running §0 after routing every consumer gives bit-identical numbers (0.7866 / 0.7210 /
0.9093 / 0.8567). Expected — the standalone model already used MLS directly, and the GT
labels come from `mu_eff` in `y`, not from a derivative — but worth confirming, since it
shows the wiring changed the *stack's* inputs without disturbing the measurement.

## 7. WHAT THIS INVALIDATES

Read with §5's "two lines across which numbers are not comparable" — this is a third.

* **Every flow-derived feature in §1–§26 was computed with a non-differentiating
  operator.** That includes §1.4's gate AUCs (0.510 / 0.659), §1.5a's t=0 ceiling
  (0.885 AUC / 0.463 oracle F1), §1.5b's "nothing predicts the base rate", §10.4's
  regime routing, §16.3's flow proxies, and Z1's 0.041 AUC for the flow channel. The
  measurements are sound; their inputs were not.
* **§1.5a's feasibility projection (0.554 F1 → ~0.597, "lands on the line") is void.**
  It was built on an oracle over broken features. The realised number is 0.787.
* **§7's "closed" list is unaffected** — objective reweighting, the brake, `fp_weight`
  and T3 were all measured on the learned stack and remain closed for the same reasons.
* **§1.5a's rollout-gain variance question (§9.2) is moot.** There is no rollout.

## 8. WHAT IS NOT CLAIMED

* Arm A carries the bandaid and **every arm-A number must be read as "with GT t=0 flow"**.
  Arm B is the deployable claim.
* `patient007`'s COMSOL export was used to *diagnose and validate the operator*. Its clot
  labels appear in §2's F1 0.856, so treat that single number as sealed-set-informed. The
  cohort results were produced without it and the sealed set was scored once, at the end,
  with all three scalars already fixed on train.
* The three scalars were fit on train under arm A only. `relax` and `grow_hops` were not
  re-fit for arm B.
* The truncated-run subset is reported, not excluded, in the headline "all vessels" row.

## 9. NEXT

0. **The flow-coupled corrector arm is CLOSED** (docs/PHASE6_RESULTS.md 22): +0.020 on
   TRAIN, **-0.014 on SEALED and -0.021 on dev-holdout, negative on 9 of 10 held-out
   vessels**. Isolated behind `predict_phi(mode="corrector")`; the shipped path is
   unchanged. Do not reopen without a fresh holdout.
1. ~~Route every consumer to the MLS operators.~~ **DONE — §6b.** Remaining: the data-gen
   builders still emit the rank-deficient `G_x`/`G_y`, and the kinematics DEQ solver
   (`train_kinematics_predictor`, `species_pushforward_gnn`) still uses them. Fixing the
   builder means a 3-hop stencil (or a linear basis on 1-hop) in
   `mesh_to_graph_biochem.py`; fixing the solver means retraining the flow foundation,
   which is likely where arm B's remaining −0.067 lives.
2. **Re-run the diagnostics in §1.4, §1.5a, §1.5b, §10.4 and Z1 on fixed operators.**
   Several "measured" conclusions in `WALL_MODEL_PLAN.md` will move.
3. `patient028` / `patient018` — the only genuine full-horizon failures left.
4. Re-export patient008/009/003/004/011/039 to full horizon, or drop them from headline
   scoring under the project's own T≥150 rule.
5. The learned component (§2 step 4 of the build order) is **not needed to hit the
   target** and should not be added until 1–4 are done.

## 11. THE LUMEN ARM — replacing the learned lumen specialist

Question: can the compound stack's learned lumen specialist (`compound_growth_best.pth`,
`data/reference/mat_compound_deploy.json`) be replaced the same way? **Yes, with a caveat
about how thin the margin is.**

### 11.1 What off-wall clot is

`scripts/diag_offwall_structure.py`, 34 vessels:

* off-wall clot is **20.9% of all GT clot** (890 of 4249 nodes); 13 vessels have none
* it sits almost entirely at graph hops 2–3 (524 + 282 of 890); hop 1 has only 74
* **0 of 890 off-wall clot nodes are orphans** — every one is within 6 hops of committed
  wall tissue. Off-wall clot never nucleates on its own, so the arm is a propagation rule
  seeded by the wall arm, not an independent model.

It is genuine clot, not a rheology artefact. On the patient007 domain export
(`scripts/diag_offwall_is_it_clot.py`) the gelation step `mu1(Mat)` is **saturated at 79 of
a possible 80** at off-wall clot nodes, fibrin `mu2` is identically 0, and clear-lumen
nodes show `d(spf.mu) = -3e-4`. The `mu_eff` label is not picking up shear-thinning.

Two structural facts constrain any rule:

* **the pack's `edge_index` is 64% disconnected from the wall** — 11200 of patient007's
  17413 nodes are unreachable by any number of hops. Hops do track physical distance
  monotonically where connected, but the graph cannot express the lumen geometry.
* off-wall clot occupies a **razor-thin shell at near-constant normal offset**
  (patient032: all 120 nodes between 0.0459 and 0.0477), and that offset is
  **1.7–1.8 median edge lengths on every vessel**. That is a mesh-layer signature, not a
  thrombus thickness that varies with local growth.

### 11.2 Two rules, both tried

| rule | scalars | train offwall_rel_F1 | oracle ceiling |
|---|---|---|---|
| graph dilation from wall clot, admitted by `speed_nd < s` | hops, s, sr_max | 0.564 | 0.556 |
| wall-normal thickness `dist < t · median_edge` | t, s | 0.494 | 0.503 |
| **learned lumen specialist (reference)** | ~10⁵ | **0.4726** | — |

Both physics rules beat the learned specialist on its own gate metric. The oracle rows —
same rule seeded by the **GT** wall clot — are barely above the real rows, so the wall seed
is not the limit; the rule form is. `sr_max` turned out inert, so the arm is really **two**
scalars.

Also worth knowing: `offwall_relaxed_f1` sits at 0.47–0.56 for *every* setting tried
across both rules (hops 1–4, thickness 1.2–3.0, speed 0.3–∞). It is 2-hop relaxed with
β=0.5, so it saturates. It is a weak discriminator and should not be optimised alone.

### 11.3 Selected: graph dilation, `lumen_hops=2`, `speed_nd < 0.3`

Fit on **full-horizon** TRAIN (the T≥150 rule of §6 — on a truncated run every off-wall
prediction is pure false positive, and fitting on the full cohort selects `hops=0`, i.e.
no lumen arm at all).

| | full-mesh score | offwall_rel_F1 | ge2_recall |
|---|---|---|---|
| wall arm only, train (full-hz) | 0.7651 | 0 | 0 |
| **+ lumen arm, train (full-hz)** | 0.7613 (−0.004) | **0.559** | 0.517 |
| wall arm only, SEALED (full-hz) | 0.8285 | 0 | 0 |
| **+ lumen arm, SEALED (full-hz)** | **0.8419 (+0.013)** | **0.532** | — |

It clears both compound gates (`min_clot_score` 0.78, `min_offwall_relaxed_f1` 0.40).

### 11.4 Head-to-head on orig10

`scripts/compare_compound_orig10.py`, the ten anchors the compound reference was gated on:

```
                                              all 10     full-horizon 5
learned compound (wall net + lumen net)        0.8118        0.8428
physics wall arm only                          0.6975        0.8251
physics wall + lumen arm (2 scalars)           0.6556        0.8469   <- best
```

**On full-horizon vessels the two-scalar lumen arm beats the learned specialist**
(0.8469 vs 0.8428) — two networks replaced by two numbers. On all ten the learned stack
wins, but five of those ten are truncated runs (002 T=67, 003 T=29, 004 T=63, 008 T=49,
011 T=45) where our model predicts the converged map against an onset-only GT (§6).

### 11.4a An autocatalytic lumen arm with physical-radius nucleation — tested, rejected

Proposed because the pack graph is disconnected: replace hop dilation with a Euclidean
nucleation ball and iterate, so growth is autocatalytic (a node ignites on its own local
exposure, then becomes a source). The committed **fraction** of the ball is the natural
brake. `src/core_physics/physics_lumen_model.py::autocatalytic_lumen`.

**The radius fix was right and matters.** Exposure — committed fraction of the ball —
ranks off-wall clot at:

```
ball radius (median edge lengths)   1.5     2.2     3.0
mean AUC over 19 vessels           0.573   0.986   0.988
```

r=1.5 is barely better than chance; r=2.2 is near-perfect ranking. My first sweep also
used the wrong threshold grid (0.25–0.55) when every vessel's oracle optimum is
**0.10–0.19** — so the first run never tested the live regime and looked like a hard
bifurcation. Re-swept properly:

| arm | train score | train offRel+ | sealed score | sealed offRel+ | ge2_recall |
|---|---|---|---|---|---|
| wall only | 0.7651 | — | 0.8285 | — | — |
| **graph dilation, 2 hops, spd<0.3** | **0.7613** | **0.5589** | **0.8419** | **0.5317** | **0.517** |
| autocat r=2.2 thr=0.10 steps=1 | 0.6906 | 0.5565 | 0.7395 | 0.5019 | 2.88 |
| autocat r=3.0 thr=0.25 steps=1 | 0.7551 | 0.3964 | 0.8017 | 0.4171 | 0.00 |

**It still loses on every axis**, and two facts say why:

* **The autocatalytic iteration is strictly harmful.** `steps=1` is optimal at every
  radius and threshold; steps 2/3/5 degrade monotonically (off-wall predictions per
  vessel go 71 → 152 → 237 → 432 while GT holds at ~40). Growth from a committed source
  has no brake that stops *at* the one-cell shell the target occupies. This is the same
  bifurcation as §3 on the wall, in a different coordinate.
* **Ranking is not the bottleneck; the level is.** With AUC 0.986 but a base rate of
  ~0.6% (patient007: 99 off-wall clot nodes among 16830 lumen nodes), the measured
  **oracle-threshold ceiling is strict F1 0.302** — and the oracle threshold itself moves
  per vessel (0.101 / 0.136 / 0.187 at the 10th/50th/90th percentile). This is §1.5b's
  level problem, and unlike on the wall (§6a) it is real here.

Kept in the module as a documented negative result, not wired into the predictor.
`scripts/sweep_lumen_autocat.py`, `scripts/diag_lumen_separability.py`.

**Worth someone's time:** exposure at r≈2.2 is an excellent *ranker*. A lumen arm that
spends a per-vessel budget on that ranking, rather than thresholding it, is the untried
option — but note the GT off-wall/wall count ratio spans 0.06–0.94 across vessels, so
the budget has the same transfer problem.

### 11.5 Honest limits

* The gain is **small and fragile**: −0.004 on train, +0.013 on sealed. Selecting on the
  full cohort would reject the arm entirely.
* The rule's ceiling is ~0.56 even with an oracle wall seed. Off-wall clot lives in a
  one-cell mesh shell, so any rule that fires on a whole neighbourhood over-paints:
  `ge2_recall` is 0.517 here against the reference's 0.735.
* `speed_nd < 0.3` does very little work (0.490 → 0.504 off-wall F1 across its whole
  range). The physics hypothesis "lumen clot fills stagnant pockets" is **not** what
  separates these nodes; proximity to committed wall does.
* **The canonical eval cannot currently score this at all.** `deploy_clot_phi_trajectory`
  multiplies pred and GT by `mask_wall` unconditionally whenever the pack has one, so
  every off-wall metric is structurally zero. All §11 numbers come from unmasked pred/GT
  with `wall_mask=` passed through, which is what the off-wall block in
  `compute_clot_relaxed_metrics` is written for. Restoring a real compound eval means
  gating that masking on the wall-only model — worth doing before anyone trusts a
  compound number, in either direction.

## 12. REPRODUCE

```bash
python scripts/parse_comsol_wall_export.py && python scripts/parse_comsol_domain_export.py
python scripts/step0_comsol_law_check.py      # law vs COMSOL, and the J0 vs d(Mat,t) gap
python scripts/step0_mls_validate.py          # operator audit, G_x vs MLS vs COMSOL
python scripts/step0_sanity_and_residual.py   # trivial-predictor baselines
python scripts/sweep_growth.py                # the two growth scalars, fit on TRAIN
python scripts/sweep_hops_both_arms.py        # MLS stencil per flow arm, fit on TRAIN
python scripts/report_phase3_results.py       # the table in section 0
python scripts/diag_offwall_structure.py      # what off-wall clot is
python scripts/diag_offwall_is_it_clot.py     # ... and that it is clot, not shear-thinning
python scripts/sweep_lumen_arm.py --full-horizon-only   # the lumen arm's two scalars
python scripts/compare_compound_orig10.py     # vs the learned compound stack
python -m pytest src/tests/test_mls_gradient_and_gates.py -q
```

---

## 13. TEMPORAL DYNAMICS — the growth CURVE, not just the final mask

Session 2026-08-09 (second pass). The mask model reproduces the final committed set at
0.79/0.91 but has **no time axis**, and the ODE behind it ignites in a flash. Any
time-resolved or longer-horizon claim needs the curve. Metrics live in
`src/core_physics/temporal_metrics.py`: `curve_l1` (L1 between the two cumulative
committed-fraction curves), `rho` (rank correlation of per-node onset time),
`spread_ratio` (model onset IQR / GT onset IQR; 1.0 = as gradual as GT). All are computed
on nodes both sides commit, so they measure timing and not the mask.

### 13.0 The baseline defect, measured

`scripts/diag_ignition_timing.py`:

```
patient043   MODEL onset [s] pct[0,25,50,75,100] = 3000 3000 3000 3000 3000   spread 0.000
             GT    onset [s]                     = 6900 9750 11250 11550 28650  spread 0.725
patient013   MODEL median 3000 s                   GT median 8925 s
```

patient043's gate is 100% low-shear, so `gate == 1` uniformly, every node has an
**identical** ODE, and they cross together. The flash is structural, not a tuning artefact.

### 13.1 `da_scale` was the bigger half of the fix, and the mask metric could not see it

§3 established that every `da_scale` above ~50 gives a bit-identical committed set. That
made the choice look free. It is not — holding the gate hard and changing nothing else:

```
da_scale 100 -> curve_l1 0.2998  spread_ratio 0.362  deploy score 0.7919
da_scale  40 -> curve_l1 0.1018  spread_ratio 0.911  deploy score 0.7919   <- identical score
```

**A 3x improvement in growth-curve fidelity at a bit-identical deploy score.** The
saturation that made §3's negative result clean also hid a real parameter. A quantity the
selection metric cannot resolve needs a second metric, not a shrug.

### 13.2 ARM 1 — graded gate

COMSOL's gate is provably a hard step, so grading it is a deliberate departure. The
justification: this model freezes the gate at t=0, and a node just below `lss` is the one
most likely to leave the gate as the flow evolves, while a node deep in a stagnation zone
is not. The right t=0 surrogate for the *time-averaged* gate is therefore a decreasing
function of the margin, not the indicator. `tau` is in units of the threshold itself, so it
is dimensionless and transfers across vessels.

Grading only the stagnation branch (`sigmoid_low`) beats grading both — the separation
branch already carries a magnitude through `|dsrx|`, so it was never the one that flashed.

```
                              curve_l1   rho    spread_ratio   train score
hard, da=100 (old default)     0.2998   0.713      0.362         0.7919
hard, da=40                    0.1018   0.713      0.911         0.7919
sigmoid_low tau=0.10, da=40    0.0799   0.608      1.178         0.7965
sigmoid_low tau=0.25, da=40    0.0649   0.651      1.481         0.7881
```

**curve_l1 0.2998 -> 0.0649, a 78% reduction, with the deploy score held.** Note `rho`
*falls*. Grading fixes **when the population ignites**, not **which node goes first** —
different failures, and only the second needs flow information.

### 13.3 ARM 2 AS SPECIFIED RESTS ON A FALSE PREMISE — measured, then corrected

The brief was: local `Mat` -> blockage -> shear rescale by a `1/r^3`-type resistance
relation. Implemented as `feedback="occlude"` in
`src/core_physics/shear_redistribution.py`. It moved `curve_l1` 0.2998 -> 0.2827. Two
measurements say why, and neither is a tuning problem.

**(a) The occluded fraction is zero.** `scripts/diag_blockage_magnitude.py`, occluded
fraction of the local cross-section at the end of the run:

```
              phi median   phi p90     shear amplification (median)   gates closed
model clot       0.000     0.02-0.24            1.00                   0-15 / ~550
GT clot          0.000     0.03-0.41            1.00                   --
```

`phi` is 0.000 **using the GT clot**, not merely the model's. The lumen is 14-27 mesh
cells deep and the clot is 1-3 cells thick. There is nothing to occlude.

**(b) The GT flow barely moves at all.** `scripts/diag_gt_shear_evolution.py` recomputes
`spf.sr` from the GT velocity at all 201 timesteps, 27 full-horizon vessels:

```
median sr(t_final)/sr(t=0)                          0.9974      <- a 0.26% change
p90    sr(t_final)/sr(t=0)                          1.0265
spearman(sr @t=0, sr @t_final) at the wall          0.838
fraction of wall nodes whose low-shear gate FLIPS   0.064
```

**The frozen-t=0-gate premise is essentially sound.** Arm 2 as specified was approximating
a 0.3% effect, and arm 3 would have spent a network on the same 0.3%.

**But the sign was backwards.** The gate open-fraction *rises* through the run
(patient032 0.000 -> 0.202, patient044 0.056 -> 0.256, patient007 0.153 -> 0.298).
Narrowing would accelerate the flow and *close* gates. Committed tissue is a no-slip
obstacle at 80x viscosity — it sheds a **stagnation wake**. So the feedback is

```
sr  ->  sr * (1 - wake * phi)      phi = committed fraction of a small neighbourhood
```

(`feedback="wake"`; one sparse matvec every 5 steps, no network). Fit on full-horizon
train: `wake=8.0`, neighbourhood radius `0.30 x local half-width`, `da=40`.

### 13.4 ARM 3 IS NOT EARNED — the oracle says so

Before paying for a learned corrector in the loop,
`scripts/diag_timevarying_gate_oracle.py` bounds it: integrate the same ODE but
re-evaluate the gates from the **GT velocity at the current timestep** — a flow model with
zero error, which no corrector can beat.

| variant | train score | train curve_l1 | sealed score |
|---|---|---|---|
| frozen hard t=0 gate | 0.7919 | 0.1018 | 0.8645 |
| **+ wake feedback (algebraic, no ML)** | **0.8513** | **0.0758** | **0.8957** |
| ORACLE: GT time-varying gates (illegal) | 0.8913 | 0.0670 | 0.9066 |

The algebraic wake rule recovers **60% of the oracle's train gain and 74% of its sealed
gain**, and arm 1 alone (`curve_l1` 0.0649) already **beats the oracle** on curve shape
(0.0670). What remains between the wake rule and a *perfect* flow model is 0.040 train and
0.011 sealed. **Arm 3 cannot exceed the oracle, so that gap is its ceiling — it should not
be built.**

### 13.5 Consolidated — and an honest split

`scripts/report_temporal_results.py`, 19 train / 8 sealed full-horizon vessels:

| variant | flow | train score | sealed score | curve_l1 (tr) | rho (tr) | spread |
|---|---|---|---|---|---|---|
| A gate + graph growth (shipped, §4) | GT | 0.8330 | **0.9093** | no time axis | | |
| A | pred | 0.7471 | **0.8567** | no time axis | | |
| B ODE + wake | GT | 0.8513 | 0.8957 | 0.0758 | 0.670 | 1.042 |
| B | pred | 0.7302 | 0.8056 | 0.0786 | 0.317 | 1.080 |
| C = B + graph growth | GT | **0.8655** | 0.9015 | 0.0857 | 0.685 | 1.260 |
| C | pred | 0.7445 | 0.8158 | 0.0930 | 0.393 | 1.359 |

**The temporal model does not replace the mask model.** C beats A on train (+0.033, GT
flow) but A still wins on **sealed** on both flow arms, and by 0.041 on the deployable one.
What B/C buy is a growth curve A cannot produce at all (`curve_l1` ~0.08, `spread_ratio`
~1.03). Use A for the final mask; use C when the question is time-resolved.

`rho` on the deployable arm collapses (0.670 -> 0.317 for B). Predicted-flow shear noise
costs far more in onset *ordering* than in the final mask — the mask needs only the gate's
sign, ordering needs its magnitude.

### 13.6 What this changes

* `da_scale` default should be **40**, not 100. Free, and 3x better on the curve.
* Arm 2 as briefed is closed — premise falsified twice by direct measurement — but the
  sign-corrected wake form is worth keeping.
* Arm 3 is closed on the oracle bound. 0.011 sealed headroom does not buy a network.
* Test suite 603 passing (+9 in `src/tests/test_temporal_wall_model.py`).
* The remaining temporal gap is **per-node ordering** (`rho` 0.670 against the oracle's
  0.795). Even perfect flow reaches only 0.795, so onset ordering is largely not a flow
  problem.

---

## 14. THE STALE DIAGNOSTICS, RE-RUN ON THE FIXED OPERATOR

§7 flagged that every flow-derived measurement in `WALL_MODEL_PLAN.md` was computed with a
non-differentiating operator. `BIOCHEM_GRAD_OPERATOR=legacy|mls` now switches it, so the
originals are reproducible and the correction is a one-line re-run. The `legacy` column
below reproduces the published numbers exactly, which is the check that the harness is
faithful.

### 14.1 §1.4 / §26.17 — gate support

`scripts/diag_physics_gate_support.py`, 35 vessels, wall + 3-hop band:

| | legacy (published) | **fixed** |
|---|---|---|
| % of committing nodes below `lss` | 60.9% | 32.3% |
| **% of NON-committing band nodes below `lss`** | **44.4%** | **7.4%** |
| mean AUC `-gamma_si` (low-shear) | 0.510 | 0.489 |
| mean AUC `neg_dgamma_dx` (separation) | 0.659 | **0.746** |

**§1.5b's headline supporting fact is gone.** "The gate is open on 45.6% of band nodes and
only ~7% ever commit — necessary-ish and wildly insufficient" was an artefact: the gate
fires on **7.4%** of non-committing nodes, not 44.4%. It was never open half the time; the
field it was thresholding was noise. The separation gate, whose input was identically zero,
gains the most (0.659 -> 0.746), and the low-shear gate remains a poor *ranker* (0.489) —
consistent with §4.1, where neither gate alone clears 0.6 and the union does.

### 14.2 §1.5a — the t=0 ceiling

`scripts/diag_t0_ceiling.py`, 35 vessels, oracle feature *and* oracle threshold:

| | legacy (published) | **fixed** |
|---|---|---|
| mean best-single-feature AUC | 0.884 | **0.921** |
| mean ORACLE-THRESHOLD F1 | 0.462 | **0.583** |
| vessels with oracle F1 >= 0.6 | 5 / 35 | **15 / 35** |
| vessels with oracle F1 >= 0.5 | 12 / 35 | **20 / 35** |

§1.5a's projection was `0.463 + 0.092 (rollout gain) = 0.554 F1 -> ~0.597 score` against a
0.600 target, read as "it lands on the line — marginal, not comfortable". Redone on the
fixed operator the same arithmetic gives `0.583 + 0.092 = 0.675 -> ~0.715`, and the
realised result is **0.787**. The margin was never thin; the instrument was.

### 14.3 Still stale

§10.4's regime routing and Z1's 0.041 AUC for the flow channel have not been re-run. Both
are downstream of the same operator and both should be expected to move.
