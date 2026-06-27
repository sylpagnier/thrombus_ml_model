# Strategy re-evaluation: how to capture the coagulation cascade / species learning

**Date:** 2026-06-18 · **Status:** decision doc (supersedes the "full 9-species ADR GNN" framing)

This re-evaluation is grounded in the now-complete COMSOL validation
([docs/COMSOL_PHYSICS_VALIDATION.md](COMSOL_PHYSICS_VALIDATION.md)), which
reconstructed every phase-2 reaction law against the patient007 exports to machine
precision. The validation changes what is worth learning.

---

## 1. What the validation tells us (the binding constraints)

1. **The clot is platelet-matrix (`Mat`) driven, not fibrin driven.** Fibrin `FI`
   peaks at ~0.013 µM on patient007 — ~46× below the 0.6 µM `viscosity_fi_crit`
   gelation threshold. `mu2(FI) ≡ 0` everywhere, always. The viscosity jump that
   *defines* the GT clot (`µ_eff` growth) comes entirely from `mu1(Mat)`.
2. **`Mat` grows by autocatalytic aggregation, not by recruitment from the bulk.**
   The exported deposition flux `J0_Mat` (fresh platelets sticking to wall) accounts
   for only ~1/133 of the true `d(Mat,t)`. ~90% of growth is the autocatalytic term
   `(Mas/Minf)·k_aa·AP` — platelets sticking to already-deposited activated platelets,
   amplified by `Mat`-driven thrombin. Effective rate ≈ 143× the `J0` deposition rate.
3. **Deposition is gated by low-shear stagnation.** The `sr < lss` (stagnation) gate
   carries ~80% of deposition; the `dsrx < sgt` shear-gradient (separation) gate is a
   minor contributor. The separation gate is also the one the graph WLS operators
   resolve worst (R1 failure cause).
4. **Units are now nailed down.** `Da = 1e-4` reconstructs `J0_Mat` exactly in either
   CGS (COMSOL) or SI (kernel). This is pinned by
   [src/tests/test_comsol_wall_deposition_calibration.py](../src/tests/test_comsol_wall_deposition_calibration.py)
   against [src/core_physics/comsol_surface_deposition.py](../src/core_physics/comsol_surface_deposition.py).

**Implication:** most of the 9-species ADR cascade is *irrelevant to the clot output*.
Only the **activation pathway → AP** and the **Mat autocatalysis** matter. Fibrinogen/
fibrin, and the fine detail of APR/APS/AT/PT transport, mostly do not move `µ_eff`.

---

## 2. Options considered

| Option | What it learns | Verdict |
|---|---|---|
| **A. Full mechanistic ADR GNN** (current `biochem_wall_residual` + 9-species transport, all backprop) | Every species field + surface ODE | **Reject as the primary path.** Stiff (143× autocatalytic gain), needs accurate graph `dsrx` (WLS under-resolves it → R1 failed), and spends ~all capacity on species that don't change the clot. |
| **B. Gray-box reduced-order (recommended)** | AP/activation field + `Mat` autocatalytic closure, with the validated CGS law as a differentiable physics prior; fibrin dropped | **Adopt.** Focuses learning where the physics actually lives; uses the exact law as prior/feature instead of a hard PINN constraint. |
| **C. Pure data-driven clot-φ** (current `predict_phi_prior_rule` / `ClotPhiMLP` on kinematic + low-shear features) | `Mat`/φ directly from geometry+flow, no species | **Keep as baseline/fallback.** Already ~0.70 F1 on p007. Strong, but can't generalize the *chemistry* (dose, agonists) since it has no species. |

---

## 3. Recommended strategy — gray-box, Mat-centric

Learn the two things that are both **necessary** and **hard**, and hard-wire the rest:

1. **Activated-platelet field `AP` (the cascade part that matters).** Predict/transport
   `AP` (and its precursor `RP`) along the flow. This is where the coagulation cascade
   genuinely enters: `AP` is set by `k_pa(ω, shear)` with `ω` from APR/APS/T. Keep the
   validated kinetics (`BiochemKinetics`) as a prior; let the model correct the AP field.
   Fibrinogen/fibrin transport can be dropped from the trigger path entirely.
2. **`Mat` autocatalytic closure (the growth part that matters).** Model
   `d(Mat,t) ≈ Da·[sr<lss]·(L/γ_m·|dsrx|·dep + dep) + autocat`, with
   `autocat = k_aa_eff·(Mas/Minf)·AP` and `k_aa_eff` a learned boost over the bare
   `Da·k_aa` (validation: ~143×). Primary gate = **low-shear stagnation**; separation
   gate optional/auxiliary (don't rely on graph `dsrx`). Nucleation = wall + 1-hop
   (already in `neighbor_supervision_mask`).
3. **Clot trigger = `mu1(Mat)` only.** Fibrin leg removed from the deploy trigger
   (`CLOT_PHI_PHYSICS_USE_FIBRIN=0` default; see §4). `µ_eff = µ_carreau · mu1(Mat)`.

Why this is the right shape: it keeps the **interpretable, validated physics** (exact
deposition law, exact gelation step, exact kinetics) as the scaffold, and spends the
learned capacity on (a) the AP/activation field and (b) the one closure term
(autocatalytic boost + nucleation) that the bare law under-predicts. It sidesteps the
two things that broke the mechanistic path: fibrin (irrelevant) and graph-`dsrx`
(unresolvable).

### Proposed ladder (fresh, small steps first)
- **S0 (law-sufficiency gate) — PASSED.** Forward-integrate `Mat` from the validated
  law (deposition + autocatalytic closure) using oracle `ap`/`Mas`/flow on the
  patient007 wall; gelate with `mu1(Mat)` only. Script:
  [scripts/s0_mat_law_sufficiency.py](../scripts/s0_mat_law_sufficiency.py) →
  `outputs/reports/comsol_validation/s0_mat_law_sufficiency.json`.
  - **bare law (no closure): gelation F1 = 0.00** (deposition alone is ~133× too weak — never gels).
  - **with autocatalytic closure (`k_aa_eff`): gelation F1 = 0.97 final frame** (P=1.00, R=0.94),
    final-`Mat` corr 0.96. *The single autocatalytic closure coefficient is the essential
    learnable piece — exactly the strategy thesis.*
  - Caveat: oracle `ap`/`Mas` (no GT clot labels as input). Deployability now hinges on
    producing `ap`/`Mas` from IC + cascade + kine flow (S1–S3), not on the trigger.
- **S1 (closure generalization) — PARTIAL PASS.** Fit `k_aa_eff` per patient on the 10
  anchor graphs (oracle `ap`/`Mas`, GT flow), forward-integrate `Mat`, score gelation F1
  vs the canonical GT clot. Script:
  [scripts/s1_kaa_closure_generalization.py](../scripts/s1_kaa_closure_generalization.py)
  → `outputs/reports/comsol_validation/s1_kaa_generalization.json`.
  - **`k_aa_eff` is reasonably stable: mean 3.7e-6, CV 0.42** (all within ~2× of mean,
    range 1.4e-6–7e-6). A single closure scalar is plausible. ✓
  - **BUT law-integrated F1 with graph-derived gates ≈ 0.44 own-fit / 0.53 p007-transfer**,
    far below the **0.97** S0 achieved with COMSOL-exact gates. Root cause = the on-graph
    shear / `dsrx` (WLS) under-resolves the gates (the R1 finding); the *clot footprint*,
    not the closure magnitude, is the bottleneck. (GT-`Mat` gelation ceiling on-graph ≈ 0.95.)
  - Net: ~0.53 transfer ≈ the deployable GNN baseline (0.48), with full interpretability,
    but does not beat it yet.
- **S1-diagnosis (shear: resolution, not calibration) — DONE.** The graph nodes are an
  exact subset of the COMSOL p007 mesh, so `spf.sr` maps onto them with zero error.
  Decomposing the wall-recall gap (`scripts/_diag_shear_resolution.py`):
  - On wall nodes graph-WLS shear is **rank-decorrelated** from exact `spf.sr`
    (Pearson(log) ~0.1, Spearman ~0.25, gate agreement 60%, WLS over-fires 59% vs true 31%).
  - **Calibration is not the lever** — a global rescale (`a=0.535`) *lowers* wall recall
    (−0.058). Swapping to exact `spf.sr` lifts wall recall **0.62→0.80**, wall F1
    0.76→0.885, full F1 0.51→0.65. Root cause: wall shear is the no-slip wall-normal
    gradient `∂u_t/∂n`, which a 1-ring WLS on a coarse linear graph cannot resolve.
  - Deployable proxy ranking vs the exact gate (`scripts/_diag_shear_proxies.py`):
    **`mu_prior` (analytic Carreau wall shear) AUC 0.82 / |ρ| 0.84** ≫ WLS 0.63; velocity
    proxies (`wss_prior`, `speed`, `shear_potential`) ≈ 0 at no-slip nodes (useless).
- **S1b (gate-source bake-off) — DONE; learned gate wins.**
  [scripts/s1b_gate_variants.py](../scripts/s1b_gate_variants.py) →
  `outputs/reports/comsol_validation/s1b_gate_variants.json`. Shared scoring = wall-surface
  law (oracle Mas/ap) → nucleate at wall → dilate 1 hop into the lumen band → F1 vs GT clot.
  Mean F1 over 10 anchors:
  - **(c) learned logistic gate (LOAO) on [carreau, wallfunc, wls, sdf, width, mu_prior]:
    0.590 — best**, rescues the WLS failure cases (p001 0.53→0.83, p005 0.57→0.72,
    p007 0.69→0.82, p011 0.00→0.21).
  - (b) wall-function near-wall shear `|u_nearwall|/d_wall`: **0.519** ≈ baseline.
  - baseline WLS: **0.505**.
  - (a) static analytic Carreau gate (inverted `mu_prior`): **0.404 — hurts** (fragile,
    fires too narrowly for the integrated law).
  - Why (b) ≈ baseline despite the operator finding: with **oracle Mas** the gate's
    precision is free (Mas masks off-target deposits) and the **band-dilation** recovers
    recall, so on well-behaved patients the gate source barely moves F1 (p007 ceiling 0.683
    ≈ baseline 0.693). The differentiator is the failure patients, where the learned gate's
    geometry features generalize. **(b)'s precision advantage should matter once Mas is no
    longer oracle (S2).**
  - Caveats: (c) is trained on GT-clot-wall labels (deployable features only, LOAO) so it is
    a supervised footprint head, not a pure shear gate; small N (10) → overfit risk.
- **S2** — replace oracle `ap`/`Mas` with the cascade forward-solve from inlet/IC + kine
  flow (deployable). This is where the real deploy number is earned vs the GNN baseline.
- **S3** — closed loop: predicted flow → AP → Mat autocat → `mu1` → `µ_eff`; eval with
  `scripts/eval_biochem_gnn_deploy_ab.py` (frozen-kine) against the locked baseline.

---

## 4. Code changes landed alongside this doc

- **Canonical unit-consistent law:** `src/core_physics/comsol_surface_deposition.py`
  (`DepositionConstants.cgs()/.si()`, `j0_mat_*`, `j0_thrombin_cgs`,
  `recover_damkohler_cgs`) — single source of truth for the deposition/thrombin laws.
- **Regression test (Da=1e-4 vs exports):**
  `src/tests/test_comsol_wall_deposition_calibration.py` + fixture
  `src/tests/fixtures/comsol_wall_deposition_patient007.csv` (649 wall points). Asserts
  recovered `Da == surface_damkohler` (CV < 1e-6), `J0_Mat`/`J0_th` reconstruction, and
  SI-law ≡ CGS-law (proving the kernel's SI convention is unit-consistent).
- **`biochem_wall_residual` unit-consistency:** now sources its adhesion constants from
  `DepositionConstants.si(cfg)` (numerically identical SI values → no training
  regression) with an explicit unit-system docstring; the law is guarded by the test.
- **Deploy clot trigger rescoped:** fibrin leg dropped by default in
  `_resolve_gelation_legs` (`clot_phi_physics_use_fibrin()`,
  `CLOT_PHI_PHYSICS_USE_FIBRIN=1` to re-enable for ablation). Trigger is now Mat-driven;
  `mu2(FI)=0`.

---

## 5. Pre-S2 diagnostics — why F1 ~0.59, and is it the model ceiling? (NOT missing physics)

Ran a diagnostic battery (`scripts/_diag_ceiling.py`, `_diag_p008.py`,
`_diag_gate_coverage.py`, `_diag_nogate.py`, `_diag_p011_fit.py`). Findings:

- **Label ceiling, not model ceiling.** Perfect `Mat` reconstruction → mean F1 **0.87**
  (label precision ~1.0 on 9/10). Our 0.59 is approximation debt (blunt 1-hop dilation +
  coarse gate), not a physics wall. Precision headroom is large.
- **Anchor sims have very different maturity.** p007 runs to 30000 s (201 frames) but
  **p008 stops at 7126 s (49 frames)** and **p011 at 6486 s (45 frames)**. "Final frame"
  is a different clot age per patient.
- **p008 is degenerate, not missing physics.** Nascent 24-node wall rim; GT `Mat` median
  **1.88e7 < crit (2e7)** — sub-critical. The growth-`mu_eff` clot label (24) and
  Mat-gelation label (6) disagree at nucleation. Label ceiling 0.40.
- **The hard gelation threshold is a cliff.** p011 reconstructs the right shape (corr 0.62)
  but undershoots magnitude ~25%; since GT `Mat` there is **2.06e7 ≈ crit**, that flips
  134→0 gelled. A 1.3× scale recovers 105/134. Brittle metric near marginal clots.
- **Under oracle `Mas` the shear gate is net-harmful.** No-gate (gate=1) mean F1 **0.577** >
  WLS-gated **0.504**; `Mas`/`ap` are present on 100% of clot nodes, so `Mas` already
  localizes the clot and the gate mostly false-vetoes (p011: carreau gate 0% on, claims
  213 s⁻¹ on a stagnant clot). **=> the gate can only be evaluated at S2, not S1/S1b.**
- **The deposition+autocat law form is sound** (shape corr ~0.6 even on the "failing"
  patients; global least-squares underweights the few clot nodes → undershoot).

**Pre-S2 actions implied:** (1) score on a **soft/continuous** clot signal (`mu_eff` or
`Mat`/crit ratio), not hard `Mat>=crit`, to kill the threshold cliff; (2) **clot-weighted**
or per-node closure fit instead of global lstsq; (3) **handle sim maturity** (matched
physical time, or clot-size-weighted aggregate, or drop immature p008); (4) reconcile the
two clot labels (Mat-gelation vs growth-`mu_eff`); (5) defer the gate to S2.

### S1c (pre-S2 fixes #1+#2 + cohorts) — DONE; law-form validated
[scripts/s1c_soft_eval.py](../scripts/s1c_soft_eval.py) →
`outputs/reports/comsol_validation/s1c_soft_eval.json`. No-gate law, Mas-weighted fit,
soft/threshold-swept scoring, complete vs early cohorts.
- **Sim maturity:** 5 **complete** (201-frame / 30000 s: p001/p005/p006/p007/p010) vs 5
  **early-terminated** (p002/p003/p004/p008/p011, COMSOL stopped at ~4-10 ks, likely solver
  divergence as the clot stiffens). **Decision: keep all for training** (early sims hold the
  nucleation/early-growth signal needed for deployment across clot ages); **exclude early
  from the headline aggregate only.**
- **Headline (complete) swept-F1 = 0.863 ≈ label ceiling 0.87** → the deposition+autocat
  law reconstructs the mature-clot footprint as well as perfect `Mat` would. **No missing
  physics for mature clots.** Deployable fixed-`crit` hard-F1 = **0.817** (vs old all-cohort
  no-gate 0.577).
- **#1 (soft/swept) is the big lever**: p011 hard-F1 0.000 -> swept-F1 **0.975** (footprint
  was right; hard `crit` cliff + ~25% undershoot hid it). **#2 (Mas-weighted fit) marginal**
  (kept; not the hero).
- Per-patient best `thr/crit` spans 0.33-1.49 = a **magnitude-calibration** gap (spatial
  footprint is right, absolute `Mat` magnitude per patient is not). swept-F1 uses a
  GT-picked threshold (footprint *ceiling*, not deployable); closing the hard->swept gap is
  an S2/S3 magnitude-calibration job. p008 stays weak (soft-Dice 0.22) = genuinely immature.

## 6. S2 — deployable closed-loop forward solve (DONE; gate is the bottleneck)

Scripts: [scripts/s2_deploy_cascade.py](../scripts/s2_deploy_cascade.py) (cascade probe),
[scripts/_diag_bulk_flat.py](../scripts/_diag_bulk_flat.py),
[scripts/_diag_s2_mag.py](../scripts/_diag_s2_mag.py),
[scripts/s2_deploy_forward.py](../scripts/s2_deploy_forward.py) →
`outputs/reports/comsol_validation/s2_deploy_forward.json`.

**Scope collapse (key finding — no bulk ADR needed).** A from-scratch ADR march is CFL-
unstable (advective time `d_bar/u_ref ~0.16 s` vs frame `dt 150 s`; COMSOL/the DEQ solve
implicitly). But it turns out we don't need it: `_diag_bulk_flat.py` shows GT **bulk
platelets barely move from the resting IC** — `RP` is exactly constant (final/rest = 1.000),
`AP` stays **0.2-1.6x** resting (mildly *depleted* on clot, never blows up). Thrombin/agonists
grow but only feed activation (`k_pa`), which barely fires locally (`omega~0`, `k_pa=0` even at
shear 100); and the deposition law uses only `rp/ap/Mas`. A per-node reactor (cascade reactions
+ residence-time washout, GT flow) gives `ap` of the **right magnitude but ~0 spatial
correlation** (corr -0.1..0.3) — advective transport is what localizes a real cascade, and we
can't (and needn't) reproduce it. **=> freeze `rp=c_RP0`, `ap=ap_rest` from the IC; integrate
only the surface system.**

**Deployable forward = parameter-free surface ODE** (`biochem_wall_residual`, Da=1e-4 + cfg
rates), the FIRST test with **`Mas` produced** (closed-loop autocat), not oracle:
```
dMas/dt = dM/dt = Da*step2t*avail*g_low*(k_rs*rp + k_as*ap)
dMat/dt = dMas/dt + Da*step2t*g_low*(Mas/Minf)*k_aa*ap,   avail = 1-(M+Mas+Mat)/Minf
```
Inputs: IC + geometry + flow-shear gate. No GT except to score.

**Results (complete cohort, swept-F1 | hard-F1 | soft-Dice):**
| gate (deployable?) | swept | hard | dice |
|---|---|---|---|
| wls (GT-flow, ref) | 0.339 | 0.329 | 0.314 |
| wallfunc (flow)    | 0.338 | 0.336 | 0.314 |
| learned (LOAO, hard>0.5) | 0.401 | 0.394 | 0.363 |
| **carreau (geometry only)** | 0.451 | 0.436 | 0.407 |
| **combo = carreau ∪ learned** | **0.474** | **0.457** | **0.435** |

vs **S1c oracle-`Mas` headline 0.863** and species-GNN deploy p007 ~0.70.

**Debug conclusions:**
- **The low-shear gate is the deployable bottleneck.** Under oracle `Mas` (S1/S1c) the `Mas`
  field localized the clot, so the gate was redundant/harmful. With **produced** `Mas`, the gate
  *is* the localizer and no current source is clean: `carreau` fires on only 0.2-0.74 of clot
  nodes (recall-capped) and **collapses to 0 on p001/p011** (analytic shear never < `lss=25`);
  `wls`/`wallfunc` fire on ~0.5-0.9 of the wall (broad, decorrelated). `combo` (union) rescues
  the collapses and wins.
- **Magnitude is uncalibrated → footprint = gate footprint.** `crit/Minf = 3e-4`, so with frozen
  `ap` (no `J_in_AP` depletion sink) and raw cfg rates the autocat runs away (`Mat~2.25e10`,
  ~450x GT) → `Mat` is effectively binary wherever the gate fires; swept-F1 ≈ hard-F1 (unlike
  S1c). So F1 here measures **gate footprint quality**, not magnitude. Soft gates over-gel under
  produced `Mas` (S1b's soft learned winner needed hard-thresholding here).

### 6.1 Step-back diagnostics — where is the gap, really? (flow is solved; the GATE FORM is the lever)
Scripts: [scripts/_diag_coupling.py](../scripts/_diag_coupling.py),
[scripts/_diag_relgate.py](../scripts/_diag_relgate.py),
[scripts/s2_kine_flow_test.py](../scripts/s2_kine_flow_test.py).

- **Clot location is set by the INITIAL flow, not by coupling.** Among wall nodes, clot-wall
  shear is ~**10x lower** than non-clot-wall shear **at t0, before any clot** (p007 2.2 vs 26.3,
  p006 2.3 vs 37.6, p010 0.0 vs 32.5). The two-way coupling (clot raises `mu_eff` **37-300x** →
  diverts flow) only *sharpens* the footprint (`low%clot` 0.70→0.73), it doesn't create it. So a
  **one-way deployable pipeline (flow → gate → law) is physically justified**; the `mu_eff`
  feedback loop is a later refinement, not the missing piece. (This corrects the earlier
  "coupling is the bottleneck" guess.)
- **The deployable flow is essentially perfect.** The kine RGP-DEQ reproduces COMSOL's t0 field
  to **velCorr 0.996 / relative-L2 ~4%**, and the gate F1 from kine flow equals the gate F1 from
  GT flow (wls 0.307 vs 0.304; wallfunc 0.305 vs 0.304). **=> flow source / flow resolution is
  NOT the bottleneck.**
### 6.2 Ceiling decomposition + where ML belongs (hybrid gray-box)
Script: [scripts/_diag_ml_direction.py](../scripts/_diag_ml_direction.py),
[scripts/_diag_exact_gate.py](../scripts/_diag_exact_gate.py) (p007, closed-loop, parameter-free).

The COMSOL gate is *known exactly* (validated): low-shear `[spf.sr<lss]` + separation
`(L/gamma_m)|d spf.sr/ds|*[d spf.sr/ds<sgt]`. **Ceiling decomposition (p007 swept-F1):**

| layer | F1 | note |
|---|---|---|
| label (perfect `Mat`, **wall+1-hop recon**) | 0.85 | recon artifact, NOT mesh — see 6.4 |
| label (perfect `Mat`, **unrestricted recon**) | **0.99** | true graph ceiling (mesh is fine) |
| **coupled-flow exact gate** (time-varying `spf.sr`) | **0.77** | realistic physics ceiling — needs flow↔clot coupling |
| initial-flow exact gate (t0) | 0.48 | deployable, best shear |
| initial-flow kine gate (t0) | 0.50 | deployable now |

**The dominant lever is the flow↔clot COUPLING (+0.29), not the shear operator (-0.02 at t0).**
As the clot grows it reroutes flow and *sharpens* the stagnation (improves precision); at the
*initial* flow even exact `spf.sr` gives only ~0.50. (This corrects 6.1's "shear operator is the
lever" — the operator only matters once you have the evolved flow to read.) The time-varying exact
gate adds the separation branch for **0.80** ([_diag_exact_gate.py](../scripts/_diag_exact_gate.py)).

**ML recovers most of the coupling from DEPLOYABLE inputs.** Predicting the *coupled* (final)
stagnation gate from *initial* geometry + kine-flow features (p007, spatial held-out):
- RandomForest gate: test-AUC **0.86**, closed-loop F1 **0.665** (features: `mu_prior` 0.44,
  `speed_ring`/`wallfunc`/`width` ~0.16, WLS 0.08).
- Regress coupled `spf.sr`: test-R2 **0.61** -> physics gate F1 **0.655**.

i.e. an ML corrector lifts the deployable 0.50 toward the 0.77 coupled ceiling — **this is exactly
where ML earns its keep** (it learns the geometry→coupled-stagnation map = the flow↔clot feedback,
which is what must generalize across geometries). Physics supplies the gate/deposition/gelation
backbone + the ceiling; ML supplies the coupling surrogate + generalization.

### 6.3 Coupling loop wired + tested — couple via GEOMETRY, not viscosity; then it's operator-limited
Scripts: [scripts/s3_coupled_loop.py](../scripts/s3_coupled_loop.py),
[scripts/_diag_couple_probe.py](../scripts/_diag_couple_probe.py),
[scripts/_diag_kine_mu_response.py](../scripts/_diag_kine_mu_response.py),
[scripts/_diag_occlusion_probe.py](../scripts/_diag_occlusion_probe.py).

We wired the faithful sequential coupling loop into the S-ladder (gate from live flow ->
deposition -> re-solve kine -> repeat) and tested two ways to feed the clot back to the flow:

1. **Viscosity injection (`mu1(Mat)` -> `MU_PRIOR`) = DEAD.** The kine RGP-DEQ does not respond
   physically to clot-scale `mu`: with the *oracle* plug gelled, **in-clot speed goes UP
   (0.059->0.086)**; uniform `mu`x100 raises *mean* speed 0.93->1.49. Trained on Carreau
   `MU_PRIOR` <=~16 nd, a plug needs ~29 nd -> out-of-distribution, wrong sign. (Answers "re-guess
   with clot viscosity?" -> no.)
2. **Geometry occlusion (clot -> wall) = PHYSICALLY CORRECT.** Re-express the clot as a solid:
   `SDF = dist(node, wall U clot)`, shrink hydraulic `WIDTH`, rescale the velocity prior ~1/R, zero
   it inside the clot. Now the kine model behaves: **in-clot speed -> ~0** (0.0595 -> 1e-6) and the
   one-shot *oracle* occlusion sharpens the gate **0.47 -> 0.53**. The model respects geometry (it
   was trained on many channel widths) even though it ignores `mu`. (Answers "make the clotted node
   a wall?" -> yes, that's the right mechanism; this is now the coupling method in `s3_coupled_loop.py`.)

**But the deployable progressive loop still flatlines (0.500 -> 0.494).** Two reasons, and both say
the binding constraint has moved off the *flow*:
- **Operator cap.** Even the one-shot *oracle* occlusion only reaches **0.53** read with WLS/wallfunc,
  vs **0.71** read with exact `spf.sr` on the *same* flow. The deployable shear operator can't read
  the sharpened stagnation. (matches 6.2: WLS time-varying 0.56 vs exact 0.77.)
- **Footprint bootstrap.** The progressive clot grows from the broad *initial* gate, so occluding the
  produced (too-broad) footprint reinforces breadth instead of correcting it; only the oracle clot
  (which we don't have at deploy) sharpens.

=> **Net: the flow side is now solvable deployably (geometry occlusion makes kine divert); the
remaining 0.53 -> 0.77 gap is the wall-shear READOUT.** That is exactly the ML corrector trained on
COMSOL's coupled `spf.sr` (export). Keep the geometry-occlusion coupling; it is the deployable
flow-update, but it only pays off once the gate can read accurate shear.

**=> S3 = hybrid gray-box, in priority order:**
1. **ML coupled-gate/shear corrector (the +0.29 lever).** Learn (geometry + initial kine flow) ->
   coupled `spf.sr` / low-shear membership, feeding the EXACT physics gate + closed-loop law.
   Train LOAO across anchors with COMSOL-evolved `spf.sr` supervision. (Within-patient signal is
   strong: AUC 0.86-0.91; cross-geometry needs the exports below.)
2. **Data need (now REQUIRED, not optional):** export **time-varying `spf.sr`** for all anchors
   (Track A) to supervise + LOAO-validate the ML corrector. The physics coupling loop is shelved
   (6.3: kine can't divert), so COMSOL's coupled shear is the only source of the +0.29 signal.
3. **Magnitude calibration.** Freeze S1 closure + `ap`-depletion (`J_in_AP` sink) so autocat
   self-limits (today `Mat` ~450x hot -> binary).
4. **(Optional, later) kine retrain for clot-scale `mu`** so a deployable physics coupling loop
   becomes viable — only if the ML corrector route stalls.

**Net model shape (hybrid):** physics backbone (validated gate + deposition + autocat +
`mu1/mu2` gelation) + already-trained kine flow + a learned coupling/shear corrector. Species are
passive (frozen at IC); ML is concentrated exactly on the coupling — the part that is hard,
geometry-dependent, and worth ~0.29 F1.

### 6.4 The 0.85 "label ceiling" is the wall+1-hop reconstruction, NOT mesh resolution
Script: [scripts/_diag_label_ceiling.py](../scripts/_diag_label_ceiling.py) (complete anchors,
perfect GT `Mat`).

The graph (p007: 17,413 nodes / 583 wall) is coarser than the COMSOL export mesh (51,240), but the
clot is **fully representable on it**: an *unrestricted* footprint (`Mat/crit>=thr` anywhere) from
perfect `Mat` scores **0.990 mean** (p010 1.000). The 0.85 only appears under the deployable
**wall-seed + 1-hop dilation** recon (`wall+1`=0.877; `wall+2`=0.799; `wall+3`=0.710 — more hops
flood false positives). Cause: only **69-80%** of the clot label is wall and **78-86%** is within
1 hop; **~15-20% of the clot extends >1 ring into the lumen** and the wall+1 seed can't reach it.

=> The 1-hop dilation is a crude stand-in for the clot **growing into the lumen**. Fix it two ways
(both on-path): (a) the **coupling loop** already grows volumetrically — each occlusion step narrows
the lumen and recruits the adjacent ring + autocat spread; (b) a **growth/nucleation footprint**
model ([docs/CLOT_ML_LADDER_V2.md](CLOT_ML_LADDER_V2.md)) instead of static dilation. Either lifts
the label ceiling 0.85 -> ~0.99; **mesh refinement is unnecessary.** So the two "ceilings" collapse
to one lever: model lumen-ward growth (coupling), then read accurate shear (corrector).

### 6.5 Consolidated lessons (2026-06-18)
Crisp symptom -> cause -> fix distilled from 6.1-6.4 (detail + scripts in those sections).

1. **ML stays in — but only where it generalizes.** Species are passive (bulk `ap`/`rp` flat at IC,
   proven); don't learn the full cascade. Concentrate ML on the **wall-shear / coupled-gate
   corrector** (a *local* operator: velocity-gradient -> `spf.sr`) — it's well-posed and transfers
   across geometries. Physics supplies the gate/deposition/`mu1(Mat)` backbone + the ceiling.
2. **The dominant lever is the flow time-evolution (coupling), +0.29; the t0 shear operator is ~0.**
   At the *initial* flow even exact `spf.sr` caps ~0.50; the clot reroutes flow as it grows, which
   *sharpens* the stagnation (precision). Earlier "shear operator is the lever" was wrong — it only
   matters once you have the evolved flow to read.
3. **Couple via GEOMETRY, not viscosity.** Injecting clot-scale `mu1(Mat)` into the kine model is a
   dead end: it's out-of-distribution (`MU_PRIOR` trained <=16 nd, plug needs ~29) and responds with
   the *wrong sign* (in-clot speed goes UP). Re-expressing the clot as a **wall/occlusion**
   (`SDF=dist(node, wall U clot)`, shrink `WIDTH`, rescale `UV_PRIOR` ~1/R, zero inside) makes the
   kine model divert physically (in-clot speed -> 0). The model respects geometry, ignores `mu`.
4. **With correct coupling, the deployable gate is OPERATOR-limited.** One-shot oracle occlusion read
   with WLS/wallfunc -> 0.53; read with exact `spf.sr` on the *same* flow -> 0.71. => the binding
   constraint is the wall-shear READOUT, which is why we export COMSOL `spf.sr` to supervise the ML
   corrector.
5. **0.77 is the trigger ceiling, not deposition outcome.** The gate is *necessary* (low shear) but
   not *sufficient*: it over-predicts (low-shear nodes that lack residence time / `ap` never gel) and
   under-predicts (autocat + moving boundary recruit non-gated neighbors). Closing 0.77 -> 0.85 needs
   an ML deposition-residual on top of the gate, not better shear.
6. **The 0.85 "label ceiling" is the wall+1-hop reconstruction, NOT mesh resolution.** Perfect `Mat`
   scores 0.99 unrestricted on the same 17k-node graph; wall+1-hop caps it because ~15-20% of the clot
   grows >1 ring into the lumen (and extra dilation floods false positives). Fix = model **lumen-ward
   growth** (the coupling loop already does this; or a growth/nucleation footprint). Mesh refinement
   is unnecessary.
7. **Net: two levers, not many.** (a) grow the clot volumetrically (coupling) -> ~0.99 representable;
   (b) read accurate wall shear (ML corrector on exported `spf.sr`) -> unlock the 0.77 trigger. The
   residual gap to 1.0 is boundary threshold-ambiguity (irreducible).

### 6.6 spf.sr exported for all anchors + corrector wired/tested — the gap is the FLOW, not the readout (2026-06-18)
Scripts: [scripts/preprocess_spfsr.py](../scripts/preprocess_spfsr.py) (->
`data/processed/spfsr_cache/`), [scripts/spfsr_lib.py](../scripts/spfsr_lib.py),
[scripts/s3_exact_gate_all.py](../scripts/s3_exact_gate_all.py),
[scripts/s3_shear_corrector.py](../scripts/s3_shear_corrector.py),
[scripts/s3_corrector_loop.py](../scripts/s3_corrector_loop.py),
[scripts/s3_geom_shear_probe.py](../scripts/s3_geom_shear_probe.py).

COMSOL `spf.sr` (+ `d(spf.sr,x/y)`) now exported time-varying (201 frames, t=0..30000) for all 10
anchors and cached onto graph nodes (median NN ~1e-7 nd = identical mesh). Four tests, decisive:

1. **Exact-gate ceiling generalizes (good).** Time-varying exact `spf.sr` gate -> closed loop,
   **complete-cohort (201-frame) mean F1 0.75** (p010 0.93, p006 0.87, p007 0.75; label ceiling
   0.88; frozen-t0 0.45). p007's 0.77 was representative, not a fluke. Coupling lever = **+0.30**.
2. **Static t0 -> coupled-gate map does NOT generalize.** LOAO RF from deployable t0 features
   (kine shear + geometry) to the *coupled final* stagnation recovers only **17%** of the lever
   and collapses on the most out-of-distribution geometry (p001 AUC 0.52, F1 0). The t0 readout
   itself is ~trivial (`mu_prior` is analytic Carreau ~ a function of shear) — irrelevant, because
   the lever is the *time-evolution*, not the t0 read.
3. **ML readout on the occluded kine flow = wallfunc (the correction WORKED, the flow didn't).**
   Wired the per-frame corrector inside the geometry-occlusion loop. Even with the **oracle GT
   clot** occluding the flow: ML-readout F1 **0.379 == wallfunc 0.380**, both vs exact **0.752**.
   Deployable progressive loop 0.328. => the learned readout adds *nothing* over analytic wallfunc.
4. **Why: the kine-occluded flow doesn't carry COMSOL's coupled shear.** Per-node low-shear
   membership AUC vs exact is only **~0.65-0.74 for every deployable feature** (wallfunc-on-kine
   0.737, geom-local 0.651, multiscale-geom+position 0.638) — none accurate, none beats wallfunc.

**=> Correction to 6.5 lessons 1 & 4.** The binding constraint is **not** the wall-shear *readout
operator* (ML readout == wallfunc on the same flow). It is **flow fidelity**: the kine surrogate
(trained on clot-free Carreau flow) + geometry occlusion does not reproduce COMSOL's *coupled*
velocity/shear field, so no local read of it — analytic or learned — recovers `spf.sr`. The earlier
"0.53 vs 0.71 on the same flow" gap was measured on COMSOL's own (GT) flow; on the *kine* flow the
ceiling is ~0.38 regardless of reader.

**=> Revised S3 fork (the real architecture choice).** To close 0.38 -> 0.75 deployably we need a
model that predicts COMSOL's **coupled `spf.sr` field** for unseen geometry, using mesh structure
(global momentum balance), not per-node local features:
  - **A (recommended): GNN coupled-shear surrogate.** Input = mesh graph + current clot-occlusion
    mask (+ optionally the kine velocity field as a prior); output = `spf.sr` field; supervise
    directly on the exported 201-frame `spf.sr`, LOAO. Feeds the validated exact-gate + deposition.
    This is the "learn the operator on our data, generalize across geometry" path — the per-node RF
    fails precisely for lack of mesh/global receptive field, which is the GNN's strength.
  - **B: retrain the kine flow model on clot-laden coupled COMSOL flow** (with `mu1(Mat)` gelation /
    moving boundary) so its occluded solve matches COMSOL; then wallfunc readout suffices. Heavier
    (needs coupled velocity training data), but yields a true flow model.
  - **C: anchor-only oracle pipeline** (use exported `spf.sr` directly) — not deployable to new
    geometry; only a within-anchor upper bound.

Net: the local readout-corrector idea is closed out (negative result, tests above). The lever is a
**geometry-generalizing coupled-shear predictor** (A or B).

### 6.7 Deploy-stack error decomposition + gate-calibration verdict (2026-06-18)
Scripts: [scripts/diag_deploy_error_decomp.py](../scripts/diag_deploy_error_decomp.py),
[scripts/diag_deploy_gate_calib.py](../scripts/diag_deploy_gate_calib.py) ->
`outputs/reports/comsol_validation/deploy_error_decomp.json`, `deploy_gate_calib.json`.

We stopped before building the GNN shear surrogate to first *confirm the lever* on the existing
deploy stack (GraphSAGE species -> `mu1(Mat)` gelation -> trigger). Findings:

1. **Where the deploy stack stands.** `deploy_ab_eval` (fi_mat): with **GT COMSOL flow** p007 F1
   **0.70**, holdout mean **0.52**; with **deployable frozen-kine flow** p007 **0.48**, holdout
   **~0.37**. So the species model is good *given good flow*; flow fidelity costs ~0.22.
2. **The dominant deployable error is PRECISION, and it lives ON THE WALL.** Every anchor: recall
   **0.85-1.0**, precision **0.06-0.37**. The false positives are overwhelmingly wall nodes
   (FP_wall >> FP_near-wall >> FP_lumen); the model over-fires clot along the wall.
3. **Those wall-FP are gate-prunable in PRINCIPLE.** 83-100% of FP fall in regions an **ideal
   (exact `spf.sr`) gate** would reject; applying it lifts the complete-sim anchors hard
   (p010 0.49->0.87, p006 0.35->0.69, p005 0.28->0.52, p007 0.63->0.72). Cohort mean 0.55 -> **0.68**.
4. **But calibration of the DEPLOYABLE gate cannot recover it.** Pruning the model's predictions by
   kine-wallfunc shear rank, even with a **per-patient oracle threshold**, recovers only **+0.015**
   (0.548 -> 0.563); the cohort-best transferable threshold is "don't prune." Pruning by exact
   `spf.sr` rank recovers **+0.13**. => it is a **ranking** failure (the deployable shear does not
   order true-clot vs false-positive wall nodes), **not** a threshold/calibration failure.
5. **~23% of the remaining misses are deep-lumen** (FN beyond 1 hop from the wall) — a separate
   *recall*/growth lever, independent of the flow-fidelity wall.

**=> Verdict.** Two orthogonal levers, with very different cost:
  - **Precision (+~0.13, deployable ceiling 0.68): BLOCKED by flow fidelity.** Confirmed a 4th
    independent way — analytic readout, learned readout, global-tabular geometry, and now oracle gate
    calibration all land at AUC ~0.74 / +0.015 F1. Clot-free kine flow (even geometry-occluded) does
    not encode coupled stagnation; the kine model cannot sharpen flow around a *tiny* nucleating clot
    (occluding a few nodes barely changes the solve, and `mu` injection is OOD / wrong-sign, see 6.5).
    Unblocking requires the heavy path: a flow model trained on COMSOL **coupled/clotted** fields.
  - **Recall (deep-lumen ~23%): NOT blocked** — a lumen-ward growth/neighbor-recruitment head is
    independent of shear fidelity.
  - **Species channels are saturated for generalization.** The rank ladder shows the best single
    add-on (FG=fibrinogen) lifts p007 guiding +0.04 and is the *only* add-on that also helps holdout;
    AP/RP/T/etc. help p007 slightly but **flatten or hurt holdout** (overfitting). More species is
    not the lever; a leaner Mat(+FG) core likely generalizes better.

**FI ablation (run 1, 2026-06-18)** — [outputs/biochem/biochem_gnn/fi_ablation/](../outputs/biochem/biochem_gnn/fi_ablation/fi_ablation_report.md),
launcher result ranked by holdout guiding (deploy_frozen / kine flow):

| Leg | p007 guiding | holdout guiding |
|-----|--------------|-----------------|
| **Mat+FG** (winner) | **0.555** | **0.393** |
| FI+Mat (old baseline) | 0.530 | 0.368 |
| Mat | 0.513 | 0.360 |
| FI+Mat+FG | 0.488 | 0.334 |

  - **FG (fibrinogen) helps** (+0.033 holdout vs Mat-only); **FI (polymerized fibrin) is dead weight
    and harmful** — FI+Mat barely beats Mat (+0.008), and adding FI on top of Mat+FG *hurts* -0.059.
  - **=> Lock the species core at Mat+FG; drop FI.** Wins on both p007 (+0.025) and holdout (+0.026)
    vs the FI+Mat baseline. Consistent with `mu1(Mat)`-dominated gelation + fibrinogen-precursor
    assist; the fibrin channel mostly adds geometry-specific noise that hurts transfer.
  - Modest absolute gain (~+0.026), as expected — species are not the big lever. The dominant
    wall-precision error is untouched; that is the next move (precision-aware footprint head).

### COMSOL physics principles established (consolidated)
- **Clot location = pre-existing low-shear STAGNATION**, sharpened over time as the growing clot
  reroutes flow (coupling, the +0.30 lever). Gelation is **`mu1(Mat)`-dominated** (Mat autocatalytic
  aggregation); `mu2(FI)` fibrin is secondary. Bulk species (`ap`/`rp`) are ~flat at the IC = passive.
- The low-shear gate is **necessary but not sufficient**: residence time + `ap` availability + autocat
  neighbor recruitment add structure a pure shear cutoff misses (trigger ceiling ~0.77; label ceiling
  0.85 wall+1hop / 0.99 unrestricted).
- Coupling must be expressed as **geometry/occlusion** (wall-ify clot nodes), never viscosity injection.
- Units: COMSOL CGS/uM vs repo SI — decode all species to SI consistently (past F1=0 bug).

### 6.8 Moves 2+3+4 A/B + the epochs win (2026-06-19)
Scripts: [scripts/go_species_moves234.ps1](../scripts/go_species_moves234.ps1),
[scripts/summarize_species_moves234.py](../scripts/summarize_species_moves234.py) ->
`outputs/biochem/biochem_gnn/moves234/`. Wiring: stagnation feats in
[species_pushforward_gnn.py](../src/core_physics/species_pushforward_gnn.py) (`SPECIES_STAGNATION_FEATS`),
weighted Tversky in [species_gelation_readout.py](../src/core_physics/species_gelation_readout.py)
(`SPECIES_FOOTPRINT_TVERSKY`). Also fixed a latent bug: `species_gelation_readout` imported the wrong
`gt_mu_anchor_cap_si` overload (crashed whenever `physics_readout` was on).

Two-leg A/B, Mat+FG, 40 ep, deploy_frozen / kine flow:

| Leg | p007 F1 | p007 prec | p007 rec | holdout F1 | holdout prec | holdout rec |
|-----|---------|-----------|----------|------------|--------------|-------------|
| Mat+FG control (moves OFF) | **0.689** | 0.645 | 0.738 | **0.591** | 0.448 | 0.915 |
| Mat+FG +moves234 | 0.553 | 0.415 | 0.828 | 0.438 | 0.299 | 0.924 |
| (ref) fi_ablation Mat+FG 12 ep | 0.548 | - | - | 0.369 | - | - |
| (ref) locked baseline | 0.701 | - | - | 0.523 | - | - |

1. **The real win = EPOCHS.** Mat+FG at 40 ep (vs the 12-ep fi_ablation) lifts holdout F1
   **0.369 -> 0.591** and p007 **0.548 -> 0.689**, overtaking the locked baseline holdout
   (0.523) by **+0.068**. Free, robust. Adopt 40-ep Mat+FG as the new deploy baseline.
2. **Moves 2+3+4 NET HURT** (holdout F1 **-0.153**), and move 2 moved precision the WRONG way
   (holdout prec 0.448 -> 0.299, recall up) = more flooding, not less.
3. **Why (confounded, but architecturally decisive):** `clot_phi_f1` was **frozen at 0.140 for all
   40 ep**. (a) **Selection confound** — `physics_readout=1` (needed for the footprint-loss hook)
   flips checkpoint selection to the `physics_on` score branch (`0.5*clot_phi_f1 + ...`); with that
   stuck, selection rode `growth_f1` noise instead of `deploy_mat_f1` (which the control climbed to
   ~0.77). (b) **The differentiable gelation phi is saturated/decoupled from the deploy trigger** —
   so the Tversky had ~no gradient traction on the deploy-relevant Mat field. (c) The physics aux
   losses instead inflated Mat broadly -> more trigger firing -> lower precision. (d) M4 stagnation
   feats confounded in the same leg (`deploy_mat` ended 0.643 vs 0.763).

**=> Decisive lesson.** Shaping the **differentiable gelation sigmoid does NOT reach the deploy
footprint** (both deploy paths use `rollout_t0_clot_phi`/`forward_physics_trigger_phi` on the Mat
field, not the sigmoid). The precision lever must act on the **trigger-physics path** (or supervise
the actual deploy footprint with matched checkpoint selection), not the gelation readout. The
loss-on-gelation-phi route for moves 2/3 is closed out. M4 features remain untested in isolation.

### 6.9 Final flow-lever ladder on the REAL deploy clot F1 — all flow routes CLOSED (2026-06-22)
Scripts: [scripts/go_clot_veto_zkin_ladder.ps1](../scripts/go_clot_veto_zkin_ladder.ps1),
[scripts/eval_clot_veto_zkin_ladder.py](../scripts/eval_clot_veto_zkin_ladder.py) ->
`outputs/biochem/corrector_coupling/veto_zkin_ladder/ladder.json`. Model:
`flow_aware_leashed_dynamic/sage` (latent-leash + flow feats). Scored on the **deploy clot F1**
(`rollout_t0_clot_phi` + relaxed metrics), 6 anchors, main eval time per anchor. Four configs:
`none`; `kine_veto` / `gt_veto` (drop predicted-clot nodes with high kine / GT shear, percentile
**swept = oracle-calibrated ceiling** of the veto); `tiled_zkin` (no veto, `z_kin` refreshed
mid-rollout by **geometry occlusion** — clot nodes -> wall, DEQ re-solve, 1-3 refreshes/anchor).

| anchor | t | none | kine_veto | gt_veto | tiled_zkin |
|--------|---|------|-----------|---------|------------|
| p001 | 200 | 0.591 | 0.591 | 0.591 | 0.583 |
| p002 |  66 | 0.688 | 0.688 | 0.688 | 0.688 |
| p003 |  28 | 0.738 | 0.738 | 0.738 | 0.738 |
| p004 |  62 | 0.622 | 0.622 | 0.622 | 0.620 |
| p006 | 200 | 0.638 | 0.638 | 0.638 | 0.638 |
| p007 | 200 | 0.606 | 0.606 | 0.606 | 0.600 |
| **holdout mean** | | **0.656** | **0.656** | **0.656** | **0.653** |

1. **The shear veto is DEAD — even at the ceiling (GT shear).** Every veto sweep picked
   percentile 100 (= veto nothing) on every anchor; `kine_veto == gt_veto == none` to 3 dp. A
   veto can only trade recall for precision, and that trade is **never net-positive**, even with
   **perfect COMSOL shear**. Mechanistic reason: these models are recall-heavy / over-predicting
   (p003 r=0.97 p=0.59; p004 r=0.97 p=0.46), yet the false positives sit in the **same low-shear
   pocket** as the true clot, so **no shear threshold separates FP from TP**. Precision is not an
   output-side flow problem.
2. **`z_kin` occlusion coupling is a no-op-to-slightly-negative** (holdout 0.656 -> 0.653; worst
   p001 -0.008). The refreshes fired (counts 1-3), so the only in-distribution way to make `z_kin`
   clot-aware genuinely changes nothing useful on the deploy footprint, and adds slight noise. The
   model already extracts what it needs from the baseline (clot-blind) latent.
3. **`z_kin` cannot be set from a corrector (u,v) field** — it is the DEQ *equilibrium*; `UV_PRIOR`
   is only a warm start the fixed-point solve washes out. The only clot-aware `z_kin` levers are
   mu-injection (OOD, §6.3) and geometry occlusion (tested here: no-op). There is no other route.

**=> Verdict: ALL deployable flow routes are now exhausted** — input-side (flow feats §6.6,
occlusion `z_kin` §6.9), output-side (shear veto §6.9, gate form §6.1), and coupling (corrector
§6.6). None move the **real deploy clot F1**. The earlier "+0.29 flow lever" was an artifact of a
different (oracle/Mat-Dice) measurement, not the deploy metric. **Stop spending budget on flow.**
The precision bottleneck is the **Mat predictor / species generalization itself** (the one robust
win to date was *epochs*, §6.8: holdout 0.369 -> 0.591). Next budget goes to the Mat predictor
(data, training length, scope), not the flow path.

## 6.10 A/B/C/D non-flow precision ladder (RESULT: geometry is a real lever; loss-tuning is not)

Flow is exhausted (6.9), so the remaining OPEN levers for wall-FP precision are non-flow. Ran a clean
2x2 factorial (`scripts/go_species_abcd.ps1` -> `summarize_species_abcd.py`), all legs fresh, same data
(6 anchors, moves234 convention so `score_clot_w` distinguishes selection), 40 ep, deploy_frozen eval,
val=p007. "rest" = mean over the 5 non-val anchors (in-sample, relative comparison).

- **A baseline** - phase recipe (`fp_weight=8`, `mature_fp_exempt`, mat-based ckpt selection).
- **B footprint_sup** - stronger Mat-field FP penalty (`fp_weight 8->16` via `ActiveGrowthHuberLoss`,
  the term that reaches the deploy trigger) + matched ckpt selection on deploy clot F1 (`score_clot_w=0.6`).
- **C geom_feats** - static NON-FLOW geometry discriminators appended to GNN inputs
  (`SPECIES_GEOM_FEATS=1` -> `_geometry_band_features`): `[width, width_gradient, wall_curvature]`,
  per-band standardized, no kine solve. `meta["geom_feats"]` round-trips to deploy/viz.
- **D both** - B + C.

| leg | F1 p007 | F1 rest | prec rest | rec rest | dF1 rest |
|-----|---------|---------|-----------|----------|----------|
| A baseline      | 0.693 | 0.597 | 0.452 | 0.917 | -      |
| B footprint_sup | 0.693 | 0.598 | 0.457 | 0.910 | +0.001 |
| **C geom_feats**| 0.687 | **0.626** | **0.502** | 0.897 | **+0.029** |
| D both          | 0.696 | 0.610 | 0.474 | 0.909 | +0.013 |

1. **C (geometry) is the first real non-flow precision lever** since flow was abandoned: rest F1
   +0.029, driven by **precision +0.050** (0.452->0.502) at near-constant recall (0.917->0.897) -- the
   favorable precision-up/recall-held signature, not a recall trade. Biggest wins p001 +0.063 /
   p002 +0.089 / p006 +0.025; tiny losses on p003/p004/p007. Cost: negligible (static feats, 1324s
   vs A 1284s). The `z_kin`-redundancy worry was **wrong** -- explicit width/expansion/curvature gives
   the GNN a discriminative handle the flow latent only carried implicitly.
2. **B (footprint supervision) is a no-op** on the real deploy metric (rest +0.001, prec +0.005) DESPITE
   its in-training selection metric (`deploy_clot_g`) climbing impressively to 0.74 (rprec 0.43->0.77).
   Cautionary: the matched-selection in-training number **overstated** the gain; the saved ckpt is no
   better than A at eval. Cranking the FP loss harder cannot fix precision because the model lacks the
   **information** to know which predictions are FPs -- it is an information problem, not a loss-weight
   problem.
3. **D < C: B interferes with C** (rest 0.610 < C 0.626; prec 0.474 < C 0.502). Once geometry supplies
   the missing info, the heavier FP loss + clot-selection just distort placement. Do NOT combine.

**=> Verdict: adopt leg C (geometry feats); drop B.** Precision was an *information* deficit, and cheap
deployable static geometry is the fix. Next: (a) confirm C on a TRUE holdout (LOAO / `--exclude-val`)
to separate fit from generalization before promoting; (b) enrich the geometry block (branch distance,
multi-hop curvature) since the minimal 3-ch version already moved precision; (c) re-promote the locked
deploy baseline with `SPECIES_GEOM_FEATS=1` if holdout confirms. Re-run any leg:
`powershell -File .\scripts\go_species_abcd.ps1 -Legs C` (artifacts in `outputs/biochem/biochem_gnn/abcd_precision/`).

## 6.11 Theory recap: what IS the clot, why FI is irrelevant, the perfect-Mat ceiling

Consolidates answers that were scattered across §5, §6.4 and `COMSOL_PHYSICS_VALIDATION.md`.

- **The clot label IS mu_eff, and mu_eff IS Mat.** GT clot = `relu(mu_eff(t) - mu_eff(0)) >= thresh`
  (`gt_growth_commit_mask_at_time`, `clot_growth_masks.py`). `mu_eff = mu1(Mat) + mu2(FI) + ...`, but
  COMSOL-validated **`mu2(FI) == 0`** and `mu1(Mat)` is a HARD step at `Mat = 2e7 plt/cm^2`
  (`PhysicsConfig.viscosity_mat_crit`, transition zone ~7e6). So the clot mask is, by construction,
  ~the Mat field crossing 2e7. Nothing else defines it.
- **=> FI cannot matter mechanistically** (zero viscosity -> zero contribution to the label). Channels:
  FG (fibrinogen, *precursor*) = 7, FI (fibrin, *product*) = 8, Mat (mature aggregate) = 11.
  Empirically (fi_ablation 2026-06-18, holdout guiding): Mat+FG **0.393** > FI+Mat 0.368 > Mat 0.360 >
  FI+Mat+FG 0.334 -- FI as a *feature* HURTS transfer (-0.059 on Mat+FG); FG (precursor) helps (+0.033)
  as a "reaction-active here" marker; FI (product) overfits geometry. Core locked at **Mat+FG**.
- **FI never reaches its trigger (measured, 2026-06-23).** The mu2(FI) soft-step fires at
  `viscosity_fi_crit = 0.6 uM` (-> `mu_ratio_max = 80`; COMSOL step Location 0.6 / To 80 / transition 0.2).
  Decoding fibrin from the GT graphs (`FI_uM = 7 * expm1(log1p_nd)`, ch `y[:,4+8]`) across all 10
  anchors: **peak FI = 0.0001-0.0161 uM (p007 highest, the most mature sim), 37x-6000x BELOW the 0.6 uM
  trigger; 0% of nodes cross it at any time** (`scripts/_probe_fi_trigger.py`). So `mu2(FI) == 0` is not
  an assumption -- it is what the data does: fibrin is ~37x sub-trigger even at maximum maturity, the
  step never activates, fibrin adds exactly nothing to `mu_eff`, and the clot is 100% `mu1(Mat)`. This
  matches the user's COMSOL plots: fibrin-concentration peak ~0.012-0.016 uM (x10^-3 colorbar) while the
  viscosity/clot (red) sits on the wall = the Mat term. (Spatial IoU(FI,clot) is moot: even where fibrin
  is relatively high it is still sub-trigger, so it cannot contribute regardless of overlap.)
- **Perfect-Mat ceiling (the answer to "if Mat were perfect, how much clot?"):**

  | layer | F1 | meaning |
  |---|---|---|
  | perfect Mat, **unrestricted graph** | **0.99** | clot ~= Mat-threshold; the Mat->clot map is essentially exact |
  | perfect Mat, **wall+1hop band** (what we model) | 0.85-0.87 | deep-lumen recall lost to the band restriction |
  | deployable trigger now (predicted Mat + gate) | ~0.60-0.70 | predicted-Mat error + gate debt |

  So **perfect Mat ~= the clot (0.99)**. The deploy gap is (a) *predicting* Mat and (b) the wall+1hop
  band + low-shear gate -- NOT the Mat->clot mapping. (The older "0.77 coupled-flow ceiling" §6.4 is a
  different swept/oracle metric; §6.9 showed flow levers do not move the real deploy clot F1, so treat
  0.77 as metric-specific, not the operative target. Operative target ~ 0.85 band-restricted.)

- **Why Mat precision is hard (intuition).** Note the asymmetry: recall ~0.85-1.0 but precision
  ~0.06-0.45 (§6.10/§6 error-decomp). Four compounding reasons, all on the FP (over-paint) side:
  1. **Hard step on a smooth field.** True Mat is near-binary -- a pocket either autocatalytically
     commits (Mas feedback ~140x, runs away past 2e7) or stays ~0, giving a SHARP contiguous blob. The
     GNN emits a smooth field then thresholds at 2e7, so small magnitude errors near the edge "bleed"
     past the crisp true boundary -> a ring of wall FPs. (A steep threshold amplifies field error into
     footprint error.)
  2. **Committing vs merely-eligible is set by COUPLED flow we cannot deploy.** The deposition gate is
     low-shear stagnation in the flow *around the growing clot*. Many wall-band nodes are low-shear in
     the baseline (clot-blind) kine flow but never actually commit; only the coupled field separates
     them. Deployably that signal is unrecoverable (§6.9), so the model paints all *eligible* wall
     pockets -> high recall, low precision. (This is exactly why geometry feats, §6.10, helped: a
     deployable proxy for *which* pockets are real.)
  3. **Recall-first under class imbalance.** Clot is a small fraction of wall-band nodes; the cheapest
     way to not miss it is to over-paint the wall. Loss-reweighting alone (leg B) can't fix this -- it
     adds pressure but not the missing discriminative information.
  4. **Temporal lock-in.** Mat is an accumulated/integrated quantity; an early over-gated node keeps
     accumulating and the mature-commit freeze locks the FP in -> errors compound forward, not cancel.
  Evidence it is genuinely a Mat-*placement* problem (not only flow): even with **GT flow** the band
  ceiling is ~0.70, ~0.15-0.30 below the perfect-Mat band ceiling (0.85) -- that residual is the GNN's
  Mat field being too diffuse/broad, independent of flow fidelity.

## 6.12 Geometry-context importance probe (RESULT: expansion + bend topology transfer; flow proxies not yet tested)

Script: [scripts/_diag_geom_context_importance.py](../scripts/_diag_geom_context_importance.py) ->
`outputs/reports/comsol_validation/geom_context_importance.json`. All anchors, final frame,
inside the **wall+3hop** deploy band (triangle6 baseline topology).

**What was tested (static, clot-blind, deployable):**
`sdf`, `width`, `width_d1`, `width_d2`, 1-hop `expansion`, 1/2-hop `wall_curv`, BFS
`downstream` from inlet; reference shear proxies `mu_prior`, `wss_prior` (analytic Carreau /
WSS priors, no kine solve).

**What was NOT tested (physical gaps -- see below):**
deployable kine-flow stagnation (`log speed`, neighbour shear proxy), graph `divergence`
(accel/decel), GT COMSOL `spf.sr`, time-varying flow from `data.y`, chemistry
(`FG`/depletion, neighbour `Mat`), branch/bifurcation distance.

### Key findings

1. **Static geometry transfers strongly (LOAO).** `geom_only` holdout mean AUC **0.874** vs
   `shear_only` **0.608**; `geom+shear` **0.883** (+0.009 over geometry alone). Geometry is the
   cross-patient signal; analytic shear priors add almost nothing once geometry is present.
2. **Dominant geometry contexts (pooled logistic |coef|):**
   `expansion` (+1.50) > `wall_curv1` (+1.16) > `wall_curv2` (-1.13) > `sdf` (-0.84) >
   `downstream` (+0.38) >> `width` / `width_d1` / `mu_prior` (~0.14). `wss_prior` = 0.
   Physically: clots sit in **expansion pockets on bent walls, closer to the wall surface**
   (negative `sdf` coef), not merely "wide" or "far downstream."
3. **Signal lives IN the band, not on the wall mask.** Wall-only univariate AUC for
   `expansion`/`wall_curv` ~**0.50** (random); full band AUC ~**0.84**. The precision problem is
   *which band node commits*, not wall vs lumen. `mu_prior` is the opposite: wall AUC 0.72, band
   0.61 -- eligibility-ish, not placement.
4. **Pooled univariate AUC (band):** `expansion` 0.81, `wall_curv1/2` 0.80/0.82, `width_d1`
   0.58, `width` 0.17, `sdf` 0.20 (univariate misleading for sdf -- multivariate strong).

### Physics interpretation

COMSOL deposition is gated by **low-shear stagnation**, but stagnation is not a static node tag --
it is a **geometry-conditioned flow phenomenon** (expansion -> recirculation; bend -> asymmetric
low-shear pocket). This probe shows the **static geometric correlates of those pockets** carry
~0.87 LOAO AUC for final clot location, while **analytic shear priors do not**. That supports leg C
and suggests architecture should encode **local expansion + multi-hop wall bend topology** (not
raw width or global position alone).

**Caveat:** we have not yet run the symmetric probe for **actual flow stagnation/acceleration**
(kine `speed`, neighbour shear proxy, graph divergence, GT `spf.sr`). Prior deploy ladders (6.9)
show post-hoc shear cannot fix precision once predicted, but we still need to measure whether
flow features add LOAO signal *on top of* geometry (oracle upper bound vs deployable kine flow).

### Recommended follow-up probes (before architecture lock-in)

**Comprehensive probe (wired):** [scripts/_diag_clot_context_comprehensive.py](../scripts/_diag_clot_context_comprehensive.py)
-> `outputs/reports/comsol_validation/clot_context_comprehensive.json`. Adds kine/GT flow
stagnation+divergence, oracle `spf.sr`, chemistry, neighbour Mat/Mas, physics gate proxies,
incremental LOAO sets, and commit-vs-eligible bifurcation probe.

```powershell
python scripts/_diag_clot_context_comprehensive.py --hops 3
```

| probe | features | answers |
|---|---|---|
| **P2 flow** (in comprehensive) | kine `log speed`, shear proxy, `divergence`; GT `y[:,u,v]` same block | does stagnation/accel add beyond geometry? |
| **P3 oracle shear** (in comprehensive) | exact `spf.sr`, `spf.sr < lss` gate | ceiling for "stagnation explains commit" within band |
| **P4 commit vs eligible** (in comprehensive) | geometry + flow, label = GT clot vs "low-shear eligible non-clot" | tests the bifurcation directly, not just clot vs all |

### Architectural implications (provisional)

- **Do enrich:** `expansion`, multi-hop `wall_curv`, `sdf` (within-band nearness).
- **Deprioritize:** `wss_prior`, raw `width`, global `downstream` alone.
- **Do not assume flow is dead without P2/P3** -- geometry may be proxying stagnation; confirm
  whether explicit flow stagnation/divergence adds incremental LOAO before finalizing inputs.

## 6.13 Comprehensive clot-context probe results (2026-06-25)

Script: [scripts/_diag_clot_context_comprehensive.py](../scripts/_diag_clot_context_comprehensive.py)
-> `outputs/reports/comsol_validation/clot_context_comprehensive.json`. 48 features / 8 groups,
all anchors, final frame, wall+3hop band.

### CRITICAL caveat: half the "winners" are CIRCULAR (they ARE the label)

The GT clot label is `relu(mu_eff(t) - mu_eff(0)) >= thresh` with `mu_eff == Carreau x (1 +
mu1(Mat) + mu2(FI))` and `mu2(FI) == 0` (6.11). So any feature read from **GT species at the
eval time** trivially reconstructs the label:

| feature | group | pooled AUC | why circular |
|---|---|---|---|
| `mat_growth_log` = Mat(t)-Mat(0) | chem | **1.000** | this IS the label definition |
| `mat_log_nd` = Mat(t) | chem | **1.000** | clot == Mat>crit |
| `m_log_nd` / `mas_log_nd` | chem | 0.980 | wall precursors tightly coupled to Mat |
| `nbr_mat_log` / `nbr_mas_log` | neighbor | 0.97 | GT Mat smeared 1 hop = label dilated |
| `fg_depletion` / `fi_log_nd` | chem | 0.847 | reaction co-products, eval-time GT, co-located |

=> The LOAO sets that include `chem`/`neighbor`/`gate` (**0.998-0.999**) are **NOT deployable**:
at deploy we do not have GT Mat at the eval time -- predicting it IS the task. The script's old
`deployable_all` label was wrong (it included eval-time GT species); treat those rows as an
**oracle sanity check** that merely re-confirms clot==Mat. `oracle`/`all` rows are **nan** because
exact `spf.sr` is cached for **p007 only** (9/10 anchors NaN -> LOAO degenerate); oracle shear is a
p007 univariate result, not a cohort LOAO.

### Honest, DEPLOYABLE result (clot-blind / t0 features only)

| feature set (deployable) | LOAO holdout AUC |
|---|---|
| geom | **0.882** |
| geom + static priors | 0.878 (priors add ~0) |
| **geom + static + kine flow** | **0.897** |
| (+ GT eval-time flow, oracle) | 0.927 |

- **Geometry is the deployable signal (LOAO 0.88).** Static analytic priors (`mu_prior`,
  `wss_prior`) add **nothing** (0.882 -> 0.878). Deployable **kine flow** (t0 speed / shear proxy /
  divergence) adds only **+0.015** (0.882 -> 0.897). Oracle eval-time GT flow adds **+0.03** more
  (0.927) -- the "coupled flow knows where the clot is" effect (6.9), unrecoverable deployably.
- This matches 6.9/6.12: **flow is a weak deployable lever on top of geometry**; geometry already
  proxies most of the stagnation structure.

### Commit-vs-eligible bifurcation (section E) -- the precision lever

Label restricted to **GT clot vs low-shear-eligible-non-clot** (the actual FP population). Excluding
circular GT-species features, the separators are **geometric**:

| deployable feature | commit-vs-eligible AUC |
|---|---|
| `wall_curv2` (2-hop bend) | 0.862 |
| `wall_curv1` (1-hop bend) | ~0.80 |
| `expansion` | ~0.80 |
| flow speed (kine/gt) | ~0.10 (i.e. low speed -> commit, |d|~0.40, but partly circular: eligible was defined by low shear) |

=> **Geometry (multi-hop wall curvature + expansion) is what distinguishes a committed pocket from a
merely-eligible one.** This is the precision bottleneck of 6.7/6.11, and it is a *geometry* signal.
Confirms leg C (6.10) at the data level and tells us *which* geometry: **curvature/expansion
context, multi-hop**, not raw width or global position.

### Wall vs band (section F)
`width` flips sign wall (0.54) vs band (0.14): geometry signal is **contextual/multivariate**, not a
raw per-node value -- it needs neighbourhood aggregation (GNN territory), not a lookup.

### What this means for "highly precise" (vs G neighbor-gate baseline)

- **Matched-budget comparison (the controlled one):** `baseline_fast` is **architecturally
  identical** to the locked triangle6 baseline (dual-head `fi_mat`, channels [8,11]) -- it differs
  ONLY in training budget (160 vs 865 windows, ~10 vs 50 epochs). In the matched FAST regime, **G
  (Mat-only + neighbour gate) BEATS the dual fi_mat head**: clot_f1 0.455 vs 0.373 (+0.082), Mat_f1
  0.599 vs 0.448 (+0.151). So the Mat-only simplification did **not** hurt; with the neighbour gate
  it helped.
- **Do NOT compare G-fast to the locked baseline (0.634).** That -0.179 gap is the **training-budget
  difference** (fast vs 50-epoch), NOT architecture -- G was never run at full budget. The robust
  prior lesson (6.8) stands: epochs are the single biggest lever. G at full budget is untested and
  is the obvious next run.
- The honest **deployable ranking ceiling is LOAO AUC ~0.90** (geom+static+kine). At wall-band clot
  prevalence ~5% (1108 clot / 22359 band), AUC 0.90 mechanically caps precision at a single
  threshold -> the deploy clot_f1 ~0.45-0.63 we see is roughly what 0.90-AUC ranking yields under
  that imbalance. **The gap to "highly precise" is NOT mainly model capacity on current inputs; it
  is (a) the AUC->F1 collapse under imbalance + thresholding, and (b) a genuine deployable-signal
  ceiling.**

### Architectural implications (updated)

1. **Geometry-context head is the right deployable lever** (curvature/expansion, multi-hop). Enrich
   beyond leg C's 3 channels: 2-hop curvature, expansion_2hop, branch/bifurcation distance.
2. **Autocatalytic recurrence is real, not leakage.** `nbr_mat` is circular only as a *static GT*
   feature; in a **closed-loop rollout with PREDICTED Mat**, neighbour-Mat feedback is legitimate
   physics (the Mas autocat term). G's neighbour gate is the right instinct -- the issue is the
   Mat-only head, not the neighbour coupling.
3. **[SUPERSEDED by 6.14]** This item proposed post-hoc calibration / per-geometry threshold / a
   commit head + two-stage narrowing. The 6.14 sweep **refuted the deployable post-hoc forms**:
   oracle per-vessel thresholds recover 0.000, and geometry adds nothing over the predicted Mat
   field. The surviving direction is improving the **predicted-Mat field itself** (training budget;
   bifurcation-aware *training* loss; in-loop autocatalysis) -- see 6.14.
4. **Stop expecting flow or more species to move it** -- both are saturated deployably (flow +0.015,
   FI/chem circular). Budget -> geometry context + commit head + calibration.

### Probe honesty fixes (wired)
Script updated: explicit `CIRCULAR_GROUPS` (chem/neighbor/gate = eval-time GT) separated from
`DEPLOYABLE_GROUPS` (geom/static/kine); oracle sets gated behind spf-cache coverage; re-run for the
corrected `deployable_t0` vs `oracle` split:
```powershell
python scripts/_diag_clot_context_comprehensive.py --hops 3
```

## 6.14 Post-hoc precision-lever sweep (RESULT: post-hoc is DEAD; the predicted-Mat FIELD is the bottleneck)

Script: [scripts/_sweep_precision_levers.py](../scripts/_sweep_precision_levers.py) ->
`outputs/reports/comsol_validation/precision_levers_sweep.json`. Ran the **locked triangle6
baseline** (`species/best.pth`) rollout once per anchor, captured the continuous predicted-Mat
field, and tested every 6.13 direction **post-hoc (no retraining)** as deploy clot F1 in the
wall+3hop band.

| lever | mean F1 | d vs default | verdict |
|---|---|---|---|
| `L_default` (model phi) | 0.643 | - | reference |
| `L_matthr_global` (ORACLE global thr) | 0.642 | -0.001 | **calibration headroom = ZERO** |
| `L_matthr_vessel` (ORACLE per-vessel thr) | 0.642 | -0.001 | **per-geometry threshold = ZERO** |
| `L_geomcommit` (LOAO geometry-only) | 0.160 | **-0.483** | geometry-only F1 COLLAPSES |
| `L_mat_x_geom` (LOAO Mat+geom) | 0.637 | -0.005 | geometry adds NOTHING over predicted Mat |
| `L_mat_x_geom_nbr` (LOAO +nbr Mat) | **0.647** | **+0.005** | only positive lever (autocat), tiny |

Clot prevalence in band raw **0.049 -> geom-gate top50% 0.097 (1.96x)**.

### What this refutes (both 6.13 "deployable fixes")

1. **Problem 1 (calibration / AUC->F1 collapse) is NOT thresholding.** Even an **oracle per-vessel
   threshold** on the predicted Mat field ties the model default (0.642 vs 0.643). The model already
   thresholds its own Mat field **optimally**; there is **no recoverable F1 in recalibration**. The
   nucleation+gelation trigger == the optimal Mat cutoff.
2. **Problem 2 (geometry commit head) is dead POST-HOC.** Geometry-only LOAO F1 = **0.160** (the
   earlier 0.88 *AUC* collapses to useless *F1* at ~5% prevalence with a transferable threshold), and
   `Mat+geom` (0.637) does **not** beat `Mat` alone. **Geometry adds nothing on top of the predicted
   Mat field** -- the field already encodes whatever static geometry contributes (it is fed `z_kin` +
   SDF + geometry-conditioned flow).

   **SCOPE CAVEAT (important): this refutes POST-HOC geometry re-ranking on a FROZEN field, NOT
   in-training geometry features.** Feeding geometry to the GNN during training (leg C,
   `SPECIES_GEOM_FEATS=1`) reshapes the learned Mat field and is a *different* lever -- it does not
   need its own transferable threshold (it rides the Mat field's physical `viscosity_mat_crit`), and
   it can improve *placement* rather than re-weight a fixed field. Leg C (6.10) DID help in training
   (+0.029 rest F1, precision +0.05) but on the old corner graph / 40ep, never confirmed on
   triangle6 or LOAO. So in-training geometry is **promising-but-unconfirmed**, not dead. The sweep
   only *hints* its headroom may be modest (the field already absorbs much geometry) -- that must be
   measured directly (triangle6 leg C + LOAO), not assumed.

### The sharpened conclusion

**The predicted-Mat FIELD is the bottleneck, and it is already near its own thresholding ceiling.**
Deploy F1 ~0.64 is the *field's* ranking quality, not a calibration or missing-feature artifact.
Two facts pin this:
- Predicted Mat has a **physically calibrated operating point** (`viscosity_mat_crit`) that transfers
  across patients -- which is exactly why geometry-logistic thresholds do NOT transfer (geom F1 0.16)
  while Mat does.
- The only positive lever is **neighbour predicted-Mat (+0.005)** = autocatalytic recruitment, the
  one mechanism that is *physics*, not a static feature.

### Per-anchor structure (where the pain is)
Default F1 spans **0.206 (p008) -> 0.864 (p002)**. The low-F1 anchors are the **lowest-prevalence /
smallest-clot** ones (p008 prev 0.011 F1 0.206; p006 0.022 F1 0.55; p004 0.039 P=0.34 R=1.0 = massive
over-paint). Precision pain concentrates where the clot is tiny: a few absolute FPs crush precision.
High-prevalence anchors (p001 0.115, p007 0.137) already do better.

### => Strategy pivot (post-hoc abandoned)

- **STOP** post-hoc threshold calibration and post-hoc geometry re-ranking -- both empirically dead.
- The lever is **the predicted-Mat field itself**. Improve it via:
  1. **Training budget (epochs)** -- the proven 6.8 lever; this is the *locked* ckpt, G/full-budget
     is untested. Cheapest high-value run.
  2. **Bifurcation/sharpening trained INTO the field** -- a commit-aware *training* loss that makes
     the Mat field bimodal at the boundary (NOT a post-hoc head, which is dead). This is the only
     surviving form of "Problem 2."
  3. **Stronger in-loop neighbour autocatalysis** -- the sole positive post-hoc signal (+0.005); G's
     neighbour gate strengthens exactly this *inside* the rollout, so a full-budget G is the natural
     test (combines levers 1 + 3).
- **Two-stage candidate narrowing** (geom gate 1.96x prevalence) survives ONLY as a recall-preserving
  *filter*, low priority given geometry's post-hoc collapse; revisit only if field-quality work
  stalls.

**Next concrete run:** ~~full-budget G (Mat-only + neighbour gate)~~ **[SUPERSEDED by 6.15]** -- the
6.15 fast sweep shows the neighbour gate is a net negative and the real lever is **Mat-only scope +
rich geometry**. New next run: **full-budget `N_mat_geom_rich`** (no gate) vs the locked fi_mat 0.634.

## 6.15 Precision sweep -- in-training levers (RESULT 2026-06-25: SCOPE is the lever; gate hurts only on fi_mat)

Post-hoc is dead (6.14), so the sweep tests the surviving levers **inside training**, each as a
one-knob flip on a fast dual `fi_mat` baseline so the delta is attributable. Launcher
`scripts/go_precision_sweep.ps1 -Fast -Fresh`; legs in `src/biochem_gnn/mat_growth_simple.py`;
summary `scripts/summarize_precision_sweep.py` ->
`outputs/biochem/biochem_gnn/precision_sweep/precision_sweep_summary.json`.

### Fast result (10 ep / 16 windows, vs matched-budget `baseline_fast` dual fi_mat = clot_f1 0.373)

| leg | scope | extra lever | deploy clot_f1 | d_clot | deploy mat | clot_score |
|-----|-------|-------------|---------------:|-------:|-----------:|-----------:|
| **N_mat_geom_rich** | **mat** | rich geom | **0.446** | **+0.074** | 0.602 | 0.379 (+0.080) |
| L_fimat_geom_rich | fi_mat | rich geom | 0.312 | -0.060 | 0.424 | 0.285 |
| M_fimat_neighbor_geom_rich | fi_mat | gate + rich geom | 0.301 | -0.071 | 0.417 | 0.282 |
| K_fimat_neighbor_gate | fi_mat | neighbour gate | 0.283 | -0.089 | 0.399 | 0.282 |

**Two findings that invert the going-in hypothesis:**
1. **Scope dominates: Mat-only >> fi_mat at matched budget.** The SAME rich geometry *hurt* on
   fi_mat (L, -0.060) but *won* on Mat-only (N, +0.074). The FI channel actively wastes capacity
   under a tight budget -- exactly what 6.11 predicts (`mu2(FI) == 0`). N is the **only** leg that
   beats baseline_fast and it also beats the prior best fast leg G (0.373).
2. **The neighbour commit gate hurts ONLY on fi_mat scope -- NOT on Mat-only** (corrected by the
   6.15b O-leg run below). On fi_mat, K (gate only) is the worst leg (-0.089) and M (gate+geom)
   undercuts L (gate-free geom). On Mat-only, the gate is neutral-to-slightly-positive (G 0.476 >=
   N 0.446). So the gate's harm is **scope-coupled** (it interacts badly with the wasted FI head),
   not intrinsic.

**Misleading val proxy (logged for triage):** the fi_mat legs have *higher* val `best_score`
(0.634-0.639) than N (0.586) yet *lower* deploy clot_f1. The early-stop score rewards fi_mat's easy
state/FI reconstruction while the real deploy target rewards Mat-only -- prefer deploy clot_f1 over
val best_score for selection on this stack.

### 6.15b N+G combo + clean G re-eval (RESULT 2026-06-25; leg `O_mat_neighbor_geom_rich`)

Launcher `scripts/go_mat_ng_combo_fast.ps1 -Fresh`. `O` = N's scope+geom **plus** G's neighbour
gate. The run also re-evaluated G with the eval-recipe fix (gate restored), giving the first clean
matched-budget G number. All three are Mat-only, fast budget:

| leg | levers (on Mat-only dual head) | deploy clot_f1 | deploy mat | clot_score |
|-----|--------------------------------|---------------:|-----------:|-----------:|
| **G_dual_mat_neighbor_gate** | neighbour gate | **0.476** | 0.617 | 0.407 |
| O_mat_neighbor_geom_rich | gate + rich geom | 0.468 | 0.615 | 0.398 |
| N_mat_geom_rich | rich geom | 0.446 | 0.602 | 0.379 |

- O vs N **+0.022** clot_f1; O vs G **-0.007**. On Mat-only scope the gate is **not** harmful
  (G >= N); this **corrects** the 6.15 headline "gate is a net negative" -- that claim holds for
  **fi_mat only**.
- **Geometry and the gate do not stack:** O (both) does not beat G (gate alone). They appear to be
  partly redundant levers for the same FP-suppression job, OR the fast budget can't exploit both.
- **All three within ~0.03** (0.446-0.476) with epoch-to-epoch val_growth swinging 0.2-0.75 -- at
  this budget G/N/O are **statistically indistinguishable**. The robust, repeatable signal is the
  **scope** (all three ~0.45-0.48, all clearly > baseline_fast 0.373). Gate-vs-geom within Mat-only
  is in the noise and must be settled at full budget.

### Next runs (priority order)
1. **Full-budget Mat-only**, run G / N / O together (50 ep / all windows) vs the locked full-budget
   fi_mat 0.634. Mat-only is the candidate to challenge the locked baseline and is *cheaper* (one
   channel). The earlier "Mat-only hurt (0.455 vs 0.634)" verdict was a budget-confound. Pick the
   gate/geom winner only at full budget (fast budget can't separate them).
2. **Isolation control:** a matched-budget **mat-only, NO geom, NO gate** leg (`P_mat_plain`?) to
   split the +0.074 into pure scope vs geom vs gate. Lower priority given (1) will largely answer it.
3. The neighbour gate stays **in** the Mat-only candidate set (it was only harmful on fi_mat);
   drop it only if full-budget G <= N.

| leg | lever under test | rationale |
|-----|------------------|-----------|
| `K_fimat_neighbor_gate` | neighbour commit gate on the **dual fi_mat** head | G's instinct was right (autocatalysis = the `Mas` term, legit in closed loop with predicted Mat) but executed on Mat-only; this keeps the gate but on the full head. |
| `L_fimat_geom_rich` | enriched geometry (`SPECIES_GEOM_FEATS_RICH`): +`width_grad_2hop`, +`curv_2hop` | 6.13 §E found **multi-hop** expansion/curvature -- not 1-hop -- separate *committed* from merely *eligible* wall pockets. Leg C had only 3 (1-hop) channels. |
| `M_fimat_neighbor_geom_rich` | K + L combined | the two surviving levers together. |
| `N_mat_geom_rich` | rich geometry on the Mat-only scope | control vs leg C (does scope or geometry-richness carry the gain?). |

Implementation notes (landed):
- `SPECIES_GEOM_FEATS_RICH` extends `_geometry_band_features` to 5 channels; `SPECIES_GEOM_FEATS`
  (3-ch, leg C) unchanged. Model in-dim auto-derives from `base_feats.shape[1]`, so warm-start
  from the locked (no-geom) ckpt partial-loads conv1 columns -- no manual dim wiring.
- **Eval-recipe fix (affected G/H/J too):** `eval_mat_growth_simple._apply_ckpt_recipe` now restores
  `geom_feats` / `geom_feats_rich` / `flow_feats` / `neighbor_commit_gate(+alpha)` from ckpt meta.
  Previously the neighbour gate (which adds +1 to the spatial-gate input dim) was NOT re-enabled at
  eval, so the trained spatial head was silently dropped by the partial loader -- a latent
  underestimate of every gated leg.
- **Not in this sweep:** a dedicated commit/nucleation **head** (Problem 2) is a larger
  architectural change; gate it behind these results -- if `L`/`M` (geometry-into-field) move
  clot_f1, the bifurcation signal is learnable from inputs and a separate head may be unnecessary;
  if they don't, that justifies the head. Adaptive/two-stage thresholding stays parked (6.14: dead
  post-hoc).

## 6.16 Over-prediction diagnosis + 6h precision ladder (DESIGN 2026-06-25)

External diagnosis reviewed and **largely agreed** -- it is grounded in our code (verified:
`pred_delta = sigmoid(spatial_logits) * softplus(magnitude)`, GraphSAGE mean-agg backbone, the
neighbour-commit fraction is appended to the **gate** input only, the gate is trained with **focal**
loss and the magnitude with Huber; the sigmoid had **no temperature**). The six structural pressures
(recall-heavy precision collapse; monotone autocatalytic lock-in; hard gelation step on a smooth
field; missing coupled-flow feedback at deploy; recall-leaning incentives; equal-weight mean
pooling) all match our runs. The "attention only on the gate path, masked to committed neighbours"
read is the correct lesson from G (selective neighbour pooling, not attention everywhere), and the
risk note (vanilla graph attention smooths/spreads -> worse over-paint) is exactly our concern.

**What we add / refine (from our own evidence):**
1. **It is a ranking/separability problem more than an incentives problem.** We already run FP
   pressure (`FP_WEIGHT=8`, focal on the gate) and 6.14 showed post-hoc thresholding yields ZERO
   recoverable F1 -- so simply "add more FP weight" risks killing recall without fixing ranking,
   unless the gate gains a signal that *separates* true- from false-wall nodes.
2. **Mechanical AUC x prevalence ceiling.** Deployable ranking caps at LOAO AUC ~0.90; at wall-band
   prevalence ~5% that mechanically caps single-threshold precision around the 0.45-0.63 clot_f1 we
   see. "Highly precise" therefore likely needs **candidate narrowing** (raise prevalence) and/or a
   separating signal, not just a better-tuned threshold.
3. **The neighbour gate is double-edged** (explains O ~= G, not O > G in 6.15b): it nucleates true
   clots but its committed-neighbour blur can *also* propagate an early wall FP forward (monotone
   lock-in). Sharpening the gate (temperature) and pressuring gate-positives on zero-growth nodes
   targets exactly this.
4. **Coupled flow is empirically a weak lever** (6.9 flow feats ~ +0.015); don't over-invest there.
5. **Selection must be on deploy clot_f1 cohort, not val score** (6.15b: val best_score ranks
   inversely to deploy on this stack; G/N/O within fast-budget noise).

**6h ladder (launcher `scripts/go_mat_only_full_overnight.ps1 -Fresh`, ~30 ep / all windows, eval vs
locked fi_mat 0.634; legs in `src/biochem_gnn/mat_growth_simple.py`):**

| leg | levers (all Mat-only dual head) | tests |
|-----|--------------------------------|-------|
| `P_mat_plain` | none (no gate, no geom) | pure-scope attribution floor -- splits the +0.074 into scope vs gate vs geom |
| `G_dual_mat_neighbor_gate` | neighbour gate | proven best fast leg, now at real budget |
| `N_mat_geom_rich` | rich 2-hop geometry | geometry-into-field |
| `O_mat_neighbor_geom_rich` | gate + geom | do they stack at budget? |
| `Q_mat_gate_sharp_fp` | gate + **gate temp 0.5** + spatial focal w=3 + focal gamma 3 | 6.16 "sparsify the gate" + spatial FP pressure (precision lives in the gate) |
| `R_mat_geom_gate_sharp_fp` | Q + rich geometry | all survivors stacked |
| `S_mat_frontier_nuc` | gate + geom + **sparse nucleation + 1-hop slow front** | "choose a few sites, propagate slowly" -- nucleate at top-k model-confidence seeds, then grow only on the k-hop frontier of *predicted* committed mass |
| `T_mat_frontier_sharp` | S + gate temp 0.5 + spatial FP pressure | max-precision nucleation front |

New knobs landed: `SPECIES_CONTINUOUS_GATE_TEMP` (`gate = sigmoid(logits / T)`, `T<1` sharpens);
`SPECIES_CONTINUOUS_FRONTIER_HOPS` (growth confined to the k-hop frontier of predicted committed
Mat -- the propagation-speed lever, 1 hop/macro step); `SPECIES_CONTINUOUS_NUCLEATION_TOPK` (top-k
of the model's own gate logits may seed when the frontier is empty). All persisted in train meta and
restored at eval.

**Leakage audit (S/T -- deployability is the design constraint):** the committed mask is read from
the **predicted** rollout `log_state` (the same signal the proven neighbour-commit gate `G` uses),
and nucleation seeds come from the model's **own gate logits** -- **never** GT clot mask, `data.y`,
or oracle phi, at train OR eval. This avoids the documented `deploy_pred` vs oracle-GT-phi frontier
bug (BIOCHEM_TRAINING_PROGRESS s "frontier" entry). "Slow propagation" is enforced *structurally*
(`FRONTIER_HOPS=1`), not via any GT runtime input, so a GT-front-rate supervision loss was
deliberately **not** added (it would be a valid training target but is unnecessary and keeps the
surface clean). Known bounded train/deploy gap (shared with G): under teacher forcing the committed
mask can be GT-seeded on forced steps, but deploy uses pred and selection is on the deploy metric.

**Deferred** (bigger modules, gate behind ladder results): gate-only GAT/attention masked to
committed neighbours; explicit per-step commit-count budget rate-matched to GT growth speed (loss
form); learned geometry clot-likelihood prior head (Design A). Run order is priority-first (control +
proven trio, then the two deployable nucleation legs `S`/`T`, then the gate-precision ablations
`Q`/`R` which `T` subsumes) so a partial 8h run still answers "does Mat-only beat 0.634", "gate vs
geom", and reaches the nucleation architecture. Launcher budget raised to an 8h cap.

### 6.16 Partial full-budget overnight (STOPPED 2026-06-26; P+G only)

Launcher `scripts/go_mat_only_full_overnight.ps1 -Fresh` (40 ep / early-stop 25 / all 865 windows).
User stopped after ~10h (~593 min after G; N aborted ~ep24). Partial summary:
`outputs/biochem/biochem_gnn/mat_only_full/mat_only_partial_summary.json`.

| leg | wall | deploy clot_f1 | vs locked 0.634 | overpaint/gt | best_ep |
|-----|-----:|---------------:|----------------:|-------------:|--------:|
| **P_mat_plain** | ~289 min | **0.762** | **+0.128** | 0.040 | 37 |
| G_dual_mat_neighbor_gate | ~304 min | 0.724 | +0.091 | 0.034 | 37 |
| N_mat_geom_rich | (ep~24) | — | — | — | — |

Pivot legs U/V/S/T/Q/R **not run**.

**What we learned (full budget, honest eval on all anchors):**
1. **Mat-only scope wins at real budget.** Both completed legs beat locked fi_mat; P is the leader.
2. **Plain beats gate at full budget.** P > G (+0.038 clot_f1) even though fast budget had G >= N
   within noise -- architecture ranking is budget-sensitive; do not promote from 10-ep sweeps alone.
3. **Overpaint drops.** P/G cut `mat_overpaint_per_gt` from baseline ~0.28 to ~0.04 and slow
   `mat_front_speed_ratio` (~0.63 vs ~0.93) -- Mat-only trains a slower, more localized front.
4. **Training val metrics mis-rank again.** `deploy_mat_t` on patient007 at best epoch: P 0.691 vs G
   0.689 (tie); `val_growth_f1` swings 0.52-0.94 ep-to-ep. **Only post-train `compare.json`
   deploy_clot_f1** is trustworthy for leg selection.

### 6.17 Fast-but-honest architecture comparison budget

Goal: separate legs without ~5h/leg full runs. Unroll curriculum is the constraint:
`5 -> 10 -> 15 -> 25` at ep 1 / 11 / 21 / 31. Gate and rollout-heavy knobs need at least the
**unroll=10** phase before comparisons mean anything.

| tier | epochs | early-stop | windows | ~time/leg | use when |
|------|-------:|-----------:|--------:|----------:|----------|
| **scope** | 10 | 6 | 16 | ~15-20 min | Mat vs fi_mat scope flip (`-Fast`); proved in 6.15 |
| **arch triage** | **16-20** | **12** | **64-128** | **~45-90 min** | P vs G vs N vs U within Mat-only |
| **promote** | 35-40 | 25 | all (865) | ~5 h | 1-2 survivors only |

**Rules:**
- Always run `eval_mat_growth_simple.py` after train (`go_mat_growth_simple.ps1` does this).
  Never pick legs from inline `deploy_mat_t` or `val_growth_f1` alone.
- For arch triage, stop no earlier than **ep 16** (first epoch after unroll bumps to 10 at ep 11).
- Scope questions can stay on `-Fast`; they are not comparable to full-budget absolute F1.

Example arch-triage leg (not yet a named preset):

```powershell
powershell -File .\scripts\go_mat_growth_simple.ps1 -Leg P_mat_plain -Fresh -Epochs 20 -EarlyStop 12 -MaxWindows 64
```

Four-leg Mat-only triage (P/G/N/U) at arch tier: ~3-6 h total vs ~20 h at full budget.

### 6.18 Precision-first mat-growth recipe (2026-06-26)

Mat-growth legs now default to **anti wall-paint** training + checkpoint selection:

- **Loss:** `FP_WEIGHT=16`, new `GATE_FP_WEIGHT=4` (BCE on spatial gate at zero-growth nodes),
  `SPATIAL_LOSS_WEIGHT=2`, `SPEED_FP_WEIGHT=6`.
- **Selection:** `CLOUT_SCORE=relaxed_prec_floor`, `SCORE_CLOUT_W=0.75`, overpaint penalty on
  `deploy_clot_pred_pos_frac` in the trainer score formula.
- **New physical triage legs:** `W_mat_flow_stagnation`, `X_mat_flow_seedfront`, `Y_mat_tight_seed`,
  `AB_mat_gelation_aux`. Launcher: `scripts/go_mat_arch_triage.ps1 -Fresh`.
- Full queue: `docs/MAT_GROWTH_SIM_TODO.md`.

Prior P=0.762 used recall-heavy `deploy_mat_f1` selection; re-run P at triage tier to compare under
the new objective.

### 6.19 Mat arch triage results (2026-06-26; 8/8 legs complete)

**Budget:** 20 ep / ES 12 / max_windows 64 / precision-first recipe (`fp_w=16`, `score_clot_w=0.75`,
`relaxed_prec_floor`). ~28 min/leg (~3.7 h total). Summary:
`outputs/biochem/biochem_gnn/mat_arch_triage/mat_arch_triage_summary.json`.

**Baseline in compare:** locked fi_mat cohort mean **clot_f1=0.441** (mat-only deploy eval vs
`species/best.pth`). Prior full-budget reference **0.634** used fi_mat closed-loop rollout with
old recall-heavy selection.

| Leg | clot_f1 | d_clot | clot_score | seedP | frontP | speed | over/gt | best_ep |
|-----|--------:|-------:|-----------:|------:|-------:|------:|--------:|--------:|
| **W_mat_flow_stagnation** | **0.764** | **+0.130** | **0.978** | 1.00 | 0.98 | 0.47 | **0.01** | 20 |
| AB_mat_gelation_aux | 0.671 | +0.037 | 0.699 | 1.00 | 0.78 | 0.83 | 0.20 | 7 |
| T_mat_frontier_sharp | 0.497 | -0.008 | 0.782 | 1.00 | 0.99 | 0.29 | 0.01 | 20 |
| V_mat_frontier_geom | 0.470 | -0.053 | 0.777 | 1.00 | 0.98 | 0.22 | 0.00 | 16 |
| U_mat_frontier_only | 0.455 | +0.014 | 0.884 | 1.00 | 0.99 | 0.20 | 0.00 | 7 |
| S_mat_frontier_nuc | 0.447 | -0.076 | 0.760 | 1.00 | 0.99 | 0.21 | 0.00 | 16 |
| X_mat_flow_seedfront | 0.443 | +0.002 | 0.747 | 1.00 | 0.99 | 0.22 | 0.00 | 16 |
| Y_mat_tight_seed | 0.415 | -0.007 | 0.775 | 1.00 | 1.00 | 0.17 | 0.00 | 18 |

**Cross-budget context (full 40 ep / all windows, old selection):** P=0.762, G=0.724 (partial overnight).
**W at triage tier matches P** with **lower overpaint** (0.01 vs P 0.04).

**Per-patient (W):** strong p001-p004, p007 (0.769), p011 (0.982); weak p005 (0.505), p008 (0.514).
**AB:** p008 collapse (0.229), high overpaint p002/p006; gelation aux trades precision for recall.

**Architecture lessons:**
1. **Flow stagnation features dominate** -- `SPECIES_FLOW_FEATS=1` alone beats all SeedFront pivots.
2. **SeedFront stack (U/V/S/T) is precision-safe** (seedP=1, over/gt~0) but **under-grows** (speed~0.2).
   Adding gate+geom (S) does not beat bare U; sharp gate (T) best in family but still << W.
3. **Flow + SeedFront (X) is anti-synergistic** at triage budget -- frontier mask blocks flow gains.
4. **Tighter seeds (Y) hurt recall** without precision gain vs U.
5. **Gelation aux (AB)** trains (after mat-scope readout fix) but **wall-paints** (over/gt 0.20,
   frontP 0.78); early-stop ep7 -- not a promotion candidate without heavy FP retuning.

**AB bug (fixed):** `band_log_state_to_species12` hardcoded `STATE_DIM=2`; mat-only is 1-ch.
Fix in `species_gelation_readout.py` via `pushforward_state_bulk_indices()`.

### 6.20 Promotion candidates (post-triage)

| Tier | Leg / combo | Rationale | Next step |
|------|-------------|-----------|-----------|
| **A** | `W_mat_flow_stagnation` | Best clot_f1 + best score + low overpaint; matches P at 1/5 budget | **40 ep / all windows** |
| **B** | `P_mat_plain` under precision-first recipe | Fair head-to-head vs W at same budget + selection | Re-run 20 ep triage or 40 ep full |
| **C** | `W + frontier` (new leg) | Flow wins growth; SeedFront wins precision -- test `flow_feats + frontier_hops=1` | Wire `WX_mat_flow_frontier` one-off |
| **D** | `U_mat_frontier_only` | Best precision fallback (clot_score 0.884, over/gt~0) if W over-grows at full budget | Hold as deploy-safe backup |
| **E** | `N_mat_geom_rich` | Geometry-only control unfinished at full budget | Finish aborted overnight leg |

**Deprioritize:** X, Y, S, V, AB (unless gelation weights retuned), Q/R (subsumed by T).

**Suggested overnight queue:**
1. `W` full budget (primary)
2. `P` full budget with precision-first recipe (control)
3. Optional `WX` or `W+frontier_hops=1` triage if W full-budget overpaint rises

## 7. Open questions / before promoting
- Confirm the Mat/AP-only path generalizes across patients (p007 -> others) before
  retiring the data-driven `predict_phi_prior_rule` baseline.
- Decide where the gate + `k_aa_eff` live (per-node MLP vs scalar) once S3 numbers exist —
  keep it minimal. Evidence says the **gate** (a multi-feature membership, not a shear cutoff)
  is what separates deployable from oracle; S3 spends its budget there first.
