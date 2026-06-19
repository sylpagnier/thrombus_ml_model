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

## 7. Open questions / before promoting
- Confirm the Mat/AP-only path generalizes across patients (p007 → others) before
  retiring the data-driven `predict_phi_prior_rule` baseline.
- Decide where the gate + `k_aa_eff` live (per-node MLP vs scalar) once S3 numbers exist —
  keep it minimal. Evidence says the **gate** (a multi-feature membership, not a shear cutoff)
  is what separates deployable from oracle; S3 spends its budget there first.
