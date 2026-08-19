# PHASE 6 HANDOFF — physics-informed ML for the TEMPORAL rollout

Written 2026-08-14 for a fresh context window. Repo: `C:\Users\pgssy\thrombus_ml_model`
(Windows; PowerShell and Git Bash both available).

> **STEPS 1–3 ARE DONE AND BOTH PHYSICS MECHANISMS ARE NEGATIVE ON THE SCORE.**
> **READ `docs/PHASE6_RESULTS.md` §10–13 FIRST — they correct this file and each other.**
>
> * **Metric of record is now MEAN-over-time**, not median. The median lands at t/T≈0.5,
>   inside a flat plateau, and could not see the failure at all (§10).
> * On 22 never-selected-on vessels, mean-over-time: AP closure **−0.0001**, two-scalar
>   Damköhler **+0.0000**, against an oracle prize of **+0.0990**. Nothing is recovered.
> * **Why**: the score pays about equally for onset *order* (−0.145 if destroyed) and onset
>   *spread* (−0.134 if compressed to the model's). The closure buys spread (0.39 → 0.74)
>   on the **shear** axis, whose ordering correlation is only ≈0.30, so it loses in `rho`
>   what it gains in spread. No value of `C` fixes that — the axis is wrong (§12).
> * The model is **not firing early**: onset bias is −0.007. It fires a step that is too
>   steep, centred about right.
> * **Go/no-go for any onset model, learned or not**: it must reach `spread_ratio` ≳ 0.6
>   *while holding* `rho` above the physics model's **0.60**. Below that it nets zero
>   however good the growth curve looks. This rules out §4's `C_i = C0*exp(GNN_i)` (§13).
>
> Also corrected: `C = 68` below is window-dependent and not reproducible as stated; the
> `consumption` formula in §2.1 is algebraically just `gate*k_as`; the Damköhler ratio is a
> cohort-wide **3.07**, not 4.40. Full list at `PHASE6_RESULTS.md` §7.

**Read this file first, then `docs/PHASE3_RESULTS.md` §13–14. Do NOT read
`docs/PHASE3_HANDOFF.md` or `docs/WALL_MODEL_PLAN.md` linearly — both contain claims this
document corrects, and §7 below lists exactly which.**

---

## 0. WHERE THINGS STAND

The spatial problem is **solved and closed**. A zero-learned-parameter physics model scores,
on the canonical wall-masked `deploy_clot_score` (fingerprint verified identical to
`scripts/eval_mat_growth_simple.py`):

| | with GT t=0 flow (arm A) | deployable, `u0_pred` (arm B) |
|---|---|---|
| **SEALED (8 vessels)** | **0.9093** | **0.8567** |
| all vessels (34 / 27) | 0.7866 | 0.7210 |

Against a flow oracle of **0.9066** — i.e. arm A is *at* the ceiling of what any
evolving-flow model could do. Target was 0.60. **There is no headroom left on the final
mask; stop optimising it.**

What is wrong is **timing**. `scripts/diag_ignition_timing.py`: patient043 commits all 84
wall nodes in a *single* step at t=3000 s, while GT climbs from ~7000 s to ~12000 s and
creeps on to 30000 s. Holding the committed set fixed and varying only onset times is
worth (`scripts/diag_time_resolved_ceiling.py`, median-over-time deploy score):

```
              flash (ships today)   perfect onset   prize
train  (19)         0.8875             0.9716      +0.0842
SEALED  (8)         0.9342             0.9701      +0.0359
```

Per-vessel the prize runs from +0.007 (p014) to **+0.275** (p028), +0.176 (p019, p025).

**Your mission: recover as much of that prize as possible with a physics-informed ML
model, and produce a rollout whose growth curve matches GT.**

---

## 1. THE PHYSICS, RE-DERIVED FROM COMSOL'S OWN EXPORT

`data/reference_local/comsol_calibration/patient007_calibration_wall.txt` holds 876 wall
nodes × 201 timesteps with **every** state variable, both exported fluxes, and all three
time derivatives. Parse with `scripts/parse_comsol_wall_export.py`, then
`scripts/diag_rederive_surface_ode.py` reproduces everything below.

**Identity checks (these correct the docs):**

```
M == Mas                        max rel diff 1.86e-10   -> TWO surface species, not three
d(M,t) == d(Mas,t)              max rel diff 5.55e-10
Sat == 1 - Mas/Minf             max abs err  2.22e-10   <- CORRECT
Sat == 1 - (M+Mas+Mat)/Minf     max abs err  5.70e+01   <- what PHASE3_HANDOFF 1.3 claims. WRONG.
```

**The autocatalytic term, isolated by differencing** (fresh-deposition terms cancel
exactly, so this is the sharpest available test of the law):

```
d(Mat,t) - d(Mas,t)  vs  gate*(Mas/Minf)*k_aa*ap   ->  C = 140.5 * Da,  R2 = 0.8905
d(Mas,t)             vs  gate*Sat*(k_rs*rp+k_as*ap) ->  C =  31.9 * Da,  R2 = 0.4648
```

The law's *structure* is confirmed. But there are **two different effective Damköhlers,
ratio 4.40, and neither is the exported `Da = 1e-4`. This is unexplained and it is the
biggest open question in the physics** — it did not matter when only the saturated mask was
being scored, and it matters directly now, because the rate is what sets onset.

---

## 2. THE FLASH IS A MODEL ARTEFACT, NOT PHYSICS

For low-shear nodes `gate == 1` *exactly*, so their ODEs are identical and they must cross
`Mat >= crit` in the same step. That is the flash, and no `da_scale` / wake / in-loop
corrector can fix it — the trajectories are the same equation.

But **GT spreads anyway**. Among nodes whose gate is exactly 1
(`scripts/diag_onset_sign_test.py`, 14 vessels with >=8 such committing nodes):

```
onset spread                       mean 0.234 of horizon (0.560 on p007)
spearman(ap @ t_final,  onset)     median -0.727   (12/14 negative)  <- the explanator
spearman(sr @ t=0,      onset)     median -0.470   (11/14 negative)
spearman(|dsrx| @ t=0,  onset)     median -0.068   ( 7/14) -- uninformative here
```

`ap` is spatially **uniform at t=0** (CV 0.0000) and develops CV 0.07–0.31, falling to as
low as **1.2% of inlet** where deposition is heavy. The model holds it constant. That
missing depletion feedback is the flash.

### 2.1 A recovered algebraic law for wall AP

```
ap / ap0  =  1 / (1 + C * consumption / sr^q)
consumption = gate * (Sat + Mas/Minf) * k_as

q = 1.000, C = 68   ->  R2 = 0.9041      <- best
q = 0.333 (Leveque) ->  R2 = 0.5099
q = 0.000           ->  R2 = 0.4172
```

A wall Damköhler balance: adhesion consumption against **shear-driven renewal**. The
exponent is 1, not the Leveque diffusive-boundary-layer 1/3 — do not name it Leveque
until that is checked on more vessels.

### 2.2 It works in the rollout, with zero learned parameters

Dropping `ap_i(t) = ap0/(1 + 68*consumption_i(t)/sr_i)` into the Euler loop:

| vessel | spread flash → closure (GT) | rho flash → closure | curve_L1 |
|---|---|---|---|
| **p043** | 0.000 → **0.415** (0.725) | nan → **0.520** | 0.1403 → **0.0520** |
| p020 | 0.000 → 0.035 (0.780) | nan → **0.627** | 0.2333 → **0.1935** |
| p007 | 0.435 → 0.800 (0.890) | 0.437 → **0.724** | 0.0937 → 0.1186 |
| p013 | 0.440 → 0.440 (0.785) | 0.925 → **0.973** | 0.0565 → **0.0404** |
| p001 | 0.530 → 0.530 (0.795) | 0.983 → 0.982 | 0.0729 → 0.0746 |

**The flash breaks.** Not uniformly good: p007's `curve_l1` worsens (overshoots) and p001
is unchanged.

---

## 3. RETRACTED — the graded gate's sign is BACKWARDS

`docs/PHASE3_RESULTS.md` §13.2 introduced `g_low = sigmoid((lss - sr)/tau)`, which boosts
**low** shear on the assumption that deeper stagnation ignites sooner. §2 above measures
the opposite in **11/14 vessels**: inside the gated band, *higher* shear ignites *earlier*,
because shear replenishes AP.

This explains its measured signature exactly — it improved `curve_l1` (it spread onsets
out) while **reducing** `rho` 0.713 → 0.651 (it spread them in the wrong order).
**Retract it; do not tune it.** The AP closure supersedes it and has the correct sign.

---

## 4. WHAT ML SHOULD AND SHOULD NOT DO HERE

Three ML attempts have failed, each for a diagnosed reason. Do not repeat them.

* **In-ODE neural corrector** (`sweep_ml_v2.py`, `sweep_temporal_only.py`). Backprop
  through a 200-step stiff ODE, supervised through a near-step readout where **0.00% of
  wall nodes fell in the sigmoid's gradient band** at `phi_temp=0.005`. Also zero-init on
  the final layer gave the conv layers *exactly* zero gradient.
* **Spatial MeshGraphNet head.** Aimed at the final mask, which is already at ceiling.
  Pooled over 4 seeds: **+0.0074**, 95% CI [−0.043, +0.053] — indistinguishable from zero.
  A single lucky seed read +0.021 and was nearly reported as a win.
* **Survival / time-to-event head** (`src/differentiable_wall_model/survival_head.py`).
  Scored +0.046 on sealed (60% of the prize) but inspection showed it **degenerate** —
  predicting ~0.33 for nearly every node, 2–8 distinct values, and `rho` *dropping* to
  0.531 vs the physics ODE's 0.639. It learned a constant delay, not an S-curve. The code
  is left in place and a continuous-expectation readout was added but never tested.

**Where ML genuinely earns its place:** the local AP closure leaves a residual whose
**6-neighbour correlation is 0.58**, while `ap` itself is an extremely smooth field
(neighbour correlation **0.999**). The remaining error is *non-local transport* — a field
problem on the graph, densely supervised (~3M node-timestep samples of `ap` across the
cohort, from `AP_log1p_nd` in every pack), with no recurrence and no thresholded readout.
That is a GNN's actual job description.

Suggested form, so the physics keeps doing the structural work:

```
ap_i(t) = ap0 / (1 + C_i * consumption_i(t) / sr_i)      C_i = C0 * exp(GNN_i)
```

with `C0` calibrated and the GNN output bounded, so `GNN = 0` recovers the calibrated
physics exactly.

---

## 5. DO THIS, IN THIS ORDER

1. **Re-calibrate `C` on the TRAIN cohort.** `C = 68` came from patient007's export, and
   **patient007 is in the SEALED set** — that is a protocol violation and the current
   numbers are provisional because of it. Every pack carries `AP_log1p_nd` at all 201
   timesteps (`src/core_physics/species_fields.py::gt_species_trajectory`), so refit
   `C` (and re-check `q`) on train vessels only. Report per-vessel spread of `C`: if it
   varies wildly, a single global constant is the wrong model.
2. **Test the AP closure properly under the clean protocol** — FIT/DEV/SEALED disjoint,
   selection on DEV, sealed opened once. Metrics: median-over-time deploy score (primary),
   `curve_l1`, `rho`, and the final mask (which must not move — assert it).
3. **Resolve the 4.4× Damköhler ratio** (§1). With AP now varying, refit both equations;
   the fresh-deposition fit's R² of 0.46 suggests its recovered constant is unreliable and
   the discrepancy may partly dissolve.
4. **Only then add the GNN residual on `C_i`** (§4), and check it against a ridge
   regression on the same features before claiming anything for the capacity. Standing
   lesson (§19.2): 187k parameters once bought +0.024 over a logistic regression.
5. **Re-run arm B** (`flow_source="pred"`). The AP closure needs `sr`, which on arm B is
   noisier; the pred arm already costs ~0.05 on the mask and collapses onset ordering
   (`rho` 0.685 → 0.393).

---

## 6. STANDING CONSTRAINTS — violating one invalidates the result

1. **SEALED SET, never train or tune against, spend ONCE:**
   `patient001, 007, 010, 013, 014, 031, 042, 043`.
   Note p007 is sealed *and* is the only vessel with a raw COMSOL export. Using it to
   validate an operator against COMSOL's own fields is defensible; using it to fit a
   constant is not. Say which you did.
2. **Exclude truncated runs (`T < 150`) and zero-GT vessels everywhere.** On a truncated
   run the final map is a different quantity; an empty-GT vessel scores **1.0000** for
   predicting nothing (`empty_gt_fp_tol=8.0`) and measures the tolerance, not the model.
   `scripts/audit_optuna_generalization.py` is the cautionary tale — an earlier sweep
   trained on one vessel, scored that same vessel, and included a free-1.0 vessel.
3. **Selection reads DEV only.** DEV must be non-empty in *both* flow arms — 039/040/041/044
   all lack `u0_pred`, which once silently produced an empty DEV and an untrained "result".
4. **NO CONCURRENT GPU JOBS.** 4 GB card; epoch time went 650 s → 1900 s under contention.
5. **Report seeds.** A single run's sealed number is a peak-pick off a noisy DEV trace.
   Pool >=3 seeds and give a paired CI.

---

## 7. CORRECTIONS TO THE OLDER DOCS — do not re-derive these

* `PHASE3_HANDOFF.md` §1.3 `Sat = 1 - M_tot/Minf` — **wrong**, it is `1 - Mas/Minf` (§1).
* `PHASE3_HANDOFF.md` §1.3 "rp/ap are near-CONSTANT, there is almost no chemistry to
  learn" — **true at t=0 only**. AP develops CV 0.31 and falls to 1.2% of inlet. This is
  the single most consequential error in the old docs; it is why the rollout has no
  depletion feedback.
* `PHASE3_HANDOFF.md` §0a / `WALL_MODEL_PLAN.md` §16.1c "`u_prior`/`v_prior`/`mu_prior`
  are the clot-affected converged solution and are illegal" — **wrong**. They are the GT
  **t=0** fields (corr 1.000 with t=0, 0.05 with t_final), legal under the bandaid. The
  real trap is that `data.x` is static, so a model run with `flow_source="pred"` still
  reads GT t=0 flow out of its own feature vector; rebuild with
  `src/differentiable_wall_model/deploy_features.py`.
* `PHASE3_RESULTS.md` §13.2 graded gate — **retracted**, sign backwards (§3).
* `WALL_MODEL_PLAN.md` pre-2026-08-09 flow-derived measurements — computed with
  `data.G_x`/`G_y`, which have a **median of one non-zero per row** and return `G_x @ x = 0`
  across the interior. Use `src/core_physics/mls_gradient.py` (spearman 0.998 vs COMSOL's
  `spf.sr`, 0.990 vs `d(spf.sr,x)`); `BIOCHEM_GRAD_OPERATOR=legacy` reproduces the old
  numbers bit-for-bit.

---

## 8. TOOLING

| file | what it is |
|---|---|
| `src/core_physics/physics_wall_model.py` | the zero-parameter model; `t0_flow_fields`, `graded_gate` (retracted mode), `integrate_mat_trajectory`, `first_crossing` |
| `src/core_physics/mls_gradient.py` | the derivative operators. **Everything flow-derived depends on these** |
| `src/core_physics/temporal_metrics.py` | `gt_onset_index`, `curve_l1`, `onset_metrics`, `spearman` |
| `src/core_physics/species_fields.py` | `gt_species_trajectory` — AP/RP at all timesteps, the AP-closure training target |
| `scripts/predict_wall_clot.py` | entry point for the shipped physics model |
| `scripts/diag_rederive_surface_ode.py` | §1 — the ODE re-derivation |
| `scripts/diag_onset_sign_test.py` | §2/§3 — the onset-vs-shear sign test |
| `scripts/diag_time_resolved_ceiling.py` | the prize measurement (flash / oracle / ceiling) |
| `scripts/sweep_temporal_only.py` | clean protocol + **parity gate** (refuses to train if the differentiable base is >0.02 below the hard physics — the guard whose absence invalidated a whole round) |
| `scripts/audit_optuna_generalization.py` | how to audit a suspicious result |

Test suite: **603 passing**. `src/tests/test_mls_gradient_and_gates.py` and
`test_temporal_wall_model.py` are the guard files — every assertion corresponds to a way
the wiring could become a silent no-op. Add to them as you build.

---

## 9. KILL CRITERIA

* **`C` varies wildly across train vessels** → a single global AP constant is wrong; either
  condition it on geometry/flow or abandon the algebraic closure for a transport solve.
* **The AP closure moves the final mask** → a bug. It must only change *when*, not *which*.
  Assert it.
* **The GNN residual does not beat ridge on the same features** → do not ship the network.
* **Onset `rho` does not exceed the physics ODE's ~0.69** → the head is a calibration trick
  and not an ordering model; say so plainly rather than reporting the score gain.
* Note the ordering ceiling is genuinely bounded: an oracle with *perfect* time-varying
  flow reaches `rho` 0.795, and with perfect flow **and** GT species 0.866. Do not expect 1.0.
