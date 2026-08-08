# PHASE 3/4 HANDOFF — build a physics-mirroring wall model

Written 2026-08-08 for a new context window.

**The mission: build a new wall-only clot model whose structure mirrors the COMSOL PDE, instead
of a generic network that must rediscover it.** The deposition law is already in this repo,
validated against COMSOL exports to machine precision. Roughly fifteen legs have been spent
teaching a 187k-parameter GraphSAGE to approximate a function we already have exactly.

Repo: `C:\Users\pgssy\thrombus_ml_model` (Windows; PowerShell and Git Bash both available).
History: `docs/WALL_MODEL_PLAN.md` (~4400 lines, every claim cites a `§N`).
**Read §26 first** (the session that produced this), then §10.1, §10.4, §19.2.
Do **not** read §1–§13 linearly — they contain retracted conclusions, cited from later sections
where still valid. §14.5 in particular is RETRACTED by §19.2.

---

## 0. Goal, and where things actually stand

`deploy_clot_score > 0.6` on **unseen vessels**, **wall clot only** (inside the 3-hop wall band).
Floor 0.50.

| | |
|---|---|
| best *deployed* result | **0.6925** — zero-shot warm start `WG_clotrich_nplus` on `patient043`, no cohort training (§9.3) |
| best *in-training* | 0.4889 — Phase 1 epoch 6 |
| fine-tune legs that beat the warm start | **none, out of ~15** |

**Read §6.3 before comparing anything to 0.6925** — it was measured with the flow coupler
working, and every number since 2026-08-06 was measured with it silently disabled.

Phase 0, 1 and 2 of the project ladder are complete. This document replaces Phase 3.

---

## 1. THE MISSION

### 1.1 What the current stack does, and why it cannot generalize

```text
rgp_deq_kine [frozen flow] -> species_graphsage [LEARNS Mat directly] -> gelation -> clot readout
```

The GraphSAGE predicts `dMat` per node per step from a 287-dim feature vector. Its delta head is

```python
pred_delta = spatial_gate * magnitude * autocat_factor
```

Three structural problems, all measured:

1. **It replaces a known law with a learned one.** §10.1's `J0_Mat` is exact and calibrated; the
   network has to rediscover it from data, per geometry.
2. **Its autocatalysis is spatial; the PDE's is local.** `_apply_autocatalytic` aggregates over
   `edge_index`. The PDE's `(Mas/Minf)·k_aa·ap` uses the node's **own** adsorbed mass. Clot does
   not spread neighbour-to-neighbour — each node ignites when its *local* shear gate fires and
   then self-accelerates. Apparent spatial clustering is inherited from the shear field being
   smooth.
3. **It barely beats a linear model.** §19.2: GNN 0.540 vs logreg-on-8-physics-features 0.516,
   and the logreg *wins* on `039` by 0.263 and on `041` by 0.099. 187k parameters, a 256-dim
   frozen latent and a 200-step rollout buy +0.024 mean F1, with per-vessel spread −0.263 to
   +0.174.

Roughly fifteen legs, five clean nulls, and a full session of objective surgery (§26) produced no
improvement. §26.8 is why: **the committed set is fixed by (input representation, per-step
supervision) and is insensitive to every other loss term.** The objective was never the lever.

### 1.2 The law — already here, already validated

`src/core_physics/comsol_surface_deposition.py` is the canonical single source of truth, pinned
by `src/tests/test_comsol_wall_deposition_calibration.py` against COMSOL's own exported `J0_*`
columns (`patient007`, 876 wall nodes × 201 timesteps). **5 tests pass.**
`docs/COMSOL_PHYSICS_VALIDATION.md`: *"Every reaction law, threshold, and scale in the COMSOL
phase-2 thrombosis model has been reconstructed from ground-truth exports and matches the repo's
`BiochemConfig` to machine precision."*

```python
j0_mat_si(cfg=..., sat_m=..., shear_sr=..., dsrx=..., rp=..., ap=..., mas=..., step2t=...)

# J0_Mat = Da * ( [dsrx<sgt]*(L/gamma_m)*|dsrx|*common  +  [sr<lss]*common ) * step2t
# common = Sat*(k_rs*rp + k_as*ap) + (Mas/Minf)*k_aa*ap
```

Constants, all real and calibrated, in `src/config.py`:

| | | |
|---|---|---|
| `lss` | 25.0 1/s | low-shear gate — **79.7% of growing nodes** |
| `sgt` | −7.5e4 1/(m·s) | separation gate — 21% |
| `k_rs` | 3.7e-5 m/s | resting-platelet adhesion |
| `k_as` | 4.5e-4 m/s | activated-platelet adhesion |
| `k_aa` | 4.5e-4 m/s | **autocatalytic** rate on own `Mas` |
| `Minf` | 7.0e10 plt/m² | saturation capacity |

~90% of Mat growth is the autocatalytic term; ~7% fresh deposition. Fibrin is provably inert
(`mu2(fi) ≡ 0`). The clot label *is* the `mu1(Mat)` hard step at `Mat = 2e7 plt/cm²`. So this is
**a gated autocatalytic ignition problem with a threshold readout**, not a deposition-rate
regression.

### 1.3 KNOWN vs LEARNED — the whole point

Every input `j0_mat_si` needs, and where it comes from:

| input | source | learned? |
|---|---|---|
| `sat_m` = 1 − M_tot/Minf | integrated state | no |
| `mas` | integrated state | no |
| `step2t` | activation-phase time gate (`surface_time_gate_scalar`) | no |
| `rp`, `ap` | **near-CONSTANT — see below** | **no** |
| **`shear_sr`, `dsrx`** | **flow field → the gates** | **YES — this is the model** |

**An earlier draft of this handoff said the learned part was `rp`/`ap` at the wall. That was
wrong, and §26.16 measured it before anything was built.** Spatial coefficient of variation
across wall-band nodes:

```
RP   0.003     <- flat to 0.3%          Mas  3.934
AP   0.095     <- ~10%                  Mat  4.399   <- 440%
```

Cross-vessel, `RP@wall` spans 0.9978e-6 … 1.0e-6 — a **0.2% spread over the entire cohort**. Both
platelet concentrations sit at essentially their inlet values everywhere, at every time sampled.
**There is almost no chemistry to learn.** With `RP0`, `AP0` constant the law collapses to

```
J0_Mat ~= Da · gates(flow) · [ Sat·(k_rs·RP0 + k_as·AP0) + (Mas/Minf)·k_aa·AP0 ]
```

so every spatial and temporal feature of `Mat` enters through **the gates** (flow-derived) and
**`Mas`/`Sat`** (integrated state). **The learnable content of this problem is the FLOW FIELD.**
That makes the corrector (§6.3) the central component, not a supporting one.

### 1.4 Do the gates actually discriminate on our data? Measured: yes — on GT flow

`scripts/diag_physics_gate_support.py`, 35 vessels, wall+3hop band, gates at t=0 (§26.17):

```
                                      mean AUC   mean |AUC-0.5|
low-shear gate  [sr < lss=25 1/s]       0.510        0.136
separation gate [d(sr,x) < sgt]         0.659        0.167
best-of-two per vessel                     --        0.203
```

**The "dominant" 79.7% mechanism averages to chance because it is bimodal, not weak.** 14 vessels
have low shear predicting MORE clot (mean AUC 0.679), 14 predicting LESS (0.347), 7 ambiguous.

§10.4's mechanism is confirmed quantitatively:

```
rho(band_speed_q25, AUC low-shear gate)  = -0.413      (10.4 predicts negative)
rho(band_speed_q25, AUC separation gate) = +0.607      (10.4 predicts positive)

group                    n   band_speed_q25   AUC low-shear   AUC separation
low shear -> MORE clot  14       0.0428            0.679           0.589
INVERTED                14       0.0874            0.347           0.714
```

Inverted vessels are **2× faster**. Slow → real stagnation zone → low-shear gate carries it.
Fast → no stagnation → deposition falls to the shear-*gradient* mechanism. So **the law's
two-gate sum may reproduce the cohort's bimodality for free**, since the separation term scales
with `|dsrx|`, which is largest exactly on the fast vessels. §10.4 separately measured the regime
itself routable from `band_speed_q25` at AUC 0.975 / **90.6% LOO**.

### 1.5 STEP 0 — the one measurement that decides the project

**Every AUC in §1.4 was computed from GROUND-TRUTH velocity** (`gamma_si` derives from
`y[:,0], y[:,1]`). Z1 measured the *predicted* flow field's marginal contribution to clot ranking
at **0.041 AUC**.

> **How much gate discrimination survives when the gates are computed from PREDICTED flow?**

* **Survives** → the physics plan works, and the learned component is the flow correction.
* **Collapses** → the flow surrogate is the entire problem. No chemistry work, no architecture
  work, and no objective work helps, and improving flow becomes the project.

It is cheap: recompute `gamma_si` and `dshear_ds` from the corrector/kinematics flow instead of
from `y`, rerun `scripts/diag_physics_gate_support.py`, and compare the two AUC columns
vessel-by-vessel.

**This supersedes the earlier Step 0** ("does the law reproduce GT `dMat` from GT species?"),
which is a weaker question now that §26.16 shows the species are constants. Note also that the
unit risk that earlier draft warned about does not exist — `gamma_si` is already SI (range
~0.006–1264 1/s, so `lss=25` sits well inside it), `dshear_ds` is already in the gate's units,
and `is_low_shear` is already computed in the t=0 feature table.

**If the gates hold up on predicted flow, then run the law end-to-end** against GT `Mat`
trajectories as a second check before building anything.

### 1.6 Kill criteria

* **Step 0 fails** (law + GT species does not reproduce GT `dMat`) → the premise is wrong; stop
  and report rather than tuning around it.
* **Gate discrimination collapses on predicted flow** (§1.5) → the flow surrogate is the whole
  problem; stop and make flow the project.
* **Whole model** against §19.2's bar: logreg **0.516**, GNN **0.540**. And against the standing
  0.6925 once §6.3 makes that comparison legitimate.

---

## 2. BUILD ORDER

0. **Resolve the metric split** (§6.1a). CPU, and it decides whether any epoch selection to date
   is trustworthy.
1. **Step 0** (§1.5). CPU. Everything depends on it.
2. **Confirm the new corrector loads** (§6.3) — 3-output `[dU, dV, dShear]`, no WARN. It supplies
   `shear_sr`/`dsrx`, so the law is only as good as it is.
3. **Flow → gates.** The learned component is the flow correction, not chemistry (§1.3). Feed the
   corrector's `u,v` (and its shear head) into `gamma_si` / `dshear_ds`, and measure the gate AUCs
   against the GT-flow numbers in §1.4. Improving that gap IS the model.
4. **Assemble**: flow → gates → `j0_mat_si` with `rp`/`ap` held at their measured constants →
   integrate `Mat` → `mu1(Mat)` step → clot readout. Compare against GT `Mat` trajectories.
   Treat `rp`/`ap` as constants first; only make them learned if the residual demands it.
5. **Then** single-variable legs from that baseline. The assembly in (4) is a **re-baseline**, not
   an A/B — label it as such in the log, as Phase 1 was.

---

## 3. OPTIONAL COMPARISON ARM — the previous Phase 3 (additive C1)

Retained deliberately as a comparison, not the main line.

```
dMat = NUCLEATION(field) + GROWTH(local committed Mat) · gate(shear)
```

* **Motivation:** the multiplicative form cannot express nucleation — a node with no committed
  neighbour and no Mat can only emit ~0.
* **Premise size, corrected (§26.13.2):** genuine nucleation is **~21% of commits** under a 2-hop
  growth rule (18.9% at 3-hop), **not** the 40.3% a strict 1-hop rule gives or §14.6's 27–58%.
  Under 5% on some vessels. Measure with `scripts/diag_nucleation_census.py --growth-hops 2`.
* **Head target:** `t=20` seeds (§20.4 — AUC 0.903 vs 0.806 for the final map). This is sound;
  §26.13's claim that it needed time conditioning was retracted by §26.13.2.
* **Already built and switched off:** `GROWTH` = `_apply_autocatalytic`
  (`autocatalytic_growth=False`); `gate(shear)` = `shear_readout_gate` (`False`). Only the
  nucleation head would be new.
* **Why it is not the main line:** its `GROWTH` term is neighbour-based where the PDE is local,
  and its additive split does not match a PDE that is multiplicative — gate × saturation ×
  (deposition + autocatalysis). It is a reasonable approximation of the right idea; §1 is the
  right idea.

If §1 stalls, this is the fallback, and the two are directly comparable on the same cohort.

---

## 4. PHASE 4 — after the physics model lands

* **Species as a residual.** §26.16 shows `rp`/`ap` are constants to 0.3%/10%. If the assembled
  model has structured residuals that the gates cannot explain, revisit whether the ~10% `AP`
  variation matters. Do not start here — start by assuming they are constants.
* **D3 — `z_kin` shrink.** Gated on a **shuffle** test, not the zero-ablation already run (§6.4).
* **C2 — per-vessel conditioning.** §19.2's spread (−0.263 to +0.174) is the strongest argument
  on record. It should largely dissolve if §1.4 is right — check whether it does.
* **B3 — longer windows.** One clean re-test at `unroll` 25–50, `curriculum_unroll=False`.
* **Re-baseline `patient043`'s 0.6925 with the coupler ON** (§6.3). Required before any deploy
  claim.

---

## 5. STANDING CONSTRAINTS — violating one invalidates the result, not just the run

1. **SEALED SET — never train on, never tune against, spend ONCE:**
   `patient001, 007, 010, 013, 014, 031, 042, 043`.
   Train cohort (26): `WALL_COHORT_V2_TRAIN`. Val anchor: `patient041` (dev, not sealed).
   `scripts/go_phase1_baseline.ps1` enforces this and refuses a sealed `-ValAnchor`.
2. **ONE VARIABLE PER LEG.** v1–v6 were lost to bundling (§9.12, §9.14). The exception is a
   deliberate **re-baseline** — several things change at once, it becomes a new reference point,
   nothing is attributable across it, and it must be labelled as such in the log.
3. **VERIFY THE MECHANISM ENGAGED before trusting any result.** The dominant failure mode here.
   **Thirteen dead constants/mechanisms are on record.** A config value appearing in the
   fingerprint is *not* sufficient — §6 has two that printed correctly and did nothing.
4. **EPOCH BUDGET AND THE PLATEAU RULE.** ~21–25 min/epoch on cohort v2. **Epochs 1–3 are a
   low-sensitivity plateau**: five legs with materially different objectives produced a
   bit-identical epoch-2 committed set (§26.8, §26.11). Both Phase 1 and the z_kin leg leave it
   at epoch 4. **An A/B read on `deploy_clot_score` inside epochs 1–3 measures nothing.** Run
   past epoch 4, or read `deploy_mat_f1`, which never entered the plateau (§6.2).
5. **NO CONCURRENT GPU JOBS.** Epoch time went 650s → 1900s under contention. 4 GB card.
6. **Two lines across which numbers are not comparable:** the Phase-1 re-baseline (cohort,
   priors, labels all changed at once), and **2026-08-06**, when the flow coupler silently
   switched off (§6.3).

---

## 6. THE INSTRUMENT — what was broken, what is fixed, what you must not trust

Thirteen dead mechanisms are on record. These five were found in the last session and change how
every number in `WALL_MODEL_PLAN.md` should be read.

### 6.1 `loss_scale` divided every rolled term by 10 (§26.4) — fixed, off by default, and it does not help

`loss_scale=0.1` multiplies every rolled term at its own site. It IS applied to the per-step loss
in `continuous_delta_loss` — but that is the **single-head** path, and every cohort leg runs
`dual_head=True`, where `dual_head_step_loss` never picked it up. `rolled_soft_f1_weight=120` was
sized in §12.6.6 as "8.2× the noise floor"; the realised multiplier was **0.82×**.

Wired as `loss_scale_unified`, default **off**. **Tested over 6 epochs (§26.11): it does not
help.** Better at epochs 1 and 3, bit-identical at 2, worse at 4–6; best 0.4424 against Phase 1's
0.4889. It drives the mass collapse *faster*. A real bug fix that produces a worse model — which
is itself evidence for the pivot in §1.

### 6.1a THE IN-TRAINING METRIC AND THE CANONICAL EVAL DISAGREE IN DIRECTION — unresolved

**Read this before trusting any "best epoch" in §21–§26.** On an *identical committed set* —
same `mass`, same `fp` — the in-training deploy score and the canonical cold eval differ, and not
by a constant:

| leg | sel ep | in-training | cold eval | delta |
|---|---|---|---|---|
| Phase 1 | 4 | 0.4593 | 0.4319 | **−0.027** |
| T5 | 4 | 0.4424 | **0.5103** | **+0.068** |

They move in **opposite directions and reorder the legs**. By the in-training metric Phase 1
beats T5; by the canonical eval T5 wins by +0.078 and clears the 0.50 floor.

Scoring *parameters* are identical in both fingerprints (`clout_score_mode=guiding`,
`clout_prec_rec_floor=0.3`, `guide_relax_hops=2`, `guide_f_beta=0.5`, `empty_gt_fp_tol=8.0`);
only `runtime_bound` differs and that is a diagnostic flag, not a scoring parameter. So this is
**not** the parameter drift §20.1 fixed.

**Blast radius:** in-training selection decides which checkpoint every leg keeps. If it disagrees
in direction with the canonical eval, epoch selection across this project may have kept the wrong
checkpoints, and every "best epoch" number in §21–§26 inherits that.

**Cheap to resolve relative to the damage:** instrument both paths on one checkpoint with one
committed set and diff the intermediates — tp/fp/fn before relaxation, the dilation, the recall
floor, the guiding blend. Do this early; it is the difference between measuring and guessing.

### 6.2 `deploy_clot_score` has a resolution floor; `deploy_mat_f1` does not

At epoch 2, four materially different objectives gave **bit-identical** `deploy_clot_score`
(0.3706237861441937) while differing on `deploy_mat_f1`. `deploy_mat_f1` is already logged, is a
genuine deploy metric (not teacher-forced), already feeds the selection score, and has **10/10
distinct values** across Phase 1's ten epochs. Use it as the primary A/B discrimination metric —
it makes legs ~3× cheaper. `deploy_clot_score` stays the *goal*; `deploy_mat_f1` is the
*instrument*.

Also: `mass`, `fp` and `fn` are **not** sufficient to characterise a committed set. At epoch 3,
T5 and Phase 1 matched on all three and on recall, yet differed by **+0.027** score because T5's
201 false positives were better *placed* (relaxed precision dilates by 2 hops). Any
committed-set comparison must include `relaxed_prec`.

Related: `val_growth_mat_f1` is identically **0.0** across all ten Phase-1 epochs — a dead metric
that still carries weight 0.05 in the `mat_only` selection branch and 0.10 in `physics_gt_fed`.

### 6.3 The flow coupler was OFF from 2026-08-06 — check the new corrector

Commit `9eba0db` widened `LocalKinematicCorrector.readout[-1]` to 3 outputs `[dU, dV, dShear]`;
the saved checkpoint still had 2. `load_local_corrector` loads strictly, raises, and **both call
sites catch it, print a WARN, and continue with `coupler = None`** — falling through to uncoupled
flow. Gated on `flow_source == "auto"`, which every cohort leg uses.

**`patient043`'s 0.6925 (artifact dated 2026-08-05) was measured with coupling working; every
Phase 1/2/3 number was measured with it off.**

The user was training a replacement 3-output corrector *with a shear head* on the evening of
2026-08-08. Check `outputs/kinematics/local_corrector/` for `readout.2.weight` of shape `(3,64)`,
and **confirm the coupler initialises with no WARN before trusting any deploy number.** That
corrector also supplies `shear_sr`/`dsrx` to the law in §1.2 — it is now load-bearing for the
model, not merely for the metric.

### 6.4 `latent_ablate` was a no-op in the canonical eval (§26.10) — fixed

`maybe_drop_latent` gates on `model.kin_latent_dim`, bound only inside the *training* process.
The eval bundle loader never set it, so at eval `ld = 0` and a leg fine-tuned on zeroed `z_kin`
was **scored on intact `z_kin`**. Fixed; scoped so no other leg's numbers move.

Caught by a cross-check worth keeping: **a leg's cold eval should reproduce its in-training
`mass`, `fp` and `relaxed_prec` exactly.** Leg A did; the z_kin leg was off by 21× in mass, which
is what made the bug legible. Free, and it caught this.

Consequence for D3: the zero-ablation result (−0.182 score at ep1) is **confounded** — 80–91% of
first-layer weight norm sits on the `z_kin` block, but per-dim weighting is 0.49–1.26× the
non-latent features, so the share is a pure *width* effect (256 dims vs 31). Zeroing destroys
~85% of input drive regardless of information content. **Z1 measured information; the ablation
measured perturbation magnitude.** The clean test is the one Z1 specified and nobody ran:
**shuffle `z_kin` across vessels**.

### 6.5 Five live loss terms were in no decomposition (§26.3)

Under the cohort runtime — not the dataclass defaults — `phi_loss_weight = 20.0` with
`physics_readout=True`. A weight-20 per-step term was hidden inside `per_step_block`; a weight-10
final phi term and a weight-4 speed-FP bleed were in **no recorded term at all**. All five are
now recorded. **Verify term shares by removal, never by reading the recorded number.**

---

## 7. CLOSED — do not reopen

* **Objective reweighting of the current model** (§26.8, §26.11). The committed set is fixed by
  (input representation, per-step block); everything else in that loss is decoration. Five legs
  confirm it, including a 10× rescaling of the rolled terms.
* **The brake / mass in the loss** (§25.5). Mass is a selection guard only.
* **`fp_weight`** (§26.2) — unreachable. Over a full epoch it selected **0 nodes**, max predicted
  raw delta 7.76e-06 against a 2e-5 threshold. Do not wire it into anything.
* **T3, unfreeze the backbone** — premise refuted (§26.8: two objectives under head-only training
  ended 128% apart in weight space), and moot under fresh init.
* **§14.5's "a linear model matches the GNN"** — RETRACTED by §19.2.
* **§26.13's "early and late nucleation are different places"** — RETRACTED by §26.13.2. Late
  "nucleation" is 2-hop growth: every late-quartile site in six vessels sat within 2 hops of
  existing clot, median exactly 2.
* **Regime routing as an open problem** (§13.4 — all six cohort vessels are normal-regime), and
  **shear decoding** (Z4, ±0.001).

---

## 8. TOOLING

| tool | use |
|---|---|
| `scripts/go_phase1_baseline.ps1` | leg launcher. Seal guard, cohort-v2 resolution, junk-list and min-size checks. **Not** `go_wg_stenosis_subcohort_ft.ps1` — it encodes a 5-vessel study's invariants. |
| `scripts/eval_mat_growth_simple.py` | canonical eval; prints `SCORING FINGERPRINT` from inside the scoring scope |
| `scripts/diag_leg_alignment.py` | Spearman + exact permutation p + jackknife + z-separation + distinct-state count; learning-curve stop/extend verdict |
| `scripts/diag_ckpt_weight_geometry.py` | weight-space geometry between checkpoints; **warns on epoch-mismatched comparisons**, which that measurement needs |
| `scripts/diag_nucleation_census.py` | growth-vs-nucleation census + time-quartile profile. Use `--growth-hops 2`. Its `seed_reach` column is degenerate — ignore it. |
| `scripts/diag_nucleation_timing_probe.py` | Q1-vs-Q4 site comparison with a permutation null |
| `src/core_physics/comsol_surface_deposition.py` | **the canonical `J0_Mat` law** |
| `src/core_physics/biochem_physics_kernels.py` | `biochem_wall_residual` — the law wired end-to-end, incl. both gates and saturation |
| `docs/COMSOL_PHYSICS_VALIDATION.md` | what was validated, against which exports |

Retention: `best_salvage.pth` keeps the top-scoring epoch even if selection rejects every one,
and is promoted with a loud warning. No leg can silently produce nothing.

Test suite: **585 passing**. `src/tests/test_vessel_scope_and_phase1.py` is the guard file —
every assertion corresponds to a specific way wiring could become a silent no-op. Add to it as
you build; that file is why the last session caught four dead mechanisms.

---

## 9. OPEN QUESTIONS

0. **Why do the in-training and canonical metrics disagree in direction?** §6.1a. It decides
   whether any "best epoch" on record is trustworthy, and it is cheap to answer. Do it alongside
   Step 0.
1. **Does the law reproduce GT `dMat` on the graph packs?** §1.5. Everything depends on it.
2. **Is the flow good enough?** The law consumes `shear_sr` and `dsrx` directly. Z1 scored the
   flow channel at 0.041 AUC on clot-ranking. If Step 0 passes and the assembled model still
   underperforms, this is the answer, and the flow surrogate becomes the project.
3. **Does `z_kin` carry vessel-specific information?** Shuffle test, §6.4. Gates D3.
4. **The goal split** (§20.5): `039`–`044` scores mean F1 0.516 where a random draw of 9 scores
   0.322. Be explicit about which set any ">0.6" claim is on.
