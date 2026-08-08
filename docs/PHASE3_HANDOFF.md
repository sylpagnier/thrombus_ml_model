# PHASE 3 HANDOFF — build C1: a fresh two-term model

Written 2026-08-08 for a new context window. You are picking up a project that has spent
~20 sections and ~15 GPU legs discovering, mostly the hard way, that its current architecture
cannot express the thing it is being asked to predict. **Your job is not to fix that model. It
is to build a new one.**

Repo: `C:\Users\pgssy\thrombus_ml_model` (Windows; PowerShell and Git Bash both available).
Long-form history: `docs/WALL_MODEL_PLAN.md`, 4300 lines, every claim cites a `§N`.
**Read §26 (this session), then §14.6, §19.2, §20.4.** Do not read §1–§13 linearly; they contain
retracted conclusions and are cited from later sections where still valid.

---

## 0. The goal

`deploy_clot_score > 0.6` on **unseen vessels**, **wall clot only** (inside the 3-hop wall band).
Stretch target from §0; the floor is 0.50.

Best *deployed* result on record is still the **zero-shot warm start** — `WG_clotrich_nplus`
applied to `patient043` with no cohort training at all: **0.6925** (§9.3). Thirteen fine-tune
legs have all been worse. That is the number to beat, with one caveat you must read in §5.3
before comparing anything to it.

---

## 1. THE MISSION — Phase 3, and why it is a fresh build

### 1.1 What is wrong with the current model, mechanically

The delta head is **purely multiplicative**:

```
pred_delta = spatial_gate * magnitude * autocat_factor
```

Every factor scales a growth signal. A node with no committed neighbour and no existing Mat can
only produce ~0. But §14.6 decomposed every commit event into growth (had a committed neighbour)
vs nucleation (did not):

| vessel | nucleation % |
|---|---|
| `039` | 51.1% |
| `040` | 47.2% |
| `041` | 30.1% |
| `042` | 26.7% |
| `043` | 58.1% |
| `044` | 31.4% |

**27–58% of all commits are nucleation, continuing into the final time quartile** — not an
initial transient. The architecture structurally cannot express them. §14.6's conclusion:
*"A seed-and-grow model has a ceiling near F1 0.58 on this cohort. That ceiling is where ten
fine-tuning legs have been stuck."*

### 1.2 Build C1

```
dMat = NUCLEATION(static field)  +  GROWTH(local committed Mat) · gate(shear)
```

The change that matters is **`+`**. Nucleation is an additive, neighbour-independent rate.

**Half of it already exists and is switched off:**

| term | code | state in `WG_phase3a_closedloop` |
|---|---|---|
| `GROWTH` | `_apply_autocatalytic` (change D) | `autocatalytic_growth = False` — **off** |
| `gate(shear)` | `shear_readout_gate` | `False` — **off** |
| `NUCLEATION` | — | **does not exist. This is the build.** |

The masks that would suppress nucleation are already off and must stay off:
`frontier_hops=0`, `nucleation_topk=0.0`, `dynamic_frontier_mask=False`.

**NUCLEATION design constraints:**

* Reads **only deploy-legal static field features**: `-anaSpd`, `-sdf`, `wgrad`,
  `shear_potential`, `curv` (§16.3's survivors). **NOT `mu_eff`** — it is anti-predictive on
  19 of 35 vessels (§16.3), and the Tier-C spec in §14.x that lists it predates that finding.
* **Must not see neighbouring clot.** If it reads the fused GNN hidden state it inherits
  neighbour aggregation and stops being nucleation. That is the whole point.
* Consequence: the nucleation rate is a **static per-node field**, computable once per vessel
  rather than per rollout step. Big efficiency win, and it is what "independent of neighbouring
  clot" actually means.

### 1.3 Why fresh (random init), not another fine-tune

| evidence | implication |
|---|---|
| 13 warm-started legs all **degraded** the warm start | fine-tuning on this cohort is actively destructive |
| additive nucleation changes the function class | warm weights are calibrated for the multiplicative form |
| §13.8 warm-start blocker on any `in_dim` change | a shrink forces fresh anyway |
| 17 existing legs already use `no_init=True` | the from-scratch path works |

Complexity bar is low, which favours a smaller well-structured model: §19.2 (the corrected
version — §14.5 is **RETRACTED**, do not cite it) has the 187k-param GNN beating a logistic
regression on 8 deploy-legal features by only **+0.024** mean F1 (0.540 vs 0.516), and *losing*
on `039` by **−0.263** and on `041` by −0.099. Per-vessel spread −0.263 to +0.174.

Suggestive support for C1's premise, computed this session: ρ(nucleation %, GNN-minus-logreg)
= **−0.429** (n=6 — directional only, `043` breaks it). The GNN tends to lose where nucleation
dominates.

### 1.4 Training target and kill criterion

**Target (§20.4, item 0f).** Same deploy-legal features, two targets, 35 vessels:

```
final map    n=35   mean best-feature AUC 0.806   sd 0.085
t=20 seeds   n=32   mean best-feature AUC 0.903   sd 0.032
```

Seeds are **+0.097 AUC more predictable and 2.7x more consistent across vessels**. §20.4:
*"train the nucleation head on early commits, and let the growth term carry it forward."*
That is settled — do not re-litigate it.

**Kill criterion, corrected.** The Tier-C spec says "if the nucleation head cannot beat the
logreg of §14.5" — **§14.5 is retracted**, so that bar is a dead number. Use instead:

* nucleation head vs a logreg predicting **t=20 seeds** on the same deploy-legal features
  (the 0.903 AUC benchmark);
* whole model vs §19.2's **0.516** logreg / **0.540** GNN.

If the nucleation head cannot beat the seed logreg, the field model is saturated and the
remaining error is temporal/calibration, not spatial.

### 1.5 Loss design — this is where the last session's findings actually pay off

**Fold C3 (ranking) into C1 rather than after it.** The ladder said "C3 only if C1 lands", but
nucleation is *inherently* a ranking problem — "which nodes seed?" — and §20.4 hands it a target
at AUC 0.903. Training it with a per-step Huber would repeat the exact category error that made
the current model growth-only. So:

* **nucleation head → ranking / AUC-surrogate loss on t=20 seeds**
* **growth term → keeps per-step supervision.** It is the only thing ever measured to move this
  model (§26.8), so do not remove it.
* **Do NOT wire `fp_weight`.** Its FP branch is unreachable: over a full epoch of real training
  it selected **0 nodes**, with max predicted raw delta **7.76e-06** against a `2e-5` threshold
  (§26.2). It has been multiplying an empty set for the entire project.
* **No mass term in the loss.** `deploy_clot_score` is relaxed *precision* gated by a recall
  floor; a mass target optimises what the metric does not reward. Mass survives as a
  **selection guard only** (§25.5).
* **Put every term on ONE scale at the point of summation, from the start.** See §5.1 — this
  bug cost the project five null results.
* **Select and discriminate on `deploy_mat_f1`, not `deploy_clot_score` alone.** See §5.2.

---

## 2. STANDING CONSTRAINTS — violating one invalidates the result, not just the run

1. **SEALED SET — never train on, never tune against, spend ONCE:**
   `patient001, 007, 010, 013, 014, 031, 042, 043`.
   Train cohort (26): `WALL_COHORT_V2_TRAIN`. Val anchor: `patient041` (dev, not sealed).
   `scripts/go_phase1_baseline.ps1` enforces this and refuses a sealed `-ValAnchor`.
   Verified disjoint this session.

2. **ONE VARIABLE PER LEG.** v1–v6 were lost to bundling (§9.12, §9.14). The single exception is
   a deliberate **re-baseline** (Phase 1 was one; C1 will be another) — several things change at
   once, it establishes a new reference point, nothing is attributable across it, and it must be
   labelled as such in the log so nobody later reads it as an experiment.

3. **VERIFY THE MECHANISM ENGAGED before trusting any result.** This project's dominant failure
   mode. **Thirteen dead constants/mechanisms are on record.** A config value appearing in the
   fingerprint is *not* sufficient — see §5 for two that printed correctly and did nothing.

4. **EPOCH BUDGET — and the plateau rule.** ~21–25 min/epoch on cohort v2. Epochs 1–3 are a
   **low-sensitivity plateau**: the committed set is fixed by (input representation, per-step
   supervision) and is insensitive to other loss terms. Three materially different objectives
   produced a **bit-identical** epoch-2 committed set (§26.8). Both Phase 1 and the z_kin leg
   leave that plateau at **epoch 4**. So: **an A/B read inside epochs 1–3 on `deploy_clot_score`
   measures nothing.** Either run past epoch 4, or read `deploy_mat_f1`, which never entered the
   plateau (§5.2).

5. **NO CONCURRENT GPU JOBS.** Measured epoch time went 650s → 1900s under contention. 4 GB card.

6. **Numbers are not comparable across two lines:** the Phase-1 re-baseline (cohort, priors,
   labels all changed) and **2026-08-06**, when the flow coupler silently switched off (§5.3).

---

## 3. WHAT IS CLOSED — do not reopen

* **The brake / mass in the loss** (§25.5). Dead → revived → still inert; removing it entirely
  moved the rollout ~1%. Mass is a selection guard only.
* **Objective reweighting of the current model** (§26.8). The committed set is determined by
  (input representation, per-step block); everything else in that loss is decoration. This is
  the door the last session closed, and it is why you are building C1 instead.
* **`fp_weight`** (§26.2) — unreachable, proven by direct count.
* **Regime routing** (§13.4 — all six cohort vessels are normal-regime).
* **Shear decoding** (Z4, ±0.001).
* **§14.5's "a linear model matches the GNN"** — RETRACTED by §19.2.
* **T3 (unfreeze the backbone)** — its premise was that head-only training is too narrow for
  objectives to differ. Refuted: two objectives under exactly that parameterisation ended
  **128%** apart in weight space (§26.8). Moot anyway once you train fresh.

---

## 4. PHASE 4 — after C1 lands

* **D3 — `z_kin` shrink.** Gated on a **shuffle test**, NOT the zero-ablation already run. See
  §5.4: the zero-ablation result is confounded and the shuffle test is the one Z1 originally
  specified.
* **C2 — per-vessel conditioning.** §19.2's per-vessel spread (−0.263 to +0.174) is the
  strongest argument for it on record. Feed a small per-vessel descriptor (`band_speed_q25`,
  median WSS, geometry class) as a global conditioning vector.
* **B3 — longer windows.** One clean re-test at `unroll` 25–50, `curriculum_unroll=False`.
  v6 tested this and found nothing, but v6 ran with the dead loss terms of §12.6.
* **Re-baseline `patient043`'s 0.6925 with the flow coupler ON** — required before any deploy
  claim (§5.3).

---

## 5. THE INSTRUMENT — what was broken, what is fixed, what you must not trust

Thirteen dead mechanisms are on record. These five were found in the last session and they
change how you read every number in `docs/WALL_MODEL_PLAN.md`.

### 5.1 `loss_scale` divided every rolled term by 10 (§26.4) — FIXED, off by default

`loss_scale=0.1` multiplies every rolled term at its own site. It *is* applied to the per-step
loss in `continuous_delta_loss` — but that is the **single-head** path, and every cohort leg runs
`dual_head=True`, where `dual_head_step_loss` never picked it up. So every weight ever set on a
rolled term was implicitly **÷10** against the only term that moves the model.
`rolled_soft_f1_weight=120` was sized in §12.6.6 as "8.2× the noise floor"; the realised
multiplier was **0.82×**. That sizing is invalid.

Wired as `loss_scale_unified` (default **off** = historical behaviour). **For C1, put all terms
on one scale from the start and never recreate this.**

Measured: turning it on moved epoch 1 from 0.4056 → **0.4334** score, fp 244 → **215**, mass
3.080 → **2.823** — *inside* the plateau where other interventions were bit-identical. A global
term does steer once it is on equal footing. This de-risks §1.5's ranking-loss design.

### 5.2 `deploy_clot_score` has a resolution floor; `deploy_mat_f1` does not

At epoch 2, three materially different objectives gave **bit-identical** `deploy_clot_score`
(0.3706237861441937) — and **all three differed** on `deploy_mat_f1`
(0.32247 / 0.32309 / 0.32301). The species is moving; the gelation threshold quantises it away.

`deploy_mat_f1` is already logged, is a genuine deploy metric (not teacher-forced), already
feeds the selection score, and has **10/10 distinct values** across Phase 1's ten epochs where
the clot score plateaus. It tracks the target at ρ = +0.552.

**Use it as the primary A/B discrimination metric.** It makes legs ~3× cheaper — you can read a
result at epoch 1–2 instead of needing 6 to clear the plateau. `deploy_clot_score` remains the
*goal*; `deploy_mat_f1` is the *instrument*.

Related: `val_growth_mat_f1` is identically **0.0** across all ten Phase-1 epochs — a dead
metric that still carries weight 0.05 in the `mat_only` selection branch and 0.10 in
`physics_gt_fed`. Fix or drop it before it silently biases C1's selection.

### 5.3 The flow coupler has been OFF since 2026-08-06 — user is fixing upstream

Commit `9eba0db` widened `LocalKinematicCorrector.readout[-1]` to 3 outputs `[dU, dV, dShear]`;
the saved corrector checkpoint still had 2. `load_local_corrector` loads strictly, raises, and
**both call sites catch the exception, print a WARN, and continue with `coupler = None`** —
falling through to uncoupled flow. Gated on `flow_source == "auto"`, which every cohort leg uses.

**Consequence:** `patient043`'s **0.6925** benchmark (artifact dated 2026-08-05) was measured
with coupling **working**; every Phase 1/2/3 number was measured with it **off**. The standing
headline "thirteen legs never beat the zero-shot warm start" spans an unrecorded deploy-path
change.

**Status:** the user was training a new 3-output corrector *with a shear head* on the evening of
2026-08-08. Check `outputs/kinematics/local_corrector/` for a checkpoint whose
`readout.2.weight` is `(3, 64)`. That corrector is also what makes C1's `gate(shear)` meaningful
— it supplies a real predicted shear field. **Confirm the coupler initialises without the WARN
before trusting any C1 deploy number.**

### 5.4 `latent_ablate` was a no-op in the canonical eval (§26.10) — FIXED

`maybe_drop_latent` gates on `model.kin_latent_dim`, which was bound only inside the *training*
process. The eval bundle loader never set it, so at eval `ld = 0`, the guard failed, and a leg
fine-tuned on zeroed `z_kin` was **scored on intact `z_kin`**. Fixed; scoped so no other leg's
numbers move.

Caught by a cross-check worth institutionalising: **a leg's cold eval should reproduce its
in-training `mass` and `fp` exactly.** Leg A did (2.7434 / 206 both); the z_kin leg was off by
21× in mass, which is what made the bug legible. Run this check on every C1 leg — it is free.

### 5.5 Five live loss terms were never in any decomposition (§26.3)

Under the cohort runtime — **not** the dataclass defaults — `phi_loss_weight = 20.0` with
`physics_readout=True`. So a weight-20 per-step term was hidden inside `per_step_block`, and a
weight-10 final phi term plus a weight-4 speed-FP bleed were added *after* all recording, i.e.
in no recorded term at all. All five are now recorded (`step_phi`, `final_phi`, `step_mu`,
`final_mu`, `speed_fp_bleed`).

This is a second, independent reason §23.7's per-term shares cannot be read directly, on top of
the §24.2 denominator bug. **Verify term shares by removal, never by reading the recorded
number.**

---

## 6. TOOLING

| tool | use |
|---|---|
| `scripts/go_phase1_baseline.ps1` | leg launcher. Seal guard, cohort-v2 resolution, junk-list and min-size checks. **Do not** use `go_wg_stenosis_subcohort_ft.ps1` — it encodes a 5-vessel study's invariants. |
| `scripts/eval_mat_growth_simple.py` | canonical eval; prints `SCORING FINGERPRINT` from inside the scoring scope |
| `scripts/diag_leg_alignment.py --logs <train_log.jsonl> --per-epoch` | Spearman + exact permutation p + jackknife + z-separation + distinct-state count; learning-curve stop/extend verdict |
| `scripts/diag_ckpt_weight_geometry.py` | weight-space geometry between checkpoints sharing a warm start; **warns on epoch-mismatched comparisons**, which that measurement needs |
| `src/core_physics/vessel_scope.py` | per-vessel priors + label scale, one primitive |

Retention: `best_salvage.pth` keeps the top-scoring epoch even if selection rejects every one,
and is promoted to `best.pth` with a loud warning. No leg can silently produce nothing.

Test suite: **583 passing**. `src/tests/test_vessel_scope_and_phase1.py` is the guard file —
every assertion there corresponds to a specific way wiring could become a silent no-op. Add to
it as you build C1; that file is why the last session caught four dead mechanisms.

---

## 7. OPEN QUESTIONS worth your attention

1. **The Z1 paradox.** Z1 measured the entire flow channel at **0.041 AUC** (GT field 0.789 vs
   zero-prior 0.748), yet zeroing `z_kin` costs 45% of the deploy score. Resolution found last
   session: **80–91% of first-layer squared weight norm sits on the `z_kin` block, but the
   per-dim weighting is 0.49–1.26× the non-latent features** — the share is a pure *width*
   effect (256 dims vs 31). So zeroing destroys ~85% of input drive regardless of information
   content. Z1 measured *information*; the ablation measured *perturbation magnitude*. They were
   never in conflict.
   **The clean test is the one Z1 actually specified and nobody ran: shuffle `z_kin` across
   vessels.** Preserves the marginal distribution and input-drive magnitude exactly, destroys
   only the vessel-specific correspondence. Shuffled ≈ intact → the latent carries no
   vessel-specific information and D3's shrink is free. This gates Phase 4's D3.

2. **The goal split (§20.5).** `039`–`044` scores mean F1 0.516 where a random draw of 9 scores
   0.322. The cohort is materially easier than a random draw. Cohort v2's sealed 8 addresses
   this, but when you claim ">0.6", be explicit about which set it is on.

3. **Nucleation continues into the last time quartile** (§14.6). A purely static nucleation rate
   may be right, but check whether the rate needs a slow time modulation before concluding the
   head underperforms.

---

## 8. IMMEDIATE NEXT ACTIONS

1. Confirm the new 3-output corrector exists and the coupler initialises **without** the WARN
   (§5.3). Until then, no deploy number is comparable to 0.6925.
2. Check `outputs/biochem/eda/t5/WG_t5_unified_scale/train_log.jsonl`. A 6-epoch run of the
   `loss_scale_unified` leg was in flight when this handoff was written; epoch 1 gave 0.4334.
   Its only remaining value is confirming §5.1's finding past the epoch-4 plateau. **It is not a
   Phase-3 dependency** — do not wait on it.
3. Build the nucleation head (§1.2), wired as the single new component against a fresh-init
   baseline. Enable `autocatalytic_growth` and `shear_readout_gate` as **separate follow-on
   legs**, so the one-variable rule survives a three-part architecture change.
4. Write the C1 re-baseline into `docs/WALL_MODEL_PLAN.md` as §27, labelled explicitly as a
   re-baseline and not an A/B.
