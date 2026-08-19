# PHASE 3/4/5 HANDOFF — a t=0-flow physics wall model

> **SUPERSEDED IN PART, 2026-08-09 — read `docs/PHASE3_RESULTS.md` first.**
> `data.G_x`/`G_y` do not compute derivatives (median ONE non-zero per row; `G_x @ x` = 0
> across the interior). Every flow-derived measurement quoted below — §1.4's gate AUCs,
> §1.5a's ceiling, §1.5b's "the LEVEL does not transfer", §9's open questions 0-3 — was
> computed on that operator. With a correct one, the t=0 gates score
> **0.9093 on the sealed set with GT t=0 flow and 0.8567 without it**, zero learned
> parameters. §1.5's Step 0 is answered; §1.5c is refuted (the level does not emerge from
> `Da`); §1.6's kill criteria are not triggered. §1.5b is refuted too — on the eight
> no-clot vessels the model predicts zero nodes, a 0.0% false-positive rate.

Written 2026-08-08 for a new context window.

**The mission (Phase 3): using the GT flow field at `t=0` ONLY, plus geometry, initial and
boundary conditions, build a new wall-only clot model that generalizes to unseen vessels at
`deploy_clot_score > 0.6`.**

**Phase 5 removes the t=0-GT-flow assumption** and substitutes the deployable ML kinematic model.
Phase 3 is deliberately scoped *inside* a bandaid so the biochem model can be developed against a
clean flow input instead of being confounded by flow-surrogate error.

Repo: `C:\Users\pgssy\thrombus_ml_model` (Windows; PowerShell and Git Bash both available).
History: `docs/WALL_MODEL_PLAN.md` (~4500 lines, every claim cites a `§N`).
**Read §26 first** (the session that produced this), then §10.1, §10.4, §19.2.
Do **not** read §1–§13 linearly — they contain retracted conclusions, cited from later sections
where still valid. §14.5 in particular is RETRACTED by §19.2.

---

## 0. Goal, and where things actually stand

`deploy_clot_score > 0.6` on **unseen vessels**, **wall clot only** (3-hop wall band). Floor 0.50.

| | |
|---|---|
| best *deployed* result | **0.6925** — zero-shot warm start on `patient043`, no cohort training (§9.3) |
| best *in-training* | 0.4889 — Phase 1 epoch 6 |
| fine-tune legs that beat the warm start | **none, out of ~15** |

**Read §6.3 before comparing anything to 0.6925** — it was measured with the flow coupler
working, and every number since 2026-08-06 was measured with it silently disabled.

Phase 0, 1 and 2 of the project ladder are complete. This document replaces Phase 3 onward.

---

## 0a. THE BANDAID — temporary, and it must be removed

**Phase 3 assumes the ground-truth flow field at `t = 0` is available for any vessel, including
unseen ones.** Nothing else from the GT solution is permitted: no flow at `t > 0`, and no GT
species beyond `t=0` initial conditions.

> **CORRECTED 2026-08-13.** This paragraph used to add "and **not** the converged
> `u_prior`/`v_prior`/`mu_prior` (§16.1c — those are the *clot-affected converged* solution and
> remain illegal)". **That is factually wrong and it was suppressing legal inputs.** Measured on
> the packs (`deploy_features.prior_channel_audit`):
>
> | field | corr vs GT `t=0` | corr vs GT `t_final` |
> |---|---|---|
> | `data.x` `u_prior` | **1.000** | 0.999 (flow barely evolves) |
> | `data.x` `mu_prior_nd` | **1.000** | 0.05 |
> | `data.mu_prior` attr | 0.41–0.53 | **−0.04 … −0.07** |
>
> The `data.x` prior channels are the **GT `t=0` fields** — legal under this bandaid, which
> grants exactly that. The `data.mu_prior` *attribute* is an approximate `t=0` prior and
> correlates ~0 with the converged field, so it is not the clot-affected solution either.
> `wss_prior_nd` is identically constant in every pack — a dead channel.
>
> **The real constraint is different and was being missed:** `data.x` is static, so it keeps
> serving GT `t=0` flow even when a model runs with `flow_source="pred"`. That silently
> re-imports the bandaid into the arm meant to be free of it. Phase-5 arms must rebuild those
> channels from predicted flow — `src/differentiable_wall_model/deploy_features.py`.

**Why it is defensible for now.** §26.16/§26.17 established that all spatial structure in `Mat`
enters through the flow-derived gates, and Z1 measured the *predicted* flow's marginal
contribution to clot ranking at 0.041 AUC. Developing the biochem model on top of a flow
surrogate that weak makes every negative result ambiguous — is the chemistry wrong, or the flow?
The bandaid removes that confound so Phase 3 answers one question at a time.

**What it costs.** A model that needs a CFD solve at `t=0` per new vessel is **not deployable** in
the sense this project targets. Every Phase 3 number is an **upper bound** on the deployable
system, and the Phase 3 → Phase 5 gap is exactly the flow surrogate's error.

**When it goes.** Phase 5. Do not let it become permanent by omission: **every result reported out
of Phase 3 must carry the words "with GT t=0 flow".**

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

### 1.5 STEP 0 — does the law + t=0 gates reproduce GT `Mat`?

Under the bandaid (§0a) the flow at `t=0` is exact, so the question "does the flow surrogate
support the gates" moves to **Phase 5** (§4a). Phase 3's Step 0 is the end-to-end physics check:

> **Integrate `j0_mat_si` forward using t=0 gates, `rp`/`ap` held at their measured constants, and
> the autocatalytic `Mas` feedback. Does the resulting `Mat` trajectory match GT?**

Run it per vessel and report **the predicted committed-node count and mass ratio against GT**,
not merely correlation — §1.5c explains why the level, not the ranking, is the thing being
tested. Also report the resulting `deploy_clot_score`. This is the whole plan in one measurement,
and it is CPU-only.

What it distinguishes:

* **Matches** → the law plus t=0 gates plus autocatalysis *is* the model. Whatever residual
  remains is what the learned component must supply, and it will be small and well-posed.
* **Systematically under/over-grows** → the `Da` scale or `step2t` activation gate needs
  calibrating per vessel. That is a small learned correction, not a new architecture.
* **Wrong spatial pattern** → the t=0 gates are insufficient because the flow *evolves* as the
  clot narrows the lumen. That is the one failure mode the bandaid cannot hide, and it would
  mean the coupled flow is needed even in Phase 3.

**The unit risk an earlier draft warned about does not exist.** `gamma_si` is already SI (range
~0.006–1264 1/s, so `lss=25` sits well inside it), `dshear_ds` is already in the gate's units, and
`is_low_shear` is already computed in the t=0 feature table
(`build_feature_table_at_time(data, 0, ...)`). The only conversion still needed is `log1p_nd`
species → SI for `mas`/`sat_m`, and §26.16 says `rp`/`ap` can simply be held constant.

**Do not fix a poor result by adjusting the constants** — they are calibrated and pinned by
`test_comsol_wall_deposition_calibration.py`. A mismatch means the plumbing is wrong, or the
premise is.

### 1.5a IS >0.6 ACTUALLY ATTAINABLE FROM t=0? — measured, and it is marginal

Do not start without reading this. `scripts/diag_t0_ceiling.py`, 35 vessels.

**The information at t=0, as a pure ranking problem.** Rank wall-band nodes by the best
deploy-legal t=0 feature and sweep the threshold — oracle feature *and* oracle threshold, so this
is doubly generous:

```
mean best-single-feature AUC : 0.885
mean ORACLE-THRESHOLD F1     : 0.463
vessels with oracle F1 >= 0.6:  5 / 35
vessels with oracle F1 >= 0.5: 12 / 35
```

**AUC 0.885 sounds excellent and yields F1 0.463**, because the base rate is 2–21% (mean ~7%). At
that imbalance, ranking quality converts poorly into F1. **The base rate, not the ranking, is what
makes this hard.**

**But the rollout beats that "ceiling", and that is the key to the whole plan.** Against §19.2's
per-vessel GNN F1 on the same vessels:

```
vessel     t=0 oracle F1   GNN F1 (19.2)   rollout gain
039            0.647           0.518          -0.129
040            0.388           0.704          +0.316
041            0.438           0.255          -0.183
042            0.402           0.513          +0.111
043            0.416           0.650          +0.234
044            0.401           0.602          +0.201
                                        mean  +0.092
```

The rollout adds ~+0.09 F1 over oracle node-ranking with the *same information*. It wins on
**inductive bias**: autocatalytic growth produces spatially coherent components where independent
per-node thresholding produces scatter. That is exactly what the physics model in §1.2–§1.4
supplies, and it is the strongest argument for building it.

**Naive projection:**

```
0.463 (t=0 oracle ranking, 35 vessels) + 0.092 (mean rollout gain) = 0.554 F1
deploy_clot_score runs ~+0.04 above f1 (p043: 0.6925 vs 0.6497)    ~ 0.597
target                                                               0.600
```

**It lands on the line.** Read that as *marginal, not comfortable*, and note it is biased
**optimistic** three ways:

* the rollout gain is n=6, range −0.183 to +0.316 — enormous variance, and two of six are
  negative;
* those six are `039`–`044`, which §20.5 measured as a materially easier subset (mean F1 0.516
  against 0.322 for a random draw of nine);
* the 0.463 baseline already used oracle feature selection *and* oracle thresholding.

**So: >0.6 on a single favourable vessel is demonstrated (`patient043`, 0.6925). >0.6 as a mean
over randomly drawn unseen vessels is not supported by current evidence.**

The honest objective for Phase 3 is therefore **not** "add 0.09 to 0.463 and hope" — it is
**close the rollout-gain variance**: turn +0.316/−0.183 into a consistent gain. If encoding the
right mechanism makes the gain consistent, the target is reachable. If the gain stays
vessel-dependent, it is not, and §20.5's goal-split question becomes unavoidable.

### 1.5b WHY GENERALIZATION FAILS — measured. Ranking transfers; the LEVEL does not.

This is the most important section in the document. "We know the physics, so why can't we
generalize?" has an answer, and it is not the one the project assumed (§26.19).

**Within a vessel, ranking is fine. Across vessels, the operating point is unknowable.**

```
mean best-single-feature AUC, within vessel : 0.884
base rate across the 35 vessels             : 2.1% .. 21.2%   (10x)

best |rho| vs base_rate over 225 real t=0 aggregates : 0.513
best |rho| vs base_rate over 225 RANDOM vectors      : 0.513   <- identical
```

**Nothing at t=0 predicts how much clot a vessel develops** — the best "predictor" is exactly what
a random search of the same size finds. And the physics does not rescue it:

```
rho(frac low-shear gate open,  base_rate) = -0.054
rho(frac separation gate open, base_rate) = +0.037
```

**This one fact explains the project's whole symptom list**: §2.7/§2.9's "no gate threshold works
on `patient037`", §9.10's mass-rejection of every epoch, Phase 1's mass collapse 3.06 → 0.43, T5's
faster collapse, and why a mass selection guard exists at all. You can rank nodes within a vessel;
you cannot know where to cut. At a ~7% base rate, precision is exquisitely sensitive to exactly
the quantity that does not transfer.

**Three supporting facts:**

* **The gate is open nearly half the time.** Low-shear gate open on **45.6%** of band nodes, and
  only ~7% ever commit. The dominant gate is necessary-ish and wildly insufficient; whatever
  selects 7% out of that 45.6% is not the gate.
* **13 different features win across 35 vessels** — pressure 5×, a boundary condition 5×,
  `speed_nd` 5×, and a **raw y-coordinate 5×**. A coordinate beating every physics feature means
  the ranker found *where in this vessel*, not *why*.
* **t=0 → final is an ignition map, not a regression.** ~90% of growth is autocatalytic, which is
  a bifurcation: marginally-above runs away, marginally-below never starts. §20.4 measured the
  degradation directly — t=0 predicts the **t=20 seeds at 0.903** but the **final map at 0.806**.
  And "before clot affects flow" holds only for initiation: §10.1 puts within-vessel onset spread
  at **0.346 of the horizon**, with `mu1(Mat)` stepping viscosity 1→80 at commitment, so for a
  third of the run committed nodes are changing the flow uncommitted nodes see.

### 1.5c THEREFORE — what the physics model is actually for

**Not better ranking.** Ranking is already 0.884 and is not the bottleneck.

**The level should EMERGE rather than be predicted.** A learned model must choose an operating
point that §1.5b shows is unknowable from t=0. The law sets the amount dynamically — through
saturation `Sat = 1 − M_tot/Minf`, the `Da` rate scale, and the finite horizon. **Mass
conservation replaces a threshold nobody can transfer.** That is the mechanism by which a
physics-mirroring model could generalize where fifteen learned legs did not.

**So Step 0's real question is not "does the law rank correctly" — it is "does the law get the
AMOUNT right, per vessel".** Report per-vessel predicted vs GT committed-node count and mass
ratio, not just correlation. If the levels match, generalization follows and §1.5a's 0.597
projection is conservative, because that projection assumed the ranking-plus-rollout path rather
than a physically-set level. If the levels do not match, no architecture fixes it, and the goal
must be restated against §20.5's split.

### 1.6 Kill criteria

* **Step 0 fails** (law + GT species does not reproduce GT `dMat`) → the premise is wrong; stop
  and report rather than tuning around it.
* **Step 0's forward model gets the LEVEL wrong per vessel** (§1.5c) → the law does not supply
  the operating point either, and since §1.5b shows nothing else does, the >0.6 goal is not
  reachable from t=0 and must be restated against §20.5's split.
* **Step 0's forward model has the wrong spatial pattern** (§1.5) → t=0 gates are insufficient
  because the flow evolves with the clot; the coupled flow is needed even under the bandaid.
* **The learned correction needs real capacity to close the residual** → the physics framing is
  not buying what it promised; fall back to §3's comparison arm.
* **Whole model** against §19.2's bar: logreg **0.516**, GNN **0.540**. And against the standing
  0.6925 once §6.3 makes that comparison legitimate.

---

## 2. BUILD ORDER

0. **Resolve the metric split** (§6.1a). CPU, and it decides whether any epoch selection to date
   is trustworthy.
1. **Step 0** (§1.5) — integrate the law on t=0 gates, compare to GT `Mat`. CPU. Everything
   depends on it, and §1.5a says the margin is thin, so measure before building.
2. **Assemble the forward model**: t=0 GT flow → `gamma_si`/`dshear_ds` → both gates →
   `j0_mat_si` with `rp`/`ap` at their measured constants → integrate `Mat` with the `Mas`
   feedback → `mu1(Mat)` step → clot readout. No learned parameters yet. Score it.
3. **Measure the residual.** Where does the pure-physics forward model differ from GT? That
   residual — and nothing else — defines what the learned component should do. Candidates, in
   order of how little they assume: per-vessel `Da` scale, the `step2t` activation gate, then the
   ~10% `AP` variation (§26.16).
4. **Add the smallest learned correction that closes the residual**, and check it against a
   logreg baseline before adding any capacity. §19.2 is the cautionary tale: 187k parameters
   bought +0.024 over a logistic regression.
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

## 4. PHASE 4 — after the physics model lands (still under the bandaid)

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

## 4a. PHASE 5 — remove the bandaid

Substitute the deployable ML kinematic model for the GT `t=0` flow, and measure what it costs.

1. **Re-run `scripts/diag_physics_gate_support.py` with predicted flow.** The GT-flow gate AUCs in
   §1.4 are the reference; the drop is the flow surrogate's error expressed in the only units that
   matter here. This is the single number that says whether the deployable system can work.
2. **Re-run the Phase 3 model end-to-end on predicted flow.** The Phase 3 → Phase 5 delta is the
   deployability gap.
3. **If the gap is large, the flow surrogate becomes the project.** Z1 put its marginal
   contribution at 0.041 AUC, so this is the likely outcome and should not be a surprise. The
   corrector with its shear head (§6.3) is the natural place to start.

Phase 5 is where the project's actual deployability claim is decided. Phase 3 only establishes
whether the *chemistry and dynamics* are right given good flow.

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
   whether any "best epoch" on record is trustworthy, and it is cheap. Do it alongside Step 0.
1. **Does the law on t=0 gates reproduce GT `Mat`?** §1.5. Everything depends on it.
2. **Can the rollout-gain variance be closed?** §1.5a — the projection lands on 0.597 against a
   0.600 target only if the +0.316/−0.183 spread becomes consistent. This is the real Phase 3
   question, not the mean.
3. **Is the flow good enough without the bandaid?** Phase 5 (§4a). Z1 scored the flow channel at
   0.041 AUC. Likely the binding constraint on the deployable system.
3. **Does `z_kin` carry vessel-specific information?** Shuffle test, §6.4. Gates D3.
4. **The goal split** (§20.5): `039`–`044` scores mean F1 0.516 where a random draw of 9 scores
   0.322. Be explicit about which set any ">0.6" claim is on.
