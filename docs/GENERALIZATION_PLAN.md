# Generalization Plan — recovering deploy clot score on unseen vessels

> **[2026-08-04] The active working plan is now [`WALL_MODEL_PLAN.md`](WALL_MODEL_PLAN.md).**
> Read that first for current scope, next steps, and parked arms. This document remains
> the historical record and the source for EDA / eval-protocol background, but several
> of its measurements use raw `Mat` activity rather than the graded viscosity-rise label
> and are corrected in `WALL_MODEL_PLAN.md` §3.

Goal: raise the compound deploy clot score on *full generalization* (held-out
vessels) toward **>0.6**, with **>0.7** as the stretch target. This document is
the working plan; it is grounded in an EDA over the anchor vessels
(`scripts/eda_generalization.py` -> `outputs/biochem/eda/generalization_eda.json`;
originally 35, now **43** after the 2026-08-03 batch — see §1b) and an audit of
the wall backbone's conditioning inputs.

TL;DR: the bottleneck is **not** model capacity or architecture (in-family score
is ~0.81, 007 ~0.93). It is (1) a tiny, flow-homogeneous dataset, (2) a held-out
set dominated by vessels that barely have any GT clot, and (3) wall-backbone
conditioning that leans on spatial position instead of transferable physics.
Fix the eval set and the wall seeding first; the growth specialist is
second-order and capped by the wall.

## Eval protocol lock (agents)

1. **Deploy-faithful always.** Generalization numbers must use cold deploy with **no GT velocity leak**: RGP-DEQ once at t=0, then local tiling / kinematic corrector for clot-aware flow updates. At eval, force `flow_feats_source=auto` + corrector coupling (`eval_mat_growth_simple._apply_ckpt_recipe` / `canonical_deploy_clot_metrics`). GT is labels / timeline only. **Never** wire COMSOL `data.y[..., 0:2]` into `model.velocity` / physics-GAT / convective upwind — use `band_uv_for_model` / `resolve_species_rollout_uv`.
2. **Wall-gen / phase1 / flow-source primary holdout = `patient020` only** (clot-rich). Do not use `patient020+patient034` means as the decision metric — `034` is near-zero clot and muddies the gate. Broader held-out tables remain optional diagnostics.

Canonical small cohort: train `005,006,010,023,002`; val = holdout = `020`. See `AGENTS.md` and `go_flow_source_ab.ps1` / `go_phase1_sweep_v3.ps1`.

**Wall-gen baseline (promoted):** `FS_ab_coupled` — typed stack in `WG_sweep_v3_*`, ckpt
`outputs/biochem/biochem_gnn/wall_gen_baseline/species/best.pth`, reference
`data/reference/mat_wall_gen_baseline.json`. Phase1 arms warm-start from this and tweak one factor
(geom / flux / mirror / teacher noise). Does **not** replace locked `WC_v7` for compound deploy.

---

## 1. What the data actually looks like (EDA)

Anchor graphs (`data/processed/graphs_biochem_anchors/*.pt`; exclude `*_mirror_y`),
PyG `Data`, `x` = 18-col `NodeFeat`, `y` = `[T, N, 16]` species; clot =
`Mat_log1p_nd` (y-channel 15), active threshold `1e-4`
(`SPECIES_SNAPSHOT_ACTIVE_LOG_ND`).

Findings that drive the plan:

1. **Reynolds number is constant at Re=450 for every vessel.** There is zero
   flow-regime diversity. Any deploy vessel at a different Re, inlet speed, or
   viscosity is fully out-of-distribution — the model has literally never seen
   another regime. This is the single largest latent generalization risk.
   *(Still true after the 2026-08-03 batch.)*

2. **Simulation length varies a lot** (`T` from 29 to 201). Short sims never
   develop clot, so their GT clot burden is near zero. Example: **`patient039`**
   (half-finished aneurysm, T=92).

3. **Clot burden splits the cohort into three buckets.** Pre-batch geometry was
   mostly mild stenotic/curved tubes; the new batch adds **real aneurysm /
   expansion** vessels (`040`, `043`, incomplete `039`) and stronger stenoses.

   | Bucket | Count (pre → now) | Vessels | Meaning for scoring |
   |---|---|---|---|
   | **clot-rich (full, T=201, off-wall ≥30%)** | 13 → **19** | prior 13 + **012,040,041,042,043,044** | deploy clot score is meaningful here |
   | low / short sim | 14 → **15** | prior 14 + **039** | little GT clot; score floors near 0 regardless of model |
   | zero / near-zero clot | 8 → **9** | prior 8 + **027** | no clot at all; F1 measures noise / empty-GT FP |

4. **007 is not geometrically special** — its value as a "shape teacher" is that
   it is the most clot-rich, full-length vessel (highest `Mat` max, 41% off-wall),
   not that it has unusual geometry. (Our earlier "007 is a complex-expansion
   outlier" read was a width-floor artifact, now corrected.)

5. **The old sealed val set was mostly empty of clot.** Val was
   `004,015,018,019,021,031,035,036` — but 6 of those 8 are low/short vessels.
   Only `021` and `035` are clot-rich. That is why the pivot run's mean val clot
   score "stalled" at ~0.34 and `patient015` sat at ~0.028: **most of the val set
   has almost no clot to recover**, so the mean is structurally capped near 0.3.
   We were grading generalization largely on vessels where the right answer is
   "≈nothing."

**Consequence:** before any retraining, we must score generalization on the set
where clot exists. The honest held-out generalization vessels are the clot-rich
ones **not** in train: `021, 032, 035, 037` (plus any sealed clot-rich from §1b
when LOAO expands).

### 1b. New-vessel batch (2026-08-03) — sealed organization

Landed: `012, 027, 039–044`. Inventory:
[`data/reference/generalization_new_vessels.json`](../data/reference/generalization_new_vessels.json).
Code: `WALL_GEN_BATCH_1B_*` + `WALL_GEN_CLOT_RICH_ANCHORS` in
`src/biochem_gnn/mat_growth_simple.py`. Velocity QA: `outputs/biochem/viz/new_anchors/`.

**Sealed roles (we chose these):**

| Role | Anchors | Rationale |
|---|---|---|
| **Train (batch add)** | `012, 040, 041, 042` | Strong stenosis (`012`) + stenosis twins (`041`/`042`) + one aneurysm (`040`) so train sees both pathologies |
| **Sealed challenge** | `043` (aneurysm), `044` (stenosis) | One held-out of each new geometry family; never in default N+ train |
| **Primary wall-gen gate** | `020` (unchanged) | Continuity with existing baselines; report `043`/`044` separately — do **not** average into the `020` gate |
| **Neg control** | `027` | Full-length empty GT (like `034`); empty-GT FP score only |
| **Exclude** | `039` | Half-finished aneurysm (T=92); never clot-rich train/holdout |

| Anchor | Sim | Pathology | Bucket | Sealed role |
|---|---|---|---|---|
| `patient012` | full | stenosis | clot-rich | **train** |
| `patient027` | full | stenosis | zero | **neg control** |
| `patient039` | half-finished | aneurysm | low/short | **exclude** |
| `patient040` | full | aneurysm | clot-rich | **train** |
| `patient041` | full | stenosis | clot-rich | **train** |
| `patient042` | full | stenosis | clot-rich | **train** |
| `patient043` | full | aneurysm | clot-rich | **challenge** |
| `patient044` | full | stenosis | clot-rich | **challenge** |

Default N+ train = `WALL_GEN_CLOT_RICH_ANCHORS − {020} − {043,044}` (see
`wall_gen_clot_rich_train_anchors`). Still missing vs +10: new **Re / inlet**
regimes (all remain Re≈450).

---

## 2. Conditioning audit — what the wall backbone sees

Wall backbone = locked `WC_v7_clot_phi_mse` GraphSAGE. Per-node band input
(`build_band_base_features`) is:

```
[ z_kin (frozen RGP-DEQ kinematic latent) , sdf_nd , FLOW_FEATS(5) ]
FLOW_FEATS = [ log1p(speed), log1p(shear_proxy), tanh(divergence), x_norm, y_norm ]
```

with dynamic flow splicing (`SPECIES_FLOW_FEATS_DYNAMIC=1`), `wall_hops=3`.
Static geometry discriminators (`SPECIES_GEOM_FEATS` / `_RICH`) are **off** in the
locked leg.

High-lift issues, ranked:

1. **`x_norm, y_norm` are in the flow block = spatial memorization risk.** On 16
   fixed training vessels the net can learn "clot appears at this normalized
   position," which does not transfer to a new vessel. There is already an
   ablation knob `SPECIES_FLOW_FEATS_DROP_XY=1` (keeps `[speed, shear, div]`,
   zeros `x,y`). **Action:** measure wall A-floor on the clot-rich held-outs with
   and without xy. If drop-xy helps or is neutral held-out, drop it — it is a
   pure generalization win.

2. **No residence-time / low-shear-dwell feature.** Nucleation is driven by
   low-shear **and** high residence. We feed instantaneous shear and a divergence
   proxy (negative = converging/stagnating), but nothing time-integrated. A
   deployable residence proxy (e.g. per-node dwell = accumulated `1/speed` or a
   low-shear persistence measure over the base flow) is physically transferable
   and cheap. **Action:** add as an optional band channel and A/B it.

3. **Static geometry discriminators are off.** `_geometry_band_features` (width,
   width-gradient/expansion, wall-curvature, + 2-hop rich variants) encode *where*
   recirculation pockets form, are static/clot-blind/deployable, and transfer
   across geometry. A prior probe (leg C) found the 2-hop expansion/curvature
   separate committed clot pockets from merely eligible wall nodes. **Action:**
   turn `SPECIES_GEOM_FEATS_RICH=1` on in the wall retrain and A/B on held-outs.

4. **`z_kin` is the primary flow channel and is a frozen per-vessel latent.** Its
   cross-vessel transfer quality is unverified. Per-vessel standardization
   (`kin_per_vessel_norm`) matters here; keep it on and confirm.

Note the flow features are trained on GT COMSOL velocity
(`SPECIES_FLOW_FEATS_SOURCE=gt`) and deployed on corrector-coupled flow — so flow
accuracy at deploy also gates wall quality.

---

## 2b. Measured A-floor (2026-07-29) -- the wall IS the ceiling

Rescore on clot-rich held-outs (`scripts/eval_mat_growth_simple.py`, deploy-faithful
WC_v7 leg; artifacts under `outputs/biochem/eda/rescore_*.json`):

| Anchor | Wall-only A-floor (score / off-wall pred/GT) | Compound (score / off-wall) | Growth headroom |
|---|---|---|---|
| `patient032` | 0.325 · **17/120** | 0.325 · 16/120 | **+0.000** |
| `patient021` | 0.256 · 14/14 | 0.277 · 17/14 | +0.021 |

On the meaningful challenge vessel the wall commits only 17 of 120 off-wall GT
nodes and the growth specialist adds **nothing** (compound == wall). `frontier_offwall`
can only grow *from* committed wall clot, so under-seeding caps everything. Conclusion:
**work the wall; the lumen/growth specialist is shelved** until the wall seeds properly.

Hardware note: this box has a **4 GB GPU** and one deploy-faithful coupled rollout is
**~25-30 min/anchor**. A single full WC_v7 wall retrain is ~2-5 GPU-hours; a 13-fold LOVO
is ~30-100+ GPU-hours. LOVO is therefore deferred in favour of tight single-family folds.

## 2b-bis. Train/test leakage in every prior wall number (found 2026-07-29)

`scripts/go_mat_growth_simple.ps1` appends `--all-anchors` in **both** branches of its
`if ($AllAnchors)` test, so the flag is inert and **every WC_v7 wall leg trained on all 35
anchors**. `train_species_pushforward_continuous.py` compounds this: a bare `--init` (or no
`--init`) falls back to `DEFAULT_S34_CKPT`, itself all-anchor trained, so only explicit
`--no-init` gives a clean start.

Consequences, in order of importance:

1. **The A-floor table in 2b is fit, not generalization.** `patient032`/`patient021` were in
   the locked wall's training set. So the wall commits 17 of 120 off-wall nodes on a vessel it
   *was trained on*. Under-seeding is therefore not primarily a data-coverage problem, and more
   data alone will not fix it. Part of the 17/120 is by design (wall + 3-hop band), but 032's
   off-wall GT mass sits inside that band, so the rest is underfitting.
2. **Any warm-start from `locked/species_gnn_best.pth` leaks the held-outs**, which invalidated
   the first `wall_family7` arm (killed). `go_wall_family7_probe.ps1` now defaults to cold
   (`--no-init`); `-WarmStart` is kept only as a deliberately-labelled fine-tune arm.
3. Earlier "sealed" splits (009/021/032/035) were never sealed at the wall level, so the
   pre-2026-07-29 generalization gap was measured against a contaminated baseline.

## 2b-ter. Leak audit of the deploy clot metric (verified 2026-07-29)

Asked whether a high held-out score could come from GT being fed to the rollout. Traced and
cleared for the `deploy_*` fields under `WC_v7_clot_phi_mse`:

| Stage | Source | Leak? |
|---|---|---|
| Initial condition | `deploy_fimat_log_init`, `SPECIES_ROLLOUT_IC_SOURCE=resting` | no -- resting zeros |
| Base flow | `data.u0_pred` or frozen RGP-DEQ (`ClotAwareFlow.base_flow`) | no -- never reads `data.y[:,0:2]` |
| Rollout flow | `VEL_SOURCE=coupled`, driven by the model's own predicted clot | no |
| Clot state | `predict_continuous_step_delta(model, ...)` per step | no -- model under test, not locked ckpt |
| Other species | `SPECIES_ROLLOUT_PIN_OTHER=rest` | no |

`resolve_species_rollout_uv` returns GT velocity only on its `src == "gt"` branch, which this
leg never takes; `train_deploy_eval_flow_source()` also defaults to `kinematics`. GT is used
only for labels and timeline length. `patient020` has **0** active GT clot nodes at t=0, so the
baseline flow field cannot encode clot position either.

Caveats that are real:

- `val_state_f1` / `val_mat_f1` / `val_growth_f1` / `init_f1` come from `eval_continuous_window`,
  which **is** handed GT velocity and GT species blocks (teacher-forced). They are diagnostics,
  **not** generalization numbers. Quote only `deploy_*`.
- This leg sets `CLOT_GUIDE_RELAX_HOPS=3`, so relaxed prec/rec tolerate a 3-hop miss. The
  dilation-free number is `clot_f1` (`= strict_f1`); prefer it when claiming generalization.
- `CLOT_PHI_CEILING_HOPS=4` caps the clot label to within 4 hops of the wall, so the target is
  ~1% of the mesh (~150-200 nodes on `patient020`), not the 3154 mat-active nodes at t=200.
  Combined with `SPECIES_SNAPSHOT_WALL_HOPS=3`, band and label are aligned by construction:
  this is a wall-band task, not full-lumen prediction.

## 2b-quater. Two rollout paths give incomparable clot scores (found 2026-07-29)

Same checkpoint (`wall_family7_cold.pth`, ep10), same anchor (`patient020`), same GT, t=200:

| Path | Entry point | Strict clot F1 | Precision | Recall |
|---|---|---|---|---|
| In-training `deploy_clot_g` | `eval_deploy_clot_f1` (`species_pushforward_continuous.py`) | **0.844** | 1.000 rel. | 0.945 rel. |
| Canonical / viz | `rollout_species_gnn_phi_trajectory` (`species_gnn_clot_rollout.py`) | **0.234** | 0.338 | 0.179 |

Counts on the canonical path: TP 22, FP 43, FN 101 vs 123 GT nodes, only 65 predicted.

Cause: they are different rollout implementations. `eval_deploy_clot_f1` builds a `ClotAwareFlow`
coupler whenever `SPECIES_CLOSED_LOOP_COUPLING=1` (set by the leg) and takes the coupled-flow
branch, so clot-aware occlusion feeds back and sustains growth. The viz/canonical path runs
`apply_deploy_env`, which forces `SPECIES_ROLLOUT_VEL_SOURCE=kinematics`, and does not build the
coupler.

**Rule: never compare an in-training `deploy_clot_g` against an `eval_mat_growth_simple.py`
number.** The A-floor table in 2b (032 = 0.325, 021 = 0.256) came from the canonical path, and
the viz agrees with it (0.23-0.28), so the in-training score is the optimistic outlier. Any
promote / leg-comparison decision must use `eval_mat_growth_simple.py` only.

Knock-on: checkpoint selection is worse than "picks a slightly wrong epoch". Under
`SPECIES_CONTINUOUS_PHYSICS_READOUT=1` the `physics_on` branch wins the if-chain before the
`--exclude-val-from-train` branch, so selection uses
`0.50*val_clot_phi_f1 + 0.25*val_growth_f1 + 0.15*val_state_f1 + 0.10*val_growth_mat_f1`
where `val_clot_phi_f1 == 0.000` every epoch (half the weight is dead) and the rest are
**GT-teacher-forced** window metrics. So no clot-deploy signal reaches selection at all.

## 2b-quinquies. Alignment fix + honest metrics (2026-07-29)

Correction to 2b-quater: `scripts/eval_mat_growth_simple.py` calls **the same**
`eval_deploy_clot_f1` as training (`rollout_species_gnn_phi_trajectory` only feeds the timeline
stats). The rollout was never the difference -- the **protocol around it** was:

1. **Flow cache.** Offline resets `reset_species_rollout_flow_cache()` *before* each anchor;
   training reset it *after*. Training therefore scored on a coupled-flow field cached from the
   preceding teacher-forced rollouts -- a more-occluded state than cold deploy ever sees.
2. **In-place `data.y`.** Closed-loop coupling writes diverted UV into `data.y`; training reuses
   one val pack every epoch (`.to(device)` returns self when resident), so it accumulates.
   `band_speed_at_time` reads those channels, so vel-decay drifts.
3. **Env restore** was not exception-safe on the training side.

Fix: both callers now go through `canonical_deploy_clot_metrics`
(`src/evaluation/canonical_clot_eval.py`), which resets the cache first, clones `data`, applies
the deploy env, and restores env in a `finally`. Identical protocol by construction.

*Not yet verified:* that this closes the 0.877 vs 0.2996 gap. Confirmation = next training run's
in-training `deploy_clot_*` matching a standalone `eval_mat_growth_simple.py` on the same ckpt.

### Metrics: report mass, not just tolerance-inflated F-scores

`clot_mass_ratio = n_pred / n_gt` added to `compute_clot_relaxed_metrics` (surfaced as
`deploy_clot_mass_ratio`, with raw `clot_pred_pos` / `clot_gt_pos`). Rationale: with
`CLOT_GUIDE_RELAX_HOPS=3`, a 65-node prediction scored relaxed prec/rec of 1.000/0.945 against
123 GT nodes -- both saturated while the model committed **half** the required mass. Report:

| Metric | Reads |
|---|---|
| `deploy_clot_f1` (strict) | dilation-free overlap -- the headline number |
| `clot_mass_ratio` | 1.0 = right amount, <1 under-commits, >1 over-paints |
| `deploy_clot_offwall_strict_f1` | whether anything is committed past the wall |
| `clot_empty_gt_score` (2c) | FP restraint on clot-free vessels |

Do **not** headline relaxed prec/rec; they saturate. Selection also no longer touches
GT-teacher-forced `val_*` metrics on a held-out anchor (`select_checkpoint_score`,
mode `deploy_only`, covered by `src/tests/test_checkpoint_selection_gt_free.py`).

### Scope simplification (2026-07-29)

Re = 450 for all simulations; **no cross-Re generalization required**. Target is geometry
generalization within straightish vessels at fixed Re.

## 2b-sexies. ROOT CAUSE: `--exclude-val-from-train` validated on a TRAINING vessel (2026-07-29)

`train_anchors` drops the val anchor when `--exclude-val-from-train` is set, and `packs` is built
only from `train_anchors`, so no pack for the held-out anchor ever existed. Resolution was:

```python
val_pack = next((p for p in packs if p["anchor"] == val_anchor), packs[0])   # silent fallback
```

With no match it returned **`packs[0]` = the first training anchor**. The flag intended to create
a held-out validation set silently destroyed it.

Proof (wall_family7 cold, ep10). In-training "val = patient020" vs a standalone canonical eval
of **patient005**:

| Metric | in-training "patient020" | canonical `patient005` | canonical `patient020` |
|---|---|---|---|
| `deploy_clot_score` | 0.8771 | **0.8771** | 0.2996 |
| strict clot F1 | 0.8444 | **0.8444** | 0.2487 |
| relaxed rec | 0.9452 | **0.945** | 0.463 |
| `deploy_mat_f1` | 0.5825 | **0.583** | 0.1708 |

Four metrics matching to four decimals. This supersedes the protocol theory in 2b-quinquies:
the rollout and protocol were fine, **the vessel was wrong**. Every in-training "held-out" number
from any LOAO-style run is training performance.

Fix: build an eval-only pack for the held-out anchor (kept out of `packs`), raise instead of
substituting on mismatch, print `[i] val pack = <anchor> (held_out=...)`, and filter the held-out
anchor out of the backprop-carrying deploy-horizon aux term. Covered by
`src/tests/test_val_pack_is_held_out.py`.

### What the fit/generalize experiment actually showed

Canonical eval of the same cold ckpt, train vs unseen:

| Set | Anchors | `deploy_clot_score` | strict F1 | off-wall strict F1 |
|---|---|---|---|---|
| **Train** | `005, 006, 010` | **0.871** | **0.850** | 0.571 / 0.632 / 0.000 |
| **Held out** | `020` | 0.300 | 0.249 | **0.000** |

So the architecture **can** fit 200-step closed-loop deploy rollouts (strict F1 ~0.85) and **can**
commit off-wall clot (up to 0.63) -- on vessels it trained on. It does not transfer to an unseen
straightish vessel at fixed Re=450.

This **reverses** the earlier "rollout stability" diagnosis: closed-loop rollout is not the
problem. The problem is **memorization** -- a ~0.60 strict-F1 train/unseen gap on 5 training
vessels. That re-prioritises exactly what was previously de-prioritised:

1. **Drop absolute coordinates** (`SPECIES_FLOW_DROP_XY`): `x_norm`/`y_norm` in the flow feats let
   the net key on vessel-specific position. Leading suspect for memorization.
2. **Regularise / shrink capacity**, augment (mirror), and widen the geometry pool.
3. Rollout-stability work is **not** indicated.

## 2c. Scoring clot-free vessels (fixed 2026-07-29)

Predicting *no* clot where there is no clot is a success, and over- vs under-prediction
should both be punished. Previously only the exactly-vacuous case was handled: GT empty +
**any** prediction collapsed every field to 0.0, scoring a 2-node blip identically to a
500-node spray. Now graded (`src/evaluation/clot_relaxed_metrics.py`):

- `empty_gt_match_score(n_pred) = 1 / (1 + n_pred / tol)`, `tol` = `CLOT_EMPTY_GT_FP_TOL`
  (default 8 nodes, i.e. 8 FPs scores 0.5). Nothing predicted -> 1.0; decays monotonically.
- Applied whole-vessel when `n_gt == 0`, and **independently to the off-wall block** when
  `offwall_n_gt == 0` -- which is what makes "no spray" measurable on wall-only-clot vessels.
- Raw counts (`clot_fp`, `offwall_n_pred`, ...) stay truthful for diagnostics; flags
  `clot_empty_gt` / `offwall_empty_gt` mark the graded rows so they are never silently
  averaged in as failures.

## 3. Evaluation sets (fix scoring first)

Add two named anchor sets and always report them separately.

- **`family_validation`** (in-distribution sanity — "can we even fit the
  family?"): clot-rich, full-length vessels that sit close to a training vessel
  in standardized geometry space. Spec: **`patient035, patient037, patient021`**
  (nearest-train distance 0.23 / 0.43 / 0.61; off-wall 38 / 35 / 49%). If we can't
  score high here, nothing else matters. This is the user's "straight-stenosis in
  both train and val at different strengths" idea, made concrete on clot-bearing
  vessels.

- **`generalization_challenge`** (sealed, clot-rich, held out): **`patient032`**
  (clot-rich, 40% off-wall, nearest-train distance 0.63). This is the real
  generalization number. Keep **`patient009`** only as a *no-harm / robustness*
  case (T=67, essentially no GT clot) — success there is "don't hallucinate," not
  a score target.

- **Batch-1b sealed challenge** (2026-08-03, §1b): **`patient043`** (aneurysm) +
  **`patient044`** (stenosis). Train-side batch mates: `012,040,041,042`. Primary
  decision gate remains **`patient020`**; report 043/044 as a separate geometry
  table. Neg control `027`; exclude incomplete `039`.

- **Report, don't average blindly:** never fold the zero/low-clot vessels into a
  single mean with clot-rich ones. Report clot-rich mean, low/short mean, and
  zero-clot false-positive rate separately.

- **Active cohort: tight straight-vessel family (chosen 2026-07-29).** First real
  test of wall generalization -- train on near-identical vessels, hold out
  near-identical vessels, and include clot-free negative controls so correct
  silence is rewarded (see 2c). All 7 are straightish (curvature 0.03-0.06,
  stenosis 1.11-1.45, expansion 1.05-1.28):

  | Role | Vessels |
  |---|---|
  | clot-rich | `010, 020, 006, 005` |
  | clot-free / near-empty (negative controls) | `023, 034` (0%), `002` (0.43%, T=67) |

  Scored as two groups: clot-rich by `deploy_clot_score` / off-wall recall;
  clot-free by the graded empty-GT score (2c). `p005` is the loosest member
  (aspect 9.8 vs ~5-7) so it belongs in train, not held out.

### Drop-xy attempt (2026-07-29) -- invalidated; honest cold rebaseline instead

Tried `SPECIES_FLOW_FEATS_DROP_XY=1` for 7 cold epochs on the same cohort. The leg
loader calls `apply_mat_growth_leg_env(..., force=True)`, which resets
`SPECIES_FLOW_FEATS_DROP_XY` to `"0"` from the mat-growth recipe, so the run was
**not** drop-xy (`meta.flow_drop_xy=false`). Re-run must set the flag *after* leg
apply (or use a WC_v7-compatible leg override).

What the run *did* give: the first **honest** cold train with real held-out
`patient020` (val-pack fix live). Best ep6: guiding **0.319**, strict F1 **0.275**,
select_score 0.290 -- matches the earlier offline held-out ~0.30. Confirms the
~0.30 ceiling under correct measurement; does not test the memorization hypothesis.

### Result: cold wall probe on the family-of-7 (2026-07-29) -- no lift

Train `005,006,010,023,002`; held out `020` (clot-rich) + `034` (clot-free). Cold
(`--no-init`), so leak-free. Canonical `eval_mat_growth_simple.py`
(`outputs/biochem/eda/wall_family7/eval_holdout_cold.json`):

| Anchor | `deploy_clot_score` | strict clot F1 | rel. prec / rec | off-wall strict F1 |
|---|---|---|---|---|
| `patient020` (clot-rich) | 0.2996 | 0.2487 | 0.400 / 0.463 | **0.000** |
| `patient034` (clot-free) | 0.3077 | 0.3077 | -- (18 FP, graded) | 1.000 (no off-wall FP) |
| **mean** | **0.3036** | 0.2782 | | |

Locked-wall A-floor on the canonical path was 0.325 (`032`) / 0.256 (`021`), so
**0.3036 is inside the existing baseline band: cold retraining on a tight
5-vessel family produced no measurable generalization lift.** Training a
same-family wall is not the missing ingredient.

Two failure modes, opposite directions, both present:

- `020`: under-seeds badly -- median FN 63 vs median FP 15.5, off-wall strict F1
  exactly 0.000 (pred hop>=2 ~1.5 nodes vs GT ~6.5). Wall commits almost nothing
  beyond hop 0.
- `034`: leaks 18 wall-adjacent FPs where GT is empty (graded score 0.3077 =
  `1/(1+18/8)`). Off-wall FP count is 0, so the leak is wall-hugging, not spray.

The diagnostic that matters most: teacher-forced `val_state_f1` was **0.913**
while canonical closed-loop strict F1 is **0.249** on the same vessel. The
one-step map is learned; the 200-step unaided rollout is what collapses. That
points at **rollout stability** (unroll length, scheduled sampling, TBPTT tail,
`fp_w` balance) rather than conditioning features (drop-xy / geom-rich), which
should be de-prioritised.

---

## 4. Plan

### Tier 1 — The Immediate Target: Fix the Wall Model
We are adopting a two-model compound architecture (`C0_compound_front_offwall_h0p5`) as our base. The compound architecture is bottlenecked entirely by its seed-finding capability (the wall model). Therefore, our **immediate target** is to tune the `WC_v7_clot_phi_mse` wall model component to predict wall clots correctly on a small cohort of simple vessels. We are temporarily adding a "wall deploy score" metric that only compares predictions on the wall vs real clots on the wall.
**Current Goal: Achieve Wall Deploy Score > 0.5.**

**Important Enforcement:** Since we are currently *only* tuning the wall model, the deployment clot score must **only depend on wall clots**. We should not even allow our wall model to predict off-wall, as off-wall growth is the responsibility of the downstream growth specialist. The evaluation metrics and **training losses** must explicitly mask out the lumen (off-wall) from both predictions and ground truth. Training loss penalties for the wall model must *only* come from the wall clots.

*   **Scope:** Tier 1 will strictly focus on vessels with **Re=450**. We will not be asked to predict outside of this flow regime for now.
*   **Baseline:** Based on recent experiments, dropping the absolute spatial coordinates (`SPECIES_FLOW_FEATS_DROP_XY=1`) provided a measurable improvement (Leg A). This drop-xy configuration is our new wall baseline going forward.
*   **Performance:** To eliminate the 15+ minute solver bottleneck during evaluation, we are now pre-computing the RGP-DEQ kinematics (t=0) for our active vessel family and caching them to disk transparently.

1. **Re-score on the right set (no training needed).** Recompute wall-only
   A-floor and compound score on `family_validation` + `generalization_challenge`.
   Expectation: the "0.34 stall" was an artifact of grading on empty vessels; the
   clot-rich held-out picture will look different (better or worse — either way,
   honest).

2. **Wall backbone leave-one-vessel-out (LOVO) retrain + per-held-out A-floor.**
   The wall is the ceiling on everything (compound ≈ wall seed + lumen fill; on
   009 the wall floor is ~0.02). Retrain the wall with LOVO discipline over the
   clot-rich vessels and record A-floor per held-out. This is the measurement
   that tells us how far 0.6/0.7 actually is.

3. **Conditioning experiments (cheap, high-upside), A/B'd on held-outs:**
   - `SPECIES_FLOW_FEATS_DROP_XY=1` (kill spatial memorization).
   - `SPECIES_FLOW_FEATS_SOURCE=kine` (train on predicted flow to fix exposure bias).
   - `SPECIES_GEOM_FEATS_RICH=1` (transferable expansion/curvature discriminators).
   - Add a deployable residence-time / low-shear-dwell channel.
   Keep each as a single-flag A/B on the wall backbone so we attribute the lift.

4. **Training Dynamics (Rollout Stability Fixes):**
   - The teacher-forced 1-step F1 is >0.90, but the autonomous rollout collapses. This strongly implicates training dynamics (exposure bias, lack of scheduled sampling, short TBPTT unroll length) rather than model capacity.
   - We will address training dynamics *after* finishing the conditioning audits in step 3.
   - Up to **+10 real vessels** — **8 landed 2026-08-03** (`012,027,039–044`);
     see §1b + `data/reference/generalization_new_vessels.json`. Six are
     clot-rich (incl. aneurysm `040`/`043`); `027` is clot-free; `039` is
     half-finished. Remaining gap: new **flow regimes** (different Re / inlet
     speed) — the Re=450 monoculture is still our biggest OOD exposure. Prefer
     more clot-rich full-length sims; avoid adding more zero-clot tubes.
   - **Safe augmentation = axis mirror only.** 2D incompressible Navier–Stokes is
     invariant under reflection `y -> -y` if we also flip `v_y -> -v_y`,
     `wall_normal_y -> -wall_normal_y`, keep pressure and all scalar species
     unchanged, and keep the inlet along x. This doubles the dataset with **zero**
     change to the underlying physics. Do **not** use rotation or anisotropic
     scaling — they change the inlet-BC orientation or the effective Re and would
     corrupt the physics.

**UPDATE (2026-07-30): Wall Generalization Legs Implemented**
We have implemented 8 specific experimental legs in `src/biochem_gnn/mat_growth_simple.py` to target the wall generalization bottleneck, split into two families:

**Family 1: Training Dynamics** (Attacking the 0.90 teacher-forced vs 0.25 autonomous rollout gap)
- `WG_sched_sample`: Scheduled sampling. Uses noisy-GT anchoring that ramps down over training, stabilizing early long-window training while converging to fully autoregressive by late epochs.
- `WG_noise_boost`: Amplifies per-step input noise and teacher blur to train error-recovery robustness.
- `WG_long_tbptt`: Longer TBPTT tail (15) and higher max unroll (120) to push gradient horizons.
- `WG_dynamics_all`: The full dynamics stack combined.

**Family 2: Conditioning + Data** (Attacking *where* to predict)
- `WG_mirror_y`: Y-axis mirror augmentation (exact N-S symmetry) to double the effective dataset. (Pre-cached via `scripts/precache_mirror_y.py`).
- `WG_geom_rich`: Enables static 2-hop geometry discriminators.
- `WG_flux_stag`: Adds the new `flux_stag` nucleation prior channel (low shear + high residence).
- `WG_full_stack`: The full stack (dynamics + conditioning + mirror) combined.

All legs inherit the `drop-xy`, `kine flow`, and `wall-mat-only` baseline configurations. You can run them via the new `scripts/go_wall_gen_probe.ps1 -Leg <LegName>` launcher.

**UPDATE (2026-07-31): 30-Leg Hyperparameter Sweep — Results & Lessons**

Ran 15 of 30 `WG_sweep_*` legs (cold start, 30 epochs, 5-vessel cohort, `--no-init`).
All legs had `SPECIES_SCHEDULED_SAMPLING=1` enabled from the start, varying
`SS_TARGET_PROB` (0.1–0.3), `SS_WARMUP_EPOCHS` (3–8), `TEACHER_NOISE` (0.02–0.08),
`TEACHER_BLUR` (0.1–0.5), and `GEOM_FEATS_RICH` (0/1).

**Key Result: ALL legs scored 0.06–0.14 deploy_clot_score — far below the 0.30 cold
baseline achieved without scheduled sampling.** Scheduled sampling on a cold-start model
with only 30 epochs prevents the model from learning basic growth dynamics at all
(`val_growth_f1 = 0.000` for all 30 epochs in every run).

**Ranked leaderboard (mean deploy_clot_score across patient020 + patient034):**

| Rank | Leg | Score | F1 | ss_prob | warmup | noise | blur | geom_rich |
|------|-----|-------|----|---------|--------|-------|------|-----------|
| 1 | WG_sweep_09 | 0.135 | 0.156 | 0.3 | 3 | 0.02 | 0.1 | 1 |
| 2 | WG_sweep_06 | 0.128 | 0.162 | 0.1 | 3 | 0.02 | 0.25 | 1 |
| 3 | WG_sweep_02 | 0.127 | 0.156 | 0.1 | 3 | 0.08 | 0.5 | 1 |
| 4 | WG_sweep_07 | 0.122 | 0.158 | 0.1 | 8 | 0.02 | 0.25 | 1 |
| 5 | WG_sweep_12 | 0.085 | 0.127 | 0.3 | 3 | 0.04 | 0.25 | 0 |
| ... | (10 more) | 0.058–0.084 | ... | ... | ... | ... | ... | 0 |

**Signals extracted despite the low absolute scores:**

1. **`GEOM_FEATS_RICH=1` is the clearest winner.** The top 4 legs all have it on.
   Mean score with geom_rich=1: **0.094** vs geom_rich=0: **0.078** (n=8 each). On
   patient034 (clot-free), geom_rich legs produce **fewer FPs** (median 16–22 vs
   36–61), suggesting geometry features help the model learn *where not* to predict.

2. **Lower teacher noise (0.02) is better.** Mean 0.100 (n=6) vs 0.04: 0.078 (n=4)
   vs 0.08: 0.085 (n=5). Cold-start models are too fragile for heavy noise injection.

3. **Shorter warmup (3 epochs) is better than 8.** Mean 0.097 vs 0.079. With only
   30 epochs total, 8 warmup epochs waste too much of the budget on full teacher
   forcing before the SS ramp even starts.

4. **SS target prob 0.1 is mildest and best.** Mean 0.102 (n=6) vs 0.3: 0.083 (n=7).
   Lower SS intensity = less disruption to a barely-learning model.

5. **Blur is not strongly discriminative** (0.1: 0.084, 0.25: 0.096, 0.5: 0.087).

6. **Two distinct failure modes on patient020 vs patient034:**
   - **geom_rich=1 legs**: low overpaint (2–3%), low FP (16–30), but *also* low
     recall — the model is conservative. Score comes from restraint on 034.
   - **geom_rich=0 legs**: high overpaint (20–40%), high FP (36–61) — spray. The
     score is dragged down by 034 false positives.

**CRITICAL LESSON FOR FUTURE WORK:**
> Scheduled sampling must NOT be applied to cold-start models at 30 epochs. It
> should only be used as a **fine-tuning phase** after the model has already learned
> growth dynamics (val_growth_f1 > 0). For the next training dynamics sweep, either
> (a) warm-start from a converged cold baseline, or (b) train for 80+ epochs with
> SS disabled for the first 40, then ramp.

**TWO-PHASE SWEEP PLAN:**

**Phase 1: Architecture / Conditioning Sweep (next GPU session)**
Goal: find the best feature set for the wall model, holding training dynamics fixed
(no scheduled sampling, standard teacher forcing). Cold start, 30 epochs, same
5-vessel cohort. Compare against the 0.30 cold baseline. Axes to sweep:
- `SPECIES_GEOM_FEATS_RICH` on/off (strong signal from sweep above)
- `SPECIES_FLUX_STAG_FEAT` on/off (residence-time prior, untested)
- `WG_mirror_y` on/off (data augmentation via y-axis reflection)
- Teacher noise level (0.0, 0.02, 0.04) — no SS, just input perturbation
- Combinations thereof

**Phase 2: Training Dynamics Sweep (subsequent session)**
Goal: improve rollout stability on the best conditioning from Phase 1. Warm-start
from the best Phase 1 checkpoint. Axes to sweep:
- Scheduled sampling (with long warmup, gentle ramp)
- TBPTT tail length (5 vs 10 vs 15)
- Max unroll (200 vs 120)
- Number of epochs (50–100)
- Closed-loop coupling strength

### Tier 2 — squeeze the growth specialist (what we've been doing)
5. FN / missed-mass reweighting, underpred tilt, two-stage training. Worth ~0.1 on lumen recall **within** the wall-limited band, but capped by Tier 1. Only worth another long run once the wall A-floor on clot-rich held-outs is known and the eval set is fixed. Train the specialist against clot-rich anchors; do not let low/zero-clot vessels dominate the val objective.

### Backup Plan — Architecture Pivot (Physics-Biased GAT)
An architectural swap to a Graph Attention Network (GAT) is kept strictly as a **backup alternative**. The current GraphSAGE model achieves >0.90 strict F1 on teacher-forced 1-step predictions, proving the architecture capacity is not the bottleneck. If we do pivot to a GAT, it must be a **Physics-Biased GAT** (where attention logits are conditioned on the dot product of edge vectors and flow velocity) to explicitly teach the model a stable upwind scheme for advection. We will not do this until we have exhausted the conditioning and training-dynamic fixes.

---

## 5. Realistic expectation

- On the **clot-rich held-outs** (021/032/035/037), after wall LOVO + conditioning
  fixes: **~0.45–0.6 is a realistic, defensible win**; **>0.6** is achievable if
  drop-xy + geom-rich + residence measurably lift the wall A-floor and the
  specialist recovers off-wall mass on 032.
- **>0.7 cohort-wide** likely needs the +10 vessels to include **new flow regimes**
  (breaking the Re=450 monoculture) plus the wall LOVO gains — it is a
  data+wall program, not a growth-only night and not an architecture swap.
- Judge success against **C0 on the clot-rich held-outs** (beat 032 under-recall,
  no spray on 009), not against 007's in-family ~0.93.

---

## 6. Artifacts

- EDA: `scripts/eda_generalization.py` -> `outputs/biochem/eda/generalization_eda.json`
  (re-run as new vessels land).
- New-vessel inventory (2026-08-03): `data/reference/generalization_new_vessels.json`
  + §1b; clot-rich code list `WALL_GEN_CLOT_RICH_ANCHORS`.
- Split builder: `scripts/build_generalization_splits.py` (extend with
  `family_validation` + `generalization_challenge` sets above).
- Wall/compound eval: `scripts/eval_mat_growth_simple.py`
  (`--two-model-route frontier_offwall --two-model-frontier-hops 0.5`).
- Sweep results: `outputs/biochem/eda/wall_gen/sweep_results.csv` and per-leg
  `eval_holdout_cold.json` files.
- New-anchor velocity QA: `outputs/biochem/viz/new_anchors/`.
