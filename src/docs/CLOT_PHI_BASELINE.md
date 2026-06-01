# Clot-phi simple probe — baseline record

Wall-local **clot phase** probe: capped GT `mu_eff`, GT kinematics for Carreau baseline, hybrid head (BCE on phi + log-mu regression). This is **not** the full biochem teacher/corrector; it is a **cheap, interpretable baseline** for mask design, feature ablations, and “can we localize clot before coupling the full ODE stack?”

## Role in the pipeline

| Layer | What it answers |
|-------|-----------------|
| **This probe** | Given GT `[u,v]` and a tight supervision shell, can a tiny head recover **where** clot is and a **local mu bump**? |
| **Passive transport (step 2a)** | Can we fit **Mat/FI ADR** on frozen GT velocity? |
| **Biochem teacher** | Can we roll out **species + mu_eff** with physics losses? |
| **Corrector** | Can we fix rollout drift on mixed graphs? |

Use clot-phi results to **lock the supervision mask and kinematic features** before betting on full multitask biochem. Promote configs here only when val metrics are stable (no predict-none / predict-all).

**Roadmap:** full viscosity/clot ladder (rungs 0–11, gates, launchers) — [BIOCHEM_TRAINING_PLAN.md](BIOCHEM_TRAINING_PLAN.md#viscosity--clot-localization-ladder-rungs-011).

## Data and mask (patient007-centric today)

- **Graphs**: `data/processed/graphs_biochem_anchors/*.pt` (default); val anchor `patient007`.
- **Mask**: `neighbor` (wall + clot seeds + 1-hop), `CLOT_PHI_CENTER_EXCLUDE_FRAC=0.10`, **`CLOT_PHI_DGAMMA_SLICE=1`** @ `t=0` (wall: GT clot OR `-dgamma/dx >= 100` SI; off-wall: 80th pct of `-dgamma/dx`).
- **Labels**: `mu_cap <= 0.10` Pa·s; soft phi from log-blend inversion optional (`CLOT_PHI_SOFT_LABELS=1`).
- **Features (minimal)**: `[sdf, log10(gamma_dot), log1p(-dgamma/dx_ref)]` — **no** FI/Mat in the default baseline (no species leak).

## Model ladder

| Step | Config | Launcher |
|------|--------|----------|
| Linear hybrid | `hidden=16`, `depth=1` | `go_clot_phi_simple.ps1 -Model linear` |
| **MLP baseline (production)** | `hidden=32`, `depth=2`, `dropout=0.15` | `go_clot_phi_simple.ps1 -Fresh` (default) |
| MLP sweep | 13 legs × 45 ep | `go_clot_phi_mlp_sweep.ps1` |

Env: `CLOT_PHI_MLP_DEPTH`, `CLOT_PHI_HIDDEN`, `CLOT_PHI_DROPOUT`, `CLOT_PHI_MU_LOG_LAMBDA`, `CLOT_PHI_LR`, `CLOT_PHI_WEIGHT_DECAY`.

## MLP sweep (2026-05-26, 45 ep/leg, patient007 val)

Summary: `outputs/biochem/sweep_clot_phi_mlp/summary.jsonl`

| Rank | Leg | score | F1 | rec | logMAE | Notes |
|------|-----|-------|-----|-----|--------|-------|
| 1 | **h32_d2** | 0.571 | 0.466 | 0.398 | 0.499 | **Winner** — wider + depth-2 |
| 2 | wd1e5 | 0.568 | 0.458 | 0.387 | **0.468** | Best logMAE; same F1 as baseline |
| 3 | baseline | 0.566 | 0.458 | 0.387 | 0.477 | h16, depth=1 |
| … | h8, drop0, lr5e4 | ≤0.564 | 0.458 | 0.387 | No gain on F1/rec |

**60-ep retrain of h32_d2** (promoted to `outputs/biochem/clot_phi_best.pth`):

| Metric | h16/d1 (prior 60 ep) | **h32/d2 (60 ep)** |
|--------|----------------------|---------------------|
| val score | 0.566 | **0.574** |
| F1 | 0.458 | **0.469** |
| precision | 0.575 | 0.575 |
| recall | 0.387 | **0.401** |
| logMAE | 0.477 | 0.495 |
| pred+ / gt+ | 0.217 / 0.345 | **0.227** / 0.345 |
| dice | 0.865 | **0.876** |

Healthy: `pred+` not collapsed; recall up without precision collapse.

## Artifacts

| File | Purpose |
|------|---------|
| `outputs/biochem/clot_phi_best.pth` | Best checkpoint (config embedded) |
| `outputs/biochem/clot_phi_best_mlp.pth` | Copy after train/retrain |
| `outputs/biochem/clot_phi_train_log.jsonl` | Per-epoch metrics |
| `outputs/biochem/clot_phi_viz_patient007.png` | Qualitative |
| `outputs/biochem/sweep_clot_phi_mlp/<leg>/` | Per-sweep-leg ckpt + log |

## How to cite / reproduce

```powershell
# Default baseline (60 ep, h32_d2)
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_phi_simple.ps1" -Fresh

# Sweep + promote winner
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_phi_mlp_sweep.ps1"
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_phi_mlp_sweep.ps1" -RetrainBest
python scripts/summarize_clot_phi_mlp_sweep.py
```

## Biology ladder (2026-05-26)

Launcher: `scripts/go_clot_phi_biology_ladder.ps1`  
Artifacts: `outputs/biochem/clot_phi_ladder/`

| Stage | Mode | patient007 val (best) | Notes |
|-------|------|------------------------|-------|
| **0 kinematic MLP** | h32/d2, 3 feat | F1 **0.469**, score 0.574 | [prior baseline] |
| **1a physics** | `mu_ratio_max=1` | F1 0.11, pred+ 0.05 | Gelation off; collapsed |
| **1b physics** | `mu_ratio_max=80` | F1 0.44, rec **0.59**, pred+ **0.82** | Global FI flood |
| **2 species feat** | +GT FI/Mat (5 feat) | F1 0.458 @ ep13 | No gain vs stage 0 |
| **3 joint bio** | species head + `L_Data_Bio` (SI) | F1 **0.474**, rec **0.41**, score **0.581** | Healthy; best overall |

Env flags: `CLOT_PHI_PHYSICS_ORACLE`, `CLOT_PHI_SPECIES_FEATURES`, `CLOT_PHI_JOINT_BIO`, `CLOT_PHI_BIO_LAMBDA`, `CLOT_PHI_PHYSICS_MU_RATIO_MAX`.

**Production default after round-2:** `joint_blend_gtsp` -> `clot_phi_best.pth`.

## Round-2 biology (2026-05-26)

Launcher: `scripts/go_clot_phi_biology_round2.ps1`

### Physics oracle sweep (GT species, no train)

| Config | F1 | rec | pred+ | score |
|--------|-----|-----|-------|-------|
| **mu_ratio=4, no gate** | **0.478** | 0.47 | 0.33 | **0.531** |
| mu_ratio=8, gate+cap2 | 0.409 | 0.32 | 0.19 | 0.522 |
| mu_ratio=80 | 0.442 | 0.59 | 0.82 | 0.44 (flood) |
| mu_ratio=1 | 0.11 | 0.06 | 0.05 | -1 |

Use **`CLOT_PHI_PHYSICS_MU_RATIO_MAX=4`** for clot-scale gelation (not 80).

### Learned models (patient007 val, 60 ep)

| Leg | F1 | rec | pred+ | score |
|-----|-----|-----|-------|-------|
| kinematic h32/d2 (r1) | 0.469 | 0.40 | 0.23 | 0.574 |
| joint_bio (r1) | 0.474 | 0.41 | 0.23 | 0.581 |
| **joint_blend_gtsp (r2)** | **0.480** | **0.42** | **0.24** | **0.584** |
| joint_pred (r2) | 0.471 | 0.40 | 0.23 | 0.579 |

**Winner recipe (`joint_blend_gtsp`):** GT species features + joint species head (`L_Data_Bio`) + physics blend (`mu_ratio_max=4`, prior gate, alpha=0.55) for phi/mu mix during training.

`joint_blend` (pred species + gate) over-predicted (pred+ 0.31) — GT species for physics branch is safer until passive transport exists.

## Next experiments

1. `go_passive_transport.ps1` then re-run `joint_blend` with `CLOT_PHI_JOINT_USE_PRED_SPECIES=1`.
2. Physics-only deploy path (no MLP) at `mu_ratio_max=4` for sanity / ablation.

## Full-horizon anchor refresh + fast teacher dump (2026-05-27)

- Re-extracted anchors with full COMSOL horizon (`BIOCHEM_EXTRACT_FULL_TIME=1`), then dumped teacher species with `--time-stride 36` and heartbeat logging.
- Effective rollout lengths after stride: patient001/006/007 `T=6`, patient002/004 `T=2`, patient003 `T=1`.
- Training on these teacher-species anchors was **healthy on patient007** (val F1 climbs to **0.631**, logMAE to **0.687**).

### Single-anchor train result (patient007 val)

| Metric | Best |
|--------|------|
| val F1 | **0.631** |
| val precision | 0.831 |
| val recall | 0.509 |
| val pred+ / gt+ | 0.357 / 0.580 |
| val logMAE | **0.687** |
| val score | **0.708** |

### Multi-anchor evaluation (`outputs/biochem/multi_anchor_eval_teacher_species.jsonl`)

| Anchor | F1 | Recall | pred+ | logMAE | score |
|--------|----|--------|-------|--------|-------|
| patient001 | 0.242 | 0.584 | 0.690 | 2.196 | 0.085 |
| patient002 | 0.207 | 0.956 | 0.819 | 2.267 | -1.000 |
| patient003 | 0.192 | 1.000 | 0.866 | 2.321 | -1.000 |
| patient004 | **0.055** | 0.933 | 0.839 | **2.401** | -1.000 |
| patient006 | 0.232 | 0.982 | 0.837 | 2.301 | -1.000 |
| patient007 | **0.418** | 0.788 | 0.749 | **1.810** | 0.146 |

Aggregate: `mean_f1=0.224`, `min_f1=0.055`, `mean_logMAE=2.216`, `mean_score=-0.628`.

### Interpretation

- This run **does not pass multi-anchor health** despite strong patient007.
- Failure mode is consistent across 4/6 anchors: **predict-all tendency** (`pred+ ~0.82-0.87`, recall ~0.93-1.00, very high logMAE), indicating teacher-species rollout quality/domain mismatch remains the dominant issue.
- Next action remains Step 2: improve passive transport teacher species quality (especially wall-band FI/Mat), then re-dump/retrain before architecture sweeps.

## Step-2 rerun (2026-05-27, passive finetune -> re-dump -> retrain)

- Passive transport finetune run: `go_passive_transport_finetune.ps1` (`run_id=20260527T110533Z`, 6 teacher epochs, resume+best init, GT velocity mode).
- Teacher metrics were effectively **flat** vs prior passive run:
  - val all-truth `mu_log_mae ~1.3966` at ep0/2/4/5,
  - wall `~2.25`, high-μ `~1.17`,
  - train `L_bio` decreased but no val μ improvement signal.
- Re-dump (`--time-stride 36`) and clot-phi retrain reproduced the same patient007 behavior:
  - best val `F1=0.631`, `score=0.708`, `logMAE~0.691`.

### Multi-anchor after Step-2 (`multi_anchor_eval_teacher_species_after_step2.jsonl`)

| Anchor | F1 | Recall | pred+ | logMAE | score |
|--------|----|--------|-------|--------|-------|
| patient001 | 0.242 | 0.584 | 0.690 | 2.188 | 0.085 |
| patient002 | 0.211 | 0.956 | 0.801 | 2.285 | -1.000 |
| patient003 | 0.195 | 1.000 | 0.850 | 2.311 | -1.000 |
| patient004 | **0.056** | 0.933 | 0.825 | **2.344** | -1.000 |
| patient006 | 0.224 | 0.943 | 0.823 | 2.318 | -1.000 |
| patient007 | **0.416** | 0.771 | 0.731 | **1.785** | 0.166 |

Aggregate: `mean_f1=0.224`, `min_f1=0.056`, `mean_logMAE=2.205`, `mean_score=-0.625`.

### Conclusion

- Step-2 rerun produced **no material multi-anchor health gain**.
- Dominant failure remains over-prediction on non-007 anchors (`pred+ ~0.80-0.85`), consistent with teacher-species domain mismatch.
- Before architecture sweeps, the largest lever is still improving species rollout quality (especially wall-band Mat/FI calibration) rather than clot-head capacity.

## Quick iterate probe (2026-05-27, pre-overnight)

Launcher: `scripts/go_clot_phi_quick_iterate.ps1` (short legs only, no long sweep).

### Leg A — oracle_gt (GT species ceiling, 25 ep)

- Checkpoint/eval: `outputs/biochem/quick_iterate/oracle_gt/multi_anchor.jsonl`
- Aggregate: `mean_f1=0.559`, `min_f1=0.206`, `mean_logMAE=0.500`, `mean_score=0.329`
- Per-anchor highlights:
  - patient001 `F1=0.580`, `logMAE=0.983`
  - patient004 `F1=0.206`, `logMAE=0.437` (still weakest F1)
  - patient007 `F1=0.733`, `logMAE=0.652`

Interpretation: with GT species, clot-phi generalization is healthy across anchors; current failure is still species-rollout quality, not clot-head capacity.

### Leg B — passive_tf08 (teacher-only 4 ep + dump, quick check)

- Passive teacher metrics remained flat during quick pass (`all logMAE~1.3966`, `wall~2.25`, `high~1.17`), matching prior Step-2 reruns.
- No evidence of species-quality movement in this short leg before dump completion; stopped this iterate pass and did not promote a new clot-phi checkpoint from this branch.

Decision: keep using short probes to validate direction, but do not expect meaningful movement from very short passive legs unless val species/μ subsets move first.

## Passive species-focus A/B (2026-05-27)

Goal: test whether focusing passive teacher backward on global species (`L_Data_Bio`) helps downstream clot decisions in the wall-local mask.

Recipe (both legs):
- Passive teacher only (GT velocity, `PRESET=passive_transport`, `TeacherEpochs=8`, val every epoch).
- Re-dump species cache (`time-stride=36`), retrain clot-phi (`20` epochs), run multi-anchor eval.

### Leg comparison

| Leg | Passive weights | Teacher val μ (all/wall/high) | Multi-anchor mean F1 | min F1 | mean logMAE |
|-----|------------------|-------------------------------|----------------------|--------|-------------|
| `lbio_on` | `PASSIVE_DATA_BIO_WEIGHT=1.0`, `DATA_KINE=0.25` | ~`1.3966 / 2.25 / 1.171` (flat) | `0.290` | `0.000` | `0.646` |
| `lbio_off` | `PASSIVE_DATA_BIO_WEIGHT=0.0`, `DATA_KINE=0.25` | ~`1.3966 / 2.25 / 1.171` (flat) | `0.290` | `0.000` | `0.659` |

Outputs:
- `outputs/biochem/passive_species_focus_compare/lbio_on/multi_anchor.jsonl`
- `outputs/biochem/passive_species_focus_compare/lbio_off/multi_anchor.jsonl`

Interpretation:
- Turning global `L_Data_Bio` on/off in this passive setup produced almost identical downstream clot behavior.
- This supports the diagnosis that what matters is not global species fit, but species calibration in the wall-relevant clot decision region.

## True clot-band species loss (2026-05-27)

Change:
- Implemented `BIOCHEM_DATA_BIO_MASK_MODE=clot_band` in `train_biochem_corrector.py` so `L_Data_Bio` is computed on the clot-phi decision region mask (wall+dgamma slice neighborhood) instead of all anchor nodes.

Quick run (teacher 6 ep, dump stride 36, clot-phi 12 ep):
- Teacher run note: `passive_transport_clotband_focus`
- Output eval: `outputs/biochem/passive_species_focus_compare/clotband_focus/multi_anchor.jsonl`

| Anchor | F1 | Recall | pred+ | logMAE | score |
|--------|----|--------|-------|--------|-------|
| patient001 | 0.541 | 0.402 | 0.327 | 1.093 | 0.557 |
| patient002 | 0.235 | 0.182 | 0.035 | 0.542 | -1.000 |
| patient003 | 0.000 | 0.000 | 0.013 | 0.492 | 0.122 |
| patient004 | 0.000 | 0.000 | 0.014 | 0.473 | 0.128 |
| patient006 | 0.410 | 0.298 | 0.080 | 0.562 | 0.505 |
| patient007 | 0.553 | 0.415 | 0.291 | 0.800 | 0.613 |

Aggregate: `mean_f1=0.290`, `min_f1=0.000`, `mean_logMAE=0.660`, `mean_score=0.154`.

Interpretation:
- Compared to `lbio_on` global species A/B (`mean_f1=0.290`, `mean_logMAE=0.646`), clot-band masking changed behavior distribution by anchor (less predict-all on several non-007 anchors) but did not yet lift aggregate mean/min F1.
- This confirms the implementation is active and materially changes supervision locality, but stronger teacher-side signal (longer run / FI-Mat channel emphasis inside the same mask) is still needed.

## Adaptive time-coverage fix (2026-05-27)

Hypothesis:
- Fixed dump stride (`--time-stride 36`) was collapsing hard anchors to very short trajectories (`T=1`/`2`), starving dynamics where clot decisions are made.

Implementation:
- `scripts/dump_teacher_species_to_anchors.py` now supports `--min-steps` and always keeps the final timestep.
- New dump used: `--time-stride 36 --min-steps 4`.
- Effective rollout lengths changed from `[6,2,1,2,6,6]` to `[7,4,5,5,7,7]` for patients `[001,002,003,004,006,007]`.

Run:
- Anchors: `outputs/biochem/passive_species_clotband_focus/anchors_clotband_adapt`
- Clot-phi leg: `clotband_adapt_tcov` (16 epochs)
- Eval: `outputs/biochem/passive_species_focus_compare/clotband_adapt_tcov/multi_anchor.jsonl`

| Anchor | F1 | Recall | pred+ | logMAE | score |
|--------|----|--------|-------|--------|-------|
| patient001 | 0.551 | 0.408 | 0.340 | 1.154 | 0.558 |
| patient002 | 0.333 | 0.257 | 0.104 | 0.557 | 0.430 |
| patient003 | 0.286 | 0.226 | 0.094 | 0.507 | 0.390 |
| patient004 | 0.262 | 0.200 | 0.125 | 0.564 | 0.358 |
| patient006 | 0.512 | 0.413 | 0.133 | 0.543 | 0.611 |
| patient007 | 0.582 | 0.441 | 0.312 | 0.817 | 0.639 |

Aggregate:
- `mean_f1=0.421`, `min_f1=0.262`, `mean_logMAE=0.690`, `mean_score=0.498`.

Interpretation:
- This is a major stability gain over prior clot-band run (`mean_f1=0.290`, `min_f1=0.000`): failing anchors `003/004` no longer collapse to predict-none.
- Time-coverage skew was a primary failure driver; adaptive dump coverage is now a strong default candidate for future teacher-species legs.

## Clot-phi on teacher species: anchor balance + FI/Mat weights (2026-05-27)

Same anchors (`anchors_clotband_adapt`, `--min-steps 4`), 35 ep, `CLOT_PHI_ANCHOR_BALANCED=1`, `CLOT_PHI_BIO_FI_WEIGHT=2`, `CLOT_PHI_BIO_MAT_WEIGHT=2`:

- Leg: `clotband_adapt_balfi2`
- Eval: `outputs/biochem/passive_species_focus_compare/clotband_adapt_balfi2/multi_anchor.jsonl`
- Aggregate: **`mean_f1=0.510`**, **`min_f1=0.338`**, `mean_logMAE=0.632`

## 7h hardening orchestrator (2026-05-28)

Launcher: `scripts/go_7h_passive_clot_hardening.ps1` (~1.1h wall on this machine; log `outputs/biochem/7h_hardening/run_log.jsonl`).

| Step | Result |
|------|--------|
| FI/Mat sweep (12 ep, adapt cache) | Best **FI=3.0**, **Mat=2.0** by `min_f1` |
| Passive teacher 14 ep (`clot_band`) | Val mu flat (**1.3966** all); species cache refreshed |
| Species dump `--min-steps 8` | All anchors **T=8-9** (vs adapt **T=4-7**) |
| Clot-phi 35 ep on new dump (`7h_final`) | **`mean_f1=0.506`**, **`min_f1=0.247`** (regressed vs balfi2 on 003/004) |
| Threshold branch (0.040-0.050, no retrain) | **`min_f1~0.099`** predict-all artifact — not usable |

**Recovery** (same adapt cache + sweep weights FI=3/Mat=2, 35 ep):

- Leg: **`recovery_adapt_fi30`** (promoted best multi-anchor clot-phi on teacher species today)
- Checkpoint: `outputs/biochem/passive_species_focus_compare/recovery_adapt_fi30/clot_phi_best.pth`
- Eval: `outputs/biochem/passive_species_focus_compare/recovery_adapt_fi30/multi_anchor.jsonl`
- Aggregate: **`mean_f1=0.526`**, **`min_f1=0.341`**, `mean_logMAE=0.591`, `mean_score=0.617`

| Anchor | F1 | Recall | pred+ | logMAE |
|--------|----|--------|-------|--------|
| patient001 | 0.551 | 0.408 | 0.340 | 1.142 |
| patient002 | 0.434 | 0.393 | 0.146 | 0.455 |
| patient003 | 0.358 | 0.349 | 0.174 | 0.417 |
| patient004 | 0.341 | 0.305 | 0.183 | 0.433 |
| patient006 | 0.811 | 0.856 | 0.181 | 0.423 |
| patient007 | 0.660 | 0.538 | 0.381 | 0.676 |

Interpretation:
- **Promote `recovery_adapt_fi30`** for cross-anchor clot-phi on passive teacher species until a new dump beats **`min_f1>=0.34`** on multi-anchor eval.
- Longer passive teacher + denser temporal dump did **not** improve species quality for 003/004; keep **`anchors_clotband_adapt`** (`--min-steps 4`) as default species cache.
- Oracle GT species multi-anchor was ~**0.56** mean F1 — head capacity exists; bottleneck remains **teacher species**, not mask geometry.
- Threshold-only eval without retrain is misleading; staged branch (`min_f1>=0.38`) was correctly skipped.
