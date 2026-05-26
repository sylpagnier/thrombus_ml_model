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
