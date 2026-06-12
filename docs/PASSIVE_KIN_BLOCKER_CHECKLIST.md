# Passive / biochem tests while Stage-A kinematics is blocked

**Full roadmap (milestones, X/Y/XY tables, Phase I vs II):** [BIOCHEM_TRAINING_PLAN.md](BIOCHEM_TRAINING_PLAN.md).

Use **`BIOCHEM_GT_KINE_VEL=1`** + **`BIOCHEM_GT_KINE_SKIP_DEQ=1`** for all teacher work below. Do **not** wait on `kinematics_best.pth` quality for these probes.

## Already exercised (keep as reference ckpts, do not re-run blindly)

| Area | Status | Artifact / note |
|------|--------|-----------------|
| M3 union mask + `transport_only` ADR | Pass (20ep) | `biochem_teacher_passive_align_locked.pth`, §126 |
| Mu-unlock probe (`MU_LOG`, delta head) | Pass bulk mu ~0.80 | `20260529T200500Z`; species OK |
| Mu-unlock finetune (wall/high weights) | Val noop | Plateau; species OK |
| Step-2 bridge from unlock init | Misconfigured | Preset + `post_pretrain` clobber; redo on explore base |
| **Explore 6h** (`go_passive_explore_6h.ps1`) | **Done** | §130: X clot-band = species OK, m3 gate false FAIL; **X_mask_global** trains; **XY_mu_unlock** PASS |
| Phase A X (species recipes) | Pass | `phaseA_X_fi2mat2`, §107; explore confirms saturated-init caveat |
| Phase A Y (isolated terms) | **Pass** (explore) | ADR_S/F OK; MU_LOG OK; W_BIO/W_PHY gate WARN |
| Phase B ramp1/ramp2 | Done | `phaseB_XY_ramp1_data`, §116–117; explore ramp legs species OK, m3 WARN |
| GT-flow species ladder + clot-phi | Ongoing | `go_gt_flow_*`, separate track |

## Still to test (kin-independent)

### X — species / data bio (substance)

| ID | What | Why |
|----|------|-----|
| X1 | M3 union + `PASSIVE` 10–20ep confirm | Canonical species baseline |
| X2 | `LOSS_ISOLATE=DATA_BIO` vs `PASSIVE` | Pure species backward |
| X3 | Mask: `clot_band` vs `global` vs `union` supervision | Mask scope sensitivity |
| X4 | FI/Mat weights (2/2 vs 3/2 vs 1/1) | Gradient balance |
| X5 | Train-anchor species eval every epoch | Catch val-only blind spots |
| X6 | Seed / init sensitivity (locked align) | Reproducibility |

### Y — isolated physics / mu terms

| ID | What | Why |
|----|------|-----|
| Y1 | `ADR_S` + `transport_only` + matched mask | Core passive ADR story |
| Y2 | `ADR_F` isolate | Fast transient path |
| Y3 | `MU_LOG` + frozen bio + `USE_DELTA_MU_HEAD` | Mu without species drift |
| Y4 | `MU_LOG` + wall/high weights | Wall/high recovery |
| Y5 | `W_BIO` / `W_PHY` / `BIO_IO` | Wall flux objectives |
| Y6 | `ADR` residual formulation ladder | `transport_only` vs `log` (see m3n) |

### XY — combinations (after X and Y singles look OK)

| ID | What | Why |
|----|------|-----|
| XY1 | `LOSS_DATA_ONLY` + modest `MU_LOG`/`MU_SI` (true bridge) | Joint step-2 without step 3 |
| XY2 | X + low-weight `PASSIVE_ADR` in backward | Species + masked ADR co-descent |
| XY3 | Mu-unlock probe -> fixed bridge | Preserve ~0.80 mu + species |
| XY4 | Phase B ramp1 (data only) -> ramp2 (+ADR) | Scheduled co-train |
| XY5 | `COMPLEXITY_STEP=2` + gelation off vs delta head | Forward policy choice |

### Downstream (teacher quality gating)

| ID | What | Blocked on |
|----|------|------------|
| D1 | `dump_teacher_species_to_anchors` + clot-phi legs | Stable X teacher |
| D2 | Clot-phi F1 multi-anchor | Species + optional mu band |
| D3 | GT-flow round2/3 ladders | Clot-phi + teacher |
| D4 | Biochem **corrector** stage | Stable teacher + pseudo bank |
| D5 | Step 3 multitask (`COMPLEXITY_STEP=3`) | Step-2 joint stable |

### Explicitly blocked until kin is ready

| Item | Reason |
|------|--------|
| `BIOCHEM_TRAIN_KIN_LORA=1` / DEQ velocity training | Needs Stage-A |
| Teacher without `GT_KINE_VEL` | Couples mu/species to bad `[u,v,p]` |
| Full pipeline `STOP_AFTER_TEACHER=0` at scale | Corrector assumes teacher |
| Kinematics geometry curriculum (L0/L1/L2) | Separate Stage-A track |

## Metrics to log per leg

- Val: `mu_log_mae` (all / wall / high / bulk), `val_species_fi_log_mae`, `val_species_mat_log_mae`
- Train: `train_L_bio_avg`, masked `train_L_ADR_S_avg`, `passive_step2_bridge`, `passive_mu_unlock`
- Gates: `check_phase_a_gate.py` (X/Y), `check_m3_align_gate.py`, `check_passive_mu_unlock_gate.py`

## Orchestrated 6h sweep

See [scripts/go_passive_explore_6h.ps1](../scripts/go_passive_explore_6h.ps1) and `outputs/biochem/explore_6h/`.
