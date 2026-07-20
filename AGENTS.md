# Agent notes (HemoGINO)

## Canonical biochem stack (active)

- **Stack id:** `biochem_deploy` (implementation package `src/biochem_gnn/`, import alias `src.biochem_deploy`).
- **Train:** `powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_biochem_gnn.ps1"` or `python -m src.bin.main train biochem-deploy`
- **Promote / lock:** `python scripts/promote_biochem_gnn.py` → `outputs/biochem/biochem_gnn/locked/` + `data/reference/biochem_gnn_baseline.json`
- **Docs:** [docs/BIOCHEM_GNN.md](docs/BIOCHEM_GNN.md), [docs/MODEL_NOMENCLATURE.md](docs/MODEL_NOMENCLATURE.md)
- **Mat-growth canonical baseline:** **WC_v7_clot_phi_mse** promoted 2026-07-19. Source of truth: `outputs/biochem/biochem_gnn/locked/species_gnn_best.pth`. Aliases: `mat_canonical_deploy/species/best.pth`, `species/best.pth`. Cohort mean clot score **~0.791**, clot F1 **~0.767**. Launchers: `go_fresh_canonical.ps1`, `go_mat_w_wc_canonical.ps1`, `go_mat_growth_simple.ps1`, `go_mat_growth_ladder.ps1`. Doc: [docs/MAT_GROWTH_SIM_TODO.md](docs/MAT_GROWTH_SIM_TODO.md).
- **Off-wall pivot (2026-07-05):** only Pivot 3 (`BIOCHEM_ROLLOUT_DYNAMIC_OCCLUSION=1`) survived; launcher `go_off_wall_clot_sweep_6h.ps1`. Viz: `go_viz_pivot3_hop_analysis.ps1`.
- **Deploy viz:** `scripts/viz_species_gnn_deploy.py` / `go_species_gnn_deploy_viz.ps1`. Steady kinematics + deploy smoke: `python -m src.evaluation.visualize_pipeline` (GNODE teacher temporal inspector **retired**).
- **Compound A/B/C:** `go_wc_v7_compound_growth_abc_9h.ps1` — expect **~20–26 h** full pipeline with all on-disk anchors (~8–10 h/specialist). Arm B trained 2026-07-20; resume A-vs-B with `-EvalOnly -SkipC` (~2–6 h).
- **Customer predict:** `go_customer_predict.ps1`
- **A/B helpers:** `go_biochem_gnn_arch_ab.ps1`, `go_biochem_gnn_gate_ab.ps1`, `check_biochem_gnn_gate.py`

## Retired ladders (do not revive without restoring modules)

Removed from the active script surface (2026-06 cleanup + 2026-07 legacy trim). Recover from git history if needed; index: [docs/archive/2026-06-16-biochem-cleanup.md](docs/archive/2026-06-16-biochem-cleanup.md), lessons: [docs/BIOCHEM_LEGACY_LESSONS.md](docs/BIOCHEM_LEGACY_LESSONS.md).

| Family | Former entry points | Deleted core |
|---|---|---|
| GNODE teacher/corrector | `go_k*`, `go_visc3h`, `go_health10h`, `go_mu_complexity_6h`, `go_passive*`, `go_gnode*` | `train_biochem_corrector`, `gnode_biochem`, `biochem_teacher_loader` |
| Clot-phi GNODE MLP | `go_clot_phi_*`, `go_rung6*`, `go_mlp_*` | `train_clot_phi_simple` |
| Clot-ML rule ladder | `train_clot_ml_*`, `go_sweep_clot_*` | `clot_ml_device`, `clot_ml_step0_coef`, … |
| Clot forecast / T0 / snapshot | `go_clot_forecast*`, `go_t0*`, `go_species_snapshot*` | matching train/eval modules |
| Empty husk | `src/clot_deploy_gnn/` | removed (use `biochem_gnn` / `biochem_deploy`) |

Historical run narrative still lives in [docs/BIOCHEM_TRAINING_PROGRESS.md](docs/BIOCHEM_TRAINING_PROGRESS.md) and [docs/BIOCHEM_TRAINING_PLAN.md](docs/BIOCHEM_TRAINING_PLAN.md) — treat launcher paths there as archive unless they match `scripts/README.md`.

## Kinematics (Stage A)

- **Foundation:** `go_kinematics_foundation.ps1 -Fresh`
- **L2 finetune:** `go_kinematics_l2_finetune.ps1`
- **Production allfix:** `go_kinematics_production_allfix.ps1` — best Rel L2 **0.087** @ ep 119 in `outputs/kinematics/production_allfix/kinematics_best.pth`
- **Precision long / clinical / stage-A ladder:** `go_kinematics_precision_long.ps1`, `go_kinematics_clinical_anchor_finetune.ps1`, `go_kinematics_stage_a_ladder.ps1`
- **Recovery sweep:** `go_kinematics_recovery12h.ps1`
- **Bend-sign A/B:** `go_kinematics_bend_ab.ps1`
- **Backfill geometry_level:** `python -m src.data_gen.backfill_kinematics_geometry_level`
- Doc + architecture: [docs/KINEMATICS_BEST_ARCHITECTURE.md](docs/KINEMATICS_BEST_ARCHITECTURE.md)
- Reference manifest: [data/reference/kinematics_best_20260426T184600Z.json](data/reference/kinematics_best_20260426T184600Z.json)
- Code: `snapshot_gino_deq_model_config` / `resolve_gino_deq_ctor_kwargs` in [src/architecture/kinematics_model_config.py](src/architecture/kinematics_model_config.py)

## Local kinematic corrector

- Doc: [docs/LOCAL_KINEMATIC_CORRECTOR.md](docs/LOCAL_KINEMATIC_CORRECTOR.md)
- Train: `python -m src.training.train_local_kinematic_corrector --epochs 600 --batch-size 4 --stride 2 --device cuda`
- Latest (2026-06-21): relative loss → global relL2 **15.9%**

## Console output (PowerShell)

- No emoji in training/script `print` output. Use ASCII tags (`[OK]`, `[WARN]`, `[i]`). See [.cursor/rules/powershell-console-ascii.mdc](.cursor/rules/powershell-console-ascii.mdc).

## Interactive kinematics demo

- `python -m src.tools.demo_kinematics_flow` or `python -m src.bin.main inspect flow -- --rheology carreau`

## Scripts layout

- Active launchers: [scripts/README.md](scripts/README.md)
- Biochem COMSOL extract: `python -m src.data_gen.lib.extract_biochem_comsol_data` (auto-pull from `comsol_models/phase2_nowound_XXX.mph`)

## Project orientation

- [docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md)

## Checkpoint evaluation (off-wall metrics)

- Save `leg` and `env_overrides` in checkpoint `meta` when training.
- On eval/rollout, restore `meta.get("env_overrides")` (or infer from leg path) before metrics.
- Off-wall metrics: Hop >= 1 via `deploy_clot_offwall_relaxed_f1` and related helpers.

## Hardware

- Enforce `require_cuda_device()` at training entry points (`train_species_pushforward_continuous.py`, etc.); fail loud if CUDA is unavailable.
