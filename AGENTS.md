# Agent notes (HemoRGP)

Short cheat sheet for agents and contributors. Full orientation: [docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md). Publishing policy: [docs/PUBLISHING.md](docs/PUBLISHING.md).

## Canonical stacks

| Stack | Train | Docs |
|-------|-------|------|
| **RGP-DEQ** (`rgp_deq_kine`) | `python -m src.bin.main train rgp-deq-kine` or `scripts/go_kinematics_*.ps1` | [docs/KINEMATICS_BEST_ARCHITECTURE.md](docs/KINEMATICS_BEST_ARCHITECTURE.md) |
| **biochem_gnn** | `scripts/go_biochem_gnn.ps1` or `python -m src.bin.main train biochem-gnn` | [docs/BIOCHEM_GNN.md](docs/BIOCHEM_GNN.md) |
| Mat-growth (research) | `go_mat_*.ps1`, `go_wc_v7_*.ps1` | [docs/MAT_GROWTH.md](docs/MAT_GROWTH.md) |
| Local corrector | `python -m src.training.train_local_kinematic_corrector` | [docs/LOCAL_KINEMATIC_CORRECTOR.md](docs/LOCAL_KINEMATIC_CORRECTOR.md) |

- **Promote biochem:** `python scripts/promote_biochem_gnn.py` → `outputs/biochem/biochem_gnn/locked/` + `data/reference/biochem_gnn_baseline.json`
- **Locked mat baseline:** `WC_v7_clot_phi_mse` (2026-07-19); cohort clot F1 **~0.767**, clot score **~0.791**
- **Customer UI:** `scripts/go_customer_predict.ps1`
- **Import:** `from src.biochem_gnn import BiochemGNN` (alias package `src.biochem_deploy`)

## Kinematics (Stage A)

- Production allfix: `go_kinematics_production_allfix.ps1` — Rel L2 **~0.087** after continuity finetune
- Manifest: [data/reference/kinematics_best_20260426T184600Z.json](data/reference/kinematics_best_20260426T184600Z.json)
- Config helpers: `snapshot_rgp_deq_model_config` / `resolve_rgp_deq_ctor_kwargs` in `src/architecture/kinematics_model_config.py` (gino/pmgp aliases retained)

## Scripts

- Active only: [scripts/README.md](scripts/README.md)
- Retired: `scripts/archive/` — do not revive GNODE / clot-ML / T0 trainers without restoring modules from git ([docs/BIOCHEM_LEGACY_LESSONS.md](docs/BIOCHEM_LEGACY_LESSONS.md))

## Console (PowerShell)

No emoji in `print` / launcher banners. Use ASCII tags (`[OK]`, `[WARN]`, `[i]`). See [.cursor/rules/powershell-console-ascii.mdc](.cursor/rules/powershell-console-ascii.mdc).

## Eval / off-wall

- Persist `leg` and `env_overrides` in checkpoint `meta`
- On eval, restore `meta.get("env_overrides")` before metrics
- Off-wall: hop >= 1 helpers (`deploy_clot_offwall_relaxed_f1`, …)

## Hardware

Training entry points should call `require_cuda_device()` and fail loud without CUDA.

## Historical training chronicles

Biochem corrector / GNODE run logs live under [docs/archive/BIOCHEM_TRAINING_PROGRESS.md](docs/archive/BIOCHEM_TRAINING_PROGRESS.md) (archive only).
