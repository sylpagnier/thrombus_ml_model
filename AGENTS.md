# Agent notes (HemoGINO)

## Biochem training progress

- Living log: [src/docs/BIOCHEM_TRAINING_PROGRESS.md](src/docs/BIOCHEM_TRAINING_PROGRESS.md)
- ~3h viscosity/velocity architecture sweep (per-leg teacher ckpts for viz): one line `powershell -NoProfile -ExecutionPolicy Bypass -File "…/scripts/go_visc3h.ps1"` → [scripts/go_visc3h.ps1](scripts/go_visc3h.ps1), [scripts/run_biochem_visc_velocity_arch_sweep_3h.ps1](scripts/run_biochem_visc_velocity_arch_sweep_3h.ps1) → `outputs/biochem/sweep_visc_velocity_3h/`
- Cursor rule: [.cursor/rules/biochem-training-progress.mdc](.cursor/rules/biochem-training-progress.mdc) — agents should update the log when the user discusses biochem teacher/corrector run results (unless they opt out).
- Run artifacts: `outputs/reports/training/biochem/<run_id>/run.jsonl` (compact `meta` / `val` / `end` events) and `outputs/reports/training/biochem/runs_index.jsonl` (one summary row per completed run). Val rows include **viz health** fields (`viz_health_score`, `viz_t0_speed_mean`, `viz_final_mu2_mean`, …) for rollout triage. Disable with `BIOCHEM_TRAINING_LOG=0`.
- Overnight health sweep: `scripts/go_health10h.ps1` (9 legs: **K0** Carreau kinematic probe first, then R0/G0/G1/S0/S1/M0/M1/M2) → `outputs/biochem/sweep_health_arch_10h/<leg>/biochem_teacher_best_high_mu.pth` (per-leg via `BIOCHEM_ARCHIVE_CHECKPOINT_DIR`).
- Teacher checkpoints: `biochem_teacher_best_high_mu.pth` (global best teacher by high-μ val), `biochem_teacher_last.pth` (latest run backup). Viz default: best high-μ teacher → last teacher. Optional legacy all-truth: `biochem_teacher_best.pth` if `BIOCHEM_TEACHER_KEEP_GLOBAL_BEST_ALL=1`.

## Kinematics (Stage A) architecture record

- Reference manifest (architecture/flags only, no weights): [data/reference/kinematics_best_20260426T184600Z.json](data/reference/kinematics_best_20260426T184600Z.json) (best run `20260426T184600Z`, epoch 84).
- Doc: [src/docs/KINEMATICS_BEST_ARCHITECTURE.md](src/docs/KINEMATICS_BEST_ARCHITECTURE.md).
- Code: `snapshot_gino_deq_model_config` / `resolve_gino_deq_ctor_kwargs` in [src/architecture/kinematics_model_config.py](src/architecture/kinematics_model_config.py).
- `train_kinematics_predictor.py` embeds `model_config` in `kinematics_best.pth` and writes `outputs/kinematics/kinematics_architecture.json`.
- `train_biochem_corrector.py` reads Stage-A `model_config` (checkpoint → reference JSON → shape inference).

## Project orientation

- [src/docs/PROJECT_CONTEXT.md](src/docs/PROJECT_CONTEXT.md) — architecture, entry points, data layout.
