# Agent notes (HemoGINO)

## Biochem training progress

- Living log: [src/docs/BIOCHEM_TRAINING_PROGRESS.md](src/docs/BIOCHEM_TRAINING_PROGRESS.md)
- ~3h viscosity/velocity architecture sweep (per-leg teacher ckpts for viz): one line `powershell -NoProfile -ExecutionPolicy Bypass -File "…/scripts/go_visc3h.ps1"` → [scripts/go_visc3h.ps1](scripts/go_visc3h.ps1), [scripts/run_biochem_visc_velocity_arch_sweep_3h.ps1](scripts/run_biochem_visc_velocity_arch_sweep_3h.ps1) → `outputs/biochem/sweep_visc_velocity_3h/`
- Cursor rule: [.cursor/rules/biochem-training-progress.mdc](.cursor/rules/biochem-training-progress.mdc) — agents should update the log when the user discusses biochem teacher/corrector run results (unless they opt out).
- Run artifacts: `outputs/reports/training/biochem/<run_id>/run.jsonl` (compact `meta` / `val` / `end` events) and `outputs/reports/training/biochem/runs_index.jsonl` (one summary row per completed run). Val rows include **viz health** fields (`viz_health_score`, `viz_t0_speed_mean`, `viz_final_mu2_mean`, …) for rollout triage. Disable with `BIOCHEM_TRAINING_LOG=0`.
- Overnight health sweep: `scripts/go_health10h.ps1` (9 legs: **K0** Carreau kinematic probe first, then R0/G0/G1/S0/S1/M0/M1/M2) → `outputs/biochem/sweep_health_arch_10h/<leg>/biochem_teacher_best_high_mu.pth` (per-leg via `BIOCHEM_ARCHIVE_CHECKPOINT_DIR`).
- **K10a** (steady-kin μ at t=0): `powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_k10a_ic_steady_kin.ps1" -Fresh` — sets `BIOCHEM_MU_IC_STEADY_KIN=1` (+ K1 `DATA_KINE` stack).
- **K10b** (K10a + split head + `BIOCHEM_MU_ADDITIVE_DELTA=1`): `powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_k10b_additive_delta_ic_steady.ps1" -Fresh`. Teacher `.pth` embeds `model_config.forward_policy`; viz needs **no** manual `BIOCHEM_*` flags (re-save old ckpts to embed policy).
- **K10c** (K10b + data-only backprop + `MU_LOG_HIGH`, no `LOSS_ISOLATE`): `powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_k10c_high_mu_aux.ps1"`.
- **K10d** (proof: `μ_eff=μ_ss+softplus(Δμ_SI)`, `LOSS_ISOLATE=MU_MSE` only): `powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_k10d_simple_mu_mse.ps1"`.
- **K10e** (wall-adjacent clots: `μ_eff=μ_ss+adj_mask×Δμ_nd`, `LOSS_ISOLATE=K10E`): `powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_k10e_wall_adjacent_mu_log.ps1" -Fresh`. Viz: `python -m src.evaluation.visualize_pipeline --teacher-only --biochem-checkpoint outputs/biochem/biochem_teacher_last.pth`
- **K10f** (K10e wide band: `D_PEAK/SIGMA=0.008`, `SDF_MAX=0.04`, `Δμ_nd_max=30`, adjacent w=6, bulk w=0.5): `powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_k10f_wide_adjacent_band.ps1" -Fresh`
- **K10g oracle viz** (GT clots in wall-adjacent band, no train): `powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_k10g_oracle_clots_viz.ps1"`
- **K10g bias sanity** (init `Δμ` bias ~17 ND + `DATA_KINE` w=1, 6ep): `powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_k10g_bias_clot_sanity.ps1" -Fresh`
- **K11** (clot gate: `μ_eff=μ_ss+p_clot×(μ_clot−μ_ss)`, `LOSS_ISOLATE=K11`): `powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_k11_clot_gate.ps1" -Fresh`. Viz: `python -m src.evaluation.visualize_pipeline --teacher-only --biochem-checkpoint outputs/biochem/biochem_teacher_last.pth --anchor patient007`
- Teacher checkpoints: `biochem_teacher_best_high_mu.pth` (global best teacher by high-μ val), `biochem_teacher_last.pth` (latest run backup). Viz default: best high-μ teacher → last teacher. Optional legacy all-truth: `biochem_teacher_best.pth` if `BIOCHEM_TEACHER_KEEP_GLOBAL_BEST_ALL=1`.
- **K4→K5 split-head staged** (wall head first, then clot + gelation): `powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_k4k5_split_head_staged.ps1"` — or stepwise `go_k4_wall_head_only.ps1` then `go_k5_clot_head_physics.ps1`. Env: `BIOCHEM_MU_TRAIN_WALL_ONLY` / `BIOCHEM_MU_TRAIN_CLOT_ONLY` (not `TRAIN_WALL_HEAD`; `BIOCHEM_MU_CARREAU_ONLY` / `BIOCHEM_USE_SIREN` are ignored).
- **K6 unified kitchen-sink** (~0.47 leash recipe, both heads together): `powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_k6_unified_kitchen_sink.ps1" -Fresh` — `SUPERVISED_DATA_LEASH` forces step-2 data backward (not step-3); use `-Multitask` only if intentionally skipping the leash.

## Kinematics (Stage A) geometry curriculum

- **Foundation** (mixed L0/L1/L2 sampling, full data): `powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_kinematics_foundation.ps1" -Fresh`
- **L2-heavy finetune** (resume + `--finetune-lr 1e-5`): `.\scripts\go_kinematics_l2_finetune.ps1`
- **Backfill** `geometry_level` on existing graphs from mesh JSON (no COMSOL): `python -m src.data_gen.backfill_kinematics_geometry_level`
- **Bend-sign A/B** (down-only vs bidirectional, isolated graph dirs): `powershell -File .\scripts\go_kinematics_bend_ab.ps1 -Arm both -NumVessels 120 -AnchorMax 0`
- Doc: [src/docs/KINEMATICS_BEST_ARCHITECTURE.md](src/docs/KINEMATICS_BEST_ARCHITECTURE.md) (geometry table)

## Kinematics (Stage A) architecture record

- Reference manifest (architecture/flags only, no weights): [data/reference/kinematics_best_20260426T184600Z.json](data/reference/kinematics_best_20260426T184600Z.json) (best run `20260426T184600Z`, epoch 84).
- Doc: [src/docs/KINEMATICS_BEST_ARCHITECTURE.md](src/docs/KINEMATICS_BEST_ARCHITECTURE.md).
- Code: `snapshot_gino_deq_model_config` / `resolve_gino_deq_ctor_kwargs` in [src/architecture/kinematics_model_config.py](src/architecture/kinematics_model_config.py).
- `train_kinematics_predictor.py` embeds `model_config` in `kinematics_best.pth` and writes `outputs/kinematics/kinematics_architecture.json`.
- `train_biochem_corrector.py` reads Stage-A `model_config` (checkpoint → reference JSON → shape inference).

## Project orientation

- [src/docs/PROJECT_CONTEXT.md](src/docs/PROJECT_CONTEXT.md) — architecture, entry points, data layout.
