# Agent notes (HemoGINO)

## Biochem training progress

- **`mu_ratio_max`**: step ceiling for COMSOL μ₁/μ₂ gelation steps (default 80), **not** clot `μ_eff`/bulk. GT clots are ~2–3× in `mu_eff_si`; fit channel 3 / `mu_log_mae`. See [src/docs/PROJECT_CONTEXT.md](src/docs/PROJECT_CONTEXT.md) § `mu_ratio_max` vs `mu_eff`.
- Training plan (milestones, X/Y/XY isolation, Phase I teacher / Phase II synthetic): [src/docs/BIOCHEM_TRAINING_PLAN.md](src/docs/BIOCHEM_TRAINING_PLAN.md)
- Loss policy (approved vs deprecated backward terms; `BIOCHEM_LEGACY_LOSSES=1`): [src/training/biochem_loss_policy.py](src/training/biochem_loss_policy.py), doc in [src/docs/BIOCHEM_TRAINING_PROGRESS.md](src/docs/BIOCHEM_TRAINING_PROGRESS.md) (Loss policy section)
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
- **Clot anchor survey** (GT μ floor + kinematic priors, regression diagnostics): `python scripts/survey_clot_anchor_patterns.py`.
- **Simple clot φ** (wall-local probe): [src/docs/CLOT_PHI_BASELINE.md](src/docs/CLOT_PHI_BASELINE.md). Default train: `go_clot_phi_simple.ps1 -Fresh` (joint_blend_gtsp). Ladders: `go_clot_phi_biology_ladder.ps1`, `go_clot_phi_biology_round2.ps1`. Best val patient007: F1~0.48, rec~0.42, pred+~0.24, score~0.58.
- **M3 ADR viability** (proved): union + `transport_only` + masked ADR; `m3_align_transport_union_12ep` (§132). Re-verify: `python scripts/check_m3_viability_pass.py`. **Optimize later:** `go_m3_block_pass.ps1` (full narrow/sweep/lock), global ramp2 raw ADR.
- **I.3 XY bridge** (viability **PASS**, §134): `go_passive_xy_block_pass.ps1` — hold 3ep + learn 6ep, `GRAD_SCALE_ON_CAP=1`; val FI **~0.013**. Lock canonical: `go_passive_lock_xy_ckpt.ps1` -> `biochem_teacher_passive_xy_locked.pth`. M5 chunks: plan **I.4** table (mu-unlock from xy_locked).
- **Passive transport (Step 2a, 1-way)**: Mat/FI on **COMSOL GT `[u,v,p]`** (`BIOCHEM_GT_KINE_VEL=1`), `mu_ratio_max=1`, `DETACH_MACRO=0`, **`BIOCHEM_PASSIVE_ADR_BACKPROP=0`** (ADR log-only until `L_Data_Bio` falls) — `.\scripts\go_passive_transport.ps1 -Fresh`. Phase B ramp: `go_phaseB_xy_passive.ps1`. **M3 align probe** (12ep): `go_m3_align_probe.ps1`. **Promote + confirm**: `go_passive_lock_align_ckpt.ps1` -> `outputs/biochem/biochem_teacher_passive_align_locked.pth`; **20ep** `go_passive_align_20ep.ps1` (train-anchor species via `BIOCHEM_PASSIVE_SPECIES_TRAIN_EVAL=1`); **step-2 bridge** `go_passive_step2_bridge.ps1` (`LOSS_DATA_ONLY=1`, `COMPLEXITY_STEP=2`, modest `MU_LOG`/`MU_SI`, low-weight `transport_only` ADR, not step 3); **mu-unlock probe** `go_passive_mu_unlock_probe.ps1` (`BIOCHEM_PASSIVE_MU_UNLOCK=1`, `LOSS_ISOLATE=MU_LOG`, `TRAIN_MU=1`, bio frozen, init locked ckpt); **finetune** `go_passive_mu_unlock_finetune.ps1` (wall+high-mu weights, init `biochem_teacher_passive_mu_unlock_best.pth`); **I.1 X probe** `go_passive_x_probe.ps1` (3ep matrix, ~30-45 min); **promote dump** `go_passive_x_block_finish.ps1 -Promote` only when probes pick a recipe. **6h explore** `go_passive_explore_6h.ps1` (isolated X/Y/XY legs, kin-blocked). Checklist: [src/docs/PASSIVE_KIN_BLOCKER_CHECKLIST.md](src/docs/PASSIVE_KIN_BLOCKER_CHECKLIST.md). Gates: `check_m3_align_gate.py`, `check_passive_step2_bridge_gate.py`, `check_passive_mu_unlock_gate.py`, `check_passive_mu_unlock_finetune_gate.py`; species table: `eval_passive_species_anchors.py`. Env: `BIOCHEM_SUPERVISION_MASK_TIMES`, `BIOCHEM_ADR_*`, `BIOCHEM_PASSIVE_STEP2_BRIDGE`, `BIOCHEM_PASSIVE_MU_UNLOCK`. Doc: [src/docs/BIOCHEM_TRAINING_PROGRESS.md](src/docs/BIOCHEM_TRAINING_PROGRESS.md).
- **GT-flow clot-phi (no kin model)**: ladder `go_gt_flow_species_ladder_6h.ps1`; round2 `go_gt_flow_round2_4h.ps1` (best so far **min F1 0.357** `long_adapt_blend`); round3 `go_gt_flow_round3_4h.ps1` (finetune toward **0.38**); chain `go_gt_flow_chain_r2finish_r3_4h.ps1`.
- **GNODE-ODE ladder (9.x, GT vel)**: doc [src/docs/GNODE_ODE_LADDER.md](src/docs/GNODE_ODE_LADDER.md); smoke **9.1** `go_gnode91_smoke.ps1`; ~8h queue **9.4-9.6** `go_gnode_8h_ladder.ps1`; **9.9** `go_gnode99.ps1` (after_94 init, FI/Mat=3/2, dump best ckpt, stride 72); headless teacher PNG `snapshot_biochem_teacher.py` + clot-band `snapshot_biochem_teacher_clotband.py`.
- **GNODE-ODE component ladder (9.0–9.9)**: [src/docs/GNODE_ODE_LADDER.md](src/docs/GNODE_ODE_LADDER.md) — simplest forward smoke -> passive -> dump/clot-phi; fast iterations first.
- **Clot-phi rollout (6a/6b)**: [src/docs/CLOT_PHI_ROLLOUT.md](src/docs/CLOT_PHI_ROLLOUT.md); `go_rung6a_clot_phi_rollout_gt.ps1`, `go_rung6b_clot_phi_rollout_kine.ps1`.
- Teacher checkpoints: `biochem_teacher_best_high_mu.pth` (global best teacher by high-μ val), `biochem_teacher_last.pth` (latest run backup). Viz default: best high-μ teacher → last teacher. Optional legacy all-truth: `biochem_teacher_best.pth` if `BIOCHEM_TEACHER_KEEP_GLOBAL_BEST_ALL=1`.
- **K4→K5 split-head staged** (wall head first, then clot + gelation): `powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_k4k5_split_head_staged.ps1"` — or stepwise `go_k4_wall_head_only.ps1` then `go_k5_clot_head_physics.ps1`. Env: `BIOCHEM_MU_TRAIN_WALL_ONLY` / `BIOCHEM_MU_TRAIN_CLOT_ONLY` (not `TRAIN_WALL_HEAD`; `BIOCHEM_MU_CARREAU_ONLY` / `BIOCHEM_USE_SIREN` are ignored).
- **K6 unified kitchen-sink** (~0.47 leash recipe, both heads together): `powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_k6_unified_kitchen_sink.ps1" -Fresh` — `SUPERVISED_DATA_LEASH` forces step-2 data backward (not step-3); use `-Multitask` only if intentionally skipping the leash.

## Kinematics (Stage A) geometry curriculum

- **Foundation** (mixed L0/L1/L2 sampling, full data): `powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_kinematics_foundation.ps1" -Fresh`
- **L2-heavy finetune** (resume + `--finetune-lr 1e-5`): `.\scripts\go_kinematics_l2_finetune.ps1`
- **Backfill** `geometry_level` on existing graphs from mesh JSON (no COMSOL): `python -m src.data_gen.backfill_kinematics_geometry_level`
- **Bend-sign A/B** (down-only vs bidirectional, isolated graph dirs): `powershell -File .\scripts\go_kinematics_bend_ab.ps1 -Arm both -NumVessels 120 -AnchorMax 0`
- **Recovery sweep ~10h** (main `graphs_kinematics/newtonian`, 8 scaled recipes, quiet logs): `powershell -File .\scripts\go_kinematics_recovery12h.ps1` (optional `-TargetHours 10`)
- **Production allfix** (100 ep, 3000 graphs, auto-resume): `powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_kinematics_production_allfix.ps1"` — best **20260601**: Rel L2 **0.126** @ ep 80 (`outputs/kinematics/production_allfix/`)
- **Synthetic polish**: `go_kinematics_production_allfix_finetune.ps1 -ContinuityFocus`
- **Clinical anchor finetune**: `go_kinematics_clinical_anchor_finetune.ps1` -> `outputs/kinematics/clinical_anchor_finetune/`; gates: `check_kinematics_promotion_gates.py`, `promote_kinematics_checkpoint.ps1`
- Doc: [src/docs/KINEMATICS_BEST_ARCHITECTURE.md](src/docs/KINEMATICS_BEST_ARCHITECTURE.md) (Stage-A ladder + geometry table)

## Kinematics (Stage A) architecture record

- Reference manifest (architecture/flags only, no weights): [data/reference/kinematics_best_20260426T184600Z.json](data/reference/kinematics_best_20260426T184600Z.json) (best run `20260426T184600Z`, epoch 84).
- **Current repo best (weights)**: `outputs/kinematics/production_allfix/kinematics_best.pth` — production allfix run `20260601T180106Z`, val Rel L2 **0.1263** @ epoch 80 (Adam; do not use post-L-BFGS epochs).
- Doc: [src/docs/KINEMATICS_BEST_ARCHITECTURE.md](src/docs/KINEMATICS_BEST_ARCHITECTURE.md).
- Code: `snapshot_gino_deq_model_config` / `resolve_gino_deq_ctor_kwargs` in [src/architecture/kinematics_model_config.py](src/architecture/kinematics_model_config.py).
- `train_kinematics_predictor.py` embeds `model_config` in `kinematics_best.pth` and writes `outputs/kinematics/kinematics_architecture.json`.
- `train_biochem_corrector.py` reads Stage-A `model_config` (checkpoint → reference JSON → shape inference).

## Console output (PowerShell)

- Do not use emoji or decorative Unicode in training/script `print` output — Windows PowerShell often renders them as mojibake. Use ASCII tags (`[OK]`, `[WARN]`, `[i]`). See [.cursor/rules/powershell-console-ascii.mdc](.cursor/rules/powershell-console-ascii.mdc).

## Scripts layout

- Active launchers and utilities: [scripts/README.md](scripts/README.md).
- Historical sweep names in `BIOCHEM_TRAINING_PROGRESS.md` referred to one-off runners since removed; use current `go_*` scripts instead.

## Project orientation

- [src/docs/PROJECT_CONTEXT.md](src/docs/PROJECT_CONTEXT.md) — architecture, entry points, data layout.
