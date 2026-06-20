# Agent notes (HemoGINO)

## Biochem training progress

- **`mu_ratio_max`**: step ceiling for COMSOL μ₁/μ₂ gelation steps (default 80), **not** clot `μ_eff`/bulk. GT clots are ~2–3× in `mu_eff_si`; fit channel 3 / `mu_log_mae`. See [docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md) § `mu_ratio_max` vs `mu_eff`.
- Training plan (milestones, X/Y/XY isolation, Phase I teacher / Phase II synthetic): [docs/BIOCHEM_TRAINING_PLAN.md](docs/BIOCHEM_TRAINING_PLAN.md)
- Loss policy (approved vs deprecated backward terms; `BIOCHEM_LEGACY_LOSSES=1`): [src/training/biochem_loss_policy.py](src/training/biochem_loss_policy.py), doc in [docs/BIOCHEM_TRAINING_PROGRESS.md](docs/BIOCHEM_TRAINING_PROGRESS.md) (Loss policy section)
- Living log: [docs/BIOCHEM_TRAINING_PROGRESS.md](docs/BIOCHEM_TRAINING_PROGRESS.md)
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
- **Mu coupling A/B/C probe** (A=baseline, B=Leg B v2 MLP mu map, C=`neighbor_wall` mu gate): `go_mlp_mu_map_v2_fast.ps1` / `go_mlp_clot_inject_probe.ps1` -> `abc_compare.json`; viz `go_mlp_mu_map_v2_viz.ps1` (v2) or `go_mlp_clot_inject_viz.ps1` (v1); env `BIOCHEM_MLP_MU_MAP=1` (v2), `BIOCHEM_MLP_CLOT_INJECT=1` (v1 trigger), `BIOCHEM_MU_NEIGHBOR_WALL_ONLY=1` (C). Quick coupled finetune: `go_mlp_mu_map_v2_coupled_train.ps1` (4ep, frozen MLP map in forward).
- **Clot anchor survey** (GT μ floor + kinematic priors, regression diagnostics): `python scripts/survey_clot_anchor_patterns.py`.
- **Simple clot φ** (wall-local probe): [docs/CLOT_PHI_BASELINE.md](docs/CLOT_PHI_BASELINE.md). Default train: `go_clot_phi_simple.ps1 -Fresh` (joint_blend_gtsp). Ladders: `go_clot_phi_biology_ladder.ps1`, `go_clot_phi_biology_round2.ps1`. Best val patient007: F1~0.48, rec~0.42, pred+~0.24, score~0.58.
- **M3 ADR viability** (proved): union + `transport_only` + masked ADR; `m3_align_transport_union_12ep` (§132). Re-verify: `python scripts/check_m3_viability_pass.py`. **Optimize later:** `go_m3_block_pass.ps1` (full narrow/sweep/lock), global ramp2 raw ADR.
- **I.3 XY bridge** (viability **PASS**, §134): `go_passive_xy_block_pass.ps1` — hold 3ep + learn 6ep, `GRAD_SCALE_ON_CAP=1`; val FI **~0.013**. Lock canonical: `go_passive_lock_xy_ckpt.ps1` -> `biochem_teacher_passive_xy_locked.pth`. M5 chunks: plan **I.4** table (mu-unlock from xy_locked).
- **Passive transport (Step 2a, 1-way)**: Mat/FI on **COMSOL GT `[u,v,p]`** (`BIOCHEM_GT_KINE_VEL=1`), `mu_ratio_max=1`, `DETACH_MACRO=0`, **`BIOCHEM_PASSIVE_ADR_BACKPROP=0`** (ADR log-only until `L_Data_Bio` falls) — `.\scripts\go_passive_transport.ps1 -Fresh`. Phase B ramp: `go_phaseB_xy_passive.ps1`. **M3 align probe** (12ep): `go_m3_align_probe.ps1`. **Promote + confirm**: `go_passive_lock_align_ckpt.ps1` -> `outputs/biochem/biochem_teacher_passive_align_locked.pth`; **20ep** `go_passive_align_20ep.ps1` (train-anchor species via `BIOCHEM_PASSIVE_SPECIES_TRAIN_EVAL=1`); **step-2 bridge** `go_passive_step2_bridge.ps1` (`LOSS_DATA_ONLY=1`, `COMPLEXITY_STEP=2`, modest `MU_LOG`/`MU_SI`, low-weight `transport_only` ADR, not step 3); **mu-unlock probe** `go_passive_mu_unlock_probe.ps1` (`BIOCHEM_PASSIVE_MU_UNLOCK=1`, `LOSS_ISOLATE=MU_LOG`, `TRAIN_MU=1`, bio frozen, init locked ckpt); **finetune** `go_passive_mu_unlock_finetune.ps1` (wall+high-mu weights, init `biochem_teacher_passive_mu_unlock_best.pth`); **I.1 X probe** `go_passive_x_probe.ps1` (3ep matrix, ~30-45 min); **promote dump** `go_passive_x_block_finish.ps1 -Promote` only when probes pick a recipe. **6h explore** `go_passive_explore_6h.ps1` (isolated X/Y/XY legs, kin-blocked). Checklist: [docs/PASSIVE_KIN_BLOCKER_CHECKLIST.md](docs/PASSIVE_KIN_BLOCKER_CHECKLIST.md). Gates: `check_m3_align_gate.py`, `check_passive_step2_bridge_gate.py`, `check_passive_mu_unlock_gate.py`, `check_passive_mu_unlock_finetune_gate.py`; species table: `eval_passive_species_anchors.py`. Env: `BIOCHEM_SUPERVISION_MASK_TIMES`, `BIOCHEM_ADR_*`, `BIOCHEM_PASSIVE_STEP2_BRIDGE`, `BIOCHEM_PASSIVE_MU_UNLOCK`. Doc: [docs/BIOCHEM_TRAINING_PROGRESS.md](docs/BIOCHEM_TRAINING_PROGRESS.md).
- **GT-flow clot-phi (no kin model)**: ladder `go_gt_flow_species_ladder_6h.ps1`; round2 `go_gt_flow_round2_4h.ps1` (best so far **min F1 0.357** `long_adapt_blend`); round3 `go_gt_flow_round3_4h.ps1` (finetune toward **0.38**); chain `go_gt_flow_chain_r2finish_r3_4h.ps1`.
- **GNODE-ODE ladder (9.x GT vel / 10-12 predicted kine)**: doc [docs/GNODE_ODE_LADDER.md](docs/GNODE_ODE_LADDER.md); **10** `go_gnode10_sweep.ps1`, `go_gnode10_finish.ps1`, `go_gnode10_kine_loop.ps1`; **11** `go_gnode11_*` + `check_gnode11_finish_gate.py`; **12 Lane A** `go_gnode12_lane_a.ps1` (mu unlock + dump + clot-phi); **12 Lane B** `go_gnode12_lane_b.ps1` (11-finish corrector dump + clot-phi) + `check_gnode12_lane_b_gate.py`; `_gnode12_env.ps1`
- **Viscosity comprehensive sweep (~6h):** `go_mu_complexity_6h.ps1` — **pred RGP-DEQ kine + GNODE Phase3**, **teacher then corrector/synth+pseudo**; legs `FULL_step2`, `FULL_step2p5`, `FULL_step3` (+ optional `FULL_overnight`); init **Lane A promoted**; `summarize_mu_complexity_6h.py` -> `outputs/biochem/sweep_mu_complexity_6h/`
- **Deploy clot ladder (2026-06):** [docs/DEPLOY_ARCHITECTURE.md](docs/DEPLOY_ARCHITECTURE.md) — **Track A** (this PC): COMSOL anchors, S0->G2 with GT flow. **Track B** (other PC): `go_clot_deploy_dump_comsol.ps1` for pred-kine dump. Launchers: `go_clot_deploy_s0_static.ps1`, `go_clot_deploy_s1_multih.ps1`, `go_clot_deploy_phase1.ps1` (G1).
- **Biochem deploy baseline (2026-06, canonical ML clot):** stack **`biochem_deploy`** — `gino_deq_kine` + **species GNN** + **gelation_beta** + **clot_trigger_physics**; config `src/biochem_deploy/` (alias `src/biochem_gnn/`); train `python -m src.bin.main train biochem-deploy`; launcher **`go_biochem_gnn.ps1`**; nomenclature [docs/MODEL_NOMENCLATURE.md](docs/MODEL_NOMENCLATURE.md); detail [docs/BIOCHEM_GNN.md](docs/BIOCHEM_GNN.md). **Distinct from** `train_biochem_corrector` (GNODE). Legacy: `biochem_gnn`, `clot_deploy_gnn`, `go_species_snapshot_s34.ps1`. p007 F1 ~0.70 vs s0 ~0.408.
- **Species pushforward arch A/B (2026-06-14, cancelled):** sage vs GINO-band trunk under same guiding recipe — **provisional winner: GraphSAGE** (sage ep19 `deploy_clot_score` 0.57 vs gnode ep1 0.43, declining by ep7). Doc [docs/BIOCHEM_GNN_ARCH_AB.md](docs/BIOCHEM_GNN_ARCH_AB.md); launcher `go_biochem_gnn_arch_ab.ps1`; partial summary `outputs/biochem/biochem_gnn/arch_ab/arch_ab_summary_partial.json`.
- **Clot ML V2 ladder (planned):** coupled band-GNN growth rate + **nucleation mask** (wall or 1-hop from commit; no ceiling). Baseline-first from V1. Doc: [docs/CLOT_ML_LADDER_V2.md](docs/CLOT_ML_LADDER_V2.md). V1 ladder: [docs/CLOT_ML_DEPLOY_TRAINING_PLAN.md](docs/CLOT_ML_DEPLOY_TRAINING_PLAN.md).
- **Teacher-only deploy (2026-06):** full model = teacher stage only; skip corrector/pseudo for now. Scalar mu: `sweep_mu_complexity_6h/FULL_step2/biochem_teacher_best_high_mu.pth` (~0.449). **Clot maps (project goal):** GNODE teacher + clot-phi MLP — **`outputs/biochem/clot_baseline/manifest.json`** now **lane_b_deploy** (closed-loop MLP mu map, **no GT clot mask**); env `src/inference/deploy_mu_map_env.py`; smoke `go_mlp_b_deploy_probe.ps1 -Fast -Leg B_deploy`; promote `go_promote_deploy_baseline.ps1`; recipe `data/reference/clot_baseline_lane_b_deploy.json`; predict `python -m src.inference` (`ClotBaselinePredictor.from_manifest()` attaches deploy injector); viz `go_mlp_abc_viz.ps1 -Leg B_deploy`. Oracle gt_clot Leg B still via `BIOCHEM_MLP_MU_MAP_MASK=gt_clot` for upper-bound eval.
- **GNODE-ODE component ladder (9.0–9.9)**: [docs/GNODE_ODE_LADDER.md](docs/GNODE_ODE_LADDER.md) — simplest forward smoke -> passive -> dump/clot-phi; fast iterations first.
- **Clot forecast ladder (R0–R6, fresh start)**: [docs/CLOT_FORECAST_LADDER.md](docs/CLOT_FORECAST_LADDER.md) — R0 label sanity → R1 one-step (`go_clot_forecast_r1.ps1` A–D, **promoted R1D** `deploy_pred`) → **R2** carry (`go_clot_forecast_r2.ps1`, R1D init) → R4 RGP-DEQ → R6 deploy. **Viz:** metrics at R1; R2+ PNG + **`clot_shape`** in logs.
- **Clot-phi rollout (6a/6b)**: [docs/CLOT_PHI_ROLLOUT.md](docs/CLOT_PHI_ROLLOUT.md); `go_rung6a_clot_phi_rollout_gt.ps1`, `go_rung6b_clot_phi_rollout_kine.ps1`.
- Teacher checkpoints: `biochem_teacher_best_high_mu.pth` (global best teacher by high-μ val), `biochem_teacher_last.pth` (latest run backup). Viz default: best high-μ teacher → last teacher. Optional legacy all-truth: `biochem_teacher_best.pth` if `BIOCHEM_TEACHER_KEEP_GLOBAL_BEST_ALL=1`.
- **K4→K5 split-head staged** (wall head first, then clot + gelation): `powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_k4k5_split_head_staged.ps1"` — or stepwise `go_k4_wall_head_only.ps1` then `go_k5_clot_head_physics.ps1`. Env: `BIOCHEM_MU_TRAIN_WALL_ONLY` / `BIOCHEM_MU_TRAIN_CLOT_ONLY` (not `TRAIN_WALL_HEAD`; `BIOCHEM_MU_CARREAU_ONLY` / `BIOCHEM_USE_SIREN` are ignored).
- **K6 unified kitchen-sink** (~0.47 leash recipe, both heads together): `powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_k6_unified_kitchen_sink.ps1" -Fresh` — `SUPERVISED_DATA_LEASH` forces step-2 data backward (not step-3); use `-Multitask` only if intentionally skipping the leash.

## Kinematics (Stage A) geometry curriculum

- **Foundation** (mixed L0/L1/L2 sampling, full data): `powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_kinematics_foundation.ps1" -Fresh`
- **L2-heavy finetune** (resume + `--finetune-lr 1e-5`): `.\scripts\go_kinematics_l2_finetune.ps1`
- **Backfill** `geometry_level` on existing graphs from mesh JSON (no COMSOL): `python -m src.data_gen.backfill_kinematics_geometry_level`
- **Bend-sign A/B** (down-only vs bidirectional, isolated graph dirs): `powershell -File .\scripts\go_kinematics_bend_ab.ps1 -Arm both -NumVessels 120 -AnchorMax 0`
- **Recovery sweep ~10h** (main `graphs_kinematics/newtonian`, 8 scaled recipes, quiet logs): `powershell -File .\scripts\go_kinematics_recovery12h.ps1` (optional `-TargetHours 10`)
- **Production allfix (default 3-phase loop)**: `powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_kinematics_production_allfix.ps1"` — phases 1-3 + promote; `-FoundationOnly` for phase 1 only. Best finetune **20260603** ep **119**: Rel L2 **0.087** (`production_allfix/`)
- **Synthetic polish only**: `go_kinematics_production_allfix_finetune.ps1 -ContinuityFocus`
- **Long precision (synth 60ep + clinical 50ep + promote)**: `go_kinematics_precision_long.ps1` (needs `graphs_kinematics_anchors/carreau/patient*.pt`)
- **Clinical geometry only**: `go_kinematics_clinical_anchor_finetune.ps1`
- **Stage-A ladder** (orchestrator): `go_kinematics_stage_a_ladder.ps1` (`-SkipFoundation` to resume after phase 1)
- Doc: [docs/KINEMATICS_BEST_ARCHITECTURE.md](docs/KINEMATICS_BEST_ARCHITECTURE.md) (Stage-A ladder + geometry table)

## Kinematics (Stage A) architecture record

- Reference manifest (architecture/flags only, no weights): [data/reference/kinematics_best_20260426T184600Z.json](data/reference/kinematics_best_20260426T184600Z.json) (best run `20260426T184600Z`, epoch 84).
- **Current repo best (weights)**: `outputs/kinematics/production_allfix/kinematics_best.pth` — finetune after production `20260601T180106Z`, val Rel L2 **0.0870** @ epoch **119** (Adam; skip LBFGS).
- Doc: [docs/KINEMATICS_BEST_ARCHITECTURE.md](docs/KINEMATICS_BEST_ARCHITECTURE.md).
- Code: `snapshot_gino_deq_model_config` / `resolve_gino_deq_ctor_kwargs` in [src/architecture/kinematics_model_config.py](src/architecture/kinematics_model_config.py).
- `train_kinematics_predictor.py` embeds `model_config` in `kinematics_best.pth` and writes `outputs/kinematics/kinematics_architecture.json`.
- `train_biochem_corrector.py` reads Stage-A `model_config` (checkpoint → reference JSON → shape inference).

## Local kinematic corrector (clot velocity diversion)

- Local k-hop GNN that predicts velocity diversion `[dU,dV]` as a residual on the frozen GINO-DEQ base flow around micro-clots. Doc + run log: [docs/LOCAL_KINEMATIC_CORRECTOR.md](docs/LOCAL_KINEMATIC_CORRECTOR.md).
- Data: COMSOL Patch Factory (`src/data_gen/lib/patch_factory_comsol.py`, no Gmsh; mapped quad grid; QC + default mesh-convergence in `patch_factory_qc.py`). Model: `LocalKinematicCorrector` (3x GATv2) in `src/core_physics/coupled_shear_gnn.py`.
- Train: `python -m src.training.train_local_kinematic_corrector --epochs 600 --batch-size 4 --stride 2 --device cuda` (5 GiB-safe). Eval vs COMSOL truth: `python -m src.tools.eval_local_corrector ...`. Live overlay vs GINO-DEQ: `python -m src.tools.verify_local_corrector_live ...`.
- Latest (2026-06-20): 800 ep -> held-out global relL2 17.6% (med 19%, p90 42%, p95 56%, max 98%); improved across the board vs 300 ep (26.7%), still trending down but tail is now the bottleneck. See doc run log.

## Console output (PowerShell)

- Do not use emoji or decorative Unicode in training/script `print` output — Windows PowerShell often renders them as mojibake. Use ASCII tags (`[OK]`, `[WARN]`, `[i]`). See [.cursor/rules/powershell-console-ascii.mdc](.cursor/rules/powershell-console-ascii.mdc).

## Interactive kinematics demo

- Parametric flow GUI: `python -m src.tools.demo_kinematics_flow` or `python -m src.bin.main inspect flow -- --rheology carreau` — Gmsh mesh, RGP-DEQ u/v/p; **Edit walls** mode drags interior wall control points (pinned ends); `--no-gui` writes PNG under `outputs/reports/figures/kinematics/`.

## Scripts layout

- **Biochem COMSOL extract:** default auto-pull from `comsol_models/phase2_nowound_XXX.mph` (`patientXXX`); auto-exports mesh `.nas`/`.msh` and boundaries from COMSOL when missing. Run `python -m src.data_gen.lib.extract_biochem_comsol_data`. Legacy manual exports: `--no-from-comsol`.
- Active launchers and utilities: [scripts/README.md](scripts/README.md).
- Historical sweep names in `BIOCHEM_TRAINING_PROGRESS.md` referred to one-off runners since removed; use current `go_*` scripts instead.

## Project orientation

- [docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md) — architecture, entry points, data layout.
