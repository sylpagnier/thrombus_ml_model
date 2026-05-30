# Scripts

## Active launchers (`go_*.ps1`)

One-liners from repo root (see `AGENTS.md` for full ladder):

| Area | Examples |
|------|----------|
| Biochem mu ladder | `go_k10a` … `go_k10g`, `go_k4` / `go_k5` / `go_k4k5`, `go_k6` |
| Biochem presets | `go_k0` … `go_k3`, `go_k1_delta_mu`, `go_k2_physics_triggers_on`, `go_passive_transport` (ADR in backward off until `L_Data_Bio` falls — see training progress doc §113) |
| GT flow (no kin model) | **I.1 X:** `go_passive_x_block_pass.ps1`; **M3 ADR:** `go_m3_block_pass.ps1` (`-Turbo`); `go_passive_explore_6h.ps1` (scale only) (**6h** X/Y/XY ladder), `go_phase_a_xy_iterate.ps1`, `go_phaseB_xy_passive.ps1`, `go_m3_align_probe.ps1`, `go_passive_lock_align_ckpt.ps1`, `go_passive_align_20ep.ps1`, `go_passive_step2_bridge.ps1`, `go_passive_mu_unlock_probe.ps1`, `go_passive_mu_unlock_finetune.ps1`, `go_m3_adr_alignment_sweep.ps1`, `go_m3_narrowing_90m.ps1`, `go_gt_flow_species_ladder_6h.ps1`, `go_gt_flow_round2_4h.ps1`, ... |
| Sweeps | `go_health10h`, `go_visc3h` |
| Kinematics | `go_kinematics_foundation`, `go_kinematics_l2_finetune`, `go_kinematics_bend_ab`, `go_kinematics_recovery12h` |

`go_*` scripts set env vars and call `python -m src.training.train_biochem_corrector` or kinematics training directly, except:

- `go_health10h` -> `run_biochem_health_arch_sweep_10h.ps1`
- `go_visc3h` -> `run_biochem_visc_velocity_arch_sweep_3h.ps1`
- `go_kinematics_recovery12h` -> `run_kinematics_recovery_sweep_12h.ps1`

## Active runners (`run_*.ps1` in this folder)

| Script | Notes |
|--------|--------|
| `run_biochem_mu_formulation_study.ps1` | Preferred teacher-only mu iteration |
| `run_biochem_thrombus_corona.ps1` | Full corona pipeline (experimental) |
| `run_biochem_comprehensive_mu.ps1` | Comprehensive mu study (experimental) |
| `run_biochem_health_arch_sweep_10h.ps1` | Used by `go_health10h` |
| `run_biochem_visc_velocity_arch_sweep_3h.ps1` | Used by `go_visc3h` |
| `run_kinematics_recovery_sweep_12h.ps1` | Used by `go_kinematics_recovery12h` |

## Utilities (Python)

| Script | Notes |
|--------|--------|
| `survey_clot_anchor_patterns.py` | Clot anchor / kinematic prior diagnostics |
| `eval_kine_cross_cohort.py` | Cross-cohort kinematics eval CSV |
| `check_units.py` | ND velocity scale check (kin vs biochem graphs) |
| `strip_console_unicode.py` | Strip emoji from logs (see `.cursor/rules/powershell-console-ascii.mdc`) |
| `check_m3_align_gate.py` | M3 gate: `L_Data_Bio` + masked `L_ADR_S` co-descent from `run.jsonl` |
| `check_passive_step2_bridge_gate.py` | Step-2 bridge gate (M3 + modest val mu + `PASSIVE_STEP2_BRIDGE`) |
| `check_passive_mu_unlock_gate.py` | Mu-unlock probe gate (mu drop + species stable + `PASSIVE_MU_UNLOCK`) |
| `check_passive_mu_unlock_finetune_gate.py` | Finetune gate (all-mu down, wall/high-mu recovery, species stable) |
| `summarize_passive_explore_6h.py` | Rank `explore_6h` legs by species FI and mu drop |
| `eval_passive_species_anchors.py` | Per-anchor FI/Mat logMAE table for a passive teacher ckpt |
| `check_m3_narrowing_gate.py` | Narrowing ladder gate (bio + ADR + stability) |
| `summarize_m3_narrowing.py` | Rank `m3n_*` legs after narrowing sweep |
| `audit_passive_adr_alignment.py` | GT ADR: masks + `--all-formulations` ablation |
| `install_torch_cuda.ps1` | CUDA torch install helper |
