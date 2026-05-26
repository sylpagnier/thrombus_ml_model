# Scripts

## Active launchers (`go_*.ps1`)

One-liners from repo root (see `AGENTS.md` for full ladder):

| Area | Examples |
|------|----------|
| Biochem mu ladder | `go_k10a` … `go_k10g`, `go_k4` / `go_k5` / `go_k4k5`, `go_k6` |
| Biochem presets | `go_k0` … `go_k3`, `go_k1_delta_mu`, `go_k2_physics_triggers_on`, `go_passive_transport` (ADR in backward off until `L_Data_Bio` falls — see training progress doc §113) |
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
| `install_torch_cuda.ps1` | CUDA torch install helper |
