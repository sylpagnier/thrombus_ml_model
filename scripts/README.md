# Scripts

## Canonical baseline entry points

- `go_biochem_gnn.ps1` — canonical `biochem_deploy` train/eval/promote flow.
- `python -m src.bin.main train biochem-deploy` — CLI route to the same stack.
- `promote_biochem_gnn.py` — lock baseline artifacts and write canonical reference manifest.

## A/B scripts kept active

- `go_biochem_gnn_arch_ab.ps1` + `summarize_biochem_gnn_arch_ab.py`
- `go_biochem_gnn_gate_ab.ps1` + `summarize_biochem_gnn_gate_ab.py`
- `check_adhesion_gate_smoke.py`

## Deploy checks and visualization

- `check_biochem_gnn_gate.py`
- `viz_species_gnn_deploy.py`

## Kinematics scripts kept active

- `go_kinematics_production_allfix.ps1`
- `go_kinematics_precision_long.ps1`
- `go_kinematics_stage_a_ladder.ps1`

## Archived legacy ladders

Legacy S/T0/rules/GNODE/passive/clot-forecast launchers were removed from the active surface in the 2026-06-16 cleanup.
See:

- `docs/BIOCHEM_LEGACY_LESSONS.md`
- `docs/archive/2026-06-16-biochem-cleanup.md`
