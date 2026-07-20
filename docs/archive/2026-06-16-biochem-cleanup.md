# Biochem Cleanup Archive (2026-06-16)

## Baseline promotion performed

- Ran: `python scripts/promote_biochem_gnn.py`
- Result: canonical baseline locked successfully.
- Canonical artifacts:
  - `outputs/biochem/biochem_gnn/locked/manifest.json`
  - `outputs/biochem/biochem_gnn/locked/species_gnn_best.pth`
  - `outputs/biochem/biochem_gnn/locked/viscosity_beta.pth`
  - `data/reference/biochem_gnn_baseline.json`

## Cleanup policy

- Keep active `biochem_deploy` train/eval/promote paths.
- Remove legacy ladder launchers and wrapper aliases from top-level active surface.
- Preserve lessons in `docs/BIOCHEM_LEGACY_LESSONS.md`.

## Archived/removed categories

- Species snapshot ladder launchers (`go_species_snapshot_s*.ps1`)
- T0 ladder launchers (`go_t0*.ps1`)
- GNODE ladder launchers (`go_gnode*.ps1`)
- Passive/M3/GT-flow ladder launchers (`go_passive*.ps1`, `go_m3*.ps1`, `go_gt_flow*.ps1`)
- Clot forecast ladder launchers (`go_clot_forecast*.ps1`)
- Legacy wrapper aliases:
  - `scripts/go_clot_deploy_gnn.ps1`
  - `scripts/go_promote_species_gnn_baseline.ps1`
  - `scripts/promote_clot_deploy_gnn.py`
  - `scripts/promote_species_gnn_baseline.py`
  - `src/clot_deploy_gnn/*`
- Legacy ladder docs (superseded by canonical baseline docs and this archive record)

## Follow-up trim (2026-07-19)

Additional dead surface removed after GraphSAGE migration left GNODE-era scripts broken:

- Slimmed `src/evaluation/visualize_pipeline.py` (~3k → ~600 lines): steady-kin + GraphSAGE deploy smoke only.
- Deleted scripts importing removed modules (`train_biochem_corrector`, `gnode_biochem`, `biochem_teacher_loader`, `train_clot_phi_simple`, `clot_ml_*`).
- Deleted broken stubs: `go_visc3h`, `go_health10h`, `go_mu_complexity_6h`, `go_mlp_*`, `go_rung6*`, clot-ML train/eval/viz trees, T0 diagnose, orphan `_clot_deploy_*_base.ps1`.
- Removed empty `src/clot_deploy_gnn/` husk; restored `src/biochem_deploy/` as a thin re-export of `src.biochem_gnn`.
- Rewrote `AGENTS.md` / `scripts/README.md` to match the active surface.

## Notes

- This archive keeps history discoverable without keeping old ladders in the active script/doc surface.
- If a path is needed again, recover from git history using this file as an index.

