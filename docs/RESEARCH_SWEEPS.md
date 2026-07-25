# Research geometry-sensitivity sweeps

Future-runnable experiments that measure how **vessel geometry / inlet physics /
deploy stack knobs** affect transferable **research parameters**, using whatever
**locked canonical** biochem model is promoted at run time.

This is **not** a neural-architecture ablation ladder. Axes are stenosis,
aneurysm, Re, width, bendiness, roughness, and optional corrector / dynamic-
occlusion stack switches.

## Model resolution (`locked_canonical`)

Registry JSON sets `"model": "locked_canonical"`. The runner does **not** pin a
checkpoint hash.

At execution, [`CustomerDeployPipeline`](../src/inference/customer_pipeline.py)
resolves:

| Knob | Default |
|------|---------|
| Wall / species ckpt | `outputs/biochem/biochem_gnn/locked/species_gnn_best.pth` |
| Mat-growth leg | `WC_v7_clot_phi_mse` (or current promote target) |

After you promote a new baseline (`scripts/promote_biochem_gnn.py`), re-run the
same sweep configs to compare against the new canonical model.

Arm-level `env_overrides` (see `15_stack_coupling`) are applied **after** the
mat-leg recipe so stack ablations win for that run only.

Overrides (rare): `--wall-ckpt`, `--mat-leg`, or env `CUSTOMER_WALL_CKPT`.

## Research parameters (shared pack)

Module: [`src/evaluation/research_parameters.py`](../src/evaluation/research_parameters.py)
(`schema_version` 2). Per-frame rows match the customer **Scientific** downloadable
CSV ([`customer_predict_metrics.py`](../src/tools/customer_predict_metrics.py)).

| Series | Meaning |
|--------|---------|
| `vessel_clot_pct` / `wall_clot_pct` / `lumen_clot_pct` | Node coverage (φ ≥ 0.5) |
| `open_lumen_pct` | % lumen nodes still open |
| `max_occlusion_pct` | Lumen hop occlusion |
| `open_lumen_residual_pct` | Remaining open radial depth / max lumen hop |
| `clot_frac_hop0/1/ge2_pct` | Hop histogram of clot (% of clot nodes) |
| `clot_mass_prox/mid/dist_pct` | Axial clot mass thirds |
| `clot_cov_prox/mid/dist_pct` | Coverage within each third |
| `clot_axis_span_norm` / `clot_axis_centroid_norm` | Axial extent / centroid |
| `clot_front_speed_per_h` | d(span)/d(t_h) |
| `has_wall_clot` / `has_lumen_clot` | Onset flags |
| `mean_vel_open_lumen` / `vel_open_lumen_drop_pct` | When velocity bookends present |

Trajectory **summary** adds peak / final / AUC, time-to-occ 25/50/75,
`t_h_to_first_wall_clot`, `t_h_to_first_lumen_clot`, and early front speed.

## Experiment matrix

Configs: [`configs/research_sweeps/`](../configs/research_sweeps/).  
Shared control (unless overridden): straight, width 0.012 m, Re=450, 8 UI hours
(`t_final_s=30000`, `n_steps=120`), seed 42.

| Id | Axis |
|----|------|
| `01_stenosis_strength` | Diameter occlusion 0 / 0.25 / 0.50 / 0.75 / 0.80 |
| `02_aneurysm_strength` | Aneurysm factor 0 → 1.0 (local width up to 3x inlet) |
| `03_inlet_re` | Re 150 / 300 / 450 / 600 / 900 |
| `04_inlet_width` | Width 0.008 → 0.020 m |
| `05_bendiness` | Straight → mild/strong arc → s_curve → hook |
| `06_stenosis_location` | Proximal / mid / distal @ 50% stenosis |
| `07_stenosis_x_re` | Stenosis × Re |
| `08_vessel_length` | `base_length` 0.05 / 0.10 / 0.15 m |
| `09_stenosis_eccentricity` | Top / bottom / both-wall stenosis |
| `10_pathology_length` | Narrow / default / broad Gaussian |
| `11_aneurysm_x_re` | Aneurysm × Re |
| `12_bend_x_stenosis` | Bend × stenosis |
| `13_width_x_re` | Width × Re |
| `14_wall_roughness` | Wall roughness amp (0–8% of width) |
| `15_stack_coupling` | Corrector × dynamic occlusion (env) |

Geometry builder + mesh cache:
[`src/evaluation/research_sweep_geometry.py`](../src/evaluation/research_sweep_geometry.py)  
Cache: `outputs/research_sweeps/_meshes/`.

## How to run

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_research_sweep.ps1 -List
powershell ... -File .\scripts\go_research_sweep.ps1 -Sweep 01_stenosis_strength
powershell ... -File .\scripts\go_research_sweep.ps1 -Sweep 15_stack_coupling
powershell ... -File .\scripts\go_research_sweep.ps1 -All
```

Or:

```text
python scripts/run_research_sweep.py --list
python scripts/run_research_sweep.py --sweep 09_stenosis_eccentricity
```

Outputs under `outputs/research_sweeps/<id>/`: `arm_*.json`, `arm_*.csv`,
`summary.json`.

## Customer Scientific CSV

Running Predict in **Scientific** mode writes / downloads the same per-frame
research columns (via `trajectory_scientific_table` → `write_scientific_csv`).

## Notes

- Rollouts are GPU-heavy; prefer CUDA.
- Coverage is **node fraction** (matches customer UI).
- `15_stack_coupling` enables velocity bookends so open-lumen velocity drop is populated.
