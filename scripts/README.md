# Scripts

## Canonical biochem deploy

- `go_biochem_gnn.ps1` — train/eval/promote for `biochem_deploy` (GraphSAGE species + gelation + clot trigger).
- `python -m src.bin.main train biochem-deploy` — same stack via CLI.
- `promote_biochem_gnn.py` — lock baseline artifacts + reference manifest.

## Mat-growth (current research path)

- `go_fresh_canonical.ps1` / `go_fresh_canonical_finish.ps1` — promote WC legs into locked baseline.
- `go_mat_w_wc_canonical.ps1`, `go_mat_growth_simple.ps1`, `go_mat_growth_ladder.ps1`
- `go_off_wall_clot_sweep_6h.ps1` — off-wall pivot A/B (Pivot 3 occlusion winner).
- **`go_wc_v7_compound_growth_abc_9h.ps1`** — WC_v7 (A) vs frontier compound (B) vs wall-route+blurring_prec (C). **Not a 9h job** with `--all-anchors` (~35 graphs): ~8–10 h per specialist train, ~20–26 h full A/B/C; eval A+B alone ~2–6 h. Partial: Arm B ckpt saved under `outputs/biochem/offwall_model/wc_v7_compound_abc_9h/`; resume compare with `-EvalOnly -SkipC`.
- `go_wc_v7_compound_growth_ab_6h.ps1` — stub redirect to the ABC launcher.
- `go_viz_mat_w_wc_canonical.ps1`, `go_viz_pivot3_hop_analysis.ps1`
- `viz_species_gnn_deploy.py` / `go_species_gnn_deploy_viz.ps1` — species/clot timeline viz.
- `eval_mat_growth_simple.py` — cohort metrics for mat-growth legs (`--offwall-ckpt` / `--two-model-route` for compound).
- `summarize_wc_v7_compound_ab.py` — Arm A/B/C metric table.

## Visualization

- Steady kinematics + GraphSAGE deploy smoke: `python -m src.evaluation.visualize_pipeline` (optional `--steady-kin-only`).
- Batch steady-kin: `steady_kin_viz_cohort.py`
- Customer Predict GUI: `go_customer_predict.ps1`

## A/B and gates

- `go_biochem_gnn_arch_ab.ps1` + `summarize_biochem_gnn_arch_ab.py`
- `go_biochem_gnn_gate_ab.ps1` + `summarize_biochem_gnn_gate_ab.py`
- `check_biochem_gnn_gate.py`

## Kinematics (Stage A)

- `go_kinematics_foundation.ps1`
- `go_kinematics_production_allfix.ps1`
- `go_kinematics_precision_long.ps1`
- `go_kinematics_stage_a_ladder.ps1`
- `go_kinematics_recovery12h.ps1`
- `go_kinematics_l2_finetune.ps1`
- `go_kinematics_clinical_anchor_finetune.ps1`
- `go_kinematics_bend_ab.ps1`

## Archived legacy ladders

GNODE teacher/corrector, passive/M3, clot-forecast, T0, clot-ML rule ladders, and MLP mu-map probe launchers were removed from the active surface (2026-06 and 2026-07). See:

- `docs/BIOCHEM_LEGACY_LESSONS.md`
- `docs/archive/2026-06-16-biochem-cleanup.md`
- `AGENTS.md` (retired table)
