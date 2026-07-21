# Scripts

Supported launchers for **HemoRGP**. Retired ladders live under [`archive/`](archive/).
Publishing policy: [`docs/PUBLISHING.md`](../docs/PUBLISHING.md).

## Canonical biochem deploy

- `go_biochem_gnn.ps1` — train/eval/promote for `biochem_gnn` (GraphSAGE species + gelation + clot trigger).
- `python -m src.bin.main train biochem-gnn` — same stack via CLI.
- `promote_biochem_gnn.py` — lock baseline artifacts + reference manifest.

## Mat-growth (current research path)

- `go_fresh_canonical.ps1` / `go_fresh_canonical_finish.ps1` — promote WC legs into locked baseline.
- `go_mat_w_wc_canonical.ps1`, `go_mat_growth_simple.ps1`, `go_mat_growth_ladder.ps1`
- `go_off_wall_clot_sweep_6h.ps1` — off-wall pivot A/B (Pivot 3 occlusion winner).
- **`go_wc_v7_compound_growth_abc_orig10_9h.ps1`** — **true ~9 h** WC_v7 (A) vs revised frontier compound (B) vs wall-route+blurring_prec (C) on original anchors 1–8,10,11. Arm B uses `loss_blurring_prec` + `offwall_balanced` (fixes 35-anchor overgrowth). No skiphop Arm D.
- **`go_wc_v7_firewall_fix_seq.ps1`** — firewall sequence on WC_v7: (1) midside-blind+hop1-smooth+sat30 finetune, (2) hop>=2 lumen-shape specialist + compound eval, (3) optional isolate/skiphop. Hop-stratified off-wall metrics in eval.
- `go_wc_v7_compound_growth_abc_9h.ps1` — all-on-disk-anchors variant (**~20–26 h**, not 9 h). Partial Arm B under `outputs/biochem/offwall_model/wc_v7_compound_abc_9h/`; resume with `-EvalOnly -SkipC`.
- `go_wc_v7_compound_growth_ab_6h.ps1` — stub redirect to the all-anchor ABC launcher.
- `go_viz_mat_w_wc_canonical.ps1`, `go_viz_pivot3_hop_analysis.ps1`
- `go_wc_v7_compound_orig10_viz.ps1` — hop-ladder viz for orig10 A/B/C compare dirs.
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

Retired GNODE / clot-ML / T0 / graybox launchers live under **`scripts/archive/`** (see that folder's README). Active entry points are listed above only.

Also see:

- `docs/MAT_GROWTH.md` — canonical mat-growth baseline
- `docs/BIOCHEM_LEGACY_LESSONS.md`
- `docs/archive/2026-06-16-biochem-cleanup.md`
- `docs/PUBLISHING.md` — public vs local artifact policy
- `AGENTS.md`
