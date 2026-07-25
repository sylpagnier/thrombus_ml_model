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
- **`go_wc_v7_offwall_limit_2h.ps1`** — ~2h limit-analysis micro-sweep (LumenPush / FrontierPush / SkipHopSpec / BlindSat): freeze-backbone specialist + compound probe on patient007; `limit_2h_summary.json`.
- **`go_wc_v7_frontier_lumen_6h.ps1`** — ~6h FrontierLumen scale-up on orig10 (freeze-backbone, `frontier_lumen` loss, compound wall-route A vs S + hop_ge2 gates).
- **`go_wc_v7_frontier_lumen_viz.ps1`** — hop-ladder viz: locked WC_v7 (A) vs WC_v7+FrontierLumen compound (S).
- **`go_wc_v7_frontier_ge2_prec_8h.ps1`** — ~8h precision Frontier-ge2 compound on orig10 (`frontier_ge2` + compound val + wall clot floor).
- **`go_wc_v7_tile_cc_explore_2h.ps1`** — ~2h explore: union tile vs per-clot-region (`--tile-mode per_component`) A/B.
- **`go_wc_v7_tile_cc_explore_viz.ps1`** — hop-ladder viz: A vs UnionTile vs PerComponent (007/004).
- **`go_wc_v7_frontier_ge2_prec_viz.ps1`** — hop-ladder viz for Frontier-ge2 prec A vs S (007/004/008).
- **`go_wc_v7_frontier_ge2_prec_compare_viz.ps1`** — comparative hop-colored GT | WC_v7 | S (same figure).
- **`eda_compound_lumen_bottlenecks.py`** — hop-ge2 prevalence + Arm S failure taxonomy EDA.
- **`diagnose_lumen_001_vs_007.py`** — deploy-time hop hist: why 001 misses lumen vs 007.
- **`go_wc_v7_lumen_recall_limit_2h.ps1`** — ~2h recall-push limit analysis (001+007 teachers vs Prec8hRef).
- **`go_wc_v7_open001_1h.ps1`** — ~1h test: open 001 (train 001/007/010, recall tilt, probe teachers+spray).
- **`go_wc_v7_crack_001_3h.ps1`** — ~3h solo-001 hypothesis ladder (freeze / unfreeze / CC tiles) to crack hop_ge2=0 lock; `summarize_crack_001.py`.
- **`diagnose_crack_001_root.py`** — root cause: compound-val full-graph static vs eval wall-band (A_floor=0).
- `go_wc_v7_compound_growth_abc_9h.ps1` — all-on-disk-anchors variant (**~20–26 h**, not 9 h). Partial Arm B under `outputs/biochem/offwall_model/wc_v7_compound_abc_9h/`; resume with `-EvalOnly -SkipC`.
- `go_wc_v7_compound_growth_ab_6h.ps1` — stub redirect to the all-anchor ABC launcher.
- `go_viz_mat_w_wc_canonical.ps1`, `go_viz_pivot3_hop_analysis.ps1`
- `go_wc_v7_compound_orig10_viz.ps1` — hop-ladder viz for orig10 A/B/C compare dirs.
- `viz_species_gnn_deploy.py` / `go_species_gnn_deploy_viz.ps1` — species/clot timeline viz.
- `eval_mat_growth_simple.py` — cohort metrics for mat-growth legs (`--offwall-ckpt` / `--two-model-route` for compound).
- `summarize_wc_v7_compound_ab.py` — Arm A/B/C metric table.

## Research geometry-sensitivity sweeps

- `go_research_sweep.ps1` / `run_research_sweep.py` — vessel geometry × physics arms (stenosis, aneurysm, Re, width, bend, …) scored with transferable research parameters against **locked canonical** biochem at run time.
- Configs: `configs/research_sweeps/*.json`
- Docs: [`docs/RESEARCH_SWEEPS.md`](../docs/RESEARCH_SWEEPS.md)

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
- `docs/RESEARCH_SWEEPS.md` — geometry-sensitivity research sweeps
- `docs/BIOCHEM_LEGACY_LESSONS.md`
- `docs/archive/2026-06-16-biochem-cleanup.md`
- `docs/PUBLISHING.md` — public vs local artifact policy
- `AGENTS.md`
