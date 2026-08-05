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
- **`go_wc_v8_improvement_sweeps.ps1`** — post-promote WC v8 compound axes: hops eval sweep, 010 FP polish, frontier-h1 retrain, 007 recall, partial unfreeze (`run_wc_v8_improvement_sweeps.py`).
- **`go_generalization_8h.ps1`** / **`run_generalization_8h.py`** — ~8h clean growth retrain: cold `--no-init`, sealed challenge `009/032`, mean compound-val on disjoint val set, `frontier_offwall h0.5`.
- **`go_flow_source_ab.ps1`** / **`run_flow_source_ab.py`** — pre-phase1 gate: GT vs RGP-DEQ kine vs coupled (local tiling) train flow on the small wall-gen cohort; cold **deploy-faithful** eval on holdout **`patient020` only**.
- **`sweep_frontier_hops.py`** — eval-time frontier hops 0 / 0.5 / 1 / 2 (+ per-vessel map).
- **`promote_compound_deploy.py`** — lock growth ckpt + `data/reference/mat_compound_deploy.json`.
- **`promote_wall_gen_baseline.py`** — lock `FS_ab_coupled` wall-gen baseline + `data/reference/mat_wall_gen_baseline.json` (phase1 warm-start; does not replace WC_v7).
- **`go_phase1_sweep_v3.ps1`** — single-factor tweaks on wall-gen baseline (`WG_sweep_v3_*`); holdout `patient020`; deploy-faithful eval.
- **`go_wg_featfix_sweep.ps1`** — re-run of geom/flux arms (`WG_featfix_01..04`) after pack-cache + `in_dim` band-extra fix; not launched by default.
- **`go_wg_clotrich_nplus.ps1`** — clot-rich N+ LOAO on featfix_03 stack; default leg `WG_clotrich_nplus_v2` (mass-gated select, deploy_horizon aux, heads-only FT); holdout `patient020`; warm-start `WG_featfix_03`.
- **`go_wg_prec_iter.ps1`** — small-cohort (005/006/010) precision iteration: step+final mass/FP loss, mass-gated select, warm-start featfix_03; fix train–deploy mismatch before revisiting N+. Also used for `WG_prec_mirror` via `-Leg WG_prec_mirror -RunRoot outputs/biochem/eda/wall_gen_prec_mirror`.
- **`go_wg_prec_seed.ps1`** — train-time sparse commitment on prec-iter stack (`frontier_hops` + `nucleation_topk` in typed config); warm-start `WG_prec_iter`; holdout `patient020`. Primary `-Leg WG_prec_seed`; A/B `WG_prec_seed_fh2` / `WG_prec_seed_tk02`.
- **`go_wg_prec_seed_aux.ps1`** — early seed-location aux on prec-iter (`seed_aux_weight`, no hard frontier); warm-start `WG_prec_iter`; select clot score + seed panel on `patient020`.
- **`go_wg_prec_front.ps1`** — front/recall FT from `WG_prec_iter` (underpred up, gate/step FP down; no hard mask; seed_aux off); select clot score + front_speed/FN on `patient020`.
- **`go_wg_prec_physfp.ps1`** — **current path**: FP geography viz then one FT from `WG_prec_iter` (`WG_prec_physfp` = physical_fp_gating, or `-Leg WG_prec_cloop`). Gate = primary `deploy_clot_f1` on `patient020` (mass hard [0.5,1.5]; never promote score alone). `-FpGeoOnly` for the cheap diagnostic.
- **`go_wg_prec_sites.ps1`** — clot-rich N+ revisit with the fixed prec loss (no freeze); holdout `patient020`.
- **`go_wg_prec_loao.ps1`** — clot-rich LOAO with tight mass/FP (stronger than prec_iter); spray-abort; init from best small-cohort ckpt.
- **`go_wg_prec_queue.ps1`** — autonomous post-prec_iter queue: finish/eval prec_iter → mirror → gated sites (~5h GPU budget).
- **`go_wg_physgat_ab.ps1`** — research-only PM-GAT vs SAGE control (`WG_physgat_01` / `WG_physgat_ctrl`). **Parked:** cold physics_gat is not an easy drop-in (spray / ~0.21 p020); keep wall-gen focus on SAGE (`WG_featfix_03` / phase1). Do not launch by default.
- **`go_wg_stenosis_subcohort_ft.ps1`** — stenosis/aneurysm sub-cohort recall FT (docs/WALL_MODEL_PLAN.md s9): warm-start `WG_clotrich_nplus`, underpred/fp loss flipped 4.0/4.0 (was 2.0/8.0), frozen backbone, select on strict F1 + front_speed/FN-FP; default train `039,040,041,042,044`, holdout `patient043`, pocket-gate pct 25. Zero-shot floor to beat: `deploy_clot_f1=0.650`. Default train set departs from the sealed `WALL_GEN_BATCH_1B_*` split (see in-script warning) — pass `-TrainAnchors` to use the sealed one instead.
- **`eval_mat_growth_simple.py`** — cohort metrics (`--offwall-ckpt` / `--two-model-route`); prints `compound_gates` when two-model ON.
- **`eda_compound_lumen_bottlenecks.py`** — hop-ge2 prevalence + Arm S failure taxonomy EDA.
- **`diagnose_lumen_001_vs_007.py`** — deploy-time hop hist: why 001 misses lumen vs 007.
- **`diagnose_crack_001_root.py`** — root cause: compound-val full-graph static vs eval wall-band (A_floor=0).
- **`diagnose_001_signs_of_life.py`** — fast (~min) teacher vs closed-loop lumen fire on 001 (band vs global).
- `go_viz_mat_w_wc_canonical.ps1`, `go_viz_pivot3_hop_analysis.ps1`
- `viz_species_gnn_deploy.py` / `go_species_gnn_deploy_viz.ps1` — species/clot timeline viz.

## Research geometry-sensitivity sweeps

- `go_research_sweep.ps1` / `run_research_sweep.py` — vessel geometry × physics arms (stenosis, aneurysm, Re, width, bend, …) scored with transferable research parameters against **locked canonical** biochem at run time.
- Configs: `configs/research_sweeps/*.json`
- Docs: [`docs/RESEARCH_SWEEPS.md`](../docs/RESEARCH_SWEEPS.md)

## Visualization

- Steady kinematics + GraphSAGE deploy smoke: `python -m src.evaluation.visualize_pipeline` (optional `--steady-kin-only`).
- Batch steady-kin: `steady_kin_viz_cohort.py`
- Customer Predict GUI: `go_customer_predict.ps1`

## A/B and gates

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
