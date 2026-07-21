# Mat-growth (canonical baseline)

Mat-growth is the active research path on top of locked **`biochem_gnn`**: warm-start the wall-band species GraphSAGE and improve **deploy clot** footprint (precision-aware) without reviving retired GNODE / clot-phi ladders.

## Locked baseline (2026-07-19)

| | |
|--|--|
| **Leg** | `WC_v7_clot_phi_mse` |
| **Checkpoint** | `outputs/biochem/biochem_gnn/locked/species_gnn_best.pth` (local) |
| **Aliases** | `mat_canonical_deploy/species/best.pth`, `species/best.pth` |
| **Manifests** | `data/reference/biochem_gnn_baseline.json`, `data/reference/mat_canonical_deploy.json` |
| **Cohort (mean)** | clot score **~0.791**, clot F1 **~0.767**, Mat F1 **~0.714** |

Selection metrics for new legs: all-anchor `deploy_clot_f1` and `deploy_clot_relaxed_prec` (see each leg `compare.json`).

New work should **warm-start from locked** (or `species/best.pth`) and apply env via `mat_growth_leg_spec("WC_v7_clot_phi_mse")` unless deliberately ablating.

## How to run

Supported launchers ([`scripts/README.md`](../scripts/README.md)):

| Launcher | Role |
|----------|------|
| `go_fresh_canonical.ps1` / `go_fresh_canonical_finish.ps1` | Promote WC legs into locked baseline |
| `go_mat_w_wc_canonical.ps1`, `go_mat_growth_simple.ps1`, `go_mat_growth_ladder.ps1` | Mat-growth training ladders |
| `go_off_wall_clot_sweep_6h.ps1` | Off-wall pivot (Pivot 3 occlusion survived) |
| `go_wc_v7_compound_growth_abc_orig10_9h.ps1` | Compound A/B/C on original anchors (~9 h) |
| `go_wc_v7_firewall_fix_seq.ps1` | Firewall / hop-stratified fix sequence |
| `eval_mat_growth_simple.py` | Cohort metrics (`--offwall-ckpt` / `--two-model-route`) |
| `go_viz_mat_w_wc_canonical.ps1`, `go_wc_v7_compound_orig10_viz.ps1` | Viz |

## Design notes (short)

- Prefer **precision-aware** clot scoring (anti wall-paint) over raw recall.
- Off-wall (hop >= 1) metrics matter for deploy claims; restore `meta.env_overrides` on eval.
- Compound / firewall sequences are **research budgets**, not the locked public baseline until promoted.

## Related

- Stack design: [BIOCHEM_GNN.md](BIOCHEM_GNN.md)
- Naming: [MODEL_NOMENCLATURE.md](MODEL_NOMENCLATURE.md)
- Historical leg tables and living TODO dump: [archive/MAT_GROWTH_SIM_TODO.md](archive/MAT_GROWTH_SIM_TODO.md)
- Historical baseline leaderboard: [archive/BIOCHEM_GNN_BASELINES.md](archive/BIOCHEM_GNN_BASELINES.md)
