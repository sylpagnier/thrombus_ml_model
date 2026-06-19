# Biochem GNN pushforward architecture A/B (sage vs gnode)

**Date:** 2026-06-14  
**Status:** **Cancelled** (user stop) — partial run; no full-anchor eval summary.

## Question

Same deploy recipe (wall-band FI/Mat pushforward, dual-head, physics clot, pred-kine guiding checkpoint). Only swap dynamics trunk:

| Arm | Trunk | Class |
|-----|-------|-------|
| **sage** | GraphSAGE | `SpeciesDualHeadContinuousGNN` |
| **gnode** | GINO derivative on band | `SpeciesGnodeDualHeadContinuousGNN` |

Code: `src/core_physics/species_gnode_pushforward.py`, launcher `scripts/go_biochem_gnn_arch_ab.ps1`.

## Recipe (both legs)

Mirrors `go_biochem_gnn_global_guiding_5h.ps1`: 10 on-disk anchors, per-vessel `t0` caps, guiding `deploy_clot_score`, shared warm-start `global_guiding_5h/species/best.pth`, shared gelation beta (species-only train to avoid 4GB OOM on beta re-fit).

Env: `SPECIES_PUSHFORWARD_ARCH=sage|gnode`, `--arch` on `train_biochem_gnn` / `train_species_pushforward_continuous`.

## Run log

| Event | Note |
|-------|------|
| Smoke sage | OK; beta calibration OOM on 4GB GPU |
| Fix | Train `--step species` only; copy shared beta |
| Gnode warm-start | Strict load failed (head dim 321 vs 325); fixed partial init |
| Full sage | Reached ep 19 before cancel; ckpt saved |
| Gnode | Stopped at ep 7 (~20 min/ep); **cancelled** |
| Full eval | **Not run** (no `arch_ab_summary.json`) |

## Checkpoints (val p007, train-time deploy metric @ t53)

Metrics from `best.json` at checkpoint selection (not post-hoc all-anchor eval).

| Leg | Best ep | `deploy_clot_score` | `deploy_clot_guiding` | `deploy_clot_f1` @ t53 | `deploy_fi_f1` @ t53 |
|-----|---------|---------------------|------------------------|-------------------------|----------------------|
| **sage** | 19 | **0.568** | 0.510 | 0.551 | 0.492 |
| **gnode** | 1 | 0.432 | 0.436 | 0.426 | 0.305 |

Gnode last logged epoch (7): `deploy_clot_score` **0.403** (monotone decline after ep 1).

Paths:

- `outputs/biochem/biochem_gnn/arch_ab/sage/species/best.pth`
- `outputs/biochem/biochem_gnn/arch_ab/gnode/species/best.pth` (ep 1 best only)

Reference baseline (completed run): `global_guiding_5h` sage ~0.525 guiding @ p007 t200.

## Provisional conclusion

**Keep GraphSAGE (`sage`) for deploy species pushforward.**

Under the same recipe, sage warm-started from the same teacher and improved to **0.57** checkpoint score; gnode partial warm-start (snapshot heads + random GINO trunk) peaked at ep 1 (**0.43**) and regressed. Not a fair full-length comparison (gnode only 7/75 ep), but the trend + FI gap (~0.49 vs ~0.30) strongly favors sage for this band pushforward use case.

GNODE-style trunk remains appropriate for **full-mesh** `GNODE_Phase3` / biochem corrector — not this wall-band closed-loop deploy loop.

## Resume / re-run

```powershell
# Sage only (resume from arch_ab ckpt if present)
.\scripts\go_biochem_gnn_arch_ab.ps1 -Leg sage

# Eval only (when both ckpts exist)
.\scripts\go_biochem_gnn_arch_ab.ps1 -SkipTrain
```

To declare a winner formally: run eval on all anchors @ t53 + full deploy horizon, then `scripts/summarize_biochem_gnn_arch_ab.py`.
