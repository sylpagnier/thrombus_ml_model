# Species GNN deploy baseline (canonical ML clot stack)

**Status:** locked baseline (2026-06-12)  
**Role:** first **fully deployable ML** clot model in this repo — compare all future clot-ML experiments against this stack.

Related: [T0_RUNG_LADDER.md](T0_RUNG_LADDER.md), [SPECIES_TEMPORAL_ML.md](SPECIES_TEMPORAL_ML.md), [DEPLOY_ARCHITECTURE.md](DEPLOY_ARCHITECTURE.md).

---

## What it is

Closed-loop deploy path with **no GT species** at inference:

```text
geometry -> RGP-DEQ kinematics (optional pred flow)
         -> wall-band species GNN rollout (s34)
         -> global viscosity beta (s35)
         -> COMSOL Carreau + gelation (FI/Mat)
         -> nucleation clot phi
```

This replaces **R4.s0 inc40 rules** (~0.408 F1 @ t=53 on patient007) as the primary clot-shape baseline for Rung 4 / T0 deploy.

**Not the same as** `outputs/biochem/clot_baseline/` (GNODE teacher + MLP mu-map clot-phi — different architecture and inference entry point).

---

## Locked artifacts

| Path | Role |
|------|------|
| `data/reference/species_gnn_deploy_baseline.json` | Canonical reference (manifest + eval snapshot + metadata) |
| `outputs/biochem/species_gnn_deploy_baseline/manifest.json` | Runtime deploy manifest |
| `outputs/biochem/species_gnn_deploy_baseline/species_gnn_best.pth` | Global s34 GNN (copy) |
| `outputs/biochem/species_gnn_deploy_baseline/viscosity_beta.pth` | s35 gelation beta (copy) |
| `outputs/biochem/species_gnn_deploy_baseline/loao/holdout_*/best.pth` | Per-anchor LOAO folds (copy) |
| `outputs/biochem/species_gnn_deploy_baseline/eval_summary.json` | Metrics at promotion time |

Training sources (not overwritten on promote): `species_snapshot_s34`, `species_snapshot_s35`, `species_gnn_loao/`.

---

## Metrics @ promotion (gt flow, manifest picks)

| Anchor | Species GNN F1@t53 | s0 rules | Notes |
|--------|-------------------|----------|-------|
| patient001 | 0.961 | 0.602 | LOAO fold |
| patient002 | 0.627 | 0.400 | LOAO fold |
| patient003 | 0.557 | 0.357 | LOAO fold |
| patient004 | 0.336 | 0.163 | global s34; wall carpet |
| patient006 | 0.133 | 0.011 | LOAO + beta=0.15 |
| **patient007** | **0.701** | **0.408** | **val anchor; global s34** |

LOAO holdout mean F1@t53 (gt): **~0.523** (+0.331 vs s0 per anchor).

---

## Commands

**Promote / lock (after LOAO train + pick + eval):**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_promote_species_gnn_baseline.ps1" -Gate -Viz
```

**Smoke predict (new vessel or anchor):**

```bash
python -m src.inference.predict_species_gnn_deploy \
  --graph data/processed/graphs_biochem_anchors/patient007.pt \
  --flow kinematics
```

**Eval vs s0:**

```bash
python scripts/eval_t0_rung4_species_gnn_loao.py --flow both
```

**Viz ladder (GT | s0 | GNN):**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_species_gnn_deploy_viz.ps1"
```

**Rung 4 step eval:**

```bash
python scripts/eval_t0_rung4_step.py --anchor patient007 --step species_gnn
```

**Gate (CI / pre-merge):**

```bash
python scripts/check_species_gnn_baseline_gate.py
```

Env helper: `src/inference/species_gnn_deploy_env.py` (`SPECIES_GNN_DEPLOY_MANIFEST` overrides default reference path).

---

## When to re-promote

Re-run `go_promote_species_gnn_baseline.ps1` when:

1. A new species GNN or beta checkpoint **beats this baseline** on LOAO eval (especially patient007 vs 0.408 s0).
2. LOAO fold picks change materially (`pick_species_gnn_loao_ckpts.py`).

Document delta in eval_summary and bump `version` in the reference JSON if the stack architecture changes.

---

## Comparison policy

| Experiment type | Compare against |
|-----------------|-----------------|
| New species GNN train (s36+) | This baseline LOAO eval + patient007 F1 |
| Rule / S-star Rung4 heads | s0 rules **and** this baseline |
| MLP mu-map clot track | Separate (`clot_baseline` manifest); not interchangeable |

Report: `delta_f1_vs_baseline` and `delta_f1_vs_s0` on the same anchor set and flow mode (gt vs kinematics).

## ~4h architecture sweep

Curated tuning + architecture legs (warm-start from locked baseline):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_species_gnn_s34_arch_sweep_4h.ps1" -PromoteWinner
python scripts/summarize_species_gnn_s34_sweep.py
```

Output: `outputs/biochem/sweep_species_gnn_s34_arch/summary.json`

New env flags (s34 train): `SPECIES_CONTINUOUS_DELTA_RESIDUAL`, `SPECIES_CONTINUOUS_TEMPORAL_OFFSET`, `SPECIES_CONTINUOUS_SCORE_CLOUT_W`.
