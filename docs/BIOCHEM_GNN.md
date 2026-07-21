# Biochem deploy stack (`biochem_gnn`)

Canonical Stage-B pipeline: frozen **RGP-DEQ** flow + wall-band **GraphSAGE** species pushforward + **gelation** scale + mechanistic **clot** readout.

Naming: [MODEL_NOMENCLATURE.md](MODEL_NOMENCLATURE.md). Baseline metrics: [MAT_GROWTH.md](MAT_GROWTH.md).

## Components

```text
rgp_deq_kine              [frozen RGP-DEQ, Stage A]
  -> species_graphsage    [trained]  wall-band GraphSAGE (FI / Mat)
  -> gelation_beta        [trained]  global Mat scale
  -> clot_trigger_physics [equations] Carreau + gelation + nucleation
  -> local_kinematic_corrector [optional] k-hop [dU, dV]
```

## Artifact layout (local)

```text
outputs/biochem/biochem_gnn/
  locked/species_gnn_best.pth     # canonical (WC_v7_clot_phi_mse, 2026-07-19)
  mat_canonical_deploy/species/best.pth
  species/best.pth
  viscosity/beta.pth
data/reference/biochem_gnn_baseline.json
data/reference/mat_canonical_deploy.json
```

## Commands

```powershell
powershell -File .\scripts\go_biochem_gnn.ps1 -Step all -Gate -Viz

python -m src.bin.main train biochem-gnn -- --step all --all-anchors
python -m src.training.train_biochem_gnn --step species
```

Promote / lock: `python scripts/promote_biochem_gnn.py`.

## Deploy conventions

- Metrics and gates use each graph's **last macro-step** (full COMSOL timeline) unless `SPECIES_CONTINUOUS_DEPLOY_HORIZON` caps the horizon.
- Deploy-faithful rollout (via `apply_deploy_env()`): resting FI/Mat ICs, pin other species to rest, velocity from predicted kinematics.
- Only **FI + Mat** are learned (`STATE_DIM = 2`); other species stay at resting IC at inference.

## Python API

```python
from src.biochem_gnn import BiochemGNN, FlowMode

model = BiochemGNN.from_manifest(anchor="patient007", flow_mode=FlowMode.COUPLED)
out = model.rollout(data)  # out.phi_by_time, out.mu_by_time, out.species_series
```

Legacy: `BiochemDeployStack = BiochemGNN` (also `src.biochem_deploy`).

## Historical note

The GNODE teacher/corrector path (`train_biochem_corrector`) is **retired**. Condensed lessons: [BIOCHEM_LEGACY_LESSONS.md](BIOCHEM_LEGACY_LESSONS.md). Detailed leaderboards: [archive/BIOCHEM_GNN_BASELINES.md](archive/BIOCHEM_GNN_BASELINES.md).
