# HemoRGP

Mesh-agnostic **scientific machine learning** for vascular hemodynamics and thrombosis.

**HemoRGP** predicts steady velocity, pressure, and effective viscosity on unstructured vessel graphs, then couples a frozen flow backbone to a GraphSAGE species pushforward and a mechanistic clot readout. The goal is CFD-quality fields at interactive cost on patient-like geometries.

| Layer | Canonical name | Role |
|-------|----------------|------|
| Product | **HemoRGP** | Hemodynamics + thrombus SciML |
| Stage A flow | **RGP-DEQ** (`rgp_deq_kine`) | Rheology-guided graph-perceiver DEQ |
| Stage B clot | **`biochem_gnn`** | Frozen flow + species GraphSAGE + gelation + clot physics |
| Optional | **Local kinematic corrector** | k-hop residual diversion around micro-clots |

Terminology and legacy aliases: [`docs/MODEL_NOMENCLATURE.md`](docs/MODEL_NOMENCLATURE.md).

---

## Architecture

### RGP-DEQ (Stage A)

Deep-equilibrium graph model for steady non-Newtonian flow `[u, v, p, mu_eff]` on vessel meshes:

- **Physics-modulated GAT** — attention biased by advection, curvature, wall normals, and SDF priors
- **Perceiver global mixing** — fixed latent tokens for long-range pressure–velocity coupling
- **Rheology inside the fixed point** — Carreau-style viscosity feedback evaluated in the DEQ loop (Picard / Anderson)

Training mixes supervised COMSOL anchors with unsupervised PDE residuals (continuity, momentum, wall BCs). A **50/50** anchor / physics-only mix is the preferred kinematics recipe.

### biochem_gnn (Stage B)

```text
RGP-DEQ (frozen)
  -> species_graphsage     wall-band GraphSAGE (FI / Mat)
  -> gelation_beta         scalar Mat scale
  -> clot_trigger_physics  Carreau + gelation + nucleation
  -> local_kinematic_corrector   [optional] local [dU, dV]
```

Deploy evaluation is **GT-species-free** at inference (resting ICs, predicted kinematics).

---

## Results (reference)

| Stage | Metric | Value | Notes |
|-------|--------|------:|-------|
| Kinematics | Val Rel L2 | **~0.087** | Production allfix + continuity finetune |
| Local corrector | Global Rel L2 | **~15.9%** | Patch-factory residuals on frozen flow |
| biochem_gnn (WC_v7) | Cohort clot F1 | **~0.767** | Locked baseline; clot score ~0.791 |

Manifests (no weights in git): [`data/reference/`](data/reference/). Checkpoints stay local under `outputs/` — see [`docs/PUBLISHING.md`](docs/PUBLISHING.md).

---

## Quick start

### Install

Python 3.9+, CUDA recommended for training.

```powershell
pip install -r requirements.txt
```

Bulk meshes, graphs, and COMSOL `.mph` files are **not** in this repository. Place them under `data/` and `comsol_models/` on your machine ([`docs/PUBLISHING.md`](docs/PUBLISHING.md)).

### Demo apps

```powershell
# Vessel Simulation Desktop UI (flow + biochem timeline)
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_customer_predict.ps1

# Parametric flow GUI (RGP-DEQ)
python -m src.bin.main inspect flow -- --rheology carreau
```

### Data pipelines

```powershell
python -m src.data_gen.pipeline_kinematics
python -m src.data_gen.pipeline_biochem
```

### Training

```powershell
# Stage A then Stage B (orchestrator)
python -m src.bin.orchestrate all

# Stage A only (RGP-DEQ)
python -m src.bin.main train rgp-deq-kine

# Stage B only (biochem_gnn)
python -m src.bin.main train biochem-gnn
```

Supported launchers: [`scripts/README.md`](scripts/README.md).

### Tests

```powershell
pytest src/tests/
```

---

## Repository layout

```text
src/                   Library: architecture, physics, training, tools, tests
scripts/               Supported launchers (+ scripts/archive/ for retired ladders)
docs/                  Active design docs (+ docs/archive/ for chronicles)
data/reference/        Small JSON manifests (tracked)
customer_geometries/   Inbox README only (uploads stay local)
outputs/               LOCAL — checkpoints, logs, figures (gitignored)
comsol_models/         LOCAL — COMSOL sources (gitignored)
```

---

## Documentation

| Doc | Contents |
|-----|----------|
| [`docs/PROJECT_CONTEXT.md`](docs/PROJECT_CONTEXT.md) | Goals, stages, source map, entry points |
| [`docs/MODEL_NOMENCLATURE.md`](docs/MODEL_NOMENCLATURE.md) | Canonical IDs and SciML naming |
| [`docs/BIOCHEM_GNN.md`](docs/BIOCHEM_GNN.md) | Deploy stack design |
| [`docs/KINEMATICS_BEST_ARCHITECTURE.md`](docs/KINEMATICS_BEST_ARCHITECTURE.md) | Locked RGP-DEQ recipe |
| [`docs/PUBLISHING.md`](docs/PUBLISHING.md) | Public vs local artifact policy |
| [`docs/README.md`](docs/README.md) | Full documentation index |

Contributor / agent shortcuts: [`AGENTS.md`](AGENTS.md).
