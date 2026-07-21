# Model nomenclature (SciML-accurate) — HemoRGP

Names should reflect **what each piece is**. Prefer **RGP-DEQ** / **biochem_gnn** over legacy GINO / PMGP / biochem_deploy labels.

**Programmatic source of truth:** [`src/model_nomenclature.py`](../src/model_nomenclature.py)

**Product name:** **HemoRGP** (Rheology-guided Graph-Perceiver hemodynamics + thrombus SciML). Former brand: HemoGINO (retired — not Li et al. GINO).

## Quick map

| Canonical ID | Acronym | SciML category | Code class | Legacy alias |
|--------------|---------|----------------|------------|--------------|
| **`rgp_deq_kine`** | **RGP-DEQ** | Rheology-coupled graph DEQ | `RGP_DEQ` | `pmgp_deq_kine`, `gino_deq_kine`, `GINO_DEQ` |
| `species_graphsage` | — | Discrete-time GraphSAGE operator | `SpeciesDualHeadContinuousGNN` | `species_gnn` |
| `gelation_beta` | — | Scalar calibration | — | `viscosity_beta` |
| `clot_trigger_physics` | — | Mechanistic physics closure | — | `clot_phi` |
| `local_kinematic_corrector` | — | Local residual GNN on frozen flow | `LocalKinematicCorrector` | `local_corrector` |
| **`biochem_gnn`** | — | Composable hybrid SciML pipeline | `BiochemGNN` | `biochem_deploy`, `BiochemDeployStack` |
| `gnode_biochem` | GNODE | Graph neural ODE (retired) | `GNODE_Phase3` | `train biochem` |

Legacy IDs still resolve in manifests, CLI aliases, and import paths.

---

## 1. Stage A flow: `rgp_deq_kine` / **RGP-DEQ** (`RGP_DEQ`)

### Name

- **RGP-DEQ** = **R**heology-guided **G**raph-**P**erceiver **DEQ**
- Canonical id: `rgp_deq_kine`
- Former ids: `pmgp_deq_kine`, `gino_deq_kine`
- Code class: `RGP_DEQ` (legacy alias `GINO_DEQ`); block: `RGPBlock` (legacy `GINOBlock`)

### Three distinguishing features (vs generic PI-GNN / implicit GNN / FNO-DEQ)

1. **Physics-modulated GAT (`MultiHeadPhysicsGATConv`)** — edge attention logits biased by **advection**, **wall-rheology**, and **curvature** priors, SDF-decayed toward the bulk.

2. **Perceiver global mixing (`AttentionGlobalMixingBlock`)** — fixed global tokens **cross-attend** the mesh, then **broadcast** back.

3. **μ feedback inside the DEQ loop** — equilibrium solve finds `z*` such that `z* = f(z*, mu(z*))`.

### Preferred phrasing

- **Paper title line:** μ-coupled PM-GAT–Perceiver DEQ for steady non-Newtonian flow on unstructured graphs  
- **Short:** RGP-DEQ  
- **Logs:** `rgp_deq_kine (RGP-DEQ)`

### What it is **not**

| Label | Why it does not fit |
|-------|---------------------|
| **GINO (Li et al.)** | No GNO→FNO operator; different architecture |
| **HemoGINO** | Retired product brand that implied GINO |

Train: `python -m src.bin.main train rgp-deq-kine` (aliases: `pmgp-deq-kine`, `gino-deq-kine`, `kinematics`).

Implementation: [`src/architecture/ginodeq.py`](../src/architecture/ginodeq.py).

---

## 2. Deploy species: `species_graphsage`

A **learned discrete-time operator** on the **wall-band subgraph** (~1-hop from ceiling):

- **Backbone:** 3-layer **GraphSAGE**.
- **Inputs:** frozen `z_kin` from `RGP_DEQ.solve_latent()` + normalized SDF.
- **Outputs:** FI and Mat only; other species pinned at inference.

---

## 3–4. `gelation_beta` / `clot_trigger_physics`

Unchanged roles (scalar Mat scale; mechanistic Carreau + gelation + nucleation).

---

## 4b. Optional coupling: `local_kinematic_corrector`

**Class:** `LocalKinematicCorrector` in `src/core_physics/coupled_shear_gnn.py`.

Cheap **k-hop residual GNN** that predicts `[dU, dV]` on top of frozen **RGP-DEQ** UV around clot nodes. Prefer this over viscosity injection into the DEQ (OOD). Wired optionally into `BiochemGNN` via `local_corrector_ckpt` / `set_local_corrector`.

Not required for the locked WC_v7 species baseline; used when coupled deploy needs flow diversion.

Doc: [LOCAL_KINEMATIC_CORRECTOR.md](LOCAL_KINEMATIC_CORRECTOR.md).

---

## 5. Full stack: `biochem_gnn`

```
rgp_deq_kine           [frozen RGP-DEQ checkpoint, Stage A]
  -> species_graphsage  [trained]  wall-band GraphSAGE pushforward (FI/Mat)
  -> gelation_beta      [trained]  global Mat scale
  -> clot_trigger_physics [equations] Carreau + gelation + nucleation phi
  -> local_kinematic_corrector [optional] k-hop [dU,dV] residual on clot nodes
  -> flow_coupling      [optional/future] broader clot->flow refresh
```

Train: `python -m src.bin.main train biochem-gnn`. Launcher: `scripts/go_biochem_gnn.ps1`.

Local corrector train: `python -m src.training.train_local_kinematic_corrector`. Doc: [LOCAL_KINEMATIC_CORRECTOR.md](LOCAL_KINEMATIC_CORRECTOR.md).

Python: `from src.biochem_gnn import BiochemGNN` (legacy: `BiochemDeployStack`, `src.biochem_deploy`).

---

## CLI and manifest conventions

| Use | Name |
|-----|------|
| Kinematics train (canonical) | `rgp-deq-kine` |
| Kinematics train (legacy) | `pmgp-deq-kine`, `gino-deq-kine` |
| Component key (canonical) | `rgp_deq_kine` |
| Stack id (canonical) | `biochem_gnn` |
| Stack id (legacy) | `biochem_deploy` |

New manifests and logs should prefer **`rgp_deq_kine` / RGP-DEQ** and **`biochem_gnn`**; old checkpoints keep working via `resolve_model_id()`.

---

## Related docs

- [BIOCHEM_GNN.md](BIOCHEM_GNN.md)
- [MAT_GROWTH.md](MAT_GROWTH.md)
- [KINEMATICS_BEST_ARCHITECTURE.md](KINEMATICS_BEST_ARCHITECTURE.md)
- [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)
- [README.md](README.md) — documentation index
