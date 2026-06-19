# Model nomenclature (SciML-accurate)

Names should reflect **what each piece is** and **what makes our Stage-A flow model distinct** from generic "GNN" or "GINO" labels.

**Programmatic source of truth:** [`src/model_nomenclature.py`](../src/model_nomenclature.py)

## Quick map

| Canonical ID | Acronym | SciML category | Code class | Legacy alias |
|--------------|---------|----------------|------------|--------------|
| **`pmgp_deq_kine`** | **RGP-DEQ** | Rheology-coupled graph DEQ | `GINO_DEQ` | `gino_deq_kine`, `gino-deq-kine`, `pmgp-deq-kine` |
| `species_graphsage` | — | Discrete-time GraphSAGE operator | `SpeciesDualHeadContinuousGNN` | `species_gnn` |
| `gelation_beta` | — | Scalar calibration | — | `viscosity_beta` |
| `clot_trigger_physics` | — | Mechanistic physics closure | — | `clot_phi` |
| `biochem_deploy` | — | Composable hybrid SciML pipeline | `BiochemDeployStack` | `biochem_gnn` |
| `gnode_biochem` | GNODE | Graph neural ODE (research) | `GNODE_Phase3` | `train biochem` |

Legacy IDs still resolve in manifests, CLI aliases, and import paths.

---

## 1. Stage A flow: `pmgp_deq_kine` / **RGP-DEQ** (`GINO_DEQ`)

### Name

- **RGP-DEQ** = **R**heology-guided **G**raph-**P**erceiver **DEQ**
- Canonical id: `pmgp_deq_kine`
- Former display acronym in older docs: **PMGP-DEQ**
- Code class remains `GINO_DEQ`; internal block `GINOBlock` is **legacy** (graph + global mixing), not Li et al. GINO

### Three distinguishing features (vs generic PI-GNN / implicit GNN / FNO-DEQ)

These are the architectural claims worth citing in papers and READMEs:

1. **Physics-modulated GAT (`MultiHeadPhysicsGATConv`)** — edge attention logits are biased by **advection**, **wall-rheology**, and **curvature** priors, SDF-decayed toward the bulk. Not plain GCN/SAGE or unmodulated GAT.

2. **Perceiver global mixing (`AttentionGlobalMixingBlock`)** — fixed global tokens **cross-attend** the mesh (strictly within each graph in a batch), MLP-process, then **broadcast** back. Captures long-range pressure/flow coupling without stacking many local layers.

3. **μ feedback inside the DEQ loop** — equilibrium solve finds `z*` such that `z* = f(z*, mu(z*))`: `mu(z)` is decoded, re-encoded into latent, and fed back **before each** `GINOBlock` step. Rheology is part of the fixed point, not only a post-decode head.

Together: Anderson/Picard root-finding on a **rheology-coupled** implicit layer built from **PM-GAT + Perceiver**, then SIREN/linear decode of `[u,v,p]`.

### Full stack (for orientation)

1. Fourier positional encoding on `(x, y, SDF, wall normal)` + priors  
2. MLP encoder → initial `z`  
3. DEQ loop (features above)  
4. Decode `[u, v, p]` and `mu_nd`

Implementation: [`src/architecture/ginodeq.py`](../src/architecture/ginodeq.py).

Train: `python -m src.bin.main train pmgp-deq-kine` (alias: `train gino-deq-kine`).

### Prior art (same family, different instantiation)

| Method | Overlap | Gap vs RGP-DEQ |
|--------|---------|-----------------|
| [ψ-GNN](https://arxiv.org/html/2302.10891) | PI + implicit GNN + unstructured mesh | Plain MPNN; no PM-GAT, Perceiver, or μ-in-loop |
| [FNO-DEQ](https://openreview.net/forum?id=FzXsSCF50t) | DEQ + steady PDE | FNO on regular grids, not vascular graph stack |
| [Physics-guided graph DEQ](https://researchportal.vub.be/en/publications/physics-guided-graph-convolutional-deep-equilibrium-network-for-e/) | PI + graph + DEQ | GCN + env PDE, not PM-GAT–Perceiver–μ coupling |
| Feedforward hemo GNNs | Vascular graphs | No DEQ / fixed point |

### What it is **not**

| Label | Why it does not fit |
|-------|---------------------|
| **GINO (Li et al.)** | No GNO→FNO operator; different architecture |
| **PI-GNN (generic)** | Underspecifies PM-GAT, Perceiver, and μ-in-loop |
| **FNO / DeepONet** | No spectral / branch-trunk operator on grids |
| **Neural ODE** | Steady fixed point, not `dz/dt` |

### Preferred phrasing

- **Paper title line:** μ-coupled PM-GAT–Perceiver DEQ for steady non-Newtonian flow on unstructured graphs  
- **Short:** RGP-DEQ  
- **Logs:** `pmgp_deq_kine (RGP-DEQ)`

---

## 2. Deploy species: `species_graphsage`

A **learned discrete-time operator** on the **wall-band subgraph** (~1-hop from ceiling):

- **Backbone:** 3-layer **GraphSAGE** (`SpeciesSnapshotGNN`).
- **Inputs:** frozen `z_kin` from `GINO_DEQ.solve_latent()` (RGP-DEQ equilibrium latent) + normalized SDF.
- **Outputs:** FI and Mat only; other species pinned at inference.
- **Deploy variant:** dual-head spatial gate × magnitude delta; autoregressive pushforward.

See [`src/core_physics/species_snapshot_gnn.py`](../src/core_physics/species_snapshot_gnn.py).

---

## 3. Gelation scale: `gelation_beta`

Single global scalar (learned offline) on Mat before physics gelation readout. Legacy dir: `outputs/biochem/biochem_gnn/viscosity/`.

---

## 4. Clot trigger: `clot_trigger_physics`

Mechanistic closure (Carreau + gelation + nucleation). Not learned in deploy baseline.

---

## 5. Full deploy stack: `biochem_deploy`

```
pmgp_deq_kine           [frozen RGP-DEQ checkpoint, Stage A]
  -> species_graphsage  [trained]  wall-band GraphSAGE pushforward (FI/Mat)
  -> gelation_beta      [trained]  global Mat scale
  -> clot_trigger_physics [equations] Carreau + gelation + nucleation phi
  -> flow_coupling      [future]   mu -> RGP-DEQ MU_PRIOR refresh
```

Train: `python -m src.bin.main train biochem-deploy`. Launcher: `scripts/go_biochem_gnn.ps1`.

Legacy package/artifact paths (`biochem_gnn/`, etc.) unchanged.

---

## 6. Research biochem: `gnode_biochem` (`GNODE_Phase3`)

Full-mesh **graph neural ODE** — distinct from deploy. Reuses RGP-style physics-GAT blocks (`GINOBlock`) inside the ODE derivative, not the full RGP-DEQ equilibrium loop.

---

## CLI and manifest conventions

| Use | Name |
|-----|------|
| Kinematics train (canonical) | `pmgp-deq-kine` |
| Kinematics train (legacy) | `gino-deq-kine` |
| Component key (canonical) | `pmgp_deq_kine` |
| Component key (legacy) | `gino_deq_kine` |
| Stack id | `biochem_deploy` |

New manifests and logs should prefer **`pmgp_deq_kine` / RGP-DEQ**; old checkpoints and scripts keep working via `resolve_model_id()`.

---

## Related docs

- [BIOCHEM_GNN.md](BIOCHEM_GNN.md)
- [KINEMATICS_BEST_ARCHITECTURE.md](KINEMATICS_BEST_ARCHITECTURE.md)
- [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)
