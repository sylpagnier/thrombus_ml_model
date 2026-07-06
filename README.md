# HemoGINO

**HemoGINO** is a state-of-the-art **Scientific Machine Learning (SciML)** framework that acts as a mesh-agnostic graph neural surrogate for vascular blood flow (hemodynamics) and coupled biochemistry (thrombosis/clotting). By combining physics-informed losses, deep equilibrium networks, and graph operators, HemoGINO provides real-time predictions of velocity, pressure, viscosity, and clot growth dynamics on complex vascular structures, running orders of magnitude faster than conventional Computational Fluid Dynamics (CFD) solvers.

---

## 🚀 Key Architectural Features

### 1. Rheology-Guided Graph-Perceiver DEQ (RGP-DEQ)
At the core of the kinematics solver is **RGP-DEQ** (`pmgp_deq_kine`), a Deep Equilibrium (DEQ) graph architecture that models steady non-Newtonian flow (`[u, v, p, μ_eff]`) on unstructured vessel graphs:
*   **Physics-Modulated GAT (`MultiHeadPhysicsGATConv`)**: Edge attention logits are dynamically biased by physical priors (advection, curvature, wall normals, and Signed Distance Fields).
*   **Perceiver Global Mixing (`AttentionGlobalMixingBlock`)**: Fixed global latent tokens cross-attend the mesh nodes, enabling long-range pressure-velocity coupling without requiring deep stacks of local message passing.
*   **Rheology-Coupled Fixed-Point Solve**: Fluid viscosity feedback is evaluated *inside* the DEQ loop (using Picard or Anderson acceleration), ensuring the non-Newtonian rheology (Carreau model) is satisfied at the fixed-point equilibrium.

### 2. Physics-Informed Joint Training (PINN)
Training blends supervised CFD labels with analytical physics constraints:
*   **50/50 Anchor Mix**: Optimizes training by mixing 50% supervised nodes (COMSOL anchors) and 50% unsupervised nodes (non-anchors).
*   **PDE Residual Losses**: Enforces mass conservation (divergence-free flow `∇·u = 0`), momentum conservation (Navier-Stokes residuals), and wall boundary conditions (no-slip) across unsupervised regions, serving as a powerful regularizer for out-of-distribution generalization.

### 3. Sim-to-Real Domain Adaptation via LoRA
Adapts models trained on synthetic vascular geometries to real clinical scans:
*   **Low-Rank Adaptation (LoRA)**: Intercepts GNN and MLP weight projections with parameter-efficient spectral/low-rank adapters.
*   **Zero-Shot Inference**: Bridges the sim-to-real gap, enabling rapid adaptation to diverse patient cohorts without catastrophic forgetting or expensive retraining.

### 4. Coupled Biochemistry Stack (`biochem_deploy`)
A modular pipeline modeling clot growth and species transport:
*   **GraphSAGE Pushforward (`species_graphsage`)**: A 3-layer GraphSAGE operator that predicts temporal species transport (Autologous Fibrin / Platelets) on the wall-band subgraph.
*   **Gelation Calibration & Readout**: Integrates a learned scalar calibration (`gelation_beta`) and a mechanistic physics closure (`clot_trigger_physics`) to map chemical concentrations to physical thrombus formation and dynamic flow occlusion.

### 5. Local Kinematic Corrector
Predicts localized velocity diversion residuals `[dU, dV]` induced by micro-clots:
*   **Local GATv2 Attention**: Instead of re-solving the expensive global flow field when a clot forms, a local k-hop GNN patch routes flow around/over the clot.
*   **Energy-Normalized Loss**: Trained on COMSOL "Patch Factory" simulations using a relative loss function that prevents small-signal/low-shear regions from degrading accuracy.

---

## 💻 Demos & Interactive Visuals

### Parametric Flow GUI Demo
HemoGINO includes an interactive GUI that allows users to design arbitrary 2D vessels, drag wall control points in real-time, generate meshes via Gmsh, and watch the RGP-DEQ solver instantly predict fluid flow.

Run the interactive flow demo:
```powershell
python -m src.bin.main inspect flow -- --rheology carreau
```
*(Add `--no-gui` to run headlessly and save static visual outputs to `outputs/reports/figures/kinematics/`)*

---

## 🛠️ Quick Start

### 1. Installation
Install the required packages (requires Python 3.9+ and PyTorch):
```powershell
pip install -r requirements.txt
```

### 2. Dataloading & Generation
Generate Newtonian/non-Newtonian kinematics graphs and biochemical meshes:
```powershell
# Kinematics pipelines
python -m src.data_gen.pipeline_kinematics

# Biochemistry pipelines
python -m src.data_gen.pipeline_biochem
```

### 3. Running Training
Orchestrate the Stage-A kinematics training followed by Stage-B biochem models:
```powershell
# Full orchestrated pipeline
python -m src.bin.orchestrate all

# Train RGP-DEQ kinematics flow surrogate only
python -m src.bin.main train kinematics

# Train biochem deploy stack only
python -m src.bin.main train biochem-deploy
```

### 4. Running Tests
Run the deterministic suite verifying physics kernels, CLI routing, boundary conditions, and GNN shape safety:
```powershell
# Full test suite
pytest src/tests/

# Kinematics suite only
pytest src/tests/ --suite=kinematics

# Biochem suite only
pytest src/tests/ --suite=biochem
```

---

## 📊 Performance & Validation

*   **Kinematics (Stage A)**: Achieves a **relative L2 error of 8.70%** (epoch 119) for velocity/pressure fields on unseen geometries.
*   **Local Corrector**: Achieves a **global relative L2 error of 15.9%** on localized clot diversion patches, showing highly robust behavior even in complex low-shear boundary regions.
*   **Biochemistry Stack**: Reaches an **F1 score of ~0.70 - 0.73** on real clinical scans (e.g. `patient007`) under zero-shot deploy conditions.

---

## 📁 Repository Structure

```text
├── src/
│   ├── architecture/     # RGP-DEQ, DEQ solvers, LoRA, GNN layers
│   ├── core_physics/     # PDE kernels, Navier-Stokes residuals, rheology models
│   ├── data_gen/         # Mesh processors, quad-patch generator, graph builders
│   ├── training/         # Dataloaders, curriculum schedules, trainers
│   ├── bin/              # Unified CLI main router and pipeline orchestrator
│   ├── tools/            # Interactive Matplotlib GUIs, live inspectors
│   └── tests/            # Deterministic regression and unit tests
├── docs/                 # In-depth architectural designs and log sweeps
├── comsol_models/        # Reference COMSOL Multiphysics source files
├── requirements.txt      # Dependency specification
└── README.md             # Landing page
```

For a deeper dive into terminology, data channels, and implementation details, see [`docs/PROJECT_CONTEXT.md`](docs/PROJECT_CONTEXT.md).
