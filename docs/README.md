# HemoGINO — Documentation Index

Welcome to the HemoGINO documentation index. This folder contains architectural records, design choices, training logs, and validation summaries for the hemodynamics and biochemistry surrogate models.

---

## 🗺️ Core Project Overview

*   **[PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)** — **Primary Entry Point:**
    *   System goals, Stage-A (kinematics) vs Stage-B (biochemistry) terminology, configuration channels, and directory structures.
    *   Guidelines for simulation boundary assumptions, anchor vs non-anchor nodes, and deterministic unit/regression test policies.
*   **[MODEL_NOMENCLATURE.md](MODEL_NOMENCLATURE.md)** — Explains our SciML acronyms and namespaces:
    *   **RGP-DEQ** (`pmgp_deq_kine`), `species_graphsage`, `gelation_beta`, `clot_trigger_physics`, and the composite `biochem_deploy` stack.
    *   Highlights key differentiators from generic GNNs/FNOs (attention biasing, global perceiver cross-attention, in-loop viscosity feedback).

---

## 🌊 Stage A: Fluid Kinematics & Rheology

*   **[KINEMATICS_BEST_ARCHITECTURE.md](KINEMATICS_BEST_ARCHITECTURE.md)** — Detailed record of the best-performing GINO-DEQ kinematics solver configuration.
*   **[LOCAL_KINEMATIC_CORRECTOR.md](LOCAL_KINEMATIC_CORRECTOR.md)** — Deep dive into the localized GATv2 velocity diversion model:
    *   Explains the **Patch Factory** data generator (Couette-freestream quad patches).
    *   Analyzes the transition from standard MSE to energy-normalized **relative loss** to improve accuracy in low-shear zones.
*   **[COMSOL_PHYSICS_VALIDATION.md](COMSOL_PHYSICS_VALIDATION.md)** — Physical validation of Newtonian and Carreau non-Newtonian flow solver boundaries against analytical limits.
*   **[COMSOL_MU_RHEOLOGY_CHECKLIST.md](COMSOL_MU_RHEOLOGY_CHECKLIST.md)** — Checklist ensuring mathematical and dimensional parity of viscosity equations between COMSOL and python code.

---

## 🩸 Stage B: Coupled Biochemistry & Clotting

*   **[BIOCHEM_GNN.md](BIOCHEM_GNN.md)** — Design of the deployment-ready biochemistry stack:
    *   Contrast between `biochem_deploy` (discrete GraphSAGE pushforward on wall-band subgraphs) and `gnode_biochem` (continuous Graph Neural ODE).
    *   Environment configuration tables and evaluation horizon metrics.
*   **[BIOCHEM_GNN_BASELINES.md](BIOCHEM_GNN_BASELINES.md)** — Comparison metrics, validation history, and leaderboard for species training models.
*   **[DEPLOY_ARCHITECTURE.md](DEPLOY_ARCHITECTURE.md)** — Production deployment strategy (Track A: static COMSOL flow fields vs Track B: dynamic GINO-DEQ flow prediction).
*   **[CLOT_PHI_BASELINE.md](CLOT_PHI_BASELINE.md)** — Evaluation notes on wall-local clot trigger models and simple boundary-distance thresholding.

---

## 🧠 Historical Lessons & Sweep Records

*   **[BIOCHEM_LEGACY_LESSONS.md](BIOCHEM_LEGACY_LESSONS.md)** — Consolidated engineering takeaways from older biochemical training ladders and learning schedules.
*   **[SPECIES_LEARNING_STRATEGY.md](SPECIES_LEARNING_STRATEGY.md)** — In-depth discussion of advection-diffusion-reaction (ADR) dynamics, temporal rollout stability, and pushforward horizons.
