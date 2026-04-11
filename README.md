# HemoGINO — Chemo-Hemodynamic Surrogate

Multi-fidelity **rGINO-DEQ** with gradient-aware physics kernels for **non-Newtonian thrombosis**: predict coupled flow fields and steady-state thrombus boundaries on patient meshes in seconds instead of full CFD.

## Architecture (high level)

- **Encoder**: Physics-informed node features (e.g. SDF, shear-rate potential).
- **Solver core**: DEQ fixed point \(Z^\* = \mathrm{GINO}(Z^\*, X_{\mathrm{geo}}; \theta)\).
- **Adaptation**: LoRA on kinematics for small patient-specific datasets.

## Physics fidelity: config “tiers” vs training “stages”

**Tiers** (`PhysicsConfig` / `VesselConfig`: `tier1`, `tier2`, `tier3`) describe **what physics is active** in the model and data:

| Tier | Role |
|------|------|
| **tier1** | Newtonian, laminar-style kinematic baseline. |
| **tier2** | Adds Carreau–Yasuda non-Newtonian rheology \(\mu(\dot\gamma)\). |
| **tier3** | Adds coupled biochemistry / clot-related dynamics on the flow backbone. |

**Stages** describe **training scripts and checkpoint layout**, formalizing a **Predictor–Corrector** architecture to stabilize highly coupled physics:

| Stage | Scripts | Core mechanics | Checkpoints (default) |
|-------|---------|----------------|------------------------|
| **Stage A — Predictor** | `src/training/train_t1_predictor.py`, `train_t2_predictor.py`, and Tier 3 **warmup** in `train_t3_corrector.py` | **Decoupled transport:** biochem ADR over a static flow field (`mu_ratio = 1.0`). Kinematic LoRA is **frozen**. | `outputs/stage_a/` (Tier 1/2); Tier 3 warmup weights use `outputs/stage_b/` |
| **Stage B — Corrector** | Tier 3 **post-warmup** in `src/training/train_t3_corrector.py` | **Coupled rheology:** `mu_ratio` ramps from neutral toward `mu_ratio_max`; LoRA **unfreezes** for kinematic co-adaptation to feedback. | `outputs/stage_b/` |

**Orchestrator**

```bash
python -m src.main a                # Tier 1 then Tier 2 (Stage A)
python -m src.main b                # Tier 3 corrector (Stage B)
python -m src.main all              # A then B
python -m src.main a --skip-tier1   # Start Stage A at Tier 2
```

Do not assume **Tier 3** is where most kinematic accuracy comes from; Tier 1/2 are the primary transport-focused stages.

## Repository layout (`src/`)

| Path | Purpose |
|------|---------|
| `src/core_physics/` | Navier–Stokes–consistent kernels, rheology, biochem coupling interfaces. |
| `src/architecture/` | GNODE / DEQ core, LoRA injection. |
| `src/data_pipeline/` | Mesh → graph, COMSOL export ingestion, Tier 3 plumbing. |
| `src/training/` | Curriculum, `train_t1_predictor`, `train_t2_predictor`, `train_t3_corrector`. |
| `src/evaluation/` | Benchmarks, visualization helpers. |
| `src/utils/` | Paths, metrics, inference, shared kinematics helpers. |
| `src/tools/` | **Interactive** inspectors (not run by pytest). |
| `src/tests/` | Pytest suite. |

**Artifacts (not source):**

| Location | Contents |
|----------|----------|
| `data/` | Raw meshes, COMSOL exports, processed graphs (`data/raw/…`, `data/processed/…`); benchmark scratch under `data/benchmark/`. |
| `outputs/stage_a/`, `outputs/stage_b/` | Current checkpoints (Stage A / B). |
| `outputs/reports/` | CSVs, validation figures, training diaries, Tier 3 logs. |
| `comsol_models/` | Reference `.mph` templates. |

Checkpoint files must live under `outputs/stage_a/` or `outputs/stage_b/` with the names the training scripts expect (for example `tier1_best_physics.pth`). Older top-level `models/` weights are no longer read automatically—copy them into the matching `outputs/stage_*` folder if you still need them.

## Diagnostics

- **Raw COMSOL anchors** (`.npz` from CFD exports):  
  `python -m src.tools.inspect_anchor_cfd --tier tier2 --scan-only`  
  Writes `outputs/reports/<tier>_anchor_health.csv`; omit `--scan-only` for interactive plots.

- **Processed graphs** (`.pt`) and overlap with CFD:  
  `python -m src.tools.inspect_graph_sample --inspect-sample --tier tier1`

- **DEQ / Anderson manual audit** (matplotlib): `python -m src.tools.verify_deq_convergence`

Full agent-oriented map: [docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md).

## Tests

```bash
pytest src/tests
```

If Tier 1/2 COMSOL anchor tests fail only on boundary (`l_bc`) caps, your masks may not match the label convention — see `PHASE1_PHYSICS_CHECK_BC` in `src/tests/test_kinematics_physics_kernels.py` and [docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md).

## Documentation

- [docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md) — stages, tiers, paths, tools, and conventions for agents and contributors.
