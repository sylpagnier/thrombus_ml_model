# HemoGINO — project context (agents & contributors)

This document gives **enough structure** to navigate the repo without re-reading the whole tree: terminology, **architecture**, entry points, artifacts, and where to change behavior safely.

## Goal

**HemoGINO** is a **mesh-agnostic** surrogate that predicts **velocity, pressure, viscosity, and clot-related fields** on vessel graphs faster than COMSOL, with physics-informed losses and a DEQ-style core.

Domain adaptation from synthetic to real patient geometries uses **LoRA**. During Stage B, LoRA bridges the sim-to-real gap across a diverse population of meshes, enabling zero-shot inference on new, unseen patient scans without retraining.

## Terminology: Tier vs Stage

- **Tier** (`tier1` | `tier2` | `tier3` / `tier3_patients` / `tier3_mix` in `PhysicsConfig` / `VesselConfig`): **which physics terms and data modes are enabled** (Newtonian → non-Newtonian → coupled biochem + transient graphs as configured).
- **Stage** (training pipeline): **Stage A** = predictor warm-up (Tier 1 + Tier 2); **Stage B** = Tier 3 corrector. This is about **checkpoint buckets and orchestration order**, not a different physical meaning of “tier.”

## Architecture (current)

### Model and physics

- **`GINO_DEQ`** (`src/architecture/ginodeq.py`): graph encoder → fixed-point **DEQ** loop (`core_physics.anderson`) over latent states; optional **global mixing** (`AttentionGlobalMixingBlock`) for long-range pressure-like coupling; optional **SIREN** spatial decoder (`architecture/siren_decoder.py`) when enabled by `PhysicsConfig`.
- **LoRA** adapters: `src/architecture/lora_injection.py` (spectral / low-rank hooks used where config enables LoRA for sim-to-real).
- **`PhysicsKernels`** (`src/core_physics/physics_kernels.py`): residual / BC / rheology interfaces shared by training and tests.
- **Kinematics losses** align with `src/utils/kinematics_physics_terms.py` (not legacy standalone `kinematics` wrappers).

### Configuration and tensor conventions

- **`PhysicsConfig`**, **`VesselConfig`**, **`BiochemConfig`**, **`CurriculumConfig`** — see `src/config.py`.
- **Predictor channels** — `PredChannels` / `STATE_CHANNEL_MU_EFF_ND`: canonical indices for `[u, v, p, μ_eff_nd, …]`.
- **Node features (Tier 1/2)** — `NodeFeat` slices: positions, SDF, wall normals, priors, optional hydraulic width `D(x)` and derivatives for geometric priors.
- **Tier 3 node features** — `Tier3NodeFeat` for clot/chemistry graphs.

### Training runtime (no monolithic “trainer” class)

Training is driven by **self-contained scripts** that own dataloading loops, schedulers, and checkpointing:

| Script | Role |
|--------|------|
| `src.training.train_t1_predictor` | Tier 1 GINO-DEQ warm-up; explorer / sweeps via `t1_explorer.py` and env vars (`TIER1_*`) |
| `src.training.train_t2_predictor` | Tier 2 non-Newtonian predictor; may bootstrap from Tier 1 `GINO_DEQ` weights in `outputs/stage_a/` |
| `src.training.train_t3_corrector` | Stage B: Tier 3 coupled corrector and graph semantics |
| `src.training.physics_curriculum` | Curriculum helpers consumed by training scripts |

Thin wrappers `train_t1.py` / `train_t2.py` delegate to the predictor entrypoints and exist so `python -m src.bin.main train t1` can resolve a small module surface.

### Routers

| Module | Role |
|--------|------|
| `python -m src.bin.orchestrate {a\|b\|all}` | Runs Stage A (`train_t1_predictor` → `train_t2_predictor` unless `--skip-tier1`) and/or Stage B (`train_t3_corrector`) |
| `python -m src.bin.main <group> <target> [-- …]` | Stable CLI map to training, datagen, eval, inspection (`MODULE_MAP` in `src/bin/main.py`) |

There is **no** separate `src.main` package for training; use **`src.bin.orchestrate`** or **`src.bin.main`** above.

### Data generation

- **`src.data_gen.pipeline_tier12`**, **`src.data_gen.pipeline_tier3`**: orchestrate mesh export, COMSOL or file I/O, and `mesh_to_graph` / `mesh_to_graph_tier3` builders under `src/data_gen/lib/`.

## Entry points

| Action | Command / module |
|--------|------------------|
| Chain training (recommended) | `python -m src.bin.orchestrate {a\|b\|all}` (`--skip-tier1` for Stage A starting at Tier 2) |
| Tier 1 only | `python -m src.training.train_t1_predictor` or `python -m src.bin.main train t1` |
| Tier 2 only | `python -m src.training.train_t2_predictor` or `python -m src.bin.main train t2` |
| Tier 3 only | `python -m src.training.train_t3_corrector` or `python -m src.bin.main train t3` |
| Tier 1 explorer | `python -m src.training.t1_explorer` or `python -m src.bin.main train explore` |
| Tier 1/2 datagen | `python -m src.data_gen.pipeline_tier12` |
| Tier 3 datagen | `python -m src.data_gen.pipeline_tier3` |

Checkpoints: `outputs/stage_a/` and `outputs/stage_b/` only (`resolve_checkpoint` returns the canonical path under those dirs).

**Path helpers**: `data_root()`, `outputs_root()`, `stage_a_dir()`, `stage_b_dir()`, `reports_dir()`, `comsol_models_dir()`, `resolve_checkpoint()`.

## Simulation boundary assumptions (current)

- `Re` is defined from the **inlet diameter** (`D_inlet`).
- `Re` is currently kept the same across simulations.
- Inlet boundary condition is **fully developed (FD) flow**, with average velocity `v_avg` computed from `Re`.
- Outlet flow boundary condition is **0 pressure**.
- Tier 3 simulations add species transport boundary conditions in addition to flow (for example, inlet concentrations and outlet flux constraints).

## Sample semantics (anchor vs non-anchor)

- **Anchor samples** are vessel geometries with full CFD labels from COMSOL across the graph (supervised fields like `u`, `v`, `p`, and related targets).
- **Non-anchor samples** are vessel geometries without COMSOL solution labels; they contribute geometry and analytical-prior/physics-based constraints only.
- **Optimal Mix (Tier 1)**: We found that a **50/50 split** of anchor and non-anchor (physics-only) nodes yields the most robust generalization and physical continuity (lowest `|∇·u|` and optimal total loss). This acts as a powerful regularizer, forcing the network to solve the PDE on blinded nodes rather than over-fitting to the labels.

## Source map

- **`src/core_physics/`** — PDE-consistent building blocks (fluid kinematics, rheology, biochem kernels, Anderson). Shared by training and tests.
- **`src/architecture/`** — `GINO_DEQ`, DEQ loop, LoRA, SIREN decoder, Tier 3 model variants.
- **`src/data_gen/`** — Top-level **pipelines** (`pipeline_tier12`, `pipeline_tier3`); **`lib/`** holds mesh/COMSOL/graph builders imported by those pipelines and re-exported from `src.data_gen`.
- **`src/training/`** — Predictor and corrector **scripts** (`train_*_predictor`, `train_t3_corrector`), `physics_curriculum.py`, `t1_explorer.py`; import LoRA helpers from `src.architecture.lora_injection` when needed.
- **`src/evaluation/`** — Benchmark drivers, tier comparison plots (may reference checkpoint paths).
- **`src/utils/`** — `paths`, `metrics`, `batching`, `rheology`, `units`, kinematics helpers (`kinematics_physics_terms`), training diary, inference helpers aligned with training.
- **`src/bin/`** — `main.py` (unified CLI router), `orchestrate.py` (stage runner).
- **`src/tools/`** — **Manual** inspection (matplotlib GUIs, optional COMSOL JDBC). Not imported by `src/tests/`.
- **`src/tests/`** — Pytest; keep tests **fast** and **deterministic** where possible.

## Data & artifacts (not in `src/`)

- **`data/`** — Canonical tree via `data_root()`: `raw/<tier>`, `processed/cfd_results_*`, `processed/graphs_*`, and **`data/benchmark/`** for temporary benchmark pipeline outputs (cleaned up by `run_benchmark` when finished).
- **`outputs/stage_a`**, **`outputs/stage_b`** — Preferred checkpoint roots (`stage_a_dir()`, `stage_b_dir()`).
- **`outputs/reports/`** — All generated reports (`reports_dir()`): CSVs, `figures/<tier>/`, `validation_<tier>/`, training diaries, Tier 3 metrics/debug logs, Tier 1 experiment JSON under `experiments/`.
- **`comsol_models/`** — Reference COMSOL projects (`comsol_models_dir()`); large binary assets.

## Interactive tools

Primary modules (also routed through `python -m src.bin.main inspect <target> -- ...`; see `src/bin/main.py` `MODULE_MAP`):

| Module | Purpose |
|--------|---------|
| `python -m src.tools.inspect_phase1_data` | **Tier 1 / 2 COMSOL anchors** (`vessel_*.npz`): **default** = full-directory health scan (flags + CSV) **then** interactive plot (random sample or `--sample-idx`; Regenerate / `r`). Also: `--summary` (compact text only), `--scan-only` (health/CSV, no GUI), `--skip-health-scan` (plot only), `--plot-static`, `--inspect-template-tags`. |
| `python -m src.tools.inspect_tier3_data` | **Tier 3** domain `.txt` + graphs: **default** = one patient at a time (brief availability line + qualitative text for that stem), one figure (domain time-slider or single-time 2×2; graph-only stems use graph views). **Regenerate Random Patient** / `r` like phase1. `--summary` = full table only. `--no-regenerate` fixes the current stem (`tier3` / `tier3_patients` / `tier3_mix`). |
| `python -m src.tools.inspect_comsol_model` | Live **COMSOL** model browser (`mph`): `--list-models`, `--model`, `--all-models`. |
| `python -m src.tools.inspect_graph_sample` | **Processed** `.pt` graphs (tier 1/2/stage A style): COMSOL overlap, WLS condition numbers, BC masks, widgets. |
| `python -m src.tools.verify_deq_convergence` | Manual Picard vs Anderson residual curves (not pytest). |

`bin.main` shortcuts: `inspect anchor` and `inspect phase1` → `inspect_phase1_data`, `inspect tier3`, `inspect comsol`, `inspect graph`, `inspect deq`.

## Conventions for edits

- **Mesh-agnostic**: never assume fixed node/edge counts or regular grids.
- **Naming**: prefer physical names (`u`, `v`, `p`, `phi`, `mu`, `gamma_dot`).
- **Tiers**: changing `tier*` in config affects both data pipeline and model heads; grep for `tier` when touching defaults.
- **Docs**: root `README.md` is the landing page; this file holds deeper orientation; `src/docs/README.md` indexes docs.

## Tests policy

- Physics and index/shape safety: unit tests in `src/tests/`.
- **COMSOL anchor strict tests** (`test_kinematics_physics_kernels.py`) compare graph labels to the same kinematics loss path as training. They optionally enforce a tight **`l_bc`** cap against wall BC masks. If your Tier 1/2 graph corpus was exported with wall labels that do not match the training mask convention, set `PHASE1_PHYSICS_CHECK_BC=0` so identity + shuffle checks still run without failing the whole suite on `l_bc`. Use **`inspect_phase1_data`** (raw anchors) and **`inspect_graph_sample`** (processed graphs) to debug masks before re-enabling BC asserts.
- **Regression / wiring**: `test_cli_routing_regression.py` (`src.bin.main`, `orchestrate`), `test_mesh_to_graph_regression.py`, `test_predictor_trainer_regression.py`, `test_predictor_trainer_split.py`, `test_vessel_generator_centerline.py` — keep in sync with training entry points and graph builders.
