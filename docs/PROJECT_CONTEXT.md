# HemoGINO — project context (agents & contributors)

This document gives **enough structure** to navigate the repo without re-reading the whole tree: terminology, entry points, artifacts, and where to change behavior safely.

## Goal

**HemoGINO** is a **mesh-agnostic** surrogate that predicts **velocity, pressure, viscosity, and clot-related fields** on vessel graphs faster than COMSOL, with physics-informed losses and a DEQ-style core. Patient-specific adaptation uses **LoRA** where configured.

## Terminology: Tier vs Stage

- **Tier** (`tier1` | `tier2` | `tier3` in `PhysicsConfig` / `VesselConfig`): **which physics terms and data modes are enabled** (Newtonian → non-Newtonian → coupled biochem).
- **Stage** (training pipeline): **Stage A** = predictor warm-up (Tier 1 + Tier 2 scripts); **Stage B** = Tier 3 corrector script. This is only about **script naming and checkpoint buckets**, not a different physical meaning of “tier.”

## Entry points

| Action | Command / module |
|--------|------------------|
| Chain training | `python -m src.main {a\|b\|all}` (`--skip-tier1` for Stage A starting at Tier 2) |
| Tier 1 only | `python -m src.training.train_t1_predictor` |
| Tier 2 only | `python -m src.training.train_t2_predictor` |
| Tier 3 only | `python -m src.training.train_t3_corrector` |

Checkpoints: `outputs/stage_a/` and `outputs/stage_b/` only (`resolve_checkpoint` returns the canonical path under those dirs).

**Path helpers**: `data_root()`, `outputs_root()`, `stage_a_dir()`, `stage_b_dir()`, `reports_dir()`, `comsol_models_dir()`, `resolve_checkpoint()`.

## Source map

- **`src/core_physics/`** — PDE-consistent building blocks (fluid kinematics, rheology, biochem kernels). Shared by training and tests.
- **`src/architecture/`** — Model classes, DEQ loop, LoRA hooks.
- **`src/data_pipeline/`** — Gmsh/mesh → PyG `Data`, WLS helpers, Tier 3-specific converters where split.
- **`src/training/`** — `physics_curriculum.py` and the three `train_*` scripts; import LoRA helpers from `src.architecture.lora_injection` when needed.
- **`src/evaluation/`** — Benchmark drivers, tier comparison plots (may reference checkpoint paths).
- **`src/utils/`** — `paths`, `metrics`, `inference`, kinematics loss helpers aligned with training.
- **`src/tools/`** — **Manual** inspection (matplotlib GUIs). Not imported by `src/tests/`.
- **`src/tests/`** — Pytest; keep tests **fast** and **deterministic** where possible.

## Data & artifacts (not in `src/`)

- **`data/`** — Canonical tree via `data_root()`: `raw/<tier>`, `processed/cfd_results_*`, `processed/graphs_*`, and **`data/benchmark/`** for temporary benchmark pipeline outputs (cleaned up by `run_benchmark` when finished).
- **`outputs/stage_a`**, **`outputs/stage_b`** — Preferred checkpoint roots (`stage_a_dir()`, `stage_b_dir()`).
- **`outputs/reports/`** — All generated reports (`reports_dir()`): CSVs, `figures/<tier>/`, `validation_<tier>/`, training diaries, Tier 3 metrics/debug logs.
- **`comsol_models/`** — Reference COMSOL projects (`comsol_models_dir()`); large binary assets.

## Interactive tools

| Tool | Purpose |
|------|---------|
| `python -m src.tools.inspect_anchor_cfd` | Scan/plot **raw** `vessel_*.npz` COMSOL anchors; `--scan-only` + CSV health report. |
| `python -m src.tools.inspect_graph_sample` | Inspect **processed** `.pt` graphs, COMSOL overlap, WLS condition numbers, BC masks. |
| `python -m src.tools.verify_deq_convergence` | Manual Picard vs Anderson residual curves (not pytest). |

## Conventions for edits

- **Mesh-agnostic**: never assume fixed node/edge counts or regular grids.
- **Naming**: prefer physical names (`u`, `v`, `p`, `phi`, `mu`, `gamma_dot`).
- **Tiers**: changing `tier*` in config affects both data pipeline and model heads; grep for `tier` when touching defaults.
- **Docs**: README stays short; this file holds deeper orientation; `.cursorrules` mirrors tier/stage paths for AI assistants.

## Tests policy

- Physics and index/shape safety: unit tests in `src/tests/`.
- **COMSOL anchor strict tests** (`test_kinematics_physics_kernels.py`) compare graph labels to the same kinematics loss path as training. They optionally enforce a tight **`l_bc`** cap against wall BC masks. If your Tier 1/2 graph corpus was exported with wall labels that do not match the training mask convention, set `PHASE1_PHYSICS_CHECK_BC=0` so identity + shuffle checks still run without failing the whole suite on `l_bc`. Use `src.tools.inspect_anchor_cfd` / `inspect_graph_sample` to debug masks before re-enabling BC asserts.
