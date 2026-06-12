# HemoGINO — project context (agents & contributors)

This document gives **enough structure** to navigate the repo without re-reading the whole tree: terminology, **architecture**, entry points, artifacts, and where to change behavior safely.

## Goal

**HemoGINO** is a **mesh-agnostic** surrogate that predicts **velocity, pressure, viscosity, and clot-related fields** on vessel graphs faster than COMSOL, with physics-informed losses and a DEQ-style core.

Domain adaptation from synthetic to real patient geometries uses **LoRA**. During Biochem phase, LoRA bridges the sim-to-real gap across a diverse population of meshes, enabling zero-shot inference on new, unseen patient scans without retraining.

## Terminology: Tier vs Stage

- **Tier** (`kinematics` | `kinematics` | `biochem` / `biochem_anchors` / `biochem_patients` (legacy alias) / `biochem_mix` in `PhysicsConfig` / `VesselConfig`): **which physics terms and data modes are enabled** (Newtonian → non-Newtonian → coupled biochem + transient graphs as configured).
- **Stage** (training pipeline): **Kine phase** = unified kinematics pretraining (curriculum spans kinematics behavior to kinematics target behavior); **Biochem phase** = biochem corrector. This is about **checkpoint buckets and orchestration order**, not a different physical meaning of “phase.”

## Architecture (current)

### Model and physics

- **`GINO_DEQ`** (`src/architecture/ginodeq.py`): graph encoder → fixed-point **DEQ** loop (`core_physics.anderson`) over latent states with a required **global message/mixing** path (`AttentionGlobalMixingBlock`) for long-range pressure-like coupling and a required **SIREN** spatial decoder (`architecture/siren_decoder.py`).
- **LoRA** adapters: `src/architecture/lora_injection.py` (spectral / low-rank hooks used where config enables LoRA for sim-to-real).
- **`PhysicsKernels`** (`src/core_physics/physics_kernels.py`): residual / BC / rheology interfaces shared by training and tests.
- **Kinematics losses** align with `src/utils/kinematics_physics_terms.py` (not legacy standalone `kinematics` wrappers).

### Configuration and tensor conventions

- **`PhysicsConfig`**, **`VesselConfig`**, **`BiochemConfig`**, **`CurriculumConfig`** — see `src/config.py`.
- **Predictor channels** — `PredChannels` / `STATE_CHANNEL_MU_EFF_ND`: canonical indices for `[u, v, p, μ_eff_nd, …]`.
- **Node features (Kinematics/2)** — `NodeFeat` slices: positions, SDF, wall normals, priors, optional hydraulic width `D(x)` and derivatives for geometric priors.
- **Biochem node features** — `BiochemNodeFeat` for clot/chemistry graphs.

### Clot-phi simple baseline (wall-local probe)

Separate from the full GNODE biochem stack: a **tiny hybrid head** on GT kinematics + capped `mu_eff` with a dgamma-sliced neighbor mask. Use it to validate masks/features before multitask biochem. Record and metrics: [CLOT_PHI_BASELINE.md](CLOT_PHI_BASELINE.md). Entry: `scripts/go_clot_phi_simple.ps1`.

### Biochem "Clot" Semantics

In this repository, a "clot" is not a discrete model class. It is an interpretive
label for a spatial region where the continuous effective-viscosity field
`mu_eff_si` has been triggered upward by the coupled platelet/fibrin rheology.
Training therefore treats `mu_eff_si` as a regression target on COMSOL anchor
nodes. Thresholded views such as `mu_eff_si > 20 * mu_inf` may be logged as
`HighMuDice@thr` diagnostics for quick sanity checks, but they should not be
used as classifier metrics (for example AUPRC) or primary checkpoint objectives.

### `mu_ratio_max` vs `mu_eff` (do not conflate)

`BiochemConfig.mu_ratio_max` (default **80**) and `BIOCHEM_TEACHER_MU_RATIO_MAX` cap the
**dimensionless COMSOL step outputs** μ₁(Mat) ∈ [0, max−1] and μ₂(FI) ∈ [0, max] used in the
gelation multiplier `(1 + μ₁ + μ₂ + …)` on top of Carreau — they are **step ceilings**, not
**clot μ / bulk μ**.

| Quantity | Meaning | Typical clot-scale value |
|----------|---------|---------------------------|
| `mu_ratio_max` | Max μ₁/μ₂ step height in the surrogate forward | **80** (headroom) |
| `mu_eff_si` (GT channel 3) | COMSOL export `mu_effective` [Pa·s] | bulk ~**0.04**, patches ~**0.10** (~**2.5×**) |

Training objectives and checkpoints should use **`mu_log_mae` / `mu_eff_si`**, not “μ₂ → 80” or
`viz_final_mu2_mean` alone. Open-loop μ₂ can saturate while rollout `mu_eff` stays near bulk.

Preferred validation quantities for Biochem high-viscosity behavior are:

- `mu_MAE_si` / `mu_RMSE_si`: absolute effective-viscosity error in physical SI units.
- `mu_log_MAE`: scale-aware error for the heavy-tailed viscosity field.
- `mu_Pearson` and `mu_R2`: spatial pattern and explained-variance diagnostics.
- `HighMuDice@thr`: optional threshold-overlap diagnostic only.

### Training runtime (no monolithic “trainer” class)

Training is driven by **self-contained scripts** that own dataloading loops, schedulers, and checkpointing:

| Script | Role |
|--------|------|
| `src.training.train_kinematics_predictor` | Unified Kine phase curriculum trainer; Stage 1 Newtonian anchor -> Stage 2 physics ramp -> Stage 3 Kinematics target |
| `src.training.train_biochem_corrector` | Biochem phase: Biochem coupled corrector and graph semantics |
| `src.training.physics_curriculum` | Curriculum helpers consumed by training scripts |

`python -m src.bin.main train kinematics` runs the same module as `train_kinematics_predictor` (see `MODULE_MAP` in `src/bin/main.py`).

### Routers

| Module | Role |
|--------|------|
| `python -m src.bin.orchestrate {a\|b\|all}` | Runs Kine phase (`train_kinematics_predictor`) and/or Biochem phase (`train_biochem_corrector`) |
| `python -m src.bin.main <group> <target> [-- …]` | Stable CLI map to training, datagen, eval, inspection (`MODULE_MAP` in `src/bin/main.py`) |

There is **no** separate `src.main` package for training; use **`src.bin.orchestrate`** or **`src.bin.main`** above.

### Data generation

- **`src.data_gen.pipeline_kinematics`**, **`src.data_gen.pipeline_biochem`**: orchestrate mesh export, COMSOL or file I/O, and `mesh_to_graph` / `mesh_to_graph_biochem` builders under `src/data_gen/lib/`.

## Entry points

| Action | Command / module |
|--------|------------------|
| Chain training (recommended) | `python -m src.bin.orchestrate {a\|b\|all}` |
| Unified Kine phase only | `python -m src.training.train_kinematics_predictor` or `python -m src.bin.main train kinematics` |
| Biochem only | `python -m src.training.train_biochem_corrector` or `python -m src.bin.main train t3` |
| Kine phase aliases | `python -m src.bin.main train t1` / `train t2` / `train explore` / `train kinematics` -> same unified Kine phase trainer |
| Kinematics/2 datagen | `python -m src.data_gen.pipeline_kinematics` |
| Biochem datagen | `python -m src.data_gen.pipeline_biochem` |

Checkpoints: `outputs/kinematics/` and `outputs/biochem/` (`resolve_checkpoint` keeps backward-compatible reads from legacy `stage_a` / `stage_b` runs). Kinematics `.pth` files embed **`model_config`** (see `src/architecture/kinematics_model_config.py`); canonical best-run manifest: `data/reference/kinematics_best_20260426T184600Z.json` and [KINEMATICS_BEST_ARCHITECTURE.md](KINEMATICS_BEST_ARCHITECTURE.md).

**Path helpers**: `data_root()`, `outputs_root()`, `kinematics_dir()`, `biochem_dir()`, `stage_a_dir()`, `stage_b_dir()`, `reports_dir()`, `comsol_models_dir()`, `resolve_checkpoint()`.

## Simulation boundary assumptions (current)

- `Re` is defined from the **inlet diameter** (`D_inlet`).
- `Re` is currently kept the same across simulations.
- Inlet boundary condition is **fully developed (FD) flow**, with average velocity `v_avg` computed from `Re`.
- Outlet flow boundary condition is **0 pressure**.
- Biochem simulations add species transport boundary conditions in addition to flow (for example, inlet concentrations and outlet flux constraints).

## Sample semantics (anchor vs non-anchor)

- **Anchor samples** are vessel geometries with full CFD labels from COMSOL across the graph (supervised fields like `u`, `v`, `p`, and related targets).
- **Non-anchor samples** are vessel geometries without COMSOL solution labels; they contribute geometry and analytical-prior/physics-based constraints only.
- **Optimal Mix (Kinematics)**: We found that a **50/50 split** of anchor and non-anchor (physics-only) nodes yields the most robust generalization and physical continuity (lowest `|∇·u|` and optimal total loss). This acts as a powerful regularizer, forcing the network to solve the PDE on blinded nodes rather than over-fitting to the labels.

## Source map

- **`src/core_physics/`** — PDE-consistent building blocks (fluid kinematics, rheology, biochem kernels, Anderson). Shared by training and tests.
- **`src/architecture/`** — `GINO_DEQ`, DEQ loop, LoRA, SIREN decoder, Biochem model variants.
- **`src/data_gen/`** — Top-level **pipelines** (`pipeline_kinematics`, `pipeline_biochem`); **`lib/`** holds mesh/COMSOL/graph builders imported by those pipelines and re-exported from `src.data_gen`.
- **`src/training/`** — Predictor and corrector **scripts** (`train_kinematics_predictor`, `train_biochem_corrector`), `physics_curriculum.py`; import LoRA helpers from `src.architecture.lora_injection` when needed.
- **`src/evaluation/`** — Benchmark drivers, phase comparison plots (may reference checkpoint paths).
- **`src/utils/`** — `paths`, `metrics`, `batching`, `rheology`, `units`, kinematics helpers (`kinematics_physics_terms`), training diary, inference helpers aligned with training.
- **`src/bin/`** — `main.py` (unified CLI router), `orchestrate.py` (phase runner).
- **`src/tools/`** — **Manual** inspection (matplotlib GUIs, optional COMSOL JDBC). Not imported by `src/tests/`.
- **`src/tests/`** — Pytest; keep tests **fast** and **deterministic** where possible.

## Data & artifacts (not in `src/`)

- **`data/`** — Canonical tree via `data_root()`: `raw/<phase>`, `processed/cfd_results_*`, `processed/graphs_*`, and **`data/benchmark/`** for temporary benchmark pipeline outputs (cleaned up by `run_benchmark` when finished).
- **`outputs/kinematics`**, **`outputs/biochem`** — Preferred checkpoint roots (`kinematics_dir()`, `biochem_dir()`). Legacy `stage_a` / `stage_b` reads remain supported.
- **`outputs/reports/`** — All generated reports (`reports_dir()`): CSVs, `figures/<phase>/`, `validation_<phase>/`, training diaries, biochem metrics/debug logs, kinematics experiment JSON under `experiments/`.
- **`comsol_models/`** — Reference COMSOL projects (`comsol_models_dir()`); large binary assets.

## Interactive tools

Primary modules (also routed through `python -m src.bin.main inspect <target> -- ...`; see `src/bin/main.py` `MODULE_MAP`):

| Module | Purpose |
|--------|---------|
| `python -m src.tools.inspect_kinematics_data` | **Kinematics / 2 COMSOL anchors** (`vessel_*.npz`): **default** = full-directory health scan (flags + CSV) **then** interactive plot (random sample or `--sample-idx`; Regenerate / `r`). Also: `--summary` (compact text only), `--scan-only` (health/CSV, no GUI), `--skip-health-scan` (plot only), `--plot-static`, `--inspect-template-tags`. |
| `python -m src.tools.extract_biochem_comsol` | **Biochem COMSOL extract** (interactive): status table; `--from-comsol` pulls solved `.mph` via `mph` into `cfd_results_biochem/` (no manual Results export), then builds `graphs_biochem_anchors/*.pt`. Save `<stem>.mph` beside the mesh in `data/raw/biochem_anchors/`. PyCharm: run this module. CLI: `python -m src.bin.main data extract-biochem -- --from-comsol`. |
| `python -m src.tools.inspect_biochem_data` | **Biochem** domain `.txt` + graphs: **default** = one anchor stem at a time (brief availability line + qualitative text for that stem), one figure (domain time-slider or single-time 2×2; graph-only stems use graph views). **Regenerate Random Anchor** / `r` like kinematics. `--summary` = full table only. `--no-regenerate` fixes the current stem (`biochem` / `biochem_anchors` / `biochem_mix`; legacy `biochem_patients` still accepted). |
| `python -m src.tools.inspect_comsol_model` | Live **COMSOL** model browser (`mph`): `--list-models`, `--model`, `--all-models`. |
| `python -m src.tools.inspect_graph_sample` | **Processed** `.pt` graphs (kinematics/stage A style): COMSOL overlap, WLS condition numbers, BC masks, widgets. |
| `python -m src.tools.demo_kinematics_flow` | **Synthetic kinematics flow demo**: parametric sliders + optional **Edit walls** drag on top/bot station polylines (pinned inlet/outlet); Gmsh mesh -> graph -> GINO-DEQ u/v/p (``inspect flow`` via `bin.main`). |
| `python -m src.tools.verify_deq_convergence` | Manual Picard vs Anderson residual curves (not pytest). |

`bin.main` shortcuts: `data extract-biochem` → `extract_biochem_comsol`; `inspect anchor` and `inspect kinematics` → `inspect_kinematics_data`, `inspect biochem`, `inspect comsol`, `inspect graph`, `inspect flow`, `inspect deq`.

## Conventions for edits

- **Mesh-agnostic**: never assume fixed node/edge counts or regular grids.
- **Naming**: prefer physical names (`u`, `v`, `p`, `phi`, `mu`, `gamma_dot`).
- **Phases**: changing `phase` in config affects both data pipeline and model heads; grep for `phase` when touching defaults.
- **Docs**: root `README.md` is the landing page; this file holds deeper orientation; `docs/README.md` indexes docs.

## Tests policy

- Physics and index/shape safety: unit tests in `src/tests/`.
- **COMSOL anchor strict tests** (`test_kinematics_physics_kernels.py`) compare graph labels to the same kinematics loss path as training. They optionally enforce a tight **`l_bc`** cap against wall BC masks. If your Kinematics/2 graph corpus was exported with wall labels that do not match the training mask convention, set `KINEMATICS_PHYSICS_CHECK_BC=0` so identity + shuffle checks still run without failing the whole suite on `l_bc`. Use **`inspect_kinematics_data`** (raw anchors) and **`inspect_graph_sample`** (processed graphs) to debug masks before re-enabling BC asserts.
- **Regression / wiring**: `test_cli_routing_regression.py` (`src.bin.main`, `orchestrate`), `test_mesh_to_graph_regression.py`, `test_predictor_trainer_regression.py`, `test_predictor_trainer_split.py`, `test_vessel_generator_centerline.py` — keep in sync with training entry points and graph builders.
