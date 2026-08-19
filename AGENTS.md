# Agent notes (HemoRGP)

Short cheat sheet for agents and contributors. Full orientation: [docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md). Publishing policy: [docs/PUBLISHING.md](docs/PUBLISHING.md).

## Canonical stacks

| Stack | Train | Docs |
|-------|-------|------|
| **RGP-DEQ** (`rgp_deq_kine`) | `python -m src.bin.main train rgp-deq-kine` or `scripts/go_kinematics_*.ps1` | [docs/KINEMATICS_BEST_ARCHITECTURE.md](docs/KINEMATICS_BEST_ARCHITECTURE.md) |
| **biochem_gnn** | `scripts/go_biochem_gnn.ps1` or `python -m src.bin.main train biochem-gnn` | [docs/BIOCHEM_GNN.md](docs/BIOCHEM_GNN.md) |
| Mat-growth (research) | `go_mat_*.ps1`, `go_wc_v7_*.ps1` | [docs/MAT_GROWTH.md](docs/MAT_GROWTH.md) |
| Local corrector | `python -m src.training.train_local_kinematic_corrector` | [docs/LOCAL_KINEMATIC_CORRECTOR.md](docs/LOCAL_KINEMATIC_CORRECTOR.md) |

- **Promote biochem:** `python scripts/promote_biochem_gnn.py` → `outputs/biochem/biochem_gnn/locked/` + `data/reference/biochem_gnn_baseline.json`
- **Locked wall mat:** `WC_v7_clot_phi_mse` (2026-07-19); cohort clot F1 **~0.767**, clot score **~0.791** — still the wall-only / compound backbone
- **Wall-gen / phase1 baseline:** `FS_ab_coupled` — deploy-faithful train (RGP-DEQ @ t=0 + local tiling), drop-xy; promote via `python scripts/promote_wall_gen_baseline.py`; manifest [data/reference/mat_wall_gen_baseline.json](data/reference/mat_wall_gen_baseline.json); ckpt `outputs/biochem/biochem_gnn/wall_gen_baseline/species/best.pth`
- **Locked compound deploy:** `WC_v8_compound_front_h1` (2026-07-27); wall + `compound_growth_best.pth`, frontier hops=1 — see [data/reference/mat_compound_deploy.json](data/reference/mat_compound_deploy.json)
- **Promote compound:** `python scripts/promote_compound_deploy.py`
- **Precision Frontier-ge2 (~8 h):** `go_wc_v7_frontier_ge2_prec_8h.ps1` (viz: `go_wc_v7_frontier_ge2_prec_viz.ps1`) — see [docs/MAT_GROWTH.md](docs/MAT_GROWTH.md)
- **Customer UI:** `scripts/go_customer_predict.ps1`
- **Import:** `from src.biochem_gnn import BiochemGNN` (alias package `src.biochem_deploy`)

## Kinematics (Stage A)

- Production allfix: `go_kinematics_production_allfix.ps1` — Rel L2 **~0.087** after continuity finetune
- Manifest: [data/reference/kinematics_best_20260426T184600Z.json](data/reference/kinematics_best_20260426T184600Z.json)
- Config helpers: `snapshot_rgp_deq_model_config` / `resolve_rgp_deq_ctor_kwargs` in `src/architecture/kinematics_model_config.py` (gino/pmgp aliases retained)
- **Solver Orchestration**: Dynamic flow patching is handled via a hybrid macro/micro architecture. See [docs/KINE_ADJUSTMENTS.md](docs/KINE_ADJUSTMENTS.md) for macro-resolves, micro-resolves, and smooth SDF fast-marching.
- **GT Leakage Policy**: `prior_mode="analytic"` MUST be used for priors. See [docs/KINE_ADJUSTMENTS.md](docs/KINE_ADJUSTMENTS.md).


## Configuration Architecture Guardrail

**GT Input Leakage Policy**: Never feed COMSOL ground-truth velocity, viscosity, or WSS into model INPUT features (data.x channels 11-14). Prior channels must always use analytical Poiseuille/Carreau formulas computed from mesh geometry. GT data may only appear in LABELS (data.y) and loss targets.

**Strict Policy**: Never mutate `os.environ` to alter model architectures, toggle features,
deploy/rollout policy, scoring, coupling, or sweeps.

Prefer **system-wide typed dataclasses** over temporary env strings:

| Concern | Dataclass | Leg field / helper |
|---------|-----------|--------------------|
| Architecture / loss / features | `PushforwardConfig` | `MatGrowthLegSpec.config_kwargs` / `get_mat_growth_config_kwargs` |
| Coupling, rollout, scoring, gelation, off-wall | `BiochemRuntimeConfig` (+ nested) | `MatGrowthLegSpec.runtime_kwargs` / `get_mat_growth_runtime_kwargs` |

Patterns:
- Define knobs as typed fields; override with `dataclasses.replace(cfg, **kwargs)` or `BiochemRuntimeConfig.with_overrides(**kwargs)`.
- Bind with `use_pushforward_config(cfg)` / `use_biochem_runtime(rt)` so helpers resolve without globals.
- Persist `config_kwargs` + `runtime_kwargs` in checkpoint `meta`; reload via `from_meta`.
- `env_overrides` is **deprecated residual only** — do not add new knobs there.
- Process/IO only (ckpt paths, tqdm, CUDA) may stay CLI/env.

**Agent bias**: favor robust, concise, **system-wide** patches (typed config + call-site wiring)
over temporary env toggles, one-off scripts, or per-leg string dictionaries.
Always-apply rule: [`.cursor/rules/robust-system-wide-changes.mdc`](.cursor/rules/robust-system-wide-changes.mdc).

## Scripts

- Active only: [scripts/README.md](scripts/README.md)
- Retired: `scripts/archive/` — do not revive GNODE / clot-ML / T0 trainers without restoring modules from git ([docs/BIOCHEM_LEGACY_LESSONS.md](docs/BIOCHEM_LEGACY_LESSONS.md))

## Debugging visualizations — "viz" means this

**If a human says "viz", "visualize", or asks to look at a vessel/model, read
[docs/VIZ_STANDARD.md](docs/VIZ_STANDARD.md) and extend that template — do not build a
new one from scratch.** Standard shape: two synced windows (Model | Ground truth), a
time slider scrubbing real simulated seconds, synced zoom/pan, wall=circle/lumen=square
with a depth-into-lumen color gradient, and the canonical deploy score shown live,
domain-restricted (wall vs off-wall) and over time — not just a final-mask number.
Reference build: `scripts/gen_offwall_temporal_data.py` + `scripts/build_offwall_temporal_artifact.py`.

## Console (PowerShell)

No emoji in `print` / launcher banners. Use ASCII tags (`[OK]`, `[WARN]`, `[i]`). See [.cursor/rules/powershell-console-ascii.mdc](.cursor/rules/powershell-console-ascii.mdc).
- **Process Management Guardrail**: When killing or canceling a background PowerShell task that is running Python processes (especially heavy GPU workloads like training scripts), `Stop-Process` on the `pwsh` task may silently orphan the child `python.exe` processes. You **must** manually verify and explicitly kill these zombie PyTorch processes (`Stop-Process -Name python -Force` or by PID) to prevent invisible GPU starvation and extreme slowdowns in subsequent runs.

## Generalization / wall-gen eval policy (mandatory)

**Deploy-faithful only — no GT velocity leak.** New vessels give geometry (and whatever we predict), not COMSOL `[u,v]`. Quote only cold-deploy metrics from `eval_mat_growth_simple.py` / `canonical_deploy_clot_metrics`:

1. **t=0 base flow:** RGP-DEQ once (`u0_pred` / kinematics checkpoint). Never re-run the heavy global solver during rollout.
2. **t>0 flow adjustments:** local kinematic corrector + tiling only (`corrector_coupling=1`, closed-loop when the leg uses it).
3. **Flow features at eval:** `flow_feats_source=auto` (predicted / coupled). Do **not** score generalization with `flow_feats_source=gt` or `train_vel_source=gt`.
4. **Model UV inputs:** never pass raw COMSOL `data.y[..., 0:2]` into `model.velocity`, physics-GAT, or convective upwind. Use `band_uv_for_model` / `resolve_species_rollout_uv` (coupled / `u0_pred` / RGP-DEQ). GT may appear in **labels / timeline length / teacher-forced diagnostics** only — never as the deploy flow channel. Ignore in-training `val_*` that are GT-teacher-forced when claiming generalization.

**Holdout for wall-gen / phase1 / flow-source legs:** single clot-rich vessel **`patient020`** only. Do not average in clot-free `patient034` (or other empty-GT tubes) for the primary gate — that dilutes the score and confuses FP noise with generalization. Broader multi-vessel tables are optional secondary diagnostics, not the decision metric.

Canonical small cohort (unless a leg explicitly overrides):

| Split | Anchors |
|-------|---------|
| Train | `patient005,patient006,patient010,patient023,patient002` |
| Val (selection, held out of train) | `patient020` |
| Holdout cold eval | **`patient020`** |

Launchers: `go_flow_source_ab.ps1`, `go_phase1_sweep_v3.ps1`, `go_baseline_validation.ps1`.

## Clot-map scoring: use the strict protocol, and respect the noise floor

**[docs/PHASE10_V4.md](docs/PHASE10_V4.md) supersedes the readout half of
[docs/PHASE9_ML.md](docs/PHASE9_ML.md).** Two rules:

1. **Score with `scripts/eval_strict.py` (final time) / `scripts/eval_strict_temporal.py`
   (mean-over-time + final).** They select every readout scalar on the *out-of-fold* scores
   of vessels outside the held-out fold. `train_time_conditioned.py`'s hard-coded
   `score >= 0.73 / 0.92` cuts are a whole-pool leak; PHASE9's numbers are ~0.02 optimistic.
   Always quote **both** mean-over-time and the last time point — they disagree.
2. **The cohort noise floor is ±0.024 wall and ±0.091 off-wall** (`scripts/eval_significance.py`;
   three configs of the *same* arm spread that much). **A 0.01–0.03 cohort-mean difference on
   these 19 vessels is not a result** — quote a paired bootstrap CI or call it undetermined.
   This applies retroactively to several PHASE9 claims.

Corollary: prefer levers that do not need a per-config effect to be detectable — ensembling,
and deterministic readout fixes. Adding selection layers (inner-CV family choice, per-domain
family choice, rule selection) was measured and all of it **loses** at this n.

**Active findings: [docs/PHASE7_FINDINGS.md](docs/PHASE7_FINDINGS.md)** — the off-wall
problem, `.mph` reading, and the corrections it makes to Phase 6. Read it first.
Earlier scope: [docs/WALL_MODEL_PLAN.md](docs/WALL_MODEL_PLAN.md) (wall model only, wall
clots only, target `deploy_clot_f1` > 0.5), ordered next steps, and the list of parked arms.
Historical context: [docs/GENERALIZATION_PLAN.md](docs/GENERALIZATION_PLAN.md).

## COMSOL ground truth — read the `.mph`, do not re-derive from exports

`.mph` files are **zip archives**; `smodel.json` inside is the full model tree. Trust the
**physics node tree** over the parameter list (stale parameters for deleted mechanisms
survive). `phase2_nowound_001/002/003.mph` are an experimental branch and are **not** the
production physics — see [docs/PHASE7_FINDINGS.md](docs/PHASE7_FINDINGS.md) §0–1.

## Eval / off-wall

- Persist `leg`, `config_kwargs`, `runtime_kwargs` (and residual `env_overrides` only if needed) in checkpoint `meta`
- On eval / load: `PushforwardConfig.from_meta(meta)` + `BiochemRuntimeConfig.from_meta(meta)` — do not inject into `os.environ`
- Off-wall: hop >= 1 helpers (`deploy_clot_offwall_relaxed_f1`, …)
- **Wall-cohort physics** (`predict_wall_clot`, Phase 3–8 scalars): FIT / DEV / SEALED from [`src/core_physics/wall_cohort_splits.py`](src/core_physics/wall_cohort_splits.py), same lists as `scripts/sweep_ml_clean_protocol.py`. Fit on **FIT**, select on **DEV** (039/040/041/044). `patient020` is FIT, not a holdout. **SEALED** (`WALL_COHORT_V2_GENERALIZATION`) is spent once — do not tune against it. Do not quote a TRAIN-mean (27, or the 19 eligible full-horizon subset) as a decision metric; it mixes FIT with DEV. Comparison table: `python scripts/eval_wall_protocol.py`.

## Hardware

Training entry points should call `require_cuda_device()` and fail loud without CUDA.

## Historical training chronicles

Biochem corrector / GNODE run logs live under [docs/archive/BIOCHEM_TRAINING_PROGRESS.md](docs/archive/BIOCHEM_TRAINING_PROGRESS.md) (archive only).

## Data Architecture / Graphs

- **Graph Dimensionality**: The simulations (via COMSOL) generate strictly **2D meshes**. Do not assume 3D spatial environments.
- **Node Features**: The `biochem_gnn` PyTorch `Data` objects do **not** use a standard `data.pos` tensor. Spatial coordinates are embedded directly in the node feature matrix `data.x` (specifically the first two columns: `x_nd` and `y_nd`). There is no Z coordinate.
- **The meshes are QUADRATIC, and ~3/4 of every pack's nodes are mid-edge nodes** (0.742–0.746 on all 19 wall-cohort vessels; 49.6% of *wall* nodes). `M/Mas/Mat` are not populated on the wall-normal ones, so GT `Mat` is zero at 44.6% of mid-side wall nodes against 17.6% of corner nodes. Any per-node metric over "all wall nodes" is therefore half-scoring a structural zero block — a rank correlation reads 0.534 that way and 0.193 on species-carrying nodes. Use `midside_nodes` / `first_corner_shell` in `src/core_physics/physics_lumen_model.py`; see [docs/PHASE7_FINDINGS.md](docs/PHASE7_FINDINGS.md) §8.
- **Baselines**: The `drop-xy` baseline zeroes out these first two columns of `data.x` to strip global positioning and force the model to learn from graph connectivity and flow features.
