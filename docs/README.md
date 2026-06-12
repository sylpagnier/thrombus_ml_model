# HemoGINO — documentation index

Start here, then open the linked files for depth.

## Primary references

1. **[PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)** — Authoritative overview:
   - **Tier vs Stage** terminology
   - **Architecture** (model, training scripts, routers, config channels)
   - Entry points (`bin.main`, `bin.orchestrate`, unified training scripts)
   - Pipelines (`pipeline_kinematics`, `pipeline_biochem`)
   - Data and checkpoint layout
   - Interactive inspection tools and pytest policy

2. **[KINEMATICS_TRAINING_HISTORY.md](KINEMATICS_TRAINING_HISTORY.md)** — Kinematics sweep history, mesh-resolution decision, V2/V3 strategy.

3. **[BIOCHEM_TRAINING_PROGRESS.md](BIOCHEM_TRAINING_PROGRESS.md)** — Biochem corrector training log: complexity ladder, μ diagnostics, env pitfalls, run table (update after experiments).

## Ladders and specialized tracks

| Doc | Topic |
|-----|-------|
| [T0_RUNG_LADDER.md](T0_RUNG_LADDER.md) | T0 isolation ladder (mu + clot physics) |
| [CLOT_TRIGGER_LADDER.md](CLOT_TRIGGER_LADDER.md) | Clot trigger / deploy-mask audit ladder |
| [CLOT_ML_DEPLOY_TRAINING_PLAN.md](CLOT_ML_DEPLOY_TRAINING_PLAN.md) | Clot ML V1 deploy training plan |
| [CLOT_ML_LADDER_V2.md](CLOT_ML_LADDER_V2.md) | Clot ML V2 (band-GNN growth + nucleation mask) |
| [CLOT_FORECAST_LADDER.md](CLOT_FORECAST_LADDER.md) | Clot forecast R0–R6 |
| [DEPLOY_ARCHITECTURE.md](DEPLOY_ARCHITECTURE.md) | Deploy clot ladder (Track A/B) |
| [GNODE_ODE_LADDER.md](GNODE_ODE_LADDER.md) | GNODE-ODE component ladder |
| [CLOT_PHI_BASELINE.md](CLOT_PHI_BASELINE.md) | Simple clot-phi wall-local probe |
| [CLOT_PHI_ROLLOUT.md](CLOT_PHI_ROLLOUT.md) | Clot-phi rollout (6a/6b) |
| [COMSOL_MU_RHEOLOGY_CHECKLIST.md](COMSOL_MU_RHEOLOGY_CHECKLIST.md) | COMSOL mu/rheology alignment checklist |
| [SPECIES_TEMPORAL_ML.md](SPECIES_TEMPORAL_ML.md) | Wall-band graph reaction rollout |
| [KINEMATICS_BEST_ARCHITECTURE.md](KINEMATICS_BEST_ARCHITECTURE.md) | Stage-A kinematics architecture record |
| [PASSIVE_KIN_BLOCKER_CHECKLIST.md](PASSIVE_KIN_BLOCKER_CHECKLIST.md) | Passive transport / kin blocker gates |

## Common commands

```text
# Orchestrated training (same order as production): kinematics then biochem
python -m src.bin.orchestrate all
python -m src.bin.orchestrate biochem

# Thin CLI router (see src/bin/main.py MODULE_MAP)
python -m src.bin.main train kinematics
python -m src.bin.main train t3
python -m src.bin.main inspect kinematics -- --summary

# Datagen
python -m src.data_gen.pipeline_kinematics
python -m src.data_gen.pipeline_biochem
```

## Source layout (short)

| Path | Role |
|------|------|
| `src/architecture/` | `GINO_DEQ`, DEQ solver hooks, LoRA, SIREN decoder |
| `src/core_physics/` | Physics kernels, Anderson acceleration, PDE-consistent terms |
| `src/config.py` | `PhysicsConfig`, `VesselConfig`, channel enums (`PredChannels`, `NodeFeat`) |
| `src/data_gen/` | Kinematics / Biochem pipelines and mesh→graph builders |
| `src/training/` | `train_kinematics_predictor`, `train_biochem_corrector`, `physics_curriculum` |
| `src/bin/` | `main` (router), `orchestrate` (phase runner) |

Training is implemented as **explicit scripts** (not a shared trainer class). Invoke unified kinematics training via `python -m src.training.train_kinematics_predictor` or `python -m src.bin.main train kinematics`.
