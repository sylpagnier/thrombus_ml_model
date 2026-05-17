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
