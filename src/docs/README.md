# HemoGINO — documentation index

Start here, then open the linked files for depth.

## Primary references

1. **[PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)** — Authoritative overview:
   - **Tier vs Stage** terminology
   - **Architecture** (model, training scripts, routers, config channels)
   - Entry points (`bin.main`, `bin.orchestrate`, `train_*_predictor`)
   - Pipelines (`pipeline_tier12`, `pipeline_tier3`)
   - Data and checkpoint layout
   - Interactive inspection tools and pytest policy

2. **[TIER1_TRAINING_HISTORY.md](TIER1_TRAINING_HISTORY.md)** — Tier 1 sweep history, mesh-resolution decision, V2/V3 strategy (complements the code paths in `train_t1_predictor` and `t1_explorer`).

## Common commands

```text
# Orchestrated training (same order as production): Stage A then Stage B
python -m src.bin.orchestrate all
python -m src.bin.orchestrate a --skip-tier1
python -m src.bin.orchestrate b

# Thin CLI router (see src/bin/main.py MODULE_MAP)
python -m src.bin.main train t1
python -m src.bin.main train t2
python -m src.bin.main train t3
python -m src.bin.main inspect phase1 -- --summary

# Datagen
python -m src.data_gen.pipeline_tier12
python -m src.data_gen.pipeline_tier3
```

## Source layout (short)

| Path | Role |
|------|------|
| `src/architecture/` | `GINO_DEQ`, DEQ solver hooks, LoRA, SIREN decoder |
| `src/core_physics/` | Physics kernels, Anderson acceleration, PDE-consistent terms |
| `src/config.py` | `PhysicsConfig`, `VesselConfig`, channel enums (`PredChannels`, `NodeFeat`) |
| `src/data_gen/` | Tier 12 / Tier 3 pipelines and mesh→graph builders |
| `src/training/` | `train_t1_predictor`, `train_t2_predictor`, `train_t3_corrector`, `physics_curriculum`, `t1_explorer` |
| `src/bin/` | `main` (router), `orchestrate` (stage runner) |

Training is implemented as **explicit scripts** (not a shared trainer class). Tier 1 may use `train_t1.py` as a small wrapper that forwards into `train_t1_predictor`.
