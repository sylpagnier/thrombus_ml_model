# HemoGINO

Mesh-agnostic graph neural surrogate for vessel hemodynamics and coupled biochemistry (Tier 1–3), with physics-informed training and DEQ-style inference.

## Documentation

| Document | Contents |
|----------|----------|
| [`docs/PROJECT_CONTEXT.md`](docs/PROJECT_CONTEXT.md) | Tiers vs stages, **architecture**, entry points, data layout, inspection tools, tests policy |
| [`docs/TIER1_TRAINING_HISTORY.md`](docs/TIER1_TRAINING_HISTORY.md) | Tier 1 experiments, sweep decisions, V2/V3 notes |
| [`docs/README.md`](docs/README.md) | Doc index and quick commands |

## Quick start

```text
# Kinematics then Biochem (phase B)
python -m src.bin.orchestrate all

# Unified kinematics pretraining only
python -m src.training.train_kinematics_predictor

# Unified CLI (train / data / eval / inspect / orchestrate)
python -m src.bin.main train kinematics

# Kinematics flow demo (sliders + GINO-DEQ inference; headless: add --no-gui)
python -m src.bin.main inspect flow -- --rheology carreau
```

Artifacts: checkpoints under `outputs/kinematics/` and `outputs/biochem/`; reports under `outputs/reports/`; datasets under `data/` via `data_root()`.

## Tests

```text
# Full suite
pytest src/tests/

# Kinematics-only suite (skips biochem/phase-3 coverage)
pytest src/tests/ --suite=kinematics

# Biochem suite (includes all kinematics tests)
pytest src/tests/ --suite=biochem
```

See `PROJECT_CONTEXT.md` for which regression modules track CLI routing and graph builders.
