# HemoGINO

Mesh-agnostic graph neural surrogate for vessel hemodynamics and coupled biochemistry (Tier 1–3), with physics-informed training and DEQ-style inference.

## Documentation

| Document | Contents |
|----------|----------|
| [`src/docs/PROJECT_CONTEXT.md`](src/docs/PROJECT_CONTEXT.md) | Tiers vs stages, **architecture**, entry points, data layout, inspection tools, tests policy |
| [`src/docs/TIER1_TRAINING_HISTORY.md`](src/docs/TIER1_TRAINING_HISTORY.md) | Tier 1 experiments, sweep decisions, V2/V3 notes |
| [`src/docs/README.md`](src/docs/README.md) | Doc index and quick commands |

## Quick start

```text
# Stage A (Tier 1 + Tier 2 predictors) then Stage B (Tier 3 corrector)
python -m src.bin.orchestrate all

# Tier 1 predictor only (full feature set: explorer env vars, diary, checkpoints)
python -m src.training.train_t1_predictor

# Unified CLI (train / data / eval / inspect / orchestrate)
python -m src.bin.main train t1
```

Artifacts: checkpoints under `outputs/stage_a/` and `outputs/stage_b/`; reports under `outputs/reports/`; datasets under `data/` via `data_root()`.

## Tests

```text
pytest src/tests/
```

See `PROJECT_CONTEXT.md` for which regression modules track CLI routing and graph builders.
