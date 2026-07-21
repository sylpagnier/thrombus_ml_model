# Bend-sign A/B (L1 convergence)

## Hypothesis

April-2026 kinematics used **down-only** arcs (`bend_sign=+1`). May-2026 added **bidirectional** bends (`bend_sign ∈ {−1,+1}`), which may explain **L1 val Rel L2 ≈ 1** while **L0** stays ~0.35–0.5.

## Modes

| Mode | Env / CLI | L1 arc/hook | L1 S-curve |
|------|-----------|-------------|------------|
| `down_only` | `KINEMATICS_BEND_SIGN_MODE=down_only` or `--bend-sign-mode down_only` | always `bend_sign=+1` | amplitude ≥ 0 |
| `bidirectional` | default | random ±1 | random sign |

`bend_sign` and `bend_sign_mode` are stored in `vessel_<id>.json` and copied onto `.pt` graphs.

## Fast A/B (other machine)

```powershell
cd D:\Users\Comsol\PycharmProjects\thrombus_ml_model
git pull

# Both arms: datagen + 12-epoch train smoke each (~isolated under ab_bend_*)
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_kinematics_bend_ab.ps1 `
  -Arm both -NumVessels 120 -AnchorMax 0 -Seed 42

# Datagen only (no train)
powershell -File .\scripts\go_kinematics_bend_ab.ps1 -Arm down -NumVessels 120 -DatagenOnly
```

Compare validation lines **`L1=`** between arms. **`AnchorMax 0`** skips COMSOL (train/val μ still run; anchor Rel L2 is weak). For meaningful val, use `-AnchorMax 60` (slow).

## Manual single arm

```powershell
$env:KINEMATICS_BEND_SIGN_MODE = "down_only"
$env:KINEMATICS_GRAPH_RHEOLOGY_DIR = "data/processed/graphs_kinematics/ab_bend_down/newtonian"

python -m src.data_gen.pipeline_kinematics --batch --rheology newtonian `
  -n 120 --mixed-levels --overwrite --bend-sign-mode down_only --seed 42 --skip-anchor

python -m src.training.train_kinematics_predictor --fresh --limit-data 120 `
  --epochs 12 --adam-epochs 10 --stage1-end-epoch 8 --l0l1-only-epochs 4
```

Your **existing 500-graph** cohort was built **bidirectional**; it cannot be re-labeled without new meshes. Use A/B dirs or regen with `--bend-sign-mode down_only`.
