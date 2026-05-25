# Kinematics best-model architecture (Stage A)

Canonical **architecture and training flags** for the production **GINO_DEQ** run that biochem should match when building the frozen kinematics backbone. Weights are not required in this repo; the committed reference JSON is the source of truth for constructor kwargs and curriculum.

## Reference files

| Path | Purpose |
|------|---------|
| [data/reference/kinematics_best_20260426T184600Z.json](../../data/reference/kinematics_best_20260426T184600Z.json) | **`model_config`** + **`training_recipe`** + best-epoch val metrics (no torch, no weights) |
| `outputs/kinematics/kinematics_architecture.json` | Optional: updated when you train kinematics here and save a new best |
| `outputs/kinematics/kinematics_best.pth` | Optional: weights; if present, may also embed the same `model_config` |

Override the reference JSON with `KINEMATICS_MODEL_CONFIG_REF=/path/to/manifest.json`.

## Best run (2026-04-26)

- **Source project**: `LadHyX_ml_cfd_thrombus_predictions`
- **Run id**: `20260426T184600Z`
- **Best epoch**: 84 (Stage 3 Carreau), **before** L-BFGS steps produced NaN validation
- **Val**: Rel L2 ≈ 0.1007, \|∇·u\| mean ≈ 0.157, composite ≈ 15.80

## GINO_DEQ constructor (must match for biochem load)

| Field | Value |
|-------|-------|
| `latent_dim` | 256 |
| `num_fourier_freqs` | 16 |
| `use_siren_decoder` | true |
| `use_hard_bcs` | true |
| `use_width_priors` | true |
| `max_iters` | 25 |
| `fourier_base` | 2.0 |
| `activation_fn` | silu |

Code: `snapshot_gino_deq_model_config` / `resolve_gino_deq_ctor_kwargs` in [src/architecture/kinematics_model_config.py](../architecture/kinematics_model_config.py).

## Geometry curriculum (L0 / L1 / L2)

Stage-A training supports **geometry-level weighted sampling** and **stratified validation** (default **on**).

| Phase | Epochs (default) | Sampling intent |
|-------|------------------|-----------------|
| `foundation` | Stage 1 (0–39) | 45% L0, 45% L1, 10% L2 — easy hot start |
| `ramp` | Stage 2 (40–59) | Blend → 30/30/40 |
| `l2_heavy` | Stage 3 (60+) | 15/15/70 — thrombus-target geometry |

- **Foundation train**: `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_kinematics_foundation.ps1 -Fresh`
- **L2 finetune** (after foundation ckpt): `.\scripts\go_kinematics_l2_finetune.ps1`
- **Backfill** `geometry_level` on existing `.pt` (no COMSOL): `python -m src.data_gen.backfill_kinematics_geometry_level`
- Disable: `--no-geometry-curriculum`

**Data:** Mixed cohort needs L0+L1 meshes (`pipeline_kinematics --mixed-levels`). L2-only disks cannot run `foundation`; use finetune-only or regen mixed vessels.

## Agents / biochem

`train_biochem_corrector.py` reads Stage-A shape from, in order: `model_config` inside `kinematics_best.pth` (if any), then **`data/reference/kinematics_best_20260426T184600Z.json`**, then tensor-shape inference. For this project, the reference JSON is enough—you do not need LadHyX weights in `outputs/kinematics/`.
