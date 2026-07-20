# Biochem deploy baseline (`biochem_deploy`)

> **Naming:** Stack id **`biochem_deploy`** = frozen **RGP-DEQ** (`pmgp_deq_kine`) + **GraphSAGE pushforward** + **gelation_beta** + **mechanistic clot readout**. See [MODEL_NOMENCLATURE.md](MODEL_NOMENCLATURE.md).

## Two biochem training paths (historical)

| | **biochem_deploy** (canonical, active) | **train_biochem_corrector** (GNODE, **retired 2026-06**) |
|---|--------------------------------------|-----------------------------------------------|
| Entry | `python -m src.bin.main train biochem-deploy` | removed (`go_biochem_gnn.ps1` is the active path) |
| Model | GraphSAGE pushforward + physics clot readout | `GNODE_Phase3` graph neural ODE (deleted) |
| Mesh scope | Ceiling / wall band (~1-hop) | Full vessel graph |
| Flow | External frozen RGP-DEQ / GINO-DEQ | Learned / GT kine in-graph |
| Mu | Physics gelation from species (`gelation_beta`) | Learned mu heads |
| Clot | `clot_trigger_physics` + mat-growth | Clot-phi probes, MLP injectors |
| Deploy | Yes (no GT species at inference) | Research only (archived) |

GNODE teacher/corrector launchers and modules were removed from the active surface; see `docs/archive/2026-06-16-biochem-cleanup.md` and `AGENTS.md`.

## Components

```
pmgp_deq_kine           [frozen RGP-DEQ ckpt, Stage A]
  -> species_graphsage  [trained]  wall-band GraphSAGE pushforward (FI/Mat)
  -> gelation_beta      [trained]  global Mat gelation scale
  -> clot_trigger_physics [equations] Carreau + gelation + nucleation phi
  -> flow_coupling      [future]   mu -> RGP-DEQ MU_PRIOR refresh
```

## Paths (legacy dirs still resolve)

```
outputs/biochem/biochem_gnn/          # active artifact tree today
  locked/species_gnn_best.pth         # CANONICAL (WC_v7_clot_phi_mse, 2026-07-19)
  mat_canonical_deploy/species/best.pth  # synced alias
  species/best.pth                    # synced warm-start alias
  viscosity/beta.pth
  loao/holdout_*/
data/reference/biochem_gnn_baseline.json
data/reference/mat_canonical_deploy.json
```

Canonical leg: **`WC_v7_clot_phi_mse`**. New mat-growth improvements warm-start from `locked/species_gnn_best.pth`.

Legacy ladders and aliases were archived from the active surface; see `BIOCHEM_LEGACY_LESSONS.md` and `archive/2026-06-16-biochem-cleanup.md`.

## Commands

```powershell
# Full baseline pipeline
powershell -File .\scripts\go_biochem_gnn.ps1 -Step all -Gate -Viz

# Python trainer
python -m src.bin.main train biochem-deploy -- --step all --all-anchors
python -m src.training.train_biochem_gnn --step species
```

## Deploy horizon convention

Deploy metrics, gelation-beta calibration, and checkpoint gates use **each graph's last macro-step** (full COMSOL timeline), not a fixed index.

| Concept | Meaning |
|---------|---------|
| **Default eval** | `deploy_eval_time_index(n_times)` -> `n_times - 1` |
| **Legacy capped regime** | ~53 consecutive macro-steps on patient007 8ks export (~2.2 h physical); F1 ~0.70-0.73 at that checkpoint |
| **Override** | `SPECIES_CONTINUOUS_DEPLOY_HORIZON=N` caps eval/aux unroll to step N |
| **Unroll VRAM cap** | `SPECIES_PUSHFORWARD_MAX_UNROLL` (default 200) limits training curriculum length, not eval time |

Helpers: `graph_last_time_index`, `default_deploy_metric_times`, `LEGACY_CAPPED_DEPLOY_HORIZON` (53) in `species_pushforward_continuous.py`.

Eval scripts (`eval_t0_rung4_species_gnn_loao.py`, `predict_species_gnn_deploy`) default to per-graph times `[0, 27, legacy_cap, last]` when `--times` is omitted.

## Deploy-faithful rollout

Set automatically by ``apply_deploy_env()``:

| Env | Default | Meaning |
|-----|---------|---------|
| `SPECIES_ROLLOUT_IC_SOURCE` | `resting` | FI/Mat t=0 from plasma IC |
| `SPECIES_ROLLOUT_PIN_OTHER` | `rest` | non-FI/Mat pinned to resting |
| `SPECIES_ROLLOUT_VEL_SOURCE` | `kinematics` | vel-decay uses pred GINO-DEQ |
| `SPECIES_ROLLOUT_DEPLOY_FAITHFUL` | `1` | Enable above defaults |
| `SPECIES_CONTINUOUS_DEPLOY_HORIZON` | `0` | Per-graph last step; set `>0` to cap eval horizon |
| `SPECIES_CONTINUOUS_DEPLOY_EVAL_FULL` | `1` | Score deploy metrics at each graph's last macro-step |
| `SPECIES_PUSHFORWARD_MAX_UNROLL` | `200` | Training unroll VRAM cap (not eval time) |

## `BiochemDeployStack` (Python)

```python
from src.biochem_deploy import BiochemDeployStack, FlowMode

model = BiochemDeployStack.from_manifest(anchor="patient007", flow_mode=FlowMode.COUPLED)
out = model.rollout(data)  # out.phi_by_time, out.mu_by_time, out.species_series
```

Alias: `BiochemGNN = BiochemDeployStack`.

Coupled mode is **not yet validated** against locked baseline F1 (~0.70).

Baseline comparison table: [BIOCHEM_GNN_BASELINES.md](BIOCHEM_GNN_BASELINES.md).

## Species channels: only FI + Mat

Deploy GNN uses ``STATE_DIM = 2``. Other species pinned to ``resting_species_log_nd`` at inference.
