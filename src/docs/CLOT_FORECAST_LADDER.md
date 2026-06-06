# Clot forecast ladder (fresh start)

Predict **clot / mu maps into the future** from current state + flow, then optionally couple **GINO-DEQ** for flow blockage. This ladder is **separate** from the full GNODE species stack and from deploy_b (no `gt_clot` oracle in forward).

## Rungs

| Rung | What | Launcher | Gate |
|------|------|----------|------|
| **R0** | GT label sanity: `mu(t) -> mu(t+dt)` pairs | `go_clot_forecast_r0.ps1` | Clot growth + `\|dlog mu\|` signal on 3 anchors |
| **R1** | One-step forecast, GT flow | `go_clot_forecast_r1.ps1 -Prong A\|B\|C` | p007 1-step F1 >= 0.40 @ multiple times |
| **R2** | Multi-step carry, GT flow | `go_rung6a_clot_phi_rollout_gt.ps1` | Late-T growth vs R1 |
| **R3** | Short rollout loss | TBD | clot_shape > 0.25 mean |
| **R4** | + GINO-DEQ coupling | `go_rung6b_clot_phi_rollout_kine.ps1` | Within 0.08 F1 of R3 on p007 |
| **R5** | Architecture fork if stuck | MPNN depth / clot-GNODE | Only if R4 spatial growth weak |
| **R6** | Deploy eval (neighbor mask) | `go_mlp_b_deploy_probe.ps1` | After R4 passes |

## R1 prongs

| Prong | Env | Model |
|-------|-----|-------|
| **A** | `CLOT_FORECAST_INPUT_MU=0` | MLP hybrid |
| **B** | `CLOT_FORECAST_INPUT_MU=1` | MLP + `log(mu_t)` input |
| **C** | `CLOT_FORECAST_INPUT_MU=1`, `CLOT_PHI_MODEL=mpnn` | 1-hop MPNN hybrid |

Core env (all R1 prongs):

- `CLOT_FORECAST_MODE=one_step`
- `CLOT_PHI_ROLLOUT=0`, `CLOT_PHI_VEL_SOURCE=gt`
- `CLOT_PHI_JOINT_BIO=0` (no species head)

Code: `src/core_physics/clot_forecast.py`, training via `train_clot_phi_simple`.

## Off-ladder (do not use as starting point)

- Full GNODE species ODE + gelation
- Deploy Leg B with `gt_clot` mask removed before R4 passes
- Coupled finetune without diagnose gate
