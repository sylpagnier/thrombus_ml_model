# Clot-phi coupled rollout (ladder rung 6)

Serial **macro** loop like `GNODE_Phase3.forward`, but with the **clot-phi MLP** instead of species ODE + full biochem:

```text
t=0:  phi_0, mu_0 = MLP(features_0)
t=1:  (optional) u_1,v_1 = KinematicsDEQ(mu_prior=mu_0)   # rung 6b
        features_1 = [geom, u_1,v_1, species, carry(phi_0,mu_0)]
        phi_1, mu_1 = MLP(features_1)
...
```

## Why this rung

| Rung | What it tests |
|------|----------------|
| 3b | Same MLP weights at every `ti` with **GT** `[u,v](t)` — not a closed loop |
| 4–5 | Species supply (GT or dumped); still per-frame |
| **6a** | **Temporal carry** + optional **serialized loss** without flow error (GT velocity) |
| **6b** | **Two-way** clot -> `MU_PRIOR` -> DEQ -> new `[u,v]` -> clot (Stage-A kine) |
| 8–11 | Full GNODE TBPTT |

This is the right bridge to **viscosity over time** and eventually **geometry at t=0 only**.

## Phases

### 6a — GT velocity (teacher forcing on flow)

- `[u,v]` at step `ti` always from COMSOL `data.y[ti]` (same as today).
- **Carry**: append `phi_{t-1}` and/or `log(mu_{t-1})` to node features (detached between steps by default).
- **Loss**: sum over `ti` in one backward pass per graph (or detach carry for stability).
- **Gate**: late-`T` phi/mu not worse than rung 3b; viz shows **second layer / growth** improving vs single-frame.

Env: `CLOT_PHI_ROLLOUT=1`, `CLOT_PHI_VEL_SOURCE=gt`, `CLOT_PHI_CARRY_PHI=1`, `CLOT_PHI_CARRY_LOG_MU=1`.

Launcher: `scripts/go_rung6a_clot_phi_rollout_gt.ps1`.

### 6b — Predicted kinematics (coupled flow)

- After `phi_{t-1}, mu_{t-1}`, set `data.x[:, MU_PRIOR] = mu_eff_nd` and run **one** steady `GINO_DEQ` forward (Stage-A `kinematics_best.pth`).
- MLP features use **predicted** `[u,v]` (not GT).
- Optional `CLOT_PHI_KINE_TF` in `(0,1)` to blend pred/GT velocity for stability while kine is imperfect.

Env: `CLOT_PHI_VEL_SOURCE=kinematics`, `CLOT_PHI_KINE_CKPT=outputs/kinematics/kinematics_best.pth`.

Launcher: `scripts/go_rung6b_clot_phi_rollout_kine.ps1` (slower than 6a — one DEQ solve per time step per graph).

## Not in scope yet

- Message passing (old rung 6) -> **rung 7** in plan.
- Species ODE or ADR backprop.
- Training kine + clot end-to-end (freeze kine for 6b v1).

## Relation to biochem teacher

`GNODE_Phase3` already does `kin_in[:, MU_PRIOR] = current_mu_eff / scale` then `_solve_kinematics_macro`. Rung **6b** reuses that **interface** with the **tiny** clot head replacing species+gelation until species quality is proven.
