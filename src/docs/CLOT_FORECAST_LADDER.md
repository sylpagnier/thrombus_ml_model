# Clot forecast ladder (legacy R0–R6)

**Canonical deploy plan (2026-06-07):** [DEPLOY_ARCHITECTURE.md](DEPLOY_ARCHITECTURE.md) — use **Phase 0 → S0 → S1 → G1 → G2 → (F) → D** for new work.

This file retains R0–R6 experiment history and env detail from the forecast ladder exploration.

**Canonical val anchor:** `patient007`  
**Promoted one-step ckpt (R1):** `outputs/biochem/clot_forecast_ladder/r1_prong_d/clot_phi_best.pth`

---

## Complexity stack (what each rung adds)

```text
R0   labels only          mu(t) -> mu(t+dt) sanity (no model)
R1   + one-step head      features @ t_in  ->  clot/mu @ t_out   [GT flow]
R2   + temporal carry     phi/mu roll forward across macro steps  [GT flow]
R3   + rollout loss       multi-step BCE in one backward pass
R4   + GINO-DEQ           u,v from predicted mu (two-way coupling)
R5   fork (if stuck)      MPNN depth / clot-GNODE
R6   deploy eval          neighbor commit mask, no gt_clot; full pipeline viz
```

**Mask vs forward (all rungs):**

| | Full mesh forward | Loss / commit |
|--|-------------------|---------------|
| MLP (R1–R2) | Yes — per-node, no message passing | Deploy / neighbor **band only** |
| GINO-DEQ (R4+) | **Full vessel** required for u,v,p | Clot μ committed in band only; bulk = Carreau |

Stripping the graph for MLP forward does **not** change band predictions (local features only). Skip unless profiling speed.

---

## Ladder table (status 2026-06-06)

| Rung | Adds | Status | p007 gate | Launcher |
|------|------|--------|-----------|----------|
| **R0** | Label pairs | **PASS** (3/3) | growth + \|dlog μ\| | `go_clot_forecast_r0.ps1` |
| **R1** | One-step forecast | **PASS** (prong D) | F1 ≥ 0.40 | `go_clot_forecast_r1.ps1 -Prong A\|B\|C\|D` |
| **R2** | Multi-step carry (hybrid mu) | **FAIL** | late-T `clot_shape` vs R1D | `go_clot_forecast_r2.ps1` |
| **R2B** | Bridge A GT carry | **FAIL** | train ok / val mismatch | `go_clot_forecast_r2b.ps1` |
| **R2-simple** | Phi rollout + fixed μ | **FAIL** shape | band F1 ~0.20 | `go_clot_forecast_r2_simple.ps1` |
| **R2α** | One-step phi, in_dim=3 | **FAIL** shape | band F1 ~0.20 | `go_clot_forecast_r2a_one_step.ps1` |
| **R2α+** | One-step phi + log(μ@t_in) | **PARTIAL** | band F1 **~0.56**; shape **0.014** | `go_clot_forecast_r2a_plus.ps1` |
| **R2β** | Phi-carry rollout (init R2α+) | planned | shape > 0.10 | TBD |
| **R4** | + GINO-DEQ kine | planned | within 0.08 F1 of R3 | `go_rung6b_clot_phi_rollout_kine.ps1` |
| **R5** | Arch fork | if stuck | — | MPNN / clot-GNODE |
| **R6** | Deploy eval | after R4 | deploy probe + mask viz | `go_mlp_b_deploy_probe.ps1` |

\* `go_rung6a_clot_phi_rollout_gt.ps1` is an alias for `go_clot_forecast_r2.ps1`.

---

## When can you viz?

| Rung | Spatial viz? | How | What you see |
|------|----------------|-----|--------------|
| **R0** | No | `r0_label_sanity.json` | Scalar growth stats only |
| **R1** | Metrics (+ optional PNG) | `multi_anchor.jsonl`; `go_clot_forecast_r1_viz.ps1` | Band F1/logMAE; PNG is same-frame sanity only |
| **R2** | **Yes** | `go_clot_forecast_r2.ps1` + `clot_shape` in train log | Rolled φ/μ at final T; deploy_pred band |
| **R4** | Flow + clot | rung6b + pipeline viz | μ blockage affecting u,v |
| **R6** | Deploy masks | `go_mlp_commit_mask_viz.ps1 -Leg B_deploy` | Oracle vs deploy commit band |

**Metric cheat sheet**

| Metric | Scope | Use |
|--------|-------|-----|
| `f1` / `pred+` | **Loss band only** (neighbor or deploy_pred) | Training gate, collapse detection |
| **`clot_shape`** | **Full mesh**, μ≥0.055 Pa·s, location-weighted F1 | North-star spatial score (rank legs) |
| `score` | Band F1 + heuristics | Legacy ckpt picker (not clot_shape) |

**Rule of thumb:** Pretty PNGs can look good on the **neighbor band** while **clot_shape** on the full mesh tells you if deploy spatial quality is real.

---

## R1 prongs (completed)

Core env (all): `CLOT_FORECAST_MODE=one_step`, `CLOT_PHI_ROLLOUT=0`, `CLOT_PHI_VEL_SOURCE=gt`, `CLOT_PHI_JOINT_BIO=0`.

| Prong | Config | p007 F1 | mean F1 | min F1 | Notes |
|-------|--------|---------|---------|--------|-------|
| **A** | MLP, no μ_t, mask=target @ t_out | 0.476 | 0.282 | 0.038 | Baseline; μ_t required |
| **B** | MLP + log(μ_t), mask=target | 0.573 | 0.366 | 0.075 | Strong backbone |
| **C** | MPNN + log(μ_t) | 0.578 | 0.365 | 0.075 | ~tie B |
| **D** | B + **deploy_pred** band | **0.584** | 0.554* | 0.038 | **Promoted**; deploy-faithful |

\* mean inflated by p001/p004 (~0.99 F1 inside nearly all-clot bands). Judge on **p007** and deploy eval, not mean F1.

| Prong | Env highlights |
|-------|----------------|
| A | `CLOT_FORECAST_INPUT_MU=0` |
| B | `CLOT_FORECAST_INPUT_MU=1` |
| C | `CLOT_PHI_MODEL=mpnn`, `CLOT_FORECAST_INPUT_MU=1` |
| D | `CLOT_FORECAST_MASK=deploy_pred`, `NEIGHBOR_REQUIRE_PHI=0`, init R1B ckpt |

**Do not use:** `deploy_input` + GT φ @ t_in + `REQUIRE_PHI=1` (degenerate F1=1.0 band).

Code: `src/core_physics/clot_forecast.py`, `train_clot_phi_simple.py`.

---

## R1 → R2 handoff

Carry into R2:

- Init: `outputs/biochem/clot_forecast_ladder/r1_prong_d/clot_phi_best.pth`
- Keep: `log(μ_t)` input, hybrid MLP, minimal features, deploy_pred loss band
- Add: `CLOT_PHI_ROLLOUT=1`, `CLOT_PHI_CARRY_PHI=1`, `CLOT_PHI_CARRY_LOG_MU=1`, still `CLOT_PHI_VEL_SOURCE=gt`

Success: late macro times show **spatial growth** (pred φ/μ extends beyond t=0 seed), not collapse to bulk.

---

## R2 rung6a result (legacy recipe, 2026-06-06)

**Not forecast-aligned:** `forecast=legacy`, `mask=neighbor`, `joint_bio=1`, `species_feat=1`, no R1D init.

| Metric (p007) | R1D | R2 rung6a |
|---------------|-----|-----------|
| Band F1 | 0.584 | 0.491 |
| Band logMAE | 0.035 | 0.479 |
| `clot_shape` | (re-run with new log) | (re-run with new log) |

Visual PNG at t=53 looks strong on **neighbor band** φ/μ — that is **not** the deployable stack (oracle band + GT flow + rolled carry). Re-run rung6a with **R1D init + forecast env** before promoting.

---

## R2 aligned result (2026-06-06)

R1D init + `deploy_pred` + pred `log(mu)` carry (no bridge): **collapse** by ep ~10.

| Symptom | Typical val |
|---------|-------------|
| Band F1 | ~0.000 (spikes ~0.2) |
| Band gt+ | ~0.003 |
| logMAE | ~1.8 |
| clot_shape | ~0.014 flat |

**Cause:** R1D 4th feature = GT `log(mu_t)`; R2 carry slot = pred `log(mu_{t-1})` (zeros @ t=0). Same `in_dim`, different semantics.

---

## R2B Bridge A (GT carry warm-up)

Train-only bridge so early epochs match R1D input semantics, then fade to pred carry (eval always pred carry).

| Env | Default (launcher) | Meaning |
|-----|-------------------|---------|
| `CLOT_PHI_CARRY_GT_WARMUP_EPOCHS` | 15 | Epochs 0..14: carry slot = log(GT mu @ ti) |
| `CLOT_PHI_CARRY_GT_FADE_EPOCHS` | 10 | Epochs 15..24: linear blend GT -> pred carry |
| `CLOT_PHI_CARRY_GT_WARMUP_STEPS` | 0 | Optional: first K macro indices also use GT (after epoch warm-up) |

Code: `src/core_physics/clot_phi_rollout.py` (`resolve_carry_log_mu_feature`), `clot_forecast.resolve_rollout_prev_mu_si`.

**Gate:** p007 val `clot_shape` should rise above ~0.1 during GT warm-up; after fade, band gt+ >> 0.003 and logMAE well below R2.

---

## R2-simple (phi-only + fixed mu_solid)

One rolled state (phi only). No hybrid head, no mu carry, no GT bridge.

| Env | Value | Meaning |
|-----|-------|---------|
| `CLOT_PHI_FIXED_MU_FROM_PHI` | 1 | `mu = log_blend(mu_c, phi, mu_solid)` |
| `CLOT_PHI_HYBRID` | 0 | Phi BCE only (`CLOT_PHI_MU_LOG_LAMBDA=0`) |
| `CLOT_PHI_CARRY_PHI` | 1 | 4th feature = phi_prev |
| `CLOT_PHI_CARRY_LOG_MU` | 0 | No mu in feature vector |
| `CLOT_PHI_MU_SOLID_SI` | 0.10 | Fixed clot viscosity |

Deploy band seeds use `mu_eff_from_carried_phi(mu_c, phi_prev)`. Ckpt score weights `clot_shape` + band F1.

Code: `mu_eff_from_carried_phi`, `go_clot_forecast_r2_simple.ps1`.

---

## R2α+ result (2026-06-06)

**Config:** one-step, phi-only + `log(GT mu @ t_in)`, fixed `mu_solid=0.10`, `deploy_pred`, cold start.

| Metric (p007) | R1D hybrid | R2α (in_dim=3) | R2α+ |
|---------------|------------|----------------|------|
| Band F1 | 0.584 | 0.205 | **~0.559** |
| `clot_shape` | — | 0.014 | **0.014** |
| logMAE (diag) | 0.035 | 1.84 | 1.83 |

**Pass:** band F1 near R1D without hybrid regression. **Fail:** `clot_shape` unchanged — band quality does not transfer to full-mesh spatial score.

**Note:** first run hit ckpt scorer bug (`pred+~0.003` rejected); fixed in train loop — re-run to save `clot_phi_best.pth`.

---

## Deploy sweep (~30 min)

Explore deploy-faithful one-step variants ranked by **`clot_shape`**:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_forecast_deploy_sweep_30m.ps1"
```

Default legs (8 ep each): `deploy_base`, `hard_labels`, `mesh_aux`, `target_mask`, `mesh_aux_hard`.

| Leg | What it tests |
|-----|----------------|
| deploy_base | R2α+ baseline (`deploy_pred`) |
| hard_labels | `CLOT_PHI_SOFT_LABELS=0` |
| mesh_aux | Full-mesh eligible-lumen aux BCE + bulk phi penalty |
| target_mask | Oracle neighbor @ t_out (curriculum) + light mesh aux |
| mesh_aux_hard | Hard labels + strong mesh aux |

**New training knobs:** `CLOT_PHI_MESH_AUX_LAMBDA`, `CLOT_PHI_MESH_BULK_LAMBDA`, `CLOT_PHI_SHAPE_USE_T_OUT=1` (fix clot_shape @ t_out).

Summary: `outputs/biochem/sweep_clot_forecast_deploy_30m/summary.jsonl` (rank = 0.65×shape + 0.35×band F1).

---

## Next commands

**R1 viz sanity (optional, today):**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_forecast_r1_viz.ps1" -Prong D
```

**R2α one-step phi-only (gate before rollout):**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_forecast_r2a_one_step.ps1" -Fresh
```

**R2-simple train (phi rollout — after R2α passes):**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_forecast_r2_simple.ps1" -Fresh
```

**R2 / R2B (hybrid mu carry — known collapse):**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_forecast_r2.ps1" -Fresh
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_forecast_r2b.ps1" -Fresh
```

**R6 deploy mask viz (not R1 — full teacher + inject stack):**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_mlp_commit_mask_viz.ps1" -Leg B_deploy -Anchor patient007
```

---

## Off-ladder (do not use as starting point)

- Full GNODE species ODE + gelation before R4 passes
- Deploy Leg B with `gt_clot` removed before R6
- Coupled finetune without `go_diagnose_deploy_gate.ps1`
- Legacy clot-phi ladder (joint bio / species-heavy rung 4–6a) as substitute for forecast R1D

## References

- Training log: [BIOCHEM_TRAINING_PROGRESS.md](BIOCHEM_TRAINING_PROGRESS.md) §180–182
- Rollout detail: [CLOT_PHI_ROLLOUT.md](CLOT_PHI_ROLLOUT.md)
- Deploy masks: `go_mlp_commit_mask_viz.ps1`, `src/core_physics/clot_phi_mu_inject.py`
