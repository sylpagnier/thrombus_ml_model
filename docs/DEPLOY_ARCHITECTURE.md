# Deploy architecture — clot growth ladder (CAVO)

**Status:** canonical (2026-06-07)  
**North-star:** full-mesh `clot_shape` (location-weighted F1, mu >= 0.055 Pa*s)  
**Val anchor:** `patient007`  
**Time cap:** COMSOL anchors use **7950 s** (`BIOCHEM_T_MAX=8000`); do **not** use `gnode_8h_ladder/anchors_stride_72` (stale 30 ks `data.t`).

Related: [CLOT_FORECAST_LADDER.md](CLOT_FORECAST_LADDER.md), [CLOT_PHI_ROLLOUT.md](CLOT_PHI_ROLLOUT.md).

---

## Goal

Geometry + flow @ t=0 -> clot shell at each **COMSOL macro time** (no extrapolation). v1 = **where/when** clot forms; pred-kine / DEQ is optional until `clot_shape >= 0.08`.

---

## Two parallel tracks

| Track | Machine | Data | Stages | Blocks? |
|-------|---------|------|--------|---------|
| **A — fast (this PC)** | Main | `data/processed/graphs_biochem_anchors` | 0.1, 0.3, **S0->G2** | **No dump** |
| **B — slow (other PC / overnight)** | GPU box | same COMSOL -> pred-kine dump | **0.2 only** | Does not block Track A |

```text
Track A (start now)                Track B (parallel)
-------------------                ------------------
0.1 R0  [done]                     0.2 pred-kine dump (7950s, T=54)
0.3 band viz (COMSOL)                    go_clot_deploy_dump_comsol.ps1
S0 static_final                           6 patients (no patient005)
S1 from_t0
G1 one-step
G2 carry rollout
       \                           merge before Stage F
        `------------------------> F pred flow (optional)
                                 D deploy package
```

**Patients in repo:** `patient001`, `002`, `003`, `004`, `006`, `007` (no `patient005`).

---

## Ladder (sequential gates on Track A)

| # | Stage | Question | Pair schedule | Flow | Gate (p007) |
|---|-------|----------|---------------|------|-------------|
| 0.1 | R0 | Labels sane? | — | — | R0 PASS |
| 0.3 | Band viz | B0 hugs GT? | — | — | visual |
| **S0** | Static shell | Final clot from t=0? | `static_final` | GT | shape >= 0.06, pred_frac < 0.25 |
| **S1** | Multi-horizon | Clot @ each t_k from t=0? | `from_t0` | GT | mean shape >= 0.04 |
| **G1** | One-step | t_k -> t_{k+1}? | `rolling` | GT + mu carry warm | shape >= 0.08, pred_frac < 0.20 |
| **G2** | Carry rollout | Closed loop on series? | rollout | GT | late-T shape >= G1 |
| **F** | Pred flow | DEQ + dump u,v? | rollout | **pred kine** | optional; needs dump |
| **D** | Package | inference manifest | — | Tier A | shape >= 0.10 |

Do **not** skip S0. Do **not** enter F before G2 passes (and dump exists).

---

## Defaults (all S/G stages)

- **Binary shell:** phi BCE + fixed `mu_solid=0.10`; no mu regression
- **Hard projection:** mu = Carreau off-band; model mu on B_t only
- **Anchor dir (Track A):** `data/processed/graphs_biochem_anchors`
- **Anchor dir (Stage F+):** `outputs/biochem/gnode10_sweep/anchors_gnode12_predkine_uvp`

---

## Commands — Track A (this computer, no dump)

**0.3 — band sanity (COMSOL):**

```powershell
python -m src.evaluation.viz_clot_phi_masks --anchor patient007 --time-index 0
```

**S0 — localization gate:**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_deploy_s0_static.ps1" -Fresh
```

**S1 — multi-horizon (after S0 passes):**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_deploy_s1_multih.ps1" -Fresh
```

**G1 — rolling one-step:**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_deploy_phase1.ps1" -Fresh `
  -InitCheckpoint "outputs/biochem/clot_deploy/s1_from_t0/clot_phi_best.pth"
```

**G2 — carry rollout:** wire `_clot_deploy_binary_base.ps1` + `go_clot_forecast_r2b.ps1` after G1 passes.

---

## Commands — Track B (other computer, dump only)

Full COMSOL horizon (7950 s, ~54 frames). **Do not** use `go_gnode12_lane_a.ps1` default June dir.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_deploy_dump_comsol.ps1"
```

Optional subset (same 6 patients):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_deploy_dump_comsol.ps1" `
  -Only "patient001,patient002,patient003,patient004,patient006,patient007"
```

Output: `outputs/biochem/gnode10_sweep/anchors_gnode12_predkine_uvp/*.pt`  
Copy/sync that folder back before Stage F.

---

## CAVO forward

```text
  mu_blend = log_blend(mu_c, phi)
  mu_deploy[B_t] = mu_blend[B_t];  elsewhere mu_c
  B_t = dgamma_wall@t0 U 1-hop(clot seeds)
```

| Env | S0-S1 | G1+ |
|-----|-------|-----|
| `CLOT_PHI_HARD_SUPPORT_PROJECTION` | 1 | 1 |
| `CLOT_FORECAST_MASK` | deploy_band | input |
| `CLOT_FORECAST_INPUT_MU` | 0 | 1 |
| `CLOT_FORECAST_MU_CARRY` | off | on |

---

## Time / data pitfalls

| Source | p007 T | t_max | Use for |
|--------|--------|-------|---------|
| `graphs_biochem_anchors` | 54 | 7950 s | **Track A, R0, S0-G2** |
| `gnode10_sweep/anchors_stride_72` | 5 | 7950 s | legacy gnode smoke only |
| `gnode_8h_ladder/anchors_stride_72` | 5 | **30000 s** | **avoid** (wrong t) |
| pred-kine dump (Option B) | 54 | 7950 s | **Stage F, deploy eval** |

---

## Key modules

| Concern | Path |
|---------|------|
| Pair schedules | `src/core_physics/clot_forecast.py` |
| Hard projection | `src/core_physics/clot_phi_simple.py` |
| Training | `src/training/train_clot_phi_simple.py` |
| Metric | `src/evaluation/clot_shape_score.py` |
| Dump launcher | `scripts/go_clot_deploy_dump_comsol.ps1` |

---

## Chronicle

| Date | Note |
|------|------|
| 2026-06-07 | CAVO ladder; hard projection; S0/S1/G1 launchers |
| 2026-06-07 | Parallel tracks: COMSOL fast path vs dump; 7950s cap documented |
