# Mat-growth simulation to-do

Living queue for clot Mat architecture experiments. **Selection metric:** all-anchor
`deploy_clot_f1` and `deploy_clot_relaxed_prec` in each leg's `compare.json`.

Locked reference: fi_mat baseline **0.634** (`outputs/biochem/biochem_gnn/species/best.pth`).

Partial full-budget log:
`outputs/biochem/biochem_gnn/mat_only_full/mat_only_partial_summary.json`

---

## Precision-first training (landed 2026-06-26)

Mat-growth recipe now **anti wall-paint by default**:

| Knob | Value | Effect |
|------|-------|--------|
| `SPECIES_CONTINUOUS_CLOUT_SCORE` | `relaxed_prec_floor` | Checkpoint score = precision with recall floor |
| `SPECIES_CONTINUOUS_SCORE_CLOUT_W` | `0.75` | 75% weight on deploy clot score vs mat F1 |
| `SPECIES_CONTINUOUS_FP_WEIGHT` | `16` | Heavy FP on magnitude at zero-growth nodes |
| `SPECIES_CONTINUOUS_GATE_FP_WEIGHT` | `4` | BCE on spatial gate at zero-growth nodes |
| `SPECIES_CONTINUOUS_SPATIAL_LOSS_WEIGHT` | `2` | Stronger gate supervision |
| `SPECIES_CONTINUOUS_SPEED_FP_WEIGHT` | `6` | Penalize growth in fast-flow false positives |
| `SPECIES_MAT_GROWTH_PRECISION_SELECT` | `1` | Enable precision checkpoint formula + clot eval |

Triage launcher: `scripts/go_mat_arch_triage.ps1 -Fresh` (20 ep / 64 windows / ~45-90 min per leg).

---

## Already done (full budget -- do not re-run unless promoting)

| Leg | deploy_clot_f1 | Note |
|-----|---------------:|------|
| **P_mat_plain** | **0.762** | Leader (old recall-heavy selection) |
| G_dual_mat_neighbor_gate | 0.724 | Full budget complete |
| N_mat_geom_rich | aborted ~ep24 | Re-run at triage tier |

---

## Architecture map

### Dense / scope baselines

| Leg | Physical idea |
|-----|---------------|
| P_mat_plain | Mat-only control (no gate, no geom) |
| G_dual_mat_neighbor_gate | Autocatalysis via committed-neighbour gate |
| N_mat_geom_rich | Vessel geometry into field (2-hop rich) |
| O_mat_neighbor_geom_rich | G + N stack |

### SeedFrontMat pivot (sparse nucleation + slow front)

| Leg | Physical idea |
|-----|---------------|
| U_mat_frontier_only | Top-k seed + 1-hop front only |
| V_mat_frontier_geom | U + rich geometry |
| Y_mat_tight_seed | U with top-2% seeds (vs 5%) |
| S_mat_frontier_nuc | Full v0: gate + geom + front |
| T_mat_frontier_sharp | S + sharp gate + spatial FP |

### Physically guided heads (NEW)

| Leg | Physical idea |
|-----|---------------|
| **W_mat_flow_stagnation** | Low-shear / stagnation flow features (`SPECIES_FLOW_FEATS`) -- clots nucleate in recirculation pockets |
| **X_mat_flow_seedfront** | Stagnation prior + SeedFront structural mask |
| **AB_mat_gelation_aux** | Differentiable gelation readout aux loss (`mu1(Mat)` physics head during train) |

### Gate-precision ablations

| Leg | Note |
|-----|------|
| Q_mat_gate_sharp_fp | Sharp gate + spatial FP, no geom |
| R_mat_geom_gate_sharp_fp | Q + rich geom; subsumed by T |

### Deferred (wait for triage results)

| Module | When |
|--------|------|
| Design A geometry clot-likelihood prior head | Pivot legs plateau below P |
| Committed-neighbour GAT gate | Overpaint persists after T |
| GT growth-rate matching aux loss | Frontier hops insufficient |

---

## PRIORITY NOW -- promote W (+ precision P control)

**Triage complete (8/8 legs, 2026-06-26).** Winner: **W** clot_f1 **0.764** (+0.130 vs locked 0.441),
clot_score **0.978**, over/gt **0.01**. Runner-up AB **0.671** but over/gt **0.20** (deprioritize).
Summary: `outputs/biochem/biochem_gnn/mat_arch_triage/mat_arch_triage_summary.json`.

| Tier | Candidate | clot_f1 | Why explore further |
|------|-----------|--------:|---------------------|
| **A** | `W_mat_flow_stagnation` | 0.764 | Primary -- matches full-budget P at 1/5 cost |
| **B** | `P_mat_plain` + precision recipe | 0.762* | Fair control (*full budget, old selection) |
| **C** | `WX` flow + frontier (new) | -- | Untested hybrid: W growth + U precision |
| **D** | `U_mat_frontier_only` | 0.455 | Precision-safe fallback (score 0.884) |
| **E** | `N_mat_geom_rich` | -- | Finish aborted full-budget geometry leg |

```powershell
# Primary promotion (~5 h)
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_mat_growth_simple.ps1 -Fresh -Leg W_mat_flow_stagnation -Epochs 40 -MaxWindows 0

# Fair control: P under same precision-first recipe (~5 h)
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_mat_growth_simple.ps1 -Fresh -Leg P_mat_plain -Epochs 40 -MaxWindows 0
```

Triage ranking (deploy_clot_f1): **W 0.764 > AB 0.671 > T 0.497 > V 0.470 > U 0.455 > S 0.447 > X 0.443 > Y 0.415**.

---

## WAIT -- longer runs

| Tier | When | Recipe |
|------|------|--------|
| **Promote 1-2 winners** | Triage picks leg within ~0.02 of best on clot_f1 + lower overpaint | 40 ep / all windows (~5 h/leg) |
| **Full ladder** | Triage inconclusive | `go_mat_only_full_overnight.ps1` (8-12 h cap) |
| **New modules** | All wired legs plateau | Design A / GAT gate |

---

## Open questions

1. ~~Does stagnation flow (W/X) beat geometry-only (N/V) for precision?~~ **Yes -- W wins triage; X flat.**
2. ~~Does tighter top-2% seed (Y) beat default 5% (U)?~~ **No -- Y 0.415 < U 0.455.**
3. ~~Does gelation aux (AB) improve ranking without overpaint?~~ **No -- AB 0.671 but over/gt 0.20.**
4. ~~Does precision-first selection change the P vs G ranking?~~ **W matches P=0.762 at triage tier.**
5. Does W hold at 40 ep / all windows? Does flow+frontier hybrid beat W alone?
