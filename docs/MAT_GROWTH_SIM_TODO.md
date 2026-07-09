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

| Leg | deploy_clot_f1 | deploy_clot_score | Note |
|-----|---------------:|------------------:|------|
| **W_mat_flow_stagnation** | **0.792** | **0.981** | **Canonical Mat deploy** (2026-06-29 full compare; **superseded 2026-07-02**) |
| **WC_mat_flow_dynamic** | 0.789 | **0.947** | **Canonical Mat deploy** (2026-07-02 FP-aware pick) |
| P_mat_plain | 0.798 | 0.946 | Precision-first control (was 0.762 old selection) |
| G_dual_mat_neighbor_gate | 0.724 | — | Full budget complete |
| N_mat_geom_rich | aborted ~ep24 | — | Re-run at triage tier |

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

## DONE -- W-physics triage (2026-06-26)

Summary: `outputs/biochem/biochem_gnn/mat_physics_triage/mat_physics_triage_summary.json`

| Leg | clot_f1 | clot_score | over/gt | Verdict |
|-----|--------:|-----------:|--------:|---------|
| **WC_mat_flow_dynamic** | **0.772** | 0.927 | 0.01 | **Promote A2** (40 ep) |
| **W_mat_flow_stagnation** | 0.771 | **0.976** | 0.01 | **Promote A** (40 ep) |
| WF_mat_flow_fg | 0.767 | 0.947 | 0.00 | Optional backup |
| WA/WI/WJ | 0.75-0.76 | 0.91-0.92 | 0.01-0.04 | Incremental; skip full budget |
| WG_mat_flow_neighbor_crit | 0.712 | 0.809 | 0.02 | Mat-heavy; deprioritize |
| WD_mat_flow_frontier | 0.001 | 0.004 | 0.00 | **Kill** (cold-start starvation) |

## DONE -- W/WC/P full compare (2026-06-29)

Summary: `outputs/biochem/biochem_gnn/mat_w_wc_p_full/mat_w_wc_p_full_summary.json`

**Promoted:** `W_mat_flow_stagnation` -> `mat_growth_ladder/W_mat_flow_stagnation/species/best.pth`

## DONE -- W-fix sweep (2026-07-01)

Summary: `outputs/biochem/biochem_gnn/mat_w_fix_sweep_10h/mat_w_fix_sweep_10h_summary.json`

8/9 legs @ 28 ep (~10.8 h). **WC** best timeline FP (medFP 14, p90 34); **W** best score 0.961.
**X/Y seedfront fail** (undergrowth). **WK/WL dropxy fail**. **WM** not run (budget cap).

## DONE -- W/WC canonical (2026-07-02)

Summary: `outputs/biochem/biochem_gnn/mat_w_wc_canonical/mat_w_wc_canonical_summary.json`

**Promoted:** `WC_mat_flow_dynamic` -> `mat_growth_ladder/WC_mat_flow_dynamic/species/best.pth`
(FP-aware pick: medFP **12**, p90 **36**, F1 **0.789** vs W medFP **56** p90 **99** F1 **0.795**).

Run `-Promote` to copy alias -> `mat_canonical_deploy/species/best.pth`.

## BACKLOG -- post canonical

| Task | Why |
|------|-----|
| Run **WM_mat_flow_seedfront_tightfp** (1 leg) | Only unfinished W-fix sweep leg |
| FP-aware winner on prior summaries | Re-rank `mat_w_fix_sweep_10h` with new `--minimize-metrics` |

## DEPRIORITIZE (W-fix evidence)

| Leg / lever | Why |
|-------------|-----|
| X / Y seed-front (without flow) | F1 0.42-0.50; cold-start undergrowth |
| WK / WL drop-x/y only | No FP improvement vs W |
| WG neighbor+crit | mat_f1 up, clot_f1 down |

---

## WAIT -- longer runs

| Tier | When | Recipe |
|------|------|--------|
| **go_mat_w_wc_canonical.ps1** | **Done 2026-07-02** | WC promoted @ 40 ep |
| **WF_mat_flow_fg** | Only if W over-grows | 40 ep optional backup |

---

## Open questions

1. ~~Does stagnation flow (W/X) beat geometry-only (N/V) for precision?~~ **Yes -- W wins triage; X flat.**
2. ~~Does tighter top-2% seed (Y) beat default 5% (U)?~~ **No -- Y 0.415 < U 0.455.**
3. ~~Does gelation aux (AB) improve ranking without overpaint?~~ **No -- AB 0.671 but over/gt 0.20.**
4. ~~Does precision-first selection change the P vs G ranking?~~ **W matches P=0.762 at triage tier.**
5. ~~Does W hold at 40 ep / all windows? Does **WC** beat W at full budget?~~ **WC wins** on FP-aware deploy pick (§191); W +0.006 F1 but 2-3x worse medFP/p90FP.
6. ~~Does flow+frontier without top-k (WD) help topology?~~ **No -- WD dead at deploy.**
