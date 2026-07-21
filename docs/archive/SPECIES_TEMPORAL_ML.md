# Species temporal ML (fresh start)

Conceptual reset for learning **how biochem species evolve over time** on vessel
graphs, ignoring rules-based s0/s4/s5 heads. Grounded in GT COMSOL anchors
(6 vessels, ~54 macro steps each).

---

## Physics (what COMSOL is doing)

At each macro time step the state is roughly:

```
y[t, node] = [u, v, p, mu_nd, 12 x log1p(species_nd)]
```

**12 species:** 9 bulk (RP..FI) + 3 wall (M, Mas, **Mat**). Clot viscosity uses
**FI** (bulk) and **Mat** (wall) through gelation `mu1(Mat) + mu2(FI)`.

Mechanisms on the mesh:

1. **Resting IC at t=0** -- PT/AT/FG at log1p~0.69 everywhere; FI/Mat/M ~ 0.
2. **Wall-local kinetics** -- cascade intermediates (APR, APS) rise first on the
   wall band under shear; FI/Mat rise later in **localized** patches.
3. **Weak mesh transport** -- neighbor vs local change ratio ~ **0.34** across
   vessels (reaction-dominated on nodes; not dye advecting around bulk).
4. **Sharp gelation** -- tiny log-ND FI/Mat moves can flip mu and clot commits.

Flow enters mainly as **shear activation** on the wall, not as carrying species
through the lumen (at least at clot-relevant magnitudes).

---

## Diagnostic (run first)

```powershell
python scripts/diagnose_species_temporal_patterns.py --max-times 14
```

Output: `outputs/biochem/diagnostics/species_temporal_patterns.json`

**patient007 @ t=53 (illustrative):**

| Region | FI (log ND) | Mat (log ND) | APS (log ND) |
|--------|-------------|--------------|--------------|
| t=0 wall | 0 | 0 | 0 |
| t=53 clot | ~1.1e-4 | ~4.1e-4 | ~1.9e-2 |

Bulk PT/AT/FG stay near resting; **APR/APS move more than FI** in log space but
FI/Mat still gate clot.

---

## What ML must learn (patterns)

| Pattern | Implication |
|---------|-------------|
| Uniform resting IC | Predict **delta from rest**, not absolute 12-ch vector |
| Spatial sparsity | Train on **wall + 1-hop band**, not full mesh |
| Time localization | Onset ~ mid-tau; need **temporal rollout**, not per-t independent |
| Channel sparsity | 4-6 moving channels (APR, APS, FI, Mat, maybe T); not all 12 |
| Local vs transport | Start with **reaction + mild graph diffusion**, not full ADR |
| Gate vs magnitude | Oracle: gate ~0.50 F1 headroom, species ~0.99 -- **where** and **how much** are coupled but identifiable |

---

## Recommended architecture (v0)

**Wall-band Graph Reaction Rollout (WGRR)**

```
S_0 = resting_species (fixed, from config)
For t = 0 .. T-1:
    feats = [S_t, u,v,p, shear, sdf, wall, tau, commits_{t-1}]  on wall band
    dS = GNN_reaction(feats, edge_index_wall)   # source + neighbor coupling
    S_{t+1} = clamp(S_t + dt * dS, rest, S_max)  on band; bulk pinned to rest
    mu, phi = Carreau + gelation(S_{t+1}, u,v)   # frozen physics
```

Components:

1. **Subset state** -- `APR, APS, FI, Mat` (4 ch); optional `T`.
2. **Mesh GNN** -- 2 layers on wall subgraph; edge features from flow direction.
3. **Temporal** -- explicit macro-step rollout (not independent per-t); optional
   GRU on nodes for slow memory.
4. **Loss** -- masked MSE on moving channels in wall band + optional commit BCE
   on FN/FP from coupled phi; **no** FP->rest wipeout.
5. **Train** -- LOAO on 6 vessels; val F1 from coupled gelation, not species MSE alone.

**Not recommended as v0:** full 12-ch GNODE teacher, s0 rule deltas, per-step
FI/Mat-only without cascade intermediates.

---

## Experiments ladder

| Step | Question | Tool |
|------|----------|------|
| D0 | Species timelines all vessels | `diagnose_species_temporal_patterns.py` |
| D1 | Which channels correlate with shear onset | extend diagnostic |
| D2 | Oracle ceiling per channel subset | masked GT species in wall band |
| M0 | WGRR 4-ch, teacher-forced 1-step | new trainer |
| M1 | Coupled multi-step rollout + gelation loss | same |
| M2 | Add graph diffusion head | ablation |

---

## Viz convention

Deploy triage panels: **GT clot | R4.s0 rules | model step** (`viz_t0_rung4_step.py`).
Oracle ceilings (R2 GT species, teacher) only with `--include-ceilings`.
