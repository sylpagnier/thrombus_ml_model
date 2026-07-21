# Biochem GNN baselines

> **Active summary:** [../MAT_GROWTH.md](../MAT_GROWTH.md) and [../BIOCHEM_GNN.md](../BIOCHEM_GNN.md). This file is an archived leaderboard notebook.

Canonical comparison table for deploy stack (`biochem_gnn`) runs. Use these rows when judging faster iteration legs.

## Active baselines

| ID | Date | Graph | Band | Anchors | Best ep | Score | deploy Mat@t_last | deploy FI@t_last | val growth F1 | Notes |
|----|------|-------|------|---------|---------|-------|-------------------|------------------|---------------|-------|
| `triangle6_wall3hop_20260624` | 2026-06-24 | triangle6 P2 | wall+3hop | 10 | 27 | **0.673** | **0.594** | **0.626** | **0.791** | First post-mesh-fix baseline; species-only complete |
| `mat_growth_simple` | 2026-06-24 | triangle6 P2 | wall+3hop | 10 | 21* | 0.767* | 0.724* | 0.602* | 0.841* | **INVALID RUN** -- recipe not applied; see run log below |

\* Training metrics @ best ep21 on **p007 val only** (fi_mat dual-head, random init -- **not** the intended Mat-only recipe).

Full JSON: `outputs/biochem/biochem_gnn/baselines/triangle6_wall3hop_20260624/baseline.json`

Mat-only simple leg artifacts: `outputs/biochem/biochem_gnn/mat_growth_simple/`

### mat_growth_simple run log (2026-06-24) -- INVALID, do not promote

Launcher: `go_mat_growth_simple.ps1 -Fresh` (50 ep, ~7852 s).

**What we intended:** Mat-only, single-head, random init, growth Huber on wall+3hop, analytical gelation clot eval.

**What actually trained** (ckpt `meta` + console banner): **fi_mat dual-head** (`dual_head=1`, channels `[8,11]`), same as baseline architecture but **random init** (no warm-start from locked ckpt). Root cause: recipe env was set in a child `python -c` subprocess and did not persist to the training process. Fixed: `--recipe mat_growth_simple` on the trainer.

**Post-hoc compare eval** (`compare.json`) is also **invalid for the simple leg**: eval forced `scope=mat` on a fi_mat ckpt. Fixed in `eval_mat_growth_simple.py` (respect ckpt meta).

| metric (10-anchor mean, pred kine, analytical clot) | baseline `species/best.pth` | invalid simple run | delta |
|---|---:|---:|---:|
| deploy_mat_f1 | 0.702 | 0.430 | -0.273 |
| deploy_clot_f1 | 0.634 | 0.432 | -0.202 |
| deploy_clot_score | 0.579 | 0.415 | -0.164 |

**p007 only** (fairer apples-to-apples on val anchor): deploy_mat **0.588 vs 0.594**, deploy_clot **0.638 vs 0.645** -- near tie despite random init.

**Worst simple anchors (deploy_mat_f1):** p011 **0.075**, p005 **0.323**, p010 **0.352**, p006 **0.361** -- generalization collapse off p007.

**Training paradox:** val `val_mat_f1` ~0.99 (teacher oracle, meaningless); selection score 0.767 driven by p007 `deploy_mat_f1=0.724` at ep21. Cohort deploy_mat mean 0.43 shows the checkpoint does not generalize.

**Lesson:** Random-init fi_mat dual-head matches p007 but fails cohort vs warm-started baseline. True Mat-only test still pending after `--recipe` fix.

Audit: `outputs/biochem/biochem_gnn/mat_growth_simple/run_record_20260624_invalid.json`

### Mat-growth ladder (3-leg comparison)

Launcher: `scripts/go_mat_growth_ladder.ps1` (runs all legs + `summarize_mat_growth_ladder.py`).

| Leg | Init | Hypothesis |
|-----|------|------------|
| `A_random` | random | Simplest Mat-only single-head baseline |
| `B_backbone` | SAGE conv from `species/best.pth` | Transfer flow/geometry representation without FI head |
| `C_geom` | random + `SPECIES_GEOM_FEATS` | Static geometry discriminators (proven precision lever) |

Single leg: `go_mat_growth_simple.ps1 -Leg B_backbone -Fresh`

Artifacts: `outputs/biochem/biochem_gnn/mat_growth_ladder/<leg>/`
Run record: `outputs/biochem/biochem_gnn/mat_growth_ladder/run_record_20260625.json`

### Fast pair check (2026-06-25)

Launcher: `scripts/go_mat_growth_fast_pair.ps1 -Fresh` (fixed fast preset: 10 ep / early-stop 6 / max-windows 16, all anchors).

- `baseline_fast` trained as expected (`fi_mat`, dual-head), early-stopped at ep8, best score **0.552**.
- `D_parity_single` trained as expected (single-head, Mat-only, mat_readout warm-start), best score **0.645**.
- `D_parity_single` compare (vs locked species baseline) stayed low: Mat **0.322**, clot F1 **0.256**, clot score **0.202**.

Run ended with compare-step path error (`FileNotFoundError`) because the pair runner expected the parity ckpt under
`mat_growth_fast_pair/...` while the leg script writes to `mat_growth_ladder/D_parity_single/...`.
Fixed in `scripts/go_mat_growth_fast_pair.ps1`.

Fast-pair run record: `outputs/biochem/biochem_gnn/mat_growth_fast_pair/run_record_20260625.json`

Fast pair compare is now complete (via `go_mat_growth_fast_pair.ps1 -EvalOnly` after path fix):

| metric (10-anchor mean) | fast baseline (`baseline_fast`) | `D_parity_single` | delta |
|---|---:|---:|---:|
| deploy_mat_f1 | 0.448 | 0.322 | -0.126 |
| deploy_clot_f1 | 0.373 | 0.256 | -0.117 |
| deploy_clot_score | 0.298 | 0.202 | -0.096 |

Interpretation: under identical fast defaults (10 ep / early-stop 6 / max-windows 16, all anchors), parity single-head Mat-only underperforms even the fast baseline-like dual-head fi_mat leg.

### Head/scope A-B check (2026-06-25)

Launcher: `scripts/go_mat_head_scope_ab.ps1 -Fast -Fresh` (fixed fast preset: 10 ep / early-stop 6 / max-windows 16, all anchors).

Tested pair:
- `E_dual_mat`: dual-head, Mat-only, baseline-like dynamics.
- `F_single_fimat`: single-head, Mat+FI, baseline-like dynamics.

Per-leg compare vs locked species baseline (`outputs/biochem/biochem_gnn/species/best.pth`):

| metric (10-anchor mean) | locked baseline | `E_dual_mat` | `F_single_fimat` |
|---|---:|---:|---:|
| deploy_mat_f1 | 0.702 | 0.582 | 0.324 |
| deploy_clot_f1 | 0.634 | 0.435 | 0.256 |
| deploy_clot_score | 0.579 | 0.372 | 0.202 |

Direct A/B compare (`F_single_fimat` vs `E_dual_mat`):

| metric (10-anchor mean) | `E_dual_mat` baseline | `F_single_fimat` simple | delta |
|---|---:|---:|---:|
| deploy_mat_f1 | 0.582 | 0.324 | -0.257 |
| deploy_clot_f1 | 0.435 | 0.256 | -0.180 |
| deploy_clot_score | 0.372 | 0.202 | -0.170 |

Interpretation:
- Holding fast settings constant, **dual-head Mat-only** (`E_dual_mat`) clearly beats **single-head Mat+FI** (`F_single_fimat`).
- Both remain below the full locked dual-head fi_mat baseline, so this does not support dropping either factor globally.
- In this regime, head architecture appears to be the larger lever than adding FI to a single-head model.
- `F_single_fimat` landing near the earlier single-head Mat-only legs indicates FI channel presence alone does not recover deploy behavior without dual-head structure.

### Physics triple ablation (2026-06-25)

Launcher: `scripts/go_mat_physics_triple_ablation.ps1 -Fast -Fresh` (fixed fast preset: 10 ep / early-stop 6 / max-windows 16, all anchors).

Baseline used inside this run:
- `baseline_fast` (`fi_mat`, dual-head): deploy mean `Mat=0.448`, `clot_f1=0.373`, `clot_score=0.298`.

Tested legs:
- `G_dual_mat_neighbor_gate`: dual-head Mat-only + neighbor commit-aware spatial gate.
- `H_dual_mat_crit_focus`: dual-head Mat-only + crit-focused loss weighting.
- `I_dual_fimat_fi_aux`: dual-head fi_mat with FI as light auxiliary target (`channel_weight_fi=0.15`, `channel_weight_mat=8.0`).

| metric (10-anchor mean) | fast baseline | `G_dual_mat_neighbor_gate` | `H_dual_mat_crit_focus` | `I_dual_fimat_fi_aux` |
|---|---:|---:|---:|---:|
| deploy_mat_f1 | 0.448 | **0.599** | 0.562 | 0.402 |
| deploy_clot_f1 | 0.373 | **0.455** | 0.412 | 0.290 |
| deploy_clot_score | 0.298 | **0.389** | 0.347 | 0.286 |

Delta vs fast baseline:
- `G`: `+0.150` Mat, `+0.082` clot_f1, `+0.090` clot_score.
- `H`: `+0.114` Mat, `+0.039` clot_f1, `+0.048` clot_score.
- `I`: `-0.046` Mat, `-0.083` clot_f1, `-0.012` clot_score.

Delta vs locked triangle6 baseline (`species/best.pth`: 0.702 / 0.634 / 0.579):
- `G`: `-0.104` Mat, `-0.179` clot_f1, `-0.190` clot_score.
- `H`: `-0.140` Mat, `-0.222` clot_f1, `-0.232` clot_score.
- `I`: `-0.300` Mat, `-0.344` clot_f1, `-0.292` clot_score.

Interpretation:
- Winner of this triple is **G (neighbor commit-aware gate)**, with the largest gains over the run's fast baseline.
- **H (crit-focused weighting)** helps, but less than G; useful secondary knob, not primary.
- **I (light FI auxiliary)** hurts under this fast setup, including a sharp collapse on patient011 (`deploy_mat_f1 ~0.066` in compare JSON), indicating this weighting/scope recipe is brittle.
- Overall direction is consistent with prior A/B findings: improving **spatial support gating** is more effective than changing FI supervision weighting.

### G+H combo check (2026-06-25)

Launcher: `scripts/go_mat_gh_combo_fast.ps1 -Fresh` (same fast preset; leg `J_dual_mat_neighbor_crit` = G+H combined).

| compare | deploy_mat_f1 | deploy_clot_f1 | deploy_clot_score | delta Mat |
|---|---:|---:|---:|---:|
| J vs `baseline_fast` | 0.548 | 0.407 | 0.340 | **+0.099** |
| J vs `G` | 0.548 | 0.407 | 0.340 | -0.051 |
| J vs `H` | 0.548 | 0.407 | 0.340 | -0.014 |
| J vs locked baseline | 0.548 | 0.407 | 0.340 | -0.155 |

Interpretation:
- **J beats `baseline_fast`**, but **does not beat G or H alone**; combining both knobs is not synergistic in fast mode.
- Best single knob remains **G** (`0.599` Mat F1 vs fast baseline).
- J lands between H and G on Mat F1, suggesting partial overlap/conflict between gate prior and crit-focused loss pressure.

Run record: `outputs/biochem/biochem_gnn/mat_gh_combo_fast/compare_J_vs_*.json`

### Fast exploration scoreboard vs `baseline_fast` (2026-06-25)

Reference `baseline_fast` (`fi_mat`, dual-head, 10 ep / early-stop 6 / max-windows 16):
- `deploy_mat_f1=0.448`, `deploy_clot_f1=0.373`, `deploy_clot_score=0.298`

| leg | deploy_mat_f1 | delta Mat | beat fast baseline? |
|---|---:|---:|:---:|
| `G_dual_mat_neighbor_gate` | **0.599** | **+0.150** | yes |
| `H_dual_mat_crit_focus` | 0.562 | +0.114 | yes |
| `J_dual_mat_neighbor_crit` (G+H) | 0.548 | +0.099 | yes |
| `E_dual_mat` | 0.582* | +0.134* | yes* |
| `I_dual_fimat_fi_aux` | 0.402 | -0.046 | no |
| `D_parity_single` | 0.322 | -0.126 | no |
| `F_single_fimat` | 0.324 | -0.126* | no |
| `A_random` / `B_backbone` / `C_geom` | ~0.32 | ~-0.13* | no |

\* `E` and `F` deltas vs `baseline_fast` are inferred from same-preset cohort means in adjacent runs (not re-evaluated in one script).

**Answer:** yes, multiple legs beat the fast baseline. Clear winners: **G > H > J**. No explored leg beats the locked full baseline (`0.702` Mat F1) in this fast regime; best gap remains about `-0.10` Mat F1 (`G`).

#### Current run (2026-06-25, triangle6 wall+3hop)
Reference (`triangle6_wall3hop_20260624`, 10-anchor deploy_frozen vs analytical gelation):
- `deploy_mat_f1` = 0.702
- `deploy_clot_f1` = 0.634
- `deploy_clot_score` = 0.579

| leg | init mode | `deploy_mat_f1` | `deploy_clot_f1` | `deploy_clot_score` |
|---|---|---:|---:|---:|
| `A_random` | random | 0.319 | 0.256 | 0.202 |
| `B_backbone` | backbone | 0.326 | 0.256 | 0.202 |
| `C_geom` | random + geom feats | 0.323 | 0.257 | 0.203 |

Observation: all three Mat-only simplicity legs collapse to essentially the same behavior and underperform baseline by ~-0.38 Mat F1 and ~-0.38 clot F1. Backbone warm-start and static geometry features do not rescue the Mat-only loss-to-deploy mapping in this configuration.

Train log: `outputs/biochem/biochem_gnn/species/train_log.jsonl`

## Config fingerprint (`triangle6_wall3hop_20260624`)

- Launcher: `go_biochem_gnn.ps1 -Step species -Fresh -AllAnchors` (50 ep default)
- `SPECIES_SNAPSHOT_WALL_HOPS=3`, `CLOT_PHI_CEILING_HOPS=3`
- Graph: full triangle6 edges + rebuilt `edge_attr` (see `scripts/patch_biochem_anchor_triangle6_edges.py`)
- Init: warm-start from `locked/species_gnn_best.pth` (corner-graph era weights)
- Score (species): `0.70*deploy_mat_f1 + 0.15*val_growth_f1 + 0.10*val_state_f1 + 0.05*val_growth_mat_f1`
- Deploy eval: patient007 `t=200` (full timeline), pred kinematics vel-decay

## How to compare a new leg

1. Train with same launcher flags unless intentionally ablating one knob.
2. Copy `outputs/biochem/biochem_gnn/species/best.json` + `train_log.jsonl` into `outputs/biochem/biochem_gnn/baselines/<leg_id>/`.
3. Record delta vs `compare_keys` in `baseline.json`.
4. Prefer **deploy Mat/FI @ t_last** for clot-relevant comparison; val Mat F1 is teacher-oracle and saturates ~0.98.

## Legacy (pre-triangle6, do not compare hop-for-hop)

Corner-only `edge_index` (~75% degree-0 mid-edge nodes on p007). Older locked ckpts and F1 ~0.70 references used that topology. Treat as a different graph regime.
