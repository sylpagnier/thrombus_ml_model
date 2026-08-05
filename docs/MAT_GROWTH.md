# Mat-growth (canonical baseline)

Mat-growth is the active research path on top of locked **`biochem_gnn`**: warm-start the wall-band species GraphSAGE and improve **deploy clot** footprint (precision-aware) without reviving retired GNODE / clot-phi ladders.

## Locked baseline (2026-07-19)

| | |
|--|--|
| **Leg** | `WC_v7_clot_phi_mse` |
| **Checkpoint** | `outputs/biochem/biochem_gnn/locked/species_gnn_best.pth` (local) |
| **Aliases** | `mat_canonical_deploy/species/best.pth`, `species/best.pth` |
| **Manifests** | `data/reference/biochem_gnn_baseline.json`, `data/reference/mat_canonical_deploy.json` |
| **Cohort (mean)** | clot score **~0.791**, clot F1 **~0.767**, Mat F1 **~0.714** |

Selection metrics for new legs: all-anchor `deploy_clot_f1` and `deploy_clot_relaxed_prec` (see each leg `compare.json`).

New work should **warm-start from locked** (or `species/best.pth`) and apply env via `mat_growth_leg_spec("WC_v7_clot_phi_mse")` unless deliberately ablating.

## Generalization eval (wall-gen / phase1)

Two hard rules (also in `AGENTS.md`):

1. **Deploy-faithful scoring only.** Cold eval uses RGP-DEQ base flow at t=0 and local kinematic tiling/corrector afterward — never COMSOL GT `[u,v]`. New vessels provide geometry only. Prefer `eval_mat_growth_simple.py` / `canonical_deploy_clot_metrics` with `flow_feats_source=auto` and corrector coupling on.
2. **Primary holdout = `patient020` only** (clot-rich). Do not gate legs on a mean that includes clot-free `patient034`. Optional multi-vessel tables are secondary.

Small cohort defaults: train `005,006,010,023,002`; val/holdout `020`. Flow-source gate: `go_flow_source_ab.ps1` / `run_flow_source_ab.py`. Phase1 tweaks on promoted **`FS_ab_coupled`**: `go_phase1_sweep_v3.ps1` (warm-start `wall_gen_baseline`). Promote: `python scripts/promote_wall_gen_baseline.py`. Plan: [`docs/GENERALIZATION_PLAN.md`](GENERALIZATION_PLAN.md).

**Trunk:** stay on **GraphSAGE**. Stage-A PM-GAT (`WG_physgat_*`, `arch=physics_gat`) is **not** an easy drop-in for wall-gen: cold physgat sprayed (~3.5× mass, cold p020 score stuck ~0.21) even after soft `physics_gat_prior_scale` + identity `edge_proj`. Best wall-gen feature stack remains SAGE **`WG_featfix_03`** (geom+flux, warm-start from `wall_gen_baseline`, cold p020 ~0.33). Do not prioritize further physgat sweeps unless investing in residual/LN trunk + warm-start/distill — default effort goes to SAGE feature / loss / coupling work.

## How to run

Supported launchers ([`scripts/README.md`](../scripts/README.md)):

| Launcher | Role |
|----------|------|
| `go_fresh_canonical.ps1` / `go_fresh_canonical_finish.ps1` | Promote WC legs into locked baseline |
| `go_mat_w_wc_canonical.ps1`, `go_mat_growth_simple.ps1`, `go_mat_growth_ladder.ps1` | Mat-growth training ladders |
| `go_off_wall_clot_sweep_6h.ps1` | Off-wall pivot (Pivot 3 occlusion survived) |
| `go_wc_v7_compound_growth_abc_orig10_9h.ps1` | Compound A/B/C on original anchors (~9 h) |
| `go_wc_v7_firewall_fix_seq.ps1` | Firewall / hop-stratified fix sequence |
| `go_wc_v7_offwall_limit_2h.ps1` | ~2h limit-analysis (extreme lumen/frontier/skiphop/blind) |
| `go_wc_v7_frontier_lumen_6h.ps1` | ~6h FrontierLumen scale-up on orig10 (compound A vs S) |
| `go_wc_v7_frontier_ge2_prec_8h.ps1` | ~8h precision Frontier-ge2 compound (orig10, compound val + wall floor) |
| `go_wc_v7_tile_cc_explore_2h.ps1` | ~2h A/B: union tile vs per-clot-region tiles |
| `go_wc_v7_open001_1h.ps1` | ~1h open-001 test (001+007+010 train, recall tilt) |
| `go_wc_v7_wall_only_retrain.ps1` | Step-3: retrain lumen specialist under wall-only vel-decay (warm-start D_Orig10) |
| `go_wc_v7_wall_lumen_target_9h.ps1` | ~9h autonomous: spray-gated specialist + WALL_ONLY → frontier-route deploy |
| `go_wc_v7_crack_001_3h.ps1` | ~3h solo-001 ladder (freeze / unfreeze / CC) to crack hop_ge2=0 |
| `go_wc_v7_frontier_ge2_prec_viz.ps1` | Hop-ladder viz for Frontier-ge2 prec A vs S |
| `eval_mat_growth_simple.py` | Cohort metrics (`--offwall-ckpt` / `--two-model-route`) |
| `go_viz_mat_w_wc_canonical.ps1`, `go_wc_v7_compound_orig10_viz.ps1` | Viz |

## Design notes (short)

- Prefer **precision-aware** clot scoring (anti wall-paint) over raw recall.
- Off-wall (hop >= 1) metrics matter for deploy claims; restore `meta.env_overrides` on eval.
- Compound / firewall sequences are **research budgets**, not the locked public baseline until promoted.

## Run log — firewall fix sequence (2026-07-22)

Launcher: `go_wc_v7_firewall_fix_seq.ps1 -Fresh` (steps 1+2; step 3 not run).  
Cohort: orig10 (`patient001–008,010,011`). Out: `outputs/biochem/offwall_model/wc_v7_firewall_fix_seq/`.

| Arm | clot_f1 | clot_score | offwall_n_pred / n_gt | offwall_strict_f1 | hop_ge2 n_pred / n_gt | hop_ge2 strict_f1 |
|-----|--------:|-----------:|----------------------:|------------------:|----------------------:|------------------:|
| **A** canonical WC_v7 | 0.769 | 0.792 | 1.1 / 20.1 | 0.185 | **0.0** / 18.5 | 0.000 |
| **Step1** `WC_v7_fw1_blind_sat` | 0.771 | 0.814 | 0.8 / 20.1 | 0.124 | **0.0** / 18.5 | 0.000 |
| **S** wall + lumen-shape specialist | 0.752 | 0.787 | 1.2 / 20.1 | 0.029 | **0.4** / 18.5 | 0.000 |

(Off-wall relaxed F1 uses the fixed offwall-only dilation metric.)

### Verdict

1. **Step 1 (blind+hop1-smooth+sat30) — wall OK, firewall intact.** Short finetune (~9 ep, ES, ~1.6 h) kept clot F1 and raised score slightly vs A, but **did not open hop≥2** (`hop_ge2 n_pred=0`). Off-wall became slightly more conservative than A.
2. **Step 2 specialist — train/val ≠ deploy.** Growth train on `hop_ge2` + `loss_lumen_shape` reached best `hop_ge2_balanced=0.289` @ ep 10 with val offwall **61/99** and RelF1 **0.73** (specialist-only val). Compound wall-route deploy still ~A volume (**1.2** offwall preds) and **hop_ge2=0.4** (essentially patient002 hop2 spray with **strict F1=0**). Wall clot F1 only −0.017 vs A.
3. **Gates:** wall F1 near A ≈ pass for Step1 / soft fail for S; `hop_ge2 n_pred→n_gt` **fail**; `hop_ge2` strict F1 **fail**.

### Lesson

Soft WC_v7 knobs + a hop≥2 Dice specialist are not enough: the specialist can match lumen **volume on its own val loop** without writing **deploy-threshold clot** at hop≥2 under compound rollout. Next: diagnose train-val vs deploy gap (threshold / horizon / saturation), try **frontier** route for the same ckpt, then Step3 isolate/skiphop if still blocked.

### Collapse autopsy (2026-07-22, patient007 probe)

Probe: `scripts/probe_firewall_collapse.py` → `probe_collapse_patient007.json`.

| Mode | clot_f1 | mat_f1 | offwall n_pred | hop_ge2 |
|------|--------:|-------:|---------------:|--------:|
| A wall alone | 0.796 | 0.707 | 0 | 0 |
| **G growth alone** (same ckpt as train-val) | **0.000** | **0.000** | 0 | 0 |
| S compound wall | 0.796 | 0.707 | 0 | 0 |
| F compound frontier | 0.557 | 0.502 | 1 | 1 (wrong) |

Train log claimed ep10 val offwall **61/99** / clot **0.51**. Reloading that ckpt under production `eval_mat_growth_simple` (and even coupled+gt flow) gives **zero Mat/clot**. So the “collapse” is not primarily wall-route blending:

1. **Train-val was non-deploy-faithful / non-reproducible.** Val static omitted `wall_mask_band` (dual-head sat/magnitude need it) and forced `flow_source=gt`. Ckpt metric optimized a phantom offwall count.
2. **hop_ge2-only supervision wrecked shared wall competence.** Solo specialist mat_f1=0 under real eval; it cannot nucleate wall clot, so it cannot be a unified deploy model.
3. **Compound wall-route therefore ≈ A.** Wall model owns wall (healthy); growth owns `~wall` but the specialist is dead there → S matches A on p007. Cohort-wide, S even **erases** A’s rare good offwall hits (e.g. patient006 6→0) by replacing WC_v7 `~wall` deltas with the dead specialist.
4. Frontier route does not rescue lumen; it damages wall (clot 0.56) for one wrong hop2.

**Code fix applied:** `train_offwall_growth` val now binds `wall_mask_band`, uses kinematics deploy flow like mat-growth eval, and logs hop_ge2 counts. Retune specialist with compound-aware / wall-protected training before another budget burn.

## Run log — off-wall limit 2h (2026-07-22)

Launcher: `go_wc_v7_offwall_limit_2h.ps1 -Fresh` (~1.5 h). p007 only; freeze-backbone + cheap-val train; compound wall-route probe.  
Out: `outputs/biochem/offwall_model/wc_v7_offwall_limit_2h/limit_2h_summary.json`.

| Arm | clot_f1 | hop_ge2 n_pred | hop_ge2 strict | Verdict |
|-----|--------:|---------------:|---------------:|---------|
| A wall | 0.796 | 0 | 0.000 | baseline |
| LumenPush | 0.797 | 2 | 0.020 | weak_volume |
| **FrontierPush** | 0.765 | **22** | **0.084** | **signal** |
| SkipHopSpec | 0.571 | 0 | 0.000 | null (wall damaged; hop_ge2 dead) |
| BlindSat | 0.797 | 2 | 0.020 | weak_volume |
| **FrontierLumen** | 0.785 | **16** | **0.071** | signal (lumen-only loss) |

**FrontierLumen** (`--supervise-mode frontier_lumen` = dilate(clot)&~wall): same extreme knobs as FrontierPush; wall clot F1 0.785 (vs A 0.796 / FrontierPush 0.765); hop_ge2 16 / strict 0.071 — slightly less volume than FrontierPush but cleaner wall. Prefer this for Track 1 scale-up if wall floor matters.

**Decision:** not fundamental-null. Extreme frontier (+ lumen-only variant) **can** move hop≥2 under honest compound deploy while roughly holding wall. Scale-up launcher: `go_wc_v7_frontier_lumen_6h.ps1` (orig10, ~6 h, wall clot floor + hop_ge2 gates). SkipHop on specialist+global env is a dead end here.

## Run log — FrontierLumen 6h scale-up (2026-07-23)

Launcher: `go_wc_v7_frontier_lumen_6h.ps1 -Fresh` (16 ep / ES8 / max-windows 48 / cheap-val / freeze-backbone).  
Out: `outputs/biochem/offwall_model/wc_v7_frontier_lumen_6h/` (`compare_frontier_lumen.json`).

| | A canon | S FrontierLumen | dS−A |
|--|--------:|----------------:|-----:|
| clot_f1 | 0.769 | 0.738 | **−0.030** |
| clot_score | 0.792 | 0.776 | −0.016 |
| offwall n_pred / n_gt | 1.1 / 20.1 | 8.0 / 20.1 | +6.9 |
| offwall strict_f1 | 0.185 | 0.055 | −0.129 |
| **hop_ge2 n_pred** | **0.0** | **5.0** | **+5.0** |
| hop_ge2 strict_f1 | 0.000 | 0.017 | +0.017 |

Gates: `hop_vol_up=True`, `hop_strict_up=True`, **`wall_ok=False`** (floor A−0.03) → verdict **`lumen_up_wall_regress`**.

Per-anchor (S hop_ge2 / clot): signal concentrated on **patient007** (10 ge2, strict **0.168**, clot≈A); spray without localization on **004** (18 ge2, strict 0), **008** (11, 0), **002** (5, 0). p001/011 unchanged.

### Verdict

1. **Limit-2h signal survives orig10** — mean hop_ge2 leaves 0; not a p007-only fluke.
2. **Localization does not scale** — cohort hop_ge2 strict only 0.017; overall offwall strict collapses vs A’s rare hop1 hits.
3. **Wall floor barely missed** (−0.0305 vs −0.03). Freeze-backbone is not enough when specialist FP on ~wall bleeds clot F1 via false lumen paint.
4. **Ckpt selection still cheap-val (−loss)** — no compound/deploy gate during train; ep16 lowest loss ≠ best hop_ge2 balance.

**Next (if continuing):** compound-aware val on p007 each epoch (or wall F1 floor in save); soften FN/underpred; reject saves that drop wall clot; optional per-anchor viz on 007 vs 004/008. Do **not** promote S over locked WC_v7.

## Run log — Frontier-ge2 precision 8h (2026-07-24)

Launcher: `go_wc_v7_frontier_ge2_prec_8h.ps1 -Fresh` (16 ep cap / ES6 / max-windows 56 / hops-k 5 / union tiles / compound-val every 2 / wall floor A−0.02).  
Stopped early at **ep14** (stale=6). Best ckpt = **ep2** (`hop_ge2_balanced` 0.318 on p007 compound).  
Out: `outputs/biochem/offwall_model/wc_v7_frontier_ge2_prec_8h/` (`compare_frontier_ge2_prec.json`).

### Orig10 mean A vs S

| | A canon | S ge2-prec | dS−A |
|--|--------:|-----------:|-----:|
| clot_f1 | 0.769 | 0.755 | **−0.014** |
| clot_score | 0.792 | 0.794 | +0.001 |
| offwall n_pred / n_gt | 1.1 / 20.1 | 6.3 / 20.1 | +5.2 |
| offwall strict_f1 | 0.185 | 0.137 | −0.047 |
| **hop_ge2 n_pred** | **0.0** | **4.3** | **+4.3** |
| hop_ge2 strict_f1 | 0.000 | **0.063** | **+0.063** |

Gates: `wall_ok=True` (floor A−0.02), `hop_vol_up=True`, `hop_strict_up=True`, `hop_strict_vs_6h=True` (0.063 ≫ 6h FrontierLumen 0.017) → launcher verdict **`pass_precision_signal`**.

### Per-anchor (S hop_ge2 / clot vs A)

| Anchor | ge2 n_pred / n_gt | ge2 strict | clot F1 (A→S) |
|--------|------------------:|-----------:|---------------|
| **007** | 17 / 97 | **0.140** | 0.796→0.786 |
| **006** | 14 / 6 | **0.200** | 0.782→0.699 |
| **010** | 2 / 12 | **0.286** | 0.833→0.852 |
| 004 | 1 / 0 | 0.000 | 0.715→0.699 |
| **008** | 9 / 0 | 0.000 | 0.393→0.365 |
| 001 | 0 / 68 | 0.000 | ~flat |

### Vs prior lumen arms

| Arm | clot Δ vs A | hop_ge2 n | hop_ge2 strict | spray |
|-----|------------:|----------:|---------------:|-------|
| FrontierLumen 6h | −0.030 | 5.0 | 0.017 | heavy 004/008 |
| Union tile 2h probe | ~0 | 4.5 | 0.038 | mild |
| **ge2-prec 8h** | **−0.014** | **4.3** | **0.063** | **004 almost clean; 008 still paints** |

### Goal check

Bar was: **high precision + lumen recall > WC_v7**, wall near A, beat 6h spray/quality.

| Criterion | Result |
|-----------|--------|
| Lumen recall > WC_v7 (hop_ge2 leaves 0) | **Yes** (0→4.3 mean; real on 007/006/010) |
| Precision better than 6h FrontierLumen | **Yes** (strict 0.063 vs 0.017; 004 spray largely gone) |
| Wall clot F1 within 0.02 of A | **Yes** (−0.014) |
| “High” absolute lumen precision | **Partial** — cohort strict still modest; overall offwall strict down vs A’s rare hop1 hits |
| Promote over locked WC_v7 | **No** — research compound only; 008 spray + ckpt picked early high-volume ep2 |

### Caveats

1. **Ckpt = ep2** under `hop_ge2_balanced`: later vals (ep8/12/14) had **higher p007 hop_ge2 strict** (0.13–0.14) but lower score, so were not saved. Volume-heavy early save may understate precision potential.
2. Train-time A_floor on p007 alone printed **0.481** (vs cohort A clot ~0.77) — floor still held for that val path, but the absolute number is not comparable to orig10 mean clot F1.
3. Residual failure mode is **008 zero-GT hop_ge2 paint** (9 preds, strict 0), not the old 004 flood.

**Decision:** 8h recipe **worked as a precision-tilted compound step** (passes scripted gates; clearly better product than FrontierLumen 6h). Goal of *usable high-precision lumen* is **only partially met** — keep as research Arm S, do not promote; next lever is ckpt score that prefers precision over volume and/or an 008 spray reject.

## EDA — compound lumen bottlenecks (2026-07-24)

Script: `scripts/eda_compound_lumen_bottlenecks.py`  
Report: `outputs/biochem/offwall_model/wc_v7_frontier_ge2_prec_8h/eda_lumen_bottlenecks.json`

### Data regime (orig10 GT, sampled timeline)

| Fact | Value |
|------|------:|
| Anchors with any peak hop_ge2 | **5 / 10** |
| Anchors with peak hop_ge2 ≥ 20 | **2 / 10** (001, 007) |
| Mean % frames with any hop_ge2 | **17.5%** |
| Peak-clot mass share wall / hop1 / hop_ge2 | **81.9% / 1.4% / 16.7%** |

Thick lumen is rare and late: most clot mass is wall; only 007 (and to a lesser extent 001) are strong hop_ge2 teachers. 002/003/004/008/011 have **peak hop_ge2 = 0**.

### Arm S failure taxonomy (8h compound vs A)

| Class | Anchors | Meaning |
|-------|---------|---------|
| `signal_localized` | 007, 010 | Real lumen hits, wall OK |
| `signal_but_wall_bleed` | 006 | Some overlap but clot F1 drops (over-volume 14 vs 6 GT) |
| `miss_all_lumen` | **001**, 005 | **001 has 68 GT hop_ge2 but S predicts 0** |
| `zero_gt_spray` | 004, 008 | Paint where GT lumen never exists |
| `no_lumen_gt_idle` | 002, 003, 011 | Correct idle (no GT lumen) |

### What is holding us back

1. **Severe class imbalance / teacher scarcity** — lumen learning is dominated by 007 (+ weak 006/010); half the cohort cannot supervise hop_ge2 at all.
2. **Wall-dominated labels** — even at peak clot, only ~17% of positive nodes are hop_ge2; loss still sees mostly wall-adjacent structure unless hard-masked (`frontier_ge2` helps but cannot invent missing teachers).
3. **Transfer failure on 001** — largest missed positive lumen reservoir (68 GT, 0 pred). Pattern learned on 007 does not generalize to 001’s thick clot.
4. **Zero-GT spray** — 004/008 get false lumen without any GT to counteract; precision tilt alone is insufficient on idle-lumen vessels.
5. **Ckpt volume bias** — ep2 save under `hop_ge2_balanced` favors early over-recall; fights the precision goal.

### Highest-leverage next experiments (ranked)

1. **Ckpt / val gate**: score on hop_ge2 **precision** (or strict) + reject zero-GT spray on 004/008; optionally multi-anchor compound val (007+001).
2. **Reweight / oversample** lumen-positive windows (001/007/010) and **downweight or mask loss** on zero-GT-lumen anchors during growth training.
3. **Diagnose 001 miss** (geometry / hop distribution / timing vs 007) — likely the biggest untapped recall gain without more spray.
4. Keep **union tiles**; do not default to per-component until spray gates exist.

### Follow-up levers (wired)

1. **Diag 001 vs 007:** `python scripts/diagnose_lumen_001_vs_007.py`  
   Deploy-time: 007 pred hop_ge2=13 (TP6/FP7/FN91); **001 pred hop_ge2=0 (FN68)**. Specialist is not globally dead — transfer fails on 001 (more compact lumen x_span ~0.97 vs 007 ~2.31).
2. **Recall limit analysis (~2h):** `go_wc_v7_lumen_recall_limit_2h.ps1 -Fresh`  
   Trains on **001+007 only** with extreme FN (`FN_W=25`, underpred=12), `--ckpt-metric hop_ge2_recall`. Probes A / Prec8hRef / RecallPush on 001,007,004,008.  
   Verdict classes: `capacity_yes_tuning_headroom` vs `null_architecture_suspect`.
3. **Open001 1h:** `go_wc_v7_open001_1h.ps1 -Fresh` — see run log below (001 still closed).

### Why 001 stays closed (deeper)

Not a missing-GT or orphan-lumen problem. Structure of 001 vs 007 is surprisingly similar:

| | 007 | 001 |
|--|----:|----:|
| Peak / deploy hop_ge2 GT | 97 | 68 |
| First wall clot → first hop_ge2 lag | 46 | 46 |
| Lumen dist to nearest wall-clot (med / p90) | 2 / 3 | 2 / 3 |
| Orphan lumen (unreachable from wall clot) | 0 | 0 |
| Lumen CCs (late) | 19 small | 10 (larger top-2: 33, 27) |
| Arm S hop_ge2 pred | 17 | **0** |
| Arm S all offwall pred | 26 | **0** |
| Open001 1h hop_ge2 pred | 27 | **0** |
| Solo001_CC (train 001 only) hop_ge2 pred | 38 | **0** |

So 001’s thick clot is attached like 007’s; earlier we read this as “specialist never turns on.” That was wrong for long-horizon deploy.

**Root cause (confirmed 2026-07-26):** `ROOT_vel_decay_wipes_001_lumen`

- Locked wall ckpt `log_vel_decay_mat` softplus ≈ **0.72** (not init ~3e-4). Per-step Mat *= `(1 - α·speed)` on the band.
- Wall clot speed ≈ **0** → wall Mat survives (healthy wall F1 ~0.81). Lumen GT speed p90: **001 ≈ 0.60** vs **007 ≈ 0.25** → ~43%/step wipe on 001 vs ~18%/step on 007.
- Ablation (`scripts/diagnose_001_root_cause.py`, growth `wc_v7_open001_6h/growth_D_Orig10_Band`): baseline **ge2=0/68**; `SPECIES_CONTINUOUS_VEL_DECAY=0` → **ge2=20/68**, offwall=44, F1 0.807→0.835. Report: `.../diagnose_root_cause.json`.
- Caveat: `load_continuous_bundle(apply_meta_env=True)` force-sets `VEL_DECAY=1` from meta — ablations must re-pin env after load.

**Rejected / secondary (still true, not the smoking gun):**

1. Not competition / freeze / union (crack_001).
2. Not orphan lumen / out-of-band GT (65/68 in band, orphan=0).
3. Compound-val blind band_static bug (fixed 2026-07-25) was real but orthogonal.
4. Solo-001 training fails under default deploy physics because the same wipe still runs at eval.

**Next fix options:** wall-only or speed-capped vel-decay; freeze/clamp α; do **not** ship blanket `VEL_DECAY=0` (see cohort below).

**Orig10 deploy ablation (D_Orig10_Band, wall-route):** `compare_vel_decay_cohort.json`

| | decay ON | decay OFF | Δ |
|--|--------:|---------:|--:|
| mean clot F1 | 0.710 | **0.582** | −0.129 |
| mean clot score | 0.776 | 0.737 | −0.039 |
| sum hop_ge2 | 99/185 | **539**/185 | +440 |
| 001 hop_ge2 | 0 | **20** | open |
| spray (002/006/010 ge2) | 22/14/0 | **100/73/114** | worse |

So decay-off unlocks 001 but **hurts cohort F1** via lumen spray on zero-GT vessels.

**Deploy contract (2026-07-26):** `SPECIES_CONTINUOUS_VEL_DECAY_WALL_ONLY=1` (default in train/deploy recipes). Vel-decay applies only on wall-band wall nodes; lumen Mat is not washed. Train unrolls + deploy rollouts share this rule; ckpt `meta.vel_decay_wall_only` / `env_overrides` restore it. Legacy full-band washout: `WALL_ONLY=0`.

**Orig10 re-eval under wall-only (same D_Orig10 weights, no retrain):** `compare_vel_decay_wall_only.json`

| | legacy full-band | wall_only | Δ |
|--|--------:|---------:|--:|
| mean clot F1 | 0.710 | **0.583** | −0.128 |
| mean clot score | 0.776 | 0.741 | −0.034 |
| sum hop_ge2 | 99/185 | **539**/185 | +440 |
| 001 hop_ge2 | 0 | **20** | open |

Wall-only ≈ decay-off here (wall speed≈0), so spray still dominates mean F1. Contract is correct for new-vessel deploy; **step 3** must retrain the lumen specialist under wall-only so it stops relying on wipe as an implicit FP filter.

### Run log — wall-only retrain step 3→4 (2026-07-26)

Launcher: `go_wc_v7_wall_only_retrain.ps1 -Fresh` (~2.9 h).  
Out: `outputs/biochem/offwall_model/wc_v7_wall_only_retrain/` (`compare_wall_only_retrain.json`).

Warm-start D_Orig10; freeze-backbone; `hop_ge2_balanced`; val=001; floor Δ=0.12. Best = **ep2** (val ge2 49/68, compound F1 0.790).

| Arm | mean clot F1 | sum hop_ge2 | 001 ge2 | 002 ge2 |
|-----|-------------:|------------:|--------:|--------:|
| A canonical (WALL_ONLY=1) | **0.780** | 59/185 | **50**/68 | 0 |
| legacy D_Orig10 (full-band) | 0.710 | 99/185 | **0**/68 | 22 |
| unretrain wall_only | 0.583 | 539/185 | 20/68 | 100 |
| **S retrain** | **0.518** | **764**/185 | **48**/68 | 93 |

**Gates:** `opened_001=True`; `mean_f1_near_A=False`; spray still worse than unretrain. **Verdict: `needs_work`.**

Notes: under WALL_ONLY, **wall-alone A already opens 001** (ge2 50) at mean F1 0.78 — stronger than S. Retrain opened 001 but increased spray (002/006/007/010). **Do not widen (step 5)** until precision/FP on zero-GT lumen is fixed; consider promoting wall-only A as interim deploy or a stronger FP-tilted / spray-aware val.

### Run log — wall+lumen target 9h (2026-07-26/27)

Launcher: `go_wc_v7_wall_lumen_target_9h.ps1` / `run_wall_lumen_target_9h.py`.  
Out: `outputs/biochem/offwall_model/wc_v7_wall_lumen_target_9h/`.

Contract: `VEL_DECAY=1` + `WALL_ONLY=1`. Goal: wall F1 near A, open 001+007 hop_ge2, zero-GT spray (002/004/008) ≈0.

| Phase | Arm | mean F1 | 001 ge2 | 007 ge2 | 002 ge2 | notes |
|-------|-----|--------:|--------:|--------:|--------:|-------|
| A | wall alone | **0.780** | 50/68 | 9/97 | 0 | spray-clean; weak lumen recall |
| A | Prec8h wall-route | 0.660 | 174/68 | 136/97 | 12 | legacy decay was FP filter |
| B | spray_max=3 retrain | 0.625 | 30/68 | 151/97 | 52 | all ckpt saves rejected |
| C | FP=12 soft spray (Prec8h init) | 0.651 | 58/68 | 109/97 | 35 | first valid save (ep6) |
| D | D_Orig10 + FP=16 | 0.582 | 20/68 | 138/97 | 89 | no valid spray-gated save |
| E | C + **frontier hops=2** | 0.683 | 60/68 | 83/97 | 13 | spray↓ but F1 < A−0.04 |
| E | C + **frontier hops=1** | **0.724** | **61**/68 | **32**/97 | **0** | spray-clean; F1 0.016 shy of 0.740 gate |
| E | C + frontier hops=0 | 0.780 | 50/68 | 9/97 | 0 | ≡ wall alone |

**EDA (`eda_spray_vs_gt_speed.json`):** spray idle band speed overlaps 001 GT (p90≈0.60) — speed washout cannot selectively kill 002 without wiping 001. **Physics pivot that worked:** deploy `two-model-route=frontier` with `frontier_hops=1` so the specialist only acts in the 1-hop neighborhood of committed Mat (not all lumen).

**Best compound so far:** `growth_C/best.pth` + WALL_ONLY + **frontier hops=1**. Gates: open 001/007 + spray-clean **pass**; mean F1 **0.724** vs need ≥0.740 (A−0.04). **Verdict: `near_target`** — do not promote yet; next lever is a short F1-recovery polish (cut 010/006 over-fire) under the same frontier-h1 deploy contract.

**Best growth ckpt:** `outputs/biochem/offwall_model/wc_v7_wall_lumen_target_9h/growth_C/best.pth`  
**Eval recipe:** `--two-model-route frontier --two-model-frontier-hops 1`

### WC v8 improvement sweeps (8h, 2026-07-27/28)

Launcher: `go_wc_v8_improvement_sweeps.ps1 -DeadlineHours 8`. Out: `outputs/biochem/offwall_model/wc_v8_improvement_sweeps/`.

| Arm | F1 | score | off_rel | 010 ge2 | 007 ge2 | target |
|-----|---:|------:|--------:|--------:|--------:|:------:|
| baseline frontier-h1 (locked) | 0.724 | **0.812** | **0.505** | 29/12 | 32/97 | no (wall guardrail) |
| frontier hops=0 | 0.780 | 0.802 | 0.439 | 0/12 | 9/97 | no (lumen strict) |
| **frontier hops=0.5** | **0.745** | **0.812** | 0.473 | 32/12 | 34/97 | **yes** |
| FP polish 010 (last-weights; wall-floor reject) | 0.693 | 0.805 | 0.426 | 41/12 | 45/97 | no |
| frontier-h1 retrain (fixed floor 0.12) | 0.721 | 0.799 | 0.446 | **15/12** | **51/97** | no |

**Headline:** same locked growth weights + **hops=0.5** (committed + off-wall 1-hop shell) recovers **+0.021 F1** vs h1 and **passes all compound gates** (including wall guardrail). 010 still over-fires; score stays ~0.812.

**Train arms:** short FP polish failed (compound F1 on 001 << A_floor−0.04; no valid ckpt). Retrain under frontier-h1 val cut 010 over-fire (15/12) and lifted 007 recall (51/97) but lost mean score/F1 vs locked. Budget expired before recall_007 / unfreeze.

**Pivot lesson:** eval-time hops=0.5 is the cheapest F1 recovery; retrain needs wall_floor_delta≥0.12 when val=patient001 (A_floor≈0.90). Next: promote/eval hops=0.5 as deploy default, or short polish under hops=0.5 compound-val.

**Target reframed (2026-07-27):** do **not** require zero hop_ge2 preds on idle vessels (002/004/008). Reward **closeness to GT** (clot or no-clot) with emphasis on **precision / clot score**, while still opening true lumen teachers (001/007) and holding mean F1 near wall. Idle vessels fail only if F1/relaxed precision collapse under false lumen paint.

**Compound eval gates:** `src/evaluation/compound_deploy_gates.py` (wired into `eval_mat_growth_simple.py` and `run_wall_lumen_target_9h.py`).

| Tier | Metrics |
|------|---------|
| Primary | `deploy_clot_score`, `deploy_clot_offwall_relaxed_f1` |
| Lumen | `deploy_clot_offwall_strict_f1_hop_ge2`, ge2 recall (sum pred / sum gt) |
| Spray (reframed) | idle 002/004/008: F1 >= 0.70 and relaxed prec >= 0.85 if hop_ge2 paint |
| Guardrail | mean `deploy_clot_f1` >= wall_floor - 0.04 |

### Canonical compound deploy (2026-07-27)

**Leg:** `WC_v8_compound_front_h1` — WC v8 compound: WC_v7 wall + frontier-h1 lumen specialist.

| Role | Artifact |
|------|----------|
| Wall backbone (unchanged) | `outputs/biochem/biochem_gnn/locked/species_gnn_best.pth` (`WC_v7_clot_phi_mse`) |
| Lumen growth | `outputs/biochem/biochem_gnn/locked/compound_growth_best.pth` |
| Manifest | `data/reference/mat_compound_deploy.json` |
| Env helper | `src/biochem_gnn/compound_deploy.py` |

Promote: `python scripts/promote_compound_deploy.py --growth-src outputs/.../growth_C/best.pth --eval-json ... --wall-floor-json ...`

**Note:** locked WC_v7 remains the strong **wall-only** model; compound stacks the growth specialist with `two-model-route=frontier`, `frontier_hops=1`, `WALL_ONLY=1`.

### WC_v7 vs growth_C + frontier_h1 (same WALL_ONLY eval)

| | Locked **WC_v7** (wall alone) | **growth_C** + frontier hops=1 |
|--|--|--|
| Stack | Single locked species GNN | Compound: WC_v7 wall + growth specialist |
| Routing | n/a | Specialist only in 1-hop of committed Mat |
| Mean clot F1 | **0.780** | 0.724 |
| Mean clot score | 0.802 | **0.812** |
| hop_ge2 strict | 0.097 | **0.181** |
| mean ge2 pred/gt | 5.9 / 18.5 | **13.1 / 18.5** |
| 001 ge2 | 50/68 | **61/68** |
| 007 ge2 | 9/97 | **32/97** |
| 002 ge2 | 0/0 | 0/0 |
| 010 ge2 | 0/12 | 29/12 (over-fire) |

**Physics / washout:** both still carry learned `log_vel_decay_mat` softplus **α ≈ 0.72**. Current deploy contract is `VEL_DECAY=1` + **`WALL_ONLY=1`**, so α washes Mat only on **wall** nodes — lumen is **not** speed-washed. Growth_C was trained and meta-tagged under that wall-only contract; locked WC_v7 meta predates the flag but eval/deploy recipes pin WALL_ONLY now.

**Architecture:** same SAGE dual-head wall backbone. Growth_C is a freeze-backbone specialist (Prec8h warm-start) trained with `frontier_ge2` + `loss_lumen_shape` (FP-heavy) under WALL_ONLY; deploy uses frontier route hops=1 (train used hops=2 supervision).

Viz: `scripts/go_wc_v7_growth_c_frontier_h1_compare_viz.ps1` → `outputs/biochem/offwall_model/wc_v7_wall_lumen_target_9h/viz_compare_wc_v7_vs_growth_c_front_h1/`.

## Run log — tile CC explore 2h (2026-07-23)

Launcher: `go_wc_v7_tile_cc_explore_2h.ps1 -Fresh` (5 ep / max-windows 12 / train 007+004+001).  
Probes completed via `-EvalOnly` after a PowerShell path bug (`Invoke-PythonRcCheck` exit `0` polluted `--offwall-ckpt`); fixed with `$null = Invoke-PythonRcCheck` + `return , $ckpt`.

Out: `outputs/biochem/offwall_model/wc_v7_tile_cc_explore_2h/` (`compare_tile_cc.json`).

**Train density:** UnionTile **36 tiles/ep**; PerComponent **258 tiles/ep** (~7.2× more local steps; max 8 CCs/window).

### Probe mean (007+004 compound wall-route)

| | A canon | UnionTile | PerComponent | d(CC−Union) |
|--|--------:|----------:|-------------:|------------:|
| clot_f1 | 0.756 | **0.756** | 0.674 | **−0.083** |
| clot_score | 0.792 | 0.796 | 0.784 | −0.012 |
| hop_ge2 n_pred | 0.0 | 4.5 | **33.5** | **+29.0** |
| hop_ge2 strict_f1 | 0.000 | 0.038 | **0.117** | **+0.079** |
| offwall strict_f1 | 0.000 | 0.037 | 0.095 | +0.058 |

Verdict: **`per_component_helps_lumen_hurts_wall`**.

### Per-anchor

| Anchor | metric | A | Union | CC |
|--------|--------|--:|-----:|---:|
| **007** | clot_f1 | 0.796 | **0.798** | 0.770 |
| 007 | hop_ge2 n / strict | 0 / 0 | 9 / **0.075** | **31 / 0.234** |
| **004** | clot_f1 | 0.715 | 0.715 | **0.577** |
| 004 | hop_ge2 n / strict | 0 / 0 | 0 / 0 | **36 / 0.000** |

### Readout

1. **Per-CC tiling does move lumen** — mean hop_ge2 volume and strict beat union on this short budget; on **p007** localization is real (strict 0.234).
2. **It also sprays** — **p004** gets 36 hop_ge2 preds with **strict 0** and clot F1 collapses (−0.14 vs A/Union). That drives the wall/clot regression.
3. **Union is the safer default for the 8h prec run** — holds clot F1 ≈ A while opening a small hop_ge2 signal (4.5 / 0.038) without 004 spray.
4. **Do not flip 8h to `per_component` alone.** If revisiting CC tiles: need wall floor in ckpt save + stronger FP / smaller `max-tiles`, or oversample vessels with clean hop_ge2 GT (007-like) and downweight spray anchors.

## Run log — Open001 1h (2026-07-24)

Launcher: `go_wc_v7_open001_1h.ps1 -Fresh` (4 ep / max-windows 16 / train **001+007+010** / freeze-backbone / `frontier_ge2` / `loss_lumen_shape` FN-heavy / `--ckpt-metric hop_ge2_recall` / cheap-val on **001**).

Out: `outputs/biochem/offwall_model/wc_v7_open001_1h/` (`compare_open001.json`).

**Budget note:** first pass hit the soft 1h cap after Probe A only and printed a false `null_architecture_suspect` (missing Prec8hRef / Open001). Completed probes via continued eval; **do not treat that first summary as evidence**.

**Train:** loss 5.136 → 5.128; every epoch `CkptScore` = −loss, `hop_ge2` / off-wall RelF1 all **0**. Best = ep4 (`growth_Open001/best.pth`).

### Probe mean (6 anchors: 001,007,010,006,004,008; wall-route compound)

| Arm | clot_f1 | hop_ge2 n | hop_ge2 strict |
|-----|--------:|----------:|---------------:|
| A (WC_v7) | **0.720** | 0.0 | 0.000 |
| Prec8hRef (8h Arm S) | 0.700 | 7.2 | **0.104** |
| Open001 | 0.665 | **12.8** | 0.083 |

(`RecallPush` row in `compare_open001.json` is a copy of Open001 — that arm was not trained this run.)

### Per-anchor hop_ge2 n_pred (strict)

| Anchor | GT | Prec8hRef | Open001 |
|--------|---:|----------:|--------:|
| **001** | 68 | **0** (0.000) | **0** (0.000) |
| **007** | 97 | 17 (0.140) | **27** (0.097) |
| 010 | 12 | 2 (0.286) | 2 (0.286) |
| 006 | 6 | 14 (0.200) | **29** (0.114) |
| 004 | 0 | 1 | **7** |
| 008 | 0 | 9 | **12** |

Gates: `opened_001=False`, `recall_up_007=True`, `spray_004_or_008=True` → **`partial_capacity_mixed_spray`**.

### Readout

1. **Primary gate failed** — 001 hop_ge2 pred stays **0** despite being in the train set and the val anchor. Teacher oversample + recall tilt did **not** open 001.
2. **Capacity exists on 007** — Open001 raises 007 hop_ge2 17→27 vs Prec8hRef, but strict falls 0.140→0.097 (volume without better localization).
3. **Spray worsens** — 004/008/006 all gain false lumen volume; mean clot F1 drops vs A (−0.055) and vs Prec8hRef (−0.035); mean hop_ge2 strict also worse than Prec8hRef.
4. **Not a null architecture** — specialist can increase lumen volume; it just **does not transfer to 001** and the short freeze-backbone recipe trades precision for spray.
5. **Next levers (ranked):** (a) make train/val rollout actually emit hop_ge2 on 001 before trusting `hop_ge2_recall` selection (unfreeze backbone / longer unroll / 001-only tile oversample); (b) spray reject on 004/008 in ckpt; (c) do not promote Open001 over Prec8hRef or locked WC_v7.

### Crack script (wired)

`go_wc_v7_crack_001_3h.ps1 -Fresh` isolates causes on **patient001 alone** with real `compound-val` + `hop_ge2_recall`:

| Stage | Hypothesis | If 001 opens |
|-------|------------|--------------|
| `Solo001_Freeze` | 007 gradient competition | Mix recipe was wrong; re-add 007 carefully |
| `Solo001_Unfreeze` | Frozen WC_v7 feature lock | Need trainable backbone for lumen |
| `Solo001_CC` | Union undersamples compact lumen | Prefer `per_component` on 001 |
| all closed | Architecture / route / IC | Not a short tuning fix |

Summarizer: `scripts/summarize_crack_001.py` → `compare_crack_001.json`.

### Run log — crack_001 3h (2026-07-24/25, complete)

Launcher: `go_wc_v7_crack_001_3h.ps1 -Fresh` then `-Resume` after cancel mid Freeze probe.

Out: `outputs/biochem/offwall_model/wc_v7_crack_001_3h/` (`compare_crack_001.json`).

**Verdict: `still_closed_architecture_suspect`** — patient001 hop_ge2 pred = **0 under every Solo001 arm**.

### Probe mean (001/007/004/008)

| Arm | clot_f1 | hop_ge2 n | hop_ge2 strict |
|-----|--------:|----------:|---------------:|
| A | **0.676** | 0.0 | 0.000 |
| Prec8hRef | 0.662 | 6.8 | 0.035 |
| Solo001_Freeze | 0.664 | 3.5 | 0.005 |
| Solo001_Unfreeze | 0.670 | 3.2 | 0.018 |
| Solo001_CC | 0.537 | **67.5** | 0.067 |

### Per-anchor hop_ge2 n_pred (strict)

| Anchor | GT | A | Prec8h | Freeze | Unfreeze | CC |
|--------|---:|--:|-------:|-------:|---------:|---:|
| **001** | 68 | **0** | **0** | **0** | **0** | **0** |
| 007 | 97 | 0 | 17 (0.140) | 14 (0.018) | 13 (0.073) | **38 (0.267)** |
| 004 | 0 | 0 | 1 | 0 | 0 | **88** |
| 008 | 0 | 0 | 9 | 0 | 0 | **144** |

(Summarizer initially printed `spray=False` because it checked Freeze only when nothing opened; CC clearly sprays. Fixed in `summarize_crack_001.py`.)

### Hypothesis outcomes

| H | Result |
|---|--------|
| H1 competition | **Rejected** — solo-001 freeze still leaves 001 at 0 |
| H2 backbone lock | **Rejected** — unfreeze also leaves 001 at 0; loss barely moves |
| H3 tile density | **Rejected for 001 gate** — CC raises volume elsewhere + spray, still 001=0 |

### Key readout

1. **001 is closed even when it is the only teacher.** Not a 007-steal problem.
2. **Bizarre transfer:** models trained *only on 001* light lumen on **007** (and CC floods 004/008) but stay silent on 001 itself at deploy. Capacity exists; the deploy path for 001 specifically does not use it.
3. **Compound-val was inert:** every arm logged `compound_f1=0.000`, `A_floor=0.000`, `hop_ge2: 0/68` on val epochs; ckpt score stuck at 0.0 (ep2 saves). So `hop_ge2_recall` never saw a positive 001 lumen signal during selection — matches broken floor vs probe-A clot_f1 ~0.80 on 001.
4. **CC is dangerous without spray gates** — mean hop_ge2 volume jumps via 004/008 paint; clot F1 collapses (−0.14 vs A).
5. **Revised failure mode:** train↔deploy mismatch on 001 (wall-route compound / IC / teacher-forced tiles vs closed-loop wall state), not short freeze/unfreeze/CC knobs. Next: debug why compound-val clot_f1 is 0 on 001; compare train-tile preds vs deploy wall-route on the same 001 frame; try growth-alone (non-compound) deploy and/or frontier route / teacher-IC deploy ablations.

### Run log — crack_001 layer-2 quick (2026-07-25, band-static val fixed)

Launcher: `go_wc_v7_crack_001_3h.ps1 -Fresh -Epochs 2 -EarlyStop 2 -MaxWindows 12`

**A_floor now healthy** (`≈0.807`) — layer-1 compound-val bug is fixed.  
**Verdict still `still_closed_architecture_suspect`:** 001 hop_ge2 = **0** on Freeze / Unfreeze / CC.

| Arm | clot_f1 | ge2_n | 001 ge2 | 007 ge2 | 004 | 008 |
|-----|--------:|------:|--------:|--------:|----:|----:|
| A | 0.676 | 0.0 | 0 | 0 | 0 | 0 |
| Prec8hRef | 0.662 | 6.8 | 0 | 17 | 1 | 9 |
| Solo001_Freeze | 0.566 | 30.2 | **0** | **63** | **47** | 11 |
| Solo001_Unfreeze | 0.563 | 31.8 | **0** | **63** | **52** | 12 |
| Solo001_CC | 0.554 | 51.5 | **0** | 34 | **66** | **106** |

Compound-val every ep2: `compound_f1≈0.80` but still `hop_ge2: 0/68` — wall OK, lumen never lights on 001.

**Read:** solo-001 training with global tile feats still sprays 007/004/008 at deploy while 001 stays hard-zero. Matches train/deploy kin feature mismatch (001 lumen cos≈0.16 global vs band).

**Next wiring (done):** `--train-feat-source band` (default) trains tiles on the same wall-band features as eval. Re-run layer-2 quick after this.

### Run log — crack_001 Resume (2026-07-25 evening) — **invalid for band-feat test**

Launcher: `go_wc_v7_crack_001_3h.ps1 -Resume -Epochs 2 ...`

Completed Prec8hRef probe only; **skipped all Solo001 train/probe** because pre-band ckpts/probes already existed (`best.json` has no `train_feat_source`). Numbers match the earlier global-feat layer-2 table exactly → verdict `still_closed_architecture_suspect` is **not** a band-feat result.

Launcher now retrains on Resume when `best.json.train_feat_source != band` (and drops the stale probe).

**True band-feat test:**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_wc_v7_crack_001_3h.ps1 -Resume -Epochs 2 -EarlyStop 2 -MaxWindows 12
```

(or `-Fresh` to wipe baselines too). Confirm train banner shows `train_feat_source: band`.

### Fast path (preferred for hard-zero bug hunting)

Do **not** re-run the full crack ladder to hunt a stiff 0. Use:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_wc_v7_001_signs_of_life.ps1
# or even faster:
powershell ... -File .\scripts\go_wc_v7_001_signs_of_life.ps1 -SkipDeploy
```

Mini-trains ~40 steps on one late 001 tile (band vs global), reports lumen Mat delta fire under teacher IC vs resting closed-loop, optional single-anchor deploy. Verdicts: `signs_of_life_band`, `life_on_teacher_dead_on_closed_loop`, `dead_everywhere_*`.

### Run log — 001 narrow ladder (2026-07-25 evening)

Scripts: `diagnose_001_narrow.py`, `diagnose_001_band_long.py`, band Solo001_Freeze retrain.

| Test | Result |
|------|--------|
| Band vs global cos(kin, lumen) | 001 **0.145**, 007 **0.198** (both severe; `kin_norm` on band only was the gap) |
| Signs-of-life band 40/300 steps | fire rises (~28→160+); **deploy hop_ge2 still 0** (compound / growth-alone / frontier) |
| Global late-tile | fire stays 0; loss flat |
| Band Solo001_Freeze 2ep/12win (`train_feat_source=band`) | **001 ge2=0**, 007 ge2=**1** (global-feat solo used to spray 007~63) |
| `out_scale`×50 late-tile | does **not** open hop_ge2 |
| Bug fixed | launcher FN tilt (`UNDERPRED=12`) was wiped by `apply_mat_growth_leg_env(force=True)` — now restored after leg apply |

**Verdict:** `tile_life_but_deploy_threshold_or_nucleation_gap`. Band feats are necessary (kill spray / unlock tile gradients) but **not sufficient** at 2ep to open 001 deploy lumen. Next: longer band Solo001 budget + instrument full-rollout Mat accumulation (fire ≠ commit).

### Run log — open001 6h autonomous ladder (2026-07-26)

Launcher: `go_wc_v7_open001_6h.ps1 -Fresh` (~6 h) + emergency `go_wc_v7_open001_prec8h_ft.ps1`.  
Out: `outputs/biochem/offwall_model/wc_v7_open001_6h/`.

| Arm | 001 ge2 | 007 ge2 | cohort ge2 | mean clot_f1 |
|-----|--------:|--------:|-----------:|-------------:|
| A/B/C Solo001 band | **0** | — | 0 | 0.80 |
| C2 Teachers 001+007+010 | **0** | — | 0 | 0.80 |
| **D_Orig10_Band** | **0** | **44** | **99**/185 | 0.71 |
| E Prec8h→001+007 FT | **0** | 9 | — | — |

**Verdict: `still_closed_001`.** Band + FN tilt + freeze/unfreeze/CC/teachers/orig10 all fail to light **patient001** hop_ge2, while the same D orig10 compound **does** write lumen elsewhere (007=44, cohort 99). Prec8h warm-start FT also keeps 001 hard-zero.

**Implication:** not a short-budget or freeze/union issue. 001 deploy lumen lock survives correct band features and multi-anchor capacity. Next: instrument full-deploy lumen Mat vs gelation log1p≈3e-4 on 001 (accumulation / vel-decay / state-conditioning), not more Solo001 ep knobs.

### Run log — crack_001 root diagnose (2026-07-25)

Script: `scripts/diagnose_crack_001_root.py`  
Report: `outputs/biochem/offwall_model/wc_v7_crack_001_3h/diagnose_root.json`

**Verdict: `compound_val_uses_wrong_static_full_graph_vs_wall_band`**

| Path | patient001 nodes | wall-only clot_f1 | notes |
|------|-----------------:|------------------:|-------|
| TRAIN-static (compound-val old) | 9490 (full) | **0.000** | matches inert A_floor |
| EVAL-static (`eval_mat_growth_simple`) | 2173 (wall-band) | **0.801** | real WC_v7 |

Same Solo001_Freeze growth on **correct** EVAL static:

| | 001 hop_ge2 | 007 hop_ge2 | clot_f1 |
|--|------------:|------------:|--------:|
| compound wall-route | **0**/68 | **14**/97 | 0.798 / 0.757 |
| growth-alone | 0 (clot_f1=0) | 0 (clot_f1=0) | nucleate fails alone |

**Root stack (two layers):**

1. **Bug (fixed):** `eval_wall_only_deploy_floor` / `eval_compound_wall_route_deploy` used full-graph `base_feats_global` + all-node `node_idx`. That zeros clot F1 on 001, so `hop_ge2_recall` never saw a usable 001 signal (A_floor=0, compound_f1=0 every epoch). Packs now store `band_static` via `build_band_base_features` (same as eval).
2. **Remaining 001 lock:** even with correct EVAL static, Solo001_Freeze compound still has **001 hop_ge2=0** while opening 007. Layer-2 quick (2 ep) confirmed spray-without-001. **Follow-up fix:** train tiles now default to `--train-feat-source band` (same `band_static` as eval) to kill global↔band kin mismatch.

## Related

- Stack design: [BIOCHEM_GNN.md](BIOCHEM_GNN.md)
- Naming: [MODEL_NOMENCLATURE.md](MODEL_NOMENCLATURE.md)
- Geometry-sensitivity research sweeps: [RESEARCH_SWEEPS.md](RESEARCH_SWEEPS.md)
- Historical leg tables and living TODO dump: [archive/MAT_GROWTH_SIM_TODO.md](archive/MAT_GROWTH_SIM_TODO.md)
- Historical baseline leaderboard: [archive/BIOCHEM_GNN_BASELINES.md](archive/BIOCHEM_GNN_BASELINES.md)
