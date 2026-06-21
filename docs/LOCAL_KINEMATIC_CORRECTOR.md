# Local kinematic corrector (clot velocity diversion)

A local, k-hop GNN that predicts the velocity diversion `[dU, dV]` a micro-clot induces,
as a **residual on the frozen GINO-DEQ base flow**. Instead of re-solving the global flow
when a clot appears, we patch the base field locally around the clot nodes (cheap,
anisotropic). Trained on synthetic COMSOL "Patch Factory" residuals.

## Why
The deploy GINO-DEQ kine model is accurate on healthy hemodynamics but OOD on the extreme
`mu` spikes a clot imposes. Rather than retrain the global model, we learn a *local*
correction so flow reroutes over/around the clot, supplying the shear/stagnation structure
the biochem clot model needs.

## Data: Patch Factory (COMSOL `mph`)
- Generator: `src/data_gen/lib/patch_factory_comsol.py` (no Gmsh; mapped structured quad grid).
- One master template `local_kine_template.mph`: a flat 2000um x ~350um box, parametric
  continuous-viscosity clot (`Clot_Mask` Heaviside, high-viscosity porous zone -- never a
  hole), inlet linear shear `u = shear_rate*y`, no-slip bottom, prescribed freestream top
  (exact Couette so the analytical baseline is clean).
- Baseline subtracted analytically -> residual `dU = U - shear_rate*y`, `dV = V`.
- QC: `src/data_gen/lib/patch_factory_qc.py` (baseline purity, BCs, mass, SNR, clot slowdown)
  + default mesh-convergence check (re-solve at refined mapped mesh; `convergence_report.json`).
- Current cohort: 1000 patches, all passing QC; convergence rel L2 ~ 1e-14 (mesh-independent).

## Architecture: `LocalKinematicCorrector`
- `src/core_physics/coupled_shear_gnn.py`. 3x `GATv2Conv` (heads=4, hidden=64, concat=False)
  + 2-layer MLP readout. Attention is used so the model can learn the anisotropic diversion
  (flow reroutes over/around a clot far more than it reverses behind it).
- Readout init near-identity (gain 0.01) so an untrained model leaves the base flow intact.
- Input features (`in_channels=6`): `[dx, dy, dist_to_wall, u0, v0, delta_mu]`, where `dx,dy`
  are clot-COM-centered (translation invariant). Assembled by the single source of truth
  `assemble_local_corrector_features` (shared by train / live verify / deploy).
- ND convention: positions by length scale (`d_bar` on patient graphs, channel height `H` on
  patches), velocity by `PhysicsConfig.get_u_ref(H)`, viscosity by `mu_viscosity_nd_scale`.

## Stage-A lessons ported (2026-06-20)
From the Stage-A kinematics curriculum ([docs/KINEMATICS_BEST_ARCHITECTURE.md](KINEMATICS_BEST_ARCHITECTURE.md)):
- **Difficulty-weighted oversampling** (clinical-anchor-boost analog): train sampler weight
  `1 + hard_boost*difficulty`, where `difficulty = patch_difficulty(clot_mu, clot_w, clot_h)`
  in `[0,1]` (single source of truth in `patch_factory_comsol.py`). `--hard-boost` (default 3).
- **Curriculum ramp** (L0L1 -> L2-heavy analog): `--curriculum-frac F` ramps the boost 0->full
  over the first `F` of epochs (easy-first, then hard). Default 0 (constant boost).
- **Cosine LR** (Stage-A: scheduler helped, LBFGS hurt): on by default; `--no-cosine` to disable.
- **Stratified (per-bucket) eval**: `eval_local_corrector` prints energy-weighted relL2 by
  terciles of difficulty / clot_mu / clot_w / clot_h / occlusion / shear -> shows *where* the
  failure tail lives.
- **Harder data generation**: `patch_factory_comsol --hard-bias B` skews clot mu/width/height
  toward the hard end for a fresh cohort (over-sample the difficult corner). `B=0` = original.

## Train / eval / viz
- Train: `python -m src.training.train_local_kinematic_corrector --patch-dir data/processed/cfd_results_patch_factory --epochs 800 --batch-size 4 --stride 2 --device cuda --hard-boost 3 --curriculum-frac 0.3`
  - 5 GiB GPU: `batch-size 4 / stride 2` fits. Reports per-epoch val MSE_nd + val relL2 + lr.
  - Difficulty-weighted sampler + cosine LR on by default; `--hard-boost 0 --no-cosine` for the
    legacy uniform/constant-LR recipe.
  - Saves `outputs/kinematics/local_corrector/local_kinematic_corrector_best.pth` (+ `_last`);
    checkpoint meta records `sampling` (hard_boost / curriculum_frac / cosine).
- Eval (held-out vs COMSOL truth): `python -m src.tools.eval_local_corrector --patch-dir data/processed/cfd_results_patch_factory --corrector outputs/kinematics/local_corrector/local_kinematic_corrector_best.pth`
  - Global + per-sample relL2; truth/pred/error maps (best/median/worst) ->
    `outputs/reports/figures/kinematics/local_corrector_eval.png`.
- Live (overlay on GINO-DEQ, dummy clot on a patient graph): `python -m src.tools.verify_local_corrector_live --graph data/processed/graphs_biochem_anchors/patient007.pt --corrector .../local_kinematic_corrector_best.pth --num-hops 5 --clot-mu 3.0`
  - Panels: base | corrected | overlay (shared arrow scale) ->
    `outputs/reports/figures/kinematics/local_corrector_diversion.png`.

## Metric note
relL2 = `sqrt( sum||pred - truth||^2 / sum||truth||^2 )`, global over the split (energy
weighted). A per-sample mean would divide by the ~0 far-field norms; the global ratio is the
robust accuracy number. `max per-sample relL2 > 100%` means a patch where predicting the
correction is worse than predicting zero (a failure-tail flag).

## Run log
| date | config | data | val MSE_nd (best) | held-out relL2 (global / med / p90 / max) | note |
|------|--------|------|-------------------|-------------------------------------------|------|
| 2026-06-19 | 300 ep, bs4, stride2, hidden64, heads4, MSE | 1000 patches (900/100) | 3.00e-6 | 26.7% / 30.0% / 54.5% / 106.3% | First end-to-end. Healthy curve, still descending at 300 ep (undertrained). Heavy failure tail (max>100%) -> likely extreme clots (largest w/h, highest mu/shear). Live diversion smooth, no artifacts, max\|dUV_nd\|~0.31. |
| 2026-06-20 | 800 ep, bs4, stride2, hidden64, heads4, MSE | 1000 patches (900/100) | 1.30e-6 | 17.6% / 19.1% / 42.4% / 97.5% | Just more epochs (same arch/loss). Every metric improved: global 26.7->17.6%, median 30->19%, p90 54->42%, max 106->98% (tail no longer worse-than-zero). val relL2 still trending down at ep799 (noisy, val n=100) -> not fully plateaued; no overfit (val~train). Tail (p90/p95) now the bottleneck. |
| 2026-06-21 | 800 ep, bs4, stride2, hidden64, **relative loss** (floor 0.5*med, grad_clip 5), cosine, hard_boost 0 | 1000 patches (900/100) | 1.06e-6 | 15.9% global (med/p90/max pending eval) | First stable relative-loss run (after fixing the 1e-12 -> median-energy floor; the naive version had collapsed, val relL2~100%). **Beats the MSE 800-ep baseline on the headline too**: global relL2 17.6->15.9%, val MSE 1.30e-6->1.06e-6. Smooth monotonic descent, cosine annealed LR->0, plateaued cleanly (~16% from ep600). The relative loss was meant to help only the low-signal tail but improved the global energy-weighted metric as well -> the low-signal patches were dragging the whole model, not just their own bucket. **Next: run `eval_local_corrector` for the bucket table to confirm the low-shear/thin-clot terciles compressed.** |

## Tail diagnosis (2026-06-20, stratified eval on the 800-ep ckpt)
The failure tail is **the low-signal regime, not the hard clots.** Energy-weighted relL2 is
worst at the *low* end of every axis; `shear_rate` has the steepest gradient (28.4% low ->
15.1% high), then `clot_h` (25.1% -> 17.5%). The deployment-relevant big clots (high
mu/size/shear) are already the best-fit (~15-16%).

Mechanism: `get_u_ref(d_bar)` is **geometry-only** (no shear), so `du_nd = du_si/u_ref` scales
with shear / clot size. Plain MSE on `du_nd` is therefore magnitude-weighted -> it fits big
signals and neglects small ones -> high *relative* error on low-shear/small/thin clots.

Consequence: `--hard-boost` (oversample high mu/size) is the **wrong direction** here (those
are already best). The principled fix is the **relative loss** (`--loss relative`), which
normalizes each patch by its own target energy so all signal scales count equally.

## Where we are / next levers (priority)
At global relL2 **15.9%** (800 ep, relative loss) -- the current best, beating the MSE
baseline (17.6%). The relative loss is now the default recipe to beat.
0. **Confirm the bucket table** (`eval_local_corrector`) for the relative-loss ckpt -- verify
   the low-shear/thin-clot terciles compressed (the whole point); log it. **(pending)**
1. **Relative loss** (implemented, `--loss relative`) -- now validated; it improved *global*
   relL2 too, so low-signal patches were dragging the whole model. **Stability note:** the
   per-patch normalization must floor the denominator at a fraction of the *median* patch
   energy (`--rel-floor-frac`, default 0.5) + grad clip (`--grad-clip`, default 5.0); a naive
   `1e-12` floor blew the loss to ~1e7 and collapsed the model (val relL2 stuck ~100%).
   Try `--rel-floor-frac 0.25` (more aggressive) if the bucket table shows the low terciles
   still lagging.
2. **Capacity** -- hidden 64 is small; try 96/128 now that the loss is settled.
3. **Hard data / oversampling** -- only if a future diagnosis shows the hard corner
   regressing. Not indicated.
2. **Decide what matters**: if only the big-diversion clots matter for biochem coupling, the
   current model is already ~15-16% there and the tail is a low-impact metric artifact -> a
   signal-weighted acceptance metric may be more honest than raw relL2.
3. **Capacity** -- hidden 64 is small; try 96/128 once the loss is settled.
4. **Hard data / oversampling** -- only if a *future* diagnosis shows the hard corner
   regressing (`--hard-bias` data gen, `--hard-boost` sampler). Not indicated now.

Target: global relL2 <~12% with p95 well under 100% before wiring into the biochem deploy
rollout (`BiochemDeployStack.set_local_corrector` / `local_corrector_ckpt`).

## Deploy coupling (intercept the flow in the rollout loop)
`src/inference/corrector_coupling.py` is the single source of truth for dynamically bending
the frozen base flow around a growing clot before it is fed to the biochem model (Steps A-F):
- **A** base flow `[u0, v0]` from the frozen GINO-DEQ kine pass (cached per graph).
- **B** clot nodes = `delta_mu_si > BIOCHEM_CORRECTOR_MU_THRESH` (default 1e-3 Pa.s), where
  `delta_mu = mu_eff - mu_bulk_carreau` (clot elevation over the *clot-free Carreau bulk*, not
  over `mu_inf` -- see the 2026-06-20 confound fix; the bulk ref keeps Δμ~0 away from the clot,
  matching the corrector's training distribution).
- **C** `k_hop_subgraph` (`BIOCHEM_CORRECTOR_NUM_HOPS`, default 4) around the clot nodes.
- **D** `assemble_local_corrector_features` (same clot-COM-centered convention as train/verify).
- **E** `corrector(x_sub, sub_edge_index)` diversion patched onto the base flow on the subset.
- **F** coupled `[u, v]` published to a per-graph registry + optionally written into
  `data.y[:, :, 0:2]` so species/shear/nucleation consumers see the diverted field.

Entry points: `CorrectorCoupledFlow.couple(data, mu_eff_si)` (provider) and
`couple_flow_with_corrector(...)` (stateless). Enable with `BIOCHEM_CORRECTOR_COUPLING=1`;
the per-step coupled clot rollout (`clot_coupled_rollout.rollout_temporal_phi_coupled`) then
uses the cheap corrector diversion instead of a full DEQ re-solve. The species GraphSAGE
stagnation features (`SPECIES_STAGNATION_FEATS`) and vel-decay flow also read the coupled
registry when coupling is on.

### Fixing the GraphSAGE *primary* input: two-tier clot-aware flow (`ClotAwareFlow`)
The local corrector only patches `u, v`; it never regenerates the DEQ latent `z_kin =
predict_kinematics_latent(kine, data)` that is the GraphSAGE teacher's primary flow input.
`ClotAwareFlow.update(data, mu_eff_si)` escalates by clot burden:
- **frozen**  -- no clot -> frozen base flow + frozen latent.
- **corrector** -- small clot (`n_clot < BIOCHEM_KINE_RESOLVE_MIN_CLOT_NODES`, default 40) ->
  cheap local diversion on `u, v`; latent stays frozen.
- **resolved** -- significant clot (node count or `BIOCHEM_KINE_RESOLVE_MIN_BAND_FRAC`) -> the
  kine model **updates itself**: the clot `mu` is injected into `data.x[:, MU_PRIOR]` and the
  GINO-DEQ is re-solved, regenerating **both** the velocity field **and** `z_kin`. Hysteresis
  (`BIOCHEM_KINE_RESOLVE_GROWTH_FACTOR`, default 1.5x growth since last solve) avoids a global
  solve every step.

The re-solved `z_kin` is threaded into the GraphSAGE band features via
`build_band_base_features(..., z_kin_override=...)` /
`prepare_species_gnn_rollout_static(..., z_kin_override=...)`, so once the clot is big enough to
reroute flow the teacher's primary input tracks the rerouted field instead of the clot-free
latent. Knobs: `BIOCHEM_KINE_RESOLVE_ON_CLOT` (default = coupling state),
`BIOCHEM_KINE_RESOLVE_MIN_CLOT_NODES`, `BIOCHEM_KINE_RESOLVE_MIN_BAND_FRAC`,
`BIOCHEM_KINE_RESOLVE_GROWTH_FACTOR`.

**Phase 2 (system-level) test** -- does coupling fix `Mat` nucleation localization?
`python -m src.tools.compare_coupled_mat_rollout --graph data/processed/graphs_biochem_anchors/patient007.pt --species-ckpt <species best.pth>`
runs the species rollout uncoupled vs corrector-coupled and reports the spatial overlap
(Dice/F1) of the active `Mat` species vs the COMSOL ground truth -> a higher coupled F1 means
the diverted stagnation zone moved `Mat` to the correct downstream location.

### Coupling experiment log
**2026-06-20 -- Run #1 (INVALID, two confounds found & fixed).** First `compare_coupled_mat_rollout`
sweep vs the GraphSAGE `arch_ab/sage` teacher, t_last Mat Dice/F1 (baseline -> coupled):
p007 0.637->0.694 (+0.058), p007+stagnation 0.642->0.692 (+0.050), p006 0.366->0.458 (+0.091),
**p004 0.628->0.403 (-0.225)**, **p008 0.365->0.142 (-0.223)** -> 3 up / 2 (large) down, net inconclusive.
Two bugs made this **not** a test of the intended architecture:
1. **Clot mask flagged the whole mesh.** `delta_mu = mu_eff - mu_inf > 1e-3` tagged ~17,378/17,413
   nodes (the non-Newtonian bulk sits well above `mu_inf=0.0035`; Carreau `mu_0~0.056`). The
   corrector -- trained on *localized* clot patches where outside-clot Δμ≈0 -- was applied
   mesh-wide (OOD, `max|div|_nd~0.18`), which is the most likely cause of the p004/p008 collapses.
   Every run logged `clot_nodes<= <N_total>`.
   *Fix:* detect clot as `mu_eff - mu_bulk_carreau` (clot-free Carreau ref from the base flow).
   On p007 GT @ t_last this drops the mask from 17,378 -> **500 nodes** (~2.9%, the real gelation).
2. **The kine re-solve tier never fired.** The tool set `BIOCHEM_CORRECTOR_COUPLING=0` for the
   baseline pass, so during the coupled refresh `kine_resolve_enabled()` (default = coupling
   state) was False -> every run logged `final_mode=corrector kine_resolved=False (z_kin frozen)`.
   So the GraphSAGE *primary* input (`z_kin`) was never updated -- only the velocity-derived
   shear/vel-decay changed. *Fix:* enable coupling/kine-resolve **before** the refresh loop.
With both fixes the 500-node clot clears the 40-node burden gate, so the `resolved` tier engages
and `z_kin` is regenerated. **Re-run Run #2** with the same command to get a valid baseline-vs-
coupled comparison (watch for `final_mode=resolved kine_resolved=True`).

**2026-06-20 -- Run #2 (clot mask fixed, kine re-solve firing): still mixed/regressing.**
t_last Mat Dice/F1 (baseline -> coupled), `kine_resolved=True` on all: p004 0.628->0.495 (-0.133),
p006 0.366->0.174 (-0.192), p008 0.365->0.488 (+0.123), **p007 crashed (CUDA OOM in the DEQ
re-solve, 4 GiB)**. Clot masks were now sane (`clot_nodes<= 686/612/730`), but `max|div|_nd`
was **0.42-0.62** -- i.e. the diversion is as large as the *entire* freestream (~0.5). Diagnosis:
1. **Corrector is OOD on patient clots.** It was trained on *micro*-clot patches (small, μ 1.5-3
   Pa.s); late-rollout patient clots are large (600+ nodes) with μ up to ~4 Pa.s. It extrapolates
   unphysically large diversions that wreck the flow (p006 t100 -0.485). *Fixes:* clamp the Δμ
   feature (`BIOCHEM_CORRECTOR_MAX_DELTA_MU`, default 3.0); the spatial-extent OOD (huge subgraph)
   remains a fundamental scope limit -- the corrector is a *micro*-clot operator.
2. **OOM** on the big graph -> CPU fallback added to `ClotAwareFlow.resolve_full`.
3. **Fundamental: there is no clean channel for the corrector to help the GraphSAGE.** The species
   teacher's primary flow input is the *opaque DEQ latent* `z_kin`, and it was **trained on the
   frozen (clot-free) latent**. The corrector improves *raw velocity*, which the GraphSAGE does
   not consume. The only way to make `z_kin` clot-aware is to re-solve the DEQ with the clot μ --
   but the DEQ is *inaccurate on clots* (the very reason the corrector exists), so that latent
   carries the DEQ's clot error AND is a distribution the GraphSAGE never trained on -> mixed-sign,
   high-variance results (NOT a controlled improvement). Phase 1 already proves the corrector
   improves *flow fidelity* (17.6% relL2 vs COMSOL); the gap is purely *consumption*.

   **Real fix (requires training, not an inference swap):** fine-tune / retrain the species
   GraphSAGE to consume the corrector-improved flow in a representation it controls -- e.g. add
   coupled-`u,v`-derived stagnation/shear features (`SPECIES_STAGNATION_FEATS`) or feed coupled
   `u,v` directly -- so flow accuracy actually translates to Mat localization. Until then,
   inference-time flow swaps on a frozen model are distribution shift.

   **Decisive isolation experiment** (added): `compare_coupled_mat_rollout --oracle-mu` drives the
   diversion from the TRUE COMSOL clot μ (removes the predicted-μ feedback confound). If Mat still
   regresses under oracle μ, the bottleneck is the `z_kin` consumption/distribution-shift, not clot
   localization -> retraining is required.

**2026-06-20 -- Run #3 (oracle-μ + corrector-only ablation): regresses regardless -> consumption is
the blocker.** Two ablations, t_last Mat Dice/F1 (baseline -> coupled):
- **A) `--oracle-mu`, z_kin re-solved** (divert around the *true* clot): p004 0.628->0.420 (-0.208),
  p006 0.366->0.344 (-0.023), p007 0.637->0.588 (-0.049), p008 0.365->0.274 (-0.090). **4/4 regress.**
- **B) corrector velocity only, `BIOCHEM_KINE_RESOLVE_ON_CLOT=0`** (z_kin frozen): p004 -0.195,
  p006 +0.013, p007 -0.057, p008 -0.189. **3/4 regress** (p006 +0.013 is noise).

Interpretation -- the two ablations isolate the two suspects and **both** still regress:
- Oracle-μ rules out predicted-μ localization error: diverting around the *correct* clot still hurts.
- Frozen-z_kin rules out the DEQ-latent swap as the sole cause: changing *only* the velocity-derived
  features (vel-decay/shear) also hurts. So **any** inference-time flow change degrades this teacher.
- `max|div|_nd` was still **0.32-0.52** in every config -> the corrector, driven as ONE subgraph over
  the whole macro-clot (single COM, dx,dy spanning hundreds of nodes), is extrapolating far OOD.

**Fixes implemented (this run):**
1. **Apply the micro-clot corrector *locally*, not as one giant subgraph** (`BIOCHEM_CORRECTOR_LOCAL_CLUSTERS=1`,
   default on). `tile_clot_nodes` greedily partitions the clot into ND-radius balls
   (`BIOCHEM_CORRECTOR_CLUSTER_RADIUS_ND`, default 0.12) capped at `BIOCHEM_CORRECTOR_CLUSTER_MAX_NODES`
   (default 64); each patch gets its OWN local COM + small k-hop subgraph, and `couple_flow_with_corrector`
   accumulates the per-node diversions (averaged on overlap). This restores the training scale -- the
   direct answer to "apply at every clot node in a subgraph?": **yes, but as many small in-distribution
   patches, not one macro subgraph.**
2. **Clean GT-flow diagnostic** (`compare_coupled_mat_rollout --gt-flow`): feeds the TRUE COMSOL velocity
   (already in `data.y[:, :, 0:2]`), NO corrector, frozen z_kin. This is the gate the corrector cannot
   provide: does the GraphSAGE benefit from *accurate* flow at all? If `--gt-flow` ALSO regresses, the
   corrector is the wrong lever and the only path is retraining the teacher to consume coupled flow.

**Strategic read (micro -> macro adaptation).** There are two separable problems and they need
different fixes:
- *Corrector OOD (operator scale):* fixed by local tiling above (no retraining needed). Optionally
  retrain the corrector on patient-scale connected clots + the deploy μ range if tiling is insufficient.
-   *Consumption / distribution shift (the real blocker):* Run #3 (oracle-μ) shows even perfect flow
  hurts the frozen teacher. This is **not** fixable at inference -- the GraphSAGE must be **retrained
  with the coupled flow in the loop** (coupled-`u,v`-derived stagnation/shear features, and ideally the
  clot-aware latent), so flow accuracy maps to Mat localization. Run `--gt-flow` first to confirm the
  upside is real before paying for that retrain.

**2026-06-20 -- Run #4 (GT-flow gate + local tiling): the consumption blocker is now PROVEN structural.**
- **A) `--gt-flow`** (TRUE COMSOL velocity, NO corrector, frozen z_kin): p004 0.628->0.628 (+0.000),
  p006 0.366->0.373 (+0.006), p007 0.637->0.648 (+0.011), p008 0.365->0.365 (+0.000). **Feeding the
  *perfect* flow is a no-op (±0.01).**
- **B) `--oracle-mu` + local tiling** (z_kin re-solved): `max|div|_nd` dropped **0.32-0.52 -> 0.26-0.30**
  (tiling works -- diversions are now physical-scale), but Mat still regresses 3/4 (p004 -0.180,
  p007 -0.034, p008 -0.038; p006 +0.028) because the only channel that moves Mat here is the OOD
  clot-aware `z_kin`.

**Root cause (confirmed in code).** The species GraphSAGE node input is `build_snapshot_features` =
`[z_kin, sdf]` -- there is **no velocity feature in the model**. Flow enters the rollout only as a
learned `vel_decay` multiplier on the growth state (`band_speed_for_rollout`); swapping that to GT
COMSOL flow barely moves Mat. So Mat is ~entirely a function of the frozen, clot-blind `z_kin` + wall
distance + state carry. The corrector edits `u,v` -> a channel the model ignores; the only channel
that matters (`z_kin`) is reachable only via an OOD/inaccurate DEQ re-solve. **Inference-time coupling
on this teacher cannot work; it is missing wiring, not mistuned.**

**Retrain plan (the actual fix).** Give the teacher a clot-aware flow channel it can learn from:
1. Add explicit flow features to `build_snapshot_features` -- speed `|u|`, shear-rate proxy, and a
   stagnation/divergence indicator -- computed from the **GT COMSOL velocity** (which is already
   clot-aware) during training. New input dim = `latent_dim + 1 + k` (breaks old ckpts -> new run id).
2. Train the species GraphSAGE with those features so flow->Mat is actually learned (currently the
   velocity signal had no training variance, so the model learned to ignore it).
3. At deploy, the corrector-coupled flow (now in-distribution thanks to local tiling) supplies the
   approximation of that clot-aware flow; keep `z_kin` frozen (do NOT feed the OOD re-solved latent).
4. Re-run `--gt-flow` on the retrained teacher: it should now show a real, positive delta. Only then
   does the corrector have ROI.

**2026-06-20 -- Retrain plumbing implemented.** Flow-aware teacher wiring is in place:
- `species_pushforward_gnn.py`: `flow_feats_enabled()` (`SPECIES_FLOW_FEATS=1`) appends a 5-ch
  clot-aware flow block `[log1p(speed), log1p(shear), tanh(div), x_n, y_n]` (`_flow_band_features`)
  to the band inputs. Velocity source via `SPECIES_FLOW_FEATS_SOURCE`: `gt` (COMSOL `data.y`,
  training), `kine`, or `auto` (kine + corrector-coupled override, deploy default); representative
  GT time `SPECIES_FLOW_FEATS_TIME` (-1=last). Model in_dim auto-derives from `base_feats.shape[1]`
  (257 -> 262 confirmed on p007), so no manual dim wiring.
- Persisted: trainer writes `flow_feats` into meta; `load_species_gnn_rollout_bundle` re-enables
  `SPECIES_FLOW_FEATS` at deploy (source left `auto`, NOT the training `gt`).
- Launcher: `scripts/go_species_flow_aware.ps1` (canonical arch_ab sage recipe + flow feats, FRESH
  since input dim changed) -> `outputs/biochem/biochem_gnn/flow_aware/sage/species/best.pth`.
- Tests: `src/tests/test_species_flow_feats.py` (shape, gt/kine source, bounded divergence).
**Next:** run the launcher, then re-gate with `compare_coupled_mat_rollout --gt-flow` on the new
ckpt; a positive delta unlocks the corrector path (keep z_kin frozen, source `auto`).

**2026-06-21 -- Run #5 (flow-aware teacher trained; gt-flow gate fixed): teacher +0.08, corrector
upside only +0.008.** 75-ep flow-aware sage (`outputs/biochem/biochem_gnn/flow_aware/sage/species/
best.pth`, `best_score=0.752`; input 257->262 confirms the 5 flow channels). p007 t200 Mat F1:
- Old teacher `[z_kin,sdf]`: **0.637**.
- New teacher, **kine** flow features (deploy baseline): **0.717 (+0.080)**.
- New teacher, **GT** flow features (corrector upper bound): **0.725 (+0.008 over its own baseline)**.

**gt-flow diagnostic bug (fixed).** The first re-gate read +0.002 (no-op) because `--gt-flow` set
only `SPECIES_ROLLOUT_VEL_SOURCE=gt` (vel-decay) and NOT `SPECIES_FLOW_FEATS_SOURCE=gt`; with
coupling off the flow features fell back to `auto`->kine in BOTH passes. Fixed
`compare_coupled_mat_rollout` to export `SPECIES_FLOW_FEATS_SOURCE=gt` in the gt-flow branch (and
clear it after). Re-gate then showed the true +0.008.

**Read.** The +0.080 is a real win but comes from the *richer feature set* (speed/shear/divergence/
geometry), NOT from clot-awareness: accurate (GT) flow beats clot-blind (kine) flow by only +0.008,
and that is the corrector's ceiling. Two structural reasons: (1) the flow features are **static**
(one representative time) so they cannot represent the corrector's *dynamic* diversion as the clot
grows; (2) stagnation localization is largely **redundant with geometry** (`z_kin`/SDF) the model
already encodes. **Actions:** (a) promote the flow-aware teacher as the new baseline regardless of
the corrector; (b) if the corrector must matter, make the flow features **dynamic** -- recompute the
speed/shear/divergence channels each rollout step from the current corrector-coupled flow -- since
the static upper bound is already only +0.008, expectations should be modest.
