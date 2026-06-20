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
At global relL2 17.6% (800 ep). Tail = low-signal patches (diagnosis above).
1. **Relative loss** (implemented, `--loss relative`) -- retrain and compare the bucket table;
   expect the low-shear/thin-clot terciles to improve. Watch for a small trade-off on the
   high-signal terciles (acceptable: those are already good and the metric we care about is
   uniform relL2). **Stability note:** the per-patch normalization must floor the denominator
   at a fraction of the *median* patch energy (`--rel-floor-frac`, default 0.5) + grad clip
   (`--grad-clip`, default 5.0); a naive `1e-12` floor lets near-zero-signal patches blow the
   loss to ~1e7 and collapse the model (val relL2 stuck ~100%, train loss oscillating).
2. **Decide what matters**: if only the big-diversion clots matter for biochem coupling, the
   current model is already ~15-16% there and the tail is a low-impact metric artifact -> a
   signal-weighted acceptance metric may be more honest than raw relL2.
3. **Capacity** -- hidden 64 is small; try 96/128 once the loss is settled.
4. **Hard data / oversampling** -- only if a *future* diagnosis shows the hard corner
   regressing (`--hard-bias` data gen, `--hard-boost` sampler). Not indicated now.

Target: global relL2 <~12% with p95 well under 100% before wiring into the biochem deploy
rollout (`BiochemDeployStack.set_local_corrector` / `local_corrector_ckpt`).
