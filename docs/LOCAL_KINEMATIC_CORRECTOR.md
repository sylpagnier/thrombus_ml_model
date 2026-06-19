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

## Train / eval / viz
- Train: `python -m src.training.train_local_kinematic_corrector --patch-dir data/processed/cfd_results_patch_factory --epochs 600 --batch-size 4 --stride 2 --device cuda`
  - 5 GiB GPU: `batch-size 4 / stride 2` fits. Reports per-epoch val MSE_nd + val relL2.
  - Saves `outputs/kinematics/local_corrector/local_kinematic_corrector_best.pth` (+ `_last`).
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

## Where we are / next levers (priority)
1. **Train longer** -- val was still descending at 300 ep; run 600-1000 ep.
2. **Characterize the tail** -- inspect eval worst panel; add per-bucket relL2 by
   `clot_w / clot_h / clot_mu / shear_rate` to locate where error concentrates.
3. **Loss for the tail** -- plain MSE is energy-weighted; try a normalized / relative loss
   so big-diversion patches don't dominate and small ones aren't ignored.
4. **Capacity** -- hidden 64 is small; try 96/128 once data/loss are set.
5. **More/!balanced data** -- 1000 patches may under-cover the extreme-clot corner; consider
   oversampling wide/tall/high-mu clots.

Target: global relL2 <~12% with p95 well under 100% before wiring into the biochem deploy
rollout (`BiochemDeployStack.set_local_corrector` / `local_corrector_ckpt`).
