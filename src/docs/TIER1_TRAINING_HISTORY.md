# Tier 1 Predictor (GINO-DEQ) Training History & Insights
*Last Updated: 2026-04-17*
*Target Goal: < 5% Relative L2 Error on Tier 1 Hemodynamics*

## 1. The 15% Pareto Frontier (The Current Bottleneck)
Initial sweeps (14+ candidates) using AdamW successfully descended but hit a hard "noise floor" at ~15.6% Relative L2 Error. AdamW is excellent for finding the general basin of the PDE landscape but struggles to minimize the stiff Navier-Stokes residuals beyond this point.

## 2. Architectural Findings (What Failed vs. What Works)
* **Kinematics Mode (CRITICAL):**
  * ❌ `stream`: Fails completely (~80-90% error). Deriving velocities requires 1st derivatives, and the NS momentum residual then requires 3rd-order spatial derivatives. Graph-WLS accumulates too much discretization noise at the 3rd order.
  * ✅ `direct_uvp`: Required. Bypasses higher-order noise. Reached ~24% error quickly.
* **Activations:**
  * ❌ `relu`: Stalls PINN training. Its 2nd derivative is zero, meaning PDE gradients vanish.
  * ✅ `silu`: Standardized across all runs. Smooth 2nd derivatives enable stable NS residuals.
* **NS Derivative Mode (`wls` vs `autograd`):**
  * `wls` acts as a smooth surrogate, allowing AdamW to drop loss fast, but suffers from unstructured mesh noise near walls.
  * `autograd` is exact but creates a highly stiff, non-convex landscape. It failed to converge better than `wls` using *only* AdamW, but is a prime candidate for L-BFGS refinement.
* **Fourier Encoding (Boundary Layers):**
  * ❌ Sparse encodings (`fourier_base=3.0`, `num_freqs=8`) aliased near the wall.
  * ✅ Dense encodings (`fourier_base=1.5`) showed immediate physical score improvements. Hemodynamics require high-frequency capacity to resolve extreme near-wall shear gradients.
* **Loss Weighting:**
  * `dynamic` loss weighting (learned uncertainty) performs best during the AdamW phase to balance continuity, momentum, and data losses dynamically.

## 3. Structural Deficiencies Addressed
To break the 15% barrier, the following structural changes to `src/architecture/ginodeq.py` are required moving forward:
1. **Pressure Propagation:** Replaced `global_mean_pool` with `GlobalAttention` to allow long-range elliptic Poisson pressure updates to flow from inlet to outlet without being washed out by the mean.
2. **Upwind Edge Stencils:** Upgraded `self.edge_proj` to a 2-layer MLP to allow the GNN to learn complex, non-linear directional stencils (upwinding) based on edge geometry.

## 4. Current Phase Strategy
* Expand `latent_dim` to 256.
* Increase `num_fourier_freqs` to 16 or 24.
* **Two-Stage Optimization:** 40 epochs of AdamW (to reach the 15% basin) -> 20 epochs of Full-Batch L-BFGS (to polish PDE residuals to < 5%).

## 5. Tier-1 Mesh Resolution Sweep (2026-04-16)

Two comparable Tier-1 explorer candidates were run with identical optimizer/loss settings and train/val split (`n_train=225`, `n_val=25`), differing primarily in dataset mesh resolution:

- `Res_Coarse_1.5` (`tier1_res_coarse`): `best_rel_l2=0.1807801053`, `best_phys_score=38.5780247591`, duration `35.84 min`.
- `Res_Medium_0.75` (`tier1_res_medium`): `best_rel_l2=0.1736430429`, `best_phys_score=46.0482775472`, duration `70.61 min`.

### Decision (Tier 1 default)

- **Optimal mesh resolution for Tier 1:** `tier1_res_medium` (mesh size factor `0.75`).
- Rationale: best validation accuracy among tested candidates (about `3.95%` lower `best_rel_l2` than coarse) with improved physics score.
- Trade-off: about `2x` candidate runtime versus coarse; accepted for default Tier-1 quality.

## 6. V2 sweep → V3 continuity strategy (2026-04-17)

**Goal:** Break the `<5%` relative L2 benchmark on Tier 1 (2D vessel) CFD.

**V2 baseline:** `Baseline_Legacy` plateau ~`19.2%` validation error; bottleneck attributed to BC truncation and weak global pressure–velocity coupling.

**V2_Hard_BCs:** Replaced soft BC penalties with SDF-masked exact constraints (`u_out = SDF × u_pred + u_inlet`). Wall `|u|` mean `0.0000` (was `~0.0520`), but validation error rose ~`21.6%` (“squeeze”: error shifts into the interior); mean `|∇·u|` rose `~0.42 → ~0.76` — rigid walls expose limited interior capacity / receptive field for the elliptic pressure field.

**V2_Attention_MultiGrid:** `global_mean_pool` → Perceiver-style cross-attention bottleneck; run preempted ~epoch 4, but faster convergence (e.g. ~`33%` error by epoch 2 vs baseline pace).

**Optimizer note:** LBFGS after AdamW warmup — prior best **~15.4%** relative L2; second-order polish on stiff NS residuals.

**V3 plan:** Pair `hard_bcs=True` with enough model capacity to restore low interior continuity: `latent_dim=256`, **40 AdamW → 20 LBFGS** curriculum. Four-way A/B: `V3_Baseline_Legacy` (control); `V3_Attention_MultiGrid` (global elliptic coupling); `V3_Geometric_Priors` (sphere-traced width `D(x)` and derivatives as geometry shortcuts); `V3_SIREN_Implicit` (SIREN decoder + analytic spatial grads via autograd vs WLS mesh noise).

**V3 success criteria:** Keep wall `|u|` at `0.0000` while driving interior mean `|∇·u|` back **below ~0.10**.
