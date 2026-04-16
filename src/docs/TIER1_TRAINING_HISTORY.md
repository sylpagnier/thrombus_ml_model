# Tier 1 Predictor (GINO-DEQ) Training History & Insights
*Last Updated: April 2026*
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
