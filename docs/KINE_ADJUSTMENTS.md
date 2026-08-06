# Kinematic Adjustments

This document outlines the hybrid architecture for capturing accurate hemodynamics as a vessel transitions from an empty state to near occlusion.

## Overview
The architecture relies on a hybrid macro/micro approach for dynamic flow coupling. The global RGP-DEQ solver handles large-scale macro-resolves by updating the Signed Distance Field (SDF), while the Local Kinematic Corrector fills in micro-scale perturbations between these resolves.

## GT Input Leakage Policy
- Phase 3 clinical anchor fine-tuning MUST use `prior_mode="analytic"` (never `"gt_flow"`).
- The old `apply_gt_flow_priors_to_kine_x()` function is DEPRECATED.
- **Why**: The residual hard-BC formulation `u_pred = u_prior + (1-exp(-λ·SDF))·Δu` makes GT priors trivially learnable (Δu ≈ 0), destroying generalization.
- Prior channels (11-14) must always contain analytical Poiseuille/Carreau formulas from geometry.

## Direct Shear Rate Prediction
- The GINODEQ model now has a `shear_decoder` head (channel 5) predicting per-node `log1p(shear_rate)`.
- This replaces the noisy WLS finite-difference approach for downstream consumers.
- Supervised by deriving GT shear rate from GT velocity via WLS operators during training.
- Downstream consumers (`clot_kinematics_fields`, `species_pushforward_gnn`) should prefer the direct prediction.

## Smooth SDF Fast-Marching
- When macro-resolves are triggered, SDF is smoothly recomputed via graph-based fast-marching.
- This replaces binary `SDF=0` masking, giving the solver the smooth gradients it expects.
- The fast-march computes Euclidean shortest paths from the combined wall+clot boundary.

## Macro-Resolves
- **Trigger policy**: When clot nodes exceed threshold or hop count changes.
- **Process**: fast-march SDF → RGP-DEQ solve → zero-velocity enforcement on clot nodes.

## Micro-Resolves
- Local Kinematic Corrector for small perturbations between macro-resolves.

## Training
- RGP-DEQ trains on analytical priors only (no GT leak).
- L2 stenosed synthetics (up to 80% occlusion) are already in the training curriculum.
- Shear decoder supervised by WLS-derived GT shear rate.
- Local corrector may need retraining on post-adjustment flow.

## Metrics
- Local 3-hop Rel L2 error around clots (target <15%).
- Shear rate Rel L2 (target <20%).
- Shear correlation, wall-specific and lumen-specific shear metrics.
