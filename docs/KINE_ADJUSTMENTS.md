# Kinematic Adjustments

This document outlines the hybrid architecture for capturing accurate hemodynamics as a vessel transitions from an empty state to near occlusion.

## Overview
Historically, kinematic coupling relied entirely on the **Local Kinematic Corrector**, which proved insufficient and out-of-distribution (OOD) for massive, late-stage clots. Concurrently, attempting to force the global **RGP-DEQ** solver to handle these large clots by injecting extreme viscosity spikes into `MU_PRIOR` caused the latent space (`z_kin`) to collapse or the GPU to run out of memory.

To resolve this, we adopt a **Geometry-Based Hybrid Architecture**:
1. **Macro Adjustments**: The RGP-DEQ solver handles large-scale flow rerouting by treating the clot as a solid wall (SDF update) rather than a viscosity anomaly.
2. **Micro Adjustments**: The Local Kinematic Corrector fills in the gaps between macro resolves.

## 1. Macro-Resolves (RGP-DEQ via SDF)
When a clot grows enough to significantly alter the global flow field (e.g., occluding a new threshold percentage of the lumen or hopping deeper into the channel), we trigger a macro-resolve.

Instead of changing the viscosity parameter `MU_PRIOR`, we recalculate the **Signed Distance Field (SDF)** of the vessel, effectively treating the gelled clot boundary as the new rigid vessel wall (`SDF = 0`). 

**Why this works**: The RGP-DEQ model was trained on healthy fluid across variously shaped vessels. By feeding it a narrower vessel via the updated SDF, the solver remains perfectly in-distribution. It outputs an accurate, updated base flow (`u0, v0`) and a clean updated latent (`z_kin`) reflecting the stenosed geometry.

## 2. Micro-Resolves (Local Kinematic Corrector)
Because a full RGP-DEQ solve is computationally heavy, we do not run it at every timestep. In the intervening rollout steps between macro-resolves, the **Local Kinematic Corrector** applies localized, micro-scale flow diversions (`u, v`) around the newly nucleating clot nodes.

## 3. Species Model Consumption
The species GraphSAGE teacher model must dynamically consume the updated `z_kin` from the macro-resolves. As the vessel narrows, the updated `z_kin` informs the species model of the shifting geometric boundaries, allowing it to accurately localize the next phase of clot growth based on the constricted flow.

## Training Implications
Because this architecture dynamically adjusts the SDF to keep the solver in-distribution, **no retraining of the RGP-DEQ kinematics model is required**. The solver naturally handles the updated `SDF = 0` geometries because it was already trained on a diverse set of vessel shapes, including stenoses.

Similarly, the **Local Kinematic Corrector** does not need retraining, as it continues to see its standard local graph patches and small delta-velocities.

If the species model (GraphSAGE teacher) requires fine-tuning on the new $z_{kin}$ dynamics, you can promote this pipeline using the standard deploy tools. However, current probes indicate the $z_{kin}$ magnitude ratio remains near 1.0 (no distribution shift), suggesting the species model will seamlessly consume the updated latent without structural retraining.
