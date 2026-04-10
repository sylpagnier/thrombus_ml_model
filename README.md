Project PIRON: The Chemo-Hemodynamic Surrogate

Multi-Fidelity rGINO-DEQ with Gradient-Aware Physics Kernels for Non-Newtonian Thrombosis
🚀 Overview

Project PIRON aims to predict coupled non-Newtonian flow fields and steady-state thrombus boundaries (ϕfinal​) in under 5 seconds. By utilizing a Graph-Independent Neural Operator (GINO) integrated with a Deep Equilibrium (DEQ) solver, we bypass traditional CFD bottlenecks while maintaining mesh-agnosticism for patient-specific geometries.
🧠 Architecture: rGINO-DEQ + LoRA

The system solves for the mutual consistency between fluid dynamics and biochemical clot growth:

    Encoder: Physics-informed features including SDF and Shear-Rate Potential.

    Solver Core: A DEQ layer finding the fixed point Z∗=GINO(Z∗,Xgeo​;θ).

    Adaptation: Low-Rank Adaptation (LoRA) for fine-tuning on N=17 patient-specific datasets.

🧪 Physics Framework
Non-Newtonian Rheology

We model blood viscosity using the Carreau-Yasuda parameters, where μ is sensitive to the local concentration of Fibrinogen (Fib) and Platelets (Plt).
Loss Landscape

The model is trained via a multi-fidelity curriculum:

    Tier 1 (LoFi): Newtonian + Laminar (Laminar mapping).

    Tier 2 (MidFi): Carreau-Yasuda Rheology (μ(γ˙​) sensitivity).

    Tier 3 (HiFi): 9-species biochemical triggers via Soft-Logic and LoRA.