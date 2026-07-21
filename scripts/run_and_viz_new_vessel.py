import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to sys.path
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.core_physics.t0_device import require_cuda_device
from src.inference.customer_pipeline import CustomerDeployPipeline
from src.data_gen.lib.customer_geometry_import import load_customer_geometry

def main():
    device = require_cuda_device()
    print(f"[i] Using device: {device}")
    
    # Checkpoint for Leg 2 (clot_phi_mse)
    leg = "WC_v7_clot_phi_mse"
    wall_ckpt = REPO / f"outputs/biochem/biochem_gnn/{leg}/species/best.pth"
    print(f"[i] Using species GNN checkpoint: {wall_ckpt}")
    
    # Input geometry
    geom_path = REPO / "customer_geometries/vessel_0_demo.pt"
    if not geom_path.exists():
        print(f"[ERR] Demo vessel not found: {geom_path}")
        return 1
        
    print(f"[i] Loading geometry: {geom_path.name}")
    # Load geometry
    re_target = 100.0
    t_final_s = 2.0 * 3600.0  # 2 hours
    n_steps = 40
    
    data = load_customer_geometry(
        geom_path, re_target=re_target, t_final_s=t_final_s, n_steps=n_steps
    )
    
    # Initialize pipeline
    print("[i] Initializing deploy pipeline...")
    pipeline = CustomerDeployPipeline(
        device=device,
        wall_ckpt=wall_ckpt,
        mat_leg=leg,
        require_cuda=True
    )
    
    # Run prediction
    print("[i] Running closed-loop rollout...")
    traj = pipeline.run(
        data,
        t_final_s=t_final_s,
        include_velocity=True
    )
    print(f"[OK] Rollout finished. Steps: {traj.n_steps}, Elapsed time: {traj.elapsed_s:.1f}s")
    
    # Plot results at 4 representative time steps (0%, 33%, 66%, 100%)
    steps_to_plot = [0, traj.n_steps // 3, 2 * traj.n_steps // 3, traj.n_steps - 1]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"HemoRGP Forecast on New Vessel (Leg: {leg})", fontsize=14, fontweight="bold")
    
    pos = traj.pos
    
    for i, step in enumerate(steps_to_plot):
        t_val = traj.t_sec[step] / 60.0  # minutes
        phi = traj.phi[step]
        vel = traj.vel_mag[step]
        
        # Plot Clot phi
        ax_phi = axes[0, i]
        sc_phi = ax_phi.scatter(
            pos[:, 0], pos[:, 1], c=phi, cmap="coolwarm", s=1.0, vmin=0.0, vmax=1.0
        )
        ax_phi.set_title(f"Step {step} (t={t_val:.1f} min)\nClot Predict", fontsize=10)
        ax_phi.axis("equal")
        ax_phi.axis("off")
        if i == 3:
            fig.colorbar(sc_phi, ax=ax_phi, label="Clot Probability (phi)")
            
        # Plot Velocity Magnitude
        ax_vel = axes[1, i]
        sc_vel = ax_vel.scatter(
            pos[:, 0], pos[:, 1], c=vel, cmap="viridis", s=1.0
        )
        ax_vel.set_title(f"Velocity field", fontsize=10)
        ax_vel.axis("equal")
        ax_vel.axis("off")
        if i == 3:
            fig.colorbar(sc_vel, ax=ax_vel, label="Velocity magnitude (ND)")
            
    out_png = REPO / f"outputs/biochem/viz/species_gnn_deploy/deploy_{leg}_new_vessel_viz.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved visualization: {out_png}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
