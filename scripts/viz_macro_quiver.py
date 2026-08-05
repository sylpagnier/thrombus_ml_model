"""Visualize flow diversion using quiver (arrows) to prove no-slip enforcement.

1. Tests SDF wallification on a synthetic clot in patient007.
2. Compares DEQ native prediction to COMSOL GT on the heavily stenosed patient041.
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.core_physics.t0_device import require_cuda_device
from src.config import PhysicsConfig, NodeFeat
from src.inference.corrector_coupling import ClotAwareFlow
from src.utils.paths import data_root

def create_quiver_plot(ax, pos_x, pos_y, u, v, title, clot_mask=None, skip=3):
    # Calculate speed for background coloring
    speed = np.sqrt(u**2 + v**2)
    
    # Plot speed as a scatter background
    sc = ax.scatter(pos_x, pos_y, c=speed, cmap='viridis', s=2, alpha=0.5)
    plt.colorbar(sc, ax=ax, label="Speed (ND)")
    
    # Subsample for quiver so it's not too dense
    idx = np.arange(0, len(pos_x), skip)
    
    # Plot arrows
    ax.quiver(pos_x[idx], pos_y[idx], u[idx], v[idx], 
              color='white', scale=25, width=0.002, headwidth=4)
              
    # Overlay clot boundary if applicable
    if clot_mask is not None:
        ax.scatter(pos_x[clot_mask], pos_y[clot_mask], color='red', s=3, alpha=0.8, label='Dynamic Clot (SDF=0)')
        ax.legend()
        
    ax.set_title(title)
    ax.set_xlabel("X (ND)")
    ax.set_ylabel("Y (ND)")
    ax.set_aspect('equal', adjustable='box')


def main():
    device = require_cuda_device()
    phys_cfg = PhysicsConfig(phase="kinematics")
    
    # Force low threshold for kine resolve so we trigger it easily
    os.environ["BIOCHEM_KINE_RESOLVE_ON_CLOT"] = "1"
    os.environ["BIOCHEM_KINE_RESOLVE_MIN_CLOT_NODES"] = "100"
    
    flow = ClotAwareFlow(device, phys_cfg=phys_cfg)
    out_dir = Path(r"C:\Users\pgssy\.gemini\antigravity\brain\d0524dc0-3837-4442-ab5f-5da07d6faaeb")
    
    # =========================================================================
    # PART 1: Synthetic Dynamic Clot on Patient 007
    # =========================================================================
    print("Loading patient007 for dynamic clot test...")
    graph_path7 = data_root() / "processed" / "graphs_biochem_anchors" / "patient007.pt"
    data7 = torch.load(graph_path7, map_location=device, weights_only=False)
    n7 = data7.num_nodes
    pos7 = data7.x[:, 0:2].cpu().numpy()
    
    # Baseline
    mu_eff_base7 = torch.full((n7,), float(phys_cfg.mu_inf), device=device)
    state_base7 = flow.update(data7, mu_eff_base7)
    u_base7 = state_base7.u.cpu().numpy()
    v_base7 = state_base7.v.cpu().numpy()
    
    # Synthetic Clot
    x_mid = (pos7[:, 0].max() + pos7[:, 0].min()) / 2
    x_span = pos7[:, 0].max() - pos7[:, 0].min()
    y_mid = (pos7[:, 1].max() + pos7[:, 1].min()) / 2
    
    clot_mask7 = (
        (pos7[:, 0] > x_mid - 0.15 * x_span) & 
        (pos7[:, 0] < x_mid + 0.15 * x_span) & 
        (pos7[:, 1] < y_mid) &
        ~data7.mask_wall.cpu().numpy()
    )
    
    mu_eff_large7 = mu_eff_base7.clone()
    mu_eff_large7[torch.from_numpy(clot_mask7).to(device)] = 2.0
    
    # Macro-Resolved
    state_clot7 = flow.update(data7, mu_eff_large7)
    u_clot7 = state_clot7.u.cpu().numpy()
    v_clot7 = state_clot7.v.cpu().numpy()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    create_quiver_plot(axes[0], pos7[:, 0], pos7[:, 1], u_base7, v_base7, "Patient 007: Baseline Flow (Unadjusted)", skip=5)
    create_quiver_plot(axes[1], pos7[:, 0], pos7[:, 1], u_clot7, v_clot7, "Patient 007: Macro-Resolved Flow (Dynamic Clot)", clot_mask7, skip=5)
    plt.tight_layout()
    out_path7 = out_dir / "quiver_patient007_dynamic_clot.png"
    plt.savefig(out_path7, dpi=150)
    print(f"Saved {out_path7}")
    
    # =========================================================================
    # PART 2: Real Stenosis GT vs DEQ on Patient 041
    # =========================================================================
    print("\nLoading patient041 for GT vs DEQ stenosis test...")
    graph_path41 = data_root() / "processed" / "graphs_biochem_anchors" / "patient041.pt"
    data41 = torch.load(graph_path41, map_location=device, weights_only=False)
    n41 = data41.num_nodes
    pos41 = data41.x[:, 0:2].cpu().numpy()
    
    # GT COMSOL Flow (from t=200, or any t since kinematics are steady-state)
    # y shape is [201, N, 16]. Channels 0,1 are u,v
    u_gt41 = data41.y[-1, :, 0].cpu().numpy()
    v_gt41 = data41.y[-1, :, 1].cpu().numpy()
    
    # RGP-DEQ Predicted Flow (native prediction on the stenosed mesh)
    flow.invalidate_base_cache()
    mu_eff_base41 = torch.full((n41,), float(phys_cfg.mu_inf), device=device)
    state_base41 = flow.update(data41, mu_eff_base41)
    u_deq41 = state_base41.u.cpu().numpy()
    v_deq41 = state_base41.v.cpu().numpy()
    
    fig2, axes2 = plt.subplots(2, 1, figsize=(14, 10))
    create_quiver_plot(axes2[0], pos41[:, 0], pos41[:, 1], u_gt41, v_gt41, "Patient 041: Ground Truth COMSOL Flow (Real Stenosis)", skip=5)
    create_quiver_plot(axes2[1], pos41[:, 0], pos41[:, 1], u_deq41, v_deq41, "Patient 041: RGP-DEQ Predicted Flow (Native Handling)", skip=5)
    plt.tight_layout()
    out_path41 = out_dir / "quiver_patient041_gt_vs_deq.png"
    plt.savefig(out_path41, dpi=150)
    print(f"Saved {out_path41}")

if __name__ == "__main__":
    sys.exit(main())
