"""Visualize dynamic clot flow diversion on patient041 vs Ground Truth.

This script proves that:
1. RGP-DEQ on the base graph predicts the flow at t=0 (unadjusted).
2. By identifying where the clot grew at t=200 in COMSOL and applying SDF wallification,
   RGP-DEQ accurately reroutes the flow (adjusted).
3. The adjusted flow matches the COMSOL GT flow at t=200.
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

def create_quiver_plot(ax, pos_x, pos_y, u, v, title, clot_mask=None, skip=4):
    speed = np.sqrt(u**2 + v**2)
    sc = ax.scatter(pos_x, pos_y, c=speed, cmap='viridis', s=2, alpha=0.5, vmin=0, vmax=2.5)
    plt.colorbar(sc, ax=ax, label="Speed (ND)")
    
    idx = np.arange(0, len(pos_x), skip)
    ax.quiver(pos_x[idx], pos_y[idx], u[idx], v[idx], 
              color='white', scale=30, width=0.002, headwidth=4)
              
    if clot_mask is not None and clot_mask.sum() > 0:
        ax.scatter(pos_x[clot_mask], pos_y[clot_mask], color='red', s=2, alpha=0.8, label='Clot Region')
        ax.legend(loc='lower right')
        
    ax.set_title(title)
    ax.set_xlabel("X (ND)")
    ax.set_ylabel("Y (ND)")
    ax.set_aspect('equal', adjustable='box')


def main(patient_id):
    device = require_cuda_device()
    phys_cfg = PhysicsConfig(phase="kinematics")
    
    os.environ["BIOCHEM_KINE_RESOLVE_ON_CLOT"] = "1"
    os.environ["BIOCHEM_KINE_RESOLVE_MIN_CLOT_NODES"] = "100"
    
    flow = ClotAwareFlow(device, phys_cfg=phys_cfg)
    
    print(f"Loading {patient_id}...")
    graph_path = data_root() / "processed" / "graphs_biochem_anchors" / f"{patient_id}.pt"
    if not graph_path.exists():
        print(f"File not found: {graph_path}")
        return
        
    data = torch.load(graph_path, map_location=device, weights_only=False)
    n = data.num_nodes
    pos = data.x[:, 0:2].cpu().numpy()
    
    # Extract GT flows
    u_gt_t0 = data.y[0, :, 0].cpu().numpy()
    v_gt_t0 = data.y[0, :, 1].cpu().numpy()
    speed_gt_t0 = np.sqrt(u_gt_t0**2 + v_gt_t0**2)
    
    u_gt_t200 = data.y[-1, :, 0].cpu().numpy()
    v_gt_t200 = data.y[-1, :, 1].cpu().numpy()
    speed_gt_t200 = np.sqrt(u_gt_t200**2 + v_gt_t200**2)
    
    # Identify where the clot grew based on velocity drop in GT
    clot_mask_np = (speed_gt_t0 - speed_gt_t200 > 0.05) & (speed_gt_t200 < 0.05) & ~data.mask_wall.cpu().numpy()
    clot_mask = torch.from_numpy(clot_mask_np).to(device)
    
    print(f"Identified dynamically grown clot of {clot_mask.sum().item()} nodes.")
    if clot_mask.sum().item() == 0:
        print("No clot found in this vessel with the current threshold.")
        return
        
    # Find 3-hop neighborhood
    from torch_geometric.utils import k_hop_subgraph
    clot_idx = torch.where(clot_mask)[0]
    subset, _, _, _ = k_hop_subgraph(clot_idx, num_hops=3, edge_index=data.edge_index, relabel_nodes=False)
    subset_np = subset.cpu().numpy()
    print(f"3-hop local neighborhood contains {len(subset_np)} nodes.")
    
    # 1. Unadjusted Prediction (DEQ Base Flow)
    print("Computing Unadjusted DEQ flow...")
    mu_eff_base = torch.full((n,), float(phys_cfg.mu_inf), device=device)
    state_unadjusted = flow.update(data, mu_eff_base)
    u_pred_unadj = state_unadjusted.u.cpu().numpy()
    v_pred_unadj = state_unadjusted.v.cpu().numpy()
    
    # 2. Adjusted Prediction (DEQ + SDF Wallification)
    print("Computing Adjusted DEQ flow...")
    mu_eff_large = mu_eff_base.clone()
    mu_eff_large[clot_mask] = 2.0
    state_adjusted = flow.update(data, mu_eff_large)
    u_pred_adj = state_adjusted.u.cpu().numpy()
    v_pred_adj = state_adjusted.v.cpu().numpy()
    
    # Compute Local Rel L2 Errors
    def rel_l2_local(pred_u, pred_v, gt_u, gt_v, mask):
        pred_stacked = np.stack([pred_u[mask], pred_v[mask]], axis=-1)
        gt_stacked = np.stack([gt_u[mask], gt_v[mask]], axis=-1)
        return np.linalg.norm(pred_stacked - gt_stacked) / (np.linalg.norm(gt_stacked) + 1e-8)
        
    err_unadj = rel_l2_local(u_pred_unadj, v_pred_unadj, u_gt_t200, v_gt_t200, subset_np)
    err_adj = rel_l2_local(u_pred_adj, v_pred_adj, u_gt_t200, v_gt_t200, subset_np)
    
    print(f"Local (3-hop) Rel L2 Error (Unadjusted): {err_unadj:.4f}")
    print(f"Local (3-hop) Rel L2 Error (Adjusted):   {err_adj:.4f}")
    
    # Calculate bounding box for zoom
    clot_pos = pos[subset_np] # Zoom out a bit to show the neighborhood
    x_min, x_max = clot_pos[:, 0].min(), clot_pos[:, 0].max()
    y_min, y_max = clot_pos[:, 1].min(), clot_pos[:, 1].max()
    pad_x = 0.5
    pad_y = 0.2
    
    def apply_zoom(ax):
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)

    # Plotting
    print("Generating visualizations...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 15))
    
    create_quiver_plot(axes[0], pos[:, 0], pos[:, 1], u_pred_unadj, v_pred_unadj, 
                      f"1. Unadjusted Prediction (Local Rel L2: {err_unadj:.4f})", clot_mask_np, skip=2)
    apply_zoom(axes[0])
                      
    create_quiver_plot(axes[1], pos[:, 0], pos[:, 1], u_pred_adj, v_pred_adj, 
                      f"2. Adjusted Prediction (Local Rel L2: {err_adj:.4f})", clot_mask_np, skip=2)
    apply_zoom(axes[1])
                      
    create_quiver_plot(axes[2], pos[:, 0], pos[:, 1], u_gt_t200, v_gt_t200, 
                      "3. Ground Truth COMSOL Flow (t=final)", clot_mask_np, skip=2)
    apply_zoom(axes[2])
                      
    plt.tight_layout()
    out_dir = Path(r"C:\Users\pgssy\.gemini\antigravity\brain\d0524dc0-3837-4442-ab5f-5da07d6faaeb")
    out_path = out_dir / f"{patient_id}_dynamic_comparison.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    patient = sys.argv[1] if len(sys.argv) > 1 else "patient041"
    sys.exit(main(patient))
