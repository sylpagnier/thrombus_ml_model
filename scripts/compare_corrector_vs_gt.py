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
from src.config import PhysicsConfig
from src.inference.corrector_coupling import CorrectorCoupledFlow
from src.utils.paths import data_root, reports_dir

def main():
    device = require_cuda_device()
    print(f"[i] Using device: {device}")
    
    graph_path = data_root() / "processed" / "graphs_biochem_anchors" / "patient007.pt"
    data = torch.load(graph_path, map_location=device, weights_only=False)
    
    # 1. Get GT data at late timestep
    t = int(data.y.shape[0]) - 1
    u_gt = data.y[t, :, 0].contiguous()
    v_gt = data.y[t, :, 1].contiguous()
    mu_eff_nd = data.y[t, :, 3].contiguous()
    
    # Also get t=0 for base flow visualization and mu_bulk reference
    u0_gt = data.y[0, :, 0].contiguous()
    v0_gt = data.y[0, :, 1].contiguous()
    mu_bulk_nd = data.y[0, :, 3].contiguous()
    
    # 2. Setup CorrectorCoupledFlow
    phys = PhysicsConfig(phase="kinematics")
    mu_eff_si = phys.viscosity_nd_to_si(mu_eff_nd)
    mu_bulk_si = phys.viscosity_nd_to_si(mu_bulk_nd)
    
    corrector_flow = CorrectorCoupledFlow(device)
    
    # Ensure base flow is loaded (will use data.u0_pred if available, else run kine model)
    u0_pred, v0_pred = corrector_flow.base_flow(data)
    
    # 3. Predict corrected flow
    print("[i] Running local corrector coupling...")
    u_pred, v_pred = corrector_flow.couple(data, mu_eff_si, mu_bulk_si=mu_bulk_si, publish=False)
    
    # Calculate error
    err_u = torch.abs(u_pred - u_gt)
    err_v = torch.abs(v_pred - v_gt)
    err_mag = torch.sqrt(err_u**2 + err_v**2)
    print(f"[OK] Mean Abs Error (magnitude): {err_mag.mean().item():.4f}")
    
    # 4. Visualization
    print("[i] Generating visualization...")
    # Find clot nodes for visualization
    delta_mu_si = mu_eff_si - mu_bulk_si
    clot_mask = delta_mu_si > corrector_flow.min_delta_mu_si
    clot_idx = torch.where(clot_mask)[0]
    
    # Random subset to keep plot clean
    subset_size = 5000
    if data.num_nodes > subset_size:
        subset = torch.randperm(data.num_nodes, device=device)[:subset_size]
    else:
        subset = torch.arange(data.num_nodes, device=device)
        
    pos = data.x[subset, 0:2].detach().cpu().numpy() * 1000 # to mm
    cpos = data.x[clot_idx, 0:2].detach().cpu().numpy() * 1000
    
    u0_n = u0_pred[subset].detach().cpu().numpy()
    v0_n = v0_pred[subset].detach().cpu().numpy()
    
    u_p_n = u_pred[subset].detach().cpu().numpy()
    v_p_n = v_pred[subset].detach().cpu().numpy()
    
    u_g_n = u_gt[subset].detach().cpu().numpy()
    v_g_n = v_gt[subset].detach().cpu().numpy()
    
    fig, axes = plt.subplots(1, 4, figsize=(22, 5), sharex=True, sharey=True)
    
    axes[0].quiver(pos[:, 0], pos[:, 1], u0_n, v0_n, color="tab:blue", alpha=0.6)
    axes[0].set_title("Base Flow (t=0)")
    
    axes[1].quiver(pos[:, 0], pos[:, 1], u_g_n, v_g_n, color="tab:green", alpha=0.6)
    axes[1].scatter(cpos[:, 0], cpos[:, 1], color="black", s=5, zorder=5)
    axes[1].set_title("Ground Truth Flow (t=end)")
    
    axes[2].quiver(pos[:, 0], pos[:, 1], u_p_n, v_p_n, color="tab:red", alpha=0.6)
    axes[2].scatter(cpos[:, 0], cpos[:, 1], color="black", s=5, zorder=5)
    axes[2].set_title("Corrected Flow (Predicted)")
    
    axes[3].quiver(pos[:, 0], pos[:, 1], u_g_n, v_g_n, color="tab:green", alpha=0.4, label="GT")
    axes[3].quiver(pos[:, 0], pos[:, 1], u_p_n, v_p_n, color="tab:red", alpha=0.7, label="Pred")
    axes[3].scatter(cpos[:, 0], cpos[:, 1], color="black", s=5, zorder=5)
    axes[3].set_title("Overlay: Pred (Red) vs GT (Green)")
    axes[3].legend()
    
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        
    plt.tight_layout()
    out_png = reports_dir() / "figures" / "kinematics" / "gt_vs_pred_flow.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()
    
    print(f"[OK] Saved plot to {out_png}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
