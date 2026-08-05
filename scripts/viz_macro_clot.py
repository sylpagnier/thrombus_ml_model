"""Visualize flow diversion around massive macro-clots using SDF wall-ification.

Tests on patient041 to verify the RGP-DEQ solver accurately reroutes flow
when a large portion of the vessel is occluded by a clot.
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.core_physics.t0_device import require_cuda_device
from src.config import PhysicsConfig, NodeFeat
from src.inference.corrector_coupling import ClotAwareFlow
from src.utils.paths import data_root

def create_streamplot(ax, pos_x, pos_y, u, v, title, clot_mask=None, sdf=None):
    # Create grid
    xi = np.linspace(pos_x.min(), pos_x.max(), 300)
    yi = np.linspace(pos_y.min(), pos_y.max(), 100)
    X, Y = np.meshgrid(xi, yi)

    # Interpolate U and V
    U = griddata((pos_x, pos_y), u, (X, Y), method='linear')
    V = griddata((pos_x, pos_y), v, (X, Y), method='linear')
    Speed = np.sqrt(U**2 + V**2)

    # Plot speed as background
    im = ax.imshow(Speed, extent=(pos_x.min(), pos_x.max(), pos_y.min(), pos_y.max()),
                   origin='lower', cmap='viridis', alpha=0.8, aspect='auto')
    plt.colorbar(im, ax=ax, label="Speed (ND)")

    # Plot streamlines
    # Mask out areas where speed is near zero to avoid messy streamlines inside the wall/clot
    mask = Speed > 0.05
    U_masked = np.where(mask, U, np.nan)
    V_masked = np.where(mask, V, np.nan)
    
    ax.streamplot(X, Y, U_masked, V_masked, color='w', density=1.5, linewidth=0.5, arrowsize=0.8)

    # Overlay clot
    if clot_mask is not None:
        ax.scatter(pos_x[clot_mask], pos_y[clot_mask], color='red', s=2, alpha=0.5, label='Clot Nodes')
        ax.legend()
        
    ax.set_title(title)
    ax.set_xlabel("X (ND)")
    ax.set_ylabel("Y (ND)")


def main():
    device = require_cuda_device()
    
    # 1. Load patient041 (stenosed vessel)
    graph_path = data_root() / "processed" / "graphs_biochem_anchors" / "patient041.pt"
    data = torch.load(graph_path, map_location=device, weights_only=False)
    n = data.num_nodes
    pos = data.x[:, 0:2].cpu().numpy()
    
    print(f"Loaded {graph_path.name} with {n} nodes.")
    
    # 2. Init ClotAwareFlow (Force macro resolves)
    phys_cfg = PhysicsConfig(phase="kinematics")
    os.environ["BIOCHEM_KINE_RESOLVE_ON_CLOT"] = "1"
    os.environ["BIOCHEM_KINE_RESOLVE_MIN_CLOT_NODES"] = "100"
    
    flow = ClotAwareFlow(device, phys_cfg=phys_cfg)
    
    # 3. Get Baseline Flow
    print("Computing baseline flow...")
    mu_eff_base = torch.full((n,), float(phys_cfg.mu_inf), device=device)
    state_base = flow.update(data, mu_eff_base)
    u_base = state_base.u.cpu().numpy()
    v_base = state_base.v.cpu().numpy()
    sdf_base = data.x[:, NodeFeat.SDF.start].cpu().numpy()
    
    # 4. Create a massive synthetic clot blocking the bottom half of the channel in the middle
    x_min, x_max = pos[:, 0].min(), pos[:, 0].max()
    y_min, y_max = pos[:, 1].min(), pos[:, 1].max()
    
    x_mid = (x_max + x_min) / 2
    x_span = x_max - x_min
    y_mid = (y_max + y_min) / 2
    
    # Block from x_mid - 15% to x_mid + 15%, lower half of vessel
    clot_mask = (
        (pos[:, 0] > x_mid - 0.15 * x_span) & 
        (pos[:, 0] < x_mid + 0.15 * x_span) & 
        (pos[:, 1] < y_mid) &
        ~data.mask_wall.cpu().numpy()
    )
    n_clot = int(clot_mask.sum())
    print(f"Created massive clot of {n_clot} nodes ({n_clot/n*100:.1f}% of mesh).")
    
    mu_eff_large = mu_eff_base.clone()
    mu_eff_large[torch.from_numpy(clot_mask).to(device)] = 2.0
    
    # 5. Get Macro-Resolved Flow
    print("Computing macro-resolved flow (SDF Wallification)...")
    state_clot = flow.update(data, mu_eff_large)
    u_clot = state_clot.u.cpu().numpy()
    v_clot = state_clot.v.cpu().numpy()
    
    assert state_clot.mode == "resolved"
    
    # 6. Plotting
    print("Generating visualizations...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    create_streamplot(axes[0], pos[:, 0], pos[:, 1], u_base, v_base, 
                     "Baseline Flow (No Clot)")
                     
    create_streamplot(axes[1], pos[:, 0], pos[:, 1], u_clot, v_clot, 
                     "Macro-Resolved Flow (Massive Clot treated as rigid SDF wall)", 
                     clot_mask)
                     
    plt.tight_layout()
    out_dir = Path(r"C:\Users\pgssy\.gemini\antigravity\brain\d0524dc0-3837-4442-ab5f-5da07d6faaeb")
    out_path = out_dir / "patient041_macro_resolve.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved visualization to {out_path}")
    
    # Repeat for patient042
    print("\nRepeating for patient042...")
    graph_path2 = data_root() / "processed" / "graphs_biochem_anchors" / "patient042.pt"
    data2 = torch.load(graph_path2, map_location=device, weights_only=False)
    n2 = data2.num_nodes
    pos2 = data2.x[:, 0:2].cpu().numpy()
    
    flow.invalidate_base_cache()
    
    mu_eff_base2 = torch.full((n2,), float(phys_cfg.mu_inf), device=device)
    state_base2 = flow.update(data2, mu_eff_base2)
    u_base2 = state_base2.u.cpu().numpy()
    v_base2 = state_base2.v.cpu().numpy()
    
    x_mid2 = (pos2[:, 0].max() + pos2[:, 0].min()) / 2
    x_span2 = pos2[:, 0].max() - pos2[:, 0].min()
    y_mid2 = (pos2[:, 1].max() + pos2[:, 1].min()) / 2
    
    # Block top half of vessel
    clot_mask2 = (
        (pos2[:, 0] > x_mid2 - 0.15 * x_span2) & 
        (pos2[:, 0] < x_mid2 + 0.15 * x_span2) & 
        (pos2[:, 1] > y_mid2) &
        ~data2.mask_wall.cpu().numpy()
    )
    mu_eff_large2 = mu_eff_base2.clone()
    mu_eff_large2[torch.from_numpy(clot_mask2).to(device)] = 2.0
    
    state_clot2 = flow.update(data2, mu_eff_large2)
    u_clot2 = state_clot2.u.cpu().numpy()
    v_clot2 = state_clot2.v.cpu().numpy()
    
    fig2, axes2 = plt.subplots(2, 1, figsize=(14, 10))
    create_streamplot(axes2[0], pos2[:, 0], pos2[:, 1], u_base2, v_base2, "Baseline Flow")
    create_streamplot(axes2[1], pos2[:, 0], pos2[:, 1], u_clot2, v_clot2, "Macro-Resolved Flow (Top-wall Clot)", clot_mask2)
    
    out_path2 = out_dir / "patient042_macro_resolve.png"
    plt.tight_layout()
    plt.savefig(out_path2, dpi=150)
    print(f"Saved visualization to {out_path2}")

if __name__ == "__main__":
    sys.exit(main())
