import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from src.config import NodeFeat

def main():
    parser = argparse.ArgumentParser(description="Verify y-mirrored versions of anchor graph .pt files.")
    parser.add_argument("--anchor", type=str, required=True, help="Anchor to verify.")
    parser.add_argument("--graph-dir", type=str, default="data/processed/graphs_biochem_anchors", help="Directory containing the graph .pt files.")
    parser.add_argument("--plot-out", type=str, default="mirror_verification.png", help="Path to save the verification plot.")
    
    args = parser.parse_args()
    
    graph_dir = Path(args.graph_dir)
    orig_path = graph_dir / f"{args.anchor}.pt"
    mirr_path = graph_dir / f"{args.anchor}_mirror_y.pt"
    
    if not orig_path.exists():
        print(f"[WARN] Original file not found: {orig_path}")
        return
    if not mirr_path.exists():
        print(f"[WARN] Mirrored file not found: {mirr_path}")
        return
        
    print(f"[i] Verifying {args.anchor}...")
    
    # Use weights_only=False for PyG Data objects
    orig = torch.load(orig_path, weights_only=False)
    mirr = torch.load(mirr_path, weights_only=False)
    
    failed = False
    
    # 1. Check Y_ND flipped
    y_match = torch.allclose(mirr.x[:, 1], -orig.x[:, 1], atol=1e-6)
    print(f"  - Y coordinates flipped: {'[OK]' if y_match else '[FAIL]'}")
    if not y_match: failed = True
        
    # 2. Check X_ND identical
    x_match = torch.allclose(mirr.x[:, 0], orig.x[:, 0], atol=1e-6)
    print(f"  - X coordinates identical: {'[OK]' if x_match else '[FAIL]'}")
    if not x_match: failed = True
        
    # 3. Check V velocity flipped (y.shape is [T, N, features], v is index 1)
    v_match = torch.allclose(mirr.y[:, :, 1], -orig.y[:, :, 1], atol=1e-6)
    print(f"  - V velocity (y-comp) flipped: {'[OK]' if v_match else '[FAIL]'}")
    if not v_match: failed = True
        
    # 4. Check U velocity identical
    u_match = torch.allclose(mirr.y[:, :, 0], orig.y[:, :, 0], atol=1e-6)
    print(f"  - U velocity (x-comp) identical: {'[OK]' if u_match else '[FAIL]'}")
    if not u_match: failed = True
        
    # 5. Check Wall Normal Y flipped (NodeFeat.WALL_NORMAL is 4:6, so y is index 5)
    ny_match = torch.allclose(mirr.x[:, 5], -orig.x[:, 5], atol=1e-6)
    print(f"  - Wall Normal Y flipped: {'[OK]' if ny_match else '[FAIL]'}")
    if not ny_match: failed = True
        
    # 6. Check Wall Normal X identical
    nx_match = torch.allclose(mirr.x[:, 4], orig.x[:, 4], atol=1e-6)
    print(f"  - Wall Normal X identical: {'[OK]' if nx_match else '[FAIL]'}")
    if not nx_match: failed = True
        
    # 7. Check scalar species (indices 2+) are identical
    species_match = torch.allclose(mirr.y[:, :, 2:], orig.y[:, :, 2:], atol=1e-6)
    print(f"  - Scalar species identical: {'[OK]' if species_match else '[FAIL]'}")
    if not species_match: failed = True
        
    # 8. Check u0_pred if it exists
    if getattr(orig, "u0_pred", None) is not None:
        u0_v_match = torch.allclose(mirr.u0_pred[:, 1], -orig.u0_pred[:, 1], atol=1e-6)
        print(f"  - u0_pred V velocity flipped: {'[OK]' if u0_v_match else '[FAIL]'}")
        if not u0_v_match: failed = True
            
    print(f"\n=> Quantitative verification: {'FAILED' if failed else 'PASSED'}")
    
    # Visualization
    print(f"[i] Generating visualization...")
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Mirror Augmentation Verification: {args.anchor}", fontsize=16)
    
    # Subsample for vector plots so they are readable
    n_nodes = orig.x.shape[0]
    sub = np.random.choice(n_nodes, min(500, n_nodes), replace=False)
    wall_mask = orig.mask_wall.numpy() if hasattr(orig, "mask_wall") else np.zeros(n_nodes, dtype=bool)
    wall_idx = np.where(wall_mask)[0]
    wall_sub = np.random.choice(wall_idx, min(200, len(wall_idx)), replace=False) if len(wall_idx) > 0 else []
    
    orig_pos = orig.x[:, :2].numpy()
    mirr_pos = mirr.x[:, :2].numpy()
    
    # Orig Geometry & Normals
    ax = axs[0, 0]
    ax.scatter(orig_pos[:, 0], orig_pos[:, 1], s=1, c='gray', alpha=0.5)
    if len(wall_sub) > 0:
        ax.quiver(orig_pos[wall_sub, 0], orig_pos[wall_sub, 1], 
                  orig.x[wall_sub, 4].numpy(), orig.x[wall_sub, 5].numpy(), 
                  color='red', scale=15, width=0.003, label='Wall Normals')
    ax.set_title("Original: Geometry & Wall Normals")
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    
    # Mirrored Geometry & Normals
    ax = axs[0, 1]
    ax.scatter(mirr_pos[:, 0], mirr_pos[:, 1], s=1, c='gray', alpha=0.5)
    if len(wall_sub) > 0:
        ax.quiver(mirr_pos[wall_sub, 0], mirr_pos[wall_sub, 1], 
                  mirr.x[wall_sub, 4].numpy(), mirr.x[wall_sub, 5].numpy(), 
                  color='red', scale=15, width=0.003, label='Wall Normals')
    ax.set_title("Mirrored: Geometry & Wall Normals")
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    
    # Orig Velocity field (t=20 if available, else 0)
    t_idx = min(20, orig.y.shape[0] - 1)
    ax = axs[1, 0]
    ax.scatter(orig_pos[:, 0], orig_pos[:, 1], s=1, c='gray', alpha=0.1)
    ax.quiver(orig_pos[sub, 0], orig_pos[sub, 1], 
              orig.y[t_idx, sub, 0].numpy(), orig.y[t_idx, sub, 1].numpy(), 
              color='blue', scale=50, width=0.002, label=f'Velocity (t={t_idx})')
    ax.set_title("Original: Velocity Field")
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    
    # Mirrored Velocity field
    ax = axs[1, 1]
    ax.scatter(mirr_pos[:, 0], mirr_pos[:, 1], s=1, c='gray', alpha=0.1)
    ax.quiver(mirr_pos[sub, 0], mirr_pos[sub, 1], 
              mirr.y[t_idx, sub, 0].numpy(), mirr.y[t_idx, sub, 1].numpy(), 
              color='blue', scale=50, width=0.002, label=f'Velocity (t={t_idx})')
    ax.set_title("Mirrored: Velocity Field")
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(args.plot_out, dpi=150)
    print(f"[OK] Saved visualization to {args.plot_out}")

if __name__ == "__main__":
    main()
