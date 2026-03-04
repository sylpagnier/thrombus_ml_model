import torch
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.utils import degree
from src.config import PhysicsConfig, VesselConfig
from src.phase1.physics.physics_kernels import PhysicsKernels, scatter_add


def analyze_geometric_quality(data):
    """Analyzes mesh quality via WLS condition numbers (2nd-order 5x5 matrix)."""
    row, col = data.edge_index
    num_nodes = data.num_nodes
    d = degree(row, num_nodes, dtype=torch.long)

    pos_diff = data.x[col, :2] - data.x[row, :2]
    dist = torch.norm(pos_diff, dim=1)

    ones = torch.ones_like(dist)
    count = scatter_add(ones, row, dim=0, dim_size=num_nodes)
    sum_dist = scatter_add(dist, row, dim=0, dim_size=num_nodes)
    avg_edge_len = (sum_dist / (count + 1e-6))[row]

    pos_diff_norm = pos_diff / (avg_edge_len.unsqueeze(1) + 1e-8)
    dx, dy = pos_diff_norm[:, 0], pos_diff_norm[:, 1]
    W = 1.0 / (dx ** 2 + dy ** 2 + 1e-8)

    V = torch.stack([dx, dy, 0.5 * dx ** 2, dx * dy, 0.5 * dy ** 2], dim=1)
    M_e = W.view(-1, 1, 1) * torch.bmm(V.unsqueeze(2), V.unsqueeze(1))
    M = scatter_add(M_e.view(-1, 25), row, dim=0, dim_size=num_nodes).view(num_nodes, 5, 5)

    try:
        eigenvalues = torch.linalg.eigvalsh(M)
        cond_numbers = eigenvalues[:, -1] / (torch.abs(eigenvalues[:, 0]) + 1e-12)
        return cond_numbers.cpu().numpy(), (d >= 5).cpu().numpy()
    except RuntimeError:
        return np.zeros(num_nodes), np.zeros(num_nodes, dtype=bool)


def inspect_sample(filename="vessel_0.pt", tier="tier2"):
    phys_cfg = PhysicsConfig(tier=tier)
    vessel_cfg = VesselConfig(tier=tier)
    data_path = vessel_cfg.graph_output_dir / filename

    if not data_path.exists():
        print(f"File {filename} not found.")
        return

    print(f"\n{'=' * 60}\n INSPECTING: {data_path.name} | TIER: {tier.upper()}\n{'=' * 60}")
    data = torch.load(data_path, weights_only=False)

    # --- 1. Structural & Feature Validation ---
    print("\n[1] Architecture & Invariants")

    # Updated to 15 channels (Pos:2, SDF:1, ShearPot:1, Normal:2, Type:4, NonNewt:1, UVPrior:2, MuPrior:1, WSSPrior:1)
    expected_channels = 15
    if data.x.shape[1] != expected_channels:
        print(f" ❌ FAIL: Feature mismatch! Expected {expected_channels}, got {data.x.shape[1]}.")
    else:
        print(f" ✅ PASS: Features aligned ({expected_channels} channels).")

    # Check for NaNs across all attributes
    for attr in ['x', 'y', 'edge_attr', 'edge_index']:
        val = getattr(data, attr)
        if val is not None and torch.isnan(val).any():
            print(f" ❌ FAIL: NaNs found in '{attr}'.")
        else:
            print(f" ✅ PASS: '{attr}' is clean.")

    # --- 2. Physical Consistency ---
    print("\n[2] Boundary & Physics Sanity")

    # No-slip condition (y[mask_wall, 0:2] should be ~0)
    if data.y is not None:
        wall_vel = torch.norm(data.y[data.mask_wall, :2], dim=1).max().item()
        status = "✅ PASS" if wall_vel < 1e-3 else "❌ FAIL"
        print(f" {status}: No-slip condition (Max Wall Vel: {wall_vel:.2e})")

    # SDF Zero-crossing check
    wall_sdf = torch.abs(data.x[data.mask_wall, 2]).max().item()
    status = "✅ PASS" if wall_sdf < 5e-3 else "❌ FAIL"
    print(f" {status}: SDF Wall Alignment (Max SDF at Wall: {wall_sdf:.2e})")

    # --- 3. Geometric Quality ---
    cond_nums, mask_valid = analyze_geometric_quality(data)
    print(f"\n[3] Mesh Stability (WLS Condition Numbers)")
    print(f" -> Mean: {np.mean(cond_nums[mask_valid]):.2e} | Max: {np.max(cond_nums[mask_valid]):.2e}")
    if np.max(cond_nums[mask_valid]) > 1e7:
        print(" ⚠️ WARNING: Extreme condition numbers. WLS gradients will likely explode.")

    # --- 4. Visualization Grid (3x4 Layout) ---
    print("\nRendering visualization...")
    pos = data.x[:, :2].numpy()

    fig, axes = plt.subplots(3, 4, figsize=(20, 12), constrained_layout=True)
    axes = axes.flatten()

    # Row 1: Inputs & Geometric Quality
    # 0: SDF
    sc0 = axes[0].scatter(pos[:, 0], pos[:, 1], c=data.x[:, 2], cmap='viridis', s=2)
    axes[0].set_title("Input: ND-SDF")
    plt.colorbar(sc0, ax=axes[0])

    # 1: WLS Condition
    sc1 = axes[1].scatter(pos[:, 0], pos[:, 1], c=np.log10(cond_nums + 1), cmap='magma', s=2)
    axes[1].set_title("Mesh: Log10(WLS Cond)")
    plt.colorbar(sc1, ax=axes[1])

    # 2: Wall Normals (Quiver)
    mask_w = data.mask_wall.numpy()
    axes[2].scatter(pos[:, 0], pos[:, 1], color='lightgray', s=1, alpha=0.1)
    axes[2].quiver(pos[mask_w, 0], pos[mask_w, 1], data.x[mask_w, 4], data.x[mask_w, 5], color='red', scale=30)
    axes[2].set_title("Input: Wall Normals")

    # 3: Boundary Masks
    axes[3].scatter(pos[data.mask_inlet, 0], pos[data.mask_inlet, 1], c='green', s=5, label='Inlet')
    axes[3].scatter(pos[data.mask_outlet, 0], pos[data.mask_outlet, 1], c='blue', s=5, label='Outlet')
    axes[3].scatter(pos[data.mask_wall, 0], pos[data.mask_wall, 1], c='black', s=2, label='Wall')
    axes[3].set_title("Input: Boundary Masks")
    axes[3].legend(loc='upper right', fontsize='x-small')

    # Row 2: Ground Truth (Labels)
    # 4: U-Velocity
    sc4 = axes[4].scatter(pos[:, 0], pos[:, 1], c=data.y[:, 0], cmap='jet', s=2)
    axes[4].set_title("GT: U-Velocity")
    plt.colorbar(sc4, ax=axes[4])

    # 5: Pressure
    sc5 = axes[5].scatter(pos[:, 0], pos[:, 1], c=data.y[:, 2], cmap='coolwarm', s=2)
    axes[5].set_title("GT: Pressure")
    plt.colorbar(sc5, ax=axes[5])

    # 6: Viscosity (Tier 2)
    sc6 = axes[6].scatter(pos[:, 0], pos[:, 1], c=data.y[:, 3], cmap='plasma', s=2)
    axes[6].set_title("GT: ND-Viscosity")
    plt.colorbar(sc6, ax=axes[6])

    # 7: WSS Magnitude (New Channel!)
    sc7 = axes[7].scatter(pos[:, 0], pos[:, 1], c=data.y[:, 4], cmap='inferno', s=2)
    axes[7].set_title("GT: Wall Shear Stress")
    plt.colorbar(sc7, ax=axes[7])

    # Row 3: Physical Priors (x-channels 11-14)
    # 8: U-Prior
    sc8 = axes[8].scatter(pos[:, 0], pos[:, 1], c=data.x[:, 11], cmap='jet', s=2)
    axes[8].set_title("Prior: U-Velocity")
    plt.colorbar(sc8, ax=axes[8])

    # 9: Mu-Prior
    sc9 = axes[9].scatter(pos[:, 0], pos[:, 1], c=data.x[:, 13], cmap='plasma', s=2)
    axes[9].set_title("Prior: Viscosity")
    plt.colorbar(sc9, ax=axes[9])

    # 10: WSS-Prior
    sc10 = axes[10].scatter(pos[:, 0], pos[:, 1], c=data.x[:, 14], cmap='inferno', s=2)
    axes[10].set_title("Prior: WSS")
    plt.colorbar(sc10, ax=axes[10])

    # 11: Info
    axes[11].axis('off')
    axes[11].text(0, 0.5, f"File: {filename}\nRe: {phys_cfg.re_target}\nModel: {phys_cfg.viscosity_model}", fontsize=10)

    for ax in axes:
        if ax.get_title():
            ax.set_aspect('equal')

    plt.show()


if __name__ == "__main__":
    inspect_sample(filename="vessel_6.pt", tier="tier1")