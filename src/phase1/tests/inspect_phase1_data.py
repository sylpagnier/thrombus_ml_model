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
    print("\n Architecture & Invariants")
    expected_channels = 15
    if data.x.shape != expected_channels:
        print(f" ❌ FAIL: Feature mismatch! Expected {expected_channels}, got {data.x.shape}.")
    else:
        print(f" ✅ PASS: Features aligned ({expected_channels} channels).")

    # --- 2. Physical Consistency ---
    print("\n Boundary & Physics Sanity")
    if data.y is not None:
        wall_vel = torch.norm(data.y[data.mask_wall, :2], dim=1).max().item()
        status = "✅ PASS" if wall_vel < 1e-3 else "❌ FAIL"
        print(f" {status}: No-slip condition (Max Wall Vel: {wall_vel:.2e})")

    # --- 3. Geometric Quality ---
    cond_nums, mask_valid = analyze_geometric_quality(data)
    print(f"\n Mesh Stability (WLS Condition Numbers)")
    print(f" -> Mean: {np.mean(cond_nums[mask_valid]):.2e} | Max: {np.max(cond_nums[mask_valid]):.2e}")

    # --- 4. Visualization Grid (3x4 Layout) ---
    print("\nRendering visualization...")
    pos = data.x[:, :2].numpy()

    # PRE-CALCULATE MAGNITUDES FOR PHYSICAL ALIGNMENT
    # GT Magnitude: sqrt(u^2 + v^2) from indices 0 and 1
    vel_mag_gt = torch.norm(data.y[:, 0:2], dim=1).cpu().numpy()
    # Prior Magnitude: sqrt(u_p^2 + v_p^2) from indices 11 and 12
    vel_mag_prior = torch.norm(data.x[:, 11:13], dim=1).cpu().numpy()

    fig, axes = plt.subplots(3, 4, figsize=(20, 12), constrained_layout=True)
    axes = axes.flatten()

    # ROW 1: INPUTS & GEOMETRY
    sc0 = axes.scatter(pos[:, 0], pos[:, 1], c=data.x[:, 2], cmap='viridis', s=2)
    axes.set_title("Input: ND-SDF")
    plt.colorbar(sc0, ax=axes)

    sc1 = axes.scatter(pos[:, 0], pos[:, 1], c=np.log10(cond_nums + 1), cmap='magma', s=2)
    axes.set_title("Mesh: Log10(WLS Cond)")
    plt.colorbar(sc1, ax=axes)

    mask_w = data.mask_wall.numpy()
    axes.scatter(pos[:, 0], pos[:, 1], color='lightgray', s=1, alpha=0.1)
    axes.quiver(pos[mask_w, 0], pos[mask_w, 1], data.x[mask_w, 4], data.x[mask_w, 5], color='red', scale=30)
    axes.set_title("Input: Wall Normals")

    axes.scatter(pos[data.mask_inlet, 0], pos[data.mask_inlet, 1], c='green', s=5, label='Inlet')
    axes.scatter(pos[data.mask_outlet, 0], pos[data.mask_outlet, 1], c='blue', s=5, label='Outlet')
    axes.scatter(pos[data.mask_wall, 0], pos[data.mask_wall, 1], c='black', s=2, label='Wall')
    axes.set_title("Input: Boundary Masks")
    axes.legend(loc='upper right', fontsize='x-small')

    # ROW 2: GROUND TRUTH (Now with Magnitude)
    sc4 = axes.scatter(pos[:, 0], pos[:, 1], c=vel_mag_gt, cmap='jet', s=2)
    axes.set_title("GT: Velocity Magnitude")
    plt.colorbar(sc4, ax=axes, label='ND Speed')

    sc5 = axes.scatter(pos[:, 0], pos[:, 1], c=data.y[:, 2], cmap='coolwarm', s=2)
    axes.set_title("GT: Pressure")
    plt.colorbar(sc5, ax=axes)

    sc6 = axes.scatter(pos[:, 0], pos[:, 1], c=data.y[:, 3], cmap='plasma', s=2)
    axes.set_title("GT: ND-Viscosity")
    plt.colorbar(sc6, ax=axes)

    sc7 = axes.scatter(pos[:, 0], pos[:, 1], c=data.y[:, 4], cmap='inferno', s=2)
    axes.set_title("GT: Wall Shear Stress")
    plt.colorbar(sc7, ax=axes)

    # ROW 3: PHYSICAL PRIORS (Now with Magnitude)
    sc8 = axes.scatter(pos[:, 0], pos[:, 1], c=vel_mag_prior, cmap='jet', s=2)
    axes.set_title("Prior: Velocity Magnitude")
    plt.colorbar(sc8, ax=axes, label='ND Speed')

    sc9 = axes.scatter(pos[:, 0], pos[:, 1], c=data.x[:, 13], cmap='plasma', s=2)
    axes.set_title("Prior: Viscosity")
    plt.colorbar(sc9, ax=axes)

    sc10 = axes.scatter(pos[:, 0], pos[:, 1], c=data.x[:, 14], cmap='inferno', s=2)
    axes.set_title("Prior: WSS")
    plt.colorbar(sc10, ax=axes)

    axes.axis('off')
    axes.text(0, 0.5, f"File: {filename}\nRe: {phys_cfg.re_target}\nModel: {phys_cfg.viscosity_model}", fontsize=10)

    for ax in axes:
        if ax.get_title():
            ax.set_aspect('equal')
            ax.axis('off')

    plt.show()


if __name__ == "__main__":
    inspect_sample(filename="vessel_6.pt", tier="tier1")