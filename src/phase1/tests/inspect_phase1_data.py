import torch
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.utils import degree
from src.config import PhysicsConfig, VesselConfig
from src.phase1.physics.physics_kernels import scatter_add


# ============================================================
# Geometry Analysis
# ============================================================
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


# ============================================================
# Visualization Helper
# ============================================================
def plot_field(ax, pos, values, title, cmap="viridis", colorbar=True, **kwargs):
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=values, cmap=cmap, s=2, **kwargs)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")

    if colorbar:
        plt.colorbar(sc, ax=ax)

    return sc


# ============================================================
# Main Inspection Function
# ============================================================
def inspect_sample(filename="vessel_0.pt", tier="tier2"):
    phys_cfg = PhysicsConfig(tier=tier)
    vessel_cfg = VesselConfig(tier=tier)
    data_path = vessel_cfg.graph_output_dir / filename

    if not data_path.exists():
        print(f"File {filename} not found.")
        return

    print(f"\n{'=' * 60}\n INSPECTING: {data_path.name} | TIER: {tier.upper()}\n{'=' * 60}")
    data = torch.load(data_path, weights_only=False)

    # ------------------------------------------------------------
    # 1. Structural Checks
    # ------------------------------------------------------------
    print("\n Architecture & Invariants")
    expected_channels = 15
    if data.x.shape[1] != expected_channels:
        print(f" ❌ FAIL: Feature mismatch! Expected {expected_channels}, got {data.x.shape}.")
    else:
        print(f" ✅ PASS: Features aligned ({expected_channels} channels).")

    # ------------------------------------------------------------
    # 2. Physics Checks
    # ------------------------------------------------------------
    print("\n Boundary & Physics Sanity")
    if data.y is not None:
        wall_vel = torch.norm(data.y[data.mask_wall, :2], dim=1).max().item()
        status = "✅ PASS" if wall_vel < 1e-3 else "❌ FAIL"
        print(f" {status}: No-slip condition (Max Wall Vel: {wall_vel:.2e})")

    # ------------------------------------------------------------
    # 3. Geometric Quality
    # ------------------------------------------------------------
    cond_nums, mask_valid = analyze_geometric_quality(data)
    print(f"\n Mesh Stability (WLS Condition Numbers)")
    print(f" -> Mean: {np.mean(cond_nums[mask_valid]):.2e} | Max: {np.max(cond_nums[mask_valid]):.2e}")

    # ------------------------------------------------------------
    # 4. Visualization
    # ------------------------------------------------------------
    print("\nRendering visualization...")

    pos = data.x[:, :2].cpu().numpy()

    vel_mag_gt = torch.norm(data.y[:, 0:2], dim=1).cpu().numpy()
    vel_mag_prior = torch.norm(data.x[:, 11:13], dim=1).cpu().numpy()

    fig, axes = plt.subplots(3, 4, figsize=(20, 12), constrained_layout=True)
    axes = axes.flatten()

    # ---- Plot definitions (clean + extensible) ----
    plots = [
        # Row 1
        ("Input: ND-SDF", data.x[:, 2], "viridis"),
        ("Mesh: Log10(WLS Cond)", np.log10(cond_nums + 1), "magma"),
        ("Input: Wall Normals", None, None),  # special case
        ("Input: Boundary Masks", None, None),

        # Row 2
        ("GT: Velocity Magnitude", vel_mag_gt, "jet"),
        ("GT: Pressure", data.y[:, 2], "coolwarm"),
        ("GT: ND-Viscosity", data.y[:, 3], "plasma"),
        ("GT: Wall Shear Stress", data.y[:, 4], "inferno"),

        # Row 3
        ("Prior: Velocity Magnitude", vel_mag_prior, "jet"),
        ("Prior: Viscosity", data.x[:, 13], "plasma"),
        ("Prior: WSS", data.x[:, 14], "inferno"),
    ]

    # ---- Loop through standard plots ----
    for i, (title, values, cmap) in enumerate(plots):
        ax = axes[i]

        if values is not None:
            if torch.is_tensor(values):
                values = values.cpu().numpy()
            plot_field(ax, pos, values, title, cmap=cmap)
        else:
            # Special cases
            if title == "Input: Wall Normals":
                mask_w = data.mask_wall.cpu().numpy()
                ax.scatter(pos[:, 0], pos[:, 1], color='lightgray', s=1, alpha=0.1)
                ax.quiver(
                    pos[mask_w, 0],
                    pos[mask_w, 1],
                    data.x[mask_w, 4],
                    data.x[mask_w, 5],
                    color='red',
                    scale=30
                )
                ax.set_title(title)
                ax.set_aspect("equal")
                ax.axis("off")

            elif title == "Input: Boundary Masks":
                ax.scatter(pos[data.mask_inlet, 0], pos[data.mask_inlet, 1], c='green', s=5, label='Inlet')
                ax.scatter(pos[data.mask_outlet, 0], pos[data.mask_outlet, 1], c='blue', s=5, label='Outlet')
                ax.scatter(pos[data.mask_wall, 0], pos[data.mask_wall, 1], c='black', s=2, label='Wall')
                ax.legend(loc='upper right', fontsize='x-small')
                ax.set_title(title)
                ax.set_aspect("equal")
                ax.axis("off")

    # ---- Metadata panel ----
    meta_ax = axes[-1]
    meta_ax.axis("off")
    meta_ax.text(
        0,
        0.5,
        f"File: {filename}\nRe: {phys_cfg.re_target}\nModel: {phys_cfg.viscosity_model}",
        fontsize=10
    )

    plt.show()


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    inspect_sample(filename="vessel_73.pt", tier="tier1")