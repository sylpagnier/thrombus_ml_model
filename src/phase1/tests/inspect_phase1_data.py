import torch
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.utils import degree
from src.config import PhysicsConfig, VesselConfig
from src.phase1.physics.physics_kernels import PhysicsKernels, scatter_add


def analyze_geometric_quality(data):
    """
    Analyzes geometric quality using 2nd-order WLS (5x5 matrix).
    Normalizes by local edge length to prevent unit-scale artifacts.
    """
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

    dist_sq_norm = dx ** 2 + dy ** 2 + 1e-8
    W = 1.0 / dist_sq_norm

    dx2, dxy, dy2 = 0.5 * dx ** 2, dx * dy, 0.5 * dy ** 2
    V = torch.stack([dx, dy, dx2, dxy, dy2], dim=1)

    V_unsqueezed = V.unsqueeze(2)
    V_T_unsqueezed = V.unsqueeze(1)
    M_e = W.view(-1, 1, 1) * torch.bmm(V_unsqueezed, V_T_unsqueezed)

    M_e_flat = M_e.view(-1, 25)
    M_flat = scatter_add(M_e_flat, row, dim=0, dim_size=num_nodes)
    M = M_flat.view(num_nodes, 5, 5)

    try:
        eigenvalues = torch.linalg.eigvalsh(M)
        cond_numbers = eigenvalues[:, -1] / (torch.abs(eigenvalues[:, 0]) + 1e-12)
        cond_nums_np = cond_numbers.cpu().numpy()
        mask_valid = (d >= 5).cpu().numpy()
        return cond_nums_np, mask_valid
    except RuntimeError:
        return np.zeros(num_nodes), np.zeros(num_nodes, dtype=bool)


def inspect_sample(filename="vessel_0.pt", tier="tier1"):
    # 1. Configuration & Path Setup
    phys_cfg = PhysicsConfig(tier=tier)
    vessel_cfg = VesselConfig(tier=tier)

    data_dir = vessel_cfg.graph_output_dir
    data_path = data_dir / filename

    if not data_path.exists():
        print(f"File {filename} not found in {data_dir}. Looking for alternatives...")
        files = sorted(list(data_dir.glob("*.pt")))
        if not files:
            print("No .pt files found. Please generate the graphs first.")
            return
        data_path = files[0]

    print(f"\n{'=' * 50}")
    print(f" LOADING: {data_path.name} ({tier.upper()})")
    print(f"{'=' * 50}")
    data = torch.load(data_path, weights_only=False)

    # 2. Strict Invariant Checks (No Zombie Logic)
    print("\n--- Structural & Physical Validation ---")

    # NaN Check
    if torch.isnan(data.x).any() or (data.y is not None and torch.isnan(data.y).any()):
        print(" ❌ FAIL: NaNs detected in graph data.")
    else:
        print(" ✅ PASS: No NaNs detected.")

    # Version Check
    if data.x.shape[1] != 11:
        print(f" ❌ FAIL: Feature mismatch! Expected 11 channels, got {data.x.shape[1]}. Data is outdated.")
        # We assign a default just so the plotting doesn't crash, but we DO NOT save/patch the data.
        phys_cfg.viscosity_model = "newtonian"
    else:
        print(" ✅ PASS: Feature format is current (11 channels).")
        if data.x[0, -1] == 1.0:
            phys_cfg.viscosity_model = "carreau"
        else:
            phys_cfg.viscosity_model = "newtonian"

    # Mask Exclusivity Check
    sum_masks = data.mask_inlet.int() + data.mask_outlet.int() + data.mask_wall.int()
    if torch.any(sum_masks > 1):
        print(" ❌ FAIL: Overlapping boundary masks detected (a node cannot be two boundaries at once).")
    elif not (data.mask_inlet.any() and data.mask_outlet.any() and data.mask_wall.any()):
        print(" ❌ FAIL: Missing one or more boundary masks (Inlet/Outlet/Wall).")
    else:
        print(" ✅ PASS: Boundary masks are exclusive and present.")

    # Geometric Prior Validation (Only if format is modern enough)
    if data.x.shape[1] >= 11:
        wall_normals = data.x[data.mask_wall, 4:6]
        magnitudes = torch.linalg.norm(wall_normals, dim=1)
        if not torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-4):
            print(" ❌ FAIL: Wall normals are not unit length.")
        else:
            print(" ✅ PASS: Wall normals are strictly unit length.")

        sdf_vals = data.x[:, 2]
        if not torch.all(sdf_vals >= -1e-5):
            print(" ❌ FAIL: Negative SDF values detected.")
        elif not torch.all(torch.abs(sdf_vals[data.mask_wall]) < 1e-3):
            print(" ❌ FAIL: SDF is not zero at the wall boundary.")
        else:
            print(" ✅ PASS: SDF values follow spatial boundaries.")

    # No-Slip Condition Check
    if data.y is not None:
        wall_velocities = data.y[data.mask_wall, :2]
        max_wall_vel = torch.max(torch.abs(wall_velocities)).item()
        if max_wall_vel >= 0.05:
            print(f" ❌ FAIL: No-slip condition violated. Max wall velocity: {max_wall_vel:.4f} [-]")
        else:
            print(f" ✅ PASS: No-slip condition holds (Max Wall Vel: {max_wall_vel:.4f} [-]).")

    # 3. Global Information
    print("\n--- File Metadata ---")
    print(f" -> Mode: {phys_cfg.viscosity_model.capitalize()}")
    if hasattr(data, 'd_bar'):
        print(f" -> Mean Vessel Diameter (D_bar): {data.d_bar.item():.6f}")
    if hasattr(data, 'u_inlet_bc'):
        u_max = torch.max(data.u_inlet_bc[:, 0]).item()
        print(f" -> Max ND U_inlet applied: {u_max:.4f} [-] (Scaled for Re={phys_cfg.re_target})")

    # 4. Geometric Quality Check
    print("\n--- Internal Mesh Stability ---")
    cond_nums, mask_valid = analyze_geometric_quality(data)
    valid_cond = cond_nums[mask_valid]
    print(f" -> Nodes with < 5 neighbors: {len(cond_nums) - len(valid_cond)} (Excluded)")
    print(f" -> Mean Condition No:  {np.mean(valid_cond):.2e}")
    print(f" -> Max Condition No:   {np.max(valid_cond):.2e}")
    if np.max(valid_cond) > 1e6:
        print(" ⚠️ WARNING: High condition numbers detected. Potential WLS gradient instability.")

    # 5. Physics Residual Check
    if hasattr(data, 'y') and data.y is not None:
        print("\n--- Physical Governing Equations (Residuals) ---")
        kernels = PhysicsKernels(phys_cfg=phys_cfg)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        data_gpu = data.clone().to(device)

        with torch.no_grad():
            res_ns = kernels.navier_stokes_residual(data_gpu.y, data_gpu)
            res_bc = kernels.boundary_condition_loss(data_gpu.y, data_gpu)
            res_io = kernels.inlet_outlet_loss(data_gpu.y, data_gpu)
            res_rheo = kernels.rheology_loss(data_gpu.y, data_gpu)

            print(f" -> Navier-Stokes: {res_ns.item():.6e}")
            print(f" -> BC Loss:       {res_bc.item():.6e}")
            print(f" -> Inlet/Outlet:  {res_io.item():.6e}")
            print(f" -> Rheology:      {res_rheo.item():.6e}")

    # 6. Dynamic Visualization
    print("\nLaunching visualization...")
    pos = data.x[:, :2].numpy()
    sdf_val = data.x[:, 2].numpy()
    shear_val = data.x[:, 3].numpy()

    cols = 3
    rows = 3 if phys_cfg.viscosity_model == "carreau" else 2

    fig = plt.figure(figsize=(6 * cols, 5 * rows), constrained_layout=True)
    gs = fig.add_gridspec(rows, cols)

    # --- Plot 1: Geometric Stability ---
    ax1 = fig.add_subplot(gs[0, 0])
    cmap = plt.cm.viridis
    cmap.set_bad(color='lightgrey')
    cond_viz = np.log10(cond_nums + 1)
    sc1 = ax1.scatter(pos[:, 0], pos[:, 1], c=cond_viz, cmap=cmap, s=5)
    plt.colorbar(sc1, ax=ax1, label="Log10(Cond No)", fraction=0.046, pad=0.04)
    ax1.set_title("Geometric Quality (WLS Condition)")
    ax1.set_aspect('equal')

    # --- Plot 2: SDF (Input Feature) ---
    ax2 = fig.add_subplot(gs[0, 1])
    sc2 = ax2.scatter(pos[:, 0], pos[:, 1], c=sdf_val, cmap='viridis', s=5)
    plt.colorbar(sc2, ax=ax2, label="ND-SDF", fraction=0.046, pad=0.04)
    ax2.set_title("Signed Distance Function")
    ax2.set_aspect('equal')

    # --- Plot 3: Shear Potential (Input Feature) ---
    ax3 = fig.add_subplot(gs[0, 2])
    sc3 = ax3.scatter(pos[:, 0], pos[:, 1], c=shear_val, cmap='plasma', s=5)
    plt.colorbar(sc3, ax=ax3, label="Shear Pot", fraction=0.046, pad=0.04)
    ax3.set_title("Shear Rate Potential (Log)")
    ax3.set_aspect('equal')

    # --- Plot 4: Velocity U (Ground Truth) ---
    ax4 = fig.add_subplot(gs[1, 0])
    if hasattr(data, 'y') and data.y is not None:
        sc4 = ax4.scatter(pos[:, 0], pos[:, 1], c=data.y[:, 0].numpy(), cmap='jet', s=5)
        plt.colorbar(sc4, ax=ax4, label="U Velocity", fraction=0.046, pad=0.04)
        ax4.set_title("Ground Truth: U Velocity")
    else:
        ax4.text(0.5, 0.5, "No Ground Truth", ha='center')
    ax4.set_aspect('equal')

    # --- Plot 5: Pressure (Ground Truth) ---
    ax5 = fig.add_subplot(gs[1, 1])
    if hasattr(data, 'y') and data.y is not None:
        sc5 = ax5.scatter(pos[:, 0], pos[:, 1], c=data.y[:, 2].numpy(), cmap='coolwarm', s=5)
        plt.colorbar(sc5, ax=ax5, label="Pressure", fraction=0.046, pad=0.04)
        ax5.set_title("Ground Truth: Pressure")
    else:
        ax5.text(0.5, 0.5, "No Ground Truth", ha='center')
    ax5.set_aspect('equal')

    # --- Plot 6: Boundary Masks (Input Labels) ---
    ax6 = fig.add_subplot(gs[1, 2])
    mask_inlet = data.mask_inlet.numpy().astype(bool)
    mask_outlet = data.mask_outlet.numpy().astype(bool)
    mask_wall = data.mask_wall.numpy().astype(bool)

    ax6.scatter(pos[:, 0], pos[:, 1], c='whitesmoke', s=15, marker='s')
    ax6.scatter(pos[mask_inlet, 0], pos[mask_inlet, 1], c='green', s=10, label='Inlet')
    ax6.scatter(pos[mask_outlet, 0], pos[mask_outlet, 1], c='red', s=10, label='Outlet')
    ax6.scatter(pos[mask_wall, 0], pos[mask_wall, 1], c='black', s=5, label='Wall')
    ax6.set_title("Boundary Masks")
    ax6.legend(loc='upper right', fontsize='small')
    ax6.set_aspect('equal')

    # --- Tier 2 Specific Plots ---
    if rows == 3:
        # Plot 7: Viscosity Field (\mu)
        ax7 = fig.add_subplot(gs[2, 0])
        if hasattr(data, 'y') and data.y is not None and data.y.shape[1] == 4:
            mu_vals = data.y[:, 3].numpy()
            sc7 = ax7.scatter(pos[:, 0], pos[:, 1], c=mu_vals, cmap='viridis', s=5)
            plt.colorbar(sc7, ax=ax7, label="Viscosity [-] (ND)", fraction=0.046, pad=0.04)
            ax7.set_title(r"Carreau-Yasuda Viscosity ($\mu$)")
        else:
            ax7.text(0.5, 0.5, "Viscosity Missing", ha='center')
        ax7.set_aspect('equal')

        # Plot 8: Wall Normal Vectors
        ax8 = fig.add_subplot(gs[2, 1])
        skip = max(1, len(pos) // 500)
        ax8.scatter(pos[:, 0], pos[:, 1], c='lightgray', s=1, alpha=0.3)
        if mask_wall.any() and data.x.shape[1] >= 11:
            wall_pos = pos[mask_wall]
            nx, ny = data.x[mask_wall, 4].numpy(), data.x[mask_wall, 5].numpy()
            ax8.quiver(wall_pos[::skip, 0], wall_pos[::skip, 1], nx[::skip], ny[::skip],
                       color='purple', scale=20, width=0.005)
            ax8.set_title(r"Geometric Prior: Wall Normals ($n_{wall}$)")
        else:
            ax8.set_title("Wall Normals Missing")
        ax8.set_aspect('equal')

        # Plot 9: Placeholder/Empty (keeps the 3x3 grid neat)
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')

    plt.show()


if __name__ == "__main__":
    inspect_sample(filename="vessel_12.pt", tier="tier1")