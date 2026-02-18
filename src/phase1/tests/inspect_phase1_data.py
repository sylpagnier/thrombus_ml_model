import torch
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.utils import degree
from src.config import PhysicsConfig, VesselConfig
from src.phase1.physics.physics_kernels import PhysicsKernels, scatter_add
from src.utils.paths import get_project_root


def analyze_geometric_quality(data):
    """
    Analyzes geometric quality using 2nd-order WLS (5x5 matrix).
    Normalizes by local edge length to prevent unit-scale artifacts.
    Returns:
        cond_nums: The condition number for every node.
        mask_valid: Boolean mask identifying nodes with enough neighbors (>=5) for valid stats.
    """
    row, col = data.edge_index
    num_nodes = data.num_nodes

    # 1. Compute Node Degree for filtering
    d = degree(row, num_nodes, dtype=torch.long)

    # 2. Geometric Setup (Normalized)
    pos_diff = data.x[col, :2] - data.x[row, :2]
    dist = torch.norm(pos_diff, dim=1)

    # Average edge length per node for normalization
    ones = torch.ones_like(dist)
    count = scatter_add(ones, row, dim=0, dim_size=num_nodes)
    sum_dist = scatter_add(dist, row, dim=0, dim_size=num_nodes)
    avg_edge_len = (sum_dist / (count + 1e-6))[row]

    pos_diff_norm = pos_diff / (avg_edge_len.unsqueeze(1) + 1e-8)
    dx, dy = pos_diff_norm[:, 0], pos_diff_norm[:, 1]

    dist_sq_norm = dx ** 2 + dy ** 2 + 1e-8
    W = 1.0 / dist_sq_norm

    # 3. Form 5x5 Matrix Basis: [dx, dy, 0.5*dx^2, dx*dy, 0.5*dy^2]
    dx2, dxy, dy2 = 0.5 * dx ** 2, dx * dy, 0.5 * dy ** 2
    V = torch.stack([dx, dy, dx2, dxy, dy2], dim=1)

    V_unsqueezed = V.unsqueeze(2)
    V_T_unsqueezed = V.unsqueeze(1)
    M_e = W.view(-1, 1, 1) * torch.bmm(V_unsqueezed, V_T_unsqueezed)

    M_e_flat = M_e.view(-1, 25)
    M_flat = scatter_add(M_e_flat, row, dim=0, dim_size=num_nodes)
    M = M_flat.view(num_nodes, 5, 5)

    # 4. Eigenvalue check
    try:
        eigenvalues = torch.linalg.eigvalsh(M)
        # Condition number = Max Eigen / Min Eigen
        cond_numbers = eigenvalues[:, -1] / (torch.abs(eigenvalues[:, 0]) + 1e-12)
        cond_nums_np = cond_numbers.cpu().numpy()

        # Filter out nodes with fewer than 5 neighbors (Rank Deficient)
        mask_valid = (d >= 5).cpu().numpy()

        return cond_nums_np, mask_valid

    except RuntimeError:
        return np.zeros(num_nodes), np.zeros(num_nodes, dtype=bool)


def inspect_sample(filename="vessel_0.pt"):
    # 1. Load Data
    root = get_project_root()
    data_dir = root / VesselConfig.graph_output_dir
    data_path = data_dir / filename

    if not data_path.exists():
        files = sorted(list(data_dir.glob("*.pt")))
        if not files:
            print(" No .pt files found.")
            return
        data_path = files[0]

    print(f"--- Loading {data_path.name} ---")
    data = torch.load(data_path, weights_only=False)

    # 2. Version & Attribute Check
    phys_cfg = PhysicsConfig()

    # Default to assuming 11 channels is the target
    if data.x.shape[1] == 10:
        print(" [VERSION WARNING] Data is old (10 channels). Missing 'is_non_newt' flag.")
        phys_cfg.viscosity_model = "newtonian"
        # Patch for viz: Add dummy column
        data.x = torch.cat([data.x, torch.zeros((data.num_nodes, 1))], dim=1)
    elif data.x.shape[1] == 11:
        print("✅ Data format is current (11 channels).")
        # Check flag at index 10
        if data.x[0, -1] == 1.0:
            print("   -> Mode: Non-Newtonian (Carreau)")
            phys_cfg.viscosity_model = "carreau"
        else:
            print("   -> Mode: Newtonian")
            phys_cfg.viscosity_model = "newtonian"

    # 3. Geometric Quality
    print("\n--- Geometric Quality Check ---")
    cond_nums, mask_valid = analyze_geometric_quality(data)

    valid_cond = cond_nums[mask_valid]
    print(f"Nodes with < 5 neighbors: {len(cond_nums) - len(valid_cond)} (Excluded from stats)")
    print(f"Mean Condition No (Valid):  {np.mean(valid_cond):.2e}")
    print(f"Max Condition No (Valid):   {np.max(valid_cond):.2e}")

    if np.max(valid_cond) > 1e6:
        print(" WARNING: Internal mesh instability detected.")
    else:
        print(" Internal mesh quality is excellent.")

    # 4. Physics Residuals
    if hasattr(data, 'y') and data.y is not None:
        print("\n--- Physics Residual Check ---")
        kernels = PhysicsKernels(phys_cfg=phys_cfg)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        data_gpu = data.clone().to(device)

        with torch.no_grad():
            res_ns = kernels.navier_stokes_residual(data_gpu.y, data_gpu)
            res_bc = kernels.boundary_condition_loss(data_gpu.y, data_gpu)
            res_io = kernels.inlet_outlet_loss(data_gpu.y, data_gpu)
            print(f"Navier-Stokes Residual: {res_ns.item():.6e}")
            print(f"BC Loss (Wall):         {res_bc.item():.6e}")
            print(f"Inlet/Outlet Loss:      {res_io.item():.6e}")

    # 5. Visualization (Restored 6-Panel Layout)
    pos = data.x[:, :2].numpy()
    sdf_val = data.x[:, 2].numpy()
    shear_val = data.x[:, 3].numpy()

    # 2 rows, 3 columns. Constrained layout prevents overlap.
    fig = plt.figure(figsize=(20, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    # --- Plot 1: Geometric Stability ---
    ax1 = fig.add_subplot(gs[0, 0])
    cmap = plt.cm.viridis
    cmap.set_bad(color='lightgrey')  # Masked nodes appear grey
    cond_viz = np.log10(cond_nums + 1)
    # Apply mask for visualization (optional, or just rely on stats)
    # Here we show all, but you can see the 'grey' boundaries if masked manually
    sc1 = ax1.scatter(pos[:, 0], pos[:, 1], c=cond_viz, cmap=cmap, s=5)
    plt.colorbar(sc1, ax=ax1, label="Log10(Cond No)", fraction=0.046, pad=0.04)
    ax1.set_title("Geometric Quality")
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
        ax4.set_title("Ground Truth: U")
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
    mask_inlet = data.mask_inlet.numpy()
    mask_outlet = data.mask_outlet.numpy()
    mask_wall = data.mask_wall.numpy()

    # Plot background
    ax6.scatter(pos[:, 0], pos[:, 1], c='whitesmoke', s=15, marker='s')
    # Overlay masks
    ax6.scatter(pos[mask_inlet, 0], pos[mask_inlet, 1], c='green', s=10, label='Inlet')
    ax6.scatter(pos[mask_outlet, 0], pos[mask_outlet, 1], c='red', s=10, label='Outlet')
    ax6.scatter(pos[mask_wall, 0], pos[mask_wall, 1], c='black', s=5, label='Wall')

    ax6.set_title("Boundary Masks")
    ax6.legend(loc='upper right', fontsize='small')
    ax6.set_aspect('equal')

    plt.show()


if __name__ == "__main__":
    inspect_sample("vessel_0.pt")