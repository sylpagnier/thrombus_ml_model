import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torch_geometric.data import Data
from src.phase1.physics.physics_kernels import PhysicsKernels


def analyze_wls_stability(data):
    """
    Analyzes the condition number of the WLS matrices for all nodes.
    High condition numbers (>1e5) indicate nearly collinear neighbors or
    sparse neighborhoods that will cause gradient blow-ups.
    """
    row, col = data.edge_index
    num_nodes = data.num_nodes
    # Coordinates are in the first two channels
    pos = data.x[:, :2]

    # Compute Geometric components identical to PhysicsKernels
    pos_diff = pos[col] - pos[row]
    dx, dy = pos_diff[:, 0], pos_diff[:, 1]
    dist_sq = dx ** 2 + dy ** 2 + 1e-8
    w = 1.0 / (dist_sq + 1e-8)

    # Replicate scatter_add logic to form the Gram Matrix M = [[m_xx, m_xy], [m_xy, m_yy]]
    def scatter_sum(src, index, num_nodes):
        out = torch.zeros(num_nodes, dtype=src.dtype, device=src.device)
        return out.scatter_add_(0, index, src)

    m_xx = scatter_sum(w * dx * dx, row, num_nodes)
    m_xy = scatter_sum(w * dx * dy, row, num_nodes)
    m_yy = scatter_sum(w * dy * dy, row, num_nodes)

    # Eigenvalue calculation for 2x2 matrix to find Condition Number
    # Det = L1 * L2, Trace = L1 + L2
    det = m_xx * m_yy - m_xy ** 2
    trace = m_xx + m_yy

    # Quadratic formula discriminant
    discriminant = torch.clamp(trace ** 2 - 4 * det, min=0)
    lambda_1 = (trace + torch.sqrt(discriminant)) / 2
    lambda_2 = (trace - torch.sqrt(discriminant)) / 2

    # Condition Number = Max Eigenvalue / Min Eigenvalue
    cond_numbers = lambda_1 / (lambda_2 + 1e-12)

    return cond_numbers.cpu().numpy()


def inspect_sample(filename="vessel_0.pt"):
    # 1. Load Data
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    data_dir = project_root / "data/processed/graphs"
    data_path = data_dir / filename

    if not data_path.exists():
        print(f"❌ File not found: {data_path}")
        files = sorted(list(data_dir.glob("*.pt")))
        if files:
            print(f"   -> Switching to found file: {files[0].name}")
            data_path = files[0]
        else:
            return

    print(f"--- Loading {data_path.name} ---")
    data = torch.load(data_path, weights_only=False)

    # 2. Attribute Verification
    print("\n--- Attribute Check ---")
    print(f"Fields found: {list(data.keys())}")
    print(f"Node Features (data.x) Shape: {data.x.shape}")

    has_y = hasattr(data, 'y') and data.y is not None
    print(f"{'✅' if has_y else '❌'} Ground Truth (y): {'Found' if has_y else 'NOT FOUND'}")

    # 3. Geometric Stability Analysis (New Functionality)
    print("\n--- Geometric Stability Check ---")
    cond_nums = analyze_wls_stability(data)
    max_cond = np.max(cond_nums)
    print(f"Mean Condition Number: {np.mean(cond_nums):.2f}")
    print(f"Max Condition Number:  {max_cond:.2e}")

    if max_cond > 1e5:
        print("⚠️  WARNING: High condition numbers detected. Check stability map.")

    # 4. Physics Kernel Sanity Check
    if has_y:
        print("\n--- Physics Sanity Check (COMSOL vs. Kernels) ---")
        kernels = PhysicsKernels(reynolds=150.0)
        with torch.no_grad():
            res_ns = kernels.navier_stokes_residual(data.y, data)
            res_bc = kernels.boundary_condition_loss(data.y, data)
            print(f"Mean NS Residual: {res_ns.mean().item():.6f}")
            print(f"BC Loss:          {res_bc.item():.6f}")

    # 5. Visualization
    pos = data.x[:, :2].numpy()
    sdf_val = data.x[:, 2].numpy()
    shear_val = data.x[:, 3].numpy()

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3)

    # Plot 1: Mesh & Stability
    ax1 = fig.add_subplot(gs[0, 0])
    sc1 = ax1.scatter(pos[:, 0], pos[:, 1], c=np.log10(cond_nums), cmap='inferno', s=3)
    plt.colorbar(sc1, ax=ax1, label="Log10(Condition Number)")
    ax1.set_title("Numerical Stability (Yellow=Unstable)")
    ax1.axis('equal')

    # Plot 2: SDF
    ax2 = fig.add_subplot(gs[0, 1])
    sc2 = ax2.scatter(pos[:, 0], pos[:, 1], c=sdf_val, cmap='viridis', s=2)
    plt.colorbar(sc2, ax=ax2, label="ND-SDF")
    ax2.set_title("Signed Distance Function")
    ax2.axis('equal')

    # Plot 3: Shear Potential
    ax3 = fig.add_subplot(gs[0, 2])
    sc3 = ax3.scatter(pos[:, 0], pos[:, 1], c=shear_val, cmap='plasma', s=2)
    plt.colorbar(sc3, ax=ax3, label="Shear Potential")
    ax3.set_title("Shear Rate Potential")
    ax3.axis('equal')

    if has_y:
        # Plot 4: U Velocity
        ax4 = fig.add_subplot(gs[1, 0])
        sc4 = ax4.scatter(pos[:, 0], pos[:, 1], c=data.y[:, 0].numpy(), cmap='jet', s=2)
        plt.colorbar(sc4, ax=ax4, label="U Velocity")
        ax4.set_title("Ground Truth: U")
        ax4.axis('equal')

        # Plot 5: Pressure
        ax5 = fig.add_subplot(gs[1, 1])
        sc5 = ax5.scatter(pos[:, 0], pos[:, 1], c=data.y[:, 2].numpy(), cmap='coolwarm', s=2)
        plt.colorbar(sc5, ax=ax5, label="Pressure")
        ax5.set_title("Ground Truth: P")
        ax5.axis('equal')

        # Plot 6: Alignment Check
        ax6 = fig.add_subplot(gs[1, 2])
        mask_inlet = data.mask_inlet.numpy()
        mask_outlet = data.mask_outlet.numpy()
        ax6.scatter(pos[:, 0], pos[:, 1], c='lightgrey', s=1, alpha=0.3)
        ax6.scatter(pos[mask_inlet, 0], pos[mask_inlet, 1], c='green', s=10, label='Inlet')
        ax6.scatter(pos[mask_outlet, 0], pos[mask_outlet, 1], c='red', s=10, label='Outlet')
        ax6.set_title("Boundary Masks Alignment")
        ax6.legend()
        ax6.axis('equal')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # You can change this to any vessel_idx.pt you want to check
    inspect_sample("vessel_1.pt")