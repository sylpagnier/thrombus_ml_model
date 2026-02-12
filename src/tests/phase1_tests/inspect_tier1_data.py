import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torch_geometric.data import Data
# New import for physics verification
from src.phase1.utils.physics_kernels import PhysicsKernels


def inspect_sample(filename="vessel_0.pt"):
    # 1. Load Data
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    data_path = project_root / "data/processed/tier1_graphs" / filename

    if not data_path.exists():
        print(f"❌ File not found: {data_path}")
        files = list(data_path.parent.glob("*.pt"))
        if files:
            print(f"   -> Switching to found file: {files[0].name}")
            data_path = files[0]
        else:
            return

    print(f"--- Loading {data_path.name} ---")
    # Load with weights_only=False to ensure custom Data attributes are preserved
    data = torch.load(data_path, weights_only=False)

    # 2. Attribute Verification
    print("\n--- Attribute Check ---")
    available_attrs = data.keys()
    print(f"Fields found in Data object: {available_attrs}")
    print(f"Node Features (data.x) Shape: {data.x.shape}")
    print("   [0,1]: Pos (ND), [2]: SDF (ND), [3]: Shear Potential")

    has_y = hasattr(data, 'y') and data.y is not None
    if has_y:
        print(f"✅ 'y' (Ground Truth) found. Shape: {data.y.shape}")
    else:
        print(f"❌ 'y' NOT FOUND. Training on anchors will result in 0.0 supervised loss.")

    # 3. Physics Kernel Sanity Check
    if has_y:
        print("\n--- Physics Sanity Check (GT vs. Kernels) ---")
        kernels = PhysicsKernels(reynolds=150.0)

        with torch.no_grad():
            res_ns = kernels.navier_stokes_residual(data.y, data)
            res_bc = kernels.boundary_condition_loss(data.y, data)

            mean_res = res_ns.mean().item()
            print(f"Mean Navier-Stokes Residual of COMSOL data: {mean_res:.6f}")
            print(f"Mean Boundary Condition Residual: {res_bc.item():.6f}")

            if mean_res > 5.0:
                print("⚠️  NOTE: High residual is expected if using 'Direct SPH' Laplacian.")
                print("   Ensure PhysicsKernels is updated to use 'Double Gradient' for accuracy.")

    # 4. Visualization
    # Extract features from data.x based on mesh_to_graph.py packing
    pos = data.x[:, :2].numpy()  # ND-Coordinates
    sdf_val = data.x[:, 2].numpy() # Extracted from Channel 2
    shear_val = data.x[:, 3].numpy() # Extracted from Channel 3

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3)

    # --- Plot 1: Mesh & Masks ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(pos[:, 0], pos[:, 1], c='lightgrey', s=1, alpha=0.5)

    # Highlight specific boundaries
    if hasattr(data, 'mask_inlet'):
        inlet = pos[data.mask_inlet]
        ax1.scatter(inlet[:, 0], inlet[:, 1], c='green', s=10, label='Inlet')
    if hasattr(data, 'mask_outlet'):
        outlet = pos[data.mask_outlet]
        ax1.scatter(outlet[:, 0], outlet[:, 1], c='red', s=10, label='Outlet')

    ax1.set_title("Mesh Boundaries")
    ax1.legend()
    ax1.axis('equal')

    # --- Plot 2: SDF ---
    ax2 = fig.add_subplot(gs[0, 1])
    sc2 = ax2.scatter(pos[:, 0], pos[:, 1], c=sdf_val, cmap='viridis', s=2)
    plt.colorbar(sc2, ax=ax2, label="ND-SDF")
    ax2.set_title("Signed Distance Function (from data.x[:, 2])")
    ax2.axis('equal')

    # --- Plot 3: Shear Potential ---
    ax3 = fig.add_subplot(gs[0, 2])
    sc3 = ax3.scatter(pos[:, 0], pos[:, 1], c=shear_val, cmap='plasma', s=2)
    plt.colorbar(sc3, ax=ax3, label="Shear Potential")
    ax3.set_title("Shear Rate Potential (from data.x[:, 3])")
    ax3.axis('equal')

    # --- Compare number of nodes in graph vs CFD ---
    if has_y:
        u_gt = data.y[:, 0].numpy()
        p_gt = data.y[:, 2].numpy()

        ax4 = fig.add_subplot(gs[1, 0])
        sc4 = ax4.scatter(pos[:, 0], pos[:, 1], c=u_gt, cmap='jet', s=2)
        plt.colorbar(sc4, ax=ax4, label="U Velocity")
        ax4.set_title("Ground Truth: U Velocity")
        ax4.axis('equal')

        ax5 = fig.add_subplot(gs[1, 1])
        sc5 = ax5.scatter(pos[:, 0], pos[:, 1], c=p_gt, cmap='coolwarm', s=2)
        plt.colorbar(sc5, ax=ax5, label="Pressure")
        ax5.set_title("Ground Truth: Pressure (Pinned)")
        ax5.axis('equal')

    plt.tight_layout()
    plt.show()

    # 1. Define Paths Robustly
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    pt_path = project_root / "data/processed/tier1_graphs/vessel_0.pt"
    npz_path = project_root / "data/raw/cfd_anchors/vessel_0.npz"

    print(f"DEBUG: Loading PT from {pt_path}")
    print(f"DEBUG: Loading NPZ from {npz_path}")

    # 2. Load Data
    pt_data = torch.load(pt_path, weights_only=False)  # weights_only=False for custom objects
    npz_data = np.load(npz_path)

    # 3. Check Alignment
    # (Rest of the diagnostic logic remains the same)
    print(f"Graph Nodes: {pt_data.x.shape[0]}")
    print(f"CFD Solution Points: {npz_data['x'].flatten().shape[0]}")


if __name__ == "__main__":
    inspect_sample("vessel_0.pt")