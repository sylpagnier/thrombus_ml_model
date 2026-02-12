import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torch_geometric.data import Data
# New import for physics verification
from src.utils.physics_kernels import PhysicsKernels


def inspect_sample(filename="vessel_0.pt"):
    # 1. Load Data
    project_root = Path(__file__).resolve().parent.parent.parent
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

    # 2. Attribute Verification (The "0.0000 Loss" Debugger)
    print("\n--- Attribute Check ---")
    available_attrs = data.keys()
    print(f"Fields found in Data object: {available_attrs}")

    has_y = hasattr(data, 'y') and data.y is not None
    if has_y:
        print(f"✅ 'y' (Ground Truth) found. Shape: {data.y.shape}")
    else:
        print(f"❌ 'y' NOT FOUND. Training on anchors will result in 0.0 supervised loss.")

    # 3. Physics Kernel Sanity Check
    # If we have GT data, the physics residual should be near zero.
    if has_y:
        print("\n--- Physics Sanity Check (GT vs. Kernels) ---")
        # Initialize kernels at your Target Re
        kernels = PhysicsKernels(reynolds=150.0)

        # We treat the Ground Truth (data.y) as the "prediction"
        # data.y contains [u, v, p] based on mesh_to_graph.py
        with torch.no_grad():
            res_ns = kernels.navier_stokes_residual(data.y, data)
            res_bc = kernels.boundary_condition_loss(data.y, data)

            mean_res = res_ns.mean().item()
            print(f"Mean Navier-Stokes Residual of COMSOL data: {mean_res:.6f}")
            print(f"Mean Boundary Condition Residual: {res_bc.item():.6f}")

            if mean_res > 1.0:
                print("⚠️  WARNING: High residual on ground truth data!")
                print("   This suggests a unit mismatch between COMSOL and the physics kernels.")
                print("   Check: Pressure scaling (1/rho), viscosity (mu), or coordinate normalization.")
            else:
                print("✅ Kernels are consistent with COMSOL data units.")

    # 4. Visualization (Existing Functionality)
    pos = data.x[:, :2].numpy()  # ND-Coordinates

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
    sc2 = ax2.scatter(pos[:, 0], pos[:, 1], c=data.sdf.numpy().flatten(), cmap='viridis', s=2)
    plt.colorbar(sc2, ax=ax2, label="ND-SDF")
    ax2.set_title("Signed Distance Function")
    ax2.axis('equal')

    # --- Plot 3: Shear Potential ---
    ax3 = fig.add_subplot(gs[0, 2])
    sc3 = ax3.scatter(pos[:, 0], pos[:, 1], c=data.shear_pot.numpy().flatten(), cmap='plasma', s=2)
    plt.colorbar(sc3, ax=ax3, label="Shear Potential")
    ax3.set_title("Shear Rate Potential (1/SDF)")
    ax3.axis('equal')

    # --- Plot 4 & 5: CFD Ground Truth (If exists) ---
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


if __name__ == "__main__":
    # Test on a specific anchor if known, otherwise default to vessel_0.pt
    inspect_sample("vessel_0.pt")