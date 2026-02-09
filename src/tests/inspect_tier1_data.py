import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torch_geometric.data import Data


def inspect_sample(filename="vessel_0.pt"):
    # 1. Load Data
    project_root = Path(__file__).resolve().parent.parent.parent
    data_path = project_root / "data/processed/tier1_graphs" / filename

    if not data_path.exists():
        print(f"❌ File not found: {data_path}")
        # Try to find any .pt file
        files = list(data_path.parent.glob("*.pt"))
        if files:
            print(f"   -> Switching to found file: {files[0].name}")
            data_path = files[0]
        else:
            return

    print(f"--- Loading {data_path.name} ---")
    data = torch.load(data_path, weights_only=False)

    # Extract Data
    pos = data.x[:, :2].numpy()  # ND-Coordinates
    masks = {
        "Inlet": data.mask_inlet.numpy(),
        "Outlet": data.mask_outlet.numpy(),
        "Wall": data.mask_wall.numpy(),
        "Fluid": ~(data.mask_inlet | data.mask_outlet | data.mask_wall).numpy()
    }

    # Setup Plot Grid
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3)

    # --- Plot 1: Boundary Masks (Critical for Bifurcations) ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(pos[masks["Fluid"], 0], pos[masks["Fluid"], 1], c='lightgrey', s=1, label="Fluid")
    ax1.scatter(pos[masks["Wall"], 0], pos[masks["Wall"], 1], c='black', s=5, label="Wall")
    ax1.scatter(pos[masks["Inlet"], 0], pos[masks["Inlet"], 1], c='red', s=15, label="Inlet")
    ax1.scatter(pos[masks["Outlet"], 0], pos[masks["Outlet"], 1], c='blue', s=15, label="Outlet")
    ax1.set_title("Boundary Masks (Check Bifurcation Tags!)")
    ax1.legend(loc='upper right')
    ax1.axis('equal')

    # --- Plot 2: SDF Field ---
    ax2 = fig.add_subplot(gs[0, 1])
    sc2 = ax2.scatter(pos[:, 0], pos[:, 1], c=data.sdf.numpy().flatten(), cmap='viridis', s=2)
    plt.colorbar(sc2, ax=ax2, label="ND-SDF")
    ax2.set_title("Signed Distance Function")
    ax2.axis('equal')

    # --- Plot 3: Shear Potential (Input Feature) ---
    ax3 = fig.add_subplot(gs[0, 2])
    sc3 = ax3.scatter(pos[:, 0], pos[:, 1], c=data.shear_pot.numpy().flatten(), cmap='plasma', s=2)
    plt.colorbar(sc3, ax=ax3, label="Shear Potential")
    ax3.set_title("Shear Rate Potential (1/SDF)")
    ax3.axis('equal')

    # --- Plot 4 & 5: CFD Ground Truth (Only if exists) ---
    if data.y is not None:
        u_gt = data.y[:, 0].numpy()
        p_gt = data.y[:, 2].numpy()

        ax4 = fig.add_subplot(gs[1, 0])
        sc4 = ax4.scatter(pos[:, 0], pos[:, 1], c=u_gt, cmap='jet', s=2)
        plt.colorbar(sc4, ax=ax4, label="U Velocity")
        ax4.set_title("Ground Truth: U Velocity")
        ax4.axis('equal')

        ax5 = fig.add_subplot(gs[1, 1])
        sc5 = ax5.scatter(pos[:, 0], pos[:, 1], c=p_gt, cmap='magma', s=2)
        plt.colorbar(sc5, ax=ax5, label="Pressure")
        ax5.set_title("Ground Truth: Pressure")
        ax5.axis('equal')
    else:
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.text(0.5, 0.5, "No CFD Labels (Physics Only Sample)", ha='center')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Change this index to check different samples (e.g., vessel_0.pt)
    # Try finding a bifurcation index if you generated 5000 samples
    inspect_sample("vessel_0.pt")