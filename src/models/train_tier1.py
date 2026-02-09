import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from src.models.ginodeq import rGINO_DEQ
from src.utils.physics_kernels import PhysicsKernels
from pathlib import Path
from tqdm import tqdm


def validate_and_plot(model, val_data, epoch, device):
    """Generates and saves a validation plot for the current epoch."""
    model.eval()
    with torch.no_grad():
        if val_data.x.dim() == 2:
            data_on_device = Batch.from_data_list([val_data]).to(device)
            pred = model(data_on_device)
            u_pred = pred[:val_data.num_nodes, 0].cpu().numpy()
            coords = val_data.x.cpu().numpy()
        else:
            return

    plt.figure(figsize=(10, 4))
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=u_pred, cmap='jet', s=5)
    plt.colorbar(sc, label="Predicted ND-Velocity (u)")
    plt.title(f"Tier 1 Validation - Epoch {epoch}")
    plt.axis('equal')

    project_root = Path(__file__).resolve().parent.parent.parent
    save_dir = project_root / "reports" / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f"val_epoch_{epoch}.png")
    plt.close()


def load_dataset():
    current_script_dir = Path(__file__).resolve().parent
    # Assuming standard structure: src/train_tier1.py -> project_root/data/...
    data_dir = current_script_dir.parent.parent / "data" / "processed" / "tier1_graphs"

    file_list = sorted(list(data_dir.glob("vessel_*.pt")))
    if not file_list:
        raise FileNotFoundError(f"No .pt files found in {data_dir}. Run mesh_to_graph.py first.")

    dataset = []
    print(f"📂 Found {len(file_list)} files. Loading into memory...")
    for f in tqdm(file_list):
        try:
            data = torch.load(f, weights_only=False)
            dataset.append(data)
        except Exception:
            pass
    return dataset


def train_tier1(epochs=50, lr=1e-4):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Starting Tier 1 Hybrid Training on {device.upper()} ---")

    # Initialize Model and Kernels
    model = rGINO_DEQ(latent_dim=64, max_iters=15).to(device)
    kernels = PhysicsKernels(reynolds=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    try:
        dataset = load_dataset()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Split: 90% Train / 10% Val
    train_size = int(0.9 * len(dataset))
    train_data, val_data = dataset[:train_size], dataset[train_size:]
    loader = DataLoader(train_data, batch_size=4, shuffle=True)
    example_val = val_data[0] if len(val_data) > 0 else None

    # Curriculum Setup: Ramp Re from 1 to 150
    target_re = 150.0
    start_re = 1.0
    ramp_epochs = 20

    for epoch in range(epochs):
        # Update Reynolds Number
        current_re = start_re + (target_re - start_re) * min(1.0, epoch / ramp_epochs)
        kernels.Re = current_re

        model.train()
        total_loss = 0
        total_anchor = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [Re={current_re:.1f}]")
        for batch_idx, data in enumerate(pbar):
            data = data.to(device)
            optimizer.zero_grad()
            pred = model(data)

            # 1. Physics Residuals (Uses new Masks)
            l_ns = kernels.navier_stokes_residual(pred, data)
            l_bc = kernels.boundary_condition_loss(pred, data)
            l_io = kernels.inlet_outlet_loss(pred, data)

            # 2. Hybrid Supervised Loss (Applied only to Anchors)
            l_data = torch.tensor(0.0, device=device)
            # Check if 'y' exists and is not empty (Anchors have y)
            if hasattr(data, 'y') and data.y is not None:
                l_data = F.mse_loss(pred, data.y)
                total_anchor += l_data.item()

            # Weighted Objective
            # High weight on Data to force correct initial structure
            loss = (1.0 * l_ns + 10.0 * l_bc + 20.0 * l_io) + (100.0 * l_data)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Anchor": f"{l_data.item():.4f}"})

        # Periodically run validation
        if epoch % 5 == 0 and example_val:
            validate_and_plot(model, example_val, epoch, device)

    # Save Model
    script_dir = Path(__file__).resolve().parent
    model_dir = script_dir.parent.parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "tier1_hybrid_backbone.pth")
    print(f"✅ Training Complete. Model saved to {model_dir}")

if __name__ == "__main__":
    train_tier1()