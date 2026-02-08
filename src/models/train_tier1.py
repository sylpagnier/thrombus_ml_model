import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from src.models.ginodeq import rGINO_DEQ
from src.utils.physics_kernels import PhysicsKernels
import os
import glob
from pathlib import Path
from tqdm import tqdm


def validate_and_plot(model, val_data, epoch, device):
    model.eval()
    with torch.no_grad():
        if val_data.x.dim() == 2:
            data_on_device = val_data.to(device)
            pred = model(data_on_device)
            u_pred = pred[:, 0].cpu().numpy()
            coords = val_data.x.cpu().numpy()
        else:
            pass

    plt.figure(figsize=(10, 4))
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=u_pred, cmap='jet', s=5)
    plt.colorbar(sc, label="Predicted ND-Velocity (u)")
    plt.title(f"Tier 1 Validation - Epoch {epoch} | Newtonian Flow")
    plt.axis('equal')

    project_root = Path(__file__).resolve().parent.parent.parent
    save_dir = project_root / "reports" / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f"val_epoch_{epoch}.png")
    plt.close()


def load_dataset():
    current_script_dir = Path(__file__).resolve().parent
    data_dir = current_script_dir.parent / "data_gen" / "data" / "processed" / "tier1_graphs"
    if not data_dir.exists():
        project_root = current_script_dir.parent.parent
        data_dir = project_root / "data" / "processed" / "tier1_graphs"

    file_list = sorted(list(data_dir.glob("vessel_*.pt")))
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
    print(f"--- Starting Tier 1 Training on {device.upper()} ---")

    # Keep batch size small for CPU safety
    model = rGINO_DEQ(latent_dim=64, max_iters=10).to(device)
    kernels = PhysicsKernels(reynolds=150.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    try:
        dataset = load_dataset()
    except FileNotFoundError as e:
        print(e);
        return

    train_size = int(0.9 * len(dataset))
    train_data, val_data = dataset[:train_size], dataset[train_size:]
    loader = DataLoader(train_data, batch_size=4, shuffle=True)
    example_val = val_data[0] if len(val_data) > 0 else None

    print(f"--- Dataset Loaded: {len(dataset)} samples ---")

    for epoch in range(epochs):
        model.train()
        total_loss, total_ns, total_bc, total_io = 0, 0, 0, 0

        pbar = tqdm(loader, desc=f"Epoch {epoch:02d}", unit="batch")

        for batch_idx, data in enumerate(pbar):
            data = data.to(device)
            optimizer.zero_grad()

            pred = model(data)

            # 1. Physics Loss
            l_ns = kernels.navier_stokes_residual(pred, data)

            # 2. Wall Boundary (No-Slip)
            l_bc = kernels.boundary_condition_loss(pred, data)

            # 3. Inlet/Outlet (The "Pump") - CRITICAL ADDITION
            l_io = kernels.inlet_outlet_loss(pred, data)

            # Weighted Sum:
            # High penalty for Inlet/Outlet (20.0) to drive flow.
            # Medium penalty for Walls (10.0).
            # Low penalty for internal physics (1.0) until boundaries are fixed.
            loss = 1.0 * l_ns + 10.0 * l_bc + 20.0 * l_io

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_ns += l_ns.item()
            total_bc += l_bc.item()
            total_io += l_io.item()

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(loader)

        # Validation Hook
        if epoch % 5 == 0 and example_val:
            validate_and_plot(model, example_val, epoch, device)
            print(f"   >>> Epoch {epoch} | NS: {total_ns / len(loader):.5f} | Pump: {total_io / len(loader):.5f}")

    script_dir = Path(__file__).resolve().parent
    model_dir = script_dir.parent.parent / "models"
    model_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_dir / "tier1_newtonian_backbone.pth")
    print(f"✅ Training Complete.")


if __name__ == "__main__":
    train_tier1()