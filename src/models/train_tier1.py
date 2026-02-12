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
    data_dir = current_script_dir.parent.parent / "data" / "processed" / "tier1_graphs"
    file_list = sorted(list(data_dir.glob("vessel_*.pt")))
    dataset = []
    print(f"📂 Loading {len(file_list)} graphs...")
    for f in tqdm(file_list):
        data = torch.load(f, weights_only=False)
        dataset.append(data)
    return dataset


def train_tier1(epochs=50, lr=1e-4):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = rGINO_DEQ(latent_dim=64, max_iters=15).to(device)

    # Initialize directly with Target Reynolds
    target_re = 150.0
    kernels = PhysicsKernels(reynolds=target_re)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    dataset = load_dataset()
    train_size = int(0.9 * len(dataset))
    train_data, val_data = dataset[:train_size], dataset[train_size:]
    loader = DataLoader(train_data, batch_size=4, shuffle=True)

    for epoch in range(epochs):
        # Reynolds Ramp logic removed; using target_re immediately
        kernels.Re = target_re
        model.train()

        pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [Re={target_re:.1f}]")
        for data in pbar:
            data = data.to(device)
            optimizer.zero_grad()
            pred = model(data)

            l_ns = kernels.navier_stokes_residual(pred, data)
            l_bc = kernels.boundary_condition_loss(pred, data)
            l_io = kernels.inlet_outlet_loss(pred, data)

            l_data = torch.tensor(0.0, device=device)
            if hasattr(data, 'y') and data.y is not None:
                l_data = F.mse_loss(pred, data.y)

            # Maintaining Stability Patch weights
            loss = (1.0 * l_ns + 10.0 * l_bc + 20.0 * l_io) + (500.0 * l_data)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Anchor": f"{l_data.item():.4f}"})

        if epoch % 5 == 0:
            validate_and_plot(model, val_data[0], epoch, device)

    torch.save(model.state_dict(), "models/tier1_hybrid_backbone.pth")


if __name__ == "__main__":
    train_tier1()