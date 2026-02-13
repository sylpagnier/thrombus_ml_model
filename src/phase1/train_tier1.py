import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from src.phase1.utils.ginodeq import rGINO_DEQ
from src.phase1.utils.physics_kernels import PhysicsKernels
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Sampler
import random


class StratifiedAnchorSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size

        # FIX: Check the actual boolean flag stored in the Data object
        self.anchor_indices = []
        self.physics_indices = []

        for i, d in enumerate(dataset):
            # is_anchor is a tensor([True]) or tensor([False])
            if hasattr(d, 'is_anchor') and d.is_anchor.item() is True:
                self.anchor_indices.append(i)
            else:
                self.physics_indices.append(i)

        # Safety Check
        if not self.anchor_indices or not self.physics_indices:
            raise ValueError(
                f"Dataset split failed. Anchors: {len(self.anchor_indices)}, "
                f"Physics: {len(self.physics_indices)}. Ensure you have some "
                "unlabeled graphs in your data folder."
            )

        self.num_batches = len(dataset) // batch_size
        self.num_anchors_per_batch = batch_size // 2
        self.num_physics_per_batch = batch_size - self.num_anchors_per_batch

    def __iter__(self):
        for _ in range(self.num_batches):
            batch_indices = []
            # Sample 50% from anchors and 50% from physics sets
            batch_indices.extend(random.sample(self.anchor_indices, self.num_anchors_per_batch))
            batch_indices.extend(random.sample(self.physics_indices, self.num_physics_per_batch))
            random.shuffle(batch_indices)
            yield from batch_indices

    def __len__(self):
        return self.num_batches * self.batch_size


def validate_and_plot(model, val_data, epoch, device):
    model.eval()
    with torch.no_grad():
        data_on_device = Batch.from_data_list([val_data]).to(device)
        pred = model(data_on_device)
        # Use first 2 columns of x (which are nodes_nd) for plotting
        coords = data_on_device.x[:, :2].cpu().numpy()
        u_pred = pred[:, 0].cpu().numpy()

    plt.figure(figsize=(10, 4))
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=u_pred, cmap='jet', s=5)
    plt.colorbar(sc, label="Predicted ND-Velocity (u)")
    plt.title(f"Tier 1 Validation - Epoch {epoch}")
    plt.axis('equal')

    save_dir = Path("reports/figures")
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


def train_tier1(epochs=50, lr=5e-5, warm_up_epochs=20):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Encoder now takes 4 channels: [x_nd, y_nd, sdf_nd, shear_pot_nd]
    model = rGINO_DEQ(in_channels=4, latent_dim=64, max_iters=15).to(device)

    target_re = 150.0
    kernels = PhysicsKernels(reynolds=target_re)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    # 1. Load Dataset
    dataset = load_dataset()
    if not dataset:
        print("Error: Dataset is empty.")
        return

    train_size = int(0.9 * len(dataset))
    train_data, val_data = dataset[:train_size], dataset[train_size:]

    # 2. Initialize Stratified Loader (Fixes the Unresolved Reference)
    # batch_size=4 to maintain the 50/50 anchor split
    sampler = StratifiedAnchorSampler(train_data, batch_size=4)
    loader = DataLoader(train_data, batch_size=4, sampler=sampler)

    for epoch in range(epochs):
        model.train()

        # Physics Warm-Up Logic
        physics_active = epoch >= warm_up_epochs
        lambda_phys = 1.0 if physics_active else 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [Re={target_re}]")

        for batch_idx, data in enumerate(pbar):
            data = data.to(device)
            optimizer.zero_grad()

            # --- DEBUG BLOCK START ---
            if epoch == 0 and batch_idx == 0:  # Run this check once
                print(f"DEBUG: Batch has anchors? {data.is_anchor.any()}")
                if data.is_anchor.any():
                    mask = data.is_anchor[data.batch]
                    print(f"DEBUG: Number of anchor nodes: {mask.sum()}")
                    print(f"DEBUG: Max Label Value (y): {data.y[mask].max()}")
                    print(f"DEBUG: Mean Label Value (y): {data.y[mask].mean()}")
            # --- DEBUG BLOCK END ---

            # Forward Pass
            pred = model(data)

            # Supervised Loss (L_data) - only for anchors
            l_data = torch.tensor(0.0, device=device)
            if hasattr(data, 'is_anchor'):
                # Expand the graph-level boolean to node-level boolean
                node_is_anchor = data.is_anchor[data.batch]

                if node_is_anchor.sum() > 0:
                    # Calculate MSE only on anchor nodes
                    l_data = F.mse_loss(pred[node_is_anchor], data.y[node_is_anchor])

            # Physics Losses
            l_ns = kernels.navier_stokes_residual(pred, data)
            l_bc = kernels.boundary_condition_loss(pred, data)
            l_io = kernels.inlet_outlet_loss(pred, data)

            # Calculate a simple roughness penalty
            # Penalize the difference between a node and its average neighbor
            row, col = data.edge_index
            # Calculate difference squared between connected nodes
            l_smoothness = torch.mean((pred[row] - pred[col]) ** 2)

            # Weighted Objective (λ_data=500.0)
            loss = (lambda_phys * l_ns + 10.0 * l_bc + 20.0 * l_io) + (10 * l_data) + (5 * l_smoothness)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            pbar.set_postfix({
                "L_total": f"{loss.item():.4f}",
                "Phys": "ON" if physics_active else "WARMUP"
            })

        if epoch % 5 == 0:
            validate_and_plot(model, val_data[0], epoch, device)

    torch.save(model.state_dict(), "utils/tier1_hybrid_backbone.pth")


if __name__ == "__main__":
    train_tier1()