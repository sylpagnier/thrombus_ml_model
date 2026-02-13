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

# --- 1. Scheduler Import ---
from torch.optim.lr_scheduler import CosineAnnealingLR


class StratifiedAnchorSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.anchor_indices = []
        self.physics_indices = []

        for i, d in enumerate(dataset):
            if hasattr(d, 'is_anchor') and d.is_anchor.item() is True:
                self.anchor_indices.append(i)
            else:
                self.physics_indices.append(i)

        # Safety Check
        if not self.anchor_indices or not self.physics_indices:
            raise ValueError("Dataset split failed. Ensure you have both anchor and physics graphs.")

        self.num_batches = len(dataset) // batch_size
        self.num_anchors_per_batch = batch_size // 2
        self.num_physics_per_batch = batch_size - self.num_anchors_per_batch

    def __iter__(self):
        for _ in range(self.num_batches):
            batch_indices = []
            # Use random.choices to allow resampling if one set is smaller
            batch_indices.extend(random.choices(self.anchor_indices, k=self.num_anchors_per_batch))
            batch_indices.extend(random.choices(self.physics_indices, k=self.num_physics_per_batch))
            random.shuffle(batch_indices)
            yield from batch_indices

    def __len__(self):
        return self.num_batches * self.batch_size


def validate_and_plot(model, val_data, epoch, device, save_name="val_epoch"):
    model.eval()
    with torch.no_grad():
        data_on_device = Batch.from_data_list([val_data]).to(device)
        pred = model(data_on_device)
        coords = data_on_device.x[:, :2].cpu().numpy()
        u_pred = pred[:, 0].cpu().numpy()

    plt.figure(figsize=(10, 4))
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=u_pred, cmap='jet', s=5)
    plt.colorbar(sc, label="Predicted ND-Velocity (u)")
    plt.title(f"Tier 1 Validation - Epoch {epoch}")
    plt.axis('equal')

    save_dir = Path("reports/figures")
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f"{save_name}_{epoch}.png")
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


def train_tier1_v2(epochs=50, lr=1e-4, warm_up_epochs=10):
    # NOTE: Increased initial LR to 1e-4 because Scheduler will decay it
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = rGINO_DEQ(in_channels=4, latent_dim=64, max_iters=15).to(device)

    target_re = 150.0
    kernels = PhysicsKernels(reynolds=target_re)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    # --- 2. Initialize Scheduler ---
    # Decays LR from 1e-4 down to 1e-6 over the course of training
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    dataset = load_dataset()
    if not dataset: return

    train_size = int(0.9 * len(dataset))
    train_data, val_data = dataset[:train_size], dataset[train_size:]

    sampler = StratifiedAnchorSampler(train_data, batch_size=4)
    loader = DataLoader(train_data, batch_size=4, sampler=sampler)

    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()

        # Physics Warm-Up: Turn on earlier (Epoch 10) to give scheduler time to work
        physics_active = epoch >= warm_up_epochs
        lambda_phys = 1.0 if physics_active else 0.0

        total_loss_epoch = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [Re={target_re}]")

        for batch_idx, data in enumerate(pbar):
            data = data.to(device)
            optimizer.zero_grad()

            pred = model(data)

            # Supervised Loss
            l_data = torch.tensor(0.0, device=device)
            if hasattr(data, 'is_anchor'):
                node_is_anchor = data.is_anchor[data.batch]
                if node_is_anchor.sum() > 0:
                    l_data = F.mse_loss(pred[node_is_anchor], data.y[node_is_anchor])

            # Physics Losses
            l_ns = kernels.navier_stokes_residual(pred, data)
            l_bc = kernels.boundary_condition_loss(pred, data)
            l_io = kernels.inlet_outlet_loss(pred, data)

            # Smoothness Penalty
            row, col = data.edge_index
            l_smoothness = torch.mean((pred[row] - pred[col]) ** 2)

            # --- 3. WEIGHT TUNING ---
            # Reduced l_data (10 -> 5.0) to prevent overfitting noise
            # Reduced l_smoothness (5 -> 2.0) to allow parabolic curves
            # Kept l_ns strong (1.0)
            loss = (lambda_phys * l_ns + 10.0 * l_bc + 20.0 * l_io) + \
                   (5.0 * l_data) + (2.0 * l_smoothness)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss_epoch += loss.item()

            pbar.set_postfix({
                "L_total": f"{loss.item():.4f}",
                "LR": f"{optimizer.param_groups[0]['lr']:.2e}"
            })

        # --- 4. Step Scheduler ---
        scheduler.step()

        # Save Best Model (Checkpointing)
        avg_loss = total_loss_epoch / len(loader)
        if avg_loss < best_loss and physics_active:
            best_loss = avg_loss
            torch.save(model.state_dict(), "models/tier1_best.pth")

        if epoch % 5 == 0:
            validate_and_plot(model, val_data[0], epoch, device)

    # Save final
    torch.save(model.state_dict(), "models/tier1_final.pth")
    print(f"Training Complete. Best Loss: {best_loss:.4f}")


if __name__ == "__main__":
    train_tier1_v2()