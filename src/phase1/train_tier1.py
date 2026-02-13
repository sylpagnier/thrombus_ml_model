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


def quantify_performance(model, val_loader, kernels, device):
    """
    Computes PIRON-specific metrics:
    1. Rel L2 Error (on Anchors)
    2. Continuity Residual (Mass Conservation)
    3. Max Wall Slip (BC Violations)
    """
    model.eval()
    metrics = {
        "rel_l2": [],
        "continuity": [],
        "wall_slip": []
    }

    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            pred = model(data)

            # 1. Relative L2 Error (only for nodes with labels/anchors)
            if hasattr(data, 'is_anchor'):
                node_mask = data.is_anchor[data.batch]
                if node_mask.any():
                    diff_norm = torch.norm(pred[node_mask] - data.y[node_mask], p=2)
                    target_norm = torch.norm(data.y[node_mask], p=2)
                    rel_l2 = diff_norm / (target_norm + 1e-8)
                    metrics["rel_l2"].append(rel_l2.item())

            # 2. Continuity Residual (Mass Conservation: div(u) = 0)
            props = kernels._get_geometric_props(data)
            u, v = pred[:, 0:1], pred[:, 1:2]
            grad_u = kernels._compute_gradients(u, props)
            grad_v = kernels._compute_gradients(v, props)
            div_u = grad_u[:, 0:1] + grad_v[:, 1:2]
            metrics["continuity"].append(torch.abs(div_u).mean().item())

            # 3. Wall Slip Violation (Velocity magnitude at walls)
            if data.mask_wall.any():
                wall_vel = torch.norm(pred[data.mask_wall, :2], p=2, dim=1)
                metrics["wall_slip"].append(wall_vel.mean().item())

    return {k: np.mean(v) if v else 0.0 for k, v in metrics.items()}


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


def train_tier1(epochs=50, lr=1e-4, warm_up_epochs=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = rGINO_DEQ(in_channels=4, latent_dim=64, max_iters=15).to(device)

    target_re = 150.0
    kernels = PhysicsKernels(reynolds=target_re)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    dataset = load_dataset()
    if not dataset: return

    train_size = int(0.9 * len(dataset))
    train_data, val_data = dataset[:train_size], dataset[train_size:]

    sampler = StratifiedAnchorSampler(train_data, batch_size=4)
    loader = DataLoader(train_data, batch_size=4, sampler=sampler)
    val_loader = DataLoader(val_data, batch_size=4, shuffle=False)

    best_phys_score = float('inf')
    best_loss = float('inf')

    # Ensure model directory exists
    Path("models").mkdir(exist_ok=True)

    for epoch in range(epochs):
        model.train()
        physics_active = epoch >= warm_up_epochs
        lambda_phys = min(1.0, max(0.0, (epoch - warm_up_epochs) / 20.0))
        total_loss_epoch = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [Re={target_re}]")
        for batch_idx, data in enumerate(pbar):
            data = data.to(device)
            optimizer.zero_grad()
            pred = model(data)

            # --- Loss Calculation ---
            l_data = torch.tensor(0.0, device=device)
            if hasattr(data, 'is_anchor'):
                node_is_anchor = data.is_anchor[data.batch]
                if node_is_anchor.sum() > 0:
                    l_data = F.mse_loss(pred[node_is_anchor], data.y[node_is_anchor])

            l_ns = kernels.navier_stokes_residual(pred, data)
            l_bc = kernels.boundary_condition_loss(pred, data)
            l_io = kernels.inlet_outlet_loss(pred, data)
            row, col = data.edge_index
            l_smoothness = torch.mean((pred[row] - pred[col]) ** 2)

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

        # --- 4. Step Scheduler & Post-Epoch Evaluation ---
        scheduler.step()

        # Quantitative Metrics every 2 epochs
        if epoch % 2 == 0:
            scores = quantify_performance(model, val_loader, kernels, device)
            print(f"\n📊 [Validation] Rel L2: {scores['rel_l2']:.4f} | Div Res: {scores['continuity']:.3e} | Wall Slip: {scores['wall_slip']:.4f}")

            # Checkpoint 1: Best "Physical Health" (Focus on consistency)
            phys_score = scores['rel_l2'] + scores['continuity']
            if phys_score < best_phys_score and physics_active:
                best_phys_score = phys_score
                torch.save(model.state_dict(), "models/tier1_best_physics.pth")
                print("⭐ Saved Best Physics Model")

        # Checkpoint 2: Best "Training Loss" (Focus on objective minimization)
        avg_loss = total_loss_epoch / len(loader)
        if avg_loss < best_loss and physics_active:
            best_loss = avg_loss
            torch.save(model.state_dict(), "models/tier1_best_loss.pth")

        # Visualization every 5 epochs
        if epoch % 5 == 0:
            validate_and_plot(model, val_data[0], epoch, device)

    # Save final weights
    torch.save(model.state_dict(), "models/tier1_final.pth")
    print(f"Training Complete. Best Physical Score: {best_phys_score:.4f} | Best Loss: {best_loss:.4f}")


if __name__ == "__main__":
    train_tier1()