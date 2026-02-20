# train_tier1.py
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from pathlib import Path
from tqdm import tqdm
from src.utils.paths import get_project_root
from src.phase1.physics.ginodeq import GINO_DEQ
from src.phase1.physics.physics_kernels import PhysicsKernels
from src.config import VesselConfig, PhysicsConfig
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.phase1.utils.samplers import StratifiedAnchorSampler
from src.phase1.utils.metrics import quantify_performance, validate_and_plot


def load_dataset():
    cfg = VesselConfig(tier="tier1")
    if not cfg.graph_output_dir.exists():
        return []
    dataset = []
    for f in tqdm(sorted(list(cfg.graph_output_dir.glob("vessel_*.pt")))):
        dataset.append(torch.load(f, weights_only=False))
    return dataset


def train_tier1(epochs=50, lr=1e-4, warm_up_epochs=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GINO_DEQ(in_channels=11, out_channels=4, latent_dim=64, max_iters=15).to(device)

    phys_cfg = PhysicsConfig(tier="tier1", re_target=150.0)
    kernels = PhysicsKernels(phys_cfg=phys_cfg)
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
    Path("models").mkdir(exist_ok=True)

    for epoch in range(epochs):
        model.train()
        physics_active = epoch >= warm_up_epochs
        lambda_phys = min(1.0, max(0.0, (epoch - warm_up_epochs) / 20.0))
        total_loss_epoch = 0.0

        current_solver = "picard" if epoch < 5 else "anderson"

        pbar = tqdm(loader, desc=f"Tier 1 Epoch {epoch:02d}")
        for data in pbar:
            data = data.to(device)
            optimizer.zero_grad()
            pred = model(data, solver=current_solver, anderson_beta=0.8)

            l_data = torch.tensor(0.0, device=device)
            if hasattr(data, 'is_anchor') and data.is_anchor[data.batch].sum() > 0:
                mask = data.is_anchor[data.batch]
                l_data = F.mse_loss(pred[mask, :3], data.y[mask, :3])

            l_ns = kernels.navier_stokes_residual(pred, data)
            l_bc = kernels.boundary_condition_loss(pred, data)
            l_io = kernels.inlet_outlet_loss(pred, data)

            row, col = data.edge_index
            l_smoothness = torch.mean((pred[row] - pred[col]) ** 2)
            l_mu_dummy = torch.mean((pred[:, 3] - 1.0) ** 2)

            loss = (lambda_phys * l_ns + 5 * l_bc + 5 * l_io) + (5.0 * l_data) + (.5 * l_smoothness) + l_mu_dummy
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss_epoch += loss.item()

        scheduler.step()

        if epoch % 2 == 0:
            # Pass the tier specifically so the unified function behaves correctly
            scores = quantify_performance(model, val_loader, kernels, device, tier="tier1")
            phys_score = scores['rel_l2'] + scores['continuity']

            if phys_score < best_phys_score and physics_active:
                best_phys_score = phys_score
                root = get_project_root()
                torch.save(model.state_dict(), root / "models/tier1_best_physics.pth")

        if epoch % 5 == 0:
            validate_and_plot(model, val_data[0], epoch, device, tier="tier1")


if __name__ == "__main__":
    train_tier1()