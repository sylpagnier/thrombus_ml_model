import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from pathlib import Path
from tqdm import tqdm
from src.utils.paths import get_project_root
from src.phase1.physics.ginodeq import GINO_DEQ
from src.phase1.physics.physics_kernels import PhysicsKernels
from src.config import VesselConfig, PhysicsConfig
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from src.phase1.utils.samplers import StratifiedAnchorSampler
from src.phase1.utils.metrics import quantify_performance, validate_and_plot
import random


class DynamicLossWeighter(nn.Module):
    """
    Dynamically weights multiple loss components using homoscedastic task uncertainty.
    Reference: Kendall et al., 2018 (Multi-Task Learning Using Uncertainty to Weigh Losses)
    """

    def __init__(self, num_losses=4):
        super().__init__()
        # Initialize log variances to 0 (which initializes the weight to 1.0)
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, losses, scales=None):
        # Default to a scale of 1.0 for all tasks if no scales are provided
        if scales is None:
            scales = [1.0] * len(losses)

        total_loss = 0
        for i, loss in enumerate(losses):
            # Safeguard: only apply to active losses to prevent log_vars from diverging
            if loss > 0.0:
                precision = torch.exp(-self.log_vars[i])

                # 1. Calculate the balanced task loss based on RAW uncertainty
                task_loss = precision * loss + self.log_vars[i]

                # 2. Apply the manual external scaling (e.g., warm-up schedules)
                total_loss += scales[i] * task_loss

        return total_loss


def load_dataset():
    cfg = VesselConfig(tier="tier1")
    if not cfg.graph_output_dir.exists():
        return []
    dataset = []
    print(f"📂 Loading Tier 1 graphs from {cfg.graph_output_dir}...")
    for f in tqdm(sorted(list(cfg.graph_output_dir.glob("vessel_*.pt")))):
        dataset.append(torch.load(f, weights_only=False))
    return dataset


def train_tier1(epochs=50, lr=1e-4, warm_up_epochs=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GINO_DEQ(in_channels=11, out_channels=4, latent_dim=64, max_iters=15).to(device)

    phys_cfg = PhysicsConfig(tier="tier1", re_target=150.0)
    kernels = PhysicsKernels(phys_cfg=phys_cfg)

    # Initialize the Dynamic Weighter for 4 losses: Data, Navier-Stokes, BC, Inlet/Outlet
    loss_weighter = DynamicLossWeighter(num_losses=4).to(device)

    # Pass BOTH the model parameters and the weighter parameters to the optimizer
    optimizer = optim.AdamW(list(model.parameters()) + list(loss_weighter.parameters()),
                            lr=lr, weight_decay=1e-5)

    # Replaced basic CosineAnnealing with Warm Restarts (restarts every 15 epochs)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=1, eta_min=1e-6)

    dataset = load_dataset()
    if not dataset: return

    # --- NEW STRATIFIED SPLIT LOGIC ---
    # 1. Separate graphs by type
    anchors = [d for d in dataset if d.is_anchor.item()]
    physics = [d for d in dataset if not d.is_anchor.item()]

    # 2. Shuffle to remove any generation-order bias
    random.seed(42)  # For reproducibility
    random.shuffle(anchors)
    random.shuffle(physics)

    # 3. Calculate 90% split bounds for BOTH classes
    split_idx_a = int(0.9 * len(anchors))
    split_idx_p = int(0.9 * len(physics))

    # 4. Combine them into train and val sets
    train_data = anchors[:split_idx_a] + physics[:split_idx_p]
    val_data = anchors[split_idx_a:] + physics[split_idx_p:]

    sampler = StratifiedAnchorSampler(train_data, batch_size=4)

    loader = DataLoader(train_data, batch_size=4, sampler=sampler)
    val_loader = DataLoader(val_data, batch_size=4, shuffle=False)

    best_phys_score = float('inf')
    best_loss = float('inf')
    Path("models").mkdir(exist_ok=True)

    for epoch in range(epochs):
        model.train()
        physics_active = epoch >= warm_up_epochs
        sampler.set_warmup_mode(not physics_active)
        lambda_phys = min(1.0, max(0.0, (epoch - warm_up_epochs) / 20.0))
        total_loss_epoch = 0.0

        current_solver = "picard" if epoch < 5 else "anderson"

        pbar = tqdm(loader, desc=f"Tier 1 Epoch {epoch:02d} [Re={phys_cfg.re_target}]")
        for batch_idx, data in enumerate(pbar):
            data = data.to(device)
            optimizer.zero_grad()
            pred = model(data, solver=current_solver, anderson_beta=0.8)

            l_data = torch.tensor(0.0, device=device)
            if hasattr(data, 'is_anchor'):
                node_is_anchor = data.is_anchor[data.batch]
                if node_is_anchor.sum() > 0:
                    mse_loss = F.mse_loss(pred[node_is_anchor, :3], data.y[node_is_anchor, :3])

                    props = kernels._get_geometric_props(data)
                    u_pred, v_pred = pred[:, 0].unsqueeze(1), pred[:, 1].unsqueeze(1)

                    c_u_pred = kernels._compute_derivatives(u_pred, props)
                    c_v_pred = kernels._compute_derivatives(v_pred, props)

                    grad_u_pred = c_u_pred[:, :2, 0]
                    grad_v_pred = c_v_pred[:, :2, 0]

                    with torch.no_grad():
                        u_true, v_true = data.y[:, 0].unsqueeze(1), data.y[:, 1].unsqueeze(1)
                        c_u_true = kernels._compute_derivatives(u_true, props)
                        c_v_true = kernels._compute_derivatives(v_true, props)

                        grad_u_true = c_u_true[:, :2, 0].detach()
                        grad_v_true = c_v_true[:, :2, 0].detach()

                    grad_mse = F.mse_loss(grad_u_pred[node_is_anchor], grad_u_true[node_is_anchor]) + \
                               F.mse_loss(grad_v_pred[node_is_anchor], grad_v_true[node_is_anchor])

                    # 🚀 CAPPED SOBOLEV WEIGHT: Reduced max from 0.5 to 0.1
                    alpha_sobolev = min(0.1, max(0.0, (epoch - warm_up_epochs) / 30.0))

                    l_data = mse_loss + (alpha_sobolev * grad_mse)

            l_ns = kernels.navier_stokes_residual(pred, data)
            l_bc = kernels.boundary_condition_loss(pred, data)
            l_io = kernels.inlet_outlet_loss(pred, data)
            l_mu_dummy = torch.mean((pred[:, 3] - 1.0) ** 2)

            #  DYNAMIC LOSS WEIGHTING
            # 1. Pass the RAW, unscaled losses to the weighter so it learns the true variance
            losses = [l_data, l_ns, l_bc, l_io]

            # 2. Apply the warm-up schedule strictly as a post-multiplier
            # Index 1 corresponds to l_ns
            scales = [1.0, lambda_phys, 1.0, 1.0]

            loss = loss_weighter(losses, scales=scales) + l_mu_dummy

            if torch.isnan(loss):
                print(f"\n⚠️ NaN detected in loss at epoch {epoch}, batch {batch_idx}! Skipping batch.")
                continue

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss_epoch += loss.item()

            pbar.set_postfix({
                "L_tot": f"{loss.item():.3f}",
                "L_data": f"{l_data.item():.3f}",
                "L_ns": f"{l_ns.item():.3f}",
                "|g|": f"{grad_norm:.2f}",
                "LR": f"{optimizer.param_groups[0]['lr']:.2e}"
            })

        scheduler.step()

        if epoch % 2 == 0:
            scores = quantify_performance(model, val_loader, kernels, device, tier="tier1")

            print(f"\n📊 [Validation] Rel L2: {scores.get('rel_l2', 0):.4f} | "
                  f"Div: {scores.get('continuity', 0):.3e} | "
                  f"Wall Slip: {scores.get('wall_slip', 0):.4f}")

            # Debugging the current learned weights
            with torch.no_grad():
                weights = torch.exp(-loss_weighter.log_vars)
                print(
                    f"⚖️ Learned Loss Weights -> Data: {weights[0]:.2f} | NS: {weights[1]:.2f} | BC: {weights[2]:.2f} | IO: {weights[3]:.2f}")

            phys_score = scores.get('rel_l2', 0) + scores.get('continuity', 0)

            if phys_score < best_phys_score and physics_active:
                best_phys_score = phys_score
                root = get_project_root()
                torch.save(model.state_dict(), root / "models/tier1_best_physics.pth")
                print("⭐ Saved Best Physics Model")

        avg_loss = total_loss_epoch / len(loader)
        if avg_loss < best_loss and physics_active:
            best_loss = avg_loss
            root = get_project_root()
            torch.save(model.state_dict(), root / "models/tier1_best_loss.pth")

        if epoch % 5 == 0:
            validate_and_plot(model, val_data[0], epoch, device, tier="tier1")

    print(f"Tier 1 Training Complete. Best Physical Score: {best_phys_score:.4f} | Best Loss: {best_loss:.4f}")


if __name__ == "__main__":
    train_tier1()