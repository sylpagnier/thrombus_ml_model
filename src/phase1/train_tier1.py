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
    print(f"📂 Loading Tier 1 graphs from {cfg.graph_output_dir}...")
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
                    # 1. Standard L2/MSE Loss STRICTLY on Kinematics (u, v, p)
                    # This prevents ground-truth noise from corrupting the viscosity sub-network
                    mse_loss = F.mse_loss(pred[node_is_anchor, :3], data.y[node_is_anchor, :3])

                    # 2. Sobolev H1 Regularization (Gradient-Enhanced Loss)
                    props = kernels._get_geometric_props(data)

                    u_pred, v_pred = pred[:, 0].unsqueeze(1), pred[:, 1].unsqueeze(1)

                    # Compute predicted spatial derivatives
                    c_u_pred = kernels._compute_derivatives(u_pred, props)  # [N, 5, 1]
                    c_v_pred = kernels._compute_derivatives(v_pred, props)

                    grad_u_pred = c_u_pred[:, :2, 0]  # [N, 2] (d/dx, d/dy)
                    grad_v_pred = c_v_pred[:, :2, 0]

                    # 🚀 OPTIMIZATION: Prevent graph tracking for ground truth
                    with torch.no_grad():
                        u_true, v_true = data.y[:, 0].unsqueeze(1), data.y[:, 1].unsqueeze(1)
                        c_u_true = kernels._compute_derivatives(u_true, props)
                        c_v_true = kernels._compute_derivatives(v_true, props)

                        grad_u_true = c_u_true[:, :2, 0].detach()
                        grad_v_true = c_v_true[:, :2, 0].detach()

                    # Compute Gradient MSE strictly on the anchor nodes
                    grad_mse = F.mse_loss(grad_u_pred[node_is_anchor], grad_u_true[node_is_anchor]) + \
                               F.mse_loss(grad_v_pred[node_is_anchor], grad_v_true[node_is_anchor])

                    # 🚀 DYNAMIC WEIGHTING: Anneal alpha_sobolev from 0.0 to 0.5
                    # Start enforcing gradients only after basic physics kicks in
                    alpha_sobolev = min(0.5, max(0.0, (epoch - warm_up_epochs) / 30.0))

                    l_data = mse_loss + (alpha_sobolev * grad_mse)

            l_ns = kernels.navier_stokes_residual(pred, data)
            l_bc = kernels.boundary_condition_loss(pred, data)
            l_io = kernels.inlet_outlet_loss(pred, data)
            l_mu_dummy = torch.mean((pred[:, 3] - 1.0) ** 2)

            loss = (lambda_phys * l_ns + 5 * l_bc + 5 * l_io) + (5.0 * l_data) + l_mu_dummy

            # --- NaN Check ---
            if torch.isnan(loss):
                print(f"\n⚠️ NaN detected in loss at epoch {epoch}, batch {batch_idx}! Skipping batch.")
                continue

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss_epoch += loss.item()

            # --- Enhanced Postfix Tracking ---
            pbar.set_postfix({
                "L_tot": f"{loss.item():.3f}",
                "L_data": f"{l_data.item():.3f}",
                "L_ns": f"{l_ns.item():.3f}",
                "|g|": f"{grad_norm:.2f}",  # Track gradient norm
                "LR": f"{optimizer.param_groups[0]['lr']:.2e}"
            })

        scheduler.step()

        if epoch % 2 == 0:
            scores = quantify_performance(model, val_loader, kernels, device, tier="tier1")

            # --- DEBUGGING: Restore robust validation prints ---
            print(f"\n📊 [Validation] Rel L2: {scores.get('rel_l2', 0):.4f} | "
                  f"Div: {scores.get('continuity', 0):.3e} | "
                  f"Wall Slip: {scores.get('wall_slip', 0):.4f}")

            phys_score = scores.get('rel_l2', 0) + scores.get('continuity', 0)

            if phys_score < best_phys_score and physics_active:
                best_phys_score = phys_score
                root = get_project_root()
                torch.save(model.state_dict(), root / "models/tier1_best_physics.pth")
                print("⭐ Saved Best Physics Model")

        # --- Restore best loss saving logic ---
        avg_loss = total_loss_epoch / len(loader)
        if avg_loss < best_loss and physics_active:
            best_loss = avg_loss
            root = get_project_root()
            torch.save(model.state_dict(), root / "models/tier1_best_loss.pth")

        if epoch % 5 == 0:
            validate_and_plot(model, val_data[0], epoch, device, tier="tier1")

    # --- Final completion print ---
    print(f"Tier 1 Training Complete. Best Physical Score: {best_phys_score:.4f} | Best Loss: {best_loss:.4f}")


if __name__ == "__main__":
    train_tier1()