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
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from src.phase1.utils.samplers import StratifiedAnchorSampler
from src.phase1.utils.metrics import quantify_performance, validate_and_plot
import random


class DynamicLossWeighter(nn.Module):
    """
    Dynamically weights multiple loss components using homoscedastic task uncertainty.
    Includes a critical clamp to prevent the variance from collapsing to negative infinity
    as PDE residuals approach zero.
    Reference: Kendall et al., 2018.
    """

    def __init__(self, num_losses=2, min_log_var=-8.0):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_losses))
        self.min_log_var = min_log_var

    def forward(self, losses, scales=None):
        if scales is None:
            scales = [1.0] * len(losses)
        total_loss = 0
        for i, loss in enumerate(losses):
            if loss > 0.0:
                safe_log_var = torch.clamp(self.log_vars[i], min=self.min_log_var)
                precision = torch.exp(-safe_log_var)
                task_loss = precision * loss + safe_log_var
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


def compute_step_loss(model, data, kernels, loss_weighter, current_solver, lambda_phys, device):
    """Extracted closure logic to support both AdamW and L-BFGS"""

    # --- UPDATE: Unpack the tuple if the model returns the Jacobian loss ---
    out = model(data, solver=current_solver, anderson_beta=0.8)
    if isinstance(out, tuple):
        pred, jac_loss = out
    else:
        pred = out
        jac_loss = torch.tensor(0.0, device=device)

    l_data = torch.tensor(0.0, device=device)
    if hasattr(data, 'is_anchor'):
        node_is_anchor = data.is_anchor[data.batch]
        if node_is_anchor.sum() > 0:
            l_data = F.mse_loss(pred[node_is_anchor, :3], data.y[node_is_anchor, :3])

    l_cont, l_mom = kernels.navier_stokes_residual(pred, data)
    l_bc = kernels.boundary_condition_loss(pred, data)
    l_io = kernels.inlet_outlet_loss(pred, data)
    l_mu_dummy = torch.mean((pred[:, 3] - 1.0) ** 2)

    pde_losses = [l_cont, l_mom]
    pde_scales = [lambda_phys, lambda_phys]

    weighted_pde_loss = loss_weighter(pde_losses, scales=pde_scales)

    loss = weighted_pde_loss + (5.0 * l_data) + (5.0 * l_bc) + (5.0 * l_io) + l_mu_dummy + (0.1 * jac_loss)

    metrics = {
        "L_data": l_data.item(),
        "L_mom": l_mom.item(),
        "L_cont": l_cont.item(),
        "L_jac": jac_loss.item()
    }

    return loss, metrics

def train_tier1(epochs=125, lr=1e-4, warm_up_epochs=10, adam_epochs=100):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GINO_DEQ(in_channels=13, out_channels=4, latent_dim=64, max_iters=15).to(device)

    phys_cfg = PhysicsConfig(tier="tier1", re_target=150.0)
    kernels = PhysicsKernels(phys_cfg=phys_cfg)

    loss_weighter = DynamicLossWeighter(num_losses=2).to(device)

    # 1. Initialize Phase 1 Optimizer (AdamW)
    optimizer = optim.AdamW(list(model.parameters()) + list(loss_weighter.parameters()),
                            lr=lr, weight_decay=1e-5)

    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warm_up_epochs)
    decay_epochs = adam_epochs - warm_up_epochs # Adjust decay to fit AdamW phase
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=decay_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warm_up_epochs])

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
    lbfgs_initialized = False

    for epoch in range(epochs):
        model.train()
        physics_active = epoch >= warm_up_epochs
        sampler.set_warmup_mode(not physics_active)
        lambda_phys = min(1.0, max(0.0, (epoch - warm_up_epochs) / 20.0))
        total_loss_epoch = 0.0

        current_solver = "picard" if epoch < 5 else "anderson"

        if epoch >= adam_epochs and not lbfgs_initialized:
            print(f"\n⚡ Switching to L-BFGS Optimizer for the final {epochs - adam_epochs} epochs...")
            optimizer = optim.LBFGS(
                list(model.parameters()) + list(loss_weighter.parameters()),
                lr=0.01,
                max_iter=20,
                history_size=50,
                line_search_fn="strong_wolfe",
                tolerance_grad=1e-7,
                tolerance_change=1e-9
            )
            lbfgs_initialized = True

        if not lbfgs_initialized:
            # --- PHASE 1: AdamW Execution (Mini-Batch) ---
            pbar = tqdm(loader, desc=f"Tier 1 Epoch {epoch:02d} [Re={phys_cfg.re_target}] (AdamW)")
            for batch_idx, data in enumerate(pbar):
                data = data.to(device)

                optimizer.zero_grad()
                loss, metrics = compute_step_loss(model, data, kernels, loss_weighter, current_solver, lambda_phys,
                                                  device)

                if torch.isnan(loss):
                    print(f"\n⚠️ NaN detected! Skipping batch.")
                    continue

                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss_epoch += loss.item()

                pbar.set_postfix({
                    "L_tot": f"{loss.item():.3f}",
                    "L_data": f"{metrics['L_data']:.3f}",
                    "L_mom": f"{metrics['L_mom']:.3f}",
                    "L_cont": f"{metrics['L_cont']:.3f}",
                    "L_jac": f"{metrics['L_jac']:.3f}",
                    "|g|": f"{grad_norm:.2f}",
                    "LR": f"{optimizer.param_groups[0]['lr']:.2e}"
                })

            scheduler.step()

        else:
            # --- PHASE 2: L-BFGS Execution (Full-Batch via Accumulation) ---
            print(f"⏳ Tier 1 Epoch {epoch:02d} [Re={phys_cfg.re_target}] (L-BFGS Line Search...)")

            def closure():
                optimizer.zero_grad()
                accumulated_loss = torch.tensor(0.0, device=device)

                # Loop through the entire dataset to build the full-batch gradient
                for closure_data in loader:
                    closure_data = closure_data.to(device)
                    loss, _ = compute_step_loss(model, closure_data, kernels, loss_weighter, current_solver,
                                                lambda_phys, device)

                    # Average the loss so the gradient magnitude is stable
                    loss = loss / len(loader)
                    loss.backward()

                    accumulated_loss += loss.detach()

                # Clip gradients over the accumulated full-batch
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                return accumulated_loss

            # Take exactly ONE step per epoch. This will internally call the closure
            # multiple times depending on the strong_wolfe line search.
            loss_tensor = optimizer.step(closure)

            # Scale it back up so the `avg_loss` calculation later in the script doesn't break
            total_loss_epoch = loss_tensor.item() * len(loader)

            print(f"✅ L-BFGS Step Complete. Accumulated Full-Batch Loss: {loss_tensor.item():.4f}")

        if epoch % 2 == 0:
            scores = quantify_performance(model, val_loader, kernels, device, tier="tier1")

            print(f"\n📊 [Validation] Rel L2: {scores.get('rel_l2', 0):.4f} | "
                  f"Div: {scores.get('continuity', 0):.3e} | "
                  f"Wall Slip: {scores.get('wall_slip', 0):.4f}")

            # Debugging the safely clamped learned weights
            with torch.no_grad():
                safe_vars = torch.clamp(loss_weighter.log_vars, min=loss_weighter.min_log_var)
                weights = torch.exp(-safe_vars)
                print(f"⚖️ Learned PDE Weights -> Cont: {weights[0]:.2f} | Mom: {weights[1]:.2f}")

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