import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from pathlib import Path
from tqdm import tqdm
import random
from src.phase1.physics.ginodeq import GINO_DEQ
from src.phase1.physics.physics_kernels import PhysicsKernels
from src.config import VesselConfig, PhysicsConfig
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from src.phase1.utils.samplers import StratifiedAnchorSampler
from src.phase1.utils.metrics import quantify_performance, validate_and_plot

class DynamicLossWeighter(nn.Module):
    """
    Dynamically weights multiple loss components using homoscedastic task uncertainty.
    """
    def __init__(self, num_losses=3):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, losses, scales=None):
        if scales is None:
            scales = [1.0] * len(losses)

        total_loss = 0
        for i, loss in enumerate(losses):
            if loss > 0.0:
                precision = torch.exp(-self.log_vars[i])
                task_loss = precision * loss + self.log_vars[i]
                total_loss += scales[i] * task_loss

        return total_loss

def load_dataset():
    cfg = VesselConfig(tier="tier2")
    data_dir = cfg.graph_output_dir

    if not data_dir.exists():
        print(f"Directory not found: {data_dir}. Please generate Tier 2 data first.")
        return []

    file_list = sorted(list(data_dir.glob("vessel_*.pt")))
    dataset = []
    print(f"📂 Loading {len(file_list)} Tier 2 graphs...")
    for f in tqdm(file_list):
        data = torch.load(f, weights_only=False)
        dataset.append(data)
    return dataset


def setup_distillation_phase(model):
    """Freezes kinematics. Trains ONLY the Viscosity sub-network."""
    print("❄️ Freezing Kinematics Backbone. Activating Viscosity Sub-network.")
    for param in model.encoder.parameters(): param.requires_grad = False
    for param in model.core.parameters(): param.requires_grad = False
    for param in model.kinematics_decoder.parameters(): param.requires_grad = False

    for param in model.mu_decoder.parameters(): param.requires_grad = True
    for param in model.mu_encoder.parameters(): param.requires_grad = True

    return optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-5)


def setup_coupled_phase(model, loss_weighter, base_lr=2e-5):
    """Unfreezes all layers for the final Operator Splitting DEQ phase."""
    print("🔥 Unfreezing All Layers. Activating Coupled DEQ Optimization.")
    for param in model.parameters(): param.requires_grad = True

    # Pass BOTH model and weighter parameters to the optimizer
    return optim.AdamW([
        {'params': model.parameters(), 'lr': base_lr},
        {'params': loss_weighter.parameters(), 'lr': 1e-3, 'weight_decay': 0.0}
    ], weight_decay=1e-5)


def train_tier2(epochs=60, distillation_epochs=15, lr=2e-5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GINO_DEQ(in_channels=11, out_channels=4, latent_dim=64, max_iters=15).to(device)

    # 1. Load Tier 1 (Newtonian) Weights
    tier1_path = Path("models/tier1_best_physics.pth")
    if tier1_path.exists():
        model.load_state_dict(torch.load(tier1_path, map_location=device, weights_only=True), strict=False)
        print("✅ Successfully loaded Tier 1 foundational physics weights.")
    else:
        print("⚠️ Warning: Tier 1 weights not found. Training Tier 2 from scratch will be unstable.")

    phys_cfg = PhysicsConfig(tier="tier2", re_target=150.0)
    kernels = PhysicsKernels(phys_cfg=phys_cfg)

    # Initialize Weighter for 3 internal physics PDEs: Cont, Mom, Rheology
    loss_weighter = DynamicLossWeighter(num_losses=3).to(device)

    dataset = load_dataset()
    if not dataset: return

    anchors = [d for d in dataset if d.is_anchor.item()]
    physics = [d for d in dataset if not d.is_anchor.item()]

    random.seed(42)
    random.shuffle(anchors)
    random.shuffle(physics)

    split_idx_a = int(0.9 * len(anchors))
    split_idx_p = int(0.9 * len(physics))

    train_data = anchors[:split_idx_a] + physics[:split_idx_p]
    val_data = anchors[split_idx_a:] + physics[split_idx_p:]

    sampler = StratifiedAnchorSampler(train_data, batch_size=4)
    loader = DataLoader(train_data, batch_size=4, sampler=sampler)
    val_loader = DataLoader(val_data, batch_size=4, shuffle=False)

    best_phys_score = float('inf')
    Path("models").mkdir(exist_ok=True)

    optimizer = None
    scheduler = None

    for epoch in range(epochs):
        is_distillation = epoch < distillation_epochs

        # --- CURRICULUM PHASE SWITCHING ---
        if epoch == 0:
            print(f"\n🚀 --- Starting Phase 1: Viscosity Distillation (Epochs 0-{distillation_epochs - 1}) ---")
            optimizer = setup_distillation_phase(model)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-5)
            sampler.set_warmup_mode(True)
        elif epoch == distillation_epochs:
            print(f"\n🚀 --- Starting Phase 2: Fully Coupled DEQ (Epochs {distillation_epochs}-{epochs - 1}) ---")
            optimizer = setup_coupled_phase(model, loss_weighter, base_lr=lr)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
            sampler.set_warmup_mode(False)

        model.train()
        total_loss_epoch = 0.0

        pbar = tqdm(loader, desc=f"Tier 2 Epoch {epoch:02d} [Re={phys_cfg.re_target}]")
        for batch_idx, data in enumerate(pbar):
            data = data.to(device)
            optimizer.zero_grad()

            # Default logging values
            l_cont = l_mom = l_rheo = torch.tensor(0.0, device=device)

            # --- PHASE 1: DISTILLATION ROUTING ---
            if is_distillation:
                pred = model(data, solver="picard", anderson_beta=1.0)

                l_data_mu = torch.tensor(0.0, device=device)
                if hasattr(data, 'is_anchor'):
                    node_is_anchor = data.is_anchor[data.batch]
                    if node_is_anchor.sum() > 0:
                        l_data_mu = F.mse_loss(pred[node_is_anchor, 3], data.y[node_is_anchor, 3])

                l_rheo = kernels.rheology_loss(pred, data)
                loss = 10.0 * l_rheo + 5.0 * l_data_mu

            # --- PHASE 2: FULLY COUPLED ROUTING ---
            else:
                pred = model(data, solver="anderson", anderson_beta=0.8)

                l_data = torch.tensor(0.0, device=device)
                if hasattr(data, 'is_anchor'):
                    node_is_anchor = data.is_anchor[data.batch]
                    if node_is_anchor.sum() > 0:
                        # Strictly MSE on primary state variables - NO Sobolev
                        l_data = F.mse_loss(pred[node_is_anchor, :4], data.y[node_is_anchor, :4])

                l_cont, l_mom = kernels.navier_stokes_residual(pred, data)
                l_bc = kernels.boundary_condition_loss(pred, data)
                l_io = kernels.inlet_outlet_loss(pred, data)
                l_rheo = kernels.rheology_loss(pred, data)

                lambda_phys = min(1.0, max(0.0, (epoch - distillation_epochs) / 20.0))

                # Dynamic Uncertainty Weighting specifically for internal PDE constraints
                pde_losses = [l_cont, l_mom, l_rheo]
                pde_scales = [lambda_phys, lambda_phys, 1.0]

                weighted_pdes = loss_weighter(pde_losses, scales=pde_scales)

                # Static anchoring for data and boundaries
                loss = weighted_pdes + (5.0 * l_data) + (5.0 * l_bc) + (5.0 * l_io)

            if torch.isnan(loss):
                print(f"\n⚠️ NaN detected in loss at epoch {epoch}, batch {batch_idx}! Skipping batch.")
                continue

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss_epoch += loss.item()

            pbar.set_postfix({
                "L_tot": f"{loss.item():.3f}",
                "L_mom": f"{l_mom.item():.3f}",
                "L_rh": f"{l_rheo.item():.3f}",
                "|g|": f"{grad_norm:.2f}",
                "LR": f"{optimizer.param_groups[0]['lr']:.2e}"
            })

        scheduler.step()

        if epoch % 2 == 0:
            scores = quantify_performance(model, val_loader, kernels, device, tier="tier2")
            print(
                f"\n📊 [Validation] Rel L2: {scores.get('rel_l2', 0):.4f} | "
                f"Div: {scores.get('continuity', 0):.3e} | "
                f"Rheo: {scores.get('rheology', 0):.3e}"
            )

            # Debugging the current learned weights for PDE residuals only
            if not is_distillation:
                with torch.no_grad():
                    weights = torch.exp(-loss_weighter.log_vars)
                    print(f"⚖️ Learned PDE Weights -> Cont: {weights[0]:.2f} | Mom: {weights[1]:.2f} | Rheo: {weights[2]:.2f}")

                phys_score = scores.get('rel_l2', 0) + scores.get('continuity', 0) + scores.get('rheology', 0)
                if phys_score < best_phys_score:
                    best_phys_score = phys_score
                    torch.save(model.state_dict(), "models/tier2_best_physics.pth")
                    print("⭐ Saved Best Physics Model")

        if epoch % 5 == 0:
            validate_and_plot(model, val_data[0], epoch, device, tier="tier2")

    torch.save(model.state_dict(), "models/tier2_final.pth")
    print(f"Tier 2 Training Complete. Best Physical Score: {best_phys_score:.4f}")

if __name__ == "__main__":
    train_tier2()