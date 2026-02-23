import torch
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
    # 1. Freeze Kinematics
    for param in model.encoder.parameters(): param.requires_grad = False
    for param in model.core.parameters(): param.requires_grad = False
    for param in model.kinematics_decoder.parameters(): param.requires_grad = False

    # 2. Ensure Viscosity layers are active
    for param in model.mu_decoder.parameters(): param.requires_grad = True
    for param in model.mu_encoder.parameters(): param.requires_grad = True

    # Use a higher learning rate for the distillation phase
    return optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)


# Added log_vars to the optimizer for dynamic loss weighting
def setup_coupled_phase(model, log_vars, base_lr=2e-5):
    """Unfreezes all layers for the final Operator Splitting DEQ phase."""
    print("🔥 Unfreezing All Layers. Activating Coupled DEQ Optimization.")
    for param in model.parameters(): param.requires_grad = True

    # We assign a slightly higher learning rate (1e-3) to the loss weights so they adapt quickly
    return optim.Adam([
        {'params': model.parameters(), 'lr': base_lr},
        {'params': [log_vars], 'lr': 1e-3}
    ])


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

    dataset = load_dataset()
    if not dataset: return

    # --- STRATIFIED SPLIT (Fixes the 0.0000 Rel L2 bug) ---
    anchors = [d for d in dataset if d.is_anchor.item()]
    physics = [d for d in dataset if not d.is_anchor.item()]

    random.seed(42)
    random.shuffle(anchors)
    random.shuffle(physics)

    split_idx_a = int(0.9 * len(anchors))
    split_idx_p = int(0.9 * len(physics))

    train_data = anchors[:split_idx_a] + physics[:split_idx_p]
    val_data = anchors[split_idx_a:] + physics[split_idx_p:]
    # -----------------------------------------------------

    sampler = StratifiedAnchorSampler(train_data, batch_size=4)
    loader = DataLoader(train_data, batch_size=4, sampler=sampler)
    val_loader = DataLoader(val_data, batch_size=4, shuffle=False)

    best_phys_score = float('inf')
    best_loss = float('inf')
    Path("models").mkdir(exist_ok=True)

    # Initialize learnable parameters for Uncertainty-based Dynamic Weighting
    # Indices map to -> 0: Navier-Stokes, 1: Boundary Conditions, 2: Inlet/Outlet, 3: Data, 4: Rheology
    log_vars = torch.zeros(5, requires_grad=True, device=device)

    # State variables for optimizers/schedulers
    optimizer = None
    scheduler = None

    for epoch in range(epochs):
        is_distillation = epoch < distillation_epochs

        # --- CURRICULUM PHASE SWITCHING ---
        if epoch == 0:
            print(f"\n🚀 --- Starting Phase 1: Viscosity Distillation (Epochs 0-{distillation_epochs - 1}) ---")
            optimizer = setup_distillation_phase(model)
            # Implemented Warm Restarts to escape sharp local minima
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-5)
            sampler.set_warmup_mode(True)  # Focus mostly on anchors initially
        elif epoch == distillation_epochs:
            print(f"\n🚀 --- Starting Phase 2: Fully Coupled DEQ (Epochs {distillation_epochs}-{epochs - 1}) ---")
            # Pass the learnable log_vars to the optimizer
            optimizer = setup_coupled_phase(model, log_vars, base_lr=lr)
            # Implemented Warm Restarts (restarts every 10 epochs, increasing length)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
            sampler.set_warmup_mode(False)  # Turn on full physics batching

        model.train()
        total_loss_epoch = 0.0

        pbar = tqdm(loader, desc=f"Tier 2 Epoch {epoch:02d} [Re={phys_cfg.re_target}]")
        for batch_idx, data in enumerate(pbar):
            data = data.to(device)
            optimizer.zero_grad()

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
                l_ns = torch.tensor(0.0)

            # --- PHASE 2: FULLY COUPLED ROUTING ---
            else:
                pred = model(data, solver="anderson", anderson_beta=0.8)

                l_data = torch.tensor(0.0, device=device)
                if hasattr(data, 'is_anchor'):
                    node_is_anchor = data.is_anchor[data.batch]
                    if node_is_anchor.sum() > 0:
                        mse_loss = F.mse_loss(pred[node_is_anchor, :4], data.y[node_is_anchor, :4])

                        # Sobolev H1 Regularization
                        props = kernels._get_geometric_props(data)
                        u_pred, v_pred = pred[:, 0].unsqueeze(1), pred[:, 1].unsqueeze(1)

                        c_u_pred = kernels._compute_derivatives(u_pred, props)
                        c_v_pred = kernels._compute_derivatives(v_pred, props)

                        grad_u_pred, grad_v_pred = c_u_pred[:, :2, 0], c_v_pred[:, :2, 0]

                        with torch.no_grad():
                            u_true, v_true = data.y[:, 0].unsqueeze(1), data.y[:, 1].unsqueeze(1)
                            c_u_true = kernels._compute_derivatives(u_true, props)
                            c_v_true = kernels._compute_derivatives(v_true, props)
                            grad_u_true, grad_v_true = c_u_true[:, :2, 0].detach(), c_v_true[:, :2, 0].detach()

                        grad_mse = F.mse_loss(grad_u_pred[node_is_anchor], grad_u_true[node_is_anchor]) + \
                                   F.mse_loss(grad_v_pred[node_is_anchor], grad_v_true[node_is_anchor])

                        alpha_sobolev = min(0.1, max(0.0, (epoch - distillation_epochs) / 20.0))
                        l_data = mse_loss + (alpha_sobolev * grad_mse)

                l_ns = kernels.navier_stokes_residual(pred, data)
                l_bc = kernels.boundary_condition_loss(pred, data)
                l_io = kernels.inlet_outlet_loss(pred, data)
                l_rheo = kernels.rheology_loss(pred, data)

                lambda_phys = min(1.0, max(0.0, (epoch - distillation_epochs) / 20.0))

                # Dynamic Uncertainty Weighting
                # Automatically balances the gradients so no single loss component dominates
                loss = (torch.exp(-log_vars[0]) * (lambda_phys * l_ns) + log_vars[0]) + \
                       (torch.exp(-log_vars[1]) * l_bc + log_vars[1]) + \
                       (torch.exp(-log_vars[2]) * l_io + log_vars[2]) + \
                       (torch.exp(-log_vars[3]) * l_data + log_vars[3]) + \
                       (torch.exp(-log_vars[4]) * l_rheo + log_vars[4])

            # --- Safety checks and Backprop ---
            if torch.isnan(loss):
                print(f"\n⚠️ NaN detected in loss at epoch {epoch}, batch {batch_idx}! Skipping batch.")
                continue

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss_epoch += loss.item()

            pbar.set_postfix({
                "L_tot": f"{loss.item():.3f}",
                "L_ns": f"{l_ns.item():.3f}",
                "L_rh": f"{l_rheo.item():.3f}",
                "|g|": f"{grad_norm:.2f}",
                "LR": f"{optimizer.param_groups[0]['lr']:.2e}"
            })

        scheduler.step()

        # Validation
        if epoch % 2 == 0:
            scores = quantify_performance(model, val_loader, kernels, device, tier="tier2")
            print(
                f"\n📊 [Validation] Rel L2: {scores.get('rel_l2', 0):.4f} | "
                f"Div: {scores.get('continuity', 0):.3e} | "
                f"Rheo: {scores.get('rheology', 0):.3e} | "
                f"Shear MSE: {scores.get('shear_mse', 0):.3e}"
            )

            if not is_distillation:
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