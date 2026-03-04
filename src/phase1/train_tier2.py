import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import random
from src.utils.paths import get_project_root
from src.phase1.physics.ginodeq import GINO_DEQ
from src.phase1.physics.physics_kernels import PhysicsKernels
from src.config import VesselConfig, PhysicsConfig
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from src.phase1.utils.samplers import StratifiedAnchorSampler
from src.phase1.utils.metrics import quantify_performance, validate_and_plot, DynamicLossWeighter

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
    print("❄️ Freezing Kinematics Backbone and Core. Unfreezing Viscosity Sub-network AND Encoder.")
    for param in model.parameters():
        param.requires_grad = False

    for param in model.mu_decoder.parameters():
        param.requires_grad = True
    for param in model.mu_encoder.parameters():
        param.requires_grad = True
    for param in model.encoder.parameters():
        param.requires_grad = True

    return optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-5)


def setup_coupled_phase(model, loss_weighter, base_lr=1e-4):
    print("🔥 Unfreezing All Layers. Activating Coupled DEQ Optimization.")
    for param in model.parameters(): param.requires_grad = True
    return optim.AdamW([
        {'params': model.parameters(), 'lr': base_lr},
        {'params': loss_weighter.parameters(), 'lr': 1e-3, 'weight_decay': 0.0}
    ], weight_decay=1e-5)


def compute_step_loss(model, data, kernels, loss_weighter, current_solver, lambda_phys, device, is_distillation):
    out = model(data, solver=current_solver, anderson_beta=1.0 if is_distillation else 0.8, anderson_warmup_iters=5)
    if isinstance(out, tuple):
        pred, jac_loss = out
    else:
        pred = out
        jac_loss = torch.tensor(0.0, device=device)

    # Precompute all spatial properties ONCE per batch
    props = kernels._get_geometric_props(data)

    # Explicit WSS gradient loss using encapsulated kernel
    l_wss = kernels.wall_shear_stress_loss(pred, data, props=props)

    # --- DISTILLATION ROUTING ---
    if is_distillation:
        l_data_mu = torch.tensor(0.0, device=device)
        if hasattr(data, 'is_anchor'):
            node_is_anchor = data.is_anchor[data.batch] if hasattr(data, 'batch') else data.is_anchor
            if node_is_anchor.sum() > 0:
                l_data_mu = F.mse_loss(pred[node_is_anchor, 3], data.y[node_is_anchor, 3])

        l_rheo = kernels.rheology_loss(pred, data, props=props)
        l_bc = kernels.boundary_condition_loss(pred, data)
        l_io = kernels.inlet_outlet_loss(pred, data)

        loss = (10.0 * l_rheo) + (5.0 * l_data_mu) + (5.0 * l_bc) + (5.0 * l_io) + (10.0 * l_wss) + (0.1 * jac_loss)

        metrics = {"L_rh": l_rheo.item(), "L_jac": jac_loss.item(), "L_mom": 0.0, "L_cont": 0.0, "L_wss": l_wss.item()}
        return loss, metrics

    # --- PHASE 2/3: FULLY COUPLED ROUTING ---
    l_data_kine = torch.tensor(0.0, device=device)
    l_data_mu = torch.tensor(0.0, device=device)
    if hasattr(data, 'is_anchor'):
        node_is_anchor = data.is_anchor[data.batch] if hasattr(data, 'batch') else data.is_anchor
        if node_is_anchor.sum() > 0:
            l_data_kine = F.mse_loss(pred[node_is_anchor, :3], data.y[node_is_anchor, :3])
            l_data_mu = F.mse_loss(pred[node_is_anchor, 3], data.y[node_is_anchor, 3])

    # 1. Compute Momentum
    l_mom = kernels.navier_stokes_residual(pred, data, props=props)

    # 2. Extract 1st-order gradients for Continuity
    c_u = kernels._compute_derivatives(pred[:, 0:1], props)
    c_v = kernels._compute_derivatives(pred[:, 1:2], props)

    du_dx, du_dy = c_u[:, 0, 0], c_u[:, 1, 0]
    dv_dx, dv_dy = c_v[:, 0, 0], c_v[:, 1, 0]
    du_ij = torch.stack([du_dx, du_dy, dv_dx, dv_dy], dim=1)

    # 3. Compute Continuity explicitly
    l_cont = kernels.continuity_loss(du_ij)

    l_bc = kernels.boundary_condition_loss(pred, data)
    l_io = kernels.inlet_outlet_loss(pred, data)
    l_rheo = kernels.rheology_loss(pred, data, props=props)

    pde_losses = [l_cont, l_mom]
    pde_scales = [lambda_phys, lambda_phys]
    weighted_pdes = loss_weighter(pde_losses, scales=pde_scales)

    loss = weighted_pdes + (1.0 * l_rheo) + (500.0 * l_data_kine) + (50.0 * l_data_mu) + (5.0 * l_bc) + (5.0 * l_io) + (
                10.0 * l_wss) + (0.1 * jac_loss)

    metrics = {
        "L_mom": l_mom.item(),
        "L_cont": l_cont.item(),
        "L_rh": l_rheo.item(),
        "L_jac": jac_loss.item(),
        "L_wss": l_wss.item()
    }
    return loss, metrics


def train_tier2(epochs=50, distillation_epochs=15, adam_epochs=50, lr=1e-4):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device being used:", device)

    # 1. Calculate physics bounds
    phys_cfg = PhysicsConfig(tier="tier2", re_target=150.0)
    kernels = PhysicsKernels(phys_cfg=phys_cfg)
    mu_inf_nd = phys_cfg.mu_inf / phys_cfg.mu_ref
    mu_0_nd = phys_cfg.mu_0 / phys_cfg.mu_ref

    # 2. Instantiate the model
    model = GINO_DEQ(
        in_channels=15,
        out_channels=5,
        latent_dim=64,
        max_iters=15,
        mu_inf_nd=mu_inf_nd,
        mu_0_nd=mu_0_nd
    ).to(device)

    # 3. Load the Tier 1 weights safely
    root = get_project_root()
    model_dir = root / "models"
    model_dir.mkdir(exist_ok=True)
    tier1_path = model_dir / "tier1_best_physics.pth"

    if tier1_path.exists():
        state_dict = torch.load(tier1_path, map_location=device, weights_only=True)

        # --- Dynamic channel expansion surgery ---
        if 'encoder.0.weight' in state_dict:
            tier1_weight = state_dict['encoder.0.weight']
            model_weight = model.encoder[0].weight
            if tier1_weight.shape[1] != model_weight.shape[1]:
                print(f"🔧 Adapting Tier 1 encoder weights ({tier1_weight.shape[1]} -> {model_weight.shape[1]})...")
                new_weight = torch.zeros_like(model_weight)
                min_dim = min(tier1_weight.shape[1], model_weight.shape[1])
                new_weight[:, :min_dim] = tier1_weight[:, :min_dim]
                state_dict['encoder.0.weight'] = new_weight
        # ------------------------------------------------------

        model.load_state_dict(state_dict, strict=False)
        print("✅ Successfully loaded Tier 1 foundational physics weights.")
    else:
        print("⚠️ Warning: Tier 1 weights not found.")

    # Synchronized to strictly handle [l_cont, l_mom]
    loss_weighter = DynamicLossWeighter(num_losses=2).to(device)

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

    # Reduce physical batch size to save memory, but maintain effective batch size
    micro_batch_size = 2
    accumulation_steps = 4  # Effective batch size = 2 * 4 = 8

    sampler = StratifiedAnchorSampler(train_data, batch_size=micro_batch_size)
    loader = DataLoader(train_data, batch_size=micro_batch_size, sampler=sampler)
    val_loader = DataLoader(val_data, batch_size=micro_batch_size, shuffle=False)

    best_phys_score = float('inf')
    best_loss = float('inf')
    optimizer = None
    scheduler = None
    lbfgs_initialized = False

    target_n = phys_cfg.n
    start_n = 0.8

    for epoch in range(epochs):
        is_distillation = epoch < distillation_epochs
        physics_active = not is_distillation
        lambda_phys = min(1.0, max(0.0, (epoch - distillation_epochs) / 20.0))

        if is_distillation:
            progress = epoch / max(1, (distillation_epochs - 1))
            current_n = start_n - progress * (start_n - target_n)
            phys_cfg.n = current_n
            print(f"🔄 Curriculum: Annealed Carreau index 'n' to {current_n:.4f}")
        else:
            phys_cfg.n = target_n

        if is_distillation or lbfgs_initialized:
            current_solver = "picard"
        else:
            current_solver = "anderson"

        if epoch == 0:
            print(f"\n🚀 --- Starting Phase 1: Viscosity Distillation (Epochs 0-{distillation_epochs - 1}) ---")
            optimizer = setup_distillation_phase(model)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-5)
            sampler.set_warmup_mode(True)
        elif epoch == distillation_epochs:
            print(
                f"\n🚀 --- Starting Phase 2: Fully Coupled DEQ via AdamW (Epochs {distillation_epochs}-{adam_epochs - 1}) ---")
            optimizer = setup_coupled_phase(model, loss_weighter, base_lr=lr)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
            sampler.set_warmup_mode(False)
        elif epoch == adam_epochs and not lbfgs_initialized:
            print(f"\n⚡ --- Starting Phase 3: L-BFGS Optimizer for final {epochs - adam_epochs} epochs ---")
            torch.cuda.empty_cache()

            for param in loss_weighter.parameters():
                param.requires_grad = False

            optimizer = optim.LBFGS(
                model.parameters(),
                lr=0.01,
                max_iter=20,
                history_size=30,
                line_search_fn=None,
                tolerance_grad=1e-6,
                tolerance_change=1e-8
            )
            lbfgs_initialized = True

        if not lbfgs_initialized:
            model.train()
        else:
            model.eval()

        total_loss_epoch = 0.0
        grad_norm = 0.0

        if not lbfgs_initialized:
            pbar = tqdm(loader, desc=f"Tier X Epoch {epoch:02d}...")

            # Zero gradients AT THE START of the epoch
            optimizer.zero_grad()

            for batch_idx, data in enumerate(pbar):
                data = data.to(device)

                # Compute loss (scaled by accumulation steps)
                loss, metrics = compute_step_loss(model, data, kernels, loss_weighter, current_solver, lambda_phys,
                                                  device, is_distillation)
                loss = loss / accumulation_steps

                if torch.isnan(loss):
                    print(f"\n⚠️ NaN detected in loss at epoch {epoch}! Skipping micro-batch.")
                    continue

                loss.backward()

                # Step optimizer ONLY when we've accumulated enough micro-batches
                if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(loader)):
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()  # Reset for the next effective batch

                # Multiply back for display purposes
                total_loss_epoch += (loss.item() * accumulation_steps)

                pbar.set_postfix({
                    "L_tot": f"{(loss.item() * accumulation_steps):.3f}",
                    "L_mom": f"{metrics['L_mom']:.3f}",
                    "L_rh": f"{metrics['L_rh']:.3f}",
                    "|g|": f"{grad_norm:.2f}",
                    "LR": f"{optimizer.param_groups[0]['lr']:.2e}"
                })
            scheduler.step()

        else:
            print(f"⏳ Tier 2 Epoch {epoch:02d} [Re={phys_cfg.re_target}] (L-BFGS)")
            if not hasattr(optimizer, 'static_batches'):
                optimizer.static_batches = [d.to(device) for d in loader]

            def closure():
                optimizer.zero_grad()
                accumulated_loss = torch.tensor(0.0, device=device)

                for closure_data in optimizer.static_batches:
                    loss, _ = compute_step_loss(model, closure_data, kernels, loss_weighter, current_solver,
                                                lambda_phys, device, is_distillation)
                    loss = loss / len(optimizer.static_batches)
                    loss.backward()
                    accumulated_loss += loss.detach()
                return accumulated_loss

            loss_tensor = optimizer.step(closure)
            total_loss_epoch = loss_tensor.item() * len(optimizer.static_batches)
            print(f"✅ L-BFGS Step Complete. Accumulated Full-Batch Loss: {loss_tensor.item():.4f}")

        if epoch % 2 == 0:
            scores = quantify_performance(model, val_loader, kernels, device, tier="tier2")
            print(
                f"\n📊 [Validation] Rel L2: {scores.get('rel_l2', 0):.4f} | Div: {scores.get('continuity', 0):.3e} | Rheo: {scores.get('rheology', 0):.3e}")

            if physics_active:
                with torch.no_grad():
                    safe_vars = torch.clamp(loss_weighter.log_vars, min=loss_weighter.min_log_var)
                    weights = torch.exp(-safe_vars)
                    print(f"⚖️ Learned PDE Weights -> Cont: {weights[0]:.2f} | Mom: {weights[1]:.2f}")

                phys_score = scores.get('rel_l2', 0) + scores.get('continuity', 0) + scores.get('rheology', 0)
                if phys_score < best_phys_score:
                    best_phys_score = phys_score
                    torch.save(model.state_dict(), model_dir / "tier2_best_physics.pth")
                    print("⭐ Saved Best Physics Model")

        avg_loss = total_loss_epoch / len(loader)
        if avg_loss < best_loss and physics_active:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_dir / "tier2_best_loss.pth")
            print(f"⭐ Saved Best Loss Model")

        if epoch % 5 == 0:
            validate_and_plot(model, val_data[0], epoch, device, tier="tier2")

    torch.save(model.state_dict(), model_dir / "tier2_final.pth")
    print(f"Tier 2 Training Complete. Best Physical Score: {best_phys_score:.4f} | Best Loss: {best_loss:.4f}")


if __name__ == "__main__":
    train_tier2()