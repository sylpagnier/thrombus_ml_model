import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from src.utils.paths import get_project_root
from src.phase1.physics.ginodeq import GINO_DEQ
from src.phase1.physics.physics_kernels import PhysicsKernels
from src.config import VesselConfig, PhysicsConfig
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from src.phase1.utils.samplers import StratifiedAnchorSampler
from src.phase1.utils.metrics import quantify_performance, validate_and_plot, DynamicLossWeighter
import random





def load_dataset():
    cfg = VesselConfig(tier="tier1")
    if not cfg.graph_output_dir.exists():
        return []
    dataset = []
    print(f"📂 Loading Tier 1 graphs from {cfg.graph_output_dir}...")
    for f in tqdm(sorted(list(cfg.graph_output_dir.glob("vessel_*.pt")))):
        dataset.append(torch.load(f, weights_only=False))
    return dataset


def compute_step_loss(model, data, kernels, loss_weighter, current_solver, lambda_phys, device, is_distillation=False):
    out = model(data, solver=current_solver, anderson_beta=0.8, anderson_warmup_iters=5)
    if isinstance(out, tuple):
        pred, jac_loss = out
    else:
        pred = out
        jac_loss = torch.tensor(0.0, device=device)

    # Precompute geometric properties ONCE per batch
    props = kernels._get_geometric_props(data)

    # Calculate explicit WSS gradient loss using precomputed props
    l_wss = kernels.wall_shear_stress_loss(pred, data, props=props)

    # Supervised Data Loss (Anchors only)
    l_data_kine = torch.tensor(0.0, device=device)
    if hasattr(data, 'is_anchor'):
        node_is_anchor = data.is_anchor[data.batch] if hasattr(data, 'batch') else data.is_anchor
        if node_is_anchor.sum() > 0:
            l_data_kine = F.mse_loss(pred[node_is_anchor, :3], data.y[node_is_anchor, :3])

    # 1. Compute Momentum using precomputed props
    l_mom = kernels.navier_stokes_residual(pred, data, props=props)

    # 2. Extract 1st-order gradients for Continuity using precomputed props
    c_u = kernels._compute_derivatives(pred[:, 0:1], props)
    c_v = kernels._compute_derivatives(pred[:, 1:2], props)

    du_dx, du_dy = c_u[:, 0, 0], c_u[:, 1, 0]
    dv_dx, dv_dy = c_v[:, 0, 0], c_v[:, 1, 0]
    du_ij = torch.stack([du_dx, du_dy, dv_dx, dv_dy], dim=1)

    # 3. Compute Continuity explicitly
    l_cont = kernels.continuity_loss(du_ij)

    l_bc = kernels.boundary_condition_loss(pred, data)
    l_io = kernels.inlet_outlet_loss(pred, data)

    pde_losses = [l_cont, l_mom]
    pde_scales = [lambda_phys, lambda_phys]
    weighted_pdes = loss_weighter(pde_losses, scales=pde_scales)

    # Combine losses cleanly
    loss = weighted_pdes + (500.0 * l_data_kine) + (10.0 * l_bc) + (5.0 * l_io) + (10.0 * l_wss) + (0.1 * jac_loss)

    metrics = {
        "L_data": l_data_kine.item(),
        "L_mom": l_mom.item(),
        "L_cont": l_cont.item(),
        "L_jac": jac_loss.item(),
        "L_wss": l_wss.item()
    }
    return loss, metrics


def train_tier1(epochs=50, lr=1e-4, warm_up_epochs=10, adam_epochs=50):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device being used:", device)
    model = GINO_DEQ(in_channels=15, out_channels=5, latent_dim=64, max_iters=15).to(device)

    phys_cfg = PhysicsConfig(tier="tier1")
    kernels = PhysicsKernels(phys_cfg=phys_cfg)

    loss_weighter = DynamicLossWeighter(num_losses=2).to(device)

    fig_dir = get_project_root() / "reports" / "figures" / "tier1"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Phase 1 Optimizer (AdamW)
    optimizer = optim.AdamW(list(model.parameters()) + list(loss_weighter.parameters()),
                            lr=lr, weight_decay=1e-5)

    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warm_up_epochs)
    decay_epochs = adam_epochs - warm_up_epochs
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=decay_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warm_up_epochs])

    dataset = load_dataset()
    if not dataset: return

    # --- NEW STRATIFIED SPLIT LOGIC ---
    anchors = [d for d in dataset if d.is_anchor.any().item()]
    physics = [d for d in dataset if not d.is_anchor.any().item()]

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
    root = get_project_root()
    model_dir = root / "models"
    model_dir.mkdir(exist_ok=True)
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
            torch.cuda.empty_cache()

            optimizer = optim.LBFGS(
                list(model.parameters()) + list(loss_weighter.parameters()),
                lr=0.01,
                max_iter=20,
                history_size=30,
                line_search_fn="strong_wolfe",
                tolerance_grad=1e-6,
                tolerance_change=1e-8
            )
            lbfgs_initialized = True

        if not lbfgs_initialized:
            # --- PHASE 1: AdamW Execution (Mini-Batch with Accumulation) ---
            pbar = tqdm(loader, desc=f"Tier 1 Epoch {epoch:02d} [Re={phys_cfg.re_target}] (AdamW)")

            # Zero gradients AT THE START of the epoch
            optimizer.zero_grad()

            for batch_idx, data in enumerate(pbar):
                data = data.to(device)

                # Compute loss (scaled by accumulation steps so the final gradient magnitude is correct)
                loss, metrics = compute_step_loss(model, data, kernels, loss_weighter, current_solver, lambda_phys,
                                                  device, is_distillation=False)
                loss = loss / accumulation_steps

                if torch.isnan(loss):
                    print(f"\n⚠️ NaN detected! Skipping micro-batch.")
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
                    "L_cont": f"{metrics['L_cont']:.3f}",
                    "L_jac": f"{metrics['L_jac']:.3f}",
                    "LR": f"{optimizer.param_groups[0]['lr']:.2e}"
                })

            scheduler.step()

        else:
            # --- PHASE 2: L-BFGS Execution (Full-Batch via Accumulation) ---
            print(f"⏳ Tier 1 Epoch {epoch:02d} [Re={phys_cfg.re_target}] (L-BFGS Line Search...)")

            if not hasattr(optimizer, 'static_batches'):
                optimizer.static_batches = [d.to(device) for d in loader]

            def closure():
                optimizer.zero_grad()
                accumulated_loss = torch.tensor(0.0, device=device)

                for closure_data in optimizer.static_batches:
                    loss, _ = compute_step_loss(model, closure_data, kernels, loss_weighter, current_solver,
                                                lambda_phys, device, is_distillation=False)
                    loss = loss / len(optimizer.static_batches)
                    loss.backward()
                    accumulated_loss += loss.detach()

                return accumulated_loss

            loss_tensor = optimizer.step(closure)
            total_loss_epoch = loss_tensor.item() * len(optimizer.static_batches)

            print(f"✅ L-BFGS Step Complete. Accumulated Full-Batch Loss: {loss_tensor.item():.4f}")

        if epoch % 2 == 0:
            scores = quantify_performance(model, val_loader, kernels, device, tier="tier1")

            print(f"\n📊 [Validation] Rel L2: {scores.get('rel_l2', 0):.4f} | "
                  f"Div: {scores.get('continuity', 0):.3e} | "
                  f"Wall Slip: {scores.get('wall_slip', 0):.4f}")

            with torch.no_grad():
                safe_vars = torch.clamp(loss_weighter.log_vars, min=loss_weighter.min_log_var)
                weights = torch.exp(-safe_vars)
                print(f"⚖️ Learned PDE Weights -> Cont: {weights[0]:.2f} | Mom: {weights[1]:.2f}")

            phys_score = scores.get('rel_l2', 0) + scores.get('continuity', 0)

            if phys_score < best_phys_score and physics_active:
                best_phys_score = phys_score
                save_path = model_dir / "tier1_best_physics.pth"
                torch.save(model.state_dict(), save_path)
                print(f"⭐ Saved Best Physics Model to {save_path}")

        avg_loss = total_loss_epoch / len(loader)
        if avg_loss < best_loss and physics_active:
            best_loss = avg_loss
            save_path = model_dir / "tier1_best_loss.pth"
            torch.save(model.state_dict(), save_path)
            print(f"⭐ Saved Best Loss Model to {save_path}")

        if epoch % 5 == 0:
            validate_and_plot(model, val_data[0], epoch, device, tier="tier1")

    print(f"Tier 1 Training Complete. Best Physical Score: {best_phys_score:.4f} | Best Loss: {best_loss:.4f}")


if __name__ == "__main__":
    train_tier1()