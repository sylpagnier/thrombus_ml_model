import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import random
from pathlib import Path

from src.utils.paths import get_project_root
from src.phase2.ginodeq_tier3 import GINO_DEQ_Tier3
from src.phase2.physics_kernels_tier3 import BiochemPhysicsKernels
from src.phase1.physics.physics_kernels import PhysicsKernels
from src.config import VesselConfig, PhysicsConfig, BiochemConfig
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from src.phase1.utils.samplers import StratifiedAnchorSampler
from src.phase1.utils.metrics import DynamicLossWeighter


def load_dataset():
    # Update tier to match the PatientDataExtractor configuration
    cfg = VesselConfig(tier="tier3_patients")
    data_dir = cfg.graph_output_dir

    if not data_dir.exists():
        print(f"Directory not found: {data_dir}. Please generate Tier 3 data first.")
        return []

    # Update glob pattern to find files without the "vessel_" prefix
    file_list = sorted(list(data_dir.glob("*.pt")))

    dataset = []
    print(f"📂 Loading {len(file_list)} Tier 3 patient graphs...")
    for f in tqdm(file_list):
        data = torch.load(f, weights_only=False)
        dataset.append(data)
    return dataset


def setup_tier3_optimization(model, loss_weighter, base_lr=1e-3):
    print("❄️  Verifying Kinematic Backbone is Frozen.")
    print("🔥 Activating LoRA layers, Biochemistry Encoders/Decoders, and Loss Weighter.")

    # Freeze everything by default to be absolutely safe
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze specifically intended modules
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            param.requires_grad = True

    for param in model.bio_encoder.parameters():
        param.requires_grad = True

    for name, param in model.biochem_decoder.named_parameters():
        if 'lora' not in name.lower():
            param.requires_grad = True

    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))

    return optim.AdamW([
        {'params': trainable_params, 'lr': base_lr},
        {'params': loss_weighter.parameters(), 'lr': 1e-3, 'weight_decay': 0.0}
    ], weight_decay=1e-5)


def compute_tier3_loss(model, data, kernels, loss_weighter, current_solver, device, phys_cfg):
    # --- 1. DYNAMIC REYNOLDS NUMBER UPDATE ---
    if hasattr(data, 're_actual'):
        # Update the core physics config for this specific graph's scaling
        phys_cfg.re_target = data.re_actual.mean().item()

    # 2. Forward Pass
    out = model(data, anderson_beta=1.0)
    if isinstance(out, tuple):
        pred, jac_loss = out
    else:
        pred = out
        jac_loss = torch.tensor(0.0, device=device)

    props = kernels.core._get_geometric_props(data)

    # We use data.batch to ensure proper mapping if multiple graphs are batched together
    if hasattr(data, 'batch') and data.batch is not None:
        props['u_ref'] = data.u_ref[data.batch]
        props['d_bar'] = data.d_bar[data.batch]
    else:
        props['u_ref'] = data.u_ref
        props['d_bar'] = data.d_bar

    velocity_fields = pred[:, 0:2]
    biochem_preds = pred[:, 4:13]
    wall_preds = pred[:, 13:16]
    mask_wall = data.mask_wall.view(-1).bool()

    # 3. Supervised Data Loss (With Dual Variance Normalization)
    l_data_kine = torch.tensor(0.0, device=device)
    l_data_bio = torch.tensor(0.0, device=device)

    if hasattr(data, 'is_anchor'):
        node_is_anchor = data.is_anchor[data.batch] if hasattr(data, 'batch') else data.is_anchor
        if node_is_anchor.sum() > 0:
            # --- Kinematics Variance Normalization (Channels 0-3) ---
            pred_kine = pred[node_is_anchor, :4]
            targ_kine = data.y[node_is_anchor, :4]
            # Add epsilon to prevent div by zero
            kine_var = torch.var(targ_kine, dim=0, keepdim=True) + 1e-6
            l_data_kine = torch.mean(((pred_kine - targ_kine) ** 2) / kine_var)

            # --- Biochemistry Variance Normalization (Channels 4-15) ---
            pred_bio = pred[node_is_anchor, 4:16]
            targ_bio = data.y[node_is_anchor, 4:16]
            bio_var = torch.var(targ_bio, dim=0, keepdim=True) + 1e-6
            l_data_bio = torch.mean(((pred_bio - targ_bio) ** 2) / bio_var)

    # 4. Fluid Mechanics (Inherited from Tiers 1 & 2)
    l_mom = kernels.core.navier_stokes_residual(pred[:, 0:4], data, props=props)

    c_u = kernels.core._compute_derivatives(pred[:, 0:1], props)
    c_v = kernels.core._compute_derivatives(pred[:, 1:2], props)
    du_ij = torch.stack([c_u[:, 0, 0], c_u[:, 1, 0], c_v[:, 0, 0], c_v[:, 1, 0]], dim=1)

    l_cont = kernels.core.continuity_loss(du_ij)
    l_rheo = kernels.core.rheology_loss(pred[:, 0:4], data, props=props)
    l_bc = kernels.core.boundary_condition_loss(pred[:, 0:4], data)
    l_io = kernels.core.inlet_outlet_loss(pred[:, 0:4], data)
    l_wss = kernels.core.wall_shear_stress_loss(pred[:, 0:5], data, props=props)

    # 5. Biochemistry (Tier 3 additions)
    l_adr = kernels.biochem_adr_residual(biochem_preds, velocity_fields, props)
    l_wall = kernels.biochem_wall_residual(biochem_preds, wall_preds, velocity_fields, props, mask_wall)

    # 6. Balance & Combine
    pde_losses = [l_mom, l_cont, l_adr]
    weighted_pdes = loss_weighter(pde_losses)

    loss = weighted_pdes + (10.0 * l_wall) + \
           (500.0 * l_data_kine) + (500.0 * l_data_bio) + \
           (10.0 * l_bc) + (5.0 * l_io) + (0.1 * jac_loss)

    metrics = {
        "L_mom": l_mom.item(),
        "L_ADR": l_adr.item(),
        "L_Wall": l_wall.item()
    }
    return loss, metrics


def calculate_validation_metrics(pred, data, kernels, device):
    props = kernels.core._get_geometric_props(data)

    # --- Morphological Metric: Clot Dice Coefficient ---
    mu_eff = pred[:, 3]
    mu_base = 0.0035  # Approximated reference viscosity
    pred_clot = (mu_eff > 1000.0 * mu_base).float()

    if data.y.shape[1] > 3:
        gt_clot = (data.y[:, 3] > 1000.0 * mu_base).float()
        intersection = (pred_clot * gt_clot).sum()
        dice = (2.0 * intersection) / (pred_clot.sum() + gt_clot.sum() + 1e-8)
    else:
        dice = torch.tensor(0.0)

    # --- Hemodynamic Metric: WSS Pearson Correlation (Patent Lumen) ---
    mask_wall = data.mask_wall.view(-1).bool()

    if mask_wall.any() and data.y.shape[1] > 1:
        # Patent Lumen Mask (Walls where there is NO ground truth clot)
        patent_wall_mask = mask_wall & (gt_clot == 0) if data.y.shape[1] > 3 else mask_wall

        if patent_wall_mask.any():
            # Compute Predicted WSS
            c_u = kernels.core._compute_derivatives(pred[:, 0:1], props)
            c_v = kernels.core._compute_derivatives(pred[:, 1:2], props)
            dudx_p, dudy_p = c_u[:, 0, 0], c_u[:, 1, 0]
            dvdx_p, dvdy_p = c_v[:, 0, 0], c_v[:, 1, 0]

            mu_wall_p = pred[patent_wall_mask, 3]
            tau_xx_p = 2.0 * mu_wall_p * dudx_p[patent_wall_mask]
            tau_yy_p = 2.0 * mu_wall_p * dvdy_p[patent_wall_mask]
            tau_xy_p = mu_wall_p * (dudy_p[patent_wall_mask] + dvdx_p[patent_wall_mask])

            nx = data.x[patent_wall_mask, 4]
            ny = data.x[patent_wall_mask, 5]

            tx_p, ty_p = tau_xx_p * nx + tau_xy_p * ny, tau_xy_p * nx + tau_yy_p * ny
            tn_p = tx_p * nx + ty_p * ny
            wss_pred = torch.sqrt((tx_p - tn_p * nx) ** 2 + (ty_p - tn_p * ny) ** 2 + 1e-8)

            # Compute Target WSS
            c_u_t = kernels.core._compute_derivatives(data.y[:, 0:1], props)
            c_v_t = kernels.core._compute_derivatives(data.y[:, 1:2], props)
            dudx_t, dudy_t = c_u_t[:, 0, 0], c_u_t[:, 1, 0]
            dvdx_t, dvdy_t = c_v_t[:, 0, 0], c_v_t[:, 1, 0]

            mu_wall_t = data.y[patent_wall_mask, 3] if data.y.shape[1] > 3 else torch.full_like(mu_wall_p, mu_base)
            tau_xx_t = 2.0 * mu_wall_t * dudx_t[patent_wall_mask]
            tau_yy_t = 2.0 * mu_wall_t * dvdy_t[patent_wall_mask]
            tau_xy_t = mu_wall_t * (dudy_t[patent_wall_mask] + dvdx_t[patent_wall_mask])

            tx_t, ty_t = tau_xx_t * nx + tau_xy_t * ny, tau_xy_t * nx + tau_yy_t * ny
            tn_t = tx_t * nx + ty_t * ny
            wss_targ = torch.sqrt((tx_t - tn_t * nx) ** 2 + (ty_t - tn_t * ny) ** 2 + 1e-8)

            # Pearson Correlation
            stacked = torch.stack([wss_pred, wss_targ])
            pearson_corr = torch.corrcoef(stacked)[0, 1]
            if torch.isnan(pearson_corr): pearson_corr = torch.tensor(0.0)
        else:
            pearson_corr = torch.tensor(0.0)
    else:
        pearson_corr = torch.tensor(0.0)

    return dice.item(), pearson_corr.item()


def train_tier3(epochs=50, lr=1e-3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device being used: {device}")

    phys_cfg = PhysicsConfig(tier="tier3", re_target=150.0)
    bio_cfg = BiochemConfig(tier="tier3")
    core_kernels = PhysicsKernels(phys_cfg=phys_cfg)
    kernels = BiochemPhysicsKernels(biochem_cfg=bio_cfg, core_physics_kernels=core_kernels)

    model = GINO_DEQ_Tier3(in_channels=12, latent_dim=64, max_outer_iters=3, max_inner_iters=15).to(device)

    # 1. Load Tier 2 Checkpoint (Frozen Kinematic Backbone)
    root = get_project_root()
    model_dir = root / "models"
    tier2_path = model_dir / "tier2_best_physics.pth"

    if tier2_path.exists():
        state_dict = torch.load(tier2_path, map_location=device, weights_only=True)

        # --- NEW: Map Tier 1/2 keys to Tier 3 kinematic keys ---
        mapped_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('encoder'):
                mapped_state_dict[key.replace('encoder', 'kin_encoder')] = value
            elif key.startswith('processor'):
                mapped_state_dict[key.replace('processor', 'kin_processor')] = value
            elif key.startswith('decoder'):
                mapped_state_dict[key.replace('decoder', 'kinematics_decoder')] = value
            elif key.startswith('mu_encoder') or key.startswith('mu_decoder'):
                pass  # Ignore standalone mu layers if they exist from Tier 2
            else:
                mapped_state_dict[key] = value

        # Load the mapped dict
        model.load_state_dict(mapped_state_dict, strict=False)
        print("✅ Successfully loaded Tier 2 kinematic weights into Tier 3 backbone.")

    # 5 targets: L_NS, L_Rheo, L_Data, L_ADR, L_Wall
    loss_weighter = DynamicLossWeighter(num_losses=3).to(device)

    dataset = load_dataset()
    if not dataset: return

    # Separate anchors (labeled) and physics-only (unlabeled)
    anchors = [d for d in dataset if d.is_anchor.item()]
    physics = [d for d in dataset if not d.is_anchor.item()]

    random.seed(42)
    random.shuffle(anchors)
    random.shuffle(physics)

    # ADJUSTED SPLIT: Ensure at least 1 graph in validation if dataset is very small
    if len(dataset) == 1:
        print("⚠️ Only one patient graph found. Using it for both Training and Validation.")
        train_data = dataset
        val_data = dataset
    else:
        split_idx_a = int(0.9 * len(anchors))
        split_idx_p = int(0.9 * len(physics))
        train_data = anchors[:split_idx_a] + physics[:split_idx_p]
        val_data = anchors[split_idx_a:] + physics[split_idx_p:]

    micro_batch_size = 2
    accumulation_steps = 4

    # Handle single-patient case (usually just for quick testing)
    if len(train_data) < micro_batch_size:
        loader = DataLoader(train_data, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
    else:
        sampler = StratifiedAnchorSampler(train_data, batch_size=micro_batch_size)
        loader = DataLoader(train_data, batch_size=micro_batch_size, sampler=sampler)
        val_loader = DataLoader(val_data, batch_size=micro_batch_size, shuffle=False)

    optimizer = setup_tier3_optimization(model, loss_weighter, base_lr=lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    best_dice = -1.0

    print("\n🚀 --- Starting Phase 3: Segregated Bio-Fluid Coupling ---")

    for epoch in range(epochs):
        # --- Curriculum Learning Scheduler for mu_ratio ---
        if epoch < 10:
            current_mu_ratio = 2.0
        else:
            progress = (epoch - 10) / max(1, (epochs - 11))
            current_mu_ratio = 2.0 + progress * (7000.0 - 2.0)

        model.mu_ratio_max = current_mu_ratio

        print(f"\n⏳ Epoch {epoch:02d} | mu_ratio Curriculum: {current_mu_ratio:.2f}x")

        model.train()
        total_loss_epoch = 0.0
        optimizer.zero_grad()

        pbar = tqdm(loader, desc=f"Tier 3 Ep {epoch:02d}")
        for batch_idx, data in enumerate(pbar):
            data = data.to(device)
            data.x.requires_grad_(True)

            loss, metrics = compute_tier3_loss(model, data, kernels, loss_weighter, "anderson", device, phys_cfg)
            loss = loss / accumulation_steps

            if torch.isnan(loss):
                print(f"\n⚠️ NaN detected in loss at epoch {epoch}! Skipping micro-batch.")
                continue

            loss.backward()

            if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(loader)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss_epoch += (loss.item() * accumulation_steps)

            pbar.set_postfix({
                "L_tot": f"{(loss.item() * accumulation_steps):.3f}",
                "L_mom": f"{metrics['L_mom']:.3f}",
                "L_ADR": f"{metrics['L_ADR']:.3f}",
                "L_Wall": f"{metrics['L_Wall']:.3f}"
            })

        scheduler.step()

        # Validation & Metrics
        if epoch % 2 == 0:
            model.eval()
            val_dice_total, val_pearson_total = 0.0, 0.0

            with torch.no_grad():
                # Print current learned loss weights
                safe_vars = torch.clamp(loss_weighter.log_vars, min=loss_weighter.min_log_var)
                weights = torch.exp(-safe_vars)
                print(
                    f"⚖️ Dynamic PDE Weights -> Mom: {weights[0]:.2f} | Cont: {weights[1]:.2f} | ADR: {weights[2]:.2f}")

                for v_data in val_loader:
                    v_data = v_data.to(device)
                    v_pred = model(v_data, anderson_beta=1.0)
                    if isinstance(v_pred, tuple): v_pred = v_pred[0]

                    d, p = calculate_validation_metrics(v_pred, v_data, kernels, device)
                    val_dice_total += d
                    val_pearson_total += p

            avg_dice = val_dice_total / len(val_loader)
            avg_pearson = val_pearson_total / len(val_loader)

            print(f"📊 [Validation] Clot Dice: {avg_dice:.4f} | Patent WSS Pearson: {avg_pearson:.4f}")

            if avg_dice > best_dice:
                best_dice = avg_dice
                torch.save(model.state_dict(), model_dir / "tier3_best_bio.pth")
                print("⭐ Saved Best Biochemical Coupling Model")


if __name__ == "__main__":
    train_tier3()