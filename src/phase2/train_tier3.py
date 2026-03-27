import os
import sys

# Only enable expandable_segments on Linux/Unix systems to prevent Windows warnings
if sys.platform != "win32":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
else:
    # Fallback for Windows if you face OOM issues, otherwise leave empty
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import random
from torch_geometric.data import Dataset
from src.utils.paths import get_project_root
from src.phase2.ginodeq_tier3 import GINO_DEQ_Tier3
from src.phase2.physics_kernels_tier3 import BiochemPhysicsKernels
from src.phase1.physics.physics_kernels import PhysicsKernels
from src.config import VesselConfig, PhysicsConfig, BiochemConfig
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from src.phase1.utils.samplers import StratifiedAnchorSampler
from src.phase1.utils.metrics import DynamicLossWeighter

class PatientDataset(Dataset):
    def __init__(self, root, file_list):
        super().__init__(root, transform=None, pre_transform=None)
        self.file_list = file_list

    def len(self):
        return len(self.file_list)

    def get(self, idx):
        # Loads data only when called by the DataLoader using the spaced brackets
        return torch.load(self.file_list[ idx ], weights_only=False)


def load_dataset():
    cfg = VesselConfig(tier="tier3_patients")
    data_dir = cfg.graph_output_dir

    if not data_dir.exists():
        print(f"Directory not found: {data_dir}. Please generate Tier 3 data first.")
        return []

    file_list = sorted(list(data_dir.glob("*.pt")))

    # Removed the for-loop loading all graph nodes into a memory list
    print(f"📂 Found {len(file_list)} Tier 3 patient graphs for lazy loading...")

    return PatientDataset(root=str(data_dir), file_list=file_list)


def setup_tier3_optimization(model, loss_weighter, base_lr=1e-3):
    print("❄️  Verifying Kinematic Backbone is Frozen.")
    print("🔥 Activating LoRA layers, Biochemistry Encoders/Decoders, and Loss Weighter.")

    # Set the frozen kinematic backbone to eval mode!
    model.kin_encoder.eval()
    model.kin_processor.eval()
    model.kinematics_decoder.eval()

    # Freeze everything by default to be absolutely safe
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze specifically intended modules
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            param.requires_grad = True

    for param in model.bio_encoder.parameters():
        param.requires_grad = True

    for param in model.bio_processor.parameters():
        param.requires_grad = True

    for name, param in model.biochem_decoder.named_parameters():
        if 'lora' not in name.lower():
            param.requires_grad = True

    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))

    return optim.AdamW([
        {'params': trainable_params, 'lr': base_lr},
        {'params': loss_weighter.parameters(), 'lr': 5e-2, 'weight_decay': 0.0}
    ], weight_decay=1e-5)


def compute_tier3_loss(model, data, kernels, loss_weighter, current_solver, device, phys_cfg):
    # --- 1. DYNAMIC REYNOLDS NUMBER UPDATE ---
    if hasattr(data, 're_actual'):
        phys_cfg.re_target = data.re_actual.mean().item()

    # 2. Forward Pass
    pred = model(data)

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

    # 3. Supervised Data Loss (With SAFELY CLAMPED Variance Normalization)
    l_data_kine = torch.tensor(0.0, device=device)
    l_data_bio = torch.tensor(0.0, device=device)

    if hasattr(data, 'is_anchor'):
        node_is_anchor = data.is_anchor[data.batch] if hasattr(data, 'batch') else data.is_anchor
        if node_is_anchor.sum() > 0:
            # --- Kinematics Variance Normalization ---
            pred_kine = pred[node_is_anchor, :4]
            targ_kine = data.y[node_is_anchor, :4]
            # FIX: Clamp variance to a physical minimum to prevent division by near-zero
            kine_var = torch.clamp(torch.var(targ_kine, dim=0, keepdim=True), min=1e-2)
            l_data_kine = torch.mean(((pred_kine - targ_kine) ** 2) / kine_var)

            # --- Biochemistry Variance Normalization ---
            pred_bio = pred[node_is_anchor, 4:16]
            targ_bio = data.y[node_is_anchor, 4:16]
            # FIX: Clamp variance to a physical minimum
            bio_var = torch.clamp(torch.var(targ_bio, dim=0, keepdim=True), min=1e-2)
            l_data_bio = torch.mean(((pred_bio - targ_bio) ** 2) / bio_var)

    # 4. Fluid Mechanics (Inherited from Tiers 1 & 2)
    l_mom = kernels.core.navier_stokes_residual(pred[:, 0:4], data, props=props)

    c_u = kernels.core._compute_derivatives(pred[:, 0:1], props)
    c_v = kernels.core._compute_derivatives(pred[:, 1:2], props)
    du_ij = torch.stack([c_u[:, 0, 0], c_u[:, 1, 0], c_v[:, 0, 0], c_v[:, 1, 0]], dim=1)

    l_cont = kernels.core.continuity_loss(du_ij)
    l_bc = kernels.core.boundary_condition_loss(pred[:, 0:4], data)
    l_io = kernels.core.inlet_outlet_loss(pred[:, 0:4], data)

    # 5. Biochemistry Residuals
    l_adr_fast, l_adr_slow = kernels.biochem_adr_residual(biochem_preds, velocity_fields, props)

    # FIX: Unpack the newly split wall losses
    l_wall_bio, l_wall_phys = kernels.biochem_wall_residual(biochem_preds, wall_preds, velocity_fields, props, data)

    # 6. Balance & Combine with PDE targets
    pde_losses = [l_adr_fast, l_adr_slow, l_wall_bio, l_wall_phys]
    weighted_pdes = loss_weighter(pde_losses)

    loss = weighted_pdes + (500.0 * l_data_bio) + l_data_kine

    metrics = {
        "L_mom": l_mom.item(),
        "L_ADR_F": l_adr_fast.item(),
        "L_ADR_S": l_adr_slow.item(),
        "L_W_Bio": l_wall_bio.item(),
        "L_W_Phy": l_wall_phys.item(),
        "L_Data": l_data_bio.item()
    }
    return loss, metrics


def calculate_validation_metrics(pred, data, kernels, device):
    props = kernels.core._get_geometric_props(data)

    # --- Morphological Metric: Clot Dice Coefficient ---
    mu_eff = pred[:, 3]
    mu_base = 0.0035  # Approximated reference viscosity
    clot_threshold = 20.0 * mu_base
    pred_clot = (mu_eff > clot_threshold).float()

    if data.y.shape[1] > 3:
        gt_clot = (data.y[:, 3] > clot_threshold).float()
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

            nx = data.x[patent_wall_mask, 3]
            ny = data.x[patent_wall_mask, 4]

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

    max_fibrin_pred = pred[:, 12].max().item()

    return dice.item(), pearson_corr.item(), max_fibrin_pred


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
    loss_weighter = DynamicLossWeighter(num_losses=4).to(device)

    dataset = load_dataset()
    if len(dataset) == 0:
        return

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
        # --- Smooth Curriculum Learning Scheduler ---
        # 1. Warmup Phase (Epochs 0-9): Keep mu_ratio low, keep step functions VERY smooth
        if epoch < 10:
            current_mu_ratio = 2.0
            # Start at T=10.0 (very smooth), gently cool to T=8.0
            current_T_scale = 10.0 - (epoch / 10.0) * 2.0

            # 2. Coupling Phase (Epochs 10+): Ramp up physics severity, sharpen step functions
        else:
            progress = (epoch - 10) / max(1, (epochs - 11))

            # Linearly scale viscosity up to COMSOL max (80.0)
            current_mu_ratio = 2.0 + progress * (80.0 - 2.0)

            # Continue cooling T_scale from 8.0 down to 0.1 (sharp, physical bounds)
            current_T_scale = 8.0 - progress * (8.0 - 0.1)

        # Push updates to the network and kernels
        model.mu_ratio_max = current_mu_ratio

        # Unify the curriculum temperature
        model.T_scale = current_T_scale
        kernels.kinetics.T_scale = current_T_scale

        # FIX: Capitalized 'T' here as well
        print(f"\n⏳ Epoch {epoch:02d} | mu_ratio: {current_mu_ratio:.1f}x | T_scale: {current_T_scale:.2f}")

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
                "L_tot": f"{(loss.item() * accumulation_steps):.2e}",
                "L_Data": f"{metrics['L_Data']:.2e}",
                "L_ADR_F": f"{metrics['L_ADR_F']:.2e}",
                "L_W_Bio": f"{metrics['L_W_Bio']:.2e}",
                "L_W_Phy": f"{metrics['L_W_Phy']:.2e}"
            })

        scheduler.step()

        # Validation & Metrics
        if epoch % 2 == 0:
            # 1. Explicitly set to evaluation mode
            model.eval()
            val_dice_total, val_pearson_total, val_fibrin_total = 0.0, 0.0, 0.0

            with torch.no_grad():
                safe_vars = torch.clamp(loss_weighter.log_vars, min=loss_weighter.min_log_var)
                weights = torch.exp(-safe_vars)
                print(f"⚖️ Learned Weights -> ADR_F: {weights[0]:.2f} | ADR_S: {weights[1]:.2f} | W_Bio: {weights[2]:.2f} | W_Phys: {weights[3]:.2f}")

                for v_data in val_loader:
                    v_data = v_data.to(device)
                    v_pred = model(v_data)
                    if isinstance(v_pred, tuple):
                        v_pred = v_pred[0]

                    # Ensure calculate_validation_metrics extracts dynamic velocity from v_pred[ :, 0:2 ]
                    d, p, f = calculate_validation_metrics(v_pred, v_data, kernels, device)
                    val_dice_total += d
                    val_pearson_total += p
                    val_fibrin_total += f

            # 2. Return to training mode
            model.train()
            avg_dice = val_dice_total / len(val_loader)
            avg_pearson = val_pearson_total / len(val_loader)
            # 3. Calculate the average before printing
            avg_fibrin = val_fibrin_total / len(val_loader)

            print(f"📊 [Validation] Clot Dice: {avg_dice:.4f} | Patent WSS Pearson: {avg_pearson:.4f} | Max Fibrin: {avg_fibrin:.2e}")

            if avg_dice > best_dice:
                best_dice = avg_dice
                model_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), model_dir / "tier3_best_bio.pth")
                print("⭐ Saved Best Biochemical Coupling Model")


if __name__ == "__main__":
    train_tier3()