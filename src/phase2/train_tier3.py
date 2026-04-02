import os
import sys
import math

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
from src.phase2.gnode_tier3 import GNODE_Tier3
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


def initialize_biochem_priors(model):
    """
    Initializes the biochem_decoder to output the exact resting physiological state
    at t=0, matching the COMSOL baseline initial values.
    """
    print("🧬 Injecting physical priors into biochemistry decoder biases...")

    # 1. Zero out the final weights so the initial output is driven entirely by the bias.
    # Assuming SpectralLinear has an underlying 'linear' attribute.
    # If it inherits directly from nn.Linear, use model.biochem_decoder.weight instead.
    target_layer = model.biochem_decoder.linear if hasattr(model.biochem_decoder, 'linear') else model.biochem_decoder
    torch.nn.init.zeros_(target_layer.weight)

    # 2. Create the precise bias vector matching your 12 species order:
    # [ 0 ]:RP, [ 1 ]:AP, [ 2 ]:APR, [ 3 ]:APS, [ 4 ]:PT, [ 5 ]:T, [ 6 ]:AT, [ 7 ]:FG, [ 8 ]:FI, [ 9 ]:M, [ 10 ]:Mas, [ 11 ]:Mat
    bias_vals = torch.zeros(12, dtype=torch.float32)

    # Resting bulk species start at non-dimensional C = 1.0.
    # Your network predicts in log1p space: log(1 + 1.0) = ln(2.0)
    resting_indices = [0, 4, 6, 7]  # RP, PT, AT, FG

    for idx in resting_indices:
        bias_vals[idx] = math.log(2.0)

    # Active/Surface species (AP, APR, APS, T, FI, M, Mas, Mat) remain 0.0

    # Apply the biases
    with torch.no_grad():
        target_layer.bias.copy_(bias_vals)

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

    for param in model.ode_func.parameters():
        param.requires_grad = True

    for name, param in model.biochem_decoder.named_parameters():
        if 'lora' not in name.lower():
            param.requires_grad = True

    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))

    return optim.AdamW([
        {'params': trainable_params, 'lr': base_lr},
        {'params': loss_weighter.parameters(), 'lr': 5e-2, 'weight_decay': 0.0}
    ], weight_decay=1e-5)


def compute_tier3_loss(model, data, kernels, loss_weighter, device, phys_cfg, bio_cfg):
    if hasattr(data, 're_actual'):
        phys_cfg.re_target = data.re_actual.mean().item()

    # Dynamically extract actual time steps and final time from the ground truth
    actual_num_steps = data.y.shape[ 0 ]
    actual_t_final = data.t[-1].item() if hasattr(data, 't') else bio_cfg.t_final

    dense_time_steps = actual_num_steps
    evaluation_times = torch.linspace(0.0, actual_t_final, steps=dense_time_steps, device=device)

    # 2. Forward Pass (Trajectory Generation)
    # Returns shape: [ Time, Nodes, 16 ]
    pred_series = model(data, evaluation_times)

    props = kernels.core._get_geometric_props(data)
    if hasattr(data, 'batch') and data.batch is not None:
        props['u_ref'] = data.u_ref[data.batch]
        props['d_bar'] = data.d_bar[data.batch]
    else:
        props['u_ref'] = data.u_ref
        props['d_bar'] = data.d_bar

    # 3. Supervised Data Loss (Evaluated ONLY at final time step T)
    pred_final = pred_series[-1]
    l_data_kine = torch.tensor(0.0, device=device)
    l_data_bio = torch.tensor(0.0, device=device)

    # FIX: Since we removed the 3x dense multiplier, the prediction frequency
    # perfectly matches the data frequency. No need to slice [::3] anymore!
    pred_series_data_freq = pred_series

    # Supervised Data Loss (Evaluated over ENTIRE TRAJECTORY)
    if hasattr(data, 'is_anchor'):
        node_is_anchor = data.is_anchor[data.batch] if hasattr(data, 'batch') else data.is_anchor
        if node_is_anchor.sum() > 0:
            pred_kine = pred_series_data_freq[:, node_is_anchor, :4]
            targ_kine = data.y[:, node_is_anchor, :4]
            kine_var = torch.clamp(torch.var(targ_kine, dim=(0, 1), keepdim=True), min=1e-2)
            # Use Huber loss to prevent massive initial gradients, scaled by variance
            l_data_kine = torch.mean(F.huber_loss(pred_kine, targ_kine, reduction='none') / kine_var)

            pred_bio = pred_series_data_freq[:, node_is_anchor, 4:16]
            targ_bio = data.y[:, node_is_anchor, 4:16]
            bio_var = torch.clamp(torch.var(targ_bio, dim=(0, 1), keepdim=True), min=1e-2)
            # Huber loss Delta of 1.0 transitions from MSE to MAE at an error of 1.0
            l_data_bio = torch.mean(F.huber_loss(pred_bio, targ_bio, reduction='none', delta=1.0) / bio_var)

    # 4. Physics PDE Loss (Evaluated over dense time sequence)
    # The dt is now based on the actual physical data steps
    dt = evaluation_times[1] - evaluation_times[0]
    d_pred_dt = (pred_series[1:] - pred_series[:-1]) / dt
    l_adr_fast, l_adr_slow, l_wall_bio, l_wall_phys, l_bio_io = 0.0, 0.0, 0.0, 0.0, 0.0
    num_steps = len(evaluation_times) - 1

    for t_idx in range(num_steps):
        # Evaluate physics at step t+1 using finite difference gradient
        pred_t = pred_series[t_idx + 1]
        d_dt_t = d_pred_dt[t_idx]

        vel_t = pred_t[:, 0:2]
        biochem_t = pred_t[:, 4:13]
        wall_t = pred_t[:, 13:16]

        dC_dt_t = d_dt_t[:, 4:13]
        dM_dt_t = d_dt_t[:, 13:16]

        # Accumulate physics residuals
        l_af, l_as = kernels.biochem_adr_residual(biochem_t, vel_t, props, data, d_pred_dt=dC_dt_t)
        l_wb, l_wp = kernels.biochem_wall_residual(biochem_t, wall_t, vel_t, props, data, dM_dt_t)
        l_bi, l_bo = kernels.biochem_inlet_outlet_residual(biochem_t, props, data)

        l_adr_fast += l_af
        l_adr_slow += l_as
        l_wall_bio += l_wb
        l_wall_phys += l_wp
        l_bio_io += (l_bi + l_bo)

    # Average physics over time steps
    l_adr_fast /= num_steps
    l_adr_slow /= num_steps
    l_wall_bio /= num_steps
    l_wall_phys /= num_steps
    l_bio_io /= num_steps

    # Fluid Mechanics (Base flow is pseudo-steady, evaluate once at t_final)
    l_mom = kernels.core.navier_stokes_residual(pred_final[:, 0:4], data, props=props)

    # Pass ALL 7 losses to the dynamic weighter
    all_losses = [
        l_adr_fast, l_adr_slow, l_wall_bio, l_wall_phys, l_bio_io,
        l_data_kine, l_data_bio
    ]
    loss = loss_weighter(all_losses)

    metrics = {
        "L_mom": l_mom.item(),
        "L_ADR_F": l_adr_fast.item(),
        "L_ADR_S": l_adr_slow.item(),
        "L_W_Bio": l_wall_bio.item(),
        "L_W_Phy": l_wall_phys.item(),
        "L_B_IO": l_bio_io.item(),
        "L_Data": l_data_bio.item()
    }
    return loss, metrics


def calculate_validation_metrics(pred, data, kernels, device):
    props = kernels.core._get_geometric_props(data)

    # --- Morphological Metric: Clot Dice Coefficient ---
    mu_eff = pred[ :, 3 ]
    mu_base = 0.0035  # Approximated reference viscosity
    clot_threshold = 20.0 * mu_base
    pred_clot = (mu_eff > clot_threshold).float()

    # FIX 1: Check the last dimension (features) instead of dimension 1 (nodes)
    if data.y.shape[ -1 ] > 3:
        # Convert GT back to dimensional before checking threshold
        mu_gt_dimensional = data.y[-1, :, 3] * mu_base if data.y.shape[-1] > 3 else torch.full_like(mu_eff, mu_base)
        gt_clot = (mu_gt_dimensional > clot_threshold).float()
        intersection = (pred_clot * gt_clot).sum()
        dice = (2.0 * intersection) / (pred_clot.sum() + gt_clot.sum() + 1e-8)
    else:
        dice = torch.tensor(0.0)

    # --- Hemodynamic Metric: WSS Pearson Correlation (Patent Lumen) ---
    mask_wall = data.mask_wall.view(-1).bool()

    if mask_wall.any() and data.y.shape[ -1 ] > 1:
        # Patent Lumen Mask (Walls where there is NO ground truth clot)
        patent_wall_mask = mask_wall & (gt_clot == 0) if data.y.shape[ -1 ] > 3 else mask_wall

        if patent_wall_mask.any():
            # Compute Predicted WSS
            c_u = kernels.core._compute_derivatives(pred[ :, 0:1 ], props)
            c_v = kernels.core._compute_derivatives(pred[ :, 1:2 ], props)
            dudx_p, dudy_p = c_u[ :, 0, 0 ], c_u[ :, 1, 0 ]
            dvdx_p, dvdy_p = c_v[ :, 0, 0 ], c_v[ :, 1, 0 ]

            mu_wall_p = pred[ patent_wall_mask, 3 ]
            tau_xx_p = 2.0 * mu_wall_p * dudx_p[ patent_wall_mask ]
            tau_yy_p = 2.0 * mu_wall_p * dvdy_p[ patent_wall_mask ]
            tau_xy_p = mu_wall_p * (dudy_p[ patent_wall_mask ] + dvdx_p[ patent_wall_mask ])

            nx = data.x[ patent_wall_mask, 3 ]
            ny = data.x[ patent_wall_mask, 4 ]

            tx_p, ty_p = tau_xx_p * nx + tau_xy_p * ny, tau_xy_p * nx + tau_yy_p * ny
            tn_p = tx_p * nx + ty_p * ny
            wss_pred = torch.sqrt((tx_p - tn_p * nx) ** 2 + (ty_p - tn_p * ny) ** 2 + 1e-8)

            # Compute Target WSS
            # FIX 3: Extract the final time step [-1] for ground truth velocity
            c_u_t = kernels.core._compute_derivatives(data.y[ -1, :, 0:1 ], props)
            c_v_t = kernels.core._compute_derivatives(data.y[ -1, :, 1:2 ], props)
            dudx_t, dudy_t = c_u_t[ :, 0, 0 ], c_u_t[ :, 1, 0 ]
            dvdx_t, dvdy_t = c_v_t[ :, 0, 0 ], c_v_t[ :, 1, 0 ]

            # FIX 4: Extract the final time step [-1] for ground truth viscosity
            mu_wall_t = data.y[ -1, patent_wall_mask, 3 ] if data.y.shape[ -1 ] > 3 else torch.full_like(mu_wall_p, mu_base)
            tau_xx_t = 2.0 * mu_wall_t * dudx_t[ patent_wall_mask ]
            tau_yy_t = 2.0 * mu_wall_t * dvdy_t[ patent_wall_mask ]
            tau_xy_t = mu_wall_t * (dudy_t[ patent_wall_mask ] + dvdx_t[ patent_wall_mask ])

            tx_t, ty_t = tau_xx_t * nx + tau_xy_t * ny, tau_xy_t * nx + tau_yy_t * ny
            tn_t = tx_t * nx + ty_t * ny
            wss_targ = torch.sqrt((tx_t - tn_t * nx) ** 2 + (ty_t - tn_t * ny) ** 2 + 1e-8)

            # Pearson Correlation
            stacked = torch.stack([wss_pred, wss_targ])
            pearson_corr = torch.corrcoef(stacked)[ 0, 1 ]
            if torch.isnan(pearson_corr): pearson_corr = torch.tensor(0.0)
        else:
            pearson_corr = torch.tensor(0.0)
    else:
        pearson_corr = torch.tensor(0.0)

    max_fibrin_pred = pred[ :, 12 ].max().item()

    return dice.item(), pearson_corr.item(), max_fibrin_pred


def train_tier3(epochs=50, lr=1e-3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device being used: {device}")

    phys_cfg = PhysicsConfig(tier="tier3", re_target=150.0)
    bio_cfg = BiochemConfig(tier="tier3")
    core_kernels = PhysicsKernels(phys_cfg=phys_cfg)
    kernels = BiochemPhysicsKernels(biochem_cfg=bio_cfg, core_physics_kernels=core_kernels)

    model = GNODE_Tier3(
        in_channels=12,
        spatial_channels=15,
        latent_dim=64,
        max_inner_iters=10
    ).to(device)

    # 1. Load Tier 2 Checkpoint (Frozen Kinematic Backbone)
    root = get_project_root()
    model_dir = root / "models"
    tier2_path = model_dir / "tier2_best_physics.pth"

    if tier2_path.exists():
        state_dict = torch.load(tier2_path, map_location=device, weights_only=True)

        mapped_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('encoder.'):
                mapped_state_dict[key.replace('encoder.', 'kin_encoder.')] = value
            elif key.startswith('core.'):
                mapped_state_dict[key.replace('core.', 'kin_processor.')] = value
            elif key.startswith('kinematics_decoder.'):
                mapped_state_dict[key] = value
            # Extract the frozen mu_encoder
            elif key.startswith('mu_encoder.'):
                mapped_state_dict[key] = value

        # --- Dynamic channel expansion surgery (Tier 2 -> Tier 3) ---
        if 'kin_encoder.0.weight' in mapped_state_dict:
            tier2_weight = mapped_state_dict['kin_encoder.0.weight']
            model_weight = model.kin_encoder[0].weight
            if tier2_weight.shape[1] != model_weight.shape[1]:
                print(f"🔧 Adapting Tier 2 encoder weights ({tier2_weight.shape[1]} -> {model_weight.shape[1]})...")
                new_weight = torch.zeros_like(model_weight)
                min_dim = min(tier2_weight.shape[1], model_weight.shape[1])
                new_weight[:, :min_dim] = tier2_weight[:, :min_dim]
                mapped_state_dict['kin_encoder.0.weight'] = new_weight
        # ------------------------------------------------------------

        model.load_state_dict(mapped_state_dict, strict=False)
        print("✅ Successfully loaded Tier 2 kinematic weights into Tier 3 backbone.")

    initialize_biochem_priors(model)
    # 7 targets: L_ADR_F, L_ADR_S, L_W_Bio, L_W_Phys, L_B_IO, L_Data_Kine, L_Data_Bio
    loss_weighter = DynamicLossWeighter(num_losses=7).to(device)

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
        progress = 0.0

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

            # Continue cooling T_scale from 8.0 down to 1.0 (Saved the ODE solver!)
            current_T_scale = 8.0 - progress * (8.0 - 1.0)

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

            loss, metrics = compute_tier3_loss(model, data, kernels, loss_weighter, device, phys_cfg, bio_cfg)
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

                print(
                    f"⚖️ Learned Weights -> ADR_F: {weights[ 0 ]:.2f} | ADR_S: {weights[ 1 ]:.2f} | "
                    f"W_Bio: {weights[ 2 ]:.2f} | W_Phys: {weights[ 3 ]:.2f} | Bio_IO: {weights[ 4 ]:.2f} | "
                    f"Data_Kine: {weights[ 5 ]:.2f} | Data_Bio: {weights[ 6 ]:.2f}"
                )

                for v_data in val_loader:
                    v_data = v_data.to(device)

                    # Define dynamic evaluation times for the validation pass
                    actual_val_steps = v_data.y.shape[0]
                    actual_val_t_final = v_data.t[-1].item() if hasattr(v_data, 't') else bio_cfg.t_final
                    val_eval_times = torch.linspace(0.0, actual_val_t_final, steps=actual_val_steps, device=device)

                    # Pass val_eval_times to the model
                    v_pred = model(v_data, val_eval_times)

                    if isinstance(v_pred, tuple):
                        v_pred = v_pred[0]

                    # v_pred is [ Time, Nodes, 16 ], slice [ -1 ] to evaluate final clot
                    d, p, f = calculate_validation_metrics(v_pred[-1], v_data, kernels, device)
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