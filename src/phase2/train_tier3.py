import os
import sys
import math
import gc
from typing import Optional

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
from src.phase2.gnode_tier3 import GNODE_Tier3, tier3_truth_node_mask
from src.phase2.physics_kernels_tier3 import BiochemPhysicsKernels
from src.phase1.physics.physics_kernels import PhysicsKernels
from src.config import (
    VesselConfig,
    PhysicsConfig,
    BiochemConfig,
    CurriculumConfig,
    STATE_CHANNEL_MU_EFF_ND,
)
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from src.phase1.utils.metrics import DynamicLossWeighter
from src.phase2.tier3_time_utils import resolve_tier3_times


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
    cfg_patients = VesselConfig(tier="tier3_patients")
    cfg_synthetic = VesselConfig(tier="tier3")

    patient_dir = cfg_patients.graph_output_dir
    synthetic_dir = cfg_synthetic.graph_output_dir

    patient_files = sorted(list(patient_dir.glob("*.pt"))) if patient_dir.exists() else []
    synthetic_files = sorted(list(synthetic_dir.glob("*.pt"))) if synthetic_dir.exists() else []

    if not patient_files and not synthetic_files:
        print(
            f"No Tier 3 graphs found in {patient_dir} or {synthetic_dir}. "
            f"Please generate/extract Tier 3 data first."
        )
        return []

    file_list = patient_files + synthetic_files
    print(
        f"📂 Found {len(patient_files)} Tier 3 anchor/patient graphs + "
        f"{len(synthetic_files)} Tier 3 synthetic graphs for lazy loading..."
    )

    # Dataset root is only a placeholder for PyG Dataset base class.
    return PatientDataset(root=str(get_project_root()), file_list=file_list)


def initialize_biochem_priors(model):
    print("🧬 Injecting physical priors into biochemistry decoder biases...")
    target_layer = model.biochem_decoder.linear if hasattr(model.biochem_decoder, 'linear') else model.biochem_decoder

    # FIX: Do not use strict zeros. It kills the backward gradient (grad_in = grad_out @ W).
    # Use a very small random initialization so the Neural ODE can learn.
    torch.nn.init.normal_(target_layer.weight, std=1e-4)

    bias_vals = torch.zeros(12, dtype=torch.float32)

    # Resting bulk species: C_nd = 1 ⇒ decoder output is log1p(1) = ln(2) (see _decode_species_log1p).
    resting_indices = [ 0, 4, 6, 7 ]  # RP, PT, AT, FG

    for idx in resting_indices:
        bias_vals[ idx ] = math.log(2.0)

    # Apply the biases
    with torch.no_grad():
        target_layer.bias.copy_(bias_vals)

    print("🛑 Initializing ODE function to near-zero derivative...")

    def _init_linear_like_near_zero(module, eps=1e-5):
        linear = module.linear if hasattr(module, 'linear') else module
        if not isinstance(linear, torch.nn.Linear):
            return
        weight = getattr(linear, 'weight_orig', None)
        if weight is None:
            weight = linear.weight
        torch.nn.init.uniform_(weight, a=-eps, b=eps)
        if linear.bias is not None:
            torch.nn.init.zeros_(linear.bias)

    # Target terminal projection layers in the ODE network so dz/dt starts near zero.
    terminal_layers = []
    for name, module in model.ode_func.named_modules():
        if not isinstance(module, (torch.nn.Linear, type(model.biochem_decoder))):
            continue

        has_linear_child = any(
            child_name and isinstance(child, (torch.nn.Linear, type(model.biochem_decoder)))
            for child_name, child in module.named_modules()
        )
        if not has_linear_child:
            terminal_layers.append((name, module))

    # Fallback safety: if architecture inspection misses terminals, damp all ODE linear projections.
    if not terminal_layers:
        terminal_layers = [
            (name, module)
            for name, module in model.ode_func.named_modules()
            if isinstance(module, (torch.nn.Linear, type(model.biochem_decoder)))
        ]

    with torch.no_grad():
        for _, layer in terminal_layers:
            _init_linear_like_near_zero(layer)
        if hasattr(model.ode_func, 'derivative_scale'):
            model.ode_func.derivative_scale.fill_(1e-5)


def make_tier3_dynamic_loss_weighter(curriculum: CurriculumConfig, device: str) -> DynamicLossWeighter:
    """Per-task Kendall bounds: cap physics weights, floor supervised data weights."""
    phys_ceiling = max(float(curriculum.tier3_physics_precision_ceiling), 1e-12)
    data_floor = max(float(curriculum.tier3_data_precision_floor), 1e-12)
    phys_min_lv = -math.log(phys_ceiling)
    data_max_lv = -math.log(data_floor)
    # 0–5: ADR_F, ADR_S, W_Bio, W_Phy, Bio_IO, NS_mom — 6–7: supervised Data_Kine, Data_Bio
    min_lv = [phys_min_lv] * 6 + [-8.0, -8.0]
    max_lv = [float("inf")] * 6 + [data_max_lv, data_max_lv]
    print(
        f"⚖️ Tier 3 loss weighter: physics prec ≤ {phys_ceiling:g} (log_var ≥ {phys_min_lv:.3f}), "
        f"data prec ≥ {data_floor:g} (log_var ≤ {data_max_lv:.3f}), "
        f"freeze_in_warmup={curriculum.tier3_weighter_freeze_during_warmup}"
    )
    return DynamicLossWeighter(num_losses=8, min_log_var=min_lv, max_log_var=max_lv).to(device)


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


def pretrain_autoencoder(model, loader, optimizer, device, epochs=5):
    print("\n🚀 --- Phase 3a: Autoencoder Pre-Training (Freezing ODE) ---")

    for param in model.ode_func.parameters():
        param.requires_grad = False

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for data in loader:
            data = data.to(device)
            mask = tier3_truth_node_mask(data, int(data.x.shape[0]), device)
            if not mask.any():
                continue

            optimizer.zero_grad()

            pred_species = model.autoencode(data)
            targ_species = data.y[0, :, 4:16]

            loss = F.mse_loss(pred_species[mask], targ_species[mask])
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        if num_batches == 0:
            print(
                f"AE Epoch {epoch:02d}: skipped (no graphs with COMSOL-labeled nodes — check is_anchor / re-extract)."
            )
            continue
        avg_loss = total_loss / num_batches
        print(f"AE Epoch {epoch:02d}: Recon Loss = {avg_loss:.4e}")

    for param in model.ode_func.parameters():
        param.requires_grad = True


def compute_tier3_loss(
    model,
    data,
    kernels,
    loss_weighter,
    device,
    bio_cfg,
    epoch=0,
    curriculum: Optional[CurriculumConfig] = None,
):
    curriculum = curriculum or CurriculumConfig()

    num_nodes_d = int(data.x.shape[0])
    truth_mask = tier3_truth_node_mask(data, num_nodes_d, device)

    re_ref = None
    if hasattr(data, 're_actual') and data.re_actual is not None:
        ra = data.re_actual
        re_ref = float(ra.mean().item()) if torch.is_tensor(ra) else float(ra)

    full_times = resolve_tier3_times(data, bio_cfg, device)

    actual_num_steps = int(data.y.shape[0])
    start_idx = 0
    end_idx = actual_num_steps
    y_true_trajectory = data.y
    teacher_forcing_ratio = 0.0

    wu = curriculum.tier3_warmup_epochs
    if model.training:
        if epoch < wu:
            teacher_forcing_ratio = 1.0
        else:
            decay_progress = (epoch - wu) / float(curriculum.tier3_teacher_force_decay_epochs)
            teacher_forcing_ratio = max(0.0, 1.0 - decay_progress)

        # Teacher forcing uses COMSOL labels only where ``tier3_truth_node_mask`` is True
        # (synthetic graphs: all False; patient graphs: spatially matched nodes only).
        if not truth_mask.any():
            teacher_forcing_ratio = 0.0

        # TBPTT windowing only when ``y`` is supervised (anchors). Synthetic placeholder ``y`` is zeros
        # and must not drive mid-trajectory times or IC slicing.
        if actual_num_steps > 2 and truth_mask.any():
            window_cap = max(2, actual_num_steps - 1)
            window_size = min(10 + (epoch // 2), window_cap)
            max_start = actual_num_steps - window_size
            if max_start > 0:
                start_idx = int(torch.randint(0, max_start, (1,), device=device).item())
            end_idx = start_idx + window_size
            y_true_trajectory = data.y[start_idx:end_idx]
            evaluation_times = full_times[start_idx:end_idx]
        else:
            evaluation_times = full_times
    else:
        evaluation_times = full_times

    # 2. Forward Pass (Trajectory Generation)
    pred_series = model(
        data,
        evaluation_times,
        y_true_trajectory=y_true_trajectory,
        teacher_forcing_ratio=teacher_forcing_ratio,
        start_idx=start_idx
    )

    props = kernels.core._get_geometric_props(data)
    if hasattr(data, 'batch') and data.batch is not None:
        props['u_ref'] = data.u_ref[data.batch]
        props['d_bar'] = data.d_bar[data.batch]
    else:
        props['u_ref'] = data.u_ref
        props['d_bar'] = data.d_bar

    # 3. Supervised data loss (full supervised time window on anchor nodes)
    pred_final = pred_series[-1]
    l_data_kine = torch.tensor(0.0, device=device)
    l_data_bio = torch.tensor(0.0, device=device)
    has_anchor_supervision = bool(truth_mask.any().item())

    # FIX: Since we removed the 3x dense multiplier, the prediction frequency
    # perfectly matches the data frequency. No need to slice [::3] anymore!
    pred_series_data_freq = pred_series
    target_series = y_true_trajectory.to(device)

    # Supervised loss only on COMSOL-trusted nodes (entire trajectory window).
    # l_data_kine: Huber on [u,v,p,mu_nd] vs batch variance scale (anchors only).
    # l_data_bio: Huber on log1p species channels vs per-channel floors (bulk + wall).
    if has_anchor_supervision:
        node_is_anchor = truth_mask
        pred_kine = pred_series_data_freq[:, node_is_anchor, :4]
        targ_kine = target_series[:, node_is_anchor, :4]
        kine_var = torch.clamp(torch.var(targ_kine, dim=(0, 1), keepdim=True), min=1e-2)
        l_data_kine = torch.mean(F.huber_loss(pred_kine, targ_kine, reduction='none') / kine_var)

        pred_bio = pred_series_data_freq[:, node_is_anchor, 4:16]
        targ_bio = target_series[:, node_is_anchor, 4:16]
        raw_bio_var = torch.var(targ_bio, dim=(0, 1), keepdim=True, unbiased=False)

        scales = bio_cfg.get_species_scales(device=device)
        apr_floor = torch.log1p((bio_cfg.APRcrit * bio_cfg.bulk_scale) / scales[2])
        aps_floor = torch.log1p((bio_cfg.APScrit * bio_cfg.bulk_scale) / scales[3])
        t_floor = torch.log1p((bio_cfg.Tcrit * bio_cfg.bulk_scale) / scales[5])
        baseline_floor = torch.tensor(0.01, dtype=targ_bio.dtype, device=device)

        bio_floors = torch.stack([
            baseline_floor, baseline_floor, apr_floor, aps_floor,
            baseline_floor, t_floor, baseline_floor, baseline_floor,
            baseline_floor, baseline_floor, baseline_floor, baseline_floor
        ]).view(1, 1, 12)

        safe_bio_var = torch.maximum(raw_bio_var, bio_floors)
        l_data_bio = torch.mean(F.huber_loss(pred_bio, targ_bio, reduction='none', delta=1.0) / safe_bio_var)

    # 4. Physics PDE Loss (Evaluated over dense time sequence)
    num_steps = len(evaluation_times) - 1
    z = torch.tensor(0.0, device=device)
    if num_steps <= 0:
        l_adr_fast = l_adr_slow = l_wall_bio = l_wall_phys = l_bio_io = z
    else:
        dt_intervals = (evaluation_times[1:] - evaluation_times[:-1]).view(-1, 1, 1)
        dt_intervals = torch.clamp(dt_intervals, min=1e-9)
        d_pred_dt = (pred_series[1:] - pred_series[:-1]) / dt_intervals
        l_adr_fast = l_adr_slow = l_wall_bio = l_wall_phys = l_bio_io = z

        for t_idx in range(num_steps):
            # Evaluate physics at step t+1 using finite difference gradient
            pred_t = pred_series[t_idx + 1]
            d_dt_t = d_pred_dt[t_idx]

            vel_t = pred_t[:, 0:2]
            biochem_t = pred_t[:, 4:13]
            wall_t = pred_t[:, 13:16]

            dC_dt_t = d_dt_t[:, 4:13]
            dM_dt_t = d_dt_t[:, 13:16]

            l_af, l_as = kernels.biochem_adr_residual(biochem_t, vel_t, props, data, d_pred_dt=dC_dt_t)
            l_wb, l_wp = kernels.biochem_wall_residual(biochem_t, wall_t, vel_t, props, data, dM_dt_t)
            l_bi, l_bo = kernels.biochem_inlet_outlet_residual(biochem_t, props, data)

            l_adr_fast = l_adr_fast + l_af
            l_adr_slow = l_adr_slow + l_as
            l_wall_bio = l_wall_bio + l_wb
            l_wall_phys = l_wall_phys + l_wp
            l_bio_io = l_bio_io + (l_bi + l_bo)

        inv = 1.0 / float(num_steps)
        l_adr_fast = l_adr_fast * inv
        l_adr_slow = l_adr_slow * inv
        l_wall_bio = l_wall_bio * inv
        l_wall_phys = l_wall_phys * inv
        l_bio_io = l_bio_io * inv

    # Fluid Mechanics (pseudo-steady snapshot at final time in the window)
    l_mom = kernels.core.navier_stokes_residual(
        pred_final[:, 0:4], data, props=props, re_ref=re_ref
    )

    # Eight Kendall tasks: skip supervised heads on non-anchor batches.
    all_losses = [
        l_adr_fast, l_adr_slow, l_wall_bio, l_wall_phys, l_bio_io, l_mom,
        l_data_kine, l_data_bio,
    ]
    task_active = [True] * 6 + [has_anchor_supervision, has_anchor_supervision]
    l_latent_reg = torch.tensor(0.0, device=device)
    if model.training and getattr(model.ode_func, "derivative_eval_count", 0) > 0:
        # Memory-safe detached metric from ODE evaluations in this forward pass.
        avg_deriv_energy = model.ode_func.derivative_energy_sum / max(model.ode_func.derivative_eval_count, 1)
        l_latent_reg = torch.tensor(avg_deriv_energy, dtype=torch.float32, device=device)
        model.ode_func.derivative_energy_sum = 0.0
        model.ode_func.derivative_eval_count = 0

    loss = loss_weighter(all_losses, task_active=task_active) + (1e-3 * l_latent_reg)

    metrics = {
        "L_mom": l_mom.item(),
        "L_ADR_F": l_adr_fast.item(),
        "L_ADR_S": l_adr_slow.item(),
        "L_W_Bio": l_wall_bio.item(),
        "L_W_Phy": l_wall_phys.item(),
        "L_B_IO": l_bio_io.item(),
        # Supervised COMSOL labels on anchor nodes only (Huber / variance-normalized).
        "L_Data_Kine": l_data_kine.item(),
        "L_Data_Bio": l_data_bio.item(),
        "L_Latent_Reg": l_latent_reg.item(),
        "TF_eff": float(teacher_forcing_ratio),
    }
    return loss, metrics


def calculate_validation_metrics(pred, data, kernels, device):
    props = kernels.core._get_geometric_props(data)

    num_nodes = int(data.num_nodes)
    truth_mask = tier3_truth_node_mask(data, num_nodes, pred.device)

    if pred.shape[0] != num_nodes:
        raise ValueError(
            "calculate_validation_metrics: pred rows must equal data.num_nodes "
            f"({pred.shape[0]} != {num_nodes})."
        )
    if data.y.dim() != 3:
        raise ValueError(
            "calculate_validation_metrics expects data.y shaped [T, N, C] (tier-3 trajectories); "
            f"got {tuple(data.y.shape)}."
        )
    if data.y.shape[1] != num_nodes:
        raise ValueError(
            "calculate_validation_metrics: data.y spatial dim must match num_nodes "
            f"({data.y.shape[1]} != {num_nodes})."
        )

    y_last = data.y[-1]

    mu_ch = STATE_CHANNEL_MU_EFF_ND
    mu_eff_nd = pred[ :, mu_ch ]
    mu_scale = kernels.core.cfg.mu_viscosity_nd_scale
    clot_threshold = 20.0 * mu_scale

    mu_pred_dimensional = kernels.core.cfg.viscosity_nd_to_si(mu_eff_nd)
    pred_clot = (mu_pred_dimensional > clot_threshold).float()

    dice = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    gt_clot = torch.zeros_like(pred_clot)

    if truth_mask.any() and data.y.shape[-1] > mu_ch + 1:
        mu_gt_dimensional = kernels.core.cfg.viscosity_nd_to_si(y_last[:, mu_ch])
        gt_clot = (mu_gt_dimensional > clot_threshold).float()
        pc = pred_clot[truth_mask]
        gc = gt_clot[truth_mask]
        intersection = (pc * gc).sum()
        dice = (2.0 * intersection) / (pc.sum() + gc.sum() + 1e-8)

    # --- Hemodynamic Metric: WSS Pearson (patent lumen, COMSOL-trusted wall nodes only) ---
    mask_wall = data.mask_wall.view(-1).bool()
    zero_pearson = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    if (
        truth_mask.any()
        and mask_wall.any()
        and data.y.shape[-1] > 1
    ):
        patent_wall_mask = mask_wall & truth_mask
        if data.y.shape[-1] > mu_ch + 1:
            patent_wall_mask = patent_wall_mask & (gt_clot == 0)

        if patent_wall_mask.any():
            # Compute Predicted WSS ([N,1] fields for WLS — same contract as physics_kernels)
            c_u = kernels.core._compute_derivatives(pred[ :, 0:1 ], props)
            c_v = kernels.core._compute_derivatives(pred[ :, 1:2 ], props)
            dudx_p, dudy_p = c_u[ :, 0, 0 ], c_u[ :, 1, 0 ]
            dvdx_p, dvdy_p = c_v[ :, 0, 0 ], c_v[ :, 1, 0 ]

            mu_wall_p = pred[ patent_wall_mask, mu_ch ]
            tau_xx_p = 2.0 * mu_wall_p * dudx_p[ patent_wall_mask ]
            tau_yy_p = 2.0 * mu_wall_p * dvdy_p[ patent_wall_mask ]
            tau_xy_p = mu_wall_p * (dudy_p[ patent_wall_mask ] + dvdx_p[ patent_wall_mask ])

            nx = data.x[ patent_wall_mask, 3 ]
            ny = data.x[ patent_wall_mask, 4 ]

            tx_p, ty_p = tau_xx_p * nx + tau_xy_p * ny, tau_xy_p * nx + tau_yy_p * ny
            tn_p = tx_p * nx + ty_p * ny
            wss_pred = torch.sqrt((tx_p - tn_p * nx) ** 2 + (ty_p - tn_p * ny) ** 2 + 1e-8)

            # Ground-truth WSS from final timestep velocities (same [N, C] layout as pred)
            c_u_t = kernels.core._compute_derivatives(y_last[ :, 0:1 ], props)
            c_v_t = kernels.core._compute_derivatives(y_last[ :, 1:2 ], props)
            dudx_t, dudy_t = c_u_t[ :, 0, 0 ], c_u_t[ :, 1, 0 ]
            dvdx_t, dvdy_t = c_v_t[ :, 0, 0 ], c_v_t[ :, 1, 0 ]

            mu_wall_t = (
                y_last[ patent_wall_mask, mu_ch ]
                if data.y.shape[ -1 ] > mu_ch + 1
                else torch.ones_like(mu_wall_p)
            )
            tau_xx_t = 2.0 * mu_wall_t * dudx_t[ patent_wall_mask ]
            tau_yy_t = 2.0 * mu_wall_t * dvdy_t[ patent_wall_mask ]
            tau_xy_t = mu_wall_t * (dudy_t[ patent_wall_mask ] + dvdx_t[ patent_wall_mask ])

            tx_t, ty_t = tau_xx_t * nx + tau_xy_t * ny, tau_xy_t * nx + tau_yy_t * ny
            tn_t = tx_t * nx + ty_t * ny
            wss_targ = torch.sqrt((tx_t - tn_t * nx) ** 2 + (ty_t - tn_t * ny) ** 2 + 1e-8)

            # Pearson is undefined for constant vectors; corrcoef returns NaN — treat as 0 for logging
            min_std = 1e-12
            if wss_pred.numel() < 2:
                pearson_corr = zero_pearson
            else:
                std_p = wss_pred.std(unbiased=False)
                std_t = wss_targ.std(unbiased=False)
                if std_p < min_std or std_t < min_std:
                    pearson_corr = zero_pearson
                else:
                    stacked = torch.stack([wss_pred, wss_targ])
                    pearson_corr = torch.corrcoef(stacked)[ 0, 1 ]
                    if torch.isnan(pearson_corr):
                        pearson_corr = zero_pearson
        else:
            pearson_corr = zero_pearson
    else:
        pearson_corr = zero_pearson

    max_fibrin_pred = pred[ :, 12 ].max().item()

    return dice.item(), pearson_corr.item(), max_fibrin_pred


def train_tier3(epochs=50, lr=1e-3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device being used: {device}")
    if device == "cuda":
        # Help avoid allocator fragmentation on long ODE runs.
        torch.cuda.empty_cache()

    phys_cfg = PhysicsConfig(tier="tier3")
    bio_cfg = BiochemConfig(tier="tier3")
    curriculum = CurriculumConfig()
    core_kernels = PhysicsKernels(phys_cfg=phys_cfg)
    kernels = BiochemPhysicsKernels(biochem_cfg=bio_cfg, core_physics_kernels=core_kernels)

    # PASS PHYS_CFG TO MODEL
    model = GNODE_Tier3(
        phys_cfg=phys_cfg,
        in_channels=12,
        spatial_channels=15,
        latent_dim=64,
        max_inner_iters=10,
        mu_ratio_max=bio_cfg.mu_ratio_max,
        mat_crit=bio_cfg.viscosity_mat_crit,
        fi_crit=bio_cfg.viscosity_fi_crit,
        temp_mat=bio_cfg.viscosity_gnode_temp_mat,
        temp_fi=bio_cfg.viscosity_gnode_temp_fi,
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
    loss_weighter = make_tier3_dynamic_loss_weighter(curriculum, device)

    dataset = load_dataset()
    if len(dataset) == 0:
        return

    # Keep loading lazy: split by file path metadata instead of materializing all graphs.
    all_files = list(dataset.file_list)
    anchors, physics = [], []
    print("🔎 Indexing Tier 3 files by anchor flag (lazy split)...")
    for graph_path in all_files:
        graph = torch.load(graph_path, map_location="cpu", weights_only=False)
        ia = getattr(graph, "is_anchor", None)
        if ia is None:
            is_anchor = False
        elif torch.is_tensor(ia):
            is_anchor = bool(ia.any().item())
        else:
            is_anchor = bool(ia)
        if is_anchor:
            anchors.append(graph_path)
        else:
            physics.append(graph_path)
        del graph
    gc.collect()

    random.seed(42)
    random.shuffle(anchors)
    random.shuffle(physics)

    # Robust split: keep at least one anchor in training whenever anchors exist.
    if len(dataset) == 1:
        print("⚠️ Only one graph found. Using it for both Training and Validation.")
        only = [all_files[0]]
        train_data = only
        val_data = only
    else:
        if len(anchors) <= 1:
            train_anchors = anchors[:]  # keep the single anchor in train to satisfy warmup sampler
            val_anchors = []
        else:
            split_idx_a = int(0.9 * len(anchors))
            split_idx_a = max(1, min(split_idx_a, len(anchors) - 1))
            train_anchors = anchors[:split_idx_a]
            val_anchors = anchors[split_idx_a:]

        if len(physics) <= 1:
            train_physics = physics[:]
            val_physics = []
        else:
            split_idx_p = int(0.9 * len(physics))
            split_idx_p = max(1, min(split_idx_p, len(physics) - 1))
            train_physics = physics[:split_idx_p]
            val_physics = physics[split_idx_p:]

        train_data = train_anchors + train_physics
        val_data = val_anchors + val_physics

        # Safety fallback if split produced empty validation
        if len(val_data) == 0:
            val_data = train_data

    train_dataset = PatientDataset(root=str(get_project_root()), file_list=train_data)
    val_dataset = PatientDataset(root=str(get_project_root()), file_list=val_data)

    # IMPORTANT:
    # Tier 3 graphs store trajectories as y: [T, N, 16]. With vanilla PyG batching,
    # x concatenates over nodes while y concatenates over time, which misaligns tensors.
    # Use batch_size=1 and gradient accumulation for stable/equivalent optimization.
    accumulation_steps = 4

    # Use simple shuffled loaders with batch_size=1 to preserve [T, N, 16] integrity.
    train_anchor_count = len(train_anchors) if len(dataset) > 1 else 1
    if train_anchor_count == 0:
        print("⚠️ No anchors in training split; running physics-only updates.")
    loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    optimizer = setup_tier3_optimization(model, loss_weighter, base_lr=lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    pretrain_autoencoder(model, loader, optimizer, device, epochs=5)

    best_dice = -1.0

    print("\n🚀 --- Starting Phase 3: Segregated Bio-Fluid Coupling ---")

    for epoch in range(epochs):
        wu = curriculum.tier3_warmup_epochs

        if epoch < wu:
            current_mu_ratio = bio_cfg.mu_ratio_init
            span = max(float(wu), 1.0)
            current_T_scale = curriculum.tier3_t_scale_warmup_initial - (epoch / span) * (
                curriculum.tier3_t_scale_warmup_initial - curriculum.tier3_t_scale_warmup_final
            )
        else:
            coupled_denom = max(1, epochs - wu - 1)
            progress = (epoch - wu) / float(coupled_denom)
            current_mu_ratio = bio_cfg.mu_ratio_init + progress * (
                bio_cfg.mu_ratio_max - bio_cfg.mu_ratio_init
            )
            current_T_scale = curriculum.tier3_t_scale_coupled_initial - progress * (
                curriculum.tier3_t_scale_coupled_initial - curriculum.tier3_t_scale_coupled_final
            )

        # Push updates to the network and kernels
        model.mu_ratio_max = current_mu_ratio

        # Unify the curriculum temperature
        model.T_scale = current_T_scale
        kernels.kinetics.T_scale = current_T_scale

        # FIX: Capitalized 'T' here as well
        print(f"\n⏳ Epoch {epoch:02d} | mu_ratio: {current_mu_ratio:.1f}x | T_scale: {current_T_scale:.2f}")

        if curriculum.tier3_weighter_freeze_during_warmup:
            learn_w = epoch >= wu
            loss_weighter.log_vars.requires_grad_(learn_w)
            if learn_w and epoch == wu:
                print("⚖️  Unfreezing loss weighter log_vars after Tier 3 warmup.")

        model.train()
        total_loss_epoch = 0.0
        optimizer.zero_grad()

        # Epoch-level TF schedule (actual per-batch TF_eff may be 0 on graphs with no labeled nodes).
        if epoch < curriculum.tier3_warmup_epochs:
            teacher_forcing_ratio = 1.0
        else:
            decay_progress = (epoch - curriculum.tier3_warmup_epochs) / float(
                curriculum.tier3_teacher_force_decay_epochs
            )
            teacher_forcing_ratio = max(0.0, 1.0 - decay_progress)

        # EMA-smoothed progress metrics for less noisy tqdm feedback.
        ema_metrics = None
        ema_alpha = 0.05

        pbar = tqdm(loader, desc=f"Tier 3 Ep {epoch:02d}")
        for batch_idx, data in enumerate(pbar):
            data = data.to(device)
            data.x.requires_grad_(True)

            loss, metrics = compute_tier3_loss(
                model, data, kernels, loss_weighter, device, bio_cfg, epoch=epoch, curriculum=curriculum
            )
            loss = loss / accumulation_steps

            if torch.isnan(loss):
                print(f"\n⚠️ NaN detected in loss at epoch {epoch}! Skipping micro-batch.")
                continue

            loss.backward()

            if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(loader)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            current_l_tot = loss.item() * accumulation_steps
            total_loss_epoch += current_l_tot

            if ema_metrics is None:
                ema_metrics = {
                    "L_tot": current_l_tot,
                    "L_Data_Kine": metrics["L_Data_Kine"],
                    "L_Data_Bio": metrics["L_Data_Bio"],
                    "L_ADR_F": metrics['L_ADR_F'],
                    "L_W_Bio": metrics['L_W_Bio'],
                    "L_W_Phy": metrics['L_W_Phy']
                }
            else:
                ema_metrics["L_tot"] = (1 - ema_alpha) * ema_metrics["L_tot"] + ema_alpha * current_l_tot
                ema_metrics["L_Data_Kine"] = (1 - ema_alpha) * ema_metrics["L_Data_Kine"] + ema_alpha * metrics["L_Data_Kine"]
                ema_metrics["L_Data_Bio"] = (1 - ema_alpha) * ema_metrics["L_Data_Bio"] + ema_alpha * metrics["L_Data_Bio"]
                ema_metrics["L_ADR_F"] = (1 - ema_alpha) * ema_metrics["L_ADR_F"] + ema_alpha * metrics['L_ADR_F']
                ema_metrics["L_W_Bio"] = (1 - ema_alpha) * ema_metrics["L_W_Bio"] + ema_alpha * metrics['L_W_Bio']
                ema_metrics["L_W_Phy"] = (1 - ema_alpha) * ema_metrics["L_W_Phy"] + ema_alpha * metrics['L_W_Phy']

            pbar.set_postfix({
                "L_tot": f"{ema_metrics['L_tot']:.2e}",
                "L_Kine": f"{ema_metrics['L_Data_Kine']:.2e}",
                "L_Bio": f"{ema_metrics['L_Data_Bio']:.2e}",
                "L_ADR_F": f"{ema_metrics['L_ADR_F']:.2e}",
                "L_W_Bio": f"{ema_metrics['L_W_Bio']:.2e}",
                "L_W_Phy": f"{ema_metrics['L_W_Phy']:.2e}",
                "TF_eff": f"{metrics['TF_eff']:.2f}",
            })

        scheduler.step()

        # Validation & Metrics
        if epoch % 2 == 0:
            # 1. Explicitly set to evaluation mode
            model.eval()
            val_dice_total, val_pearson_total, val_fibrin_total = 0.0, 0.0, 0.0

            with torch.no_grad():
                safe_vars = loss_weighter.clamped_log_vars()
                weights = torch.exp(-safe_vars)

                print(
                    f"⚖️ Learned Weights -> ADR_F: {weights[0]:.2f} | ADR_S: {weights[1]:.2f} | "
                    f"W_Bio: {weights[2]:.2f} | W_Phys: {weights[3]:.2f} | Bio_IO: {weights[4]:.2f} | "
                    f"NS_mom: {weights[5]:.2f} | Data_Kine: {weights[6]:.2f} | Data_Bio: {weights[7]:.2f}"
                )

                for v_data in val_loader:
                    v_data = v_data.to(device)

                    actual_val_steps = int(v_data.y.shape[0])
                    val_eval_times = resolve_tier3_times(v_data, bio_cfg, device)

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