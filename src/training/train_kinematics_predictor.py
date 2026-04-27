"""
Unified Kinematics Predictor Training with Mathematical Continuation Ramp.
Implements dynamic dataset swapping, Carreau-Yasuda parameter ramping,
and curriculum-based loss isolation.
"""
import argparse
import json
import os
import random
import re
import time
import warnings
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.architecture.ginodeq import GINO_DEQ
from src.config import VesselConfig, PhysicsConfig, PredChannels
from src.core_physics.physics_kernels import PhysicsKernels
from src.utils.anchor_mask import graph_has_anchor, anchor_node_mask
from src.utils.kinematics_physics_terms import compute_kinematics_physics_terms
from src.utils.metrics import DynamicLossWeighter, quantify_performance
from src.utils.paths import stage_a_dir
from src.utils.training_diary import TrainingDiary

# Ignore known PyTorch scheduler deprecation noise in training logs.
warnings.filterwarnings("ignore", category=UserWarning, message="The epoch parameter.*")

# -------------------------------------------------------------------------
# Curriculum Definitions
# -------------------------------------------------------------------------
STAGE1_END_EPOCH = 40
STAGE2_END_EPOCH = 60


def get_stage_physics(epoch: int, s1_end: int, s2_end: int):
    """
    Returns (stage, n, mu_0, target_rheology) based on the training epoch.
    Stage 1: Newtonian Anchor
    Stage 2: Soft Transition (Linear Ramp)
    Stage 3: Target State (Full Carreau-Yasuda)
    """
    if epoch < s1_end:
        return 1, 1.0, 0.0035, "newtonian"
    elif epoch < s2_end:
        alpha = (epoch - s1_end) / float(s2_end - s1_end)
        n = 1.0 - alpha * (1.0 - 0.6)
        mu_0 = 0.0035 + alpha * (0.035 - 0.0035)
        # Stage-2 keeps Newtonian labels while physics ramps internally.
        return 2, n, mu_0, "newtonian"
    else:
        return 3, 0.358, 0.056, "carreau"


# -------------------------------------------------------------------------
# Data Loading & Management
# -------------------------------------------------------------------------
def load_dataset(phase: str, rheology: str | None = None):
    cfg = VesselConfig(phase=phase)
    data_dir = cfg.graph_output_dir
    if rheology:
        data_dir = data_dir / str(rheology).lower()

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}. "
            "Expected rheology-split graphs under graphs_kinematics/<newtonian|carreau>."
        )

    paths = sorted(data_dir.glob("vessel_*.pt"))
    if not paths:
        raise RuntimeError(
            f"No graph files found in dataset directory: {data_dir}. "
            "Expected at least one vessel_*.pt file."
        )
    dataset = []
    print(f"📂 Loading {len(paths)} graphs from {data_dir}...")
    for f in tqdm(paths, leave=False):
        dataset.append(torch.load(f, weights_only=False))
    return dataset


def split_anchor_physics(dataset, seed=42, train_ratio=0.9):
    anchors = [d for d in dataset if d.is_anchor.any().item()]
    physics = [d for d in dataset if not d.is_anchor.any().item()]
    rng = random.Random(seed)
    rng.shuffle(anchors)
    rng.shuffle(physics)
    split_a = int(train_ratio * len(anchors))
    split_p = int(train_ratio * len(physics))

    return {
        "train": anchors[:split_a] + physics[:split_p],
        "val": anchors[split_a:] + physics[split_p:],
        "n_anchors": len(anchors[:split_a]),
        "n_physics": len(physics[:split_p]),
    }


def evaluate_mass_flow_health(model, dataset, device, max_graphs=12):
    """Fallback physics diagnostic when anchor labels are unavailable.

    Uses boundary velocity magnitudes as a proxy for volumetric flux:
    - inlet_flux ≈ mean(|u| on inlet nodes)
    - outlet_flux ≈ mean(|u| on outlet nodes)
    Reports normalized inlet/outlet imbalance and a collapse score indicating
    how close both boundary fluxes are to zero (trivial stagnant solution risk).
    """
    model.eval()
    eps = 1e-8
    in_means = []
    out_means = []
    n_used = 0
    with torch.no_grad():
        for d in dataset:
            if n_used >= max_graphs:
                break
            if not hasattr(d, "mask_inlet") or not hasattr(d, "mask_outlet"):
                continue
            if int(d.mask_inlet.sum().item()) == 0 or int(d.mask_outlet.sum().item()) == 0:
                continue
            dd = d.clone().to(device)
            out = model(dd, solver="anderson")
            pred = out[0] if isinstance(out, tuple) else out
            speed = torch.norm(pred[:, :2], dim=1)
            in_flux = float(speed[dd.mask_inlet].mean().item())
            out_flux = float(speed[dd.mask_outlet].mean().item())
            in_means.append(in_flux)
            out_means.append(out_flux)
            n_used += 1
    model.train()

    if n_used == 0:
        return None

    inlet_mean = sum(in_means) / n_used
    outlet_mean = sum(out_means) / n_used
    flow_ref = max(inlet_mean, outlet_mean, eps)
    imbalance = abs(inlet_mean - outlet_mean) / (inlet_mean + outlet_mean + eps)
    collapse_score = 1.0 - ((inlet_mean + outlet_mean) / (2.0 * flow_ref + eps))
    return {
        "n_graphs": n_used,
        "inlet_flux": inlet_mean,
        "outlet_flux": outlet_mean,
        "imbalance": imbalance,
        "collapse_score": max(0.0, min(1.0, collapse_score)),
    }


# -------------------------------------------------------------------------
# Forward & Loss Computation
# -------------------------------------------------------------------------
def compute_step_loss(
    model,
    data,
    kernels,
    loss_weighter,
    solver,
    device,
    stage,
    current_n,
    current_mu_0,
    weight_data_base: float,
    weight_mu_base: float,
    weight_wss_base: float,
):
    # 1. Inject dynamic physics parameters into the kernels
    # mu_viscosity_nd_scale is typically mu_inf (0.0035)
    mu_nd_scale = kernels.cfg.mu_viscosity_nd_scale
    kernels.mu_0_nd = current_mu_0 / mu_nd_scale

    # 2. Forward pass
    out = model(
        data,
        solver=solver,
        anderson_beta=0.8,
        anderson_warmup_iters=5,
        current_n=current_n,
    )
    pred, jac_loss = out if isinstance(out, tuple) else (out, torch.tensor(0.0, device=device))

    # 3. Get generic terms
    terms = compute_kinematics_physics_terms(
        pred,
        data,
        kernels,
        phase="kinematics",
        distillation=False,
        carreau_n=current_n,
    )

    # 4. Curriculum Biochem phaseranching
    l_mom = terms["l_mom"]
    l_cont = terms["l_cont"]
    l_bc = terms["l_bc"]
    l_io = terms["l_io"]
    l_wss = terms.get("l_wss", torch.tensor(0.0, device=device))
    l_data_kine = terms.get("l_data_kine", torch.tensor(0.0, device=device))
    p_grad_loss = torch.tensor(0.0, device=device)

    if stage in (1, 3):
        props = kernels._get_geometric_props(data)
        c_p_pred = kernels._compute_derivatives(pred[:, PredChannels.P:PredChannels.P + 1], props)
        c_p_true = kernels._compute_derivatives(data.y[:, PredChannels.P:PredChannels.P + 1], props)
        p_pred_grad = c_p_pred[:, 0:2, 0]
        p_true_grad = c_p_true[:, 0:2, 0]
        node_is_anchor = anchor_node_mask(data)
        if node_is_anchor is not None and int(node_is_anchor.sum().item()) > 0:
            # Non-dimensionalize physical gradients prior to squaring in MSE.
            if hasattr(data, "d_bar"):
                d_bar = data.d_bar
                if torch.is_tensor(d_bar):
                    d_bar_flat = d_bar.view(-1)
                    if d_bar_flat.numel() == data.num_nodes:
                        length_scale = d_bar_flat[node_is_anchor].view(-1, 1)
                    else:
                        length_scale = d_bar_flat[:1].reshape(1, 1)
                else:
                    length_scale = torch.tensor([[float(d_bar)]], device=device, dtype=p_pred_grad.dtype)
            else:
                length_scale = torch.tensor([[1e-4]], device=device, dtype=p_pred_grad.dtype)
            p_pred_grad_nd = p_pred_grad[node_is_anchor] * length_scale
            p_true_grad_nd = p_true_grad[node_is_anchor] * length_scale
            p_grad_loss = torch.nn.functional.mse_loss(
                p_pred_grad_nd,
                p_true_grad_nd,
            )

    # Stage-specific loss manipulation
    if stage in (1, 2):
        # Constant-field preconditioning: supervise mu decoder toward the curriculum viscosity target.
        target_mu_nd = torch.full_like(pred[:, PredChannels.MU_EFF_ND], current_mu_0 / mu_nd_scale)
        l_data_mu = torch.nn.functional.mse_loss(
            pred[:, PredChannels.MU_EFF_ND], target_mu_nd
        )
        if stage == 1:
            weight_data = weight_data_base
            weight_mu = weight_mu_base
            weight_wss = weight_wss_base
        else:
            # Stage 2 keeps PDE-only kinematics but preserves rheology supervision while ramping.
            l_data_kine = l_data_kine * 0.0
            l_wss = l_wss * 0.0
            weight_data = 0.0
            weight_mu = weight_mu_base
            weight_wss = 0.0
    else:
        # Stage 3: Target phase. Both data (now matching physics) and PDEs.
        l_data_mu = terms.get("l_data_mu", torch.tensor(0.0, device=device))
        weight_data = weight_data_base
        weight_mu = weight_mu_base
        weight_wss = weight_wss_base

    # 5. Kendall Loss Weighting for PDEs
    raw_pdes = [l_mom, l_cont]
    weighted_pdes = loss_weighter(raw_pdes)

    # 6. Final Composite Loss
    loss = (
        weighted_pdes
        + (weight_data * l_data_kine)
        + (weight_mu * l_data_mu)
        + (5.0 * l_bc)
        + (5.0 * l_io)
        + (1.0 * p_grad_loss)
        + (weight_wss * l_wss)
        + (0.1 * jac_loss)
    )

    weighted_data_kine = weight_data * l_data_kine
    weighted_data_mu = weight_mu * l_data_mu
    weighted_bc = 5.0 * l_bc
    weighted_io = 5.0 * l_io
    weighted_pgrad = 1.0 * p_grad_loss
    weighted_wss = weight_wss * l_wss
    weighted_jac = 0.1 * jac_loss
    metrics = {
        "L_mom": l_mom.item(),
        "L_cont": l_cont.item(),
        "L_data": l_data_kine.item(),
        "L_mu": l_data_mu.item(),
        "L_bc": l_bc.item(),
        "L_io": l_io.item(),
        "L_wss": l_wss.item(),
        "L_jac": jac_loss.item(),
        "L_pgrad": p_grad_loss.item(),
        "L_total": loss.item(),
        "C_weighted_pde": weighted_pdes.item(),
        "C_data_kine": weighted_data_kine.item(),
        "C_data_mu": weighted_data_mu.item(),
        "C_bc": weighted_bc.item(),
        "C_io": weighted_io.item(),
        "C_pgrad": weighted_pgrad.item(),
        "C_wss": weighted_wss.item(),
        "C_jac": weighted_jac.item(),
    }
    return loss, metrics


# -------------------------------------------------------------------------
# Training Loop
# -------------------------------------------------------------------------
def train_kinematics(
    *,
    epochs: int = 100,
    adam_epochs: int = 85,
    stage1_end_epoch: int = STAGE1_END_EPOCH,
    stage2_end_epoch: int = STAGE2_END_EPOCH,
    resume_from: str | None = None,
    accum_steps: int = 2,
    weight_data: float = 500.0,
    weight_mu: float = 10.0,
    weight_wss: float = 10.0,
    max_lbfgs_graphs: int = 4,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Handoff to L-BFGS in late Stage 3 by default.

    phys_cfg = PhysicsConfig(phase="kinematics")  # Kinematics supports Carreau
    kernels = PhysicsKernels(phys_cfg=phys_cfg)
    model = GINO_DEQ(
        in_channels=15,
        out_channels=5,
        latent_dim=256,
        max_iters=25,
        num_fourier_freqs=16,
        phys_cfg=phys_cfg,
        activation_fn="silu",
        use_hard_bcs=True,
        use_siren_decoder=True,
        use_width_priors=True,
    ).to(device)

    # Kendall loss weighter bounds
    loss_weighter = DynamicLossWeighter(num_losses=2).to(device)
    opt_params = list(model.parameters()) + list(loss_weighter.parameters())
    optimizer = optim.AdamW(opt_params, lr=1e-4, weight_decay=1e-5)
    warm_up_epochs = 5
    decay_epochs = max(1, adam_epochs - warm_up_epochs)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warm_up_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=decay_epochs, eta_min=1e-6)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warm_up_epochs],
    )

    # State tracking
    current_phase_loaded = None
    train_data, val_data = [], []
    hard_anchor_multiplier = {}
    lbfgs_initialized = False
    static_batches = []
    n_anchors, n_physics = 0, 0
    best_val_composite_loss = float("inf")
    accum_steps = max(1, int(accum_steps))
    max_lbfgs_graphs = max(1, int(max_lbfgs_graphs))
    start_epoch = 0

    if resume_from:
        print(f"🔁 Resuming training from: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            if "loss_weighter_state_dict" in ckpt:
                loss_weighter.load_state_dict(ckpt["loss_weighter_state_dict"])
            if "optimizer_state_dict" in ckpt and ckpt.get("optimizer_name", "AdamW") == "AdamW":
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                except (ValueError, RuntimeError):
                    print("⚠️ Could not restore AdamW optimizer state; continuing with fresh optimizer.")
            if "scheduler_state_dict" in ckpt:
                try:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                except (ValueError, RuntimeError):
                    print("⚠️ Could not restore scheduler state; continuing with fresh scheduler.")
            start_epoch = int(ckpt.get("epoch", -1)) + 1
            best_val_composite_loss = float(ckpt.get("best_val_composite_loss", best_val_composite_loss))
            # Always re-enter LBFGS via normal handoff so static batches are rebuilt deterministically.
            lbfgs_initialized = False
            print(f"✅ Loaded full training state (next epoch: {start_epoch})")
        else:
            model.load_state_dict(ckpt)
            m = re.search(r"kinematics_ckpt_(\d+)\.pth$", str(resume_from))
            if m:
                start_epoch = int(m.group(1))
            print(f"✅ Loaded model-only checkpoint (next epoch: {start_epoch})")

    diary = TrainingDiary("kinematics")
    diary.log_run_start(
        epochs=int(epochs),
        adam_epochs=int(adam_epochs),
        stage1_end_epoch=int(stage1_end_epoch),
        stage2_end_epoch=int(stage2_end_epoch),
        device=str(device),
    )

    def make_loader(data_split, n_anchors, n_physics):
        # 50/50 Weighted Random Sampler logic extracted from Kinematics
        if n_anchors > 0 and n_physics > 0:
            w_anchor = 0.5 / n_anchors
            w_phys = 0.5 / n_physics
            weights = []
            for d in data_split:
                if graph_has_anchor(d):
                    gkey = int(getattr(d, "config_id", 0))
                    weights.append(w_anchor * hard_anchor_multiplier.get(gkey, 1.0))
                else:
                    weights.append(w_phys)
            sampler = torch.utils.data.WeightedRandomSampler(weights, len(data_split), replacement=True)
            return DataLoader(data_split, batch_size=1, sampler=sampler)
        return DataLoader(data_split, batch_size=1, shuffle=True)

    def refresh_hard_mining(epoch, dataset):
        _ = epoch  # reserved for parity with legacy hooks
        model.eval()
        rows = []
        with torch.no_grad():
            for d in dataset:
                if not graph_has_anchor(d):
                    continue
                dd = d.clone().to(device)
                out = model(dd, solver="anderson")
                pred = out if isinstance(out, tuple) else out
                mask = anchor_node_mask(dd)
                if mask is not None and mask.sum() > 0:
                    rel = torch.norm(pred[mask, :2] - dd.y[mask, :2]) / torch.clamp(
                        torch.norm(dd.y[mask, :2]), min=1e-8
                    )
                    gkey = int(getattr(dd, "config_id", 0))
                    rows.append((gkey, float(rel.item())))
        if rows:
            errs = torch.tensor([r[1] for r in rows], dtype=torch.float32)
            q = float(torch.quantile(errs, torch.tensor(0.7)))
            for gkey, err in rows:
                hard_anchor_multiplier[gkey] = (1.0 + 0.8) if err >= q else 1.0  # hard_alpha = 0.8
        model.train()

    print("🚀 Starting Unified Kinematics Training...")

    for epoch in range(start_epoch, epochs):
        stage, current_n, current_mu_0, target_rheology = get_stage_physics(
            epoch, int(stage1_end_epoch), int(stage2_end_epoch)
        )
        target_phase = "kinematics"

            # 1. Dynamic DataLoader Swapping
        # 1. Dynamic DataLoader Swapping
        if current_phase_loaded != target_rheology:
            print(
                f"\n🔄 Swapping Dataset to {target_phase.upper()}/{target_rheology.upper()} for Stage {stage} "
                f"(n={current_n:.3f}, μ0={current_mu_0:.4f})"
            )
            dataset = load_dataset(target_phase, target_rheology)
            splits = split_anchor_physics(dataset)
            train_data, val_data = splits["train"], splits["val"]
            n_anchors, n_physics = splits["n_anchors"], splits["n_physics"]
            current_phase_loaded = target_rheology

            # Reset hard mining when swapping datasets
            hard_anchor_multiplier.clear()
            if stage == 3 and not lbfgs_initialized:
                print("🧹 Resetting AdamW momentum buffers for Stage 3 Target Phase...")
                optimizer.state.clear()

        # 2. Hard Mining Management
        if stage in (1, 3) and epoch % 4 == 0 and not lbfgs_initialized:
            print("⛏️ Refreshing Hard Negative Anchor Weights...")
            refresh_hard_mining(epoch, train_data)
        elif epoch % 4 == 0 and not lbfgs_initialized:
            # During ramp/no-anchor phases, anchor rel-L2 is not informative.
            flow_diag = evaluate_mass_flow_health(model, train_data, device)
            if flow_diag is None:
                print("🧪 Flow diagnostic skipped (missing inlet/outlet masks).")
            else:
                print(
                    "🧪 Flow diagnostic "
                    f"(graphs={flow_diag['n_graphs']}): "
                    f"flux_in={flow_diag['inlet_flux']:.3e}, "
                    f"flux_out={flow_diag['outlet_flux']:.3e}, "
                    f"imbalance={flow_diag['imbalance']:.3f}, "
                    f"collapse={flow_diag['collapse_score']:.3f}"
                )
        loader = make_loader(train_data, n_anchors, n_physics)

        # 3. Kendall Loss Weighter Management
        if stage == 2:
            loss_weighter.requires_grad_(False)  # Freeze during physics shift
        else:
            loss_weighter.requires_grad_(True)

        # 4. L-BFGS Handoff (Kinematics preservation)
        if epoch >= adam_epochs and not lbfgs_initialized:
            print("\n⚡ Switching to L-BFGS Optimizer for final fixed-point refinement...")
            loss_weighter.requires_grad_(False)
            lbfgs_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.LBFGS(
                lbfgs_params, lr=0.01, max_iter=20, history_size=30, line_search_fn="strong_wolfe"
            )
            static_batches = []
            for d in list(loader)[:max_lbfgs_graphs]:
                static_batches.append(d.clone().to(device))
            if not static_batches:
                raise RuntimeError("L-BFGS initialization failed: no batches available to cache.")
            lbfgs_initialized = True

        model.train()
        total_loss = 0.0
        component_sums = {
            "C_weighted_pde": 0.0,
            "C_data_kine": 0.0,
            "C_data_mu": 0.0,
            "C_bc": 0.0,
            "C_io": 0.0,
            "C_pgrad": 0.0,
            "C_wss": 0.0,
            "C_jac": 0.0,
        }

        if not lbfgs_initialized:
            pbar = tqdm(loader, desc=f"Ep {epoch:02d} [S{stage}: n={current_n:.3f}, μ0={current_mu_0:.4f}]")
            optimizer.zero_grad()
            accum_counter = 0
            for idx, data in enumerate(pbar):
                loss, metrics = compute_step_loss(
                    model,
                    data.to(device),
                    kernels,
                    loss_weighter,
                    "anderson" if epoch > 5 else "picard",
                    device,
                    stage,
                    current_n,
                    current_mu_0,
                    weight_data,
                    weight_mu,
                    weight_wss,
                )
                if torch.isnan(loss):
                    continue
                scaled_loss = loss / accum_steps
                scaled_loss.backward()
                accum_counter += 1
                grad_norm = 0.0

                if (idx + 1) % accum_steps == 0 or (idx + 1) == len(loader):
                    if accum_counter == 0:
                        continue
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(opt_params, 1.0))
                    optimizer.step()
                    optimizer.zero_grad()
                    accum_counter = 0

                total_loss += loss.item()
                for k in component_sums:
                    component_sums[k] += metrics.get(k, 0.0)
                lr_val = (
                    optimizer.param_groups[0]["lr"]
                    if hasattr(optimizer, "param_groups") and len(optimizer.param_groups) > 0
                    else float("nan")
                )
                pbar.set_postfix(
                    {
                        "L_tot": f"{metrics['L_total']:.3f}",
                        "L_data": f"{metrics['L_data']:.3f}",
                        "L_mu": f"{metrics['L_mu']:.3f}",
                        "L_mom": f"{metrics['L_mom']:.3f}",
                        "L_cont": f"{metrics['L_cont']:.3f}",
                        "L_bc": f"{metrics['L_bc']:.3f}",
                        "L_io": f"{metrics['L_io']:.3f}",
                        "L_wss": f"{metrics['L_wss']:.3f}",
                        "L_pgrad": f"{metrics['L_pgrad']:.3f}",
                        "L_jac": f"{metrics['L_jac']:.3f}",
                        "|g|": f"{grad_norm:.2f}",
                        "LR": f"{lr_val:.2e}",
                    }
                )
            scheduler.step()
        else:
            print(f"⏳ L-BFGS Step (Ep {epoch:02d}) [S{stage}: n={current_n:.3f}]")
            # static_batches is frozen during LBFGS initialization and already on device.

            def closure():
                optimizer.zero_grad()
                accumulated_loss = torch.tensor(0.0, device=device)
                for c_data in static_batches:
                    loss, _ = compute_step_loss(
                        model,
                        c_data,
                        kernels,
                        loss_weighter,
                        "anderson",
                        device,
                        stage,
                        current_n,
                        current_mu_0,
                        weight_data,
                        weight_mu,
                        weight_wss,
                    )
                    loss.backward()
                    accumulated_loss += loss.detach() / len(static_batches)
                return accumulated_loss

            loss_tensor = optimizer.step(closure)
            total_loss = loss_tensor.item()

        # Simple save
        if epoch % 5 == 0 or epoch == epochs - 1:
            os.makedirs(stage_a_dir(), exist_ok=True)
            ckpt_path = stage_a_dir() / f"kinematics_ckpt_{epoch + 1}.pth"
            torch.save(model.state_dict(), ckpt_path)
            torch.save(model.state_dict(), stage_a_dir() / "kinematics_ckpt_latest.pth")
            state_path = stage_a_dir() / f"kinematics_state_{epoch + 1}.pth"
            state_payload = {
                "epoch": int(epoch),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": (
                    optimizer.state_dict()
                    if hasattr(optimizer, "state_dict")
                    else None
                ),
                "scheduler_state_dict": (
                    scheduler.state_dict()
                    if hasattr(scheduler, "state_dict")
                    else None
                ),
                "loss_weighter_state_dict": loss_weighter.state_dict(),
                "best_val_composite_loss": float(best_val_composite_loss),
                "optimizer_name": optimizer.__class__.__name__,
            }
            torch.save(state_payload, state_path)
            torch.save(state_payload, stage_a_dir() / "kinematics_state_latest.pth")

        if epoch % 2 == 0 and len(val_data) > 0:
            val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
            scores = quantify_performance(model, val_loader, kernels, device, phase="kinematics")
            rel_l2 = float(scores.get("rel_l2", float("nan")))
            continuity = float(scores.get("continuity", float("nan")))
            val_comp = rel_l2 + 100.0 * continuity
            print(
                f"📊 [Validation] Rel L2: {rel_l2:.4f} | "
                f"|∇·u| mean: {continuity:.3e} | composite: {val_comp:.4f}"
            )
            if stage == 3 and val_comp < best_val_composite_loss:
                best_val_composite_loss = val_comp
                torch.save(model.state_dict(), stage_a_dir() / "kinematics_best.pth")
                print("⭐ Saved New Best Kinematics Model")
            try:
                os.makedirs(stage_a_dir(), exist_ok=True)
                with open(stage_a_dir() / "kinematics_validation.jsonl", "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "epoch": int(epoch),
                                "stage": int(stage),
                                "rheology": str(target_rheology),
                                "lr": float(
                                    optimizer.param_groups[0]["lr"]
                                    if hasattr(optimizer, "param_groups")
                                    and len(optimizer.param_groups) > 0
                                    else float("nan")
                                ),
                                "rel_l2": rel_l2,
                                "continuity": continuity,
                                "composite": val_comp,
                                "best_so_far": float(best_val_composite_loss),
                            }
                        )
                        + "\n"
                    )
            except OSError:
                pass
            diary.log_validation(
                epoch,
                {
                    "rel_l2": rel_l2,
                    "continuity": continuity,
                    "composite": val_comp,
                },
                stage=int(stage),
                lr=float(
                    optimizer.param_groups[0]["lr"]
                    if hasattr(optimizer, "param_groups") and len(optimizer.param_groups) > 0
                    else float("nan")
                ),
                best_so_far=float(best_val_composite_loss),
            )

        num_steps = max(1, len(loader))
        avg_epoch_loss = total_loss / num_steps
        print(f"Epoch {epoch:03d} complete | stage={stage} | loss={avg_epoch_loss:.6f}")
        avg_components = {k: v / num_steps for k, v in component_sums.items()}
        component_total = sum(avg_components.values())
        if component_total > 0.0:
            print(
                "   ↳ Loss breakdown (avg/step): "
                f"PDE={avg_components['C_weighted_pde']:.3f} ({100.0 * avg_components['C_weighted_pde'] / component_total:5.1f}%), "
                f"data_u={avg_components['C_data_kine']:.3f} ({100.0 * avg_components['C_data_kine'] / component_total:5.1f}%), "
                f"data_mu={avg_components['C_data_mu']:.3f} ({100.0 * avg_components['C_data_mu'] / component_total:5.1f}%), "
                f"bc={avg_components['C_bc']:.3f} ({100.0 * avg_components['C_bc'] / component_total:5.1f}%), "
                f"io={avg_components['C_io']:.3f} ({100.0 * avg_components['C_io'] / component_total:5.1f}%), "
                f"pgrad={avg_components['C_pgrad']:.3f} ({100.0 * avg_components['C_pgrad'] / component_total:5.1f}%), "
                f"wss={avg_components['C_wss']:.3f} ({100.0 * avg_components['C_wss'] / component_total:5.1f}%), "
                f"jac={avg_components['C_jac']:.3f} ({100.0 * avg_components['C_jac'] / component_total:5.1f}%)"
            )
        else:
            print("   ↳ Loss breakdown skipped (non-positive total weighted contribution).")
        diary.log_epoch_end(
            epoch,
            stage=int(stage),
            train_loss=float(avg_epoch_loss),
            lr=float(
                optimizer.param_groups[0]["lr"]
                if hasattr(optimizer, "param_groups") and len(optimizer.param_groups) > 0
                else float("nan")
            ),
        )
    diary.log_run_end(best_val_composite_loss=float(best_val_composite_loss))


if __name__ == "__main__":
    _ = (time, Path, quantify_performance)  # kept for API parity/future hooks
    parser = argparse.ArgumentParser(description="Train kinematics predictor with optional resume UX.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--adam-epochs", type=int, default=85)
    parser.add_argument("--stage1-end-epoch", type=int, default=STAGE1_END_EPOCH)
    parser.add_argument("--stage2-end-epoch", type=int, default=STAGE2_END_EPOCH)
    parser.add_argument("--accum-steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--weight-data", type=float, default=500.0, help="Supervised data weight")
    parser.add_argument("--weight-mu", type=float, default=10.0, help="Viscosity supervision weight")
    parser.add_argument("--weight-wss", type=float, default=10.0, help="Wall shear stress weight")
    parser.add_argument(
        "--max-lbfgs-graphs",
        type=int,
        default=4,
        help="Number of cached graphs for L-BFGS closure.",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        help="Resume from checkpoint path or use 'latest' (default when flag provided without value).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start a fresh run and disable interactive resume prompt.",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Disable interactive prompt; starts fresh unless --resume is explicitly set.",
    )
    args = parser.parse_args()

    if args.fresh and args.resume is not None:
        raise ValueError("Cannot use --fresh together with --resume.")

    ckpt_dir = stage_a_dir()
    latest_state = ckpt_dir / "kinematics_state_latest.pth"
    latest_model = ckpt_dir / "kinematics_ckpt_latest.pth"

    resume_from = None
    if args.resume is not None:
        if args.resume == "latest":
            if latest_state.exists():
                resume_from = str(latest_state)
            elif latest_model.exists():
                resume_from = str(latest_model)
            else:
                print("ℹ️ No latest checkpoint found; starting fresh.")
        else:
            resume_path = Path(args.resume)
            if not resume_path.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
            resume_from = str(resume_path)
    elif not args.fresh and not args.no_prompt:
        latest = latest_state if latest_state.exists() else (latest_model if latest_model.exists() else None)
        if latest is not None:
            try:
                choice = input(f"Found checkpoint '{latest}'. Resume? [Y/n]: ").strip().lower()
            except EOFError:
                choice = "n"
            if choice in ("", "y", "yes"):
                resume_from = str(latest)

    train_kinematics(
        epochs=int(args.epochs),
        adam_epochs=int(args.adam_epochs),
        stage1_end_epoch=int(args.stage1_end_epoch),
        stage2_end_epoch=int(args.stage2_end_epoch),
        resume_from=resume_from,
        accum_steps=int(args.accum_steps),
        weight_data=float(args.weight_data),
        weight_mu=float(args.weight_mu),
        weight_wss=float(args.weight_wss),
        max_lbfgs_graphs=int(args.max_lbfgs_graphs),
    )
