import argparse
import atexit
import math
import os
import sys
if sys.platform != "win32":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import random
from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.config import VesselConfig, PhysicsConfig, CurriculumConfig
from src.architecture.ginodeq import GINO_DEQ
from src.core_physics.physics_kernels import PhysicsKernels
from src.utils.metrics import DynamicLossWeighter, quantify_performance, validate_and_plot
from src.utils.kinematics_physics_terms import compute_kinematics_physics_terms
from src.utils.samplers import StratifiedAnchorSampler
from src.utils.training_diary import TrainingDiary, env_snapshot
from src.utils.paths import get_project_root, stage_a_dir, resolve_checkpoint
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau

# Validation composite for ``tier2_best_physics.pth``: ``rel_l2 + continuity + rheology`` (all from
# ``quantify_performance``). Lower is better — not the same as training loss. Logged as
# ``best_val_composite_loss`` in checkpoints / run_end; per-validation rows use ``val_composite_loss``.


def _env_truthy(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def _load_tier1_bootstrap(model: GINO_DEQ, tier1_path: Path, device: str) -> bool:
    """Load Tier 1 ``GINO_DEQ`` weights into this Tier 2 model (encoder input-width surgery if needed)."""
    if not tier1_path.is_file():
        print(
            f"⚠️ Tier 1 weights not found at {tier1_path}. "
            "Train Tier 1 first or place tier1_best_physics.pth under outputs/stage_a/; Tier 2 will use random init."
        )
        return False
    state_dict = torch.load(tier1_path, map_location=device, weights_only=True)
    if "encoder.0.weight" in state_dict:
        w_t1 = state_dict["encoder.0.weight"]
        w_t2 = model.encoder[0].weight
        if w_t1.shape[1] != w_t2.shape[1]:
            print(f"🔧 Adapting Tier 1 encoder width ({w_t1.shape[1]} -> {w_t2.shape[1]}) for Tier 2 node features.")
            new_w = torch.zeros_like(w_t2)
            n = min(w_t1.shape[1], w_t2.shape[1])
            new_w[:, :n] = w_t1[:, :n]
            state_dict["encoder.0.weight"] = new_w
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(
            f"ℹ️ load_state_dict(strict=False): {len(missing)} missing keys, {len(unexpected)} unexpected keys."
        )
    print(f"✅ Loaded Tier 1 bootstrap weights from {tier1_path.name}")
    return True


def _assert_tier2_train_split(train_data: list, val_data: list) -> None:
    """Fail fast on splits that break the sampler or coupled phase."""
    if not train_data:
        raise ValueError("train_data is empty after split.")
    if not val_data:
        raise ValueError("val_data is empty after split; need at least one validation graph.")
    n_anchor_tr = sum(1 for d in train_data if d.is_anchor.any().item())
    n_phys_tr = len(train_data) - n_anchor_tr
    if n_anchor_tr == 0:
        raise ValueError(
            "Training split has no anchor (COMSOL-labeled) graphs; StratifiedAnchorSampler requires them."
        )
    if n_phys_tr == 0:
        raise ValueError(
            "Training split has no physics-only graphs. After distillation the sampler uses a "
            "50/50 anchor/physics mix — add unlabeled meshes or change the split."
        )

def load_dataset():
    cfg = VesselConfig(tier="tier2")
    data_dir = cfg.graph_output_dir
    if not data_dir.exists():
        print(f"Directory not found: {data_dir}. Please generate Tier 2 data first.")
        return []
    file_list = sorted(list(data_dir.glob("vessel_*.pt")))
    dataset = []
    print(f"📂 Loading {len(file_list)} Tier 2 graphs from {data_dir}...")
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


def _tier2_dynamic_loss_weighter(device: str, mom_precision_floor: float) -> DynamicLossWeighter:
    """Kendall weights for [momentum]; floor momentum precision (default >= 0.8)."""
    floor = max(float(mom_precision_floor), 1e-6)
    max_lv_mom = float(-math.log(floor))
    return DynamicLossWeighter(
        num_losses=1,
        max_log_var=[max_lv_mom],
    ).to(device)


def compute_step_loss(
    model,
    data,
    kernels,
    loss_weighter,
    current_solver,
    lambda_phys,
    device,
    is_distillation,
    carreau_n=None,
    *,
    tier2_kine_p_weight: float = 1.0,
    coupled_io_scale: float = 6.0,
):
    out = model(data, solver=current_solver, anderson_beta=1.0 if is_distillation else 0.8, anderson_warmup_iters=5)
    if isinstance(out, tuple):
        pred, jac_loss = out
    else:
        pred = out
        jac_loss = torch.tensor(0.0, device=device)

    terms = compute_kinematics_physics_terms(
        pred,
        data,
        kernels,
        tier="tier2",
        tier2_distillation=is_distillation,
        carreau_n=carreau_n,
        tier2_kine_p_weight=tier2_kine_p_weight,
    )
    l_wss = terms["l_wss"]
    l_data_kine = terms["l_data_kine"]
    l_data_mu = terms["l_data_mu"]
    l_mom = terms["l_mom"]
    l_cont = terms["l_cont"]
    l_bc = terms["l_bc"]
    l_io = terms["l_io"]
    l_rheo = terms["l_rheo"]

    # --- DISTILLATION ROUTING ---
    if is_distillation:
        loss = (
            (10.0 * l_rheo)
            + (5.0 * l_data_mu)
            + (5.0 * l_bc)
            + (5.0 * l_io)
            + (10.0 * l_wss)
            + (0.1 * jac_loss)
        )

        metrics = {"L_rh": l_rheo.item(), "L_jac": jac_loss.item(), "L_mom": 0.0, "L_cont": 0.0, "L_wss": l_wss.item()}
        return loss, metrics

    # --- PHASE 2/3: FULLY COUPLED ROUTING ---
    pde_losses = [l_mom]
    pde_scales = [lambda_phys]
    weighted_pdes = loss_weighter(pde_losses, scales=pde_scales)

    loss = (
        weighted_pdes
        + (1.0 * l_rheo)
        + (500.0 * l_data_kine)
        + (50.0 * l_data_mu)
        + (5.0 * l_bc)
        + (float(coupled_io_scale) * l_io)
        + (10.0 * l_wss)
        + (0.1 * jac_loss)
    )

    metrics = {
        "L_mom": l_mom.item(),
        "L_cont": l_cont.item(),
        "L_rh": l_rheo.item(),
        "L_jac": jac_loss.item(),
        "L_wss": l_wss.item()
    }
    return loss, metrics


def train_t2_predictor(epochs=80, distillation_epochs=12, adam_epochs=50, lr=1e-4):
    """Train Tier 2 (Carreau) GINO-DEQ.

    **Weights**
        Tier 1 ``tier1_best_physics.pth`` under ``outputs/stage_a/`` is always loaded when **not** restoring from
        ``tier2_latest_checkpoint.pth`` in the same stage directory (bootstrap from Newtonian pretraining).

    **Environment**
        ``TIER2_RESUME=1`` — restore optimizer/scheduler/epoch from ``tier2_latest_checkpoint.pth``
        only. Tier 1 bootstrap is skipped in that case (checkpoint already contains trained weights).
        Default ``TIER2_RESUME=0`` — new run: Tier 1 bootstrap + fresh optimizer (ignores stale
        checkpoint file on disk unless resume is enabled).
        ``TIER2_USE_LBFGS=1`` — after ``adam_epochs``, switch to L-BFGS (same pattern as Tier 1:
        ``epoch >= adam_epochs``, ``strong_wolfe`` line search, joint ``model`` + ``loss_weighter``
        parameters). Checkpoints saved in the L-BFGS phase do not restore the L-BFGS optimizer
        state on resume (fresh AdamW for the current phase, then L-BFGS activates again when due).
        **Early stopping (coupled phase only; validation every 2 epochs):**
        ``TIER2_EARLY_STOP_PATIENCE`` (default ``8``) and ``TIER2_EARLY_STOP_MIN_DELTA`` (default
        ``0.002``) — stop if anchor Rel L2 fails to beat the prior *material* best by more than
        ``min_delta`` for ``patience`` validation checks (mirrors Tier 1 logic).
        ``TIER2_EARLY_STOP_FLAT_EPS`` (default ``3e-4``) and ``TIER2_EARLY_STOP_FLAT_PATIENCE``
        (default ``6``) — additionally stop if Rel L2 changes by less than ``flat_eps`` versus the
        *previous* validation for ``flat_patience`` checks in a row (catches a fully flat tail).

        **Coupled-phase stability / pressure emphasis (defaults tuned for Carreau DEQ):**
        ``TIER2_MOM_PRECISION_FLOOR`` (default ``0.8``) — Kendall momentum precision cannot drop
        below this (caps ``log_var`` for the momentum task).
        ``TIER2_KINE_P_WEIGHT`` (default ``1.35``) — extra weight on the pressure channel in anchor
        kinematic MSE vs ``u,v``.
        ``TIER2_GRAD_CLIP_COUPLED`` (default ``0.5``) — global grad norm clip in the coupled phase
        (distillation still uses ``1.0``).
        ``TIER2_COSINE_ETA_MIN`` / ``TIER2_PLATEAU_MIN_LR`` (default ``5e-6``) — floor for
        ``CosineAnnealingWarmRestarts`` and ``ReduceLROnPlateau`` in the coupled phase.
        ``TIER2_COUPLED_IO_SCALE`` (default ``6.0``) — multiplier on ``l_io`` (inlet velocity + outlet
        ``p=0`` soft penalty) after distillation.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device being used:", device)

    phys_cfg = PhysicsConfig(tier="tier2")
    kernels = PhysicsKernels(phys_cfg=phys_cfg)
    model = GINO_DEQ(
        in_channels=15,
        out_channels=5,
        latent_dim=64,
        max_iters=15,
        phys_cfg=phys_cfg,
    ).to(device)

    root = get_project_root()
    model_dir = stage_a_dir()
    tier1_path = resolve_checkpoint("a", "tier1_best_physics.pth")
    latest_ckpt_path = resolve_checkpoint("a", "tier2_latest_checkpoint.pth")
    latest_ckpt_save = model_dir / "tier2_latest_checkpoint.pth"
    resume_training = _env_truthy("TIER2_RESUME")

    mom_precision_floor = float(os.environ.get("TIER2_MOM_PRECISION_FLOOR", "0.8"))
    loss_weighter = _tier2_dynamic_loss_weighter(device, mom_precision_floor)

    dataset = load_dataset()
    if not dataset:
        return

    anchors = [d for d in dataset if d.is_anchor.any().item()]
    physics = [d for d in dataset if not d.is_anchor.any().item()]

    random.seed(42)
    random.shuffle(anchors)
    random.shuffle(physics)

    split_idx_a = int(0.9 * len(anchors))
    split_idx_p = int(0.9 * len(physics))

    train_data = anchors[:split_idx_a] + physics[:split_idx_p]
    val_data = anchors[split_idx_a:] + physics[split_idx_p:]

    if distillation_epochs < 1:
        raise ValueError("distillation_epochs must be >= 1.")
    if adam_epochs > epochs:
        raise ValueError(f"adam_epochs ({adam_epochs}) must be <= epochs ({epochs}).")
    if adam_epochs < distillation_epochs:
        print(
            f"⚠️ adam_epochs ({adam_epochs}) < distillation_epochs ({distillation_epochs}): "
            "L-BFGS activates when epoch >= adam_epochs and may overlap distillation. "
            "Prefer adam_epochs >= distillation_epochs."
        )
    _assert_tier2_train_split(train_data, val_data)

    micro_batch_size = 2
    accumulation_steps = 4

    sampler = StratifiedAnchorSampler(train_data, batch_size=micro_batch_size)
    loader = DataLoader(train_data, batch_size=micro_batch_size, sampler=sampler)
    val_loader = DataLoader(val_data, batch_size=micro_batch_size, shuffle=False)

    best_val_composite_loss = float("inf")
    best_loss = float("inf")
    optimizer = None
    scheduler = None
    lbfgs_initialized = False
    use_lbfgs = _env_truthy("TIER2_USE_LBFGS")
    stop_after_adam = _env_truthy("TIER2_STOP_AFTER_ADAM")
    start_epoch = 0
    ckpt_every = max(1, int(os.environ.get("TIER2_CKPT_EVERY", "1")))

    curriculum = CurriculumConfig()
    target_n = phys_cfg.n
    start_n = curriculum.tier2_carreau_n_distill_start

    tier2_kine_p_weight = float(os.environ.get("TIER2_KINE_P_WEIGHT", "1.35"))
    tier2_cosine_eta_min = float(os.environ.get("TIER2_COSINE_ETA_MIN", "5e-6"))
    tier2_plateau_min_lr = float(os.environ.get("TIER2_PLATEAU_MIN_LR", "5e-6"))
    tier2_grad_clip_coupled = float(os.environ.get("TIER2_GRAD_CLIP_COUPLED", "0.5"))
    tier2_coupled_io_scale = float(os.environ.get("TIER2_COUPLED_IO_SCALE", "6.0"))

    resumed_full_checkpoint = False
    tier1_bootstrap_loaded = False
    plateau_scheduler = None

    if resume_training and latest_ckpt_path.is_file():
        print(f"🔄 TIER2_RESUME: restoring full training state from {latest_ckpt_path.name}")
        ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        try:
            loss_weighter.load_state_dict(ckpt["loss_weighter_state_dict"])
        except RuntimeError:
            # Backward compatibility: old checkpoints may store two PDE log_vars (cont, mom).
            print("ℹ️ Reinitializing Tier 2 PDE loss weighter for momentum-only setup.")
        best_val_composite_loss = float(ckpt.get("best_val_composite_loss", best_val_composite_loss))
        best_loss = float(ckpt.get("best_loss", best_loss))
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        ckpt_optimizer_type = ckpt.get("optimizer_type", "AdamW")

        if start_epoch >= distillation_epochs:
            optimizer = setup_coupled_phase(model, loss_weighter, base_lr=lr)
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=tier2_cosine_eta_min
            )
            plateau_scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=3, threshold=5e-4, min_lr=tier2_plateau_min_lr
            )
            sampler.set_warmup_mode(False)
        else:
            optimizer = setup_distillation_phase(model)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-5)
            plateau_scheduler = None
            sampler.set_warmup_mode(True)

        if ckpt_optimizer_type == "LBFGS":
            print("ℹ️ Ignoring stored LBFGS optimizer state in checkpoint; continuing with AdamW for this phase.")
            lbfgs_initialized = False
        elif ckpt_optimizer_type == "AdamW" and start_epoch < adam_epochs:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if ckpt.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        resumed_full_checkpoint = True
        print(f"✅ Tier 2 training resume complete at epoch {start_epoch} (Tier 1 bootstrap skipped).")
    else:
        if resume_training:
            print(
                f"ℹ️ TIER2_RESUME is set but no checkpoint file at {latest_ckpt_path}. "
                "Starting a new run with Tier 1 bootstrap."
            )
        tier1_bootstrap_loaded = _load_tier1_bootstrap(model, tier1_path, device)

    cfg_t2 = VesselConfig(tier="tier2")
    n_anchor_train_t2 = len([d for d in train_data if d.is_anchor.any().item()])
    n_phys_train_t2 = max(0, len(train_data) - n_anchor_train_t2)
    plateau_patience = max(1, int(os.environ.get("TIER2_EARLY_STOP_PATIENCE", "8")))
    plateau_min_delta = float(os.environ.get("TIER2_EARLY_STOP_MIN_DELTA", "0.002"))
    flat_eps = float(os.environ.get("TIER2_EARLY_STOP_FLAT_EPS", "3e-4"))
    flat_patience = max(1, int(os.environ.get("TIER2_EARLY_STOP_FLAT_PATIENCE", "6")))
    best_rel_l2 = float("inf")
    val_no_improve = 0
    prev_val_rel_l2: Optional[float] = None
    flat_streak = 0
    early_stopped = False

    diary = TrainingDiary("tier2")
    diary.log_run_start(
        device=device,
        re_target=float(phys_cfg.re_target),
        graph_dir=str(cfg_t2.graph_output_dir),
        n_graphs_total=len(dataset),
        n_train=len(train_data),
        n_val=len(val_data),
        n_anchor_train=n_anchor_train_t2,
        n_phys_train=n_phys_train_t2,
        n_val_anchors=sum(1 for d in val_data if d.is_anchor.any().item()),
        micro_batch_size=micro_batch_size,
        accumulation_steps=accumulation_steps,
        distillation_epochs=distillation_epochs,
        adam_epochs=adam_epochs,
        epochs=epochs,
        lr=lr,
        carreau_n_curriculum_start=float(start_n),
        carreau_n_target=float(target_n),
        start_epoch=start_epoch,
        resume_training=bool(resume_training),
        resumed_full_checkpoint=bool(resumed_full_checkpoint),
        tier1_bootstrap_loaded=bool(tier1_bootstrap_loaded),
        best_val_composite_loss_checkpoint=float(best_val_composite_loss),
        best_loss_checkpoint=float(best_loss),
        use_lbfgs=use_lbfgs,
        ckpt_every=ckpt_every,
        plateau_patience=plateau_patience,
        plateau_min_delta=plateau_min_delta,
        early_stop_flat_eps=flat_eps,
        early_stop_flat_patience=flat_patience,
        env_tier2_phase1=env_snapshot("TIER2_", "PHASE1_"),
        mom_precision_floor=float(mom_precision_floor),
        tier2_kine_p_weight=float(tier2_kine_p_weight),
        tier2_cosine_eta_min=float(tier2_cosine_eta_min),
        tier2_plateau_min_lr=float(tier2_plateau_min_lr),
        tier2_grad_clip_coupled=float(tier2_grad_clip_coupled),
        tier2_coupled_io_scale=float(tier2_coupled_io_scale),
    )

    tier2_final_path = model_dir / "tier2_final.pth"
    run_end_emitted = False
    saved_final_weights = False
    last_epoch_completed: Optional[int] = None

    def _emit_tier2_run_end(interrupted: bool = False) -> None:
        nonlocal run_end_emitted
        if run_end_emitted or not diary.enabled:
            return
        run_end_emitted = True
        if interrupted:
            print("\n⚠️ Training interrupted; appending training diary run_end (JSONL report).")
        diary.log_run_end(
            best_val_composite_loss=float(best_val_composite_loss),
            best_loss=float(best_loss),
            best_rel_l2=float(best_rel_l2),
            early_stopped=bool(early_stopped),
            interrupted=bool(interrupted),
            last_epoch_completed=last_epoch_completed,
            diary_path=str(diary.path) if diary.path else None,
            final_weights_path=str(tier2_final_path) if saved_final_weights else None,
        )

    atexit.register(lambda: _emit_tier2_run_end(True))

    for epoch in range(start_epoch, epochs):
        last_epoch_completed = epoch
        if use_lbfgs and stop_after_adam and epoch >= adam_epochs:
            checkpoint = {
                "epoch": epoch - 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": (scheduler.state_dict() if scheduler is not None else None),
                "loss_weighter_state_dict": loss_weighter.state_dict(),
                "best_val_composite_loss": best_val_composite_loss,
                "best_loss": best_loss,
                "optimizer_type": "AdamW",
                "handoff_for_lbfgs": True,
            }
            torch.save(checkpoint, latest_ckpt_save)
            print(
                f"🧳 Saved Tier 2 Adam handoff checkpoint at epoch {epoch - 1} -> {latest_ckpt_save.name}. "
                "Resume with TIER2_STOP_AFTER_ADAM=0 (or unset) to continue L-BFGS."
            )
            break
        is_distillation = epoch < distillation_epochs
        physics_active = not is_distillation
        lambda_phys = min(1.0, max(0.0, (epoch - distillation_epochs) / 20.0))

        if is_distillation:
            progress = epoch / max(1, (distillation_epochs - 1))
            carreau_n = start_n - progress * (start_n - target_n)
            print(f"🔄 Curriculum: Annealed Carreau index 'n' to {carreau_n:.4f}")
        else:
            carreau_n = target_n

        # Ensure Anderson remains active during L-BFGS to maintain gradient accuracy
        if is_distillation:
            current_solver = "picard"
        else:
            current_solver = "anderson"

        if epoch == 0 and not resumed_full_checkpoint:
            print(f"\n🚀 --- Starting Phase 1: Viscosity Distillation (Epochs 0-{distillation_epochs - 1}) ---")
            optimizer = setup_distillation_phase(model)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-5)
            plateau_scheduler = None
            sampler.set_warmup_mode(True)
        elif epoch == distillation_epochs:
            print(
                f"\n🚀 --- Starting Phase 2: Fully Coupled DEQ via AdamW (Epochs {distillation_epochs}-{adam_epochs - 1}) ---")
            optimizer = setup_coupled_phase(model, loss_weighter, base_lr=lr)
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=tier2_cosine_eta_min
            )
            plateau_scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=3, threshold=5e-4, min_lr=tier2_plateau_min_lr
            )
            sampler.set_warmup_mode(False)
        elif use_lbfgs and epoch >= adam_epochs and not lbfgs_initialized:
            print(f"\n⚡ Switching to L-BFGS Optimizer for the final {max(0, epochs - adam_epochs)} epoch(s)...")
            torch.cuda.empty_cache()

            optimizer = optim.LBFGS(
                list(model.parameters()) + list(loss_weighter.parameters()),
                lr=0.01,
                max_iter=20,
                history_size=30,
                line_search_fn="strong_wolfe",
                tolerance_grad=1e-6,
                tolerance_change=1e-8,
            )
            lbfgs_initialized = True

        model.train()

        total_loss_epoch = 0.0
        grad_norm = 0.0

        if not lbfgs_initialized:
            phase_tag = "Distill" if is_distillation else "AdamW"
            pbar = tqdm(loader, desc=f"Tier 2 Epoch {epoch:02d} [Re={phys_cfg.re_target}] ({phase_tag})")

            # Zero gradients AT THE START of the epoch
            optimizer.zero_grad()

            for batch_idx, data in enumerate(pbar):
                data = data.to(device)

                # Compute loss (scaled by accumulation steps)
                loss, metrics = compute_step_loss(
                    model,
                    data,
                    kernels,
                    loss_weighter,
                    current_solver,
                    lambda_phys,
                    device,
                    is_distillation,
                    carreau_n=carreau_n,
                    tier2_kine_p_weight=tier2_kine_p_weight,
                    coupled_io_scale=(5.0 if is_distillation else tier2_coupled_io_scale),
                )
                loss = loss / accumulation_steps

                if torch.isnan(loss):
                    print(f"\n⚠️ NaN detected in loss at epoch {epoch}! Skipping micro-batch.")
                    continue

                loss.backward()

                # Step optimizer ONLY when we've accumulated enough micro-batches
                if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(loader)):
                    clip_params = [p for g in optimizer.param_groups for p in g["params"]]
                    clip_max = 1.0 if is_distillation else tier2_grad_clip_coupled
                    grad_norm = torch.nn.utils.clip_grad_norm_(clip_params, max_norm=clip_max)
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
            if scheduler is not None:
                scheduler.step()

        else:
            print(f"⏳ Tier 2 Epoch {epoch:02d} [Re={phys_cfg.re_target}] (L-BFGS Line Search...)")
            # L-BFGS may reevaluate closure many times; keep batch set fixed per epoch.
            # Snapshot batches on CPU and transfer one-by-one in closure to avoid VRAM spikes.
            epoch_batches_cpu = list(loader)
            epoch_batches_device = [b.to(device) for b in epoch_batches_cpu]
            n_batches = max(len(epoch_batches_device), 1)

            def closure():
                optimizer.zero_grad()
                accumulated_loss = torch.tensor(0.0, device=device)

                for closure_data in epoch_batches_device:
                    loss, _ = compute_step_loss(
                        model,
                        closure_data,
                        kernels,
                        loss_weighter,
                        current_solver,
                        lambda_phys,
                        device,
                        is_distillation,
                        carreau_n=carreau_n,
                        tier2_kine_p_weight=tier2_kine_p_weight,
                        coupled_io_scale=(5.0 if is_distillation else tier2_coupled_io_scale),
                    )
                    loss = loss / n_batches
                    loss.backward()
                    accumulated_loss += loss.detach()
                return accumulated_loss

            loss_tensor = optimizer.step(closure)
            total_loss_epoch = loss_tensor.item() * n_batches
            print(f"✅ L-BFGS Step Complete. Accumulated Full-Batch Loss: {loss_tensor.item():.4f}")

        avg_loss = total_loss_epoch / max(1, len(loader))
        if avg_loss < best_loss and physics_active:
            best_loss = avg_loss
            save_bl = model_dir / "tier2_best_loss.pth"
            torch.save(model.state_dict(), save_bl)
            print(f"⭐ Saved Best Loss Model to {save_bl}")

        if epoch % 5 == 0 and val_data:
            validate_and_plot(model, val_data[0], epoch, device, tier="tier2")

        should_save_ckpt = ((epoch + 1) % ckpt_every == 0) or (epoch == epochs - 1)
        if should_save_ckpt:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": (scheduler.state_dict() if not lbfgs_initialized and scheduler is not None else None),
                "loss_weighter_state_dict": loss_weighter.state_dict(),
                "best_val_composite_loss": best_val_composite_loss,
                "best_loss": best_loss,
                "optimizer_type": ("LBFGS" if lbfgs_initialized else "AdamW"),
            }
            torch.save(checkpoint, latest_ckpt_save)
            print(f"💾 Saved Tier 2 checkpoint -> {latest_ckpt_save.name} (every {ckpt_every} epoch(s))")

        diary.log_epoch_end(
            epoch,
            avg_epoch_loss=float(avg_loss),
            lr=float(optimizer.param_groups[0]["lr"]),
            lambda_phys=float(lambda_phys),
            carreau_n=float(carreau_n),
            phase=("distillation" if is_distillation else ("lbfgs" if lbfgs_initialized else "coupled")),
            solver=current_solver,
            physics_active=bool(physics_active),
            lbfgs=bool(lbfgs_initialized),
            best_loss_so_far=float(best_loss),
            best_val_composite_loss_so_far=float(best_val_composite_loss),
            best_rel_l2_so_far=float(best_rel_l2),
            val_no_improve=int(val_no_improve),
        )

        if epoch % 2 == 0:
            scores = quantify_performance(model, val_loader, kernels, device, tier="tier2")
            print(
                f"\n📊 [Validation] Rel L2 (anchor): {scores.get('rel_l2', float('nan')):.4f} "
                f"(σ {scores.get('rel_l2_std', float('nan')):.4f}, p90 {scores.get('rel_l2_p90', float('nan')):.4f})"
            )
            print(
                f"   Components u / v / p: {scores.get('rel_l2_u', float('nan')):.4f} / "
                f"{scores.get('rel_l2_v', float('nan')):.4f} / {scores.get('rel_l2_p', float('nan')):.4f}"
            )
            print(
                f"   |∇·u| mean (fluid interior): {scores.get('continuity', float('nan')):.3e} "
                f"(p90 {scores.get('continuity_p90', float('nan')):.3e}) | "
                f"Rheo residual mean: {scores.get('rheology', float('nan')):.3e} "
                f"(p90 {scores.get('rheology_p90', float('nan')):.3e})"
            )
            print(
                f"   μ MAE: {scores.get('mu_mae', float('nan')):.4e} | "
                f"log-μ MSE: {scores.get('mu_log_mse', float('nan')):.4e} | "
                f"Wall |u| mean: {scores.get('wall_slip', float('nan')):.4f} "
                f"(p90 {scores.get('wall_slip_p90', float('nan')):.4f})"
            )
            print(
                f"   γ̇ MSE (anchor): {scores.get('shear_mse', float('nan')):.4e} "
                f"(p90 {scores.get('shear_mse_p90', float('nan')):.4e}) | "
                f"Batches w/ anchors: {int(scores.get('val_anchor_batches', 0))}/"
                f"{int(scores.get('val_total_batches', 0))}"
            )

            w_cont = None
            w_mom = None
            n_val_anchor = sum(1 for d in val_data if d.is_anchor.any().item())
            n_val_physics = max(0, len(val_data) - n_val_anchor)

            with torch.no_grad():
                safe_vars = loss_weighter.clamped_log_vars()
                weights = torch.exp(-safe_vars)
                w_cont = 0.0
                w_mom = float(weights[0].item())
                print(f"⚖️ Learned PDE Weights -> Cont: {w_cont:.2f} | Mom: {w_mom:.2f}")
            print(f"📌 Val split: anchors={n_val_anchor} | physics={n_val_physics}")

            val_composite_loss = float(
                scores.get("rel_l2", 0)
                + scores.get("continuity", 0)
                + scores.get("rheology", 0)
            )

            diary.log_validation(
                epoch,
                scores,
                weight_cont=w_cont,
                weight_mom=w_mom,
                val_composite_loss=val_composite_loss,
                carreau_n=float(carreau_n),
                phase_distillation=bool(is_distillation),
                best_rel_l2=float(best_rel_l2),
                val_no_improve=int(val_no_improve),
                n_val_anchors=int(n_val_anchor),
                n_val_physics=int(n_val_physics),
            )

            if plateau_scheduler is not None and not lbfgs_initialized:
                plateau_scheduler.step(float(scores.get("rel_l2", 0.0)))

            rel_l2 = float(scores.get("rel_l2", float("inf")))
            if not physics_active:
                if rel_l2 < best_rel_l2:
                    best_rel_l2 = rel_l2
            else:
                if (best_rel_l2 - rel_l2) > plateau_min_delta:
                    best_rel_l2 = rel_l2
                    val_no_improve = 0
                    flat_streak = 0
                else:
                    val_no_improve += 1

                if prev_val_rel_l2 is not None and abs(rel_l2 - prev_val_rel_l2) < flat_eps:
                    flat_streak += 1
                else:
                    flat_streak = 0
                prev_val_rel_l2 = rel_l2

                stop_material = val_no_improve >= plateau_patience
                stop_flat = flat_streak >= flat_patience
                if stop_material or stop_flat:
                    if stop_material and stop_flat:
                        reason = "val_rel_l2_plateau_and_flat"
                    elif stop_flat:
                        reason = "val_rel_l2_flatline"
                    else:
                        reason = "val_rel_l2_plateau"
                    print(
                        f"🛑 Early stopping ({reason}): material plateau min_delta={plateau_min_delta:.4f} "
                        f"patience={plateau_patience} | flat_eps={flat_eps:.2e} flat_patience={flat_patience} "
                        f"(last Rel L2={rel_l2:.4f})."
                    )
                    early_stopped = True
                    diary.log_event(
                        "early_stop",
                        epoch=epoch,
                        reason=reason,
                        plateau_min_delta=plateau_min_delta,
                        plateau_patience=plateau_patience,
                        flat_eps=flat_eps,
                        flat_patience=flat_patience,
                        last_rel_l2=rel_l2,
                    )
                    break

            if physics_active and val_composite_loss < best_val_composite_loss:
                best_val_composite_loss = val_composite_loss
                save_bp = model_dir / "tier2_best_physics.pth"
                torch.save(model.state_dict(), save_bp)
                print(f"⭐ Saved Best Physics Model to {save_bp}")

    torch.save(model.state_dict(), tier2_final_path)
    saved_final_weights = True
    _emit_tier2_run_end(interrupted=False)
    print(
        f"Tier 2 Training Complete. Best val composite loss: {best_val_composite_loss:.4f} "
        f"(rel_l2 + continuity + rheology; lower is better) | Best Loss: {best_loss:.4f}"
    )


def _parse_args():
    p = argparse.ArgumentParser(description="Tier 2 GINO-DEQ predictor training.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Resume from tier2_latest_checkpoint.pth (sets TIER2_RESUME=1).",
    )
    mode.add_argument(
        "--new",
        action="store_true",
        help="Start a new run (sets TIER2_RESUME=0).",
    )
    return p.parse_args()


def _prompt_resume_or_new_t2() -> bool:
    """Ask user whether to resume checkpointed training."""
    while True:
        raw = input("Training mode [1=resume / 2=start new] [1]: ").strip()
        if raw in ("", "1"):
            return True
        if raw == "2":
            return False
        print("  Enter 1 or 2.")


if __name__ == "__main__":
    args = _parse_args()
    if args.resume:
        resume_enabled = True
    elif args.new:
        resume_enabled = False
    else:
        resume_enabled = _prompt_resume_or_new_t2()
    os.environ["TIER2_RESUME"] = "1" if resume_enabled else "0"
    print(
        "🔄 Resuming Tier 2 from latest checkpoint."
        if resume_enabled
        else "🆕 Starting a new Tier 2 run."
    )
    train_t2_predictor()