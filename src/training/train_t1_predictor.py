import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import atexit
import csv
from typing import Optional

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from src.utils.paths import get_project_root, reports_dir, stage_a_dir, resolve_checkpoint
from src.architecture.ginodeq import GINO_DEQ
from src.core_physics.physics_kernels import PhysicsKernels
from src.config import VesselConfig, PhysicsConfig
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, ReduceLROnPlateau
from src.utils.metrics import quantify_performance, validate_and_plot, DynamicLossWeighter
from src.utils.anchor_mask import graph_has_anchor, anchor_node_mask
from src.utils.kinematics_physics_terms import compute_kinematics_physics_terms
from src.utils.training_diary import TrainingDiary, env_snapshot
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


def compute_step_loss(
    model,
    data,
    kernels,
    loss_weighter,
    current_solver,
    lambda_phys,
    device,
    data_scale=500.0,
    bc_scale=10.0,
    io_scale=5.0,
    wss_scale=10.0,
    boundary_data_weight=2.0,
    is_distillation=False,
):
    out = model(data, solver=current_solver, anderson_beta=0.8, anderson_warmup_iters=5)
    if isinstance(out, tuple):
        pred, jac_loss = out
    else:
        pred = out
        jac_loss = torch.tensor(0.0, device=device)

    terms = compute_kinematics_physics_terms(
        pred,
        data,
        kernels,
        tier="tier1",
        boundary_data_weight=boundary_data_weight,
    )
    l_wss = terms["l_wss"]
    l_data_kine = terms["l_data_kine"]
    l_mom = terms["l_mom"]
    l_cont = terms["l_cont"]
    l_bc = terms["l_bc"]
    l_io = terms["l_io"]

    node_is_anchor = anchor_node_mask(data)
    pde_losses = [l_cont, l_mom]
    pde_scales = [lambda_phys, lambda_phys]
    weighted_pdes = loss_weighter(pde_losses, scales=pde_scales)

    # Combine losses cleanly
    loss = (
        weighted_pdes
        + (float(data_scale) * l_data_kine)
        + (float(bc_scale) * l_bc)
        + (float(io_scale) * l_io)
        + (float(wss_scale) * l_wss)
        + (0.1 * jac_loss)
    )

    metrics = {
        "L_data": l_data_kine.item(),
        "L_mom": l_mom.item(),
        "L_cont": l_cont.item(),
        "L_jac": jac_loss.item(),
        "L_wss": l_wss.item(),
        "A_nodes": int(node_is_anchor.sum().item()) if node_is_anchor is not None else 0,
    }
    return loss, metrics


def train_t1_predictor(epochs=60, lr=1e-4, warm_up_epochs=10, adam_epochs=60):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device being used:", device)
    model = GINO_DEQ(in_channels=15, out_channels=5, latent_dim=64, max_iters=15).to(device)

    phys_cfg = PhysicsConfig(tier="tier1")
    kernels = PhysicsKernels(phys_cfg=phys_cfg)

    loss_weighter = DynamicLossWeighter(num_losses=2).to(device)

    fig_dir = reports_dir() / "figures" / "tier1"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Phase 1 Optimizer (AdamW)
    optimizer = optim.AdamW(list(model.parameters()) + list(loss_weighter.parameters()),
                            lr=lr, weight_decay=1e-5)

    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warm_up_epochs)
    decay_epochs = adam_epochs - warm_up_epochs
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=decay_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warm_up_epochs])
    plateau_scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, threshold=5e-4, min_lr=1e-6
    )

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

    target_anchor_fraction = float(os.environ.get("TIER1_TARGET_ANCHOR_FRACTION", "0.5"))
    target_anchor_fraction = min(max(target_anchor_fraction, 0.0), 1.0)
    hard_alpha = max(float(os.environ.get("TIER1_HARD_MINING_ALPHA", "0.8")), 0.0)
    hard_refresh = max(int(os.environ.get("TIER1_HARD_MINING_REFRESH_EPOCHS", "4")), 1)
    boundary_data_weight = max(float(os.environ.get("TIER1_BOUNDARY_DATA_WEIGHT", "2.0")), 1.0)
    use_lbfgs = (os.environ.get("TIER1_USE_LBFGS", "0").strip().lower() in ("1", "true", "yes", "on"))
    n_anchor_train = len([d for d in train_data if d.is_anchor.any().item()])
    n_phys_train = max(0, len(train_data) - n_anchor_train)
    hard_anchor_multiplier = {}

    def _graph_sampling_key(data, list_idx: int) -> int:
        cid = getattr(data, "config_id", None)
        if cid is None:
            return list_idx
        if torch.is_tensor(cid):
            return int(cid.view(-1)[0].item()) if cid.numel() else list_idx
        return int(cid)

    def _make_train_loader():
        if n_anchor_train > 0 and n_phys_train > 0:
            w_anchor = target_anchor_fraction / max(n_anchor_train, 1)
            w_phys = (1.0 - target_anchor_fraction) / max(n_phys_train, 1)
            sample_weights = []
            for gi, d in enumerate(train_data):
                if graph_has_anchor(d):
                    gkey = _graph_sampling_key(d, gi)
                    hard_mult = float(hard_anchor_multiplier.get(gkey, 1.0))
                    sample_weights.append(w_anchor * hard_mult)
                else:
                    sample_weights.append(w_phys)
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=max(len(train_data), accumulation_steps * 4),
                replacement=True,
            )
            return DataLoader(train_data, batch_size=micro_batch_size, sampler=sampler)
        return DataLoader(train_data, batch_size=micro_batch_size, shuffle=True)

    loader = _make_train_loader()
    print(
        f"🎯 Weighted sampling enabled (target anchor fraction ~{target_anchor_fraction:.2f}; "
        f"anchors={n_anchor_train}, physics={n_phys_train})."
    )
    val_loader = DataLoader(val_data, batch_size=micro_batch_size, shuffle=False)

    best_phys_score = float('inf')
    best_loss = float('inf')
    root = get_project_root()
    model_dir = stage_a_dir()
    lbfgs_initialized = False
    start_epoch = 0
    latest_ckpt_save = model_dir / "tier1_latest_checkpoint.pth"
    best_physics_save = model_dir / "tier1_best_physics.pth"
    latest_ckpt_path = resolve_checkpoint("a", "tier1_latest_checkpoint.pth")
    best_physics_path = resolve_checkpoint("a", "tier1_best_physics.pth")
    ckpt_every = max(1, int(os.environ.get("TIER1_CKPT_EVERY", "1")))
    resume_enabled = (os.environ.get("TIER1_RESUME", "0").strip().lower() in ("1", "true", "yes", "on"))
    init_from_best_enabled = (os.environ.get("TIER1_INIT_FROM_BEST", "0").strip().lower() in ("1", "true", "yes", "on"))
    init_done = False

    if init_from_best_enabled and best_physics_path.exists():
        print(f"🧷 Initializing Tier 1 weights from best physics checkpoint: {best_physics_path}")
        best_state = torch.load(best_physics_path, map_location=device, weights_only=True)
        model.load_state_dict(best_state, strict=False)
        init_done = True

    if resume_enabled and latest_ckpt_path.exists():
        print(f"🔄 Resuming Tier 1 from checkpoint: {latest_ckpt_path}")
        ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        loss_weighter.load_state_dict(ckpt["loss_weighter_state_dict"])

        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_phys_score = float(ckpt.get("best_phys_score", best_phys_score))
        best_loss = float(ckpt.get("best_loss", best_loss))

        ckpt_optimizer_type = ckpt.get("optimizer_type", "AdamW")
        if start_epoch < adam_epochs and ckpt_optimizer_type == "AdamW":
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if ckpt.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        elif ckpt_optimizer_type == "LBFGS":
            print("ℹ️ Ignoring stored LBFGS optimizer state in checkpoint; continuing with AdamW.")

        print(f"✅ Tier 1 resume complete at epoch {start_epoch}.")
    elif resume_enabled:
        print(f"ℹ️ TIER1_RESUME is enabled but no checkpoint found at {latest_ckpt_path}. Starting fresh.")
    elif init_done:
        print("✅ Started from tier1_best_physics.pth (fresh optimizer/schedulers).")

    best_rel_l2 = float("inf")
    plateau_patience = int(os.environ.get("TIER1_EARLY_STOP_PATIENCE", "10"))
    plateau_min_delta = float(os.environ.get("TIER1_EARLY_STOP_MIN_DELTA", "0.005"))
    val_no_improve = 0
    early_stopped = False
    cfg_paths = VesselConfig(tier="tier1")
    diary = TrainingDiary("tier1")
    diary.log_run_start(
        device=device,
        re_target=float(phys_cfg.re_target),
        graph_dir=str(cfg_paths.graph_output_dir),
        n_graphs_total=len(dataset),
        n_train=len(train_data),
        n_val=len(val_data),
        n_anchor_train=n_anchor_train,
        n_phys_train=n_phys_train,
        n_val_anchors=sum(1 for d in val_data if graph_has_anchor(d)),
        micro_batch_size=micro_batch_size,
        accumulation_steps=accumulation_steps,
        target_anchor_fraction=target_anchor_fraction,
        hard_alpha=hard_alpha,
        hard_refresh_epochs=hard_refresh,
        boundary_data_weight=boundary_data_weight,
        epochs=epochs,
        warm_up_epochs=warm_up_epochs,
        adam_epochs=adam_epochs,
        lr=lr,
        stage_split_epochs=int(os.environ.get("TIER1_DATA_STAGE_EPOCHS", "12")),
        plateau_patience=plateau_patience,
        plateau_min_delta=plateau_min_delta,
        ckpt_every=ckpt_every,
        resume_enabled=resume_enabled,
        init_from_best_enabled=init_from_best_enabled,
        init_from_best_applied=init_done,
        use_lbfgs=use_lbfgs,
        start_epoch=start_epoch,
        resumed_checkpoint=bool(resume_enabled and latest_ckpt_path.exists()),
        best_phys_score_checkpoint=float(best_phys_score),
        best_loss_checkpoint=float(best_loss),
        env_tier1_phase1=env_snapshot("TIER1_", "PHASE1_"),
    )

    run_end_emitted = False
    last_epoch_completed: Optional[int] = None

    def _emit_tier1_run_end(interrupted: bool = False) -> None:
        nonlocal run_end_emitted
        if run_end_emitted or not diary.enabled:
            return
        run_end_emitted = True
        if interrupted:
            print("\n⚠️ Training interrupted; appending training diary run_end (JSONL report).")
        diary.log_run_end(
            best_phys_score=float(best_phys_score),
            best_loss=float(best_loss),
            best_rel_l2=float(best_rel_l2),
            early_stopped=bool(early_stopped),
            interrupted=bool(interrupted),
            last_epoch_completed=last_epoch_completed,
            diary_path=str(diary.path) if diary.path else None,
        )

    atexit.register(lambda: _emit_tier1_run_end(True))

    hard_csv_path = reports_dir() / "tier1_anchor_hardness.csv"
    hard_csv_path.parent.mkdir(parents=True, exist_ok=True)

    def _refresh_hard_mining(epoch_idx: int):
        if n_anchor_train == 0:
            return
        model.eval()
        rows = []
        with torch.no_grad():
            for gi, d in enumerate(train_data):
                if not graph_has_anchor(d):
                    continue
                # IMPORTANT: never move cached dataset items in-place to GPU.
                # DataLoader expects CPU tensors and will fail on mixed CPU/CUDA batches.
                dd = d.clone().to(device)
                out = model(dd, solver="anderson", anderson_beta=0.8, anderson_warmup_iters=5)
                pred = out[0] if isinstance(out, tuple) else out
                mask = anchor_node_mask(dd)
                if mask is None or int(mask.sum().item()) == 0:
                    continue
                rel = torch.norm(pred[mask, :3] - dd.y[mask, :3]) / torch.clamp(torch.norm(dd.y[mask, :3]), min=1e-8)
                gkey = _graph_sampling_key(dd, gi)
                rows.append((gkey, float(rel.item())))
        if not rows:
            model.train()
            return
        errs = torch.tensor([r[1] for r in rows], dtype=torch.float32)
        q = float(torch.quantile(errs, torch.tensor(0.7)))
        for gkey, err in rows:
            hard_anchor_multiplier[gkey] = (1.0 + hard_alpha) if err >= q else 1.0
        write_header = (not hard_csv_path.exists()) or (hard_csv_path.stat().st_size == 0)
        with open(hard_csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["epoch", "config_id", "anchor_rel_l2", "hard_mult"])
            for gkey, err in rows:
                w.writerow([epoch_idx, gkey, err, hard_anchor_multiplier.get(gkey, 1.0)])
        model.train()

    for epoch in range(start_epoch, epochs):
        last_epoch_completed = epoch
        if epoch == start_epoch or (epoch % hard_refresh == 0):
            _refresh_hard_mining(epoch)
            loader = _make_train_loader()
        model.train()
        physics_active = epoch >= warm_up_epochs
        lambda_phys = min(1.0, max(0.0, (epoch - warm_up_epochs) / 20.0))
        stage_split = int(os.environ.get("TIER1_DATA_STAGE_EPOCHS", "12"))
        if epoch < stage_split:
            # Stage A: fit kinematics harder while physics ramps in.
            data_scale = float(os.environ.get("TIER1_DATA_SCALE_STAGE1", "700.0"))
            bc_scale = float(os.environ.get("TIER1_BC_SCALE_STAGE1", "8.0"))
            io_scale = float(os.environ.get("TIER1_IO_SCALE_STAGE1", "4.0"))
            wss_scale = float(os.environ.get("TIER1_WSS_SCALE_STAGE1", "8.0"))
        else:
            # Stage B: restore stronger physics regularization terms.
            data_scale = float(os.environ.get("TIER1_DATA_SCALE_STAGE2", "500.0"))
            bc_scale = float(os.environ.get("TIER1_BC_SCALE_STAGE2", "10.0"))
            io_scale = float(os.environ.get("TIER1_IO_SCALE_STAGE2", "5.0"))
            wss_scale = float(os.environ.get("TIER1_WSS_SCALE_STAGE2", "10.0"))
        total_loss_epoch = 0.0

        current_solver = "picard" if epoch < 5 else "anderson"

        if use_lbfgs and epoch >= adam_epochs and not lbfgs_initialized:
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
                                                  device, data_scale=data_scale, bc_scale=bc_scale,
                                                  io_scale=io_scale, wss_scale=wss_scale,
                                                  boundary_data_weight=boundary_data_weight, is_distillation=False)
                loss = loss / accumulation_steps

                if torch.isnan(loss):
                    print(f"\n⚠️ NaN detected! Skipping micro-batch.")
                    continue

                loss.backward()

                # Step optimizer ONLY when we've accumulated enough micro-batches
                if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(loader)):
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()) + list(loss_weighter.parameters()),
                        max_norm=1.0,
                    )
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
            # L-BFGS calls closure multiple times per step; keep closure data fixed.
            # Snapshot once on CPU (safe for VRAM), then move one batch at a time in closure.
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
                        data_scale=data_scale,
                        bc_scale=bc_scale,
                        io_scale=io_scale,
                        wss_scale=wss_scale,
                        boundary_data_weight=boundary_data_weight,
                        is_distillation=False,
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
            save_path = model_dir / "tier1_best_loss.pth"
            torch.save(model.state_dict(), save_path)
            print(f"⭐ Saved Best Loss Model to {save_path}")

        if epoch % 5 == 0:
            validate_and_plot(model, val_data[0], epoch, device, tier="tier1")

        should_save_ckpt = ((epoch + 1) % ckpt_every == 0) or (epoch == epochs - 1)
        if should_save_ckpt:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": (scheduler.state_dict() if not lbfgs_initialized else None),
                "loss_weighter_state_dict": loss_weighter.state_dict(),
                "best_phys_score": best_phys_score,
                "best_loss": best_loss,
                "optimizer_type": ("LBFGS" if lbfgs_initialized else "AdamW"),
            }
            torch.save(checkpoint, latest_ckpt_save)
            print(f"💾 Saved Tier 1 checkpoint -> {latest_ckpt_save.name} (every {ckpt_every} epoch(s))")

        diary.log_epoch_end(
            epoch,
            avg_epoch_loss=float(avg_loss),
            lr=float(optimizer.param_groups[0]["lr"]),
            lambda_phys=float(lambda_phys),
            data_scale=float(data_scale),
            bc_scale=float(bc_scale),
            io_scale=float(io_scale),
            wss_scale=float(wss_scale),
            solver=current_solver,
            lbfgs=bool(lbfgs_initialized),
            physics_active=bool(physics_active),
            best_loss_so_far=float(best_loss),
            best_phys_score_so_far=float(best_phys_score),
            best_rel_l2_so_far=float(best_rel_l2),
        )

        if epoch % 2 == 0:
            scores = quantify_performance(model, val_loader, kernels, device, tier="tier1")

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
                f"Wall |u| mean: {scores.get('wall_slip', float('nan')):.4f} "
                f"(p90 {scores.get('wall_slip_p90', float('nan')):.4f})"
            )
            print(
                f"   γ̇ MSE (anchor): {scores.get('shear_mse', float('nan')):.4e} "
                f"(p90 {scores.get('shear_mse_p90', float('nan')):.4e}) | "
                f"Batches w/ anchors: {int(scores.get('val_anchor_batches', 0))}/"
                f"{int(scores.get('val_total_batches', 0))}"
            )

            with torch.no_grad():
                safe_vars = loss_weighter.clamped_log_vars()
                weights = torch.exp(-safe_vars)
                w_cont = float(weights[0].item())
                w_mom = float(weights[1].item())
                print(f"⚖️ Learned PDE Weights -> Cont: {w_cont:.2f} | Mom: {w_mom:.2f}")
                n_val_anchor = sum(1 for d in val_data if graph_has_anchor(d))
                print(f"📌 Val split: anchors={n_val_anchor} | physics={max(0, len(val_data) - n_val_anchor)}")

            phys_score = scores.get('rel_l2', 0) + scores.get('continuity', 0)
            diary.log_validation(
                epoch,
                scores,
                weight_cont=w_cont,
                weight_mom=w_mom,
                best_rel_l2=best_rel_l2,
                val_no_improve=val_no_improve,
                phys_score=float(phys_score),
                n_val_anchors=n_val_anchor,
                n_val_physics=max(0, len(val_data) - n_val_anchor),
            )
            if not lbfgs_initialized:
                plateau_scheduler.step(float(scores.get('rel_l2', 0)))

            rel_l2 = float(scores.get("rel_l2", float("inf")))
            if (best_rel_l2 - rel_l2) > plateau_min_delta:
                best_rel_l2 = rel_l2
                val_no_improve = 0
            else:
                val_no_improve += 1
                if val_no_improve >= plateau_patience:
                    print(
                        f"🛑 Early stopping: validation Rel L2 did not improve by > {plateau_min_delta:.4f} "
                        f"for {plateau_patience} validation checks."
                    )
                    early_stopped = True
                    diary.log_event(
                        "early_stop",
                        epoch=epoch,
                        reason="val_rel_l2_plateau",
                        plateau_min_delta=plateau_min_delta,
                        plateau_patience=plateau_patience,
                        last_rel_l2=rel_l2,
                    )
                    break

            if phys_score < best_phys_score and physics_active:
                best_phys_score = phys_score
                save_path = model_dir / "tier1_best_physics.pth"
                torch.save(model.state_dict(), save_path)
                print(f"⭐ Saved Best Physics Model to {save_path}")

    _emit_tier1_run_end(interrupted=False)
    print(f"Tier 1 Training Complete. Best Physical Score: {best_phys_score:.4f} | Best Loss: {best_loss:.4f}")


if __name__ == "__main__":
    train_t1_predictor()