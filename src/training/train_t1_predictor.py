"""Tier 1 predictor training (GINO-DEQ).

**Explorer / sweeps:** see ``src/training/t1_explorer.py`` and env vars ``TIER1_EXPERIMENT_NAME``,
``TIER1_KINE_WEIGHT_MODE``, ``TIER1_GEOMETRY_LEVEL``, ``TIER1_LATENT_DIM``, …
Use ``TIER1_EPOCHS`` (default 60) for short exploratory runs, e.g. ``25``.
Each run writes ``outputs/reports/experiments/tier1_<name>_<ts>.json`` for comparing runs.

Example (SDF-gradient–weighted kinematic loss, level-1 vessels only)::

    set TIER1_EXPERIMENT_NAME=sdf_grad_l1
    set TIER1_KINE_WEIGHT_MODE=sdf_grad
    set TIER1_GEOMETRY_LEVEL=1
    set TIER1_EPOCHS=25
    py -m src.training.train_t1_predictor
"""
import argparse
import os
import sys
if sys.platform != "win32":
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
from src.utils.kinematics_physics_terms import (
    compute_anchor_kinematic_importance,
    compute_kinematics_physics_terms,
)
from src.utils.training_diary import TrainingDiary, env_snapshot
from src.training.t1_explorer import (
    T1ExplorerConfig,
    filter_graph_paths_by_geometry_level,
    write_t1_experiment_artifact,
)
from src.training.trainer import _resolve_t1_dataset_tier
import random


def load_dataset(explorer: Optional[T1ExplorerConfig] = None):
    dataset_tier = _resolve_t1_dataset_tier(explorer)
    cfg = VesselConfig(tier=dataset_tier)
    if not cfg.graph_output_dir.exists():
        print(f"⚠️ Tier 1 graph dir not found: {cfg.graph_output_dir} (dataset_tier={dataset_tier})")
        return []
    paths = sorted(cfg.graph_output_dir.glob("vessel_*.pt"))
    if explorer is not None and explorer.geometry_level is not None:
        raw_dir = cfg.mesh_input_dir
        before = len(paths)
        paths = filter_graph_paths_by_geometry_level(list(paths), raw_dir, explorer.geometry_level)
        print(
            f"🔎 TIER1_GEOMETRY_LEVEL={explorer.geometry_level}: "
            f"{len(paths)}/{before} graphs (json \"level\" filter)."
        )
    dataset = []
    print(f"📂 Loading Tier 1 graphs from {cfg.graph_output_dir} (dataset_tier={dataset_tier})...")
    for f in tqdm(paths):
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
    explorer: Optional[T1ExplorerConfig] = None,
    tier1_kine_p_weight: float = 1.0,
    re_ref: Optional[float] = None,
    re_scale: Optional[float] = None,
    train_loss_weighter: bool = True,
):
    if explorer is not None and explorer.ns_derivative_mode == "autograd":
        data.x = data.x.clone().detach().requires_grad_(True)
    anderson_beta = float(explorer.anderson_beta) if explorer is not None else 0.8
    out = model(data, solver=current_solver, anderson_beta=anderson_beta, anderson_warmup_iters=5)
    if isinstance(out, tuple):
        pred, jac_loss = out
    else:
        pred = out
        jac_loss = torch.tensor(0.0, device=device)

    node_is_anchor = anchor_node_mask(data)
    kine_imp = None
    if explorer is not None:
        props = kernels._get_geometric_props(data)
        kine_imp = compute_anchor_kinematic_importance(
            data,
            node_is_anchor,
            mode=explorer.kine_weight_mode,
            sdf_wall_beta=explorer.sdf_wall_beta,
            sdf_wall_tau=explorer.sdf_wall_tau,
            sdf_grad_beta=explorer.sdf_grad_beta,
            shear_true_alpha=explorer.shear_true_alpha,
            kernels=kernels,
            props=props,
        )
    terms = compute_kinematics_physics_terms(
        pred,
        data,
        kernels,
        tier="tier1",
        boundary_data_weight=boundary_data_weight,
        tier1_kine_p_weight=float(tier1_kine_p_weight),
        anchor_kine_importance=kine_imp,
        re_ref=re_ref,
        re_scale=re_scale,
        kinematics_mode=(explorer.kinematics_mode if explorer is not None else None),
    )
    l_wss = terms["l_wss"]
    l_data_kine = terms["l_data_kine"]
    l_mom = terms["l_mom"]
    l_cont = terms["l_cont"]
    l_bc = terms["l_bc"]
    l_io = terms["l_io"]

    lambda_cont = float(explorer.lambda_cont) if explorer is not None else 1.0
    loss_weight_mode = (explorer.loss_weight_mode if explorer is not None else "dynamic").strip().lower()

    # Optional gauge-invariant pressure supervision via pressure gradients on anchor nodes.
    p_grad_loss = torch.tensor(0.0, device=device)
    p_grad_weight = float(explorer.p_grad_supervision) if explorer is not None else 0.0
    if p_grad_weight > 0.0:
        props = kernels._get_geometric_props(data)
        c_p_pred = kernels._compute_derivatives(pred[:, 2:3], props)
        c_p_true = kernels._compute_derivatives(data.y[:, 2:3], props)
        p_pred_grad = c_p_pred[:, 0:2, 0]
        p_true_grad = c_p_true[:, 0:2, 0]
        if node_is_anchor is not None and int(node_is_anchor.sum().item()) > 0:
            p_grad_loss = torch.nn.functional.mse_loss(p_pred_grad[node_is_anchor], p_true_grad[node_is_anchor])
        else:
            # Never supervise pressure gradients on physics-only batches with no anchor labels.
            p_grad_loss = torch.tensor(0.0, device=device)

    pde_losses = [l_mom, lambda_cont * l_cont]
    pde_scales = [lambda_phys, lambda_phys]
    data_terms = [
        float(data_scale) * l_data_kine,
        float(bc_scale) * l_bc,
        float(io_scale) * l_io,
        float(wss_scale) * l_wss,
        float(p_grad_weight) * p_grad_loss,
    ]
    if loss_weight_mode == "fixed" or loss_weighter is None:
        weighted_pdes = (lambda_phys * l_mom) + (lambda_phys * lambda_cont * l_cont)
        weighted_data = sum(data_terms)
    elif loss_weight_mode == "grad_norm":
        pde_term = l_mom + (lambda_cont * l_cont)
        ref_param = next((p for p in model.parameters() if p.requires_grad), None)
        if ref_param is None:
            weighted_pdes = lambda_phys * pde_term
        else:
            g_data = torch.autograd.grad(
                l_data_kine,
                ref_param,
                retain_graph=True,
                allow_unused=True,
            )[0]
            g_pde = torch.autograd.grad(
                pde_term,
                ref_param,
                retain_graph=True,
                allow_unused=True,
            )[0]
            g_data_norm = torch.linalg.vector_norm(g_data) if g_data is not None else torch.tensor(0.0, device=device)
            g_pde_norm = torch.linalg.vector_norm(g_pde) if g_pde is not None else torch.tensor(0.0, device=device)
            ratio = torch.clamp(g_data_norm / (g_pde_norm + 1e-8), min=0.1, max=10.0).detach()
            weighted_pdes = lambda_phys * ratio * pde_term
        weighted_data = sum(data_terms)
    else:
        # Dynamically weight only PDE residuals; keep supervised terms on rigid static scales.
        pde_losses = [
            lambda_phys * l_mom,
            lambda_phys * lambda_cont * l_cont,
        ]
        safe_log_vars = loss_weighter.clamped_log_vars()
        if not train_loss_weighter:
            safe_log_vars = safe_log_vars.detach()
        weighted_terms = []
        for idx, lv in enumerate(safe_log_vars):
            precision = torch.exp(-lv)
            weighted_terms.append(precision * pde_losses[idx] + lv)
        weighted_pdes = sum(weighted_terms)
        weighted_data = sum(data_terms)

    # Combine losses cleanly
    loss = (
        weighted_pdes
        + weighted_data
        + (0.1 * jac_loss)
    )

    metrics = {
        "L_data": l_data_kine.item(),
        "L_mom": l_mom.item(),
        "L_cont": l_cont.item(),
        "L_bc": l_bc.item(),
        "L_io": l_io.item(),
        "L_jac": jac_loss.item(),
        "L_wss": l_wss.item(),
        "L_pgrad": p_grad_loss.item(),
        "A_nodes": int(node_is_anchor.sum().item()) if node_is_anchor is not None else 0,
    }
    return loss, metrics


def train_t1_predictor(
    epochs: Optional[int] = None,
    lr: float = 1e-4,
    warm_up_epochs: Optional[int] = None,
    adam_epochs: Optional[int] = None,
    explorer: Optional[T1ExplorerConfig] = None,
):
    explorer = explorer or T1ExplorerConfig.from_env()
    if epochs is None:
        epochs = max(1, int(os.environ.get("TIER1_EPOCHS", "60")))
    if adam_epochs is None:
        raw_adam = os.environ.get("TIER1_ADAM_EPOCHS", "").strip()
        adam_epochs = int(raw_adam) if raw_adam else epochs
    adam_epochs = max(1, min(int(adam_epochs), int(epochs)))
    if warm_up_epochs is None:
        raw_w = os.environ.get("TIER1_WARM_UP_EPOCHS", "").strip()
        if raw_w:
            warm_up_epochs = int(raw_w)
        else:
            warm_up_epochs = min(10, max(1, adam_epochs // 3))
    warm_up_epochs = max(0, min(int(warm_up_epochs), adam_epochs - 1))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device being used:", device)
    print(
        f"⏱️ epochs={epochs} | warm_up={warm_up_epochs} | adam_phase={adam_epochs} "
        f"(set TIER1_EPOCHS / TIER1_WARM_UP_EPOCHS / TIER1_ADAM_EPOCHS to override)"
    )
    print(
        f"🔬 Explorer: name={explorer.experiment_name!r} | "
        f"kine_weight={explorer.kine_weight_mode} | "
        f"latent={explorer.latent_dim} | deq_iters={explorer.deq_max_iters} | "
        f"kinematics_mode={explorer.kinematics_mode} | ns_derivatives={explorer.ns_derivative_mode} | "
        f"act={explorer.activation_fn} | fourier_base={explorer.fourier_base:.2f} | "
        f"loss_weight={explorer.loss_weight_mode} | anderson_beta={explorer.anderson_beta:.2f} | "
        f"lambda_cont={explorer.lambda_cont:.2f} | re_curriculum={explorer.re_curriculum} | "
        f"p_grad_sup={explorer.p_grad_supervision:.3f}"
    )
    phys_cfg = PhysicsConfig(tier="tier1")
    model = GINO_DEQ(
        in_channels=15,
        out_channels=5,
        latent_dim=explorer.latent_dim,
        max_iters=explorer.deq_max_iters,
        num_fourier_freqs=explorer.num_fourier_freqs,
        phys_cfg=phys_cfg,
        kinematics_mode=explorer.kinematics_mode,
        activation_fn=explorer.activation_fn,
        fourier_base=explorer.fourier_base,
    ).to(device)

    kernels = PhysicsKernels(phys_cfg=phys_cfg)
    kernels.ns_derivative_mode = explorer.ns_derivative_mode
    kernels.cfg.kinematics_mode = explorer.kinematics_mode
    kernels.advect_detach = bool(explorer.advect_detach)

    # Keep dynamic weighing available for momentum+continuity when selected.
    loss_weighter = None
    if explorer.loss_weight_mode == "dynamic":
        loss_weighter = DynamicLossWeighter(num_losses=2).to(device)

    disable_figures = (os.environ.get("TIER1_DISABLE_FIGURES", "0").strip().lower() in ("1", "true", "yes", "on"))
    if not disable_figures:
        fig_dir = reports_dir() / "figures" / "tier1"
        fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Phase 1 Optimizer (AdamW)
    opt_params = list(model.parameters())
    if loss_weighter is not None:
        opt_params += list(loss_weighter.parameters())
    optimizer = optim.AdamW(opt_params, lr=lr, weight_decay=1e-5)

    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warm_up_epochs)
    decay_epochs = adam_epochs - warm_up_epochs
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=decay_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warm_up_epochs])
    plateau_scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, threshold=5e-4, min_lr=1e-6
    )

    dataset_tier = _resolve_t1_dataset_tier(explorer)
    dataset_cfg = VesselConfig(tier=dataset_tier)
    dataset = load_dataset(explorer)
    if not dataset:
        return {
            "status": "no_data",
            "experiment_name": explorer.experiment_name,
            "dataset_tier": dataset_tier,
            "graph_dir": str(dataset_cfg.graph_output_dir),
        }

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
    micro_batch_size = int(os.environ.get("TIER1_MICRO_BATCH_SIZE", "2"))
    accumulation_steps = int(os.environ.get("TIER1_ACCUMULATION_STEPS", "4"))

    target_anchor_fraction = float(os.environ.get("TIER1_TARGET_ANCHOR_FRACTION", "0.5"))
    target_anchor_fraction = min(max(target_anchor_fraction, 0.0), 1.0)
    hard_alpha = max(float(os.environ.get("TIER1_HARD_MINING_ALPHA", "0.8")), 0.0)
    hard_refresh = max(int(os.environ.get("TIER1_HARD_MINING_REFRESH_EPOCHS", "4")), 1)
    boundary_data_weight = max(float(os.environ.get("TIER1_BOUNDARY_DATA_WEIGHT", "2.0")), 1.0)
    tier1_kine_p_weight = max(float(os.environ.get("TIER1_KINE_P_WEIGHT", "1.0")), 0.0)
    use_lbfgs_requested = (os.environ.get("TIER1_USE_LBFGS", "0").strip().lower() in ("1", "true", "yes", "on"))
    use_lbfgs = bool(use_lbfgs_requested)
    if use_lbfgs:
        print("✅ TIER1_USE_LBFGS=1: AdamW warm-up/phase then L-BFGS refinement is enabled.")
        print("ℹ️ Safety note: L-BFGS path snapshots the full loader and can be memory-heavy on large graph datasets.")
    dynamic_freeze_during_warmup = (os.environ.get("TIER1_DYNAMIC_FREEZE_DURING_WARMUP", "1").strip().lower() in ("1", "true", "yes", "on"))
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
        try:
            if loss_weighter is not None and ckpt.get("loss_weighter_state_dict") is not None:
                loss_weighter.load_state_dict(ckpt["loss_weighter_state_dict"])
        except RuntimeError:
            print("ℹ️ Reinitializing Tier 1 PDE loss weighter for current setup.")

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
        t1_explorer=explorer.to_serializable(),
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
                out = model(dd, solver="anderson", anderson_beta=float(explorer.anderson_beta), anderson_warmup_iters=5)
                pred = out[0] if isinstance(out, tuple) else out
                mask = anchor_node_mask(dd)
                if mask is None or int(mask.sum().item()) == 0:
                    continue
                rel = torch.norm(pred[mask, :2] - dd.y[mask, :2]) / torch.clamp(torch.norm(dd.y[mask, :2]), min=1e-8)
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
        if explorer.re_curriculum and epoch == start_epoch:
            print("⚠️ TIER1_RE_CURRICULUM requested, but disabled for supervised Tier 1 to avoid Re/data contradiction.")
        re_scale_epoch = 1.0
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
            if loss_weighter is not None:
                loss_weighter.requires_grad_(False)
            lbfgs_params = [p for p in model.parameters() if p.requires_grad]

            optimizer = optim.LBFGS(
                lbfgs_params,
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
            accum_counter = 0

            for batch_idx, data in enumerate(pbar):
                data = data.to(device)

                # Compute loss (scaled by accumulation steps so the final gradient magnitude is correct)
                loss, metrics = compute_step_loss(
                    model,
                    data,
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
                    explorer=explorer,
                    tier1_kine_p_weight=tier1_kine_p_weight,
                    re_scale=re_scale_epoch,
                    train_loss_weighter=not (dynamic_freeze_during_warmup and epoch < warm_up_epochs),
                )
                accum_counter += 1
                loss = loss / float(accumulation_steps)

                if torch.isnan(loss):
                    print(f"\n⚠️ NaN detected! Skipping micro-batch.")
                    accum_counter = max(0, accum_counter - 1)
                    continue

                loss.backward()

                # Step optimizer ONLY when we've accumulated enough micro-batches
                is_step = ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(loader))
                if is_step:
                    if accum_counter > 0 and accum_counter < accumulation_steps:
                        scale = float(accumulation_steps) / float(accum_counter)
                        for p in opt_params:
                            if p.grad is not None:
                                p.grad.mul_(scale)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        opt_params,
                        max_norm=1.0,
                    )
                    optimizer.step()
                    optimizer.zero_grad()  # Reset for the next effective batch
                    accum_counter = 0

                # Multiply back for display purposes
                total_loss_epoch += (loss.item() * accumulation_steps)

                pbar.set_postfix({
                    "L_tot": f"{(loss.item() * accumulation_steps):.3f}",
                    "L_data": f"{metrics['L_data']:.3f}",
                    "L_mom": f"{metrics['L_mom']:.3f}",
                    "L_cont": f"{metrics['L_cont']:.3f}",
                    "L_bc": f"{metrics['L_bc']:.3f}",
                    "L_io": f"{metrics['L_io']:.3f}",
                    "L_wss": f"{metrics['L_wss']:.3f}",
                    "L_pgrad": f"{metrics['L_pgrad']:.3f}",
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
            n_batches = max(len(epoch_batches_cpu), 1)

            def closure():
                optimizer.zero_grad()
                accumulated_loss = torch.tensor(0.0, device=device)

                for closure_data_cpu in epoch_batches_cpu:
                    closure_data = closure_data_cpu.to(device)
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
                        explorer=explorer,
                        tier1_kine_p_weight=tier1_kine_p_weight,
                        re_scale=re_scale_epoch,
                        train_loss_weighter=not (dynamic_freeze_during_warmup and epoch < warm_up_epochs),
                    )
                    loss = loss / n_batches
                    loss.backward()
                    accumulated_loss += loss.detach()
                    del closure_data

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

        if (not disable_figures) and epoch % 5 == 0:
            validate_and_plot(model, val_data[0], epoch, device, tier="tier1")

        should_save_ckpt = ((epoch + 1) % ckpt_every == 0) or (epoch == epochs - 1)
        if should_save_ckpt:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": (scheduler.state_dict() if not lbfgs_initialized else None),
                "loss_weighter_state_dict": (loss_weighter.state_dict() if loss_weighter is not None else None),
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
            print(
                f"   Rel L2 near-wall (anchor low-|SDF|): {scores.get('rel_l2_near_wall', float('nan')):.4f} "
                f"(p90 {scores.get('rel_l2_near_wall_p90', float('nan')):.4f}) | "
                f"Rel L2 high-|∇SDF| (anchor): {scores.get('rel_l2_high_sdf_grad', float('nan')):.4f} "
                f"(p90 {scores.get('rel_l2_high_sdf_grad_p90', float('nan')):.4f})"
            )

            with torch.no_grad():
                if loss_weighter is not None:
                    safe_vars = loss_weighter.clamped_log_vars()
                    weights = torch.exp(-safe_vars)
                    w_mom = float(weights[0].item()) if weights.numel() > 0 else 1.0
                    w_cont = float(weights[1].item()) if weights.numel() > 1 else float(explorer.lambda_cont)
                else:
                    w_mom = 1.0
                    w_cont = float(explorer.lambda_cont)
                print(f"⚖️ Learned PDE Weights -> Cont: {w_cont:.2f} | Mom: {w_mom:.2f}")
                n_val_anchor = sum(1 for d in val_data if graph_has_anchor(d))
                print(f"📌 Val split: anchors={n_val_anchor} | physics={max(0, len(val_data) - n_val_anchor)}")

            phys_score = scores.get('rel_l2', 0) + (100.0 * scores.get('continuity', 0))
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

    if os.environ.get("TIER1_SKIP_EXPERIMENT_ARTIFACT", "0").strip().lower() not in ("1", "true", "yes", "on"):
        write_t1_experiment_artifact(
            explorer,
            best_rel_l2=best_rel_l2,
            best_phys_score=best_phys_score,
            best_loss=best_loss,
            early_stopped=early_stopped,
            n_graphs=len(dataset),
            n_train=len(train_data),
            n_val=len(val_data),
            graph_dir=str(cfg_paths.graph_output_dir),
        )

    return {
        "status": "ok",
        "experiment_name": explorer.experiment_name,
        "best_rel_l2": float(best_rel_l2),
        "best_phys_score": float(best_phys_score),
        "best_loss": float(best_loss),
        "early_stopped": bool(early_stopped),
        "n_graphs": int(len(dataset)),
        "n_train": int(len(train_data)),
        "n_val": int(len(val_data)),
        "graph_dir": str(cfg_paths.graph_output_dir),
    }


def _parse_args():
    p = argparse.ArgumentParser(description="Tier 1 GINO-DEQ predictor training (explorer-friendly).")
    p.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Sets TIER1_EXPERIMENT_NAME for reports/experiments JSON (same as env).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.experiment_name:
        os.environ["TIER1_EXPERIMENT_NAME"] = args.experiment_name
    train_t1_predictor()