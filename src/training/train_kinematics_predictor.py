"""
Unified Kinematics Predictor Training with Mathematical Continuation Ramp.
Implements dynamic dataset swapping, Carreau-Yasuda parameter ramping,
and curriculum-based loss isolation.
"""
import argparse
import json
import math
import os
import random
import re
import shutil
import sys
import time
import warnings
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.architecture.ginodeq import GINO_DEQ
from src.architecture.kinematics_model_config import (
    build_gino_deq_from_ctor,
    kinematics_checkpoint_tensors,
    resolve_gino_deq_ctor_kwargs,
    save_kinematics_checkpoint_file,
    snapshot_gino_deq_model_config,
    write_kinematics_architecture_manifest,
)
from src.config import VesselConfig, PhysicsConfig, PredChannels
from src.core_physics.physics_kernels import PhysicsKernels
from src.utils.anchor_mask import graph_has_anchor, anchor_node_mask
from src.utils.kinematics_physics_terms import compute_kinematics_physics_terms
from src.utils.metrics import DynamicLossWeighter, quantify_performance
from src.utils.paths import kinematics_dir
from src.utils.training_diary import TrainingDiary
from src.utils.channel_schema import KINE_Y_SCHEMA, assert_graph_schema, infer_missing_schema
from src.utils.kinematics_console import (
    kinematics_skip_lbfgs,
    kinematics_tqdm_enabled,
    kinematics_val_every,
)
from src.utils.kinematics_geometry import (
    GeometryCurriculumConfig,
    attach_geometry_metadata,
    cohort_level_counts,
    count_anchor_physics,
    geometry_sample_weight,
    split_anchor_physics_stratified,
    train_pool_for_epoch,
    warn_if_single_level_cohort,
)

# Ignore known PyTorch scheduler deprecation noise in training logs.
warnings.filterwarnings("ignore", category=UserWarning, message="The epoch parameter.*")

# -------------------------------------------------------------------------
# Curriculum Definitions
# -------------------------------------------------------------------------
STAGE1_END_EPOCH = 40
STAGE2_END_EPOCH = 60


def _prune_kine_training_artifacts(target_dir: Path, *, keep: int = 3) -> int:
    """Keep only the newest numbered kinematics checkpoint/state files."""
    if keep < 1:
        return 0
    target = Path(target_dir)
    groups = {
        "kinematics_ckpt": re.compile(r"^kinematics_ckpt_(\d+)\.pth$"),
        "kinematics_state": re.compile(r"^kinematics_state_(\d+)\.pth$"),
    }
    removed = 0
    for pattern in groups.values():
        matches = []
        for path in target.glob("*.pth"):
            found = pattern.match(path.name)
            if found:
                matches.append((int(found.group(1)), path))
        matches.sort(key=lambda item: item[0], reverse=True)
        for _, old_path in matches[keep:]:
            try:
                old_path.unlink()
                removed += 1
            except OSError:
                pass
    return removed


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
        return 3, 0.3568, 0.056, "carreau"


# -------------------------------------------------------------------------
# Data Loading & Management
# -------------------------------------------------------------------------
def load_dataset(
    phase: str,
    rheology: str | None = None,
    limit: int | None = None,
    *,
    attach_geometry: bool = True,
    shuffle_graphs: bool = False,
    graph_load_seed: int = 42,
):
    cfg = VesselConfig(phase=phase)
    if phase == "kinematics" and rheology:
        from src.utils.kinematics_paths import kinematics_graph_rheology_dir

        data_dir = kinematics_graph_rheology_dir(rheology)
    else:
        data_dir = cfg.graph_output_dir
        if rheology:
            data_dir = data_dir / str(rheology).lower()

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}. "
            "Expected rheology-split graphs under graphs_kinematics/<newtonian|carreau>."
        )

    paths = sorted(data_dir.glob("vessel_*.pt"))
    if shuffle_graphs:
        paths = list(paths)
        rng = random.Random(int(graph_load_seed))
        rng.shuffle(paths)
        print(f"[kin] Shuffled graph load order (seed={int(graph_load_seed)}).")
    if limit is not None:
        paths = paths[:limit]
        print(f"[kin] WARN limit-data active: only loading {limit} graphs.")
    cap_raw = os.environ.get("KINEMATICS_GRAPH_CAP", "").strip()
    if cap_raw:
        n_cap = int(cap_raw)
        n_total = len(paths)
        if n_total > n_cap:
            rng = random.Random(int(graph_load_seed))
            paths = rng.sample(list(paths), n_cap)
            print(
                f"[kin] KINEMATICS_GRAPH_CAP={n_cap}: sampled {n_cap}/{n_total} graphs "
                f"(seed={int(graph_load_seed)})."
            )
    if not paths:
        raise RuntimeError(
            f"No graph files found in dataset directory: {data_dir}. "
            "Expected at least one vessel_*.pt file."
        )
    dataset = []
    print(f"[kin] Loading {len(paths)} graphs from {data_dir}...")
    file_iter = paths
    if kinematics_tqdm_enabled():
        file_iter = tqdm(paths, leave=False, ascii=sys.platform == "win32")
    for i, f in enumerate(file_iter):
        data = torch.load(f, weights_only=False)
        data = infer_missing_schema(data, phase_hint=phase)
        assert_graph_schema(data, expected_y_schema=(KINE_Y_SCHEMA,))
        if attach_geometry:
            data.graph_stem = f.stem
            attach_geometry_metadata(data, mesh_input_dir=cfg.mesh_input_dir, stem=f.stem)
        dataset.append(data)
        if not kinematics_tqdm_enabled() and (i + 1) % 500 == 0:
            print(f"[kin]   loaded {i + 1}/{len(paths)} graphs")
    counts = cohort_level_counts(dataset)
    print(
        f"   Geometry levels: L0={counts.get(0, 0)}, L1={counts.get(1, 0)}, "
        f"L2={counts.get(2, 0)}, unknown={counts.get(-1, 0)}"
    )
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
    # Canonical ND viscosity scale (shared with biochem channel encoding).
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

    # 5. Static PDE weights (Kendall weighter disabled — avoids negative weighted PDE collapse)
    _ = loss_weighter
    weighted_pdes = 1.0 * l_mom + 1.0 * l_cont

    # Scale up IO/BC weight severely ONLY in Stage 2 when interior data supervision is removed.
    io_weight = 100.0 if stage == 2 else 5.0
    bc_weight = 50.0 if stage == 2 else 5.0

    # 6. Final Composite Loss
    loss = (
        weighted_pdes
        + (weight_data * l_data_kine)
        + (weight_mu * l_data_mu)
        + (bc_weight * l_bc)
        + (io_weight * l_io)
        + (1.0 * p_grad_loss)
        + (weight_wss * l_wss)
        + (0.1 * jac_loss)
    )

    weighted_data_kine = weight_data * l_data_kine
    weighted_data_mu = weight_mu * l_data_mu
    weighted_bc = bc_weight * l_bc
    weighted_io = io_weight * l_io
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
def resolve_kinematics_device(*, require_cuda: bool = True) -> str:
    """Pick training device and refuse CPU-only runs when require_cuda is set."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"[kin] Training device: CUDA - {device_name}")
        return "cuda"
    print("[kin] Training device: CPU (CUDA not available)")
    if require_cuda:
        print(
            "[kin] ERROR: kinematics training requires CUDA. "
            "Use a CUDA-enabled PyTorch build with a visible GPU."
        )
        sys.exit(1)
    print("[kin] WARN continuing on CPU (require_cuda=False).")
    return "cpu"


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
    limit_data: int | None = None,
    shuffle_graphs: bool = False,
    graph_load_seed: int = 42,
    geometry_curriculum: GeometryCurriculumConfig | None = None,
    finetune_lr: float | None = None,
    require_cuda: bool = True,
):
    geometry_cfg = geometry_curriculum or GeometryCurriculumConfig()
    device = resolve_kinematics_device(require_cuda=require_cuda)
    # Handoff to L-BFGS in late Stage 3 by default.

    phys_cfg = PhysicsConfig(phase="kinematics")  # Kinematics supports Carreau
    kernels = PhysicsKernels(phys_cfg=phys_cfg)
    default_ctor = resolve_gino_deq_ctor_kwargs(None, {})
    model = build_gino_deq_from_ctor(phys_cfg, default_ctor).to(device)
    training_manifest = {
        "epochs": int(epochs),
        "adam_epochs": int(adam_epochs),
        "stage1_end_epoch": int(stage1_end_epoch),
        "stage2_end_epoch": int(stage2_end_epoch),
        "accum_steps": int(accum_steps),
        "weight_data": float(weight_data),
        "weight_mu": float(weight_mu),
        "weight_wss": float(weight_wss),
        "max_lbfgs_graphs": int(max_lbfgs_graphs),
        "limit_data": limit_data,
        "shuffle_graphs": bool(shuffle_graphs),
        "graph_load_seed": int(graph_load_seed),
        "geometry_curriculum": {
            "enabled": bool(geometry_cfg.enabled),
            "phase": str(geometry_cfg.phase),
            "foundation_mix": list(geometry_cfg.foundation_mix),
            "ramp_end_mix": list(geometry_cfg.ramp_end_mix),
            "l2_heavy_mix": list(geometry_cfg.l2_heavy_mix),
            "hard_mining_start_epoch": int(geometry_cfg.hard_mining_start_epoch),
            "l0l1_only_epochs": int(geometry_cfg.l0l1_only_epochs),
        },
        "finetune_lr": finetune_lr,
        "model_config": snapshot_gino_deq_model_config(model),
    }

    # Legacy checkpoint field only; PDE terms use fixed 1:1 weights in compute_step_loss.
    loss_weighter = DynamicLossWeighter(num_losses=2).to(device)
    opt_params = list(model.parameters())
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
        print(f"[kin] Resuming training from: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            resume_meta, resume_state = kinematics_checkpoint_tensors(ckpt)
            resume_ctor = resolve_gino_deq_ctor_kwargs(resume_meta, resume_state)
            if resume_meta.get("model_config"):
                print("[kin] GINO_DEQ architecture from checkpoint model_config.")
            model = build_gino_deq_from_ctor(phys_cfg, resume_ctor).to(device)
            opt_params = list(model.parameters())
            optimizer = optim.AdamW(opt_params, lr=1e-4, weight_decay=1e-5)
            warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warm_up_epochs)
            cosine_scheduler = CosineAnnealingLR(optimizer, T_max=decay_epochs, eta_min=1e-6)
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warm_up_epochs],
            )
            model.load_state_dict(resume_state)
            if isinstance(ckpt.get("training_manifest"), dict):
                training_manifest.update(ckpt["training_manifest"])
            if "loss_weighter_state_dict" in ckpt:
                loss_weighter.load_state_dict(ckpt["loss_weighter_state_dict"])
            if "optimizer_state_dict" in ckpt and ckpt.get("optimizer_name", "AdamW") == "AdamW":
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                except (ValueError, RuntimeError):
                    print("[kin] WARN could not restore AdamW optimizer state; fresh optimizer.")
            if "scheduler_state_dict" in ckpt:
                try:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                except (ValueError, RuntimeError):
                    print("[kin] WARN could not restore scheduler state; fresh scheduler.")
            start_epoch = int(ckpt.get("epoch", ckpt.get("best_epoch", -1))) + 1
            best_val_composite_loss = float(ckpt.get("best_val_composite_loss", best_val_composite_loss))
            # Always re-enter LBFGS via normal handoff so static batches are rebuilt deterministically.
            lbfgs_initialized = False
            print(f"[kin] Loaded full training state (next epoch: {start_epoch})")
        else:
            model.load_state_dict(ckpt)
            m = re.search(r"kinematics_ckpt_(\d+)\.pth$", str(resume_from))
            if m:
                start_epoch = int(m.group(1))
            print(f"[kin] Loaded model-only checkpoint (next epoch: {start_epoch})")

    if finetune_lr is not None and finetune_lr > 0:
        for pg in optimizer.param_groups:
            pg["lr"] = float(finetune_lr)
        print(f"[kin] Finetune LR set to {float(finetune_lr):.2e}")

    diary = TrainingDiary("kinematics")
    diary.log_run_start(
        epochs=int(epochs),
        adam_epochs=int(adam_epochs),
        stage1_end_epoch=int(stage1_end_epoch),
        stage2_end_epoch=int(stage2_end_epoch),
        device=str(device),
        model_config=training_manifest.get("model_config"),
    )
    val_every = kinematics_val_every(int(epochs))
    try:
        arch_path = diary.run_dir / "kinematics_architecture.json"
        with open(arch_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_id": diary.run_dir.name,
                    "training_manifest": training_manifest,
                },
                f,
                indent=2,
            )
            f.write("\n")
        print(f"[kin] Architecture manifest: {arch_path}")
    except OSError:
        pass

    def make_loader(data_split, n_anchors, n_physics, epoch: int, stage: int):
        # 50/50 Weighted Random Sampler logic extracted from Kinematics
        level_weights = geometry_cfg.level_weights(
            epoch,
            stage,
            stage1_end=int(stage1_end_epoch),
            stage2_end=int(stage2_end_epoch),
        )
        if n_anchors > 0 and n_physics > 0:
            w_anchor = 0.5 / n_anchors
            w_phys = 0.5 / n_physics
            weights = []
            for d in data_split:
                geo = geometry_sample_weight(d, level_weights) if geometry_cfg.enabled else 1.0
                if graph_has_anchor(d):
                    gkey = int(getattr(d, "config_id", 0))
                    weights.append(w_anchor * geo * hard_anchor_multiplier.get(gkey, 1.0))
                else:
                    weights.append(w_phys * geo)
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

    print("[kin] Starting unified kinematics training...")

    for epoch in range(start_epoch, epochs):
        stage, current_n, current_mu_0, target_rheology = get_stage_physics(
            epoch, int(stage1_end_epoch), int(stage2_end_epoch)
        )
        target_phase = "kinematics"

            # 1. Dynamic DataLoader Swapping
        # 1. Dynamic DataLoader Swapping
        if current_phase_loaded != target_rheology:
            print(
                f"\n[kin] Swapping dataset to {target_phase.upper()}/{target_rheology.upper()} "
                f"for stage {stage} (n={current_n:.3f}, mu0={current_mu_0:.4f})"
            )
            dataset = load_dataset(
                target_phase,
                target_rheology,
                limit=limit_data,
                shuffle_graphs=shuffle_graphs,
                graph_load_seed=graph_load_seed,
            )
            if geometry_cfg.enabled:
                splits = split_anchor_physics_stratified(dataset)
            else:
                splits = split_anchor_physics(dataset)
            train_data, val_data = splits["train"], splits["val"]
            n_anchors, n_physics = splits["n_anchors"], splits["n_physics"]
            current_phase_loaded = target_rheology
            warn_if_single_level_cohort(
                dataset,
                curriculum=geometry_cfg,
                epoch=epoch,
                stage=stage,
                stage1_end=int(stage1_end_epoch),
                stage2_end=int(stage2_end_epoch),
            )

            # Reset hard mining when swapping datasets
            hard_anchor_multiplier.clear()
            if stage == 3 and not lbfgs_initialized:
                print("[kin] Resetting AdamW momentum buffers for stage 3...")
                optimizer.state.clear()

        mining_interval = int(geometry_cfg.hard_mining_interval)
        mining_start = (
            int(geometry_cfg.hard_mining_start_epoch) if geometry_cfg.enabled else 4
        )
        train_epoch_data = train_pool_for_epoch(
            train_data,
            curriculum=geometry_cfg,
            epoch=epoch,
            stage=stage,
            stage1_end=int(stage1_end_epoch),
            stage2_end=int(stage2_end_epoch),
        )
        n_anchors_ep, n_physics_ep = count_anchor_physics(train_epoch_data)

        if geometry_cfg.enabled:
            geo_line = geometry_cfg.describe(
                epoch, stage, stage1_end=int(stage1_end_epoch), stage2_end=int(stage2_end_epoch)
            )
            pool_counts = cohort_level_counts(train_epoch_data)
            print(
                f"[kin] {geo_line} | train_pool={len(train_epoch_data)} "
                f"(L0={pool_counts.get(0, 0)}, L1={pool_counts.get(1, 0)}, L2={pool_counts.get(2, 0)})"
            )

        # 2. Hard Mining Management
        if (
            stage in (1, 3)
            and epoch >= mining_start
            and epoch % mining_interval == 0
            and not lbfgs_initialized
        ):
            print("[kin] Refreshing hard-negative anchor weights...")
            refresh_hard_mining(epoch, train_epoch_data)
        elif epoch >= mining_start and epoch % mining_interval == 0 and not lbfgs_initialized:
            # During ramp/no-anchor phases, anchor rel-L2 is not informative.
            flow_diag = evaluate_mass_flow_health(model, train_data, device)
            if flow_diag is None:
                print("[kin] Flow diagnostic skipped (missing inlet/outlet masks).")
            else:
                print(
                    "[kin] Flow diagnostic "
                    f"(graphs={flow_diag['n_graphs']}): "
                    f"flux_in={flow_diag['inlet_flux']:.3e}, "
                    f"flux_out={flow_diag['outlet_flux']:.3e}, "
                    f"imbalance={flow_diag['imbalance']:.3f}, "
                    f"collapse={flow_diag['collapse_score']:.3f}"
                )
        loader = make_loader(train_epoch_data, n_anchors_ep, n_physics_ep, epoch, stage)

        # 3. L-BFGS Handoff (Kinematics preservation)
        if epoch >= adam_epochs and not lbfgs_initialized and not kinematics_skip_lbfgs():
            print("\n[kin] Switching to L-BFGS optimizer for final refinement...")
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
            ema_metrics: dict[str, float] | None = None
            ema_alpha = 0.1
            use_bar = kinematics_tqdm_enabled()
            step_iter = loader
            if use_bar:
                step_iter = tqdm(
                    loader,
                    desc=f"Ep {epoch:02d} [S{stage}: n={current_n:.3f}, mu0={current_mu_0:.4f}]",
                    ascii=sys.platform == "win32",
                )
            elif epoch == 0 or (epoch + 1) % max(1, epochs // 5) == 0:
                print(
                    f"[kin] ep {epoch:02d} stage {stage} train "
                    f"(n={current_n:.3f}, mu0={current_mu_0:.4f}, steps={len(loader)})"
                )
            optimizer.zero_grad()
            accum_counter = 0
            for idx, data in enumerate(step_iter):
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

                if ema_metrics is None:
                    ema_metrics = {k: float(v) for k, v in metrics.items()}
                else:
                    for k, v in metrics.items():
                        ema_metrics[k] = (ema_alpha * float(v)) + ((1.0 - ema_alpha) * ema_metrics[k])

                lr_val = (
                    optimizer.param_groups[0]["lr"]
                    if hasattr(optimizer, "param_groups") and len(optimizer.param_groups) > 0
                    else float("nan")
                )
                if use_bar:
                    step_iter.set_postfix(
                        {
                            "L_tot": f"{ema_metrics['L_total']:.3f}",
                            "L_data": f"{ema_metrics['L_data']:.3f}",
                            "L_mu": f"{ema_metrics['L_mu']:.3f}",
                            "L_mom": f"{ema_metrics['L_mom']:.3f}",
                            "L_cont": f"{ema_metrics['L_cont']:.3f}",
                            "L_bc": f"{ema_metrics['L_bc']:.3f}",
                            "L_io": f"{ema_metrics['L_io']:.3f}",
                            "|g|": f"{grad_norm:.2f}",
                            "LR": f"{lr_val:.2e}",
                        }
                    )
            scheduler.step()
        else:
            print(f"[kin] L-BFGS step (ep {epoch:02d}) [S{stage}: n={current_n:.3f}]")
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
            os.makedirs(kinematics_dir(), exist_ok=True)
            ckpt_path = kinematics_dir() / f"kinematics_ckpt_{epoch + 1}.pth"
            save_kinematics_checkpoint_file(
                ckpt_path,
                model,
                checkpoint_role=f"kinematics_ckpt_{epoch + 1}",
                best_epoch=int(epoch),
                training_manifest=training_manifest,
            )
            save_kinematics_checkpoint_file(
                kinematics_dir() / "kinematics_ckpt_latest.pth",
                model,
                checkpoint_role="kinematics_ckpt_latest",
                best_epoch=int(epoch),
                training_manifest=training_manifest,
            )
            state_path = kinematics_dir() / f"kinematics_state_{epoch + 1}.pth"
            state_payload = {
                "epoch": int(epoch),
                "model_state_dict": model.state_dict(),
                "model_config": snapshot_gino_deq_model_config(model),
                "training_manifest": dict(training_manifest),
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
            torch.save(state_payload, kinematics_dir() / "kinematics_state_latest.pth")
            _prune_kine_training_artifacts(kinematics_dir(), keep=3)

        run_val = len(val_data) > 0 and (
            epoch % val_every == 0
            or epoch == epochs - 1
            or (epoch == adam_epochs - 1 and not lbfgs_initialized)
        )
        if run_val:
            val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
            scores = quantify_performance(model, val_loader, kernels, device, phase="kinematics")
            rel_l2 = float(scores.get("rel_l2", float("nan")))
            continuity = float(scores.get("continuity", float("nan")))
            val_comp = rel_l2 + 100.0 * continuity
            level_bits = []
            for lvl in (0, 1, 2):
                key = f"rel_l2_level_{lvl}"
                val = scores.get(key)
                if val is not None and val == val:
                    level_bits.append(f"L{lvl}={float(val):.3f}")
            level_msg = f" | {' '.join(level_bits)}" if level_bits else ""
            if math.isfinite(rel_l2) and math.isfinite(continuity):
                print(
                    f"[kin] [Validation] Rel L2: {rel_l2:.4f} | "
                    f"div_u mean: {continuity:.3e} | composite: {val_comp:.4f}{level_msg}"
                )
            else:
                print(
                    f"[kin] [Validation] non-finite metrics "
                    f"(rel_l2={rel_l2}, continuity={continuity}); best ckpt unchanged"
                )
            if stage == 3 and math.isfinite(val_comp) and val_comp < best_val_composite_loss:
                best_val_composite_loss = val_comp
                save_kinematics_checkpoint_file(
                    kinematics_dir() / "kinematics_best.pth",
                    model,
                    checkpoint_role="kinematics_best",
                    best_epoch=int(epoch),
                    rel_l2=rel_l2,
                    continuity=continuity,
                    composite=val_comp,
                    run_id=str(getattr(diary, "run_dir", Path(".")).name),
                    training_manifest=training_manifest,
                )
                manifest_path = write_kinematics_architecture_manifest(
                    snapshot_gino_deq_model_config(model),
                    best_epoch=int(epoch),
                    rel_l2=rel_l2,
                    continuity=continuity,
                    composite=val_comp,
                    run_id=str(getattr(diary, "run_dir", Path(".")).name),
                    extra={"training_manifest": training_manifest},
                )
                print("[kin] Saved new best kinematics model")
                print(f"[kin] Updated {manifest_path.name} in {kinematics_dir()}")
            try:
                os.makedirs(kinematics_dir(), exist_ok=True)
                val_record = {
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
                for lvl in (0, 1, 2):
                    key = f"rel_l2_level_{lvl}"
                    lvl_val = scores.get(key)
                    if lvl_val is not None and lvl_val == lvl_val:
                        val_record[key] = float(lvl_val)
                with open(kinematics_dir() / "kinematics_validation.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(val_record) + "\n")
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
                "   -> Loss breakdown (avg/step): "
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
            print("   -> Loss breakdown skipped (non-positive total weighted contribution).")
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

    best_path = kinematics_dir() / "kinematics_best.pth"
    if not best_path.exists():
        latest = kinematics_dir() / "kinematics_ckpt_latest.pth"
        if latest.exists():
            shutil.copy2(latest, best_path)
            print(f"[kin] WARN no Carreau best saved; copied {latest.name} -> kinematics_best.pth")

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
        "--limit-data",
        type=int,
        default=None,
        help="Max graphs to load for fast debugging (not for production runs).",
    )
    parser.add_argument(
        "--shuffle-graphs",
        action="store_true",
        help="Shuffle vessel_*.pt order before limit/split (avoids sorted-prefix bias).",
    )
    parser.add_argument(
        "--graph-load-seed",
        type=int,
        default=42,
        help="RNG seed for --shuffle-graphs.",
    )
    parser.add_argument(
        "--no-geometry-curriculum",
        action="store_true",
        help="Disable L0/L1/L2 weighted sampling and stratified val split.",
    )
    parser.add_argument(
        "--geometry-phase",
        choices=("auto", "foundation", "ramp", "l2_heavy", "off"),
        default="auto",
        help="Geometry curriculum: auto=foundation->ramp->l2_heavy by stage; off=uniform.",
    )
    parser.add_argument(
        "--hard-mining-start-epoch",
        type=int,
        default=16,
        help="First epoch for hard-negative anchor mining (with geometry curriculum).",
    )
    parser.add_argument(
        "--l0l1-only-epochs",
        type=int,
        default=6,
        help="Stage-1 Newtonian epochs using only L0+L1 graphs (no L2 in train pool).",
    )
    parser.add_argument(
        "--finetune-lr",
        type=float,
        default=None,
        help="Override AdamW LR after resume (L2-heavy finetune, e.g. 1e-5).",
    )
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
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="No tqdm bars; epoch/validation lines only (copy-paste friendly logs).",
    )
    args = parser.parse_args()

    if args.quiet:
        os.environ["KINEMATICS_QUIET"] = "1"
        os.environ["KINEMATICS_VAL_PROGRESS"] = "0"
        os.environ["KINEMATICS_TQDM"] = "0"

    if args.fresh and args.resume is not None:
        raise ValueError("Cannot use --fresh together with --resume.")

    ckpt_dir = kinematics_dir()
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
                print("[kin] No latest checkpoint found; starting fresh.")
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

    geom_enabled = not args.no_geometry_curriculum
    geom_phase = "off" if args.no_geometry_curriculum else str(args.geometry_phase)
    geometry_curriculum = GeometryCurriculumConfig(
        enabled=geom_enabled,
        phase=geom_phase,
        hard_mining_start_epoch=int(args.hard_mining_start_epoch),
        l0l1_only_epochs=int(args.l0l1_only_epochs),
    )

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
        limit_data=args.limit_data,
        shuffle_graphs=bool(args.shuffle_graphs),
        graph_load_seed=int(args.graph_load_seed),
        geometry_curriculum=geometry_curriculum,
        finetune_lr=args.finetune_lr,
    )
