"""Shared DEQ predictor trainer for Tier 1 and Tier 2 loops."""

from __future__ import annotations

import atexit
import csv
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    ReduceLROnPlateau,
    SequentialLR,
)
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.architecture.ginodeq import GINO_DEQ
from src.config import CurriculumConfig, PhysicsConfig, VesselConfig
from src.core_physics.physics_kernels import PhysicsKernels
from src.training.t1_explorer import (
    T1ExplorerConfig,
    filter_graph_paths_by_geometry_level,
    write_t1_experiment_artifact,
)
from src.utils.anchor_mask import anchor_node_mask, graph_has_anchor
from src.utils.kinematics_physics_terms import (
    compute_anchor_kinematic_importance,
    compute_kinematics_physics_terms,
)
from src.utils.metrics import DynamicLossWeighter, quantify_performance, validate_and_plot
from src.utils.paths import get_project_root, reports_dir, resolve_checkpoint, stage_a_dir
from src.utils.samplers import StratifiedAnchorSampler
from src.utils.training_diary import TrainingDiary, env_snapshot

if sys.platform != "win32":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


@dataclass
class DatasetSplit:
    train: list
    val: list
    train_anchors: int
    train_physics: int


def _resolve_t1_dataset_tier(explorer: Optional[T1ExplorerConfig] = None) -> str:
    if explorer is not None:
        tier = str(getattr(explorer, "dataset_tier", "")).strip()
        if tier:
            return tier
    return os.environ.get("TIER1_DATASET_TIER", "tier1").strip() or "tier1"


def _load_t1_dataset(explorer: Optional[T1ExplorerConfig] = None):
    dataset_tier = _resolve_t1_dataset_tier(explorer)
    cfg = VesselConfig(tier=dataset_tier)
    if not cfg.graph_output_dir.exists():
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


def _compute_step_loss_t1(
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
    lambda_cont_override: Optional[float] = None,
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

    if lambda_cont_override is not None:
        lambda_cont = float(lambda_cont_override)
    else:
        lambda_cont = float(explorer.lambda_cont) if explorer is not None else 1.0
    loss_weight_mode = (explorer.loss_weight_mode if explorer is not None else "dynamic").strip().lower()

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
            p_grad_loss = torch.tensor(0.0, device=device)

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
            g_data = torch.autograd.grad(l_data_kine, ref_param, retain_graph=True, allow_unused=True)[0]
            g_pde = torch.autograd.grad(pde_term, ref_param, retain_graph=True, allow_unused=True)[0]
            g_data_norm = torch.linalg.vector_norm(g_data) if g_data is not None else torch.tensor(0.0, device=device)
            g_pde_norm = torch.linalg.vector_norm(g_pde) if g_pde is not None else torch.tensor(0.0, device=device)
            ratio = torch.clamp(g_data_norm / (g_pde_norm + 1e-8), min=0.1, max=10.0).detach()
            weighted_pdes = lambda_phys * ratio * pde_term
        weighted_data = sum(data_terms)
    else:
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

    loss = weighted_pdes + weighted_data + (0.1 * jac_loss)

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


def _env_truthy(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def _load_tier1_bootstrap(model: GINO_DEQ, tier1_path: Path, device: str) -> bool:
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
        print(f"ℹ️ load_state_dict(strict=False): {len(missing)} missing keys, {len(unexpected)} unexpected keys.")
    print(f"✅ Loaded Tier 1 bootstrap weights from {tier1_path.name}")
    return True


def _assert_tier2_train_split(train_data: list, val_data: list) -> None:
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


def _load_t2_dataset():
    cfg = VesselConfig(tier="tier2")
    data_dir = cfg.graph_output_dir
    if not data_dir.exists():
        print(f"Directory not found: {data_dir}. Please generate Tier 2 data first.")
        return []
    file_list = sorted(list(data_dir.glob("vessel_*.pt")))
    dataset = []
    print(f"📂 Loading {len(file_list)} Tier 2 graphs from {data_dir}...")
    for f in tqdm(file_list):
        dataset.append(torch.load(f, weights_only=False))
    return dataset


def _setup_distillation_phase(model):
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


def _setup_coupled_phase(model, loss_weighter, base_lr=1e-4):
    print("🔥 Unfreezing All Layers. Activating Coupled DEQ Optimization.")
    for param in model.parameters():
        param.requires_grad = True
    return optim.AdamW(
        [
            {"params": model.parameters(), "lr": base_lr},
            {"params": loss_weighter.parameters(), "lr": 1e-3, "weight_decay": 0.0},
        ],
        weight_decay=1e-5,
    )


def _tier2_dynamic_loss_weighter(device: str, mom_precision_floor: float) -> DynamicLossWeighter:
    floor = max(float(mom_precision_floor), 1e-6)
    max_lv_mom = float(-math.log(floor))
    return DynamicLossWeighter(num_losses=1, max_log_var=[max_lv_mom]).to(device)


def _compute_step_loss_t2(
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
        "L_wss": l_wss.item(),
    }
    return loss, metrics


class DEQPredictorTrainer:
    """Centralized trainer for Tier 1 and Tier 2 predictor training loops."""

    def __init__(self, seed: int = 42, train_ratio: float = 0.9):
        self.seed = int(seed)
        self.train_ratio = float(train_ratio)

    def split_anchor_physics(self, dataset: Sequence) -> DatasetSplit:
        anchors = [d for d in dataset if d.is_anchor.any().item()]
        physics = [d for d in dataset if not d.is_anchor.any().item()]
        rng = random.Random(self.seed)
        rng.shuffle(anchors)
        rng.shuffle(physics)
        split_idx_a = int(self.train_ratio * len(anchors))
        split_idx_p = int(self.train_ratio * len(physics))
        train_data = anchors[:split_idx_a] + physics[:split_idx_p]
        val_data = anchors[split_idx_a:] + physics[split_idx_p:]
        n_train_anchors = len([d for d in train_data if d.is_anchor.any().item()])
        return DatasetSplit(
            train=train_data,
            val=val_data,
            train_anchors=n_train_anchors,
            train_physics=max(0, len(train_data) - n_train_anchors),
        )

    def train_t1_predictor(
        self,
        epochs: Optional[int] = None,
        lr: float = 1e-4,
        warm_up_epochs: Optional[int] = None,
        adam_epochs: Optional[int] = None,
        explorer: Optional[T1ExplorerConfig] = None,
    ):
        explorer = explorer or T1ExplorerConfig.from_env()
        dataset_tier = _resolve_t1_dataset_tier(explorer)
        dataset_cfg = VesselConfig(tier=dataset_tier)
        if epochs is None:
            epochs = max(1, int(os.environ.get("TIER1_EPOCHS", "60")))
        if adam_epochs is None:
            raw_adam = os.environ.get("TIER1_ADAM_EPOCHS", "").strip()
            adam_epochs = int(raw_adam) if raw_adam else min(40, int(epochs))
        adam_epochs = max(1, min(int(adam_epochs), int(epochs)))
        if warm_up_epochs is None:
            raw_w = os.environ.get("TIER1_WARM_UP_EPOCHS", "").strip()
            warm_up_epochs = int(raw_w) if raw_w else min(10, max(1, adam_epochs // 3))
        warm_up_epochs = max(0, min(int(warm_up_epochs), adam_epochs - 1))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Device being used:", device)
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

        loss_weighter = DynamicLossWeighter(num_losses=2).to(device) if explorer.loss_weight_mode == "dynamic" else None
        opt_params = list(model.parameters()) + (list(loss_weighter.parameters()) if loss_weighter is not None else [])
        optimizer = optim.AdamW(opt_params, lr=lr, weight_decay=1e-5)

        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warm_up_epochs)
        decay_epochs = adam_epochs - warm_up_epochs
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=decay_epochs, eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warm_up_epochs])
        plateau_scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, threshold=5e-4, min_lr=1e-6
        )

        def _safe_float(v, default=float("nan")) -> float:
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        def _fmt_fixed(v, digits=4, nan_text="nan") -> str:
            vv = _safe_float(v)
            if math.isnan(vv):
                return nan_text
            return f"{vv:.{digits}f}"

        def _fmt_sci(v, digits=3, nan_text="nan") -> str:
            vv = _safe_float(v)
            if math.isnan(vv):
                return nan_text
            return f"{vv:.{digits}e}"

        def _print_t1_validation_block(scores: dict, n_anchor_val: int, n_phys_val: int) -> None:
            rel_l2 = _safe_float(scores.get("rel_l2"))
            rel_l2_std = _safe_float(scores.get("rel_l2_std"))
            rel_l2_p90 = _safe_float(scores.get("rel_l2_p90"))
            rel_u = _safe_float(scores.get("rel_l2_u"))
            rel_v = _safe_float(scores.get("rel_l2_v"))
            rel_p = _safe_float(scores.get("rel_l2_p"))
            cont = _safe_float(scores.get("continuity"))
            cont_p90 = _safe_float(scores.get("continuity_p90"))
            wall = _safe_float(scores.get("wall_slip"))
            wall_p90 = _safe_float(scores.get("wall_slip_p90"))
            shear = _safe_float(scores.get("shear_mse"))
            shear_p90 = _safe_float(scores.get("shear_mse_p90"))
            rel_near = _safe_float(scores.get("rel_l2_near_wall"))
            rel_near_p90 = _safe_float(scores.get("rel_l2_near_wall_p90"))
            rel_grad = _safe_float(scores.get("rel_l2_high_sdf_grad"))
            rel_grad_p90 = _safe_float(scores.get("rel_l2_high_sdf_grad_p90"))
            val_anchor_batches = int(_safe_float(scores.get("val_anchor_batches"), 0.0))
            val_total_batches = int(_safe_float(scores.get("val_total_batches"), 0.0))

            tqdm.write(
                f"\n📊 [Validation] Rel L2 (anchor): {_fmt_fixed(rel_l2)} "
                f"(σ {_fmt_fixed(rel_l2_std)}, p90 {_fmt_fixed(rel_l2_p90)})"
            )
            tqdm.write(
                f"   Components u / v / p: {_fmt_fixed(rel_u)} / {_fmt_fixed(rel_v)} / {_fmt_fixed(rel_p)}"
            )
            tqdm.write(
                f"   |∇·u| mean (fluid interior): {_fmt_sci(cont)} (p90 {_fmt_sci(cont_p90)}) | "
                f"Wall |u| mean: {_fmt_fixed(wall)} (p90 {_fmt_fixed(wall_p90)})"
            )
            tqdm.write(
                f"   γ̇ MSE (anchor): {_fmt_sci(shear)} (p90 {_fmt_sci(shear_p90)}) | "
                f"Batches w/ anchors: {val_anchor_batches}/{val_total_batches}"
            )
            tqdm.write(
                f"   Rel L2 near-wall: {_fmt_fixed(rel_near)} (p90 {_fmt_fixed(rel_near_p90)}) | "
                f"Rel L2 high-|∇SDF|: {_fmt_fixed(rel_grad)} (p90 {_fmt_fixed(rel_grad_p90)})"
            )
            tqdm.write(f"📌 Val split: anchors={n_anchor_val} | physics={n_phys_val}")

        dataset = _load_t1_dataset(explorer)
        if not dataset:
            return {
                "status": "no_data",
                "experiment_name": explorer.experiment_name,
                "dataset_tier": dataset_tier,
                "graph_dir": str(dataset_cfg.graph_output_dir),
            }
        split = self.split_anchor_physics(dataset)
        train_data = split.train
        val_data = split.val

        micro_batch_size = int(os.environ.get("TIER1_MICRO_BATCH_SIZE", "2"))
        accumulation_steps = int(os.environ.get("TIER1_ACCUMULATION_STEPS", "4"))
        target_anchor_fraction = min(max(float(os.environ.get("TIER1_TARGET_ANCHOR_FRACTION", "0.5")), 0.0), 1.0)
        hard_alpha = max(float(os.environ.get("TIER1_HARD_MINING_ALPHA", "0.8")), 0.0)
        hard_refresh = max(int(os.environ.get("TIER1_HARD_MINING_REFRESH_EPOCHS", "4")), 1)
        boundary_data_weight = max(float(os.environ.get("TIER1_BOUNDARY_DATA_WEIGHT", "2.0")), 1.0)
        tier1_kine_p_weight = max(float(os.environ.get("TIER1_KINE_P_WEIGHT", "1.0")), 0.0)
        use_lbfgs = _env_truthy("TIER1_USE_LBFGS")
        dynamic_freeze_during_warmup = _env_truthy("TIER1_DYNAMIC_FREEZE_DURING_WARMUP", "1")
        n_anchor_train = split.train_anchors
        n_phys_train = split.train_physics
        n_anchor_val = len([d for d in val_data if d.is_anchor.any().item()])
        n_phys_val = max(0, len(val_data) - n_anchor_val)
        hard_anchor_multiplier = {}

        print(
            f"⏱️ epochs={epochs} | warm_up={warm_up_epochs} | adam_phase={adam_epochs} "
            "(set TIER1_EPOCHS / TIER1_WARM_UP_EPOCHS / TIER1_ADAM_EPOCHS to override)"
        )
        print(
            f"🔬 Explorer: name='{explorer.experiment_name}' | "
            f"kine_weight={explorer.kine_weight_mode} | "
            f"latent={explorer.latent_dim} | deq_iters={explorer.deq_max_iters} | "
            f"kinematics_mode={explorer.kinematics_mode} | ns_derivatives={explorer.ns_derivative_mode} | "
            f"act={explorer.activation_fn} | fourier_base={explorer.fourier_base:.2f} | "
            f"loss_weight={explorer.loss_weight_mode} | anderson_beta={explorer.anderson_beta:.2f} | "
            f"lambda_cont={explorer.lambda_cont:.2f} | re_curriculum={bool(explorer.re_curriculum)} | "
            f"p_grad_sup={explorer.p_grad_supervision:.3f}"
        )
        if n_anchor_train > 0 and n_phys_train > 0:
            print(
                f"🎯 Weighted sampling enabled (target anchor fraction ~{target_anchor_fraction:.2f}; "
                f"anchors={n_anchor_train}, physics={n_phys_train})."
            )

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
                        sample_weights.append(w_anchor * float(hard_anchor_multiplier.get(gkey, 1.0)))
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
        val_loader = DataLoader(val_data, batch_size=micro_batch_size, shuffle=False)

        best_phys_score = float("inf")
        best_loss = float("inf")
        # Allow sweep runner to redirect checkpoints into a per-candidate directory.
        ckpt_dir_override = os.environ.get("TIER1_CKPT_DIR", "").strip()
        if ckpt_dir_override:
            model_dir = Path(ckpt_dir_override)
            model_dir.mkdir(parents=True, exist_ok=True)
        else:
            model_dir = stage_a_dir()
        lbfgs_initialized = False
        start_epoch = 0
        latest_ckpt_save = model_dir / "tier1_latest_checkpoint.pth"
        best_physics_path = resolve_checkpoint("a", "tier1_best_physics.pth")
        latest_ckpt_path = model_dir / "tier1_latest_checkpoint.pth"
        ckpt_every = max(1, int(os.environ.get("TIER1_CKPT_EVERY", "1")))
        disable_stage_a_artifacts = _env_truthy("TIER1_DISABLE_STAGE_A_ARTIFACTS")
        resume_enabled = _env_truthy("TIER1_RESUME")
        init_from_best_enabled = _env_truthy("TIER1_INIT_FROM_BEST")

        if init_from_best_enabled and best_physics_path.exists():
            best_state = torch.load(best_physics_path, map_location=device, weights_only=True)
            model.load_state_dict(best_state, strict=False)

        if resume_enabled and latest_ckpt_path.exists():
            ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            if loss_weighter is not None and ckpt.get("loss_weighter_state_dict") is not None:
                try:
                    loss_weighter.load_state_dict(ckpt["loss_weighter_state_dict"])
                except RuntimeError:
                    print("ℹ️ Reinitializing Tier 1 PDE loss weighter for current setup.")
            start_epoch = int(ckpt.get("epoch", -1)) + 1
            best_phys_score = float(ckpt.get("best_phys_score", best_phys_score))
            best_loss = float(ckpt.get("best_loss", best_loss))

        best_rel_l2 = float("inf")
        plateau_patience = int(os.environ.get("TIER1_EARLY_STOP_PATIENCE", "10"))
        plateau_min_delta = float(os.environ.get("TIER1_EARLY_STOP_MIN_DELTA", "0.005"))
        val_no_improve = 0
        early_stopped = False

        diary = TrainingDiary("tier1")
        run_end_emitted = False
        last_epoch_completed: Optional[int] = None

        def _emit_tier1_run_end(interrupted: bool = False) -> None:
            nonlocal run_end_emitted
            if run_end_emitted or not diary.enabled:
                return
            run_end_emitted = True
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
                    dd = d.clone().to(device)
                    out = model(dd, solver="anderson", anderson_beta=float(explorer.anderson_beta), anderson_warmup_iters=5)
                    pred = out[0] if isinstance(out, tuple) else out
                    mask = anchor_node_mask(dd)
                    if mask is None or int(mask.sum().item()) == 0:
                        continue
                    rel = torch.norm(pred[mask, :2] - dd.y[mask, :2]) / torch.clamp(torch.norm(dd.y[mask, :2]), min=1e-8)
                    rows.append((_graph_sampling_key(dd, gi), float(rel.item())))
            if rows:
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
            model.train() if not lbfgs_initialized else model.eval()
            physics_active = epoch >= warm_up_epochs
            lambda_phys = min(1.0, max(0.0, (epoch - warm_up_epochs) / 20.0))
            lambda_cont_epoch = float(explorer.lambda_cont)
            total_loss_epoch = 0.0
            current_solver = "picard" if epoch < 5 else "anderson"

            if use_lbfgs and epoch >= adam_epochs and not lbfgs_initialized:
                if loss_weighter is not None:
                    loss_weighter.requires_grad_(False)
                optimizer = optim.LBFGS(
                    [p for p in model.parameters() if p.requires_grad],
                    lr=0.01,
                    max_iter=20,
                    history_size=30,
                    line_search_fn="strong_wolfe",
                    tolerance_grad=1e-6,
                    tolerance_change=1e-8,
                )
                lbfgs_initialized = True

            if not lbfgs_initialized:
                optimizer.zero_grad()
                accum_counter = 0
                pbar = tqdm(loader, desc=f"Tier 1 Epoch {epoch:02d} [Re={phys_cfg.re_target}] (AdamW)")
                for batch_idx, data in enumerate(pbar):
                    data = data.to(device)
                    loss, metrics = _compute_step_loss_t1(
                        model,
                        data,
                        kernels,
                        loss_weighter,
                        current_solver,
                        lambda_phys,
                        device,
                        boundary_data_weight=boundary_data_weight,
                        explorer=explorer,
                        tier1_kine_p_weight=tier1_kine_p_weight,
                        re_scale=1.0,
                        train_loss_weighter=not (dynamic_freeze_during_warmup and epoch < warm_up_epochs),
                        lambda_cont_override=lambda_cont_epoch,
                    )
                    accum_counter += 1
                    loss = loss / float(accumulation_steps)
                    if torch.isnan(loss):
                        accum_counter = max(0, accum_counter - 1)
                        continue
                    loss.backward()
                    if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(loader)):
                        if accum_counter > 0 and accum_counter < accumulation_steps:
                            scale = float(accumulation_steps) / float(accum_counter)
                            for p in opt_params:
                                if p.grad is not None:
                                    p.grad.mul_(scale)
                        torch.nn.utils.clip_grad_norm_(opt_params, max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        accum_counter = 0
                    total_loss_epoch += (loss.item() * accumulation_steps)
                    current_lr = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else float("nan")
                    pbar.set_postfix(
                        {
                            "L_tot": f"{(loss.item() * accumulation_steps):.3f}",
                            "L_data": f"{metrics['L_data']:.3f}",
                            "L_mom": f"{metrics['L_mom']:.3f}",
                            "L_cont": f"{metrics['L_cont']:.3f}",
                            "L_bc": f"{metrics['L_bc']:.3f}",
                            "L_io": f"{metrics['L_io']:.3f}",
                            "L_wss": f"{metrics['L_wss']:.3f}",
                            "L_pgrad": f"{metrics['L_pgrad']:.3f}",
                            "L_jac": f"{metrics['L_jac']:.3f}",
                            "LR": f"{current_lr:.2e}",
                        }
                    )
                scheduler.step()
            else:
                epoch_batches_cpu = list(loader)
                n_batches = max(len(epoch_batches_cpu), 1)

                def closure():
                    optimizer.zero_grad()
                    accumulated_loss = torch.tensor(0.0, device=device)
                    for closure_data_cpu in epoch_batches_cpu:
                        closure_data = closure_data_cpu.to(device)
                        loss, _ = _compute_step_loss_t1(
                            model,
                            closure_data,
                            kernels,
                            loss_weighter,
                            current_solver,
                            lambda_phys,
                            device,
                            boundary_data_weight=boundary_data_weight,
                            explorer=explorer,
                            tier1_kine_p_weight=tier1_kine_p_weight,
                            re_scale=1.0,
                            train_loss_weighter=not (dynamic_freeze_during_warmup and epoch < warm_up_epochs),
                            lambda_cont_override=lambda_cont_epoch,
                        )
                        loss = loss / n_batches
                        loss.backward()
                        accumulated_loss += loss.detach()
                    return accumulated_loss

                loss_tensor = optimizer.step(closure)
                total_loss_epoch = loss_tensor.item() * n_batches

            avg_loss = total_loss_epoch / max(1, len(loader))
            if avg_loss < best_loss and physics_active:
                best_loss = avg_loss
                if not disable_stage_a_artifacts:
                    torch.save(model.state_dict(), model_dir / "tier1_best_loss.pth")

            if epoch % 2 == 0:
                scores = quantify_performance(model, val_loader, kernels, device, tier="tier1")
                if not lbfgs_initialized:
                    plateau_scheduler.step(float(scores.get("rel_l2", 0)))
                _print_t1_validation_block(scores, n_anchor_val=n_anchor_val, n_phys_val=n_phys_val)
                if loss_weighter is not None and explorer.loss_weight_mode == "dynamic":
                    safe_lvs = loss_weighter.clamped_log_vars().detach().cpu()
                    if safe_lvs.numel() >= 2:
                        mom_w = float(torch.exp(-safe_lvs[0]).item())
                        cont_w = float(torch.exp(-safe_lvs[1]).item())
                        tqdm.write(
                            f"⚖️ Learned PDE Weights -> Cont: {cont_w:.2f} | Mom: {mom_w:.2f}"
                        )
                rel_l2 = float(scores.get("rel_l2", float("inf")))
                if (best_rel_l2 - rel_l2) > plateau_min_delta:
                    best_rel_l2 = rel_l2
                    val_no_improve = 0
                else:
                    val_no_improve += 1
                    if val_no_improve >= plateau_patience:
                        early_stopped = True
                        break
                phys_score = scores.get("rel_l2", 0) + (100.0 * scores.get("continuity", 0))
                if phys_score < best_phys_score and physics_active:
                    best_phys_score = phys_score
                    if not disable_stage_a_artifacts:
                        torch.save(model.state_dict(), model_dir / "tier1_best_physics.pth")

            if ((epoch + 1) % ckpt_every == 0) or (epoch == epochs - 1):
                if not disable_stage_a_artifacts:
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

        _emit_tier1_run_end(interrupted=False)
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
                graph_dir=str(dataset_cfg.graph_output_dir),
            )
        return {
            "status": "ok",
            "experiment_name": explorer.experiment_name,
            "dataset_tier": dataset_tier,
            "best_rel_l2": float(best_rel_l2),
            "best_phys_score": float(best_phys_score),
            "best_loss": float(best_loss),
            "early_stopped": bool(early_stopped),
            "n_graphs": int(len(dataset)),
            "n_train": int(len(train_data)),
            "n_val": int(len(val_data)),
            "graph_dir": str(dataset_cfg.graph_output_dir),
        }

    def train_t2_predictor(self, epochs=80, distillation_epochs=12, adam_epochs=50, lr=1e-4):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        phys_cfg = PhysicsConfig(tier="tier2")
        kernels = PhysicsKernels(phys_cfg=phys_cfg)
        model = GINO_DEQ(
            in_channels=15,
            out_channels=5,
            latent_dim=64,
            max_iters=15,
            phys_cfg=phys_cfg,
        ).to(device)

        model_dir = stage_a_dir()
        tier1_path = resolve_checkpoint("a", "tier1_best_physics.pth")
        latest_ckpt_path = resolve_checkpoint("a", "tier2_latest_checkpoint.pth")
        latest_ckpt_save = model_dir / "tier2_latest_checkpoint.pth"
        resume_training = _env_truthy("TIER2_RESUME")
        mom_precision_floor = float(os.environ.get("TIER2_MOM_PRECISION_FLOOR", "0.8"))
        loss_weighter = _tier2_dynamic_loss_weighter(device, mom_precision_floor)

        dataset = _load_t2_dataset()
        if not dataset:
            return
        split = self.split_anchor_physics(dataset)
        train_data = split.train
        val_data = split.val
        _assert_tier2_train_split(train_data, val_data)

        micro_batch_size = 2
        accumulation_steps = 4
        sampler = StratifiedAnchorSampler(train_data, batch_size=micro_batch_size)
        loader = DataLoader(train_data, batch_size=micro_batch_size, sampler=sampler)
        val_loader = DataLoader(val_data, batch_size=micro_batch_size, shuffle=False)

        best_phys_score = float("inf")
        best_loss = float("inf")
        optimizer = None
        scheduler = None
        lbfgs_initialized = False
        use_lbfgs = _env_truthy("TIER2_USE_LBFGS")
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

        if resume_training and latest_ckpt_path.is_file():
            ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            try:
                loss_weighter.load_state_dict(ckpt["loss_weighter_state_dict"])
            except RuntimeError:
                pass
            best_phys_score = float(ckpt.get("best_phys_score", best_phys_score))
            best_loss = float(ckpt.get("best_loss", best_loss))
            start_epoch = int(ckpt.get("epoch", -1)) + 1
            resumed_full_checkpoint = True
        else:
            _load_tier1_bootstrap(model, tier1_path, device)

        if start_epoch >= distillation_epochs:
            optimizer = _setup_coupled_phase(model, loss_weighter, base_lr=lr)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=tier2_cosine_eta_min)
            plateau_scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=3, threshold=5e-4, min_lr=tier2_plateau_min_lr
            )
            sampler.set_warmup_mode(False)
        else:
            optimizer = _setup_distillation_phase(model)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-5)
            plateau_scheduler = None
            sampler.set_warmup_mode(True)

        best_rel_l2 = float("inf")
        plateau_patience = max(1, int(os.environ.get("TIER2_EARLY_STOP_PATIENCE", "8")))
        plateau_min_delta = float(os.environ.get("TIER2_EARLY_STOP_MIN_DELTA", "0.002"))
        flat_eps = float(os.environ.get("TIER2_EARLY_STOP_FLAT_EPS", "3e-4"))
        flat_patience = max(1, int(os.environ.get("TIER2_EARLY_STOP_FLAT_PATIENCE", "6")))
        val_no_improve = 0
        prev_val_rel_l2: Optional[float] = None
        flat_streak = 0
        early_stopped = False
        run_end_emitted = False
        saved_final_weights = False
        last_epoch_completed: Optional[int] = None
        tier2_final_path = model_dir / "tier2_final.pth"
        diary = TrainingDiary("tier2")

        def _emit_tier2_run_end(interrupted: bool = False) -> None:
            nonlocal run_end_emitted
            if run_end_emitted or not diary.enabled:
                return
            run_end_emitted = True
            diary.log_run_end(
                best_phys_score=float(best_phys_score),
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
            is_distillation = epoch < distillation_epochs
            physics_active = not is_distillation
            lambda_phys = min(1.0, max(0.0, (epoch - distillation_epochs) / 20.0))
            if is_distillation:
                progress = epoch / max(1, (distillation_epochs - 1))
                carreau_n = start_n - progress * (start_n - target_n)
            else:
                carreau_n = target_n
            current_solver = "picard" if is_distillation else "anderson"

            if epoch == 0 and not resumed_full_checkpoint:
                optimizer = _setup_distillation_phase(model)
                scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-5)
                plateau_scheduler = None
                sampler.set_warmup_mode(True)
            elif epoch == distillation_epochs:
                optimizer = _setup_coupled_phase(model, loss_weighter, base_lr=lr)
                scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=tier2_cosine_eta_min)
                plateau_scheduler = ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=3, threshold=5e-4, min_lr=tier2_plateau_min_lr
                )
                sampler.set_warmup_mode(False)
            elif use_lbfgs and epoch >= adam_epochs and not lbfgs_initialized:
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
            if not lbfgs_initialized:
                optimizer.zero_grad()
                pbar = tqdm(loader, desc=f"Tier 2 Epoch {epoch:02d} [Re={phys_cfg.re_target}]")
                for batch_idx, data in enumerate(pbar):
                    data = data.to(device)
                    loss, _metrics = _compute_step_loss_t2(
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
                        continue
                    loss.backward()
                    if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(loader)):
                        clip_params = [p for g in optimizer.param_groups for p in g["params"]]
                        clip_max = 1.0 if is_distillation else tier2_grad_clip_coupled
                        torch.nn.utils.clip_grad_norm_(clip_params, max_norm=clip_max)
                        optimizer.step()
                        optimizer.zero_grad()
                    total_loss_epoch += (loss.item() * accumulation_steps)
                if scheduler is not None:
                    scheduler.step()
            else:
                epoch_batches_cpu = list(loader)
                n_batches = max(len(epoch_batches_cpu), 1)

                def closure():
                    optimizer.zero_grad()
                    accumulated_loss = torch.tensor(0.0, device=device)
                    for closure_data_cpu in epoch_batches_cpu:
                        closure_data = closure_data_cpu.to(device)
                        loss, _ = _compute_step_loss_t2(
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

            avg_loss = total_loss_epoch / max(1, len(loader))
            if avg_loss < best_loss and physics_active:
                best_loss = avg_loss
                torch.save(model.state_dict(), model_dir / "tier2_best_loss.pth")

            if ((epoch + 1) % ckpt_every == 0) or (epoch == epochs - 1):
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": (scheduler.state_dict() if not lbfgs_initialized and scheduler is not None else None),
                    "loss_weighter_state_dict": loss_weighter.state_dict(),
                    "best_phys_score": best_phys_score,
                    "best_loss": best_loss,
                    "optimizer_type": ("LBFGS" if lbfgs_initialized else "AdamW"),
                }
                torch.save(checkpoint, latest_ckpt_save)

            if epoch % 2 == 0:
                scores = quantify_performance(model, val_loader, kernels, device, tier="tier2")
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
                    if (val_no_improve >= plateau_patience) or (flat_streak >= flat_patience):
                        early_stopped = True
                        break
                phys_score = float(scores.get("rel_l2", 0) + scores.get("continuity", 0) + scores.get("rheology", 0))
                if physics_active and phys_score < best_phys_score:
                    best_phys_score = phys_score
                    torch.save(model.state_dict(), model_dir / "tier2_best_physics.pth")

        torch.save(model.state_dict(), tier2_final_path)
        saved_final_weights = True
        _emit_tier2_run_end(interrupted=False)
        print(f"Tier 2 Training Complete. Best Physical Score: {best_phys_score:.4f} | Best Loss: {best_loss:.4f}")
