"""Train wall-local clot phase (phi) with capped GT mu and GT kinematics.

Usage (from repo root)::

    python -m src.training.train_clot_phi_simple

Checkpoint: ``outputs/biochem/clot_phi_best.pth``
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND, VesselConfig
from src.evaluation.clot_shape_score import compute_clot_shape_metrics
from src.core_physics.clot_forecast import (
    build_clot_forecast_pair_step,
    build_deploy_eligible_phi_gt,
    clot_forecast_deploy_loss_enabled,
    clot_forecast_mask_mode,
    clot_forecast_mu_carry_detach,
    clot_forecast_mu_carry_enabled,
    clot_forecast_one_step_enabled,
    clot_forecast_pair_stride,
    iter_forecast_pairs,
    resolve_forecast_deploy_mask_from_model,
    resolve_rollout_prev_mu_si,
    snapshot_clot_forecast_config,
)
from src.core_physics.clot_phi_simple import (
    ClotPhiSpeciesHead,
    build_clot_phi_model,
    build_clot_phi_step,
    cap_mu_eff_si,
    clot_phi_dropout,
    clot_phi_feature_dim,
    clot_phi_fixed_mu_from_phi_enabled,
    clot_phi_forward_apply_region,
    clot_phi_mesh_aux_lambda,
    clot_phi_mesh_bulk_lambda,
    clot_phi_mlp_depth,
    clot_phi_hybrid_enabled,
    clot_phi_joint_bio_enabled,
    clot_phi_mask_mode,
    clot_phi_minimal_features_enabled,
    clot_phi_model_kind,
    clot_phi_model_uses_mpnn,
    clot_phi_mu_cap_si,
    clot_phi_oracle_mu_enabled,
    clot_phi_physics_oracle_enabled,
    clot_phi_prior_feature_count,
    clot_phi_species_features_enabled,
    clot_phi_thresh_si,
    clot_phi_use_prior_features,
    clot_phi_hard_support_projection_enabled,
    log_blend_mu_eff_si,
    lumen_eligible_mask,
    mu_eff_from_delta_log_si,
    snapshot_clot_support_config,
    snapshot_mesh_aux_config,
    snapshot_phi_only_rollout_config,
    physics_mu_eff_si,
    physics_phi_from_mu,
    rule_phi_from_mu_cap,
)
from src.core_physics.clot_phi_rollout import (
    ClotPhiRolloutState,
    clot_phi_carry_log_mu_enabled,
    clot_phi_carry_phi_enabled,
    clot_phi_rollout_detach_carry,
    clot_phi_rollout_enabled,
    clot_phi_vel_source,
    snapshot_carry_gt_warmup_config,
)
from src.training.clot_trigger_stack import (
    clot_phi_trigger_rollout_enabled,
    forward_trigger_rollout_step,
)
from src.utils.channel_schema import BIO_Y_SCHEMA, assert_graph_schema, build_y_valid_mask, infer_missing_schema
from src.utils.paths import get_project_root


class AnchorFileDataset(Dataset):
    def __init__(self, file_list: List[str]):
        self.file_list = list(file_list)

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int):
        data = torch.load(self.file_list[idx], weights_only=False)
        data = infer_missing_schema(data, phase_hint="biochem")
        assert_graph_schema(data, expected_y_schema=(BIO_Y_SCHEMA,))
        return data


def _list_anchor_paths(root: Path) -> List[str]:
    paths = sorted(str(p) for p in root.glob("*.pt") if p.is_file())
    if not paths:
        raise FileNotFoundError(f"No anchor graphs in {root}")
    return paths


def _split_train_val(paths: List[str], val_stem: str) -> Tuple[List[str], List[str]]:
    val_stem = val_stem.strip().lower()
    val_paths = [p for p in paths if Path(p).stem.lower() == val_stem]
    train_paths = [p for p in paths if Path(p).stem.lower() != val_stem]
    if not val_paths:
        split = max(1, len(paths) // 5)
        return paths[split:], paths[:split]
    if not train_paths:
        train_paths = paths[:]
    return train_paths, val_paths


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _loss_indices(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    mask: torch.Tensor,
    *,
    balanced: bool,
) -> torch.Tensor:
    """Node indices for BCE; optional 1:1 pos/neg subsample inside the supervision mask."""
    idx = mask.nonzero(as_tuple=False).view(-1)
    if not balanced or idx.numel() == 0:
        return idx
    pos = idx[(tgt[idx] > 0.5)]
    neg = idx[(tgt[idx] <= 0.5)]
    if pos.numel() == 0 or neg.numel() == 0:
        return idx
    k = int(min(pos.numel(), neg.numel()))
    pos_pick = pos[torch.randperm(pos.numel(), device=pos.device)[:k]]
    neg_pick = neg[torch.randperm(neg.numel(), device=neg.device)[:k]]
    return torch.cat([pos_pick, neg_pick])


def _dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pb = (pred.reshape(-1) > 0.5).float()
    tb = (target.reshape(-1) > 0.5).float()
    inter = float((pb * tb).sum().item())
    return (2.0 * inter + eps) / (float(pb.sum()) + float(tb.sum()) + eps)


def _state_with_pred_mu(y_slice: torch.Tensor, mu_si: torch.Tensor, phys_cfg: PhysicsConfig) -> torch.Tensor:
    """Build a pseudo rollout state with predicted mu_eff in channel 3 (SI -> ND)."""
    state = y_slice.clone().to(dtype=torch.float32)
    mu_cap = cap_mu_eff_si(mu_si.reshape(-1))
    state[:, STATE_CHANNEL_MU_EFF_ND] = phys_cfg.viscosity_si_to_nd(mu_cap)
    return state


def _accumulate_clot_shape(
    *,
    shape_sums: dict[str, float],
    data,
    phi_pred: torch.Tensor,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    time_index: int,
    t_out: int | None,
    forecast_one_step: bool,
    pair_stride: int,
    t_steps: int,
    fixed_mu_from_phi: bool,
    hybrid: bool,
    mu_pred: torch.Tensor,
) -> None:
    """Update shape_sums with one clot_shape sample (one-step uses mu_c @ t_out)."""
    from src.core_physics.clot_phi_simple import clot_phi_shape_use_t_out_mu

    t_shape = int(time_index)
    if forecast_one_step and clot_phi_shape_use_t_out_mu():
        if t_out is not None:
            t_shape = int(t_out)
        else:
            t_shape = min(int(time_index) + int(pair_stride), int(t_steps) - 1)
    if t_shape < 0 or t_shape >= int(t_steps):
        return
    if clot_phi_hard_support_projection_enabled() and mu_pred is not None:
        mu_shape = mu_pred.reshape(-1)
    elif forecast_one_step and clot_phi_shape_use_t_out_mu() and fixed_mu_from_phi:
        step_out = build_clot_phi_step(data, t_shape, phys_cfg, bio_cfg, device)
        mu_shape = log_blend_mu_eff_si(step_out.mu_c_si, phi_pred)
    elif hybrid:
        mu_shape = mu_pred.reshape(-1)
    elif fixed_mu_from_phi:
        step_t = build_clot_phi_step(data, t_shape, phys_cfg, bio_cfg, device)
        mu_shape = log_blend_mu_eff_si(step_t.mu_c_si, phi_pred)
    else:
        mu_shape = mu_pred.reshape(-1)
    y_sl = data.y[t_shape].to(device=device, dtype=torch.float32)
    pred_state = _state_with_pred_mu(y_sl, mu_shape, phys_cfg)
    sm = compute_clot_shape_metrics(
        pred_state=pred_state,
        gt_state=y_sl,
        edge_index=data.edge_index.to(device),
        phys_cfg=phys_cfg,
    )
    shape_sums["clot_shape"] += float(sm["clot_shape"])
    shape_sums["clot_shape_rec"] += float(sm["clot_recall"])
    shape_sums["clot_shape_pred_frac"] += float(sm["clot_pred_frac"])
    shape_sums["clot_shape_gt_frac"] += float(sm["clot_gt_frac"])


def _maybe_project_deploy_mu(
    *,
    data,
    step,
    mu_pred: torch.Tensor,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    forecast_one_step: bool,
    time_index: int | None = None,
    bulk_time_index: int | None = None,
    phi_pred_by_time: dict[int, torch.Tensor] | None = None,
) -> torch.Tensor:
    from src.core_physics.clot_phi_simple import project_deploy_mu_with_support

    return project_deploy_mu_with_support(
        data=data,
        step=step,
        mu_pred=mu_pred,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        device=device,
        forecast_one_step=forecast_one_step,
        time_index=time_index,
        bulk_time_index=bulk_time_index,
        phi_pred_by_time=phi_pred_by_time,
    )


def _mesh_aux_losses(
    *,
    phi_pred: torch.Tensor,
    data,
    step,
    ti: int,
    t_out: int | None,
    forecast_one_step: bool,
    pair_stride: int,
    t_steps: int,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    use_soft: bool,
) -> torch.Tensor:
    """Full-mesh eligible-lumen aux BCE + bulk phi suppression at forecast target time."""
    aux_w = clot_phi_mesh_aux_lambda()
    bulk_w = clot_phi_mesh_bulk_lambda()
    if aux_w <= 0.0 and bulk_w <= 0.0:
        return torch.tensor(0.0, device=device)
    t_mesh = int(ti)
    if forecast_one_step:
        if t_out is not None:
            t_mesh = int(t_out)
        else:
            t_mesh = min(int(ti) + int(pair_stride), int(t_steps) - 1)
    elig = lumen_eligible_mask(data, device)
    if not bool(elig.any().item()):
        return torch.tensor(0.0, device=device)
    loss = torch.tensor(0.0, device=device)
    if aux_w > 0.0:
        phi_mesh_gt = build_deploy_eligible_phi_gt(
            data,
            t_mesh,
            phys_cfg,
            bio_cfg,
            device,
            use_soft=use_soft,
        )
        idx = elig.nonzero(as_tuple=False).view(-1)
        pm = phi_pred[idx].clamp(1e-6, 1.0 - 1e-6)
        tm = phi_mesh_gt[idx]
        loss = loss + aux_w * F.binary_cross_entropy(pm, tm)
    if bulk_w > 0.0:
        thr = clot_phi_thresh_si(phys_cfg)
        mu_gt_t = step.mu_gt_cap.reshape(-1)
        bulk = elig & (mu_gt_t < thr)
        if bool(bulk.any().item()):
            idx_b = bulk.nonzero(as_tuple=False).view(-1)
            pm = phi_pred[idx_b].clamp(1e-6, 1.0 - 1e-6)
            loss = loss + bulk_w * F.binary_cross_entropy(pm, torch.zeros_like(pm))
    return loss


def _clot_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    """Precision/recall/F1 and positive rate inside the supervision mask."""
    if not bool(mask.any().item()):
        return {
            "clot_prec": 0.0,
            "clot_rec": 0.0,
            "clot_f1": 0.0,
            "pred_pos_frac": 0.0,
            "gt_pos_frac": 0.0,
        }
    pb = (pred[mask] > 0.5).float()
    tb = (target[mask] > 0.5).float()
    tp = float((pb * tb).sum().item())
    fp = float((pb * (1.0 - tb)).sum().item())
    fn = float(((1.0 - pb) * tb).sum().item())
    prec = tp / max(tp + fp, 1e-6)
    rec = tp / max(tp + fn, 1e-6)
    f1 = (2.0 * prec * rec) / max(prec + rec, 1e-6)
    return {
        "clot_prec": prec,
        "clot_rec": rec,
        "clot_f1": f1,
        "pred_pos_frac": float(pb.mean().item()),
        "gt_pos_frac": float(tb.mean().item()),
    }


def _checkpoint_rank_score(va: dict[str, float]) -> float:
    """North-star rank for forecast / fixed-mu legs (always non-negative if f1>0)."""
    shape = float(va.get("clot_shape", 0.0))
    f1 = float(va.get("clot_f1", 0.0))
    return shape * 0.65 + f1 * 0.35


def _checkpoint_score(va: dict[str, float]) -> float:
    """Prefer non-collapsed val F1; penalize predict-none / predict-all / memorization."""
    if _env_bool("CLOT_PHI_REGRESSION_ONLY", False):
        return -float(va.get("mu_log_mae", 99.0))
    if clot_phi_trigger_rollout_enabled():
        scaled = float(
            va.get("full_mesh_f1_scaled", va.get("loss_mask_f1_scaled", float("nan")))
        )
        pp = float(va.get("pred_pos_frac", 0.0))
        gt = float(va.get("gt_pos_frac", 0.0))
        if math.isfinite(scaled):
            if pp > 0.92:
                return -1.0
            if gt < 0.04 and pp > 0.35:
                return -1.0
            return scaled
    if clot_phi_fixed_mu_from_phi_enabled() or clot_forecast_one_step_enabled():
        shape = float(va.get("clot_shape", 0.0))
        f1 = float(va.get("clot_f1", 0.0))
        pp = float(va.get("pred_pos_frac", 0.0))
        gt = float(va.get("gt_pos_frac", 0.0))
        if pp > 0.95:
            return -1.0
        if gt >= 0.05 and pp > 3.5 * gt:
            return -1.0
        if gt < 0.05 and pp > 0.35:
            return -1.0
        return _checkpoint_rank_score(va)
    f1 = float(va.get("clot_f1", 0.0))
    pp = float(va.get("pred_pos_frac", 0.0))
    gt = float(va.get("gt_pos_frac", 0.0))
    rec = float(va.get("clot_rec", 0.0))
    prec = float(va.get("clot_prec", 0.0))
    mae = float(va.get("mu_log_mae", 99.0))
    # Near-zero positive anchors: allow low pp when gt itself is tiny; score mainly from mu/logMAE.
    if gt < 0.04:
        # Treat blatant overprediction as failure.
        if pp > 0.25:
            return -1.0
        mae_bonus = max(0.0, 0.9 - mae) * 0.3
        return max(0.0, f1 * 0.5) + mae_bonus
    if pp < 0.05 or pp > 0.92:
        return -1.0
    if rec > 0.92:
        return -1.0
    if gt > 1e-6 and (pp < 0.35 * gt or pp > 2.5 * gt):
        return f1 * 0.35
    if rec > 0.72 and prec < 0.45:
        return f1 * 0.4
    mae_bonus = max(0.0, 1.2 - mae) * 0.15
    return f1 + mae_bonus


def _bio_lambda() -> float:
    return max(float(os.environ.get("CLOT_PHI_BIO_LAMBDA", "1.0") or "0"), 0.0)


def _physics_blend_alpha() -> float:
    return max(0.0, min(float(os.environ.get("CLOT_PHI_PHYSICS_BLEND_ALPHA", "0.5") or "0.5"), 1.0))


def _regression_only() -> bool:
    return _env_bool("CLOT_PHI_REGRESSION_ONLY", False)


def _freeze_mu_branch(model: torch.nn.Module) -> None:
    """Stage-B: train phi head only; keep learned mu branch fixed."""
    if not _env_bool("CLOT_PHI_FREEZE_MU_BRANCH", False):
        return
    if hasattr(model, "dlog_fc"):
        for p in model.dlog_fc.parameters():
            p.requires_grad = False
    for name, p in model.named_parameters():
        if "phi_fc" not in name:
            p.requires_grad = False


def _load_init_checkpoint(
    model: torch.nn.Module,
    species_head: torch.nn.Module | None,
    path: Path,
    device: torch.device,
) -> None:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if species_head is not None and "species_head_state_dict" in ckpt:
        species_head.load_state_dict(ckpt["species_head_state_dict"], strict=False)


def _species_hidden() -> int:
    return max(int(os.environ.get("CLOT_PHI_SPECIES_HIDDEN", "32")), 8)


def _species_data_mse(
    pred_log: torch.Tensor,
    tgt_log: torch.Tensor,
    idx: torch.Tensor,
    bio_cfg: BiochemConfig,
) -> torch.Tensor:
    """SI-scaled species fit (same spirit as biochem ``L_Data_Bio`` on anchors)."""
    p_log = pred_log[idx].clamp(-10.0, 8.0)
    t_log = tgt_log[idx].clamp(-10.0, 8.0)
    # Log1p-ND MSE (stable); optionally upweight FI/Mat channels in-band.
    fi_w = max(float(os.environ.get("CLOT_PHI_BIO_FI_WEIGHT", "1.0") or "1.0"), 0.0)
    mat_w = max(float(os.environ.get("CLOT_PHI_BIO_MAT_WEIGHT", "1.0") or "1.0"), 0.0)
    if abs(fi_w - 1.0) < 1e-8 and abs(mat_w - 1.0) < 1e-8:
        return F.mse_loss(p_log, t_log)
    w = torch.ones((1, 12), device=p_log.device, dtype=p_log.dtype)
    w[:, 8] = fi_w
    w[:, 11] = mat_w
    return torch.mean(((p_log - t_log) ** 2) * w)


def _model_logits(model: torch.nn.Module, feats: torch.Tensor, edge_index: torch.Tensor | None) -> torch.Tensor:
    if clot_phi_model_uses_mpnn(model):
        if edge_index is None:
            raise ValueError("mpnn model requires edge_index")
        return model.forward_logits(feats, edge_index)
    return model.forward_logits(feats)


def _model_delta_log_mu(model: torch.nn.Module, feats: torch.Tensor, edge_index: torch.Tensor | None) -> torch.Tensor:
    if clot_phi_model_uses_mpnn(model):
        if edge_index is None:
            raise ValueError("mpnn model requires edge_index")
        return model.forward_delta_log_mu(feats, edge_index)
    return model.forward_delta_log_mu(feats)


def _build_step(
    data,
    ti: int,
    *,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    rollout_state: ClotPhiRolloutState | None,
    forecast_state: ClotPhiRolloutState | None,
    t_steps: int,
    pair_stride: int,
    t_out: int | None = None,
    train_epoch: int | None = None,
):
    if clot_forecast_one_step_enabled():
        t_out_i = int(t_out if t_out is not None else ti + pair_stride)
        if t_out_i >= t_steps or t_out_i < 0:
            return None
        return build_clot_forecast_pair_step(
            data,
            ti,
            t_out_i,
            phys_cfg,
            bio_cfg,
            device,
            forecast_state=forecast_state,
            train_epoch=train_epoch,
        )
    return build_clot_phi_step(
        data,
        ti,
        phys_cfg,
        bio_cfg,
        device,
        rollout_state=rollout_state,
        train_epoch=train_epoch,
    )


def _resolve_step_loss_mask(
    step,
    data,
    model: torch.nn.Module | None,
    edge_index: torch.Tensor | None,
    *,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    hybrid: bool,
) -> torch.Tensor:
    """Deploy-faithful forecast band (pred phi @ t_in); static for legacy modes."""
    mode = clot_forecast_mask_mode()
    if mode not in ("deploy_input", "deploy_pred"):
        return step.loss_mask
    if model is None:
        return step.loss_mask
    return resolve_forecast_deploy_mask_from_model(
        data,
        step=step,
        model=model,
        edge_index=edge_index,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        device=device,
        hybrid=hybrid,
    )


def _run_epoch(
    model: torch.nn.Module | None,
    paths: List[str],
    *,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    train: bool,
    time_stride: int,
    pos_weight: float,
    balanced: bool,
    rule_baseline: bool = False,
    physics_oracle: bool = False,
    species_head: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    train_epoch: int | None = None,
) -> dict[str, float]:
    if train:
        if model is None or rule_baseline or physics_oracle:
            raise ValueError("rule/physics oracle cannot train")
        model.train()
        if species_head is not None:
            species_head.train()
    elif model is not None:
        model.eval()
        if species_head is not None:
            species_head.eval()

    bce_sum = 0.0
    mu_mse_sum = 0.0
    bio_mse_sum = 0.0
    dice_sum = 0.0
    log_mae_sum = 0.0
    hybrid = clot_phi_hybrid_enabled()
    mu_log_lambda = max(float(os.environ.get("CLOT_PHI_MU_LOG_LAMBDA", "1.0") or "0"), 0.0)
    bio_lambda = _bio_lambda()
    joint_bio = species_head is not None
    use_soft = _env_bool("CLOT_PHI_SOFT_LABELS", False)
    regression_only = _regression_only()
    metric_sums = {
        "clot_prec": 0.0,
        "clot_rec": 0.0,
        "clot_f1": 0.0,
        "pred_pos_frac": 0.0,
        "gt_pos_frac": 0.0,
        "_full_scaled": 0.0,
    }
    shape_sums = {
        "clot_shape": 0.0,
        "clot_shape_rec": 0.0,
        "clot_shape_pred_frac": 0.0,
        "clot_shape_gt_frac": 0.0,
    }
    n_steps = 0
    n_graphs = 0
    n_shape_graphs = 0
    dice_lambda = max(float(os.environ.get("CLOT_PHI_DICE_LAMBDA", "0") or "0"), 0.0)
    anchor_balanced = _env_bool("CLOT_PHI_ANCHOR_BALANCED", False)
    path_step_scale: dict[str, float] = {}
    if train and anchor_balanced and paths:
        approx_steps = []
        for p in paths:
            d = torch.load(p, map_location="cpu", weights_only=False)
            if hasattr(d, "y") and torch.is_tensor(d.y) and d.y.dim() == 3:
                approx_steps.append(max(int(d.y.shape[0] // max(1, time_stride)), 1))
            else:
                approx_steps.append(1)
        target = float(sum(approx_steps)) / float(max(len(approx_steps), 1))
        for p, n in zip(paths, approx_steps):
            path_step_scale[p] = float(target / float(max(n, 1)))

    pair_stride = clot_forecast_pair_stride()
    forecast_one_step = clot_forecast_one_step_enabled()
    step_train_epoch = train_epoch if train else None

    with torch.set_grad_enabled(train):
        for path in paths:
            data = torch.load(path, weights_only=False).to(device)
            data = infer_missing_schema(data, phase_hint="biochem")
            if hasattr(data, "y") and torch.is_tensor(data.y):
                if hasattr(data, "y_valid_mask") and torch.is_tensor(data.y_valid_mask):
                    if tuple(data.y_valid_mask.shape) != tuple(data.y.shape):
                        data.y_valid_mask = build_y_valid_mask(
                            data.y, data.y_schema, getattr(data, "mask_wall", None)
                        )
            assert_graph_schema(data, expected_y_schema=(BIO_Y_SCHEMA,))
            if not hasattr(data, "y") or data.y.dim() != 3:
                continue
            n_graphs += 1
            t_steps = data.y.shape[0]
            edge_index = data.edge_index.to(device) if model is not None and clot_phi_model_uses_mpnn(model) else None
            trigger_rollout_on = (
                clot_phi_trigger_rollout_enabled()
                and not rule_baseline
                and not physics_oracle
                and not forecast_one_step
                and model is not None
            )
            if trigger_rollout_on:
                phi_prev: torch.Tensor | None = None
                phi_by_t: dict[int, torch.Tensor] = {}
                mu_anchor_si: torch.Tensor | None = None
                for ti in range(0, t_steps, max(1, time_stride)):
                    fwd = forward_trigger_rollout_step(
                        model,
                        data,
                        ti,
                        phys_cfg=phys_cfg,
                        bio_cfg=bio_cfg,
                        device=device,
                        phi_prev=phi_prev,
                        phi_pred_by_time=phi_by_t,
                        mu_anchor_si=mu_anchor_si,
                        edge_index=edge_index,
                        use_soft=use_soft,
                        hard_commit=not train,
                    )
                    step = fwd["step"]
                    mu_anchor_si = fwd["mu_anchor_si"]
                    bundle = fwd["bundle"]
                    phi_pred = fwd["phi"]
                    phi_loss = bundle["phi_hybrid"]
                    tgt = step.phi_gt
                    m = step.loss_mask
                    if not bool(m.any().item()):
                        phi_by_t[int(ti)] = phi_pred.detach()
                        phi_prev = phi_pred.detach()
                        continue
                    mu_pred = bundle["mu_hybrid"]
                    if train:
                        if optimizer is None:
                            raise ValueError("optimizer is required when train=True")
                        optimizer.zero_grad(set_to_none=True)
                    idx = m.nonzero(as_tuple=False).view(-1)
                    pm = phi_loss[idx].clamp(1e-6, 1.0 - 1e-6)
                    tm = tgt[idx]
                    pw = torch.tensor([pos_weight], device=device, dtype=pm.dtype)
                    if regression_only:
                        bce = torch.tensor(0.0, device=device)
                    else:
                        bce_n = F.binary_cross_entropy(pm, tm, reduction="none")
                        w = torch.where(tm > 0.5, pw, torch.ones_like(tm))
                        bce = (bce_n * w).mean()
                    mu_mse = torch.tensor(0.0, device=device)
                    if hybrid and mu_log_lambda > 0.0:
                        log_mu_p = torch.log(mu_pred.clamp(min=1e-8))
                        log_mu_t = torch.log(step.mu_gt_cap.clamp(min=1e-8))
                        mu_mse = F.mse_loss(log_mu_p[idx], log_mu_t[idx])
                    if train and (not regression_only) and dice_lambda > 0.0:
                        inter = (pm * tm).sum()
                        dice_loss = 1.0 - (2.0 * inter + 1e-6) / (pm.sum() + tm.sum() + 1e-6)
                        loss = bce + dice_lambda * dice_loss + mu_log_lambda * mu_mse
                    else:
                        loss = (
                            mu_log_lambda * mu_mse
                            if regression_only
                            else bce + mu_log_lambda * mu_mse
                        )
                    if train:
                        loss = loss * float(path_step_scale.get(path, 1.0))
                        loss.backward()
                        optimizer.step()
                    with torch.no_grad():
                        log_mae = (
                            torch.log(mu_pred[m].clamp(min=1e-8))
                            - torch.log(step.mu_gt_cap[m].clamp(min=1e-8))
                        ).abs().mean()
                        dice_sum += _dice_score(phi_pred[m], tgt[m])
                        log_mae_sum += float(log_mae.item())
                        bce_sum += float(bce.item())
                        mu_mse_sum += float(mu_mse.item())
                        from src.evaluation.clot_trigger_metrics import (
                            clot_trigger_step_metrics,
                        )

                        step_m = clot_trigger_step_metrics(
                            phi_pred,
                            tgt,
                            data=data,
                            device=device,
                            bio_cfg=bio_cfg,
                            loss_mask=m,
                        )
                        loss_m = _clot_metrics(phi_pred.reshape(-1), tgt.reshape(-1), m)
                        for k in ("clot_prec", "clot_rec", "clot_f1", "pred_pos_frac", "gt_pos_frac"):
                            metric_sums[k] += float(loss_m[k])
                        scaled = step_m.get("loss_mask_f1_scaled", step_m.get("ceiling_f1_scaled"))
                        metric_sums["_full_scaled"] += float(
                            scaled if scaled is not None else float(loss_m["clot_f1"])
                        )
                        n_steps += 1
                    phi_by_t[int(ti)] = phi_pred.detach()
                    phi_prev = phi_pred.detach()
                continue
            rollout_on = (
                clot_phi_rollout_enabled()
                and not rule_baseline
                and not physics_oracle
                and not forecast_one_step
            )
            rollout_state = ClotPhiRolloutState() if rollout_on else None
            forecast_carry_on = forecast_one_step and clot_forecast_mu_carry_enabled()
            forecast_state = ClotPhiRolloutState() if forecast_carry_on else None
            if forecast_one_step:
                pair_list = iter_forecast_pairs(
                    t_steps, time_stride=max(1, time_stride), pair_stride=pair_stride
                )
            else:
                pair_list = [(ti, ti) for ti in range(0, t_steps, max(1, time_stride))]
            last_mu_pred_si: torch.Tensor | None = None
            last_gt_y: torch.Tensor | None = None
            for ti, t_out in pair_list:
                step = _build_step(
                    data,
                    ti,
                    phys_cfg=phys_cfg,
                    bio_cfg=bio_cfg,
                    device=device,
                    rollout_state=rollout_state,
                    forecast_state=forecast_state,
                    t_steps=t_steps,
                    pair_stride=pair_stride,
                    t_out=t_out if forecast_one_step else None,
                    train_epoch=step_train_epoch,
                )
                if step is None:
                    continue
                tgt = step.phi_gt
                if clot_forecast_deploy_loss_enabled() and not forecast_one_step:
                    mu_in = resolve_rollout_prev_mu_si(
                        rollout_state,
                        step,
                        device,
                        time_index=ti,
                        train_epoch=step_train_epoch,
                    )
                    step = replace(
                        step,
                        mu_in_cap=mu_in,
                        phi_in_gt=step.phi_gt,
                    )
                    tgt = build_deploy_eligible_phi_gt(
                        data,
                        ti,
                        phys_cfg,
                        bio_cfg,
                        device,
                        use_soft=use_soft,
                    )
                m = _resolve_step_loss_mask(
                    step,
                    data,
                    model,
                    edge_index,
                    phys_cfg=phys_cfg,
                    bio_cfg=bio_cfg,
                    device=device,
                    hybrid=hybrid,
                )
                if not bool(m.any().item()):
                    continue
                if rule_baseline:
                    phi_pred = rule_phi_from_mu_cap(step.mu_gt_cap, step.region, phys_cfg)
                    mu_pred = log_blend_mu_eff_si(step.mu_c_si, phi_pred)
                    idx = m.nonzero(as_tuple=False).view(-1)
                    pm = phi_pred[idx].clamp(1e-6, 1.0 - 1e-6)
                    tm = tgt[idx]
                    bce = F.binary_cross_entropy(pm, tm)
                    mu_mse = torch.tensor(0.0, device=device)
                    bio_mse = torch.tensor(0.0, device=device)
                elif physics_oracle:
                    y_sl = data.y[ti].to(device=device)
                    mu_pred = cap_mu_eff_si(
                        physics_mu_eff_si(
                            step.mu_c_si,
                            step.species_log_gt,
                            bio_cfg,
                            device=device,
                            data=data,
                            u_nd=y_sl[:, 0],
                            v_nd=y_sl[:, 1],
                        )
                    )
                    phi_pred = physics_phi_from_mu(
                        mu_pred, step.mu_c_si, step.region, phys_cfg, soft=use_soft
                    )
                    idx = m.nonzero(as_tuple=False).view(-1)
                    pm = phi_pred[idx].clamp(1e-6, 1.0 - 1e-6)
                    tm = tgt[idx]
                    bce = F.binary_cross_entropy(pm, tm)
                    mu_mse = F.mse_loss(
                        torch.log(mu_pred[idx].clamp(min=1e-8)),
                        torch.log(step.mu_gt_cap[idx].clamp(min=1e-8)),
                    )
                    bio_mse = torch.tensor(0.0, device=device)
                else:
                    assert model is not None
                    if train:
                        if optimizer is None:
                            raise ValueError("optimizer is required when train=True")
                        optimizer.zero_grad(set_to_none=True)
                    feats = step.features
                    bio_mse = torch.tensor(0.0, device=device)
                    sp_for_physics = step.species_log_gt
                    if joint_bio:
                        assert species_head is not None
                        sp_pred = species_head(feats).clamp(-10.0, 8.0)
                        idx_all = m.nonzero(as_tuple=False).view(-1)
                        bio_mse = _species_data_mse(
                            sp_pred, step.species_log_gt, idx_all, bio_cfg
                        )
                        if _env_bool("CLOT_PHI_JOINT_USE_PRED_SPECIES", True):
                            sp_for_physics = sp_pred
                    u_sl, v_sl = step.u_flow_nd, step.v_flow_nd
                    logits = _model_logits(model, feats, edge_index)
                    idx = _loss_indices(logits, tgt, m, balanced=balanced and train)
                    tm = tgt[idx]
                    pw = torch.tensor([pos_weight], device=device, dtype=logits.dtype)
                    blend = _env_bool("CLOT_PHI_PHYSICS_BLEND", False)
                    alpha = _physics_blend_alpha()
                    phi_ml = torch.sigmoid(logits)
                    mu_mse = torch.tensor(0.0, device=device)
                    if hybrid:
                        dlog = _model_delta_log_mu(model, feats, edge_index)
                        mu_ml = mu_eff_from_delta_log_si(step.mu_c_si, dlog)
                    else:
                        mu_ml = log_blend_mu_eff_si(step.mu_c_si, phi_ml)
                    if blend:
                        mu_phys = cap_mu_eff_si(
                            physics_mu_eff_si(
                                step.mu_c_si,
                                sp_for_physics,
                                bio_cfg,
                                device=device,
                                data=data,
                                u_nd=u_sl,
                                v_nd=v_sl,
                            )
                        )
                        from src.core_physics.clot_phi_simple import (
                            clot_phi_physics_subtract_t0_mu,
                            gt_mu_anchor_cap_si,
                        )

                        mu_anchor = (
                            gt_mu_anchor_cap_si(data, phys_cfg, device)
                            if clot_phi_physics_subtract_t0_mu()
                            else None
                        )
                        phi_phys = physics_phi_from_mu(
                            mu_phys,
                            step.mu_c_si,
                            step.region if clot_phi_forward_apply_region() else None,
                            phys_cfg,
                            soft=use_soft,
                            mu_anchor_si=mu_anchor,
                        )
                        phi_mix = (alpha * phi_ml + (1.0 - alpha) * phi_phys).clamp(1e-6, 1.0 - 1e-6)
                        if regression_only:
                            bce = torch.tensor(0.0, device=device)
                        else:
                            pm = phi_mix[idx]
                            bce_n = F.binary_cross_entropy(pm, tm, reduction="none")
                            w = torch.where(tm > 0.5, pw, torch.ones_like(tm))
                            bce = (bce_n * w).mean()
                        if mu_log_lambda > 0.0:
                            log_mu_p = (
                                alpha * torch.log(mu_ml.clamp(min=1e-8))
                                + (1.0 - alpha) * torch.log(mu_phys.clamp(min=1e-8))
                            )
                            log_mu_t = torch.log(step.mu_gt_cap.clamp(min=1e-8))
                            mu_mse = F.mse_loss(log_mu_p[idx], log_mu_t[idx])
                        phi_pred = phi_mix
                        mu_pred = alpha * mu_ml + (1.0 - alpha) * mu_phys
                    else:
                        lm = logits[idx]
                        if regression_only:
                            bce = torch.tensor(0.0, device=device)
                        else:
                            bce = F.binary_cross_entropy_with_logits(lm, tm, pos_weight=pw)
                        if hybrid and mu_log_lambda > 0.0:
                            log_mu_p = torch.log(step.mu_c_si.clamp(min=1e-8)) + dlog
                            log_mu_t = torch.log(step.mu_gt_cap.clamp(min=1e-8))
                            mu_mse = F.mse_loss(log_mu_p[idx], log_mu_t[idx])
                        phi_pred = phi_ml
                        mu_pred = mu_ml
                    bulk_t = int(t_out) if forecast_one_step else None
                    support_t = int(t_out) if forecast_one_step else int(ti)
                    mu_pred = _maybe_project_deploy_mu(
                        data=data,
                        step=step,
                        mu_pred=mu_pred,
                        phys_cfg=phys_cfg,
                        bio_cfg=bio_cfg,
                        device=device,
                        forecast_one_step=forecast_one_step,
                        time_index=support_t,
                        bulk_time_index=bulk_t,
                    )
                    if train and (not regression_only) and dice_lambda > 0.0:
                        pm = phi_pred[idx].clamp(1e-6, 1.0 - 1e-6)
                        inter = (pm * tm).sum()
                        dice_loss = 1.0 - (2.0 * inter + 1e-6) / (pm.sum() + tm.sum() + 1e-6)
                        loss = bce + dice_lambda * dice_loss + mu_log_lambda * mu_mse + bio_lambda * bio_mse
                    else:
                        if regression_only:
                            loss = mu_log_lambda * mu_mse + bio_lambda * bio_mse
                        else:
                            loss = bce + mu_log_lambda * mu_mse + bio_lambda * bio_mse
                    if train and not rule_baseline and not physics_oracle:
                        mesh_loss = _mesh_aux_losses(
                            phi_pred=phi_pred,
                            data=data,
                            step=step,
                            ti=ti,
                            t_out=t_out if forecast_one_step else None,
                            forecast_one_step=forecast_one_step,
                            pair_stride=pair_stride,
                            t_steps=t_steps,
                            phys_cfg=phys_cfg,
                            bio_cfg=bio_cfg,
                            device=device,
                            use_soft=use_soft,
                        )
                        loss = loss + mesh_loss
                    if train:
                        loss = loss * float(path_step_scale.get(path, 1.0))
                        loss.backward()
                        optimizer.step()
                with torch.no_grad():
                    log_mae = (torch.log(mu_pred[m].clamp(min=1e-8)) - torch.log(step.mu_gt_cap[m].clamp(min=1e-8))).abs().mean()
                    dice_sum += _dice_score(phi_pred[m], tgt[m])
                    log_mae_sum += float(log_mae.item())
                    bce_sum += float(bce.item())
                    mu_mse_sum += float(mu_mse.item())
                    bio_mse_sum += float(bio_mse.item())
                    for k, v in _clot_metrics(phi_pred, tgt, m).items():
                        metric_sums[k] += v
                    n_steps += 1
                    t_gt = int(t_out) if forecast_one_step else int(ti)
                    t_gt = min(int(t_gt), t_steps - 1)
                    last_gt_y = data.y[t_gt].to(device=device, dtype=torch.float32)
                    last_mu_pred_si = cap_mu_eff_si(mu_pred.reshape(-1))
                    if rollout_state is not None:
                        rollout_state.update_from_pred(
                            phi_pred,
                            mu_pred,
                            detach=clot_phi_rollout_detach_carry() or not train,
                        )
                    if forecast_state is not None:
                        forecast_state.update_from_pred(
                            phi_pred,
                            mu_pred,
                            detach=clot_forecast_mu_carry_detach() or not train,
                        )
                    if forecast_one_step:
                        _accumulate_clot_shape(
                            shape_sums=shape_sums,
                            data=data,
                            phi_pred=phi_pred,
                            phys_cfg=phys_cfg,
                            bio_cfg=bio_cfg,
                            device=device,
                            time_index=ti,
                            t_out=t_out,
                            forecast_one_step=forecast_one_step,
                            pair_stride=pair_stride,
                            t_steps=t_steps,
                            fixed_mu_from_phi=clot_phi_fixed_mu_from_phi_enabled(),
                            hybrid=hybrid,
                            mu_pred=mu_pred,
                        )
                        n_shape_graphs += 1

            if (
                not forecast_one_step
                and last_mu_pred_si is not None
                and last_gt_y is not None
            ):
                pred_state = _state_with_pred_mu(last_gt_y, last_mu_pred_si, phys_cfg)
                gt_state = last_gt_y.clone()
                ei_shape = data.edge_index.to(device)
                sm = compute_clot_shape_metrics(
                    pred_state=pred_state,
                    gt_state=gt_state,
                    edge_index=ei_shape,
                    phys_cfg=phys_cfg,
                )
                shape_sums["clot_shape"] += float(sm["clot_shape"])
                shape_sums["clot_shape_rec"] += float(sm["clot_recall"])
                shape_sums["clot_shape_pred_frac"] += float(sm["clot_pred_frac"])
                shape_sums["clot_shape_gt_frac"] += float(sm["clot_gt_frac"])
                n_shape_graphs += 1

    denom = max(n_steps, 1)
    out = {
        "bce": bce_sum / denom,
        "mu_log_mse": mu_mse_sum / denom,
        "bio_mse": bio_mse_sum / denom,
        "dice": dice_sum / denom,
        "mu_log_mae": log_mae_sum / denom,
        "n_steps": float(n_steps),
        "n_graphs": float(n_graphs),
    }
    full_scaled = metric_sums.pop("_full_scaled", None)
    out.update({k: v / denom for k, v in metric_sums.items()})
    if full_scaled is not None:
        scaled_avg = float(full_scaled) / denom
        out["full_mesh_f1_scaled"] = scaled_avg
        out["loss_mask_f1_scaled"] = scaled_avg
    shape_denom = max(n_shape_graphs, 1)
    out.update({k: v / shape_denom for k, v in shape_sums.items()})
    out["n_shape_graphs"] = float(n_shape_graphs)
    return out


def main() -> None:
    root = get_project_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phys_cfg = PhysicsConfig(phase="biochem")
    bio_cfg = BiochemConfig(phase="biochem")

    raw_dir = (os.environ.get("CLOT_PHI_ANCHOR_DIR") or "").strip()
    if raw_dir:
        anchor_dir = Path(raw_dir).expanduser()
        if not anchor_dir.is_absolute():
            anchor_dir = root / anchor_dir
    else:
        anchor_dir = root / VesselConfig(phase="biochem_anchors").graph_output_dir
    anchor_dir = anchor_dir.resolve()
    paths = _list_anchor_paths(anchor_dir)
    val_stem = (os.environ.get("CLOT_PHI_VAL_ANCHOR") or "patient007").strip()
    train_paths, val_paths = _split_train_val(paths, val_stem)

    epochs = max(int(os.environ.get("CLOT_PHI_EPOCHS", "40")), 1)
    lr = float(os.environ.get("CLOT_PHI_LR", "3e-3"))
    time_stride = max(int(os.environ.get("CLOT_PHI_TIME_STRIDE", "2")), 1)
    auto_time_stride = _env_bool("CLOT_PHI_TIME_STRIDE_AUTO", True)
    if auto_time_stride and train_paths:
        min_t = None
        for p in train_paths:
            d = torch.load(p, map_location="cpu", weights_only=False)
            t_steps = int(getattr(d, "y").shape[0]) if hasattr(d, "y") else 0
            min_t = t_steps if min_t is None else min(min_t, t_steps)
        # Short-anchor safeguard: avoid skipping informative snapshots when
        # any train anchor is short after cache subsampling.
        if min_t is not None and min_t <= 8:
            time_stride = 1
    hidden = max(int(os.environ.get("CLOT_PHI_HIDDEN", "64")), 4)
    rule_baseline = _env_bool("CLOT_PHI_RULE_BASELINE", False)
    physics_oracle = clot_phi_physics_oracle_enabled()
    joint_bio = clot_phi_joint_bio_enabled()
    in_dim = clot_phi_feature_dim()

    if rule_baseline:
        print("[i]  clot_phi_simple: RULE BASELINE (no training)", flush=True)
        va = _run_epoch(
            None,
            val_paths,
            phys_cfg=phys_cfg,
            bio_cfg=bio_cfg,
            device=device,
            train=False,
            time_stride=1,
            pos_weight=1.0,
            balanced=False,
            rule_baseline=True,
        )
        print(
            f"[OK]  rule val dice={va['dice']:.3f} bce={va['bce']:.4f} logMAE={va['mu_log_mae']:.4f}",
            flush=True,
        )
        return

    if physics_oracle:
        print("[i]  clot_phi_simple: PHYSICS ORACLE (Carreau x gelation, no training)", flush=True)
        for label, paths_eval in (("val", val_paths), ("train", train_paths[:2])):
            va = _run_epoch(
                None,
                paths_eval,
                phys_cfg=phys_cfg,
                bio_cfg=bio_cfg,
                device=device,
                train=False,
                time_stride=1,
                pos_weight=1.0,
                balanced=False,
                physics_oracle=True,
            )
            print(
                f"[OK]  physics {label} dice={va['dice']:.3f} f1={va['clot_f1']:.3f} "
                f"prec={va['clot_prec']:.3f} rec={va['clot_rec']:.3f} "
                f"logMAE={va['mu_log_mae']:.4f} bio_mse={va.get('bio_mse', 0):.4f} "
                f"pred+={va['pred_pos_frac']:.3f} score={_checkpoint_score(va):.3f}",
                flush=True,
            )
        return

    model = build_clot_phi_model(in_dim=in_dim, hidden=hidden).to(device)
    species_head = None
    if joint_bio:
        species_head = ClotPhiSpeciesHead(in_dim=in_dim, hidden=_species_hidden()).to(device)
    init_raw = (os.environ.get("CLOT_PHI_INIT_CHECKPOINT") or "").strip()
    if init_raw:
        init_path = Path(init_raw)
        if not init_path.is_absolute():
            init_path = root / init_path
        if init_path.is_file():
            _load_init_checkpoint(model, species_head, init_path, device)
            print(f"[i]  loaded init checkpoint {init_path}", flush=True)
    _freeze_mu_branch(model)
    wd = float(os.environ.get("CLOT_PHI_WEIGHT_DECAY", "0.0") or "0.0")
    params = [p for p in model.parameters() if p.requires_grad]
    if species_head is not None:
        params += [p for p in species_head.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    # pos_weight from a quick pass on train set
    pos = 0.0
    tot = 0.0
    for path in train_paths[: min(4, len(train_paths))]:
        data = torch.load(path, weights_only=False)
        for ti in range(0, data.y.shape[0], time_stride):
            step = build_clot_phi_step(data, ti, phys_cfg, bio_cfg, torch.device("cpu"))
            m = step.loss_mask
            if m.any():
                pos += float(step.phi_gt[m].sum().item())
                tot += float(m.sum().item())
    pos_frac = pos / max(tot, 1.0)
    pos_weight = min(max((tot - pos) / max(pos, 1.0), 1.0), float(os.environ.get("CLOT_PHI_POS_WEIGHT_CAP", "15")))
    balanced = _env_bool("CLOT_PHI_BALANCED", False)
    mu_log_lambda = max(float(os.environ.get("CLOT_PHI_MU_LOG_LAMBDA", "1.0") or "0"), 0.0)
    use_soft = _env_bool("CLOT_PHI_SOFT_LABELS", False)
    if balanced:
        # If we subsample 1:1 pos/neg, do NOT also upweight positives.
        pos_weight = 1.0

    # Final-layer bias ~ logit(prior) so the net does not start at sigmoid(0)=0.5 everywhere.
    prior = max(min(pos_frac, 0.45), 0.02)
    with torch.no_grad():
        if hasattr(model, "phi_fc") and isinstance(model.phi_fc, torch.nn.Linear):
            model.phi_fc.bias.fill_(float(torch.log(torch.tensor(prior / (1.0 - prior)))))
        elif hasattr(model, "net"):
            last = model.net[-1]
            if isinstance(last, torch.nn.Linear) and last.bias is not None:
                last.bias.fill_(float(torch.log(torch.tensor(prior / (1.0 - prior)))))

    print(
        f"[i]  clot_phi_simple: mask={clot_phi_mask_mode()} cap={clot_phi_mu_cap_si():.3f} Pa*s "
        f"thr={clot_phi_thresh_si(phys_cfg):.3f} soft={int(use_soft)} balanced={int(balanced)} "
        f"hybrid={int(clot_phi_hybrid_enabled())} fixed_mu={int(clot_phi_fixed_mu_from_phi_enabled())} "
        f"model={clot_phi_model_kind()} "
        f"hidden={hidden} depth={clot_phi_mlp_depth()} dropout={clot_phi_dropout():.2f} "
        f"minimal={int(clot_phi_minimal_features_enabled())} species_feat={int(clot_phi_species_features_enabled())} "
        f"joint_bio={int(joint_bio)} in_dim={in_dim} "
        f"rollout={int(clot_phi_rollout_enabled())} trigger_rollout={int(clot_phi_trigger_rollout_enabled())} "
        f"vel={clot_phi_vel_source()} "
        f"carry_bridge={snapshot_carry_gt_warmup_config()} "
        f"forecast={snapshot_clot_forecast_config()} "
        f"train={len(train_paths)} val={len(val_paths)} "
        f"prior={prior:.3f} pos_weight={pos_weight:.2f} mu_log_w={mu_log_lambda:.2f} "
        f"bio_w={_bio_lambda():.2f}",
        flush=True,
    )

    sweep_root = (os.environ.get("CLOT_PHI_SWEEP_DIR") or "").strip()
    sweep_leg = (os.environ.get("CLOT_PHI_SWEEP_LEG") or "").strip()
    if sweep_root and sweep_leg:
        out_dir = (root / sweep_root / sweep_leg).resolve()
    else:
        out_dir = (root / "outputs" / "biochem").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = (os.environ.get("CLOT_PHI_CKPT_NAME") or "clot_phi_best.pth").strip()
    ckpt_path = out_dir / ckpt_name
    best_score = -1.0
    best_dice = -1.0
    log_name = (os.environ.get("CLOT_PHI_LOG_NAME") or "clot_phi_train_log.jsonl").strip()
    log_path = out_dir / log_name
    if ckpt_path.is_file():
        log_path.unlink(missing_ok=True)

    for epoch in range(epochs):
        tr = _run_epoch(
            model,
            train_paths,
            phys_cfg=phys_cfg,
            bio_cfg=bio_cfg,
            device=device,
            train=True,
            time_stride=time_stride,
            pos_weight=pos_weight,
            balanced=balanced,
            species_head=species_head,
            optimizer=opt,
            train_epoch=epoch,
        )
        with torch.no_grad():
            va = _run_epoch(
                model,
                val_paths,
                phys_cfg=phys_cfg,
                bio_cfg=bio_cfg,
                device=device,
                train=False,
                time_stride=1,
                pos_weight=pos_weight,
                balanced=False,
                species_head=species_head,
            )
        score = _checkpoint_score(va)
        print(
            f"Ep {epoch:02d} | train bce={tr['bce']:.4f} mu_mse={tr.get('mu_log_mse', 0):.4f} "
            f"bio_mse={tr.get('bio_mse', 0):.4f} dice={tr['dice']:.3f} f1={tr['clot_f1']:.3f} "
            f"shape={tr.get('clot_shape', 0):.3f} logMAE={tr['mu_log_mae']:.3f} pred+={tr['pred_pos_frac']:.3f} | "
            f"val bce={va['bce']:.4f} mu_mse={va.get('mu_log_mse', 0):.4f} bio_mse={va.get('bio_mse', 0):.4f} "
            f"dice={va['dice']:.3f} "
            f"f1={va['clot_f1']:.3f} shape={va.get('clot_shape', 0):.3f} shape_rec={va.get('clot_shape_rec', 0):.3f} "
            f"prec={va['clot_prec']:.3f} rec={va['clot_rec']:.3f} "
            f"logMAE={va['mu_log_mae']:.3f} pred+={va['pred_pos_frac']:.3f} gt+={va['gt_pos_frac']:.3f} score={score:.3f}",
            flush=True,
        )
        row = {"epoch": epoch, "train": tr, "val": va, "val_score": score, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        ckpt_config = {
            "mu_cap_si": clot_phi_mu_cap_si(),
            "mu_thresh_si": clot_phi_thresh_si(phys_cfg),
            "hidden": hidden,
            "in_dim": in_dim,
            "oracle_mu": clot_phi_oracle_mu_enabled(),
            "species_features": clot_phi_species_features_enabled(),
            "joint_bio": joint_bio,
            "use_prior_features": clot_phi_use_prior_features(),
            "prior_n": clot_phi_prior_feature_count(),
            "hybrid": clot_phi_hybrid_enabled(),
            "minimal_features": clot_phi_minimal_features_enabled(),
            "model_kind": clot_phi_model_kind(),
            "dropout": clot_phi_dropout(),
            "mlp_depth": clot_phi_mlp_depth(),
            "lr": lr,
            "weight_decay": wd,
            "mu_log_lambda": mu_log_lambda,
            "bio_lambda": _bio_lambda(),
            "regression_only": _regression_only(),
            "anchor_dir": str(anchor_dir),
            "physics_blend": _env_bool("CLOT_PHI_PHYSICS_BLEND", False),
            "physics_blend_alpha": _physics_blend_alpha(),
            "rollout": clot_phi_rollout_enabled(),
            "rollout_vel_source": clot_phi_vel_source(),
            "rollout_carry_phi": clot_phi_rollout_enabled() and clot_phi_carry_phi_enabled(),
            "rollout_carry_log_mu": clot_phi_rollout_enabled() and clot_phi_carry_log_mu_enabled(),
            "rollout_detach": clot_phi_rollout_detach_carry(),
            **snapshot_phi_only_rollout_config(),
            **snapshot_mesh_aux_config(),
            **snapshot_clot_support_config(),
            **snapshot_carry_gt_warmup_config(),
            "dgamma_feature_time": (os.environ.get("CLOT_PHI_DGAMMA_FEATURE_TIME") or "ref").strip(),
            **snapshot_clot_forecast_config(),
        }
        from src.training.clot_trigger_stack import snapshot_trigger_train_config

        ckpt_config.update(snapshot_trigger_train_config())
        last_payload: dict[str, Any] = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_dice": float(va["dice"]),
            "val_f1": float(va["clot_f1"]),
            "val_score": score,
            "config": ckpt_config,
        }
        if species_head is not None:
            last_payload["species_head_state_dict"] = species_head.state_dict()
        torch.save(last_payload, out_dir / "clot_phi_last.pth")

        if score > best_score:
            best_score = score
            best_dice = float(va["dice"])
            payload = dict(last_payload)
            payload["val_score"] = best_score
            payload["val_dice"] = best_dice
            torch.save(payload, ckpt_path)
            print(
                f"   [OK]  saved {ckpt_path} (val f1={va['clot_f1']:.3f} dice={best_dice:.3f} score={best_score:.3f})",
                flush=True,
            )

    if best_score < 0 and (out_dir / "clot_phi_last.pth").is_file():
        print("[WARN] no val ckpt passed scorer; run recover_clot_phi_best_from_log.py", flush=True)

    print(f"[OK]  Done. Best val score={best_score:.3f} dice={best_dice:.3f} -> {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
