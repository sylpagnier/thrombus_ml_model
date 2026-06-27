"""Differentiable deploy (mlp_band) forward + losses for coupled clot-phi finetune."""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn.functional as F

from src.config import BiochemConfig, PhysicsConfig
from src.utils import species_channels as sc
from src.core_physics.clot_phi_mu_inject import (
    mlp_clot_use_pred_species,
    resolve_clot_mu_commit_thresh_si,
    resolve_mu_map_baselines_si,
)
from src.core_physics.clot_phi_simple import (
    build_clot_phi_step,
    clot_phi_hybrid_enabled,
    log_blend_mu_eff_si,
    mu_eff_from_delta_log_si,
)


def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return bool(default)
    return raw in ("1", "true", "yes", "on")


def _zero_like(x: torch.Tensor) -> torch.Tensor:
    """Grad-connected scalar zero (safe when all loss terms skip)."""
    return x.reshape(-1)[:1].sum() * 0.0


def forward_mlp_fields_at_rollout_frame(
    clot_model: torch.nn.Module,
    data: Any,
    time_index: int,
    *,
    u_nd: torch.Tensor,
    v_nd: torch.Tensor,
    species_log: torch.Tensor,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Differentiable mirror of ``predict_mlp_fields_at_rollout_frame`` (deploy diagnose path)."""
    y_slice = data.y[time_index].to(device).clone()
    y_slice[:, 0] = u_nd.reshape(-1).to(dtype=y_slice.dtype)
    y_slice[:, 1] = v_nd.reshape(-1).to(dtype=y_slice.dtype)
    if mlp_clot_use_pred_species():
        y_slice[:, sc.SPECIES_BLOCK] = species_log.to(device=device, dtype=y_slice.dtype)
    step = build_clot_phi_step(
        data,
        time_index,
        phys_cfg,
        bio_cfg,
        device,
        u_nd_override=u_nd.reshape(-1),
        v_nd_override=v_nd.reshape(-1),
        y_slice_override=y_slice,
    )
    mu_bulk, mu_mlp_anchor = resolve_mu_map_baselines_si(
        data, u_nd.reshape(-1), v_nd.reshape(-1), phys_cfg
    )
    logits: torch.Tensor | None = None
    if clot_phi_hybrid_enabled() and hasattr(clot_model, "forward_delta_log_mu"):
        logits = clot_model.forward_logits(step.features)
        phi = torch.sigmoid(logits).reshape(-1)
        mu_mlp = mu_eff_from_delta_log_si(
            mu_mlp_anchor, clot_model.forward_delta_log_mu(step.features)
        ).reshape(-1)
    else:
        phi = clot_model(step.features).reshape(-1)
        mu_mlp = log_blend_mu_eff_si(mu_mlp_anchor, phi).reshape(-1)
    return {
        "phi": phi.reshape(-1),
        "mu_mlp": mu_mlp.reshape(-1),
        "mu_bulk": mu_bulk.reshape(-1),
        "mu_mlp_anchor": mu_mlp_anchor.reshape(-1),
        "region": step.region.reshape(-1).bool(),
        "mu_gt_cap": step.mu_gt_cap.reshape(-1),
        "phi_gt": step.phi_gt.reshape(-1),
        "logits": logits.reshape(-1) if logits is not None else None,
        "features": step.features,
    }


def resolve_supervise_clot_mask(
    *,
    gt_clot: torch.Tensor,
    phi_gt: torch.Tensor,
    allowed: torch.Tensor,
    phi_thresh: float,
) -> torch.Tensor:
    """GT clot nodes inside allowed; else high-phi allowed. Empty if neither."""
    gt = gt_clot.reshape(-1).bool()
    if bool(gt.any().item()):
        return gt
    soft = (phi_gt.reshape(-1) >= float(phi_thresh)) & allowed.reshape(-1).bool()
    return soft


def soft_mlp_band_gate(
    allowed: torch.Tensor,
    phi: torch.Tensor,
    mu_mlp_si: torch.Tensor,
    *,
    phi_thresh: float,
    mu_thresh_si: float,
    phi_sharp: float = 20.0,
    mu_sharp: float = 200.0,
) -> torch.Tensor:
    """Differentiable surrogate for ``mlp_band`` hard gate (allowed & phi & mu_mlp)."""
    allowed_f = allowed.reshape(-1).float()
    phi_s = torch.sigmoid(phi_sharp * (phi.reshape(-1) - float(phi_thresh)))
    mu_s = torch.sigmoid(mu_sharp * (mu_mlp_si.reshape(-1) - float(mu_thresh_si)))
    return (allowed_f * phi_s * mu_s).clamp(0.0, 1.0)


def soft_committed_mu_si(
    mu_bulk_si: torch.Tensor,
    mu_mlp_si: torch.Tensor,
    soft_gate: torch.Tensor,
    *,
    blend: float = 1.0,
) -> torch.Tensor:
    """``mu = mu_bulk + (blend * gate) * (mu_mlp - mu_bulk)`` (Leg B v2 blend)."""
    alpha = max(0.0, min(float(blend), 1.0))
    mu_b = mu_bulk_si.reshape(-1)
    mu_m = mu_mlp_si.reshape(-1)
    g = soft_gate.reshape(-1).to(dtype=mu_b.dtype)
    return (mu_b + (alpha * g) * (mu_m - mu_b)).clamp(min=1e-8)


def compute_allowed_deploy_metrics(
    *,
    phi: torch.Tensor,
    mu_mlp: torch.Tensor,
    mu_rollout_si: torch.Tensor,
    allowed: torch.Tensor,
    mu_thresh: float,
    phi_thresh: float,
) -> dict[str, float]:
    """Metrics on frozen deploy allowed mask (matches diagnose gate)."""
    allow = allowed.reshape(-1).bool()
    n = max(int(allow.sum().item()), 1)
    phi_a = phi.reshape(-1)[allow]
    mu_a = mu_mlp.reshape(-1)[allow]
    roll_a = mu_rollout_si.reshape(-1)[allow]
    both = (phi_a >= phi_thresh) & (mu_a >= mu_thresh)
    return {
        "frac_mu_ok_allowed": float((mu_a >= mu_thresh).float().mean().item()),
        "frac_both_allowed": float(both.float().mean().item()),
        "frac_rollout_mu_ok_allowed": float((roll_a >= mu_thresh).float().mean().item()),
        "mu_mlp_p90_allowed": float(torch.quantile(mu_a, 0.9).item()) if mu_a.numel() else 0.0,
        "n_allowed": float(n),
    }


def compute_deploy_coupled_step_losses(
    *,
    phi_pred: torch.Tensor,
    mu_mlp: torch.Tensor,
    mu_bulk: torch.Tensor,
    mu_gt_cap: torch.Tensor,
    phi_gt: torch.Tensor,
    allowed: torch.Tensor,
    gt_clot: torch.Tensor,
    phys_cfg: PhysicsConfig,
    phi_thresh: float,
    mu_log_lambda: float,
    hinge_lambda: float,
    allowed_hinge_lambda: float,
    soft_commit_lambda: float,
    phi_lambda: float,
    pos_weight: float,
    logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Losses on frozen deploy allowed mask (diagnose-aligned ``mlp_band``)."""
    device = mu_mlp.device
    mu_thresh = resolve_clot_mu_commit_thresh_si(phys_cfg)
    allowed_b = allowed.reshape(-1).bool()
    supervise = resolve_supervise_clot_mask(
        gt_clot=gt_clot, phi_gt=phi_gt, allowed=allowed_b, phi_thresh=phi_thresh
    )
    z = _zero_like(mu_mlp)

    bce = z
    if phi_lambda > 0.0 and logits is not None:
        idx = allowed_b.nonzero(as_tuple=False).view(-1)
        if idx.numel() > 0:
            tm = phi_gt[idx]
            pw = torch.tensor([pos_weight], device=device, dtype=logits.dtype)
            bce = F.binary_cross_entropy_with_logits(logits[idx], tm, pos_weight=pw)

    mu_mse = z
    if mu_log_lambda > 0.0 and bool(supervise.any().item()):
        log_p = torch.log(mu_mlp[supervise].clamp(min=1e-8))
        log_t = torch.log(mu_gt_cap[supervise].clamp(min=1e-8))
        mu_mse = F.mse_loss(log_p, log_t)

    hinge = z
    if hinge_lambda > 0.0 and bool(supervise.any().item()):
        gap = F.relu(mu_thresh - mu_mlp[supervise])
        hinge = (gap * gap).mean()

    allowed_hinge = z
    if allowed_hinge_lambda > 0.0 and bool(allowed_b.any().item()):
        phi_ok = phi_pred.reshape(-1)[allowed_b] >= float(phi_thresh)
        if bool(phi_ok.any().item()):
            idx = allowed_b.nonzero(as_tuple=False).view(-1)[phi_ok]
            gap = F.relu(mu_thresh - mu_mlp[idx])
            allowed_hinge = (gap * gap).mean()

    soft_gate = soft_mlp_band_gate(
        allowed_b,
        phi_pred,
        mu_mlp,
        phi_thresh=phi_thresh,
        mu_thresh_si=mu_thresh,
        phi_sharp=_env_float("MLP_DEPLOY_COUPLED_PHI_SHARP", 20.0),
        mu_sharp=_env_float("MLP_DEPLOY_COUPLED_MU_SHARP", 200.0),
    )
    mu_commit = soft_committed_mu_si(mu_bulk, mu_mlp, soft_gate)

    soft_commit = z
    if soft_commit_lambda > 0.0 and bool(supervise.any().item()):
        target = torch.ones_like(soft_gate[supervise])
        soft_commit = F.binary_cross_entropy(soft_gate[supervise].clamp(1e-6, 1.0 - 1e-6), target)

    commit_log = z
    if soft_commit_lambda > 0.0 and bool(supervise.any().item()):
        log_p = torch.log(mu_commit[supervise].clamp(min=1e-8))
        log_t = torch.log(mu_gt_cap[supervise].clamp(min=1e-8))
        commit_log = F.mse_loss(log_p, log_t)

    loss = (
        phi_lambda * bce
        + mu_log_lambda * mu_mse
        + hinge_lambda * hinge
        + allowed_hinge_lambda * allowed_hinge
        + soft_commit_lambda * (soft_commit + commit_log)
    )
    terms = {
        "bce": bce,
        "mu_mse": mu_mse,
        "hinge": hinge,
        "allowed_hinge": allowed_hinge,
        "soft_commit": soft_commit,
        "commit_log": commit_log,
        "mu_commit": mu_commit.detach(),
        "soft_gate": soft_gate.detach(),
    }
    return loss, terms


def resolve_mu_baselines_for_step(
    data,
    u_nd: torch.Tensor,
    v_nd: torch.Tensor,
    phys_cfg: PhysicsConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    mu_bulk, mu_anchor = resolve_mu_map_baselines_si(
        data, u_nd.reshape(-1), v_nd.reshape(-1), phys_cfg
    )
    return mu_bulk.reshape(-1), mu_anchor.reshape(-1)


def deploy_coupled_promote_score(val_metrics: dict[str, float]) -> float:
    """Rank ckpts by diagnose-aligned allowed metrics (gate_frac_mu is primary)."""
    gate_mu = float(val_metrics.get("gate_frac_mu", 0.0))
    both = float(val_metrics.get("frac_both_allowed", val_metrics.get("frac_both_band", 0.0)))
    mu_ok = float(val_metrics.get("frac_mu_ok_allowed", val_metrics.get("frac_mu_ok_band", 0.0)))
    roll_ok = float(
        val_metrics.get("frac_rollout_mu_ok_allowed", val_metrics.get("frac_rollout_mu_ok_band", 0.0))
    )
    p90 = float(val_metrics.get("gate_mu_p90", val_metrics.get("mu_mlp_p90_allowed", 0.0)))
    hinge = float(val_metrics.get("allowed_hinge", val_metrics.get("mu_hinge", 0.0)))
    log_mae = float(val_metrics.get("mu_log_mae", 1.0))
    mu_thr = 0.055
    p90_bonus = min(max(p90 / max(mu_thr, 1e-6), 0.0), 2.0) * 0.15
    return (
        (4.0 * gate_mu)
        + (2.0 * both)
        + mu_ok
        + roll_ok
        + p90_bonus
        + max(0.0, 0.5 - hinge) * 0.5
        - log_mae * 0.05
    )
