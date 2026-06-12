"""Clot-trigger F1 metrics: full vessel (baseline-scaled) + ceiling band."""

from __future__ import annotations

import math

import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_growth_masks import resolve_ceiling_mask
from src.training.train_clot_phi_simple import _clot_metrics


def f1_zero_prediction_baseline(
    target: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """F1 when predicting zero clot everywhere (trash baseline on ``mask``)."""
    n = int(mask.sum().item()) if bool(mask.any().item()) else 0
    if n <= 0:
        return float("nan")
    zeros = torch.zeros_like(target.reshape(-1))
    return float(_clot_metrics(zeros, target.reshape(-1), mask.reshape(-1))["clot_f1"])


def f1_scaled_above_baseline(f1: float, f1_baseline: float) -> float:
    """Scale F1 so baseline (predict-none) maps to 0 and perfect maps to 1."""
    if not math.isfinite(f1) or not math.isfinite(f1_baseline):
        return float("nan")
    denom = max(1.0 - float(f1_baseline), 1e-6)
    return max(0.0, (float(f1) - float(f1_baseline)) / denom)


def resolve_ceiling_eval_mask(
    data,
    device: torch.device,
    bio_cfg: BiochemConfig | None = None,
) -> torch.Tensor:
    """Fixed ceiling band: wall + ``CLOT_PHI_CEILING_HOPS`` lumen hops."""
    bio = bio_cfg or BiochemConfig(phase="biochem")
    return resolve_ceiling_mask(data, device, bio).reshape(-1).to(device=device).bool()


def clot_trigger_metric_bundle(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    full_mesh_mask: torch.Tensor | None = None,
    ceiling_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """F1 on full vessel (raw + zero-baseline scaled) and inside ceiling mask."""
    pred_f = pred.reshape(-1)
    tgt_f = target.reshape(-1)
    n = int(pred_f.numel())
    device = pred_f.device
    full = (
        full_mesh_mask.reshape(-1).to(device=device).bool()
        if full_mesh_mask is not None
        else torch.ones(n, device=device, dtype=torch.bool)
    )

    m_full = _clot_metrics(pred_f, tgt_f, full)
    f1_full = float(m_full["clot_f1"])
    f1_base = f1_zero_prediction_baseline(tgt_f, full)
    out: dict[str, float] = {
        "clot_f1": f1_full,
        "clot_prec": float(m_full["clot_prec"]),
        "clot_rec": float(m_full["clot_rec"]),
        "pred_pos_frac": float(m_full["pred_pos_frac"]),
        "gt_pos_frac": float(m_full["gt_pos_frac"]),
        "full_mesh_f1": f1_full,
        "full_mesh_f1_baseline_zero": f1_base,
        "full_mesh_f1_scaled": f1_scaled_above_baseline(f1_full, f1_base),
    }

    if ceiling_mask is not None:
        ceil = ceiling_mask.reshape(-1).to(device=device).bool()
        m_ceil = _clot_metrics(pred_f, tgt_f, ceil)
        f1_ceil = float(m_ceil["clot_f1"])
        f1_ceil_base = f1_zero_prediction_baseline(tgt_f, ceil)
        out.update(
            {
                "ceiling_f1": f1_ceil,
                "ceiling_f1_baseline_zero": f1_ceil_base,
                "ceiling_f1_scaled": f1_scaled_above_baseline(f1_ceil, f1_ceil_base),
            }
        )
    return out


def clot_trigger_step_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    data,
    device: torch.device,
    bio_cfg: BiochemConfig | None = None,
    loss_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Per-step metrics for T0+ eval/viz (full mesh primary + ceiling band)."""
    ceil = resolve_ceiling_eval_mask(data, device, bio_cfg)
    out = clot_trigger_metric_bundle(
        pred,
        target,
        ceiling_mask=ceil,
    )
    if loss_mask is not None:
        m_loss = _clot_metrics(
            pred.reshape(-1),
            target.reshape(-1),
            loss_mask.reshape(-1).to(device=pred.device).bool(),
        )
        f1_loss = float(m_loss["clot_f1"])
        f1_loss_base = f1_zero_prediction_baseline(
            target.reshape(-1),
            loss_mask.reshape(-1).to(device=pred.device).bool(),
        )
        out.update(
            {
                "loss_mask_f1": f1_loss,
                "loss_mask_f1_baseline_zero": f1_loss_base,
                "loss_mask_f1_scaled": f1_scaled_above_baseline(f1_loss, f1_loss_base),
            }
        )
    return out
