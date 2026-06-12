"""Shared clot-trigger visualization helpers."""

from __future__ import annotations

import os

import numpy as np
import torch
from matplotlib.axes import Axes

from src.config import BiochemConfig, PhysicsConfig
from src.evaluation.viz_clot_phi_simple import _scatter_fullmesh_region
from src.training.clot_trigger_stack import (
    forward_clot_trigger_hybrid,
    forward_physics_trigger_phi,
    physics_blend_alpha,
    physics_blend_enabled,
)
from src.training.train_clot_phi_simple import _clot_metrics


def clot_trigger_viz_f1(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
) -> dict[str, float]:
    """F1 inside the step loss/support mask (matches train/eval; viz is full vessel)."""
    mask = loss_mask.reshape(-1).to(device=pred.device, dtype=torch.bool)
    return _clot_metrics(pred.reshape(-1), target.reshape(-1), mask)


def _clot_phi_soft_labels_enabled() -> bool:
    return (os.environ.get("CLOT_PHI_SOFT_LABELS") or "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


@torch.no_grad()
def clot_trigger_viz_phis(
    step,
    data,
    *,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    model=None,
    edge_index: torch.Tensor | None = None,
    species_log1p: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Full-vessel display phis (unmasked). Train/eval F1 uses ``step.loss_mask`` separately."""
    use_soft = _clot_phi_soft_labels_enabled()
    phi_gt = step.phi_gt.reshape(-1)

    phi_phys, _ = forward_physics_trigger_phi(
        step,
        data,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        device=device,
        species_log1p=species_log1p,
        use_soft=use_soft,
        apply_region=False,
    )
    out: dict[str, torch.Tensor] = {
        "phi_gt": phi_gt.reshape(-1),
        "phi_phys": phi_phys.reshape(-1),
    }
    if model is None:
        return out

    bundle = forward_clot_trigger_hybrid(
        model,
        step,
        data,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        device=device,
        edge_index=edge_index,
        species_log1p=species_log1p,
        use_soft=use_soft,
    )
    if physics_blend_enabled():
        alpha = physics_blend_alpha()
        phi_hybrid = (alpha * bundle["phi_ml"] + (1.0 - alpha) * phi_phys).clamp(1e-6, 1.0 - 1e-6)
    else:
        phi_hybrid = bundle["phi_ml"]
    out["phi_hybrid"] = phi_hybrid.reshape(-1)
    return out


def scatter_clot_vessel(
    ax: Axes,
    pos: np.ndarray,
    phi: np.ndarray,
    title: str,
    *,
    scatter_size: float = 4.0,
    mask_outside_region: bool = False,
    region: np.ndarray | None = None,
    display_min: float = 0.0,
) -> None:
    """Scatter clot phi on the **full vessel**; optional ``--band-mask`` greys outside support."""
    n = int(pos.shape[0])
    reg = region.reshape(n).astype(bool) if region is not None else np.ones(n, dtype=bool)
    vals = phi.reshape(-1).astype(np.float64).copy()
    if float(display_min) > 0.0:
        vals = np.where(vals >= float(display_min), vals, 0.0)
    _scatter_fullmesh_region(
        ax,
        pos,
        vals,
        reg,
        title,
        cmap="bwr",
        vmin=0,
        vmax=1,
        s=float(scatter_size),
        layer_positive_on_top=True,
        mask_outside_region=bool(mask_outside_region),
    )


def apply_phi_viz_display(
    phi_by_time: dict[int, torch.Tensor],
    *,
    subtract_t0: bool = False,
) -> dict[int, torch.Tensor]:
    """Optional viz-only transforms (do not change eval tensors)."""
    if not subtract_t0 or not phi_by_time:
        return phi_by_time
    t0 = min(phi_by_time.keys())
    base = phi_by_time[t0].reshape(-1).float()
    return {t: (p.reshape(-1).float() - base).clamp(min=0.0) for t, p in phi_by_time.items()}
