"""Differentiable gelation readout for closed-loop species pushforward (Phase 3).

Maps accumulated FI/Mat log-ND on the ceiling band through soft Mat/FI gelation
sigmoids (deployable thresholds from ``BiochemConfig``), then optional mu_eff
coupling. Used as an auxiliary loss so species deltas feel downstream viscosity
threshold effects during training.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND
from src.core_physics.clot_phi_simple import (
    clot_phi_physics_mu_ratio_max,
    mat_si_for_gelation_from_log1p,
    species_log1p_nd_to_si,
)
from src.core_physics.species_pushforward_gnn import STATE_DIM
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time, gt_mu_anchor_cap_si
from src.training.biochem_species_scope import FI_CHANNEL, MAT_CHANNEL
from src.utils.rheology import multiplicative_clot_mu_eff_nd, phi_clot_from_mat_fi


def continuous_physics_readout() -> bool:
    raw = (os.environ.get("SPECIES_CONTINUOUS_PHYSICS_READOUT") or "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def continuous_phi_loss_weight() -> float:
    raw = (os.environ.get("SPECIES_CONTINUOUS_PHI_LOSS_WEIGHT") or "1.0").strip()
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 1.0


def continuous_mu_loss_weight() -> float:
    raw = (os.environ.get("SPECIES_CONTINUOUS_MU_LOSS_WEIGHT") or "0.25").strip()
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 0.25


def gelation_temperature_scale() -> float:
    """Override sigmoid sharpness (1.0 = use biochem gnode temps)."""
    raw = (os.environ.get("SPECIES_GELATION_TEMP_SCALE") or "1.0").strip()
    try:
        return max(float(raw), 0.1)
    except ValueError:
        return 1.0


def band_log_state_to_species12(
    log_state: torch.Tensor,
    rest: torch.Tensor,
) -> torch.Tensor:
    """Embed band FI/Mat log state into 12-ch species block (rest + updates)."""
    out = rest.clone()
    st = log_state.reshape(-1, STATE_DIM)
    out[:, FI_CHANNEL] = st[:, 0]
    out[:, MAT_CHANNEL] = st[:, 1]
    return out.clamp(min=0.0)


def differentiable_clot_phi_from_species12(
    species_log12: torch.Tensor,
    bio_cfg: BiochemConfig,
) -> torch.Tensor:
    """Soft clot indicator in [0, 1] from FI/Mat log1p ND (differentiable)."""
    mat_si = mat_si_for_gelation_from_log1p(species_log12[:, MAT_CHANNEL], bio_cfg)
    fi_si = species_log1p_nd_to_si(species_log12, bio_cfg)[:, FI_CHANNEL]
    t_scale = max(float(bio_cfg.soft_step_T_scale), 1e-5)
    temp_scale = gelation_temperature_scale()
    temp_mat = max(float(bio_cfg.viscosity_gnode_temp_mat) * t_scale / temp_scale, 1e-8)
    temp_fi = max(float(bio_cfg.viscosity_gnode_temp_fi) * t_scale / temp_scale, 1e-8)
    return phi_clot_from_mat_fi(
        mat_si,
        fi_si,
        mat_crit=float(bio_cfg.viscosity_mat_crit),
        fi_crit=float(bio_cfg.viscosity_fi_crit),
        temp_mat=temp_mat,
        temp_fi=temp_fi,
        combine="max",
    ).reshape(-1)


def differentiable_mu_eff_from_species12(
    species_log12: torch.Tensor,
    mu_carreau_si: torch.Tensor,
    phi_clot: torch.Tensor,
    bio_cfg: BiochemConfig,
) -> torch.Tensor:
    """mu_eff = mu_carreau * (1 + (ratio_max - 1) * phi_clot)."""
    ratio = max(float(clot_phi_physics_mu_ratio_max(bio_cfg)), 1.0)
    mu_c = mu_carreau_si.reshape(-1).to(device=species_log12.device, dtype=species_log12.dtype)
    return multiplicative_clot_mu_eff_nd(mu_c, phi_clot, ratio).reshape(-1).clamp(min=1e-8)


def gelation_frontier_boost() -> float:
    raw = (os.environ.get("SPECIES_GELATION_FRONTIER_BOOST") or "2.0").strip()
    try:
        return max(float(raw), 1.0)
    except ValueError:
        return 2.0


def gelation_phi_loss(
    pred_phi: torch.Tensor,
    gt_phi: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    m = mask.reshape(-1).to(device=pred_phi.device).bool()
    if not bool(m.any().item()):
        return pred_phi.sum() * 0.0
    p = pred_phi[m].clamp(1e-6, 1.0 - 1.0e-6)
    t = gt_phi[m].clamp(0.0, 1.0)
    # Upweight transition band (pre-gelation desync lives here).
    boost = gelation_frontier_boost()
    w = 1.0 + (boost - 1.0) * (4.0 * t * (1.0 - t))
    return (F.binary_cross_entropy(p, t, reduction="none") * w).mean()


def gelation_mu_log_loss(
    pred_mu: torch.Tensor,
    gt_mu: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    m = mask.reshape(-1).to(device=pred_mu.device).bool()
    if not bool(m.any().item()):
        return pred_mu.sum() * 0.0
    p = pred_mu[m].clamp(min=1e-8)
    t = gt_mu[m].clamp(min=1e-8)
    return F.mse_loss(torch.log(p), torch.log(t))


@dataclass(frozen=True)
class SpeciesPhysicsCtx:
    data: object
    phys_cfg: PhysicsConfig
    bio_cfg: BiochemConfig
    node_idx: torch.Tensor
    time_window: list[int]
    rest_band: torch.Tensor
    mu_anchor_si: torch.Tensor


def build_species_physics_ctx(
    data,
    *,
    time_window: list[int],
    node_idx: torch.Tensor,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
) -> SpeciesPhysicsCtx:
    from src.core_physics.t0_rung4_ladder import resting_species_log_nd

    rest_full = resting_species_log_nd(data, device)
    rest_band = rest_full[node_idx]
    anchor = gt_mu_anchor_cap_si(data, phys_cfg, device)
    return SpeciesPhysicsCtx(
        data=data,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        node_idx=node_idx,
        time_window=[int(t) for t in time_window],
        rest_band=rest_band,
        mu_anchor_si=anchor[node_idx],
    )


def gt_phi_band_at_time(
    ctx: SpeciesPhysicsCtx,
    time_index: int,
    device: torch.device,
) -> torch.Tensor:
    phi_full = gt_clot_phi_at_time(ctx.data, int(time_index), ctx.phys_cfg, device)
    return phi_full[ctx.node_idx].reshape(-1)


def gt_mu_band_at_time(
    ctx: SpeciesPhysicsCtx,
    time_index: int,
    device: torch.device,
) -> torch.Tensor:
    y = ctx.data.y[int(time_index)].to(device=device, dtype=torch.float32)
    mu_gt = ctx.phys_cfg.viscosity_nd_to_si(y[:, STATE_CHANNEL_MU_EFF_ND])
    return mu_gt[ctx.node_idx].reshape(-1)


def gt_mu_carreau_band_at_time(
    ctx: SpeciesPhysicsCtx,
    time_index: int,
    device: torch.device,
) -> torch.Tensor:
    """Bulk Carreau reference from GT mu anchor (per-node t0 cap)."""
    return ctx.mu_anchor_si.reshape(-1).clamp(min=1e-8)


def physics_readout_losses(
    log_state: torch.Tensor,
    ctx: SpeciesPhysicsCtx,
    train_mask: torch.Tensor,
    *,
    time_index: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (phi_loss, mu_loss) on band train_mask."""
    sp12 = band_log_state_to_species12(log_state, ctx.rest_band)
    phi_pred = differentiable_clot_phi_from_species12(sp12, ctx.bio_cfg)
    phi_gt = gt_phi_band_at_time(ctx, time_index, device)
    phi_l = gelation_phi_loss(phi_pred, phi_gt, train_mask)

    mu_l = log_state.sum() * 0.0
    if continuous_mu_loss_weight() > 0.0:
        mu_c = gt_mu_carreau_band_at_time(ctx, time_index, device)
        mu_pred = differentiable_mu_eff_from_species12(sp12, mu_c, phi_pred, ctx.bio_cfg)
        mu_gt = gt_mu_band_at_time(ctx, time_index, device)
        mu_l = gelation_mu_log_loss(mu_pred, mu_gt, train_mask)
    return phi_l, mu_l
