"""Clot forecast ladder (R0-R1): one-step mu(t) -> mu(t+dt) on GT flow."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND
from src.core_physics.clot_phi_simple import (
    ClotPhiStepBatch,
    build_clot_phi_step,
    cap_mu_eff_si,
    clot_phi_thresh_si,
    supervision_region_mask,
)

if TYPE_CHECKING:
    pass


def _env_bool(name: str, default: bool = False) -> bool:
    raw = (os.environ.get(name) or ("1" if default else "0")).strip().lower()
    return raw in ("1", "true", "yes", "on")


def clot_forecast_mode() -> str:
    """``one_step`` = features @ t_in, labels @ t_out; empty = legacy per-frame."""
    return (os.environ.get("CLOT_FORECAST_MODE") or "").strip().lower()


def clot_forecast_one_step_enabled() -> bool:
    return clot_forecast_mode() in ("one_step", "1", "next", "forecast")


def clot_forecast_input_mu_enabled() -> bool:
    """Prong B: append log(mu_t) from the input frame to node features."""
    return _env_bool("CLOT_FORECAST_INPUT_MU", False)


def clot_forecast_pair_stride() -> int:
    """COMSOL index gap between input and target (1 = adjacent snapshots)."""
    try:
        return max(1, int(os.environ.get("CLOT_FORECAST_PAIR_STRIDE", "1") or "1"))
    except ValueError:
        return 1


def clot_forecast_extra_feature_dim() -> int:
    if not clot_forecast_one_step_enabled():
        return 0
    return 1 if clot_forecast_input_mu_enabled() else 0


def append_forecast_input_features(
    feats: torch.Tensor,
    log_mu_t: torch.Tensor,
    *,
    n_nodes: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Concatenate ``log(mu_t)`` when ``CLOT_FORECAST_INPUT_MU=1``."""
    if not clot_forecast_input_mu_enabled():
        return feats
    col = log_mu_t.reshape(-1, 1).to(device=device, dtype=dtype)
    if col.shape[0] != n_nodes:
        col = col[:n_nodes]
    return torch.cat([feats, col], dim=1)


def build_clot_forecast_pair_step(
    data,
    t_in: int,
    t_out: int,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
) -> ClotPhiStepBatch:
    """One-step pair: GT flow/features at ``t_in``, supervision labels at ``t_out``."""
    step_in = build_clot_phi_step(data, int(t_in), phys_cfg, bio_cfg, device)
    y_out = data.y[int(t_out)].to(device)
    mu_gt_out = phys_cfg.viscosity_nd_to_si(y_out[:, STATE_CHANNEL_MU_EFF_ND])
    mu_cap_out = cap_mu_eff_si(mu_gt_out)
    region_out = supervision_region_mask(data, device, mu_cap_out, phys_cfg)
    from src.core_physics.clot_phi_simple import phi_gt_binary, phi_gt_soft

    use_soft = _env_bool("CLOT_PHI_SOFT_LABELS", False)
    if use_soft:
        phi_gt_out = phi_gt_soft(mu_cap_out, step_in.mu_c_si, region_out)
    else:
        phi_gt_out = phi_gt_binary(mu_cap_out, region_out, phys_cfg)

    feats = step_in.features
    log_mu_in = torch.log(step_in.mu_gt_cap.clamp(min=1e-8))
    feats = append_forecast_input_features(
        feats,
        log_mu_in,
        n_nodes=int(data.num_nodes),
        device=device,
        dtype=feats.dtype,
    )
    loss_mask = region_out.bool()
    species_out = y_out[:, 4:16].to(device=device, dtype=torch.float32)
    return ClotPhiStepBatch(
        features=feats,
        phi_gt=phi_gt_out,
        mu_c_si=step_in.mu_c_si,
        mu_gt_cap=mu_cap_out,
        region=region_out,
        loss_mask=loss_mask,
        species_log_gt=species_out,
        u_flow_nd=step_in.u_flow_nd,
        v_flow_nd=step_in.v_flow_nd,
    )


def snapshot_clot_forecast_config() -> dict[str, object]:
    return {
        "forecast_mode": clot_forecast_mode() or "legacy",
        "forecast_one_step": clot_forecast_one_step_enabled(),
        "forecast_input_mu": clot_forecast_input_mu_enabled(),
        "forecast_pair_stride": clot_forecast_pair_stride(),
    }
