"""Lightweight clot forecast pair utilities used by deploy clot readouts.

This module intentionally keeps only the pair-step API needed by the active
GraphSAGE biochem deploy stack and clot-phi helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import torch

from src.config import STATE_CHANNEL_MU_EFF_ND, BiochemConfig, PhysicsConfig
from src.utils import species_channels as sc
from src.core_physics.clot_growth_masks import resolve_ceiling_mask


@dataclass
class ClotForecastPairStep:
    t_in: int
    t_out: int
    features: torch.Tensor
    phi_gt: torch.Tensor
    loss_mask: torch.Tensor
    region: torch.Tensor
    mu_c_si: torch.Tensor
    mu_gt_cap: torch.Tensor
    mu_in_cap: torch.Tensor
    species_log_gt: torch.Tensor
    u_flow_nd: torch.Tensor
    v_flow_nd: torch.Tensor


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else int(default)


def clot_forecast_one_step_enabled() -> bool:
    return _env_bool("CLOT_FORECAST_ONE_STEP", True)


def clot_forecast_mu_carry_enabled() -> bool:
    return _env_bool("CLOT_FORECAST_MU_CARRY", False)


def clot_forecast_mu_carry_detach() -> bool:
    return _env_bool("CLOT_FORECAST_MU_CARRY_DETACH", True)


def clot_forecast_deploy_loss_enabled() -> bool:
    return _env_bool("CLOT_FORECAST_DEPLOY_LOSS", True)


def clot_forecast_mask_mode() -> str:
    return (os.environ.get("CLOT_FORECAST_MASK_MODE") or "ceiling").strip().lower()


def clot_forecast_pair_stride() -> int:
    return max(1, _env_int("CLOT_FORECAST_PAIR_STRIDE", 1))


def clot_forecast_pair_schedule() -> str:
    return (os.environ.get("CLOT_FORECAST_PAIR_SCHEDULE") or "rolling").strip().lower()


def snapshot_clot_forecast_config() -> dict[str, object]:
    return {
        "one_step": clot_forecast_one_step_enabled(),
        "mu_carry": clot_forecast_mu_carry_enabled(),
        "mu_carry_detach": clot_forecast_mu_carry_detach(),
        "deploy_loss": clot_forecast_deploy_loss_enabled(),
        "mask_mode": clot_forecast_mask_mode(),
        "pair_stride": clot_forecast_pair_stride(),
        "pair_schedule": clot_forecast_pair_schedule(),
    }


def resolve_forecast_deploy_mask_from_model(step: ClotForecastPairStep, phi_pred: torch.Tensor) -> torch.Tensor:
    """Deploy-eligible mask for loss/eval; defaults to the forecast region."""
    del phi_pred
    return step.loss_mask.reshape(-1).bool()


def build_deploy_eligible_phi_gt(step: ClotForecastPairStep) -> torch.Tensor:
    """GT phi restricted to deploy-eligible region."""
    m = step.loss_mask.reshape(-1).bool()
    out = torch.zeros_like(step.phi_gt.reshape(-1))
    out[m] = step.phi_gt.reshape(-1)[m]
    return out


def resolve_rollout_prev_mu_si(step: ClotForecastPairStep, prev_mu_si: torch.Tensor | None) -> torch.Tensor:
    """Resolve previous mu carry input for temporal rollouts."""
    if prev_mu_si is None or not clot_forecast_mu_carry_enabled():
        return step.mu_c_si.reshape(-1)
    mu = prev_mu_si.reshape(-1)
    if clot_forecast_mu_carry_detach():
        mu = mu.detach()
    return mu


def iter_forecast_pairs(
    n_times: int,
    *,
    time_stride: int = 1,
    pair_stride: int = 1,
    t_out_max: int | None = None,
) -> list[tuple[int, int]]:
    """Generate (t_in, t_out) pairs for one-step forecast style rollouts."""
    n = max(0, int(n_times))
    if n < 2:
        return []
    dt = max(1, int(time_stride))
    ps = max(1, int(pair_stride))
    tmax = (n - 1) if t_out_max is None else min(int(t_out_max), n - 1)
    pairs: list[tuple[int, int]] = []
    for t_out in range(dt, tmax + 1, ps):
        t_in = max(0, t_out - dt)
        pairs.append((t_in, t_out))
    return pairs


def _mu_clot_threshold_si(phys_cfg: PhysicsConfig) -> float:
    for key in ("mu_clot_threshold_si", "mu_clot_threshold"):
        if hasattr(phys_cfg, key):
            try:
                return float(getattr(phys_cfg, key))
            except Exception:
                pass
    return 0.055


def build_clot_forecast_pair_step(
    data,
    t_in: int,
    t_out: int,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
) -> ClotForecastPairStep:
    """Assemble a forecast pair slice from anchor timeline tensors."""
    t_in = int(max(0, t_in))
    t_out = int(max(0, t_out))
    tmax = int(data.y.shape[0]) - 1
    t_in = min(t_in, tmax)
    t_out = min(t_out, tmax)

    y_in = data.y[t_in].to(device=device, dtype=torch.float32)
    y_out = data.y[t_out].to(device=device, dtype=torch.float32)
    x = data.x.to(device=device, dtype=torch.float32)

    u_in = y_in[:, 0].reshape(-1, 1)
    v_in = y_in[:, 1].reshape(-1, 1)
    mu_in_nd = y_in[:, STATE_CHANNEL_MU_EFF_ND].reshape(-1, 1)
    mu_out_nd = y_out[:, STATE_CHANNEL_MU_EFF_ND].reshape(-1, 1)

    mu_in_si = phys_cfg.viscosity_nd_to_si(mu_in_nd).reshape(-1)
    mu_out_si = phys_cfg.viscosity_nd_to_si(mu_out_nd).reshape(-1)

    from src.core_physics.clot_phi_simple import (
        cap_mu_eff_si,
        clot_phi_thresh_si,
        gt_mu_anchor_cap_si,
    )

    mu_out_cap = cap_mu_eff_si(mu_out_si)
    anchor = gt_mu_anchor_cap_si(data, phys_cfg, device).reshape(-1)
    growth = (mu_out_cap.reshape(-1) - anchor).clamp(min=0.0)
    phi_gt = (growth >= clot_phi_thresh_si(phys_cfg)).to(dtype=torch.float32)
    loss_mask = resolve_ceiling_mask(data, device, bio_cfg).reshape(-1).to(dtype=torch.bool)
    region = loss_mask.clone()

    # Keep the same feature contract used by existing clot heads:
    # [x, u, v, mu_in, species_in(12)]
    species_in = (
        y_in[:, sc.SPECIES_BLOCK]
        if y_in.shape[1] >= sc.Y_WIDTH
        else torch.zeros(y_in.shape[0], sc.SPECIES_BLOCK_WIDTH, device=device)
    )
    features = torch.cat([x, u_in, v_in, mu_in_nd, species_in], dim=1)

    return ClotForecastPairStep(
        t_in=t_in,
        t_out=t_out,
        features=features,
        phi_gt=phi_gt,
        loss_mask=loss_mask,
        region=region,
        mu_c_si=mu_in_si,
        mu_gt_cap=mu_out_si,
        mu_in_cap=mu_in_si,
        species_log_gt=species_in,
        u_flow_nd=u_in.reshape(-1),
        v_flow_nd=v_in.reshape(-1),
    )

