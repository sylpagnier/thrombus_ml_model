"""Compatibility shim for legacy t0 mu physics helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from src.config import STATE_CHANNEL_MU_EFF_ND, BiochemConfig, PhysicsConfig
from src.utils import species_channels as sc

_ROLLOUT_CACHE: dict[tuple, dict[int, dict[str, torch.Tensor]]] = {}


@dataclass
class T0ClotStep:
    phi: torch.Tensor
    mu_pred_si: torch.Tensor
    mu_gt_si: torch.Tensor | None = None


def gt_mu_anchor_cap_si(
    data,
    time_index: int,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return GT capped mu in SI units from anchor timeline."""
    del bio_cfg
    dev = device or (data.y.device if torch.is_tensor(data.y) else torch.device("cpu"))
    ti = max(0, min(int(time_index), int(data.y.shape[0]) - 1))
    y = data.y[ti].to(device=dev, dtype=torch.float32)
    mu_nd = y[:, STATE_CHANNEL_MU_EFF_ND]
    mu_si = phys_cfg.viscosity_nd_to_si(mu_nd)
    return torch.clamp(mu_si, min=1e-8)


def gt_clot_phi_at_time(
    data,
    time_index: int,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Binary GT clot target: growth-only ``relu(mu_eff(t) - mu_eff(t=0)) >= thresh``."""
    from src.core_physics.clot_phi_simple import (
        cap_mu_eff_si,
        clot_phi_thresh_si,
        gt_mu_anchor_cap_si,
    )

    dev = device or (data.y.device if torch.is_tensor(data.y) else torch.device("cpu"))
    ti = max(0, min(int(time_index), int(data.y.shape[0]) - 1))
    y = data.y[ti].to(device=dev, dtype=torch.float32)
    mu = cap_mu_eff_si(phys_cfg.viscosity_nd_to_si(y[:, STATE_CHANNEL_MU_EFF_ND])).reshape(-1)
    anchor = gt_mu_anchor_cap_si(data, phys_cfg, dev).reshape(-1)
    growth = (mu - anchor).clamp(min=0.0)
    return (growth >= clot_phi_thresh_si(phys_cfg)).to(dtype=torch.float32)


def resolve_t0_gamma_mode() -> str:
    return (os.environ.get("T0_GAMMA_MODE") or "proxy").strip().lower()


def resolve_t0_flow_uv_nd(
    data,
    time_index: int,
    phys_cfg: PhysicsConfig | torch.device | None = None,
    device: torch.device | None = None,
    *,
    flow_source: str = "gt",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return flow [u,v] ND at requested time index."""
    del flow_source
    if isinstance(phys_cfg, torch.device):
        device, phys_cfg = phys_cfg, None
    dev = device or (data.y.device if torch.is_tensor(data.y) else torch.device("cpu"))
    ti = max(0, min(int(time_index), int(data.y.shape[0]) - 1))
    y = data.y[ti].to(device=dev, dtype=torch.float32)
    return y[:, 0].reshape(-1), y[:, 1].reshape(-1)


def predict_mu_si_at_time(
    data,
    time_index: int,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig | None = None,
    *,
    device: torch.device | None = None,
    gamma_mode: str | None = None,
    flow_source: str = "gt",
    pred_species_series: torch.Tensor | None = None,
) -> T0ClotStep:
    """Predict mu_si at one time index (deploy / eval hook)."""
    bio = bio_cfg or BiochemConfig(phase="biochem")
    dev = device or (data.y.device if torch.is_tensor(data.y) else torch.device("cpu"))
    _, step = predict_clot_phi_at_time(
        data,
        int(time_index),
        phys_cfg,
        bio,
        dev,
        gamma_mode=gamma_mode or resolve_t0_gamma_mode(),
        flow_source=flow_source,
        pred_species_series=pred_species_series,
    )
    return step


@torch.no_grad()
def predict_clot_phi_at_time(
    data,
    time_index: int,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig | None = None,
    device: torch.device | None = None,
    *,
    gamma_mode: str = "proxy",
    flow_source: str = "gt",
    pred_species_series: torch.Tensor | None = None,
    gelation_beta: torch.Tensor | float | None = None,
    nucleation: bool = True,
) -> tuple[torch.Tensor, T0ClotStep]:
    """Physics clot trigger at one time index; returns (phi_raw, step)."""
    bio = bio_cfg or BiochemConfig(phase="biochem")
    dev = device or (data.y.device if torch.is_tensor(data.y) else torch.device("cpu"))
    t = int(time_index)
    beta_key = None
    if gelation_beta is not None:
        beta_key = float(gelation_beta.item() if torch.is_tensor(gelation_beta) else gelation_beta)
    cache_key = (id(data), id(pred_species_series), flow_source, gamma_mode, beta_key)
    if t == 0:
        _ROLLOUT_CACHE.pop(cache_key, None)
    traj = _ROLLOUT_CACHE.get(cache_key)
    if traj is None:
        traj = rollout_t0_clot_phi(
            data,
            phys_cfg,
            bio,
            dev,
            gamma_mode=gamma_mode,
            flow_source=flow_source,
            pred_species_series=pred_species_series,
            nucleation=nucleation,
            gelation_beta=gelation_beta,
        )
        _ROLLOUT_CACHE[cache_key] = traj
    entry = traj[t]
    phi_raw = entry["phi_raw"]
    mu_pred = entry["mu"]
    if gelation_beta is not None and pred_species_series is not None:
        from src.core_physics.species_viscosity_calibration import predict_mu_at_time_with_beta

        mu_pred, mu_gt = predict_mu_at_time_with_beta(
            data,
            pred_species_series,
            gelation_beta,
            t,
            phys_cfg=phys_cfg,
            bio_cfg=bio,
            device=dev,
            gamma_mode=gamma_mode,
        )
    else:
        mu_gt = gt_mu_anchor_cap_si(data, t, phys_cfg, bio_cfg=bio, device=dev)
    step = T0ClotStep(phi=entry["phi"], mu_pred_si=mu_pred.reshape(-1), mu_gt_si=mu_gt.reshape(-1))
    return phi_raw.reshape(-1), step


@torch.no_grad()
def rollout_t0_clot_phi(
    data,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    *,
    gamma_mode: str = "proxy",
    flow_source: str = "gt",
    pred_species_series: torch.Tensor | None = None,
    nucleation: bool = True,
    nucleation_hops: int = 1,
    gelation_beta: torch.Tensor | None = None,
) -> dict[int, dict[str, torch.Tensor]]:
    """Physics clot trigger rollout from a predicted species timeline."""
    del gamma_mode, flow_source, gelation_beta
    from src.core_physics.clot_phi_simple import build_clot_phi_step
    from src.core_physics.clot_trigger_rollout import _project_step_phi
    from src.core_physics.clot_trigger_rollout import clot_trigger_forward_seed_mode
    from src.training.clot_trigger_stack import forward_physics_trigger_phi

    n_steps = int(data.y.shape[0])
    out: dict[int, dict[str, torch.Tensor]] = {}
    phi_prev: torch.Tensor | None = None
    phi_by_t: dict[int, torch.Tensor] = {}
    seed = clot_trigger_forward_seed_mode()
    mu_anchor_si: torch.Tensor | None = None

    for t in range(n_steps):
        species_log1p = None
        if pred_species_series is not None:
            species_log1p = pred_species_series[t, :, sc.SPECIES_BLOCK].to(device=device, dtype=torch.float32)
        step = build_clot_phi_step(
            data,
            t,
            phys_cfg,
            bio_cfg,
            device,
            species_log_override=species_log1p,
        )
        phi_raw, mu_raw = forward_physics_trigger_phi(
            step,
            data,
            phys_cfg=phys_cfg,
            bio_cfg=bio_cfg,
            device=device,
            species_log1p=species_log1p,
            use_soft=True,
            apply_region=False,
            time_index=int(t),
            mu_anchor_si=mu_anchor_si,
        )
        if mu_anchor_si is None:
            mu_anchor_si = mu_raw.reshape(-1).clone()
        phi_proj = _project_step_phi(
            phi_raw,
            phi_prev,
            data,
            t,
            device=device,
            phys_cfg=phys_cfg,
            bio_cfg=bio_cfg,
            phi_pred_by_time=phi_by_t,
            growth_seed=seed,
        ) if nucleation else phi_raw.reshape(-1)
        phi_by_t[int(t)] = phi_proj.reshape(-1)
        out[int(t)] = {
            "phi_raw": phi_raw.reshape(-1),
            "phi": phi_proj.reshape(-1),
            "mu": mu_raw.reshape(-1),
        }
        phi_prev = phi_proj.reshape(-1)
    del nucleation_hops
    return out


def _mu_log_mae(
    mu_pred: torch.Tensor,
    mu_gt: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> float:
    """Mean absolute log error between predicted and GT viscosity (Pa*s)."""
    p = mu_pred.reshape(-1).clamp(min=1e-8)
    t = mu_gt.reshape(-1).clamp(min=1e-8)
    if mask is not None:
        m = mask.reshape(-1).bool()
        if not bool(m.any().item()):
            return 0.0
        p, t = p[m], t[m]
    return float((torch.log(p) - torch.log(t)).abs().mean().item())


def _pearson(mu_pred: torch.Tensor, mu_gt: torch.Tensor) -> float:
    """Pearson r on flattened viscosity fields."""
    p = mu_pred.reshape(-1).to(dtype=torch.float64)
    t = mu_gt.reshape(-1).to(dtype=torch.float64)
    if int(p.numel()) < 2:
        return 0.0
    p = p - p.mean()
    t = t - t.mean()
    denom = p.norm() * t.norm()
    if float(denom) < 1e-12:
        return 0.0
    return float((p * t).sum().item() / float(denom))


def _region_masks(
    data,
    time_index: int,
    phys_cfg: PhysicsConfig,
    device: torch.device,
    mu_gt: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Evaluation region tags for viscosity calibration (growth / wall)."""
    del mu_gt
    from src.core_physics.clot_growth_masks import gt_growth_commit_mask_at_time

    n = int(data.num_nodes)
    growth = gt_growth_commit_mask_at_time(data, int(time_index), phys_cfg, device)
    if hasattr(data, "mask_wall") and data.mask_wall is not None:
        wall = data.mask_wall.view(-1).to(device=device).bool()
    else:
        wall = torch.zeros(n, dtype=torch.bool, device=device)
    return {"growth": growth.reshape(-1).bool(), "wall": wall.reshape(-1).bool()}

