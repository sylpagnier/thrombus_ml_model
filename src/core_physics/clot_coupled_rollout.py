"""Step 5b/5c: per-macro-step mu -> GINO-DEQ feedback for temporal clot rollout."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_continuous_time import rollout_time_indices
from src.core_physics.clot_growth_masks import resolve_ceiling_mask
from src.core_physics.clot_phi_rollout import KinematicsUvProvider
from src.core_physics.clot_temporal_growth_rules import (
    TemporalGrowthRuleConfig,
    _resolve_pool_risk,
    predict_phi_temporal_at_time,
    reset_temporal_kinematics_cache,
    temporal_vel_source,
)
from src.training.clot_ml_step5a_mu_readout import mu_eff_carreau_blend_from_phi

if TYPE_CHECKING:
    pass

_coupled_uv: tuple[torch.Tensor, torch.Tensor] | None = None
_coupled_uv_key: tuple[int, int, int] | None = None


def _graph_key(data) -> tuple[int, int, int]:
    n = int(data.num_nodes)
    e = int(data.edge_index.shape[1])
    ptr = 0
    if hasattr(data, "x") and torch.is_tensor(data.x) and data.x.numel() > 0:
        ptr = int(data.x.untyped_storage().data_ptr())
    return (n, e, ptr)


def reset_coupled_uv_cache() -> None:
    global _coupled_uv, _coupled_uv_key
    _coupled_uv = None
    _coupled_uv_key = None


def set_coupled_uv_cache(data, u: torch.Tensor, v: torch.Tensor) -> None:
    global _coupled_uv, _coupled_uv_key
    _coupled_uv = (u.detach(), v.detach())
    _coupled_uv_key = _graph_key(data)


def get_coupled_uv(data, device: torch.device) -> tuple[torch.Tensor, torch.Tensor] | None:
    if _coupled_uv is None or _coupled_uv_key != _graph_key(data):
        return None
    u, v = _coupled_uv
    return u.to(device=device), v.to(device=device)


def coupled_vel_mode_enabled() -> bool:
    return temporal_vel_source() == "coupled"


@torch.no_grad()
def _init_coupled_uv_from_frozen_kine(data, device: torch.device) -> None:
    prev = os.environ.get("CLOT_TEMPORAL_VEL_SOURCE")
    os.environ["CLOT_TEMPORAL_VEL_SOURCE"] = "kinematics"
    reset_temporal_kinematics_cache()
    from src.core_physics.clot_temporal_growth_rules import _resolve_uv_for_temporal_risk

    u, v = _resolve_uv_for_temporal_risk(data, 0, device)
    set_coupled_uv_cache(data, u, v)
    if prev is not None:
        os.environ["CLOT_TEMPORAL_VEL_SOURCE"] = prev
    else:
        os.environ.pop("CLOT_TEMPORAL_VEL_SOURCE", None)


@torch.no_grad()
def rollout_temporal_phi_coupled(
    data,
    cfg: TemporalGrowthRuleConfig,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    time_stride: int = 1,
    sim_end_scale: float | None = None,
    kine_provider: KinematicsUvProvider | None = None,
) -> dict[int, torch.Tensor]:
    """Temporal rule rollout with mu-prior DEQ refresh after each commit (Step 5b/5c)."""
    if cfg.kind == "threshold_accum":
        raise NotImplementedError("coupled rollout supports progressive shell kinds only")

    os.environ["CLOT_TEMPORAL_VEL_SOURCE"] = "coupled"
    reset_coupled_uv_cache()
    reset_temporal_kinematics_cache()
    _init_coupled_uv_from_frozen_kine(data, device)

    provider = kine_provider or KinematicsUvProvider(device)
    from src.core_physics.clot_continuous_time import feature_time_index

    n_times = int(data.y.shape[0])
    t_indices = rollout_time_indices(data, time_stride=time_stride, sim_end_scale=sim_end_scale)
    t_final = n_times - 1
    ceiling = resolve_ceiling_mask(data, device, bio_cfg)
    phi_by_t: dict[int, torch.Tensor] = {}
    phi_prev: torch.Tensor | None = None
    scale = float(sim_end_scale if sim_end_scale is not None else 1.0)

    for t_out in t_indices:
        reset_temporal_kinematics_cache()
        pool, risk = _resolve_pool_risk(
            data,
            device=device,
            bio_cfg=bio_cfg,
            ceiling=ceiling,
            cfg=cfg,
            t_out=feature_time_index(data, int(t_out)),
        )
        phi = predict_phi_temporal_at_time(
            data,
            t_out,
            device=device,
            bio_cfg=bio_cfg,
            cfg=cfg,
            ceiling=ceiling,
            risk=risk,
            phi_prev=phi_prev,
            t_final=t_final,
            sim_end_scale=scale,
        )
        phi_by_t[int(t_out)] = phi
        phi_prev = phi

        mu = mu_eff_carreau_blend_from_phi(
            data,
            phi,
            int(t_out),
            device=device,
            phys_cfg=phys_cfg,
            bio_cfg=bio_cfg,
        )
        batch = data.to(device)
        u, v = provider.uv_nd_from_mu_si(batch, mu)
        set_coupled_uv_cache(data, u, v)

    return phi_by_t
