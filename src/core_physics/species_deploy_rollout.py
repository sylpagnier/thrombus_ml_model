"""Deploy-faithful species rollout helpers (no GT leak at inference).

Verified on biochem anchors: FI/Mat log1p ND at t=0 is uniformly 0 on the wall band
(COMSOL common IC). Matches ``resting_species_log_nd`` FI/Mat channels.
"""

from __future__ import annotations

import os
from typing import Literal

import torch

from src.config import PhysicsConfig
from src.utils import species_channels as sc
from src.training.biochem_species_scope import (
    FI_CHANNEL,
    MAT_CHANNEL,
    pushforward_state_bulk_indices,
    pushforward_state_dim,
    scatter_log_state_to_species_block,
)
from src.core_physics.species_snapshot_gnn import species_log_targets
from src.core_physics.t0_rung4_ladder import resting_species_log_nd

PinOtherSpecies = Literal["rest", "gt"]

_pred_uv_cache: tuple[torch.Tensor, torch.Tensor] | None = None
_pred_uv_key: tuple[int, int, int] | None = None
_kine_model = None


def species_rollout_deploy_faithful() -> bool:
    raw = (os.environ.get("SPECIES_ROLLOUT_DEPLOY_FAITHFUL") or "1").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _normalize_vel_source(raw: str) -> str:
    r = raw.strip().lower()
    if r in ("pred", "kine", "kinematics", "deq", "gino"):
        return "kinematics"
    if r in ("coupled", "mu_coupled", "feedback"):
        return "coupled"
    return "gt"


def species_rollout_vel_source() -> str:
    default = "kinematics" if species_rollout_deploy_faithful() else "gt"
    raw = (os.environ.get("SPECIES_ROLLOUT_VEL_SOURCE") or default).strip().lower()
    return _normalize_vel_source(raw)


def species_train_vel_source() -> str:
    """Velocity for species unroll loss (may use GT to save VRAM during train)."""
    raw = (os.environ.get("SPECIES_TRAIN_VEL_SOURCE") or "gt").strip()
    if raw:
        return _normalize_vel_source(raw)
    return species_rollout_vel_source()


def species_rollout_pin_other() -> PinOtherSpecies:
    if not species_rollout_deploy_faithful():
        raw = (os.environ.get("SPECIES_ROLLOUT_PIN_OTHER") or "gt").strip().lower()
    else:
        raw = (os.environ.get("SPECIES_ROLLOUT_PIN_OTHER") or "rest").strip().lower()
    return "gt" if raw == "gt" else "rest"


def species_rollout_ic_source() -> str:
    default = "resting" if species_rollout_deploy_faithful() else "gt"
    return (os.environ.get("SPECIES_ROLLOUT_IC_SOURCE") or default).strip().lower()


def _graph_key(data) -> tuple[int, int, int]:
    n = int(data.num_nodes)
    e = int(data.edge_index.shape[1])
    ptr = 0
    if hasattr(data, "x") and torch.is_tensor(data.x) and data.x.numel() > 0:
        ptr = int(data.x.untyped_storage().data_ptr())
    return (n, e, ptr)


def reset_species_rollout_flow_cache() -> None:
    global _pred_uv_cache, _pred_uv_key, _kine_model
    _pred_uv_cache = None
    _pred_uv_key = None
    _kine_model = None


def deploy_fimat_log_init(
    data,
    device: torch.device,
    node_idx: torch.Tensor,
    *,
    override: torch.Tensor | None = None,
) -> torch.Tensor:
    """Log-ND initial state on wall-band nodes (deploy: resting zeros, not GT)."""
    idx = node_idx.reshape(-1)
    sd = pushforward_state_dim()
    if override is not None:
        st = override.to(device=device, dtype=torch.float32).reshape(-1, sd)
        if st.shape[0] == idx.numel():
            return st.clone()
    ic = species_rollout_ic_source()
    if ic in ("gt", "truth", "comsol"):
        return species_log_targets(data, 0, device)[idx]
    rest = resting_species_log_nd(data, device)
    bulk = pushforward_state_bulk_indices()
    cols = [rest[idx, int(ch)] for ch in bulk]
    return torch.stack(cols, dim=-1)


def band_speed_from_uv(
    u_nd: torch.Tensor,
    v_nd: torch.Tensor,
    node_idx: torch.Tensor,
) -> torch.Tensor:
    speed = torch.sqrt(u_nd.reshape(-1) ** 2 + v_nd.reshape(-1) ** 2 + 1e-12)
    spd = speed[node_idx.reshape(-1)]
    mx = spd.max().clamp(min=1e-6)
    return (spd / mx).clamp(0.0, 1.0)


@torch.no_grad()
def resolve_species_rollout_uv(
    data,
    time_index: int,
    device: torch.device,
    *,
    for_training: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ND [u, v] for species rollout vel-decay (pred kine by default in deploy mode)."""
    src = species_train_vel_source() if for_training else species_rollout_vel_source()
    ti = max(0, min(int(time_index), int(data.y.shape[0]) - 1))
    y = data.y[ti].to(device=device, dtype=torch.float32)
    u_gt = y[:, 0]
    v_gt = y[:, 1]
    if src == "gt":
        return u_gt, v_gt
    if src == "coupled":
        from src.core_physics.clot_coupled_rollout import get_coupled_uv

        coupled = get_coupled_uv(data, device)
        if coupled is not None:
            return coupled
    from src.inference.corrector_coupling import corrector_coupling_enabled, get_coupled_flow

    if corrector_coupling_enabled():
        coupled = get_coupled_flow(data, device)
        if coupled is not None:
            return coupled
            
    if hasattr(data, "u0_pred") and data.u0_pred is not None:
        return data.u0_pred.to(device=device, dtype=torch.float32), data.v0_pred.to(device=device, dtype=torch.float32)
        
    global _pred_uv_cache, _pred_uv_key, _kine_model
    key = _graph_key(data)
    if _pred_uv_cache is None or _pred_uv_key != key:
        from src.utils.kinematics_inference import (
            load_kinematics_predictor,
            predict_kinematics,
            resolve_kinematics_checkpoint,
        )

        ckpt = (os.environ.get("KINEMATICS_CHECKPOINT") or os.environ.get("CLOT_PHI_KINE_CKPT") or "").strip()
        if not ckpt:
            ckpt = str(resolve_kinematics_checkpoint())
        if _kine_model is None:
            _kine_model = load_kinematics_predictor(
                ckpt,
                device,
                phys_cfg=PhysicsConfig(phase="kinematics"),
            )
        batch = data.to(device)
        pred = predict_kinematics(_kine_model, batch)
        _pred_uv_cache = (pred[:, 0], pred[:, 1])
        _pred_uv_key = key
    return _pred_uv_cache


def band_speed_for_rollout(
    data,
    time_index: int,
    device: torch.device,
    node_idx: torch.Tensor,
    *,
    u_nd: torch.Tensor | None = None,
    v_nd: torch.Tensor | None = None,
    for_training: bool = False,
) -> torch.Tensor:
    if u_nd is not None and v_nd is not None:
        return band_speed_from_uv(u_nd, v_nd, node_idx)
    u, v = resolve_species_rollout_uv(data, time_index, device, for_training=for_training)
    return band_speed_from_uv(u, v, node_idx)


def alloc_species_y_series(data, device: torch.device) -> torch.Tensor:
    """Allocate species timeline buffer without copying GT ``data.y``."""
    n_steps = int(data.y.shape[0])
    n_nodes = int(data.num_nodes)
    n_ch = int(data.y.shape[2]) if data.y.ndim >= 3 else 16
    return torch.zeros(n_steps, n_nodes, n_ch, device=device, dtype=torch.float32)


def resting_species_block(data, device: torch.device) -> torch.Tensor:
    """12-ch log1p ND block (FI/Mat at plasma rest; only those two are later updated)."""
    return resting_species_log_nd(data, device)


def pin_species_block(
    data,
    time_index: int,
    device: torch.device,
    *,
    pin_other: PinOtherSpecies | None = None,
) -> torch.Tensor:
    """Non-modeled species channels for one timestep."""
    mode = pin_other or species_rollout_pin_other()
    if mode == "gt":
        return data.y[int(time_index), :, sc.SPECIES_BLOCK].to(device=device, dtype=torch.float32).clone()
    return resting_species_log_nd(data, device).clone()
