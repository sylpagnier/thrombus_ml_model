"""Shared supervision masks for passive biochem (data bio vs ADR residuals)."""

from __future__ import annotations

import os
from typing import Optional

import torch

from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def supervision_mask_times_mode() -> str:
    """How clot-band / matched masks span the TBPTT window: last | union | per_step."""
    mode = (os.environ.get("BIOCHEM_SUPERVISION_MASK_TIMES") or "last").strip().lower()
    aliases = {
        "final": "last",
        "single": "last",
        "union_tbptt": "union",
        "all_steps": "union",
        "stepwise": "per_step",
        "per_timestep": "per_step",
    }
    mode = aliases.get(mode, mode)
    if mode not in ("last", "union", "per_step"):
        raise ValueError(
            f"Unknown BIOCHEM_SUPERVISION_MASK_TIMES={mode!r}; use last|union|per_step"
        )
    return mode


def resolve_clot_band_region_mask_at_step(
    *,
    data,
    device: torch.device,
    target_series: torch.Tensor,
    step_idx: int,
    kernels,
) -> torch.Tensor:
    """GT mu_eff at ``step_idx`` -> clot-phi supervision region (dgamma / neighbor band)."""
    from src.core_physics.clot_phi_simple import cap_mu_eff_si, supervision_region_mask

    mu_ch = STATE_CHANNEL_MU_EFF_ND
    t = max(0, min(int(step_idx), int(target_series.shape[0]) - 1))
    mu_nd = target_series[t, :, mu_ch].to(device=device, dtype=torch.float32)
    phys_cfg = kernels.core.cfg
    mu_si = phys_cfg.viscosity_nd_to_si(mu_nd)
    mu_cap_si = cap_mu_eff_si(mu_si)
    return supervision_region_mask(data, device, mu_cap_si, phys_cfg).view(-1).bool()


def resolve_clot_band_region_mask_union(
    *,
    data,
    device: torch.device,
    target_series: torch.Tensor,
    kernels,
) -> torch.Tensor:
    """OR of clot-band regions across all supervised timesteps in ``target_series``."""
    n_steps = int(target_series.shape[0])
    n = int(data.num_nodes)
    union = torch.zeros(n, dtype=torch.bool, device=device)
    for t in range(n_steps):
        union = union | resolve_clot_band_region_mask_at_step(
            data=data,
            device=device,
            target_series=target_series,
            step_idx=t,
            kernels=kernels,
        )
    return union


def resolve_data_bio_supervision_mask(
    *,
    data,
    device: torch.device,
    truth_mask: torch.Tensor,
    target_series: torch.Tensor,
    bio_cfg: BiochemConfig,
    kernels,
    step_idx: Optional[int] = None,
) -> torch.Tensor:
    """Same node set as ``L_Data_Bio`` (anchor + optional clot_band / wall_only)."""
    bio_mask_mode = (os.environ.get("BIOCHEM_DATA_BIO_MASK_MODE") or "global").strip().lower()
    # ``neighbor`` = clot-phi neighbor shell (wall + GT clot seeds + 1-hop); needs CLOT_PHI_* env.
    if bio_mask_mode in ("neighbor", "neighbor_band", "neighbor_shell"):
        bio_mask_mode = "clot_band"
    node_is_bio_target = truth_mask.view(-1).bool().to(device=device)
    times_mode = supervision_mask_times_mode()

    if bio_mask_mode in ("clot_band", "clotdec_band", "wall_clot_band", "decision_band"):
        try:
            if times_mode == "union":
                region_mask = resolve_clot_band_region_mask_union(
                    data=data,
                    device=device,
                    target_series=target_series,
                    kernels=kernels,
                )
            else:
                idx = (
                    int(target_series.shape[0]) - 1
                    if step_idx is None
                    else max(0, min(int(step_idx), int(target_series.shape[0]) - 1))
                )
                region_mask = resolve_clot_band_region_mask_at_step(
                    data=data,
                    device=device,
                    target_series=target_series,
                    step_idx=idx,
                    kernels=kernels,
                )
            if tuple(region_mask.shape) == tuple(node_is_bio_target.shape):
                node_is_bio_target = node_is_bio_target & region_mask
        except Exception:
            pass
    elif bio_mask_mode in ("wall_only", "wall"):
        if hasattr(data, "mask_wall") and data.mask_wall is not None:
            wall_mask = data.mask_wall.view(-1).to(device=device).bool()
            node_is_bio_target = node_is_bio_target & wall_mask

    return node_is_bio_target


def resolve_adr_node_mask(
    *,
    data,
    device: torch.device,
    truth_mask: torch.Tensor,
    target_series: torch.Tensor,
    bio_cfg: BiochemConfig,
    kernels,
    step_idx: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Node mask for bulk ADR residual means. ``None`` = all nodes (legacy global)."""
    mode = (os.environ.get("BIOCHEM_ADR_MASK_MODE") or "global").strip().lower()
    times_mode = supervision_mask_times_mode()

    if mode in ("global", "all", "none", ""):
        base: Optional[torch.Tensor] = None
    elif mode in ("match_data_bio", "data_bio", "same_as_data", "clot_band"):
        if times_mode == "union":
            bio_mask_mode = (os.environ.get("BIOCHEM_DATA_BIO_MASK_MODE") or "global").strip().lower()
            if bio_mask_mode in (
                "clot_band",
                "clotdec_band",
                "wall_clot_band",
                "decision_band",
            ):
                region = resolve_clot_band_region_mask_union(
                    data=data,
                    device=device,
                    target_series=target_series,
                    kernels=kernels,
                )
                base = truth_mask.view(-1).bool().to(device=device) & region
            else:
                base = resolve_data_bio_supervision_mask(
                    data=data,
                    device=device,
                    truth_mask=truth_mask,
                    target_series=target_series,
                    bio_cfg=bio_cfg,
                    kernels=kernels,
                    step_idx=step_idx,
                )
        else:
            base = resolve_data_bio_supervision_mask(
                data=data,
                device=device,
                truth_mask=truth_mask,
                target_series=target_series,
                bio_cfg=bio_cfg,
                kernels=kernels,
                step_idx=step_idx,
            )
    elif mode in ("anchor", "anchors", "truth"):
        base = truth_mask.view(-1).bool().to(device=device)
    elif mode in ("interior", "bulk_nonwall"):
        n = int(data.num_nodes)
        base = torch.ones(n, dtype=torch.bool, device=device)
        if hasattr(data, "mask_wall") and data.mask_wall is not None:
            base = base & (~data.mask_wall.view(-1).bool().to(device=device))
    else:
        raise ValueError(
            f"Unknown BIOCHEM_ADR_MASK_MODE={mode!r}; use global|match_data_bio|anchor|interior"
        )

    if base is None:
        if _env_truthy("BIOCHEM_ADR_EXCLUDE_WALL", default=False):
            n = int(data.num_nodes)
            m = torch.ones(n, dtype=torch.bool, device=device)
            if hasattr(data, "mask_wall") and data.mask_wall is not None:
                m = m & (~data.mask_wall.view(-1).bool().to(device=device))
            return m
        return None

    if _env_truthy("BIOCHEM_ADR_EXCLUDE_WALL", default=False):
        if hasattr(data, "mask_wall") and data.mask_wall is not None:
            base = base & (~data.mask_wall.view(-1).bool().to(device=device))

    return base


def adr_mask_use_per_timestep() -> bool:
    """Recompute ADR mask at each PDE step (``per_step`` times mode or explicit flag)."""
    if _env_truthy("BIOCHEM_ADR_MASK_PER_STEP", default=False):
        return True
    return supervision_mask_times_mode() == "per_step"


def align_target_trajectory_to_eval_times(
    data,
    eval_times: torch.Tensor,
    bio_cfg: BiochemConfig,
    device: torch.device,
) -> torch.Tensor:
    """Index ``data.y`` at the time stamps used for validation rollout (handles VAL_TIME_STRIDE)."""
    from src.utils.nondim import to_t_nd

    y = data.y.to(device)
    t_full = to_t_nd(bio_cfg.resolve_biochem_times(data, device), bio_cfg.t_final)
    eval_t = eval_times.to(device=device, dtype=t_full.dtype).view(-1)
    idxs = [(t_full - te).abs().argmin().item() for te in eval_t]
    return y[idxs]


def compute_supervised_species_log_mae(
    *,
    pred_series: torch.Tensor,
    target_series: torch.Tensor,
    node_mask: torch.Tensor,
) -> dict[str, float]:
    """Mean |log1p pred - log1p GT| for FI (bulk ch 8) and Mat (bulk ch 11) in bio slice 4:16."""
    if not bool(node_mask.any().item()):
        return {
            "species_fi_log_mae": float("nan"),
            "species_mat_log_mae": float("nan"),
            "species_mask_n": 0.0,
        }
    t_pred = int(pred_series.shape[0])
    t_targ = int(target_series.shape[0])
    if t_pred != t_targ:
        raise ValueError(
            f"species val: pred_series T={t_pred} != target_series T={t_targ}; "
            "align GT with align_target_trajectory_to_eval_times() before calling"
        )
    pred_bio = pred_series[:, node_mask, 4:16]
    targ_bio = target_series[:, node_mask, 4:16].to(device=pred_bio.device, dtype=pred_bio.dtype)
    fi_mae = (pred_bio[..., 8] - targ_bio[..., 8]).abs().mean()
    mat_mae = (pred_bio[..., 11] - targ_bio[..., 11]).abs().mean()
    return {
        "species_fi_log_mae": float(fi_mae.item()),
        "species_mat_log_mae": float(mat_mae.item()),
        "species_mask_n": float(int(node_mask.sum().item())),
    }


def passive_species_val_enabled() -> bool:
    return _env_truthy("BIOCHEM_PASSIVE_SPECIES_VAL", default=False)


def passive_species_val_only_enabled() -> bool:
    """Skip full mu/viz val rollout; one forward per anchor for FI/Mat + flow sanity."""
    return _env_truthy("BIOCHEM_PASSIVE_SPECIES_VAL_ONLY", default=False)


def passive_species_train_eval_enabled() -> bool:
    return _env_truthy("BIOCHEM_PASSIVE_SPECIES_TRAIN_EVAL", default=False)


def passive_step2_bridge_enabled() -> bool:
    """Step-2 data-only backward + optional scaled passive ADR (not COMPLEXITY_STEP=3)."""
    return _env_truthy("BIOCHEM_PASSIVE_STEP2_BRIDGE", default=False)


def passive_adr_backprop_weight() -> float:
    if not _env_truthy("BIOCHEM_PASSIVE_ADR_BACKPROP", default=False):
        return 0.0
    return max(float(os.environ.get("BIOCHEM_PASSIVE_ADR_WEIGHT", "1e-4") or "1e-4"), 0.0)


def adr_fast_transient_enabled() -> bool:
    return _env_truthy("BIOCHEM_ADR_FAST_TRANSIENT", default=False)


def passive_wall_in_backprop() -> bool:
    return _env_truthy("BIOCHEM_PASSIVE_WALL_BACKPROP", default=False)


def passive_mu_unlock_finetune_enabled() -> bool:
    """Post-probe finetune: MU_LOG + wall/high-mu weights, init from unlock-best ckpt."""
    return _env_truthy("BIOCHEM_PASSIVE_MU_UNLOCK_FINETUNE", default=False)


def passive_mu_unlock_enabled() -> bool:
    """MU_LOG-only probe or finetune after passive species align (no ``passive_transport`` preset)."""
    return _env_truthy("BIOCHEM_PASSIVE_MU_UNLOCK", default=False) or passive_mu_unlock_finetune_enabled()


def passive_mu_unlock_freeze_bio_train() -> bool:
    """Freeze bio encoder/decoder + ODE during mu-unlock (default on when unlock enabled)."""
    if not passive_mu_unlock_enabled():
        return False
    raw = os.environ.get("BIOCHEM_PASSIVE_MU_UNLOCK_FREEZE_BIO")
    if raw is None or str(raw).strip() == "":
        return True
    return _env_truthy("BIOCHEM_PASSIVE_MU_UNLOCK_FREEZE_BIO", default=True)
