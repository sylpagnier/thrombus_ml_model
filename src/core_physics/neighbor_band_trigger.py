"""Neighbor-band species rollout + explicit physics trigger vs GT clot labels.

GT flow at each macro step; species from biochem teacher; clot from
``physics_mu_eff_si`` (Mat/FI gelation sigmoids) or optional kinematic gate.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import torch

from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND
from src.core_physics.clot_phi_simple import (
    cap_mu_eff_si,
    carreau_mu_si_from_uv,
    physics_mu_eff_si,
    physics_phi_from_mu,
    supervision_region_mask,
)
from src.evaluation.clot_shape_score import compute_clot_shape_metrics
from src.training.biochem_supervision_masks import (
    align_target_trajectory_to_eval_times,
    compute_supervised_species_log_mae,
    resolve_data_bio_supervision_mask,
)
from src.architecture.gnode_biochem import biochem_truth_node_mask


def _env_bool(name: str, default: bool = False) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def apply_neighbor_band_clot_phi_env() -> None:
    """Align biochem clot_band mask with clot-phi neighbor + dgamma slice defaults."""
    defaults = {
        "CLOT_PHI_MASK_MODE": "neighbor",
        "CLOT_PHI_CLOT_TOUCH_HOPS": "1",
        "CLOT_PHI_CENTER_EXCLUDE_FRAC": "0.10",
        "CLOT_PHI_DGAMMA_SLICE": "1",
        "CLOT_PHI_DGAMMA_REF_TIME": "0",
        "CLOT_PHI_DGAMMA_WALL_MIN_SI": "100",
        "CLOT_PHI_DGAMMA_OFFWALL_PCT": "80",
        "CLOT_PHI_MU_CAP_SI": "0.10",
        "CLOT_PHI_THRESH_SI": "0.055",
    }
    for key, val in defaults.items():
        os.environ.setdefault(key, val)


def apply_neighbor_band_species_train_env() -> None:
    """GT-flow neighbor-band species teacher defaults."""
    apply_neighbor_band_clot_phi_env()
    train_defaults = {
        "BIOCHEM_GT_KINE_VEL": "1",
        "BIOCHEM_GT_KINE_SKIP_DEQ": "1",
        "BIOCHEM_TEACHER_MU_RATIO_MAX": "1.0",
        "BIOCHEM_DATA_BIO_MASK_MODE": "neighbor",
        "BIOCHEM_SUPERVISION_MASK_TIMES": "union",
        "BIOCHEM_PASSIVE_ADR_BACKPROP": "0",
        "BIOCHEM_MU_DISABLE_EXPLICIT_GELATION": "1",
        "BIOCHEM_VAL_TIME_STRIDE": "10",
    }
    for key, val in train_defaults.items():
        os.environ.setdefault(key, val)


def apply_physics_trigger_baseline_env() -> None:
    """Explicit Mat/FI gelation trigger baseline (COMSOL spf.mu-faithful default)."""
    os.environ.setdefault("CLOT_PHI_PHYSICS_MU_BASE", "comsol_carreau")
    os.environ.setdefault("CLOT_PHI_PHYSICS_GAMMA_MODE", "max")
    os.environ.setdefault("CLOT_PHI_PHYSICS_MU_RATIO_MAX", "4")
    os.environ.setdefault("CLOT_PHI_PHYSICS_GELATION_GATE", "0")
    os.environ.setdefault("CLOT_TRIGGER_IC_PHI_ZERO", "1")
    if _env_bool("NEIGHBOR_BAND_PHYSICS_GATE", default=False):
        os.environ["CLOT_PHI_PHYSICS_GELATION_GATE"] = "1"


def _clot_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    if not bool(mask.any().item()):
        return {
            "clot_prec": 0.0,
            "clot_rec": 0.0,
            "clot_f1": 0.0,
            "pred_pos_frac": 0.0,
            "gt_pos_frac": 0.0,
        }
    pb = (pred[mask] > 0.5).float()
    tb = (target[mask] > 0.5).float()
    tp = float((pb * tb).sum().item())
    fp = float((pb * (1.0 - tb)).sum().item())
    fn = float(((1.0 - pb) * tb).sum().item())
    prec = tp / max(tp + fp, 1e-6)
    rec = tp / max(tp + fn, 1e-6)
    f1 = (2.0 * prec * rec) / max(prec + rec, 1e-6)
    n = float(mask.sum().item())
    return {
        "clot_prec": prec,
        "clot_rec": rec,
        "clot_f1": f1,
        "pred_pos_frac": float(pb.sum().item()) / max(n, 1.0),
        "gt_pos_frac": float(tb.sum().item()) / max(n, 1.0),
    }


def _log_mae_in_mask(pred: torch.Tensor, targ: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any().item()):
        return float("nan")
    p = pred.reshape(-1)[mask.reshape(-1)]
    t = targ.reshape(-1)[mask.reshape(-1)]
    return float((torch.log(p.clamp(min=1e-8)) - torch.log(t.clamp(min=1e-8))).abs().mean().item())


@dataclass
class NeighborBandStepMetrics:
    time_index: int
    species_fi_log_mae: float
    species_mat_log_mae: float
    species_mask_n: float
    band_f1: float
    band_rec: float
    band_prec: float
    band_log_mae: float
    clot_shape: float
    pred_pos_frac: float
    gt_pos_frac: float
    trigger_mode: str


def eval_neighbor_band_at_step(
    *,
    data,
    pred_y: torch.Tensor,
    target_y: torch.Tensor,
    time_index: int,
    device: torch.device,
    bio_cfg: BiochemConfig,
    phys_cfg: PhysicsConfig,
    kernels,
    species_log1p: torch.Tensor | None = None,
    trigger_mode: str = "physics",
) -> NeighborBandStepMetrics:
    """Compare species + physics-trigger mu vs GT at one macro time index."""
    t = max(0, min(int(time_index), int(target_y.shape[0]) - 1))
    y_gt = target_y[t].to(device=device)
    mu_gt_nd = y_gt[:, STATE_CHANNEL_MU_EFF_ND]
    mu_gt_si = phys_cfg.viscosity_nd_to_si(mu_gt_nd)
    mu_cap = cap_mu_eff_si(mu_gt_si)
    region = supervision_region_mask(data, device, mu_cap, phys_cfg)
    region_b = region.view(-1).bool()

    u_gt = y_gt[:, 0].reshape(-1).to(device=device, dtype=torch.float32)
    v_gt = y_gt[:, 1].reshape(-1).to(device=device, dtype=torch.float32)

    if species_log1p is None:
        species_log1p = pred_y[t, :, 4:16].to(device=device, dtype=torch.float32)
    else:
        species_log1p = species_log1p.to(device=device, dtype=torch.float32)

    mu_c = carreau_mu_si_from_uv(data, u_gt, v_gt, phys_cfg)
    mode = (trigger_mode or "physics").strip().lower()
    if mode in ("physics", "explicit", "oracle"):
        mu_pred = physics_mu_eff_si(
            mu_c,
            species_log1p,
            bio_cfg,
            device=device,
            data=data,
            u_nd=u_gt,
            v_nd=v_gt,
        )
    elif mode in ("gt_species", "gt"):
        mu_pred = physics_mu_eff_si(
            mu_c,
            y_gt[:, 4:16].to(device=device, dtype=torch.float32),
            bio_cfg,
            device=device,
            data=data,
            u_nd=u_gt,
            v_nd=v_gt,
        )
    else:
        raise ValueError(f"Unknown trigger_mode={trigger_mode!r}")

    use_soft = _env_bool("CLOT_PHI_SOFT_LABELS", default=True)
    phi_gt = physics_phi_from_mu(mu_cap, mu_c, region, phys_cfg, soft=use_soft)
    phi_pred = physics_phi_from_mu(mu_pred, mu_c, region, phys_cfg, soft=use_soft)

    n_nodes = int(data.num_nodes)
    truth_m = biochem_truth_node_mask(data, n_nodes, device)
    y_on_eval = target_y[t : t + 1]
    sup_mask = resolve_data_bio_supervision_mask(
        data=data,
        device=device,
        truth_mask=truth_m,
        target_series=y_on_eval,
        bio_cfg=bio_cfg,
        kernels=kernels,
        step_idx=0,
    )
    sp = compute_supervised_species_log_mae(
        pred_series=pred_y[t : t + 1],
        target_series=y_on_eval,
        node_mask=sup_mask,
    )
    band = _clot_metrics(phi_pred.view(-1), phi_gt.view(-1), region_b)
    band_log = _log_mae_in_mask(mu_pred, mu_cap, region_b)

    n_nodes = int(data.num_nodes)
    pred_state = torch.zeros(n_nodes, 4, device=device, dtype=torch.float32)
    gt_state = torch.zeros(n_nodes, 4, device=device, dtype=torch.float32)
    pred_state[:, 0] = u_gt
    pred_state[:, 1] = v_gt
    gt_state[:, 0] = u_gt
    gt_state[:, 1] = v_gt
    pred_state[:, STATE_CHANNEL_MU_EFF_ND] = phys_cfg.viscosity_si_to_nd(mu_pred.reshape(-1, 1)).reshape(-1)
    gt_state[:, STATE_CHANNEL_MU_EFF_ND] = mu_gt_nd.reshape(-1).to(dtype=torch.float32)

    shape = compute_clot_shape_metrics(
        pred_state=pred_state,
        gt_state=gt_state,
        edge_index=data.edge_index.to(device=device),
        phys_cfg=phys_cfg,
    )

    return NeighborBandStepMetrics(
        time_index=t,
        species_fi_log_mae=float(sp["species_fi_log_mae"]),
        species_mat_log_mae=float(sp["species_mat_log_mae"]),
        species_mask_n=float(sp["species_mask_n"]),
        band_f1=float(band["clot_f1"]),
        band_rec=float(band["clot_rec"]),
        band_prec=float(band["clot_prec"]),
        band_log_mae=float(band_log),
        clot_shape=float(shape["clot_shape"]),
        pred_pos_frac=float(band["pred_pos_frac"]),
        gt_pos_frac=float(band["gt_pos_frac"]),
        trigger_mode=mode,
    )


def eval_neighbor_band_rollout(
    *,
    data,
    pred_series: torch.Tensor,
    bio_cfg: BiochemConfig,
    phys_cfg: PhysicsConfig,
    kernels,
    device: torch.device,
    eval_times: torch.Tensor | None = None,
    trigger_mode: str = "physics",
    final_only: bool = False,
) -> dict[str, Any]:
    """Species + physics trigger metrics over validation rollout times."""
    if eval_times is None:
        target_series = data.y.to(device)
    else:
        target_series = align_target_trajectory_to_eval_times(data, eval_times, bio_cfg, device)

    n_steps = int(pred_series.shape[0])
    step_indices = [n_steps - 1] if final_only else list(range(n_steps))

    per_step: list[NeighborBandStepMetrics] = []
    for t in step_indices:
        per_step.append(
            eval_neighbor_band_at_step(
                data=data,
                pred_y=pred_series,
                target_y=target_series,
                time_index=t,
                device=device,
                bio_cfg=bio_cfg,
                phys_cfg=phys_cfg,
                kernels=kernels,
                trigger_mode=trigger_mode,
            )
        )

    def _mean(key: str) -> float:
        vals = [getattr(s, key) for s in per_step if math.isfinite(getattr(s, key))]
        if not vals:
            return float("nan")
        return float(sum(vals) / len(vals))

    last = per_step[-1] if per_step else None
    return {
        "per_step": [s.__dict__ for s in per_step],
        "mean_species_fi_log_mae": _mean("species_fi_log_mae"),
        "mean_species_mat_log_mae": _mean("species_mat_log_mae"),
        "mean_band_f1": _mean("band_f1"),
        "mean_band_log_mae": _mean("band_log_mae"),
        "mean_clot_shape": _mean("clot_shape"),
        "final_band_f1": float(last.band_f1) if last else float("nan"),
        "final_clot_shape": float(last.clot_shape) if last else float("nan"),
        "final_species_fi_log_mae": float(last.species_fi_log_mae) if last else float("nan"),
        "trigger_mode": trigger_mode,
    }
