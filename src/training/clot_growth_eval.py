"""Trajectory-aware eval for clot growth rollouts (V3+)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_forecast import build_clot_forecast_pair_step, iter_forecast_pairs
from src.core_physics.clot_nucleation_mask import resolve_wall_mask
from src.core_physics.clot_phi_simple import _hop_distance_from_seed
from src.core_physics.clot_temporal_growth_rules import (
    _shape_from_phi_at_time,
    _time_frac_at_index,
    deploy_score_from_eval_row,
)
from src.evaluation.clot_shape_score import graph_hop_distance_from_seeds
from src.evaluation.clot_relaxed_metrics import legacy_clot_f1_metrics as _clot_metrics


def _mean_key(rows: list[dict], key: str) -> float:
    if not rows:
        return float("nan")
    return float(sum(float(r[key]) for r in rows) / len(rows))


def _wall_ring_frac(
    data,
    phi: torch.Tensor,
    phi_gt: torch.Tensor,
    *,
    device: torch.device,
    commit_thresh: float = 0.5,
) -> float:
    """Fraction of pred+ wall nodes that are >3 hops from any GT+ (ring-paint proxy)."""
    n = int(data.num_nodes)
    pred = (phi.reshape(-1).float() >= commit_thresh).cpu().numpy()
    gt = (phi_gt.reshape(-1).float() >= commit_thresh).cpu().numpy()
    wall = resolve_wall_mask(data, device).reshape(-1).cpu().numpy().astype(bool)
    pred_wall = pred & wall
    if not bool(pred_wall.any()):
        return 0.0
    hop = graph_hop_distance_from_seeds(
        data.edge_index, n, gt.astype(bool), max_hops=8
    )
    distant = pred_wall & (hop > 3)
    return float(distant.sum()) / float(max(pred_wall.sum(), 1))


@torch.no_grad()
def eval_phi_trajectory_on_anchor(
    phi_by_t: dict[int, torch.Tensor],
    data,
    *,
    anchor: str,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    pair_stride: int = 1,
    rule_tag: str = "growth",
) -> dict[str, Any]:
    """Full-timeline eval: mean band F1, mean clot_shape, early recall, wall-ring."""
    pairs = iter_forecast_pairs(int(data.y.shape[0]), time_stride=1, pair_stride=pair_stride)
    if not pairs:
        return {"anchor": anchor, "rule": rule_tag, "n_pairs": 0}

    rows: list[dict[str, float]] = []
    shape_rows: list[float] = []
    recall_rows: list[float] = []
    for t_in, t_out in pairs:
        phi = phi_by_t[int(t_out)]
        step = build_clot_forecast_pair_step(data, t_in, t_out, phys_cfg, bio_cfg, device)
        band = _clot_metrics(phi, step.phi_gt, step.loss_mask)
        shape_m = _shape_from_phi_at_time(
            data, phi, t_in, int(t_out), device=device, phys_cfg=phys_cfg, bio_cfg=bio_cfg
        )
        shape_v = float(shape_m.get("clot_shape", float("nan")))
        shape_rows.append(shape_v)
        gt_pos = float(band.get("gt_pos_frac", 0.0))
        pred_pos = float(band.get("pred_pos_frac", 0.0))
        if gt_pos > 1e-6:
            recall_rows.append(min(pred_pos / gt_pos, 1.0))
        else:
            recall_rows.append(1.0 if pred_pos < 1e-4 else 0.0)
        rows.append(
            {
                "t_frac": _time_frac_at_index(data, int(t_out)),
                "clot_shape": shape_v,
                **{k: float(v) for k, v in band.items()},
            }
        )

    early = [r for r in rows if r["t_frac"] <= 0.5]
    mid = [r for r in rows if 0.35 < r["t_frac"] <= 0.75]
    tfinal = rows[-1]
    t_final_idx = int(pairs[-1][1])
    t_in_final = int(pairs[-1][0])
    phi_final = phi_by_t[t_final_idx]
    step_f = build_clot_forecast_pair_step(data, t_in_final, t_final_idx, phys_cfg, bio_cfg, device)
    wall_ring = _wall_ring_frac(data, phi_final, step_f.phi_gt, device=device)

    mean_shape = _mean_key(rows, "clot_shape")
    mean_band_f1 = _mean_key(rows, "clot_f1")
    early_recall = _mean_key(early, "clot_f1") if early else float("nan")
    early_cov = (
        float(sum(recall_rows[i] for i, r in enumerate(rows) if r["t_frac"] <= 0.5))
        / max(len(early), 1)
    )

    out = {
        "anchor": anchor,
        "rule": rule_tag,
        "n_pairs": len(rows),
        "mean_band_f1": mean_band_f1,
        "mean_clot_shape": mean_shape,
        "mid_mean_band_f1": _mean_key(mid, "clot_f1"),
        "mid_mean_clot_shape": _mean_key(mid, "clot_shape"),
        "early_mean_band_f1": _mean_key(early, "clot_f1"),
        "early_recall_cov": early_cov,
        "early_mean_pred_frac": _mean_key(early, "pred_pos_frac"),
        "early_mean_gt_pos_frac": _mean_key(early, "gt_pos_frac"),
        "tfinal_band_f1": float(tfinal["clot_f1"]),
        "tfinal_band_pred_frac": float(tfinal["pred_pos_frac"]),
        "tfinal_gt_pos_frac": float(tfinal["gt_pos_frac"]),
        "tfinal_clot_shape": float(tfinal["clot_shape"]),
        "tfinal_clot_shape_bal": float(
            _shape_from_phi_at_time(
                data, phi_final, t_in_final, t_final_idx,
                device=device, phys_cfg=phys_cfg, bio_cfg=bio_cfg,
            ).get("clot_shape_balanced", float("nan"))
        ),
        "tfinal_wall_ring_frac": float(wall_ring),
    }
    out["deploy_score"] = deploy_score_from_eval_row(out)
    out["trajectory_score"] = trajectory_score_from_row(out)
    return out


def trajectory_score_from_row(row: dict[str, Any]) -> float:
    """Primary model-select score: timeline quality, not t_final only."""
    m_f1 = float(row.get("mean_band_f1", 0.0))
    m_sh = float(row.get("mean_clot_shape", 0.0))
    early = float(row.get("early_recall_cov", 0.0))
    ring = float(row.get("tfinal_wall_ring_frac", 0.0))
    if m_f1 != m_f1:
        m_f1 = 0.0
    if m_sh != m_sh:
        m_sh = 0.0
    if early != early:
        early = 0.0
    ring_pen = min(max(ring, 0.0), 1.0)
    return float(0.35 * m_f1 + 0.35 * m_sh + 0.20 * early + 0.10 * (1.0 - ring_pen))
