"""Rung 4 rollout health: catch frozen wall-ring clot predictions.

Final-t F1 alone is misleading when GT clots are wall-adjacent: a model can
paint the entire wall at t=0 and keep it, gaining recall without real growth.
"""

from __future__ import annotations

from typing import Any

import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_continuous_time import macro_tau_at_index
from src.core_physics.clot_phi_simple import _wall_mask_from_data
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time
from src.evaluation.clot_relaxed_metrics import legacy_clot_f1_metrics as _clot_metrics


def _binary_jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).bool()
    b = b.reshape(-1).bool()
    inter = float((a & b).sum().item())
    union = float((a | b).sum().item())
    return inter / max(union, 1.0)


def compute_rung4_rollout_health(
    phi_traj: dict[int, torch.Tensor],
    data,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    *,
    times: list[int] | None = None,
) -> dict[str, Any]:
    """Health metrics for a full phi trajectory (one row per requested time)."""
    n_steps = int(data.y.shape[0])
    times = sorted(set(times or range(n_steps)))
    if not times:
        times = [0, n_steps - 1]

    n_nodes = int(data.num_nodes)
    wall = _wall_mask_from_data(data, device, n_nodes).reshape(-1).bool()
    bulk = ~wall
    mask = torch.ones(n_nodes, device=device, dtype=torch.bool)

    rows: list[dict[str, Any]] = []
    for t in times:
        phi_gt = gt_clot_phi_at_time(data, t, phys_cfg, device).reshape(-1)
        phi_p = phi_traj[int(t)].reshape(-1)
        m = _clot_metrics(phi_p, phi_gt, mask)
        pred_pos = phi_p >= 0.5
        pred_on_wall = float((pred_pos & wall).sum().item())
        pred_total = float(pred_pos.sum().item())
        rows.append({
            "time": int(t),
            "tau": float(macro_tau_at_index(data, t, bio_cfg=bio_cfg)),
            "clot_f1": float(m["clot_f1"]),
            "clot_prec": float(m["clot_prec"]),
            "clot_rec": float(m["clot_rec"]),
            "pred_pos_frac": float(m["pred_pos_frac"]),
            "gt_pos_frac": float(m["gt_pos_frac"]),
            "pred_wall_frac": pred_on_wall / max(pred_total, 1.0),
            "phi_mean_wall": float(phi_p[wall].mean().item()) if wall.any() else 0.0,
            "phi_mean_bulk": float(phi_p[bulk].mean().item()) if bulk.any() else 0.0,
            "gt_phi_mean_wall": float(phi_gt[wall].mean().item()) if wall.any() else 0.0,
        })

    t_early = times[0]
    t_late = times[-1]
    phi_early = phi_traj[int(t_early)].reshape(-1)
    phi_late = phi_traj[int(t_late)].reshape(-1)
    gt_early = gt_clot_phi_at_time(data, t_early, phys_cfg, device).reshape(-1)

    # Earliest time with zero GT clot (usually t=0).
    t0_row = rows[0]
    wall_ring_t0 = float(t0_row["phi_mean_wall"]) if t0_row["gt_pos_frac"] <= 1e-9 else 0.0
    early_commit_frac = float(t0_row["pred_pos_frac"]) if t0_row["gt_pos_frac"] <= 1e-9 else 0.0

    commit_jaccard = _binary_jaccard(phi_early >= 0.5, phi_late >= 0.5)
    phi_corr = 0.0
    if phi_early.numel() > 1 and float(phi_early.std()) > 1e-8 and float(phi_late.std()) > 1e-8:
        phi_corr = float(torch.corrcoef(torch.stack([phi_early, phi_late]))[0, 1].item())

    f1_vals = [r["clot_f1"] for r in rows]
    min_f1 = min(f1_vals) if f1_vals else 0.0
    final_f1 = rows[-1]["clot_f1"]
    final_row = rows[-1]

    early_false_commit_max = 0.0
    early_phi_wall_max = 0.0
    for r in rows:
        if r["gt_pos_frac"] <= 1e-6:
            early_false_commit_max = max(early_false_commit_max, r["pred_pos_frac"])
            early_phi_wall_max = max(early_phi_wall_max, r["phi_mean_wall"])

    frozen_wall_ring = bool(
        early_phi_wall_max > 0.15
        and early_false_commit_max > 0.005
        and final_row["pred_wall_frac"] > 0.85
        and commit_jaccard > 0.5
    )

    # High recall + low precision + all-wall commits = wall carpet (degenerate for wall-adjacent GT).
    wall_carpet = bool(
        final_row["pred_wall_frac"] > 0.95
        and final_row["clot_rec"] > 0.85
        and final_row["clot_prec"] < 0.45
    )

    # Lower is better; penalize early wall heat, frozen commits, and wall carpet.
    health_score = (
        final_f1
        - 2.0 * early_phi_wall_max
        - 1.5 * early_false_commit_max
        - 0.5 * max(0.0, commit_jaccard - 0.3)
        - 0.35 * (1.0 if wall_carpet else 0.0)
    )

    health_pass = bool(
        not frozen_wall_ring
        and not wall_carpet
        and early_phi_wall_max < 0.1
        and early_false_commit_max < 0.002
    )

    return {
        "wall_ring_t0": wall_ring_t0,
        "early_commit_frac": early_commit_frac,
        "early_false_commit_max": early_false_commit_max,
        "early_phi_wall_max": early_phi_wall_max,
        "pred_wall_frac_late": final_row["pred_wall_frac"],
        "commit_jaccard_early_late": commit_jaccard,
        "phi_corr_early_late": phi_corr,
        "min_f1": min_f1,
        "final_f1": final_f1,
        "frozen_wall_ring": frozen_wall_ring,
        "wall_carpet": wall_carpet,
        "health_score": health_score,
        "health_pass": health_pass,
        "timeline": rows,
    }
