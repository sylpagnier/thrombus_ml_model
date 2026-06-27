"""Clot shape scorecard: spatial overlap + location-weighted false positives + flow guards.

Visual clot definition (canonical, matches ``gt_clot_phi_at_time``):
  node is "clot" when ``relu(mu_eff(t) - mu_eff(t=0)) >= CLOT_SHAPE_MU_THRESH_SI``.

North-star metric ``clot_shape`` is F1 with hop-graded precision on predictions:
each predicted clot node earns proximity credit by graph hops to the nearest GT clot
(full credit on GT, linear decay through ``CLOT_SHAPE_PROX_MAX_HOPS`` (default 5),
zero at ``CLOT_SHAPE_PROX_MAX_HOPS + 1`` hops and beyond). Recall stays binary on GT
clot nodes. ``flow_score``, ``clot_recall``, and ``flow_ok`` are reported separately.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

from src.config import PhysicsConfig, STATE_CHANNEL_MU_EFF_ND
from src.core_physics.clot_phi_simple import clot_phi_thresh_si, mu_growth_clot_binary_mask
from src.utils.metrics import rel_l2_uvp


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def resolve_clot_shape_mu_thresh_si(phys_cfg: PhysicsConfig) -> float:
    """Threshold for binary growth clot on full mesh (Pa*s)."""
    override = os.environ.get("CLOT_SHAPE_MU_THRESH_SI", "").strip()
    if override:
        return max(float(override), float(phys_cfg.mu_inf))
    return float(clot_phi_thresh_si(phys_cfg))


def mu_clot_binary_mask(
    mu_si: torch.Tensor,
    thresh_si: float,
    *,
    mu_anchor_si: torch.Tensor,
) -> torch.Tensor:
    """Growth-only clot mask (canonical GT definition)."""
    return mu_growth_clot_binary_mask(mu_si, mu_anchor_si, thresh_si)


def _mu_anchor_from_state(state: torch.Tensor, phys_cfg: PhysicsConfig) -> torch.Tensor:
    mu_ch = STATE_CHANNEL_MU_EFF_ND
    return phys_cfg.viscosity_nd_to_si(state[:, mu_ch]).reshape(-1)


def graph_hop_distance_from_seeds(
    edge_index: torch.Tensor,
    n_nodes: int,
    seed_mask: np.ndarray,
    *,
    max_hops: int = 128,
) -> np.ndarray:
    """BFS hop distance from seed nodes on the biochem mesh (undirected)."""
    seed_mask = np.asarray(seed_mask, dtype=bool).reshape(-1)
    if seed_mask.shape[0] != n_nodes:
        raise ValueError(f"seed_mask length {seed_mask.shape[0]} != n_nodes {n_nodes}")
    dist = np.full(n_nodes, max_hops + 1, dtype=np.int32)
    if not bool(seed_mask.any()):
        return dist
    row = edge_index[0].detach().cpu().numpy()
    col = edge_index[1].detach().cpu().numpy()
    adj: list[list[int]] = [[] for _ in range(n_nodes)]
    for i, j in zip(row, col):
        ii, jj = int(i), int(j)
        if 0 <= ii < n_nodes and 0 <= jj < n_nodes:
            adj[ii].append(jj)
    queue = np.where(seed_mask)[0].tolist()
    for q in queue:
        dist[q] = 0
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        du = int(dist[u])
        if du >= max_hops:
            continue
        for v in adj[u]:
            if dist[v] > du + 1:
                dist[v] = du + 1
                queue.append(v)
    return dist


def _safe_div(num: float, den: float) -> float:
    if den <= 0.0:
        return 0.0
    return float(num / den)


def clot_shape_proximity_max_hops() -> int:
    """Last hop distance that receives any precision credit (default 5)."""
    return max(_env_int("CLOT_SHAPE_PROX_MAX_HOPS", 5), 0)


def proximity_weight_from_gt_hops(hop_dist: int, *, max_hop: int | None = None) -> float:
    """Credit for a predicted clot node: 1.0 on GT, linear decay to max_hop, 0 beyond."""
    mh = clot_shape_proximity_max_hops() if max_hop is None else max(int(max_hop), 0)
    h = int(hop_dist)
    if h > mh:
        return 0.0
    return float(mh + 1 - h) / float(mh + 1)


def compute_clot_shape_metrics(
    *,
    pred_state: torch.Tensor,
    gt_state: torch.Tensor,
    edge_index: torch.Tensor,
    phys_cfg: PhysicsConfig,
    mu_thresh_si: float | None = None,
    node_mask: torch.Tensor | None = None,
    mu_anchor_si: torch.Tensor | None = None,
    gt_anchor_state: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Binary clot overlap on full mesh from rollout channel-3 mu_eff (growth-only)."""
    mu_ch = STATE_CHANNEL_MU_EFF_ND
    pred_mu = phys_cfg.viscosity_nd_to_si(pred_state[:, mu_ch]).reshape(-1)
    gt_mu = phys_cfg.viscosity_nd_to_si(gt_state[:, mu_ch]).reshape(-1)
    n_nodes = int(pred_mu.numel())
    thresh = float(mu_thresh_si) if mu_thresh_si is not None else resolve_clot_shape_mu_thresh_si(phys_cfg)

    if mu_anchor_si is None:
        if gt_anchor_state is None:
            raise ValueError("mu_anchor_si or gt_anchor_state required for growth-only clot labels")
        mu_anchor_si = _mu_anchor_from_state(gt_anchor_state, phys_cfg)
    anchor = mu_anchor_si.reshape(-1).to(device=gt_mu.device, dtype=gt_mu.dtype)

    gt_clot = mu_clot_binary_mask(gt_mu, thresh, mu_anchor_si=anchor).cpu().numpy()
    pred_clot = mu_clot_binary_mask(pred_mu, thresh, mu_anchor_si=anchor).cpu().numpy()

    eval_n = n_nodes
    if node_mask is not None:
        nm = node_mask.reshape(-1).detach().cpu().numpy().astype(bool)
        if nm.shape[0] != n_nodes:
            raise ValueError(f"node_mask length {nm.shape[0]} != n_nodes {n_nodes}")
        gt_clot = np.logical_and(gt_clot, nm)
        pred_clot = np.logical_and(pred_clot, nm)
        eval_n = int(nm.sum())

    tp = int(np.logical_and(gt_clot, pred_clot).sum())
    fp = int(np.logical_and(~gt_clot, pred_clot).sum())
    fn = int(np.logical_and(gt_clot, ~pred_clot).sum())
    tn = int(np.logical_and(~gt_clot, ~pred_clot).sum())

    dice = _safe_div(2.0 * tp, 2.0 * tp + fp + fn)
    iou = _safe_div(float(tp), float(tp + fp + fn))
    precision = _safe_div(float(tp), float(tp + fp))
    recall = _safe_div(float(tp), float(tp + fn))
    f1 = _safe_div(2.0 * precision * recall, precision + recall)

    prox_max = clot_shape_proximity_max_hops()
    hop_dist = graph_hop_distance_from_seeds(edge_index, n_nodes, gt_clot, max_hops=prox_max + 2)

    pred_idx = np.where(pred_clot)[0]
    if pred_idx.size > 0:
        prox_w = np.array([proximity_weight_from_gt_hops(int(hop_dist[i]), max_hop=prox_max) for i in pred_idx])
        graded_precision = float(prox_w.mean())
        prox_zero = int((prox_w <= 0.0).sum())
        prox_full = int((prox_w >= 1.0 - 1e-9).sum())
    else:
        graded_precision = 0.0
        prox_zero = 0
        prox_full = 0

    fp_mask = np.logical_and(~gt_clot, pred_clot)
    fp_adj = int(np.logical_and(fp_mask, hop_dist <= 2).sum())
    fp_dist = int(np.logical_and(fp_mask, hop_dist > prox_max).sum())
    fp_near = int(np.logical_and(fp_mask, (hop_dist > 0) & (hop_dist <= prox_max)).sum())

    loc_precision = graded_precision
    clot_shape_score = _safe_div(2.0 * graded_precision * recall, graded_precision + recall)

    gt_frac = _safe_div(float(gt_clot.sum()), float(eval_n))
    pred_frac = _safe_div(float(pred_clot.sum()), float(eval_n))
    # Penalize painting volume, not distant FP quality (graded_precision already ignores hop>=6).
    clot_shape_efficiency = _safe_div(clot_shape_score, max(pred_frac, 1e-6))
    gt_safe = max(gt_frac, 1e-6)
    clot_shape_balanced = clot_shape_score * min(1.0, gt_safe / max(pred_frac, 1e-6))
    frac_ratio = _safe_div(pred_frac, gt_frac) if gt_frac > 0.0 else (0.0 if pred_frac <= 0.0 else float("inf"))

    bulk_mask = ~gt_clot
    bulk_vals = pred_mu.detach().cpu().numpy()[bulk_mask]
    bulk_mu_p50 = float(np.median(bulk_vals)) if bulk_vals.size > 0 else float("nan")

    u_pred = pred_state[:, 0:2].detach().cpu().numpy()
    speed = np.linalg.norm(u_pred, axis=1)
    t0_speed_mean = float(speed.mean()) if speed.size > 0 else 0.0

    rel_l2 = rel_l2_uvp(pred_state, gt_state, node_mask=node_mask)

    rel_l2_max = _env_float("CLOT_SHAPE_REL_L2_MAX", _env_float("CLOT_SHAPE_UVP_MAX", 1.2))
    t0_speed_min = _env_float("CLOT_SHAPE_T0_SPEED_MIN", 0.15)
    bulk_mu_min = _env_float("CLOT_SHAPE_BULK_MU_MIN", 0.003)
    bulk_mu_max = _env_float("CLOT_SHAPE_BULK_MU_MAX", 0.06)
    frac_ratio_lo = _env_float("CLOT_SHAPE_FRAC_RATIO_LO", 0.25)
    frac_ratio_hi = _env_float("CLOT_SHAPE_FRAC_RATIO_HI", 4.0)

    flow_trivial = t0_speed_mean < t0_speed_min
    rel_l2_ok = rel_l2 <= rel_l2_max if rel_l2 == rel_l2 else False
    bulk_ok = bulk_mu_min <= bulk_mu_p50 <= bulk_mu_max if bulk_mu_p50 == bulk_mu_p50 else False
    frac_ok = frac_ratio_lo <= frac_ratio <= frac_ratio_hi if gt_frac > 0.0 else pred_frac <= 0.01
    flow_ok = bool(rel_l2_ok and not flow_trivial and bulk_ok and frac_ok)

    flow_score = max(0.0, 1.0 - rel_l2 / max(rel_l2_max, 1e-6)) if rel_l2 == rel_l2 else 0.0

    return {
        "clot_mu_thresh_si": thresh,
        "clot_tp": tp,
        "clot_fp": fp,
        "clot_fn": fn,
        "clot_tn": tn,
        "clot_dice": dice,
        "clot_iou": iou,
        "clot_f1": f1,
        "clot_precision": precision,
        "clot_recall": recall,
        "clot_fp_adjacent": fp_adj,
        "clot_fp_distant": fp_dist,
        "clot_fp_near_gt": fp_near,
        "clot_loc_precision": loc_precision,
        "clot_graded_precision": graded_precision,
        "clot_prox_full": prox_full,
        "clot_prox_zero": prox_zero,
        "clot_prox_max_hops": prox_max,
        "clot_shape": clot_shape_score,
        "clot_shape_efficiency": clot_shape_efficiency,
        "clot_shape_balanced": clot_shape_balanced,
        "clot_gt_frac": gt_frac,
        "clot_pred_frac": pred_frac,
        "clot_frac_ratio": frac_ratio,
        "bulk_mu_p50": bulk_mu_p50,
        "t0_speed_mean": t0_speed_mean,
        "rel_l2": rel_l2,
        "flow_ok": flow_ok,
        "flow_trivial": flow_trivial,
        "rel_l2_ok": rel_l2_ok,
        "bulk_ok": bulk_ok,
        "clot_frac_ok": frac_ok,
        "flow_score": flow_score,
    }


def compute_clot_shape_trajectory(
    *,
    pred_traj: torch.Tensor,
    gt_traj: torch.Tensor,
    edge_index: torch.Tensor,
    phys_cfg: PhysicsConfig,
    pred_time_indices: list[int] | None = None,
    node_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Aggregate clot shape over rollout time steps (pred may be shorter than GT)."""
    n_gt = int(gt_traj.shape[0])
    n_pred = int(pred_traj.shape[0])
    if pred_time_indices is None:
        pred_time_indices = list(range(n_pred))
    gt_anchor_state = gt_traj[0]
    per_step: list[dict[str, Any]] = []
    for pi in pred_time_indices:
        pi = max(0, min(int(pi), n_pred - 1))
        gi = max(0, min(int(pi), n_gt - 1))
        m = compute_clot_shape_metrics(
            pred_state=pred_traj[pi],
            gt_state=gt_traj[gi],
            edge_index=edge_index,
            phys_cfg=phys_cfg,
            node_mask=node_mask,
            gt_anchor_state=gt_anchor_state,
        )
        m["pred_time_index"] = pi
        m["gt_time_index"] = gi
        per_step.append(m)

    def _mean_key(key: str) -> float | None:
        vals = [s[key] for s in per_step if s.get(key) is not None and s[key] == s[key]]
        return sum(vals) / len(vals) if vals else None

    final = per_step[-1] if per_step else {}
    out: dict[str, Any] = {
        "clot_shape_final": final.get("clot_shape"),
        "clot_dice_final": final.get("clot_dice"),
        "clot_recall_final": final.get("clot_recall"),
        "clot_shape_mean": _mean_key("clot_shape"),
        "clot_dice_mean": _mean_key("clot_dice"),
        "clot_recall_mean": _mean_key("clot_recall"),
        "n_clot_shape_steps": len(per_step),
    }
    for k, v in final.items():
        if k not in out:
            out[k] = v
    return out
