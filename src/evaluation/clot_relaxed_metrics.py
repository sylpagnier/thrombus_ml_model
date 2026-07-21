"""Relaxed clot metrics for deploy training and eval (full-vessel, TN-free).

Design (see project clot metric spec):
- **Relaxed precision**: each predicted clot node must have GT clot within ``relax_hops``.
- **Relaxed recall**: each GT clot node must have a prediction within ``relax_hops``.
- **F_beta** (default F0.5): precision-weighted; punishes over-prediction.
- **Dilation IoU**: IoU(pred, dilate(GT, relax_hops)) on the full mesh.
- **Vacuous match**: when both pred and GT have zero clot nodes, all scores are **1.0**
  (correct silence), not 0.0 from empty denominators.

Combined **clot_guiding** score (default checkpoint target):
  ``iou_w * dilation_iou + fbeta_w * relaxed_f_beta``

Env:
  ``CLOT_GUIDE_RELAX_HOPS`` (default 2)
  ``CLOT_GUIDE_F_BETA`` (default 0.5)
  ``CLOT_GUIDE_IOU_W`` / ``CLOT_GUIDE_F05_W`` (default 0.5 each)
  ``SPECIES_CONTINUOUS_CLOUT_SCORE`` = guiding | relaxed_f05 | dilation_iou | legacy_f1
"""

from __future__ import annotations

import os
from typing import Any

import torch


def legacy_clot_f1_metrics(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> dict[str, float]:
    """Precision/recall/F1 inside a supervision mask (legacy helper)."""
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
    if tp + fp + fn == 0.0:
        return {
            "clot_prec": 1.0,
            "clot_rec": 1.0,
            "clot_f1": 1.0,
            "pred_pos_frac": 0.0,
            "gt_pos_frac": 0.0,
        }
    prec = tp / max(tp + fp, 1e-6)
    rec = tp / max(tp + fn, 1e-6)
    f1 = (2.0 * prec * rec) / max(prec + rec, 1e-6)
    return {
        "clot_prec": prec,
        "clot_rec": rec,
        "clot_f1": f1,
        "pred_pos_frac": float(pb.mean().item()),
        "gt_pos_frac": float(tb.mean().item()),
    }


from src.core_physics.clot_growth_masks import graph_dilate_hops


def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def clot_guide_relax_hops() -> int:
    return max(_env_int("CLOT_GUIDE_RELAX_HOPS", 2), 0)


def clot_guide_f_beta() -> float:
    return max(_env_float("CLOT_GUIDE_F_BETA", 0.5), 1e-6)


def clot_guide_iou_weight() -> float:
    return max(_env_float("CLOT_GUIDE_IOU_W", 0.5), 0.0)


def clot_guide_fbeta_weight() -> float:
    return max(_env_float("CLOT_GUIDE_F05_W", 0.5), 0.0)


def species_continuous_clout_score_mode() -> str:
    raw = (os.environ.get("SPECIES_CONTINUOUS_CLOUT_SCORE") or "guiding").strip().lower()
    if raw in ("legacy", "legacy_f1", "f1", "strict"):
        return "legacy_f1"
    if raw in ("relaxed_f05", "f05", "f0.5"):
        return "relaxed_f05"
    if raw in ("dilation_iou", "iou", "dil_iou"):
        return "dilation_iou"
    if raw in ("relaxed_prec_floor", "prec_floor", "relaxed_prec", "precision_floor"):
        return "relaxed_prec_floor"
    return "guiding"


def clot_prec_recall_floor() -> float:
    """Min relaxed recall required before precision is rewarded at full weight."""
    return max(_env_float("SPECIES_CLOUT_PREC_REC_FLOOR", 0.30), 0.0)


def relaxed_prec_floor_score(relaxed_prec: float, relaxed_rec: float) -> float:
    """Precision-first score that still demands the model predict *some* true clots.

    - recall 0 -> 0 (degenerate empty / all-miss prediction is worthless)
    - recall >= floor -> full relaxed precision
    - 0 < recall < floor -> precision linearly ramped by recall/floor
    """
    p = float(relaxed_prec)
    r = float(relaxed_rec)
    if r <= 0.0:
        return 0.0
    floor = clot_prec_recall_floor()
    if floor <= 0.0 or r >= floor:
        return p
    return p * (r / floor)


def _safe_div(num: float, den: float) -> float:
    if den <= 0.0:
        return 0.0
    return float(num / den)


def _vacuous_clot_match_metrics(*, relax_hops: int, beta: float) -> dict[str, float]:
    """Both pred and GT empty: perfect agreement (not a failure)."""
    return {
        "clot_relaxed_prec": 1.0,
        "clot_relaxed_rec": 1.0,
        "clot_relaxed_f05": 1.0,
        "clot_relaxed_f_beta": 1.0,
        "clot_dilation_iou": 1.0,
        "clot_guiding": 1.0,
        "clot_prec": 1.0,
        "clot_rec": 1.0,
        "clot_f1": 1.0,
        "clot_iou": 1.0,
        "clot_tp": 0.0,
        "clot_fp": 0.0,
        "clot_fn": 0.0,
        "clot_pred_pos": 0.0,
        "clot_gt_pos": 0.0,
        "clot_relax_hops": float(relax_hops),
        "clot_f_beta": float(beta),
        "pred_pos_frac": 0.0,
        "gt_pos_frac": 0.0,
        "clot_vacuous_match": 1.0,
    }


def f_beta_score(precision: float, recall: float, *, beta: float) -> float:
    b2 = float(beta) ** 2
    p, r = float(precision), float(recall)
    den = b2 * p + r
    if den <= 0.0:
        return 0.0
    return (1.0 + b2) * p * r / den


def clot_guiding_score(dilation_iou: float, relaxed_f_beta: float) -> float:
    iw = clot_guide_iou_weight()
    fw = clot_guide_fbeta_weight()
    norm = iw + fw
    if norm <= 0.0:
        return 0.5 * float(dilation_iou) + 0.5 * float(relaxed_f_beta)
    return (iw * float(dilation_iou) + fw * float(relaxed_f_beta)) / norm


def clot_score_from_deploy_dict(m: dict[str, float]) -> float:
    mode = species_continuous_clout_score_mode()
    if mode == "legacy_f1":
        return float(m.get("deploy_clot_f1", m.get("clot_f1", 0.0)))
    if mode == "relaxed_f05":
        return float(m.get("deploy_clot_relaxed_f05", m.get("clot_relaxed_f05", 0.0)))
    if mode == "dilation_iou":
        return float(m.get("deploy_clot_dil_iou", m.get("clot_dilation_iou", 0.0)))
    if mode == "relaxed_prec_floor":
        prec = float(m.get("deploy_clot_relaxed_prec", m.get("clot_relaxed_prec", 0.0)))
        rec = float(m.get("deploy_clot_relaxed_rec", m.get("clot_relaxed_rec", 0.0)))
        return relaxed_prec_floor_score(prec, rec)
    return float(m.get("deploy_clot_guiding", m.get("clot_guiding", 0.0)))


def compute_clot_relaxed_metrics(
    phi_pred: torch.Tensor,
    phi_gt: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    relax_hops: int | None = None,
    f_beta: float | None = None,
    phi_thresh: float = 0.5,
    wall_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Full-vessel relaxed clot metrics from binary phi masks (entire mesh)."""
    hops = clot_guide_relax_hops() if relax_hops is None else max(int(relax_hops), 0)
    beta = clot_guide_f_beta() if f_beta is None else float(f_beta)

    device = phi_pred.device
    pred = phi_pred.reshape(-1).to(device=device, dtype=torch.float32)
    gt = phi_gt.reshape(-1).to(device=device, dtype=torch.float32)
    if pred.shape[0] != gt.shape[0]:
        raise ValueError(f"phi_pred length {pred.shape[0]} != phi_gt {gt.shape[0]}")

    pred_pos = pred > phi_thresh
    gt_pos = gt > phi_thresh
    n_pred = int(pred_pos.sum().item())
    n_gt = int(gt_pos.sum().item())

    if n_pred == 0 and n_gt == 0:
        return _vacuous_clot_match_metrics(
            relax_hops=hops,
            beta=beta,
        )

    gt_dil = graph_dilate_hops(gt_pos, edge_index, hops)
    pred_dil = graph_dilate_hops(pred_pos, edge_index, hops)

    tp_prec = int((pred_pos & gt_dil).sum().item())
    tp_rec = int((gt_pos & pred_dil).sum().item())

    strict_tp = int((pred_pos & gt_pos).sum().item())
    strict_fp = int((pred_pos & ~gt_pos).sum().item())
    strict_fn = int((~pred_pos & gt_pos).sum().item())

    relaxed_prec = _safe_div(float(tp_prec), float(n_pred))
    relaxed_rec = _safe_div(float(tp_rec), float(n_gt))
    strict_prec = _safe_div(float(strict_tp), float(strict_tp + strict_fp))
    strict_rec = _safe_div(float(strict_tp), float(strict_tp + strict_fn))
    strict_f1 = f_beta_score(strict_prec, strict_rec, beta=1.0)

    relaxed_f_beta = f_beta_score(relaxed_prec, relaxed_rec, beta=beta)

    # Symmetric dilation IoU: overlap of n-hop envelopes (exact match -> 1.0).
    dil_inter = int((pred_dil & gt_dil).sum().item())
    dil_union = int((pred_dil | gt_dil).sum().item())
    dilation_iou = _safe_div(float(dil_inter), float(dil_union))

    strict_inter = int((pred_pos & gt_pos).sum().item())
    strict_union = int((pred_pos | gt_pos).sum().item())
    strict_iou = _safe_div(float(strict_inter), float(strict_union))

    guiding = clot_guiding_score(dilation_iou, relaxed_f_beta)

    res = {
        "clot_relaxed_prec": relaxed_prec,
        "clot_relaxed_rec": relaxed_rec,
        "clot_relaxed_f05": f_beta_score(relaxed_prec, relaxed_rec, beta=0.5),
        "clot_relaxed_f_beta": relaxed_f_beta,
        "clot_dilation_iou": dilation_iou,
        "clot_guiding": guiding,
        "clot_prec": strict_prec,
        "clot_rec": strict_rec,
        "clot_f1": strict_f1,
        "clot_iou": strict_iou,
        "clot_tp": float(strict_tp),
        "clot_fp": float(strict_fp),
        "clot_fn": float(strict_fn),
        "clot_pred_pos": float(n_pred),
        "clot_gt_pos": float(n_gt),
        "clot_relax_hops": float(hops),
        "clot_f_beta": float(beta),
        "pred_pos_frac": _safe_div(float(n_pred), float(pred.numel())),
        "gt_pos_frac": _safe_div(float(n_gt), float(gt.numel())),
    }

    if wall_mask is not None:
        # Off-wall relaxed metrics must only reward off-wall predictions.
        # The previous implementation used relax dilation computed from the full
        # prediction mask, allowing wall predictions to "rescue" off-wall recall.
        offwall = ~wall_mask.reshape(-1).to(device=pred_pos.device).bool()
        pred_pos_off = pred_pos & offwall
        gt_pos_off = gt_pos & offwall

        n_pred_off = int(pred_pos_off.sum().item())
        n_gt_off = int(gt_pos_off.sum().item())

        # Build relaxed neighborhoods from off-wall-only pred/GT masks.
        gt_dil_off = graph_dilate_hops(gt_pos_off, edge_index, hops)
        pred_dil_off = graph_dilate_hops(pred_pos_off, edge_index, hops)

        tp_prec_off = int((pred_pos_off & gt_dil_off).sum().item())
        tp_rec_off = int((gt_pos_off & pred_dil_off).sum().item())

        relaxed_prec_off = _safe_div(float(tp_prec_off), float(n_pred_off))
        relaxed_rec_off = _safe_div(float(tp_rec_off), float(n_gt_off))
        relaxed_f1_off = f_beta_score(relaxed_prec_off, relaxed_rec_off, beta=beta)

        strict_tp_off = int((pred_pos & gt_pos & offwall).sum().item())
        strict_fp_off = int((pred_pos & ~gt_pos & offwall).sum().item())
        strict_fn_off = int((~pred_pos & gt_pos & offwall).sum().item())
        strict_prec_off = _safe_div(float(strict_tp_off), float(strict_tp_off + strict_fp_off))
        strict_rec_off = _safe_div(float(strict_tp_off), float(strict_tp_off + strict_fn_off))
        strict_f1_off = f_beta_score(strict_prec_off, strict_rec_off, beta=1.0)

        res["offwall_relaxed_f1"] = relaxed_f1_off
        res["offwall_strict_f1"] = strict_f1_off
        res["offwall_relaxed_prec"] = relaxed_prec_off
        res["offwall_relaxed_rec"] = relaxed_rec_off
        res["offwall_n_pred"] = float(n_pred_off)
        res["offwall_n_gt"] = float(n_gt_off)

        # Hop-stratified off-wall counts (metric discipline for firewall work).
        hop_dist = _bfs_hops_from_wall(edge_index, wall_mask.reshape(-1).bool(), int(pred_pos.numel()))
        for h in (1, 2, 3, 4):
            at_h = hop_dist == int(h)
            res[f"offwall_n_pred_hop{h}"] = float((pred_pos & at_h).sum().item())
            res[f"offwall_n_gt_hop{h}"] = float((gt_pos & at_h).sum().item())
            tp_h = int((pred_pos & gt_pos & at_h).sum().item())
            fp_h = int((pred_pos & ~gt_pos & at_h).sum().item())
            fn_h = int((~pred_pos & gt_pos & at_h).sum().item())
            prec_h = _safe_div(float(tp_h), float(tp_h + fp_h))
            rec_h = _safe_div(float(tp_h), float(tp_h + fn_h))
            res[f"offwall_strict_f1_hop{h}"] = f_beta_score(prec_h, rec_h, beta=1.0)
        lumen = hop_dist >= 2
        n_pred_lumen = int((pred_pos & lumen).sum().item())
        n_gt_lumen = int((gt_pos & lumen).sum().item())
        tp_lumen = int((pred_pos & gt_pos & lumen).sum().item())
        fp_lumen = int((pred_pos & ~gt_pos & lumen).sum().item())
        fn_lumen = int((~pred_pos & gt_pos & lumen).sum().item())
        res["offwall_n_pred_hop_ge2"] = float(n_pred_lumen)
        res["offwall_n_gt_hop_ge2"] = float(n_gt_lumen)
        res["offwall_strict_f1_hop_ge2"] = f_beta_score(
            _safe_div(float(tp_lumen), float(tp_lumen + fp_lumen)),
            _safe_div(float(tp_lumen), float(tp_lumen + fn_lumen)),
            beta=1.0,
        )

    return res


def _bfs_hops_from_wall(
    edge_index: torch.Tensor,
    wall_mask: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """BFS hop distance from wall nodes (unreachable -> 99). Local to avoid import cycles."""
    device = edge_index.device
    hops = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    wall_m = wall_mask.to(device=device).bool().reshape(-1)
    hops[wall_m] = 0
    row, col = edge_index
    current = wall_m.clone()
    cur_h = 0
    while True:
        nbr = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        nbr[col[current[row]]] = True
        nxt = nbr & (hops == -1)
        if not bool(nxt.any().item()):
            break
        cur_h += 1
        hops[nxt] = cur_h
        current = nxt
    hops[hops == -1] = 99
    return hops


def compute_clot_relaxed_metrics_full_mesh(
    phi_pred: torch.Tensor,
    phi_gt: torch.Tensor,
    edge_index: torch.Tensor,
    **kwargs: Any,
) -> dict[str, float]:
    """Alias: relaxed metrics on the entire vessel (deploy default)."""
    return compute_clot_relaxed_metrics(phi_pred, phi_gt, edge_index, **kwargs)


def metrics_to_deploy_prefix(m: dict[str, float], *, prefix: str = "deploy_") -> dict[str, float]:
    """Map generic clot metric keys to deploy-prefixed train-log keys."""
    out: dict[str, float] = {}
    mapping = {
        "clot_f1": f"{prefix}clot_f1",
        "clot_prec": f"{prefix}clot_prec",
        "clot_rec": f"{prefix}clot_rec",
        "clot_relaxed_prec": f"{prefix}clot_relaxed_prec",
        "clot_relaxed_rec": f"{prefix}clot_relaxed_rec",
        "clot_relaxed_f05": f"{prefix}clot_relaxed_f05",
        "clot_relaxed_f_beta": f"{prefix}clot_relaxed_f_beta",
        "clot_dilation_iou": f"{prefix}clot_dil_iou",
        "clot_guiding": f"{prefix}clot_guiding",
        "clot_iou": f"{prefix}clot_iou",
        "pred_pos_frac": f"{prefix}clot_pred_pos_frac",
        "offwall_relaxed_f1": f"{prefix}clot_offwall_relaxed_f1",
        "offwall_strict_f1": f"{prefix}clot_offwall_strict_f1",
        "offwall_relaxed_prec": f"{prefix}clot_offwall_relaxed_prec",
        "offwall_relaxed_rec": f"{prefix}clot_offwall_relaxed_rec",
        "offwall_n_pred": f"{prefix}clot_offwall_n_pred",
        "offwall_n_gt": f"{prefix}clot_offwall_n_gt",
        "offwall_n_pred_hop1": f"{prefix}clot_offwall_n_pred_hop1",
        "offwall_n_pred_hop2": f"{prefix}clot_offwall_n_pred_hop2",
        "offwall_n_pred_hop3": f"{prefix}clot_offwall_n_pred_hop3",
        "offwall_n_pred_hop_ge2": f"{prefix}clot_offwall_n_pred_hop_ge2",
        "offwall_n_gt_hop_ge2": f"{prefix}clot_offwall_n_gt_hop_ge2",
        "offwall_strict_f1_hop2": f"{prefix}clot_offwall_strict_f1_hop2",
        "offwall_strict_f1_hop3": f"{prefix}clot_offwall_strict_f1_hop3",
        "offwall_strict_f1_hop_ge2": f"{prefix}clot_offwall_strict_f1_hop_ge2",
    }
    for src, dst in mapping.items():
        if src in m:
            out[dst] = float(m[src])
    if "time_index" in m:
        out["time_index"] = float(m["time_index"])
    return out
