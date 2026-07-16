"""Deploy clot phi error counts over a macro-step timeline (FP/FN/TP/TN).

Complements final-horizon ``deploy_clot_f1``: sparse inlet FPs can hide behind low
``mat_overpaint_per_gt`` when GT clot mass is large. Timeline medians / p90 surface
persistent localized errors (e.g. FP=75, FN=4 at mid-time).
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import torch

from src.core_physics.species_gnn_ladder_viz import ladder_viz_times


def clot_binary_masks(
    phi_pred: torch.Tensor,
    phi_gt: torch.Tensor,
    *,
    thresh: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Per-node FP/FN/TP/TN masks (1-D, same length)."""
    pr = phi_pred.reshape(-1) >= float(thresh)
    gt = phi_gt.reshape(-1) >= float(thresh)
    return {
        "fp": pr & ~gt,
        "fn": ~pr & gt,
        "tp": pr & gt,
        "tn": ~pr & ~gt,
    }


def clot_mask_counts(masks: dict[str, torch.Tensor]) -> dict[str, int]:
    return {k: int(v.sum().item()) for k, v in masks.items()}


def clot_frame_metrics(
    phi_pred: torch.Tensor,
    phi_gt: torch.Tensor,
    *,
    thresh: float = 0.5,
    n_band: int | None = None,
) -> dict[str, float]:
    """Counts + rates for one time slice."""
    masks = clot_binary_masks(phi_pred, phi_gt, thresh=thresh)
    counts = clot_mask_counts(masks)
    gt_pos = max(counts["tp"] + counts["fn"], 1)
    pred_pos = max(counts["tp"] + counts["fp"], 1)
    n = int(n_band if n_band is not None else masks["fp"].numel())
    n = max(n, 1)
    err = counts["fp"] + counts["fn"]
    return {
        "clot_fp": float(counts["fp"]),
        "clot_fn": float(counts["fn"]),
        "clot_tp": float(counts["tp"]),
        "clot_tn": float(counts["tn"]),
        "clot_err": float(err),
        "clot_fp_per_gt": float(counts["fp"]) / float(gt_pos),
        "clot_fn_per_gt": float(counts["fn"]) / float(gt_pos),
        "clot_fp_rate_band": float(counts["fp"]) / float(n),
        "clot_fn_rate_band": float(counts["fn"]) / float(n),
        "clot_err_rate_band": float(err) / float(n),
        "clot_prec": float(counts["tp"]) / float(pred_pos),
        "clot_rec": float(counts["tp"]) / float(gt_pos),
    }


def _percentile(vals: Sequence[float], q: float) -> float:
    if not vals:
        return 0.0
    return float(np.percentile(np.asarray(vals, dtype=np.float64), q))


def summarize_clot_timeline(frames: list[dict[str, Any]]) -> dict[str, float]:
    """Robust aggregates over a list of per-time frame dicts (from :func:`clot_frame_metrics`)."""
    if not frames:
        return {}

    def series(key: str) -> list[float]:
        return [float(f.get(key, 0.0)) for f in frames]

    fp = series("clot_fp")
    fn = series("clot_fn")
    err = series("clot_err")
    fp_pg = series("clot_fp_per_gt")
    fn_pg = series("clot_fn_per_gt")
    fp_rb = series("clot_fp_rate_band")

    out: dict[str, float] = {
        "clot_fp_median": float(np.median(fp)),
        "clot_fp_mean": float(np.mean(fp)),
        "clot_fp_p90": _percentile(fp, 90.0),
        "clot_fp_max": float(max(fp)),
        "clot_fn_median": float(np.median(fn)),
        "clot_fn_mean": float(np.mean(fn)),
        "clot_fn_p90": _percentile(fn, 90.0),
        "clot_fn_max": float(max(fn)),
        "clot_err_median": float(np.median(err)),
        "clot_err_p90": _percentile(err, 90.0),
        "clot_err_max": float(max(err)),
        "clot_fp_per_gt_median": float(np.median(fp_pg)),
        "clot_fp_per_gt_p90": _percentile(fp_pg, 90.0),
        "clot_fn_per_gt_median": float(np.median(fn_pg)),
        "clot_fp_rate_band_median": float(np.median(fp_rb)),
        "clot_fp_rate_band_p90": _percentile(fp_rb, 90.0),
    }
    # Penalize early transient FPs: mean of first 3 frames (cold-start wall paint).
    early = frames[: min(3, len(frames))]
    if early:
        out["clot_fp_early_mean"] = float(np.mean([float(f.get("clot_fp", 0.0)) for f in early]))
        out["clot_err_early_mean"] = float(np.mean([float(f.get("clot_err", 0.0)) for f in early]))
    return out


@torch.no_grad()
def eval_clot_timeline_on_grid(
    phi_traj: dict[int, torch.Tensor],
    data,
    phys_cfg,
    device: torch.device,
    *,
    max_frames: int = 10,
    times: list[int] | None = None,
    thresh: float = 0.5,
) -> dict[str, Any]:
    """GT vs predicted clot phi on ``ladder_viz_times`` grid."""
    from src.core_physics.t0_mu_physics import gt_clot_phi_at_time

    n_steps = int(data.y.shape[0])
    grid = times if times is not None else ladder_viz_times(n_steps, max_frames=max_frames)
    n_nodes = int(data.num_nodes)
    frames: list[dict[str, Any]] = []
    for t in grid:
        if int(t) not in phi_traj:
            continue
        phi_gt = gt_clot_phi_at_time(data, int(t), phys_cfg, device)
        phi_pred = phi_traj[int(t)]
        fm = clot_frame_metrics(phi_pred, phi_gt, thresh=thresh, n_band=n_nodes)
        fm["time"] = int(t)
        frames.append(fm)
    summary = summarize_clot_timeline(frames)
    return {"times": grid, "frames": frames, "summary": summary}
