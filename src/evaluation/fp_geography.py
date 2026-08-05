"""Classify deploy FPs as adjacent overpaint vs distant wrong-pocket (wall-gen gate).

Used by the cheap patient020 diagnostic before choosing physfp vs closed-loop FT.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


# Graph hops from nearest GT clot: <= this => adjacent overpaint; above => distant pocket.
DEFAULT_ADJACENT_MAX_HOPS = 2


def classify_fp_geography(
    phi_pred: np.ndarray,
    phi_gt: np.ndarray,
    edge_index,
    *,
    thresh: float = 0.5,
    adjacent_max_hops: int = DEFAULT_ADJACENT_MAX_HOPS,
    n_nodes: int | None = None,
) -> dict[str, Any]:
    """Return FP hop-to-GT stats and a recommended FT arm.

    Recommendation:
    - ``physfp`` when most FPs are distant from GT (wrong pocket / high-flow spray)
    - ``cloop`` when most FPs hug the true pocket (adjacent overpaint / multi-step drift)
    """
    from src.evaluation.clot_shape_score import graph_hop_distance_from_seeds

    pred = np.asarray(phi_pred, dtype=np.float64).reshape(-1)
    gt = np.asarray(phi_gt, dtype=np.float64).reshape(-1)
    n = int(n_nodes) if n_nodes is not None else int(pred.shape[0])
    pr = pred >= float(thresh)
    gtm = gt >= float(thresh)
    fp = pr & ~gtm
    fn = ~pr & gtm
    tp = pr & gtm

    n_fp = int(fp.sum())
    n_fn = int(fn.sum())
    n_tp = int(tp.sum())

    if n_fp == 0:
        return {
            "n_fp": 0,
            "n_fn": n_fn,
            "n_tp": n_tp,
            "n_adjacent_fp": 0,
            "n_distant_fp": 0,
            "adjacent_frac": 0.0,
            "distant_frac": 0.0,
            "fp_hop_to_gt_median": None,
            "fp_hop_to_gt_mean": None,
            "adjacent_max_hops": int(adjacent_max_hops),
            "mode": "no_fp",
            "recommend_leg": "cloop",
            "hint": "no FPs -- try closed-loop FT for recall/front, not physics FP gating",
        }

    if not bool(gtm.any()):
        return {
            "n_fp": n_fp,
            "n_fn": n_fn,
            "n_tp": n_tp,
            "n_adjacent_fp": 0,
            "n_distant_fp": n_fp,
            "adjacent_frac": 0.0,
            "distant_frac": 1.0,
            "fp_hop_to_gt_median": None,
            "fp_hop_to_gt_mean": None,
            "adjacent_max_hops": int(adjacent_max_hops),
            "mode": "distant",
            "recommend_leg": "physfp",
            "hint": "empty GT -- all FPs are distant by definition; use physfp",
        }

    dist = graph_hop_distance_from_seeds(edge_index, n, gtm)
    fp_hops = dist[fp]
    adj_max = int(adjacent_max_hops)
    n_adj = int((fp_hops <= adj_max).sum())
    n_dist = int(n_fp - n_adj)
    adj_frac = float(n_adj) / float(n_fp)
    dist_frac = float(n_dist) / float(n_fp)
    med = float(np.median(fp_hops))
    mean = float(np.mean(fp_hops))

    if dist_frac >= 0.55 or med > float(adj_max) + 0.5:
        mode = "distant"
        leg = "physfp"
        hint = "FPs are mostly distant wrong pockets -- train physical_fp_gating from WG_prec_iter"
    elif adj_frac >= 0.55:
        mode = "adjacent"
        leg = "cloop"
        hint = "FPs hug the true pocket -- closed-loop dynamics FT (no new loss)"
    else:
        mode = "mixed"
        leg = "physfp"
        hint = "mixed FP geography -- prefer physfp first (precision ceiling); cloop if that stalls"

    return {
        "n_fp": n_fp,
        "n_fn": n_fn,
        "n_tp": n_tp,
        "n_adjacent_fp": n_adj,
        "n_distant_fp": n_dist,
        "adjacent_frac": adj_frac,
        "distant_frac": dist_frac,
        "fp_hop_to_gt_median": med,
        "fp_hop_to_gt_mean": mean,
        "adjacent_max_hops": adj_max,
        "mode": mode,
        "recommend_leg": leg,
        "hint": hint,
    }


def format_fp_geography(summary: Mapping[str, Any], *, label: str = "") -> str:
    tag = f" {label}" if label else ""
    med = summary.get("fp_hop_to_gt_median")
    med_s = f"{float(med):.1f}" if med is not None else "n/a"
    return (
        f"[fp_geo{tag}] mode={summary.get('mode', '?')} "
        f"fp={int(summary.get('n_fp', 0))} "
        f"adj={int(summary.get('n_adjacent_fp', 0))} "
        f"dist={int(summary.get('n_distant_fp', 0))} "
        f"adj_frac={float(summary.get('adjacent_frac', 0.0)):.2f} "
        f"hop_med={med_s} "
        f"-> recommend={summary.get('recommend_leg', '?')} "
        f"| {summary.get('hint', '')}"
    )
