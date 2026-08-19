"""The metric of record, wired once so every arm is scored identically.

Domain-restricted ``deploy_clot_score`` exactly as ``scripts/eval_domain_targets.py``
computes it: zero the prediction and the GT outside the domain, then the canonical
relaxed score.  Targets: wall > 0.9, off-wall > 0.7.
"""
from __future__ import annotations

import numpy as np
import torch

from src.evaluation.clot_relaxed_metrics import (
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
)

WALL_TARGET = 0.9
OFF_TARGET = 0.7


def domain_score(pred: np.ndarray, gt: np.ndarray, ei: torch.Tensor,
                 domain: np.ndarray, wall: np.ndarray) -> float:
    if int((gt & domain).sum()) == 0:
        return float("nan")
    dom = torch.tensor(domain.astype(np.float32))
    m = compute_clot_relaxed_metrics(
        torch.tensor(pred.astype(np.float32)) * dom,
        torch.tensor(gt.astype(np.float32)) * dom,
        ei, wall_mask=torch.tensor(wall))
    return float(clot_score_from_deploy_dict(metrics_to_deploy_prefix(m)))


def full_score(pred: np.ndarray, gt: np.ndarray, ei: torch.Tensor, wall: np.ndarray) -> float:
    m = compute_clot_relaxed_metrics(
        torch.tensor(pred.astype(np.float32)), torch.tensor(gt.astype(np.float32)),
        ei, wall_mask=torch.tensor(wall))
    return float(clot_score_from_deploy_dict(metrics_to_deploy_prefix(m)))


def f1(pred: np.ndarray, gt: np.ndarray) -> float:
    if gt.sum() == 0:
        return float("nan")
    tp = int((pred & gt).sum())
    p, r = tp / max(int(pred.sum()), 1), tp / max(int(gt.sum()), 1)
    return 2 * p * r / max(p + r, 1e-9)


def score_vessel(pred: np.ndarray, S: dict) -> dict:
    """``S`` is a sample dict from the cache; ``pred`` a boolean full-mesh mask."""
    ei = torch.tensor(S["edge_index"])
    wall, gt = S["wall"], S["y"] > 0.5
    return dict(
        wall=domain_score(pred, gt, ei, wall, wall),
        off=domain_score(pred, gt, ei, ~wall, wall),
        full=full_score(pred, gt, ei, wall),
        wall_f1=f1(pred & wall, gt & wall),
        off_f1=f1(pred & ~wall, gt & ~wall),
    )


def summarise(rows: dict[str, dict], anchors_fit, anchors_dev) -> dict:
    out = {}
    for split, anchors in (("fit", anchors_fit), ("dev", anchors_dev)):
        vals = {k: [] for k in ("wall", "off", "full", "wall_f1", "off_f1")}
        for a in anchors:
            if a not in rows:
                continue
            for k in vals:
                v = rows[a].get(k, float("nan"))
                if v == v:
                    vals[k].append(v)
        out[split] = {k: (float(np.mean(v)) if v else float("nan")) for k, v in vals.items()}
        out[split]["n"] = len(
            [a for a in anchors if a in rows and rows[a].get("wall", float("nan")) == rows[a].get("wall", float("nan"))])
    return out


def banner(tag: str, s: dict) -> str:
    f, d = s["fit"], s["dev"]
    return ("%-26s | FIT wall %.4f off %.4f full %.4f | DEV wall %.4f off %.4f full %.4f"
            % (tag, f["wall"], f["off"], f["full"], d["wall"], d["off"], d["full"]))
