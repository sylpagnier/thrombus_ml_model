"""Evaluation helpers for the differentiable wall model using canonical deploy metrics."""
from __future__ import annotations

from pathlib import Path
import torch

from src.config import PhysicsConfig
from src.core_physics.species_pushforward_continuous import resolve_deploy_eval_time_index
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time
from src.differentiable_wall_model.differentiable_ode import DifferentiableWallModel
from src.evaluation.clot_relaxed_metrics import (
    clot_score_from_deploy_dict,
    compute_clot_relaxed_metrics,
    metrics_to_deploy_prefix,
)


def evaluate_vessel(
    model: DifferentiableWallModel,
    data,
    phys_cfg: PhysicsConfig | None = None,
    *,
    flow_source: str = "pred",
    threshold: float = 0.5,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Score a single vessel graph using canonical deploy_clot_score."""
    device = device or torch.device("cpu")
    phys_cfg = phys_cfg or PhysicsConfig(phase="biochem")

    model.eval()
    with torch.no_grad():
        out = model(data, flow_source=flow_source, device=device)
        prob = out["prob_clot"]
        binary_pred = (prob >= threshold).float()

        wall = data.mask_wall.reshape(-1).bool().to(device)
        t_eval = resolve_deploy_eval_time_index(int(data.y.shape[0]))
        gt = gt_clot_phi_at_time(data, t_eval, phys_cfg, device=device).reshape(-1)
        gt = gt * wall.float()

        m = compute_clot_relaxed_metrics(
            binary_pred,
            gt,
            data.edge_index.to(device),
            wall_mask=wall,
        )
        d = metrics_to_deploy_prefix(m)
        score = clot_score_from_deploy_dict(d)
        d["deploy_clot_score"] = score
        d["n_pred_wall"] = float((binary_pred * wall.float()).sum().item())
        d["n_gt_wall"] = float(gt.sum().item())
        return d


def evaluate_cohort(
    model: DifferentiableWallModel,
    vessel_names: list[str],
    graph_dir: Path,
    phys_cfg: PhysicsConfig | None = None,
    *,
    flow_source: str = "pred",
    threshold: float = 0.5,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Evaluate model across a cohort of vessels."""
    results = {}
    scores = []
    f1s = []

    for name in vessel_names:
        p = graph_dir / f"{name}.pt"
        if not p.exists():
            continue
        try:
            data = torch.load(p, map_location=device or "cpu", weights_only=False)
            if int(data.y.shape[0]) < 150:
                continue
            if flow_source == "pred" and getattr(data, "u0_pred", None) is None:
                continue
            res = evaluate_vessel(model, data, phys_cfg, flow_source=flow_source,
                                  threshold=threshold, device=device)
            results[name] = res
            scores.append(res["deploy_clot_score"])
            f1s.append(res["deploy_clot_f1"])
        except ValueError:
            continue

    mean_score = float(sum(scores) / len(scores)) if scores else 0.0
    mean_f1 = float(sum(f1s) / len(f1s)) if f1s else 0.0

    return {
        "mean_score": mean_score,
        "mean_f1": mean_f1,
        "per_vessel": results,
        "n_vessels": len(scores),
    }
