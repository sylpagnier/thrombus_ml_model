"""Shared helpers for clot ML side-pivot experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_forecast import build_clot_forecast_pair_step, iter_forecast_pairs
from src.core_physics.clot_temporal_growth_rules import (
    _shape_from_phi_at_time,
    _time_frac_at_index,
    deploy_score_from_eval_row,
    reset_temporal_kinematics_cache,
)
from src.training.train_clot_phi_simple import _clot_metrics
from src.utils.paths import get_project_root


def pivot_out_dir(name: str) -> Path:
    return get_project_root() / f"outputs/biochem/clot_ml_ladder/pivot_{name}"


def save_pivot_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    meta: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "meta": meta}, path)


def load_pivot_checkpoint(
    path: Path | str,
    *,
    device: torch.device,
    model_ctor: Callable[[dict[str, Any]], torch.nn.Module],
) -> tuple[torch.nn.Module, dict[str, Any]]:
    raw = torch.load(Path(path), map_location=device, weights_only=False)
    meta = dict(raw.get("meta") or {})
    model = model_ctor(meta).to(device)
    model.load_state_dict(raw["model"], strict=True)
    model.eval()
    return model, meta


@torch.no_grad()
def eval_phi_rollout_on_anchor(
    phi_by_t: dict[int, torch.Tensor],
    *,
    graph_path: Path,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    rule_tag: str,
    pair_stride: int = 1,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reset_temporal_kinematics_cache()
    data = torch.load(graph_path, map_location=device, weights_only=False)
    stem = graph_path.stem
    pairs = iter_forecast_pairs(int(data.y.shape[0]), time_stride=1, pair_stride=pair_stride)
    if not pairs:
        return {"anchor": stem, "n_pairs": 0}

    rows: list[dict[str, float]] = []
    for t_in, t_out in pairs:
        phi = phi_by_t[int(t_out)]
        step = build_clot_forecast_pair_step(data, t_in, t_out, phys_cfg, bio_cfg, device)
        band = _clot_metrics(phi, step.phi_gt, step.loss_mask)
        rows.append(
            {
                "t_frac": _time_frac_at_index(data, int(t_out)),
                **{k: float(v) for k, v in band.items()},
            }
        )

    tfinal = rows[-1]
    early = [r for r in rows if r["t_frac"] <= 0.35]
    t_final_idx = int(pairs[-1][1])
    t_in_final = int(pairs[-1][0])
    phi_final = phi_by_t[t_final_idx]
    shape_final = _shape_from_phi_at_time(
        data,
        phi_final,
        t_in_final,
        t_final_idx,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
    )

    def _mean(key: str, subset: list[dict]) -> float:
        if not subset:
            return float("nan")
        return float(sum(r[key] for r in subset) / len(subset))

    row: dict[str, Any] = {
        "anchor": stem,
        "rule": rule_tag,
        "n_pairs": len(rows),
        "mean_band_f1": _mean("clot_f1", rows),
        "early_mean_pred_frac": _mean("pred_pos_frac", early),
        "early_mean_gt_pos_frac": _mean("gt_pos_frac", early),
        "tfinal_band_f1": float(tfinal["clot_f1"]),
        "tfinal_band_pred_frac": float(tfinal["pred_pos_frac"]),
        "tfinal_gt_pos_frac": float(tfinal["gt_pos_frac"]),
        "tfinal_clot_shape": float(shape_final.get("clot_shape", float("nan"))),
        "tfinal_clot_shape_bal": float(shape_final.get("clot_shape_balanced", float("nan"))),
    }
    row["deploy_score"] = deploy_score_from_eval_row(row)
    if extra:
        row.update(extra)
    return row
