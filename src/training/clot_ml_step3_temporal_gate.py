"""Step 3 ML ladder: learned per-vessel onset gate + frozen Step-0 risk shell."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_forecast import build_clot_forecast_pair_step, iter_forecast_pairs
from src.core_physics.clot_growth_masks import resolve_ceiling_mask
from src.core_physics.clot_temporal_growth_rules import (
    _resolve_pool_risk,
    _shape_from_phi_at_time,
    _time_frac_at_index,
    deploy_score_from_eval_row,
    predict_phi_temporal_at_time,
    reset_temporal_kinematics_cache,
)
from src.training.clot_ml_step0_coef import load_step0_coef_json
from src.training.clot_ml_step2_band_gnn import (
    ClotBandRiskGNN,
    band_features_pred_kine,
    step2_feature_dim,
)
from src.training.train_clot_phi_simple import _clot_metrics
from src.utils.paths import get_project_root


def default_onset_bounds() -> tuple[float, float]:
    """Match Step-0 coef search bounds (``onset`` in BOUNDS)."""
    return 0.25, 0.50


def map_onset_logit(logit: torch.Tensor, onset_min: float, onset_max: float) -> torch.Tensor:
    return float(onset_min) + (float(onset_max) - float(onset_min)) * torch.sigmoid(logit)


class ClotTemporalGateModel(nn.Module):
    """Band GNN pools to vessel stats -> scalar onset (replaces fixed inc40)."""

    def __init__(
        self,
        in_dim: int = 3,
        hidden: int = 32,
        *,
        onset_min: float = 0.25,
        onset_max: float = 0.50,
    ):
        super().__init__()
        self.onset_min = float(onset_min)
        self.onset_max = float(onset_max)
        self.encoder = ClotBandRiskGNN(in_dim=in_dim, hidden=hidden)
        stat_dim = 3  # pool_frac, span_nd, n_macro_norm
        self.onset_head = nn.Sequential(
            nn.Linear(hidden + stat_dim, max(hidden, 8)),
            nn.SiLU(),
            nn.Linear(max(hidden, 8), 1),
        )
        for m in self.onset_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.35)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Init ~0.40 onset when range [0.25, 0.50] -> sigmoid target 0.6
        nn.init.zeros_(self.onset_head[-1].weight)
        nn.init.constant_(self.onset_head[-1].bias, math.log(0.6 / 0.4))

    def vessel_stats(
        self,
        data,
        pool: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        pool_f = pool.reshape(-1).float()
        pool_frac = pool_f.mean()
        if hasattr(data, "x") and torch.is_tensor(data.x) and data.x.shape[1] >= 2:
            xy = data.x[:, :2].to(device=device, dtype=torch.float32)
            span = torch.linalg.vector_norm(xy.max(dim=0).values - xy.min(dim=0).values)
        else:
            span = torch.tensor(0.01, device=device, dtype=torch.float32)
        n_macro = float(max(int(data.y.shape[0]) - 1, 1))
        n_norm = torch.tensor(n_macro / 53.0, device=device, dtype=torch.float32)
        return torch.stack([pool_frac, span, n_norm])

    def forward_onset_logit(
        self,
        data,
        pool: torch.Tensor,
        feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        h = self.encoder.forward_hidden(feats, edge_index)
        pool_f = pool.reshape(-1).float()
        if bool(pool.any().item()):
            h_pool = (h * pool_f.unsqueeze(1)).sum(dim=0) / pool_f.sum().clamp(min=1.0)
        else:
            h_pool = h.mean(dim=0)
        stats = self.vessel_stats(data, pool, device=feats.device)
        return self.onset_head(torch.cat([h_pool, stats], dim=0)).squeeze(-1)

    def forward_onset(
        self,
        data,
        pool: torch.Tensor,
        feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        logit = self.forward_onset_logit(data, pool, feats, edge_index)
        return map_onset_logit(logit, self.onset_min, self.onset_max)


@torch.no_grad()
def compute_gt_onset_frac(
    data,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    pos_frac_threshold: float = 0.06,
) -> float:
    """First macro ``t_frac`` where band GT positive fraction exceeds threshold."""
    pairs = iter_forecast_pairs(int(data.y.shape[0]), time_stride=1)
    if not pairs:
        return 0.95
    for t_in, t_out in pairs:
        step = build_clot_forecast_pair_step(data, t_in, t_out, phys_cfg, bio_cfg, device)
        mask = step.loss_mask.reshape(-1).bool()
        if not bool(mask.any().item()):
            continue
        pos_frac = float(step.phi_gt[mask].mean().item())
        if pos_frac >= float(pos_frac_threshold):
            return float(_time_frac_at_index(data, int(t_out)))
    return float(_time_frac_at_index(data, int(pairs[-1][1])))


def rollout_step3_phi(
    data,
    rule_cfg,
    model: ClotTemporalGateModel,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    time_stride: int = 1,
) -> dict[int, torch.Tensor]:
    data = data.to(device)
    n_times = int(data.y.shape[0])
    t_final = n_times - 1
    ceiling = resolve_ceiling_mask(data, device, bio_cfg)
    pool, _ = _resolve_pool_risk(
        data,
        device=device,
        bio_cfg=bio_cfg,
        ceiling=ceiling,
        cfg=rule_cfg,
        t_out=0,
    )
    feats = band_features_pred_kine(
        data,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        flow_time=rule_cfg.risk_flow_time,
    )
    edge_index = data.edge_index.to(device)
    onset = float(model.forward_onset(data, pool, feats, edge_index).item())

    phi_by_t: dict[int, torch.Tensor] = {}
    phi_prev: torch.Tensor | None = None
    for t_out in range(0, n_times, max(int(time_stride), 1)):
        _, risk = _resolve_pool_risk(
            data,
            device=device,
            bio_cfg=bio_cfg,
            ceiling=ceiling,
            cfg=rule_cfg,
            t_out=t_out,
        )
        phi = predict_phi_temporal_at_time(
            data,
            t_out,
            device=device,
            bio_cfg=bio_cfg,
            cfg=rule_cfg,
            ceiling=ceiling,
            risk=risk,
            phi_prev=phi_prev,
            t_final=t_final,
            onset_override=onset,
        )
        phi_by_t[int(t_out)] = phi
        phi_prev = phi
    return phi_by_t


@dataclass
class Step3TrainConfig:
    step0_json: str = "outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json"
    hidden: int = 32
    lr: float = 1e-3
    epochs: int = 50
    onset_min: float = 0.25
    onset_max: float = 0.50
    gt_onset_threshold: float = 0.06
    early_onset_weight: float = 0.35
    late_onset_weight: float = 0.15


def train_one_graph(
    model: ClotTemporalGateModel,
    data,
    rule_cfg,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    gt_onset_threshold: float,
    early_onset_weight: float,
    late_onset_weight: float,
) -> torch.Tensor:
    reset_temporal_kinematics_cache()
    data = data.to(device)
    ceiling = resolve_ceiling_mask(data, device, bio_cfg)
    pool, _ = _resolve_pool_risk(
        data,
        device=device,
        bio_cfg=bio_cfg,
        ceiling=ceiling,
        cfg=rule_cfg,
        t_out=0,
    )
    feats = band_features_pred_kine(
        data,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        flow_time=rule_cfg.risk_flow_time,
    )
    edge_index = data.edge_index.to(device)
    onset_pred = model.forward_onset(data, pool, feats, edge_index)
    onset_gt = torch.tensor(
        compute_gt_onset_frac(
            data,
            device=device,
            phys_cfg=phys_cfg,
            bio_cfg=bio_cfg,
            pos_frac_threshold=gt_onset_threshold,
        ),
        device=device,
        dtype=onset_pred.dtype,
    )
    loss = F.smooth_l1_loss(onset_pred, onset_gt)
    # Penalize predicted onset earlier than GT (anti early-paint timing).
    loss = loss + early_onset_weight * F.relu(onset_gt - onset_pred) ** 2
    # Penalize predicted onset much later than GT (missed growth window).
    loss = loss + late_onset_weight * F.relu(onset_pred - onset_gt - 0.08) ** 2
    return loss


@torch.no_grad()
def eval_step3_on_anchor(
    model: ClotTemporalGateModel,
    rule_cfg,
    *,
    graph_path: Path,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    pair_stride: int = 1,
) -> dict[str, Any]:
    reset_temporal_kinematics_cache()
    data = torch.load(graph_path, map_location=device, weights_only=False)
    stem = graph_path.stem
    phi_by_t = rollout_step3_phi(
        data,
        rule_cfg,
        model,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
    )
    pairs = iter_forecast_pairs(int(data.y.shape[0]), time_stride=1, pair_stride=pair_stride)
    if not pairs:
        return {"anchor": stem, "n_pairs": 0}

    ceiling = resolve_ceiling_mask(data, device, bio_cfg)
    pool, _ = _resolve_pool_risk(
        data, device=device, bio_cfg=bio_cfg, ceiling=ceiling, cfg=rule_cfg, t_out=0
    )
    feats = band_features_pred_kine(
        data, device=device, phys_cfg=phys_cfg, bio_cfg=bio_cfg, flow_time=rule_cfg.risk_flow_time
    )
    onset_pred = float(
        model.forward_onset(data, pool, feats, data.edge_index.to(device)).item()
    )
    onset_gt = compute_gt_onset_frac(
        data, device=device, phys_cfg=phys_cfg, bio_cfg=bio_cfg
    )

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

    out = {
        "anchor": stem,
        "rule": "ml_step3_temporal_gate",
        "n_pairs": len(rows),
        "onset_pred": onset_pred,
        "onset_gt": onset_gt,
        "mean_band_f1": _mean("clot_f1", rows),
        "early_mean_pred_frac": _mean("pred_pos_frac", early),
        "early_mean_gt_pos_frac": _mean("gt_pos_frac", early),
        "tfinal_band_f1": float(tfinal["clot_f1"]),
        "tfinal_band_pred_frac": float(tfinal["pred_pos_frac"]),
        "tfinal_gt_pos_frac": float(tfinal["gt_pos_frac"]),
        "tfinal_clot_shape": float(shape_final.get("clot_shape", float("nan"))),
        "tfinal_clot_shape_bal": float(shape_final.get("clot_shape_balanced", float("nan"))),
    }
    out["deploy_score"] = deploy_score_from_eval_row(out)
    return out


def default_step3_out_dir() -> Path:
    return get_project_root() / "outputs/biochem/clot_ml_ladder/step3_temporal_gate"


def save_step3_checkpoint(
    path: Path,
    *,
    model: ClotTemporalGateModel,
    meta: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "meta": meta}, path)


def load_step3_checkpoint(
    path: Path | str,
    *,
    device: torch.device,
) -> tuple[ClotTemporalGateModel, dict[str, Any]]:
    raw = torch.load(Path(path), map_location=device, weights_only=False)
    meta = dict(raw.get("meta") or {})
    hidden = int(meta.get("hidden", 32))
    onset_min = float(meta.get("onset_min", default_onset_bounds()[0]))
    onset_max = float(meta.get("onset_max", default_onset_bounds()[1]))
    model = ClotTemporalGateModel(
        in_dim=step2_feature_dim(),
        hidden=hidden,
        onset_min=onset_min,
        onset_max=onset_max,
    ).to(device)
    model.load_state_dict(raw["model"], strict=True)
    model.eval()
    return model, meta


def resolve_step3_rule_cfg(step0_json: str | Path) -> Any:
    return load_step0_coef_json(step0_json).to_rule_config(name="ml_step0_coef")
