"""Step 2 ML ladder: 2-layer band GNN risk ranker + frozen Step-0 progressive shell."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_forecast import build_clot_forecast_pair_step, iter_forecast_pairs
from src.core_physics.clot_growth_masks import resolve_ceiling_mask
from src.core_physics.clot_localized_spatial import normalize_risk_per_wall_half
from src.core_physics.clot_phi_simple import node_features_from_gt
from src.core_physics.clot_temporal_growth_rules import (
    _resolve_pool_risk,
    _resolve_uv_for_temporal_risk,
    _shape_from_phi_at_time,
    _time_frac_at_index,
    deploy_score_from_eval_row,
    predict_phi_temporal_at_time,
    reset_temporal_kinematics_cache,
)
from src.training.clot_ml_step0_coef import load_step0_coef_json
from src.training.train_clot_phi_simple import _clot_metrics
from src.utils.paths import get_project_root


def step2_feature_dim() -> int:
    return 3  # minimal pred-kine physics (sdf, log gamma, neg_dgamma_dx)


class _BandGNNConv(MessagePassing):
    """One-hop conv for ceiling-band risk (same pattern as ClotPhiMPNNHybrid)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__(aggr="add")
        self.lin_nei = nn.Linear(in_dim, out_dim)
        self.lin_self = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.propagate(edge_index, x=x)

    def message(self, x_j: torch.Tensor, x_i: torch.Tensor) -> torch.Tensor:
        return F.silu(self.lin_nei(x_j) + self.lin_self(x_i))


class ClotBandRiskGNN(nn.Module):
    """Two-layer GNN on biochem mesh -> per-node risk logit delta."""

    def __init__(self, in_dim: int = 3, hidden: int = 32):
        super().__init__()
        h = max(int(hidden), 8)
        self.conv1 = _BandGNNConv(in_dim, h)
        self.conv2 = _BandGNNConv(h, h)
        self.head = nn.Linear(h, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.35)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward_hidden(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.conv1(x, edge_index))
        return F.silu(self.conv2(h, edge_index))

    def forward_logits(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_hidden(x, edge_index)).squeeze(-1)


def band_features_pred_kine(
    data,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    flow_time: int = 0,
) -> torch.Tensor:
    u, v = _resolve_uv_for_temporal_risk(data, flow_time, device)
    ti = max(0, min(int(flow_time), int(data.y.shape[0]) - 1))
    y = data.y[ti].to(device=device, dtype=torch.float32)
    return node_features_from_gt(
        data,
        y,
        phys_cfg,
        bio_cfg,
        device=device,
        time_index=ti,
        u_nd_override=u,
        v_nd_override=v,
    )


def hand_risk_on_pool(
    data,
    rule_cfg,
    *,
    device: torch.device,
    bio_cfg: BiochemConfig,
    ceiling: torch.Tensor,
    t_out: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    pool, risk = _resolve_pool_risk(
        data,
        device=device,
        bio_cfg=bio_cfg,
        ceiling=ceiling,
        cfg=rule_cfg,
        t_out=int(t_out),
    )
    return pool, risk


def combine_gnn_risk(
    hand_risk: torch.Tensor,
    gnn_logit: torch.Tensor,
    pool: torch.Tensor,
    data,
    rule_cfg,
    *,
    device: torch.device,
    delta_scale: float = 0.30,
) -> torch.Tensor:
    """Residual on hand risk; zero outside pool; optional per-half renorm."""
    delta = torch.tanh(gnn_logit.reshape(-1)) * float(delta_scale)
    risk = (hand_risk.reshape(-1) + delta).clamp(0.0, 1.0) * pool.reshape(-1).float()
    loc = rule_cfg.localized
    if loc is not None and loc.normalize_risk_per_half:
        risk = normalize_risk_per_wall_half(risk, data, device, pool, loc)
    return risk


def gnn_risk_from_model(
    model: ClotBandRiskGNN,
    data,
    rule_cfg,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    ceiling: torch.Tensor,
    t_out: int,
    delta_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    pool, hand = hand_risk_on_pool(
        data, rule_cfg, device=device, bio_cfg=bio_cfg, ceiling=ceiling, t_out=t_out
    )
    feats = band_features_pred_kine(
        data, device=device, phys_cfg=phys_cfg, bio_cfg=bio_cfg, flow_time=rule_cfg.risk_flow_time
    )
    edge_index = data.edge_index.to(device)
    logits = model.forward_logits(feats, edge_index)
    risk = combine_gnn_risk(
        hand,
        logits,
        pool,
        data,
        rule_cfg,
        device=device,
        delta_scale=delta_scale,
    )
    return pool, risk


def rollout_step2_phi(
    data,
    rule_cfg,
    model: ClotBandRiskGNN,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    delta_scale: float = 0.30,
    time_stride: int = 1,
) -> dict[int, torch.Tensor]:
    n_times = int(data.y.shape[0])
    t_final = n_times - 1
    ceiling = resolve_ceiling_mask(data, device, bio_cfg)
    phi_by_t: dict[int, torch.Tensor] = {}
    phi_prev: torch.Tensor | None = None
    for t_out in range(0, n_times, max(int(time_stride), 1)):
        _, risk = gnn_risk_from_model(
            model,
            data,
            rule_cfg,
            device=device,
            phys_cfg=phys_cfg,
            bio_cfg=bio_cfg,
            ceiling=ceiling,
            t_out=t_out,
            delta_scale=delta_scale,
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
            use_provided_risk=True,
        )
        phi_by_t[int(t_out)] = phi
        phi_prev = phi
    return phi_by_t


def _pairwise_rank_loss(
    risk: torch.Tensor,
    phi_gt: torch.Tensor,
    pool: torch.Tensor,
    *,
    margin: float = 0.05,
    max_pairs: int = 384,
) -> torch.Tensor:
    idx = pool.reshape(-1).bool().nonzero(as_tuple=True)[0]
    if int(idx.numel()) < 2:
        return risk.sum() * 0.0
    r = risk[idx]
    y = phi_gt[idx].reshape(-1)
    pos = y > 0.35
    neg = y < 0.08
    if not bool(pos.any().item()) or not bool(neg.any().item()):
        return F.mse_loss(r, y.clamp(0.0, 1.0))
    rp = risk[idx[pos]]
    rn = risk[idx[neg]]
    n_pos = int(rp.numel())
    n_neg = int(rn.numel())
    n_sample = min(max_pairs, max(n_pos * n_neg, 1))
    gp = torch.randint(0, n_pos, (n_sample,), device=risk.device)
    gn = torch.randint(0, n_neg, (n_sample,), device=risk.device)
    return F.relu(float(margin) + rn[gn] - rp[gp]).mean()


@dataclass
class Step2TrainConfig:
    step0_json: str = "outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json"
    hidden: int = 32
    lr: float = 1e-3
    epochs: int = 50
    delta_scale: float = 0.30
    early_paint_weight: float = 0.20
    final_rank_weight: float = 2.0


def train_one_graph(
    model: ClotBandRiskGNN,
    data,
    rule_cfg,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    delta_scale: float,
    early_paint_weight: float,
    final_rank_weight: float,
) -> torch.Tensor:
    reset_temporal_kinematics_cache()
    data = data.to(device)
    ceiling = resolve_ceiling_mask(data, device, bio_cfg)
    pairs = iter_forecast_pairs(int(data.y.shape[0]), time_stride=1)
    if not pairs:
        return torch.tensor(0.0, device=device)
    t_final = int(pairs[-1][1])
    total = torch.tensor(0.0, device=device)
    n = 0
    edge_index = data.edge_index.to(device)
    feats = band_features_pred_kine(
        data,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        flow_time=rule_cfg.risk_flow_time,
    )
    logits = model.forward_logits(feats, edge_index)
    for t_in, t_out in pairs:
        pool, hand = hand_risk_on_pool(
            data,
            rule_cfg,
            device=device,
            bio_cfg=bio_cfg,
            ceiling=ceiling,
            t_out=int(t_out),
        )
        risk = combine_gnn_risk(
            hand,
            logits,
            pool,
            data,
            rule_cfg,
            device=device,
            delta_scale=delta_scale,
        )
        step = build_clot_forecast_pair_step(data, t_in, t_out, phys_cfg, bio_cfg, device)
        phi_gt = step.phi_gt.reshape(-1).to(risk.dtype)
        t_frac = _time_frac_at_index(data, int(t_out))
        w = final_rank_weight if int(t_out) == t_final else 1.0
        loss = w * _pairwise_rank_loss(risk, phi_gt, pool)
        if t_frac <= 0.35:
            if bool(pool.any().item()):
                loss = loss + early_paint_weight * risk[pool].mean()
        total = total + loss
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def eval_step2_on_anchor(
    model: ClotBandRiskGNN,
    rule_cfg,
    *,
    graph_path: Path,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    delta_scale: float,
    pair_stride: int = 1,
) -> dict[str, Any]:
    reset_temporal_kinematics_cache()
    data = torch.load(graph_path, map_location=device, weights_only=False)
    stem = graph_path.stem
    phi_by_t = rollout_step2_phi(
        data,
        rule_cfg,
        model,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        delta_scale=delta_scale,
    )
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

    out = {
        "anchor": stem,
        "rule": "ml_step2_band_gnn",
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
    out["deploy_score"] = deploy_score_from_eval_row(out)
    return out


def default_step2_out_dir() -> Path:
    return get_project_root() / "outputs/biochem/clot_ml_ladder/step2_band_gnn"


def save_step2_checkpoint(
    path: Path,
    *,
    model: ClotBandRiskGNN,
    meta: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "meta": meta}, path)


def load_step2_checkpoint(
    path: Path | str,
    *,
    device: torch.device,
) -> tuple[ClotBandRiskGNN, dict[str, Any]]:
    raw = torch.load(Path(path), map_location=device, weights_only=False)
    meta = dict(raw.get("meta") or {})
    hidden = int(meta.get("hidden", 32))
    model = ClotBandRiskGNN(in_dim=step2_feature_dim(), hidden=hidden).to(device)
    model.load_state_dict(raw["model"], strict=True)
    model.eval()
    return model, meta


def resolve_step2_rule_cfg(step0_json: str | Path) -> Any:
    return load_step0_coef_json(step0_json).to_rule_config(name="ml_step0_coef")
