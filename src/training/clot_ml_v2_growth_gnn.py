"""V3: band GNN growth rate head + nucleation projection (no ceiling)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_continuous_time import (
    feature_time_index,
    growth_time_frac,
    macro_tau_at_index,
    rollout_time_indices,
    sim_end_scale_from_env,
)
from src.core_physics.clot_forecast import build_clot_forecast_pair_step, iter_forecast_pairs
from src.core_physics.clot_growth_masks import pred_clot_mask
from src.core_physics.clot_nucleation_mask import (
    catalytic_rate_multiplier,
    project_phi_with_nucleation,
    resolve_catalytic_hood,
    resolve_nucleation_eligibility,
    snapshot_nucleation_config,
)
from src.core_physics.clot_phi_simple import (
    _hop_distance_from_seed,
    _wall_mask_from_data,
    carreau_mu_si_from_uv,
)
from src.core_physics.clot_temporal_growth_rules import (
    _resolve_uv_for_temporal_risk,
    _time_frac_at_index,
    reset_temporal_kinematics_cache,
)
from src.training.clot_ml_step1_residual import apply_step1_eval_env, resolve_step1_rule_cfg
from src.training.clot_ml_step2_band_gnn import band_features_pred_kine
from src.training.clot_ml_v2_step1_nucleation import eval_phi_by_t_on_anchor
from src.utils.paths import get_project_root


def growth_gnn_feature_dim() -> int:
    """kine(3) + phi_prev + catalytic_hood + log10_mu_c + hop_wall + tau."""
    return 8


def rate_scale_from_env() -> float:
    raw = (os.environ.get("CLOT_V3_RATE_SCALE") or "5.0").strip()
    try:
        return max(float(raw), 0.1)
    except ValueError:
        return 5.0


def v31_recipe_enabled() -> bool:
    return (os.environ.get("CLOT_V31_RECIPE") or "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def rate_scale_for_recipe() -> float:
    if v31_recipe_enabled():
        raw = (os.environ.get("CLOT_V3_RATE_SCALE") or "2.5").strip()
    else:
        raw = (os.environ.get("CLOT_V3_RATE_SCALE") or "5.0").strip()
    try:
        return max(float(raw), 0.1)
    except ValueError:
        return 2.5 if v31_recipe_enabled() else 5.0


def max_step_delta_from_env() -> float:
    raw = (os.environ.get("CLOT_V31_MAX_STEP_DELTA") or "0.10").strip()
    try:
        return max(float(raw), 0.01)
    except ValueError:
        return 0.10


def hard_commit_for_recipe() -> bool:
    if v31_recipe_enabled():
        return (os.environ.get("CLOT_V31_HARD_COMMIT") or "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
    return True


def apply_step3_v3_env(*, sim_end_scale: float = 1.0, v31: bool = False) -> None:
    """Pred kine + macro tau + nucleation (no ceiling in forward)."""
    apply_step1_eval_env()
    os.environ["CLOT_V2_NUCLEATION"] = "1"
    os.environ.setdefault("CLOT_V2_NUCLEATION_HOPS", "1")
    os.environ.setdefault("CLOT_V2_CATALYTIC_HOPS", "1")
    os.environ.setdefault("CLOT_ML_USE_MACRO_TAU", "1")
    os.environ["CLOT_ML_CONTINUOUS_EXTRAP"] = "0"
    os.environ["CLOT_ML_SIM_END_SCALE"] = str(max(float(sim_end_scale), 1.0))
    if v31:
        os.environ["CLOT_V31_RECIPE"] = "1"
        os.environ.setdefault("CLOT_V3_RATE_SCALE", "2.5")
        os.environ.setdefault("CLOT_V31_MAX_STEP_DELTA", "0.10")


class _GrowthGNNConv(MessagePassing):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__(aggr="add")
        self.lin_nei = nn.Linear(in_dim, out_dim)
        self.lin_self = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.propagate(edge_index, x=x)

    def message(self, x_j: torch.Tensor, x_i: torch.Tensor) -> torch.Tensor:
        return F.silu(self.lin_nei(x_j) + self.lin_self(x_i))


class ClotGrowthRateGNN(nn.Module):
    """Two-layer band GNN -> growth rate logit (Euler integration in rollout)."""

    def __init__(self, in_dim: int = 8, hidden: int = 32):
        super().__init__()
        h = max(int(hidden), 8)
        self.conv1 = _GrowthGNNConv(in_dim, h)
        self.conv2 = _GrowthGNNConv(h, h)
        self.head = nn.Linear(h, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.35)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, -0.5)

    def forward_rate_logits(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.conv1(x, edge_index))
        h = F.silu(self.conv2(h, edge_index))
        return self.head(h).squeeze(-1)


def _dtau_between(
    data,
    t_prev: int,
    t_cur: int,
    *,
    bio_cfg: BiochemConfig,
) -> float:
    tau_cur = macro_tau_at_index(data, int(t_cur), bio_cfg=bio_cfg)
    tau_prev = macro_tau_at_index(data, int(t_prev), bio_cfg=bio_cfg) if int(t_prev) >= 0 else 0.0
    return max(float(tau_cur - tau_prev), 1e-6)


def growth_gnn_features(
    data,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    phi_prev: torch.Tensor | None,
    catalytic_hood: torch.Tensor,
    t_out: int,
    flow_time: int,
) -> torch.Tensor:
    n = int(data.num_nodes)
    t_feat = max(0, min(int(flow_time), int(data.y.shape[0]) - 1))
    u, v = _resolve_uv_for_temporal_risk(data, t_feat, device)
    kine = band_features_pred_kine(
        data,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        flow_time=t_feat,
    )
    prev = (
        phi_prev.reshape(-1).float().to(device=device)
        if phi_prev is not None
        else torch.zeros(n, device=device, dtype=kine.dtype)
    )
    hood = catalytic_hood.reshape(-1).float().to(device=device, dtype=kine.dtype)
    mu_c = carreau_mu_si_from_uv(data, u, v, phys_cfg).reshape(-1).clamp(min=1e-8)
    log_mu = torch.log10(mu_c).to(dtype=kine.dtype)
    wall = _wall_mask_from_data(data, device, n)
    hop = _hop_distance_from_seed(wall, data.edge_index.to(device=device), max_hops=8)
    hop_f = (hop.float() / 8.0).clamp(0.0, 1.0)
    tau = torch.full(
        (n,),
        float(growth_time_frac(data, int(t_out), bio_cfg=bio_cfg)),
        device=device,
        dtype=kine.dtype,
    )
    return torch.cat(
        [
            kine,
            prev.reshape(-1, 1),
            hood.reshape(-1, 1),
            log_mu.reshape(-1, 1),
            hop_f.reshape(-1, 1),
            tau.reshape(-1, 1),
        ],
        dim=1,
    )


def eligibility_union_mask(
    data,
    phi_by_t: dict[int, torch.Tensor],
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
) -> torch.Tensor:
    """Union of E_seed over rollout (training supervision support)."""
    n = int(data.num_nodes)
    union = torch.zeros(n, device=device, dtype=torch.bool)
    hist: dict[int, torch.Tensor] = {}
    for t in sorted(int(k) for k in phi_by_t.keys()):
        hist[int(t)] = phi_by_t[int(t)].detach()
        elig = resolve_nucleation_eligibility(
            data,
            int(t),
            device,
            phys_cfg,
            bio_cfg,
            growth_seed="pred",
            phi_pred_by_time=hist,
        )
        union = union | elig.reshape(-1).bool()
    return union


def rollout_v3_growth_gnn(
    model: ClotGrowthRateGNN,
    data,
    rule_cfg,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    sim_end_scale: float | None = None,
    commit_thresh: float = 0.5,
) -> dict[int, torch.Tensor]:
    """Euler phi += dtau * sigmoid(rate) * catalytic boost; nucleation project each step."""
    scale = float(sim_end_scale if sim_end_scale is not None else sim_end_scale_from_env())
    apply_step3_v3_env(sim_end_scale=scale)
    data = data.to(device)
    edge_index = data.edge_index.to(device)
    n = int(data.num_nodes)
    onset = float(getattr(rule_cfg, "global_onset_frac", 0.0) or 0.0)
    rate_scale = rate_scale_for_recipe()
    max_delta = max_step_delta_from_env() if v31_recipe_enabled() else None
    hard_commit = hard_commit_for_recipe()
    t_indices = rollout_time_indices(data, sim_end_scale=scale)
    out: dict[int, torch.Tensor] = {}
    phi_prev: torch.Tensor | None = None

    for i, t_out in enumerate(t_indices):
        t_out = int(t_out)
        if i == 0:
            phi0 = torch.zeros(n, device=device, dtype=torch.float32)
            out[t_out] = phi0
            phi_prev = phi0
            continue

        t_frac = growth_time_frac(data, t_out, bio_cfg=bio_cfg)
        if onset > 0.0 and t_frac < onset:
            phi = phi_prev.clone() if phi_prev is not None else torch.zeros(n, device=device)
            out[t_out] = phi
            phi_prev = phi
            continue

        t_prev = int(t_indices[i - 1])
        dtau = _dtau_between(data, t_prev, t_out, bio_cfg=bio_cfg)
        t_feat = feature_time_index(data, t_out)
        commits = (
            pred_clot_mask(phi_prev, thresh=commit_thresh)
            if phi_prev is not None
            else torch.zeros(n, device=device, dtype=torch.bool)
        )
        hood = resolve_catalytic_hood(commits, edge_index)
        feats = growth_gnn_features(
            data,
            device=device,
            phys_cfg=phys_cfg,
            bio_cfg=bio_cfg,
            phi_prev=phi_prev,
            catalytic_hood=hood,
            t_out=t_out,
            flow_time=t_feat,
        )
        rate = model.forward_rate_logits(feats, edge_index)
        cat_mult = catalytic_rate_multiplier(hood)
        assert phi_prev is not None
        delta = float(dtau) * torch.sigmoid(rate) * cat_mult * rate_scale
        if max_delta is not None:
            delta = delta.clamp(max=float(max_delta))
        phi_prop = (phi_prev + delta).clamp(0.0, 1.0)
        elig = resolve_nucleation_eligibility(
            data,
            t_out,
            device,
            phys_cfg,
            bio_cfg,
            growth_seed="pred",
            phi_pred_by_time=out,
            commit_thresh=commit_thresh,
        )
        phi = project_phi_with_nucleation(
            phi_prop,
            phi_prev,
            elig,
            commit_thresh=commit_thresh,
            hard_commit=hard_commit,
        )
        out[t_out] = phi
        phi_prev = phi if not model.training else phi

    return out


@dataclass
class V31TrainConfig:
    step0_json: str = "outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json"
    hidden: int = 32
    lr: float = 8e-4
    epochs: int = 32
    init_ckpt: str = ""
    teacher_weight: float = 0.08
    teacher_ckpt: str = "outputs/biochem/sweep_clot_ml_physics_6h/step1_a35/clot_ml_step1_best.pth"
    final_frame_boost: float = 1.5


@dataclass
class V3TrainConfig:
    step0_json: str = "outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json"
    hidden: int = 32
    lr: float = 1e-3
    epochs: int = 32
    early_paint_weight: float = 0.35
    final_bce_weight: float = 2.0
    teacher_weight: float = 0.15
    teacher_ckpt: str = "outputs/biochem/sweep_clot_ml_physics_6h/step1_a35/clot_ml_step1_best.pth"


def train_one_graph(
    model: ClotGrowthRateGNN,
    data,
    rule_cfg,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    early_paint_weight: float,
    final_bce_weight: float,
    teacher_phi_by_t: dict[int, torch.Tensor] | None = None,
    teacher_weight: float = 0.0,
) -> torch.Tensor:
    reset_temporal_kinematics_cache()
    data = data.to(device)
    phi_by_t = rollout_v3_growth_gnn(
        model,
        data,
        rule_cfg,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        sim_end_scale=1.0,
    )
    union = eligibility_union_mask(
        data, phi_by_t, device=device, phys_cfg=phys_cfg, bio_cfg=bio_cfg
    )
    if not bool(union.any().item()):
        return phi_by_t[int(min(phi_by_t.keys()))].sum() * 0.0

    pairs = iter_forecast_pairs(int(data.y.shape[0]), time_stride=1)
    if not pairs:
        return phi_by_t[int(min(phi_by_t.keys()))].sum() * 0.0
    t_final = int(pairs[-1][1])
    total = torch.tensor(0.0, device=device)
    n = 0
    for t_in, t_out in pairs:
        phi = phi_by_t[int(t_out)]
        step = build_clot_forecast_pair_step(data, t_in, t_out, phys_cfg, bio_cfg, device)
        mask = union
        if not bool(mask.any().item()):
            continue
        target = step.phi_gt.reshape(-1).to(phi.dtype)
        w = final_bce_weight if int(t_out) == t_final else 1.0
        bce = F.binary_cross_entropy(phi[mask], target[mask], reduction="mean")
        loss = w * bce
        t_frac = _time_frac_at_index(data, int(t_out))
        if t_frac <= 0.35:
            loss = loss + early_paint_weight * phi[mask].mean()
        if teacher_phi_by_t is not None and teacher_weight > 0.0:
            t_key = int(t_out)
            if t_key in teacher_phi_by_t:
                teach = teacher_phi_by_t[t_key].reshape(-1).to(device=device, dtype=phi.dtype)
                loss = loss + float(teacher_weight) * F.mse_loss(phi[mask], teach[mask])
        total = total + loss
        n += 1
    return total / max(n, 1)


def train_one_graph_v31(
    model: ClotGrowthRateGNN,
    data,
    rule_cfg,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    loss_cfg: Any | None = None,
    teacher_phi_by_t: dict[int, torch.Tensor] | None = None,
    teacher_weight: float = 0.0,
    final_frame_boost: float = 1.5,
) -> torch.Tensor:
    from src.core_physics.clot_nucleation_mask import resolve_wall_mask
    from src.training.clot_growth_loss import (
        ClotGrowthLossConfig,
        clot_growth_frame_loss,
        hop_penalty_from_gt_clot,
        temporal_frame_weight,
    )

    reset_temporal_kinematics_cache()
    apply_step3_v3_env(v31=True)
    data = data.to(device)
    phi_by_t = rollout_v3_growth_gnn(
        model,
        data,
        rule_cfg,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        sim_end_scale=1.0,
    )
    pairs = iter_forecast_pairs(int(data.y.shape[0]), time_stride=1)
    if not pairs:
        return phi_by_t[int(min(phi_by_t.keys()))].sum() * 0.0
    t_final = int(pairs[-1][1])
    lc = loss_cfg or ClotGrowthLossConfig()
    wall = resolve_wall_mask(data, device)
    edge_index = data.edge_index
    n_nodes = int(data.num_nodes)
    total = torch.tensor(0.0, device=device)
    wsum = 0.0
    for t_in, t_out in pairs:
        phi = phi_by_t[int(t_out)]
        step = build_clot_forecast_pair_step(data, t_in, t_out, phys_cfg, bio_cfg, device)
        mask = step.loss_mask.reshape(-1).bool()
        if not bool(mask.any().item()):
            continue
        target = step.phi_gt.reshape(-1).to(phi.dtype)
        gt_bin = (target >= 0.5).detach().cpu().numpy()
        hop_np = hop_penalty_from_gt_clot(edge_index, n_nodes, gt_bin)
        hop_t = torch.from_numpy(hop_np).to(device=device, dtype=phi.dtype)
        parts = clot_growth_frame_loss(
            phi, target, mask, hop_t, cfg=lc, wall_mask=wall
        )
        t_frac = _time_frac_at_index(data, int(t_out))
        fw = temporal_frame_weight(t_frac, equal=lc.temporal_equal)
        if int(t_out) == t_final:
            fw *= float(final_frame_boost)
        loss = parts["loss"]
        if teacher_phi_by_t is not None and teacher_weight > 0.0 and int(t_out) in teacher_phi_by_t:
            teach = teacher_phi_by_t[int(t_out)].reshape(-1).to(device=device, dtype=phi.dtype)
            loss = loss + float(teacher_weight) * F.mse_loss(phi[mask], teach[mask])
        total = total + fw * loss
        wsum += fw
    return total / max(wsum, 1e-6)


def val_score_from_eval_row(row: dict[str, Any]) -> float:
    """Composite selection: deploy + spatial shape (punish ring-paint)."""
    deploy = float(row.get("deploy_score", 0.0))
    shape = float(row.get("tfinal_clot_shape", 0.0))
    bal = float(row.get("tfinal_clot_shape_bal", 0.0))
    if not (shape == shape):
        shape = 0.0
    if not (bal == bal):
        bal = shape
    return 0.35 * deploy + 0.40 * shape + 0.25 * bal


@torch.no_grad()
def eval_v3_on_anchor(
    model: ClotGrowthRateGNN,
    rule_cfg,
    *,
    graph_path: Path,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    sim_end_scale: float = 1.0,
    pair_stride: int = 1,
) -> dict[str, Any]:
    reset_temporal_kinematics_cache()
    data = torch.load(graph_path, map_location=device, weights_only=False)
    phi_by_t = rollout_v3_growth_gnn(
        model,
        data,
        rule_cfg,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        sim_end_scale=float(sim_end_scale),
    )
    return eval_phi_by_t_on_anchor(
        phi_by_t,
        data,
        anchor=graph_path.stem,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        pair_stride=pair_stride,
        rule_tag="ml_v3_growth_gnn",
    )


def default_v3_out_dir() -> Path:
    return get_project_root() / "outputs/biochem/clot_ml_ladder_v2/v3_growth_gnn"


def default_v31_out_dir() -> Path:
    return get_project_root() / "outputs/biochem/clot_ml_ladder_v2/v31_growth_gnn"


def save_v3_checkpoint(path: Path, *, model: ClotGrowthRateGNN, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "meta": meta}, path)


def load_v3_checkpoint(
    path: Path | str,
    *,
    device: torch.device,
    v31: bool = False,
) -> tuple[ClotGrowthRateGNN, dict[str, Any]]:
    apply_step3_v3_env(v31=v31)
    raw = torch.load(Path(path), map_location=device, weights_only=False)
    meta = dict(raw.get("meta") or {})
    hidden = int(meta.get("hidden", 32))
    in_dim = int(meta.get("in_dim", growth_gnn_feature_dim()))
    model = ClotGrowthRateGNN(in_dim=in_dim, hidden=hidden).to(device)
    model.load_state_dict(raw["model"], strict=True)
    model.eval()
    return model, meta


def v31_manifest_dict(
    *,
    ckpt: str,
    step0_json: str = "outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json",
) -> dict[str, Any]:
    base = v3_manifest_dict(ckpt=ckpt, step0_json=step0_json)
    base.update(
        {
            "name": "clot_ml_v2_s31_growth_gnn",
            "step": "v31",
            "phi_shell": "v31_growth_gnn",
            "recipe": "v31",
            "env": {
                **base.get("env", {}),
                "CLOT_V31_RECIPE": "1",
                "CLOT_V3_RATE_SCALE": str(rate_scale_for_recipe()),
                "CLOT_V31_MAX_STEP_DELTA": str(max_step_delta_from_env()),
            },
        }
    )
    return base


def v3_manifest_dict(
    *,
    ckpt: str,
    step0_json: str = "outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json",
) -> dict[str, Any]:
    return {
        "name": "clot_ml_v2_s3_growth_gnn",
        "track": "v2",
        "step": "v3",
        "phi_shell": "v3_growth_gnn",
        "step0_json": step0_json,
        "v3_ckpt": ckpt,
        "kine_ckpt": "outputs/kinematics/kinematics_best.pth",
        "vel_source": "kinematics",
        "use_macro_tau": True,
        "continuous_extrap": False,
        "sim_end_scale": 1.0,
        "coupled": False,
        "env": {
            "CLOT_V2_NUCLEATION": "1",
            "CLOT_ML_USE_MACRO_TAU": "1",
            "CLOT_V3_RATE_SCALE": str(rate_scale_from_env()),
        },
        "rollout_growth_seed": "pred",
        "nucleation_config": snapshot_nucleation_config(),
    }


def resolve_v3_rule_cfg(step0_json: str | Path) -> Any:
    return resolve_step1_rule_cfg(get_project_root() / step0_json)


@torch.no_grad()
def teacher_phi_by_t_from_step1(
    data,
    rule_cfg,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    teacher_ckpt: str | Path,
    alpha: float = 0.35,
) -> dict[int, torch.Tensor]:
    from src.training.clot_ml_step1_residual import load_step1_checkpoint
    from src.training.clot_ml_v2_step1_nucleation import rollout_step1_v1_nucleation

    model, meta = load_step1_checkpoint(Path(teacher_ckpt), device=device)
    a = float(meta.get("alpha", alpha))
    return rollout_step1_v1_nucleation(
        data,
        rule_cfg,
        model,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        alpha=a,
        sim_end_scale=1.0,
    )
