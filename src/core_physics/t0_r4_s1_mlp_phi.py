"""Rung 4 step s1: residual MLP on physics phi inside nucleation mask E(t).

Deploy path: s0 species -> mu physics -> phi_physics; then

    phi_raw = phi_physics + alpha * sigmoid(MLP(feats))   (feats masked to E(t) at apply)

Checkpoint: ``outputs/biochem/t0_r4_s1_mlp_phi/best.pth`` (override ``T0_R4_S1_CKPT``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_continuous_time import macro_tau_at_index
from src.core_physics.clot_nucleation_mask import project_phi_with_nucleation, resolve_nucleation_eligibility
from src.core_physics.clot_phi_simple import (
    _anchor_flow_props,
    _wall_mask_from_data,
    predict_phi_prior_rule,
    sdf_nd_from_data,
)
from src.core_physics.kinematics_clot_prior import clot_prior_score_flat
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time, predict_clot_phi_at_time
from src.core_physics.t0_rung4_ladder import (
    FI_SLICE_IDX,
    MAT_SLICE_IDX,
    _build_s0_deploy_species,
    _s0_risk_normalized,
    rung4_use_dgamma_wall_seed,
)
from src.core_physics.t0_rung_config import RUNG2_GAMMA_MODE, t0_rung2_env
from src.utils.paths import get_project_root
from src.utils.rheology import compute_shear_rate

DEFAULT_S1_CKPT = "outputs/biochem/t0_r4_s1_mlp_phi/best.pth"
FEATURE_NAMES = (
    "sdf_nd",
    "risk_norm",
    "tau",
    "phi_physics",
    "s0_fi_nd",
    "s0_mat_nd",
    "s0_gate",
    "wall",
    "log10_gamma",
    "rule_phi",
)


def s1_ckpt_path() -> Path:
    raw = (os.environ.get("T0_R4_S1_CKPT") or DEFAULT_S1_CKPT).strip()
    p = Path(raw)
    if not p.is_absolute():
        p = get_project_root() / p
    return p


def s1_residual_alpha() -> float:
    raw = (os.environ.get("T0_R4_S1_RESIDUAL_ALPHA") or "0.85").strip()
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 0.85


class T0R4S1PhiMLP(nn.Module):
    """Tiny per-node residual head (2 hidden layers)."""

    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        h = max(int(hidden), 8)
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, 1),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(x))


@dataclass(frozen=True)
class T0R4S1Bundle:
    model: T0R4S1PhiMLP
    alpha: float
    in_dim: int
    hidden: int
    device: torch.device


def feature_dim() -> int:
    return len(FEATURE_NAMES)


@torch.no_grad()
def build_s1_phi_features(
    data,
    time_index: int,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    *,
    elig: torch.Tensor,
    phi_physics: torch.Tensor,
    s0_species: torch.Tensor,
    s0_gate: torch.Tensor,
) -> torch.Tensor:
    """Deploy features ``[N, in_dim]`` for s1 residual MLP."""
    t = int(time_index)
    n = int(data.num_nodes)
    y = data.y[t].to(device=device, dtype=torch.float32)
    u_nd = y[:, 0]
    v_nd = y[:, 1]
    props = _anchor_flow_props(data, device)
    risk = clot_prior_score_flat(data, u_nd, v_nd, bio_cfg, props).reshape(-1)
    risk_n = _s0_risk_normalized(risk, elig)
    tau = float(macro_tau_at_index(data, t, bio_cfg=bio_cfg))
    sdf = sdf_nd_from_data(data, device, n)
    wall = _wall_mask_from_data(data, device, n).to(dtype=torch.float32)
    u_col = u_nd.reshape(-1, 1)
    du_dx = torch.sparse.mm(data.G_x, u_col).squeeze(1)
    du_dy = torch.sparse.mm(data.G_y, u_col).squeeze(1)
    dv_dx = torch.sparse.mm(data.G_x, v_nd.reshape(-1, 1)).squeeze(1)
    dv_dy = torch.sparse.mm(data.G_y, v_nd.reshape(-1, 1)).squeeze(1)
    log_g = torch.log10(compute_shear_rate(du_dx, du_dy, dv_dx, dv_dy).clamp(min=1e-8))
    rule_phi, _ = predict_phi_prior_rule(data, device, bio_cfg, t_in=t)
    cols = [
        sdf.reshape(-1),
        risk_n.reshape(-1),
        torch.full((n,), tau, device=device, dtype=torch.float32),
        phi_physics.reshape(-1).float(),
        s0_species[:, FI_SLICE_IDX].reshape(-1),
        s0_species[:, MAT_SLICE_IDX].reshape(-1),
        s0_gate.reshape(-1).float(),
        wall.reshape(-1),
        log_g.reshape(-1),
        rule_phi.reshape(-1).float(),
    ]
    return torch.stack(cols, dim=1)


def _s0_gate_from_species(
    s0_species: torch.Tensor,
    data,
    time_index: int,
    device: torch.device,
    bio_cfg: BiochemConfig,
    elig: torch.Tensor,
) -> torch.Tensor:
    from src.core_physics.t0_rung4_ladder import (
        _log1p_nd_for_fi_si,
        _log1p_nd_for_mat_si,
        _s0_fi_mat_gain,
        resting_species_log_nd,
    )

    rest = resting_species_log_nd(data, device)
    gain = _s0_fi_mat_gain()
    fi_tgt = _log1p_nd_for_fi_si(float(bio_cfg.viscosity_fi_crit) * gain, bio_cfg, device)
    mat_tgt = _log1p_nd_for_mat_si(float(bio_cfg.viscosity_mat_crit) * gain, bio_cfg)
    fi_d = (s0_species[:, FI_SLICE_IDX] - rest[:, FI_SLICE_IDX]).clamp(min=0.0)
    mat_d = (s0_species[:, MAT_SLICE_IDX] - rest[:, MAT_SLICE_IDX]).clamp(min=0.0)
    fi_g = fi_d / max(fi_tgt - float(rest[0, FI_SLICE_IDX].item()), 1e-12)
    mat_g = mat_d / max(mat_tgt - float(rest[0, MAT_SLICE_IDX].item()), 1e-12)
    gate = torch.maximum(fi_g, mat_g).clamp(0.0, 1.0)
    return gate * elig.to(dtype=torch.float32)


def load_s1_bundle(
    ckpt_path: str | Path | None = None,
    *,
    device: torch.device | None = None,
    quiet: bool = False,
) -> T0R4S1Bundle | None:
    path = Path(ckpt_path) if ckpt_path is not None else s1_ckpt_path()
    if not path.is_file():
        if not quiet:
            print(f"[WARN] s1 checkpoint missing: {path}")
        return None
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(path, map_location=dev, weights_only=False)
    in_dim = int(payload.get("in_dim", feature_dim()))
    hidden = int(payload.get("hidden", 32))
    alpha = float(payload.get("alpha", s1_residual_alpha()))
    model = T0R4S1PhiMLP(in_dim=in_dim, hidden=hidden).to(dev)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return T0R4S1Bundle(model=model, alpha=alpha, in_dim=in_dim, hidden=hidden, device=dev)


def save_s1_checkpoint(
    path: str | Path,
    model: T0R4S1PhiMLP,
    *,
    alpha: float,
    hidden: int,
    meta: dict[str, Any] | None = None,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "in_dim": feature_dim(),
        "hidden": int(hidden),
        "alpha": float(alpha),
        "feature_names": list(FEATURE_NAMES),
        "meta": meta or {},
    }
    torch.save(payload, out)
    sidecar = out.with_suffix(".json")
    sidecar.write_text(json.dumps({k: v for k, v in payload.items() if k != "model_state"}, indent=2), encoding="utf-8")


def predict_s1_phi_raw_at_time(
    data,
    time_index: int,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    bundle: T0R4S1Bundle,
    *,
    commits_prev: torch.Tensor | None,
    pred_species_series: torch.Tensor | None = None,
    nucleation_hops: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(phi_raw, phi_physics, elig)`` before nucleation projection."""
    t = int(time_index)
    elig = resolve_nucleation_eligibility(
        data,
        t,
        device,
        phys_cfg,
        bio_cfg,
        commits_prev=commits_prev,
        growth_seed="pred",
        nucleation_hops=nucleation_hops,
        use_dgamma_wall_seed=rung4_use_dgamma_wall_seed(),
    ).reshape(-1).bool()
    s0_sp = _build_s0_deploy_species(
        data, t, device, bio_cfg, elig=elig, commits_prev=commits_prev
    )
    if pred_species_series is not None:
        series = pred_species_series
    else:
        series = data.y.clone().to(device=device)
        series[t, :, 4:16] = s0_sp
    with t0_rung2_env():
        phi_phys, _ = predict_clot_phi_at_time(
            data,
            t,
            phys_cfg,
            bio_cfg,
            device,
            gamma_mode=RUNG2_GAMMA_MODE,
            flow_source="gt",
            pred_species_series=series,
        )
    gate = _s0_gate_from_species(s0_sp, data, t, device, bio_cfg, elig)
    feats = build_s1_phi_features(
        data,
        t,
        device,
        phys_cfg,
        bio_cfg,
        elig=elig,
        phi_physics=phi_phys,
        s0_species=s0_sp,
        s0_gate=gate,
    )
    delta = bundle.alpha * bundle.model(feats)
    mask_f = elig.to(dtype=phi_phys.dtype)
    phi_raw = (phi_phys.reshape(-1) + delta.reshape(-1) * mask_f).clamp(0.0, 1.0)
    return phi_raw, phi_phys.reshape(-1), elig


def rollout_s1_phi_trajectory(
    data,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    bundle: T0R4S1Bundle,
    *,
    nucleation_hops: int = 1,
) -> dict[int, torch.Tensor]:
    """Coupled s1 rollout with nucleation projection."""
    n_steps = int(data.y.shape[0])
    phi_prev: torch.Tensor | None = None
    commits_prev: torch.Tensor | None = None
    pred_series = data.y.clone().to(device=device)
    out: dict[int, torch.Tensor] = {}

    with t0_rung2_env():
        for t in range(n_steps):
            elig = resolve_nucleation_eligibility(
                data,
                t,
                device,
                phys_cfg,
                bio_cfg,
                commits_prev=commits_prev,
                growth_seed="pred",
                nucleation_hops=nucleation_hops,
                use_dgamma_wall_seed=rung4_use_dgamma_wall_seed(),
            ).reshape(-1).bool()
            s0_sp = _build_s0_deploy_species(
                data, t, device, bio_cfg, elig=elig, commits_prev=commits_prev
            )
            pred_series[t, :, 4:16] = s0_sp
            phi_raw, _, _ = predict_s1_phi_raw_at_time(
                data,
                t,
                device,
                phys_cfg,
                bio_cfg,
                bundle,
                commits_prev=commits_prev,
                pred_species_series=pred_series,
                nucleation_hops=nucleation_hops,
            )
            phi = project_phi_with_nucleation(phi_raw, phi_prev, elig)
            out[t] = phi
            commits_prev = (phi.reshape(-1) >= 0.5).bool()
            phi_prev = phi.detach()
    return out


def s1_train_target_phi(
    data,
    time_index: int,
    phys_cfg: PhysicsConfig,
    device: torch.device,
) -> torch.Tensor:
    """GT growth clot phi (supervision only)."""
    return gt_clot_phi_at_time(data, int(time_index), phys_cfg, device).reshape(-1)
