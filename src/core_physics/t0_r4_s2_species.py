"""Rung 4 step s2: species correction inside E(t) on top of s0 rules.

Modes (``T0_R4_S2_MODE``):

  loc (default) -- relearn **where** via risk reweighting before s0 top-frac gate
  delta         -- deprecated FI/Mat residual (wall-carpet failure)

Deploy (loc)::

    risk_adj = risk_norm * (1 + scale * tanh(MLP(feats))) * onset(tau)
    sp = s0_recipe(risk_adj)   # same FI/Mat ramp, new hotspot mask
    clot = gelation_physics(sp, GT flow)

Checkpoint: ``outputs/biochem/t0_r4_s2_loc/best.pth`` (loc) or ``.../t0_r4_s2_species/`` (delta).
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
from src.core_physics.clot_nucleation_mask import resolve_nucleation_eligibility
from src.core_physics.clot_phi_simple import (
    _anchor_flow_props,
    _wall_mask_from_data,
    predict_phi_prior_rule,
    sdf_nd_from_data,
)
from src.core_physics.kinematics_clot_prior import clot_prior_score_flat
from src.core_physics.t0_mu_physics import predict_clot_phi_at_time
from src.core_physics.t0_rung4_ladder import (
    FI_SLICE_IDX,
    MAT_SLICE_IDX,
    _build_s0_deploy_species,
    _s0_onset_factor,
    _s0_risk_normalized,
    rung4_use_dgamma_wall_seed,
)
from src.core_physics.t0_rung_config import RUNG2_GAMMA_MODE, t0_rung2_env
from src.utils.paths import get_project_root
from src.utils.rheology import compute_shear_rate

DEFAULT_S2_CKPT = "outputs/biochem/t0_r4_s2_loc/best.pth"
DEFAULT_S2_DELTA_CKPT = "outputs/biochem/t0_r4_s2_species/best.pth"
S2_MODE_LOC = "loc"
S2_MODE_DELTA = "delta"
FEATURE_NAMES = (
    "sdf_nd",
    "risk_norm",
    "tau",
    "s0_fi_nd",
    "s0_mat_nd",
    "s0_gate",
    "wall",
    "log10_gamma",
    "rule_phi",
)


def s2_mode_from_env() -> str:
    raw = (os.environ.get("T0_R4_S2_MODE") or S2_MODE_LOC).strip().lower()
    if raw in ("delta", "fi_mat", "v2"):
        return S2_MODE_DELTA
    return S2_MODE_LOC


def s2_loc_scale() -> float:
    raw = (os.environ.get("T0_R4_S2_LOC_SCALE") or "1.5").strip()
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 1.5


def s2_ckpt_path() -> Path:
    mode = s2_mode_from_env()
    default = DEFAULT_S2_DELTA_CKPT if mode == S2_MODE_DELTA else DEFAULT_S2_CKPT
    raw = (os.environ.get("T0_R4_S2_CKPT") or default).strip()
    p = Path(raw)
    if not p.is_absolute():
        p = get_project_root() / p
    return p


def s2_delta_scale() -> float:
    raw = (os.environ.get("T0_R4_S2_DELTA_SCALE") or "1.0").strip()
    try:
        return max(float(raw), 1e-6)
    except ValueError:
        return 1.0


class T0R4S2LocMLP(nn.Module):
    """Per-node risk reweight logit (1 output); zero-init -> s0 hotspots unchanged."""

    def __init__(self, in_dim: int, hidden: int = 48):
        super().__init__()
        h = max(int(hidden), 16)
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
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class T0R4S2SpeciesMLP(nn.Module):
    """Predict FI/Mat log1p ND residuals (2 outputs) on s0 base [deprecated delta mode]."""

    def __init__(self, in_dim: int, hidden: int = 48):
        super().__init__()
        h = max(int(hidden), 16)
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, 2),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # zero init last layer -> start at s0
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class T0R4S2Bundle:
    model: T0R4S2LocMLP | T0R4S2SpeciesMLP
    mode: str
    delta_scale: float
    loc_scale: float
    in_dim: int
    hidden: int
    device: torch.device


def feature_dim() -> int:
    return len(FEATURE_NAMES)


def _s0_gate_from_species(
    s0_species: torch.Tensor,
    data,
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


def build_s2_features(
    data,
    time_index: int,
    device: torch.device,
    bio_cfg: BiochemConfig,
    *,
    elig: torch.Tensor,
    s0_species: torch.Tensor,
    s0_gate: torch.Tensor,
) -> torch.Tensor:
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
        s0_species[:, FI_SLICE_IDX].reshape(-1),
        s0_species[:, MAT_SLICE_IDX].reshape(-1),
        s0_gate.reshape(-1),
        wall.reshape(-1),
        log_g.reshape(-1),
        rule_phi.reshape(-1).float(),
    ]
    return torch.stack(cols, dim=1)


def s3_feature_dim() -> int:
    return len(FEATURE_NAMES) + 2


def build_s3_features(
    data,
    time_index: int,
    device: torch.device,
    bio_cfg: BiochemConfig,
    *,
    elig: torch.Tensor,
    s0_species: torch.Tensor,
    s0_gate: torch.Tensor,
    commits_prev: torch.Tensor | None,
    phi_prev: torch.Tensor | None,
) -> torch.Tensor:
    """s2 wall-band feats + local commit memory (``commit_prev``, ``phi_prev``)."""
    base = build_s2_features(
        data, time_index, device, bio_cfg, elig=elig, s0_species=s0_species, s0_gate=s0_gate
    )
    n = int(data.num_nodes)
    if commits_prev is None:
        cp = torch.zeros(n, device=device, dtype=torch.float32)
    else:
        cp = commits_prev.reshape(-1).to(device=device, dtype=torch.float32)
    if phi_prev is None:
        pp = torch.zeros(n, device=device, dtype=torch.float32)
    else:
        pp = phi_prev.reshape(-1).to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    return torch.cat([base, cp.reshape(-1, 1), pp.reshape(-1, 1)], dim=1)


def apply_s2_species_delta(
    s0_species: torch.Tensor,
    delta: torch.Tensor,
    elig: torch.Tensor,
    *,
    delta_scale: float,
) -> torch.Tensor:
    """Add masked FI/Mat delta to s0 species (log1p ND)."""
    sp = s0_species.clone()
    m = elig.reshape(-1).bool()
    if not bool(m.any().item()):
        return sp
    scale = float(delta_scale)
    d = delta.to(dtype=sp.dtype)
    sp[m, FI_SLICE_IDX] = sp[m, FI_SLICE_IDX] + scale * d[m, 0]
    sp[m, MAT_SLICE_IDX] = sp[m, MAT_SLICE_IDX] + scale * d[m, 1]
    return sp.clamp(min=0.0)


def _risk_n_at_time(
    data,
    time_index: int,
    device: torch.device,
    bio_cfg: BiochemConfig,
    *,
    elig: torch.Tensor,
) -> torch.Tensor:
    t = int(time_index)
    y = data.y[t].to(device=device, dtype=torch.float32)
    props = _anchor_flow_props(data, device)
    risk = clot_prior_score_flat(data, y[:, 0], y[:, 1], bio_cfg, props).reshape(-1).clamp(min=0.0)
    return _s0_risk_normalized(risk, elig)


def _apply_loc_risk_adjustment(
    risk_n: torch.Tensor,
    logit: torch.Tensor,
    elig: torch.Tensor,
    *,
    onset: float,
    loc_scale: float,
) -> torch.Tensor:
    e = elig.reshape(-1).bool()
    adj = risk_n.reshape(-1).clone()
    scale = float(loc_scale) * float(onset)
    boost = 1.0 + scale * torch.tanh(logit.reshape(-1))
    adj = torch.where(e, adj * boost, torch.zeros_like(adj))
    return adj.clamp(min=0.0)


def _apply_loc_gate_residual(
    s0_gate: torch.Tensor,
    logit: torch.Tensor,
    elig: torch.Tensor,
    *,
    onset: float,
    loc_scale: float,
) -> torch.Tensor:
    """Direct gate delta on s0 FI/Mat ramp (differentiable; bypasses top-8% rank cut)."""
    e = elig.reshape(-1).bool()
    delta = float(loc_scale) * float(onset) * torch.tanh(logit.reshape(-1))
    g = s0_gate.reshape(-1).clone()
    return torch.where(e, (g + delta).clamp(0.0, 1.0), g)


def species_from_gate(
    data,
    device: torch.device,
    bio_cfg: BiochemConfig,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Build FI/Mat species from explicit per-node gate in [0, 1]."""
    from src.core_physics.t0_rung4_ladder import (
        _log1p_nd_for_fi_si,
        _log1p_nd_for_mat_si,
        _s0_fi_mat_gain,
        resting_species_log_nd,
    )

    sp = resting_species_log_nd(data, device)
    gain = _s0_fi_mat_gain()
    fi_tgt = _log1p_nd_for_fi_si(float(bio_cfg.viscosity_fi_crit) * gain, bio_cfg, device)
    mat_tgt = _log1p_nd_for_mat_si(float(bio_cfg.viscosity_mat_crit) * gain, bio_cfg)
    g = gate.reshape(-1).clamp(0.0, 1.0)
    fi_rest = sp[:, FI_SLICE_IDX]
    mat_rest = sp[:, MAT_SLICE_IDX]
    sp = sp.clone()
    sp[:, FI_SLICE_IDX] = fi_rest + g * (fi_tgt - fi_rest)
    sp[:, MAT_SLICE_IDX] = mat_rest + g * (mat_tgt - mat_rest)
    return sp


def build_s2_species_log_nd_at_time(
    data,
    time_index: int,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    bundle: T0R4S2Bundle,
    *,
    commits_prev: torch.Tensor | None,
    nucleation_hops: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    gate = _s0_gate_from_species(s0_sp, data, device, bio_cfg, elig)
    feats = build_s2_features(
        data, t, device, bio_cfg, elig=elig, s0_species=s0_sp, s0_gate=gate
    )
    tau = float(macro_tau_at_index(data, t, bio_cfg=bio_cfg))
    onset = float(_s0_onset_factor(tau))
    if bundle.mode == S2_MODE_LOC:
        risk_n = _risk_n_at_time(data, t, device, bio_cfg, elig=elig)
        logit = bundle.model(feats)
        risk_adj = _apply_loc_risk_adjustment(
            risk_n, logit, elig, onset=onset, loc_scale=bundle.loc_scale
        )
        sp = _build_s0_deploy_species(
            data,
            t,
            device,
            bio_cfg,
            elig=elig,
            commits_prev=commits_prev,
            risk_n_override=risk_adj,
        )
    else:
        delta = bundle.model(feats) * onset
        sp = apply_s2_species_delta(s0_sp, delta, elig, delta_scale=bundle.delta_scale)
    return sp, elig


def load_s2_bundle(
    ckpt_path: str | Path | None = None,
    *,
    device: torch.device | None = None,
    quiet: bool = False,
) -> T0R4S2Bundle | None:
    path = Path(ckpt_path) if ckpt_path is not None else s2_ckpt_path()
    if not path.is_file():
        if not quiet:
            print(f"[WARN] s2 checkpoint missing: {path}")
        return None
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(path, map_location=dev, weights_only=False)
    in_dim = int(payload.get("in_dim", feature_dim()))
    hidden = int(payload.get("hidden", 48))
    mode = str(payload.get("mode", s2_mode_from_env()))
    delta_scale = float(payload.get("delta_scale", s2_delta_scale()))
    loc_scale = float(payload.get("loc_scale", s2_loc_scale()))
    if mode == S2_MODE_LOC:
        model = T0R4S2LocMLP(in_dim=in_dim, hidden=hidden).to(dev)
    else:
        model = T0R4S2SpeciesMLP(in_dim=in_dim, hidden=hidden).to(dev)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return T0R4S2Bundle(
        model=model,
        mode=mode,
        delta_scale=delta_scale,
        loc_scale=loc_scale,
        in_dim=in_dim,
        hidden=hidden,
        device=dev,
    )


def save_s2_checkpoint(
    path: str | Path,
    model: T0R4S2LocMLP | T0R4S2SpeciesMLP,
    *,
    mode: str,
    hidden: int,
    delta_scale: float = 1.0,
    loc_scale: float = 1.5,
    meta: dict[str, Any] | None = None,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "mode": str(mode),
        "in_dim": feature_dim(),
        "hidden": int(hidden),
        "delta_scale": float(delta_scale),
        "loc_scale": float(loc_scale),
        "feature_names": list(FEATURE_NAMES),
        "meta": meta or {},
    }
    torch.save(payload, out)
    sidecar = out.with_suffix(".json")
    sidecar.write_text(
        json.dumps({k: v for k, v in payload.items() if k != "model_state"}, indent=2),
        encoding="utf-8",
    )


@torch.no_grad()
def rollout_s2_species_series(
    data,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    bundle: T0R4S2Bundle,
    *,
    nucleation_hops: int = 1,
) -> torch.Tensor:
    """Coupled s2 species rollout; returns ``(T, N, 16)`` y-shaped tensor."""
    out = data.y.clone().to(device=device)
    commits_prev: torch.Tensor | None = None
    with t0_rung2_env():
        for t in range(int(data.y.shape[0])):
            sp, _ = build_s2_species_log_nd_at_time(
                data,
                t,
                device,
                phys_cfg,
                bio_cfg,
                bundle,
                commits_prev=commits_prev,
                nucleation_hops=nucleation_hops,
            )
            out[t, :, 4:16] = sp
            phi_raw, _ = predict_clot_phi_at_time(
                data,
                t,
                phys_cfg,
                bio_cfg,
                device,
                gamma_mode=RUNG2_GAMMA_MODE,
                flow_source="gt",
                pred_species_series=out,
            )
            commits_prev = (phi_raw.reshape(-1) >= 0.5).bool()
    return out
