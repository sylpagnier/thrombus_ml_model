"""Rung 4 step s4: 2-layer band GNN gate residual in E(t).

Deploy::

    logit = GNN(feats, edge_index)              # feats incl. commit_prev, phi_prev
    gate = clamp(gate_s0 + loc_scale * tanh(logit * onset(tau)), 0, 1) in E(t)
    sp = FI/Mat ramp from gate
    commits = gelation_physics(sp, GT flow)

Checkpoint: ``outputs/biochem/t0_r4_s4_band_ml/best.pth`` (``T0_R4_S4_CKPT``).
"""

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
from src.core_physics.clot_continuous_time import macro_tau_at_index
from src.core_physics.clot_nucleation_mask import resolve_nucleation_eligibility
from src.core_physics.t0_mu_physics import predict_clot_phi_at_time
from src.core_physics.t0_r4_s2_species import (
    FEATURE_NAMES,
    _apply_loc_gate_residual,
    _s0_gate_from_species,
    build_s3_features,
    s3_feature_dim,
    species_from_gate,
)
from src.core_physics.t0_r4_s3_temporal import S3_EXTRA_FEATURE_NAMES
from src.core_physics.t0_rung4_ladder import (
    _build_s0_deploy_species,
    _s0_onset_factor,
    rung4_use_dgamma_wall_seed,
)
from src.core_physics.t0_rung_config import RUNG2_GAMMA_MODE, t0_rung2_env
from src.utils.paths import get_project_root

DEFAULT_S4_CKPT = "outputs/biochem/t0_r4_s4_band_ml/best.pth"


def s4_ckpt_path() -> Path:
    raw = (os.environ.get("T0_R4_S4_CKPT") or DEFAULT_S4_CKPT).strip()
    p = Path(raw)
    if not p.is_absolute():
        p = get_project_root() / p
    return p


def s4_loc_scale() -> float:
    raw = (os.environ.get("T0_R4_S4_LOC_SCALE") or "0.75").strip()
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 0.75


class _BandConv(MessagePassing):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__(aggr="add")
        self.lin_nei = nn.Linear(in_dim, out_dim)
        self.lin_self = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.propagate(edge_index, x=x)

    def message(self, x_j: torch.Tensor, x_i: torch.Tensor) -> torch.Tensor:
        return F.silu(self.lin_nei(x_j) + self.lin_self(x_i))


class T0R4S4BandGNN(nn.Module):
    """Two-hop mesh GNN -> per-node gate logit (zero-init head)."""

    def __init__(self, in_dim: int, *, hidden: int = 32):
        super().__init__()
        h = max(int(hidden), 16)
        self.in_dim = int(in_dim)
        self.hidden = h
        self.conv1 = _BandConv(self.in_dim, h)
        self.conv2 = _BandConv(h, h)
        self.head = nn.Linear(h, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.35)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.conv1(x, edge_index))
        h = F.silu(self.conv2(h, edge_index))
        return self.head(h).reshape(-1)


@dataclass(frozen=True)
class T0R4S4Bundle:
    model: T0R4S4BandGNN
    loc_scale: float
    in_dim: int
    hidden: int
    device: torch.device


def s4_feature_dim() -> int:
    return s3_feature_dim()


def species_from_band_logit(
    data,
    time_index: int,
    device: torch.device,
    bio_cfg: BiochemConfig,
    *,
    elig: torch.Tensor,
    commits_prev: torch.Tensor | None,
    logit: torch.Tensor,
    loc_scale: float,
    s0_sp: torch.Tensor | None = None,
    s0_gate: torch.Tensor | None = None,
) -> torch.Tensor:
    t = int(time_index)
    if s0_sp is None:
        s0_sp = _build_s0_deploy_species(
            data, t, device, bio_cfg, elig=elig, commits_prev=commits_prev
        )
    if s0_gate is None:
        s0_gate = _s0_gate_from_species(s0_sp, data, device, bio_cfg, elig)
    onset = float(_s0_onset_factor(float(macro_tau_at_index(data, t, bio_cfg=bio_cfg))))
    gate = _apply_loc_gate_residual(
        s0_gate, logit, elig, onset=onset, loc_scale=loc_scale
    )
    return species_from_gate(data, device, bio_cfg, gate)


def build_s4_species_log_nd_at_time(
    data,
    time_index: int,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    bundle: T0R4S4Bundle,
    *,
    commits_prev: torch.Tensor | None,
    phi_prev: torch.Tensor | None = None,
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
    feats = build_s3_features(
        data, t, device, bio_cfg, elig=elig, s0_species=s0_sp, s0_gate=gate,
        commits_prev=commits_prev, phi_prev=phi_prev,
    )
    edge_index = data.edge_index.to(device=device)
    logit = bundle.model(feats, edge_index)
    sp = species_from_band_logit(
        data, t, device, bio_cfg,
        elig=elig, commits_prev=commits_prev, logit=logit,
        loc_scale=bundle.loc_scale, s0_sp=s0_sp, s0_gate=gate,
    )
    return sp, elig


def load_s4_bundle(
    ckpt_path: str | Path | None = None,
    *,
    device: torch.device | None = None,
    quiet: bool = False,
) -> T0R4S4Bundle | None:
    path = Path(ckpt_path) if ckpt_path is not None else s4_ckpt_path()
    if not path.is_file():
        if not quiet:
            print(f"[WARN] s4 checkpoint missing: {path}")
        return None
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(path, map_location=dev, weights_only=False)
    in_dim = int(payload.get("in_dim", s4_feature_dim()))
    hidden = int(payload.get("hidden", 32))
    loc_scale = float(payload.get("loc_scale", s4_loc_scale()))
    model = T0R4S4BandGNN(in_dim=in_dim, hidden=hidden).to(dev)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return T0R4S4Bundle(
        model=model, loc_scale=loc_scale, in_dim=in_dim, hidden=hidden, device=dev
    )


def save_s4_checkpoint(
    path: str | Path,
    model: T0R4S4BandGNN,
    *,
    loc_scale: float,
    hidden: int,
    meta: dict[str, Any] | None = None,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    feat_names = list(FEATURE_NAMES) + list(S3_EXTRA_FEATURE_NAMES)
    payload = {
        "model_state": model.state_dict(),
        "in_dim": s4_feature_dim(),
        "hidden": int(hidden),
        "loc_scale": float(loc_scale),
        "feature_names": feat_names,
        "meta": meta or {},
    }
    torch.save(payload, out)
    out.with_suffix(".json").write_text(
        json.dumps({k: v for k, v in payload.items() if k != "model_state"}, indent=2),
        encoding="utf-8",
    )


@torch.no_grad()
def rollout_s4_species_series(
    data,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    bundle: T0R4S4Bundle,
    *,
    nucleation_hops: int = 1,
) -> torch.Tensor:
    out = data.y.clone().to(device=device)
    commits_prev: torch.Tensor | None = None
    phi_prev: torch.Tensor | None = None
    with t0_rung2_env():
        for t in range(int(data.y.shape[0])):
            sp, _ = build_s4_species_log_nd_at_time(
                data,
                t,
                device,
                phys_cfg,
                bio_cfg,
                bundle,
                commits_prev=commits_prev,
                phi_prev=phi_prev,
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
            phi_prev = phi_raw.reshape(-1).clamp(0.0, 1.0)
            commits_prev = (phi_raw.reshape(-1) >= 0.5).bool()
    return out
