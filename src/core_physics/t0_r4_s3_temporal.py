"""Rung 4 step s3: GRU temporal gate residual inside E(t).

Deploy (default actuator ``gate``)::

    h_0 = 0
    for t in macro_times:
        logit_t = head(GRU(feats_t, h_{t-1}))   # feats include commit_prev, phi_prev
        gate_t = clamp(gate_s0 + loc_scale * tanh(logit_t * onset), 0, 1) in E(t)
        sp_t = FI/Mat ramp from gate_t
        commits_t = physics(sp_t, GT flow)

Legacy ``risk`` actuator reweights risk before s0 top-8% (rank barrier; not recommended).

Checkpoint: ``outputs/biochem/t0_r4_s3_temporal/best.pth`` (``T0_R4_S3_CKPT``).
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
from src.core_physics.t0_mu_physics import predict_clot_phi_at_time
from src.core_physics.t0_r4_s2_species import (
    FEATURE_NAMES,
    T0R4S2LocMLP,
    _apply_loc_gate_residual,
    _apply_loc_risk_adjustment,
    _risk_n_at_time,
    _s0_gate_from_species,
    build_s2_features,
    build_s3_features,
    feature_dim,
    s3_feature_dim,
    species_from_gate,
)
from src.core_physics.t0_rung4_ladder import (
    _build_s0_deploy_species,
    _s0_onset_factor,
    rung4_use_dgamma_wall_seed,
)
from src.core_physics.t0_rung_config import RUNG2_GAMMA_MODE, t0_rung2_env
from src.utils.paths import get_project_root

DEFAULT_S3_CKPT = "outputs/biochem/t0_r4_s3_temporal/best.pth"
S3_EXTRA_FEATURE_NAMES = ("commit_prev", "phi_prev")
S3_ACTUATOR_GATE = "gate"
S3_ACTUATOR_RISK = "risk"


def s3_actuator() -> str:
    raw = (os.environ.get("T0_R4_S3_ACTUATOR") or S3_ACTUATOR_GATE).strip().lower()
    return raw if raw in (S3_ACTUATOR_GATE, S3_ACTUATOR_RISK) else S3_ACTUATOR_GATE


def s3_ckpt_path() -> Path:
    raw = (os.environ.get("T0_R4_S3_CKPT") or DEFAULT_S3_CKPT).strip()
    p = Path(raw)
    if not p.is_absolute():
        p = get_project_root() / p
    return p


def s3_loc_scale() -> float:
    raw = (os.environ.get("T0_R4_S3_LOC_SCALE") or os.environ.get("T0_R4_S2_LOC_SCALE") or "1.5").strip()
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 1.5


def s3_res_scale() -> float:
    raw = (os.environ.get("T0_R4_S3_RES_SCALE") or "1.0").strip()
    try:
        return float(raw)
    except ValueError:
        return 1.0


class T0R4S3TemporalModel(nn.Module):
    """Per-node GRU on s2 features + optional zero-init residual s2_loc path."""

    def __init__(
        self,
        in_dim: int,
        *,
        gru_hidden: int = 32,
        res_hidden: int = 48,
        use_residual: bool = True,
    ):
        super().__init__()
        gh = max(int(gru_hidden), 8)
        self.gru_hidden = gh
        self.in_dim = int(in_dim)
        self.use_residual = bool(use_residual)
        self.gru = nn.GRUCell(self.in_dim, gh)
        self.head = nn.Linear(gh, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.res_mlp = T0R4S2LocMLP(in_dim=self.in_dim, hidden=res_hidden) if self.use_residual else None

    def init_hidden(self, n_nodes: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(int(n_nodes), self.gru_hidden, device=device, dtype=dtype)

    def forward_step(
        self,
        feats: torch.Tensor,
        h_prev: torch.Tensor,
        *,
        res_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_new = self.gru(feats, h_prev)
        logit = self.head(h_new).reshape(-1)
        if self.res_mlp is not None:
            logit = logit + float(res_scale) * self.res_mlp(feats).reshape(-1)
        return logit, h_new


@dataclass(frozen=True)
class T0R4S3Bundle:
    model: T0R4S3TemporalModel
    loc_scale: float
    res_scale: float
    in_dim: int
    gru_hidden: int
    res_hidden: int
    device: torch.device
    actuator: str = S3_ACTUATOR_GATE


def _species_from_logit(
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
    tau = float(macro_tau_at_index(data, int(time_index), bio_cfg=bio_cfg))
    onset = float(_s0_onset_factor(tau))
    t = int(time_index)
    if s0_sp is None:
        s0_sp = _build_s0_deploy_species(
            data, t, device, bio_cfg, elig=elig, commits_prev=commits_prev
        )
    if s0_gate is None:
        s0_gate = _s0_gate_from_species(s0_sp, data, device, bio_cfg, elig)

    if s3_actuator() == S3_ACTUATOR_GATE:
        gate = _apply_loc_gate_residual(
            s0_gate, logit, elig, onset=onset, loc_scale=loc_scale
        )
        return species_from_gate(data, device, bio_cfg, gate)

    risk_n = _risk_n_at_time(data, t, device, bio_cfg, elig=elig)
    risk_adj = _apply_loc_risk_adjustment(
        risk_n, logit, elig, onset=onset, loc_scale=loc_scale
    )
    return _build_s0_deploy_species(
        data,
        t,
        device,
        bio_cfg,
        elig=elig,
        commits_prev=commits_prev,
        risk_n_override=risk_adj,
    )


def build_s3_species_log_nd_at_time(
    data,
    time_index: int,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    bundle: T0R4S3Bundle,
    *,
    commits_prev: torch.Tensor | None,
    h_prev: torch.Tensor,
    nucleation_hops: int = 1,
    phi_prev: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single step; caller must replay from t=0 or pass ``h_prev`` from coupled rollout."""
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
    if bundle.in_dim >= s3_feature_dim():
        feats = build_s3_features(
            data, t, device, bio_cfg, elig=elig, s0_species=s0_sp, s0_gate=gate,
            commits_prev=commits_prev, phi_prev=phi_prev,
        )
    else:
        feats = build_s2_features(
            data, t, device, bio_cfg, elig=elig, s0_species=s0_sp, s0_gate=gate
        )
    logit, h_new = bundle.model.forward_step(
        feats, h_prev, res_scale=bundle.res_scale
    )
    sp = _species_from_logit(
        data, t, device, bio_cfg,
        elig=elig, commits_prev=commits_prev, logit=logit, loc_scale=bundle.loc_scale,
        s0_sp=s0_sp, s0_gate=gate,
    )
    return sp, elig, h_new


def load_s3_bundle(
    ckpt_path: str | Path | None = None,
    *,
    device: torch.device | None = None,
    quiet: bool = False,
) -> T0R4S3Bundle | None:
    path = Path(ckpt_path) if ckpt_path is not None else s3_ckpt_path()
    if not path.is_file():
        if not quiet:
            print(f"[WARN] s3 checkpoint missing: {path}")
        return None
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(path, map_location=dev, weights_only=False)
    in_dim = int(payload.get("in_dim", s3_feature_dim()))
    gru_hidden = int(payload.get("gru_hidden", 32))
    res_hidden = int(payload.get("res_hidden", 48))
    use_residual = bool(payload.get("use_residual", True))
    loc_scale = float(payload.get("loc_scale", s3_loc_scale()))
    res_scale = float(payload.get("res_scale", s3_res_scale()))
    actuator = str(payload.get("actuator", S3_ACTUATOR_RISK if int(payload.get("in_dim", s3_feature_dim())) <= feature_dim() else S3_ACTUATOR_GATE))
    os.environ["T0_R4_S3_ACTUATOR"] = actuator
    model = T0R4S3TemporalModel(
        in_dim=in_dim,
        gru_hidden=gru_hidden,
        res_hidden=res_hidden,
        use_residual=use_residual,
    ).to(dev)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return T0R4S3Bundle(
        model=model,
        loc_scale=loc_scale,
        res_scale=res_scale,
        in_dim=in_dim,
        gru_hidden=gru_hidden,
        res_hidden=res_hidden,
        device=dev,
        actuator=actuator,
    )


def save_s3_checkpoint(
    path: str | Path,
    model: T0R4S3TemporalModel,
    *,
    loc_scale: float,
    res_scale: float,
    gru_hidden: int,
    res_hidden: int,
    use_residual: bool,
    meta: dict[str, Any] | None = None,
    actuator: str | None = None,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    act = actuator or s3_actuator()
    feat_names = list(FEATURE_NAMES) + list(S3_EXTRA_FEATURE_NAMES)
    payload = {
        "model_state": model.state_dict(),
        "in_dim": s3_feature_dim(),
        "gru_hidden": int(gru_hidden),
        "res_hidden": int(res_hidden),
        "use_residual": bool(use_residual),
        "loc_scale": float(loc_scale),
        "res_scale": float(res_scale),
        "actuator": act,
        "feature_names": feat_names,
        "meta": meta or {},
    }
    torch.save(payload, out)
    out.with_suffix(".json").write_text(
        json.dumps({k: v for k, v in payload.items() if k != "model_state"}, indent=2),
        encoding="utf-8",
    )


def load_s3_residual_from_s2(
    model: T0R4S3TemporalModel,
    s2_ckpt: str | Path,
    *,
    device: torch.device,
) -> bool:
    """Init ``res_mlp`` from s2_loc checkpoint (optional warm start)."""
    if model.res_mlp is None:
        return False
    path = Path(s2_ckpt)
    if not path.is_file():
        return False
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if str(payload.get("mode", "loc")) != "loc":
        return False
    state = payload["model_state"]
    model.res_mlp.load_state_dict(state)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return True


@torch.no_grad()
def rollout_s3_species_series(
    data,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    bundle: T0R4S3Bundle,
    *,
    nucleation_hops: int = 1,
) -> torch.Tensor:
    """Coupled s3 species rollout; returns ``(T, N, 16)`` y-shaped tensor."""
    out = data.y.clone().to(device=device)
    n = int(data.num_nodes)
    commits_prev: torch.Tensor | None = None
    h = bundle.model.init_hidden(n, device, out.dtype)
    phi_prev: torch.Tensor | None = None
    with t0_rung2_env():
        for t in range(int(data.y.shape[0])):
            sp, _, h = build_s3_species_log_nd_at_time(
                data,
                t,
                device,
                phys_cfg,
                bio_cfg,
                bundle,
                commits_prev=commits_prev,
                h_prev=h,
                nucleation_hops=nucleation_hops,
                phi_prev=phi_prev,
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
