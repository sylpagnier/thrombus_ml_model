"""Phase 6: global Mat boost beta between frozen species GNN and Carreau gelation.

The GNN supplies spatiotemporal FI/Mat shape; a single scalar beta calibrates
Mat magnitude before the differentiable gelation gate and COMSOL Carreau mu readout.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND
from src.core_physics.clot_phi_simple import (
    carreau_mu_si_from_gamma_nd,
    clot_phi_physics_mu_blood_si,
    clot_phi_physics_mu_ratio_max,
    mat_si_for_gelation_from_log1p,
    resolve_gamma_dot_nd_for_carreau,
)
from src.core_physics.species_gelation_readout import (
    differentiable_mu_eff_from_species12,
    gelation_temperature_scale,
)
from src.core_physics.clot_phi_simple import comsol_carreau_mu_si_from_uv
from src.core_physics.t0_mu_physics import (
    predict_mu_si_at_time,
    resolve_t0_flow_uv_nd,
    resolve_t0_gamma_mode,
)
from src.training.biochem_species_scope import FI_CHANNEL, MAT_CHANNEL
from src.utils.rheology import phi_clot_from_mat_fi
from src.utils.paths import get_project_root

DEFAULT_S35_DIR = "outputs/biochem/species_snapshot_s35"
DEFAULT_S34_GNN_CKPT = "outputs/biochem/species_snapshot_s34/best.pth"
SPECIES_SLICE_START = 4


def viscosity_calibration_dir() -> Path:
    raw = (os.environ.get("SPECIES_VISCOSITY_CALIB_DIR") or DEFAULT_S35_DIR).strip()
    p = Path(raw)
    if not p.is_absolute():
        p = get_project_root() / p
    return p


def viscosity_calibration_enabled() -> bool:
    raw = (os.environ.get("SPECIES_VISCOSITY_CALIB") or "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def mat_log1p_from_si(mat_si: torch.Tensor, bio_cfg: BiochemConfig) -> torch.Tensor:
    minf = max(float(bio_cfg.Minf), 1e-12)
    return torch.log1p(mat_si.clamp(min=0.0) / minf)


def viscosity_beta_bounds() -> tuple[float, float]:
    lo_raw = (os.environ.get("SPECIES_VISCOSITY_BETA_MIN") or "0.1").strip()
    hi_raw = (os.environ.get("SPECIES_VISCOSITY_BETA_MAX") or "2.0").strip()
    try:
        lo = float(lo_raw)
        hi = float(hi_raw)
    except ValueError:
        lo, hi = 0.5, 1.5
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def apply_gelation_beta_scale(
    gel: torch.Tensor,
    beta: torch.Tensor | float,
) -> torch.Tensor:
    """Scale COMSOL gel factor: gel_eff = 1 + beta * (gel - 1)."""
    g = gel.reshape(-1).to(dtype=torch.float32)
    if isinstance(beta, torch.Tensor):
        b = beta.reshape(()).to(device=g.device, dtype=g.dtype)
    else:
        b = torch.tensor(float(beta), device=g.device, dtype=g.dtype)
    return (1.0 + b * (g - 1.0)).clamp(min=1e-8)


class MatViscosityCalibrator(nn.Module):
    """Learnable global viscosity calibration beta in [lo, hi] (default 0.1..2.0)."""

    def __init__(
        self,
        beta_init: float = 1.5,
        *,
        beta_min: float | None = None,
        beta_max: float | None = None,
    ):
        super().__init__()
        env_lo, env_hi = viscosity_beta_bounds()
        lo = float(beta_min if beta_min is not None else env_lo)
        hi = float(beta_max if beta_max is not None else env_hi)
        if hi <= lo:
            hi = lo + 1.0
        self.register_buffer("_beta_lo", torch.tensor(lo, dtype=torch.float32))
        self.register_buffer("_beta_hi", torch.tensor(hi, dtype=torch.float32))
        frac = (float(beta_init) - lo) / max(hi - lo, 1e-6)
        frac = min(max(frac, 1e-4), 1.0 - 1e-4)
        logit = float(torch.logit(torch.tensor(frac)).item())
        self.logit_beta = nn.Parameter(torch.tensor(logit, dtype=torch.float32))

    @property
    def beta(self) -> torch.Tensor:
        lo = self._beta_lo
        hi = self._beta_hi
        return lo + torch.sigmoid(self.logit_beta) * (hi - lo)

    def forward(self, mat_log1p: torch.Tensor, bio_cfg: BiochemConfig) -> torch.Tensor:
        mat_si = mat_si_for_gelation_from_log1p(mat_log1p, bio_cfg)
        boosted = mat_si * self.beta.clamp(min=1e-3)
        return mat_log1p_from_si(boosted, bio_cfg)


def _mat_slice_index() -> int:
    return int(MAT_CHANNEL - SPECIES_SLICE_START)


def apply_mat_beta_to_species_row(
    species_row: torch.Tensor,
    beta: torch.Tensor | float,
    bio_cfg: BiochemConfig,
) -> torch.Tensor:
    """Boost Mat log1p ND on one ``(N, 16)`` species row (differentiable in beta)."""
    out = species_row.clone()
    sp = out[:, SPECIES_SLICE_START : SPECIES_SLICE_START + 12].clone()
    mat_local = _mat_slice_index()
    mat_log = sp[:, mat_local]
    if isinstance(beta, torch.Tensor):
        b = beta.to(device=mat_log.device, dtype=mat_log.dtype)
    else:
        b = torch.tensor(float(beta), device=mat_log.device, dtype=mat_log.dtype)
    sp[:, mat_local] = _boost_mat_log(mat_log, b, bio_cfg)
    out[:, SPECIES_SLICE_START : SPECIES_SLICE_START + 12] = sp
    return out


def _boost_mat_log(
    mat_log1p: torch.Tensor,
    beta: torch.Tensor,
    bio_cfg: BiochemConfig,
) -> torch.Tensor:
    mat_si = mat_si_for_gelation_from_log1p(mat_log1p, bio_cfg)
    boosted = mat_si * beta.clamp(min=1e-3)
    return mat_log1p_from_si(boosted, bio_cfg)


def apply_mat_beta_to_species_series(
    species_series: torch.Tensor,
    beta: torch.Tensor | float,
    bio_cfg: BiochemConfig,
    *,
    time_index: int,
) -> torch.Tensor:
    """Return full series with boosted Mat at ``time_index`` only."""
    out = species_series.clone()
    ti = int(time_index)
    out[ti] = apply_mat_beta_to_species_row(out[ti], beta, bio_cfg)
    return out


def _carreau_baseline_mu_si(
    data,
    time_index: int,
    phys_cfg: PhysicsConfig,
    device: torch.device,
    *,
    gamma_mode: str,
) -> torch.Tensor:
    """Bulk Carreau mu (no gelation) from GT flow at ``time_index``."""
    t = int(time_index)
    u_nd, v_nd = resolve_t0_flow_uv_nd(data, t, device, flow_source="gt")
    gamma_nd = resolve_gamma_dot_nd_for_carreau(
        data, u_nd, v_nd, device=device, mode=gamma_mode, time_index=t
    )
    mu_0 = torch.full(
        (int(data.num_nodes),),
        float(phys_cfg.mu_0),
        device=device,
        dtype=torch.float32,
    )
    mu_inf = torch.full_like(mu_0, clot_phi_physics_mu_blood_si(phys_cfg))
    return carreau_mu_si_from_gamma_nd(
        gamma_nd, mu_0, mu_inf, phys_cfg, data=data
    ).reshape(-1).clamp(min=1e-8)


def differentiable_clot_phi_from_full_y(
    row16: torch.Tensor,
    bio_cfg: BiochemConfig,
) -> torch.Tensor:
    """Soft clot phi from full ``y`` row using FI/Mat channels 8 and 11."""
    mat_si = mat_si_for_gelation_from_log1p(row16[:, MAT_CHANNEL], bio_cfg)
    scales = bio_cfg.get_species_scales(device=row16.device)[:12].to(
        device=row16.device, dtype=row16.dtype
    )
    fi_log = row16[:, FI_CHANNEL].clamp(min=-10.0, max=8.0)
    fi_si = torch.expm1(fi_log) * scales[FI_CHANNEL]
    t_scale = max(float(bio_cfg.soft_step_T_scale), 1e-5)
    temp_scale = gelation_temperature_scale()
    temp_mat = max(float(bio_cfg.viscosity_gnode_temp_mat) * t_scale / temp_scale, 1e-8)
    temp_fi = max(float(bio_cfg.viscosity_gnode_temp_fi) * t_scale / temp_scale, 1e-8)
    return phi_clot_from_mat_fi(
        mat_si,
        fi_si,
        mat_crit=float(bio_cfg.viscosity_mat_crit),
        fi_crit=float(bio_cfg.viscosity_fi_crit),
        temp_mat=temp_mat,
        temp_fi=temp_fi,
        combine="max",
    ).reshape(-1)


def predict_mu_soft_gelation_with_beta(
    data,
    species_series: torch.Tensor,
    beta: torch.Tensor | float,
    time_index: int,
    *,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    gamma_mode: str | None = None,
    anchor: str = "",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable mu via soft gelation (Phase 3 readout) + beta-boosted Mat."""
    t = int(time_index)
    ti = max(0, min(t, int(species_series.shape[0]) - 1))
    boosted_row = apply_mat_beta_to_species_row(species_series[ti], beta, bio_cfg)
    gm = gamma_mode or (resolve_t0_gamma_mode(anchor) if anchor else "max")
    mu_c = _carreau_baseline_mu_si(data, t, phys_cfg, device, gamma_mode=gm)
    phi = differentiable_clot_phi_from_full_y(boosted_row, bio_cfg)
    ratio = max(float(clot_phi_physics_mu_ratio_max(bio_cfg)), 1.0)
    b = beta if isinstance(beta, torch.Tensor) else torch.tensor(float(beta), device=phi.device, dtype=phi.dtype)
    # beta scales gelation coupling to Carreau baseline (grad flows even when phi ~ 1).
    mu_pred = mu_c * (1.0 + b * (ratio - 1.0) * phi).clamp(min=1e-8)
    y = data.y[t].to(device=device, dtype=torch.float32)
    mu_gt = phys_cfg.viscosity_nd_to_si(y[:, STATE_CHANNEL_MU_EFF_ND]).reshape(-1)
    return mu_pred.reshape(-1), mu_gt.reshape(-1)


def predict_mu_at_time_with_beta(
    data,
    species_series: torch.Tensor,
    beta: torch.Tensor | float,
    time_index: int,
    *,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    gamma_mode: str | None = None,
    anchor: str = "",
    soft_gelation: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict mu_si at ``time_index`` with beta-boosted Mat; returns (mu_pred, mu_gt)."""
    if soft_gelation:
        return predict_mu_soft_gelation_with_beta(
            data,
            species_series,
            beta,
            time_index,
            phys_cfg=phys_cfg,
            bio_cfg=bio_cfg,
            device=device,
            gamma_mode=gamma_mode,
            anchor=anchor,
        )
    boosted = apply_mat_beta_to_species_series(
        species_series, beta, bio_cfg, time_index=time_index
    )
    gm = gamma_mode or resolve_t0_gamma_mode(anchor) if anchor else "max"
    step = predict_mu_si_at_time(
        data,
        int(time_index),
        phys_cfg,
        bio_cfg,
        device,
        gamma_mode=gm,
        flow_source="gt",
        pred_species_series=boosted,
    )
    gel_eff = apply_gelation_beta_scale(step.gel_m, beta)
    mu_pred = comsol_carreau_mu_si_from_uv(
        data,
        step.u_nd,
        step.v_nd,
        gel_eff,
        phys_cfg,
        device=device,
        gamma_mode=gm,
        time_index=int(time_index),
    ).reshape(-1).clamp(min=1e-8)
    return mu_pred, step.mu_gt_si.reshape(-1)


def mu_calibration_loss(
    mu_pred: torch.Tensor,
    mu_gt: torch.Tensor,
    *,
    log_space: bool = True,
) -> torch.Tensor:
    p = mu_pred.reshape(-1).clamp(min=1e-8)
    t = mu_gt.reshape(-1).clamp(min=1e-8)
    if log_space:
        return F.mse_loss(torch.log(p), torch.log(t))
    return F.mse_loss(p, t)


@dataclass(frozen=True)
class ViscosityCalibrationBundle:
    beta: float
    gnn_ckpt: str
    time_index: int
    beta_min: float = 0.1
    beta_max: float = 2.0
    phase: str = "s35_viscosity_calibration"


def save_viscosity_calibration(
    path: Path | str,
    calibrator: MatViscosityCalibrator,
    *,
    gnn_ckpt: str,
    time_index: int = 53,
    meta: dict | None = None,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    beta_val = float(calibrator.beta.detach().cpu().item())
    beta_min = float(calibrator._beta_lo.detach().cpu().item())
    beta_max = float(calibrator._beta_hi.detach().cpu().item())
    payload = {
        "beta": beta_val,
        "beta_min": beta_min,
        "beta_max": beta_max,
        "gnn_ckpt": str(gnn_ckpt),
        "time_index": int(time_index),
        "phase": "s35_viscosity_calibration",
        "model_state": {"logit_beta": calibrator.logit_beta.detach().cpu()},
        "meta": dict(meta or {}),
    }
    torch.save(payload, p)
    side = p.with_suffix(".json")
    side.write_text(
        json.dumps(
            {
                "beta": beta_val,
                "beta_min": beta_min,
                "beta_max": beta_max,
                "gnn_ckpt": str(gnn_ckpt),
                "time_index": int(time_index),
                "phase": "s35_viscosity_calibration",
                **dict(meta or {}),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def resolve_deploy_gelation_beta(
    device: torch.device,
    *,
    cal_path: Path | str | None = None,
) -> torch.Tensor | None:
    """Global s35 beta, or per-anchor ``SPECIES_GELATION_BETA_OVERRIDE`` env."""
    if not viscosity_calibration_enabled():
        return None
    override = (os.environ.get("SPECIES_GELATION_BETA_OVERRIDE") or "").strip()
    if override:
        return torch.tensor(float(override), device=device, dtype=torch.float32)
    raw = cal_path or os.environ.get("SPECIES_VISCOSITY_CALIB_PATH") or str(
        viscosity_calibration_dir() / "beta.pth"
    )
    p = Path(raw)
    if not p.is_absolute():
        p = get_project_root() / p
    if not p.is_file():
        return None
    cal, _ = load_viscosity_calibration(p, device=device)
    return cal.beta


def load_viscosity_calibration(
    path: Path | str | None = None,
    *,
    device: torch.device | None = None,
) -> tuple[MatViscosityCalibrator, ViscosityCalibrationBundle]:
    p = Path(path) if path is not None else viscosity_calibration_dir() / "beta.pth"
    if not p.is_absolute():
        p = get_project_root() / p
    if not p.is_file():
        raise FileNotFoundError(f"missing viscosity calibration: {p}")
    dev = device or torch.device("cpu")
    payload = torch.load(p, map_location=dev, weights_only=False)
    beta_init = float(payload.get("beta", 1.5))
    beta_min = float(payload.get("beta_min", 0.1))
    beta_max = float(payload.get("beta_max", 2.0))
    cal = MatViscosityCalibrator(
        beta_init=beta_init, beta_min=beta_min, beta_max=beta_max
    ).to(dev)
    state = dict(payload.get("model_state") or {})
    if "logit_beta" in state:
        cal.logit_beta.data = state["logit_beta"].to(dev, dtype=torch.float32)
    elif "beta" in state:
        lo, hi = viscosity_beta_bounds()
        frac = (float(state["beta"].reshape(-1)[0].item()) - lo) / max(hi - lo, 1e-6)
        frac = min(max(frac, 1e-4), 1.0 - 1e-4)
        cal.logit_beta.data = torch.tensor(
            float(torch.logit(torch.tensor(frac)).item()), device=dev, dtype=torch.float32
        )
    bundle = ViscosityCalibrationBundle(
        beta=float(cal.beta.detach().cpu().item()),
        gnn_ckpt=str(payload.get("gnn_ckpt") or DEFAULT_S34_GNN_CKPT),
        time_index=int(payload.get("time_index", 53)),
        beta_min=beta_min,
        beta_max=beta_max,
        phase=str(payload.get("phase", "s35_viscosity_calibration")),
    )
    return cal, bundle
