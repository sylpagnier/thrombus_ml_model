"""T0 viscosity oracle: GT flow + GT species -> mu_pred (no GT mu as input).

Question: given perfect kinematics and biochemistry, does COMSOL-faithful
Carreau + gelation reproduce exported ``spf.mu`` including clot growth?

Uses ``BiochemConfig.mu_ratio_max`` (default 80) and optional COMSOL ``spf.sr``
sidecar for shear rate. No deploy caps (``CLOT_PHI_MU_CAP_SI``) or clot-phi
``ratio_max=4`` override.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import torch

from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND
from src.core_physics.clot_growth_masks import gt_growth_commit_mask_at_time
from src.core_physics.clot_nucleation_mask import (
    project_phi_with_nucleation,
    resolve_nucleation_eligibility,
)
from src.core_physics.clot_phi_simple import (
    _resolve_gelation_legs,
    carreau_mu_si_from_gamma_nd,
    clot_phi_thresh_si,
    comsol_carreau_mu_si_from_uv,
    gt_mu_anchor_cap_si,
    mu1_comsol_from_mat_si,
    mu2_comsol_from_fi_si,
    resolve_gamma_dot_nd_for_carreau,
    mat_si_for_gelation_from_log1p,
    species_log1p_nd_to_si,
)
from src.utils.paths import get_project_root


def comsol_sr_sidecar_path(anchor: str, *, root: Path | None = None) -> Path | None:
    base = root or get_project_root()
    for rel in (
        f"data/processed/cfd_results_biochem_diag/{anchor}_sr.pt",
        f"outputs/biochem/diagnostics/{anchor}_sr.pt",
    ):
        p = base / rel
        if p.is_file():
            return p
    return None


def comsol_sr_sidecar_available(anchor: str, *, root: Path | None = None) -> bool:
    return comsol_sr_sidecar_path(anchor, root=root) is not None


def resolve_t0_gamma_mode(anchor: str, *, root: Path | None = None) -> str:
    """``comsol_sr`` when sidecar exists; else graph/poiseuille/kinematic proxy."""
    if comsol_sr_sidecar_available(anchor, root=root):
        return "comsol_sr"
    return "max"


_T0_ENV_KEYS = (
    "CLOT_PHI_PHYSICS_MU_BASE",
    "CLOT_PHI_PHYSICS_GAMMA_MODE",
    "CLOT_PHI_PHYSICS_COMSOL_SR_ANCHOR",
    "CLOT_PHI_PHYSICS_MU_RATIO_MAX",
    "CLOT_PHI_MU_CAP_SI",
    "CLOT_PHI_PHYSICS_HARD_STEP",
    "CLOT_PHI_PHYSICS_GELATION_GATE",
    "CLOT_PHI_PHYSICS_GELATION_ONSET_FRAC",
)


@contextlib.contextmanager
def t0_physics_env(
    anchor: str,
    *,
    gamma_mode: str | None = None,
    hard_step: bool = True,
    root: Path | None = None,
) -> Iterator[dict[str, str]]:
    """Set COMSOL-faithful physics env; restore prior values on exit."""
    saved = {k: os.environ.get(k) for k in _T0_ENV_KEYS}
    mode = (gamma_mode or resolve_t0_gamma_mode(anchor, root=root)).strip().lower()
    os.environ["CLOT_PHI_PHYSICS_MU_BASE"] = "comsol_carreau"
    os.environ["CLOT_PHI_PHYSICS_GAMMA_MODE"] = mode
    if mode in ("comsol_sr", "spf_sr", "spf.sr"):
        os.environ["CLOT_PHI_PHYSICS_COMSOL_SR_ANCHOR"] = anchor
    else:
        os.environ.pop("CLOT_PHI_PHYSICS_COMSOL_SR_ANCHOR", None)
    os.environ.pop("CLOT_PHI_PHYSICS_MU_RATIO_MAX", None)
    os.environ.pop("CLOT_PHI_MU_CAP_SI", None)
    os.environ["CLOT_PHI_PHYSICS_GELATION_GATE"] = "0"
    os.environ["CLOT_PHI_PHYSICS_GELATION_ONSET_FRAC"] = "0"
    os.environ["CLOT_PHI_PHYSICS_HARD_STEP"] = "1" if hard_step else "0"
    cfg = {
        "mu_base": "comsol_carreau",
        "gamma_mode": mode,
        "hard_step": "1" if hard_step else "0",
        "mu_ratio_max": "bio_cfg",
        "mu_cap": "none",
    }
    try:
        yield cfg
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@dataclass
class T0MuStep:
    time_index: int
    mu_pred_si: torch.Tensor
    mu_gt_si: torch.Tensor
    gel_m: torch.Tensor
    mu1: torch.Tensor
    mu2: torch.Tensor
    gamma_nd: torch.Tensor
    u_nd: torch.Tensor
    v_nd: torch.Tensor
    ratio_max: float
    gamma_mode: str


def clot_phi_binary_from_mu_growth(
    mu_si: torch.Tensor,
    mu_anchor_si: torch.Tensor,
    phys_cfg: PhysicsConfig,
) -> torch.Tensor:
    """Binary clot field [0,1] from viscosity rise above per-node t=0 anchor."""
    growth = (mu_si.reshape(-1) - mu_anchor_si.reshape(-1)).clamp(min=0.0)
    thr = clot_phi_thresh_si(phys_cfg)
    return (growth >= thr).to(dtype=torch.float32)


def gt_clot_phi_at_time(
    data,
    time_index: int,
    phys_cfg: PhysicsConfig,
    device: torch.device,
) -> torch.Tensor:
    """GT clot labels from ``spf.mu`` channel (evaluation only, not a model input)."""
    y = data.y[int(time_index)].to(device)
    mu_gt = phys_cfg.viscosity_nd_to_si(y[:, STATE_CHANNEL_MU_EFF_ND])
    anchor = gt_mu_anchor_cap_si(data, phys_cfg, device)
    return clot_phi_binary_from_mu_growth(mu_gt, anchor, phys_cfg)


def resolve_t0_flow_uv_nd(
    data,
    time_index: int,
    device: torch.device,
    *,
    flow_source: str = "gt",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve ND ``[u,v]`` for T0 physics (GT COMSOL or steady GINO-DEQ)."""
    src = (flow_source or "gt").strip().lower()
    y = data.y[int(time_index)].to(device=device, dtype=torch.float32)
    if src in ("gt", "comsol", "truth"):
        return y[:, 0].reshape(-1), y[:, 1].reshape(-1)
    if src in ("kinematics", "pred", "kine", "deq", "gino"):
        from src.core_physics.clot_temporal_growth_rules import _resolve_uv_for_temporal_risk

        u, v = _resolve_uv_for_temporal_risk(data, int(time_index), device)
        return u.reshape(-1), v.reshape(-1)
    raise ValueError(f"unknown T0 flow_source={flow_source!r} (use gt|kinematics)")


def resolve_t0_species_log_nd(
    data,
    time_index: int,
    device: torch.device,
    *,
    pred_species_series: torch.Tensor | None = None,
) -> torch.Tensor:
    """Species log1p ND for T0 gelation (GT graph or teacher rollout series)."""
    t = int(time_index)
    if pred_species_series is None:
        return data.y[t, :, 4:16].to(device=device, dtype=torch.float32)
    ti = max(0, min(t, int(pred_species_series.shape[0]) - 1))
    return pred_species_series[ti, :, 4:16].to(device=device, dtype=torch.float32)


def predict_clot_phi_at_time(
    data,
    time_index: int,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    *,
    gamma_mode: str,
    flow_source: str = "gt",
    pred_species_series: torch.Tensor | None = None,
    ratio_max: float | None = None,
    gelation_beta: float | torch.Tensor | None = None,
) -> tuple[torch.Tensor, T0MuStep]:
    """Clot phi [0,1] from physics mu (flow + species only)."""
    step = predict_mu_si_at_time(
        data,
        time_index,
        phys_cfg,
        bio_cfg,
        device,
        gamma_mode=gamma_mode,
        flow_source=flow_source,
        pred_species_series=pred_species_series,
        ratio_max=ratio_max,
        gelation_beta=gelation_beta,
    )
    anchor = gt_mu_anchor_cap_si(data, phys_cfg, device)
    phi = clot_phi_binary_from_mu_growth(step.mu_pred_si, anchor, phys_cfg)
    return phi, step


def predict_mu_si_at_time(
    data,
    time_index: int,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    *,
    gamma_mode: str,
    flow_source: str = "gt",
    pred_species_series: torch.Tensor | None = None,
    ratio_max: float | None = None,
    gelation_beta: float | torch.Tensor | None = None,
) -> T0MuStep:
    """Predict dynamic viscosity from flow + species (no GT mu channel input)."""
    t = int(time_index)
    y = data.y[t].to(device)
    u_nd, v_nd = resolve_t0_flow_uv_nd(data, t, device, flow_source=flow_source)
    species_log = resolve_t0_species_log_nd(
        data, t, device, pred_species_series=pred_species_series
    )
    rm = float(ratio_max if ratio_max is not None else bio_cfg.mu_ratio_max)

    prev_rm = os.environ.get("CLOT_PHI_PHYSICS_MU_RATIO_MAX")
    os.environ["CLOT_PHI_PHYSICS_MU_RATIO_MAX"] = f"{rm:g}"
    try:
        mu1, mu2, gel = _resolve_gelation_legs(
            species_log,
            bio_cfg,
            device=device,
            data=data,
            u_nd=u_nd,
            v_nd=v_nd,
            time_index=t,
            base_mode="comsol_carreau",
        )
        if gelation_beta is not None:
            from src.core_physics.species_viscosity_calibration import apply_gelation_beta_scale

            gel = apply_gelation_beta_scale(gel, gelation_beta)
        mu_pred = comsol_carreau_mu_si_from_uv(
            data,
            u_nd,
            v_nd,
            gel,
            phys_cfg,
            device=device,
            gamma_mode=gamma_mode,
            time_index=t,
        ).reshape(-1).clamp(min=1e-8)
    finally:
        if prev_rm is None:
            os.environ.pop("CLOT_PHI_PHYSICS_MU_RATIO_MAX", None)
        else:
            os.environ["CLOT_PHI_PHYSICS_MU_RATIO_MAX"] = prev_rm

    gamma_nd = resolve_gamma_dot_nd_for_carreau(
        data, u_nd, v_nd, device=device, mode=gamma_mode, time_index=t
    )
    mu_gt = phys_cfg.viscosity_nd_to_si(y[:, STATE_CHANNEL_MU_EFF_ND]).reshape(-1)
    return T0MuStep(
        time_index=t,
        mu_pred_si=mu_pred,
        mu_gt_si=mu_gt,
        gel_m=gel.reshape(-1),
        mu1=mu1.reshape(-1),
        mu2=mu2.reshape(-1),
        gamma_nd=gamma_nd.reshape(-1),
        u_nd=u_nd,
        v_nd=v_nd,
        ratio_max=rm,
        gamma_mode=gamma_mode,
    )


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    ac = a - a.mean()
    bc = b - b.mean()
    den = ac.pow(2).sum().sqrt() * bc.pow(2).sum().sqrt()
    if float(den.item()) < 1e-12:
        return float("nan")
    return float((ac * bc).sum().item() / den.item())


def _rel_l2(pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    p = pred.reshape(-1).float()
    t = true.reshape(-1).float()
    if mask is not None:
        m = mask.reshape(-1).bool()
        if not bool(m.any().item()):
            return float("nan")
        p, t = p[m], t[m]
    num = (p - t).pow(2).sum().sqrt()
    den = t.pow(2).sum().sqrt().clamp(min=1e-12)
    return float((num / den).item())


def _mu_log_mae(pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    p = pred.reshape(-1).float().clamp(min=1e-8)
    t = true.reshape(-1).float().clamp(min=1e-8)
    if mask is not None:
        m = mask.reshape(-1).bool()
        if not bool(m.any().item()):
            return float("nan")
        p, t = p[m], t[m]
    return float((torch.log(p) - torch.log(t)).abs().mean().item())


def _median_ratio(gt: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any().item()):
        return float("nan")
    return float((gt[mask] / pred[mask].clamp(min=1e-8)).median().item())


def _region_masks(
    data,
    time_index: int,
    phys_cfg: PhysicsConfig,
    device: torch.device,
    mu_gt: torch.Tensor,
) -> dict[str, torch.Tensor]:
    n = int(data.num_nodes)
    wall = (
        data.mask_wall.view(-1).bool().to(device)
        if hasattr(data, "mask_wall")
        else torch.zeros(n, dtype=torch.bool, device=device)
    )
    interior = ~wall
    mu_t0 = phys_cfg.viscosity_nd_to_si(data.y[0, :, STATE_CHANNEL_MU_EFF_ND]).to(device).reshape(-1)
    bulk = mu_t0 < 0.012
    growth = gt_growth_commit_mask_at_time(data, time_index, phys_cfg, device)
    high_mu = mu_gt.reshape(-1) > 0.015
    return {
        "all": torch.ones(n, dtype=torch.bool, device=device),
        "bulk": bulk,
        "interior": interior,
        "wall": wall,
        "growth": growth,
        "high_mu": high_mu,
    }


@dataclass
class T0MuTimeMetrics:
    time_index: int
    pearson_all: float
    pearson_bulk: float
    pearson_growth: float
    pearson_high_mu: float
    rel_l2_all: float
    rel_l2_growth: float
    mu_log_mae_all: float
    mu_log_mae_growth: float
    ratio_median_bulk: float
    ratio_median_growth: float
    ratio_median_high_mu: float
    n_growth: int
    gel_m_median_growth: float = float("nan")


def metrics_for_step(
    step: T0MuStep,
    data,
    phys_cfg: PhysicsConfig,
    device: torch.device,
) -> T0MuTimeMetrics:
    masks = _region_masks(data, step.time_index, phys_cfg, device, step.mu_gt_si)
    pred = step.mu_pred_si.to(device)
    gt = step.mu_gt_si.to(device)
    growth = masks["growth"]
    return T0MuTimeMetrics(
        time_index=step.time_index,
        pearson_all=_pearson(pred, gt),
        pearson_bulk=_pearson(pred[masks["bulk"]], gt[masks["bulk"]]) if masks["bulk"].any() else float("nan"),
        pearson_growth=_pearson(pred[growth], gt[growth]) if int(growth.sum()) > 10 else float("nan"),
        pearson_high_mu=_pearson(pred[masks["high_mu"]], gt[masks["high_mu"]])
        if masks["high_mu"].any()
        else float("nan"),
        rel_l2_all=_rel_l2(pred, gt),
        rel_l2_growth=_rel_l2(pred, gt, growth),
        mu_log_mae_all=_mu_log_mae(pred, gt),
        mu_log_mae_growth=_mu_log_mae(pred, gt, growth),
        ratio_median_bulk=_median_ratio(gt, pred, masks["bulk"]),
        ratio_median_growth=_median_ratio(gt, pred, growth),
        ratio_median_high_mu=_median_ratio(gt, pred, masks["high_mu"]),
        n_growth=int(growth.sum().item()),
        gel_m_median_growth=float(step.gel_m[growth].median().item()) if growth.any() else float("nan"),
    )


def _pass_band(ratio: float, *, lo: float = 0.90, hi: float = 1.10) -> bool:
    return math.isfinite(ratio) and lo <= ratio <= hi


@dataclass
class T0AnchorReport:
    anchor: str
    physics: dict[str, str]
    gamma_sidecar: bool
    ratio_max: float
    times: list[dict] = field(default_factory=list)
    summary: dict[str, float] = field(default_factory=dict)
    pass_gates: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "anchor": self.anchor,
            "physics": self.physics,
            "gamma_sidecar": self.gamma_sidecar,
            "ratio_max": self.ratio_max,
            "times": self.times,
            "summary": self.summary,
            "pass_gates": self.pass_gates,
        }


def eval_anchor_t0_mu(
    graph_path: Path,
    *,
    times: list[int] | None = None,
    gamma_mode: str | None = None,
    hard_step: bool = True,
    ratio_max: float | None = None,
    device: torch.device | None = None,
) -> T0AnchorReport:
    anchor = graph_path.stem
    root = get_project_root()
    dev = device or torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    rm = float(ratio_max if ratio_max is not None else bio.mu_ratio_max)

    data = torch.load(graph_path, map_location=dev, weights_only=False)
    n_steps = int(data.y.shape[0])
    if times is None:
        times = [0, n_steps // 4, n_steps // 2, n_steps - 1]
    times = sorted({max(0, min(int(t), n_steps - 1)) for t in times})

    with t0_physics_env(anchor, gamma_mode=gamma_mode, hard_step=hard_step, root=root) as physics:
        mode = physics["gamma_mode"]
        rows: list[dict] = []
        for t in times:
            step = predict_mu_si_at_time(
                data, t, phys, bio, dev, gamma_mode=mode, ratio_max=rm
            )
            m = metrics_for_step(step, data, phys, dev)
            rows.append(
                {
                    "time": t,
                    "mu_gt_median": float(step.mu_gt_si.median().item()),
                    "mu_pred_median": float(step.mu_pred_si.median().item()),
                    "gel_m_median": float(step.gel_m.median().item()),
                    "gel_m_max": float(step.gel_m.max().item()),
                    **{k: v for k, v in m.__dict__.items() if k != "time_index"},
                }
            )

    t0_row = next((r for r in rows if r["time"] == 0), rows[0] if rows else {})
    bulk_ok = _pass_band(float(t0_row.get("ratio_median_bulk", float("nan"))))
    growth_ratios = [float(r["ratio_median_growth"]) for r in rows if r.get("n_growth", 0) > 0]
    growth_ratio_med = (
        sorted(growth_ratios)[len(growth_ratios) // 2] if growth_ratios else float("nan")
    )
    summary = {
        "mean_pearson_all": sum(r["pearson_all"] for r in rows) / max(len(rows), 1),
        "mean_pearson_growth": sum(
            r["pearson_growth"] for r in rows if math.isfinite(r["pearson_growth"])
        )
        / max(sum(1 for r in rows if math.isfinite(r["pearson_growth"])), 1),
        "mean_mu_log_mae_growth": sum(
            r["mu_log_mae_growth"] for r in rows if math.isfinite(r["mu_log_mae_growth"])
        )
        / max(sum(1 for r in rows if math.isfinite(r["mu_log_mae_growth"])), 1),
        "t0_ratio_median_bulk": float(t0_row.get("ratio_median_bulk", float("nan"))),
        "median_ratio_growth": growth_ratio_med,
    }
    pass_gates = {
        "bulk_mu_t0": bulk_ok,
        "gamma_sidecar": comsol_sr_sidecar_available(anchor, root=root),
        "growth_ratio_in_band": _pass_band(growth_ratio_med, lo=0.75, hi=1.33),
    }
    return T0AnchorReport(
        anchor=anchor,
        physics=physics,
        gamma_sidecar=comsol_sr_sidecar_available(anchor, root=root),
        ratio_max=rm,
        times=rows,
        summary=summary,
        pass_gates=pass_gates,
    )


def write_report(report: T0AnchorReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")


def debug_sidecar_path(anchor: str, *, root: Path | None = None) -> Path | None:
    base = root or get_project_root()
    p = base / f"data/processed/cfd_results_biochem_diag/{anchor}_debug.pt"
    return p if p.is_file() else None


def load_debug_sidecar(anchor: str, *, root: Path | None = None) -> dict | None:
    path = debug_sidecar_path(anchor, root=root)
    if path is None:
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload if isinstance(payload, dict) else None


def _gamma_nd_from_si(data, gamma_si: torch.Tensor, device: torch.device) -> torch.Tensor:
    u_ref = float(data.u_ref.view(-1)[0].item())
    d_bar = float(data.d_bar.view(-1)[0].item())
    return (gamma_si.reshape(-1).to(device=device) * (d_bar / max(u_ref, 1e-8))).clamp(min=1e-12)


def predict_mu_si_from_comsol_export_legs(
    data,
    time_index: int,
    phys_cfg: PhysicsConfig,
    debug: dict,
    device: torch.device,
) -> torch.Tensor:
    """Carreau mu using COMSOL-exported mu1, mu2, spf.sr (oracle gelation legs)."""
    t = int(time_index)
    mu1 = debug["mu1"][t].reshape(-1).to(device=device, dtype=torch.float32)
    mu2 = debug["mu2"][t].reshape(-1).to(device=device, dtype=torch.float32)
    gel = (mu1 + mu2).clamp(min=1e-8)
    gamma_si = debug["gamma_si"][t].reshape(-1).to(device=device, dtype=torch.float32)
    gamma_nd = _gamma_nd_from_si(data, gamma_si, device)
    from src.core_physics.clot_phi_simple import clot_phi_physics_mu_blood_si

    mu_0_si = float(phys_cfg.mu_0) * gel
    mu_inf_si = clot_phi_physics_mu_blood_si(phys_cfg) * gel
    return carreau_mu_si_from_gamma_nd(
        gamma_nd, mu_0_si, mu_inf_si, phys_cfg, data=data
    ).clamp(min=1e-8)


def predict_mu_si_from_graph_species_legs(
    data,
    time_index: int,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    debug: dict,
    device: torch.device,
    *,
    mat_source: str = "graph",
    fi_source: str = "graph",
    ratio_max: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """mu from Python mu1/mu2 with selectable Mat/FI source (graph vs COMSOL export)."""
    t = int(time_index)
    rm = float(ratio_max if ratio_max is not None else bio_cfg.mu_ratio_max)
    if mat_source == "comsol":
        mat_si = debug["mat_si"][t].reshape(-1).to(device=device, dtype=torch.float32)
    else:
        mat_si = mat_si_for_gelation_from_log1p(
            data.y[t, :, 15].to(device), bio_cfg
        )
    if fi_source == "comsol":
        fi_si = debug["fi_si"][t].reshape(-1).to(device=device, dtype=torch.float32)
    else:
        sp = species_log1p_nd_to_si(data.y[t, :, 4:16].to(device), bio_cfg)
        fi_si = sp[:, 8]
    mu1 = mu1_comsol_from_mat_si(mat_si, bio_cfg, rm)
    mu2 = mu2_comsol_from_fi_si(fi_si, bio_cfg, rm)
    gel = mu1 + mu2
    y = data.y[t].to(device)
    gamma_nd = _gamma_nd_from_si(data, debug["gamma_si"][t], device)
    mu_0_si = float(phys_cfg.mu_0) * gel
    from src.core_physics.clot_phi_simple import clot_phi_physics_mu_blood_si

    mu_inf_si = clot_phi_physics_mu_blood_si(phys_cfg) * gel
    mu = carreau_mu_si_from_gamma_nd(gamma_nd, mu_0_si, mu_inf_si, phys_cfg, data=data)
    return mu.clamp(min=1e-8), mu1, mu2


def rollout_t0_clot_phi(
    data,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    *,
    gamma_mode: str,
    flow_source: str = "gt",
    pred_species_series: torch.Tensor | None = None,
    nucleation: bool = False,
    nucleation_hops: int = 1,
    use_dgamma_wall_seed: bool = False,
    ratio_max: float | None = None,
    gelation_beta: float | torch.Tensor | None = None,
) -> dict[int, dict[str, torch.Tensor]]:
    """Time series of raw and optionally nucleation-projected clot phi from physics mu."""
    n_steps = int(data.y.shape[0])
    anchor = gt_mu_anchor_cap_si(data, phys_cfg, device)
    phi_prev: torch.Tensor | None = None
    commits_prev: torch.Tensor | None = None
    out: dict[int, dict[str, torch.Tensor]] = {}
    for t in range(n_steps):
        phi_raw, step = predict_clot_phi_at_time(
            data,
            t,
            phys_cfg,
            bio_cfg,
            device,
            gamma_mode=gamma_mode,
            flow_source=flow_source,
            pred_species_series=pred_species_series,
            ratio_max=ratio_max,
            gelation_beta=gelation_beta,
        )
        phi = phi_raw
        if nucleation:
            elig = resolve_nucleation_eligibility(
                data,
                t,
                device,
                phys_cfg,
                bio_cfg,
                commits_prev=commits_prev,
                growth_seed="pred",
                nucleation_hops=nucleation_hops,
                use_dgamma_wall_seed=use_dgamma_wall_seed,
            )
            phi = project_phi_with_nucleation(phi_raw, phi_prev, elig)
            commits_prev = (phi.reshape(-1) >= 0.5).bool()
        phi_prev = phi.detach()
        out[t] = {"phi_raw": phi_raw, "phi": phi, "mu_pred": step.mu_pred_si}
    return out
