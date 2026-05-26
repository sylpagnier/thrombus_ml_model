"""Survey COMSOL clot nodes across biochem anchor graphs (where / when they form)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND
from src.core_physics.clot_kinematics_fields import (
    adjacent_band_mask,
    compute_clot_kinematics_fields,
    score_clot_risk_from_fields,
)
def _graph_props(data, device: torch.device) -> dict[str, torch.Tensor]:
    if isinstance(data.u_ref, torch.Tensor) and data.u_ref.numel() == data.num_nodes:
        u_ref = data.u_ref.to(device=device, dtype=torch.float32).reshape(-1)[:1]
        d_bar = data.d_bar.to(device=device, dtype=torch.float32).reshape(-1)[:1]
    else:
        u_ref = torch.as_tensor(data.u_ref, device=device, dtype=torch.float32).reshape(1)
        d_bar = torch.as_tensor(data.d_bar, device=device, dtype=torch.float32).reshape(1)
    return {"u_ref": u_ref, "d_bar": d_bar}


def _sdf_nd(data, device: torch.device, n: int) -> torch.Tensor:
    if hasattr(data, "x") and torch.is_tensor(data.x) and data.x.dim() == 2 and data.x.shape[1] > 2:
        return data.x[:, 2].to(device=device, dtype=torch.float32).clamp(min=0.0)
    return torch.zeros(n, device=device, dtype=torch.float32)


def _wall_mask(data, device: torch.device, n: int) -> torch.Tensor:
    if hasattr(data, "mask_wall") and data.mask_wall is not None:
        return data.mask_wall.view(-1).to(device=device).bool()
    return torch.zeros(n, dtype=torch.bool, device=device)


def _mu_si_trajectory(data, phys: PhysicsConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(mu_si [T,N], t_sec [T])``."""
    if not hasattr(data, "y") or data.y is None:
        raise ValueError("graph has no data.y trajectory")
    y = data.y.to(dtype=torch.float32)
    mu = phys.viscosity_nd_to_si(y[:, :, STATE_CHANNEL_MU_EFF_ND])
    if hasattr(data, "t") and data.t is not None:
        t = data.t.to(dtype=torch.float32).reshape(-1)
    else:
        t = torch.arange(mu.shape[0], dtype=torch.float32)
    return mu, t


def _first_clot_index_per_node(mu_si: torch.Tensor, mu_floor: float) -> torch.Tensor:
    """Per-node first time index where ``μ >= mu_floor``; ``T`` if never."""
    clot = mu_si >= mu_floor
    t_steps = mu_si.shape[0]
    idx = torch.arange(t_steps, device=mu_si.device, dtype=torch.long).view(t_steps, 1)
    masked = torch.where(clot, idx, torch.full_like(idx, t_steps))
    return masked.min(dim=0).values


@dataclass
class AnchorClotSurvey:
    stem: str
    n_nodes: int
    n_times: int
    t_final_s: float
    mu_floor_si: float
    n_clot_strict_t0: int
    n_clot_strict_tfinal: int
    n_clot_p90_tfinal: int
    first_any_clot_time_s: float | None
    first_any_clot_frac: float | None
    pct_clot_on_wall_tfinal: float
    pct_clot_adjacent_tfinal: float
    pct_clot_off_wall_tfinal: float
    pct_clot_bulk_tfinal: float
    pct_clot_near_wall_sdf_tfinal: float
    median_sdf_nd_clot_tfinal: float
    median_sdf_nd_nonclot_adjacent: float
    dgamma_dx_clot_mean_tfinal: float
    dgamma_dx_non_mean_tfinal: float
    dshear_ds_clot_mean_tfinal: float
    gamma_clot_mean_tfinal: float
    prior_clot_mean_tfinal: float
    prior_non_mean_tfinal: float
    suggested_dx_thresh_p10: float
    inception_dgamma_dx_mean: float
    inception_n_nodes: int
    notes: list[str] = field(default_factory=list)


def survey_anchor_graph(
    data,
    *,
    stem: str = "unknown",
    bio_cfg: BiochemConfig | None = None,
    phys_cfg: PhysicsConfig | None = None,
    device: torch.device | None = None,
) -> AnchorClotSurvey:
    """Characterize clot nodes on one anchor graph (GT ``data.y``)."""
    bio_cfg = bio_cfg or BiochemConfig(phase="biochem")
    phys_cfg = phys_cfg or PhysicsConfig()
    device = device or torch.device("cpu")

    n = int(data.num_nodes)
    mu_traj, t_sec = _mu_si_trajectory(data, phys_cfg)
    mu_traj = mu_traj.to(device)
    t_sec = t_sec.to(device)
    t_final_idx = int(t_sec.argmax().item())
    t0_idx = int((t_sec - t_sec.min()).abs().argmin().item())

    mu_floor = max(
        float(os.environ.get("BIOCHEM_K11_CLOT_MU_SI_MIN", "0.055")),
        float(phys_cfg.mu_inf),
    )
    clot_t = mu_traj >= mu_floor
    clot_t0 = clot_t[t0_idx]
    clot_tf = clot_t[t_final_idx]
    n_clot_t0 = int(clot_t0.sum())
    n_clot_tf = int(clot_tf.sum())

    mu_tf = mu_traj[t_final_idx]
    mu_cut = torch.quantile(mu_tf, 0.9)
    n_p90 = int((mu_tf >= mu_cut).sum())

    first_idx = _first_clot_index_per_node(mu_traj, mu_floor)
    ever = first_idx < mu_traj.shape[0]
    if bool(ever.any().item()):
        global_first = int(first_idx[ever].min().item())
        first_any_clot_time_s = float(t_sec[global_first].item())
        span = max(float(t_sec[-1].item()) - float(t_sec[0].item()), 1e-6)
        first_any_clot_frac = (first_any_clot_time_s - float(t_sec[0].item())) / span
    else:
        first_any_clot_time_s = None
        first_any_clot_frac = None

    sdf = _sdf_nd(data, device, n)
    wall = _wall_mask(data, device, n)
    adj = adjacent_band_mask(sdf, data.mask_wall if hasattr(data, "mask_wall") else None)

    props = _graph_props(data, device)
    y_tf = data.y[t_final_idx].to(device)
    u = y_tf[:, 0]
    v = y_tf[:, 1]
    fields = compute_clot_kinematics_fields(data, u, v, bio_cfg, props)
    prior, _, _ = score_clot_risk_from_fields(fields, bio_cfg)

    clot_tf_dev = clot_tf.to(device)
    non_tf = ~clot_tf_dev
    adj_dev = adj.to(device)

    def _pct(mask: torch.Tensor) -> float:
        n_c = int(clot_tf_dev.sum())
        if n_c == 0:
            return 0.0
        return 100.0 * float((mask & clot_tf_dev).sum()) / n_c

    pct_wall = _pct(wall)
    pct_adj = _pct(adj_dev)
    pct_off_wall = _pct(~wall)
    pct_bulk = _pct(~wall & ~adj_dev)
    near_wall = sdf <= 0.02
    pct_near_sdf = _pct(near_wall)

    if n_clot_tf > 0:
        med_sdf_clot = float(sdf[clot_tf_dev].median())
        med_sdf_non = float(sdf[non_tf & adj_dev].median()) if (non_tf & adj_dev).any() else float("nan")
        dx_clot = float(fields.dgamma_dx_phys[clot_tf_dev].mean())
        dx_non = float(fields.dgamma_dx_phys[non_tf & adj_dev].mean()) if (non_tf & adj_dev).any() else float("nan")
        ds_clot = float(fields.dshear_ds_phys[clot_tf_dev].mean())
        g_clot = float(fields.gamma_si[clot_tf_dev].mean())
        pr_clot = float(prior[clot_tf_dev].mean())
        pr_non = float(prior[non_tf & adj_dev].mean()) if (non_tf & adj_dev).any() else float("nan")
        calib_mask = clot_tf_dev & (near_wall if near_wall.any() else adj_dev)
        if not bool(calib_mask.any().item()):
            calib_mask = clot_tf_dev
        dx_c = fields.dgamma_dx_phys[calib_mask]
        neg_mag = (-dx_c[dx_c < 0]).clamp(min=0.0)
        if neg_mag.numel() >= 3:
            suggested = float(torch.quantile(neg_mag, 0.10).item())
        else:
            suggested = float("nan")
    else:
        med_sdf_clot = med_sdf_non = dx_clot = dx_non = ds_clot = g_clot = pr_clot = pr_non = suggested = float("nan")

    # Kinematics at each clot node's first clot time (where it "started").
    inception_dx: list[float] = []
    clot_nodes = torch.where(clot_tf_dev)[0]
    if clot_nodes.numel() > 0:
        onset_idx = first_idx[clot_nodes]
        for fi in torch.unique(onset_idx[onset_idx < mu_traj.shape[0]]):
            fi_i = int(fi.item())
            y_slice = data.y[fi_i].to(device)
            f_inc = compute_clot_kinematics_fields(
                data, y_slice[:, 0], y_slice[:, 1], bio_cfg, props
            )
            mask = onset_idx == fi_i
            nodes_at_t = clot_nodes[mask]
            inception_dx.extend(
                f_inc.dgamma_dx_phys[nodes_at_t].detach().cpu().tolist()
            )

    notes: list[str] = []
    if n_clot_tf == 0:
        notes.append("no strict clots at t_final")
    if n_clot_t0 > 0:
        notes.append(f"{n_clot_t0} strict clots already at t=0")
    if first_any_clot_frac is not None and first_any_clot_frac < 0.5:
        notes.append("first clot before mid-trajectory")
    elif first_any_clot_frac is not None:
        notes.append("first clot late in trajectory")

    return AnchorClotSurvey(
        stem=stem,
        n_nodes=n,
        n_times=int(mu_traj.shape[0]),
        t_final_s=float(t_sec[t_final_idx].item()),
        mu_floor_si=mu_floor,
        n_clot_strict_t0=n_clot_t0,
        n_clot_strict_tfinal=n_clot_tf,
        n_clot_p90_tfinal=n_p90,
        first_any_clot_time_s=first_any_clot_time_s,
        first_any_clot_frac=first_any_clot_frac,
        pct_clot_on_wall_tfinal=pct_wall,
        pct_clot_adjacent_tfinal=pct_adj,
        pct_clot_off_wall_tfinal=pct_off_wall,
        pct_clot_bulk_tfinal=pct_bulk,
        pct_clot_near_wall_sdf_tfinal=pct_near_sdf,
        median_sdf_nd_clot_tfinal=med_sdf_clot,
        median_sdf_nd_nonclot_adjacent=med_sdf_non,
        dgamma_dx_clot_mean_tfinal=dx_clot,
        dgamma_dx_non_mean_tfinal=dx_non,
        dshear_ds_clot_mean_tfinal=ds_clot,
        gamma_clot_mean_tfinal=g_clot,
        prior_clot_mean_tfinal=pr_clot,
        prior_non_mean_tfinal=pr_non,
        suggested_dx_thresh_p10=suggested,
        inception_dgamma_dx_mean=float(sum(inception_dx) / max(len(inception_dx), 1)) if inception_dx else float("nan"),
        inception_n_nodes=len(inception_dx),
        notes=notes,
    )


def discover_anchor_paths(anchor_dir: Path | None = None) -> list[Path]:
    root = anchor_dir or (Path(__file__).resolve().parents[2] / "data" / "processed" / "graphs_biochem_anchors")
    if not root.is_dir():
        return []
    return sorted(root.glob("patient*.pt"))


def survey_all_anchors(anchor_dir: Path | None = None) -> list[AnchorClotSurvey]:
    out: list[AnchorClotSurvey] = []
    for path in discover_anchor_paths(anchor_dir):
        data = torch.load(path, map_location="cpu", weights_only=False)
        out.append(survey_anchor_graph(data, stem=path.stem))
    return out


def format_survey_table(surveys: list[AnchorClotSurvey]) -> str:
    if not surveys:
        return "(no anchor graphs found)"
    lines = [
        "Anchor clot survey (strict μ >= K11 floor; GT u,v kinematics)",
        "-" * 100,
    ]
    hdr = (
        f"{'stem':<14} {'clot@tf':>8} {'clot@t0':>8} {'1st%':>6} "
        f"{'%wall':>6} {'%offW':>6} {'%adj':>6} {'%sdf≤2e-2':>8} "
        f"{'dγ/dx c':>8} {'dγ/dx n':>8} {'dx_p10':>7}"
    )
    lines.append(hdr)
    for s in surveys:
        first_pct = f"{100.0 * s.first_any_clot_frac:.0f}" if s.first_any_clot_frac is not None else "—"
        lines.append(
            f"{s.stem:<14} {s.n_clot_strict_tfinal:>8} {s.n_clot_strict_t0:>8} {first_pct:>6} "
            f"{s.pct_clot_on_wall_tfinal:>5.1f}% {s.pct_clot_off_wall_tfinal:>5.1f}% "
            f"{s.pct_clot_adjacent_tfinal:>5.1f}% {s.pct_clot_near_wall_sdf_tfinal:>7.1f}% "
            f"{s.dgamma_dx_clot_mean_tfinal:>8.2f} {s.dgamma_dx_non_mean_tfinal:>8.2f} "
            f"{s.suggested_dx_thresh_p10:>7.2f}"
        )
    # Aggregate calibration hint
    p10_vals = [s.suggested_dx_thresh_p10 for s in surveys if s.suggested_dx_thresh_p10 == s.suggested_dx_thresh_p10]
    if p10_vals:
        med = float(torch.tensor(p10_vals).median())
        lines.append("-" * 100)
        lines.append(
            f"Suggested BIOCHEM_PRIOR_DGAMMA_DX_THRESH (median p10 adverse |dγ/dx| on clot nodes, sdf≤0.02): ~{med:.1f} "
            f"(COMSOL plot used 800; mesh dγ/dx is typically ~O(1–100))"
        )
    return "\n".join(lines)


def suggest_prior_dx_threshold(
    anchor_dir: Path | None = None,
    *,
    fallback: float = 35.0,
    quantile: float = 0.10,
    clamp_min: float = 5.0,
    clamp_max: float = 120.0,
) -> float:
    """Data-driven ``BIOCHEM_PRIOR_DGAMMA_DX_THRESH`` from GT clot adverse dγ/dx on anchors."""
    import os

    os.environ.setdefault("BIOCHEM_PRIOR_COMSOL_ALIGNED", "1")
    vals: list[float] = []
    for s in survey_all_anchors(anchor_dir):
        if s.n_clot_strict_tfinal < 5:
            continue
        v = s.suggested_dx_thresh_p10
        if v == v and v > 0.0:
            vals.append(v)
    if not vals:
        return float(fallback)
    raw = float(torch.tensor(vals).median().item())
    return float(min(max(raw, clamp_min), clamp_max))


def aggregate_separability(surveys: list[AnchorClotSurvey]) -> dict[str, Any]:
    """Pool anchors that have strict clots at t_final."""
    active = [s for s in surveys if s.n_clot_strict_tfinal >= 5]
    if not active:
        return {"n_anchors": 0}
    dx_delta = []
    for s in active:
        d = s.dgamma_dx_clot_mean_tfinal - s.dgamma_dx_non_mean_tfinal
        if d == d:
            dx_delta.append(d)
    med_thresh = [
        s.suggested_dx_thresh_p10 for s in active if s.suggested_dx_thresh_p10 == s.suggested_dx_thresh_p10
    ]
    return {
        "n_anchors": len(active),
        "mean_pct_near_wall_sdf": sum(s.pct_clot_near_wall_sdf_tfinal for s in active) / len(active),
        "mean_pct_adjacent": sum(s.pct_clot_adjacent_tfinal for s in active) / len(active),
        "mean_pct_wall_flag": sum(s.pct_clot_on_wall_tfinal for s in active) / len(active),
        "mean_dx_delta": sum(dx_delta) / len(dx_delta) if dx_delta else float("nan"),
        "n_anchors_dx_negative": sum(1 for d in dx_delta if d < 0),
        "median_suggested_dx_thresh": float(torch.tensor(med_thresh).median()) if med_thresh else float("nan"),
    }
