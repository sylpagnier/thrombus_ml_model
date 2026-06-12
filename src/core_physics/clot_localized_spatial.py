"""Localized wall spatial support: segment ranking, recess gate, arc skip, species."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from src.config import BiochemConfig
from src.core_physics.clot_phi_simple import (
    ClotPriorRuleConfig,
    _hop_distance_from_seed,
    _top_frac_mask,
    _wall_mask_from_data,
    sdf_nd_from_data,
)
from src.utils.channel_schema import BIO_Y_SCHEMA, KINE_X_SCHEMA, X_SCHEMAS, Y_SCHEMAS


@dataclass(frozen=True)
class LocalizedSpatialConfig:
    """Per-segment spatial pool + support (not global ceiling ranking)."""

    mode: str = "wall_half"  # wall_half | arc_bins
    segment_top_frac: float = 0.25
    skip_wall_arc_frac: float = 0.15
    n_arc_bins: int = 4
    recess_gate: bool = False
    recess_width_d2_q: float = 0.55
    recess_y_within_half_q: float = 0.0
    # GT species oracle (upper bound); 0 = off
    species_gt_top_q: float = 0.0
    species_gt_time: str = "t_out"  # t_out | t0 | t_in
    species_channels: tuple[str, ...] = ("FI_log1p_nd", "Mat_log1p_nd")
    species_risk_weight: float = 0.0  # blend normalized species into risk @ species time
    normalize_risk_per_half: bool = True  # rank within each wall half fairly
    wall_halves: tuple[str, ...] = ("lower", "upper")  # lower | upper | both via tuple
    neg_dx_risk_weight: float = 0.45  # vs 0.25 global; favors lower-wall shear signal
    sep_stream_risk_weight: float = 0.0  # high-shear -> low-shear along stream
    stasis_risk_weight: float = 0.0  # low shear rate + residence (large aneurysm)
    low_grad_risk_weight: float = 0.0  # |grad gamma| below threshold [1/s per m ~ Pa/s scale]
    low_shear_thresh_si: float = 0.0  # 0 = biochem lss; user target often 10 Pa/s
    low_grad_thresh_si: float = 10.0
    aneurysm_size_mode: str = ""  # "" | auto | small_neg_dx | large_stasis

    def describe(self) -> str:
        parts = [self.mode, f"top{self.segment_top_frac:.2f}"]
        if self.skip_wall_arc_frac > 0:
            parts.append(f"skip_arc{self.skip_wall_arc_frac:.2f}")
        if self.mode == "arc_bins":
            parts.append(f"bins{self.n_arc_bins}")
        if self.recess_gate:
            parts.append(f"recess_d2q{self.recess_width_d2_q:.2f}")
            if self.recess_y_within_half_q > 0:
                parts.append(f"yq{self.recess_y_within_half_q:.2f}")
        if self.species_gt_top_q > 0:
            parts.append(f"sp_gt_q{self.species_gt_top_q:.2f}")
        if self.species_risk_weight > 0:
            parts.append(f"sp_w{self.species_risk_weight:.2f}")
        if not self.normalize_risk_per_half:
            parts.append("global_norm")
        if self.wall_halves != ("lower", "upper"):
            parts.append("walls=" + "+".join(self.wall_halves))
        if self.sep_stream_risk_weight > 0:
            parts.append(f"sep{self.sep_stream_risk_weight:.2f}")
        if self.stasis_risk_weight > 0:
            parts.append(f"stag{self.stasis_risk_weight:.2f}")
        if self.low_grad_risk_weight > 0:
            parts.append(f"lgrad{self.low_grad_risk_weight:.2f}")
        if self.low_shear_thresh_si > 0:
            parts.append(f"lss{self.low_shear_thresh_si:.0f}")
        if self.aneurysm_size_mode:
            parts.append(f"sz={self.aneurysm_size_mode}")
        return "|".join(parts)


def _x_kine_channel(data, name: str, device: torch.device) -> torch.Tensor | None:
    if not hasattr(data, "x") or data.x is None:
        return None
    if getattr(data, "x_schema", None) != KINE_X_SCHEMA:
        return None
    try:
        idx = X_SCHEMAS[KINE_X_SCHEMA].channels.index(name)
    except ValueError:
        return None
    return data.x[:, idx].to(device=device, dtype=torch.float32)


def _y_channel_at(data, time_index: int, name: str, device: torch.device) -> torch.Tensor | None:
    try:
        idx = Y_SCHEMAS[BIO_Y_SCHEMA].channels.index(name)
    except ValueError:
        return None
    ti = max(0, min(int(time_index), int(data.y.shape[0]) - 1))
    return data.y[ti, :, idx].to(device=device, dtype=torch.float32)


def wall_half_masks(data, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(data.num_nodes)
    wall = _wall_mask_from_data(data, device, n)
    pos = data.x[:, :2].to(device)
    ym = pos[wall, 1].median()
    lower = wall & (pos[:, 1] <= ym)
    upper = wall & (pos[:, 1] > ym)
    return lower, upper


def wall_arc_fraction(data, device: torch.device, wall_half: torch.Tensor) -> torch.Tensor:
    """Normalized [0,1] streamwise position along a wall half.

    Uses inlet-hop when it spans the half; falls back to ``x_nd`` rank when hop
    is degenerate (common on lower wall far from inlet in curved anchors).
    """
    n = int(data.num_nodes)
    arc = torch.zeros(n, device=device)
    if not bool(wall_half.any().item()):
        return arc

    hop = torch.zeros(n, device=device)
    if hasattr(data, "mask_inlet") and data.mask_inlet is not None:
        inlet = data.mask_inlet.view(-1).to(device).bool()
        if bool(inlet.any().item()):
            hop = _hop_distance_from_seed(inlet, data.edge_index.to(device)).float()
    if not bool(hop[wall_half].any().item()) or float(hop[wall_half].max() - hop[wall_half].min()) < 1e-6:
        hop = data.x[:, 0].to(device).float()

    hw = hop[wall_half]
    hmin = float(hw.min())
    hmax = float(hw.max())
    hop_arc = torch.zeros(n, device=device)
    hop_arc[wall_half] = (hop[wall_half] - hmin) / max(hmax - hmin, 1e-12)

    # Degenerate hop: lower wall often sits downstream (hop ~constant). Use x_nd.
    n_half = int(wall_half.sum())
    n_unique = len(torch.unique(hop_arc[wall_half].round(decimals=4)))
    hop_span = float(hop_arc[wall_half].max() - hop_arc[wall_half].min())
    if n_half >= 8 and (hop_span < 0.25 or n_unique < max(8, n_half // 10)):
        xs = data.x[wall_half, 0].to(device).float()
        xmin = float(xs.min())
        xmax = float(xs.max())
        arc[wall_half] = (xs - xmin) / max(xmax - xmin, 1e-12)
        return arc

    arc[wall_half] = hop_arc[wall_half]
    return arc


def active_wall_halves(data, device: torch.device, loc: LocalizedSpatialConfig) -> tuple[torch.Tensor, ...]:
    lower, upper = wall_half_masks(data, device)
    out: list[torch.Tensor] = []
    for name in loc.wall_halves:
        if name == "lower":
            out.append(lower)
        elif name == "upper":
            out.append(upper)
    if not out:
        return (lower, upper)
    return tuple(out)


def apply_skip_early_wall_arc(
    pool: torch.Tensor,
    data,
    device: torch.device,
    skip_frac: float,
    *,
    loc: LocalizedSpatialConfig | None = None,
) -> torch.Tensor:
    """Drop first ``skip_frac`` of arc length on each active wall half."""
    if skip_frac <= 0:
        return pool
    halves = active_wall_halves(data, device, loc or LocalizedSpatialConfig())
    out = pool.clone()
    for half in halves:
        seg = pool & half
        if not bool(seg.any().item()):
            continue
        arc = wall_arc_fraction(data, device, half)
        out = out & ~(seg & (arc < float(skip_frac)))
    return out


def apply_recess_gate(
    pool: torch.Tensor,
    data,
    device: torch.device,
    *,
    width_d2_q: float,
    y_within_half_q: float,
) -> torch.Tensor:
    """Keep concave-pocket candidates (width_d2 tail + optional y rank within half)."""
    if not bool(pool.any().item()):
        return pool
    lower, upper = wall_half_masks(data, device)
    width_d2 = _x_kine_channel(data, "width_d2", device)
    out = torch.zeros_like(pool)
    pos_y = data.x[:, 1].to(device).float()

    for half in (lower, upper):
        seg = pool & half
        if not bool(seg.any().item()):
            continue
        keep = seg.clone()
        if width_d2 is not None:
            wd = width_d2.reshape(-1)
            thr = torch.quantile(wd[seg], float(width_d2_q))
            keep = keep & (wd <= thr)
        if y_within_half_q > 0:
            y_thr = torch.quantile(pos_y[seg], float(y_within_half_q))
            keep = keep & (pos_y >= y_thr)
        out = out | keep
    return out


def _norm_pool(v: torch.Tensor, pool: torch.Tensor) -> torch.Tensor:
    if not bool(pool.any().item()):
        return torch.zeros_like(v)
    vp = v[pool]
    return (v - vp.min()) / (vp.max() - vp.min() + 1e-12)


def species_score_at_time(
    data,
    time_index: int,
    device: torch.device,
    channels: tuple[str, ...],
    pool: torch.Tensor,
) -> torch.Tensor:
    n = int(data.num_nodes)
    acc = torch.zeros(n, device=device)
    n_ch = 0
    for ch in channels:
        val = _y_channel_at(data, time_index, ch, device)
        if val is None:
            continue
        acc = acc + _norm_pool(val.reshape(-1), pool)
        n_ch += 1
    if n_ch == 0:
        return torch.zeros(n, device=device)
    return (acc / n_ch).clamp(0, 1)


def species_gt_mask(
    data,
    time_index: int,
    device: torch.device,
    pool: torch.Tensor,
    *,
    top_q: float,
    channels: tuple[str, ...],
) -> torch.Tensor:
    if top_q <= 0 or not bool(pool.any().item()):
        return torch.ones(int(data.num_nodes), dtype=torch.bool, device=device)
    score = species_score_at_time(data, time_index, device, channels, pool)
    thr = torch.quantile(score[pool], float(top_q))
    return pool & (score >= thr)


def build_eligible_pool(
    data,
    device: torch.device,
    ceiling: torch.Tensor,
    spatial_rule: ClotPriorRuleConfig | None,
    loc: LocalizedSpatialConfig,
) -> torch.Tensor:
    n = int(data.num_nodes)
    pool = ceiling.reshape(-1).bool()
    rule = spatial_rule
    if rule and rule.rank_sdf_max_nd is not None:
        sdf = sdf_nd_from_data(data, device, n)
        pool = pool & (sdf <= float(rule.rank_sdf_max_nd))
    if rule and rule.skip_inlet_quantile is not None:
        if hasattr(data, "mask_inlet") and data.mask_inlet is not None:
            inlet = data.mask_inlet.view(-1).to(device).bool()
            if int(inlet.numel()) == n and bool(inlet.any().item()):
                hin = _hop_distance_from_seed(inlet, data.edge_index.to(device)).float()
                eligible = pool & (hin > 0)
                if bool(eligible.any().item()):
                    thr = torch.quantile(hin[eligible], float(rule.skip_inlet_quantile))
                    pool = pool & (hin >= thr)
    pool = apply_skip_early_wall_arc(pool, data, device, loc.skip_wall_arc_frac, loc=loc)
    if loc.recess_gate:
        pool = apply_recess_gate(
            pool,
            data,
            device,
            width_d2_q=loc.recess_width_d2_q,
            y_within_half_q=loc.recess_y_within_half_q,
        )
    return pool


def segment_topk_mask(
    risk: torch.Tensor,
    data,
    device: torch.device,
    pool: torch.Tensor,
    loc: LocalizedSpatialConfig,
    *,
    top_frac_override: float | None = None,
) -> torch.Tensor:
    """Top fraction within each wall half or arc bin (not global ceiling)."""
    n = int(risk.numel())
    out = torch.zeros(n, dtype=torch.bool, device=device)
    top_frac = max(min(float(top_frac_override if top_frac_override is not None else loc.segment_top_frac), 0.95), 0.01)
    halves = active_wall_halves(data, device, loc)

    if loc.mode == "wall_half":
        for half in halves:
            seg = pool & half
            if bool(seg.any().item()):
                out = out | _top_frac_mask(risk, seg, top_frac)
        return out

    if loc.mode == "arc_bins":
        n_bins = max(int(loc.n_arc_bins), 1)
        for half in halves:
            seg = pool & half
            if not bool(seg.any().item()):
                continue
            arc = wall_arc_fraction(data, device, half)
            for b in range(n_bins):
                lo = b / n_bins
                hi = (b + 1) / n_bins
                if b == n_bins - 1:
                    bin_m = seg & (arc >= lo) & (arc <= hi)
                else:
                    bin_m = seg & (arc >= lo) & (arc < hi)
                if bool(bin_m.any().item()):
                    out = out | _top_frac_mask(risk, bin_m, top_frac)
        return out

    raise ValueError(f"unknown localized mode {loc.mode}")


def normalize_risk_per_wall_half(
    risk: torch.Tensor,
    data,
    device: torch.device,
    pool: torch.Tensor,
    loc: LocalizedSpatialConfig,
) -> torch.Tensor:
    """Re-rank risk within each wall half so lower/upper compete fairly."""
    out = torch.zeros_like(risk)
    for half in active_wall_halves(data, device, loc):
        seg = pool & half
        if not bool(seg.any().item()):
            continue
        rv = risk[seg]
        rmin = rv.min()
        rmax = rv.max()
        out[seg] = (rv - rmin) / (rmax - rmin + 1e-12)
    return out.clamp(0, 1) * pool.float()


def resolve_species_time_index(data, when: str, t_out: int, t_in: int = 0) -> int:
    n = int(data.y.shape[0]) - 1
    w = when.strip().lower()
    if w == "t0":
        return 0
    if w in ("t_in", "tin"):
        return max(0, min(int(t_in), n))
    return max(0, min(int(t_out), n))


def build_localized_static_support(
    risk: torch.Tensor,
    data,
    device: torch.device,
    pool: torch.Tensor,
    loc: LocalizedSpatialConfig,
    *,
    t_out: int = 0,
    t_in: int = 0,
) -> torch.Tensor:
    """Segment-local spatial flag map (replaces global predict_phi_prior_rule support)."""
    support = segment_topk_mask(risk, data, device, pool, loc)
    if loc.species_gt_top_q > 0:
        ti = resolve_species_time_index(data, loc.species_gt_time, t_out, t_in)
        sp = species_gt_mask(
            data,
            ti,
            device,
            pool,
            top_q=loc.species_gt_top_q,
            channels=loc.species_channels,
        )
        support = support & sp
    return support


def blend_species_into_risk(
    risk: torch.Tensor,
    data,
    device: torch.device,
    pool: torch.Tensor,
    loc: LocalizedSpatialConfig,
    time_index: int,
) -> torch.Tensor:
    if loc.species_risk_weight <= 0:
        return risk
    sp = species_score_at_time(data, time_index, device, loc.species_channels, pool)
    w = float(loc.species_risk_weight)
    return (risk * (1.0 - w) + sp * w).clamp(0, 1) * pool.float()


def probe_species_oracle_auc(
    data,
    *,
    stem: str,
    t_out: int,
    device: torch.device,
    phys,
    bio_cfg: BiochemConfig,
) -> list[dict[str, Any]]:
    """AUC of GT species @ t for clot vs non-clot inside localized pool."""
    from src.core_physics.clot_growth_masks import gt_clot_mask_at_time, resolve_ceiling_mask
    from src.core_physics.clot_t0_pattern_probe import _binary_auc

    ceiling = resolve_ceiling_mask(data, device, bio_cfg)
    loc = LocalizedSpatialConfig(skip_wall_arc_frac=0.15)
    pool = build_eligible_pool(data, device, ceiling, None, loc)
    gt = gt_clot_mask_at_time(data, t_out, phys, device)
    pos = pool & gt
    neg = pool & ~gt
    if int(pos.sum()) < 3 or int(neg.sum()) < 3:
        return []

    rows = []
    for ch in ("FI_log1p_nd", "Mat_log1p_nd", "M_log1p_nd", "T_log1p_nd"):
        val = _y_channel_at(data, t_out, ch, device)
        if val is None:
            continue
        v = val.reshape(-1).float()
        auc = max(_binary_auc(v, pos.float()), _binary_auc(-v, pos.float()))
        rows.append(
            {
                "anchor": stem,
                "species": ch,
                "t_out": t_out,
                "auc": auc,
                "clot_mean": float(v[pos].mean()),
                "non_mean": float(v[neg].mean()),
            }
        )
    combo = species_score_at_time(data, t_out, device, ("FI_log1p_nd", "Mat_log1p_nd"), pool)
    auc = max(_binary_auc(combo, pos.float()), _binary_auc(-combo, pos.float()))
    rows.append(
        {
            "anchor": stem,
            "species": "FI+Mat_combo",
            "t_out": t_out,
            "auc": auc,
            "clot_mean": float(combo[pos].mean()),
            "non_mean": float(combo[neg].mean()),
        }
    )
    return rows
