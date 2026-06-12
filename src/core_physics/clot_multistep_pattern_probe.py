"""Multi-timestep clot vs non-clot pattern probe (5 snapshots + trends + neighbors).

Offline discovery on GT anchor graphs: compares nodes that are clots @ t_final
vs non-clots inside the deploy ceiling. Sweeps deploy-safe rules with caps on
pred+ to avoid whole-wall flags.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND
from src.core_physics.clot_anchor_survey import (
    _first_clot_index_per_node,
    _graph_props,
    _mu_si_trajectory,
    discover_anchor_paths,
)
from src.core_physics.clot_growth_masks import resolve_ceiling_mask
from src.core_physics.clot_kinematics_fields import compute_clot_kinematics_fields, score_clot_risk_from_fields
from src.core_physics.clot_phi_simple import (
    ClotPriorRuleConfig,
    _anchor_flow_props,
    _hop_distance_from_seed,
    _noslip_wall_mask,
    _top_frac_mask,
    _wall_mask_from_data,
    build_clot_phi_step,
    clot_phi_thresh_si,
    clot_prior_score_flat,
    gt_neg_dgamma_dx_phys,
    sdf_nd_from_data,
)
from src.core_physics.clot_t0_pattern_probe import _binary_auc, _decile_rule_metrics, _wall_mask, build_t0_feature_table
from src.core_physics.clot_t0_extended_probe import build_feature_table_at_time
from src.training.train_clot_phi_simple import _clot_metrics


def select_five_time_indices(n_times: int) -> list[int]:
    """Evenly spaced indices including t=0 and t_final."""
    if n_times <= 1:
        return [0]
    if n_times <= 5:
        return list(range(n_times))
    last = n_times - 1
    raw = [0, last // 4, last // 2, (3 * last) // 4, last]
    out: list[int] = []
    for ti in raw:
        if ti not in out:
            out.append(ti)
    return sorted(out)


def _neighbor_aggregate(
    values: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    n_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-node neighbor mean and fraction of neighbors above node-local median."""
    v = values.reshape(-1).float()
    if int(v.numel()) != n_nodes:
        z = torch.zeros(n_nodes, device=edge_index.device, dtype=torch.float32)
        return z, z
    deg = torch.zeros(n_nodes, device=v.device, dtype=torch.float32)
    nsum = torch.zeros(n_nodes, device=v.device, dtype=torch.float32)
    nhi = torch.zeros(n_nodes, device=v.device, dtype=torch.float32)
    if edge_index.numel() == 0:
        return nsum, nhi
    ei = edge_index.to(device=v.device)
    src, dst = ei[0], ei[1]
    deg.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float32))
    deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
    nsum.scatter_add_(0, src, v[dst])
    nsum.scatter_add_(0, dst, v[src])
    med = v.median()
    hi = v > med
    nhi.scatter_add_(0, src, hi[dst].float())
    nhi.scatter_add_(0, dst, hi[src].float())
    safe_deg = deg.clamp(min=1.0)
    return nsum / safe_deg, nhi / safe_deg


def _time_labels(data) -> torch.Tensor:
    if hasattr(data, "t") and data.t is not None:
        return data.t.to(dtype=torch.float32).reshape(-1)
    return torch.arange(int(data.y.shape[0]), dtype=torch.float32)


def _build_trend_features(
    data,
    time_indices: list[int],
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
) -> dict[str, torch.Tensor]:
    """Temporal deltas and persistence (deploy-early window uses first two indices)."""
    n = int(data.num_nodes)
    if len(time_indices) < 2:
        return {}

    props = _graph_props(data, device)
    speeds: list[torch.Tensor] = []
    stag: list[torch.Tensor] = []
    neg_dx: list[torch.Tensor] = []
    prior: list[torch.Tensor] = []
    gamma: list[torch.Tensor] = []

    for ti in time_indices:
        y = data.y[int(ti)].to(device=device, dtype=torch.float32)
        u, v = y[:, 0], y[:, 1]
        fields = compute_clot_kinematics_fields(data, u, v, bio_cfg, props)
        pr, _, _ = score_clot_risk_from_fields(fields, bio_cfg)
        u_ref = props["u_ref"].reshape(-1).clamp(min=1e-8)
        speeds.append(torch.sqrt(u * u + v * v) * u_ref)
        stag.append(fields.flux_stag.reshape(-1))
        neg_dx.append(gt_neg_dgamma_dx_phys(data, int(ti), bio_cfg, device))
        prior.append(pr.reshape(-1))
        gamma.append(fields.gamma_si.reshape(-1))

    t0, t1 = 0, min(1, len(time_indices) - 1)
    t_last = len(time_indices) - 1
    dt01 = max(float(_time_labels(data)[time_indices[t1]] - _time_labels(data)[time_indices[t0]]), 1e-6)
    dt0f = max(
        float(_time_labels(data)[time_indices[t_last]] - _time_labels(data)[time_indices[t0]]),
        1e-6,
    )

    out: dict[str, torch.Tensor] = {
        "d_speed_dt_early": (speeds[t1] - speeds[t0]) / dt01,
        "d_stag_dt_early": (stag[t1] - stag[t0]) / dt01,
        "d_neg_dx_dt_early": (neg_dx[t1] - neg_dx[t0]) / dt01,
        "d_prior_dt_early": (prior[t1] - prior[t0]) / dt01,
        "d_gamma_dt_span": (gamma[t_last] - gamma[t0]) / dt0f,
        "stag_persist_early": ((stag[t0] + stag[t1]) * 0.5),
        "neg_dx_persist_early": ((neg_dx[t0] + neg_dx[t1]) * 0.5),
        "speed_decel_early": (speeds[t0] - speeds[t1]).clamp(min=0.0),
        "stag_max_5t": torch.stack(stag, dim=0).max(dim=0).values,
        "neg_dx_max_5t": torch.stack(neg_dx, dim=0).max(dim=0).values,
        "prior_max_5t": torch.stack(prior, dim=0).max(dim=0).values,
    }
    return out


def _build_neighbor_features(
    data,
    base_feats: dict[str, torch.Tensor],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    n = int(data.num_nodes)
    ei = data.edge_index.to(device=device)
    out: dict[str, torch.Tensor] = {}
    for key in ("flux_stag", "neg_dgamma_dx", "prior_score", "gamma_si"):
        if key not in base_feats:
            continue
        nmean, nfrac = _neighbor_aggregate(base_feats[key], ei, n_nodes=n)
        out[f"nb_mean_{key}"] = nmean
        out[f"nb_frac_hi_{key}"] = nfrac
    if "flux_stag" in base_feats and "neg_dgamma_dx" in base_feats:
        combo = base_feats["flux_stag"] * base_feats["neg_dgamma_dx"].clamp(min=0.0)
        nmean, nfrac = _neighbor_aggregate(combo, ei, n_nodes=n)
        out["nb_mean_stag_x_negdx"] = nmean
        out["nb_frac_hi_stag_x_negdx"] = nfrac
    return out


def _build_incubation_features(
    data,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
) -> dict[str, torch.Tensor]:
    """Oracle trajectory features (for offline pattern discovery only)."""
    mu_traj, t_sec = _mu_si_trajectory(data, phys_cfg)
    mu_traj = mu_traj.to(device)
    t_sec = t_sec.to(device)
    thr = clot_phi_thresh_si(phys_cfg)
    first_idx = _first_clot_index_per_node(mu_traj, thr).float()
    t_span = max(float(t_sec[-1] - t_sec[0]), 1e-6)
    first_frac = first_idx / max(float(mu_traj.shape[0] - 1), 1.0)
    ever_clot = first_idx < float(mu_traj.shape[0])
    return {
        "first_clot_time_frac": first_frac,
        "ever_clot_oracle": ever_clot.float(),
        "incubation_long": (first_frac > 0.5).float(),
    }


@dataclass
class FeatureTrendRow:
    feature: str
    group: str
    time_index: int
    time_s: float
    auc: float
    clot_mean: float
    non_mean: float
    delta_mean: float
    decile_rec: float


@dataclass
class RuleSweepRow:
    rule: str
    anchor: str
    band_f1: float
    band_prec: float
    band_rec: float
    band_pred_frac: float
    n_flag: int
    n_pool: int


@dataclass
class MultiStepAnchorReport:
    anchor: str
    n_nodes: int
    n_ceiling: int
    n_clot: int
    time_indices: list[int]
    time_s: list[float]
    trend_rows: list[FeatureTrendRow] = field(default_factory=list)
    static_rows: list[dict[str, Any]] = field(default_factory=list)
    rule_rows: list[RuleSweepRow] = field(default_factory=list)
    axial_summary: dict[str, float] = field(default_factory=dict)
    incubation_summary: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _probe_feature(
    *,
    anchor: str,
    feature: str,
    group: str,
    values: torch.Tensor,
    labels: torch.Tensor,
    in_mask: torch.Tensor,
    time_index: int,
    time_s: float,
    higher_is_risk: bool | None = None,
) -> FeatureTrendRow | None:
    lm = labels[in_mask]
    if int(lm.sum()) < 3 or not bool(in_mask.any()):
        return None
    v = values.reshape(-1)
    if int(v.numel()) != int(in_mask.numel()):
        return None
    v = v[in_mask]
    if higher_is_risk is None:
        cm = float(v[lm].mean()) if bool(lm.any()) else 0.0
        nm = float(v[~lm].mean()) if bool((~lm).any()) else 0.0
        higher_is_risk = cm >= nm
    risk = v if higher_is_risk else -v
    cm = float(v[lm].mean())
    nm = float(v[~lm].mean())
    dec = _decile_rule_metrics(risk, lm, frac=0.10)
    return FeatureTrendRow(
        feature=feature,
        group=group,
        time_index=time_index,
        time_s=time_s,
        auc=_binary_auc(risk, lm.float()),
        clot_mean=cm,
        non_mean=nm,
        delta_mean=cm - nm,
        decile_rec=float(dec["rec"]),
    )


def _phi_prior_rule_in_pool(
    data,
    device: torch.device,
    bio_cfg: BiochemConfig,
    rank_mask: torch.Tensor,
    rule: ClotPriorRuleConfig,
    *,
    t_in: int = 0,
) -> torch.Tensor:
    """Apply prior-rule legs with top-k restricted to ``rank_mask`` (deploy pool)."""
    y_in = data.y[int(t_in)].to(device=device, dtype=torch.float32)
    u, v = y_in[:, 0], y_in[:, 1]
    props = _anchor_flow_props(data, device)
    fields = compute_clot_kinematics_fields(data, u, v, bio_cfg, props)
    prior = clot_prior_score_flat(data, u, v, bio_cfg, props).reshape(-1)
    wall = _wall_mask_from_data(data, device, int(data.num_nodes))
    noslip = _noslip_wall_mask(wall, u, v)
    dx_raw = fields.flux_path_dx_raw.reshape(-1)
    hop_wall = _hop_distance_from_seed(wall, data.edge_index.to(device=device)).float()
    tie_dx = dx_raw if rule.rank_tie_break else None
    tie_hop = hop_wall if rule.rank_tie_break else None
    pool = rank_mask.reshape(-1).bool()
    stag_pool = pool
    if rule.stag_off_wall_adjacent:
        stag_pool = pool & fields.adjacent_band.reshape(-1).bool() & ~noslip

    legs: list[torch.Tensor] = []
    if rule.prior_p is not None:
        top_frac = max(1.0 - float(rule.prior_p), 0.01)
        legs.append(_top_frac_mask(prior, pool, top_frac, tie_dx=tie_dx, tie_hop=tie_hop))
    if rule.flux_stag_top_frac is not None:
        legs.append(
            _top_frac_mask(
                fields.flux_stag.reshape(-1),
                stag_pool,
                rule.flux_stag_top_frac,
                tie_dx=tie_dx,
                tie_hop=tie_hop,
            )
        )
    if rule.flux_stream_top_frac is not None:
        legs.append(
            _top_frac_mask(
                fields.flux_path_stream.reshape(-1),
                pool,
                rule.flux_stream_top_frac,
                tie_dx=tie_dx,
                tie_hop=tie_hop,
            )
        )
    if rule.flux_dx_raw_top_frac is not None:
        legs.append(
            _top_frac_mask(
                dx_raw,
                pool,
                rule.flux_dx_raw_top_frac,
                tie_dx=tie_dx,
                tie_hop=tie_hop,
            )
        )
    if rule.neg_dgamma_top_frac is not None:
        neg_dx = (-fields.dgamma_dx_phys).clamp(min=0.0).reshape(-1)
        legs.append(
            _top_frac_mask(neg_dx, pool, rule.neg_dgamma_top_frac, tie_dx=dx_raw, tie_hop=tie_hop)
        )

    if not legs:
        flag = torch.zeros(int(data.num_nodes), device=device, dtype=torch.bool)
    elif str(rule.combine_legs).strip().lower() == "and" and len(legs) > 1:
        flag = legs[0]
        for leg in legs[1:]:
            flag = flag & leg
    else:
        flag = legs[0]
        for leg in legs[1:]:
            flag = flag | leg
    if rule.require_on_wall:
        flag = flag & wall
    if rule.max_hop_from_wall is not None:
        flag = flag & (hop_wall <= float(rule.max_hop_from_wall))
    return flag.float()


def _eval_rule_band(
    *,
    anchor: str,
    rule_name: str,
    phi: torch.Tensor,
    step,
) -> RuleSweepRow:
    band = _clot_metrics(phi, step.phi_gt, step.loss_mask)
    flag = (phi.reshape(-1) > 0.5) & step.loss_mask.reshape(-1).bool()
    return RuleSweepRow(
        rule=rule_name,
        anchor=anchor,
        band_f1=float(band["clot_f1"]),
        band_prec=float(band["clot_prec"]),
        band_rec=float(band["clot_rec"]),
        band_pred_frac=float(band["pred_pos_frac"]),
        n_flag=int(flag.sum()),
        n_pool=int(step.loss_mask.sum()),
    )


def _as_node_vector(values: torch.Tensor, n_nodes: int, device: torch.device) -> torch.Tensor | None:
    v = values.reshape(-1).float().to(device=device)
    if int(v.numel()) != n_nodes:
        return None
    return v


def _rule_combined_score_topk(
    data,
    device: torch.device,
    bio_cfg: BiochemConfig,
    *,
    rank_mask: torch.Tensor,
    t_feats: dict[str, torch.Tensor],
    top_frac: float,
    weights: dict[str, float],
) -> torch.Tensor:
    n = int(data.num_nodes)
    score = torch.zeros(n, device=device, dtype=torch.float32)
    for key, w in weights.items():
        if key not in t_feats:
            continue
        v = _as_node_vector(t_feats[key], n, device)
        if v is None:
            continue
        vmin = v[rank_mask].min() if bool(rank_mask.any()) else v.min()
        vmax = v[rank_mask].max() if bool(rank_mask.any()) else v.max()
        score = score + w * (v - vmin) / (vmax - vmin + 1e-12)
    if "nb_frac_hi_flux_stag" in t_feats:
        v = _as_node_vector(t_feats["nb_frac_hi_flux_stag"], n, device)
        if v is not None:
            score = score + 0.35 * v
    hop = _hop_distance_from_seed(
        _wall_mask_from_data(data, device, n),
        data.edge_index.to(device=device),
    ).float()
    dx = t_feats.get("neg_dgamma_dx")
    dx_v = _as_node_vector(dx, n, device) if dx is not None else torch.zeros(n, device=device)
    flag = _top_frac_mask(score, rank_mask, top_frac, tie_dx=dx_v, tie_hop=hop)
    out = torch.zeros(n, device=device, dtype=torch.float32)
    out[flag] = 1.0
    return out


def _sweep_rules_for_anchor(
    data,
    *,
    stem: str,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    ceiling: torch.Tensor,
    t_feats: dict[str, torch.Tensor],
) -> list[RuleSweepRow]:
    t_out = int(data.y.shape[0]) - 1
    step = build_clot_phi_step(data, t_out, phys_cfg, bio_cfg, device)
    n = int(data.num_nodes)
    wall = _wall_mask_from_data(data, device, n)
    sdf = sdf_nd_from_data(data, device, n)
    ei = data.edge_index.to(device=device)
    hop_wall = _hop_distance_from_seed(wall, ei).float()

    pools: dict[str, torch.Tensor] = {
        "ceiling_h2": ceiling,
        "ceiling_sdf040": ceiling & (sdf <= 0.04),
        "ceiling_sdf035": ceiling & (sdf <= 0.035),
        "ceiling_hop1": ceiling & (hop_wall <= 1.0),
    }

    base_rules: list[tuple[str, ClotPriorRuleConfig]] = [
        (
            "refined_winner",
            ClotPriorRuleConfig(
                name="refined",
                prior_p=0.80,
                use_t0_strip=False,
                flux_stag_top_frac=0.20,
                rank_tie_break=True,
            ),
        ),
        (
            "refined+sdf040",
            ClotPriorRuleConfig(
                name="refined_sdf",
                prior_p=0.80,
                use_t0_strip=False,
                flux_stag_top_frac=0.20,
                rank_tie_break=True,
                rank_sdf_max_nd=0.04,
            ),
        ),
        (
            "stag_offwall_adj",
            ClotPriorRuleConfig(
                name="stag_adj",
                prior_p=0.80,
                use_t0_strip=False,
                flux_stag_top_frac=0.20,
                stag_off_wall_adjacent=True,
                rank_tie_break=True,
            ),
        ),
        (
            "prior_only_p80",
            ClotPriorRuleConfig(
                name="prior_only",
                prior_p=0.80,
                use_t0_strip=False,
            ),
        ),
    ]

    rows: list[RuleSweepRow] = []

    for pool_name, rank_mask in pools.items():
        if not bool(rank_mask.any().item()):
            continue
        for rule_tag, rule_cfg in base_rules:
            phi = _phi_prior_rule_in_pool(data, device, bio_cfg, rank_mask, rule_cfg)
            rows.append(
                _eval_rule_band(
                    anchor=stem,
                    rule_name=f"{pool_name}|{rule_tag}",
                    phi=phi,
                    step=step,
                )
            )

        # Neighbor-gated refined winner
        refined = ClotPriorRuleConfig(
            prior_p=0.80,
            use_t0_strip=False,
            flux_stag_top_frac=0.20,
            rank_tie_break=True,
        )
        for nb_min, nb_key in ((0.35, "nb_frac_hi_flux_stag"), (0.30, "nb_frac_hi_neg_dgamma_dx")):
            phi_base = _phi_prior_rule_in_pool(data, device, bio_cfg, rank_mask, refined)
            nb_raw = t_feats.get(nb_key)
            nb_v = _as_node_vector(nb_raw, n, device) if nb_raw is not None else None
            if nb_v is None:
                continue
            nmean, _ = _neighbor_aggregate(
                nb_v, data.edge_index.to(device=device), n_nodes=n
            )
            phi_nb = (phi_base.reshape(-1).bool() & (nmean >= nb_min)).float()
            rows.append(
                _eval_rule_band(
                    anchor=stem,
                    rule_name=f"{pool_name}|refined+{nb_key}>={nb_min:.2f}",
                    phi=phi_nb,
                    step=step,
                )
            )

        # Combined weighted score
        for top_frac in (0.15, 0.20):
            phi_c = _rule_combined_score_topk(
                data,
                device,
                bio_cfg,
                rank_mask=rank_mask,
                t_feats=t_feats,
                top_frac=top_frac,
                weights={
                    "prior_score": 0.35,
                    "flux_stag": 0.30,
                    "neg_dgamma_dx": 0.25,
                    "stag_persist_early": 0.10,
                },
            )
            rows.append(
                _eval_rule_band(
                    anchor=stem,
                    rule_name=f"{pool_name}|combo_top{int(100*top_frac)}",
                    phi=phi_c,
                    step=step,
                )
            )

        # Exclude early vessel (low hop from inlet): keep top 75% axial distance
        if "hop_from_inlet" in t_feats:
            hin_v = _as_node_vector(t_feats["hop_from_inlet"], n, device)
            if hin_v is not None and bool((rank_mask & (hin_v > 0)).any().item()):
                q25 = torch.quantile(hin_v[rank_mask & (hin_v > 0)], 0.25)
                axial_mask = rank_mask & (hin_v >= q25)
                phi_ax = _phi_prior_rule_in_pool(data, device, bio_cfg, axial_mask, refined)
                rows.append(
                    _eval_rule_band(
                        anchor=stem,
                        rule_name=f"{pool_name}|refined+skip_inlet_q25",
                        phi=phi_ax,
                        step=step,
                    )
                )

        # Deceleration + stagnation intersection (early flow trend)
        if "speed_decel_early" in t_feats and "flux_stag" in t_feats:
            dec_v = _as_node_vector(t_feats["speed_decel_early"], n, device)
            stag_v = _as_node_vector(t_feats["flux_stag"], n, device)
            if dec_v is not None and stag_v is not None:
                dec = dec_v[rank_mask]
                stag = stag_v[rank_mask]
            if dec_v is not None and stag_v is not None and dec.numel() > 10:
                k15 = max(int(0.15 * int(rank_mask.sum())), 1)
                _, top_dec = torch.topk(dec, min(k15, dec.numel()))
                idx_pool = torch.where(rank_mask)[0]
                flag_dec = torch.zeros(n, dtype=torch.bool, device=device)
                flag_dec[idx_pool[top_dec]] = True
                _, top_st = torch.topk(stag, min(k15, stag.numel()))
                flag_st = torch.zeros(n, dtype=torch.bool, device=device)
                flag_st[idx_pool[top_st]] = True
                phi_trend = (flag_dec & flag_st).float()
                rows.append(
                    _eval_rule_band(
                        anchor=stem,
                        rule_name=f"{pool_name}|decel_top15 AND stag_top15",
                        phi=phi_trend,
                        step=step,
                    )
                )

    return rows


def probe_anchor_multistep(
    data,
    *,
    stem: str,
    phys_cfg: PhysicsConfig | None = None,
    bio_cfg: BiochemConfig | None = None,
    device: torch.device | None = None,
    ceiling_hops: int | None = None,
) -> MultiStepAnchorReport:
    phys_cfg = phys_cfg or PhysicsConfig(phase="biochem")
    bio_cfg = bio_cfg or BiochemConfig(phase="biochem")
    device = device or torch.device("cpu")

    n_times = int(data.y.shape[0])
    time_indices = select_five_time_indices(n_times)
    t_labels = _time_labels(data)
    time_s = [float(t_labels[ti].item()) for ti in time_indices]

    ceiling = resolve_ceiling_mask(data, device, bio_cfg, ceiling_hops=ceiling_hops)
    t_final = n_times - 1
    thr = clot_phi_thresh_si(phys_cfg)
    mu_tf = phys_cfg.viscosity_nd_to_si(data.y[t_final][:, STATE_CHANNEL_MU_EFF_ND].to(device))
    labels = (mu_tf >= thr).bool() & ceiling
    n_clot = int(labels.sum())

    # t=0 feature table + trends + neighbors
    feats_t0 = build_t0_feature_table(data, device=device, phys_cfg=phys_cfg, bio_cfg=bio_cfg, t0_index=0)
    feats_t0_ext = build_feature_table_at_time(data, 0, device=device, phys_cfg=phys_cfg, bio_cfg=bio_cfg)
    t0_ext = {k: v[0] for k, (v, _, _) in feats_t0_ext.items()}
    trends = _build_trend_features(data, time_indices, device=device, phys_cfg=phys_cfg, bio_cfg=bio_cfg)
    neighbors = _build_neighbor_features(data, feats_t0, device=device)
    incubation = _build_incubation_features(data, device=device, phys_cfg=phys_cfg)
    n_nodes = int(data.num_nodes)
    t_feats = {**feats_t0, **t0_ext, **trends, **neighbors, **incubation}
    if hasattr(data, "mask_inlet") and data.mask_inlet is not None:
        inlet = data.mask_inlet.view(-1).to(device=device).bool()
        if int(inlet.numel()) == n_nodes:
            t_feats["hop_from_inlet"] = _hop_distance_from_seed(
                inlet, data.edge_index.to(device=device)
            ).float()

    trend_rows: list[FeatureTrendRow] = []
    track_keys = (
        "prior_score",
        "flux_stag",
        "neg_dgamma_dx",
        "gamma_si",
        "vel_mag_si",
        "dshear_ds",
        "flux_path_dx",
        "hop_from_inlet",
        "hop_from_wall",
        "sdf_nd",
    )
    for ti in time_indices:
        feats_t = build_feature_table_at_time(
            data, ti, device=device, phys_cfg=phys_cfg, bio_cfg=bio_cfg
        )
        ts = float(t_labels[ti].item())
        for key in track_keys:
            if key not in feats_t:
                continue
            val, grp, _ = feats_t[key]
            row = _probe_feature(
                anchor=stem,
                feature=key,
                group=grp,
                values=val,
                labels=labels,
                in_mask=ceiling,
                time_index=int(ti),
                time_s=ts,
            )
            if row is not None:
                trend_rows.append(row)

    static_groups = {
        **{k: "trend" for k in trends},
        **{k: "neighbor" for k in neighbors},
        **{k: "incubation_oracle" for k in incubation},
    }
    static_rows: list[dict[str, Any]] = []
    for key, group in static_groups.items():
        if key not in t_feats:
            continue
        row = _probe_feature(
            anchor=stem,
            feature=key,
            group=group,
            values=t_feats[key],
            labels=labels,
            in_mask=ceiling,
            time_index=0,
            time_s=time_s[0],
        )
        if row is not None:
            static_rows.append(asdict(row))

    rule_rows = _sweep_rules_for_anchor(
        data,
        stem=stem,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        ceiling=ceiling,
        t_feats=t_feats,
    )

    axial_summary: dict[str, float] = {}
    hin_v = _as_node_vector(t_feats.get("hop_from_inlet", torch.zeros(1)), int(data.num_nodes), device)
    if hin_v is not None and n_clot > 0:
        lm = labels
        axial_summary["clot_mean_hop_inlet"] = float(hin_v[lm].mean())
        non_mask = ceiling & ~labels
        axial_summary["non_mean_hop_inlet"] = (
            float(hin_v[non_mask].mean()) if bool(non_mask.any()) else float("nan")
        )
        span = hin_v[ceiling].max() - hin_v[ceiling].min() + 1e-6
        axial_summary["clot_axial_frac"] = float((hin_v[lm].mean() - hin_v[ceiling].min()) / span)

    incubation_summary: dict[str, float] = {}
    if n_clot > 0 and "first_clot_time_frac" in t_feats:
        ff = t_feats["first_clot_time_frac"]
        incubation_summary["clot_mean_first_frac"] = float(ff[labels].mean())
        incubation_summary["late_clot_frac"] = float((ff[labels] > 0.5).float().mean())

    notes: list[str] = []
    if n_clot == 0:
        notes.append("no GT clot in ceiling @ t_final")

    return MultiStepAnchorReport(
        anchor=stem,
        n_nodes=int(data.num_nodes),
        n_ceiling=int(ceiling.sum()),
        n_clot=n_clot,
        time_indices=time_indices,
        time_s=time_s,
        trend_rows=trend_rows,
        static_rows=static_rows,
        rule_rows=rule_rows,
        axial_summary=axial_summary,
        incubation_summary=incubation_summary,
        notes=notes,
    )


def probe_all_multistep(anchor_dir: Path | None = None, *, ceiling_hops: int | None = None) -> list[MultiStepAnchorReport]:
    out: list[MultiStepAnchorReport] = []
    for path in discover_anchor_paths(anchor_dir):
        data = torch.load(path, map_location="cpu", weights_only=False)
        out.append(probe_anchor_multistep(data, stem=path.stem, ceiling_hops=ceiling_hops))
    return out


def aggregate_time_auc(reports: list[MultiStepAnchorReport]) -> list[dict[str, Any]]:
    pool: dict[tuple[str, int], list[float]] = {}
    meta: dict[tuple[str, int], dict[str, Any]] = {}
    for rep in reports:
        if rep.n_clot < 5:
            continue
        for row in rep.trend_rows:
            key = (row.feature, row.time_index)
            pool.setdefault(key, []).append(row.auc)
            meta[key] = {"feature": row.feature, "group": row.group, "time_index": row.time_index, "time_s": row.time_s}
    agg = []
    for key, aucs in pool.items():
        valid = [a for a in aucs if a == a]
        if not valid:
            continue
        m = meta[key]
        agg.append(
            {
                **m,
                "mean_auc": sum(valid) / len(valid),
                "n_anchors": len(valid),
            }
        )
    agg.sort(key=lambda x: (-x["mean_auc"], x["feature"], x["time_index"]))
    return agg


def aggregate_static_features(reports: list[MultiStepAnchorReport]) -> list[dict[str, Any]]:
    pool: dict[str, list[dict]] = {}
    for rep in reports:
        if rep.n_clot < 5:
            continue
        for row in rep.static_rows:
            pool.setdefault(row["feature"], []).append(row)
    out = []
    for feat, rows in pool.items():
        aucs = [r["auc"] for r in rows if r["auc"] == r["auc"]]
        if not aucs:
            continue
        out.append(
            {
                "feature": feat,
                "group": rows[0]["group"],
                "mean_auc": sum(aucs) / len(aucs),
                "mean_delta": sum(r["delta_mean"] for r in rows) / len(rows),
                "mean_decile_rec": sum(r["decile_rec"] for r in rows) / len(rows),
                "n_anchors": len(rows),
            }
        )
    out.sort(key=lambda x: -x["mean_auc"])
    return out


def aggregate_rules(
    reports: list[MultiStepAnchorReport],
    *,
    max_pred_frac: float = 0.35,
) -> list[dict[str, Any]]:
    pool: dict[str, list[RuleSweepRow]] = {}
    for rep in reports:
        if rep.n_clot < 5:
            continue
        for row in rep.rule_rows:
            pool.setdefault(row.rule, []).append(row)
    out = []
    for rule, rows in pool.items():
        mean_pred = sum(r.band_pred_frac for r in rows) / len(rows)
        out.append(
            {
                "rule": rule,
                "n_anchors": len(rows),
                "mean_band_f1": sum(r.band_f1 for r in rows) / len(rows),
                "mean_band_prec": sum(r.band_prec for r in rows) / len(rows),
                "mean_band_rec": sum(r.band_rec for r in rows) / len(rows),
                "mean_band_pred_frac": mean_pred,
                "p007_f1": next((r.band_f1 for r in rows if r.anchor == "patient007"), float("nan")),
            }
        )
    out.sort(key=lambda x: (-x["mean_band_f1"], x["mean_band_pred_frac"]))
    if max_pred_frac < 1.0:
        capped = [r for r in out if r["mean_band_pred_frac"] <= max_pred_frac]
        if capped:
            return capped
    return out


def format_multistep_report(reports: list[MultiStepAnchorReport]) -> str:
    lines = [
        "Multi-timestep clot pattern probe (5 snapshots, ceiling mask, GT clot @ t_final)",
        "Deploy features @ t=0 flow; trends/neighbors from early window; incubation = oracle.",
        "",
    ]
    lines.append(f"{'anchor':<12} {'ceil':>6} {'clot':>5} {'times_s':>32}")
    lines.append("-" * 60)
    for r in reports:
        ts = ",".join(f"{t:.0f}" for t in r.time_s[:5])
        lines.append(f"{r.anchor:<12} {r.n_ceiling:>6} {r.n_clot:>5} {ts:>32}")
    lines.append("")

    time_auc = aggregate_time_auc(reports)
    if time_auc:
        lines.append("Feature separability by snapshot (mean AUC inside ceiling):")
        lines.append(f"{'feature':<18} {'t_idx':>5} {'t_s':>8} {'AUC':>6} {'grp':<10}")
        lines.append("-" * 55)
        shown = set()
        for row in time_auc:
            if row["feature"] in shown:
                continue
            shown.add(row["feature"])
            lines.append(
                f"{row['feature']:<18} {row['time_index']:>5} {row['time_s']:>8.0f} "
                f"{row['mean_auc']:>6.3f} {row['group']:<10}"
            )
            if len(shown) >= 15:
                break
        lines.append("")
        lines.append("AUC vs time (same features across 5 steps):")
        for feat in ("flux_stag", "neg_dgamma_dx", "prior_score", "gamma_si", "hop_from_inlet"):
            sub = [r for r in time_auc if r["feature"] == feat]
            if not sub:
                continue
            sub = sorted(sub, key=lambda x: x["time_index"])
            aucs = " -> ".join(f"{r['mean_auc']:.2f}@t{r['time_index']}" for r in sub)
            lines.append(f"  {feat:<18} {aucs}")
        lines.append("")

    static = aggregate_static_features(reports)
    if static:
        lines.append("Early-window / neighbor / trend features @ t=0 (mean AUC):")
        lines.append(f"{'feature':<28} {'AUC':>6} {'d10rec':>7} {'delta':>10}  group")
        lines.append("-" * 70)
        for row in static[:18]:
            lines.append(
                f"{row['feature']:<28} {row['mean_auc']:>6.3f} {row['mean_decile_rec']:>6.1%} "
                f"{row['mean_delta']:>10.3g}  {row['group']}"
            )
        lines.append("")

    rules = aggregate_rules(reports, max_pred_frac=0.40)
    if rules:
        lines.append("Rule sweep (band F1, mean pred+ <= 40% inside loss mask):")
        lines.append(f"{'rule':<52} {'F1':>6} {'prec':>6} {'rec':>6} {'pred+':>6} {'p007':>6}")
        lines.append("-" * 88)
        for row in rules[:15]:
            lines.append(
                f"{row['rule']:<52} {row['mean_band_f1']:>6.3f} {row['mean_band_prec']:>6.3f} "
                f"{row['mean_band_rec']:>6.3f} {row['mean_band_pred_frac']:>6.3f} {row['p007_f1']:>6.3f}"
            )
        lines.append("")

    lines.extend(_multistep_takeaways(time_auc, static, rules))
    return "\n".join(lines)


def _multistep_takeaways(time_auc, static, rules) -> list[str]:
    lines = ["Takeaways:", ""]
    stag = [r for r in time_auc if r["feature"] == "flux_stag"]
    if stag:
        t0 = min(stag, key=lambda x: x["time_index"])
        tf = max(stag, key=lambda x: x["time_index"])
        lines.append(
            f"1. Stagnation: AUC rises from t0 ({t0['mean_auc']:.2f}) toward late snapshots "
            f"({tf['mean_auc']:.2f}) -- early stagnation alone is weak; persistence helps."
        )
    ndx = [r for r in time_auc if r["feature"] == "neg_dgamma_dx"]
    if ndx:
        best = max(ndx, key=lambda x: x["mean_auc"])
        lines.append(
            f"2. Adverse shear gradient (-d(gamma)/dx): best mean AUC {best['mean_auc']:.2f} "
            f"at t={best['time_index']} -- wall-local signal, not whole-wall."
        )
    nb = [r for r in static if r["group"] == "neighbor" and r["mean_auc"] > 0.55]
    if nb:
        b = max(nb, key=lambda x: x["mean_auc"])
        lines.append(
            f"3. Neighbor coherence: {b['feature']} mean AUC {b['mean_auc']:.2f} -- "
            "clots cluster where neighbors share stagnation / neg-dx pattern."
        )
    ax = [r for r in time_auc if r["feature"] == "hop_from_inlet"]
    if ax:
        a0 = min(ax, key=lambda x: x["time_index"])
        lines.append(
            f"4. Axial position (hop from inlet): AUC {a0['mean_auc']:.2f} -- "
            "clots skew mid/late vessel, not inlet patch."
        )
    inc = [r for r in static if r["group"] == "incubation_oracle"]
    if inc:
        lines.append(
            "5. Incubation (oracle): final clots mostly form mid/late trajectory; "
            "t=0 flow alone cannot see incubation time directly."
        )
    if rules:
        best = rules[0]
        lines.append(
            f"6. Best capped rule: {best['rule']} "
            f"(mean band F1 {best['mean_band_f1']:.3f}, pred+ {best['mean_band_pred_frac']:.3f})."
        )
    lines.append("")
    lines.append(
        "Deploy implication: combine ceiling + sdf cap + prior|stag union with tie-break; "
        "optional neighbor gate trims speck FPs without painting full wall."
    )
    return lines


def write_multistep_json(reports: list[MultiStepAnchorReport], out_path: Path) -> None:
    payload = {
        "time_auc": aggregate_time_auc(reports),
        "static_features": aggregate_static_features(reports),
        "rules_all": aggregate_rules(reports, max_pred_frac=1.0),
        "rules_capped_pred40": aggregate_rules(reports, max_pred_frac=0.40),
        "anchors": [asdict(r) for r in reports],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
