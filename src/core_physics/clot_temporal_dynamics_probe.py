"""Temporal clot dynamics probe: new-clot events, autocatalysis, time/flow/geometry patterns."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_anchor_survey import _first_clot_index_per_node, _graph_props, _mu_si_trajectory
from src.core_physics.clot_growth_masks import gt_clot_mask_at_time, resolve_ceiling_mask
from src.core_physics.clot_kinematics_fields import compute_clot_kinematics_fields, score_clot_risk_from_fields
from src.core_physics.clot_phi_simple import (
    _hop_distance_from_seed,
    _wall_mask_from_data,
    clot_phi_thresh_si,
    gt_neg_dgamma_dx_phys,
    sdf_nd_from_data,
)
from src.core_physics.clot_t0_pattern_probe import _binary_auc, discover_anchor_paths
from src.core_physics.clot_t0_extended_probe import build_feature_table_at_time


def _time_frac(data, time_index: int) -> float:
    if hasattr(data, "t") and data.t is not None:
        t = data.t.to(dtype=torch.float32).reshape(-1)
        span = max(float(t[-1] - t[0]), 1e-6)
        return (float(t[int(time_index)]) - float(t[0])) / span
    t_steps = int(data.y.shape[0])
    return float(time_index) / max(t_steps - 1, 1)


def _neighbor_mean(values: torch.Tensor, edge_index: torch.Tensor, n: int) -> torch.Tensor:
    v = values.reshape(-1).float()
    if int(v.numel()) != n or edge_index.numel() == 0:
        return torch.zeros(n, device=v.device)
    ei = edge_index.to(device=v.device)
    deg = torch.zeros(n, device=v.device)
    acc = torch.zeros(n, device=v.device)
    src, dst = ei[0], ei[1]
    deg.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float32))
    deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
    acc.scatter_add_(0, src, v[dst])
    acc.scatter_add_(0, dst, v[src])
    return acc / deg.clamp(min=1.0)


@dataclass
class TemporalFeatureRow:
    feature: str
    group: str
    mean_auc: float
    mean_delta: float
    n_event_anchors: int


@dataclass
class AnchorTemporalReport:
    anchor: str
    n_times: int
    n_new_events: int
    n_new_in_ceiling: int
    mean_first_clot_frac_clot_nodes: float
    mean_lag_new_after_neighbor_clot: float
    pct_new_adjacent_to_existing: float
    feature_rows: list[dict[str, Any]] = field(default_factory=list)


def _build_event_features(
    data,
    time_index: int,
    *,
    device: torch.device,
    phys: PhysicsConfig,
    bio: BiochemConfig,
    clot_prev: torch.Tensor,
    ceiling: torch.Tensor,
) -> dict[str, torch.Tensor]:
    n = int(data.num_nodes)
    feats = build_feature_table_at_time(data, time_index, device=device, phys_cfg=phys, bio_cfg=bio)
    out: dict[str, torch.Tensor] = {}
    for key, (val, _, _) in feats.items():
        if int(val.numel()) == n:
            out[key] = val.reshape(-1).float()
    out["t_frac"] = torch.full((n,), _time_frac(data, time_index), device=device)
    out["clot_prev"] = clot_prev.float()
    out["nb_clot_prev"] = _neighbor_mean(clot_prev.float(), data.edge_index.to(device), n)
    out["nb_clot_frac_prev"] = out["nb_clot_prev"].clamp(0, 1)
    hop_exist = _hop_distance_from_seed(clot_prev, data.edge_index.to(device)).float()
    out["hop_from_existing_clot"] = hop_exist
    out["adjacent_to_clot"] = (hop_exist <= 1.0).float()
    wall = _wall_mask_from_data(data, device, n)
    out["on_wall"] = wall.float()
    out["sdf_nd"] = sdf_nd_from_data(data, device, n)
    if hasattr(data, "mask_inlet") and data.mask_inlet is not None:
        inlet = data.mask_inlet.view(-1).to(device).bool()
        if int(inlet.numel()) == n:
            out["hop_from_inlet"] = _hop_distance_from_seed(
                inlet, data.edge_index.to(device)
            ).float()
    y = data.y[int(time_index)].to(device=device, dtype=torch.float32)
    props = _graph_props(data, device)
    fields = compute_clot_kinematics_fields(data, y[:, 0], y[:, 1], bio, props)
    prior, _, _ = score_clot_risk_from_fields(fields, bio)
    out["prior_score"] = prior.reshape(-1)
    out["flux_stag"] = fields.flux_stag.reshape(-1)
    out["neg_dgamma_dx"] = gt_neg_dgamma_dx_phys(data, time_index, bio, device)
    return out


def probe_anchor_temporal_dynamics(
    data,
    *,
    stem: str,
    phys: PhysicsConfig | None = None,
    bio: BiochemConfig | None = None,
    device: torch.device | None = None,
) -> AnchorTemporalReport:
    phys = phys or PhysicsConfig(phase="biochem")
    bio = bio or BiochemConfig(phase="biochem")
    device = device or torch.device("cpu")

    n_times = int(data.y.shape[0])
    ceiling = resolve_ceiling_mask(data, device, bio)
    thr = clot_phi_thresh_si(phys)

    mu_traj, _ = _mu_si_trajectory(data, phys)
    mu_traj = mu_traj.to(device)
    first_idx = _first_clot_index_per_node(mu_traj, thr)
    clot_nodes = first_idx < n_times
    mean_first_frac = (
        float(first_idx[clot_nodes].float().mean() / max(n_times - 1, 1))
        if bool(clot_nodes.any())
        else float("nan")
    )

    pool: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    n_new = 0
    n_new_ceil = 0
    adj_hits = 0
    lag_vals: list[float] = []

    for t in range(1, n_times):
        clot_t = gt_clot_mask_at_time(data, t, phys, device)
        clot_tm1 = gt_clot_mask_at_time(data, t - 1, phys, device)
        new_ev = clot_t & ~clot_tm1
        n_new += int(new_ev.sum())
        new_ceil = new_ev & ceiling
        n_new_ceil += int(new_ceil.sum())
        if not bool(new_ceil.any()):
            continue

        feats = _build_event_features(
            data, t - 1, device=device, phys=phys, bio=bio, clot_prev=clot_tm1, ceiling=ceiling
        )
        labels = new_ceil
        negatives = ceiling & ~new_ceil
        eval_mask = labels | negatives
        if int(eval_mask.sum()) < 10:
            continue

        for key, val in feats.items():
            vm = val[eval_mask]
            lm = labels[eval_mask]
            if int(lm.sum()) < 1:
                continue
            pool.setdefault(key, []).append((vm, lm))

        adj_hits += int((new_ceil & (feats["hop_from_existing_clot"] <= 1.0)).sum())
        lag_vals.append(_time_frac(data, t))

    feature_rows: list[dict[str, Any]] = []
    for key, batches in pool.items():
        aucs: list[float] = []
        deltas: list[float] = []
        for vm, lm in batches:
            cm = float(vm[lm].mean()) if bool(lm.any()) else float("nan")
            nm = float(vm[~lm].mean()) if bool((~lm).any()) else float("nan")
            higher = cm >= nm if (cm == cm and nm == nm) else True
            risk = vm if higher else -vm
            auc = _binary_auc(risk, lm.float())
            if auc == auc:
                aucs.append(auc)
            if cm == cm and nm == nm:
                deltas.append(cm - nm)
        if not aucs:
            continue
        grp = "autocatalysis" if "clot" in key or key.startswith("nb_") else "flow"
        if key == "t_frac":
            grp = "time"
        elif key in ("sdf_nd", "on_wall", "hop_from_inlet", "hop_from_wall"):
            grp = "geometry"
        feature_rows.append(
            {
                "feature": key,
                "group": grp,
                "auc": sum(aucs) / len(aucs),
                "delta": sum(deltas) / len(deltas) if deltas else float("nan"),
            }
        )

    pct_adj = 100.0 * adj_hits / max(n_new_ceil, 1)
    return AnchorTemporalReport(
        anchor=stem,
        n_times=n_times,
        n_new_events=n_new,
        n_new_in_ceiling=n_new_ceil,
        mean_first_clot_frac_clot_nodes=mean_first_frac,
        mean_lag_new_after_neighbor_clot=sum(lag_vals) / max(len(lag_vals), 1),
        pct_new_adjacent_to_existing=pct_adj,
        feature_rows=sorted(feature_rows, key=lambda r: -r["auc"]),
    )


def probe_all_temporal(anchor_dir: Path | None = None) -> list[AnchorTemporalReport]:
    out: list[AnchorTemporalReport] = []
    for path in discover_anchor_paths(anchor_dir):
        data = torch.load(path, map_location="cpu", weights_only=False)
        out.append(probe_anchor_temporal_dynamics(data, stem=path.stem))
    return out


def aggregate_temporal_features(reports: list[AnchorTemporalReport]) -> list[TemporalFeatureRow]:
    pool: dict[str, list[dict]] = {}
    for rep in reports:
        if rep.n_new_in_ceiling < 5:
            continue
        for row in rep.feature_rows:
            pool.setdefault(row["feature"], []).append(row)
    agg: list[TemporalFeatureRow] = []
    for feat, rows in pool.items():
        deltas = [r["delta"] for r in rows if r["delta"] == r["delta"]]
        agg.append(
            TemporalFeatureRow(
                feature=feat,
                group=rows[0]["group"],
                mean_auc=sum(r["auc"] for r in rows) / len(rows),
                mean_delta=sum(deltas) / len(deltas) if deltas else float("nan"),
                n_event_anchors=len(rows),
            )
        )
    agg.sort(key=lambda x: -x.mean_auc)
    return agg


def format_temporal_probe_report(reports: list[AnchorTemporalReport]) -> str:
    lines = [
        "Temporal clot dynamics probe (NEW clot events: mu>=thr @ t and not @ t-1)",
        "Pool: deploy ceiling. Features @ t-1 flow/state.",
        "",
    ]
    lines.append(f"{'anchor':<12} {'new_ev':>7} {'new_ceil':>9} {'adj%':>6} {'1st_frac':>9}")
    lines.append("-" * 50)
    for r in reports:
        lines.append(
            f"{r.anchor:<12} {r.n_new_events:>7} {r.n_new_in_ceiling:>9} "
            f"{r.pct_new_adjacent_to_existing:>5.1f}% {r.mean_first_clot_frac_clot_nodes:>9.3f}"
        )
    lines.append("")
    agg = aggregate_temporal_features(reports)
    if agg:
        lines.append("Top predictors of NEW clot (next step, inside ceiling):")
        lines.append(f"{'feature':<24} {'AUC':>6} {'delta':>10} {'grp':<14}")
        lines.append("-" * 58)
        for row in agg[:18]:
            lines.append(
                f"{row.feature:<24} {row.mean_auc:>6.3f} {row.mean_delta:>10.3g} {row.group:<14}"
            )
        auto = [r for r in agg if r.group == "autocatalysis"]
        if auto:
            best = max(auto, key=lambda x: x.mean_auc)
            mean_adj = sum(r.pct_new_adjacent_to_existing for r in reports) / max(len(reports), 1)
            lines.append("")
            lines.append(
                f"[i] Autocatalysis: {best.feature} AUC {best.mean_auc:.2f}; "
                f"~{mean_adj:.0f}% of new ceiling events hop-adjacent to existing clot."
            )
    return "\n".join(lines)


def write_temporal_probe_json(reports: list[AnchorTemporalReport], path: Path) -> None:
    payload = {
        "aggregated": [asdict(r) for r in aggregate_temporal_features(reports)],
        "anchors": [asdict(r) for r in reports],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
