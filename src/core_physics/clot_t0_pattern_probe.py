"""Exploratory probe: t=0 kinematics vs GT clot @ t_final inside deploy ceiling mask.

Goal: find interpretable rules (shear grad, stagnation, geometry) that separate
clot from non-clot nodes **without using GT at inference** — only for offline
pattern discovery on anchor graphs.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND
from src.core_physics.clot_anchor_survey import _graph_props, discover_anchor_paths
from src.core_physics.clot_growth_masks import (
    clot_ceiling_hops,
    graph_dilate_hops,
    resolve_ceiling_mask,
    resolve_t0_dgamma_wall_mask,
)
from src.core_physics.clot_kinematics_fields import compute_clot_kinematics_fields, score_clot_risk_from_fields
from src.core_physics.clot_phi_simple import clot_phi_thresh_si, gt_neg_dgamma_dx_phys, sdf_nd_from_data
from src.core_physics.kinematics_clot_prior import clot_prior_score_flat
from src.utils.channel_schema import KINE_X_SCHEMA, X_SCHEMAS


def _wall_mask(data, device: torch.device, n: int) -> torch.Tensor:
    if hasattr(data, "mask_wall") and data.mask_wall is not None:
        return data.mask_wall.view(-1).to(device=device).bool()
    return torch.zeros(n, dtype=torch.bool, device=device)


def _hop_distance_from_seed(seed: torch.Tensor, edge_index: torch.Tensor, max_hops: int = 64) -> torch.Tensor:
    """BFS hop count from seed nodes; unreachable -> max_hops+1."""
    n = int(seed.numel())
    dist = torch.full((n,), max_hops + 1, dtype=torch.long)
    if not bool(seed.any().item()):
        return dist
    dist[seed] = 0
    active = seed.clone()
    ei = edge_index
    for h in range(max_hops):
        nxt = graph_dilate_hops(active, ei, 1) & ~active
        if not bool(nxt.any().item()):
            break
        dist[nxt] = h + 1
        active = active | nxt
    return dist


def _x_channel(data, name: str, device: torch.device) -> torch.Tensor | None:
    if not hasattr(data, "x") or data.x is None or not torch.is_tensor(data.x):
        return None
    if getattr(data, "x_schema", None) != KINE_X_SCHEMA:
        if name == "sdf_nd" and data.x.shape[1] > 2:
            return data.x[:, 2].to(device=device, dtype=torch.float32)
        return None
    try:
        idx = X_SCHEMAS[KINE_X_SCHEMA].channels.index(name)
    except ValueError:
        return None
    return data.x[:, idx].to(device=device, dtype=torch.float32)


@dataclass(frozen=True)
class FeatureSpec:
    key: str
    label: str
    higher_is_risk: bool
    group: str


FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec("neg_dgamma_dx", "-d(gamma)/dx @ t0 [1/s/m]", True, "shear_grad"),
    FeatureSpec("dgamma_dx", "d(gamma)/dx @ t0", False, "shear_grad"),
    FeatureSpec("dgamma_dy", "d(gamma)/dy @ t0", False, "shear_grad"),
    FeatureSpec("dshear_ds", "streamwise d(gamma)/ds", False, "shear_grad"),
    FeatureSpec("gamma_si", "shear rate gamma [1/s]", False, "flow"),
    FeatureSpec("vel_mag_si", "speed |u,v| [m/s]", False, "flow"),
    FeatureSpec("flux_path_dx", "adverse dx flux (prior)", True, "prior"),
    FeatureSpec("flux_stag", "stagnation flux", True, "prior"),
    FeatureSpec("flux_path_stream", "stream separation flux", True, "prior"),
    FeatureSpec("prior_score", "comsol_hybrid prior", True, "prior"),
    FeatureSpec("is_low_shear", "low-shear sigmoid", True, "flow"),
    FeatureSpec("is_separation", "stream separation sigmoid", True, "flow"),
    FeatureSpec("sdf_nd", "wall distance sdf", False, "geometry"),
    FeatureSpec("wall_proximity", "exp(-sdf/lambda)", True, "geometry"),
    FeatureSpec("on_wall", "mask_wall", True, "geometry"),
    FeatureSpec("hop_from_wall", "graph hops from wall", False, "geometry"),
    FeatureSpec("hop_from_t0_dgamma", "hops from t0 dgamma strip", False, "geometry"),
    FeatureSpec("width_nd", "hydraulic width", False, "geometry"),
    FeatureSpec("width_d1", "d(width)/ds", False, "geometry"),
    FeatureSpec("width_d2", "d2(width)/ds2 (curvature proxy)", True, "geometry"),
    FeatureSpec("wss_prior_nd", "WSS prior", False, "flow"),
    FeatureSpec("log10_gamma", "log10(gamma)", False, "flow"),
    FeatureSpec("log1p_neg_dx", "log1p(-d(gamma)/dx)+", True, "shear_grad"),
    FeatureSpec("fi_t0", "FI species @ t0", True, "species"),
    FeatureSpec("mat_t0", "Mat species @ t0", True, "species"),
)


def _risk_score(raw: torch.Tensor, spec: FeatureSpec) -> torch.Tensor:
    v = raw.reshape(-1).float()
    if spec.higher_is_risk:
        return v
    return -v


def _binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Rank AUC; scores higher => positive."""
    s = scores.detach().cpu().float().reshape(-1)
    y = labels.detach().cpu().float().reshape(-1)
    n_pos = int(y.sum())
    n_neg = int((1.0 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(s, descending=False)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, len(s) + 1, dtype=torch.float32)
    sum_pos_ranks = float(ranks[y > 0.5].sum())
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _decile_rule_metrics(
    risk: torch.Tensor,
    labels: torch.Tensor,
    *,
    frac: float = 0.10,
) -> dict[str, float]:
    """Flag top ``frac`` risk nodes; report precision/recall vs GT clot in mask."""
    n = int(labels.numel())
    n_pos = int(labels.sum())
    if n_pos == 0:
        return {"prec": float("nan"), "rec": float("nan"), "n_flag": 0.0}
    k = max(int(math.ceil(frac * n)), 1)
    k = min(k, n)
    _, idx = torch.topk(risk, k)
    pred = torch.zeros(n, dtype=torch.bool)
    pred[idx] = True
    tp = int((pred & labels).sum())
    return {
        "prec": tp / max(int(pred.sum()), 1),
        "rec": tp / max(n_pos, 1),
        "n_flag": float(k),
    }


@dataclass
class FeatureProbeRow:
    anchor: str
    feature: str
    label: str
    group: str
    n_mask: int
    n_clot: int
    clot_frac: float
    clot_mean: float
    non_mean: float
    delta_mean: float
    auc: float
    decile_prec: float
    decile_rec: float


@dataclass
class RuleProbeRow:
    anchor: str
    rule: str
    n_flag: int
    prec: float
    rec: float
    f1: float


@dataclass
class AnchorPatternReport:
    anchor: str
    n_nodes: int
    n_ceiling: int
    n_clot_ceiling: int
    clot_recall_in_ceiling: float
    t_final_s: float
    feature_rows: list[FeatureProbeRow] = field(default_factory=list)
    rule_rows: list[RuleProbeRow] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def build_t0_feature_table(
    data,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    t0_index: int = 0,
) -> dict[str, torch.Tensor]:
    """All probe features at t=0 (deploy-visible inputs)."""
    n = int(data.num_nodes)
    y0 = data.y[int(t0_index)].to(device=device, dtype=torch.float32)
    u = y0[:, 0]
    v = y0[:, 1]
    props = _graph_props(data, device)
    fields = compute_clot_kinematics_fields(data, u, v, bio_cfg, props)
    prior, _, _ = score_clot_risk_from_fields(fields, bio_cfg)
    neg_dx = gt_neg_dgamma_dx_phys(data, int(t0_index), bio_cfg, device)
    sdf = sdf_nd_from_data(data, device, n)
    wall = _wall_mask(data, device, n)
    ei = data.edge_index.to(device=device)
    hop_wall = _hop_distance_from_seed(wall, ei).float()
    t0_strip = resolve_t0_dgamma_wall_mask(data, device, bio_cfg)
    hop_t0 = _hop_distance_from_seed(t0_strip, ei).float()
    vel_mag_si = torch.sqrt(u * u + v * v) * props["u_ref"].reshape(-1).clamp(min=1e-8)

    out: dict[str, torch.Tensor] = {
        "neg_dgamma_dx": neg_dx,
        "dgamma_dx": fields.dgamma_dx_phys,
        "dgamma_dy": fields.dgamma_dy_phys,
        "dshear_ds": fields.dshear_ds_phys,
        "gamma_si": fields.gamma_si,
        "vel_mag_si": vel_mag_si,
        "flux_path_dx": fields.flux_path_dx,
        "flux_stag": fields.flux_stag,
        "flux_path_stream": fields.flux_path_stream,
        "prior_score": prior,
        "is_low_shear": fields.is_low_shear,
        "is_separation": fields.is_separation_stream,
        "sdf_nd": sdf,
        "wall_proximity": fields.wall_proximity,
        "on_wall": wall.float(),
        "hop_from_wall": hop_wall,
        "hop_from_t0_dgamma": hop_t0,
        "log10_gamma": torch.log10(fields.gamma_si.clamp(min=1e-6)),
        "log1p_neg_dx": torch.log1p(neg_dx.clamp(min=0.0)),
        "fi_t0": y0[:, 12],
        "mat_t0": y0[:, 15],
    }
    for ch in ("width_nd", "width_d1", "width_d2", "wss_prior_nd"):
        xch = _x_channel(data, ch, device)
        if xch is not None:
            out[ch] = xch
    return out


def probe_anchor_patterns(
    data,
    *,
    stem: str,
    phys_cfg: PhysicsConfig | None = None,
    bio_cfg: BiochemConfig | None = None,
    device: torch.device | None = None,
    ceiling_hops: int | None = None,
    t0_index: int = 0,
) -> AnchorPatternReport:
    phys_cfg = phys_cfg or PhysicsConfig(phase="biochem")
    bio_cfg = bio_cfg or BiochemConfig(phase="biochem")
    device = device or torch.device("cpu")

    n = int(data.num_nodes)
    t_final = int(data.y.shape[0]) - 1
    t_final_s = float(data.t[t_final].item()) if hasattr(data, "t") and data.t is not None else float(t_final)

    ceiling = resolve_ceiling_mask(data, device, bio_cfg, ceiling_hops=ceiling_hops)
    thr = clot_phi_thresh_si(phys_cfg)
    mu_tf = phys_cfg.viscosity_nd_to_si(data.y[t_final][:, STATE_CHANNEL_MU_EFF_ND].to(device))
    clot = (mu_tf >= thr).bool()
    in_mask = ceiling
    labels = clot & in_mask

    n_mask = int(in_mask.sum())
    n_clot = int(labels.sum())
    recall = float(n_clot / max(int(clot.sum()), 1))

    feats = build_t0_feature_table(data, device=device, phys_cfg=phys_cfg, bio_cfg=bio_cfg, t0_index=t0_index)
    spec_by_key = {s.key: s for s in FEATURE_SPECS}

    feature_rows: list[FeatureProbeRow] = []
    for key, tensor in feats.items():
        spec = spec_by_key.get(key)
        if spec is None:
            continue
        m = in_mask
        if not bool(m.any().item()) or n_clot == 0:
            continue
        v = tensor.reshape(-1)
        risk = _risk_score(v, spec)
        vm = v[m]
        lm = labels[m]
        clot_mean = float(vm[lm].mean()) if bool(lm.any()) else float("nan")
        non_mean = float(vm[~lm].mean()) if bool((~lm).any()) else float("nan")
        dec = _decile_rule_metrics(risk[m], lm, frac=0.10)
        feature_rows.append(
            FeatureProbeRow(
                anchor=stem,
                feature=key,
                label=spec.label,
                group=spec.group,
                n_mask=n_mask,
                n_clot=n_clot,
                clot_frac=n_clot / max(n_mask, 1),
                clot_mean=clot_mean,
                non_mean=non_mean,
                delta_mean=clot_mean - non_mean,
                auc=_binary_auc(risk[m], lm.float()),
                decile_prec=float(dec["prec"]),
                decile_rec=float(dec["rec"]),
            )
        )

    rule_rows: list[RuleProbeRow] = []
    if n_clot > 0 and n_mask > 0:
        m = in_mask
        lm = labels[m]
        risk_dx = _risk_score(feats["neg_dgamma_dx"], spec_by_key["neg_dgamma_dx"])[m]
        risk_prior = _risk_score(feats["prior_score"], spec_by_key["prior_score"])[m]
        risk_stag = _risk_score(feats["flux_stag"], spec_by_key["flux_stag"])[m]
        on_wall_m = feats["on_wall"][m] > 0.5
        t0_m = t0_strip[m] if (t0_strip := resolve_t0_dgamma_wall_mask(data, device, bio_cfg)) is not None else torch.zeros_like(lm)

        def _eval_rule(name: str, flag: torch.Tensor) -> None:
            flag = flag.reshape(-1).bool()
            tp = int((flag & lm).sum())
            fp = int((flag & ~lm).sum())
            fn = int((~flag & lm).sum())
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = (2 * prec * rec) / max(prec + rec, 1e-6)
            rule_rows.append(
                RuleProbeRow(
                    anchor=stem,
                    rule=name,
                    n_flag=int(flag.sum()),
                    prec=prec,
                    rec=rec,
                    f1=f1,
                )
            )

        k10 = max(int(0.10 * n_mask), 1)
        _, top_dx = torch.topk(risk_dx, min(k10, int(risk_dx.numel())))
        flag_dx = torch.zeros(n_mask, dtype=torch.bool)
        flag_dx[top_dx] = True
        _eval_rule("top10pct neg_dgamma_dx", flag_dx)

        _, top_pr = torch.topk(risk_prior, min(k10, int(risk_prior.numel())))
        flag_pr = torch.zeros(n_mask, dtype=torch.bool)
        flag_pr[top_pr] = True
        _eval_rule("top10pct prior_score", flag_pr)

        _, top_st = torch.topk(risk_stag, min(k10, int(risk_stag.numel())))
        flag_st = torch.zeros(n_mask, dtype=torch.bool)
        flag_st[top_st] = True
        _eval_rule("top10pct flux_stag", flag_st)

        _eval_rule("t0_dgamma_strip", t0_m)

        # wall + bottom 20% dgamma_dx (most negative dx in mask)
        k20 = max(int(0.20 * n_mask), 1)
        _, top20_dx = torch.topk(risk_dx, min(k20, int(risk_dx.numel())))
        flag_wdx = torch.zeros(n_mask, dtype=torch.bool)
        flag_wdx[top20_dx] = True
        flag_wdx = flag_wdx & on_wall_m
        _eval_rule("on_wall & top20pct neg_dgamma_dx", flag_wdx)

        # union: t0 strip OR top15% prior
        k15 = max(int(0.15 * n_mask), 1)
        _, top15_pr = torch.topk(risk_prior, min(k15, int(risk_prior.numel())))
        flag_u = t0_m.clone()
        flag_u[top15_pr] = True
        _eval_rule("t0_strip OR top15pct prior", flag_u)

    notes: list[str] = []
    if recall < 0.9:
        notes.append(f"ceiling misses {100*(1-recall):.1f}% of GT clot")
    if n_clot == 0:
        notes.append("no GT clot in ceiling at t_final")

    return AnchorPatternReport(
        anchor=stem,
        n_nodes=n,
        n_ceiling=n_mask,
        n_clot_ceiling=n_clot,
        clot_recall_in_ceiling=recall,
        t_final_s=t_final_s,
        feature_rows=feature_rows,
        rule_rows=rule_rows,
        notes=notes,
    )


def probe_all_anchors(anchor_dir: Path | None = None, *, ceiling_hops: int | None = None) -> list[AnchorPatternReport]:
    reports: list[AnchorPatternReport] = []
    for path in discover_anchor_paths(anchor_dir):
        data = torch.load(path, map_location="cpu", weights_only=False)
        reports.append(
            probe_anchor_patterns(data, stem=path.stem, ceiling_hops=ceiling_hops)
        )
    return reports


def aggregate_feature_rankings(reports: list[AnchorPatternReport]) -> list[dict[str, Any]]:
    """Mean AUC and decile recall per feature across anchors with clots."""
    pool: dict[str, list[FeatureProbeRow]] = {}
    for rep in reports:
        if rep.n_clot_ceiling < 5:
            continue
        for row in rep.feature_rows:
            pool.setdefault(row.feature, []).append(row)
    out: list[dict[str, Any]] = []
    for key, rows in pool.items():
        aucs = [r.auc for r in rows if r.auc == r.auc]
        recs = [r.decile_rec for r in rows if r.decile_rec == r.decile_rec]
        deltas = [r.delta_mean for r in rows if r.delta_mean == r.delta_mean]
        if not aucs:
            continue
        spec = next(s for s in FEATURE_SPECS if s.key == key)
        out.append(
            {
                "feature": key,
                "label": spec.label,
                "group": spec.group,
                "higher_is_risk": spec.higher_is_risk,
                "n_anchors": len(rows),
                "mean_auc": sum(aucs) / len(aucs),
                "mean_decile_rec": sum(recs) / len(recs) if recs else float("nan"),
                "mean_delta": sum(deltas) / len(deltas) if deltas else float("nan"),
            }
        )
    out.sort(key=lambda d: (-d["mean_auc"], -d["mean_decile_rec"]))
    return out


def aggregate_rule_rankings(reports: list[AnchorPatternReport]) -> list[dict[str, Any]]:
    pool: dict[str, list[RuleProbeRow]] = {}
    for rep in reports:
        if rep.n_clot_ceiling < 5:
            continue
        for row in rep.rule_rows:
            pool.setdefault(row.rule, []).append(row)
    out: list[dict[str, Any]] = []
    for rule, rows in pool.items():
        out.append(
            {
                "rule": rule,
                "n_anchors": len(rows),
                "mean_f1": sum(r.f1 for r in rows) / len(rows),
                "mean_prec": sum(r.prec for r in rows) / len(rows),
                "mean_rec": sum(r.rec for r in rows) / len(rows),
            }
        )
    out.sort(key=lambda d: -d["mean_f1"])
    return out


def format_pattern_report(reports: list[AnchorPatternReport]) -> str:
    hops = clot_ceiling_hops()
    lines = [
        f"Clot t=0 -> t_final pattern probe (ceiling = wall + {hops} hops)",
        "Labels: GT mu >= thresh @ t_final, evaluated inside ceiling only.",
        "Features: all computed from graph @ t=0 (flow u,v + geometry); no GT mu leak.",
        "",
    ]
    lines.append(f"{'anchor':<12} {'ceil':>6} {'clot':>5} {'recall':>7} {'t_final_s':>10}")
    lines.append("-" * 48)
    for r in reports:
        lines.append(
            f"{r.anchor:<12} {r.n_ceiling:>6} {r.n_clot_ceiling:>5} {r.clot_recall_in_ceiling:>6.1%} {r.t_final_s:>10.1f}"
        )
    lines.append("")

    agg_feat = aggregate_feature_rankings(reports)
    if agg_feat:
        lines.append("Top separability @ t=0 (inside ceiling, pooled anchors with >=5 clot nodes):")
        lines.append(f"{'feature':<18} {'group':<10} {'AUC':>6} {'d10_rec':>8} {'delta':>10}  label")
        lines.append("-" * 90)
        for row in agg_feat[:12]:
            lines.append(
                f"{row['feature']:<18} {row['group']:<10} {row['mean_auc']:>6.3f} "
                f"{row['mean_decile_rec']:>7.1%} {row['mean_delta']:>10.3g}  {row['label']}"
            )
        lines.append("")

    agg_rules = aggregate_rule_rankings(reports)
    if agg_rules:
        lines.append("Simple deploy-style rules (no ML, top-k inside ceiling):")
        lines.append(f"{'rule':<36} {'F1':>6} {'prec':>6} {'rec':>6}")
        lines.append("-" * 58)
        for row in agg_rules:
            lines.append(
                f"{row['rule']:<36} {row['mean_f1']:>6.3f} {row['mean_prec']:>6.3f} {row['mean_rec']:>6.3f}"
            )
        lines.append("")

    lines.extend(_narrative_takeaways(reports, agg_feat, agg_rules))
    return "\n".join(lines)


def _narrative_takeaways(
    reports: list[AnchorPatternReport],
    agg_feat: list[dict[str, Any]],
    agg_rules: list[dict[str, Any]],
) -> list[str]:
    lines = ["Intuitive takeaways:", ""]
    active = [r for r in reports if r.n_clot_ceiling >= 5]
    if not active:
        lines.append("[i] No anchors with enough clot nodes in ceiling.")
        return lines

    mean_recall = sum(r.clot_recall_in_ceiling for r in active) / len(active)
    lines.append(
        f"1. Geometry envelope: wall+{clot_ceiling_hops()} hops captures "
        f"{100*mean_recall:.0f}% of final clots on average -- clots stay wall-local."
    )

    if agg_feat:
        best = agg_feat[0]
        lines.append(
            f"2. Strongest single cue @ t=0: {best['label']} "
            f"(mean AUC {best['mean_auc']:.2f}, top-10% rule recalls {100*best['mean_decile_rec']:.0f}% of clots)."
        )
        shear = [r for r in agg_feat if r["group"] == "shear_grad" and r["mean_auc"] == r["mean_auc"]]
        if shear:
            s = max(shear, key=lambda r: r["mean_auc"])
            direction = "higher" if s["higher_is_risk"] else "lower"
            lines.append(
                f"3. Shear-gradient @ t=0: {s['label']} -- clot nodes tend toward {direction} values "
                f"(mean AUC {s['mean_auc']:.2f}, delta {s['mean_delta']:.3g}). "
                "Signal is weaker than wall/prior geometry at t=0."
            )
        prior = next((r for r in agg_feat if r["feature"] == "prior_score"), None)
        if prior:
            lines.append(
                f"4. Combined prior (separation + adverse dx + stagnation) AUC {prior['mean_auc']:.2f} — "
                "usable as a hand-crafted risk map without GT."
            )

    if agg_rules:
        best_rule = agg_rules[0]
        lines.append(
            f"5. Best simple rule: \"{best_rule['rule']}\" "
            f"(mean F1 {best_rule['mean_f1']:.2f}, prec {best_rule['mean_prec']:.2f}, rec {best_rule['mean_rec']:.2f})."
        )

    lines.append("")
    lines.append(
        "Deploy implication: a point-wise MLP on 3 minimal features alone is under-specified; "
        "a t=0 risk map from adverse d(gamma)/dx + wall proximity (+ optional stagnation) "
        "inside the ceiling mask is the simplest first rung."
    )
    return lines


def write_probe_json(reports: list[AnchorPatternReport], out_path: Path) -> None:
    payload = {
        "ceiling_hops": clot_ceiling_hops(),
        "anchors": [asdict(r) for r in reports],
        "feature_rankings": aggregate_feature_rankings(reports),
        "rule_rankings": aggregate_rule_rankings(reports),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
