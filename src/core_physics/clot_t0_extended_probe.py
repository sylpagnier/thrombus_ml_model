"""Extended t=0 vs t_final feature sweep: graph x, biochem BC, topology, flow derivs.

Finds deployable signals (strong @ t=0) vs oracle-only signals (strong @ t_final only).
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
from src.core_physics.clot_growth_masks import resolve_ceiling_mask
from src.core_physics.clot_kinematics_fields import (
    adjacent_band_mask,
    compute_clot_kinematics_fields,
    score_clot_risk_from_fields,
)
from src.core_physics.clot_phi_simple import clot_phi_thresh_si, gt_neg_dgamma_dx_phys, sdf_nd_from_data
from src.core_physics.clot_t0_pattern_probe import (
    _binary_auc,
    _decile_rule_metrics,
    _hop_distance_from_seed,
    _wall_mask,
    build_t0_feature_table,
)
from src.core_physics.kinematics_clot_prior import clot_prior_score_flat
from src.utils.channel_schema import BIO_X_SCHEMA, BIO_Y_SCHEMA, KINE_X_SCHEMA, X_SCHEMAS, Y_SCHEMAS
from src.utils.rheology import compute_shear_rate


def _x_kine_channel(data, name: str, device: torch.device) -> torch.Tensor | None:
    if not hasattr(data, "x") or data.x is None:
        return None
    schema = getattr(data, "x_schema", None)
    if schema != KINE_X_SCHEMA:
        return None
    try:
        idx = X_SCHEMAS[KINE_X_SCHEMA].channels.index(name)
    except ValueError:
        return None
    return data.x[:, idx].to(device=device, dtype=torch.float32)


def _x_biochem_channel(data, name: str, device: torch.device) -> torch.Tensor | None:
    if not hasattr(data, "x_biochem") or data.x_biochem is None:
        return None
    schema = getattr(data, "x_biochem_schema", None)
    if schema != BIO_X_SCHEMA:
        return None
    try:
        idx = X_SCHEMAS[BIO_X_SCHEMA].channels.index(name)
    except ValueError:
        return None
    return data.x_biochem[:, idx].to(device=device, dtype=torch.float32)


def _y_channel(y_slice: torch.Tensor, name: str) -> torch.Tensor | None:
    try:
        idx = Y_SCHEMAS[BIO_Y_SCHEMA].channels.index(name)
    except ValueError:
        return None
    return y_slice[:, idx]


def _flow_derivatives(
    data,
    u: torch.Tensor,
    v: torch.Tensor,
    props: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    u = u.reshape(-1).float()
    v = v.reshape(-1).float()
    du_dx = torch.sparse.mm(data.G_x, u.unsqueeze(1)).squeeze(1)
    du_dy = torch.sparse.mm(data.G_y, u.unsqueeze(1)).squeeze(1)
    dv_dx = torch.sparse.mm(data.G_x, v.unsqueeze(1)).squeeze(1)
    dv_dy = torch.sparse.mm(data.G_y, v.unsqueeze(1)).squeeze(1)
    gamma = compute_shear_rate(du_dx, du_dy, dv_dx, dv_dy)
    u_ref = props["u_ref"].reshape(-1).clamp(min=1e-8)
    d_bar = props["d_bar"].reshape(-1).clamp(min=1e-8)
    scale = u_ref / d_bar
    return {
        "div_uv": du_dx + dv_dy,
        "vorticity": dv_dx - du_dy,
        "gamma_dot_nd": gamma,
        "gamma_si_raw": gamma * scale,
        "du_dx": du_dx * scale,
        "du_dy": du_dy * scale,
        "dv_dx": dv_dx * scale,
        "dv_dy": dv_dy * scale,
        "speed_nd": torch.sqrt(u * u + v * v),
    }


def build_feature_table_at_time(
    data,
    time_index: int,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
) -> dict[str, tuple[torch.Tensor, str, str]]:
    """Return feature -> (values, group, source)."""
    n = int(data.num_nodes)
    ti = int(time_index)
    y = data.y[ti].to(device=device, dtype=torch.float32)
    u = y[:, 0]
    v = y[:, 1]
    props = _graph_props(data, device)
    fields = compute_clot_kinematics_fields(data, u, v, bio_cfg, props)
    prior, _, _ = score_clot_risk_from_fields(fields, bio_cfg)
    neg_dx = gt_neg_dgamma_dx_phys(data, ti, bio_cfg, device)
    sdf = sdf_nd_from_data(data, device, n)
    wall = _wall_mask(data, device, n)
    ei = data.edge_index.to(device=device)
    derivs = _flow_derivatives(data, u, v, props)
    u_ref = props["u_ref"].reshape(-1).clamp(min=1e-8)

    out: dict[str, tuple[torch.Tensor, str, str]] = {}

    def put(key: str, val: torch.Tensor, group: str, source: str) -> None:
        if val is None:
            return
        out[key] = (val.reshape(-1).float(), group, source)

    # Core kinematic / prior (time-varying)
    for key, val, group, source in (
        ("prior_score", prior, "prior", "computed"),
        ("dgamma_dx", fields.dgamma_dx_phys, "shear_grad", "computed"),
        ("dgamma_dy", fields.dgamma_dy_phys, "shear_grad", "computed"),
        ("neg_dgamma_dx", neg_dx, "shear_grad", "computed"),
        ("dshear_ds", fields.dshear_ds_phys, "shear_grad", "computed"),
        ("gamma_si", fields.gamma_si, "flow", "computed"),
        ("flux_path_dx", fields.flux_path_dx, "prior", "computed"),
        ("flux_stag", fields.flux_stag, "prior", "computed"),
        ("flux_path_stream", fields.flux_path_stream, "prior", "computed"),
        ("is_low_shear", fields.is_low_shear, "flow", "computed"),
        ("is_separation", fields.is_separation_stream, "flow", "computed"),
        ("wall_proximity", fields.wall_proximity, "geometry", "computed"),
        ("adjacent_band", fields.adjacent_band.float(), "geometry", "computed"),
    ):
        put(key, val, group, source)

    put("vel_mag_si", derivs["speed_nd"] * u_ref, "flow", "computed")
    put("p_t", y[:, 2], "flow", "y_slice")
    if y.shape[1] >= 16:
        for ch in Y_SCHEMAS[BIO_Y_SCHEMA].channels[4:]:
            put(f"y_{ch}@t", _y_channel(y, ch), "species", "y_slice")

    for key, val in derivs.items():
        put(key, val, "flow_derived", "computed")

    # Static graph / kine x
    put("sdf_nd", sdf, "geometry", "data.x")
    put("on_wall", wall.float(), "geometry", "mask_wall")
    put("hop_from_wall", _hop_distance_from_seed(wall, ei).float(), "topology", "graph")

    if hasattr(data, "mask_inlet") and data.mask_inlet is not None:
        inlet = data.mask_inlet.view(-1).to(device).bool()
        put("hop_from_inlet", _hop_distance_from_seed(inlet, ei).float(), "topology", "graph")
        put("on_inlet", inlet.float(), "topology", "mask_inlet")
    if hasattr(data, "mask_outlet") and data.mask_outlet is not None:
        outlet = data.mask_outlet.view(-1).to(device).bool()
        put("hop_from_outlet", _hop_distance_from_seed(outlet, ei).float(), "topology", "graph")
        put("on_outlet", outlet.float(), "topology", "mask_outlet")

    deg = torch.zeros(n, device=device)
    if ei.numel():
        deg.scatter_add_(0, ei[0], torch.ones(ei.shape[1], device=device))
        deg.scatter_add_(0, ei[1], torch.ones(ei.shape[1], device=device))
    put("graph_degree", deg, "topology", "edge_index")

    wnx = _x_kine_channel(data, "wall_normal_x", device)
    wny = _x_kine_channel(data, "wall_normal_y", device)
    if wnx is not None and wny is not None:
        flow_wall_align = torch.abs(u * wnx + v * wny) / derivs["speed_nd"].clamp(min=1e-8)
        put("flow_wall_alignment", flow_wall_align, "flow_derived", "computed")
        put("wall_normal_x", wnx, "geometry", "data.x")
        put("wall_normal_y", wny, "geometry", "data.x")

    u_pr = _x_kine_channel(data, "u_prior", device)
    v_pr = _x_kine_channel(data, "v_prior", device)
    if u_pr is not None and v_pr is not None:
        put("u_prior", u_pr, "kine_x", "data.x")
        put("v_prior", v_pr, "kine_x", "data.x")
        put("speed_mismatch_nd", torch.sqrt((u - u_pr) ** 2 + (v - v_pr) ** 2), "kine_x", "computed")

    for ch in X_SCHEMAS[KINE_X_SCHEMA].channels:
        val = _x_kine_channel(data, ch, device)
        if val is None:
            continue
        key = ch if ch in ("sdf_nd", "on_wall") else f"kine_x_{ch}"
        if ch == "sdf_nd":
            continue  # already set
        put(key, val, "kine_x", "data.x")

    for ch in X_SCHEMAS[BIO_X_SCHEMA].channels:
        val = _x_biochem_channel(data, ch, device)
        if val is not None:
            put(f"bio_x_{ch}", val, "bio_x", "data.x_biochem")

    # wss from bio y channel 4 is wss_nd in kine schema - bio y has u,v,p,mu then species
    # Fix wss: use kine y if 5ch else skip
    return out


@dataclass
class ExtendedFeatureRow:
    anchor: str
    feature: str
    group: str
    source: str
    auc_t0: float
    auc_tfinal: float
    delta_auc: float
    decile_rec_t0: float
    decile_rec_tfinal: float
    clot_mean_t0: float
    non_mean_t0: float
    higher_is_risk: bool


@dataclass
class ComboRuleRow:
    anchor: str
    rule: str
    f1_t0: float
    prec: float
    rec: float


@dataclass
class ExtendedProbeReport:
    anchor: str
    n_ceiling: int
    n_clot: int
    graph_attrs: dict[str, bool]
    rows: list[ExtendedFeatureRow] = field(default_factory=list)
    combos: list[ComboRuleRow] = field(default_factory=list)


def _probe_feature_row(
    anchor: str,
    feature: str,
    group: str,
    source: str,
    val_t0: torch.Tensor,
    val_tf: torch.Tensor,
    labels: torch.Tensor,
    in_mask: torch.Tensor,
) -> ExtendedFeatureRow | None:
    lm = labels[in_mask]
    if not bool(lm.any()) or int(lm.sum()) < 3:
        return None
    v0 = val_t0[in_mask]
    v1 = val_tf[in_mask]
    cm0 = float(v0[lm].mean()) if bool(lm.any()) else float("nan")
    nm0 = float(v0[~lm].mean()) if bool((~lm).any()) else float("nan")
    higher = (cm0 >= nm0) if (cm0 == cm0 and nm0 == nm0) else True
    r0 = v0 if higher else -v0
    r1 = v1 if higher else -v1
    return ExtendedFeatureRow(
        anchor=anchor,
        feature=feature,
        group=group,
        source=source,
        auc_t0=_binary_auc(r0, lm.float()),
        auc_tfinal=_binary_auc(r1, lm.float()),
        delta_auc=_binary_auc(r1, lm.float()) - _binary_auc(r0, lm.float()),
        decile_rec_t0=float(_decile_rule_metrics(r0, lm, frac=0.10)["rec"]),
        decile_rec_tfinal=float(_decile_rule_metrics(r1, lm, frac=0.10)["rec"]),
        clot_mean_t0=cm0,
        non_mean_t0=nm0,
        higher_is_risk=higher,
    )


def probe_anchor_extended(
    data,
    *,
    stem: str,
    phys_cfg: PhysicsConfig | None = None,
    bio_cfg: BiochemConfig | None = None,
    device: torch.device | None = None,
    ceiling_hops: int | None = None,
) -> ExtendedProbeReport:
    phys_cfg = phys_cfg or PhysicsConfig(phase="biochem")
    bio_cfg = bio_cfg or BiochemConfig(phase="biochem")
    device = device or torch.device("cpu")
    t0 = 0
    t_final = int(data.y.shape[0]) - 1

    ceiling = resolve_ceiling_mask(data, device, bio_cfg, ceiling_hops=ceiling_hops)
    thr = clot_phi_thresh_si(phys_cfg)
    mu_tf = phys_cfg.viscosity_nd_to_si(data.y[t_final][:, STATE_CHANNEL_MU_EFF_ND].to(device))
    labels = (mu_tf >= thr).bool() & ceiling

    feats_t0 = build_feature_table_at_time(data, t0, device=device, phys_cfg=phys_cfg, bio_cfg=bio_cfg)
    feats_tf = build_feature_table_at_time(data, t_final, device=device, phys_cfg=phys_cfg, bio_cfg=bio_cfg)
    keys = sorted(set(feats_t0.keys()) | set(feats_tf.keys()))

    rows: list[ExtendedFeatureRow] = []
    for key in keys:
        if key not in feats_t0 or key not in feats_tf:
            continue
        v0, grp, src = feats_t0[key]
        v1, _, _ = feats_tf[key]
        row = _probe_feature_row(stem, key, grp, src, v0, v1, labels, ceiling)
        if row is not None:
            rows.append(row)

    # Combo rules from best deployable features @ t0
    combos: list[ComboRuleRow] = []
    base = build_t0_feature_table(data, device=device, phys_cfg=phys_cfg, bio_cfg=bio_cfg, t0_index=0)
    lm = labels[ceiling]
    if int(lm.sum()) >= 5:
        n_m = int(ceiling.sum())

        def _eval_combo(name: str, flag_m: torch.Tensor) -> None:
            flag_m = flag_m.reshape(-1).bool()
            tp = int((flag_m & lm).sum())
            fp = int((flag_m & ~lm).sum())
            fn = int((~flag_m & lm).sum())
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = (2 * prec * rec) / max(prec + rec, 1e-6)
            combos.append(ComboRuleRow(anchor=stem, rule=name, f1_t0=f1, prec=prec, rec=rec))

        k10 = max(int(0.10 * n_m), 1)
        prior = base["prior_score"][ceiling]
        _, top_p = torch.topk(prior, min(k10, n_m))
        flag_p = torch.zeros(n_m, dtype=torch.bool)
        flag_p[top_p] = True

        if "width_d2" in base:
            w2 = base["width_d2"][ceiling]
            _, top_w2 = torch.topk(w2, min(k10, n_m))
            flag_w2 = torch.zeros(n_m, dtype=torch.bool)
            flag_w2[top_w2] = True
            _eval_combo("top10pct prior AND top10pct width_d2", flag_p & flag_w2)

        if "kine_x_shear_potential" not in base:
            sp = feats_t0.get("kine_x_shear_potential", (None, "", ""))[0]
            if sp is not None:
                sp_m = sp[ceiling]
                _, top_sp = torch.topk(sp_m, min(k10, n_m))
                flag_sp = torch.zeros(n_m, dtype=torch.bool)
                flag_sp[top_sp] = True
                _eval_combo("top10pct prior AND top10pct shear_potential", flag_p & flag_sp)

        on_wall = base["on_wall"][ceiling] > 0.5
        _eval_combo("top10pct prior AND on_wall", flag_p & on_wall)

        if "hop_from_wall" in base:
            near_wall = base["hop_from_wall"][ceiling] <= 1.0
            _eval_combo("top10pct prior AND hop_from_wall<=1", flag_p & near_wall)

    graph_attrs = {
        "x_biochem": hasattr(data, "x_biochem") and data.x_biochem is not None,
        "x_schema_kine": getattr(data, "x_schema", None) == KINE_X_SCHEMA,
        "mask_inlet": hasattr(data, "mask_inlet") and data.mask_inlet is not None,
        "mask_outlet": hasattr(data, "mask_outlet") and data.mask_outlet is not None,
        "pos": hasattr(data, "pos") and data.pos is not None,
        "t_vector": hasattr(data, "t") and data.t is not None,
    }

    return ExtendedProbeReport(
        anchor=stem,
        n_ceiling=int(ceiling.sum()),
        n_clot=int(labels.sum()),
        graph_attrs=graph_attrs,
        rows=rows,
        combos=combos,
    )


def probe_all_extended(anchor_dir: Path | None = None, *, ceiling_hops: int | None = None) -> list[ExtendedProbeReport]:
    out: list[ExtendedProbeReport] = []
    for path in discover_anchor_paths(anchor_dir):
        data = torch.load(path, map_location="cpu", weights_only=False)
        out.append(probe_anchor_extended(data, stem=path.stem, ceiling_hops=ceiling_hops))
    return out


def aggregate_extended_rows(reports: list[ExtendedProbeReport]) -> list[dict[str, Any]]:
    pool: dict[str, list[ExtendedFeatureRow]] = {}
    for rep in reports:
        if rep.n_clot < 5:
            continue
        for row in rep.rows:
            pool.setdefault(row.feature, []).append(row)
    agg: list[dict[str, Any]] = []
    for feat, rows in pool.items():
        a0 = [r.auc_t0 for r in rows if r.auc_t0 == r.auc_t0]
        a1 = [r.auc_tfinal for r in rows if r.auc_tfinal == r.auc_tfinal]
        d = [r.delta_auc for r in rows if r.delta_auc == r.delta_auc]
        if not a0:
            continue
        agg.append(
            {
                "feature": feat,
                "group": rows[0].group,
                "source": rows[0].source,
                "n_anchors": len(rows),
                "mean_auc_t0": sum(a0) / len(a0),
                "mean_auc_tfinal": sum(a1) / len(a1) if a1 else float("nan"),
                "mean_delta_auc": sum(d) / len(d) if d else float("nan"),
                "mean_decile_rec_t0": sum(r.decile_rec_t0 for r in rows) / len(rows),
            }
        )
    agg.sort(key=lambda x: (-x["mean_auc_t0"], -x["mean_decile_rec_t0"]))
    return agg


def find_oracle_only_features(agg: list[dict[str, Any]], *, t0_max: float = 0.58, tf_min: float = 0.68) -> list[dict]:
    return [
        r
        for r in agg
        if r["mean_auc_t0"] <= t0_max
        and r["mean_auc_tfinal"] >= tf_min
        and r["mean_delta_auc"] >= 0.12
    ]


def find_deployable_gaps(agg: list[dict[str, Any]], current_features: set[str], *, min_auc: float = 0.62) -> list[dict]:
    """Strong @ t0 but not in current minimal rule set."""
    aliases = {f.replace("kine_x_", "").replace("bio_x_", "").replace("y_", "") for f in current_features}
    gaps = []
    for r in agg:
        if r["mean_auc_t0"] < min_auc:
            continue
        base = r["feature"].replace("kine_x_", "").replace("bio_x_", "")
        if base in aliases or r["feature"] in current_features:
            continue
        if r["feature"] in ("prior_score", "neg_dgamma_dx", "log1p_neg_dx", "sdf_nd", "on_wall"):
            continue
        gaps.append(r)
    return sorted(gaps, key=lambda x: -x["mean_auc_t0"])


def format_extended_report(reports: list[ExtendedProbeReport]) -> str:
    agg = aggregate_extended_rows(reports)
    lines = [
        "Extended clot feature sweep (ceiling mask, GT clot @ t_final)",
        "Compare AUC @ t=0 (deployable) vs AUC @ t_final (oracle flow).",
        "",
    ]

    attrs = reports[0].graph_attrs if reports else {}
    lines.append("Graph attrs on first anchor: " + ", ".join(f"{k}={v}" for k, v in attrs.items()))
    lines.append("")

    lines.append("Top deployable @ t=0 (mean AUC inside ceiling):")
    lines.append(f"{'feature':<28} {'grp':<12} {'AUC_t0':>7} {'AUC_tf':>7} {'dAUC':>7} {'d10rec':>7}  source")
    lines.append("-" * 95)
    for r in agg[:20]:
        lines.append(
            f"{r['feature']:<28} {r['group']:<12} {r['mean_auc_t0']:>7.3f} {r['mean_auc_tfinal']:>7.3f} "
            f"{r['mean_delta_auc']:>7.3f} {r['mean_decile_rec_t0']:>6.1%}  {r['source']}"
        )
    lines.append("")

    oracle = find_oracle_only_features(agg)
    if oracle:
        lines.append("Oracle-only (weak @ t=0, strong @ t_final -- NOT deployable from t=0 flow alone):")
        for r in sorted(oracle, key=lambda x: -x["mean_delta_auc"])[:12]:
            lines.append(
                f"  {r['feature']:<28} AUC_t0={r['mean_auc_t0']:.3f} AUC_tf={r['mean_auc_tfinal']:.3f} "
                f"d={r['mean_delta_auc']:+.3f} ({r['group']})"
            )
        lines.append("")

    current = {
        "prior_score",
        "neg_dgamma_dx",
        "log1p_neg_dx",
        "sdf_nd",
        "on_wall",
        "flux_path_dx",
        "dgamma_dx",
    }
    gaps = find_deployable_gaps(agg, current)
    if gaps:
        lines.append("Missing from current rules (strong @ t=0, not in minimal set):")
        for r in gaps[:12]:
            lines.append(
                f"  {r['feature']:<28} AUC_t0={r['mean_auc_t0']:.3f} d10rec={r['mean_decile_rec_t0']:.1%} ({r['group']}/{r['source']})"
            )
        lines.append("")

    # Combo rules pooled
    combo_pool: dict[str, list[ComboRuleRow]] = {}
    for rep in reports:
        for c in rep.combos:
            combo_pool.setdefault(c.rule, []).append(c)
    if combo_pool:
        lines.append("Combo rules @ t=0 (inside ceiling):")
        lines.append(f"{'rule':<42} {'mean_F1':>8} {'prec':>7} {'rec':>7}")
        lines.append("-" * 68)
        combo_agg = [
            {
                "rule": k,
                "mean_f1": sum(x.f1_t0 for x in v) / len(v),
                "mean_prec": sum(x.prec for x in v) / len(v),
                "mean_rec": sum(x.rec for x in v) / len(v),
            }
            for k, v in combo_pool.items()
        ]
        combo_agg.sort(key=lambda x: -x["mean_f1"])
        for c in combo_agg[:8]:
            lines.append(f"{c['rule']:<42} {c['mean_f1']:>8.3f} {c['mean_prec']:>7.3f} {c['mean_rec']:>7.3f}")
        lines.append("")

    lines.extend(_extended_takeaways(agg, oracle, gaps))
    return "\n".join(lines)


def _extended_takeaways(agg, oracle, gaps) -> list[str]:
    lines = ["Takeaways:", ""]
    if agg:
        best = agg[0]
        lines.append(
            f"1. Best deployable signal remains {best['feature']} (mean AUC_t0={best['mean_auc_t0']:.2f}, "
            f"source={best['source']})."
        )
    if oracle:
        top = max(oracle, key=lambda x: x["mean_delta_auc"])
        lines.append(
            f"2. Largest oracle gap: {top['feature']} (dAUC={top['mean_delta_auc']:+.2f}) -- "
            "needs time evolution, not t=0 snapshot."
        )
    if gaps:
        g = gaps[0]
        lines.append(
            f"3. Top missing graph parameter for rules: {g['feature']} ({g['group']}, AUC_t0={g['mean_auc_t0']:.2f})."
        )
    else:
        lines.append("3. No major static graph channel beats prior+wall beyond what we already probe.")

    bio = [r for r in agg if r["group"] == "bio_x" and r["mean_auc_t0"] > 0.55]
    species = [r for r in agg if r["group"] == "species" and r["mean_auc_t0"] > 0.55]
    if not bio and not species:
        lines.append("4. bio_x BC channels and y species @ t=0 carry little clot signal (as expected pre-gelation).")
    else:
        lines.append(f"4. Some bio_x/species signal @ t=0: {[r['feature'] for r in (bio + species)[:3]]}")

    topo = [r for r in agg if r["group"] == "topology" and r["mean_auc_t0"] > 0.65]
    if topo:
        lines.append(
            f"5. Topology matters: {', '.join(r['feature'] for r in topo[:3])} -- "
            "consider hop-from-inlet/outlet in rules."
        )
    return lines


def write_extended_json(reports: list[ExtendedProbeReport], out_path: Path) -> None:
    payload = {
        "aggregated": aggregate_extended_rows(reports),
        "oracle_only": find_oracle_only_features(aggregate_extended_rows(reports)),
        "deployable_gaps": find_deployable_gaps(
            aggregate_extended_rows(reports),
            {"prior_score", "neg_dgamma_dx", "sdf_nd", "on_wall"},
        ),
        "anchors": [
            {
                "anchor": r.anchor,
                "n_ceiling": r.n_ceiling,
                "n_clot": r.n_clot,
                "graph_attrs": r.graph_attrs,
                "rows": [asdict(x) for x in r.rows],
                "combos": [asdict(x) for x in r.combos],
            }
            for r in reports
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
