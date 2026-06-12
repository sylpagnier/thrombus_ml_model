"""Diagnose localized rule: arc skip, top vs lower wall preds, risk components."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_growth_masks import resolve_ceiling_mask
from src.core_physics.clot_localized_spatial import (
    LocalizedSpatialConfig,
    apply_skip_early_wall_arc,
    build_eligible_pool,
    segment_topk_mask,
    wall_arc_fraction,
    wall_half_masks,
)
from src.core_physics.clot_phi_simple import prior_rule_config_from_env
from src.core_physics.clot_temporal_growth_rules import (
    TemporalGrowthRuleConfig,
    _resolve_pool_risk,
    compute_spatial_risk_score,
    predict_phi_temporal_at_time,
    temporal_rule_config_from_env,
)
from src.core_physics.clot_t0_pattern_probe import build_t0_feature_table
from src.utils.channel_schema import infer_missing_schema


def main() -> None:
    data = torch.load(REPO / "data/processed/graphs_biochem_anchors/patient007.pt", map_location="cpu", weights_only=False)
    data = infer_missing_schema(data, phase_hint="biochem")
    device = torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    spatial = prior_rule_config_from_env()
    loc = LocalizedSpatialConfig(mode="wall_half", segment_top_frac=0.25, skip_wall_arc_frac=0.15)
    cfg = TemporalGrowthRuleConfig(
        name="loc_prog_half_top25_skip15",
        kind="progressive_topk",
        spatial_rule=spatial,
        localized=loc,
        start_frac=0.05,
        end_frac=0.22,
        power=1.5,
    )
    ceiling = resolve_ceiling_mask(data, device, bio)
    lower, upper = wall_half_masks(data, device)
    pos = data.x[:, :2]
    feats = build_t0_feature_table(data, device=device, phys_cfg=phys, bio_cfg=bio)

    pool_raw = ceiling.bool()
    pool_loc = build_eligible_pool(data, device, ceiling, spatial, loc)
    pool_skip = apply_skip_early_wall_arc(pool_raw, data, device, 0.15)

    print("=== pool sizes ===")
    print(f"ceiling={int(pool_raw.sum())} localized_eligible={int(pool_loc.sum())}")
    print(f"  lower ceil={int((pool_raw & lower).sum())} lower eligible={int((pool_loc & lower).sum())}")
    print(f"  upper ceil={int((pool_raw & upper).sum())} upper eligible={int((pool_loc & upper).sum())}")

    for half_name, half in [("lower", lower), ("upper", upper)]:
        arc = wall_arc_fraction(data, device, half)
        seg = pool_loc & half
        if not bool(seg.any()):
            continue
        a = arc[seg]
        print(f"\n=== {half_name} wall arc (eligible pool) ===")
        print(f"  arc min={float(a.min()):.3f} p10={float(torch.quantile(a,0.1)):.3f} "
              f"med={float(a.median()):.3f} max={float(a.max()):.3f}")
        early = seg & (arc < 0.15)
        print(f"  nodes with arc<0.15 still in pool: {int(early.sum())} (should be 0)")

    pool, risk = _resolve_pool_risk(data, device=device, bio_cfg=bio, ceiling=ceiling, cfg=cfg, t_out=0)
    static_support = segment_topk_mask(risk, data, device, pool, loc)
    print(f"\n=== segment top25 support (t=0 risk) ===")
    print(f"  total flags={int(static_support.sum())} lower={int((static_support & lower).sum())} "
          f"upper={int((static_support & upper).sum())}")

    for t_out in [0, 7, 37, 53]:
        pool_t, risk_t = _resolve_pool_risk(data, device=device, bio_cfg=bio, ceiling=ceiling, cfg=cfg, t_out=t_out)
        phi = predict_phi_temporal_at_time(
            data, t_out, device=device, bio_cfg=bio, cfg=cfg, ceiling=ceiling, risk=risk_t,
            phi_prev=None, t_final=int(data.y.shape[0]) - 1,
        )
        flag = phi > 0.5
        print(f"\n=== t_out={t_out} phi flags={int(flag.sum())} ===")
        print(f"  lower={int((flag & lower).sum())} upper={int((flag & upper).sum())}")
        for half_name, half in [("lower", lower), ("upper", upper)]:
            fh = flag & half
            if not bool(fh.any()):
                continue
            arc = wall_arc_fraction(data, device, half)
            af = arc[fh]
            px = pos[fh, 0]
            print(f"  {half_name} arc: min={float(af.min()):.3f} med={float(af.median()):.3f} max={float(af.max()):.3f}")
            print(f"  {half_name} x_nd: min={float(px.min()):.3f} med={float(px.median()):.3f} max={float(px.max()):.3f}")
            viol = fh & (arc < 0.15)
            print(f"  {half_name} arc<0.15 violations: {int(viol.sum())}")

    print("\n=== risk components @ t0 on wall (mean) ===")
    risk_full = compute_spatial_risk_score(data, device=device, bio_cfg=bio, t_in=0, ceiling=ceiling, spatial_rule=spatial)
    for half_name, half in [("lower", lower), ("upper", upper)]:
        w = half & pool_loc
        if not bool(w.any()):
            continue
        print(f"  {half_name}: risk={float(risk_full[w].mean()):.4f} prior={float(feats['prior_score'][w].mean()):.4f} "
              f"stag={float(feats['flux_stag'][w].mean()):.4f} neg_dx={float(feats['neg_dgamma_dx'][w].mean()):.4f}")

    print("\n=== flagged @ t37 vs wall means ===")
    pool_t, risk_t = _resolve_pool_risk(data, device=device, bio_cfg=bio, ceiling=ceiling, cfg=cfg, t_out=37)
    phi = predict_phi_temporal_at_time(
        data, 37, device=device, bio_cfg=bio, cfg=cfg, ceiling=ceiling, risk=risk_t,
        phi_prev=None, t_final=int(data.y.shape[0]) - 1,
    )
    flag = phi > 0.5
    for half_name, half in [("lower", lower), ("upper", upper)]:
        w = half & pool_loc
        f = flag & half
        print(f"  {half_name} eligible={int(w.sum())} flagged={int(f.sum())} "
              f"neg_dx elig={float(feats['neg_dgamma_dx'][w].mean()):.2f} flagged={float(feats['neg_dgamma_dx'][f].mean()):.2f}")


if __name__ == "__main__":
    main()
