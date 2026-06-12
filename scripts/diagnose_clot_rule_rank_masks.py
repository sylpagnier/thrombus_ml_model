"""Sweep deploy-safe rank pools for prior rules: coverage + band F1."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault("CLOT_FORECAST_MODE", "one_step")
os.environ.setdefault("CLOT_FORECAST_PAIR_SCHEDULE", "static_final")
os.environ.setdefault("CLOT_PHI_VEL_SOURCE", "gt")
os.environ.setdefault("CLOT_PHI_FIXED_MU_FROM_PHI", "1")
os.environ.setdefault("CLOT_PHI_HARD_SUPPORT_PROJECTION", "1")
os.environ.setdefault("CLOT_PHI_SUPPORT_BAND", "ceiling_growth")
os.environ.setdefault("CLOT_FORECAST_MASK", "ceiling_growth")
os.environ.setdefault("CLOT_PHI_CEILING_HOPS", "2")
os.environ.setdefault("CLOT_PHI_DGAMMA_SLICE", "1")
os.environ.setdefault("CLOT_PHI_DGAMMA_REF_TIME", "0")
os.environ.setdefault("CLOT_PHI_DGAMMA_WALL_MIN_SI", "100")
os.environ.setdefault("CLOT_PHI_DGAMMA_OFFWALL_PCT", "80")
os.environ.setdefault("BIOCHEM_PRIOR_COMSOL_ALIGNED", "1")

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.clot_growth_masks import graph_dilate_hops, resolve_ceiling_mask  # noqa: E402
from src.core_physics.clot_phi_simple import (  # noqa: E402
    _anchor_flow_props,
    _hop_distance_from_seed,
    _noslip_wall_mask,
    _top_frac_mask,
    _wall_mask_from_data,
    adjacent_band_mask,
    clot_prior_score_flat,
    compute_clot_kinematics_fields,
    dgamma_dx_slice_mask,
    predict_prior_rule_deploy,
    build_clot_phi_step,
    sdf_nd_from_data,
)
from src.training.train_clot_phi_simple import _clot_metrics, _list_anchor_paths  # noqa: E402
from src.utils.paths import get_project_root


@dataclass(frozen=True)
class PoolSpec:
    name: str
    description: str
    build: Callable[..., torch.Tensor]


def _lumen_eligible(wall: torch.Tensor, sdf: torch.Tensor, center_frac: float) -> torch.Tensor:
    n = int(wall.numel())
    eligible = wall.clone()
    if center_frac <= 0.0:
        return torch.ones(n, device=wall.device, dtype=torch.bool)
    lumen = (~wall) & (sdf > 1e-8)
    if not bool(lumen.any().item()):
        return eligible
    thr = torch.quantile(sdf[lumen], 1.0 - center_frac)
    eligible |= lumen & (sdf <= thr)
    return eligible


def _build_pool_specs() -> list[PoolSpec]:
    specs: list[PoolSpec] = []

    def add(name: str, desc: str, fn: Callable) -> None:
        specs.append(PoolSpec(name=name, description=desc, build=fn))

    add("ceiling_h2", "wall + 2 hops (current default)", lambda d, dev, bio, ctx: ctx["ceiling"])

    for h in (0, 1):
        add(
            f"ceiling_h{h}",
            f"wall + {h} hop(s) only",
            lambda d, dev, bio, ctx, hops=h: resolve_ceiling_mask(
                d, dev, bio, ceiling_hops=h
            ),
        )

    for frac in (0.05, 0.10, 0.15, 0.20, 0.30):
        add(
            f"ceiling_center_excl_{int(100*frac):02d}",
            f"ceiling minus inner {int(100*frac)}% lumen SDF (centerline gap)",
            lambda d, dev, bio, ctx, f=frac: ctx["ceiling"]
            & _lumen_eligible(ctx["wall"], ctx["sdf"], f),
        )

    for sdf_cap in (0.025, 0.030, 0.035, 0.040, 0.050):
        add(
            f"ceiling_sdf_le_{sdf_cap:.3f}",
            f"ceiling with sdf_nd <= {sdf_cap}",
            lambda d, dev, bio, ctx, cap=sdf_cap: ctx["ceiling"] & (ctx["sdf"] <= cap),
        )

    add(
        "ceiling_adjacent_band",
        "ceiling intersect Gaussian adjacent band (K11-style)",
        lambda d, dev, bio, ctx: ctx["ceiling"]
        & adjacent_band_mask(ctx["sdf"], ctx["wall_raw"], peak_nd=0.008, sigma_nd=0.008),
    )

    add(
        "ceiling_hop_wall_le1",
        "ceiling with hop-from-wall <= 1",
        lambda d, dev, bio, ctx: ctx["ceiling"] & (ctx["hop_wall"] <= 1.0),
    )

    add(
        "ceiling_no_noslip",
        "ceiling excluding no-slip wall nodes (u=v=0)",
        lambda d, dev, bio, ctx: ctx["ceiling"] & ~ctx["noslip"],
    )

    add(
        "ceiling_offwall_lumen",
        "ceiling minus all on-wall nodes (off-wall lumen only)",
        lambda d, dev, bio, ctx: ctx["ceiling"] & ~ctx["wall"],
    )

    add(
        "ceiling_dgamma_deploy",
        "ceiling intersect -dgamma/dx adhesion @ t0 (no GT seeds)",
        lambda d, dev, bio, ctx: dgamma_dx_slice_mask(
            d, dev, ctx["ceiling"], ctx["empty_seed"], bio
        ),
    )

    add(
        "ceiling_center10_dgamma",
        "center-excl 10% + dgamma deploy",
        lambda d, dev, bio, ctx: dgamma_dx_slice_mask(
            d,
            dev,
            ctx["ceiling"] & _lumen_eligible(ctx["wall"], ctx["sdf"], 0.10),
            ctx["empty_seed"],
            bio,
        ),
    )

    add(
        "wall_only",
        "mask_wall only",
        lambda d, dev, bio, ctx: ctx["wall"],
    )

    add(
        "wall_plus_1hop",
        "wall + 1 graph hop",
        lambda d, dev, bio, ctx: graph_dilate_hops(
            ctx["wall"], ctx["edge_index"], 1
        ),
    )

    return specs


def _rule_phi_in_pool(data, device, bio, rank_mask: torch.Tensor) -> torch.Tensor:
    y0 = data.y[0]
    u, v = y0[:, 0], y0[:, 1]
    props = _anchor_flow_props(data, device)
    fields = compute_clot_kinematics_fields(data, u, v, bio, props)
    prior = clot_prior_score_flat(data, u, v, bio, props)
    hop = _hop_distance_from_seed(
        _wall_mask_from_data(data, device, int(data.num_nodes)),
        data.edge_index.to(device=device),
    ).float()
    dx = fields.flux_path_dx_raw.reshape(-1)
    leg_p = _top_frac_mask(prior, rank_mask, 0.20, tie_dx=dx, tie_hop=hop)
    leg_s = _top_frac_mask(fields.flux_stag.reshape(-1), rank_mask, 0.20, tie_dx=dx, tie_hop=hop)
    return torch.clamp(leg_p.float() + leg_s.float(), 0, 1)


def _eval_anchor(path: Path, pool: PoolSpec, *, phys, bio, device) -> dict | None:
    from src.core_physics.clot_phi_simple import build_clot_phi_step

    data = torch.load(path, map_location=device, weights_only=False)
    t_out = int(data.y.shape[0]) - 1
    step = build_clot_phi_step(data, t_out, phys, bio, device)

    wall = _wall_mask_from_data(data, device, int(data.num_nodes))
    y0 = data.y[0]
    u, v = y0[:, 0], y0[:, 1]
    ctx = {
        "ceiling": resolve_ceiling_mask(data, device, bio),
        "wall": wall,
        "wall_raw": data.mask_wall if hasattr(data, "mask_wall") else None,
        "sdf": sdf_nd_from_data(data, device, int(data.num_nodes)),
        "hop_wall": _hop_distance_from_seed(wall, data.edge_index.to(device=device)).float(),
        "noslip": _noslip_wall_mask(wall, u, v),
        "empty_seed": torch.zeros(int(data.num_nodes), device=device, dtype=torch.bool),
        "edge_index": data.edge_index.to(device=device),
    }
    rank_mask = pool.build(data, device, bio, ctx).reshape(-1).bool()
    if not bool(rank_mask.any().item()):
        return None

    phi = _rule_phi_in_pool(data, device, bio, rank_mask)
    loss_m = step.loss_mask.reshape(-1).bool()
    phi_gt = step.phi_gt.reshape(-1).bool()
    gt_in_loss = phi_gt & loss_m
    n_gt = int(gt_in_loss.sum())
    gt_in_pool = int((gt_in_loss & rank_mask).sum())
    band = _clot_metrics(phi, step.phi_gt, step.loss_mask)
    return {
        "anchor": path.stem,
        "pool": pool.name,
        "n_pool": int(rank_mask.sum()),
        "gt_in_loss": n_gt,
        "gt_in_pool": gt_in_pool,
        "gt_pool_frac": gt_in_pool / max(n_gt, 1),
        "n_flag": int((phi >= 0.5).sum()),
        "band_f1": float(band["clot_f1"]),
        "band_prec": float(band["clot_prec"]),
        "band_rec": float(band["clot_rec"]),
        "band_pred_frac": float(band["pred_pos_frac"]),
        "band_gt_frac": float(band["gt_pos_frac"]),
    }


def _score_row(mean: dict) -> float:
    """Prefer high F1 with decent GT pool coverage; penalize pred+ >> gt+."""
    f1 = mean["mean_band_f1"]
    cover = mean["mean_gt_pool_frac"]
    pred = mean["mean_pred_frac"]
    gt = mean["mean_gt_frac"]
    s = f1
    if cover < 0.85:
        s -= 0.08 * (0.85 - cover)
    if gt >= 0.05 and pred > 2.2 * gt:
        s -= 0.12
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose deploy rank pools for prior rules")
    ap.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument(
        "--out-json",
        default="outputs/biochem/diagnostics/clot_rule_rank_mask_diagnostic.json",
    )
    args = ap.parse_args()

    root = get_project_root()
    anchor_dir = Path(args.anchor_dir)
    if not anchor_dir.is_absolute():
        anchor_dir = root / anchor_dir
    paths = [Path(p) for p in _list_anchor_paths(anchor_dir.resolve()) if Path(p).is_file()]
    pools = _build_pool_specs()
    device = torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")

    per_anchor: list[dict] = []
    for pool in pools:
        for path in paths:
            row = _eval_anchor(path, pool, phys=phys, bio=bio, device=device)
            if row:
                per_anchor.append(row)

    by_pool: dict[str, list[dict]] = {}
    for row in per_anchor:
        by_pool.setdefault(row["pool"], []).append(row)

    summary: list[dict] = []
    for pool in pools:
        rows = by_pool.get(pool.name, [])
        if len(rows) < 2:
            continue
        n = len(rows)
        mean = {
            "pool": pool.name,
            "description": pool.description,
            "n_anchors": n,
            "mean_n_pool": sum(r["n_pool"] for r in rows) / n,
            "mean_gt_pool_frac": sum(r["gt_pool_frac"] for r in rows) / n,
            "mean_band_f1": sum(r["band_f1"] for r in rows) / n,
            "mean_band_prec": sum(r["band_prec"] for r in rows) / n,
            "mean_band_rec": sum(r["band_rec"] for r in rows) / n,
            "mean_pred_frac": sum(r["band_pred_frac"] for r in rows) / n,
            "mean_gt_frac": sum(r["band_gt_frac"] for r in rows) / n,
            "mean_n_flag": sum(r["n_flag"] for r in rows) / n,
        }
        mean["score"] = _score_row(mean)
        mean["per_anchor"] = rows
        summary.append(mean)

    summary.sort(key=lambda x: (-x["score"], -x["mean_band_f1"]))
    baseline = next((s for s in summary if s["pool"] == "ceiling_h2"), None)

    print(f"[i] {len(pools)} pools x {len(paths)} anchors (rule: p0.80|st20|tie)")
    print(f"{'rank':>4} {'score':>6} {'F1':>6} {'gt_cov':>6} {'pred+':>6} {'pool_n':>7}  pool")
    print("-" * 88)
    for i, row in enumerate(summary[: args.top], start=1):
        print(
            f"{i:>4} {row['score']:>6.3f} {row['mean_band_f1']:>6.3f} "
            f"{row['mean_gt_pool_frac']:>6.3f} {row['mean_pred_frac']:>6.3f} "
            f"{row['mean_n_pool']:>7.0f}  {row['pool']}"
        )

    if summary and baseline:
        best = summary[0]
        print()
        print(f"[OK] best: {best['pool']} -- {best['description']}")
        print(
            f"     vs ceiling_h2: F1 {baseline['mean_band_f1']:.3f}->{best['mean_band_f1']:.3f} "
            f"gt_cov {baseline['mean_gt_pool_frac']:.3f}->{best['mean_gt_pool_frac']:.3f} "
            f"pred+ {baseline['mean_pred_frac']:.3f}->{best['mean_pred_frac']:.3f}"
        )

    out_path = root / args.out_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"summary": [{k: v for k, v in s.items() if k != "per_anchor"} for s in summary], "per_anchor": per_anchor}, indent=2),
        encoding="utf-8",
    )
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
