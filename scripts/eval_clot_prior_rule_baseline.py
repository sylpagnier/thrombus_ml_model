"""Eval prior rule baseline (sweep winner: p95 + flux_stag top10) on biochem anchors."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig, VesselConfig  # noqa: E402
from src.core_physics.clot_phi_simple import (  # noqa: E402
    clot_phi_thresh_si,
    predict_prior_rule_deploy,
    prior_rule_config_from_env,
)
from src.evaluation.clot_shape_score import compute_clot_shape_metrics  # noqa: E402
from src.training.train_clot_phi_simple import _clot_metrics, _list_anchor_paths  # noqa: E402
from src.utils.paths import get_project_root


def _apply_deploy_env() -> None:
    os.environ.setdefault("CLOT_FORECAST_MODE", "one_step")
    os.environ.setdefault("CLOT_FORECAST_PAIR_SCHEDULE", "static_final")
    os.environ.setdefault("CLOT_PHI_VEL_SOURCE", "gt")
    os.environ.setdefault("CLOT_PHI_FIXED_MU_FROM_PHI", "1")
    os.environ.setdefault("CLOT_PHI_HYBRID", "0")
    os.environ.setdefault("CLOT_PHI_HARD_SUPPORT_PROJECTION", "1")
    os.environ.setdefault("CLOT_PHI_SUPPORT_BAND", "ceiling_growth")
    os.environ.setdefault("CLOT_FORECAST_MASK", "ceiling_growth")
    os.environ.setdefault("CLOT_PHI_CEILING_HOPS", "2")
    os.environ.setdefault("CLOT_PHI_DGAMMA_SLICE", "1")
    os.environ.setdefault("CLOT_PHI_PRIOR_RULE_P", "0.80")
    os.environ.setdefault("CLOT_PHI_PRIOR_RULE_T0_STRIP", "0")
    os.environ.setdefault("BIOCHEM_PRIOR_COMSOL_ALIGNED", "1")
    os.environ.setdefault("BIOCHEM_PRIOR_NORM_MASK", "adjacent")


def eval_anchor(
    path: Path,
    *,
    phys: PhysicsConfig,
    bio: BiochemConfig,
    device: torch.device,
) -> dict:
    data = torch.load(path, map_location=device, weights_only=False)
    t_out = int(data.y.shape[0]) - 1
    rule = prior_rule_config_from_env()
    step, phi, mu, meta = predict_prior_rule_deploy(
        data, t_out, phys_cfg=phys, bio_cfg=bio, device=device, t_in=0, rule=rule
    )
    loss_m = step.loss_mask.reshape(-1).bool()
    band = _clot_metrics(phi, step.phi_gt, loss_m)
    y_sl = data.y[t_out].to(device=device, dtype=torch.float32)
    pred_state = y_sl.clone()
    pred_state[:, 3] = phys.viscosity_si_to_nd(mu.reshape(-1).to(device))
    shape = compute_clot_shape_metrics(
        pred_state=pred_state,
        gt_state=y_sl,
        edge_index=data.edge_index.to(device),
        phys_cfg=phys,
        gt_anchor_state=data.y[0].to(device=device, dtype=torch.float32),
    )
    thr = clot_phi_thresh_si(phys)
    return {
        "anchor": path.stem,
        "t_out": t_out,
        "rule": str(meta.get("rule", rule.describe())),
        "prior_p": meta["prior_p"],
        "prior_thr": meta["prior_thr"],
        "n_ceiling": meta["n_ceiling"],
        "n_flag": meta["n_flag"],
        "n_t0_strip": meta["n_t0_strip"],
        "n_prior_hit": meta["n_prior_hit"],
        "band_f1": band["clot_f1"],
        "band_prec": band["clot_prec"],
        "band_rec": band["clot_rec"],
        "band_pred_frac": band["pred_pos_frac"],
        "band_gt_frac": band["gt_pos_frac"],
        "clot_shape": shape["clot_shape"],
        "clot_shape_rec": shape["clot_recall"],
        "clot_mu_thresh_si": thr,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Eval prior rule clot baseline (no training)")
    p.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    p.add_argument("--anchor", default="", help="Single stem e.g. patient007")
    p.add_argument(
        "--out-json",
        default="outputs/biochem/diagnostics/clot_prior_rule_baseline.json",
    )
    args = p.parse_args()
    _apply_deploy_env()

    root = get_project_root()
    anchor_dir = Path(args.anchor_dir)
    if not anchor_dir.is_absolute():
        anchor_dir = root / anchor_dir

    device = torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")

    if args.anchor.strip():
        paths = [anchor_dir / f"{args.anchor.strip()}.pt"]
    else:
        paths = [Path(p) for p in _list_anchor_paths(anchor_dir.resolve())]

    rows = []
    for path in paths:
        if not path.is_file():
            print(f"[WARN] skip missing {path}")
            continue
        row = eval_anchor(path, phys=phys, bio=bio, device=device)
        rows.append(row)
        print(
            f"{row['anchor']:<12} flag={row['n_flag']:>4} "
            f"band F1={row['band_f1']:.3f} prec={row['band_prec']:.3f} rec={row['band_rec']:.3f} "
            f"shape={row['clot_shape']:.3f} pred+={row['band_pred_frac']:.3f} gt+={row['band_gt_frac']:.3f}"
        )

    if not rows:
        print("[ERR] no anchors evaluated")
        sys.exit(1)

    out_path = Path(args.out_json)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
