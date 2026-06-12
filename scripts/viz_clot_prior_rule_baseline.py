"""Viz prior rule baseline (no checkpoint): phi inside ceiling, sweep-winner default."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND, VesselConfig  # noqa: E402
from src.core_physics.clot_growth_masks import (  # noqa: E402
    resolve_ceiling_mask,
    resolve_growth_support_at_time,
    resolve_t0_dgamma_wall_mask,
)
from src.core_physics.clot_phi_simple import (  # noqa: E402
    clot_phi_thresh_si,
    predict_prior_rule_deploy,
    prior_rule_config_from_env,
)
from src.evaluation.clot_shape_score import compute_clot_shape_metrics  # noqa: E402
from src.evaluation.viz_clot_phi_simple import (  # noqa: E402
    _clot_binary,
    _scatter_fullmesh,
    _scatter_fullmesh_region,
    _write_mask_overlay_figure,
)
from src.training.train_clot_phi_simple import _clot_metrics  # noqa: E402
from src.utils.channel_schema import BIO_Y_SCHEMA, assert_graph_schema, infer_missing_schema  # noqa: E402
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Viz prior rule clot baseline (no training)")
    parser.add_argument("--anchor", default="patient007")
    parser.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    parser.add_argument("--time-index", type=int, default=-1)
    parser.add_argument("--scatter-size", type=float, default=6.0)
    parser.add_argument("--out", default="")
    parser.add_argument("--mask-overlay-out", default="")
    args = parser.parse_args()
    _apply_deploy_env()

    root = get_project_root()
    device = torch.device("cpu")
    phys_cfg = PhysicsConfig(phase="biochem")
    bio_cfg = BiochemConfig(phase="biochem")
    rule = prior_rule_config_from_env()

    anchor_dir = Path(args.anchor_dir)
    if not anchor_dir.is_absolute():
        anchor_dir = root / anchor_dir
    graph_path = anchor_dir / f"{args.anchor}.pt"
    if not graph_path.is_file():
        raise FileNotFoundError(graph_path)

    data = torch.load(graph_path, map_location=device, weights_only=False)
    data = infer_missing_schema(data, phase_hint="biochem")
    assert_graph_schema(data, expected_y_schema=(BIO_Y_SCHEMA,))

    ti = args.time_index if args.time_index >= 0 else int(data.y.shape[0]) - 1
    step, phi_pred, mu_pred, meta = predict_prior_rule_deploy(
        data, ti, phys_cfg=phys_cfg, bio_cfg=bio_cfg, device=device, t_in=0, rule=rule
    )

    pos = data.x[:, :2].detach().cpu().numpy()
    loss_m = step.loss_mask.detach().cpu().numpy().astype(bool)
    ceiling_m = resolve_ceiling_mask(data, device, bio_cfg).detach().cpu().numpy().astype(bool)
    phi_gt = step.phi_gt.detach().cpu().numpy()
    phi_pr = phi_pred.detach().cpu().numpy()
    mu_gt = step.mu_gt_cap.detach().cpu().numpy()
    mu_pr = mu_pred.detach().cpu().numpy()
    thresh = clot_phi_thresh_si(phys_cfg)

    band = _clot_metrics(phi_pred, step.phi_gt, step.loss_mask)
    y_sl = data.y[ti].to(device=device, dtype=torch.float32)
    pred_state = y_sl.clone()
    pred_state[:, STATE_CHANNEL_MU_EFF_ND] = phys_cfg.viscosity_si_to_nd(
        torch.tensor(mu_pr, device=device, dtype=torch.float32)
    )
    shape_m = compute_clot_shape_metrics(
        pred_state=pred_state,
        gt_state=y_sl,
        edge_index=data.edge_index.to(device),
        phys_cfg=phys_cfg,
    )

    rule_label = str(meta.get("rule", rule.describe()))
    print(
        f"[i]  rule={rule_label} thr={meta['prior_thr']:.4f} "
        f"flag={meta['n_flag']} ceiling={meta['n_ceiling']} "
        f"| band F1={band['clot_f1']:.3f} prec={band['clot_prec']:.3f} rec={band['clot_rec']:.3f} "
        f"pred+={band['pred_pos_frac']:.3f} gt+={band['gt_pos_frac']:.3f} | "
        f"clot_shape={shape_m['clot_shape']:.3f} rec={shape_m['clot_recall']:.3f}",
        flush=True,
    )

    t_label = "tfinal" if ti == int(data.y.shape[0]) - 1 else f"t{ti}"
    out_default = root / f"outputs/biochem/viz/clot_deploy/prior_rule_{args.anchor}_{t_label}_fullmesh.png"
    out_path = Path(args.out) if args.out else out_default
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gt_bin = _clot_binary(mu_gt, thresh)
    phi_gt_band = phi_gt
    phi_pr_band = phi_pr
    dot = float(args.scatter_size)
    fig = plt.figure(figsize=(12, 10))
    axes = fig.subplots(2, 2)
    _scatter_fullmesh_region(
        axes[0, 0],
        pos,
        phi_gt_band,
        loss_m,
        f"GT phi (band t={ti})",
        cmap="bwr",
        vmin=0,
        vmax=1,
        s=dot,
    )
    _scatter_fullmesh_region(
        axes[0, 1],
        pos,
        phi_pr,
        ceiling_m,
        f"Rule phi ({rule_label})",
        cmap="bwr",
        vmin=0,
        vmax=1,
        s=dot,
    )
    cap = float(os.environ.get("CLOT_PHI_MU_CAP_SI", "0.10"))
    _scatter_fullmesh(
        axes[1, 0],
        pos,
        np.clip(mu_gt, 0, cap),
        f"GT mu_eff cap={cap}",
        cmap="viridis",
        vmin=0,
        vmax=cap,
        s=max(dot, dot * 1.25),
    )
    mu_pr_ceil = np.clip(mu_pr, 0, cap)
    _scatter_fullmesh_region(
        axes[1, 1],
        pos,
        mu_pr_ceil,
        ceiling_m,
        f"Pred mu_eff (fixed_mu, ceiling only)",
        cmap="viridis",
        vmin=0,
        vmax=cap,
        s=max(dot, dot * 1.25),
    )
    fig.suptitle(
        f"prior rule -- {args.anchor} t={ti} | band F1={band['clot_f1']:.3f} "
        f"GT mu>={thresh:.3f} fullmesh rec={float(gt_bin.mean()):.2f}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")

    if args.mask_overlay_out or os.environ.get("CLOT_PHI_VIZ_MASK_OVERLAY", "").strip() in ("1", "true"):
        overlay_path = Path(args.mask_overlay_out) if args.mask_overlay_out else out_path.with_name(
            out_path.stem.replace("_fullmesh", "") + "_masks.png"
        )
        if not overlay_path.is_absolute():
            overlay_path = root / overlay_path
        t0_m = resolve_t0_dgamma_wall_mask(data, device, bio_cfg).detach().cpu().numpy().astype(bool)
        support_m = (
            resolve_growth_support_at_time(data, ti, device, phys_cfg, bio_cfg)
            .detach()
            .cpu()
            .numpy()
            .astype(bool)
        )
        pred_clot_phi = (phi_pr.reshape(-1) >= 0.5).astype(np.float32)
        _write_mask_overlay_figure(
            pos=pos,
            t0_mask=t0_m,
            support=support_m,
            ceiling=ceiling_m,
            pred_clot=pred_clot_phi,
            anchor=args.anchor,
            ti=ti,
            out=overlay_path,
            scatter_size=dot,
        )
        print(f"[save] {overlay_path}")


if __name__ == "__main__":
    main()
