"""Side-by-side viz: ceiling wall+2 hops vs ceiling + sdf_nd cap rank pool."""

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

from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND  # noqa: E402
from src.core_physics.clot_growth_masks import resolve_ceiling_mask  # noqa: E402
from src.core_physics.clot_phi_simple import (  # noqa: E402
    ClotPriorRuleConfig,
    clot_phi_thresh_si,
    predict_prior_rule_deploy,
)
from src.evaluation.viz_clot_phi_simple import _scatter_fullmesh_region  # noqa: E402
from src.training.train_clot_phi_simple import _clot_metrics  # noqa: E402
from src.utils.channel_schema import BIO_Y_SCHEMA, assert_graph_schema, infer_missing_schema  # noqa: E402
from src.utils.paths import get_project_root


def _apply_deploy_env() -> None:
    os.environ.setdefault("CLOT_FORECAST_MODE", "one_step")
    os.environ.setdefault("CLOT_FORECAST_PAIR_SCHEDULE", "static_final")
    os.environ.setdefault("CLOT_PHI_VEL_SOURCE", "gt")
    os.environ.setdefault("CLOT_PHI_FIXED_MU_FROM_PHI", "1")
    os.environ.setdefault("CLOT_PHI_HARD_SUPPORT_PROJECTION", "1")
    os.environ.setdefault("CLOT_PHI_SUPPORT_BAND", "ceiling_growth")
    os.environ.setdefault("CLOT_FORECAST_MASK", "ceiling_growth")
    os.environ.setdefault("CLOT_PHI_CEILING_HOPS", "2")
    os.environ.setdefault("BIOCHEM_PRIOR_COMSOL_ALIGNED", "1")
    os.environ.setdefault("BIOCHEM_PRIOR_NORM_MASK", "adjacent")


def _refined_rule(*, rank_sdf_max_nd: float | None = None) -> ClotPriorRuleConfig:
    return ClotPriorRuleConfig(
        name="ceiling_h2" if rank_sdf_max_nd is None else f"ceiling_h2_sdf{rank_sdf_max_nd:.3f}",
        prior_p=0.80,
        flux_stag_top_frac=0.20,
        rank_tie_break=True,
        use_t0_strip=False,
        rank_sdf_max_nd=rank_sdf_max_nd,
    )


def _run_case(data, ti, rule, *, phys, bio, device):
    step, phi, mu, meta = predict_prior_rule_deploy(
        data, ti, phys_cfg=phys, bio_cfg=bio, device=device, t_in=0, rule=rule
    )
    band = _clot_metrics(phi, step.phi_gt, step.loss_mask)
    return step, phi, mu, meta, band


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ceiling vs sdf rank pool viz")
    parser.add_argument("--anchor", default="patient007")
    parser.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    parser.add_argument("--time-index", type=int, default=-1)
    parser.add_argument("--sdf-max", type=float, default=0.040)
    parser.add_argument("--scatter-size", type=float, default=6.0)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    _apply_deploy_env()

    root = get_project_root()
    device = torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")

    anchor_dir = Path(args.anchor_dir)
    if not anchor_dir.is_absolute():
        anchor_dir = root / anchor_dir
    data = torch.load(anchor_dir / f"{args.anchor}.pt", map_location=device, weights_only=False)
    data = infer_missing_schema(data, phase_hint="biochem")
    assert_graph_schema(data, expected_y_schema=(BIO_Y_SCHEMA,))

    ti = args.time_index if args.time_index >= 0 else int(data.y.shape[0]) - 1
    pos = data.x[:, :2].detach().cpu().numpy()
    ceiling_m = resolve_ceiling_mask(data, device, bio).detach().cpu().numpy().astype(bool)

    rule_ceiling = _refined_rule()
    rule_sdf = _refined_rule(rank_sdf_max_nd=float(args.sdf_max))

    step0, phi0, mu0, meta0, band0 = _run_case(data, ti, rule_ceiling, phys=phys, bio=bio, device=device)
    step1, phi1, mu1, meta1, band1 = _run_case(data, ti, rule_sdf, phys=phys, bio=bio, device=device)

    phi_gt = step0.phi_gt.detach().cpu().numpy()
    loss_m = step0.loss_mask.detach().cpu().numpy().astype(bool)
    cap = float(os.environ.get("CLOT_PHI_MU_CAP_SI", "0.10"))
    dot = float(args.scatter_size)

    fig = plt.figure(figsize=(14, 10))
    axes = fig.subplots(2, 3)

    _scatter_fullmesh_region(
        axes[0, 0], pos, phi_gt, loss_m, f"GT phi (band t={ti})", cmap="bwr", vmin=0, vmax=1, s=dot
    )
    _scatter_fullmesh_region(
        axes[0, 1],
        pos,
        phi0.detach().cpu().numpy(),
        ceiling_m,
        f"Rule phi ceiling h=2\nF1={band0['clot_f1']:.3f} flag={meta0['n_flag']} pool={meta0['n_rank_mask']}",
        cmap="bwr",
        vmin=0,
        vmax=1,
        s=dot,
    )
    _scatter_fullmesh_region(
        axes[0, 2],
        pos,
        phi1.detach().cpu().numpy(),
        ceiling_m,
        f"Rule phi + sdf<={args.sdf_max:.3f}\nF1={band1['clot_f1']:.3f} flag={meta1['n_flag']} pool={meta1['n_rank_mask']}",
        cmap="bwr",
        vmin=0,
        vmax=1,
        s=dot,
    )

    mu_gt = step0.mu_gt_cap.detach().cpu().numpy()
    _scatter_fullmesh_region(
        axes[1, 0],
        pos,
        np.clip(mu_gt, 0, cap),
        loss_m,
        f"GT mu cap={cap}",
        cmap="viridis",
        vmin=0,
        vmax=cap,
        s=max(dot, dot * 1.2),
    )
    _scatter_fullmesh_region(
        axes[1, 1],
        pos,
        np.clip(mu0.detach().cpu().numpy(), 0, cap),
        ceiling_m,
        f"Pred mu ceiling h=2\nprec={band0['clot_prec']:.3f} rec={band0['clot_rec']:.3f} pred+={band0['pred_pos_frac']:.3f}",
        cmap="viridis",
        vmin=0,
        vmax=cap,
        s=max(dot, dot * 1.2),
    )
    _scatter_fullmesh_region(
        axes[1, 2],
        pos,
        np.clip(mu1.detach().cpu().numpy(), 0, cap),
        ceiling_m,
        f"Pred mu + sdf<={args.sdf_max:.3f}\nprec={band1['clot_prec']:.3f} rec={band1['clot_rec']:.3f} pred+={band1['pred_pos_frac']:.3f}",
        cmap="viridis",
        vmin=0,
        vmax=cap,
        s=max(dot, dot * 1.2),
    )

    thr = clot_phi_thresh_si(phys)
    fig.suptitle(
        f"S0 rank pool compare -- {args.anchor} t={ti} | "
        f"ceiling F1={band0['clot_f1']:.3f} vs sdf F1={band1['clot_f1']:.3f} | GT mu>={thr:.3f}",
        fontsize=12,
    )
    fig.tight_layout()

    out_default = root / f"outputs/biochem/viz/clot_deploy/prior_rule_rank_compare_{args.anchor}_tfinal.png"
    out_path = Path(args.out) if args.out else out_default
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(
        f"[i]  ceiling_h2: pool={meta0['n_rank_mask']} flag={meta0['n_flag']} "
        f"F1={band0['clot_f1']:.3f} pred+={band0['pred_pos_frac']:.3f}",
        flush=True,
    )
    print(
        f"[i]  sdf<={args.sdf_max:.3f}: pool={meta1['n_rank_mask']} flag={meta1['n_flag']} "
        f"F1={band1['clot_f1']:.3f} pred+={band1['pred_pos_frac']:.3f}",
        flush=True,
    )
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
