"""6-panel mask diagram (training-style) + deploy rank pools + rule phi."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch

from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND, VesselConfig
from src.core_physics.clot_growth_masks import resolve_ceiling_mask
from src.core_physics.clot_phi_simple import (
    ClotPriorRuleConfig,
    _anchor_flow_props,
    _top_frac_mask,
    cap_mu_eff_si,
    clot_phi_thresh_si,
    clot_prior_score_flat,
    compute_clot_kinematics_fields,
    dgamma_dx_slice_mask,
    neighbor_supervision_mask,
    predict_phi_prior_rule,
)
from src.training.train_clot_phi_simple import build_clot_phi_step
from src.utils.channel_schema import BIO_Y_SCHEMA, assert_graph_schema, infer_missing_schema
from src.utils.paths import get_project_root


def _ensure_mask_env() -> None:
    for key, val in {
        "CLOT_PHI_MASK_MODE": "neighbor",
        "CLOT_PHI_DGAMMA_SLICE": "1",
        "CLOT_PHI_DGAMMA_REF_TIME": "0",
        "CLOT_PHI_DGAMMA_WALL_MIN_SI": "100",
        "CLOT_PHI_DGAMMA_OFFWALL_PCT": "80",
        "CLOT_PHI_MU_CAP_SI": "0.10",
        "BIOCHEM_PRIOR_COMSOL_ALIGNED": "1",
    }.items():
        os.environ.setdefault(key, val)


def _tri_panel(fig, plot_i, pos, vals, title, *, cmap="Greens", vmin=0, vmax=1) -> None:
    ax = fig.add_subplot(2, 3, plot_i)
    triang = mtri.Triangulation(pos[:, 0], pos[:, 1])
    tri_pts = pos[triang.triangles]
    d1 = np.sum((tri_pts[:, 0, :] - tri_pts[:, 1, :]) ** 2, axis=1)
    d2 = np.sum((tri_pts[:, 1, :] - tri_pts[:, 2, :]) ** 2, axis=1)
    d3 = np.sum((tri_pts[:, 2, :] - tri_pts[:, 0, :]) ** 2, axis=1)
    max_edge_sq = np.max(np.vstack([d1, d2, d3]), axis=0)
    triang.set_mask(max_edge_sq > (np.median(max_edge_sq) * 10.0))
    tc = ax.tripcolor(triang, vals, cmap=cmap, vmin=vmin, vmax=vmax, shading="gouraud")
    fig.colorbar(tc, ax=ax, fraction=0.046)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy rule mask comparison (training-style panels)")
    parser.add_argument("--anchor", default="patient007")
    parser.add_argument("--time-index", type=int, default=-1)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    _ensure_mask_env()

    root = get_project_root()
    device = torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")

    anchor_dir = root / VesselConfig(phase="biochem_anchors").graph_output_dir
    data = torch.load(anchor_dir / f"{args.anchor}.pt", map_location=device, weights_only=False)
    data = infer_missing_schema(data, phase_hint="biochem")
    assert_graph_schema(data, expected_y_schema=(BIO_Y_SCHEMA,))

    ti = args.time_index if args.time_index >= 0 else int(data.y.shape[0]) - 1
    step = build_clot_phi_step(data, ti, phys, bio, device)
    pos = data.x[:, :2].detach().cpu().numpy()

    mu_cap = cap_mu_eff_si(phys.viscosity_nd_to_si(data.y[ti, :, STATE_CHANNEL_MU_EFF_ND]))
    clot_gt = mu_cap.reshape(-1) >= clot_phi_thresh_si(phys)
    empty = torch.zeros(int(data.num_nodes), device=device, dtype=torch.bool)

    neighbor_oracle = neighbor_supervision_mask(data, device, clot_gt)
    neighbor_deploy = neighbor_supervision_mask(data, device, empty)
    dgamma_oracle = dgamma_dx_slice_mask(data, device, neighbor_oracle, clot_gt, bio)
    dgamma_deploy = dgamma_dx_slice_mask(data, device, neighbor_deploy, empty, bio)
    ceiling = resolve_ceiling_mask(data, device, bio)

    rule = ClotPriorRuleConfig(
        prior_p=0.80, flux_stag_top_frac=0.20, rank_tie_break=True, use_t0_strip=False
    )
    phi_ceiling, _ = predict_phi_prior_rule(data, device, bio, rule=rule, t_in=0)
    phi_dgamma_rank, _ = predict_phi_prior_rule(
        data,
        device,
        bio,
        rule=ClotPriorRuleConfig(
            prior_p=0.80,
            flux_stag_top_frac=0.20,
            rank_tie_break=True,
            use_t0_strip=False,
            rank_dgamma_slice=True,
        ),
        t_in=0,
    )

    wall = (
        data.mask_wall.view(-1).bool().cpu().numpy().astype(float)
        if hasattr(data, "mask_wall")
        else np.zeros(int(data.num_nodes))
    )
    phi_gt = step.phi_gt.detach().cpu().numpy()

    fig = plt.figure(figsize=(14, 9))
    pi = 1
    _tri_panel(fig, pi, pos, wall, f"mask_wall n={int(wall.sum())}", cmap="Blues")
    pi += 1
    _tri_panel(
        fig,
        pi,
        pos,
        neighbor_oracle.cpu().numpy().astype(float),
        f"neighbor + GT seeds (train) n={int(neighbor_oracle.sum())}",
    )
    pi += 1
    _tri_panel(
        fig,
        pi,
        pos,
        dgamma_oracle.cpu().numpy().astype(float),
        f"loss pool -dgamma/dx (train) n={int(dgamma_oracle.sum())}",
    )
    pi += 1
    _tri_panel(
        fig,
        pi,
        pos,
        dgamma_deploy.cpu().numpy().astype(float),
        f"deploy dgamma@neighbor n={int(dgamma_deploy.sum())}",
    )
    pi += 1
    _tri_panel(fig, pi, pos, phi_gt, "GT phi @ tfinal", cmap="Reds")
    pi += 1
    _tri_panel(
        fig,
        pi,
        pos,
        phi_ceiling.cpu().numpy().astype(float),
        f"rule phi ceiling rank n={int((phi_ceiling>=0.5).sum())}",
        cmap="Reds",
    )

    fig.suptitle(
        f"deploy vs train masks — {args.anchor} t={ti} | "
        f"ceiling={int(ceiling.sum())} deploy_dgamma={int(dgamma_deploy.sum())} "
        f"train_loss={int(dgamma_oracle.sum())}",
        fontsize=11,
    )
    fig.tight_layout()

    out = (
        Path(args.out)
        if args.out
        else root / f"outputs/biochem/viz/clot_deploy/clot_rule_masks_compare_{args.anchor}.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
