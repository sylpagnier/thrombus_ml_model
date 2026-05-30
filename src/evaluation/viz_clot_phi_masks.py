"""Visualize clot-phi supervision masks and GT labels on an anchor graph."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch

from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND, VesselConfig
from src.core_physics.clot_phi_simple import (
    build_clot_phi_step,
    cap_mu_eff_si,
    clot_phi_dgamma_slice_enabled,
    clot_phi_mask_mode,
    clot_phi_mu_cap_si,
    clot_phi_shear_min_frac,
    clot_phi_thresh_si,
    dgamma_dx_slice_mask,
    neighbor_supervision_mask,
    shear_activity_mask,
    supervision_region_mask,
)
from src.utils.channel_schema import BIO_Y_SCHEMA, assert_graph_schema, infer_missing_schema
from src.utils.paths import get_project_root


def _ensure_training_mask_env() -> None:
    """Match scripts/_clot_phi_shared_env.ps1 so viz matches train/eval."""
    defaults = {
        "CLOT_PHI_MASK_MODE": "neighbor",
        "CLOT_PHI_CENTER_EXCLUDE_FRAC": "0.10",
        "CLOT_PHI_CLOT_TOUCH_HOPS": "1",
        "CLOT_PHI_DGAMMA_SLICE": "1",
        "CLOT_PHI_DGAMMA_REF_TIME": "0",
        "CLOT_PHI_DGAMMA_WALL_MIN_SI": "100",
        "CLOT_PHI_DGAMMA_OFFWALL_PCT": "80",
        "CLOT_PHI_SHEAR_MIN_FRAC": "0",
        "CLOT_PHI_MU_CAP_SI": "0.10",
        "CLOT_PHI_THRESH_SI": "0.055",
    }
    for key, val in defaults.items():
        os.environ.setdefault(key, val)


def _tri_panel(
    fig,
    plot_i: list[int],
    pos: np.ndarray,
    vals: np.ndarray,
    title: str,
    *,
    cmap: str = "gray",
    vmin=None,
    vmax=None,
) -> None:
    ax = fig.add_subplot(2, 3, plot_i[0])
    plot_i[0] += 1
    triang = mtri.Triangulation(pos[:, 0], pos[:, 1])
    tri_pts = pos[triang.triangles]
    d1 = np.sum((tri_pts[:, 0, :] - tri_pts[:, 1, :]) ** 2, axis=1)
    d2 = np.sum((tri_pts[:, 1, :] - tri_pts[:, 2, :]) ** 2, axis=1)
    d3 = np.sum((tri_pts[:, 2, :] - tri_pts[:, 0, :]) ** 2, axis=1)
    max_edge_sq = np.max(np.vstack([d1, d2, d3]), axis=0)
    triang.set_mask(max_edge_sq > (np.median(max_edge_sq) * 10.0))
    tc = ax.tripcolor(triang, vals, cmap=cmap, vmin=vmin, vmax=vmax, shading="gouraud")
    fig.colorbar(tc, ax=ax, fraction=0.046)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser(description="Viz clot-phi supervision masks")
    parser.add_argument("--anchor", default="patient007")
    parser.add_argument("--time-index", type=int, default=-1)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    _ensure_training_mask_env()

    root = get_project_root()
    device = torch.device("cpu")
    phys_cfg = PhysicsConfig(phase="biochem")
    bio_cfg = BiochemConfig(phase="biochem")

    raw_dir = (os.environ.get("CLOT_PHI_ANCHOR_DIR") or "").strip()
    if raw_dir:
        anchor_dir = Path(raw_dir).expanduser()
        if not anchor_dir.is_absolute():
            anchor_dir = root / anchor_dir
    else:
        anchor_dir = root / VesselConfig(phase="biochem_anchors").graph_output_dir
    graph_path = anchor_dir.resolve() / f"{args.anchor}.pt"
    if not graph_path.is_file():
        raise FileNotFoundError(graph_path)

    data = torch.load(graph_path, weights_only=False).to(device)
    data = infer_missing_schema(data, phase_hint="biochem")
    assert_graph_schema(data, expected_y_schema=(BIO_Y_SCHEMA,))

    ti = args.time_index if args.time_index >= 0 else int(data.y.shape[0]) - 1
    step = build_clot_phi_step(data, ti, phys_cfg, bio_cfg, device)

    if hasattr(data, "mask_wall") and data.mask_wall is not None:
        wall_t = data.mask_wall.view(-1).to(device=device).bool()
    else:
        wall_t = torch.zeros(int(data.num_nodes), device=device, dtype=torch.bool)

    mu_nd = data.y[ti, :, STATE_CHANNEL_MU_EFF_ND]
    mu_cap_t = cap_mu_eff_si(phys_cfg.viscosity_nd_to_si(mu_nd))
    clot_seed = mu_cap_t.reshape(-1) >= clot_phi_thresh_si(phys_cfg)

    neighbor_base = neighbor_supervision_mask(data, device, clot_seed)
    region_final = supervision_region_mask(data, device, mu_cap_t, phys_cfg)

    # Show intermediate filters (dgamma vs shear are mutually exclusive in training code).
    if clot_phi_dgamma_slice_enabled():
        after_activity = dgamma_dx_slice_mask(
            data, device, neighbor_base, clot_seed, bio_cfg
        )
        activity_label = "after -dgamma/dx @ t=0"
    elif clot_phi_shear_min_frac() > 0.0:
        after_activity = neighbor_base & shear_activity_mask(
            data, device, wall_t, pool=neighbor_base
        )
        activity_label = f"after shear>={clot_phi_shear_min_frac():.2f}*max"
    else:
        after_activity = neighbor_base
        activity_label = "no activity filter"

    pos = data.x[:, :2].detach().cpu().numpy()
    wall = wall_t.cpu().numpy().astype(float)
    n_base = neighbor_base.cpu().numpy().astype(float)
    n_act = after_activity.cpu().numpy().astype(float)
    region = region_final.cpu().numpy().astype(float)
    loss_m = step.loss_mask.detach().cpu().numpy().astype(float)
    phi_gt = step.phi_gt.detach().cpu().numpy()
    mu_cap = step.mu_gt_cap.detach().cpu().numpy()
    cap = clot_phi_mu_cap_si()
    band = clot_phi_mask_mode()

    n_wall = int(wall_t.sum())
    n_base_i = int(neighbor_base.sum())
    n_act_i = int(after_activity.sum())
    n_final_i = int(region_final.sum())

    fig = plt.figure(figsize=(14, 9))
    plot_i = [1]
    _tri_panel(fig, plot_i, pos, wall, "mask_wall", cmap="Blues", vmin=0, vmax=1)
    _tri_panel(
        fig,
        plot_i,
        pos,
        n_base,
        f"neighbor ({band}) n={n_base_i}",
        cmap="Greens",
        vmin=0,
        vmax=1,
    )
    _tri_panel(
        fig,
        plot_i,
        pos,
        n_act,
        f"{activity_label} n={n_act_i}",
        cmap="Greens",
        vmin=0,
        vmax=1,
    )
    _tri_panel(
        fig,
        plot_i,
        pos,
        region,
        f"loss region n={n_final_i}",
        cmap="Greens",
        vmin=0,
        vmax=1,
    )
    _tri_panel(fig, plot_i, pos, phi_gt, "GT phi", cmap="Reds", vmin=0, vmax=1)
    _tri_panel(fig, plot_i, pos, mu_cap, f"GT mu cap {cap:.3f} Pa*s", cmap="bwr", vmin=0, vmax=cap)

    gt_pos = float(phi_gt[loss_m > 0.5].mean()) if (loss_m > 0.5).any() else 0.0
    dg = int(clot_phi_dgamma_slice_enabled())
    sh = clot_phi_shear_min_frac()
    fig.suptitle(
        f"clot_phi masks — {args.anchor} label_t={ti} | dgamma={dg} shear_frac={sh:.2f} "
        f"| wall={n_wall} gt+ in loss={gt_pos:.3f}",
        fontsize=11,
    )
    fig.tight_layout()

    out = (
        Path(args.out)
        if args.out
        else root / "outputs" / "biochem" / f"clot_phi_masks_{args.anchor}.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(
        f"[OK]  Wrote {out.resolve()} | neighbor={n_base_i} activity={n_act_i} final={n_final_i}",
        flush=True,
    )


if __name__ == "__main__":
    main()
