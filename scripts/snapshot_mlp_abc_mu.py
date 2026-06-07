"""Headless A/B/C mu coupling snapshots (matches go_mlp_abc_compare_1h legs).

Saves per-leg PNGs under outputs/biochem/viz/abc_mu/:
  - dynamic mu_eff @ t_final (COMSOL vs rollout channel 3)
  - clot-band mu @ t_final (GT vs pred, supervision region only)

Usage:
  python scripts/snapshot_mlp_abc_mu.py --anchor patient007
  python scripts/snapshot_mlp_abc_mu.py --legs A,B,C --anchor patient007
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.run_mlp_clot_inject_probe import (  # noqa: E402
    _configure_leg,
    _load_teacher,
    _normalize_probe_leg,
    _rollout,
)
from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND
from src.core_physics.clot_phi_simple import build_clot_phi_step, cap_mu_eff_si
from src.evaluation.clot_shape_score import (
    compute_clot_shape_metrics,
    mu_clot_binary_mask,
    resolve_clot_shape_mu_thresh_si,
)
from src.utils.paths import get_project_root


def _viz_mu_clim(*arrays: np.ndarray) -> tuple[float, float]:
    raw = (os.environ.get("VIZ_MU_CLIM") or "fixed").strip().lower()
    if raw in ("auto", "data", "1", "true", "yes", "on"):
        flat = np.concatenate([a.reshape(-1) for a in arrays if a.size > 0])
        if flat.size < 1:
            return 0.04, 0.10
        return float(np.min(flat)), float(np.max(flat))
    vmin_s = (os.environ.get("VIZ_MU_VMIN") or "0.04").strip()
    vmax_s = (os.environ.get("VIZ_MU_VMAX") or "0.10").strip()
    return float(vmin_s), float(vmax_s)


def _scatter_mesh(ax, pos, vals, title, *, vmin, vmax, cmap="bwr"):
    sc = ax.scatter(
        pos[:, 0],
        pos[:, 1],
        c=vals,
        s=8,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
    )
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")


def _leg_label(leg: str) -> str:
    return {
        "A": "A baseline (full-domain GNODE mu)",
        "B": "B MLP mu map v2 (gt_clot + cap_low_shear)",
        "B_deploy": "B_deploy MLP mu map (neighbor wall + 1-hop pred clot/phi)",
        "C": "C GNODE mask-only (gt_clot + cap_low_shear)",
    }.get(leg, leg)


@torch.no_grad()
def _snapshot_one_leg(
    *,
    leg: str,
    teacher,
    data,
    bio_cfg,
    phys: PhysicsConfig,
    device,
    anchor: str,
    out_dir: Path,
    time_stride: int,
) -> Path:
    pred = _rollout(teacher, data, bio_cfg, device, time_stride=time_stride, fast=False)
    gt = data.y.to(device)
    ti = int(gt.shape[0]) - 1
    pred_ti = min(ti, int(pred.shape[0]) - 1)

    pred_mu = phys.viscosity_nd_to_si(pred[pred_ti, :, STATE_CHANNEL_MU_EFF_ND]).detach().cpu().numpy()
    gt_mu = phys.viscosity_nd_to_si(gt[ti, :, STATE_CHANNEL_MU_EFF_ND]).detach().cpu().numpy()
    pos = data.x[:, :2].detach().cpu().numpy()

    t_si = float(bio_cfg.resolve_biochem_times(data, device)[ti].item())
    vmin, vmax = _viz_mu_clim(gt_mu, pred_mu)

    fig1, axs1 = plt.subplots(1, 2, figsize=(14, 5.5))
    _scatter_mesh(axs1[0], pos, gt_mu, f"COMSOL mu_eff @ t~{t_si:.0f}s", vmin=vmin, vmax=vmax)
    _scatter_mesh(
        axs1[1],
        pos,
        pred_mu,
        f"Rollout mu_eff ch3 ({leg})",
        vmin=vmin,
        vmax=vmax,
    )
    fig1.suptitle(f"{_leg_label(leg)} -- {anchor}", fontsize=12)
    fig1.tight_layout(rect=(0, 0, 1, 0.92))
    out_dyn = out_dir / f"leg_{leg}_dynamic_mu_{anchor}.png"
    fig1.savefig(out_dyn, dpi=140, bbox_inches="tight")
    plt.close(fig1)

    thresh = resolve_clot_shape_mu_thresh_si(phys)
    gt_clot = mu_clot_binary_mask(torch.from_numpy(gt_mu), thresh).numpy()
    pred_clot = mu_clot_binary_mask(torch.from_numpy(pred_mu), thresh).numpy()
    shape_m = compute_clot_shape_metrics(
        pred_state=pred[pred_ti],
        gt_state=gt[ti],
        edge_index=data.edge_index,
        phys_cfg=phys,
        mu_thresh_si=thresh,
    )

    fig3, axs3 = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, mask, title in (
        (axs3[0], gt_clot, f"GT clot mask (mu>={thresh:.3f})"),
        (axs3[1], pred_clot, f"Pred clot mask ({leg})"),
    ):
        colors = np.where(mask, "#c0392b", "#3498db")
        ax.scatter(pos[:, 0], pos[:, 1], c=colors, s=8, linewidths=0)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.axis("off")
    fig3.suptitle(
        f"Clot shape -- clot_shape={shape_m['clot_shape']:.3f} "
        f"dice={shape_m['clot_dice']:.3f} recall={shape_m['clot_recall']:.3f} "
        f"-- {_leg_label(leg)}",
        fontsize=11,
    )
    fig3.tight_layout(rect=(0, 0, 1, 0.92))
    out_shape = out_dir / f"leg_{leg}_clot_shape_{anchor}.png"
    fig3.savefig(out_shape, dpi=140, bbox_inches="tight")
    plt.close(fig3)

    step = build_clot_phi_step(data, ti, phys, bio_cfg, device, rollout_state=None)
    region = step.region.detach().cpu().numpy().astype(bool)
    idx = np.where(region)[0]
    mu_gt_cap = step.mu_gt_cap.detach().cpu().numpy()
    mu_pr_cap = cap_mu_eff_si(
        torch.from_numpy(pred_mu).to(device=device, dtype=torch.float32)
    ).detach().cpu().numpy()
    cap = 0.10

    fig2, axs2 = plt.subplots(1, 2, figsize=(14, 5.5))
    if idx.size > 0:
        sc0 = axs2[0].scatter(
            pos[idx, 0], pos[idx, 1], c=mu_gt_cap[idx], s=14, cmap="RdBu_r", vmin=0.0, vmax=cap, linewidths=0
        )
        plt.colorbar(sc0, ax=axs2[0], fraction=0.046)
        sc1 = axs2[1].scatter(
            pos[idx, 0], pos[idx, 1], c=mu_pr_cap[idx], s=14, cmap="RdBu_r", vmin=0.0, vmax=cap, linewidths=0
        )
        plt.colorbar(sc1, ax=axs2[1], fraction=0.046)
    for ax, title in zip(
        axs2,
        ("GT mu cap 0.10 Pa*s (region)", f"Pred mu cap rollout ({leg})"),
    ):
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.axis("off")
    fig2.suptitle(f"Clot-band mu -- {_leg_label(leg)} -- {anchor} t~{t_si:.0f}s", fontsize=12)
    fig2.tight_layout(rect=(0, 0, 1, 0.92))
    out_band = out_dir / f"leg_{leg}_clotband_{anchor}.png"
    fig2.savefig(out_band, dpi=140, bbox_inches="tight")
    plt.close(fig2)

    print(
        f"[OK]  leg {leg} {anchor}: {out_dyn.name} + {out_shape.name} + {out_band.name}",
        flush=True,
    )
    return out_dyn


def main() -> int:
    ap = argparse.ArgumentParser(description="Headless A/B/C mu viz snapshots")
    ap.add_argument("--teacher-checkpoint", default="outputs/biochem/clot_baseline/teacher_best_high_mu.pth")
    ap.add_argument("--clot-phi-checkpoint", default="outputs/biochem/clot_baseline/clot_phi_best.pth")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--legs", default="A,B,C", help="Comma-separated subset of A,B,C")
    ap.add_argument("--mu-ratio-max", type=float, default=20.0)
    ap.add_argument("--out-dir", default="outputs/biochem/viz/abc_mu")
    ap.add_argument("--time-stride", type=int, default=5, help="Match abc_compare_1h default")
    args = ap.parse_args()

    root = get_project_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["BIOCHEM_GT_KINE_VEL"] = "0"
    os.environ["BIOCHEM_ROLLOUT_PROGRESS"] = "1"

    ckpt = root / args.teacher_checkpoint.replace("/", os.sep)
    clot_ckpt = root / args.clot_phi_checkpoint.replace("/", os.sep)
    if not ckpt.is_file():
        print(f"[ERR] teacher ckpt missing: {ckpt}", file=sys.stderr)
        return 1
    if not clot_ckpt.is_file():
        print(f"[ERR] clot-phi ckpt missing: {clot_ckpt}", file=sys.stderr)
        return 1

    graph_path = root / "data/processed/graphs_biochem_anchors" / f"{args.anchor}.pt"
    if not graph_path.is_file():
        print(f"[ERR] anchor missing: {graph_path}", file=sys.stderr)
        return 1

    out_dir = root / args.out_dir.replace("/", os.sep)
    out_dir.mkdir(parents=True, exist_ok=True)

    legs = [_normalize_probe_leg(x) for x in args.legs.split(",") if x.strip()]
    known = {"A", "B", "B_deploy", "C"}
    unknown = [x for x in legs if x not in known]
    if unknown:
        print(f"[ERR] unknown legs: {unknown!r}", file=sys.stderr)
        return 1

    mu_ratio = args.mu_ratio_max
    teacher, phys, bio = _load_teacher(ckpt, device, mu_ratio, fast=False)

    # Patch stride into rollout via monkey - _rollout uses time_stride param
    for leg in legs:
        print(f"[NEW] leg {leg} snapshot {args.anchor} (stride={args.time_stride})", flush=True)
        _configure_leg(teacher, device, leg, clot_ckpt=clot_ckpt)
        data = torch.load(graph_path, map_location=device, weights_only=False)
        _snapshot_one_leg(
            leg=leg,
            teacher=teacher,
            data=data,
            bio_cfg=bio,
            phys=phys,
            device=device,
            anchor=args.anchor,
            out_dir=out_dir,
            time_stride=max(1, int(args.time_stride)),
        )

    print(f"[OK]  wrote under {out_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
