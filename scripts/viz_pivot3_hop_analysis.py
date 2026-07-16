"""Hop-distance analysis for the dynamic occlusion (Pivot 3) model.

Produces a figure with two sections:
  Row 1-3: Spatial scatter panels (GT clot | Pred clot | Error) coloured by wall-hop distance.
  Row 4  : Bar chart of clot node counts per hop bucket (GT vs Pred) at final time step.

Usage:
  python scripts/viz_pivot3_hop_analysis.py
  python scripts/viz_pivot3_hop_analysis.py --leg WC_mat_3hop --anchor patient007
  python scripts/viz_pivot3_hop_analysis.py --leg WC_canonical_v2 --compare-leg WC_mat_3hop
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.eval_mat_growth_simple import _apply_ckpt_recipe  # noqa: E402
from src.biochem_gnn.config import apply_deploy_env  # noqa: E402
from src.biochem_gnn.mat_growth_simple import leg_out_ckpt  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.species_gnn_clot_rollout import (  # noqa: E402
    load_species_gnn_rollout_bundle,
    prepare_species_gnn_rollout_static,
    rollout_species_gnn_phi_trajectory,
)
from src.core_physics.t0_device import require_cuda_device  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402


HOP_CMAP = plt.cm.get_cmap("RdYlGn_r", 6)  # hop 0=red (wall), 5+=green (far interior)
CLOT_THRESH = 0.5


def compute_wall_hops(edge_index: torch.Tensor, wall_mask: torch.Tensor, n_nodes: int, max_hops: int = 8) -> np.ndarray:
    """BFS from wall nodes; returns hop distance array (shape [n_nodes], unreachable=max_hops+1)."""
    hops = np.full(n_nodes, max_hops + 1, dtype=np.int32)
    row, col = edge_index.cpu().numpy()
    frontier = np.where(wall_mask.cpu().numpy())[0]
    hops[frontier] = 0
    for h in range(1, max_hops + 1):
        neighbor_mask = np.zeros(n_nodes, dtype=bool)
        # vectorised: for each edge (r->c), if r is in current frontier, mark c
        in_frontier = hops[row] == h - 1
        neighbor_mask[col[in_frontier]] = True
        new_nodes = np.where(neighbor_mask & (hops > h))[0]
        if len(new_nodes) == 0:
            break
        hops[new_nodes] = h
    return hops


def _scatter_hop_coloured(ax, pos: np.ndarray, clot_mask: np.ndarray, hops: np.ndarray,
                           title: str, *, max_hop: int = 5, s: float = 4.0) -> None:
    """Scatter plot: non-clot nodes grey, clot nodes coloured by hop distance."""
    non_clot = ~clot_mask
    ax.scatter(pos[non_clot, 0], pos[non_clot, 1], c="#cccccc", s=s * 0.3, linewidths=0)
    if clot_mask.any():
        hop_clot = np.clip(hops[clot_mask], 0, max_hop)
        sc = ax.scatter(pos[clot_mask, 0], pos[clot_mask, 1], c=hop_clot,
                        cmap="RdYlGn_r", vmin=0, vmax=max_hop,
                        s=s, linewidths=0)
        plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.01, label="Wall hop")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _hop_bar_chart(ax, hops_all: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray,
                   *, max_hop: int = 6) -> None:
    """Bar chart of clot node counts per hop bucket."""
    buckets = list(range(max_hop + 1))
    gt_counts = [int(np.sum(gt_mask & (hops_all == h))) for h in buckets]
    pred_counts = [int(np.sum(pred_mask & (hops_all == h))) for h in buckets]
    labels = [str(h) if h < max_hop else f"{max_hop}+" for h in buckets]
    x = np.arange(len(buckets))
    w = 0.38
    ax.bar(x - w / 2, gt_counts, width=w, label="GT clot", color="#2196f3", alpha=0.85)
    ax.bar(x + w / 2, pred_counts, width=w, label="Pred clot", color="#f44336", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Wall-hop distance")
    ax.set_ylabel("Node count")
    ax.set_title("Clot nodes per wall-hop bucket (final frame)", fontsize=9)
    ax.legend(fontsize=8)
    # annotate zeros in pred where GT has clots
    for i, (g, p) in enumerate(zip(gt_counts, pred_counts)):
        if g > 0 and p == 0:
            ax.text(x[i] + w / 2, 1, "0!", ha="center", va="bottom",
                    color="#f44336", fontsize=8, fontweight="bold")


def _run_rollout(ckpt: Path, data, phys, bio, device, flow_source="kinematics"):
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    meta = dict(payload.get("meta") or {})
    _apply_ckpt_recipe(meta, label="mat_growth_simple", ckpt_path=ckpt)
    apply_deploy_env(overrides={"T0_R4_FLOW_SOURCE": flow_source})
    bundle = load_species_gnn_rollout_bundle(ckpt, device=device)
    if bundle is None:
        raise SystemExit(f"[ERR] could not load bundle from {ckpt}")
    static = prepare_species_gnn_rollout_static(data, device=device)
    traj = rollout_species_gnn_phi_trajectory(
        data, bundle, static, phys_cfg=phys, bio_cfg=bio,
        device=device, flow_source=flow_source,
    )
    return traj


def main() -> int:
    ap = argparse.ArgumentParser(description="Hop-distance clot viz for Pivot 3 / WC_canonical_v2")
    ap.add_argument("--leg", default="WC_pivot3_occlusion")
    ap.add_argument("--compare-leg", default="WC_mat_3hop",
                    help="Second leg to compare against (baseline). Set empty to skip.")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--compare-ckpt", default="")
    ap.add_argument("--flow", default="kinematics", choices=("gt", "kinematics"))
    ap.add_argument("--t", type=int, default=-1, help="Time step to visualize (-1 = final)")
    ap.add_argument("--max-hop", type=int, default=5)
    ap.add_argument("--scatter-size", type=float, default=4.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = require_cuda_device()
    root = get_project_root()

    ckpt = Path(args.ckpt.strip()) if args.ckpt.strip() else root / leg_out_ckpt(args.leg.strip())
    if not ckpt.is_file():
        raise SystemExit(f"[ERR] missing ckpt: {ckpt}")

    compare_leg = args.compare_leg.strip()
    compare_ckpt: Path | None = None
    if compare_leg:
        compare_ckpt = (Path(args.compare_ckpt.strip()) if args.compare_ckpt.strip()
                        else root / leg_out_ckpt(compare_leg))
        if not compare_ckpt.is_file():
            print(f"[WARN] compare ckpt not found: {compare_ckpt} — skipping comparison", flush=True)
            compare_ckpt = None

    data = torch.load(
        root / "data/processed/graphs_biochem_anchors" / f"{args.anchor}.pt",
        map_location=device, weights_only=False,
    )
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    pos = data.x[:, :2].detach().cpu().numpy()
    n_nodes = int(data.num_nodes)
    T_final = int(data.y.shape[0]) - 1
    t_viz = args.t if args.t >= 0 else T_final

    # ---- Wall-hop distances from full-graph edge_index ----
    # wall nodes = SDF-based mask stored in data if available, else hop 0 = boundary nodes
    wall_mask = None
    if hasattr(data, "mask_wall"):
        wall_mask = data.mask_wall.bool().to("cpu")
    elif hasattr(data, "wall_mask"):
        wall_mask = data.wall_mask.bool().to("cpu")
    else:
        # fallback: approximate from degree (low-degree boundary nodes)
        ei_cpu = data.edge_index.cpu()
        deg = torch.zeros(n_nodes, dtype=torch.long)
        deg.scatter_add_(0, ei_cpu[0], torch.ones(ei_cpu.shape[1], dtype=torch.long))
        wall_mask = (deg <= deg.float().quantile(0.10)).bool()

    hops_all = compute_wall_hops(data.edge_index, wall_mask, n_nodes, max_hops=args.max_hop + 2)

    # ---- Rollout primary leg ----
    print(f"[i] Rolling out {args.leg} ...", flush=True)
    traj_primary = _run_rollout(ckpt, data, phys, bio, device, args.flow)

    phi_gt = gt_clot_phi_at_time(data, t_viz, phys, device).detach().cpu().numpy().ravel()
    phi_pred = traj_primary[t_viz].detach().cpu().numpy().ravel()
    gt_mask = phi_gt > CLOT_THRESH
    pred_mask = phi_pred > CLOT_THRESH

    traj_compare = None
    phi_pred_cmp = None
    pred_mask_cmp = None
    if compare_ckpt is not None:
        print(f"[i] Rolling out {compare_leg} ...", flush=True)
        traj_compare = _run_rollout(compare_ckpt, data, phys, bio, device, args.flow)
        phi_pred_cmp = traj_compare[t_viz].detach().cpu().numpy().ravel()
        pred_mask_cmp = phi_pred_cmp > CLOT_THRESH

    # ---- Print hop breakdown ----
    print(f"\n[i] Clot node hop breakdown at t={t_viz} for {args.anchor}:", flush=True)
    print(f"  {'Hop':>4}  {'GT':>6}  {'Pred(' + args.leg + ')':>20}  {'Pred(' + compare_leg + ')':>20}", flush=True)
    for h in range(args.max_hop + 1):
        g_c = int(np.sum(gt_mask & (hops_all == h)))
        p_c = int(np.sum(pred_mask & (hops_all == h)))
        c_c = int(np.sum(pred_mask_cmp & (hops_all == h))) if pred_mask_cmp is not None else -1
        cmp_str = f"{c_c:>6}" if c_c >= 0 else "     N/A"
        print(f"  {h:>4}  {g_c:>6}  {p_c:>20}  {cmp_str:>20}", flush=True)

    # ---- Figure ----
    n_compare_cols = 1 if compare_ckpt is None else 2
    n_rows = 4  # GT scatter | Primary pred | (Compare pred) | Bar chart
    fig_w = max(10, 4.5 * n_compare_cols + 2)
    fig, axes = plt.subplots(n_rows, max(n_compare_cols, 1),
                             figsize=(fig_w, 14),
                             squeeze=False)

    leg_tag = args.leg.strip()
    fig.suptitle(
        f"Hop-distance clot analysis -- {args.anchor} | t={t_viz}\n"
        f"Primary: {leg_tag}" + (f"  vs  {compare_leg}" if compare_ckpt else ""),
        fontsize=11, y=1.01,
    )

    s = float(args.scatter_size)

    # Row 0: GT
    _scatter_hop_coloured(axes[0, 0], pos, gt_mask, hops_all,
                           f"GT clot (t={t_viz})", max_hop=args.max_hop, s=s)
    if n_compare_cols > 1:
        _scatter_hop_coloured(axes[0, 1], pos, gt_mask, hops_all,
                               f"GT clot (t={t_viz})", max_hop=args.max_hop, s=s)

    # Row 1: Primary pred
    fp_mask = pred_mask & ~gt_mask
    fn_mask = ~pred_mask & gt_mask
    _scatter_hop_coloured(axes[1, 0], pos, pred_mask, hops_all,
                           f"Pred: {leg_tag}  F1={np.nan:.2f}", max_hop=args.max_hop, s=s)
    # compute quick F1
    tp = int(np.sum(pred_mask & gt_mask))
    prec = tp / max(int(np.sum(pred_mask)), 1)
    rec = tp / max(int(np.sum(gt_mask)), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    axes[1, 0].set_title(
        f"Pred: {leg_tag}\nF1={f1:.3f}  Prec={prec:.3f}  Rec={rec:.3f}", fontsize=9)

    # Row 2: Compare or error
    if compare_ckpt is not None:
        tp2 = int(np.sum(pred_mask_cmp & gt_mask))
        prec2 = tp2 / max(int(np.sum(pred_mask_cmp)), 1)
        rec2 = tp2 / max(int(np.sum(gt_mask)), 1)
        f1_2 = 2 * prec2 * rec2 / max(prec2 + rec2, 1e-9)
        _scatter_hop_coloured(axes[1, 1], pos, pred_mask_cmp, hops_all,
                               f"Pred: {compare_leg}", max_hop=args.max_hop, s=s)
        axes[1, 1].set_title(
            f"Pred: {compare_leg}\nF1={f1_2:.3f}  Prec={prec2:.3f}  Rec={rec2:.3f}", fontsize=9)
        # Row 2: error panels
        err_primary = np.zeros(n_nodes)
        err_primary[fp_mask] = 1.0   # FP = red
        err_primary[fn_mask] = -1.0  # FN = blue
        ax_err = axes[2, 0]
        ax_err.scatter(pos[:, 0], pos[:, 1], c="#eeeeee", s=s * 0.3, linewidths=0)
        if fp_mask.any():
            ax_err.scatter(pos[fp_mask, 0], pos[fp_mask, 1], c="#f44336", s=s, label="FP")
        if fn_mask.any():
            ax_err.scatter(pos[fn_mask, 0], pos[fn_mask, 1], c="#2196f3", s=s, label="FN")
        ax_err.set_aspect("equal"); ax_err.axis("off")
        ax_err.set_title(f"Error: {leg_tag}  FP={int(fp_mask.sum())} FN={int(fn_mask.sum())}", fontsize=9)

        fp2 = pred_mask_cmp & ~gt_mask
        fn2 = ~pred_mask_cmp & gt_mask
        ax_err2 = axes[2, 1]
        ax_err2.scatter(pos[:, 0], pos[:, 1], c="#eeeeee", s=s * 0.3, linewidths=0)
        if fp2.any():
            ax_err2.scatter(pos[fp2, 0], pos[fp2, 1], c="#f44336", s=s, label="FP")
        if fn2.any():
            ax_err2.scatter(pos[fn2, 0], pos[fn2, 1], c="#2196f3", s=s, label="FN")
        ax_err2.set_aspect("equal"); ax_err2.axis("off")
        ax_err2.set_title(f"Error: {compare_leg}  FP={int(fp2.sum())} FN={int(fn2.sum())}", fontsize=9)
    else:
        # single-leg error panel
        ax_err = axes[2, 0]
        ax_err.scatter(pos[:, 0], pos[:, 1], c="#eeeeee", s=s * 0.3, linewidths=0)
        if fp_mask.any():
            ax_err.scatter(pos[fp_mask, 0], pos[fp_mask, 1], c="#f44336", s=s, label="FP")
        if fn_mask.any():
            ax_err.scatter(pos[fn_mask, 0], pos[fn_mask, 1], c="#2196f3", s=s, label="FN")
        ax_err.set_aspect("equal"); ax_err.axis("off")
        ax_err.set_title(f"Error  FP={int(fp_mask.sum())} FN={int(fn_mask.sum())}", fontsize=9)

    # Row 3: hop bar chart (primary + optional compare)
    ax_bar = axes[3, 0]
    _hop_bar_chart(ax_bar, hops_all, gt_mask, pred_mask, max_hop=args.max_hop)
    if compare_ckpt is not None and pred_mask_cmp is not None:
        ax_bar2 = axes[3, 1]
        _hop_bar_chart(ax_bar2, hops_all, gt_mask, pred_mask_cmp, max_hop=args.max_hop)
        ax_bar2.set_title(f"Clot per hop bucket: {compare_leg} (final frame)", fontsize=9)

    fig.tight_layout()
    out_dir = root / "outputs/biochem/viz/mat_growth"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.out.strip():
        out = Path(args.out.strip())
        if not out.is_absolute():
            out = root / out
    else:
        out = out_dir / f"pivot3_hop_analysis_{leg_tag}_{args.anchor}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[save] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
