"""Visualize the Phase-6 wall+lumen model on vessels with substantial off-wall GT clot.

Entry point under test: ``scripts/predict_wall_clot.py::predict_wall_clot``. Renders, per
vessel, two full-mesh maps side by side -- wall arm alone vs wall+lumen -- so the lumen
arm's contribution is directly visible as new correct (or incorrect) markers appearing in
the thin off-wall shell next to the vessel wall. Marker SHAPE encodes wall vs lumen;
marker COLOR encodes TP/FN/FP/TN, scored on the FULL mesh (not wall-masked) against
PHASE6_RESULTS §20.3's finding that up to 48% of a vessel's clot sits off-wall.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.mls_gradient import node_positions  # noqa: E402
from src.core_physics.species_pushforward_continuous import resolve_deploy_eval_time_index  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.evaluation.clot_relaxed_metrics import (  # noqa: E402
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics_full_mesh, metrics_to_deploy_prefix,
)
from scripts.predict_wall_clot import predict_wall_clot  # noqa: E402

DIR = Path("data/processed/graphs_biochem_anchors")

# (anchor, flow arm to use -- "pred" where the pack has u0_pred, else "gt")
VESSELS = [
    ("patient012", "gt"),
    ("patient044", "gt"),
    ("patient042", "gt"),
    ("patient007", "pred"),
    ("patient032", "pred"),
]


def score_full_mesh(pred, gt, edge_index, wall):
    m = compute_clot_relaxed_metrics_full_mesh(
        torch.tensor(pred.astype(np.float32)), torch.tensor(gt.astype(np.float32)),
        edge_index, wall_mask=torch.tensor(wall))
    o = metrics_to_deploy_prefix(m)
    return clot_score_from_deploy_dict(o), o


def render(anchor: str, flow: str, out: Path, bio, phys):
    d = torch.load(DIR / f"{anchor}.pt", map_location="cpu", weights_only=False)
    pos = node_positions(d)
    wall = d.mask_wall.reshape(-1).bool().numpy()
    t_eval = resolve_deploy_eval_time_index(int(d.y.shape[0]))
    gt = (gt_clot_phi_at_time(d, t_eval, phys, device=torch.device("cpu"))
          .reshape(-1).numpy() > 0.5)

    pred_wall, _ = predict_wall_clot(d, bio, flow=flow, lumen=False)
    pred_wall = np.asarray(pred_wall).astype(bool)
    pred_full, _ = predict_wall_clot(d, bio, flow=flow, lumen=True)
    pred_full = np.asarray(pred_full).astype(bool)

    score_wall, o_wall = score_full_mesh(pred_wall, gt, d.edge_index, wall)
    score_full, o_full = score_full_mesh(pred_full, gt, d.edge_index, wall)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 6.0), dpi=150)
    interior = ~wall
    for ax, pred, title, score in (
        (axes[0], pred_wall, "wall arm only", score_wall),
        (axes[1], pred_full, "wall + lumen", score_full),
    ):
        tp = pred & gt
        fn = (~pred) & gt
        fp = pred & (~gt)
        tn = (~pred) & (~gt)
        ax.scatter(pos[interior & tn, 0], pos[interior & tn, 1], s=1.4,
                  c="#d8dde3", linewidths=0, zorder=1)

        def layer(mask_sel, on_wall_mask, color, label, marker):
            m = mask_sel & on_wall_mask
            ax.scatter(pos[m, 0], pos[m, 1], s=16 if marker == "s" else 14,
                      c=color, linewidths=0.25, edgecolors="white" if marker == "s" else "none",
                      marker=marker, zorder=4 if marker == "s" else 3, label=label)

        # wall nodes: circles.  off-wall (lumen) nodes: squares.
        layer(tn, wall, "#9aa5b1", None, "o")
        layer(fp, wall, "#f2a53a", f"FP wall", "o")
        layer(fn, wall, "#e0463f", f"FN wall", "o")
        layer(tp, wall, "#2f9e5c", f"TP wall", "o")
        layer(fp, interior, "#f2a53a", f"FP lumen", "s")
        layer(fn, interior, "#e0463f", f"FN lumen", "s")
        layer(tp, interior, "#2f9e5c", f"TP lumen", "s")

        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{title}\nscore={score:.3f}  (offwall relF1={o_full.get('deploy_clot_offwall_relaxed_f1', 0) if title!='wall arm only' else o_wall.get('deploy_clot_offwall_relaxed_f1', 0):.3f})",
                     fontsize=10)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2f9e5c", markersize=7, label="TP wall"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#2f9e5c", markersize=7, label="TP lumen"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#e0463f", markersize=7, label="FN wall"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#e0463f", markersize=7, label="FN lumen"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#f2a53a", markersize=7, label="FP wall"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#f2a53a", markersize=7, label="FP lumen"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8, frameon=False,
              bbox_to_anchor=(0.5, -0.02))
    flow_label = "predicted flow (deployable)" if flow == "pred" else "GT flow at t=0 (bandaid)"
    fig.suptitle(f"{anchor}  —  {flow_label}", fontsize=11, y=1.01)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    n_off_gt = int((gt & ~wall).sum())
    n_off_pred_full = int((pred_full & ~wall).sum())
    return dict(anchor=anchor, flow=flow, score_wall=score_wall, score_full=score_full,
               n_gt=int(gt.sum()), n_off_gt=n_off_gt, n_off_pred=n_off_pred_full,
               offwall_relf1_wall=o_wall.get("deploy_clot_offwall_relaxed_f1", 0.0),
               offwall_relf1_full=o_full.get("deploy_clot_offwall_relaxed_f1", 0.0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs/viz_phase6_offwall")
    args = ap.parse_args()
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    outdir = Path(args.outdir)
    rows = []
    for anchor, flow in VESSELS:
        out = outdir / f"{anchor}.png"
        r = render(anchor, flow, out, bio, phys)
        rows.append(r)
        print("%-12s flow=%-4s  offwall GT %3d/%3d (%.0f%%)  score wall-only %.4f -> wall+lumen %.4f  (%+.4f)  offwallF1 %.3f -> %.3f"
              % (r["anchor"], r["flow"], r["n_off_gt"], r["n_gt"],
                 100 * r["n_off_gt"] / max(r["n_gt"], 1), r["score_wall"], r["score_full"],
                 r["score_full"] - r["score_wall"], r["offwall_relf1_wall"], r["offwall_relf1_full"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
