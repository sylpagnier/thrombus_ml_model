"""Render predicted-vs-GT wall clot maps for sealed (never-trained-on) vessels.

Produces one PNG per vessel: wall nodes colored TP/FN/FP/TN against a faint outline of
the full vessel mesh, for both flow arms.  Deploy-legal inputs only.
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
from src.core_physics.physics_wall_model import node_positions  # noqa: E402
from src.core_physics.species_pushforward_continuous import resolve_deploy_eval_time_index  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.evaluation.clot_relaxed_metrics import (  # noqa: E402
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
)
from scripts.predict_wall_clot import predict_wall_clot  # noqa: E402

DIR = Path("data/processed/graphs_biochem_anchors")


def render(anchor: str, flow: str, out: Path, bio, phys):
    d = torch.load(DIR / f"{anchor}.pt", map_location="cpu", weights_only=False)
    pos = node_positions(d)
    wall = d.mask_wall.reshape(-1).bool().numpy()
    pred, f = predict_wall_clot(d, bio, flow=flow)

    t_eval = resolve_deploy_eval_time_index(int(d.y.shape[0]))
    gt_t = gt_clot_phi_at_time(d, t_eval, phys, device=torch.device("cpu")).reshape(-1).numpy()
    gt = (gt_t > 0.5) & wall

    m = compute_clot_relaxed_metrics(torch.tensor(pred.astype(np.float32)),
                                     torch.tensor((gt_t * wall).astype(np.float32)),
                                     d.edge_index, wall_mask=torch.tensor(wall))
    o = metrics_to_deploy_prefix(m)
    score = clot_score_from_deploy_dict(o)

    tp = pred & gt
    fn = (~pred) & gt & wall
    fp = pred & (~gt) & wall
    tn = (~pred) & (~gt) & wall

    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=150)
    interior = ~wall
    ax.scatter(pos[interior, 0], pos[interior, 1], s=2, c="#d8dde3", linewidths=0, zorder=1)
    ax.scatter(pos[tn, 0], pos[tn, 1], s=7, c="#9aa5b1", linewidths=0, zorder=2, label="wall (correct: no clot)")
    ax.scatter(pos[fp, 0], pos[fp, 1], s=14, c="#f2a53a", linewidths=0, zorder=3, label=f"false positive ({int(fp.sum())})")
    ax.scatter(pos[fn, 0], pos[fn, 1], s=14, c="#e0463f", linewidths=0, zorder=4, label=f"missed clot ({int(fn.sum())})")
    ax.scatter(pos[tp, 0], pos[tp, 1], s=14, c="#2f9e5c", linewidths=0, zorder=5, label=f"correct clot ({int(tp.sum())})")
    ax.set_aspect("equal")
    ax.axis("off")
    flow_label = "GT flow at t=0 (bandaid)" if flow == "gt" else "predicted flow (deployable)"
    ax.set_title(f"{anchor}  —  {flow_label}\ndeploy_clot_score = {score:.3f}   "
                 f"(GT clot nodes: {int(gt.sum())}, wall: {int(wall.sum())})", fontsize=10)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9, markerscale=1.6)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return score, int(gt.sum()), int(pred.sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", default="patient001,patient007,patient010,patient013,patient014,patient031,patient042,patient043")
    ap.add_argument("--flow", default="pred", choices=["pred", "gt", "both"])
    ap.add_argument("--outdir", default="outputs/viz_generalization")
    args = ap.parse_args()
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    flows = ["gt", "pred"] if args.flow == "both" else [args.flow]
    outdir = Path(args.outdir)
    for a in args.anchors.split(","):
        a = a.strip()
        for fl in flows:
            out = outdir / f"{a}_{fl}.png"
            try:
                score, ngt, npred = render(a, fl, out, bio, phys)
            except ValueError as e:
                print(f"  {a} [{fl}]  skip: {e}")
                continue
            print(f"  {a:12s} [{fl:4s}]  score={score:.4f}  gt={ngt:4d}  pred={npred:4d}  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
