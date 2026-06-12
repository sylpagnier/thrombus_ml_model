"""T0 mu physics clot viz: GT vs physics prediction (0-1 bwr, timeline rows).

Row 0 = GT clot from spf.mu growth (label only).
Row 1+ = clot from physics mu (GT flow + species, no GT mu input).

Layout matches ``viz_clot_trigger_t0_oracle`` / ``viz_clot_ml_v32_sweep``.

Usage::

    python scripts/viz_t0_mu_physics.py --anchor patient007
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.clot_continuous_time import macro_tau_at_index  # noqa: E402
from src.core_physics.t0_mu_physics import (  # noqa: E402
    gt_clot_phi_at_time,
    metrics_for_step,
    predict_clot_phi_at_time,
    rollout_t0_clot_phi,
    t0_physics_env,
)
from src.evaluation.viz_clot_phi_simple import _scatter_fullmesh_region  # noqa: E402
from src.evaluation.viz_clot_trigger import clot_trigger_viz_f1  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402


def _pick_times(n_steps: int, max_frames: int) -> list[int]:
    if max_frames <= 0 or n_steps <= max_frames:
        return list(range(n_steps))
    idx = np.linspace(0, n_steps - 1, num=max_frames, dtype=int)
    return sorted({int(i) for i in idx.tolist()})


def main() -> int:
    ap = argparse.ArgumentParser(description="T0 mu physics clot timeline viz")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--max-frames", type=int, default=10)
    ap.add_argument("--scatter-size", type=float, default=4.0)
    ap.add_argument("--gamma-mode", default="auto")
    ap.add_argument("--no-hard-step", action="store_true")
    ap.add_argument("--pred-label", default="", help="Row-1 legend (default: gamma mode)")
    ap.add_argument("--nucleation-row", action="store_true", help="Add row: nucleation-projected phi")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = get_project_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")

    graph_path = root / args.anchor_dir / f"{args.anchor}.pt"
    data = torch.load(graph_path, map_location=device, weights_only=False)
    pos = data.x[:, :2].detach().cpu().numpy()
    n_nodes = int(data.num_nodes)
    n_steps = int(data.y.shape[0])
    times = _pick_times(n_steps, int(args.max_frames))
    full_region = np.ones(n_nodes, dtype=bool)

    gamma_arg = None if args.gamma_mode == "auto" else args.gamma_mode
    hard_step = not args.no_hard_step

    with t0_physics_env(args.anchor, gamma_mode=gamma_arg, hard_step=hard_step) as physics:
        gamma_mode = physics["gamma_mode"]
        pred_label = (args.pred_label or f"mu physics ({gamma_mode})").strip()
        use_nuc = bool(args.nucleation_row)
        traj_nuc = (
            rollout_t0_clot_phi(
                data,
                phys,
                bio,
                device,
                gamma_mode=gamma_mode,
                nucleation=True,
                nucleation_hops=1,
                use_dgamma_wall_seed=False,
            )
            if use_nuc
            else None
        )

        frames: list[dict] = []
        for t in times:
            phi_gt = gt_clot_phi_at_time(data, int(t), phys, device)
            phi_pred, step = predict_clot_phi_at_time(
                data, int(t), phys, bio, device, gamma_mode=gamma_mode
            )
            phi_nuc = traj_nuc[int(t)]["phi"] if traj_nuc is not None else None
            full_mask = torch.ones(n_nodes, device=device, dtype=torch.bool)
            m = clot_trigger_viz_f1(phi_pred, phi_gt, full_mask)
            m_nuc = (
                clot_trigger_viz_f1(phi_nuc, phi_gt, full_mask) if phi_nuc is not None else None
            )
            frames.append(
                {
                    "t": int(t),
                    "tau": float(macro_tau_at_index(data, int(t), bio_cfg=bio)),
                    "phi_gt": phi_gt.detach().cpu().numpy(),
                    "phi_pred": phi_pred.detach().cpu().numpy(),
                    "phi_nuc": phi_nuc.detach().cpu().numpy() if phi_nuc is not None else None,
                    "metrics": {
                        "f1": float(m["clot_f1"]),
                        "prec": float(m["clot_prec"]),
                        "rec": float(m["clot_rec"]),
                        "pred_pos_frac": float(m["pred_pos_frac"]),
                        "gt_pos_frac": float(m["gt_pos_frac"]),
                        "pearson_growth_mu": float(
                            metrics_for_step(step, data, phys, device).pearson_growth
                        ),
                        "f1_nuc": float(m_nuc["clot_f1"]) if m_nuc else float("nan"),
                    },
                }
            )

        nrows = 3 if use_nuc else 2
        ncols = len(frames)
        fig, axes = plt.subplots(nrows, ncols, figsize=(max(2.5 * ncols, 10), 2.8 * nrows), squeeze=False)
        row1 = "nucleation wall+1hop" if use_nuc else pred_label
        fig.suptitle(
            f"T0 mu physics -- {args.anchor} | row0=GT | row1={pred_label}"
            + (" | row2=nucleation" if use_nuc else ""),
            fontsize=11,
        )

        for j, fr in enumerate(frames):
            mx = fr["metrics"]
            title = (
                f"t={fr['t']} tau={fr['tau']:.2f}\n"
                f"F1={mx['f1']:.2f} prec={mx['prec']:.2f} rec={mx['rec']:.2f}\n"
                f"pred+={mx['pred_pos_frac']:.3f} gt+={mx['gt_pos_frac']:.3f}"
            )
            if use_nuc and math.isfinite(mx.get("f1_nuc", float("nan"))):
                title += f"\nF1_nuc={mx['f1_nuc']:.2f}"
            rows_data = [
                (fr["phi_gt"], "GT" if j == 0 else ""),
                (fr["phi_pred"], pred_label if j == 0 else ""),
            ]
            if use_nuc and fr["phi_nuc"] is not None:
                rows_data.append((fr["phi_nuc"], "nucleation" if j == 0 else ""))
            for i, (vals, ylab) in enumerate(rows_data):
                _scatter_fullmesh_region(
                    axes[i, j],
                    pos,
                    vals,
                    full_region,
                    ylab,
                    cmap="bwr",
                    vmin=0,
                    vmax=1,
                    s=float(args.scatter_size),
                    layer_positive_on_top=True,
                )
            axes[0, j].set_title(title, fontsize=7)

    fig.tight_layout()
    out_path = Path(args.out) if args.out.strip() else (
        root / f"outputs/biochem/viz/clot_trigger/t0_mu_{args.anchor}.png"
    )
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")

    summary_path = out_path.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "anchor": args.anchor,
                "gamma_mode": gamma_mode,
                "ratio_max": bio.mu_ratio_max,
                "physics": physics,
                "pred_label": pred_label,
                "frames": [{k: v for k, v in fr.items() if k not in ("phi_gt", "phi_pred")} for fr in frames],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[save] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
