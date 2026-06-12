"""Rung 1 vs Rung 2 timeline viz (mu + clot).

Rows:
  0 GT spf.mu
  1 Rung1 mu (COMSOL spf.sr)
  2 Rung2 mu (proxy gamma)
  3 GT clot phi
  4 Rung1 clot (mu + nucleation)
  5 Rung2 clot (mu + nucleation)

Usage::

    python scripts/viz_t0_rung12.py --anchor patient007
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.core_physics.t0_rung_config import (  # noqa: E402
    RUNG2_GAMMA_MODE,
    RUNG2_GAMMA_SCALE,
    RUNG2_POISEUILLE_SCALE,
)
from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND  # noqa: E402
from src.core_physics.clot_continuous_time import macro_tau_at_index  # noqa: E402
from src.core_physics.t0_clot_predictor import t0_gt_baseline_env  # noqa: E402
from src.core_physics.t0_mu_physics import (  # noqa: E402
    gt_clot_phi_at_time,
    predict_mu_si_at_time,
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
    ap = argparse.ArgumentParser(description="Rung 1/2 mu+clot viz")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--max-frames", type=int, default=10)
    ap.add_argument("--scatter-size", type=float, default=3.5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = get_project_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")

    graph_path = root / "data/processed/graphs_biochem_anchors" / f"{args.anchor}.pt"
    data = torch.load(graph_path, map_location=device, weights_only=False)
    pos = data.x[:, :2].detach().cpu().numpy()
    n_nodes = int(data.num_nodes)
    n_steps = int(data.y.shape[0])
    times = _pick_times(n_steps, int(args.max_frames))
    full_region = np.ones(n_nodes, dtype=bool)

    row_labels = [
        "GT spf.mu",
        "Rung1 mu (spf.sr)",
        f"Rung2 mu ({RUNG2_GAMMA_MODE} poi={RUNG2_POISEUILLE_SCALE})",
        "GT clot",
        "Rung1 clot+nuc",
        "Rung2 clot+nuc",
    ]
    nrows = len(row_labels)
    ncols = len(times)

    with t0_physics_env(args.anchor, gamma_mode="comsol_sr") as _env1:
        traj1 = rollout_t0_clot_phi(
            data, phys, bio, device, gamma_mode="comsol_sr", nucleation=True, nucleation_hops=1
        )
        steps1 = {
            t: predict_mu_si_at_time(data, t, phys, bio, device, gamma_mode="comsol_sr")
            for t in times
        }

    with t0_gt_baseline_env(
        gamma_mode=RUNG2_GAMMA_MODE,
        gamma_scale=RUNG2_GAMMA_SCALE,
        poiseuille_scale=RUNG2_POISEUILLE_SCALE,
    ):
        traj2 = rollout_t0_clot_phi(
            data,
            phys,
            bio,
            device,
            gamma_mode=RUNG2_GAMMA_MODE,
            nucleation=True,
            nucleation_hops=1,
        )
        steps2 = {
            t: predict_mu_si_at_time(data, t, phys, bio, device, gamma_mode=RUNG2_GAMMA_MODE)
            for t in times
        }

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.6 * ncols, 2.1 * nrows), squeeze=False
    )
    fig.suptitle(
        f"T0 Rung 1 vs 2 -- {args.anchor} | GT flow+species | R1=COMSOL sr | R2=proxy gamma",
        fontsize=10,
    )

    mu_vmax = 0.12
    frames_meta: list[dict] = []
    mask = torch.ones(n_nodes, device=device, dtype=torch.bool)

    for j, t in enumerate(times):
        y = data.y[int(t)].to(device)
        mu_gt = phys.viscosity_nd_to_si(y[:, STATE_CHANNEL_MU_EFF_ND]).detach().cpu().numpy()
        mu1 = steps1[t].mu_pred_si.detach().cpu().numpy()
        mu2 = steps2[t].mu_pred_si.detach().cpu().numpy()
        phi_gt = gt_clot_phi_at_time(data, int(t), phys, device).detach().cpu().numpy()
        phi1 = traj1[int(t)]["phi"].detach().cpu().numpy()
        phi2 = traj2[int(t)]["phi"].detach().cpu().numpy()
        tau = float(macro_tau_at_index(data, int(t), bio_cfg=bio))
        m1 = clot_trigger_viz_f1(traj1[int(t)]["phi"], gt_clot_phi_at_time(data, int(t), phys, device), mask)
        m2 = clot_trigger_viz_f1(traj2[int(t)]["phi"], gt_clot_phi_at_time(data, int(t), phys, device), mask)
        bulk = steps1[t].mu_gt_si < 0.012
        growth = steps1[t].mu_gt_si > 0.015
        ratio_g1 = float(
            (steps1[t].mu_gt_si[growth] / steps1[t].mu_pred_si[growth].clamp(min=1e-8)).median().item()
        ) if growth.any() else float("nan")
        ratio_g2 = float(
            (steps2[t].mu_gt_si[growth] / steps2[t].mu_pred_si[growth].clamp(min=1e-8)).median().item()
        ) if growth.any() else float("nan")

        col_title = (
            f"t={t} tau={tau:.2f}\n"
            f"R1 g_ratio={ratio_g1:.3f} F1={m1['clot_f1']:.2f}\n"
            f"R2 g_ratio={ratio_g2:.3f} F1={m2['clot_f1']:.2f}"
        )
        panels = [
            (mu_gt, "viridis", 0.0, mu_vmax),
            (mu1, "viridis", 0.0, mu_vmax),
            (mu2, "viridis", 0.0, mu_vmax),
            (phi_gt, "bwr", 0.0, 1.0),
            (phi1, "bwr", 0.0, 1.0),
            (phi2, "bwr", 0.0, 1.0),
        ]
        for i, (vals, cmap, vmin, vmax) in enumerate(panels):
            _scatter_fullmesh_region(
                axes[i, j],
                pos,
                vals,
                full_region,
                row_labels[i] if j == 0 else "",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                s=float(args.scatter_size),
                layer_positive_on_top=(i >= 3),
            )
        axes[0, j].set_title(col_title, fontsize=6)

        frames_meta.append(
            {
                "time": int(t),
                "tau": tau,
                "rung1_growth_mu_ratio": ratio_g1,
                "rung2_growth_mu_ratio": ratio_g2,
                "rung1_f1": float(m1["clot_f1"]),
                "rung2_f1": float(m2["clot_f1"]),
            }
        )

    fig.tight_layout()
    out_path = Path(args.out) if args.out.strip() else (
        root / f"outputs/biochem/viz/clot_trigger/t0_rung12_{args.anchor}.png"
    )
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")

    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                "anchor": args.anchor,
                "rung1": "comsol_sr",
                "rung2": {
                    "gamma_mode": RUNG2_GAMMA_MODE,
                    "gamma_scale": RUNG2_GAMMA_SCALE,
                    "poiseuille_scale": RUNG2_POISEUILLE_SCALE,
                },
                "frames": frames_meta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[save] {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
