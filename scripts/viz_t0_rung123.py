"""Rung 2 vs Rung 3 clot comparison (GT reference row).

Rows:
  0 GT clot phi
  1 Rung2 clot+nuc (GT flow + proxy gamma)
  2 Rung3 clot+nuc (pred kine + proxy gamma)

Usage::

    python scripts/viz_t0_rung123.py --anchor patient007
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

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.clot_continuous_time import macro_tau_at_index  # noqa: E402
from src.core_physics.t0_mu_physics import (  # noqa: E402
    gt_clot_phi_at_time,
    rollout_t0_clot_phi,
)
from src.core_physics.t0_rung_config import (  # noqa: E402
    DEFAULT_KINE_CKPT,
    RUNG2_GAMMA_MODE,
    RUNG2_POISEUILLE_SCALE,
    t0_rung2_env,
    t0_rung3_env,
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
    ap = argparse.ArgumentParser(description="GT vs Rung2 vs Rung3 clot viz")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--max-frames", type=int, default=10)
    ap.add_argument("--scatter-size", type=float, default=3.5)
    ap.add_argument("--kine-ckpt", default=DEFAULT_KINE_CKPT)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = get_project_root()
    kine = root / args.kine_ckpt
    if not kine.is_file():
        print(f"[ERR] missing kinematics ckpt {kine}", file=sys.stderr)
        return 1

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
    mask = torch.ones(n_nodes, device=device, dtype=torch.bool)

    row_labels = [
        "GT clot",
        f"Rung2 clot+nuc (GT flow, {RUNG2_GAMMA_MODE})",
        f"Rung3 clot+nuc (pred kine, {RUNG2_GAMMA_MODE})",
    ]
    nrows = len(row_labels)
    ncols = len(times)

    with t0_rung2_env():
        traj2 = rollout_t0_clot_phi(
            data,
            phys,
            bio,
            device,
            gamma_mode=RUNG2_GAMMA_MODE,
            flow_source="gt",
            nucleation=True,
            nucleation_hops=1,
        )

    with t0_rung3_env(kine_ckpt=str(kine)):
        traj3 = rollout_t0_clot_phi(
            data,
            phys,
            bio,
            device,
            gamma_mode=RUNG2_GAMMA_MODE,
            flow_source="kinematics",
            nucleation=True,
            nucleation_hops=1,
        )

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.6 * ncols, 2.4 * nrows), squeeze=False
    )
    fig.suptitle(
        f"T0 clot ladder -- {args.anchor} | GT vs R2 (GT flow) vs R3 (pred kine)",
        fontsize=10,
    )

    frames_meta: list[dict] = []
    for j, t in enumerate(times):
        phi_gt = gt_clot_phi_at_time(data, int(t), phys, device)
        phi2 = traj2[int(t)]["phi"]
        phi3 = traj3[int(t)]["phi"]
        tau = float(macro_tau_at_index(data, int(t), bio_cfg=bio))
        m2 = clot_trigger_viz_f1(phi2, phi_gt, mask)
        m3 = clot_trigger_viz_f1(phi3, phi_gt, mask)

        col_title = (
            f"t={t} tau={tau:.2f}\n"
            f"R2 F1={m2['clot_f1']:.2f} rec={m2['clot_rec']:.2f}\n"
            f"R3 F1={m3['clot_f1']:.2f} rec={m3['clot_rec']:.2f}"
        )

        panels = [
            phi_gt.detach().cpu().numpy(),
            phi2.detach().cpu().numpy(),
            phi3.detach().cpu().numpy(),
        ]
        for i, vals in enumerate(panels):
            _scatter_fullmesh_region(
                axes[i, j],
                pos,
                vals,
                full_region,
                row_labels[i] if j == 0 else "",
                cmap="bwr",
                vmin=0.0,
                vmax=1.0,
                s=float(args.scatter_size),
                layer_positive_on_top=True,
            )
        axes[0, j].set_title(col_title, fontsize=6)

        frames_meta.append(
            {
                "time": int(t),
                "tau": tau,
                "rung2_f1": float(m2["clot_f1"]),
                "rung2_rec": float(m2["clot_rec"]),
                "rung3_f1": float(m3["clot_f1"]),
                "rung3_rec": float(m3["clot_rec"]),
            }
        )

    fig.tight_layout()
    out_path = Path(args.out) if args.out.strip() else (
        root / f"outputs/biochem/viz/clot_trigger/t0_rung123_{args.anchor}.png"
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
                "kine_ckpt": str(kine.relative_to(root)) if kine.is_relative_to(root) else str(kine),
                "rung2": {
                    "flow": "gt",
                    "gamma_mode": RUNG2_GAMMA_MODE,
                    "poiseuille_scale": RUNG2_POISEUILLE_SCALE,
                },
                "rung3": {
                    "flow": "kinematics",
                    "gamma_mode": RUNG2_GAMMA_MODE,
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
