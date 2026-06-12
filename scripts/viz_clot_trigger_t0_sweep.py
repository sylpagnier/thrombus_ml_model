"""Viz top T0 physics sweep legs for patient007 (+ optional second anchor)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_continuous_time import macro_tau_at_index
from src.core_physics.clot_phi_simple import build_clot_phi_step
from src.core_physics.clot_physics_trigger_sweep import apply_physics_sweep_leg, physics_sweep_legs
from src.core_physics.clot_trigger_rollout import lumen_false_positive_frac, rollout_clot_trigger_physics
from src.core_physics.neighbor_band_trigger import apply_physics_trigger_baseline_env
from src.evaluation.viz_clot_trigger import (
    apply_phi_viz_display,
    clot_trigger_viz_f1,
    scatter_clot_vessel,
)
from src.training.clot_trigger_stack import apply_clot_trigger_honest_env
from src.utils.channel_schema import BIO_Y_SCHEMA, assert_graph_schema, infer_missing_schema
from src.utils.paths import get_project_root


def _pick_times(n_steps: int, max_frames: int) -> list[int]:
    if max_frames <= 0 or n_steps <= max_frames:
        return list(range(n_steps))
    idx = np.linspace(0, n_steps - 1, num=max_frames, dtype=int)
    return sorted({int(i) for i in idx.tolist()})


def _leg_by_id(leg_id: str) -> dict:
    for leg in physics_sweep_legs():
        if leg["id"] == leg_id:
            return leg
    raise KeyError(f"unknown leg {leg_id}")


def _render_leg(
    leg: dict,
    data,
    *,
    anchor: str,
    device: torch.device,
    phys: PhysicsConfig,
    bio: BiochemConfig,
    max_frames: int,
    scatter_size: float,
    out_path: Path,
    subtract_t0: bool = False,
    display_min: float = 0.0,
) -> dict:
    apply_clot_trigger_honest_env()
    apply_physics_trigger_baseline_env()
    apply_physics_sweep_leg(leg)

    pos = data.x[:, :2].detach().cpu().numpy()
    n_steps = int(data.y.shape[0])
    full_ones = torch.ones(int(data.num_nodes), device=device, dtype=torch.bool)

    if leg.get("gt_mu_oracle"):
        traj = None
        phi_display_by_t = None
    else:
        traj = rollout_clot_trigger_physics(
            data, phys_cfg=phys, bio_cfg=bio, device=device, time_stride=1
        )
        phi_display_by_t = apply_phi_viz_display(
            {t: v["phi"] for t, v in traj.items()},
            subtract_t0=subtract_t0,
        )

    frames: list[dict] = []
    for t in _pick_times(n_steps, max_frames):
        step = build_clot_phi_step(data, t, phys, bio, device)
        support = step.loss_mask.reshape(-1).bool()
        phi_gt = step.phi_gt.reshape(-1)
        if traj is not None and phi_display_by_t is not None:
            phi_deploy = phi_display_by_t[int(t)]
            phi_eval = traj[int(t)]["phi"]
        else:
            phi_deploy = phi_gt
            phi_eval = phi_gt
        m_full = clot_trigger_viz_f1(phi_eval, phi_gt, full_ones)
        frames.append(
            {
                "t": int(t),
                "tau": float(macro_tau_at_index(data, int(t), bio_cfg=bio)),
                "phi_gt": phi_gt.detach().cpu().numpy(),
                "phi_deploy": phi_deploy.detach().cpu().numpy(),
                "f1_full": float(m_full["clot_f1"]),
                "lumen_fp": lumen_false_positive_frac(phi_eval, phi_gt, data=data, device=device),
            }
        )

    ncols = len(frames)
    fig, axes = plt.subplots(2, ncols, figsize=(max(2.5 * ncols, 10), 5.5), squeeze=False)
    suffix = ""
    if subtract_t0:
        suffix += " | viz: phi-phi_t0"
    if display_min > 0.0:
        suffix += f" | min={display_min:g}"
    fig.suptitle(
        f"T0 sweep {leg['id']} -- {anchor} | row0=GT | row1=deploy{suffix}",
        fontsize=10,
    )
    for j, fr in enumerate(frames):
        title = f"t={fr['t']} F1={fr['f1_full']:.2f} fp={fr['lumen_fp']:.2f}"
        scatter_clot_vessel(axes[0, j], pos, fr["phi_gt"], "GT" if j == 0 else "", scatter_size=scatter_size)
        scatter_clot_vessel(
            axes[1, j],
            pos,
            fr["phi_deploy"],
            leg["id"] if j == 0 else "",
            scatter_size=scatter_size,
            display_min=display_min,
        )
        axes[0, j].set_title(title, fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"leg_id": leg["id"], "anchor": anchor, "out": str(out_path), "frames": len(frames)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Viz top T0 physics sweep legs")
    ap.add_argument("--sweep-dir", default="outputs/biochem/clot_trigger/t0_physics_sweep")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--legs", default="", help="Override: comma-separated leg ids")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--anchor2", default="patient002")
    ap.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--max-frames", type=int, default=10)
    ap.add_argument("--scatter-size", type=float, default=4.0)
    ap.add_argument("--viz-dir", default="outputs/biochem/viz/clot_trigger/t0_sweep")
    ap.add_argument(
        "--subtract-t0",
        action="store_true",
        help="Viz only: display phi(t) - phi(t=0)",
    )
    ap.add_argument(
        "--display-min",
        type=float,
        default=0.0,
        help="Hide phi below this value in colormap (e.g. 0.5 = commit view)",
    )
    args = ap.parse_args()

    root = get_project_root()
    sweep_dir = root / args.sweep_dir
    index_path = sweep_dir / "sweep_index.json"
    if not index_path.is_file():
        print(f"[ERR] missing {index_path}; run sweep first", file=sys.stderr)
        return 2
    index = json.loads(index_path.read_text(encoding="utf-8"))

    if args.legs.strip():
        leg_ids = [x.strip() for x in args.legs.split(",") if x.strip()]
    else:
        leg_ids = [
            r["leg_id"]
            for r in index.get("ranking", [])
            if not r.get("gt_mu_oracle")
        ][: max(1, int(args.top_k))]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    viz_dir = root / args.viz_dir
    viz_dir.mkdir(parents=True, exist_ok=True)

    rendered: list[dict] = []
    for leg_id in leg_ids:
        leg = _leg_by_id(leg_id)
        for anc in [args.anchor, args.anchor2]:
            if not anc:
                continue
            graph_path = root / args.anchor_dir / f"{anc}.pt"
            data = torch.load(graph_path, map_location=device, weights_only=False)
            data = infer_missing_schema(data, phase_hint="biochem")
            assert_graph_schema(data, expected_y_schema=(BIO_Y_SCHEMA,))
            out_path = viz_dir / f"{leg_id}_{anc}.png"
            meta = _render_leg(
                leg,
                data,
                anchor=anc,
                device=device,
                phys=phys,
                bio=bio,
                max_frames=int(args.max_frames),
                scatter_size=float(args.scatter_size),
                out_path=out_path,
                subtract_t0=bool(args.subtract_t0),
                display_min=float(args.display_min),
            )
            rendered.append(meta)
            print(f"[save] {out_path}", flush=True)

    manifest_path = viz_dir / "viz_manifest.json"
    manifest_path.write_text(json.dumps({"rendered": rendered, "leg_ids": leg_ids}, indent=2), encoding="utf-8")
    print(f"[save] {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
