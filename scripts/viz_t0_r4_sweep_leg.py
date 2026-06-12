"""Viz one T0 Rung4 sweep leg: GT clot | R4.s0 deploy | R4.<leg>."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.t0_device import require_cuda_device  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.core_physics.t0_mu_physics import rollout_t0_clot_phi  # noqa: E402
from src.core_physics.t0_r4_sweep import load_sweep_bundle, rollout_sweep_species_series  # noqa: E402
from src.core_physics.t0_rung4_ladder import rollout_rung4_species_series  # noqa: E402
from src.core_physics.t0_rung_config import RUNG2_GAMMA_MODE, t0_rung2_env  # noqa: E402
from src.evaluation.viz_clot_phi_simple import _scatter_fullmesh_region  # noqa: E402
from src.evaluation.viz_clot_trigger import clot_trigger_viz_f1  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402


def _phi_traj_from_species(data, phys, bio, device, pred_series):
    with t0_rung2_env():
        traj = rollout_t0_clot_phi(
            data, phys, bio, device,
            gamma_mode=RUNG2_GAMMA_MODE, flow_source="gt",
            pred_species_series=pred_series, nucleation=True, nucleation_hops=1,
        )
    return {int(t): v["phi"] for t, v in traj.items()}


def _pick_times(n_steps: int, max_frames: int) -> list[int]:
    if max_frames <= 0 or n_steps <= max_frames:
        return list(range(n_steps))
    idx = np.linspace(0, n_steps - 1, num=max_frames, dtype=int)
    return sorted({int(i) for i in idx.tolist()})


def main() -> int:
    ap = argparse.ArgumentParser(description="Viz T0 Rung4 sweep leg")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--max-frames", type=int, default=10)
    ap.add_argument("--scatter-size", type=float, default=3.0)
    ap.add_argument("--leg-dir", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--teacher-ckpt", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = require_cuda_device()
    root = get_project_root()
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")

    if args.ckpt.strip():
        ckpt = Path(args.ckpt)
    else:
        ckpt = Path(args.leg_dir) / "best.pth"
    if not ckpt.is_absolute():
        ckpt = root / ckpt
    leg_dir = ckpt.parent
    leg_id = leg_dir.name

    bundle = load_sweep_bundle(ckpt, device=device)
    if bundle is None:
        raise SystemExit(f"[ERR] missing checkpoint: {ckpt}")

    graph_path = root / "data/processed/graphs_biochem_anchors" / f"{args.anchor}.pt"
    data = torch.load(graph_path, map_location=device, weights_only=False)
    pos = data.x[:, :2].detach().cpu().numpy()
    n_nodes = int(data.num_nodes)
    times = _pick_times(int(data.y.shape[0]), int(args.max_frames))
    full_region = np.ones(n_nodes, dtype=bool)

    mask = torch.ones(n_nodes, device=device, dtype=torch.bool)

    print(f"[i] CUDA: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"[i] recipe={bundle.recipe.id} family={bundle.recipe.family}", flush=True)

    t0 = time.perf_counter()
    pred = rollout_sweep_species_series(data, phys, bio, device, bundle)
    pred_s0 = rollout_rung4_species_series(data, phys, bio, device, step="s0")
    print(f"[i] sweep rollout {time.perf_counter() - t0:.1f}s", flush=True)

    phi_gt_traj = {
        int(t): gt_clot_phi_at_time(data, int(t), phys, device)
        for t in times
    }
    phi_s0_traj = _phi_traj_from_species(data, phys, bio, device, pred_s0)
    phi_leg_traj = _phi_traj_from_species(data, phys, bio, device, pred)

    phi_rows = [
        ("GT clot", phi_gt_traj),
        ("R4.s0 (rules)", phi_s0_traj),
        (f"R4.{leg_id}", phi_leg_traj),
    ]

    n_rows = len(phi_rows)
    n_cols = len(times)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.4 * n_rows), squeeze=False)
    fig.suptitle(
        f"T0 Rung4 arch sweep -- {args.anchor} | {leg_id} ({bundle.recipe.family}) | CUDA",
        fontsize=11,
    )

    for ri, (label, phi_traj) in enumerate(phi_rows):
        for ci, t in enumerate(times):
            ax = axes[ri, ci]
            phi = phi_traj[int(t)].reshape(-1).detach().cpu().numpy()
            title = ""
            if ri == 0:
                phi_gt = phi_gt_traj[int(t)]
                m_s0 = clot_trigger_viz_f1(phi_s0_traj[int(t)], phi_gt, mask)
                m_leg = clot_trigger_viz_f1(phi_leg_traj[int(t)], phi_gt, mask)
                title = f"t={t}\ns0={m_s0['clot_f1']:.2f} {leg_id}={m_leg['clot_f1']:.2f}"
            _scatter_fullmesh_region(
                ax, pos, phi, full_region, title,
                cmap="bwr", vmin=0.0, vmax=1.0,
                s=float(args.scatter_size), layer_positive_on_top=True,
            )
            if ci == 0:
                ax.set_ylabel(label, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    out_png = Path(args.out) if args.out.strip() else (
        root / "outputs/biochem/viz/sweep_t0_r4_arch_6h" / f"{leg_id}_{args.anchor}.png"
    )
    if not out_png.is_absolute():
        out_png = root / out_png
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_png}", flush=True)

    meta = {
        "anchor": args.anchor,
        "leg_id": leg_id,
        "recipe_id": bundle.recipe.id,
        "png": str(out_png),
    }
    side = out_png.with_suffix(".json")
    side.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
