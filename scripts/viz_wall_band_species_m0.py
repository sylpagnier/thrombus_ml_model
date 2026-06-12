"""Viz wall-band M0: GT clot | R4.s0 | M0.<channel_set>."""

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
from src.core_physics.clot_continuous_time import macro_tau_at_index  # noqa: E402
from src.core_physics.t0_device import require_cuda_device  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.core_physics.t0_rung4_ladder import rollout_rung4_phi_trajectory  # noqa: E402
from src.core_physics.wall_band_species_m0 import load_m0_bundle, rollout_m0_phi_trajectory  # noqa: E402
from src.evaluation.viz_clot_phi_simple import _scatter_fullmesh_region  # noqa: E402
from src.evaluation.viz_clot_trigger import clot_trigger_viz_f1  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402


def _pick_times(n_steps: int, max_frames: int) -> list[int]:
    if max_frames <= 0 or n_steps <= max_frames:
        return list(range(n_steps))
    idx = np.linspace(0, n_steps - 1, num=max_frames, dtype=int)
    return sorted({int(i) for i in idx.tolist()})


def main() -> int:
    ap = argparse.ArgumentParser(description="Viz wall-band M0 clot phi")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--max-frames", type=int, default=10)
    ap.add_argument("--scatter-size", type=float, default=3.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = require_cuda_device()
    root = get_project_root()
    bundle = load_m0_bundle(Path(args.ckpt) if args.ckpt.strip() else None, device=device)
    if bundle is None:
        raise SystemExit("[ERR] missing M0 checkpoint")

    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    data = torch.load(
        root / "data/processed/graphs_biochem_anchors" / f"{args.anchor}.pt",
        map_location=device,
        weights_only=False,
    )
    pos = data.x[:, :2].detach().cpu().numpy()
    n_nodes = int(data.num_nodes)
    times = _pick_times(int(data.y.shape[0]), int(args.max_frames))
    full_region = np.ones(n_nodes, dtype=bool)
    mask = torch.ones(n_nodes, device=device, dtype=torch.bool)

    print(f"[i] CUDA: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"[i] M0 channel_set={bundle.channel_set} channels={bundle.channel_names}", flush=True)

    t0 = time.perf_counter()
    phi_m0 = rollout_m0_phi_trajectory(data, bundle, phys_cfg=phys, bio_cfg=bio, device=device)
    phi_s0 = rollout_rung4_phi_trajectory(data, phys, bio, device, step="s0")
    print(f"[i] rollout {time.perf_counter() - t0:.1f}s", flush=True)

    row_labels = [
        "GT clot",
        "R4.s0 (rules)",
        f"M0.{bundle.channel_set}",
    ]
    fig, axes = plt.subplots(
        len(row_labels), len(times),
        figsize=(2.5 * len(times), 2.2 * len(row_labels)),
        squeeze=False,
    )
    fig.suptitle(
        f"Wall-band M0 -- {args.anchor} | {bundle.channel_set} "
        f"({','.join(bundle.channel_names)}) | CUDA",
        fontsize=9,
    )

    frames: list[dict] = []
    for j, t in enumerate(times):
        phi_gt = gt_clot_phi_at_time(data, int(t), phys, device)
        p_s0 = phi_s0[int(t)]
        p_m0 = phi_m0[int(t)]
        tau = float(macro_tau_at_index(data, int(t), bio_cfg=bio))
        m_s0 = clot_trigger_viz_f1(p_s0, phi_gt, mask)
        m_m0 = clot_trigger_viz_f1(p_m0, phi_gt, mask)
        title = f"t={t} tau={tau:.2f}\ns0={m_s0['clot_f1']:.2f} M0={m_m0['clot_f1']:.2f}"
        panels = [
            phi_gt.detach().cpu().numpy(),
            p_s0.detach().cpu().numpy(),
            p_m0.detach().cpu().numpy(),
        ]
        for i, vals in enumerate(panels):
            _scatter_fullmesh_region(
                axes[i, j], pos, vals, full_region,
                row_labels[i] if j == 0 else "",
                cmap="bwr", vmin=0.0, vmax=1.0,
                s=float(args.scatter_size), layer_positive_on_top=True,
            )
        axes[0, j].set_title(title, fontsize=5)
        frames.append({
            "time": int(t), "tau": tau,
            "s0_f1": float(m_s0["clot_f1"]),
            "m0_f1": float(m_m0["clot_f1"]),
        })

    fig.tight_layout()
    out = Path(args.out) if args.out.strip() else (
        root / f"outputs/biochem/viz/clot_trigger/wall_band_m0_{bundle.channel_set}_{args.anchor}.png"
    )
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out}")
    meta = out.with_suffix(".json")
    meta.write_text(
        json.dumps({
            "anchor": args.anchor,
            "channel_set": bundle.channel_set,
            "channel_names": bundle.channel_names,
            "frames": frames,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"[save] {meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
