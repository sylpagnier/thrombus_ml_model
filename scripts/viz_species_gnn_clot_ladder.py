"""GT clot | inc40 rules | Species GNN clot phi ladder (CUDA).

Three rows x time columns, matching ``viz_t0_rung4_step`` / ``viz_wall_band_species_m0``.
"""

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
from src.core_physics.species_gnn_clot_rollout import (  # noqa: E402
    load_species_gnn_rollout_bundle,
    prepare_species_gnn_rollout_static,
    rollout_species_gnn_phi_trajectory,
)
from src.core_physics.species_gnn_ladder_viz import ladder_viz_times  # noqa: E402
from src.core_physics.species_snapshot_gnn import species_gnn_viz_dir  # noqa: E402
from src.core_physics.t0_device import require_cuda_device  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.core_physics.clot_temporal_growth_rules import rollout_inc40_phi_trajectory  # noqa: E402
from src.evaluation.viz_clot_phi_simple import _scatter_fullmesh_region  # noqa: E402
from src.evaluation.viz_clot_trigger import clot_trigger_viz_f1  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Species GNN clot phi ladder (GT | inc40 | GNN)")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--max-frames", type=int, default=10)
    ap.add_argument("--scatter-size", type=float, default=3.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = require_cuda_device()
    root = get_project_root()
    ckpt = Path(args.ckpt) if args.ckpt.strip() else None
    bundle = load_species_gnn_rollout_bundle(ckpt, device=device)
    if bundle is None:
        raise SystemExit("[ERR] missing species GNN checkpoint (biochem_gnn baseline)")

    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    data = torch.load(
        root / "data/processed/graphs_biochem_anchors" / f"{args.anchor}.pt",
        map_location=device,
        weights_only=False,
    )
    pos = data.x[:, :2].detach().cpu().numpy()
    n_nodes = int(data.num_nodes)
    times = ladder_viz_times(int(data.y.shape[0]), max_frames=int(args.max_frames))
    full_region = np.ones(n_nodes, dtype=bool)
    mask = torch.ones(n_nodes, device=device, dtype=torch.bool)

    gnn_tag = f"SpeciesGNN.{bundle.label}"
    print(f"[i] CUDA: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"[i] rollout kind={bundle.kind} label={bundle.label}", flush=True)

    t0 = time.perf_counter()
    static = prepare_species_gnn_rollout_static(data, device=device)
    phi_gnn = rollout_species_gnn_phi_trajectory(
        data, bundle, static, phys_cfg=phys, bio_cfg=bio, device=device,
    )
    phi_rules = rollout_inc40_phi_trajectory(data, phys, bio, device, vel_source="kinematics")
    print(f"[i] rollout {time.perf_counter() - t0:.1f}s", flush=True)

    row_labels = [
        "GT clot",
        "inc40 rules",
        gnn_tag,
    ]
    fig, axes = plt.subplots(
        len(row_labels), len(times),
        figsize=(2.5 * len(times), 2.2 * len(row_labels)),
        squeeze=False,
    )
    fig.suptitle(
        f"Species GNN clot ladder -- {args.anchor} | {gnn_tag} ({bundle.kind}) | CUDA",
        fontsize=9,
    )

    frames: list[dict] = []
    for j, t in enumerate(times):
        phi_gt = gt_clot_phi_at_time(data, int(t), phys, device)
        p_rules = phi_rules[int(t)]
        p_gnn = phi_gnn[int(t)]
        tau = float(macro_tau_at_index(data, int(t), bio_cfg=bio))
        m_rules = clot_trigger_viz_f1(p_rules, phi_gt, mask)
        m_gnn = clot_trigger_viz_f1(p_gnn, phi_gt, mask)
        title = f"t={t} tau={tau:.2f}\ninc40={m_rules['clot_f1']:.2f} gnn={m_gnn['clot_f1']:.2f}"
        panels = [
            phi_gt.detach().cpu().numpy(),
            p_rules.detach().cpu().numpy(),
            p_gnn.detach().cpu().numpy(),
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
            "inc40_f1": float(m_rules["clot_f1"]),
            "gnn_f1": float(m_gnn["clot_f1"]),
        })

    fig.tight_layout()
    if args.out.strip():
        out = Path(args.out)
    else:
        out = species_gnn_viz_dir() / f"clot_ladder_{bundle.label}_{args.anchor}.png"
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
            "kind": bundle.kind,
            "label": bundle.label,
            "times": times,
            "frames": frames,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"[save] {meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
