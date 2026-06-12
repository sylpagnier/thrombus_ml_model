"""GT vs GNN FI/Mat species ladder on the same time grid as clot ladder viz.

Four rows (GT FI | Pred FI | GT Mat | Pred Mat) x ``max_frames`` time columns.
Uses full-timeline closed-loop rollout (not a short unroll window).
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
    rollout_species_gnn_species_series,
)
from src.core_physics.species_gnn_ladder_viz import ladder_viz_times  # noqa: E402
from src.core_physics.species_pushforward_continuous import log_state_to_active  # noqa: E402
from src.core_physics.species_snapshot_gnn import (  # noqa: E402
    fi_mat_active_labels,
    fi_mat_log_targets,
    species_gnn_viz_dir,
    trigger_metrics,
)
from src.core_physics.t0_device import require_cuda_device  # noqa: E402
from src.evaluation.viz_clot_phi_simple import _scatter_fullmesh_region  # noqa: E402
from src.training.biochem_species_scope import FI_CHANNEL, MAT_CHANNEL  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402

CH_NAMES = ("FI", "Mat")


def _full_band(n: int, node_idx: torch.Tensor, band_vals: torch.Tensor, *, ch: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.float32)
    idx = node_idx.reshape(-1).cpu().numpy()
    out[idx] = band_vals[:, ch].reshape(-1).detach().cpu().numpy()
    return out


def _band_log_fi_mat(y_t: torch.Tensor, node_idx: torch.Tensor) -> torch.Tensor:
    """FI/Mat log1p ND on band nodes from full ``y[t]`` (channels ``4:16``)."""
    sp = y_t[:, 4:16]
    return torch.stack([sp[:, FI_CHANNEL], sp[:, MAT_CHANNEL]], dim=-1)[node_idx]


def _band_active_from_series(
    y_t: torch.Tensor,
    node_idx: torch.Tensor,
    *,
    deploy_readout: bool = True,
) -> torch.Tensor:
    log_fi_mat = _band_log_fi_mat(y_t, node_idx)
    if deploy_readout:
        return log_state_to_active(log_fi_mat)
    return fi_mat_active_labels(log_fi_mat)


def main() -> int:
    ap = argparse.ArgumentParser(description="Species GNN FI/Mat ladder (GT vs pred, clot time grid)")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--max-frames", type=int, default=10)
    ap.add_argument("--scatter-size", type=float, default=5.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = require_cuda_device()
    root = get_project_root()
    ckpt = Path(args.ckpt) if args.ckpt.strip() else None
    bundle = load_species_gnn_rollout_bundle(ckpt, device=device)
    if bundle is None:
        raise SystemExit("[ERR] missing species GNN checkpoint")

    bio = BiochemConfig(phase="biochem")
    data = torch.load(
        root / "data/processed/graphs_biochem_anchors" / f"{args.anchor}.pt",
        map_location=device,
        weights_only=False,
    )
    n_steps = int(data.y.shape[0])
    n_nodes = int(data.num_nodes)
    times = ladder_viz_times(n_steps, max_frames=int(args.max_frames))
    pos = data.x[:, :2].detach().cpu().numpy()
    gnn_tag = f"SpeciesGNN.{bundle.label}"
    print(f"[i] CUDA: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"[i] rollout kind={bundle.kind} label={bundle.label} times={times}", flush=True)

    t0 = time.perf_counter()
    static = prepare_species_gnn_rollout_static(data, device=device)
    node_idx = static.node_idx
    band_np = static.band.detach().cpu().numpy()
    pred_series = rollout_species_gnn_species_series(
        data, bundle, static, device=device,
    )
    print(f"[i] species rollout {time.perf_counter() - t0:.1f}s", flush=True)

    row_labels = [
        "GT FI",
        gnn_tag + " FI",
        "GT Mat",
        gnn_tag + " Mat",
    ]
    fig, axes = plt.subplots(
        len(row_labels), len(times),
        figsize=(2.5 * len(times), 2.0 * len(row_labels)),
        squeeze=False,
    )
    fig.suptitle(
        f"Species GNN species ladder -- {args.anchor} | {gnn_tag} ({bundle.kind}) | CUDA",
        fontsize=9,
    )

    full_mask = torch.ones(int(node_idx.numel()), device=device, dtype=torch.bool)
    frames: list[dict] = []

    for j, t in enumerate(times):
        ti = int(t)
        gt_log = fi_mat_log_targets(data, ti, device)[node_idx]
        gt_active = log_state_to_active(gt_log)
        pr_active = _band_active_from_series(pred_series[ti], node_idx, deploy_readout=True)
        tau = float(macro_tau_at_index(data, ti, bio_cfg=bio))
        m = trigger_metrics(pr_active, gt_active, full_mask)
        title = (
            f"t={ti} tau={tau:.2f}\n"
            f"FI={m['fi_f1']:.2f} Mat={m['mat_f1']:.2f} trig={m['trigger_f1']:.2f}"
        )
        panels: list[tuple[torch.Tensor, int, str]] = [
            (gt_active, 0, "fi_f1"),
            (pr_active, 0, "fi_f1"),
            (gt_active, 1, "mat_f1"),
            (pr_active, 1, "mat_f1"),
        ]
        for i, (vals, ch, metric_key) in enumerate(panels):
            f1 = float(m[metric_key])
            _scatter_fullmesh_region(
                axes[i, j],
                pos,
                _full_band(n_nodes, node_idx, vals, ch=ch),
                band_np,
                row_labels[i] if j == 0 else "",
                cmap="Reds",
                vmin=0.0,
                vmax=1.0,
                s=float(args.scatter_size),
                mask_outside_region=True,
            )
            if i in (1, 3):
                axes[i, j].set_ylabel(f"F1={f1:.2f}", fontsize=5)
        axes[0, j].set_title(title, fontsize=5)
        frames.append({
            "time": ti,
            "tau": tau,
            "fi_f1": float(m["fi_f1"]),
            "mat_f1": float(m["mat_f1"]),
            "trigger_f1": float(m["trigger_f1"]),
        })

    fig.tight_layout()
    if args.out.strip():
        out = Path(args.out)
    else:
        out = species_gnn_viz_dir() / f"species_ladder_{bundle.label}_{args.anchor}.png"
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
