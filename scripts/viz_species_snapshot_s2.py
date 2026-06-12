"""Viz Phase 2 pushforward: GT vs pred cumulative state after unrolled rollout."""

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

from src.config import PhysicsConfig, VesselConfig  # noqa: E402
from src.core_physics.clot_phi_simple import sdf_nd_from_data  # noqa: E402
from src.core_physics.species_pushforward_gnn import (  # noqa: E402
    active_series_on_band,
    growth_active_labels,
    growth_metrics,
    load_pushforward_bundle,
    rollout_pushforward_states,
)
from src.core_physics.species_snapshot_gnn import (  # noqa: E402
    build_snapshot_features,
    induced_subgraph,
    species_gnn_viz_dir,
    species_gnn_viz_stem,
    trigger_metrics,
    wall_band_mask,
)
from src.evaluation.viz_clot_phi_simple import _scatter_fullmesh_region  # noqa: E402
from src.utils.kinematics_inference import (  # noqa: E402
    load_kinematics_predictor,
    predict_kinematics_latent,
    resolve_kinematics_checkpoint,
)
from src.utils.paths import get_project_root  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Species pushforward phase 2 viz")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--t0", type=int, default=28, help="Rollout start time index (28 -> ~4950s for patient007)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = get_project_root()
    ckpt = args.ckpt.strip() or str(root / "outputs/biochem/species_snapshot_s2/best.pth")
    bundle = load_pushforward_bundle(ckpt, device=device)
    if bundle is None:
        print(f"[ERR] missing checkpoint: {ckpt}", flush=True)
        return 1

    data = torch.load(
        root / VesselConfig(phase="biochem_anchors").graph_output_dir / f"{args.anchor}.pt",
        map_location=device,
        weights_only=False,
    )
    n_times = int(data.y.shape[0])
    unroll = bundle.unroll
    stride = bundle.stride
    t0 = max(0, min(int(args.t0), n_times - 1))
    window = [t0 + i * stride for i in range(unroll + 1)]
    if window[-1] >= n_times:
        window = [t0 + i * stride for i in range(unroll + 1) if t0 + i * stride < n_times]
    if len(window) < 2:
        print("[ERR] window too short for rollout", flush=True)
        return 1

    kine = load_kinematics_predictor(
        resolve_kinematics_checkpoint(), device, phys_cfg=PhysicsConfig(phase="kinematics")
    )
    n = int(data.num_nodes)
    band = wall_band_mask(data, device, wall_hops=2)
    node_idx, edge_sub, _ = induced_subgraph(band, data.edge_index)
    z_kin = predict_kinematics_latent(kine, data.to(device))
    sdf = sdf_nd_from_data(data, device, n)
    base_feats = build_snapshot_features(z_kin, sdf)[node_idx]
    series = active_series_on_band(data, window, device, node_idx)

    with torch.no_grad():
        states, growths = rollout_pushforward_states(
            bundle.model,
            base_feats=base_feats,
            edge_index=edge_sub,
            active_series=series,
            state0=series[0],
        )

    pos = data.x[:, :2].detach().cpu().numpy()
    band_np = band.detach().cpu().numpy()

    def _full(sub: torch.Tensor) -> np.ndarray:
        out = np.zeros(n, dtype=np.float32)
        out[node_idx.cpu().numpy()] = sub.reshape(-1).detach().cpu().numpy()
        return out

    gt_final = _full(series[-1][:, 1])
    pr_final = _full(states[-1][:, 1])
    gt_growth = _full(growth_active_labels(series[-2], series[-1])[:, 1])
    pr_growth = _full(growths[-1][:, 1]) if growths else np.zeros(n, dtype=np.float32)

    full_mask = torch.ones(len(node_idx), device=device, dtype=torch.bool)
    pred_state = states[-1]
    m_state = trigger_metrics(pred_state, series[-1], full_mask)
    if growths:
        m_growth = growth_metrics(
            growths[-1],
            growth_active_labels(series[-2], series[-1]),
            full_mask,
        )
    else:
        m_growth = {"trigger_f1": 0.0, "mat_f1": 0.0}

    if args.out.strip():
        out_png = Path(args.out.strip())
        if not out_png.is_absolute():
            out_png = root / out_png
    else:
        out_png = species_gnn_viz_dir() / f"{species_gnn_viz_stem(phase='s2', anchor=args.anchor)}.png"
    out_json = out_png.with_suffix(".json")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    panels = [
        ("GT Mat state @ tfinal", gt_final),
        ("Pred Mat state (unroll)", pr_final),
        ("GT growth last step", gt_growth),
        ("Pred growth last step", pr_growth),
    ]
    for ax, (title, vals) in zip(axes.ravel(), panels):
        _scatter_fullmesh_region(
            ax, pos, vals, band_np, title, cmap="Reds", vmin=0.0, vmax=1.0, s=6.0, mask_outside_region=True
        )

    fig.suptitle(
        f"{args.anchor} t0={t0} unroll={len(window)-1} "
        f"state_mat_f1={m_state['mat_f1']:.3f} growth_f1={m_growth.get('trigger_f1', 0):.3f}",
        fontsize=11,
    )
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

    payload = {
        "anchor": args.anchor,
        "window": window,
        "ckpt": ckpt,
        "state_metrics": m_state,
        "growth_metrics": m_growth,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] png={out_png}", flush=True)
    print(f"[OK] json={out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
