"""Viz Phase 1 species snapshot GNN: GT vs pred FI/Mat trigger on wall band."""

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
from src.core_physics.species_snapshot_gnn import (  # noqa: E402
    build_snapshot_features,
    fi_mat_active_labels,
    fi_mat_log_targets,
    induced_subgraph,
    load_snapshot_bundle,
    logits_to_probs,
    resolve_time_index,
    snapshot_active_log_nd,
    species_gnn_viz_dir,
    species_gnn_viz_stem,
    trigger_metrics,
    wall_band_mask,
)
from src.utils.kinematics_inference import (  # noqa: E402
    load_kinematics_predictor,
    predict_kinematics_latent,
    resolve_kinematics_checkpoint,
)
from src.evaluation.viz_clot_phi_simple import _scatter_fullmesh_region  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Species snapshot GNN viz")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--time-s", type=float, default=None)
    ap.add_argument("--time-index", type=int, default=None)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = get_project_root()
    ckpt = args.ckpt.strip() or str(root / "outputs/biochem/species_snapshot_s1/best.pth")
    bundle = load_snapshot_bundle(ckpt, device=device)
    if bundle is None:
        print(f"[ERR] missing checkpoint: {ckpt}", flush=True)
        return 1

    graph_path = root / VesselConfig(phase="biochem_anchors").graph_output_dir / f"{args.anchor}.pt"
    data = torch.load(graph_path, map_location=device, weights_only=False)
    t_idx = (
        int(args.time_index)
        if args.time_index is not None
        else int(bundle.time_index if args.time_s is None else resolve_time_index(data, time_s=args.time_s))
    )
    if args.time_s is not None:
        t_idx = resolve_time_index(data, time_s=args.time_s)

    meta_path = Path(ckpt).with_suffix(".json")
    kine_ckpt = ""
    if meta_path.is_file():
        meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
        kine_ckpt = str(meta_payload.get("kine_ckpt") or "")
    if not kine_ckpt:
        kine_ckpt = str(resolve_kinematics_checkpoint())
    kine_model = load_kinematics_predictor(
        kine_ckpt, device, phys_cfg=PhysicsConfig(phase="kinematics")
    )

    n = int(data.num_nodes)
    band = wall_band_mask(data, device, wall_hops=bundle.wall_hops)
    node_idx, edge_sub, remap = induced_subgraph(band, data.edge_index)
    z_kin = predict_kinematics_latent(kine_model, data.to(device))
    sdf = sdf_nd_from_data(data, device, n)
    feats = build_snapshot_features(z_kin, sdf)[node_idx]
    tgt_log = fi_mat_log_targets(data, t_idx, device)[node_idx]
    tgt_act = fi_mat_active_labels(tgt_log, thresh_log_nd=bundle.active_log_nd)

    with torch.no_grad():
        logits = bundle.model(feats, edge_sub)
        pred_sub = logits_to_probs(logits, loss_mode=bundle.loss_mode)

    pred_full = torch.zeros(n, 2, device=device)
    tgt_full = torch.zeros(n, 2, device=device)
    pred_full[node_idx] = pred_sub
    tgt_full[node_idx] = tgt_act

    metrics_band = trigger_metrics(pred_sub, tgt_act, torch.ones(pred_sub.shape[0], device=device, dtype=torch.bool))
    metrics = trigger_metrics(
        pred_full,
        fi_mat_active_labels(fi_mat_log_targets(data, t_idx, device), thresh_log_nd=bundle.active_log_nd),
        band,
    )

    pos = data.x[:, :2].detach().cpu().numpy()
    if args.out.strip():
        out_png = Path(args.out.strip())
        if not out_png.is_absolute():
            out_png = root / out_png
        out_json = out_png.with_suffix(".json")
    else:
        viz_dir = species_gnn_viz_dir()
        stem = species_gnn_viz_stem(phase="s1", anchor=args.anchor)
        out_png = viz_dir / f"{stem}.png"
        out_json = viz_dir / f"{stem}.json"
    out_png.parent.mkdir(parents=True, exist_ok=True)

    band_np = band.detach().cpu().numpy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    panels = [
        ("GT FI trigger", tgt_full[:, 0].cpu().numpy()),
        ("Pred FI trigger", pred_full[:, 0].cpu().numpy()),
        ("GT Mat trigger", tgt_full[:, 1].cpu().numpy()),
        ("Pred Mat trigger", pred_full[:, 1].cpu().numpy()),
    ]
    for ax, (title, vals) in zip(axes.ravel(), panels):
        _scatter_fullmesh_region(
            ax,
            pos,
            vals,
            band_np,
            title,
            cmap="Reds",
            vmin=0.0,
            vmax=1.0,
            s=6.0,
            mask_outside_region=True,
        )

    fig.suptitle(
        f"{args.anchor} t={t_idx} band_f1={metrics_band['trigger_f1']:.3f} "
        f"full_band_mask_f1={metrics['trigger_f1']:.3f}",
        fontsize=11,
    )
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

    payload = {
        "anchor": args.anchor,
        "time_index": t_idx,
        "ckpt": ckpt,
        "metrics_band": metrics_band,
        "metrics_full_band_mask": metrics,
        "wall_hops": bundle.wall_hops,
        "active_log_nd": bundle.active_log_nd,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] png={out_png}", flush=True)
    print(f"[OK] json={out_json}", flush=True)
    print(
        f"[i] band trigger_f1={metrics_band['trigger_f1']:.3f} "
        f"rec={metrics_band['trigger_rec']:.3f} prec={metrics_band['trigger_prec']:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
