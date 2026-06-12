"""Phase 2 pushforward growth timeline: GT vs pred growth (0->1) at each unroll step."""

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
    apply_growth_thresholds,
    growth_active_labels,
    growth_metrics,
    load_pushforward_bundle,
    rollout_pushforward_states,
)
from src.core_physics.species_snapshot_gnn import (  # noqa: E402
    build_snapshot_features,
    induced_subgraph,
    species_gnn_viz_dir,
    wall_band_mask,
)
from src.evaluation.viz_clot_phi_simple import _scatter_fullmesh_region  # noqa: E402
from src.utils.kinematics_inference import (  # noqa: E402
    load_kinematics_predictor,
    predict_kinematics_latent,
    resolve_kinematics_checkpoint,
)
from src.utils.paths import get_project_root  # noqa: E402

CH_NAMES = ("FI", "Mat")


def _time_label(data, t_idx: int) -> str:
    if hasattr(data, "t") and data.t is not None:
        t = data.t.reshape(-1)
        if t.numel() > t_idx:
            return f"{int(t_idx)} ({float(t[t_idx]):.0f}s)"
    return str(int(t_idx))


def _full_band(n: int, node_idx: torch.Tensor, sub: torch.Tensor, *, ch: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.float32)
    out[node_idx.cpu().numpy()] = sub[:, ch].reshape(-1).detach().cpu().numpy()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Species pushforward s2 growth timeline viz")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--t0", type=int, default=10, help="Rollout start (10=active growth)")
    ap.add_argument("--hard-thresh", action="store_true", help="Apply growth sigmoid thresholds for display")
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
    window = [t0 + i * stride for i in range(unroll + 1) if t0 + i * stride < n_times]
    if len(window) < 2:
        print("[ERR] window too short", flush=True)
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
    full_mask = torch.ones(len(node_idx), device=device, dtype=torch.bool)

    with torch.no_grad():
        _, growths = rollout_pushforward_states(
            bundle.model,
            base_feats=base_feats,
            edge_index=edge_sub,
            active_series=series,
            state0=series[0],
        )

    pos = data.x[:, :2].detach().cpu().numpy()
    band_np = band.detach().cpu().numpy()
    n_cols = len(growths)

    if args.out.strip():
        out_png = Path(args.out.strip())
        if not out_png.is_absolute():
            out_png = root / out_png
    else:
        out_png = species_gnn_viz_dir() / f"s2_{args.anchor}_growth_t{t0}.png"
    out_json = out_png.with_suffix(".json")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig_w = max(3.2 * n_cols, 10.0)
    fig, axes = plt.subplots(4, n_cols, figsize=(fig_w, 9.5), constrained_layout=True)
    if n_cols == 1:
        axes = np.array(axes).reshape(4, 1)

    step_metrics: list[dict] = []
    for col, step in enumerate(growths):
        t_from = window[col]
        t_to = window[col + 1]
        gt_g = growth_active_labels(series[col], series[col + 1])
        pr_g = apply_growth_thresholds(step) if args.hard_thresh else step
        m = growth_metrics(step, gt_g, full_mask, apply_thresh=not args.hard_thresh)
        step_row = {
            "from_index": t_from,
            "to_index": t_to,
            "label": f"{_time_label(data, t_from)}->{_time_label(data, t_to)}",
            "growth_f1": float(m["trigger_f1"]),
            "mat_growth_f1": float(m["mat_f1"]),
            "fi_growth_f1": float(m["fi_f1"]),
            "gt_growth_nodes": int((gt_g > 0.5).any(dim=-1).sum().item()),
        }
        step_metrics.append(step_row)

        for ch, name in enumerate(CH_NAMES):
            gt_full = _full_band(n, node_idx, gt_g, ch=ch)
            pr_full = _full_band(n, node_idx, pr_g, ch=ch)
            f1 = float(m["fi_f1"] if ch == 0 else m["mat_f1"])
            _scatter_fullmesh_region(
                axes[ch * 2, col],
                pos,
                gt_full,
                band_np,
                f"GT {name} growth\n{step_row['label']}",
                cmap="Reds",
                vmin=0.0,
                vmax=1.0,
                s=5.0,
                mask_outside_region=True,
            )
            _scatter_fullmesh_region(
                axes[ch * 2 + 1, col],
                pos,
                pr_full,
                band_np,
                f"Pred {name} growth\nF1={f1:.3f}",
                cmap="Reds",
                vmin=0.0,
                vmax=1.0,
                s=5.0,
                mask_outside_region=True,
            )

    mean_gf1 = sum(s["growth_f1"] for s in step_metrics) / max(len(step_metrics), 1)
    mean_mf1 = sum(s["mat_growth_f1"] for s in step_metrics) / max(len(step_metrics), 1)
    fig.suptitle(
        f"{args.anchor} growth timeline t0={t0}  mean growth F1={mean_gf1:.3f}  mean Mat growth F1={mean_mf1:.3f}",
        fontsize=11,
    )
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

    payload = {
        "anchor": args.anchor,
        "window": window,
        "ckpt": ckpt,
        "per_step_growth": step_metrics,
        "mean_growth_f1": mean_gf1,
        "mean_mat_growth_f1": mean_mf1,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] png={out_png}", flush=True)
    print(f"[OK] json={out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
