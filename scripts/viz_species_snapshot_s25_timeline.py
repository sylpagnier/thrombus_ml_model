"""Phase 2.5 continuous pushforward: GT vs pred FI/Mat state at each unroll step."""

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

from src.config import PhysicsConfig, VesselConfig  # noqa: E402
from src.core_physics.species_pushforward_continuous import (  # noqa: E402
    continuous_dual_head,
    load_continuous_bundle,
    log_series_on_band,
    rollout_continuous_states,
)
from src.core_physics.species_pushforward_gnn import build_band_base_features  # noqa: E402
from src.core_physics.species_snapshot_gnn import (  # noqa: E402
    fi_mat_active_labels,
    species_gnn_viz_dir,
    trigger_metrics,
    wall_band_mask,
)
from src.evaluation.viz_clot_phi_simple import _scatter_fullmesh_region  # noqa: E402
from src.utils.kinematics_inference import (  # noqa: E402
    load_kinematics_predictor,
    resolve_kinematics_checkpoint,
)
from src.utils.paths import get_project_root  # noqa: E402

CH_NAMES = ("FI", "Mat")


def _time_label(data, t_idx: int) -> str:
    if hasattr(data, "t") and data.t is not None:
        t = data.t.reshape(-1)
        if t.numel() > t_idx:
            return f"t={int(t_idx)} ({float(t[t_idx]):.0f}s)"
    return f"t={int(t_idx)}"


def _full_band(n: int, node_idx: torch.Tensor, sub: torch.Tensor, *, ch: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.float32)
    out[node_idx.cpu().numpy()] = sub[:, ch].reshape(-1).detach().cpu().numpy()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Species continuous s25 timeline viz")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--t0", type=int, default=10)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = get_project_root()
    ckpt = args.ckpt.strip() or str(root / "outputs/biochem/species_snapshot_s25/best.pth")
    bundle = load_continuous_bundle(ckpt, device=device)
    if bundle is None:
        print(f"[ERR] missing checkpoint: {ckpt}", flush=True)
        return 1
    ckpt_meta = {}
    ckpt_path = Path(ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = root / ckpt_path
    if ckpt_path.is_file():
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt_meta = dict(payload.get("meta") or {})
        if bool(ckpt_meta.get("kin_per_vessel_norm")):
            os.environ["SPECIES_KIN_PER_VESSEL_NORM"] = "1"
        if bool(ckpt_meta.get("dual_head")):
            os.environ["SPECIES_CONTINUOUS_DUAL_HEAD"] = "1"

    data = torch.load(
        root / VesselConfig(phase="biochem_anchors").graph_output_dir / f"{args.anchor}.pt",
        map_location=device,
        weights_only=False,
    )
    t0 = max(0, min(int(args.t0), int(data.y.shape[0]) - 1))
    window = [t0 + i * bundle.stride for i in range(bundle.unroll + 1) if t0 + i * bundle.stride < int(data.y.shape[0])]
    if len(window) < 2:
        print("[ERR] window too short", flush=True)
        return 1

    kine = load_kinematics_predictor(
        resolve_kinematics_checkpoint(), device, phys_cfg=PhysicsConfig(phase="kinematics")
    )
    n = int(data.num_nodes)
    band = wall_band_mask(data, device, wall_hops=2)
    stat = build_band_base_features(data, kine, device, wall_hops=2)
    node_idx = stat["node_idx"]
    edge_sub = stat["edge_index"]
    base_feats = stat["base_feats"]
    log_series = log_series_on_band(data, window, device, node_idx)
    gt_active = [fi_mat_active_labels(log_series[i]) for i in range(len(window))]

    with torch.no_grad():
        _, _, pred_active = rollout_continuous_states(
            bundle.model,
            base_feats=base_feats,
            edge_index=edge_sub,
            log_series=log_series,
            log_state0=log_series[0],
        )

    pos = data.x[:, :2].detach().cpu().numpy()
    band_np = band.detach().cpu().numpy()
    n_cols = len(pred_active)
    full_mask = torch.ones(len(node_idx), device=device, dtype=torch.bool)

    if args.out.strip():
        out_png = Path(args.out.strip())
        if not out_png.is_absolute():
            out_png = root / out_png
    else:
        phase_tag = "s31" if continuous_dual_head() else "s25"
        out_png = species_gnn_viz_dir() / f"{phase_tag}_{args.anchor}_timeline_t{t0}.png"
    out_json = out_png.with_suffix(".json")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, n_cols, figsize=(max(3.2 * n_cols, 10), 9.5), constrained_layout=True)
    if n_cols == 1:
        axes = np.array(axes).reshape(4, 1)

    step_metrics: list[dict] = []
    for col in range(n_cols):
        gt = gt_active[col]
        pr = pred_active[col]
        m = trigger_metrics(pr, gt, full_mask)
        step_metrics.append(
            {
                "time_index": window[col],
                "fi_f1": float(m["fi_f1"]),
                "mat_f1": float(m["mat_f1"]),
                "trigger_f1": float(m["trigger_f1"]),
            }
        )
        for ch, name in enumerate(CH_NAMES):
            gt_full = _full_band(n, node_idx, gt, ch=ch)
            pr_full = _full_band(n, node_idx, pr, ch=ch)
            f1 = float(m["fi_f1"] if ch == 0 else m["mat_f1"])
            _scatter_fullmesh_region(
                axes[ch * 2, col], pos, gt_full, band_np, f"GT {name}\n{_time_label(data, window[col])}",
                cmap="Reds", vmin=0.0, vmax=1.0, s=5.0, mask_outside_region=True,
            )
            _scatter_fullmesh_region(
                axes[ch * 2 + 1, col], pos, pr_full, band_np, f"Pred {name}\nF1={f1:.3f}",
                cmap="Reds", vmin=0.0, vmax=1.0, s=5.0, mask_outside_region=True,
            )

    final_m = trigger_metrics(pred_active[-1], gt_active[-1], full_mask)
    fig.suptitle(
        f"{args.anchor} {('s31' if continuous_dual_head() else 's25')} continuous t0={t0} "
        f"final FI F1={final_m['fi_f1']:.3f} Mat F1={final_m['mat_f1']:.3f}",
        fontsize=11,
    )
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

    payload = {
        "anchor": args.anchor,
        "window": window,
        "ckpt": ckpt,
        "per_step": step_metrics,
        "final_state_metrics": final_m,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] png={out_png}", flush=True)
    print(f"[OK] json={out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
