"""Probe deploy clot FP/FN sites: flow + geometry context vs GT clot labels.

Compares feature distributions at FP / FN / TP / TN nodes on the wall+3hop band
to explain localized over-prediction (e.g. inlet stagnation blob on patient007).

Usage::

    python scripts/probe_mat_clot_fp_sites.py --leg W_mat_flow_stagnation --anchor patient007
    python scripts/probe_mat_clot_fp_sites.py --ckpt path/to/best.pth --anchor patient003
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.eval_mat_growth_simple import _apply_ckpt_recipe  # noqa: E402
from src.biochem_gnn.config import apply_deploy_env  # noqa: E402
from src.biochem_gnn.mat_growth_simple import leg_out_ckpt  # noqa: E402
from src.config import BiochemConfig, NodeFeat, PhysicsConfig  # noqa: E402
from src.core_physics.clot_phi_simple import sdf_nd_from_data  # noqa: E402
from src.core_physics.species_gnn_clot_rollout import (  # noqa: E402
    load_species_gnn_rollout_bundle,
    prepare_species_gnn_rollout_static,
    rollout_species_gnn_phi_trajectory,
)
from src.core_physics.species_gnn_ladder_viz import ladder_viz_times  # noqa: E402
from src.core_physics.species_pushforward_continuous import train_deploy_eval_flow_source  # noqa: E402
from src.core_physics.species_pushforward_gnn import _flow_feats_from_uv  # noqa: E402
from src.core_physics.species_snapshot_gnn import wall_band_mask  # noqa: E402
from src.core_physics.t0_device import require_cuda_device  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.evaluation.clot_timeline_metrics import clot_binary_masks, clot_mask_counts  # noqa: E402
from src.utils.kinematics_inference import (  # noqa: E402
    load_kinematics_predictor,
    predict_kinematics,
    resolve_kinematics_checkpoint,
)
from src.utils.paths import get_project_root  # noqa: E402

FLOW_NAMES = ("log_speed", "log_shear", "tanh_div", "x_norm", "y_norm")
GEOM_NAMES = ("sdf_nd", "width_nd", "downstream_hop")


def _downstream_hop(data, device: torch.device) -> torch.Tensor:
    """Normalized BFS hop distance from inlet (static, clot-blind)."""
    n = int(data.num_nodes)
    row = torch.cat([data.edge_index[0], data.edge_index[1]]).to(device)
    col = torch.cat([data.edge_index[1], data.edge_index[0]]).to(device)
    src = None
    for attr in ("mask_inlet", "inlet_mask"):
        m = getattr(data, attr, None)
        if m is not None:
            src = m.reshape(-1).bool().to(device)
            break
    if src is None or not bool(src.any().item()):
        return torch.zeros(n, device=device)
    inf = n + 5
    dist = torch.full((n,), inf, dtype=torch.long, device=device)
    dist[src] = 0
    frontier = src.clone()
    d = 0
    while bool(frontier.any()):
        d += 1
        nxt = torch.zeros(n, dtype=torch.bool, device=device)
        nxt[col[frontier[row]]] = True
        upd = nxt & (dist > d)
        if not bool(upd.any()):
            break
        dist[upd] = d
        frontier = upd
        if d > n:
            break
    finite = dist[dist < inf]
    dmax = float(finite.max().item()) if finite.numel() else 1.0
    out = dist.clamp(max=int(dmax)).to(torch.float32)
    return out / max(dmax, 1.0)


def _class_stats(
    values: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Mean/median per feature for nodes where ``mask`` is True."""
    m = mask.astype(bool)
    out: dict[str, dict[str, float]] = {}
    if not m.any():
        return {k: {"mean": 0.0, "median": 0.0, "n": 0} for k in values}
    for name, arr in values.items():
        v = arr[m]
        out[name] = {
            "mean": float(np.mean(v)),
            "median": float(np.median(v)),
            "n": int(m.sum()),
        }
    return out


def _fp_vs_tn_lift(fp_stats: dict, tn_stats: dict, key: str) -> float:
    tn_m = float(tn_stats.get(key, {}).get("median", 0.0))
    fp_m = float(fp_stats.get(key, {}).get("median", 0.0))
    return fp_m - tn_m


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe clot FP/FN site features (deploy)")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--leg", default="W_mat_flow_stagnation")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--times", default="", help="comma times; default ladder grid")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = require_cuda_device()
    root = get_project_root()
    ckpt = Path(args.ckpt.strip()) if args.ckpt.strip() else root / leg_out_ckpt(args.leg.strip())
    if not ckpt.is_file():
        raise SystemExit(f"[ERR] missing ckpt: {ckpt}")

    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    meta = dict(payload.get("meta") or {})
    _apply_ckpt_recipe(meta, label="mat_growth_simple")
    flow_eval = train_deploy_eval_flow_source()
    apply_deploy_env(overrides={"T0_R4_FLOW_SOURCE": flow_eval})

    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    data = torch.load(
        root / "data/processed/graphs_biochem_anchors" / f"{args.anchor}.pt",
        map_location=device,
        weights_only=False,
    )
    wall_hops = int(meta.get("wall_hops", 3))
    band_m = wall_band_mask(data, device, wall_hops=wall_hops).reshape(-1).bool()

    if args.times.strip():
        times = [int(x.strip()) for x in args.times.split(",") if x.strip()]
    else:
        times = ladder_viz_times(int(data.y.shape[0]), max_frames=10)

    print(f"[i] probe leg={args.leg} anchor={args.anchor} times={times}", flush=True)
    t0 = time.perf_counter()
    bundle = load_species_gnn_rollout_bundle(ckpt, device=device)
    if bundle is None:
        raise SystemExit(f"[ERR] bundle load failed: {ckpt}")
    static = prepare_species_gnn_rollout_static(data, device=device, wall_hops=wall_hops)
    phi_traj = rollout_species_gnn_phi_trajectory(
        data, bundle, static, phys_cfg=phys, bio_cfg=bio, device=device, flow_source="kinematics",
    )
    print(f"[i] rollout {time.perf_counter() - t0:.1f}s", flush=True)

    kine = load_kinematics_predictor(resolve_kinematics_checkpoint(), device)
    uv = predict_kinematics(kine, data.clone()).to(device=device)
    flow = _flow_feats_from_uv(data, uv[:, 0], uv[:, 1], device, static.node_idx)
    flow_np = flow.detach().cpu().numpy()
    node_idx = static.node_idx.detach().cpu().numpy()

    sdf = sdf_nd_from_data(data, device, int(data.num_nodes)).detach().cpu().numpy()
    width = data.x[:, NodeFeat.WIDTH_ND].reshape(-1).detach().cpu().numpy()
    downstream = _downstream_hop(data, device).detach().cpu().numpy()

    full_feats: dict[str, np.ndarray] = {}
    for i, name in enumerate(FLOW_NAMES):
        full_feats[name] = np.zeros(int(data.num_nodes), dtype=np.float64)
        full_feats[name][node_idx] = flow_np[:, i]
    full_feats["sdf_nd"] = sdf
    full_feats["width_nd"] = width
    full_feats["downstream_hop"] = downstream

    per_time: list[dict] = []
    fp_lift_acc: dict[str, list[float]] = {k: [] for k in FLOW_NAMES + GEOM_NAMES}

    for t in times:
        phi_gt = gt_clot_phi_at_time(data, int(t), phys, device)
        phi_pred = phi_traj[int(t)]
        masks = clot_binary_masks(phi_pred, phi_gt)
        counts = clot_mask_counts(masks)
        band_masks = {k: (masks[k] & band_m).detach().cpu().numpy() for k in masks}

        stats = {
            cls: _class_stats(full_feats, band_masks[cls])
            for cls in ("fp", "fn", "tp", "tn")
        }
        lifts = {
            f"fp_minus_tn_{name}": _fp_vs_tn_lift(stats["fp"], stats["tn"], name)
            for name in FLOW_NAMES + GEOM_NAMES
        }
        for name in FLOW_NAMES + GEOM_NAMES:
            fp_lift_acc[name].append(lifts[f"fp_minus_tn_{name}"])

        # Top FP sites (for intuition): highest pred phi among FPs
        fp_idx = np.where(band_masks["fp"])[0]
        top_fp: list[dict] = []
        if fp_idx.size > 0:
            pred_np = phi_pred.reshape(-1).detach().cpu().numpy()
            order = fp_idx[np.argsort(-pred_np[fp_idx])][:8]
            for ni in order:
                top_fp.append({
                    "node": int(ni),
                    "pred_phi": float(pred_np[ni]),
                    "log_speed": float(full_feats["log_speed"][ni]),
                    "log_shear": float(full_feats["log_shear"][ni]),
                    "tanh_div": float(full_feats["tanh_div"][ni]),
                    "x_norm": float(full_feats["x_norm"][ni]),
                    "y_norm": float(full_feats["y_norm"][ni]),
                    "sdf_nd": float(full_feats["sdf_nd"][ni]),
                    "downstream_hop": float(full_feats["downstream_hop"][ni]),
                })

        per_time.append({
            "time": int(t),
            "counts_band": counts,
            "counts_band_on_wall": {k: int(band_masks[k].sum()) for k in band_masks},
            "class_stats_band": stats,
            "fp_minus_tn_lift": lifts,
            "top_fp_sites": top_fp,
        })
        print(
            f"  t={t:3d} FP={band_masks['fp'].sum():4d} FN={band_masks['fn'].sum():4d} "
            f"lift_speed={lifts['fp_minus_tn_log_speed']:+.3f} "
            f"lift_div={lifts['fp_minus_tn_tanh_div']:+.3f} "
            f"lift_x={lifts['fp_minus_tn_x_norm']:+.3f}",
            flush=True,
        )

    mean_lift = {k: float(np.mean(v)) for k, v in fp_lift_acc.items() if v}
    ranked = sorted(mean_lift.items(), key=lambda kv: abs(kv[1]), reverse=True)

    report = {
        "anchor": args.anchor,
        "leg": args.leg,
        "ckpt": str(ckpt),
        "wall_hops": wall_hops,
        "times": times,
        "per_time": per_time,
        "mean_fp_minus_tn_lift": mean_lift,
        "top_lifts": [{"feature": k, "fp_minus_tn_median": v} for k, v in ranked[:8]],
        "interpretation_hints": [
            "fp_minus_tn_log_speed > 0: FPs sit in slower flow than true negatives (stagnation prior).",
            "fp_minus_tn_tanh_div < 0: FPs in more converging / recirculation-like flow.",
            "fp_minus_tn_x_norm / y_norm: spatial memorization vs inlet/outlet geometry.",
            "fp_minus_tn_downstream_hop low: FPs cluster near inlet (entrance length artifact).",
            "fp_minus_tn_sdf_nd low: FPs on/near wall band (expected); compare to FN sdf.",
        ],
    }

    if args.out.strip():
        out = Path(args.out)
    else:
        out = (
            root
            / "outputs/reports/comsol_validation"
            / f"mat_clot_fp_probe_{args.leg}_{args.anchor}.json"
        )
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[save] {out}", flush=True)
    print("[i] top mean FP-vs-TN lifts:", flush=True)
    for item in report["top_lifts"][:5]:
        print(f"    {item['feature']}: {item['fp_minus_tn_median']:+.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
