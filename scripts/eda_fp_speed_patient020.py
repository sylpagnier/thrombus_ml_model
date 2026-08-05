"""Cheap check: are patient020 FPs low-speed (like GT) or high-speed (physfp regime)?

Confirms whether physical_fp_gating was a category error for WG_prec_iter on 020.
Uses the existing fp_geography report if present; otherwise rolls once.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.eval_mat_growth_simple import _apply_ckpt_recipe  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.species_gnn_clot_rollout import (  # noqa: E402
    load_species_gnn_rollout_bundle,
    prepare_species_gnn_rollout_static,
    rollout_species_gnn_phi_trajectory,
)
from src.core_physics.species_pushforward_continuous import clear_offwall_model_cache  # noqa: E402
from src.core_physics.t0_device import require_cuda_device  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.evaluation.fp_geography import classify_fp_geography  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402

DEFAULT_CKPT = "outputs/biochem/eda/wall_gen_prec_iter/WG_prec_iter/best.pth"
ANCHOR_DIR = get_project_root() / "data/processed/graphs_biochem_anchors"


def main() -> int:
    ap = argparse.ArgumentParser(description="FP vs speed correlation on holdout")
    ap.add_argument("--anchor", default="patient020")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = get_project_root()
    ckpt = Path(args.ckpt)
    if not ckpt.is_absolute():
        ckpt = root / ckpt
    device = require_cuda_device()
    clear_offwall_model_cache()
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    meta = dict(payload.get("meta") or {})
    _apply_ckpt_recipe(meta, label="fp_speed", ckpt_path=ckpt)

    anc = args.anchor.strip()
    data = torch.load(ANCHOR_DIR / f"{anc}.pt", map_location=device, weights_only=False)
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    bundle = load_species_gnn_rollout_bundle(ckpt, device=device, quiet=True)
    if bundle is None:
        print(f"[ERR] could not load bundle: {ckpt}", flush=True)
        return 1
    static = prepare_species_gnn_rollout_static(
        data, device=device, wall_hops=int(meta.get("wall_hops", 3))
    )
    traj = rollout_species_gnn_phi_trajectory(
        data, bundle, static, phys_cfg=phys, bio_cfg=bio, device=device, flow_source="kinematics"
    )
    t_eval = int(data.y.shape[0]) - 1
    if torch.is_tensor(traj):
        phi_pred = traj[t_eval].detach().cpu().numpy().reshape(-1)
    else:
        entry = traj[t_eval]
        phi_pred = (entry if torch.is_tensor(entry) else entry["phi"]).detach().cpu().numpy().reshape(-1)
    phi_gt = gt_clot_phi_at_time(data, t_eval, phys, device).detach().cpu().numpy().reshape(-1)
    if hasattr(data, "mask_wall") and data.mask_wall is not None:
        wall = data.mask_wall.bool().cpu().numpy().reshape(-1)
        phi_pred = phi_pred * wall
        phi_gt = phi_gt * wall

    # Prefer t=0 predicted / kinematics UV if present; else COMSOL labels (diagnostic only).
    y0 = data.y[0].detach().cpu().numpy()
    u = y0[:, 0]
    v = y0[:, 1]
    if hasattr(data, "u0_pred") and data.u0_pred is not None:
        u = data.u0_pred.detach().cpu().numpy().reshape(-1)
        v = data.v0_pred.detach().cpu().numpy().reshape(-1)
        speed_src = "u0_pred"
    else:
        speed_src = "y0_uv_label"
    speed = np.sqrt(u * u + v * v)

    pred_b = phi_pred >= float(args.thresh)
    gt_b = phi_gt >= float(args.thresh)
    fp = pred_b & ~gt_b
    tp = pred_b & gt_b
    fn = ~pred_b & gt_b
    bg = ~pred_b & ~gt_b

    def _stats(mask: np.ndarray) -> dict[str, float]:
        if not bool(mask.any()):
            return {"n": 0.0, "speed_mean": float("nan"), "speed_median": float("nan"), "speed_p90": float("nan")}
        s = speed[mask]
        return {
            "n": float(s.size),
            "speed_mean": float(np.mean(s)),
            "speed_median": float(np.median(s)),
            "speed_p90": float(np.percentile(s, 90)),
        }

    geo = classify_fp_geography(phi_pred, phi_gt, data.edge_index, thresh=float(args.thresh))
    report = {
        "anchor": anc,
        "ckpt": str(ckpt),
        "speed_source": speed_src,
        "fp_geography_mode": geo.get("mode"),
        "gt": _stats(gt_b),
        "tp": _stats(tp),
        "fp": _stats(fp),
        "fn": _stats(fn),
        "bg": _stats(bg),
    }
    fp_med = report["fp"]["speed_median"]
    gt_med = report["gt"]["speed_median"]
    if np.isfinite(fp_med) and np.isfinite(gt_med):
        if fp_med <= 1.5 * max(gt_med, 1e-6):
            verdict = "low_speed_like_gt"
            hint = "FPs are stagnant like GT -- physfp (high-speed FP penalty) is a category error; use pocket-contrast"
        else:
            verdict = "high_speed_vs_gt"
            hint = "FPs are faster than GT -- physical_fp_gating may still help"
    else:
        verdict = "unknown"
        hint = "insufficient FP/GT for speed compare"
    report["verdict"] = verdict
    report["hint"] = hint

    print(
        f"[fp_speed {anc}] src={speed_src} "
        f"gt_med={gt_med:.4g} fp_med={fp_med:.4g} tp_med={report['tp']['speed_median']:.4g} "
        f"n_fp={int(report['fp']['n'])} n_gt={int(report['gt']['n'])} "
        f"verdict={verdict}",
        flush=True,
    )
    print(f"[i] {hint}", flush=True)

    out = Path(args.out) if args.out.strip() else (
        root / f"outputs/biochem/viz/mat_growth/fp_speed_{anc}.json"
    )
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] report -> {out}", flush=True)
    print(f"VERDICT={verdict}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
