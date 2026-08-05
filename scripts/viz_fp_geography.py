"""Cheap FP geography diagnostic: distant wrong-pocket vs adjacent overpaint.

Deploy-faithful cold rollout of WG_prec_iter (or any ckpt) on patient020, then classify
FPs by graph hop distance to nearest GT clot. Prints recommend_leg = physfp | cloop.

Example:
  python scripts/viz_fp_geography.py --anchor patient020 \\
    --ckpt outputs/biochem/eda/wall_gen_prec_iter/WG_prec_iter/best.pth
"""

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
from src.evaluation.clot_shape_score import graph_hop_distance_from_seeds  # noqa: E402
from src.evaluation.fp_geography import classify_fp_geography, format_fp_geography  # noqa: E402
from src.evaluation.seed_growth_diagnostics import (  # noqa: E402
    passes_wall_gen_gate,
    seed_growth_diagnostic_panel,
)
from src.utils.paths import get_project_root  # noqa: E402

DEFAULT_CKPT = "outputs/biochem/eda/wall_gen_prec_iter/WG_prec_iter/best.pth"
ANCHOR_DIR = get_project_root() / "data/processed/graphs_biochem_anchors"


def _pos_xy(data) -> np.ndarray:
    if hasattr(data, "pos") and data.pos is not None:
        return data.pos.detach().cpu().numpy()[:, :2]
    x = data.x.detach().cpu().numpy()
    return x[:, :2]


def main() -> int:
    ap = argparse.ArgumentParser(description="FP geography viz (distant vs adjacent)")
    ap.add_argument("--anchor", default="patient020")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out", default="")
    ap.add_argument("--adjacent-max-hops", type=int, default=2)
    ap.add_argument("--thresh", type=float, default=0.5)
    args = ap.parse_args()

    root = get_project_root()
    ckpt = Path(args.ckpt)
    if not ckpt.is_absolute():
        ckpt = root / ckpt
    if not ckpt.is_file():
        print(f"[ERR] missing ckpt: {ckpt}", flush=True)
        return 1

    device = require_cuda_device()
    clear_offwall_model_cache()
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    meta = dict(payload.get("meta") or {})
    _apply_ckpt_recipe(meta, label="fp_geography", ckpt_path=ckpt)

    anc = args.anchor.strip()
    data = torch.load(ANCHOR_DIR / f"{anc}.pt", map_location=device, weights_only=False)
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    bundle = load_species_gnn_rollout_bundle(ckpt, device=device, quiet=True)
    if bundle is None:
        print(f"[ERR] could not load bundle: {ckpt}", flush=True)
        return 1
    static = prepare_species_gnn_rollout_static(data, device=device, wall_hops=int(meta.get("wall_hops", 3)))
    traj = rollout_species_gnn_phi_trajectory(
        data,
        bundle,
        static,
        phys_cfg=phys,
        bio_cfg=bio,
        device=device,
        flow_source="kinematics",
    )
    t_eval = int(data.y.shape[0]) - 1
    # Trajectory is a tensor [T,N] or list/dict of per-time phi.
    if torch.is_tensor(traj):
        phi_pred = traj[t_eval].detach().cpu().numpy().reshape(-1)
    elif isinstance(traj, dict) and t_eval in traj:
        entry = traj[t_eval]
        phi_pred = (entry["phi"] if isinstance(entry, dict) else entry).detach().cpu().numpy().reshape(-1)
    else:
        entry = traj[t_eval]
        phi_pred = (entry["phi"] if isinstance(entry, dict) else entry).detach().cpu().numpy().reshape(-1)
    phi_gt = gt_clot_phi_at_time(data, t_eval, phys, device).detach().cpu().numpy().reshape(-1)
    wall = None
    if hasattr(data, "mask_wall") and data.mask_wall is not None:
        wall = data.mask_wall.bool().cpu().numpy().reshape(-1)
        phi_pred = phi_pred * wall
        phi_gt = phi_gt * wall

    summary = classify_fp_geography(
        phi_pred,
        phi_gt,
        data.edge_index,
        thresh=float(args.thresh),
        adjacent_max_hops=int(args.adjacent_max_hops),
        n_nodes=int(phi_pred.shape[0]),
    )
    print(format_fp_geography(summary, label=anc), flush=True)

    # Gate panel from binary counts (same t_eval).
    pred_b = phi_pred >= float(args.thresh)
    gt_b = phi_gt >= float(args.thresh)
    fp_n = int((pred_b & ~gt_b).sum())
    fn_n = int((~pred_b & gt_b).sum())
    tp_n = int((pred_b & gt_b).sum())
    prec = tp_n / max(tp_n + fp_n, 1)
    rec = tp_n / max(tp_n + fn_n, 1)
    f1 = 2.0 * prec * rec / max(prec + rec, 1e-6)
    mass = float(pred_b.sum()) / max(float(gt_b.sum()), 1.0)
    gate_row = {
        "deploy_clot_f1": f1,
        "deploy_clot_score": f1,  # local proxy; formal score is guiding elsewhere
        "deploy_clot_mass_ratio": mass,
        "deploy_clot_fp": float(fp_n),
        "deploy_clot_fn": float(fn_n),
    }
    ok, reason = passes_wall_gen_gate(gate_row)
    print(f"[gate] ok={ok} {reason}", flush=True)
    panel = seed_growth_diagnostic_panel(
        {
            **gate_row,
            "mat_seed_prec": 1.0,
            "mat_seed_count": 1.0,
            "mat_front_speed_ratio": 0.5,
            "mat_overpaint_frac": 0.0,
        }
    )
    print(
        f"[i] recommend_leg={summary['recommend_leg']} "
        f"(mode={summary['mode']}) | next: go_wg_prec_physfp.ps1 -Leg WG_prec_{summary['recommend_leg']}",
        flush=True,
    )

    pos = _pos_xy(data)
    dist = graph_hop_distance_from_seeds(data.edge_index, int(pos.shape[0]), gt_b)
    fp = pred_b & ~gt_b
    adj_max = int(args.adjacent_max_hops)
    adj = fp & (dist <= adj_max)
    distant = fp & (dist > adj_max)

    out = Path(args.out) if args.out.strip() else (
        root / f"outputs/biochem/viz/mat_growth/fp_geography_{anc}.png"
    )
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, mask, title, color in (
        (axes[0], gt_b, "GT clot", "#4a148c"),
        (axes[1], pred_b, "Pred clot", "#e65100"),
        (axes[2], None, f"FP adj(red)/dist(orange) FN(blue)\n-> {summary['recommend_leg']}", None),
    ):
        ax.set_aspect("equal")
        ax.axis("off")
        ax.scatter(pos[:, 0], pos[:, 1], c="#d9d9d9", s=2.0, linewidths=0, alpha=0.35, zorder=1)
        if mask is not None:
            ax.scatter(pos[mask, 0], pos[mask, 1], c=color, s=6.0, linewidths=0, alpha=0.95, zorder=3)
        else:
            if distant.any():
                ax.scatter(
                    pos[distant, 0], pos[distant, 1], c="#ff7f0e", s=8.0, linewidths=0, alpha=0.95, zorder=3, label="distant FP"
                )
            if adj.any():
                ax.scatter(
                    pos[adj, 0], pos[adj, 1], c="#d62728", s=8.0, linewidths=0, alpha=0.95, zorder=4, label="adjacent FP"
                )
            fn = ~pred_b & gt_b
            if fn.any():
                ax.scatter(pos[fn, 0], pos[fn, 1], c="#1f77b4", s=6.0, linewidths=0, alpha=0.9, zorder=2, label="FN")
            ax.legend(loc="upper right", fontsize=7, frameon=False)
        ax.set_title(title, fontsize=10)
    fig.suptitle(
        f"FP geography {anc} | adj_frac={summary['adjacent_frac']:.2f} "
        f"dist_frac={summary['distant_frac']:.2f} hop_med={summary.get('fp_hop_to_gt_median')}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)

    report = {
        "anchor": anc,
        "ckpt": str(ckpt),
        "t_eval": t_eval,
        "fp_geography": summary,
        "gate": {"ok": ok, "reason": reason, **gate_row},
        "panel_mode": panel.get("mode"),
        "out_png": str(out),
    }
    report_path = out.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] viz -> {out}", flush=True)
    print(f"[OK] report -> {report_path}", flush=True)
    print(f"RECOMMEND_LEG=WG_prec_{summary['recommend_leg']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
