"""Rollout health diagnostics: catch frozen wall-saturated clot predictions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_continuous_time import macro_tau_at_index
from src.core_physics.clot_phi_simple import _wall_mask_from_data
from src.core_physics.t0_device import require_cuda_device
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time
from src.core_physics.t0_rung4_ladder import rollout_rung4_phi_trajectory
from src.evaluation.viz_clot_trigger import clot_trigger_viz_f1
from src.training.train_clot_phi_simple import _clot_metrics
from src.utils.paths import get_project_root


def _binary_jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).bool()
    b = b.reshape(-1).bool()
    inter = float((a & b).sum().item())
    union = float((a | b).sum().item())
    return inter / max(union, 1.0)


def compute_rollout_health(
    phi_traj: dict[int, torch.Tensor],
    data,
    phys: PhysicsConfig,
    device: torch.device,
    *,
    bio: BiochemConfig,
    times: list[int] | None = None,
) -> dict:
    n_steps = int(data.y.shape[0])
    times = times or list(range(n_steps))
    wall = _wall_mask_from_data(data, device, int(data.num_nodes)).reshape(-1).bool()
    mask = torch.ones(int(data.num_nodes), device=device, dtype=torch.bool)

    rows = []
    for t in times:
        phi_gt = gt_clot_phi_at_time(data, t, phys, device)
        phi_p = phi_traj[int(t)]
        m = _clot_metrics(phi_p.reshape(-1), phi_gt.reshape(-1), mask)
        pred_pos = phi_p.reshape(-1) >= 0.5
        gt_pos = phi_gt.reshape(-1) >= 0.5
        pred_on_wall = float((pred_pos & wall).sum().item())
        pred_total = float(pred_pos.sum().item())
        wall_frac = pred_on_wall / max(pred_total, 1.0)
        rows.append({
            "time": int(t),
            "tau": float(macro_tau_at_index(data, t, bio_cfg=bio)),
            "clot_f1": float(m["clot_f1"]),
            "clot_prec": float(m["clot_prec"]),
            "clot_rec": float(m["clot_rec"]),
            "pred_pos_frac": float(m["pred_pos_frac"]),
            "gt_pos_frac": float(m["gt_pos_frac"]),
            "pred_wall_frac": wall_frac,
            "n_pred": int(pred_total),
            "n_gt": int(gt_pos.sum().item()),
        })

    t0 = times[0]
    t1 = times[-1]
    phi_early = phi_traj[int(t0)].reshape(-1) >= 0.5
    phi_late = phi_traj[int(t1)].reshape(-1) >= 0.5
    gt_early = gt_clot_phi_at_time(data, t0, phys, device).reshape(-1) >= 0.5
    gt_late = gt_clot_phi_at_time(data, t1, phys, device).reshape(-1) >= 0.5

    f1_vals = [r["clot_f1"] for r in rows]
    min_f1 = min(f1_vals) if f1_vals else 0.0
    min_f1_t = rows[f1_vals.index(min_f1)]["time"] if f1_vals else 0

    health = {
        "commit_jaccard_early_late": _binary_jaccard(phi_early, phi_late),
        "gt_commit_jaccard_early_late": _binary_jaccard(gt_early, gt_late),
        "pred_pos_frac_early": rows[0]["pred_pos_frac"],
        "pred_pos_frac_late": rows[-1]["pred_pos_frac"],
        "pred_wall_frac_late": rows[-1]["pred_wall_frac"],
        "min_f1": min_f1,
        "min_f1_time": min_f1_t,
        "final_f1": rows[-1]["clot_f1"],
        "frozen_wall_saturation": bool(
            rows[0]["pred_pos_frac"] > 0.01
            and _binary_jaccard(phi_early, phi_late) > 0.95
            and rows[-1]["pred_wall_frac"] > 0.85
        ),
        "timeline": rows,
    }
    return health


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--step", default="s2_species")
    ap.add_argument("--compare", default="s0")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = require_cuda_device()
    root = get_project_root()
    graph = root / "data/processed/graphs_biochem_anchors" / f"{args.anchor}.pt"
    data = torch.load(graph, map_location=device, weights_only=False)
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    times = [0, 5, 11, 17, 23, 29, 35, 41, 47, 53]

    results = {}
    for step in [args.step, args.compare]:
        phi_traj = rollout_rung4_phi_trajectory(data, phys, bio, device, step=step)
        h = compute_rollout_health(phi_traj, data, phys, device, bio=bio, times=times)
        results[step] = h
        print(f"\n[{step}]")
        print(f"  final_f1={h['final_f1']:.3f} min_f1={h['min_f1']:.3f} @ t={h['min_f1_time']}")
        print(f"  pred_pos early={h['pred_pos_frac_early']:.4f} late={h['pred_pos_frac_late']:.4f}")
        print(f"  commit_jaccard early/late={h['commit_jaccard_early_late']:.3f} (gt={h['gt_commit_jaccard_early_late']:.3f})")
        print(f"  pred_wall_frac late={h['pred_wall_frac_late']:.3f}")
        print(f"  FROZEN_WALL_SATURATION={h['frozen_wall_saturation']}")

    out = Path(args.out) if args.out.strip() else (
        root / f"outputs/biochem/clot_trigger/t0_rung4_{args.step}_{args.anchor}_health.json"
    )
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[save] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
