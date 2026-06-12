"""~15 min LOAO sweep of T0 physics trigger variants (deploy nucleation projection)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.config import BiochemConfig, PhysicsConfig, VesselConfig
from src.core_physics.clot_phi_simple import build_clot_phi_step, snapshot_clot_physics_trigger_config
from src.core_physics.clot_physics_trigger_sweep import (
    apply_physics_sweep_leg,
    physics_sweep_legs,
    sweep_score,
)
from src.core_physics.clot_trigger_rollout import (
    lumen_false_positive_frac,
    rollout_clot_trigger_physics,
    snapshot_clot_trigger_rollout_config,
)
from src.core_physics.neighbor_band_trigger import apply_physics_trigger_baseline_env
from src.training.clot_growth_eval import eval_phi_trajectory_on_anchor
from src.training.clot_trigger_stack import apply_clot_trigger_honest_env
from src.training.train_clot_phi_simple import _clot_metrics


def _env_bool(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("1", "true", "yes", "on")


def _mean(rows: list[dict], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r and r[key] == r[key]]
    return sum(vals) / len(vals) if vals else float("nan")


def eval_anchor_leg(
    graph_path: Path,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    gt_mu_oracle: bool,
) -> dict:
    data = torch.load(graph_path, map_location=device, weights_only=False)
    n_steps = int(data.y.shape[0])
    full_ones = torch.ones(int(data.num_nodes), device=device, dtype=torch.bool)

    if gt_mu_oracle:
        phi_by_t: dict[int, torch.Tensor] = {}
        per_step = []
        for t in range(n_steps):
            step = build_clot_phi_step(data, t, phys_cfg, bio_cfg, device)
            phi_gt = step.phi_gt.reshape(-1)
            phi_pred = phi_gt.clone()
            phi_by_t[t] = phi_pred
            support = step.loss_mask.reshape(-1).bool()
            m_full = _clot_metrics(phi_pred, phi_gt, full_ones)
            per_step.append(
                {
                    "t": t,
                    "full_mesh_f1": float(m_full["clot_f1"]),
                    "lumen_fp_frac_deploy": 0.0,
                }
            )
    else:
        traj = rollout_clot_trigger_physics(
            data, phys_cfg=phys_cfg, bio_cfg=bio_cfg, device=device, time_stride=1
        )
        phi_by_t = {t: v["phi"] for t, v in traj.items()}
        per_step = []
        for t in range(n_steps):
            if t not in traj:
                continue
            step = build_clot_phi_step(data, t, phys_cfg, bio_cfg, device)
            phi_gt = step.phi_gt.reshape(-1)
            phi_deploy = traj[t]["phi"].reshape(-1)
            support = step.loss_mask.reshape(-1).bool()
            m_sup = _clot_metrics(phi_deploy, phi_gt, support)
            m_full = _clot_metrics(phi_deploy, phi_gt, full_ones)
            per_step.append(
                {
                    "t": t,
                    "support_f1": float(m_sup["clot_f1"]),
                    "full_mesh_f1": float(m_full["clot_f1"]),
                    "lumen_fp_frac_deploy": lumen_false_positive_frac(
                        phi_deploy, phi_gt, data=data, device=device
                    ),
                }
            )

    traj_score = eval_phi_trajectory_on_anchor(
        phi_by_t,
        data,
        anchor=graph_path.stem,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        rule_tag="t0_physics_sweep",
    )
    return {
        "anchor": graph_path.stem,
        "mean_support_f1": _mean(per_step, "support_f1"),
        "mean_full_mesh_f1": _mean(per_step, "full_mesh_f1"),
        "mean_lumen_fp_deploy": _mean(per_step, "lumen_fp_frac_deploy"),
        "final_full_mesh_f1": per_step[-1]["full_mesh_f1"] if per_step else float("nan"),
        "trajectory_score": float(traj_score.get("trajectory_score", float("nan"))),
    }


def run_leg(
    leg: dict,
    paths: list[Path],
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    val_anchor: str,
) -> dict:
    apply_physics_sweep_leg(leg)
    gt_oracle = bool(leg.get("gt_mu_oracle")) or _env_bool("CLOT_TRIGGER_GT_MU_ORACLE")
    rows = [
        eval_anchor_leg(p, device=device, phys_cfg=phys_cfg, bio_cfg=bio_cfg, gt_mu_oracle=gt_oracle)
        for p in paths
    ]
    val_row = next((r for r in rows if r["anchor"] == val_anchor), None)
    summary = {
        "leg_id": leg["id"],
        "note": leg.get("note", ""),
        "physics_config": snapshot_clot_physics_trigger_config(),
        "rollout_config": snapshot_clot_trigger_rollout_config(),
        "gt_mu_oracle": gt_oracle,
        "mean_support_f1": _mean(rows, "mean_support_f1"),
        "mean_full_mesh_f1": _mean(rows, "mean_full_mesh_f1"),
        "mean_lumen_fp_deploy": _mean(rows, "mean_lumen_fp_deploy"),
        "mean_trajectory_score": _mean(rows, "trajectory_score"),
        "val_anchor": val_anchor,
        "val_full_mesh_f1": float(val_row["mean_full_mesh_f1"]) if val_row else float("nan"),
        "val_lumen_fp": float(val_row["mean_lumen_fp_deploy"]) if val_row else float("nan"),
        "score": float("nan"),
        "per_anchor": rows,
    }
    summary["score"] = sweep_score(summary)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="T0 physics trigger variant sweep")
    ap.add_argument("--anchor-dir", default="")
    ap.add_argument("--val", default="patient007")
    ap.add_argument("--out-dir", default="outputs/biochem/clot_trigger/t0_physics_sweep")
    ap.add_argument("--legs", default="", help="Comma-separated leg ids (default: all)")
    args = ap.parse_args()

    apply_clot_trigger_honest_env()
    apply_physics_trigger_baseline_env()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phys_cfg = PhysicsConfig(phase="biochem")
    bio_cfg = BiochemConfig(phase="biochem")

    anchor_dir = Path(args.anchor_dir) if args.anchor_dir else (
        _REPO / VesselConfig(phase="biochem_anchors").graph_output_dir
    )
    paths = sorted(anchor_dir.glob("*.pt"))
    if not paths:
        print(f"[ERR] no graphs in {anchor_dir}", file=sys.stderr)
        return 2

    legs = physics_sweep_legs()
    if args.legs.strip():
        wanted = {x.strip() for x in args.legs.split(",") if x.strip()}
        legs = [leg for leg in legs if leg["id"] in wanted]
        if not legs:
            print(f"[ERR] no legs matched {wanted}", file=sys.stderr)
            return 2

    out_dir = _REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results: list[dict] = []

    for leg in legs:
        print(f"[NEW] leg {leg['id']}: {leg.get('note', '')}", flush=True)
        row = run_leg(leg, paths, device=device, phys_cfg=phys_cfg, bio_cfg=bio_cfg, val_anchor=args.val)
        results.append(row)
        leg_path = out_dir / f"{leg['id']}.json"
        leg_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
        print(
            f"[OK] {leg['id']}: score={row['score']:.3f} "
            f"full_F1={row['mean_full_mesh_f1']:.3f} "
            f"lumen_fp={row['mean_lumen_fp_deploy']:.3f} "
            f"p007={row['val_full_mesh_f1']:.3f}",
            flush=True,
        )

    results.sort(key=lambda r: float(r.get("score", float("-inf"))), reverse=True)
    physics_rows = [r for r in results if not r.get("gt_mu_oracle")]
    index = {
        "step": "t0_physics_sweep",
        "elapsed_s": time.time() - t0,
        "n_legs": len(results),
        "val_anchor": args.val,
        "ranking": [
            {
                "leg_id": r["leg_id"],
                "score": r["score"],
                "mean_full_mesh_f1": r["mean_full_mesh_f1"],
                "mean_lumen_fp_deploy": r["mean_lumen_fp_deploy"],
                "val_full_mesh_f1": r["val_full_mesh_f1"],
                "note": r.get("note", ""),
                "gt_mu_oracle": bool(r.get("gt_mu_oracle")),
            }
            for r in results
        ],
        "best_leg_id": physics_rows[0]["leg_id"] if physics_rows else "",
        "oracle_leg_id": "Z_gt_mu_oracle",
    }
    index_path = out_dir / "sweep_index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"[save] {index_path}", flush=True)
    if physics_rows:
        best = physics_rows[0]
        print(
            f"[summary] best_physics={best['leg_id']} score={best['score']:.3f} "
            f"full_F1={best['mean_full_mesh_f1']:.3f} elapsed={index['elapsed_s']:.0f}s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
