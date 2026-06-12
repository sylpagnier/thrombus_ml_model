"""T0 clot trigger eval: physics gelation + deploy nucleation projection.

Default: pred-seed forward envelope (no GT commits), support + full-mesh F1.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.config import BiochemConfig, PhysicsConfig, VesselConfig
from src.core_physics.clot_phi_simple import build_clot_phi_step, clot_phi_loss_scope
from src.core_physics.clot_trigger_rollout import (
    forward_path_uses_gt_commits,
    lumen_false_positive_frac,
    rollout_clot_trigger_physics,
    snapshot_clot_trigger_rollout_config,
)
from src.core_physics.neighbor_band_trigger import apply_physics_trigger_baseline_env
from src.training.clot_growth_eval import eval_phi_trajectory_on_anchor
from src.training.clot_trigger_stack import (
    apply_clot_trigger_honest_env,
    apply_clot_trigger_oracle_forward_env,
    apply_oracle_neighbor_mask_env,
)
from src.training.train_clot_phi_simple import _clot_metrics


def _mean(rows: list[dict], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r and math.isfinite(float(r[key]))]
    return sum(vals) / len(vals) if vals else float("nan")


def eval_anchor_t0_oracle(
    graph_path: Path,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    prior_gate: bool = False,
) -> dict:
    """GT flow + species; physics trigger with nucleation projection rollout."""
    data = torch.load(graph_path, map_location=device, weights_only=False)
    traj = rollout_clot_trigger_physics(
        data,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        device=device,
        time_stride=1,
    )
    phi_by_t = {t: v["phi"] for t, v in traj.items()}
    per_step: list[dict] = []
    n_steps = int(data.y.shape[0])
    full_ones = torch.ones(int(data.num_nodes), device=device, dtype=torch.bool)

    for t in range(n_steps):
        if t not in traj:
            continue
        step = build_clot_phi_step(data, t, phys_cfg, bio_cfg, device)
        phi_gt = step.phi_gt.reshape(-1)
        phi_deploy = traj[t]["phi"].reshape(-1)
        phi_raw = traj[t]["phi_raw"].reshape(-1)
        support = step.loss_mask.reshape(-1).bool()
        m_deploy_sup = _clot_metrics(phi_deploy, phi_gt, support)
        m_deploy_full = _clot_metrics(phi_deploy, phi_gt, full_ones)
        m_raw_full = _clot_metrics(phi_raw, phi_gt, full_ones)
        per_step.append(
            {
                "t": t,
                "support_f1": float(m_deploy_sup["clot_f1"]),
                "full_mesh_f1": float(m_deploy_full["clot_f1"]),
                "raw_full_mesh_f1": float(m_raw_full["clot_f1"]),
                "lumen_fp_frac_deploy": lumen_false_positive_frac(
                    phi_deploy, phi_gt, data=data, device=device
                ),
                "lumen_fp_frac_raw": lumen_false_positive_frac(
                    phi_raw, phi_gt, data=data, device=device
                ),
                "pred_pos_frac_deploy": float(m_deploy_full["pred_pos_frac"]),
                "pred_pos_frac_raw": float(m_raw_full["pred_pos_frac"]),
                "gt_pos_frac": float(m_deploy_sup["gt_pos_frac"]),
            }
        )

    traj_score = eval_phi_trajectory_on_anchor(
        phi_by_t,
        data,
        anchor=graph_path.stem,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        rule_tag="t0_physics_nucleation",
    )
    return {
        "anchor": graph_path.stem,
        "trigger_mode": "physics",
        "prior_gate": prior_gate,
        "loss_scope": clot_phi_loss_scope(),
        "rollout_config": snapshot_clot_trigger_rollout_config(),
        "forward_uses_gt_commits": forward_path_uses_gt_commits(),
        "n_steps": n_steps,
        "mean_support_f1": _mean(per_step, "support_f1"),
        "mean_full_mesh_f1": _mean(per_step, "full_mesh_f1"),
        "mean_lumen_fp_deploy": _mean(per_step, "lumen_fp_frac_deploy"),
        "mean_lumen_fp_raw": _mean(per_step, "lumen_fp_frac_raw"),
        "final_support_f1": per_step[-1]["support_f1"] if per_step else float("nan"),
        "final_full_mesh_f1": per_step[-1]["full_mesh_f1"] if per_step else float("nan"),
        "trajectory_score": float(traj_score.get("trajectory_score", float("nan"))),
        "tfinal_wall_ring_frac": float(traj_score.get("tfinal_wall_ring_frac", float("nan"))),
        "per_step": per_step,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="T0 clot trigger (GT flow + GT species, nucleation deploy)")
    ap.add_argument("--anchor-dir", default="")
    ap.add_argument("--val", default="patient007")
    ap.add_argument("--prior-gate", action="store_true")
    ap.add_argument(
        "--oracle-band",
        action="store_true",
        help="Legacy: GT-mu seeds + dgamma slice for loss/F1 (debug only)",
    )
    ap.add_argument(
        "--oracle-forward",
        action="store_true",
        help="Debug: forward envelope seeded from GT commits (not deploy)",
    )
    ap.add_argument("--out", default="outputs/biochem/clot_trigger/t0_oracle.json")
    ap.add_argument("--min-mean-f1", type=float, default=None)
    ap.add_argument(
        "--min-full-mesh-f1",
        type=float,
        default=None,
        help="Gate on deploy full-mesh F1 (honest deploy metric)",
    )
    args = ap.parse_args()

    apply_clot_trigger_honest_env()
    apply_physics_trigger_baseline_env()
    if args.oracle_band:
        apply_oracle_neighbor_mask_env()
    if args.oracle_forward:
        apply_clot_trigger_oracle_forward_env()
    if args.prior_gate:
        os.environ["CLOT_PHI_PHYSICS_GELATION_GATE"] = "1"

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

    rows = [
        eval_anchor_t0_oracle(
            p,
            device=device,
            phys_cfg=phys_cfg,
            bio_cfg=bio_cfg,
            prior_gate=bool(args.prior_gate),
        )
        for p in paths
    ]
    for r in rows:
        print(
            f"[OK] {r['anchor']}: sup_F1={r['mean_support_f1']:.3f} "
            f"full_F1={r['mean_full_mesh_f1']:.3f} "
            f"lumen_fp={r['mean_lumen_fp_deploy']:.3f} "
            f"traj={r['trajectory_score']:.3f}",
            flush=True,
        )

    val_row = next((r for r in rows if r["anchor"] == args.val), None)
    cfg = snapshot_clot_trigger_rollout_config()
    summary = {
        "step": "t0_oracle",
        "inputs": "GT_flow + GT_species (no mu input, no teacher)",
        "trigger": "physics_mu_eff_si + nucleation projection",
        "loss_scope": clot_phi_loss_scope(),
        "rollout_config": cfg,
        "deploy_faithful_forward": not forward_path_uses_gt_commits(),
        "prior_gate": bool(args.prior_gate),
        "mean_support_f1": _mean(rows, "mean_support_f1"),
        "mean_full_mesh_f1": _mean(rows, "mean_full_mesh_f1"),
        "mean_lumen_fp_deploy": _mean(rows, "mean_lumen_fp_deploy"),
        "mean_lumen_fp_raw": _mean(rows, "mean_lumen_fp_raw"),
        "mean_trajectory_score": _mean(rows, "trajectory_score"),
        "val_anchor": args.val,
        "val_mean_support_f1": float(val_row["mean_support_f1"]) if val_row else float("nan"),
        "val_mean_full_mesh_f1": float(val_row["mean_full_mesh_f1"]) if val_row else float("nan"),
        "val_mean_lumen_fp_deploy": float(val_row["mean_lumen_fp_deploy"]) if val_row else float("nan"),
        "val_trajectory_score": float(val_row["trajectory_score"]) if val_row else float("nan"),
        "per_anchor": [{k: v for k, v in r.items() if k != "per_step"} for r in rows],
    }
    out_path = _REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[save] {out_path}", flush=True)
    print(
        f"[summary] sup_F1={summary['mean_support_f1']:.3f} "
        f"full_F1={summary['mean_full_mesh_f1']:.3f} "
        f"lumen_fp={summary['mean_lumen_fp_deploy']:.3f} "
        f"deploy_forward={summary['deploy_faithful_forward']}",
        flush=True,
    )

    if args.min_mean_f1 is not None and summary["mean_support_f1"] < float(args.min_mean_f1):
        print(f"[FAIL] mean support F1 {summary['mean_support_f1']:.3f} < {args.min_mean_f1}", flush=True)
        return 1
    if args.min_full_mesh_f1 is not None and summary["mean_full_mesh_f1"] < float(args.min_full_mesh_f1):
        print(
            f"[FAIL] mean full-mesh F1 {summary['mean_full_mesh_f1']:.3f} < {args.min_full_mesh_f1}",
            flush=True,
        )
        return 1
    if not summary["deploy_faithful_forward"]:
        print("[WARN] forward envelope uses GT commits (--oracle-forward); not deploy-faithful", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
