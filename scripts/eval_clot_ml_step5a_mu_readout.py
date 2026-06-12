"""F1-F3: Step 5a mu readout LOAO eval (frozen phi shell -> mu_eff field)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.core_physics.clot_temporal_growth_rules import reset_temporal_kinematics_cache, temporal_vel_source
from src.training.clot_ml_step5a_mu_readout import PhiShellKind, Step5aEvalConfig, eval_loao_step5a
from src.training.clot_ml_device import resolve_clot_ml_eval_device


def main() -> int:
    ap = argparse.ArgumentParser(description="Step 5a mu readout LOAO eval")
    ap.add_argument("--shell", choices=[s.value for s in PhiShellKind], default="step1")
    ap.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--step0-json", default="outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json")
    ap.add_argument("--step1-ckpt", default="outputs/biochem/sweep_clot_ml_physics_6h/step1_a35/clot_ml_step1_best.pth")
    ap.add_argument("--lane-b-ckpt", default="")
    ap.add_argument("--kine-ckpt", default="")
    ap.add_argument("--out", default="outputs/biochem/clot_ml_ladder/step5a_mu_readout/summary.json")
    args = ap.parse_args()

    os.environ["CLOT_TEMPORAL_VEL_SOURCE"] = "kinematics"
    if args.kine_ckpt.strip():
        os.environ["CLOT_PHI_KINE_CKPT"] = args.kine_ckpt.strip()

    device = resolve_clot_ml_eval_device()
    reset_temporal_kinematics_cache()

    cfg = Step5aEvalConfig(
        anchor_dir=args.anchor_dir,
        step0_json=args.step0_json,
        step1_ckpt=args.step1_ckpt,
        lane_b_ckpt=args.lane_b_ckpt,
        shell=PhiShellKind(args.shell),
    )
    summary = eval_loao_step5a(cfg, device=device)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[i] shell={args.shell} vel={temporal_vel_source()} device={device}")
    print(f"[i] mean_deploy={summary['mean_deploy']:.3f}")
    for row in summary["per_anchor"]:
        print(
            f"  {row['anchor']:12} deploy={row['deploy_score']:.3f} "
            f"F1={row['tfinal_band_f1']:.3f} mu_shape={row.get('mu_shape_f1', float('nan')):.3f} "
            f"pred+={row['tfinal_band_pred_frac']:.3f}"
        )
    print(f"[save] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
