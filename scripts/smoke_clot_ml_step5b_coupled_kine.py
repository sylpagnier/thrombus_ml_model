"""F4: Step 5b smoke -- mu_eff prior vs frozen GINO-DEQ on one anchor."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.core_physics.clot_temporal_growth_rules import reset_temporal_kinematics_cache
from src.training.clot_ml_step5b_coupled_smoke import smoke_coupled_kine_one_anchor
from src.training.clot_ml_device import resolve_clot_ml_eval_device


def main() -> int:
    ap = argparse.ArgumentParser(description="Step 5b coupled kine smoke")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--shell", choices=["inc40", "step1"], default="step1")
    ap.add_argument("--step0-json", default="outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json")
    ap.add_argument("--step1-ckpt", default="outputs/biochem/sweep_clot_ml_physics_6h/step1_a35/clot_ml_step1_best.pth")
    ap.add_argument("--kine-ckpt", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    os.environ["CLOT_TEMPORAL_VEL_SOURCE"] = "kinematics"
    if args.kine_ckpt.strip():
        os.environ["CLOT_PHI_KINE_CKPT"] = args.kine_ckpt.strip()

    device = resolve_clot_ml_eval_device()
    reset_temporal_kinematics_cache()

    graph = REPO / args.anchor_dir / f"{args.anchor}.pt"
    row = smoke_coupled_kine_one_anchor(
        graph,
        shell=args.shell,
        step0_json=args.step0_json,
        step1_ckpt=args.step1_ckpt,
        kine_ckpt=args.kine_ckpt,
        device=device,
    )
    if args.out.strip():
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = REPO / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
        print(f"[save] {out_path}")

    print(f"[i] anchor={row['anchor']} shell={row['shell']} t_out={row['t_out']}")
    print(f"[i] rel_L2 frozen vs GT uvp={row['rel_l2_frozen_kine']:.4f}")
    print(f"[i] rel_L2 coupled vs GT uvp={row['rel_l2_coupled_kine']:.4f}")
    print(f"[i] rel_L2 coupled vs frozen={row['rel_l2_coupled_vs_frozen']:.4f}")
    print(f"[i] phi_commit_frac={row['phi_commit_frac']:.3f} mu_max/mean={row['mu_eff_max_over_mean']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
