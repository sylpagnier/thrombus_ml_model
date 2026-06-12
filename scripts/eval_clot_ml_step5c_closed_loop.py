"""Step 5c: frozen vs coupled LOAO deploy comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.training.clot_ml_device import resolve_clot_ml_eval_device
from src.training.clot_ml_step5c_closed_loop import compare_frozen_vs_coupled


def main() -> int:
    ap = argparse.ArgumentParser(description="Step 5c closed-loop LOAO")
    ap.add_argument("--recipe", default="data/reference/clot_ml_deploy_v1.json")
    ap.add_argument("--sim-end-scale", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--out", default="outputs/biochem/clot_ml_ladder/deploy_v1/step5c_compare.json")
    args = ap.parse_args()

    device = resolve_clot_ml_eval_device()
    summary = compare_frozen_vs_coupled(
        REPO / args.recipe,
        device=device,
        sim_end_scale=args.sim_end_scale,
        deploy_drop_tol=args.tol,
    )

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[i] frozen={summary['frozen_mean_deploy']:.3f} coupled={summary['coupled_mean_deploy']:.3f}")
    print(f"[i] delta={summary['mean_deploy_delta']:+.3f} pass_5c={summary['pass_5c_gate']}")
    print(f"[save] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
