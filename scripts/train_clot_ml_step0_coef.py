"""Step 0 ladder: learn clot rule coefficients (pred kine, LOAO optional).

Usage:
  python scripts/train_clot_ml_step0_coef.py
  python scripts/train_clot_ml_step0_coef.py --fast --anchor patient007
  python scripts/train_clot_ml_step0_coef.py --loao
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.training.clot_ml_device import resolve_clot_ml_training_device  # noqa: E402
from src.training.clot_ml_step0_coef import (  # noqa: E402
    Step0RuleCoefs,
    clear_step0_eval_cache,
    default_out_dir,
    discover_anchor_paths,
    eval_coef_on_anchors,
    optimize_coef_on_anchors,
    run_loao_step0,
)


def _apply_deploy_env() -> None:
    os.environ["BIOCHEM_PRIOR_COMSOL_ALIGNED"] = "1"
    os.environ["BIOCHEM_PRIOR_NORM_MASK"] = "adjacent"
    os.environ["CLOT_PHI_DGAMMA_SLICE"] = "1"
    os.environ["CLOT_PHI_CEILING_HOPS"] = "2"
    os.environ["CLOT_FORECAST_MODE"] = "one_step"
    os.environ["CLOT_FORECAST_MASK"] = "ceiling_growth"
    os.environ["CLOT_FORECAST_PAIR_SCHEDULE"] = "from_t0"
    os.environ["CLOT_FORECAST_PAIR_STRIDE"] = "1"
    os.environ["CLOT_TEMPORAL_VEL_SOURCE"] = "kinematics"
    os.environ.setdefault("CLOT_PHI_KINE_CKPT", "outputs/kinematics/kinematics_best.pth")


def _sanitize(obj: object) -> object:
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    return obj


def main() -> int:
    ap = argparse.ArgumentParser(description="Step 0: learn clot rule coefficients")
    ap.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--anchor", default="", help="Single-anchor fast mode (no LOAO)")
    ap.add_argument("--loao", action="store_true", help="Leave-one-anchor-out optimization")
    ap.add_argument("--fast", action="store_true", help="Fewer DE iterations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--maxiter", type=int, default=0)
    ap.add_argument("--popsize", type=int, default=0)
    ap.add_argument(
        "--de",
        action="store_true",
        help="Use differential evolution (slow; default is random search + polish)",
    )
    args = ap.parse_args()

    _apply_deploy_env()
    clear_step0_eval_cache()
    anchor_dir = REPO / args.anchor_dir
    if not anchor_dir.is_dir():
        print(f"[ERR] missing anchor dir {anchor_dir}", file=sys.stderr)
        return 1

    maxiter = args.maxiter or (12 if args.fast else 28)
    popsize = args.popsize or (8 if args.fast else 12)
    method = "de" if args.de else "search"
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_clot_ml_training_device()

    if args.loao:
        print(f"[i] Step0 LOAO train maxiter={maxiter} popsize={popsize} pred-kine", flush=True)
        payload = run_loao_step0(
            anchor_dir,
            device=device,
            seed=args.seed,
            maxiter=maxiter,
            popsize=popsize,
            method=method,
        )
    else:
        paths = discover_anchor_paths(anchor_dir)
        if args.anchor.strip():
            paths = [anchor_dir / f"{args.anchor.strip()}.pt"]
        if not paths:
            print("[ERR] no anchors", file=sys.stderr)
            return 1
        print(
            f"[i] Step0 train anchors={[p.stem for p in paths]} "
            f"maxiter={maxiter} popsize={popsize}",
            flush=True,
        )
        baseline = Step0RuleCoefs.inc40_baseline()
        phys = PhysicsConfig(phase="biochem")
        bio = BiochemConfig(phase="biochem")
        base_rows = eval_coef_on_anchors(
            baseline, anchor_paths=paths, device=device, phys_cfg=phys, bio_cfg=bio
        )
        base_mean = sum(r["deploy_score"] for r in base_rows) / len(base_rows)
        best, _, rows = optimize_coef_on_anchors(
            paths,
            device=device,
            seed=args.seed,
            maxiter=maxiter,
            popsize=popsize,
            method=method,
        )
        mean_deploy = sum(r["deploy_score"] for r in rows) / len(rows)
        payload = {
            "step": 0,
            "mode": "single" if args.anchor else "full",
            "anchor_dir": str(anchor_dir),
            "vel_source": "kinematics",
            "baseline_inc40_mean_deploy": base_mean,
            "full_train_mean_deploy": mean_deploy,
            "coef": best.to_dict(),
            "vector": best.to_vector(),
            "per_anchor_final": rows,
            "pass_vs_baseline": mean_deploy >= base_mean + 0.01,
        }

    out_json = out_dir / "best_coef.json"
    out_json.write_text(json.dumps(_sanitize(payload), indent=2), encoding="utf-8")
    coef = payload["coef"]
    print(
        f"[OK] mean_deploy={payload.get('loao_mean_deploy') or payload.get('full_train_mean_deploy'):.3f} "
        f"baseline_inc40={payload['baseline_inc40_mean_deploy']:.3f}",
        flush=True,
    )
    print(
        f"[OK] coef onset={coef['onset']:.3f} neg_dx={coef['neg_dx']:.3f} "
        f"top={coef['top_frac']:.3f} end={coef['end_frac']:.3f}",
        flush=True,
    )
    print(f"[save] {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
