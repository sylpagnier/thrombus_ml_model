"""V3 LOAO eval: growth GNN vs V1 nucleation baseline."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.training.clot_ml_device import resolve_clot_ml_eval_device  # noqa: E402
from src.training.clot_ml_step0_coef import discover_anchor_paths  # noqa: E402
from src.training.clot_ml_step1_residual import load_step1_checkpoint  # noqa: E402
from src.training.clot_ml_v2_growth_gnn import (  # noqa: E402
    default_v3_out_dir,
    eval_v3_on_anchor,
    load_v3_checkpoint,
    resolve_v3_rule_cfg,
    v3_manifest_dict,
)
from src.training.clot_ml_v2_step1_nucleation import eval_step1_v1_on_anchor  # noqa: E402


def _sanitize(obj: object) -> object:
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    return obj


def main() -> int:
    ap = argparse.ArgumentParser(description="V3 growth GNN LOAO eval vs V1")
    ap.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--step0-json", default="outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json")
    ap.add_argument("--v3-ckpt", default="outputs/biochem/clot_ml_ladder_v2/v3_growth_gnn/clot_ml_v3_growth_gnn_best.pth")
    ap.add_argument("--step1-ckpt", default="outputs/biochem/sweep_clot_ml_physics_6h/step1_a35/clot_ml_step1_best.pth")
    ap.add_argument("--val", default="patient007")
    ap.add_argument("--v1-mean-baseline", type=float, default=-1.0, help="Override V1 mean deploy for gate")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = REPO
    device = resolve_clot_ml_eval_device()
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    rule_cfg = resolve_v3_rule_cfg(root / args.step0_json)

    v3_model, _ = load_v3_checkpoint(root / args.v3_ckpt, device=device)
    step1_model, step1_meta = load_step1_checkpoint(root / args.step1_ckpt, device=device)
    alpha = float(step1_meta.get("alpha", 0.35))

    paths = [str(p) for p in discover_anchor_paths(root / args.anchor_dir)]
    rows = []
    for p in paths:
        gp = Path(p)
        v1_row = eval_step1_v1_on_anchor(
            step1_model,
            rule_cfg,
            graph_path=gp,
            device=device,
            phys_cfg=phys,
            bio_cfg=bio,
            alpha=alpha,
        )
        v3_row = eval_v3_on_anchor(
            v3_model,
            rule_cfg,
            graph_path=gp,
            device=device,
            phys_cfg=phys,
            bio_cfg=bio,
        )
        rows.append(
            {
                "anchor": gp.stem,
                "v1": v1_row,
                "v3": v3_row,
                "delta_deploy": float(v3_row["deploy_score"] - v1_row["deploy_score"]),
                "delta_tfinal_f1": float(v3_row["tfinal_band_f1"] - v1_row["tfinal_band_f1"]),
            }
        )

    mean_v1 = sum(r["v1"]["deploy_score"] for r in rows) / max(len(rows), 1)
    mean_v3 = sum(r["v3"]["deploy_score"] for r in rows) / max(len(rows), 1)
    mean_delta = mean_v3 - mean_v1
    v1_ref = float(args.v1_mean_baseline) if float(args.v1_mean_baseline) >= 0 else mean_v1
    val_row = next((r for r in rows if r["anchor"] == args.val), None)

    pass_lift = mean_v3 >= v1_ref + 0.03
    pass_p007 = False
    if val_row is not None:
        pass_p007 = float(val_row["v3"]["tfinal_band_f1"]) >= 0.50
    pass_gate_v3 = pass_lift or pass_p007

    payload = {
        "step": "v3_growth_gnn",
        "v3_ckpt": str(args.v3_ckpt),
        "step1_ckpt": str(args.step1_ckpt),
        "val": args.val,
        "mean_deploy_v1": mean_v1,
        "mean_deploy_v3": mean_v3,
        "mean_delta_deploy": mean_delta,
        "pass_lift_vs_v1": pass_lift,
        "pass_p007_f1": pass_p007,
        "pass_gate_v3": pass_gate_v3,
        "anchors": rows,
        "manifest": v3_manifest_dict(ckpt=str(args.v3_ckpt), step0_json=str(args.step0_json)),
    }

    out_dir = default_v3_out_dir()
    out_path = Path(args.out) if args.out.strip() else out_dir / "eval_loao.json"
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_sanitize(payload), indent=2), encoding="utf-8")

    print(f"[i] anchors={len(rows)} v1={mean_v1:.3f} v3={mean_v3:.3f} delta={mean_delta:+.3f}")
    if val_row:
        print(
            f"[i] val {args.val}: v3 F1={val_row['v3']['tfinal_band_f1']:.3f} "
            f"d_deploy={val_row['delta_deploy']:+.3f}",
            flush=True,
        )
    print(f"[i] pass_gate_v3={pass_gate_v3} (lift={pass_lift} p007_f1={pass_p007})")
    for r in rows:
        print(
            f"  {r['anchor']:12s} v1={r['v1']['deploy_score']:.3f} "
            f"v3={r['v3']['deploy_score']:.3f} d={r['delta_deploy']:+.3f} "
            f"f1_d={r['delta_tfinal_f1']:+.3f}",
            flush=True,
        )
    print(f"[save] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
