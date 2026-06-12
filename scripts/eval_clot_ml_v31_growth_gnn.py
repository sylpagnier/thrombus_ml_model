"""V3.1 LOAO eval vs V3 and V1."""

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
    default_v31_out_dir,
    eval_v3_on_anchor,
    load_v3_checkpoint,
    resolve_v3_rule_cfg,
    v31_manifest_dict,
    val_score_from_eval_row,
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
    ap = argparse.ArgumentParser(description="V3.1 eval vs V3/V1")
    ap.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--step0-json", default="outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json")
    ap.add_argument("--v31-ckpt", default="outputs/biochem/clot_ml_ladder_v2/v31_growth_gnn/clot_ml_v31_growth_gnn_best.pth")
    ap.add_argument("--v3-ckpt", default="outputs/biochem/clot_ml_ladder_v2/v3_growth_gnn/clot_ml_v3_growth_gnn_best.pth")
    ap.add_argument("--step1-ckpt", default="outputs/biochem/sweep_clot_ml_physics_6h/step1_a35/clot_ml_step1_best.pth")
    ap.add_argument("--val", default="patient007")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = REPO
    device = resolve_clot_ml_eval_device()
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    rule_cfg = resolve_v3_rule_cfg(root / args.step0_json)

    v31_model, _ = load_v3_checkpoint(root / args.v31_ckpt, device=device, v31=True)
    v3_model, _ = load_v3_checkpoint(root / args.v3_ckpt, device=device, v31=False)
    step1_model, step1_meta = load_step1_checkpoint(root / args.step1_ckpt, device=device)
    alpha = float(step1_meta.get("alpha", 0.35))

    rows = []
    for p in discover_anchor_paths(root / args.anchor_dir):
        gp = Path(p)
        v1 = eval_step1_v1_on_anchor(
            step1_model, rule_cfg, graph_path=gp, device=device, phys_cfg=phys, bio_cfg=bio, alpha=alpha
        )
        v3 = eval_v3_on_anchor(
            v3_model, rule_cfg, graph_path=gp, device=device, phys_cfg=phys, bio_cfg=bio
        )
        v31 = eval_v3_on_anchor(
            v31_model, rule_cfg, graph_path=gp, device=device, phys_cfg=phys, bio_cfg=bio
        )
        v31["val_score"] = val_score_from_eval_row(v31)
        v3["val_score"] = val_score_from_eval_row(v3)
        rows.append(
            {
                "anchor": gp.stem,
                "v1": v1,
                "v3": v3,
                "v31": v31,
                "delta_shape_v31_v3": float(v31["tfinal_clot_shape"] - v3["tfinal_clot_shape"]),
                "delta_deploy_v31_v3": float(v31["deploy_score"] - v3["deploy_score"]),
            }
        )

    val_row = next((r for r in rows if r["anchor"] == args.val), None)
    mean_shape_v31 = sum(r["v31"]["tfinal_clot_shape"] for r in rows) / max(len(rows), 1)
    mean_shape_v3 = sum(r["v3"]["tfinal_clot_shape"] for r in rows) / max(len(rows), 1)
    pass_gate = val_row is not None and (
        float(val_row["v31"]["tfinal_clot_shape"]) >= float(val_row["v3"]["tfinal_clot_shape"])
        and float(val_row["v31"]["tfinal_band_pred_frac"]) <= float(val_row["v3"]["tfinal_band_pred_frac"]) + 0.05
    )

    payload = {
        "step": "v31_growth_gnn",
        "v31_ckpt": str(args.v31_ckpt),
        "mean_clot_shape_v31": mean_shape_v31,
        "mean_clot_shape_v3": mean_shape_v3,
        "pass_gate_v31": pass_gate,
        "anchors": rows,
        "manifest": v31_manifest_dict(ckpt=str(args.v31_ckpt)),
    }

    out_path = Path(args.out) if args.out.strip() else default_v31_out_dir() / "eval_loao.json"
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_sanitize(payload), indent=2), encoding="utf-8")

    print(f"[i] mean_shape v31={mean_shape_v31:.3f} v3={mean_shape_v3:.3f} pass_gate_v31={pass_gate}")
    if val_row:
        print(
            f"[i] val {args.val}: shape v31={val_row['v31']['tfinal_clot_shape']:.3f} "
            f"v3={val_row['v3']['tfinal_clot_shape']:.3f} pred+={val_row['v31']['tfinal_band_pred_frac']:.3f}",
            flush=True,
        )
    print(f"[save] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
