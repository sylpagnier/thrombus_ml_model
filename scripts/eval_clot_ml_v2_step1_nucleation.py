"""V1 LOAO: ceiling step1 vs nucleation step1 (frozen step1_a35 weights)."""

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
from src.training.clot_ml_step1_residual import load_step1_checkpoint, resolve_step1_rule_cfg  # noqa: E402
from src.training.clot_ml_v2_step1_nucleation import (  # noqa: E402
    compare_ceiling_vs_nucleation_on_anchor,
    default_v1_out_dir,
    v1_manifest_dict,
)
from src.training.train_clot_phi_simple import _split_train_val  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402


def _sanitize(obj: object) -> object:
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    return obj


def main() -> int:
    ap = argparse.ArgumentParser(description="V1 step1 nucleation vs ceiling LOAO")
    ap.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--step0-json", default="outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json")
    ap.add_argument("--step1-ckpt", default="outputs/biochem/sweep_clot_ml_physics_6h/step1_a35/clot_ml_step1_best.pth")
    ap.add_argument("--val", default="patient007")
    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = get_project_root()
    device = resolve_clot_ml_eval_device()
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    rule_cfg = resolve_step1_rule_cfg(root / args.step0_json)
    model, meta = load_step1_checkpoint(root / args.step1_ckpt, device=device)
    alpha = float(meta.get("alpha", args.alpha))

    paths = [str(p) for p in discover_anchor_paths(root / args.anchor_dir)]
    train_paths, val_paths = _split_train_val(paths, args.val)

    rows = []
    for p in paths:
        rows.append(
            compare_ceiling_vs_nucleation_on_anchor(
                model,
                rule_cfg,
                graph_path=Path(p),
                device=device,
                phys_cfg=phys,
                bio_cfg=bio,
                alpha=alpha,
            )
        )

    def _mean(key: str, branch: str) -> float:
        vals = [float(r[branch][key]) for r in rows if r.get(branch)]
        return sum(vals) / max(len(vals), 1)

    mean_ceil = _mean("deploy_score", "ceiling")
    mean_nuc = _mean("deploy_score", "nucleation")
    mean_delta = mean_nuc - mean_ceil
    val_row = next((r for r in rows if r["anchor"] == args.val), None)

    pass_gate = abs(mean_delta) <= 0.03
    if val_row is not None:
        pass_gate = pass_gate or (
            float(val_row["delta_tfinal_f1"]) >= -0.05
            and float(val_row["delta_tfinal_pred_frac"]) <= 0.02
        )

    payload = {
        "step": "v1_nucleation",
        "step1_ckpt": str(args.step1_ckpt),
        "val": args.val,
        "alpha": alpha,
        "mean_deploy_ceiling": mean_ceil,
        "mean_deploy_nucleation": mean_nuc,
        "mean_delta_deploy": mean_delta,
        "pass_gate_v1": pass_gate,
        "anchors": rows,
        "manifest": v1_manifest_dict(step1_ckpt=str(args.step1_ckpt)),
    }

    out_dir = default_v1_out_dir()
    out_path = Path(args.out) if args.out.strip() else out_dir / "eval_loao.json"
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_sanitize(payload), indent=2), encoding="utf-8")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload["manifest"], indent=2) + "\n", encoding="utf-8")

    print(f"[i] anchors={len(rows)} mean_deploy ceiling={mean_ceil:.3f} nucleation={mean_nuc:.3f} delta={mean_delta:+.3f}")
    if val_row:
        print(
            f"[i] val {args.val}: d_deploy={val_row['delta_deploy_score']:+.3f} "
            f"d_f1={val_row['delta_tfinal_f1']:+.3f} d_pred+={val_row['delta_tfinal_pred_frac']:+.4f}",
            flush=True,
        )
    print(f"[i] pass_gate_v1={pass_gate}")
    for r in rows:
        print(
            f"  {r['anchor']:12s} ceil={r['ceiling']['deploy_score']:.3f} "
            f"nuc={r['nucleation']['deploy_score']:.3f} d={r['delta_deploy_score']:+.3f} "
            f"f1_d={r['delta_tfinal_f1']:+.3f}",
            flush=True,
        )
    print(f"[save] {out_path}")
    print(f"[save] {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
