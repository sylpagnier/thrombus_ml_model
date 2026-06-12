"""V2 LOAO: V1 nucleation vs V2 continuous tau (in-window parity + extrap metrics)."""

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
from src.training.clot_ml_v2_continuous_tau import (  # noqa: E402
    compare_v1_vs_v2_on_anchor,
    default_v2_out_dir,
    resolve_step1_rule_cfg_from_root,
    v2_manifest_dict,
)
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
    ap = argparse.ArgumentParser(description="V2 continuous tau vs V1 nucleation LOAO")
    ap.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--step0-json", default="outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json")
    ap.add_argument("--step1-ckpt", default="outputs/biochem/sweep_clot_ml_physics_6h/step1_a35/clot_ml_step1_best.pth")
    ap.add_argument("--val", default="patient007")
    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--sim-end-scale", type=float, default=2.0, help="Extrap horizon for axis-C metrics")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = get_project_root()
    device = resolve_clot_ml_eval_device()
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    rule_cfg = resolve_step1_rule_cfg_from_root(args.step0_json)
    model, meta = load_step1_checkpoint(root / args.step1_ckpt, device=device)
    alpha = float(meta.get("alpha", args.alpha))

    paths = [str(p) for p in discover_anchor_paths(root / args.anchor_dir)]
    rows = []
    for p in paths:
        rows.append(
            compare_v1_vs_v2_on_anchor(
                model,
                rule_cfg,
                graph_path=Path(p),
                device=device,
                phys_cfg=phys,
                bio_cfg=bio,
                alpha=alpha,
                sim_end_scale_extrap=float(args.sim_end_scale),
            )
        )

    def _mean(key: str, branch: str) -> float:
        vals = [float(r[branch][key]) for r in rows if r.get(branch)]
        return sum(vals) / max(len(vals), 1)

    mean_v1 = _mean("deploy_score", "v1")
    mean_v2 = _mean("deploy_score", "v2_inwindow")
    mean_delta = mean_v2 - mean_v1
    val_row = next((r for r in rows if r["anchor"] == args.val), None)

    pass_inwindow = abs(mean_delta) <= 0.02
    mono_ok = all(bool(r["v2_extrap"].get("monotone_commit_frac", False)) for r in rows)
    pass_gate_v2 = pass_inwindow and mono_ok

    payload = {
        "step": "v2_continuous_tau",
        "step1_ckpt": str(args.step1_ckpt),
        "val": args.val,
        "alpha": alpha,
        "sim_end_scale_extrap": float(args.sim_end_scale),
        "mean_deploy_v1": mean_v1,
        "mean_deploy_v2_inwindow": mean_v2,
        "mean_delta_deploy": mean_delta,
        "pass_inwindow_parity": pass_inwindow,
        "pass_extrap_monotone": mono_ok,
        "pass_gate_v2": pass_gate_v2,
        "anchors": rows,
        "manifest": v2_manifest_dict(
            step1_ckpt=str(args.step1_ckpt),
            sim_end_scale=float(args.sim_end_scale),
        ),
    }

    out_dir = default_v2_out_dir()
    out_path = Path(args.out) if args.out.strip() else out_dir / "eval_loao.json"
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_sanitize(payload), indent=2), encoding="utf-8")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload["manifest"], indent=2) + "\n", encoding="utf-8")

    print(
        f"[i] anchors={len(rows)} mean_deploy v1={mean_v1:.3f} v2_in={mean_v2:.3f} "
        f"delta={mean_delta:+.3f} extrap_scale={float(args.sim_end_scale):.1f}",
        flush=True,
    )
    if val_row:
        ex = val_row["v2_extrap"]
        print(
            f"[i] val {args.val}: d_deploy={val_row['delta_deploy_v1_v2']:+.3f} "
            f"extrap_dfrac={ex.get('commit_frac_delta_extrap', float('nan')):+.4f} "
            f"mono={ex.get('monotone_commit_frac')}",
            flush=True,
        )
    print(f"[i] pass_inwindow={pass_inwindow} pass_extrap_mono={mono_ok} pass_gate_v2={pass_gate_v2}")
    for r in rows:
        ex = r["v2_extrap"]
        print(
            f"  {r['anchor']:12s} v1={r['v1']['deploy_score']:.3f} "
            f"v2={r['v2_inwindow']['deploy_score']:.3f} d={r['delta_deploy_v1_v2']:+.3f} "
            f"ex_dfrac={ex.get('commit_frac_delta_extrap', float('nan')):+.4f}",
            flush=True,
        )
    print(f"[save] {out_path}")
    print(f"[save] {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
