"""Promote compound deploy stack (WC_v7 wall + lumen growth specialist).

Copies growth ckpt to locked alias, writes data/reference manifest, snapshots eval.
Wall backbone remains locked/species_gnn_best.pth (WC_v7_clot_phi_mse).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.evaluation.compound_deploy_gates import (  # noqa: E402
    format_gate_summary,
    gate_compound_eval_report,
)
from src.utils.paths import get_project_root  # noqa: E402

DEFAULT_LEG = "WC_v8_compound_front_h1"
DEFAULT_LABEL = "WC v8 compound: WC_v7 wall + frontier-h1 lumen specialist"
LOCKED_GROWTH = "outputs/biochem/biochem_gnn/locked/compound_growth_best.pth"
LOCKED_WALL = "outputs/biochem/biochem_gnn/locked/species_gnn_best.pth"
REFERENCE_JSON = "data/reference/mat_compound_deploy.json"
COMPOUND_ROOT = "outputs/biochem/biochem_gnn/compound_deploy"


def _copy_ckpt(src: Path, dst: Path, *, skip_copy: bool) -> bool:
    if not src.is_file():
        print(f"[ERR] missing growth ckpt: {src}", file=sys.stderr)
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if skip_copy and dst.is_file():
        print(f"[skip] {dst.name} exists", flush=True)
        return True
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    print(f"[OK] {dst.relative_to(REPO)} <- {src.relative_to(REPO)}", flush=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Promote compound deploy (wall + growth)")
    ap.add_argument("--growth-src", required=True, help="Growth specialist best.pth")
    ap.add_argument("--eval-json", default="", help="orig10 compound eval JSON")
    ap.add_argument("--wall-floor-json", default="", help="wall-alone probe for guardrail")
    ap.add_argument("--leg", default=DEFAULT_LEG)
    ap.add_argument("--label", default=DEFAULT_LABEL)
    ap.add_argument(
        "--route",
        default="frontier",
        choices=("wall", "frontier", "frontier_offwall", "frontier_lumen_only"),
    )
    ap.add_argument("--frontier-hops", type=float, default=1.0)
    ap.add_argument("--skip-copy", action="store_true")
    args = ap.parse_args()

    root = get_project_root()
    growth_src = Path(args.growth_src)
    if not growth_src.is_absolute():
        growth_src = root / growth_src

    locked_growth = root / LOCKED_GROWTH
    compound_dir = root / COMPOUND_ROOT
    if not _copy_ckpt(growth_src, locked_growth, skip_copy=args.skip_copy):
        return 1
    _copy_ckpt(growth_src, compound_dir / "growth" / "best.pth", skip_copy=args.skip_copy)
    meta_src = growth_src.with_suffix(".json")
    if meta_src.is_file():
        shutil.copy2(meta_src, compound_dir / "growth" / "best.json")

    wall_floor_f1 = None
    if args.wall_floor_json.strip():
        wp = Path(args.wall_floor_json)
        if not wp.is_absolute():
            wp = root / wp
        if wp.is_file():
            rep = json.loads(wp.read_text(encoding="utf-8"))
            simple = rep.get("simple") or rep
            wall_floor_f1 = float((simple.get("mean") or {}).get("deploy_clot_f1", 0) or 0)

    gate = {}
    eval_path = None
    if args.eval_json.strip():
        eval_path = Path(args.eval_json)
        if not eval_path.is_absolute():
            eval_path = root / eval_path
        if eval_path.is_file():
            rep = json.loads(eval_path.read_text(encoding="utf-8"))
            gate = gate_compound_eval_report(rep, wall_floor_f1=wall_floor_f1)
            print(f"[i] gates: {format_gate_summary(gate)}", flush=True)

    promoted_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest = {
        "stack": "compound_deploy",
        "leg": args.leg,
        "label": args.label,
        "promoted_at": promoted_at,
        "wall_ckpt": LOCKED_WALL,
        "wall_leg": "WC_v7_clot_phi_mse",
        "wall_note": (
            "Locked WC_v7 remains the wall backbone and strong wall-only deploy. "
            "Compound stacks growth specialist on top via two-model routing."
        ),
        "growth_ckpt": LOCKED_GROWTH,
        "growth_source": str(growth_src.relative_to(root)).replace("\\", "/"),
        "deploy": {
            "SPECIES_TWO_MODEL_MODE": "1",
            "SPECIES_TWO_MODEL_ROUTE": args.route,
            "SPECIES_TWO_MODEL_FRONTIER_HOPS": str(float(args.frontier_hops)),
            "SPECIES_CONTINUOUS_VEL_DECAY": "1",
            "SPECIES_CONTINUOUS_VEL_DECAY_WALL_ONLY": "1",
            "mat_leg": "WC_v7_clot_phi_mse",
        },
        "eval_json": str(eval_path.relative_to(root)).replace("\\", "/") if eval_path else None,
        "compound_gates": gate,
        "prior_wall_canonical": "WC_v7_clot_phi_mse (2026-07-19)",
        "prior_growth_lineage": "wc_v7_wall_lumen_target_9h/growth_C (Prec8h FP polish, WALL_ONLY)",
    }
    if gate:
        manifest["cohort_mean"] = {
            "deploy_clot_f1": gate.get("mean_f1"),
            "deploy_clot_score": gate.get("mean_score"),
            "deploy_clot_offwall_relaxed_f1": gate.get("mean_offwall_relaxed_f1"),
            "deploy_clot_offwall_strict_f1_hop_ge2": gate.get("mean_hop_ge2_strict"),
            "ge2_recall": gate.get("ge2_recall"),
        }

    ref_path = root / REFERENCE_JSON
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    ref_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (compound_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if eval_path and eval_path.is_file():
        shutil.copy2(eval_path, compound_dir / "eval_orig10.json")

    print(f"[save] {ref_path.relative_to(root)}", flush=True)
    print(f"[OK] promoted compound leg={args.leg} route={args.route} hops={args.frontier_hops}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
