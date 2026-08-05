"""Eval-time frontier hops sweep for compound deploy (WC v8 improvement axis).

Sweeps hops 0, 0.5 (tight off-wall shell), 1, 2 and optional per-vessel map.
Writes gate JSON per setting under --out-dir.

Usage:
  python -u scripts/sweep_frontier_hops.py
  python -u scripts/sweep_frontier_hops.py --growth outputs/biochem/biochem_gnn/locked/compound_growth_best.pth
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

ORIG10 = (
    "patient001,patient002,patient003,patient004,patient005,"
    "patient006,patient007,patient008,patient010,patient011"
)
WALL = REPO / "outputs/biochem/biochem_gnn/locked/species_gnn_best.pth"
DEFAULT_GROWTH = REPO / "outputs/biochem/biochem_gnn/locked/compound_growth_best.pth"
WALL_FLOOR = REPO / "outputs/biochem/offwall_model/wc_v7_wall_lumen_target_9h/probe_A_wall_alone.json"

from src.evaluation.compound_deploy_gates import (  # noqa: E402
    format_gate_summary,
    gate_compound_eval_report,
)


def _run_eval(
    *,
    out: Path,
    growth: Path,
    wall: Path,
    hops: float,
    hops_map: str = "",
) -> dict:
    env = dict(os.environ)
    env["SPECIES_CONTINUOUS_VEL_DECAY"] = "1"
    env["SPECIES_CONTINUOUS_VEL_DECAY_WALL_ONLY"] = "1"
    cmd = [
        sys.executable,
        "-u",
        str(REPO / "scripts" / "eval_mat_growth_simple.py"),
        "--ckpt",
        str(wall),
        "--mat-leg",
        "WC_v7_clot_phi_mse",
        "--no-baseline",
        "--out",
        str(out),
        "--anchors",
        ORIG10,
        "--offwall-ckpt",
        str(growth),
        "--two-model-route",
        "frontier",
        "--two-model-frontier-hops",
        str(hops),
    ]
    if hops_map.strip():
        cmd.extend(["--two-model-frontier-hops-map", hops_map.strip()])
    if WALL_FLOOR.is_file():
        cmd.extend(["--wall-floor-json", str(WALL_FLOOR)])
    print(f"[RUN] hops={hops} map={hops_map or '-'} -> {out.name}", flush=True)
    rc = subprocess.call(cmd, cwd=str(REPO), env=env)
    if rc != 0:
        raise RuntimeError(f"eval failed rc={rc} hops={hops}")
    rep = json.loads(out.read_text(encoding="utf-8"))
    gates = rep.get("compound_gates") or gate_compound_eval_report(rep)
    gate_path = out.with_name(out.stem + "_gate.json")
    gate_path.write_text(json.dumps(gates, indent=2), encoding="utf-8")
    print(f"[i] {format_gate_summary(gates)}", flush=True)
    return {"eval": str(out), "gates": gates}


def main() -> int:
    ap = argparse.ArgumentParser(description="Frontier hops eval sweep for compound deploy")
    ap.add_argument("--growth", default=str(DEFAULT_GROWTH))
    ap.add_argument("--wall", default=str(WALL))
    ap.add_argument(
        "--out-dir",
        default="outputs/biochem/offwall_model/wc_v8_improvement_sweeps/hops_sweep",
    )
    ap.add_argument(
        "--hops",
        default="0,0.5,1,2",
        help="Comma-separated hops values to sweep",
    )
    ap.add_argument(
        "--per-vessel-map",
        default="patient010:0.5,patient006:0.5,default:1",
        help="Optional per-vessel map eval (single extra probe)",
    )
    ap.add_argument("--skip-per-vessel", action="store_true")
    args = ap.parse_args()

    growth = Path(args.growth)
    wall = Path(args.wall)
    if not growth.is_absolute():
        growth = REPO / growth
    if not wall.is_absolute():
        wall = REPO / wall
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {"growth": str(growth), "wall": str(wall), "sweeps": {}}
    for raw in args.hops.split(","):
        piece = raw.strip()
        if not piece:
            continue
        hops = float(piece)
        tag = str(hops).replace(".", "p")
        out = out_dir / f"eval_frontier_h{tag}.json"
        results["sweeps"][f"h{tag}"] = _run_eval(
            out=out, growth=growth, wall=wall, hops=hops
        )

    if not args.skip_per_vessel and args.per_vessel_map.strip():
        out = out_dir / "eval_frontier_per_vessel_map.json"
        results["per_vessel"] = _run_eval(
            out=out,
            growth=growth,
            wall=wall,
            hops=1.0,
            hops_map=args.per_vessel_map,
        )

    summary_path = out_dir / "hops_sweep_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[save] {summary_path}", flush=True)

    # Pick best by deploy_clot_score among spray-clean runs.
    best_key = None
    best_score = -1.0
    for key, item in results.get("sweeps", {}).items():
        gates = (item or {}).get("gates") or {}
        if not gates.get("spray_clean", False):
            continue
        score = float(gates.get("mean_score") or 0.0)
        if score > best_score:
            best_score = score
            best_key = key
    if best_key:
        print(f"[OK] best spray-clean hops={best_key} score={best_score:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
