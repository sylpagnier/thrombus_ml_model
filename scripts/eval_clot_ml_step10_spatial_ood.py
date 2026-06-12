"""Step 10: spatial OOD proxy -- per-anchor deploy + physics checklist."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.inference.clot_ml_deploy_v1 import eval_deploy_v1_on_graph, load_deploy_v1_recipe
from src.training.clot_ml_device import resolve_clot_ml_eval_device

import torch


def _physics_checklist(row: dict) -> dict:
    pred = float(row.get("tfinal_band_pred_frac", 0.0))
    f1 = float(row.get("tfinal_band_f1", 0.0))
    no_paint = pred < 0.85
    not_empty = pred > 0.001 or f1 > 0.01
    return {
        "no_ceiling_paint": no_paint,
        "not_zero_commit": not_empty,
        "pass_physics_checklist": no_paint and (not_empty or f1 < 0.05),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Step 10 spatial OOD gate (anchor proxy)")
    ap.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--recipe", default="data/reference/clot_ml_deploy_v1.json")
    ap.add_argument("--holdout", default="", help="comma-separated holdout stems (OOD)")
    ap.add_argument("--out", default="outputs/biochem/clot_ml_ladder/deploy_v1/step10_ood.json")
    args = ap.parse_args()

    device = resolve_clot_ml_eval_device()
    recipe = load_deploy_v1_recipe(REPO / args.recipe)
    adir = REPO / args.anchor_dir
    holdouts = {s.strip() for s in args.holdout.split(",") if s.strip()}

    rows: list[dict] = []
    for p in sorted(adir.glob("patient*.pt")):
        data = torch.load(p, map_location=device, weights_only=False)
        row = eval_deploy_v1_on_graph(data, recipe, device=device, anchor=p.stem)
        row["ood_holdout"] = p.stem in holdouts if holdouts else False
        row["physics"] = _physics_checklist(row)
        rows.append(row)

    in_window = [r for r in rows if not r.get("ood_holdout")]
    ood = [r for r in rows if r.get("ood_holdout")]
    mean_all = sum(r["deploy_score"] for r in rows) / max(len(rows), 1)
    mean_in = sum(r["deploy_score"] for r in in_window) / max(len(in_window), 1) if in_window else mean_all

    summary = {
        "step": "10",
        "mean_deploy_all": mean_all,
        "mean_deploy_in_window": mean_in,
        "per_anchor": rows,
        "ood_holdouts": sorted(holdouts),
        "pass_step10": all(r["physics"]["pass_physics_checklist"] for r in rows),
    }
    if ood:
        for r in ood:
            if abs(r["deploy_score"] - mean_in) > 0.05:
                summary["pass_step10"] = False

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[i] mean_deploy={mean_all:.3f} pass_step10={summary['pass_step10']}")
    for r in rows:
        print(
            f"  {r['anchor']:12} deploy={r['deploy_score']:.3f} "
            f"pred+={r['tfinal_band_pred_frac']:.3f} physics={r['physics']['pass_physics_checklist']}"
        )
    print(f"[save] {out_path}")
    return 0 if summary["pass_step10"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
