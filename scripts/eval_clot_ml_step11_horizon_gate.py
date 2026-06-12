"""Step 11: temporal extrapolation gate across sim_end scales."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.inference.clot_ml_deploy_v1 import eval_deploy_v1_on_graph, load_deploy_v1_recipe
from src.training.clot_ml_device import resolve_clot_ml_eval_device


def main() -> int:
    ap = argparse.ArgumentParser(description="Step 11 horizon gate")
    ap.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--recipe", default="data/reference/clot_ml_deploy_v1.json")
    ap.add_argument("--scales", default="1.0,1.25,1.5,2.0")
    ap.add_argument("--anchor", default="", help="single anchor or all")
    ap.add_argument("--out", default="outputs/biochem/clot_ml_ladder/deploy_v1/step11_horizon.json")
    args = ap.parse_args()

    device = resolve_clot_ml_eval_device()
    recipe = load_deploy_v1_recipe(REPO / args.recipe)
    scales = [float(s.strip()) for s in args.scales.split(",") if s.strip()]
    adir = REPO / args.anchor_dir
    paths = sorted(adir.glob("patient*.pt"))
    if args.anchor.strip():
        paths = [adir / f"{args.anchor.strip()}.pt"]

    baseline_h1: float | None = None
    per_anchor: list[dict] = []

    for p in paths:
        data = torch.load(p, map_location=device, weights_only=False)
        by_scale: list[dict] = []
        for sc in scales:
            row = eval_deploy_v1_on_graph(
                data, recipe, device=device, sim_end_scale=sc, anchor=p.stem
            )
            by_scale.append(row)
        if baseline_h1 is None and by_scale:
            baseline_h1 = by_scale[0]["deploy_score"]
        per_anchor.append({"anchor": p.stem, "by_scale": by_scale})

    h1_mean = sum(a["by_scale"][0]["deploy_score"] for a in per_anchor) / max(len(per_anchor), 1)
    h15_rows = [s for a in per_anchor for s in a["by_scale"] if abs(s["sim_end_scale"] - 1.5) < 0.01]
    h20_rows = [s for a in per_anchor for s in a["by_scale"] if abs(s["sim_end_scale"] - 2.0) < 0.01]

    # In-window deploy scored at COMSOL t_final: must not shift when horizon extends.
    pass_h1_in_window_stable = True
    max_in_window_shift = 0.0
    for a in per_anchor:
        h1_deploy = float(a["by_scale"][0]["deploy_score"])
        for s in a["by_scale"][1:]:
            shift = abs(float(s["deploy_score"]) - h1_deploy)
            max_in_window_shift = max(max_in_window_shift, shift)
            if shift > 0.02:
                pass_h1_in_window_stable = False

    pass_h15 = all(
        (s.get("extrap") or {}).get("pass_h15_pred_frac", True) for s in h15_rows
    ) if h15_rows else True
    pass_h20 = all(
        (s.get("extrap") or {}).get("pass_no_ceiling_paint", True) for s in h20_rows
    ) if h20_rows else True
    pass_phi_monotone = all(
        (s.get("extrap") or {}).get("phi_monotone", 1.0) >= 1.0 - 1e-6 for s in h15_rows
    ) if h15_rows else True

    summary = {
        "step": "11",
        "scales": scales,
        "mean_deploy_h1": h1_mean,
        "max_in_window_deploy_shift": max_in_window_shift,
        "per_anchor": per_anchor,
        "pass_h1_in_window_stable": pass_h1_in_window_stable,
        "pass_h15_extrap": pass_h15,
        "pass_h20_sanity": pass_h20,
        "pass_phi_monotone_h15": pass_phi_monotone,
        "pass_step11": pass_h1_in_window_stable and pass_h15 and pass_h20 and pass_phi_monotone,
    }

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[i] scales={scales} mean_h1={h1_mean:.3f} pass_step11={summary['pass_step11']}")
    for a in per_anchor:
        h15 = next((s for s in a["by_scale"] if abs(s["sim_end_scale"] - 1.5) < 0.01), None)
        ext = (h15 or {}).get("extrap") or {}
        print(
            f"  {a['anchor']:12} h1={a['by_scale'][0]['deploy_score']:.3f} "
            f"h15_delta={ext.get('pred_frac_delta', float('nan')):.3f}"
        )
    print(f"[save] {out_path}")
    return 0 if summary["pass_step11"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
