"""Step 9: deploy v1 forward-path audit (no GT leakage)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.inference.clot_ml_deploy_v1 import audit_y_invariance, load_deploy_v1_recipe
from src.training.clot_ml_device import resolve_clot_ml_eval_device


def main() -> int:
    ap = argparse.ArgumentParser(description="Step 9 deploy forward audit")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--recipe", default="data/reference/clot_ml_deploy_v1.json")
    ap.add_argument("--out", default="outputs/biochem/clot_ml_ladder/deploy_v1/step9_audit.json")
    args = ap.parse_args()

    device = resolve_clot_ml_eval_device()
    recipe = load_deploy_v1_recipe(REPO / args.recipe)
    recipe.apply_env()

    graph = REPO / args.anchor_dir / f"{args.anchor}.pt"
    data = torch.load(graph, map_location=device, weights_only=False)

    checks: dict = {
        "anchor": args.anchor,
        "vel_source_enforced": recipe.vel_source == "kinematics" or not recipe.coupled,
        "phi_shell": recipe.phi_shell,
        "forbidden_gt_flow": recipe.vel_source != "gt",
    }
    inv = audit_y_invariance(data, recipe, device=device)
    checks.update(inv)
    checks["pass_step9"] = bool(
        checks["forbidden_gt_flow"]
        and inv["pass_y_invariance"]
        and checks["vel_source_enforced"]
    )

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(checks, indent=2), encoding="utf-8")

    print(f"[i] anchor={args.anchor} max_phi_delta={inv['max_phi_delta']:.2e}")
    print(f"[i] pass_y_invariance={inv['pass_y_invariance']} pass_step9={checks['pass_step9']}")
    print(f"[save] {out_path}")
    return 0 if checks["pass_step9"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
