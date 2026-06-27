"""One-off: W_mat_flow_stagnation deploy_clot_score vs F1 per anchor."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.eval_mat_growth_simple import _apply_ckpt_recipe, _eval_ckpt
from src.biochem_gnn.mat_growth_simple import apply_mat_growth_simple_recipe_env
from src.core_physics.species_pushforward_continuous import discover_biochem_anchors
from src.core_physics.t0_device import require_cuda_device


def main() -> int:
    ckpt = REPO / "outputs/biochem/biochem_gnn/mat_growth_ladder/W_mat_flow_stagnation/species/best.pth"
    device = require_cuda_device()
    apply_mat_growth_simple_recipe_env(force=True)
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    meta = dict(payload.get("meta") or {})
    _apply_ckpt_recipe(meta, label="mat_growth_simple")
    anchors = discover_biochem_anchors()
    r = _eval_ckpt(ckpt, anchors, device, label="mat_growth_simple")
    print("anchor          clot_f1  score   rprec   rrec    pos_frac  dil_iou")
    print("-" * 72)
    for anc in anchors:
        m = r["per_anchor"][anc]
        print(
            f"{anc:14}  {m.get('deploy_clot_f1', 0):.3f}   "
            f"{m.get('deploy_clot_score', 0):.3f}   "
            f"{m.get('deploy_clot_relaxed_prec', 0):.3f}   "
            f"{m.get('deploy_clot_relaxed_rec', 0):.3f}   "
            f"{m.get('deploy_clot_pred_pos_frac', 0):.4f}   "
            f"{m.get('deploy_clot_dil_iou', 0):.3f}"
        )
    print("-" * 72)
    print("mean", {k: round(v, 4) for k, v in r["mean"].items() if "clot" in k or "pos" in k})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
