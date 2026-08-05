"""Where are the deploy FPs? One rollout -> FP geography + growth-timing structure.

The scoring failure on patient020 is 172 FP against 110 GT (mass 2.42x). Beta cannot
touch it (both TP and FP are fully saturated in Mat, AUC 0.500) and the closed-loop
corrector is directionally inverted, so neither readout gain nor flow coupling explains
it. That leaves the spatial/temporal structure of the overpaint itself.

This reports, at the deploy horizon:
  * FP hop-distance to the nearest GT clot node -- adjacent halo vs distant wrong-pocket
  * the same for FN, so we can see whether recall loss is at the frontier or elsewhere
  * WHEN each predicted node first commits vs when the GT node first clots, which
    separates "grows in the right place but too fast/too long" from "grows in the
    wrong place".

Reuses the exact rollout + grading path the scored metric uses.

    python scripts/diag_fp_geography.py --ckpt <ckpt> --anchors patient020
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.eval_mat_growth_simple import _apply_ckpt_recipe, _load_static  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.species_deploy_rollout import reset_species_rollout_flow_cache  # noqa: E402
from src.core_physics.species_pushforward_continuous import (  # noqa: E402
    clear_offwall_model_cache,
    deploy_clot_phi_fields,
    deploy_species_rollout_series,
    load_continuous_bundle,
)
from src.core_physics.t0_device import require_cuda_device  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.evaluation.fp_geography import classify_fp_geography  # noqa: E402
from src.utils.kinematics_inference import (  # noqa: E402
    load_kinematics_predictor,
    resolve_kinematics_checkpoint,
)
from src.utils.paths import get_project_root  # noqa: E402

ANCHOR_DIR = get_project_root() / "data/processed/graphs_biochem_anchors"


def main() -> int:
    ap = argparse.ArgumentParser(description="FP geography + growth timing for deploy clot")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--out", default="outputs/biochem/eda/fp_geo/diag.json")
    args = ap.parse_args()

    root = get_project_root()
    device = require_cuda_device()
    anchors = [a.strip() for a in args.anchors.split(",") if a.strip()]
    ckpt = Path(args.ckpt)
    if not ckpt.is_absolute():
        ckpt = root / ckpt

    clear_offwall_model_cache()
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    meta = dict(payload.get("meta") or {})
    _apply_ckpt_recipe(meta, label="mat_growth_simple", ckpt_path=ckpt)
    bundle = load_continuous_bundle(ckpt, device=device, quiet=True)
    model = bundle.model
    wall_hops = int(meta.get("wall_hops", 3))
    kine = load_kinematics_predictor(
        str(resolve_kinematics_checkpoint()), device, phys_cfg=PhysicsConfig(phase="kinematics")
    )
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")

    report: dict = {"ckpt": str(ckpt), "per_anchor": {}}
    for anc in anchors:
        print(f"\n=== {anc} ===", flush=True)
        reset_species_rollout_flow_cache()
        data = torch.load(ANCHOR_DIR / f"{anc}.pt", map_location=device, weights_only=False)
        static = _load_static(data, device, kine, wall_hops, anc)
        static["n_times"] = int(data.y.shape[0])
        print("  rolling out...", flush=True)
        series, data = deploy_species_rollout_series(
            model, data, static, phys, bio, device, flow_source="kinematics"
        )
        phi_pred, phi_gt, wall_mask, t_eval = deploy_clot_phi_fields(
            data, series, static, phys, bio, device, flow_source="kinematics"
        )
        ei = data.edge_index.cpu()
        geo = classify_fp_geography(
            phi_pred.cpu().numpy(), phi_gt.cpu().numpy(), ei, n_nodes=int(data.num_nodes)
        )
        print(f"  t_eval={t_eval}  TP={geo['n_tp']} FP={geo['n_fp']} FN={geo['n_fn']}")
        print(f"  FP hop-to-GT: median={geo['fp_hop_to_gt_median']} mean={geo['fp_hop_to_gt_mean']}")
        print(f"  adjacent(<=2hop)={geo['n_adjacent_fp']} ({geo['adjacent_frac']:.1%})  "
              f"distant={geo['n_distant_fp']} ({geo['distant_frac']:.1%})")
        print(f"  mode={geo['mode']}  recommend={geo['recommend_leg']}")

        # ---- growth timing: when does each node first cross, pred vs GT? ----
        pr = (phi_pred > 0.5).cpu().numpy()
        gtm = (phi_gt > 0.5).cpu().numpy()
        n_times = int(data.y.shape[0])
        gt_first = np.full(int(data.num_nodes), -1, dtype=np.int64)
        probe = list(range(0, n_times, max(1, n_times // 25))) + [n_times - 1]
        for ti in sorted(set(probe)):
            g = (gt_clot_phi_at_time(data, ti, phys, device=device).reshape(-1) > 0.5).cpu().numpy()
            newly = g & (gt_first < 0)
            gt_first[newly] = ti
        tp_m, fp_m = pr & gtm, pr & ~gtm
        gt_on_tp = gt_first[tp_m]
        gt_on_tp = gt_on_tp[gt_on_tp >= 0]
        print(f"\n  GT first-clot time on TP nodes: median t={np.median(gt_on_tp) if gt_on_tp.size else 'na'}")
        print(f"  GT nodes never clotting that we predicted (FP): {int(fp_m.sum())}")

        report["per_anchor"][anc] = {
            "t_eval": int(t_eval), "geography": geo,
            "gt_first_clot_median_on_tp": float(np.median(gt_on_tp)) if gt_on_tp.size else None,
        }
        clear_offwall_model_cache()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\n[save] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
