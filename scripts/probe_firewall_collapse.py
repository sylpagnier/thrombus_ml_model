"""Probe why lumen-shape specialist collapses under compound deploy.

Compares on one anchor (default patient007):
  A) wall/canonical alone
  G) growth specialist alone  (train-val path)
  S) compound wall-route
  F) compound frontier-route

Reuses eval_mat_growth_simple loading so env/ckpt recipes match production eval.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument(
        "--wall-ckpt",
        default="outputs/biochem/biochem_gnn/locked/species_gnn_best.pth",
    )
    ap.add_argument(
        "--growth-ckpt",
        default=(
            "outputs/biochem/offwall_model/wc_v7_firewall_fix_seq/"
            "growth_hop_ge2_lumen_shape/best.pth"
        ),
    )
    ap.add_argument(
        "--out",
        default=(
            "outputs/biochem/offwall_model/wc_v7_firewall_fix_seq/"
            "probe_collapse_patient007.json"
        ),
    )
    args = ap.parse_args()

    root = _repo_root()
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.biochem_gnn.mat_growth_simple import apply_mat_growth_leg_env
    from src.core_physics.species_pushforward_continuous import clear_offwall_model_cache
    from src.core_physics.t0_device import require_cuda_device
    from scripts.eval_mat_growth_simple import _eval_ckpt

    device = require_cuda_device()
    wall = (root / args.wall_ckpt).resolve()
    growth = (root / args.growth_ckpt).resolve()
    if not wall.is_file():
        raise FileNotFoundError(wall)
    if not growth.is_file():
        raise FileNotFoundError(growth)

    anchors = [args.anchor]
    modes = []

    def run(label: str, ckpt: Path, *, offwall: Path | None, route: str | None) -> dict:
        clear_offwall_model_cache()
        # Match launcher: force WC_v7 leg, then two-model flags.
        apply_mat_growth_leg_env("WC_v7_clot_phi_mse", force=True)
        if offwall is not None:
            os.environ["SPECIES_TWO_MODEL_MODE"] = "1"
            os.environ["SPECIES_OFFWALL_MODEL_CKPT"] = str(offwall).replace("\\", "/")
            os.environ["SPECIES_TWO_MODEL_ROUTE"] = str(route or "wall")
            os.environ["SPECIES_TWO_MODEL_FRONTIER_HOPS"] = "2"
            print(f"[i] {label}: two-model route={route} growth={offwall.name}", flush=True)
        else:
            os.environ["SPECIES_TWO_MODEL_MODE"] = "0"
            os.environ.pop("SPECIES_OFFWALL_MODEL_CKPT", None)
            print(f"[i] {label}: single model ckpt={ckpt.name}", flush=True)

        res = _eval_ckpt(ckpt, anchors, device, label="mat_growth_simple")
        row = res["per_anchor"][args.anchor]
        mean = res["mean"]
        keep = {
            k: row.get(k, mean.get(k))
            for k in (
                "deploy_clot_f1",
                "deploy_clot_score",
                "deploy_mat_f1",
                "deploy_clot_offwall_n_pred",
                "deploy_clot_offwall_n_gt",
                "deploy_clot_offwall_strict_f1",
                "deploy_clot_offwall_relaxed_f1",
                "deploy_clot_offwall_n_pred_hop1",
                "deploy_clot_offwall_n_pred_hop2",
                "deploy_clot_offwall_n_pred_hop3",
                "deploy_clot_offwall_n_pred_hop_ge2",
                "deploy_clot_offwall_n_gt_hop_ge2",
                "deploy_clot_offwall_strict_f1_hop_ge2",
            )
        }
        print(
            f"  clot_f1={float(keep.get('deploy_clot_f1') or 0):.3f} "
            f"off={float(keep.get('deploy_clot_offwall_n_pred') or 0):.0f}/"
            f"{float(keep.get('deploy_clot_offwall_n_gt') or 0):.0f} "
            f"h1={float(keep.get('deploy_clot_offwall_n_pred_hop1') or 0):.0f} "
            f"h2={float(keep.get('deploy_clot_offwall_n_pred_hop2') or 0):.0f} "
            f"h3={float(keep.get('deploy_clot_offwall_n_pred_hop3') or 0):.0f} "
            f"ge2={float(keep.get('deploy_clot_offwall_n_pred_hop_ge2') or 0):.0f} "
            f"strict_off={float(keep.get('deploy_clot_offwall_strict_f1') or 0):.3f}",
            flush=True,
        )
        clear_offwall_model_cache()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return {
            "label": label,
            "ckpt": str(ckpt),
            "offwall_ckpt": str(offwall) if offwall else None,
            "route": route,
            "metrics": keep,
        }

    modes.append(run("A_wall_alone", wall, offwall=None, route=None))
    modes.append(run("G_growth_alone", growth, offwall=None, route=None))
    modes.append(run("S_compound_wall", wall, offwall=growth, route="wall"))
    modes.append(run("F_compound_frontier", wall, offwall=growth, route="frontier"))

    # Collapse ratios vs growth-alone
    g = modes[1]["metrics"]
    s = modes[2]["metrics"]
    analysis = {
        "growth_alone_offwall_n_pred": g.get("deploy_clot_offwall_n_pred"),
        "compound_wall_offwall_n_pred": s.get("deploy_clot_offwall_n_pred"),
        "growth_alone_hop_ge2": g.get("deploy_clot_offwall_n_pred_hop_ge2"),
        "compound_wall_hop_ge2": s.get("deploy_clot_offwall_n_pred_hop_ge2"),
        "growth_alone_clot_f1": g.get("deploy_clot_f1"),
        "compound_wall_clot_f1": s.get("deploy_clot_f1"),
        "interpretation": [],
    }
    g_off = float(g.get("deploy_clot_offwall_n_pred") or 0)
    s_off = float(s.get("deploy_clot_offwall_n_pred") or 0)
    if g_off >= 10 and s_off < 0.2 * g_off:
        analysis["interpretation"].append(
            "TRAJECTORY/BLEND COLLAPSE: growth-alone produces substantial off-wall clot, "
            "but wall-route compound suppresses it on the same anchor. Specialists trained "
            "solo cannot rely on wall-owned state; closed-loop wall trajectory differs."
        )
    if float(g.get("deploy_clot_f1") or 0) < 0.6:
        analysis["interpretation"].append(
            "GROWTH-ALONE WALL DAMAGE: solo specialist clot_f1 is weak; train-val reward "
            "offwall volume while wall footprint collapses — ckpt metric not compound-aware."
        )

    report = {
        "anchor": args.anchor,
        "modes": modes,
        "analysis": analysis,
    }
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out}", flush=True)
    for line in analysis["interpretation"]:
        print(f"[i] {line}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
