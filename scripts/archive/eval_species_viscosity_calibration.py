"""Eval T=53 mu with frozen GNN species + learned beta calibration (s35).

Usage::

    python scripts/eval_species_viscosity_calibration.py --anchor patient007
    python scripts/eval_species_viscosity_calibration.py --all-anchors
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.species_gnn_clot_rollout import (  # noqa: E402
    load_species_gnn_rollout_bundle,
    prepare_species_gnn_rollout_static,
    rollout_species_gnn_species_series,
)
from src.core_physics.species_pushforward_continuous import BIOCHEM_ANCHORS_6  # noqa: E402
from src.core_physics.species_viscosity_calibration import (  # noqa: E402
    DEFAULT_S34_GNN_CKPT,
    load_viscosity_calibration,
    predict_mu_at_time_with_beta,
)
from src.core_physics.t0_mu_physics import _mu_log_mae, _pearson, _region_masks  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402
from src.core_physics.t0_device import require_cuda_device  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Eval s35 viscosity calibration (GNN + beta)")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--anchors", default="")
    ap.add_argument("--all-anchors", action="store_true")
    ap.add_argument("--gnn-ckpt", default=DEFAULT_S34_GNN_CKPT)
    ap.add_argument("--calib", default="outputs/biochem/species_snapshot_s35/beta.pth")
    ap.add_argument("--time-index", type=int, default=0, help="0 = use calib bundle time")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = require_cuda_device()
    root = get_project_root()
    if args.all_anchors:
        anchors = list(BIOCHEM_ANCHORS_6)
    elif args.anchors.strip():
        anchors = [x.strip() for x in args.anchors.split(",") if x.strip()]
    else:
        anchors = [args.anchor.strip() or "patient007"]

    gnn_ckpt = Path(args.gnn_ckpt)
    if not gnn_ckpt.is_absolute():
        gnn_ckpt = root / gnn_ckpt
    calib_path = Path(args.calib)
    if not calib_path.is_absolute():
        calib_path = root / calib_path

    calibrator, bundle = load_viscosity_calibration(calib_path, device=device)
    t_eval = int(args.time_index) if int(args.time_index) > 0 else int(bundle.time_index)
    beta = calibrator.beta

    rollout_bundle = load_species_gnn_rollout_bundle(gnn_ckpt, device=device)
    if rollout_bundle is None:
        print(f"[ERR] missing GNN ckpt {gnn_ckpt}", file=sys.stderr)
        return 1

    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    rows: list[dict] = []

    for anc in anchors:
        graph = root / "data/processed/graphs_biochem_anchors" / f"{anc}.pt"
        data = torch.load(graph, map_location=device, weights_only=False)
        static = prepare_species_gnn_rollout_static(data, device=device)
        with torch.no_grad():
            series = rollout_species_gnn_species_series(
                data, rollout_bundle, static, device=device
            )
        t_i = min(t_eval, int(series.shape[0]) - 1)
        mu_pred_b, mu_gt = predict_mu_at_time_with_beta(
            data, series, beta, t_i,
            phys_cfg=phys, bio_cfg=bio, device=device, anchor=anc,
            soft_gelation=False,
        )
        mu_pred0, _ = predict_mu_at_time_with_beta(
            data, series, 1.0, t_i,
            phys_cfg=phys, bio_cfg=bio, device=device, anchor=anc,
            soft_gelation=False,
        )
        mu_comsol_b, mu_comsol_gt = mu_pred_b, mu_gt
        mu_comsol0, _ = mu_pred0, mu_gt
        masks = _region_masks(data, t_i, phys, device, mu_gt)
        growth = masks["growth"]
        row = {
            "anchor": anc,
            "time_index": t_i,
            "beta": float(beta.detach().cpu().item()),
            "mu_log_mae_all_beta": _mu_log_mae(mu_pred_b, mu_gt),
            "mu_log_mae_all_raw": _mu_log_mae(mu_pred0, mu_gt),
            "pearson_all_beta": _pearson(mu_pred_b, mu_gt),
            "pearson_all_raw": _pearson(mu_pred0, mu_gt),
            "mu_log_mae_growth_beta": _mu_log_mae(mu_pred_b, mu_gt, growth)
            if bool(growth.any().item())
            else float("nan"),
            "mu_log_mae_growth_raw": _mu_log_mae(mu_pred0, mu_gt, growth)
            if bool(growth.any().item())
            else float("nan"),
            "n_growth": int(growth.sum().item()),
            "comsol_mu_log_mae_beta": _mu_log_mae(mu_comsol_b, mu_comsol_gt),
            "comsol_mu_log_mae_raw": _mu_log_mae(mu_comsol0, mu_comsol_gt),
            "comsol_mu_log_mae_growth_beta": _mu_log_mae(mu_comsol_b, mu_comsol_gt, growth)
            if bool(growth.any().item())
            else float("nan"),
            "comsol_mu_log_mae_growth_raw": _mu_log_mae(mu_comsol0, mu_comsol_gt, growth)
            if bool(growth.any().item())
            else float("nan"),
        }
        rows.append(row)
        print(
            f"[i] {anc} beta={row['beta']:.3f} "
            f"soft logMAE {row['mu_log_mae_all_raw']:.3f}->{row['mu_log_mae_all_beta']:.3f} "
            f"comsol logMAE {row['comsol_mu_log_mae_raw']:.3f}->{row['comsol_mu_log_mae_beta']:.3f} "
            f"growth {row['comsol_mu_log_mae_growth_raw']:.3f}->{row['comsol_mu_log_mae_growth_beta']:.3f}",
            flush=True,
        )

    out = Path(args.out) if args.out.strip() else calib_path.parent / "mu_eval.json"
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
