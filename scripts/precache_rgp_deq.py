"""Precache RGP-DEQ t=0 flow onto biochem graph packs.

    python scripts/precache_rgp_deq.py --only patient040,patient041,patient044,patient012
    python scripts/precache_rgp_deq.py --only patient040 --force
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import PredChannels  # noqa: E402
from src.core_physics.t0_device import require_cuda_device  # noqa: E402
from src.utils.kinematics_inference import (  # noqa: E402
    load_kinematics_predictor,
    predict_kinematics_and_latent,
    resolve_kinematics_checkpoint,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Attach RGP-DEQ u0_pred/v0_pred/z_kin_pred onto graph packs.")
    parser.add_argument("--graph-dir", default="data/processed/graphs_biochem_anchors")
    parser.add_argument("--only", default="",
                        help="comma-separated anchors. Default: every *.pt in graph-dir.")
    parser.add_argument("--cohort", default="", choices=["", "fitdev"],
                        help="fitdev = FIT+DEV wall-cohort anchors (still skips missing files)")
    parser.add_argument("--force", action="store_true",
                        help="recompute even when u0_pred is already attached")
    args = parser.parse_args()

    graph_dir = Path(args.graph_dir)
    if not graph_dir.exists():
        print("[ERR] no graph dir at %s" % graph_dir)
        return 1

    if args.only.strip():
        pt_files = [graph_dir / f"{a.strip()}.pt" for a in args.only.split(",") if a.strip()]
    elif args.cohort == "fitdev":
        from src.core_physics.wall_cohort_splits import DEV, FIT
        pt_files = [graph_dir / f"{a}.pt" for a in list(FIT) + list(DEV)]
    else:
        pt_files = sorted(graph_dir.glob("*.pt"))
    if not pt_files:
        print("[WARN] no packs to cache")
        return 1

    device = require_cuda_device()
    ckpt_path = resolve_kinematics_checkpoint()
    print("[i] kinematics ckpt %s" % ckpt_path)
    kine = load_kinematics_predictor(ckpt_path, device)
    kine.eval()

    n_ok = n_skip = 0
    for file_path in pt_files:
        if not file_path.exists():
            print("[ERR] missing %s" % file_path.name)
            continue
        anchor = file_path.stem
        data = torch.load(file_path, map_location="cpu", weights_only=False)
        if (not args.force) and getattr(data, "u0_pred", None) is not None:
            print("[skip] %s already has u0_pred" % anchor)
            n_skip += 1
            continue
        print("[%s] RGP-DEQ t=0 ..." % anchor)
        data_cuda = data.to(device)
        with torch.no_grad():
            pred, z_kin = predict_kinematics_and_latent(kine, data_cuda)
        u0 = pred[:, PredChannels.U].contiguous()
        v0 = pred[:, PredChannels.V].contiguous()
        data.u0_pred = u0.detach().cpu()
        data.v0_pred = v0.detach().cpu()
        data.z_kin_pred = z_kin.detach().cpu()
        # Direct shear head (nd), converted to 1/s.  Cached for a later head; t0_flow_fields
        # still MLS-differentiates u0_pred (PHASE7_FINDINGS 10.7).
        if pred.shape[1] > PredChannels.SHEAR_RATE:
            u_ref = float(data.u_ref.reshape(-1)[0])
            d_bar = float(data.d_bar.reshape(-1)[0])
            sr_nd = pred[:, PredChannels.SHEAR_RATE].reshape(-1).clamp(min=0)
            data.sr0_pred = (sr_nd * (u_ref / max(d_bar, 1e-12))).detach().cpu()
        rel = ""
        if getattr(data, "y", None) is not None:
            u = data.y[0, :, 0].detach().cpu().numpy()
            v = data.y[0, :, 1].detach().cpu().numpy()
            up = data.u0_pred.reshape(-1).numpy()
            vp = data.v0_pred.reshape(-1).numpy()
            den = float((u * u + v * v).mean() ** 0.5) + 1e-12
            rel_l2 = float((((up - u) ** 2 + (vp - v) ** 2).mean() ** 0.5) / den)
            rel = "  RelL2=%.3f" % rel_l2
        torch.save(data, file_path)
        print("[OK] %s  n=%d%s  (saved u,v,sr0_pred)" % (anchor, int(data.num_nodes), rel))
        n_ok += 1
    print("[i] wrote %d  skipped %d" % (n_ok, n_skip))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
