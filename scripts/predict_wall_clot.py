"""Predict the wall clot map for a vessel. Entry point for the Phase-3 physics model.

Deploy-legal inputs only: node positions, mesh connectivity, ``u_ref``/``d_bar``, the
initial/boundary conditions, and a t=0 velocity field.

    --flow pred   uses the pack's ``u0_pred``/``v0_pred``  (DEPLOYABLE -- no GT anywhere)
    --flow gt     uses the GT velocity at t=0              (the PHASE3_HANDOFF 0a bandaid)

Zero learned parameters.  The three scalars below were fit on WALL_COHORT_V2_TRAIN;
see docs/PHASE3_RESULTS.md.

    python scripts/predict_wall_clot.py --anchor patient043 --flow pred
    python scripts/predict_wall_clot.py --pack path/to/vessel.pt --flow pred --save out.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.physics_wall_model import t0_flow_fields  # noqa: E402

# Fit on WALL_COHORT_V2_TRAIN, arm A. Stencil is per flow arm: a noisier field needs a
# wider one (docs/PHASE3_RESULTS.md 5).
RELAX = 2.0
GROW_HOPS = 6
STENCIL = {"gt": 3, "pred": 4}
# Lumen arm (docs/PHASE3_RESULTS.md 11), fit on full-horizon TRAIN.
LUMEN_HOPS = 2
LUMEN_SPEED = 0.3


def predict_wall_clot(data, bio_cfg, *, flow: str = "pred", lumen: bool = False):
    """Binary clot mask [N] plus the intermediate fields.

    ``lumen=False`` -> wall arm only.  ``lumen=True`` -> also propagate into stagnant
    lumen, replacing the learned lumen specialist of the compound stack.
    """
    f = t0_flow_fields(data, bio_cfg, hops=STENCIL[flow], flow_source=flow)
    wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
    n = len(wall)
    ei = data.edge_index.cpu().numpy()
    A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.int8)
    cur = (f.gate > 0) & wall                                   # ignition: the two gates
    adm = (f.sr < float(bio_cfg.lss) * RELAX) & wall            # clot-front admission
    for _ in range(GROW_HOPS):
        cur = cur | (((A @ cur.astype(np.int8)) > 0) & adm)
    if not lumen:
        return cur, f
    from src.core_physics.physics_lumen_model import grow_into_lumen, speed_nd, speed_nd_pred

    spd = speed_nd_pred(data) if flow == "pred" else speed_nd(data)
    off = grow_into_lumen(cur, wall, A, spd, f.sr,
                          lumen_hops=LUMEN_HOPS, speed_thresh=LUMEN_SPEED)
    return cur | off, f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default="")
    ap.add_argument("--pack", default="")
    ap.add_argument("--flow", default="pred", choices=["pred", "gt"])
    ap.add_argument("--save", default="")
    ap.add_argument("--lumen", action="store_true", help="add the off-wall lumen arm")
    ap.add_argument("--score", action="store_true", help="also score against GT (needs data.y)")
    args = ap.parse_args()
    path = Path(args.pack) if args.pack else Path(
        f"data/processed/graphs_biochem_anchors/{args.anchor}.pt")
    if not path.exists():
        print(f"[ERR] no pack at {path}")
        return 1
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    data = torch.load(path, map_location="cpu", weights_only=False)
    pred, f = predict_wall_clot(data, bio, flow=args.flow, lumen=args.lumen)
    wall = data.mask_wall.reshape(-1).bool().numpy()
    print("vessel %s  flow=%s  lumen=%s  nodes=%d wall=%d"
          % (path.stem, args.flow, args.lumen, len(wall), wall.sum()))
    print("  gates open at t=0 : low-shear %d, separation %d, union %d"
          % (int((f.gate_low[wall] > 0).sum()), int((f.gate_sep[wall] > 0).sum()),
             int(((f.gate > 0) & wall).sum())))
    print("  PREDICTED CLOT    : %d nodes (%d on wall, %d off-wall)"
          % (int(pred.sum()), int((pred & wall).sum()), int((pred & ~wall).sum())))

    if args.score:
        from src.core_physics.species_pushforward_continuous import resolve_deploy_eval_time_index
        from src.core_physics.t0_mu_physics import gt_clot_phi_at_time
        from src.evaluation.clot_relaxed_metrics import (
            clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
        )
        t_eval = resolve_deploy_eval_time_index(int(data.y.shape[0]))
        gt = gt_clot_phi_at_time(data, t_eval, phys, device=torch.device("cpu")).reshape(-1)
        gt = gt * torch.tensor(wall.astype(np.float32))
        m = compute_clot_relaxed_metrics(torch.tensor(pred.astype(np.float32)), gt,
                                         data.edge_index, wall_mask=torch.tensor(wall))
        o = metrics_to_deploy_prefix(m)
        print("  GT clot           : %d wall nodes" % int(gt.sum()))
        print("  deploy_clot_score : %.4f   (strict F1 %.4f, relaxed P %.3f R %.3f)"
              % (clot_score_from_deploy_dict(o), o["deploy_clot_f1"],
                 o["deploy_clot_relaxed_prec"], o["deploy_clot_relaxed_rec"]))
    if args.save:
        np.savez_compressed(args.save, clot=pred, sr=f.sr, dsrx=f.dsrx,
                            gate_low=f.gate_low, gate_sep=f.gate_sep, wall=wall)
        print("  wrote %s" % args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
