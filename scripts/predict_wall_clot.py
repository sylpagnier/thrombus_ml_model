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
# GROW_HOPS 6 -> 20 on 2026-08-16.  The wall mask is the t=0 gate plus shear-admitted
# graph growth; every wall false negative is a CLOSED t=0 gate (19.3% of GT), and of those
# 181/339 sit inside the admission band (sr < 2*lss) but more than 6 hops from a seed.
# Filling the whole admission component overshoots (patient020 0.725 -> 0.596 at 66 hops).
# The hop cap is a crude front-speed: unimodal on TRAIN (6: 0.783, 20: 0.802, 80: 0.789)
# and leave-one-vessel-out picks 20 on every fold (+0.018).  Under deployable predicted
# t=0 flow the same change is +0.025 (0.708 -> 0.733, 15 vessels with u0_pred).
# docs/PHASE7_FINDINGS.md 10.
GROW_HOPS = 20
STENCIL = {"gt": 3, "pred": 4}
# Lumen arm (docs/PHASE3_RESULTS.md 11), fit on full-horizon TRAIN.
#
# LUMEN_SPEED 0.3 -> 0.2 on 2026-08-16.  Every score in Phase 6 was wall-masked, which hid
# the fact that **17% of GT clot is off-wall** and that at 0.3 this arm was *worse than not
# running it*.  Swept on the full mesh over 19 train vessels
# (docs/PHASE6_RESULTS.md 21.1); the response is unimodal in speed:
#
#     hops=2  speed 0.05  0.10  0.20  0.30  0.50
#     score        0.759 0.771 0.783 0.761 0.738      (wall-only alone = 0.7651)
#
# hops=3 is worse at every threshold (0.708 at 0.30), so 2 stays.  In-sample on train and
# unconfirmed on SEALED -- one scalar on a smooth 1-D curve, so the risk is small, but say so.
LUMEN_HOPS = 2
LUMEN_SPEED = 0.2
# Mat-magnitude lumen arm (docs/PHASE7_FINDINGS.md).  ``lumen="mat"`` selects it.
# The shell comes from ``resolve_offwall_shell``, which navigates the quadratic mesh's own
# layering rather than measuring a distance: everything within 2.1 median edge lengths also
# swept up the wall-normal mid-edge family, which carries no species data and is almost pure
# false positive (off-wall precision 0.036).  Off-wall F1 0.409 -> 0.561 under a GT-Mat
# oracle, and no mesh-unit constant survives in the default path.  On a linear mesh there is
# no mid-edge family to navigate, and it falls back to the calibrated
# ``SHELL_SPECIES_LO/HI`` band.  See src.core_physics.physics_lumen_model.first_corner_shell.
LUMEN_MAT_FILL_HOPS = 6     # give graph-grown wall nodes an inherited Mat magnitude


def node_pos(data) -> np.ndarray:
    p = getattr(data, "siren_pos", None)
    if p is None:
        p = data.x[:, :2]
    return p.detach().cpu().numpy().astype(np.float64)


def predict_wall_clot(data, bio_cfg, *, flow: str = "pred", lumen: bool = False,
                      mat_wall=None):
    """Binary clot mask [N] plus the intermediate fields.

    ``lumen=False`` -> wall arm only.
    ``lumen=True`` (or ``"speed"``) -> the shipped stagnant-lumen dilation.
    ``lumen="mat"`` -> the Mat-magnitude arm; requires ``mat_wall`` (per-node Mat in COMSOL
    model units, e.g. the last row of ``integrate_mat_trajectory``).
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
    if lumen == "mat":
        if mat_wall is None:
            raise ValueError("lumen='mat' requires mat_wall")
        from src.core_physics.physics_lumen_model import (
            fill_grown_wall_mat, grow_into_lumen_by_mat,
        )

        mw = fill_grown_wall_mat(mat_wall, cur, wall, A, hops=LUMEN_MAT_FILL_HOPS)
        off = grow_into_lumen_by_mat(mw, wall, node_pos(data), ei,
                                     float(bio_cfg.viscosity_mat_crit))
        return cur | off, f
    from src.core_physics.physics_lumen_model import grow_into_lumen, speed_nd, speed_nd_pred

    spd = speed_nd_pred(data) if flow == "pred" else speed_nd(data)
    off = grow_into_lumen(cur, wall, A, spd, f.sr,
                          lumen_hops=LUMEN_HOPS, speed_thresh=LUMEN_SPEED)
    return cur | off, f


def wall_mat_field(data, bio_cfg, f, *, da_scale=None, ap_closure=True,
                   washout: float = 0.0) -> np.ndarray:
    """The rollout's final wall ``Mat`` [N], in COMSOL model units.  Zero learned params.

    ``washout`` -- the removal term's dimensionless coefficient; ``0.0`` is the
    accumulate-only trajectory.  See ``integrate_mat_trajectory``'s ``washout`` docstring for
    why the accumulate-only form cannot order GT ``Mat`` even with oracle inputs.
    """
    from src.core_physics.ap_closure import SHIPPED, SHIPPED_DA_SCALE, make_rollout_hook
    from src.core_physics.physics_wall_model import integrate_mat_trajectory

    wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
    hook = make_rollout_hook(SHIPPED, bio_cfg, f.sr) if ap_closure else None
    traj, _ = integrate_mat_trajectory(
        data, bio_cfg, f.gate * wall,
        da_scale=SHIPPED_DA_SCALE if da_scale is None else float(da_scale),
        ap_closure=hook, washout=float(washout), washout_sr=f.sr)
    return traj[-1]


def predict_wall_onset(data, bio_cfg, *, flow: str = "pred", ap_closure=True,
                       washout: float = 0.0):
    """The clot mask AND the time each node commits.  Zero learned parameters.

    The mask is exactly ``predict_wall_clot``'s -- the two t=0 gates plus shear-admitted
    graph growth, which sits at the flow-oracle ceiling and is not touched here.  What this
    adds is *when*, from the surface ODE with the wall-AP closure
    (:mod:`src.core_physics.ap_closure`) applied to the rate.

    ``ap_closure=False`` reproduces the older frozen-``ap`` rollout, in which every node
    whose gate is exactly 1 integrates an identical ODE and they all commit in the same step
    -- the flash.  It is kept only so the two can be compared; do not deploy it.

    Returns ``(mask [N] bool, onset [N] int, t [T])``.  ``onset == -1`` means "in the mask
    but never crossed in the ODE"; those nodes are given the ODE's own median onset by the
    scoring convention, which is a known weakness on the 15-26% of the mask that arrives by
    graph growth rather than by ignition.
    """
    from src.core_physics.ap_closure import SHIPPED, SHIPPED_DA_SCALE, make_rollout_hook
    from src.core_physics.physics_wall_model import (
        first_crossing, integrate_mat_trajectory,
    )

    mask, f = predict_wall_clot(data, bio_cfg, flow=flow, lumen=False)
    wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
    hook = make_rollout_hook(SHIPPED, bio_cfg, f.sr) if ap_closure else None
    traj, t = integrate_mat_trajectory(data, bio_cfg, f.gate * wall,
                                       da_scale=SHIPPED_DA_SCALE, ap_closure=hook,
                                       washout=float(washout), washout_sr=f.sr)
    idx = first_crossing(traj, float(bio_cfg.viscosity_mat_crit))
    return mask, np.where(mask, idx, -1), t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default="")
    ap.add_argument("--pack", default="")
    ap.add_argument("--flow", default="pred", choices=["pred", "gt"])
    ap.add_argument("--save", default="")
    ap.add_argument("--lumen", action="store_true", help="add the off-wall lumen arm")
    ap.add_argument("--score", action="store_true", help="also score against GT (needs data.y)")
    ap.add_argument("--temporal", action="store_true",
                    help="also emit the growth CURVE (onset time per node, AP closure on)")
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

    onset = t_grid = None
    if args.temporal:
        m2, onset, t_grid = predict_wall_onset(data, bio, flow=args.flow)
        assert np.array_equal(m2, pred if not args.lumen else m2), \
            "the AP closure must change WHEN, never WHICH (kill criterion 9)"
        hot = onset >= 0
        print("  GROWTH CURVE      : %d of %d masked nodes ignite in the ODE"
              % (int(hot.sum()), int(pred.sum())))
        tt = t_grid.reshape(-1).cpu().numpy() if hasattr(t_grid, "cpu") else np.asarray(t_grid)
        for q in (0.1, 0.25, 0.5, 0.75, 0.9):
            print("     %3.0f%% committed by t = %8.0f s" % (100 * q, np.quantile(tt[onset[hot]], q))
                  if hot.any() else "     (nothing ignites)")

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
        extra = {} if onset is None else {"onset": onset, "t": np.asarray(t_grid).reshape(-1)}
        np.savez_compressed(args.save, clot=pred, sr=f.sr, dsrx=f.dsrx,
                            gate_low=f.gate_low, gate_sep=f.gate_sep, wall=wall, **extra)
        print("  wrote %s" % args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
