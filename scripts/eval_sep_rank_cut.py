"""Within-vessel sep-gate strength cut: drop the weakest separation seeds.

Global ``sgt`` tightening helps FIT over-ignition and hurts DEV.  A rank cut is
vessel-local: keep stagnation as-is, keep only the strongest fraction of the
separation branch.  Still a t=0 algebraic rule, no GT flow.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from predict_wall_clot import GROW_HOPS, LUMEN_HOPS, LUMEN_SPEED, RELAX, STENCIL  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.physics_lumen_model import grow_into_lumen, speed_nd  # noqa: E402
from src.core_physics.physics_wall_model import M_TO_CM, t0_flow_fields  # noqa: E402
from src.core_physics.species_pushforward_continuous import resolve_deploy_eval_time_index  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.core_physics.wall_cohort_splits import (  # noqa: E402
    DEV, FIT, MIN_T, format_split_means, mean_by_split, split_of,
)
from src.evaluation.clot_relaxed_metrics import (  # noqa: E402
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
)

DIR = REPO / "data/processed/graphs_biochem_anchors"


def dscore(pred, gt, ei, domain, wall):
    if int((gt & domain).sum()) == 0:
        return float("nan")
    dom = torch.tensor(domain.astype(np.float32))
    m = compute_clot_relaxed_metrics(
        torch.tensor(pred.astype(np.float32)) * dom,
        torch.tensor(gt.astype(np.float32)) * dom,
        ei, wall_mask=torch.tensor(wall))
    return float(clot_score_from_deploy_dict(metrics_to_deploy_prefix(m)))


def grow(seed, wall, A, sr, bio):
    cur = seed.copy()
    adm = (sr < float(bio.lss) * RELAX) & wall
    for _ in range(GROW_HOPS):
        cur = cur | (((A @ cur.astype(np.int8)) > 0) & adm)
    return cur


def main() -> int:
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    sgt_cgs = float(bio.sgt) / M_TO_CM
    packs = []
    for anchor in list(FIT) + list(DEV):
        pth = DIR / f"{anchor}.pt"
        if not pth.exists():
            continue
        d = torch.load(pth, map_location="cpu", weights_only=False)
        if int(d.y.shape[0]) < MIN_T:
            continue
        wall = d.mask_wall.reshape(-1).bool().numpy()
        t_eval = resolve_deploy_eval_time_index(int(d.y.shape[0]))
        gt = gt_clot_phi_at_time(d, t_eval, phys, device=torch.device("cpu")).reshape(-1).numpy() > 0.5
        if (gt & wall).sum() == 0:
            continue
        ei = d.edge_index.detach().cpu().numpy()
        n = len(wall)
        A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
        A = ((A + A.T) > 0).astype(np.int8)
        f = t0_flow_fields(d, bio, hops=STENCIL["gt"], flow_source="gt")
        packs.append(dict(anchor=anchor, d=d, wall=wall, A=A, gt=gt, f=f,
                          spd=speed_nd(d), ei=d.edge_index))

    def pred_from(seed, p):
        msk = grow(seed, p["wall"], p["A"], p["f"].sr, bio)
        return msk | grow_into_lumen(msk, p["wall"], p["A"], p["spd"], p["f"].sr,
                                     lumen_hops=LUMEN_HOPS, speed_thresh=LUMEN_SPEED)

    shipped_w, shipped_f = {}, {}
    for p in packs:
        seed = (p["f"].gate > 0) & p["wall"]
        pred = pred_from(seed, p)
        shipped_w[p["anchor"]] = dscore(pred, p["gt"], p["ei"], p["wall"], p["wall"])
        shipped_f[p["anchor"]] = dscore(pred, p["gt"], p["ei"],
                                        np.ones(len(p["wall"]), dtype=bool), p["wall"])
    print("shipped wall %s" % format_split_means(shipped_w))
    print("        full %s" % format_split_means(shipped_f))

    print("\n=== drop weakest sep-gate quantile (stagnation untouched) ===")
    for q in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        w, fsc = {}, {}
        for p in packs:
            wall, sr, dsrx = p["wall"], p["f"].sr, p["f"].dsrx
            stag = (sr < float(bio.lss)) & wall
            sep = (dsrx < sgt_cgs) & wall
            if sep.any() and q > 0:
                thr = float(np.quantile(np.abs(dsrx[sep]), q))
                sep = sep & (np.abs(dsrx) >= thr)
            pred = pred_from(stag | sep, p)
            w[p["anchor"]] = dscore(pred, p["gt"], p["ei"], p["wall"], p["wall"])
            fsc[p["anchor"]] = dscore(pred, p["gt"], p["ei"],
                                      np.ones(len(p["wall"]), dtype=bool), p["wall"])
        dw = {a: w[a] - shipped_w[a] for a in w}
        df = {a: fsc[a] - shipped_f[a] for a in fsc}
        print("   drop p<%.1f  wall %s" % (q, format_split_means(w)))
        print("              dW   %s  dFull %s" % (format_split_means(dw), format_split_means(df)))
        mw, mf = mean_by_split(dw), mean_by_split(df)
        if ((mw["fit"]["mean"] or 0) > 1e-6 and (mw["dev"]["mean"] or 0) > 1e-6
                and (mf["fit"]["mean"] or 0) > -1e-4 and (mf["dev"]["mean"] or 0) > -1e-4):
            print("      [same-sign FIT+DEV wall, full not down]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
