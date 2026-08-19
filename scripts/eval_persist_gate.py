"""t=0 gate AND later gates -- a precision oracle for over-ignition FP.

Union-over-time can only ADD seeds (recall).  Over-ignition vessels (018/019/025) already
have recall 1.0, so the wall-0.9 remainder is dropping t=0 gates that later flow closes.
Illegal as a deploy model (needs GT flow at later t); a ceiling on any corrector/wake that
is allowed to SHUT a gate.
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
from src.core_physics.physics_wall_model import t0_flow_fields  # noqa: E402
from src.core_physics.species_pushforward_continuous import resolve_deploy_eval_time_index  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.core_physics.wall_cohort_splits import DEV, FIT, MIN_T, format_split_means, split_of  # noqa: E402
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
    wall_ship, wall_and, wall_and3, full_ship, full_and = {}, {}, {}, {}, {}
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
        f0 = t0_flow_fields(d, bio, hops=STENCIL["gt"], flow_source="gt", time_index=0)
        t_mid = int(d.y.shape[0]) // 2
        t_last = int(d.y.shape[0]) - 1
        fm = t0_flow_fields(d, bio, hops=STENCIL["gt"], flow_source="gt", time_index=t_mid)
        fl = t0_flow_fields(d, bio, hops=STENCIL["gt"], flow_source="gt", time_index=t_last)
        spd = speed_nd(d)
        seed0 = (f0.gate > 0) & wall
        seed_and = seed0 & (fl.gate > 0)
        seed_and3 = seed0 & (fm.gate > 0) & (fl.gate > 0)
        def pred_of(seed):
            msk = grow(seed, wall, A, f0.sr, bio)
            return msk | grow_into_lumen(msk, wall, A, spd, f0.sr,
                                         lumen_hops=LUMEN_HOPS, speed_thresh=LUMEN_SPEED)
        p0, pa, p3 = pred_of(seed0), pred_of(seed_and), pred_of(seed_and3)
        ones = np.ones(n, dtype=bool)
        wall_ship[anchor] = dscore(p0, gt, d.edge_index, wall, wall)
        wall_and[anchor] = dscore(pa, gt, d.edge_index, wall, wall)
        wall_and3[anchor] = dscore(p3, gt, d.edge_index, wall, wall)
        full_ship[anchor] = dscore(p0, gt, d.edge_index, ones, wall)
        full_and[anchor] = dscore(pa, gt, d.edge_index, ones, wall)
        print("%-12s %-4s wall t0 %.3f  AND_end %.3f  AND_3 %.3f"
              % (anchor, split_of(anchor), wall_ship[anchor], wall_and[anchor], wall_and3[anchor]))
    print("shipped     wall %s" % format_split_means(wall_ship))
    print("AND t0&end  wall %s" % format_split_means(wall_and))
    print("AND t0,mid,end wall %s" % format_split_means(wall_and3))
    print("shipped     full %s" % format_split_means(full_ship))
    print("AND t0&end  full %s" % format_split_means(full_and))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
