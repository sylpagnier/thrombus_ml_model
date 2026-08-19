"""Confirm hops=20 still wins under deployable (predicted) t=0 flow."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from predict_wall_clot import LUMEN_HOPS, LUMEN_SPEED, RELAX, STENCIL  # noqa: E402
from src.biochem_gnn.mat_growth_simple import WALL_COHORT_V2_TRAIN  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.physics_lumen_model import grow_into_lumen, speed_nd_pred  # noqa: E402
from src.core_physics.physics_wall_model import t0_flow_fields  # noqa: E402
from src.core_physics.species_pushforward_continuous import resolve_deploy_eval_time_index  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.evaluation.clot_relaxed_metrics import (  # noqa: E402
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
)

DIR = REPO / "data/processed/graphs_biochem_anchors"


def score(pred, gt_t, ei):
    m = compute_clot_relaxed_metrics(torch.tensor(pred.astype(np.float32)), gt_t, ei)
    return float(clot_score_from_deploy_dict(metrics_to_deploy_prefix(m)))


def main():
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    acc = {6: [], 20: []}
    n = 0
    for anchor in WALL_COHORT_V2_TRAIN:
        p = DIR / f"{anchor}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        if int(d.y.shape[0]) < 150 or getattr(d, "u0_pred", None) is None:
            continue
        wall = d.mask_wall.reshape(-1).bool().numpy()
        ei = d.edge_index.cpu().numpy()
        N = len(wall)
        A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(N, N)).tocsr()
        A = ((A + A.T) > 0).astype(np.int8)
        t_eval = resolve_deploy_eval_time_index(int(d.y.shape[0]))
        gt_f = gt_clot_phi_at_time(d, t_eval, phys, device=torch.device("cpu")).reshape(-1)
        if (gt_f.numpy() > 0.5).sum() == 0:
            continue
        f = t0_flow_fields(d, bio, hops=STENCIL["pred"], flow_source="pred")
        spd = speed_nd_pred(d)
        seed = (f.gate > 0) & wall
        adm = (f.sr < float(bio.lss) * RELAX) & wall
        n += 1
        for hops in (6, 20):
            cur = seed.copy()
            for _ in range(hops):
                cur = cur | (((A @ cur.astype(np.int8)) > 0) & adm)
            off = grow_into_lumen(cur, wall, A, spd, f.sr,
                                  lumen_hops=LUMEN_HOPS, speed_thresh=LUMEN_SPEED)
            acc[hops].append(score(cur | off, gt_f, d.edge_index))
    print("[i] %d vessels with u0_pred" % n)
    print("   hops=6  (shipped) %.4f" % np.mean(acc[6]))
    print("   hops=20           %.4f  %+.4f" % (np.mean(acc[20]), np.mean(acc[20]) - np.mean(acc[6])))


if __name__ == "__main__":
    main()
