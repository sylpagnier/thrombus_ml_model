"""Adversarial check: vessels excluded from every cohort because they have NO clot.

PHASE3_HANDOFF 1.5b's central claim is that the LEVEL -- how much clot a vessel develops
-- is unknowable at t=0.  The sharpest test of a zero-parameter physics model is a vessel
where the answer is "none": it has no threshold to transfer and nothing to calibrate, so
if it invents clot here, the level really is being smuggled in from the cohort.

``mat_growth_simple`` excludes patient017/022/023/026/027/030/033/034 as having no clot,
and patient002 on a data-quality call.  None were used to fit anything.
"""
from __future__ import annotations

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
from src.core_physics.species_pushforward_continuous import resolve_deploy_eval_time_index  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

DIR = Path("data/processed/graphs_biochem_anchors")
EXCLUDED = ("patient002", "patient017", "patient022", "patient023", "patient026",
            "patient027", "patient030", "patient033", "patient034")
RELAX, GROW = 2.0, 6


def main() -> int:
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    print("%12s %5s %7s %7s %8s %8s %9s"
          % ("vessel", "T", "nWall", "nGT", "predA", "predB", "MatMax/crit"))
    tot = {"gt": [], "pred": []}
    for a in EXCLUDED:
        p = DIR / f"{a}.pt"
        if not p.exists():
            print("%12s  (no pack)" % a)
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        wall = d.mask_wall.reshape(-1).bool().numpy()
        n = len(wall)
        ei = d.edge_index.numpy()
        A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
        A = ((A + A.T) > 0).astype(np.int8)
        t_eval = resolve_deploy_eval_time_index(int(d.y.shape[0]))
        gt = gt_clot_phi_at_time(d, t_eval, phys, device=torch.device("cpu")).reshape(-1).numpy()
        gt = (gt > 0.5) & wall
        nm = d.y_channel_names.split(",")
        mat = torch.expm1(d.y[-1, :, nm.index("Mat_log1p_nd")].clamp(-10, 8)).numpy() * float(bio.Minf)
        counts = {}
        for arm, st in (("gt", 3), ("pred", 4)):
            try:
                f = t0_flow_fields(d, bio, hops=st, flow_source=arm)
            except ValueError:
                counts[arm] = None
                continue
            cur = (f.gate > 0) & wall
            adm = (f.sr < float(bio.lss) * RELAX) & wall
            for _ in range(GROW):
                cur = cur | (((A @ cur.astype(np.int8)) > 0) & adm)
            counts[arm] = int(cur.sum())
            tot[arm].append((int(cur.sum()), int(wall.sum())))
        print("%12s %5d %7d %7d %8s %8s %9.3f"
              % (a, int(d.y.shape[0]), int(wall.sum()), int(gt.sum()),
                 counts["gt"], "--" if counts["pred"] is None else counts["pred"],
                 float(mat.max()) / float(bio.viscosity_mat_crit)))

    for arm in ("gt", "pred"):
        v = tot[arm]
        if not v:
            continue
        fp = sum(x for x, _ in v)
        nw = sum(y for _, y in v)
        print("\n  arm %-5s total predicted %d wall nodes out of %d (%.1f%% false-positive rate)"
              % (arm, fp, nw, 100.0 * fp / max(nw, 1)))
    print("\n  (a model that transfers a fixed operating point would fire at its cohort"
          "\n   base rate here, ~25-35%% of wall nodes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
