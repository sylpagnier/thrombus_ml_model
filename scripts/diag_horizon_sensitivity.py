"""Is the 0.9093 final-mask score a property of the MODEL, or of where the clock stops?

The shipped physics model commits its whole set in one step at t~3000 s and never changes
again.  GT climbs from ~7000 s to ~12000 s and creeps on to 30000 s.  So the final-mask
score compares a mask that has been static for 90% of the run against GT's *asymptotic*
extent -- and the dataset's horizon (30000 s) is an arbitrary choice of the COMSOL runs,
not a physical endpoint.

If the score is a property of the model, it should be flat in the horizon.  If it is a
property of the horizon, it will rise as GT catches up to a mask that was already finished.
This measures exactly that: the SHIPPED mask, scored against GT at every fraction of the
run.  Nothing is fitted, nothing is selected -- it is a diagnostic of a model that already
exists, so it spends no protocol budget.

    python scripts/diag_horizon_sensitivity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.biochem_gnn.mat_growth_simple import (  # noqa: E402
    WALL_COHORT_V2_GENERALIZATION, WALL_COHORT_V2_TRAIN,
)
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.physics_wall_model import t0_flow_fields  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.core_physics.temporal_metrics import gt_onset_index  # noqa: E402
from src.evaluation.clot_relaxed_metrics import (  # noqa: E402
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
)

DIR = Path("data/processed/graphs_biochem_anchors")
OUT = Path("outputs/ap_closure")
RELAX, GROW, HOPS = 2.0, 6, 3
FRACS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00)


def main() -> int:
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    print("Shipped physics mask (frozen after t~3000 s) scored against GT at each horizon.")
    print("gt_frac = fraction of GT's FINAL committed set that exists yet at that time.\n")
    print("%-12s | %s" % ("vessel", " ".join("%11s" % ("t/T=%.1f" % f) for f in FRACS)))
    for n in sorted(set(WALL_COHORT_V2_TRAIN) | set(WALL_COHORT_V2_GENERALIZATION)):
        p = DIR / f"{n}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        T = int(d.y.shape[0])
        if T < 150:
            continue
        w = d.mask_wall.reshape(-1).bool().numpy()
        gt_on = gt_onset_index(d, phys, w)
        if not ((gt_on >= 0) & w).any():
            continue
        f = t0_flow_fields(d, bio, hops=HOPS, flow_source="gt")
        ei = d.edge_index.numpy()
        nn = len(w)
        A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(nn, nn)).tocsr()
        A = ((A + A.T) > 0).astype(np.int8)
        cur = (f.gate > 0) & w
        adm = (f.sr < float(bio.lss) * RELAX) & w
        for _ in range(GROW):
            cur = cur | (((A @ cur.astype(np.int8)) > 0) & adm)
        wt = torch.tensor(w.astype(np.float32))
        pred = torch.tensor(cur.astype(np.float32)) * wt
        n_final = float(((gt_on >= 0) & w).sum())
        sc, gf = [], []
        for fr in FRACS:
            ti = int(round(fr * (T - 1)))
            gt = gt_clot_phi_at_time(d, ti, phys, device=torch.device("cpu")).reshape(-1) * wt
            m = compute_clot_relaxed_metrics(pred, gt, d.edge_index, wall_mask=torch.tensor(w))
            sc.append(clot_score_from_deploy_dict(metrics_to_deploy_prefix(m)))
            gf.append(float(((gt_on >= 0) & (gt_on <= ti) & w).sum()) / max(n_final, 1.0))
        rows.append(dict(name=n, sealed=n in WALL_COHORT_V2_GENERALIZATION,
                         score=sc, gt_frac=gf))
        print("%-12s | %s" % (n, " ".join("%11.4f" % v for v in sc)))

    for lbl, sel in (("ALL", rows), ("train", [r for r in rows if not r["sealed"]]),
                     ("SEALED", [r for r in rows if r["sealed"]])):
        if not sel:
            continue
        s = np.array([r["score"] for r in sel])
        g = np.array([r["gt_frac"] for r in sel])
        print("\n%-7s n=%2d" % (lbl, len(sel)))
        print("   median deploy score : %s"
              % " ".join("%11.4f" % v for v in np.median(s, 0)))
        print("   median GT completed : %s"
              % " ".join("%11.4f" % v for v in np.median(g, 0)))
        best = int(np.argmax(np.median(s, 0)))
        print("   best horizon for this model: t/T = %.1f  (score %.4f);  at t/T=0.5 it is %.4f"
              % (FRACS[best], np.median(s, 0)[best], np.median(s, 0)[4]))

    (OUT / "horizon_sensitivity.json").write_text(
        json.dumps(dict(fracs=list(FRACS), rows=rows), indent=2, default=float),
        encoding="utf-8")
    print("\nwrote %s" % (OUT / "horizon_sensitivity.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
