"""Head-to-head: which WALL model should the lumen arm be seeded from?

Three wall variants, all deploy-legal, same lumen arm on top (grow_into_lumen,
lumen_hops=2, speed_thresh=0.3):

  A. gate + spatial front-growth (predict_wall_clot.py's shipped model): gate-open wall
     nodes dilated 6 hops along the wall graph, admitting shear < 2*lss.
  B. autocatalytic ODE (physics_wall_model.integrate_mat, mode="ode"): integrates the
     COMSOL surface ODE with the Mas->Mat feedback term.  docs/PHASE3_RESULTS.md 3 found
     this SATURATES for da_scale>=50 and becomes bit-identical to the bare gate (no
     spatial growth at all) -- included here to confirm that in situ and show what
     compounding with it costs.
  C. bare gate, no growth of any kind (the floor both A and B sit above).

Reference: learned compound (wall net + lumen net) on orig10 = 0.8118 all / 0.8428 full-hz.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.physics_lumen_model import adjacency, grow_into_lumen, speed_nd  # noqa: E402
from src.core_physics.physics_wall_model import integrate_mat, node_positions, t0_flow_fields  # noqa: E402
from src.core_physics.species_pushforward_continuous import resolve_deploy_eval_time_index  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.evaluation.clot_relaxed_metrics import (  # noqa: E402
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
)

DIR = Path("data/processed/graphs_biochem_anchors")
ORIG10 = ["patient001", "patient002", "patient003", "patient004", "patient005",
          "patient006", "patient007", "patient008", "patient010", "patient011"]
REF = {"patient001": 0.9535, "patient002": 0.9406, "patient003": 0.8447, "patient004": 0.7120,
       "patient005": 0.7547, "patient006": 0.7313, "patient007": 0.9308, "patient008": 0.4406,
       "patient010": 0.8438, "patient011": 0.9662}
RELAX, GROW = 2.0, 6
LUMEN_HOPS, LUMEN_SPEED = 2, 0.3


def score(d, pred, gt, wall):
    m = compute_clot_relaxed_metrics(torch.tensor(pred.astype(np.float32)),
                                     torch.tensor(gt.astype(np.float32)),
                                     d.edge_index, wall_mask=torch.tensor(wall))
    o = metrics_to_deploy_prefix(m)
    return (clot_score_from_deploy_dict(o), o["deploy_clot_f1"],
            o.get("deploy_clot_offwall_relaxed_f1", 0.0),
            o.get("deploy_clot_offwall_n_pred", 0.0))


def main() -> int:
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    rows = []
    print("%12s %5s | %8s | %8s %8s | %8s %8s | %8s %8s"
          % ("vessel", "T", "learned", "A wall", "A+lumen", "B wall(ode)", "B+lumen",
             "C bare", "C+lumen"))
    for a in ORIG10:
        p = DIR / f"{a}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        wall = d.mask_wall.reshape(-1).bool().numpy()
        f = t0_flow_fields(d, bio, hops=3)
        A = adjacency(d.edge_index.numpy(), len(wall))
        t_eval = resolve_deploy_eval_time_index(int(d.y.shape[0]))
        gt = (gt_clot_phi_at_time(d, t_eval, phys, device=torch.device("cpu"))
              .reshape(-1).numpy() > 0.5)
        spd = speed_nd(d)

        # A: gate + spatial front-growth (the shipped wall model)
        wall_a = (f.gate > 0) & wall
        adm = (f.sr < float(bio.lss) * RELAX) & wall
        for _ in range(GROW):
            wall_a = wall_a | (((A @ wall_a.astype(np.int8)) > 0) & adm)

        # B: autocatalytic ODE, da_scale well into the saturated regime
        mat = integrate_mat(d, bio, f, da_scale=100.0)
        wall_b = (mat >= float(bio.viscosity_mat_crit)) & wall

        # C: bare gate, no growth mechanism at all
        wall_c = (f.gate > 0) & wall

        results = {}
        for tag, seed in (("A", wall_a), ("B", wall_b), ("C", wall_c)):
            off = grow_into_lumen(seed, wall, A, spd, f.sr,
                                  lumen_hops=LUMEN_HOPS, speed_thresh=LUMEN_SPEED)
            s_wall = score(d, seed, gt, wall)
            s_comp = score(d, seed | off, gt, wall)
            results[tag] = (s_wall, s_comp)

        rows.append((a, int(d.y.shape[0]), REF.get(a, np.nan), results))
        print("%12s %5d | %8.4f | %8.4f %8.4f | %8.4f %8.4f | %8.4f %8.4f"
              % (a, int(d.y.shape[0]), REF.get(a, np.nan),
                 results["A"][0][0], results["A"][1][0],
                 results["B"][0][0], results["B"][1][0],
                 results["C"][0][0], results["C"][1][0]))

    print("\n%-40s %s" % ("mean over orig10 (n=%d)" % len(rows), ""))
    learned = np.nanmean([r[2] for r in rows])
    print("  learned compound (wall net + lumen net)                : %.4f" % learned)
    for tag, label in (("A", "gate + spatial front-growth (shipped)"),
                       ("B", "autocatalytic ODE (saturated, da=100)"),
                       ("C", "bare gate, no growth")):
        wonly = np.nanmean([r[3][tag][0][0] for r in rows])
        wcomp = np.nanmean([r[3][tag][1][0] for r in rows])
        offrel = np.nanmean([r[3][tag][1][2] for r in rows])
        noff = np.nanmean([r[3][tag][1][3] for r in rows])
        print("  %-1s %-52s wall-only %.4f  +lumen %.4f  (offRel %.4f, nOffPred %.1f)"
              % (tag, label, wonly, wcomp, offrel, noff))

    full = [r for r in rows if r[1] >= 150]
    if full:
        print("\n  full-horizon subset only (T>=150, n=%d):" % len(full))
        print("    learned %.4f" % np.nanmean([r[2] for r in full]))
        for tag, label in (("A", "gate+growth"), ("B", "ODE(sat)"), ("C", "bare")):
            wcomp = np.nanmean([r[3][tag][1][0] for r in full])
            print("    %s (%s) +lumen: %.4f" % (tag, label, wcomp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
