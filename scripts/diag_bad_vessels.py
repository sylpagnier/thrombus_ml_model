"""Why do patient008/009 over-predict 10x while 32 other vessels do not?

The t=0 two-gate model scores 0.19 / 0.13 on these two and 0.75+ elsewhere.  Either the
reconstructed field is wrong there, or the gate is genuinely open where nothing commits.
Compares gate composition, flow scale and geometry across the cohort.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.biochem_gnn.mat_growth_simple import (  # noqa: E402
    WALL_COHORT_V2_GENERALIZATION, WALL_COHORT_V2_TRAIN,
)
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.physics_wall_model import t0_flow_fields  # noqa: E402
from src.core_physics.species_pushforward_continuous import resolve_deploy_eval_time_index  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

DIR = Path("data/processed/graphs_biochem_anchors")


def main() -> int:
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    names = sorted(set(WALL_COHORT_V2_TRAIN) | set(WALL_COHORT_V2_GENERALIZATION)
                   | {"patient042", "patient043"})
    print("%12s %8s %8s %6s %6s %7s %7s %7s %7s %7s"
          % ("vessel", "u_ref", "d_bar", "nGT", "nPred", "srWmed", "srWq10",
             "%low", "%sep", "gtMatN"))
    rows = []
    for a in names:
        p = DIR / f"{a}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        wall = d.mask_wall.reshape(-1).bool().numpy()
        f = t0_flow_fields(d, bio, hops=3)
        t_eval = resolve_deploy_eval_time_index(int(d.y.shape[0]))
        gt = gt_clot_phi_at_time(d, t_eval, phys, device=torch.device("cpu")).reshape(-1).numpy()
        gt = (gt > 0.5) & wall
        pred = (f.gate > 0) & wall
        nm = d.y_channel_names.split(",")
        mat = torch.expm1(d.y[-1, :, nm.index("Mat_log1p_nd")].clamp(-10, 8)).numpy() * float(bio.Minf)
        r = dict(a=a, u_ref=float(d.u_ref.reshape(-1)[0]), d_bar=float(d.d_bar.reshape(-1)[0]),
                 ngt=int(gt.sum()), npred=int(pred.sum()),
                 srmed=float(np.median(f.sr[wall])), srq10=float(np.percentile(f.sr[wall], 10)),
                 flow=float(np.percentile(np.hypot(d.y[0, :, 0].numpy(), d.y[0, :, 1].numpy())[~wall], 50)),
                 plow=float(f.gate_low[wall].mean()), psep=float(f.gate_sep[wall].mean()),
                 nmat=int(((mat >= float(bio.viscosity_mat_crit)) & wall).sum()))
        rows.append(r)
        print("%12s %8.4f %8.4f %6d %6d %7.2f %7.3f %7.3f %7.3f %7d"
              % (a, r["u_ref"], r["d_bar"], r["ngt"], r["npred"], r["srmed"], r["srq10"],
                 r["plow"], r["psep"], r["nmat"]))

    print("\n[GT label agreement] does gt_clot_phi (mu_eff growth) match Mat>=2e7?")
    bad = [r for r in rows if r["a"] in ("patient008", "patient009")]
    for r in rows:
        tag = "  <<<" if r["a"] in ("patient008", "patient009") else ""
        if abs(r["nmat"] - r["ngt"]) > 0.3 * max(r["ngt"], 1) or tag:
            print("   %12s  n_phi_gt %4d   n_Mat>=crit %4d%s" % (r["a"], r["ngt"], r["nmat"], tag))

    print("\n[flow scale] u_ref of the two bad vessels vs cohort")
    ur = np.array([r["u_ref"] for r in rows])
    print("   cohort u_ref pct[5,50,95] = %s" % np.round(np.percentile(ur, [5, 50, 95]), 4))
    for r in bad:
        print("   %s u_ref %.4f  (percentile %.0f%%)  median wall sr %.2f 1/s"
              % (r["a"], r["u_ref"], 100 * (ur < r["u_ref"]).mean(), r["srmed"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
