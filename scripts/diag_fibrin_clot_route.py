"""PHASE 8: GT clot has TWO routes and the repo models one of them.

``gt_clot_phi_at_time`` does not threshold ``Mat``.  It thresholds VISCOSITY GROWTH,
``relu(mu_eff(t) - mu_eff(0)) >= thresh``, on COMSOL's own ``mu_eff`` field.  And the ``.mph``
says what sets that field (``scripts/diag_mph_surface_law.py`` 4b):

    mu = mu_b * ( mu2(FI) + mu1(Mat) )

    mu1(Mat)   step, 1 -> 80  at Mat = 2e7   width 7e6     the platelet route
    mu2(FI)    step, 0 -> 80  at FI  = 0.6   width 0.1     the FIBRIN route

Both routes are worth 80x on their own.  The repo predicts clot as ``Mat >= crit``, i.e. the
``mu1`` route only; ``mu2`` is absent from the model entirely.  ``docs/PHASE7_FINDINGS.md``
records fibrin as "inert" -- that was inferred from a parameter list, and 0 warns not to trust
those over the node tree.

THE UNIT-FREE TEST.  ``Mat``'s scaling into COMSOL model units is known (x7e10, the same
constant the off-wall arm uses) but ``FI``'s is not, and guessing it would make this circular.
So test the complement instead: on nodes that GT calls clot, how many have ``Mat`` BELOW the
platelet threshold?  For those, the viscosity growth cannot have come from ``mu1``, so it came
from fibrin.  That needs no assumption about ``FI``'s units at all.

Split wall vs off-wall, because the off-wall arm is the one that fails: it tries to explain
off-wall clot as wall ``Mat`` attenuated by 0.16 into the first shell, and if off-wall clot is
actually fibrin it is explaining the wrong field.

    python scripts/diag_fibrin_clot_route.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.biochem_gnn.mat_growth_simple import WALL_COHORT_V2_TRAIN  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.species_pushforward_continuous import (  # noqa: E402
    resolve_deploy_eval_time_index,
)
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

DIR = REPO / "data/processed/graphs_biochem_anchors"
MAT_S = 7e10        # Mat_log1p_nd -> COMSOL model units [plt/cm^2]
MU1_LOC, MU1_W = 2.0e7, 7.0e6      # the platelet viscosity step
MU2_LOC, MU2_W = 0.6, 0.1          # the fibrin viscosity step, in COMSOL's FI units


def chan(d, names, key, scale=1.0):
    return np.expm1(d.y[-1, :, names.index(key)].double().numpy()) * scale


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default="outputs/phase8_fibrin_route.json")
    args = ap.parse_args()
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    rows = []

    print("%-12s %7s %9s %9s   %9s %9s   %9s"
          % ("anchor", "GTclot", "wall:sub", "off:sub", "wall_n", "off_n", "FI@sub/FI@sup"))
    for anchor in WALL_COHORT_V2_TRAIN:
        p = DIR / f"{anchor}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        if int(d.y.shape[0]) < 150:
            continue
        names = d.y_channel_names.split(",")
        wall = d.mask_wall.reshape(-1).bool().numpy()
        t_eval = resolve_deploy_eval_time_index(int(d.y.shape[0]))
        gt = gt_clot_phi_at_time(d, t_eval, phys, device=torch.device("cpu")
                                 ).reshape(-1).numpy() > 0.5
        if gt.sum() == 0:
            continue
        mat = chan(d, names, "Mat_log1p_nd", MAT_S)
        fi_nd = chan(d, names, "FI_log1p_nd")
        # "sub-threshold" = GT says clot, but Mat is below where mu1 can lift the viscosity.
        # Use the BOTTOM of the smoothing zone so this is the conservative reading: anything
        # below loc - w/2 has mu1 identically 1 and contributes exactly nothing.
        sub = gt & (mat < (MU1_LOC - MU1_W / 2.0))
        gw, go = gt & wall, gt & ~wall
        sw, so = sub & wall, sub & ~wall
        fr = lambda a, b: float(a.sum()) / max(int(b.sum()), 1)
        fi_sub = float(np.median(fi_nd[sub])) if sub.sum() else np.nan
        fi_sup = float(np.median(fi_nd[gt & ~sub])) if (gt & ~sub).sum() else np.nan
        # THE CONVERSE.  If no clot node is below the step and no node above the step is
        # clot-free, then GT clot IS the level set {Mat >= crit} and nothing else -- which
        # makes the whole deploy score a question about one field's magnitude.
        hot = mat >= (MU1_LOC + MU1_W / 2.0)
        leak = hot & ~gt
        print("%-12s %7d %9.3f %9.3f   %9d %9d   %9.3f"
              % (anchor, int(gt.sum()), fr(sw, gw), fr(so, go), int(gw.sum()),
                 int(go.sum()), fi_sub / max(fi_sup, 1e-30)))
        rows.append(dict(anchor=anchor, n_gt=int(gt.sum()), n_wall=int(gw.sum()),
                         n_off=int(go.sum()), sub_wall=fr(sw, gw), sub_off=fr(so, go),
                         sub_all=fr(sub, gt), fi_sub=fi_sub, fi_sup=fi_sup,
                         fi_max=float(fi_nd.max()), mat_max=float(mat.max()),
                         n_hot=int(hot.sum()), leak=fr(leak, hot)))

    if not rows:
        print("no vessels")
        return 1
    g = lambda k: np.array([r[k] for r in rows], dtype=float)
    nw, no = g("n_wall").sum(), g("n_off").sum()
    print("\n=== HOW MUCH GT CLOT THE PLATELET ROUTE CANNOT EXPLAIN (%d vessels) ==="
          % len(rows))
    print("   'sub-threshold' = GT clot AND Mat < %.3g, where mu1 is identically 1"
          % (MU1_LOC - MU1_W / 2.0))
    print("      wall  GT clot nodes      %6d   sub-threshold %5.1f%%"
          % (nw, 100 * (g("sub_wall") * g("n_wall")).sum() / max(nw, 1)))
    print("      off   GT clot nodes      %6d   sub-threshold %5.1f%%"
          % (no, 100 * (g("sub_off") * g("n_off")).sum() / max(no, 1)))
    print("      pooled                   %6d   sub-threshold %5.1f%%"
          % (nw + no, 100 * (g("sub_all") * g("n_gt")).sum() / max(g("n_gt").sum(), 1)))
    print("   FI_nd max over cohort %.4g   (mu2 fires at FI = %.2f in COMSOL units)"
          % (np.nanmax(g("fi_max")), MU2_LOC))
    print("\n=== THE CONVERSE: is every high-Mat node clot? ===")
    print("   nodes with Mat >= %.3g       %6d   of those, NOT GT clot %5.2f%%"
          % (MU1_LOC + MU1_W / 2.0, g("n_hot").sum(),
             100 * (g("leak") * g("n_hot")).sum() / max(g("n_hot").sum(), 1)))
    print("\n   Both directions clean means GT clot IS the level set {Mat >= crit}: one field,")
    print("   one threshold, no second mechanism.  The entire deploy score -- wall AND off-wall")
    print("   -- is then a question about how well Mat's MAGNITUDE is reproduced near 2e7,")
    print("   which is what FINDINGS 7.2 measured as 53% of the gap and 9 could not fix with")
    print("   ordering.  It also means off-wall GT clot nodes carry real Mat >= crit of their")
    print("   own; they are not a faint attenuated echo of the wall, so the off-wall arm's")
    print("   static MAT_ATTENUATION = 0.16 is standing in for wall-normal TRANSPORT.")

    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
