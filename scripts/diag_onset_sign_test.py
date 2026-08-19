"""Within the gated set, does LOWER shear ignite sooner or LATER?

The graded gate (13.2) assumes deeper-in-the-stagnation-zone ignites sooner: it sets
``g_low = sigmoid((lss - sr)/tau)``, which is LARGER for smaller ``sr``.  COMSOL's own
patient007 export says the opposite -- among the 139 nodes whose gate is exactly 1
(low-shear only, so the law is identical for all of them), onset spreads over 0.560 of
the horizon and

    spearman(sr at t=0, onset time) = -0.585      i.e. HIGHER sr ignites EARLIER
    spearman(ap at onset, onset time) = -0.822    i.e. LOWER ap ignites LATER

which is consistent with transport: within the stagnation band, more shear delivers more
activated platelets, so deposition runs faster.  If that sign holds across the cohort the
graded gate is anti-correlated with the truth, which would explain why it improved the
aggregate curve (it spread onsets out) while REDUCING rank correlation (it spread them in
the wrong order).

One vessel is not enough for a claim this consequential, so this re-tests it on every
full-horizon vessel using the packs.
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
from src.core_physics.species_fields import gt_species_trajectory  # noqa: E402
from src.core_physics.temporal_metrics import gt_onset_index, spearman  # noqa: E402

DIR = Path("data/processed/graphs_biochem_anchors")


def main() -> int:
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    lss, sgt = float(bio.lss), float(bio.sgt) / 100.0
    print("%-12s %6s | %8s %8s | %8s %8s %8s"
          % ("vessel", "n_g1", "spread", "rho(sr)", "rho(ap0)", "rho(apF)", "rho(|dsx|)"))
    rows = []
    for n in sorted(set(WALL_COHORT_V2_TRAIN) | set(WALL_COHORT_V2_GENERALIZATION)):
        p = DIR / f"{n}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        T = int(d.y.shape[0])
        if T < 150:
            continue
        w = d.mask_wall.reshape(-1).bool().numpy()
        f = t0_flow_fields(d, bio, hops=3, flow_source="gt")
        on = gt_onset_index(d, phys, w)
        t = d.t.reshape(-1).numpy()
        # nodes whose gate is EXACTLY 1: low-shear open, separation shut
        g1 = w & (f.sr < lss) & ~(f.dsrx < sgt) & (on >= 0)
        if g1.sum() < 8:
            continue
        _, ap = gt_species_trajectory(d, bio)
        ot = t[on[g1]]
        r = dict(name=n, n=int(g1.sum()), spread=float((ot.max() - ot.min()) / t[-1]),
                 rho_sr=spearman(f.sr[g1], ot),
                 rho_ap0=spearman(ap[0][g1], ot),
                 rho_apF=spearman(ap[-1][g1], ot),
                 rho_dsx=spearman(np.abs(f.dsrx)[g1], ot),
                 sealed=n in WALL_COHORT_V2_GENERALIZATION)
        rows.append(r)
        print("%-12s %6d | %8.3f %8.3f | %8.3f %8.3f %8.3f"
              % (n, r["n"], r["spread"], r["rho_sr"], r["rho_ap0"], r["rho_apF"], r["rho_dsx"]))

    if not rows:
        print("no vessel had enough gate==1 nodes")
        return 1
    print("\nn=%d vessels with >=8 identically-gated (gate==1) committing nodes" % len(rows))
    for k, lbl in (("spread", "onset spread, fraction of horizon"),
                   ("rho_sr", "spearman(sr@t0, onset)"),
                   ("rho_ap0", "spearman(ap@t0, onset)"),
                   ("rho_apF", "spearman(ap@t_final, onset)"),
                   ("rho_dsx", "spearman(|dsrx|@t0, onset)")):
        v = np.array([r[k] for r in rows])
        v = v[np.isfinite(v)]
        print("   %-34s mean %+.3f  median %+.3f  (%d/%d negative)"
              % (lbl, v.mean(), np.median(v), int((v < 0).sum()), len(v)))
    sr = np.array([r["rho_sr"] for r in rows])
    sr = sr[np.isfinite(sr)]
    print("\n  VERDICT on the graded gate's sign:")
    if (sr < 0).mean() > 0.7:
        print("     %d/%d vessels have HIGHER sr igniting EARLIER inside the gated band."
              % (int((sr < 0).sum()), len(sr)))
        print("     The graded gate boosts LOW sr, so its ordering is BACKWARDS.")
    elif (sr > 0).mean() > 0.7:
        print("     lower sr does ignite earlier -- the graded gate's sign is right.")
    else:
        print("     the sign is vessel-dependent (%d/%d negative); no cohort-wide rule."
              % (int((sr < 0).sum()), len(sr)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
