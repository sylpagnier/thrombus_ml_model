"""Does the ADVECTION operator explain off-wall `Mat` better than the 0.16 owner rule?

PHASE7 3.2 measured `Mat_off / Mat_owner = 0.16` on all 12 clot-carrying vessels and the
whole off-wall stack was built on it.  PHASE7 12.5 then measured that the residual is the
**variance** of that ratio (0.12-0.19 within a vessel) and guessed it was a mesh quantity.

The `.mph` (PHASE7 1.1) says off-wall `Mat` is an advected field, so the variance should be
a *flow* quantity: two shell nodes with the same owner see different `Mat` if one sits in a
recirculation and the other is swept.  `src/clot_ml/transport.py` solves COMSOL's own
operator; this script asks whether that beats the owner rule at ordering GT `Mat` off-wall.

    python scripts/diag_advective_offwall.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.clot_ml.data import load_cache  # noqa: E402
from src.clot_ml.transport import transport_fields  # noqa: E402


def rho(a, b):
    if len(a) < 8 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    return float(spearmanr(a, b).statistic)


def main() -> int:
    cache = load_cache("gt")
    rows = []
    t0 = time.time()
    for a, S in sorted(cache.items()):
        wall, ei, pos = S["wall"], S["edge_index"], S["pos"].astype(np.float64)
        u, v = S["u"].astype(np.float64), S["v"].astype(np.float64)
        mat_phys, owner = S["mat_phys"].astype(np.float64), S["owner"]
        y, mat_gt = S["y"] > 0.5, S["mat_gt"].astype(np.float64)

        # horizon in the same nondimensional length/velocity units the pack uses: the
        # domain crossing time at the mean speed, times a factor for the run length.
        L = float(np.ptp(pos[:, 0]) + np.ptp(pos[:, 1]))
        spd = float(np.median(np.hypot(u, v)[~wall])) + 1e-12
        H = L / spd

        T = transport_fields(pos, ei, u, v, wall, mat_phys, horizon=H)
        off = ~wall
        live = off & (mat_gt > 0)                 # nodes that carry any species at all
        if live.sum() < 30:
            continue
        r = dict(
            anchor=a, n_off_gt=int((y & off).sum()), n_live=int(live.sum()),
            owner=rho(mat_phys[owner][live], mat_gt[live]),
            adv=rho(T["mat_adv"][live], mat_gt[live]),
            adv_n=rho(T["mat_adv_n"][live], mat_gt[live]),
            tau=rho(T["tau"][live], mat_gt[live]),
            reach=rho(T["src_reach"][live], mat_gt[live]),
        )
        # ratio variance the owner rule cannot see, against what advection says
        both = off & (mat_gt > 0) & (mat_phys[owner] > 0)
        r["ratio_iqr"] = float(np.subtract(*np.percentile(
            np.expm1(mat_gt[both]) / np.maximum(mat_phys[owner][both] / 2e7, 1e-12),
            [75, 25]))) if both.sum() > 8 else float("nan")
        rows.append(r)
        print("  %-11s off_gt %4d  owner %+.3f  adv %+.3f  adv_n %+.3f  tau %+.3f  reach %+.3f"
              % (a, r["n_off_gt"], r["owner"], r["adv"], r["adv_n"], r["tau"], r["reach"]),
              flush=True)

    print("\n%d vessels, %.0fs" % (len(rows), time.time() - t0))
    print("%-10s %8s" % ("field", "mean rho"))
    for k in ("owner", "adv", "adv_n", "tau", "reach"):
        print("%-10s %+8.4f" % (k, np.nanmean([r[k] for r in rows])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
