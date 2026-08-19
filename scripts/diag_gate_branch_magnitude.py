"""PHASE 8: the SEPARATION branch is what breaks wall-Mat ordering, not a missing sink.

`docs/PHASE7_FINDINGS.md` 9.2 measured an oracle per-node-ODE ceiling of ~0.31 against GT
wall Mat and concluded the equation is short a REMOVAL term (9.3, `-lambda*sr*Mat`, oracle
0.310 -> 0.447 LOO).  `diag_wall_mat_closure_terms.py` finds a larger and simpler effect
that the removal story does not touch.

COMSOL's gate is a SUM of two structurally different things:

    gate = A + B      A = (L/gamma_m)*|d(sr,x)|  where d(sr,x) < sgt      <- a MAGNITUDE
                      B = 1                      where sr    < lss        <- an INDICATOR

`L/gamma_m = 0.05` and `|d(sr,x)|` is order 1e2-1e4 on these meshes, so **A outweighs B by
one to two orders of magnitude wherever it fires**, and wall Mat magnitude is therefore set
almost entirely by an MLS estimate of a derivative whose RANK was validated (0.990 vs
COMSOL on patient007) but whose MAGNITUDE never was.

This script sweeps how much of A the rate is allowed to see, holding the MASK fixed (the
mask uses `A > 0 or B > 0` and is not touched here -- 10.3 shows graph-growth FP is 2 nodes
and every wall FN is a closed t=0 gate, so mask and rate are separable).

    python scripts/diag_gate_branch_magnitude.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.biochem_gnn.mat_growth_simple import WALL_COHORT_V2_TRAIN  # noqa: E402
from src.config import BiochemConfig  # noqa: E402
from src.core_physics.temporal_metrics import spearman  # noqa: E402

CACHE = REPO / "outputs/wall_species_cache"
M_TO_CM = 100.0
PER_M2_TO_PER_CM2 = 1.0e-4


def load(anchor, bio):
    cf = CACHE / f"{anchor}.npz"
    if not cf.exists():
        return None
    z = np.load(cf)
    if "sr_t" not in z.files or int(z["t"].shape[0]) < 150:
        return None
    lss, sgt = float(bio.lss), float(bio.sgt) / M_TO_CM
    coef = float(bio.L_char) * M_TO_CM / float(bio.gamma_m)
    sr, dsrx = z["sr_t"], z["dsrx_t"]
    A = (dsrx < sgt) * coef * np.abs(dsrx)
    B = (sr < lss) * 1.0
    minf = float(bio.Minf) * PER_M2_TO_PER_CM2
    sat = np.clip(1.0 - z["m_tot"] / minf, 0.0, 1.0)
    C = (sat * (float(bio.k_rs) * M_TO_CM * z["rp"] + float(bio.k_as) * M_TO_CM * z["ap"])
         + (z["mas"] / minf) * float(bio.k_aa) * M_TO_CM * z["ap"])
    return dict(anchor=anchor, t=z["t"], A=A, B=B, C=C, mat=z["mat"][-1],
                wall_edges=z["wall_edges"], nw=len(z["wall_idx"]))


def integ(src, t):
    out = np.zeros(src.shape[1])
    return out + np.sum(src[:-1] * np.diff(t)[:, None], axis=0)


def lap(v):
    e = v["wall_edges"]
    A = sp.coo_matrix((np.ones(e.shape[1]), (e[0], e[1])), shape=(v["nw"], v["nw"])).tocsr()
    A = ((A + A.T) > 0).astype(np.float64)
    A.setdiag(0.0)
    A.eliminate_zeros()
    return sp.diags(np.asarray(A.sum(axis=1)).reshape(-1)) - A


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default="outputs/phase8_gate_branch_magnitude.json")
    args = ap.parse_args()
    bio = BiochemConfig(phase="biochem")
    V = [x for x in (load(a, bio) for a in WALL_COHORT_V2_TRAIN) if x is not None]

    # A_CAP: the largest value branch A may contribute to the RATE.  inf = shipped law.
    CAPS = [np.inf, 100.0, 10.0, 3.0, 1.0, 0.3, 0.1, 0.0]
    res = {c: [] for c in CAPS}
    sep_only, both_live, names = [], [], []

    for v in V:
        live = integ((v["A"] + v["B"]) * v["C"], v["t"]) > 0
        if live.sum() < 20:
            continue
        names.append(v["anchor"])
        # is this a separation-driven vessel?  (does the low-shear branch ever fire)
        sep_only.append(not (v["B"][:, live] > 0).any())
        for c in CAPS:
            g = v["B"] + (v["A"] if np.isinf(c) else np.minimum(v["A"], c))
            res[c].append(spearman(integ(g * v["C"], v["t"])[live], v["mat"][live]))
        both_live.append(live)

    n = len(names)
    sep_only = np.array(sep_only)
    print("=== oracle per-node ODE vs GT wall Mat, %d vessels ===" % n)
    print("rate sees  gate = B + min(A, cap).  MASK is untouched.\n")
    print("%9s %9s %9s %9s %9s" % ("A cap", "all", "low-shear", "sep-only", "anti-corr"))
    print("%9s %9s %9s %9s %9s"
          % ("", "n=%d" % n, "n=%d" % int((~sep_only).sum()), "n=%d" % int(sep_only.sum()), ""))
    for c in CAPS:
        r = np.array(res[c], dtype=float)
        print("%9s %9.3f %9.3f %9.3f %9d"
              % ("inf (ships)" if np.isinf(c) else "%.1f" % c, np.nanmean(r),
                 np.nanmean(r[~sep_only]), np.nanmean(r[sep_only]) if sep_only.any() else np.nan,
                 int((r < 0).sum())))

    print("\nper vessel (sep-only vessels marked *):")
    print("%-12s %8s %8s %8s" % ("anchor", "shipped", "cap=1.0", "cap=0"))
    for i, a in enumerate(names):
        print("%-12s %8.3f %8.3f %8.3f%s"
              % (a, res[np.inf][i], res[1.0][i], res[0.0][i], "  *" if sep_only[i] else ""))

    # Does neighbour mixing stack on top of the best cap?
    print("\nneighbour mixing (I + kappa*L)^-1 on top of the best cap:")
    best = min(CAPS, key=lambda c: -np.nanmean(np.array(res[c], dtype=float)))
    for k in (0.0, 1.0, 4.0, 16.0):
        rr = []
        for v, live in zip(V, both_live):
            g = v["B"] + (v["A"] if np.isinf(best) else np.minimum(v["A"], best))
            x = integ(g * v["C"], v["t"])
            if k:
                L = lap(v)
                x = np.asarray(spla.spsolve((sp.eye(v["nw"], format="csr") + k * L).tocsc(), x))
            rr.append(spearman(x[live], v["mat"][live]))
        print("   cap=%s kappa=%5.1f   %.3f"
              % ("inf" if np.isinf(best) else "%.1f" % best, k, np.nanmean(np.array(rr, float))))

    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"vessels": names, "sep_only": sep_only.tolist(),
         "rho": {("inf" if np.isinf(c) else str(c)): [float(x) for x in res[c]] for c in CAPS}},
        indent=2))
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
