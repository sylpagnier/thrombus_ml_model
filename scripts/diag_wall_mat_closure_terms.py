"""PHASE 8 verification: WHY the oracle per-node ODE only ranks GT wall Mat at ~0.31.

`docs/PHASE7_FINDINGS.md` 9.2 established the ceiling and concluded "the equation is short a
term", then tested a REMOVAL term (`-lambda*sr*Mat`).  This script tests three other
candidates that the removal story does not cover, all on the same perfect oracle:

  A. GATE SHAPE.  The gate is ``A + B`` with ``A = (L/gamma_m)*|dsrx|`` (a MAGNITUDE, only
     where ``dsrx < sgt``) and ``B = 1`` (an INDICATOR, where ``sr < lss``).  With
     ``L/gamma_m = 0.05`` and typical ``|dsrx| ~ 1e3``, branch A can outweigh branch B by
     ~50x, so wall Mat magnitude is dominated by an MLS-estimated derivative.  The repo only
     ever validated its RANK (0.990 vs COMSOL).  If the magnitude is the problem, capping or
     re-shaping A should move the oracle rank a lot.

  B. NEIGHBOUR MIXING.  COMSOL's ``tds2`` is a FEM solve with a CONSISTENT mass matrix and
     Do Carmo/Galeao crosswind stabilisation.  Even with ``D = 0`` and no-slip (``u = 0`` at
     the wall, which is why 9.2's tangential-advection test correctly read 0.000), the
     nodal equation ``M dc/dt = f - K c`` couples a wall node to its neighbours through
     ``M^-1``.  That predicts GT Mat is a SMOOTHED version of the local flux integral --
     a redistribution, not a removal.  Tested as ``(I + kappa*L)^-1`` applied to the local
     integral, sweeping kappa.

  C. TIME-WEIGHTING.  The local integral weights every timestep equally.  If the gate
     migrates, what matters is how long a node was gated, not the final gate.

    python scripts/diag_wall_mat_closure_terms.py
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


def branches(z, bio):
    """The two gate branches, kept separate.  [T, W] each."""
    lss = float(bio.lss)
    sgt = float(bio.sgt) / M_TO_CM
    coef = float(bio.L_char) * M_TO_CM / float(bio.gamma_m)
    sr, dsrx = z["sr_t"], z["dsrx_t"]
    A = (dsrx < sgt) * coef * np.abs(dsrx)      # separation, a MAGNITUDE
    B = (sr < lss) * 1.0                        # stagnation, an INDICATOR
    return A, B


def chem(z, bio):
    k_rs, k_as, k_aa = (float(bio.k_rs) * M_TO_CM, float(bio.k_as) * M_TO_CM,
                        float(bio.k_aa) * M_TO_CM)
    minf = float(bio.Minf) * PER_M2_TO_PER_CM2
    sat = np.clip(1.0 - z["m_tot"] / minf, 0.0, 1.0)
    return sat * (k_rs * z["rp"] + k_as * z["ap"]) + (z["mas"] / minf) * k_aa * z["ap"]


def integrate(src, t):
    out = np.zeros_like(src)
    out[1:] = np.cumsum(src[:-1] * np.diff(t)[:, None], axis=0)
    return out[-1]


def wall_laplacian(z):
    """Unnormalised graph Laplacian on the wall subgraph (the wall is a 1-D chain)."""
    w = len(z["wall_idx"])
    e = z["wall_edges"]
    A = sp.coo_matrix((np.ones(e.shape[1]), (e[0], e[1])), shape=(w, w)).tocsr()
    A = ((A + A.T) > 0).astype(np.float64)
    A.setdiag(0.0)
    A.eliminate_zeros()
    deg = np.asarray(A.sum(axis=1)).reshape(-1)
    return sp.diags(deg) - A


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default="outputs/phase8_wall_mat_closure_terms.json")
    args = ap.parse_args()
    bio = BiochemConfig(phase="biochem")
    KAPPAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    arms = ["local (9.2 baseline)", "A: sep as indicator", "A: sep magnitude capped",
            "B: low-shear branch only", "C: gated-time only"]
    acc = {a: [] for a in arms}
    smooth = {k: [] for k in KAPPAS}
    per_vessel = {}

    for anchor in WALL_COHORT_V2_TRAIN:
        cf = CACHE / f"{anchor}.npz"
        if not cf.exists():
            continue
        z = np.load(cf)
        if "sr_t" not in z.files or int(z["t"].shape[0]) < 150:
            continue
        t, mat_gt = z["t"], z["mat"][-1]
        A, B = branches(z, bio)
        C = chem(z, bio)
        base = integrate((A + B) * C, t)
        live = base > 0
        if live.sum() < 20:
            continue
        cand = {
            "local (9.2 baseline)": base,
            "A: sep as indicator": integrate(((A > 0) * 1.0 + B) * C, t),
            "A: sep magnitude capped": integrate((np.minimum(A, 1.0) + B) * C, t),
            "B: low-shear branch only": integrate(B * C, t),
            "C: gated-time only": integrate(((A + B) > 0) * 1.0, t),
        }
        row = {}
        for a in arms:
            r = spearman(cand[a][live], mat_gt[live])
            acc[a].append(r)
            row[a] = float(r)
        # B: neighbour mixing.  (I + kappa L)^-1 base -- redistribute, do not remove.
        L = wall_laplacian(z)
        I = sp.eye(L.shape[0], format="csr")
        for k in KAPPAS:
            sm = base if k == 0 else spla.spsolve((I + k * L).tocsc(), base)
            r = spearman(np.asarray(sm)[live], mat_gt[live])
            smooth[k].append(r)
            row[f"smooth_{k}"] = float(r)
        per_vessel[anchor] = row
        print("%-12s n=%4d | local %.3f | sep-ind %.3f | lowshear %.3f | smooth(k=4) %.3f"
              % (anchor, int(live.sum()), row["local (9.2 baseline)"],
                 row["A: sep as indicator"], row["B: low-shear branch only"],
                 row["smooth_4.0"]))

    n = len(per_vessel)
    print("\n=== %d vessels, spearman(oracle prediction, GT wall Mat) ===" % n)
    print("A/C  gate-shape and time-weighting variants")
    for a in arms:
        v = np.array(acc[a], dtype=float)
        print("   %-28s %.3f   (anti-correlated on %d/%d)"
              % (a, np.nanmean(v), int((v < 0).sum()), n))
    print("B    neighbour mixing: (I + kappa*L)^-1 applied to the local integral")
    for k in KAPPAS:
        v = np.array(smooth[k], dtype=float)
        print("   kappa = %5.2f                 %.3f   (anti-correlated on %d/%d)"
              % (k, np.nanmean(v), int((v < 0).sum()), n))

    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(per_vessel, indent=2))
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
