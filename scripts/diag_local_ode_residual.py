"""PHASE 8: what is the per-node surface ODE's residual MADE of?

``scripts/diag_local_ode_closure.py`` shows the per-node ODE cannot reproduce GT ``Mat``
even when handed perfect GT ``RP/AP/M/Mas/sr/dsrx`` at every timestep: rank 0.63, log-R2
negative, and the per-node rate scalar needed to close the local balance varies ~67% WITHIN
a single vessel.  So the deficit is structural.  This script asks what the structure is.

The flux-to-concentration conversion is the suspect.  ``J0_Mat`` is a surface flux
[plt/(cm^2 s)] and ``Mat`` is a domain concentration, so the balance carries a 1/LENGTH:

    d(Mat_i)/dt = J0_Mat_i / h_i

``docs/PHASE7_FINDINGS.md`` 2 identified that length as the cell size and folded it into a
SINGLE GLOBAL SCALAR, ``da_scale = 40``, for the whole mesh.  But ``h_i`` is local, and these
meshes are graded -- refined at stenoses, coarse in straight runs.  If that is what is
happening, the model carries a multiplicative per-node error of exactly ``h_bar/h_i``, which
is a pure ORDERING error, is spatially organised, is invisible to any global calibration,
and is the thing that would make ``da_scale`` mesh-dependent (8's portability worry).

Competing explanations tested alongside it, so this is an attribution and not a fishing
expedition:

    h_local      local element size            -> the 1/h_i hypothesis above
    upstream     upwind-accumulated J0         -> tangential advection of a D=0 domain field
    nbr_smear    neighbour mean of the prediction -> mass-matrix / stabilisation smearing
    midside      corner vs mid-edge node       -> the 8 P2 structure
    wall_dist    distance along the wall       -> any residual global trend

Each is scored by how much of the log residual's variance it removes, and by what it does to
the rank correlation against GT ``Mat`` -- the number 8.5 measured at 0.193 for the shipped
model and this script measures at 0.633 for the local oracle.

    python scripts/diag_local_ode_residual.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.biochem_gnn.mat_growth_simple import WALL_COHORT_V2_TRAIN  # noqa: E402
from src.config import BiochemConfig  # noqa: E402
from src.core_physics.physics_lumen_model import midside_nodes  # noqa: E402
from src.core_physics.physics_wall_model import gate_from_shear  # noqa: E402
from src.core_physics.temporal_metrics import spearman  # noqa: E402

DIR = REPO / "data/processed/graphs_biochem_anchors"
CACHE = REPO / "outputs/wall_species_cache"
M_TO_CM = 100.0
PER_M2_TO_PER_CM2 = 1.0e-4
EXPLANATORS = ("h_local", "upstream", "nbr_smear", "midside", "wall_dist")


def j0_mat(z, bio):
    """COMSOL's ``J0_Mat`` on GT fields, [T, W], CGS, including ``Da``."""
    k_rs = float(bio.k_rs) * M_TO_CM
    k_as = float(bio.k_as) * M_TO_CM
    k_aa = float(bio.k_aa) * M_TO_CM
    minf = float(bio.Minf) * PER_M2_TO_PER_CM2
    da = float(bio.surface_damkohler)
    gate = gate_from_shear(z["sr_t"], z["dsrx_t"], bio)
    sat = np.clip(1.0 - z["m_tot"] / minf, 0.0, 1.0)
    chem = sat * (k_rs * z["rp"] + k_as * z["ap"]) + (z["mas"] / minf) * k_aa * z["ap"]
    return da * gate * chem


def wall_graph(z):
    """Row-stochastic adjacency over the WALL subgraph, and per-node local edge length."""
    w = int(z["wall_idx"].shape[0])
    ei = np.asarray(z["wall_edges"])
    A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(w, w)).tocsr()
    A = ((A + A.T) > 0).astype(np.float64)
    A.setdiag(0.0)
    A.eliminate_zeros()
    pos = z["pos"]
    src, dst = A.nonzero()
    ln = np.linalg.norm(pos[src] - pos[dst], axis=1)
    h = np.zeros(w)
    for i in range(w):
        m = src == i
        h[i] = np.median(ln[m]) if m.any() else np.nan
    h[~np.isfinite(h)] = np.nanmedian(h)
    deg = np.asarray(A.sum(1)).reshape(-1)
    P = sp.diags(1.0 / np.maximum(deg, 1.0)) @ A
    return A, P, h


def upstream_accum(z, d, A, integ, *, decay=0.5, hops=6):
    """Upwind-gathered ``integ`` along the wall, using the near-wall GT flow direction."""
    widx = z["wall_idx"]
    u = d.y[:, :, 0].double().numpy()[:, widx].mean(0)
    v = d.y[:, :, 1].double().numpy()[:, widx].mean(0)
    pos = z["pos"]
    src, dst = A.nonzero()
    dvec = pos[dst] - pos[src]
    # j is UPSTREAM of i when the step j->i goes with the flow at i.
    flow = np.stack([u[src], v[src]], 1)
    w = -(dvec * flow).sum(1)
    w = np.clip(w, 0.0, None)
    U = sp.coo_matrix((w, (src, dst)), shape=A.shape).tocsr()
    rs = np.asarray(U.sum(1)).reshape(-1)
    U = sp.diags(1.0 / np.maximum(rs, 1e-30)) @ U
    out, cur = np.zeros_like(integ), integ.copy()
    for k in range(hops):
        cur = U @ cur
        out += (decay ** (k + 1)) * cur
    return out


def r2_of(y, X):
    """log-space R2 of an OLS fit of ``y`` on ``[1, X]``."""
    A = np.column_stack([np.ones(len(y))] + [x for x in X])
    good = np.isfinite(A).all(1) & np.isfinite(y)
    if good.sum() < 12:
        return float("nan"), None
    beta, *_ = np.linalg.lstsq(A[good], y[good], rcond=None)
    res = y[good] - A[good] @ beta
    ss, st = float((res ** 2).sum()), float(((y[good] - y[good].mean()) ** 2).sum())
    return 1.0 - ss / max(st, 1e-30), (A, beta, good)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default="outputs/phase8_local_ode_residual.json")
    args = ap.parse_args()
    bio = BiochemConfig(phase="biochem")
    lg = lambda a: np.log10(np.maximum(np.asarray(a, dtype=np.float64), 1e-30))
    rows, per_vessel = [], {}

    hdr = ("anchor", "n", "k_med", "IQR/med", "corr(k,1/h)", "rho_loc", "rho_h", "dR2_h")
    print("%-12s %5s %9s %8s %12s %8s %8s %8s" % hdr)
    for anchor in WALL_COHORT_V2_TRAIN:
        pk, cf = DIR / f"{anchor}.pt", CACHE / f"{anchor}.npz"
        if not (pk.exists() and cf.exists()):
            continue
        z = np.load(cf)
        if "sr_t" not in z.files:
            continue
        d = torch.load(pk, map_location="cpu", weights_only=False)
        t, mat_gt = z["t"], z["mat"]
        j0 = j0_mat(z, bio)
        integ = np.zeros_like(j0)
        integ[1:] = np.cumsum(j0[:-1] * np.diff(t)[:, None], axis=0)
        fin_i, fin_g = integ[-1], mat_gt[-1]
        live = (fin_i > 0) & (fin_g > 0)
        if live.sum() < 25:
            continue

        A, P, h = wall_graph(z)
        k_i = np.full(len(fin_i), np.nan)
        k_i[live] = fin_g[live] / fin_i[live]
        k_med = float(np.nanmedian(k_i))
        q1, q3 = np.nanpercentile(k_i, [25, 75])

        # Does the per-node rate scalar behave like 1/h_i?
        ck = spearman(k_i[live], 1.0 / h[live])

        pred = fin_i * k_med
        y = lg(fin_g[live])
        x_loc = lg(pred[live])
        feats = {
            "h_local": lg(1.0 / h[live]),
            "upstream": lg(upstream_accum(z, d, A, fin_i)[live] + 1e-30),
            "nbr_smear": lg((P @ pred)[live] + 1e-30),
            "wall_dist": (z["pos"][live, 0] - z["pos"][live, 0].mean())
            / max(z["pos"][:, 0].std(), 1e-9),
        }
        ms = midside_nodes(z["pos"], np.asarray(z["wall_edges"]))
        feats["midside"] = ms[live].astype(np.float64) if ms.size == len(fin_i) \
            else np.zeros(int(live.sum()))

        r2_base, _ = r2_of(y, [x_loc])
        gains = {}
        for name, f in feats.items():
            r2_f, _ = r2_of(y, [x_loc, f])
            gains[name] = float(r2_f - r2_base)

        # What does the 1/h correction do to ORDERING (the 8.5 number)?
        pred_h = pred * (np.nanmedian(h) / np.maximum(h, 1e-30))
        rho_loc = spearman(pred[live], fin_g[live])
        rho_h = spearman(pred_h[live], fin_g[live])

        print("%-12s %5d %9.2f %8.3f %12.3f %8.3f %8.3f %8.3f"
              % (anchor, int(live.sum()), k_med, float((q3 - q1) / max(k_med, 1e-30)),
                 ck, rho_loc, rho_h, gains["h_local"]))
        per_vessel[anchor] = dict(n=int(live.sum()), k_med=k_med,
                                  k_iqr_rel=float((q3 - q1) / max(k_med, 1e-30)),
                                  corr_k_invh=float(ck), rho_local=float(rho_loc),
                                  rho_hcorr=float(rho_h), r2_base=float(r2_base),
                                  gains={k: float(v) for k, v in gains.items()})
        rows.append(per_vessel[anchor])

    if not rows:
        print("no vessels")
        return 1
    g = lambda k: np.array([r[k] for r in rows], dtype=float)
    print("\n=== SUMMARY over %d vessels ===" % len(rows))
    print("   implied per-node rate k = Mat_GT / integral(J0_Mat dt)")
    print("      median                     %.1f    (1/h_cell in FINDINGS 2 = 28.1)"
          % np.nanmedian(g("k_med")))
    print("      WITHIN-vessel IQR/median   %.3f" % np.nanmean(g("k_iqr_rel")))
    print("      spearman(k_i, 1/h_i)       %.3f  <- 1.0 would mean k IS the local cell size"
          % np.nanmean(g("corr_k_invh")))
    print("\n   log-R2 gain from adding each explanator to the local oracle:")
    for name in EXPLANATORS:
        v = np.array([r["gains"][name] for r in rows], dtype=float)
        print("      %-12s %+.4f   (per-vessel %+.3f .. %+.3f)"
              % (name, np.nanmean(v), np.nanmin(v), np.nanmax(v)))
    print("\n   rank correlation against GT Mat:")
    print("      local oracle               %.3f" % np.nanmean(g("rho_local")))
    print("      + 1/h_i correction         %.3f" % np.nanmean(g("rho_hcorr")))

    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(per_vessel, indent=2))
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
