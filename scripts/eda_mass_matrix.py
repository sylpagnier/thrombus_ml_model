"""EDA: is the missing non-locality in wall `Mat` a FEM MASS MATRIX, not transport?

PHASE7 9.2 found that a per-node ODE fed a PERFECT oracle (GT RP/AP/M/Mas/sr/dsrx at every
step) still ranks final GT `Mat` at only 0.310, and concluded the equation is missing
non-local transport.  That diagnosis is suspicious: at the wall `u = 0` (no-slip) and
`D_Mat = 0`, so there is no physical mechanism to move `Mat` between wall nodes.

But COMSOL's TDS uses a **consistent mass matrix** and Do Carmo-Galeao crosswind
stabilisation on **quadratic** elements.  The semi-discrete system is then

    M . dMat/dt = f        NOT     dMat/dt = f / h

and `M` has positive off-diagonals, so each node's rate is coupled to its neighbours'.
That is a *numerical* non-locality -- a fixed local linear operator -- not a transport
closure.  It would also explain why the off-wall shell attenuation is so uncannily
constant across vessels: a shape-function ratio, not a transport coefficient.

THE TEST, in the FORWARD direction so it is linear in the unknowns and needs no inverse:

    local (current model)   f_i  ~  a * Mat_i
    mass matrix             f_i  ~  a * Mat_i + b * (A Mat)_i

where `f_i` is the time-integrated local flux and `A` is the row-normalised WALL adjacency.
If `b` is significantly non-zero, consistent in sign across vessels, and lifts R2, the
coupling is real.  The inverse direction is reported too: does smoothing the flux integral
improve its RANK correlation with GT `Mat` -- which is the quantity that actually limits us
(`rho_corner` = 0.193).

    python scripts/eda_mass_matrix.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from diag_local_ode_closure import comsol_j0  # noqa: E402
from src.clot_ml.geometry_splits import eligible_pool  # noqa: E402
from src.config import BiochemConfig  # noqa: E402
from src.core_physics.temporal_metrics import spearman  # noqa: E402

CACHE = REPO / "outputs/wall_species_cache"


def wall_adjacency(wall_edges, wall_idx):
    """Row-normalised adjacency on the WALL subgraph.

    ``wall_edges`` is already in LOCAL wall indexing (0..W-1) -- verified against the cache,
    where it spans 0..560 for 561 wall nodes while ``wall_idx`` holds the global ids.
    Remapping through ``wall_idx`` produces an EMPTY graph and silently makes every mixing
    term zero, which is how the first run of this script read b/a = 0.000 on 16 of 18
    vessels.
    """
    n = len(wall_idx)
    src, dst = wall_edges[0].astype(int), wall_edges[1].astype(int)
    keep = (src >= 0) & (src < n) & (dst >= 0) & (dst < n)
    src, dst = src[keep], dst[keep]
    if not len(src):
        return sp.csr_matrix((n, n))
    A = sp.coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.float64)
    A.setdiag(0.0)
    A.eliminate_zeros()
    deg = np.asarray(A.sum(axis=1)).reshape(-1)
    deg[deg == 0] = 1.0
    return sp.diags(1.0 / deg) @ A


def r2(y, X):
    """R2 of an OLS fit of y on X (no intercept -- both sides vanish together)."""
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss = float(((y - pred) ** 2).sum())
    st = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss / max(st, 1e-30), coef


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default="outputs/eda_mass_matrix.json")
    args = ap.parse_args()
    bio = BiochemConfig(phase="biochem")
    pool = [a for a in eligible_pool() if (CACHE / f"{a}.npz").exists()]

    print("FORWARD  f ~ a*Mat + b*(A Mat)      |  INVERSE rank(Mat) vs smoothed flux")
    print("%-11s %6s | %7s %7s %7s | %7s %7s %7s %7s"
          % ("vessel", "n", "R2 loc", "R2 mix", "b/a", "rho I", "rho AI", "rho A2I", "rho mix"))
    rows = {}
    for a in pool:
        z = np.load(CACHE / f"{a}.npz")
        if "sr_t" not in z.files:
            continue
        t = z["t"]
        j0 = comsol_j0(z, bio)                       # [T, W] oracle local flux
        integ = np.zeros_like(j0)
        integ[1:] = np.cumsum(j0[:-1] * np.diff(t)[:, None], axis=0)
        f = integ[-1]
        mat = z["mat"][-1]
        A = wall_adjacency(z["wall_edges"], z["wall_idx"])
        live = (f > 0) & (mat > 0)
        if live.sum() < 40:
            continue
        Am = A @ mat
        # forward: does the neighbour term earn its place?
        r_loc, _ = r2(f[live], mat[live][:, None])
        r_mix, c = r2(f[live], np.stack([mat[live], Am[live]], 1))
        ba = float(c[1] / c[0]) if abs(c[0]) > 1e-30 else np.nan
        # inverse: does smoothing the flux integral improve its RANK vs GT Mat?
        AI, A2I = A @ f, A @ (A @ f)
        rho_i = spearman(f[live], mat[live])
        rho_ai = spearman(AI[live], mat[live])
        rho_a2 = spearman(A2I[live], mat[live])
        best = max(
            (spearman(((1 - w) * f + w * AI)[live], mat[live]), w)
            for w in np.linspace(0, 1, 21))
        rows[a] = dict(n=int(live.sum()), r2_local=r_loc, r2_mix=r_mix, b_over_a=ba,
                       rho_local=rho_i, rho_A=rho_ai, rho_A2=rho_a2,
                       rho_best=best[0], w_best=best[1])
        print("%-11s %6d | %7.3f %7.3f %7.3f | %7.3f %7.3f %7.3f %7.3f"
              % (a, int(live.sum()), r_loc, r_mix, ba, rho_i, rho_ai, rho_a2, best[0]))

    v = lambda k: np.array([r[k] for r in rows.values()], float)
    print("\nMEAN        %6s | %7.3f %7.3f %7.3f | %7.3f %7.3f %7.3f %7.3f"
          % ("", v("r2_local").mean(), v("r2_mix").mean(), np.nanmean(v("b_over_a")),
             v("rho_local").mean(), v("rho_A").mean(), v("rho_A2").mean(),
             v("rho_best").mean()))
    print("\nb/a sign consistency: %d of %d vessels positive"
          % (int((v("b_over_a") > 0).sum()), len(rows)))
    print("mean optimal smoothing weight w: %.2f (0 = pure local, 1 = pure 1-hop mean)"
          % np.nanmean(v("w_best")))
    print("rank gain from smoothing: %+.3f" % (v("rho_best").mean() - v("rho_local").mean()))

    Path(args.save).write_text(json.dumps(rows, indent=2, default=float))
    print("\nwrote %s" % args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
