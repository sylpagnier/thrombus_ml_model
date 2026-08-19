"""PHASE 8: can a PER-NODE surface ODE reproduce GT Mat at all, given perfect inputs?

``integrate_mat_trajectory`` integrates every wall node independently.  COMSOL's ``Mat`` is
not a surface state -- it is a *Transport of Diluted Species* DOMAIN field with ``D = 0``,
advected by the Reacting Flow coupling and sourced by the wall flux ``J0_Mat``
(docs/PHASE7_FINDINGS.md 1.1).  A domain field with a wall source and tangential flow moves
material DOWNSTREAM along the wall; an independent per-node ODE cannot represent that at all.

That would show up as exactly the two symptoms Phase 7 measured and could not explain:

    d(Mat,t)/J0_Mat = 145.6  against  1/h = 28.1   (FINDINGS 2, "5.2x, one term")
    spearman(model Mat, GT Mat) = 0.193 on corner wall nodes   (FINDINGS 8.5)

both of which are what you get if a node's Mat is set partly by what happened UPSTREAM of it
rather than only by its own flux.

THE TEST.  Give the per-node ODE a perfect oracle -- GT ``RP``, ``AP``, ``M``, ``Mas``,
``sr`` and ``d(sr,x)`` at every timestep, i.e. every input it could ever want -- and
integrate COMSOL's own ``J0_Mat`` expression locally.  Then compare with GT ``Mat``.

    residual is small      -> the structure is right and Phase 7's problem is input error
    residual is large      -> the per-node ODE is structurally wrong, and the missing term
                              is non-local

Reported per vessel: the implied per-node rate scalar (which should be the constant ``1/h``
if the local balance closes), its spatial spread, the oracle ODE's rank correlation against
GT Mat, and whether the residual is organised along the flow direction.

    python scripts/diag_local_ode_closure.py
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
from src.config import BiochemConfig  # noqa: E402
from src.core_physics.physics_wall_model import gate_from_shear  # noqa: E402
from src.core_physics.temporal_metrics import spearman  # noqa: E402

DIR = REPO / "data/processed/graphs_biochem_anchors"
CACHE = REPO / "outputs/wall_species_cache"
M_TO_CM = 100.0
PER_M2_TO_PER_CM2 = 1.0e-4


def comsol_j0(z, bio, *, t_idx=None):
    """COMSOL's J0_Mat / Da, evaluated on GT fields.  [T, W] in CGS.

    J0_Mat = Da * gate * (Sat(M)*k_rs*RP + Sat(M)*k_as*AP + (Mas/M_inf)*k_aa*AP) * step2t(t)

    with ``gate = A + B``, ``A = (L/gamma_m)*|d(sr,x)|`` where ``d(sr,x) < sgt`` and ``B = 1``
    where ``sr < lss``.  Both branches can be live at once -- the .mph adds them, it does not
    choose.  ``step2t`` is the function tagged ``step4`` (location 12 s, width 2.5 s), which
    against a 150 s sampling interval is 1 at every stored timestep.
    """
    k_rs = float(bio.k_rs) * M_TO_CM
    k_as = float(bio.k_as) * M_TO_CM
    k_aa = float(bio.k_aa) * M_TO_CM
    minf = float(bio.Minf) * PER_M2_TO_PER_CM2

    gate = gate_from_shear(z["sr_t"], z["dsrx_t"], bio)
    sat = np.clip(1.0 - z["m_tot"] / minf, 0.0, 1.0)
    chem = sat * (k_rs * z["rp"] + k_as * z["ap"]) + (z["mas"] / minf) * k_aa * z["ap"]
    return gate * chem


def wall_tangent_flow(d, widx):
    """Near-wall tangential speed at each wall node, from GT velocity.  Oracle diagnostic."""
    u = d.y[:, :, 0].double().numpy()[:, widx]
    v = d.y[:, :, 1].double().numpy()[:, widx]
    return np.hypot(u, v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default="outputs/phase8_local_ode_closure.json")
    args = ap.parse_args()
    bio = BiochemConfig(phase="biochem")
    crit = float(bio.viscosity_mat_crit)
    rows, per_vessel = [], {}

    print("%-12s %8s %10s %10s %8s %8s %8s"
          % ("anchor", "n_wall", "k_med", "k_iqr/med", "rho", "rho_hot", "r2_log"))
    for anchor in WALL_COHORT_V2_TRAIN:
        pk, cf = DIR / f"{anchor}.pt", CACHE / f"{anchor}.npz"
        if not pk.exists() or not cf.exists():
            continue
        z = np.load(cf)
        if "sr_t" not in z.files:
            continue
        d = torch.load(pk, map_location="cpu", weights_only=False)
        if int(d.y.shape[0]) < 150:
            continue
        t = z["t"]
        j0 = comsol_j0(z, bio)
        # Local balance: dMat/dt = k * J0 with ONE scalar k (= 1/h if the balance closes).
        # Integrate J0 in time, then k_i is the per-node ratio to GT Mat at the end.
        integ = np.zeros_like(j0)
        integ[1:] = np.cumsum(j0[:-1] * np.diff(t)[:, None], axis=0)
        mat_gt = z["mat"]
        fin_i, fin_g = integ[-1], mat_gt[-1]
        live = fin_i > 0
        if live.sum() < 20:
            continue
        k = fin_g[live] / fin_i[live]
        k_med = float(np.median(k))
        q1, q3 = np.percentile(k, [25, 75])
        # The oracle ODE's own prediction: one global scalar, the cohort's best case.
        pred = integ[-1] * k_med
        hot = fin_g >= crit
        rho = spearman(pred, fin_g)
        rho_hot = spearman(pred[hot], fin_g[hot]) if hot.sum() > 8 else float("nan")
        lg = lambda a: np.log10(np.maximum(a, 1e-30))
        ss = float(((lg(pred[live]) - lg(fin_g[live])) ** 2).sum())
        st = float(((lg(fin_g[live]) - lg(fin_g[live]).mean()) ** 2).sum())
        r2 = 1.0 - ss / max(st, 1e-30)
        print("%-12s %8d %10.2f %10.3f %8.3f %8.3f %8.3f"
              % (anchor, int(live.sum()), k_med, float((q3 - q1) / max(k_med, 1e-30)),
                 rho, rho_hot, r2))
        per_vessel[anchor] = dict(n_wall=int(live.sum()), k_med=k_med,
                                  k_iqr_rel=float((q3 - q1) / max(k_med, 1e-30)),
                                  rho=float(rho), rho_hot=float(rho_hot), r2_log=float(r2))
        rows.append(per_vessel[anchor])

    if not rows:
        print("no vessels")
        return 1
    g = lambda k: np.array([r[k] for r in rows], dtype=float)
    print("\n=== SUMMARY: %d vessels ===" % len(rows))
    print("   implied rate scalar k = Mat_GT / integral(J0_Mat dt)")
    print("      median over vessels        %.2f      (1/h from FINDINGS 2 = 28.1)" %
          np.median(g("k_med")))
    print("      per-vessel spread          %.2f - %.2f" % (g("k_med").min(), g("k_med").max()))
    print("      WITHIN-vessel IQR/median   %.3f  <- 0 would mean one scalar closes it"
          % np.nanmean(g("k_iqr_rel")))
    print("   oracle per-node ODE vs GT Mat (perfect RP/AP/M/Mas/sr/dsrx):")
    print("      spearman, all wall nodes   %.3f" % np.nanmean(g("rho")))
    print("      spearman, GT-committed     %.3f" % np.nanmean(g("rho_hot")))
    print("      R2 in log10                %.3f" % np.nanmean(g("r2_log")))

    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(per_vessel, indent=2))
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
