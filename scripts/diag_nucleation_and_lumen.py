"""Two questions the wall-masked numbers cannot answer.

A. WHY DOES A t=0 SNAPSHOT WORK AT ALL, and where could autoregression help?
   The gates are ``sr < lss`` (stagnation) and ``dsrx < sgt`` (separation) -- both properties
   of the GEOMETRY's flow, not of the clot.  If clot forms where the flow is already
   pathological at t=0, a static rule is the right model and a rollout adds nothing.  If a
   large share of GT commits at nodes that were NOT gated at t=0 and only opened later, that
   share is exactly what autoregression is for.  This splits GT's committed nodes into
   NUCLEATION (gated at t=0) and CREEP (gated only later) and reports when each ignites.

B. DOES ANY OF THIS SURVIVE OFF THE WALL?
   Every score in PHASE6 passed ``wall_mask=wall``, so off-wall clot has been invisible
   throughout.  The repo has a lumen arm (``physics_lumen_model.grow_into_lumen``, wired as
   ``predict_wall_clot(..., lumen=True)``).  This measures the off-wall GT burden and scores
   wall-only vs wall+lumen on the FULL mesh, which is the number a user would actually see.

SEALED is not opened.

    python scripts/diag_nucleation_and_lumen.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.species_pushforward_continuous import (  # noqa: E402
    resolve_deploy_eval_time_index,
)
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.evaluation.clot_relaxed_metrics import (  # noqa: E402
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
)

sys.path.insert(0, str(REPO / "scripts"))
from predict_wall_clot import predict_wall_clot  # noqa: E402

DIR = Path("data/processed/graphs_biochem_anchors")
CACHE = Path("outputs/wall_species_cache")
OUT = Path("outputs/rollout_trackA")
M_TO_CM = 100.0


def main() -> int:
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    prot = json.load(open("outputs/ap_closure/protocol_gt_meanovertime.json"))
    names = prot["fit"] + prot["dev"]
    lss, sgt = float(bio.lss), float(bio.sgt) / M_TO_CM
    coef = float(bio.L_char) * M_TO_CM / float(bio.gamma_m)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ------------------------------------------------------- A. nucleation vs creep
    print("=" * 88)
    print("A. OF GT's COMMITTED WALL NODES, HOW MANY WERE ALREADY GATED AT t=0?")
    print("=" * 88)
    print("%-12s %6s | %8s %8s %8s | %9s %9s"
          % ("vessel", "n_gt", "nucl%", "creep%", "never%", "t_nucl", "t_creep"))
    rowsA = []
    for n in names:
        p = CACHE / f"{n}.npz"
        if not p.exists() or "sr_t" not in np.load(p).files:
            continue
        z = np.load(p)
        gt = z["gt_onset"]
        g = gt >= 0
        if not g.any():
            continue
        nt = len(z["t"])
        gate_t = (z["dsrx_t"] < sgt) * coef * np.abs(z["dsrx_t"]) + (z["sr_t"] < lss)
        open0 = gate_t[0] > 0
        ever = (gate_t > 0).any(axis=0)
        nucl = g & open0
        creep = g & ~open0 & ever
        never = g & ~ever
        f = gt[g].astype(float) / (nt - 1)
        rowsA.append(dict(name=n, n_gt=int(g.sum()), nucl=float(nucl.sum() / g.sum()),
                          creep=float(creep.sum() / g.sum()), never=float(never.sum() / g.sum()),
                          t_nucl=float(np.median(gt[nucl] / (nt - 1))) if nucl.any() else np.nan,
                          t_creep=float(np.median(gt[creep] / (nt - 1))) if creep.any() else np.nan))
        r = rowsA[-1]
        print("%-12s %6d | %7.0f%% %7.0f%% %7.0f%% | %9.2f %9.2f"
              % (n, r["n_gt"], 100 * r["nucl"], 100 * r["creep"], 100 * r["never"],
                 r["t_nucl"], r["t_creep"]))
    for k in ("nucl", "creep", "never"):
        v = np.array([r[k] for r in rowsA])
        print("   mean %-6s %.0f%%" % (k, 100 * v.mean()))
    tn = np.nanmean([r["t_nucl"] for r in rowsA])
    tc = np.nanmean([r["t_creep"] for r in rowsA])
    print("   median onset:  nucleation t/T %.2f   creep t/T %.2f   (creep is %+.2f later)"
          % (tn, tc, tc - tn))

    # --------------------------------------------------------------- B. the lumen arm
    print("\n" + "=" * 88)
    print("B. OFF-WALL CLOT: WHAT THE WALL-MASKED SCORES HAVE BEEN HIDING")
    print("=" * 88)
    print("%-12s %7s %8s | %9s %9s | %9s %9s"
          % ("vessel", "gt_wall", "gt_off", "wall-only", "wall+lum", "n_pred_w", "n_pred_l"))
    rowsB = []
    for n in names:
        p = DIR / f"{n}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        te = resolve_deploy_eval_time_index(int(d.y.shape[0]))
        gt = gt_clot_phi_at_time(d, te, phys, device=torch.device("cpu")).reshape(-1)
        w = d.mask_wall.reshape(-1).bool()
        gtb = gt > 0.5
        n_w, n_off = int((gtb & w).sum()), int((gtb & ~w).sum())
        if n_w + n_off == 0:
            continue
        pw, _ = predict_wall_clot(d, bio, flow="gt", lumen=False)
        pl, _ = predict_wall_clot(d, bio, flow="gt", lumen=True)

        def score(pred):
            m = compute_clot_relaxed_metrics(torch.tensor(pred.astype(np.float32)), gt,
                                             d.edge_index)          # FULL mesh, no wall mask
            return clot_score_from_deploy_dict(metrics_to_deploy_prefix(m))

        s_w, s_l = score(pw), score(pl)
        rowsB.append(dict(name=n, gt_wall=n_w, gt_off=n_off,
                          off_frac=n_off / max(n_w + n_off, 1),
                          wall_only=s_w, wall_lumen=s_l,
                          n_pred_w=int(pw.sum()), n_pred_l=int(pl.sum())))
        print("%-12s %7d %8d | %9.4f %9.4f | %9d %9d"
              % (n, n_w, n_off, s_w, s_l, int(pw.sum()), int(pl.sum())))
    off = np.array([r["off_frac"] for r in rowsB])
    sw = np.array([r["wall_only"] for r in rowsB])
    sl = np.array([r["wall_lumen"] for r in rowsB])
    print("\n   GT clot that is OFF-WALL: mean %.0f%%  median %.0f%%  max %.0f%%"
          % (100 * off.mean(), 100 * np.median(off), 100 * off.max()))
    print("   FULL-MESH deploy score   wall-only %.4f   wall+lumen %.4f   (%+.4f)"
          % (sw.mean(), sl.mean(), sl.mean() - sw.mean()))
    print("   for contrast, the WALL-MASKED score this project reports is ~0.91 on arm A.")
    print("   %d/%d vessels improve with the lumen arm" % (int((sl > sw).sum()), len(sl)))

    (OUT / "nucleation_and_lumen.json").write_text(json.dumps(
        dict(nucleation=rowsA, lumen=rowsB), indent=2, default=float), encoding="utf-8")
    print("\nwrote %s   (%.0fs)" % (OUT / "nucleation_and_lumen.json", time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
