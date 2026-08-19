"""PHASE 8: the off-wall arm's static attenuation is standing in for transport. How much?

``scripts/diag_fibrin_clot_route.py`` establishes that GT clot IS the level set
``{Mat >= 2e7}`` in both directions (0.0% of clot nodes below the step, 0.19% of nodes above
it not clot), and that the 583 off-wall GT clot nodes carry real ``Mat >= crit`` of their own.
So off-wall clot is not a faint echo of the wall -- ``Mat`` genuinely gets into the lumen, and
in the ``.mph`` it does so the only way it can: ``tds2`` convection, from a wall flux source.

``grow_into_lumen_by_mat`` models that transport as ONE STATIC SCALAR:

    off-wall node commits  <=>  MAT_ATTENUATION * Mat_owner >= crit        (att = 0.16)

which, for constant ``att``, is just a threshold on the owning wall node's ``Mat``. This
script separates the two off-wall errors that the Phase-7 table conflates:

    model Mat  + const att   off F1 0.023   \\  wall-Mat magnitude error
    GT Mat     + const att   off F1 0.561   /
    GT Mat     + BEST const  off F1 ?       <- ceiling of the current functional FORM
    GT Mat     + flow-aware  off F1 ?       <- is the constant hiding flow structure?

The last two are the question. If sweeping the constant to its optimum is already near the
form's ceiling, and adding flow does not move it, then off-wall is purely a wall-``Mat``
magnitude problem and the transport model is fine as a scalar. If flow moves it, then
wall-normal transport is real, and -- unlike 9's washout, which needs evolving chemistry the
deploy model does not have -- it is drivable from the **t=0** flow field the model already
predicts, so it is deployable.

    python scripts/diag_offwall_transport.py
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

from predict_wall_clot import node_pos  # noqa: E402
from src.biochem_gnn.mat_growth_simple import WALL_COHORT_V2_TRAIN  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.physics_lumen_model import (  # noqa: E402
    MAT_ATTENUATION, midside_nodes, resolve_offwall_shell, wall_normal_projection,
)
from src.core_physics.physics_wall_model import t0_flow_fields  # noqa: E402
from src.core_physics.temporal_metrics import spearman  # noqa: E402

DIR = REPO / "data/processed/graphs_biochem_anchors"
MAT_S = 7e10
CRIT = 2.0e7


def f1_pr(pred, gt):
    tp = int((pred & gt).sum())
    p = tp / max(int(pred.sum()), 1)
    r = tp / max(int(gt.sum()), 1)
    return 2 * p * r / max(p + r, 1e-9), p, r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default="outputs/phase8_offwall_transport.json")
    args = ap.parse_args()
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    packs, rows = [], []

    for anchor in WALL_COHORT_V2_TRAIN:
        p = DIR / f"{anchor}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        if int(d.y.shape[0]) < 150:
            continue
        names = d.y_channel_names.split(",")
        wall = d.mask_wall.reshape(-1).bool().numpy()
        pos, ei = node_pos(d), d.edge_index.detach().cpu().numpy()
        mat = np.expm1(d.y[-1, :, names.index("Mat_log1p_nd")].double().numpy()) * MAT_S
        shell = resolve_offwall_shell(pos, wall, ei)
        dist, owner = wall_normal_projection(pos, wall)
        # ``wall_normal_projection`` takes the NEAREST wall node, and half the wall is
        # mid-edge nodes on which 44.6% of GT Mat is a structural zero (FINDINGS 8.5).  So the
        # owner is often a node that carries no species, which inflates Mat_off/Mat_owner
        # without anything physical happening.  Restrict the owner to CORNER wall nodes -- an
        # element-order statement, not a tune.
        ms = midside_nodes(pos, ei)
        corner_wall = wall & ~ms
        _, owner_c = wall_normal_projection(pos, corner_wall)
        f = t0_flow_fields(d, bio, hops=3, flow_source="gt")
        u = d.y[0, :, 0].double().numpy()
        v = d.y[0, :, 1].double().numpy()
        spd = np.hypot(u, v)
        live = shell & (mat[owner] > 0)
        live_c = shell & (mat[owner_c] > 0)
        if live.sum() < 20 or live_c.sum() < 20:
            continue
        ratio = mat[live] / mat[owner][live]
        ratio_c = mat[live_c] / mat[owner_c][live_c]
        packs.append(dict(anchor=anchor, shell=shell, live=live, mat=mat, owner=owner,
                          owner_c=owner_c, gt_off=(mat >= CRIT) & shell, spd=spd, sr=f.sr,
                          dist=dist, ratio=ratio, ratio_c=ratio_c, live_c=live_c,
                          n_ms_wall=int((wall & ms).sum()), n_wall=int(wall.sum())))
        q = lambda a: float(np.subtract(*np.percentile(a, [75, 25])))
        rows.append(dict(anchor=anchor, n=int(live.sum()),
                         r_med=float(np.median(ratio)), r_iqr=q(ratio),
                         r_p90=float(np.percentile(ratio, 90)),
                         rc_med=float(np.median(ratio_c)), rc_iqr=q(ratio_c),
                         rc_p90=float(np.percentile(ratio_c, 90))))

    print("=== 1. THE ATTENUATION RATIO Mat_offwall / Mat_owner, measured ===")
    print("   nearest-wall owner (shipped)      |  corner-wall owner")
    print("   %-12s %6s %8s %8s %8s  | %8s %8s %8s"
          % ("anchor", "n", "median", "IQR", "p90", "median", "IQR", "p90"))
    for r in rows:
        print("   %-12s %6d %8.3f %8.3f %8.3f  | %8.3f %8.3f %8.3f"
              % (r["anchor"], r["n"], r["r_med"], r["r_iqr"], r["r_p90"],
                 r["rc_med"], r["rc_iqr"], r["rc_p90"]))
    allr = np.concatenate([p["ratio"] for p in packs])
    allc = np.concatenate([p["ratio_c"] for p in packs])
    q = lambda a: float(np.subtract(*np.percentile(a, [75, 25])))
    print("   pooled  nearest: median %.3f IQR %.3f   corner: median %.3f IQR %.3f"
          % (np.median(allr), q(allr), np.median(allc), q(allc)))
    print("   per-vessel median spread  nearest %.3f-%.3f   corner %.3f-%.3f"
          % (min(r["r_med"] for r in rows), max(r["r_med"] for r in rows),
             min(r["rc_med"] for r in rows), max(r["rc_med"] for r in rows)))
    print("   (shipped MAT_ATTENUATION = %.2f, a single cohort constant)" % MAT_ATTENUATION)

    # === 2. Ceiling of the current FORM: the best single threshold on the owner's Mat. ===
    print("\n=== 2. CEILING OF A CONSTANT ATTENUATION (GT wall Mat, sweep the constant) ===")
    atts = np.concatenate([[MAT_ATTENUATION], np.logspace(-2, 0, 41)])
    curves = {}
    for okey in ("owner", "owner_c"):
        best_o, curve = None, []
        for att in atts:
            fs, ps, rs = [], [], []
            for pk in packs:
                pred = pk["shell"] & (att * pk["mat"][pk[okey]] >= CRIT)
                a, b, c = f1_pr(pred, pk["gt_off"])
                fs.append(a), ps.append(b), rs.append(c)
            row = dict(att=float(att), f1=float(np.mean(fs)), prec=float(np.mean(ps)),
                       rec=float(np.mean(rs)))
            curve.append(row)
            if best_o is None or row["f1"] > best_o["f1"]:
                best_o = row
        curves[okey] = dict(curve=curve, best=best_o,
                            ship=[c for c in curve if c["att"] == MAT_ATTENUATION][0])
    for okey, lbl in (("owner", "nearest-wall owner (shipped)"),
                      ("owner_c", "corner-wall owner")):
        s, bo = curves[okey]["ship"], curves[okey]["best"]
        print("   %-30s att=%.2f  off F1 %.4f  P %.3f R %.3f"
              % (lbl, s["att"], s["f1"], s["prec"], s["rec"]))
        print("   %-30s att=%.4f off F1 %.4f  P %.3f R %.3f   <- best constant"
              % ("", bo["att"], bo["f1"], bo["prec"], bo["rec"]))
    best = curves["owner"]["best"]
    ship = curves["owner"]["ship"]

    # === 3. Does the ratio carry flow structure the constant throws away? ===
    print("\n=== 3. WHAT PREDICTS THE PER-NODE RATIO (spearman, per vessel then mean) ===")
    # ``spd`` at a wall node is 0 by no-slip, so ``spd_owner`` is a constant and its rank
    # correlation is undefined -- that is a fact about the boundary condition, not a gap in
    # the data, so it is dropped rather than reported as a NaN.
    feats = ("sr_owner", "spd_local", "dist", "mat_owner")
    acc = {k: [] for k in feats}
    for pk in packs:
        lv, ow = pk["live"], pk["owner"]
        cand = dict(sr_owner=pk["sr"][ow][lv], spd_local=pk["spd"][lv],
                    dist=pk["dist"][lv], mat_owner=pk["mat"][ow][lv])
        for k in feats:
            acc[k].append(spearman(cand[k], pk["ratio"]))
    for k in feats:
        a = np.array(acc[k], dtype=float)
        print("   %-12s %+.3f   (per-vessel %+.3f .. %+.3f)"
              % (k, np.nanmean(a), np.nanmin(a), np.nanmax(a)))

    # === 4. Would a flow-aware attenuation beat the best constant? ===
    # Keep the same functional form -- a threshold on the owner's Mat -- but let the
    # threshold be modulated by the owner's shear.  One extra exponent, fitted globally.
    print("\n=== 4. FLOW-AWARE ATTENUATION: att = a * (sr_owner / sr_ref)^b ===")
    print("   %8s %10s %10s %10s" % ("b", "best a", "off F1", "vs const"))
    grid_b = [0.0, -0.1, -0.25, -0.5, -0.75, -1.0, 0.25, 0.5]
    best_fa = None
    for b in grid_b:
        inner = None
        for a in np.logspace(-2, 0.3, 31):
            fs = []
            for pk in packs:
                ow = pk["owner"]
                sr_ref = np.median(pk["sr"][pk["sr"] > 0]) if (pk["sr"] > 0).any() else 1.0
                att = a * np.power(np.maximum(pk["sr"][ow], 1e-3) / sr_ref, b)
                pred = pk["shell"] & (att * pk["mat"][ow] >= CRIT)
                fs.append(f1_pr(pred, pk["gt_off"])[0])
            v = float(np.mean(fs))
            if inner is None or v > inner[1]:
                inner = (float(a), v)
        print("   %8.2f %10.4f %10.4f %+10.4f"
              % (b, inner[0], inner[1], inner[1] - best["f1"]))
        if best_fa is None or inner[1] > best_fa[2]:
            best_fa = (b, inner[0], inner[1])

    print("\n=== SUMMARY: off-wall F1, GT wall Mat throughout, %d vessels ===" % len(packs))
    print("      shipped: nearest owner, att=0.16     %.4f" % ship["f1"])
    print("      best constant, nearest owner         %.4f" % best["f1"])
    print("      best constant, CORNER owner          %.4f" % curves["owner_c"]["best"]["f1"])
    print("      best flow-aware (b=%.2f)              %.4f" % (best_fa[0], best_fa[2]))
    print("      flow-aware gain over best constant   %+.4f" % (best_fa[2] - best["f1"]))
    print("      corner-owner gain over best constant %+.4f"
          % (curves["owner_c"]["best"]["f1"] - best["f1"]))

    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        ratio_rows=rows, best_const=best, best_const_corner=curves["owner_c"]["best"],
        att_curve=curves["owner"]["curve"], att_curve_corner=curves["owner_c"]["curve"],
        best_flow=dict(b=best_fa[0], a=best_fa[1], f1=best_fa[2]),
        ratio_corr={k: float(np.nanmean(acc[k])) for k in feats}), indent=2))
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
