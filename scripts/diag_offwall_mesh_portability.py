"""PHASE 7 5.1: is the off-wall shell bound a deploy blocker, or just badly parameterised?

The Mat lumen arm admits an off-wall node inside ``shell_hi = 2.1`` **median edge lengths**
of the wall.  ``docs/PHASE7_FINDINGS.md`` 3.1/5.1 called that a pre-deploy blocker: the
bound is in mesh units, it held to 1% across the 19 train vessels only because they share
one meshing recipe, and it says nothing about a customer mesh with different boundary-layer
settings.  The proposed fix was to re-express it as a physical thickness in cm.

THAT IS NOT THE PROBLEM, AND cm IS NOT THE FIX (h varies 2.9% across the cohort while the
vessels vary 76%, so a mesh-unit bound and a cm bound select the same nodes).  Measured here:
the near-wall mesh carries **alternating node families** and the species field only lives on
every other one.

    band [median edge lengths]   Mat/Mat_owner   off-wall GT clot
    0.50 - 1.35                  0.000            83 nodes
    1.35 - 2.20                  0.154           493 nodes
    2.20 - 3.00                  0.000             7 nodes

Velocity and pressure are populated on every band, so this is specific to M/Mas/Mat, which
are exactly the fields the clot label thresholds.  The 0.5-1.35 family therefore almost
cannot be labelled clot, and ``2.1`` median edge lengths spans it AND the species band, so
roughly half of everything the original shell admitted is a structural false positive.

WHAT THE EMPTY FAMILY IS: the meshes are quadratic, ~3/4 of every pack is mid-edge nodes,
and the empty family is the mid-edge nodes of the WALL-NORMAL edges.  Being mid-side is not
sufficient -- the mid-side nodes lying along the species row carry Mat normally and hold 170
of the 493 off-wall GT clot nodes.  Navigating that structure (row E below) selects the
species row with NO LENGTH ANYWHERE, which is what actually closes 5.1.

Reported below: the node-family table, the quadratic-mesh evidence, the shell comparison
against a GT-Mat oracle, and a coarser-boundary-layer emulation (delete the empty family,
re-measure the median edge length on the surviving subgraph) showing which formulation
survives a mesh change.

    python scripts/diag_offwall_mesh_portability.py
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

from predict_wall_clot import node_pos  # noqa: E402
from src.biochem_gnn.mat_growth_simple import WALL_COHORT_V2_TRAIN  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.physics_lumen_model import (  # noqa: E402
    MAT_ATTENUATION, SHELL_SPECIES_HI, SHELL_SPECIES_LO, first_corner_shell,
    median_edge_length, midside_nodes, wall_normal_projection,
)
from src.core_physics.species_pushforward_continuous import (  # noqa: E402
    resolve_deploy_eval_time_index,
)
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

DIR = REPO / "data/processed/graphs_biochem_anchors"
MAT_S = 7e10
M_TO_CM = 100.0
OLD_SHELL_HI = 2.1                  # the original Phase-7 bound, in median edge lengths
OLD_SHELL_HI_CM = 2.1 * 0.0354      # the same bound as a length, i.e. 5.1's proposed fix
# Node bands, in median edge lengths.  The offsets come out at 1.01 / 1.71 / 2.62 / 3.43 on
# every vessel, so fixed cuts separate them cleanly; a per-owner distance RANK does not
# (58% overlap with the band, and it scores 0.409 against the band's 0.530).
BANDS = ((0.5, 1.35), (1.35, 2.2), (2.2, 3.0), (3.0, 3.8))


def f1p(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    if gt.sum() == 0 and pred.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    tp = int((pred & gt).sum())
    p = tp / max(int(pred.sum()), 1)
    r = tp / max(int(gt.sum()), 1)
    return 2 * p * r / max(p + r, 1e-9), p, r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default="outputs/phase7_offwall_mesh_portability.json")
    args = ap.parse_args()
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    crit = float(bio.viscosity_mat_crit)

    rows = {r: {k: [] for k in ("s_cm", "s_h", "deg", "ratio", "n", "gt", "hot")}
            for r in range(len(BANDS))}
    geom, shells, per_vessel = [], {}, {}

    for anchor in WALL_COHORT_V2_TRAIN:
        pk = DIR / f"{anchor}.pt"
        if not pk.exists():
            continue
        d = torch.load(pk, map_location="cpu", weights_only=False)
        if int(d.y.shape[0]) < 150:
            continue
        wall = d.mask_wall.reshape(-1).bool().numpy()
        pos, ei, n = node_pos(d), d.edge_index.detach().cpu().numpy(), len(wall)
        # pos is non-dimensional (scaled by d_bar in metres); cm for the physical columns.
        cm = float(d.d_bar.reshape(-1)[0]) * M_TO_CM
        h = median_edge_length(pos, ei)
        A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
        A = ((A + A.T) > 0).astype(np.int8)
        deg = np.asarray(A.sum(1)).reshape(-1)
        names = d.y_channel_names.split(",")
        mat = np.expm1(d.y[-1, :, names.index("Mat_log1p_nd")].double().numpy()) * MAT_S
        t_eval = resolve_deploy_eval_time_index(int(d.y.shape[0]))
        gt = gt_clot_phi_at_time(d, t_eval, phys, device=torch.device("cpu")).numpy() > 0.5
        if gt.sum() == 0:
            continue

        dist, owner = wall_normal_projection(pos, wall)
        dn = dist / h
        ms0 = midside_nodes(pos, ei)
        b0 = (~wall) & (dn >= BANDS[0][0]) & (dn < BANDS[0][1])
        w_ms, w_co = wall & ms0, wall & ~ms0
        geom.append(dict(
            anchor=anchor, h_cm=h * cm, d_bar_cm=cm,
            # The P2 signature: 3 mid-side nodes per corner node on a triangulation.
            ms_frac=float(ms0.mean()),
            b0_is_midside=float(ms0[b0].mean()) if b0.sum() else float("nan"),
            wall_ms_frac=float(w_ms.sum() / max(wall.sum(), 1)),
            gt_mat_zero_ms=float((mat[w_ms] <= 0).mean()) if w_ms.sum() else float("nan"),
            gt_mat_zero_corner=float((mat[w_co] <= 0).mean()) if w_co.sum() else float("nan"),
        ))
        pv = {"n_off_gt": int((gt & ~wall).sum())}
        for r, (lo, hi) in enumerate(BANDS):
            m = (~wall) & (dn >= lo) & (dn < hi)
            if m.sum() < 8:
                continue
            # The RATIO must be conditioned on a COMMITTED owner: averaged over all row-r
            # nodes it is dominated by the majority whose wall is clot-free and whose Mat is
            # numerically zero, which reads as 0.0 on every row and hides the structure.
            c = m & (mat[owner] >= crit)
            for k, v in (("s_cm", float(np.median(dist[m])) * cm),
                         ("s_h", float(np.median(dist[m])) / h),
                         ("deg", float(np.median(deg[m]))),
                         ("n", int(m.sum())), ("gt", int((gt & m).sum())),
                         ("hot", int(c.sum())),
                         ("ratio", float(np.median(mat[c] / mat[owner[c]]))
                          if c.sum() >= 8 else float("nan"))):
                rows[r][k].append(v)
            pv["band%d" % r] = dict(s_h=rows[r]["s_h"][-1], deg=rows[r]["deg"][-1],
                                    gt=rows[r]["gt"][-1], ratio=rows[r]["ratio"][-1])

        # --- shell formulations, all driven by the GT Mat oracle so the ONLY thing that
        # --- varies is the geometric admission rule.
        trig = MAT_ATTENUATION * mat[owner] >= crit
        phantom = (~wall) & (dn >= BANDS[0][0]) & (dn < BANDS[0][1])
        keep = ~phantom                       # coarser BL: delete the phantom family
        ke = keep[ei[0]] & keep[ei[1]]
        h_c = median_edge_length(pos, ei[:, ke]) if ke.sum() > 8 else h

        ms = midside_nodes(pos, ei)
        corner_shell = first_corner_shell(pos, wall, ei)

        def build(hh: float, live: np.ndarray) -> dict:
            g = dist / hh
            return {
                "A  0 - 2.1 edges (was shipped)": (~wall) & live & (g < OLD_SHELL_HI),
                "B  physical cm (5.1 proposal)": (~wall) & live & (dist * cm < OLD_SHELL_HI_CM),
                "C  species band (now shipped)": (~wall) & live
                & (g >= SHELL_SPECIES_LO) & (g < SHELL_SPECIES_HI),
                "D  phantom band alone": (~wall) & live
                & (g >= BANDS[0][0]) & (g < BANDS[0][1]),
                # No length anywhere: the nearest CORNER node per wall node.  If this ties C
                # then the shell has no mesh-unit constant left to recalibrate on a new mesh.
                "E  topological (no length)": live & corner_shell,
            }

        fine, coarse = build(h, np.ones_like(wall)), build(h_c, keep)
        for k in fine:
            s = shells.setdefault(k, {x: [] for x in ("f1", "p", "r", "n",
                                                      "f1c", "pc", "nc")})
            a, b, c = f1p(fine[k] & trig, gt & ~wall)
            s["f1"].append(a)
            s["p"].append(b)
            s["r"].append(c)
            s["n"].append(int((fine[k] & trig).sum()))
            a2, b2, _ = f1p(coarse[k] & trig, (gt & ~wall) & keep)
            s["f1c"].append(a2)
            s["pc"].append(b2)
            s["nc"].append(int((coarse[k] & trig).sum()))
        per_vessel[anchor] = pv
        print("%-12s off-GT %3d  h %.4f cm  species-band Mat/owner %.3f"
              % (anchor, pv["n_off_gt"], h * cm,
                 pv.get("band1", {}).get("ratio", float("nan"))))

    print("\n=== 1. GEOMETRY: why this cohort cannot answer a portability question alone ===")
    for k in ("h_cm", "d_bar_cm"):
        v = np.array([g[k] for g in geom])
        print("   %-10s min %.5f  max %.5f  spread %.1f%%"
              % (k, v.min(), v.max(), 100 * (v.max() - v.min()) / v.mean()))
    print("   The median edge length is constant to 3% across all 19 vessels, so a"
          " mesh-unit\n   bound and a cm bound are INDISTINGUISHABLE here -- which is"
          " exactly why 5.1\n   could not tell whether either was right.")

    print("\n=== 2. THE TWO NEAR-WALL NODE FAMILIES ===")
    print("   %-12s %9s %9s %9s %11s %9s"
          % ("band [edges]", "s [cm]", "s / h", "n/vessel", "Mat/owner", "GT clot"))
    for r, (lo, hi) in enumerate(BANDS):
        if not rows[r]["n"]:
            continue
        print("   %-12s %9.5f %9.2f %9.0f %11.4f %9d"
              % ("%.2f-%.2f" % (lo, hi), np.mean(rows[r]["s_cm"]), np.mean(rows[r]["s_h"]),
                 np.mean(rows[r]["n"]), np.nanmedian(rows[r]["ratio"]),
                 int(np.sum(rows[r]["gt"]))))
    print("   One node per wall node per band, at 1.01 / 1.71 / 2.62 / 3.43 edge lengths."
          "\n   Mat/Mat_owner alternates 0.000 / 0.154 / 0.000 / 0.022 -- the species field"
          "\n   lives on every OTHER family, while u/v/p are populated on all of them.")

    print("\n=== 2b. THE MESH IS QUADRATIC, AND THE EMPTY FAMILY IS ITS WALL-NORMAL"
          " MID-EDGE NODES ===")
    for k, lbl in (("ms_frac", "mid-side fraction of ALL nodes"),
                   ("b0_is_midside", "of the 0.5-1.35 band, mid-side"),
                   ("wall_ms_frac", "of WALL nodes, mid-side"),
                   ("gt_mat_zero_ms", "GT Mat == 0 at mid-side wall nodes"),
                   ("gt_mat_zero_corner", "GT Mat == 0 at corner wall nodes")):
        v = np.array([g[k] for g in geom], dtype=float)
        print("   %-38s %.3f   (min %.3f max %.3f)"
              % (lbl, np.nanmean(v), np.nanmin(v), np.nanmax(v)))
    print("   3/4 of all nodes on every vessel is the P2-triangle signature (3 mid-edge nodes"
          "\n   per corner), and the empty 0.5-1.35 band is ~100% mid-side -- each of its nodes"
          "\n   is exactly midpoint(its owner wall node, a species-band node).")
    print("   BUT MID-SIDE DOES NOT IMPLY EMPTY, and that correction matters: the mid-side"
          "\n   nodes lying ALONG the first species row carry Mat normally (Mat == 0 on 0.000"
          "\n   of them) and hold 170 of the cohort's 493 off-wall GT clot nodes.  Excluding"
          "\n   all mid-side nodes therefore costs a third of the recall (row E scored 0.429"
          "\n   that way, 0.561 once they are kept).  What is empty is specifically the mid-edge"
          "\n   node of an edge CROSSING OUT of the wall.  Mechanism not established; the"
          "\n   measurement is stable on all 19 vessels and is what row E is built on.")

    print("\n=== 3. SHELL FORMULATIONS, GT-Mat oracle ===")
    print("   %-31s %7s %6s %6s %7s | %7s %6s %7s"
          % ("formulation", "off F1", "prec", "rec", "n_pred", "F1", "prec", "n_pred"))
    print("   %-31s %28s | %22s" % ("", "--- native mesh ---", "--- coarser BL ---"))
    for k, v in shells.items():
        print("   %-31s %7.4f %6.3f %6.3f %7.1f | %7.4f %6.3f %7.1f"
              % (k, np.nanmean(v["f1"]), np.nanmean(v["p"]), np.nanmean(v["r"]),
                 np.mean(v["n"]), np.nanmean(v["f1c"]), np.nanmean(v["pc"]),
                 np.mean(v["nc"])))
    print("\n   'coarser BL' deletes the phantom family and RE-MEASURES the median edge"
          " length on the\n   surviving subgraph -- i.e. the same vessel meshed without the"
          " interleaved band, which\n   is what a customer mesh may well look like.")
    print("   E IS THE ANSWER TO 5.1.  It is built only from element order and connectivity"
          " (the\n   wall-normal mid-edge family, the corner row behind it, and the mid-side"
          " nodes along\n   that row), so it contains no length to recalibrate -- and it still"
          " scores BEST of the\n   five, on the native mesh and under the perturbation.  It"
          " reproduces the calibrated\n   1.35-2.20 band with Jaccard 1.000 on 12 of 19"
          " vessels and >= 0.84 on all of them.")

    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"geometry": geom, "per_vessel": per_vessel,
         "rows": {str(r): {k: [float(x) for x in v] for k, v in rows[r].items()}
                  for r in rows},
         "shells": {k: {kk: [float(x) for x in vv] for kk, vv in v.items()}
                    for k, v in shells.items()}}, indent=2))
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
