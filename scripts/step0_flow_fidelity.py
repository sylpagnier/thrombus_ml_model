"""Is the pack's t=0 GT flow faithful, and where does the shear reconstruction break?

Matches the pack's 17413 nodes onto COMSOL's 51240-node domain export (t=0) and asks:
  1. does ``y[0,:,0:2] * u_ref`` equal COMSOL's u,v?
  2. does a graph-operator shear reconstruction match COMSOL ``spf.sr`` in the INTERIOR?
  3. ... and at the WALL, where no-slip makes one-sided stencils fail?
  4. which wall-shear estimator best reproduces COMSOL's wall ``spf.sr``?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> int:
    dm = np.load("outputs/comsol_p007_domain.npz")
    data = torch.load("data/processed/graphs_biochem_anchors/patient007.pt",
                      map_location="cpu", weights_only=False)
    pos = data.siren_pos.detach().numpy().astype(np.float64)
    wall = data.mask_wall.reshape(-1).bool().numpy()
    u_ref = float(data.u_ref.reshape(-1)[0])

    cx, cy = dm["x"][0], dm["y"][0]
    L0 = (cx.max() - cx.min()) / (pos[:, 0].max() - pos[:, 0].min())   # cm per pack unit
    # map pack coords -> cm
    px = (pos[:, 0] - pos[:, 0].min()) * L0 + cx.min()
    py = (pos[:, 1] - pos[:, 1].min()) * L0 + cy.min()
    tree = cKDTree(np.stack([cx, cy], 1))
    dist, nn = tree.query(np.stack([px, py], 1))
    print("pack->comsol NN match: median %.5f cm, p99 %.5f cm, mesh L0=%.4f cm"
          % (np.median(dist), np.percentile(dist, 99), L0))

    cu, cv, csr = dm["u"][0][nn], dm["v"][0][nn], dm["sr"][0][nn]
    cdx = dm["dsrx"][0][nn]
    pu = data.y[0, :, 0].numpy() * u_ref * 100.0     # m/s -> cm/s
    pv = data.y[0, :, 1].numpy() * u_ref * 100.0
    interior = ~wall

    print("\n[1] velocity fidelity (all nodes)  u: pearson %.4f  spearman %.4f  scale(lsq) %.4f"
          % (np.corrcoef(pu, cu)[0, 1], spearman(pu, cu), (pu @ cu) / (pu @ pu)))
    print("    speed: pearson %.4f  |pack|/|comsol| median %.4f"
          % (np.corrcoef(np.hypot(pu, pv), np.hypot(cu, cv))[0, 1],
             np.median(np.hypot(pu, pv)[interior] / np.maximum(np.hypot(cu, cv)[interior], 1e-9))))

    # --- 2/3 graph shear reconstruction --------------------------------------
    u = data.y[0, :, 0].float()
    v = data.y[0, :, 1].float()

    def mmv(G, f):
        return torch.sparse.mm(G, f.reshape(-1, 1)).reshape(-1).numpy()

    ux, uy = mmv(data.G_x, u), mmv(data.G_y, u)
    vx, vy = mmv(data.G_x, v), mmv(data.G_y, v)
    sr_nd = np.sqrt(2 * ux ** 2 + 2 * vy ** 2 + (uy + vx) ** 2)
    scale = u_ref * 100.0 / L0            # cm/s per cm -> 1/s
    sr_rec = sr_nd * scale
    for nm, m in (("interior", interior), ("wall", wall)):
        print("[%s] sr recon vs COMSOL: spearman %.3f pearson %.3f  median ratio %.3f"
              % (nm, spearman(sr_rec[m], csr[m]), np.corrcoef(sr_rec[m], csr[m])[0, 1],
                 np.median(sr_rec[m] / np.maximum(csr[m], 1e-9))))

    # --- 4 wall-shear estimators ---------------------------------------------
    ei = data.edge_index.numpy()
    N = len(pos)
    # neighbour lists
    nbr = [[] for _ in range(N)]
    for a, b in zip(ei[0], ei[1]):
        nbr[a].append(b)
    wall_idx = np.where(wall)[0]
    speed_cm = np.hypot(pu, pv)

    est = {}
    # (a) |u| at nearest interior neighbour / distance
    a_val = np.zeros(N)
    for i in wall_idx:
        cand = [j for j in nbr[i] if not wall[j]]
        if not cand:
            cand = list(nbr[i])
        if not cand:
            continue
        dists = np.hypot(px[cand] - px[i], py[cand] - py[i])
        a_val[i] = np.max(speed_cm[cand] / np.maximum(dists, 1e-9))
    est["|u|_nbr/dist (max)"] = a_val
    b_val = np.zeros(N)
    for i in wall_idx:
        cand = [j for j in nbr[i] if not wall[j]] or list(nbr[i])
        if not cand:
            continue
        dists = np.hypot(px[cand] - px[i], py[cand] - py[i])
        b_val[i] = np.mean(speed_cm[cand] / np.maximum(dists, 1e-9))
    est["|u|_nbr/dist (mean)"] = b_val
    # (c) graph sr evaluated at interior neighbours, averaged back to wall
    c_val = np.zeros(N)
    for i in wall_idx:
        cand = [j for j in nbr[i] if not wall[j]]
        c_val[i] = np.mean(sr_rec[cand]) if cand else sr_rec[i]
    est["sr at interior nbrs"] = c_val
    # (d) 2-hop interior mean
    d_val = np.zeros(N)
    for i in wall_idx:
        s = set()
        for j in nbr[i]:
            if not wall[j]:
                s.add(j)
            for k in nbr[j]:
                if not wall[k]:
                    s.add(k)
        d_val[i] = np.mean(sr_rec[list(s)]) if s else sr_rec[i]
    est["sr 2-hop interior mean"] = d_val
    est["sr at wall (raw)"] = sr_rec

    print("\n[4] wall spf.sr estimators (n=%d wall nodes)" % wall.sum())
    for nm, val in est.items():
        w = val[wall]
        print("   %-24s spearman %.3f  pearson %.3f  lsq scale %.4f"
              % (nm, spearman(w, csr[wall]), np.corrcoef(w, csr[wall])[0, 1],
                 (w @ csr[wall]) / max(w @ w, 1e-12)))

    # --- gradient of the best field ------------------------------------------
    print("\n[5] d(sr,x) from G_x applied to each estimator (vs COMSOL dsrx at wall)")
    for nm, val in est.items():
        # fill interior with sr_rec so the x-derivative is meaningful
        fld = np.where(wall, val, sr_rec)
        g = mmv(data.G_x, torch.tensor(fld, dtype=torch.float32)) / L0
        print("   %-24s spearman %.3f  pearson %.3f  lsq scale %.4f"
              % (nm, spearman(g[wall], cdx[wall]), np.corrcoef(g[wall], cdx[wall])[0, 1],
                 (g[wall] @ cdx[wall]) / max(g[wall] @ g[wall], 1e-12)))
    print("   COMSOL sr itself -> G_x:")
    g = mmv(data.G_x, torch.tensor(csr, dtype=torch.float32)) / L0
    print("      spearman %.3f pearson %.3f lsq %.4f"
          % (spearman(g[wall], cdx[wall]), np.corrcoef(g[wall], cdx[wall])[0, 1],
             (g[wall] @ cdx[wall]) / max(g[wall] @ g[wall], 1e-12)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
