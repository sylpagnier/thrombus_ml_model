"""Shear + Mat by wall-hop on biochem anchors (hop-1 washout probe).

For each anchor at t_final:
  - BFS hop distance from wall
  - GT growth clot mask
  - WLS shear from GT u,v (1/s)
  - exact COMSOL spf.sr when cached (1/s)
  - Mat_log1p_nd

Run:
  python scripts/_diag_hop_shear_clot.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.clot_growth_masks import gt_growth_commit_mask_at_time  # noqa: E402
from src.core_physics.clot_phi_simple import clot_phi_thresh_si, gt_gamma_dot_nd  # noqa: E402
import scripts.spfsr_lib as spfsr  # noqa: E402

MAT_IDX = 15  # Mat_log1p_nd


def bfs_hop(edge_index: torch.Tensor, wall: torch.Tensor, n: int) -> torch.Tensor:
    dist = torch.full((n,), -1, dtype=torch.long)
    dist[wall] = 0
    row, col = edge_index[0], edge_index[1]
    frontier = wall.clone()
    h = 0
    while bool(frontier.any()):
        h += 1
        nxt = torch.zeros(n, dtype=torch.bool)
        nxt[col[frontier[row]]] = True
        nxt[row[frontier[col]]] = True
        nxt &= dist < 0
        dist[nxt] = h
        frontier = nxt
        if h > 10000:
            break
    return dist


def summarize(vals) -> dict:
    vals = np.asarray(vals, dtype=float)
    if vals.size == 0:
        return dict(n=0, mean=np.nan, med=np.nan, p10=np.nan, p90=np.nan, frac_lt_lss=np.nan)
    return dict(
        n=int(vals.size),
        mean=float(vals.mean()),
        med=float(np.median(vals)),
        p10=float(np.percentile(vals, 10)),
        p90=float(np.percentile(vals, 90)),
        frac_lt_lss=float((vals < LSS).mean()),
    )


def hop_mask(dist_np: np.ndarray, h: int) -> np.ndarray:
    if h < 4:
        return dist_np == h
    return dist_np >= 4


def main() -> None:
    global LSS
    root = REPO / "data" / "processed" / "graphs_biochem_anchors"
    paths = sorted(root.glob("patient*.pt"))
    phys = PhysicsConfig()
    bio = BiochemConfig(phase="biochem")
    thresh = clot_phi_thresh_si(phys)
    LSS = float(bio.lss)
    dev = torch.device("cpu")

    pools_wls = {h: {"clot": [], "all": []} for h in range(5)}
    pools_exact = {h: {"clot": [], "all": []} for h in range(5)}
    mat_pools = {h: {"clot": [], "all": []} for h in range(5)}
    hop_clot_counts = {h: 0 for h in range(5)}
    hop_node_counts = {h: 0 for h in range(5)}
    # among low-shear nodes only: clot prevalence
    low_shear_clot = {h: 0 for h in range(5)}
    low_shear_nodes = {h: 0 for h in range(5)}
    n_exact = 0
    n_graphs = 0

    print(
        f"lss={LSS:.1f} 1/s  clot_thresh={thresh:.4f} Pa*s  n_graphs={len(paths)}"
    )
    print("Shear: WLS from GT u,v at t_final; exact=COMSOL spf.sr when cached")
    print()
    hdr = (
        f"{'patient':<12}{'Nc':>5}{'h0':>5}{'h1':>5}{'h2':>5}{'h3':>5} | "
        f"{'sr_h0':>7}{'sr_h1':>7}{'sr_h2':>7} | "
        f"{'cl_h1':>7}{'cl_h2':>7} | "
        f"{'<lss_h1':>7}{'<lss_h2':>7}{'ex':>3}"
    )
    print(hdr)
    print("-" * len(hdr))

    for p in paths:
        data = torch.load(p, map_location="cpu", weights_only=False)
        n = int(data.num_nodes)
        tf = int(data.y.shape[0]) - 1
        clot = gt_growth_commit_mask_at_time(data, tf, phys, dev)
        wall = (
            data.mask_wall.view(-1).bool()
            if data.mask_wall is not None
            else torch.zeros(n, dtype=torch.bool)
        )
        dist = bfs_hop(data.edge_index, wall, n)
        dist_np = dist.numpy()
        clot_np = clot.numpy()

        g_nd = gt_gamma_dot_nd(data, tf, dev)
        u_ref = float(data.u_ref.view(-1)[0])
        d_bar = float(data.d_bar.view(-1)[0])
        sr_wls = (g_nd * (u_ref / d_bar)).detach().cpu().numpy()

        mat = data.y[tf, :, MAT_IDX].float().cpu().numpy()

        sr_ex = None
        if spfsr.has_cache(p.stem):
            try:
                al = spfsr.aligned(data, "cpu", p.stem)
                sr_ex = al["sr"][tf].cpu().numpy()
                n_exact += 1
            except Exception:
                sr_ex = None

        hop_counts = [int(((dist == h) & clot).sum()) for h in range(4)]
        hop_counts.append(int(((dist >= 4) & clot).sum()))

        def med_all(sr: np.ndarray, h: int) -> float:
            vals = sr[hop_mask(dist_np, h)]
            return float(np.median(vals)) if vals.size else float("nan")

        def med_clot(sr: np.ndarray, h: int) -> float:
            vals = sr[hop_mask(dist_np, h) & clot_np]
            return float(np.median(vals)) if vals.size else float("nan")

        def frac_lt(sr: np.ndarray, h: int) -> float:
            vals = sr[hop_mask(dist_np, h)]
            return float((vals < LSS).mean()) if vals.size else float("nan")

        for h in range(5):
            m = hop_mask(dist_np, h)
            hop_node_counts[h] += int(m.sum())
            hop_clot_counts[h] += int((m & clot_np).sum())
            pools_wls[h]["all"].extend(sr_wls[m].tolist())
            pools_wls[h]["clot"].extend(sr_wls[m & clot_np].tolist())
            mat_pools[h]["all"].extend(mat[m].tolist())
            mat_pools[h]["clot"].extend(mat[m & clot_np].tolist())
            low = m & (sr_wls < LSS)
            low_shear_nodes[h] += int(low.sum())
            low_shear_clot[h] += int((low & clot_np).sum())
            if sr_ex is not None:
                pools_exact[h]["all"].extend(sr_ex[m].tolist())
                pools_exact[h]["clot"].extend(sr_ex[m & clot_np].tolist())

        n_graphs += 1
        print(
            f"{p.stem:<12}{int(clot.sum()):>5}"
            f"{hop_counts[0]:>5}{hop_counts[1]:>5}{hop_counts[2]:>5}{hop_counts[3]:>5} | "
            f"{med_all(sr_wls, 0):>7.1f}{med_all(sr_wls, 1):>7.1f}{med_all(sr_wls, 2):>7.1f} | "
            f"{med_clot(sr_wls, 1):>7.1f}{med_clot(sr_wls, 2):>7.1f} | "
            f"{frac_lt(sr_wls, 1):>7.2f}{frac_lt(sr_wls, 2):>7.2f}"
            f"{'Y' if sr_ex is not None else 'n':>3}"
        )

    print()
    print("=" * 100)
    print("POOLED WLS shear (1/s) by hop  [all nodes / clot nodes]")
    print(
        f"{'hop':>4}{'N_all':>8}{'N_clot':>8}{'pct_clot':>9} | "
        f"{'med_all':>8}{'med_clot':>9}{'p10_all':>8}{'p90_all':>8} | "
        f"{'<lss_all':>9}{'<lss_clot':>10} | "
        f"{'medMat_all':>10}{'medMat_clot':>11} | "
        f"{'clot|lowSR':>10}"
    )
    for h in range(5):
        a = summarize(pools_wls[h]["all"])
        c = summarize(pools_wls[h]["clot"])
        ma = summarize(mat_pools[h]["all"])
        mc = summarize(mat_pools[h]["clot"])
        pct_c = hop_clot_counts[h] / max(hop_node_counts[h], 1)
        clot_given_low = low_shear_clot[h] / max(low_shear_nodes[h], 1)
        label = str(h) if h < 4 else "4+"
        print(
            f"{label:>4}{a['n']:>8}{c['n']:>8}{100 * pct_c:>8.2f}% | "
            f"{a['med']:>8.1f}{c['med']:>9.1f}{a['p10']:>8.1f}{a['p90']:>8.1f} | "
            f"{a['frac_lt_lss']:>9.3f}{c['frac_lt_lss']:>10.3f} | "
            f"{ma['med']:>10.3f}{mc['med']:>11.3f} | "
            f"{100 * clot_given_low:>9.2f}%"
        )

    if n_exact:
        print()
        print(
            f"POOLED exact COMSOL spf.sr (1/s) by hop  "
            f"[n_graphs_with_cache={n_exact}/{n_graphs}]"
        )
        print(
            f"{'hop':>4}{'N_all':>8}{'N_clot':>8} | "
            f"{'med_all':>8}{'med_clot':>9}{'p10_all':>8}{'p90_all':>8} | "
            f"{'<lss_all':>9}{'<lss_clot':>10}"
        )
        for h in range(5):
            a = summarize(pools_exact[h]["all"])
            c = summarize(pools_exact[h]["clot"])
            label = str(h) if h < 4 else "4+"
            print(
                f"{label:>4}{a['n']:>8}{c['n']:>8} | "
                f"{a['med']:>8.1f}{c['med']:>9.1f}{a['p10']:>8.1f}{a['p90']:>8.1f} | "
                f"{a['frac_lt_lss']:>9.3f}{c['frac_lt_lss']:>10.3f}"
            )

    print()
    print("[i] Interpretation helpers")
    m1 = summarize(pools_wls[1]["all"])["med"]
    m2 = summarize(pools_wls[2]["all"])["med"]
    print(
        f"  WLS median shear hop1/hop2 = {m1 / max(m2, 1e-12):.2f}x  "
        f"(hop1 med={m1:.1f}, hop2 med={m2:.1f})"
    )
    print(
        f"  clot prevalence hop1="
        f"{100 * hop_clot_counts[1] / max(hop_node_counts[1], 1):.2f}%  "
        f"hop2={100 * hop_clot_counts[2] / max(hop_node_counts[2], 1):.2f}%"
    )
    print(
        f"  low-shear fraction (sr<lss={LSS}) "
        f"hop1={summarize(pools_wls[1]['all'])['frac_lt_lss']:.3f}  "
        f"hop2={summarize(pools_wls[2]['all'])['frac_lt_lss']:.3f}"
    )
    print(
        f"  pooled clot counts: h0={hop_clot_counts[0]} h1={hop_clot_counts[1]} "
        f"h2={hop_clot_counts[2]} h3={hop_clot_counts[3]} h4+={hop_clot_counts[4]}"
    )
    if n_exact:
        e1 = summarize(pools_exact[1]["all"])["med"]
        e2 = summarize(pools_exact[2]["all"])["med"]
        print(
            f"  exact median shear hop1/hop2 = {e1 / max(e2, 1e-12):.2f}x  "
            f"(hop1 med={e1:.1f}, hop2 med={e2:.1f})"
        )
        print(
            f"  exact low-shear frac "
            f"hop1={summarize(pools_exact[1]['all'])['frac_lt_lss']:.3f}  "
            f"hop2={summarize(pools_exact[2]['all'])['frac_lt_lss']:.3f}"
        )


if __name__ == "__main__":
    main()
