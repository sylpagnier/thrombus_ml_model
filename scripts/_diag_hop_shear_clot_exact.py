"""Refined hop-1 shear washout probe on COMSOL-cached anchors only."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.clot_growth_masks import gt_growth_commit_mask_at_time  # noqa: E402
import scripts.spfsr_lib as spfsr  # noqa: E402


def bfs(edge_index: torch.Tensor, wall: torch.Tensor, n: int) -> torch.Tensor:
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


def main() -> None:
    phys = PhysicsConfig()
    bio = BiochemConfig(phase="biochem")
    lss = float(bio.lss)
    stems = sorted(p.stem for p in (REPO / "data/processed/spfsr_cache").glob("*.pt"))
    print(f"anchors with spfsr: {stems}")
    print(f"lss={lss} 1/s")

    pools: dict = defaultdict(lambda: defaultdict(list))
    counts: dict = defaultdict(lambda: defaultdict(int))

    for stem in stems:
        path = REPO / "data/processed/graphs_biochem_anchors" / f"{stem}.pt"
        if not path.exists():
            print(f"[WARN] missing graph {stem}")
            continue
        d = torch.load(path, map_location="cpu", weights_only=False)
        n = int(d.num_nodes)
        tf = int(d.y.shape[0]) - 1
        clot = gt_growth_commit_mask_at_time(d, tf, phys, torch.device("cpu")).numpy()
        wall = d.mask_wall.view(-1).bool()
        dist = bfs(d.edge_index, wall, n).numpy()
        al = spfsr.aligned(d, "cpu", stem)
        sr = al["sr"][tf].numpy()
        sr0 = al["sr"][0].numpy()
        sdf = d.x[:, 2].numpy()
        dbar = float(d.d_bar.view(-1)[0])
        sdf_um = sdf * dbar * 1e6
        speed = np.sqrt(d.y[tf, :, 0].numpy() ** 2 + d.y[tf, :, 1].numpy() ** 2)
        mat = d.y[tf, :, 15].numpy()

        print(f"\n=== {stem} N={n} Nclot={int(clot.sum())} d_bar={dbar*1e3:.2f}mm ===")
        hdr = (
            f"{'hop':>4}{'N':>7}{'Nclot':>7} | "
            f"{'sr_med':>8}{'sr0_med':>8}{'sr_cl':>8} | "
            f"{'<lss%':>6}{'cl_lss%':>8} | "
            f"{'sdf_um':>8}{'spd':>8}{'mat':>8}{'mat_cl':>8}"
        )
        print(hdr)
        for h in range(5):
            m = (dist == h) if h < 4 else (dist >= 4)
            if int(m.sum()) == 0:
                continue
            nc = int((m & clot).sum())
            sr_m = sr[m]
            sr_c = sr[m & clot]
            sr0_m = sr0[m]
            low = sr_m < lss
            low_clot = int((m & clot & (sr < lss)).sum())
            label = str(h) if h < 4 else "4+"
            print(
                f"{label:>4}{int(m.sum()):>7}{nc:>7} | "
                f"{np.median(sr_m):>8.1f}{np.median(sr0_m):>8.1f}"
                f"{(np.median(sr_c) if nc else float('nan')):>8.1f} | "
                f"{100.0 * float(low.mean()):>5.1f}%"
                f"{100.0 * low_clot / max(nc, 1):>7.1f}% | "
                f"{np.median(sdf_um[m]):>8.1f}{np.median(speed[m]):>8.3f}"
                f"{np.median(mat[m]):>8.3f}"
                f"{(np.median(mat[m & clot]) if nc else float('nan')):>8.3f}"
            )
            key = h if h < 4 else 4
            pools[key]["sr"].extend(sr_m.tolist())
            pools[key]["sr0"].extend(sr0_m.tolist())
            pools[key]["sr_clot"].extend(sr_c.tolist())
            pools[key]["sdf_um"].extend(sdf_um[m].tolist())
            pools[key]["mat"].extend(mat[m].tolist())
            pools[key]["mat_clot"].extend(mat[m & clot].tolist())
            pools[key]["speed"].extend(speed[m].tolist())
            counts[key]["N"] += int(m.sum())
            counts[key]["Nc"] += nc
            counts[key]["low"] += int(low.sum())
            counts[key]["low_clot"] += low_clot

    print("\n======== POOLED (spf.sr cached anchors only) ========")
    med1 = float(np.median(pools[1]["sr"])) if pools[1]["sr"] else float("nan")
    print(
        f"{'hop':>4}{'N':>8}{'Nc':>6}{'pct':>7} | "
        f"{'sr_med':>8}{'sr0_med':>8}{'sr_cl':>8} | "
        f"{'<lss%':>6} | "
        f"{'sdf_um':>8}{'mat':>8}{'mat_cl':>8} | "
        f"{'vs_h1':>8}"
    )
    for h in range(5):
        if counts[h]["N"] == 0:
            continue
        sr = np.asarray(pools[h]["sr"])
        sr0 = np.asarray(pools[h]["sr0"])
        src = np.asarray(pools[h]["sr_clot"])
        sdf = np.asarray(pools[h]["sdf_um"])
        mat = np.asarray(pools[h]["mat"])
        matc = np.asarray(pools[h]["mat_clot"])
        pct = 100.0 * counts[h]["Nc"] / counts[h]["N"]
        label = str(h) if h < 4 else "4+"
        print(
            f"{label:>4}{counts[h]['N']:>8}{counts[h]['Nc']:>6}{pct:>6.2f}% | "
            f"{np.median(sr):>8.1f}{np.median(sr0):>8.1f}"
            f"{(np.median(src) if src.size else float('nan')):>8.1f} | "
            f"{100.0 * float((sr < lss).mean()):>5.1f}% | "
            f"{np.median(sdf):>8.1f}{np.median(mat):>8.3f}"
            f"{(np.median(matc) if matc.size else float('nan')):>8.3f} | "
            f"{np.median(sr) / max(med1, 1e-12):>7.2f}x"
        )

    print("\n[i] Clot rate CONDITIONAL on low-shear (spf.sr < lss) at t_final:")
    for h in range(5):
        if counts[h]["low"] == 0:
            continue
        label = str(h) if h < 4 else "4+"
        rate = 100.0 * counts[h]["low_clot"] / counts[h]["low"]
        print(
            f"  hop {label}: {counts[h]['low_clot']}/{counts[h]['low']} = {rate:.2f}%"
        )

    print("\n[i] SDF (wall distance, um) percentiles by hop:")
    for h in range(4):
        sdf = np.asarray(pools[h]["sdf_um"])
        if sdf.size == 0:
            continue
        print(
            f"  hop {h}: p10={np.percentile(sdf, 10):.1f} "
            f"med={np.median(sdf):.1f} p90={np.percentile(sdf, 90):.1f} um"
        )

    if pools[2]["sdf_um"]:
        thr = float(np.percentile(pools[2]["sdf_um"], 10))
        sdf1 = np.asarray(pools[1]["sdf_um"])
        print(
            f"\n[i] hop2 p10 sdf={thr:.1f} um; "
            f"fraction of hop1 below that: {100.0 * float((sdf1 < thr).mean()):.1f}%"
        )

    # Key contrast: hop1 vs hop2 shear distributions
    print("\n[i] Exact spf.sr distribution contrast hop1 vs hop2 (all nodes):")
    for h, name in [(1, "hop1"), (2, "hop2")]:
        sr = np.asarray(pools[h]["sr"])
        print(
            f"  {name}: mean={sr.mean():.1f} med={np.median(sr):.1f} "
            f"p25={np.percentile(sr, 25):.1f} p75={np.percentile(sr, 75):.1f} "
            f"frac<lss={100*(sr<lss).mean():.1f}%"
        )

    # Among hop1, are the rare clots special?
    print("\n[i] Rare hop1 clots: shear vs hop1 non-clot:")
    sr_c = np.asarray(pools[1]["sr_clot"])
    sr_all = np.asarray(pools[1]["sr"])
    if sr_c.size:
        # approximate non-clot = all (clots are rare)
        print(
            f"  hop1 clot n={sr_c.size} med_sr={np.median(sr_c):.2f} "
            f"frac<lss={100*(sr_c<lss).mean():.1f}%"
        )
        print(
            f"  hop1 all  n={sr_all.size} med_sr={np.median(sr_all):.2f} "
            f"frac<lss={100*(sr_all<lss).mean():.1f}%"
        )


if __name__ == "__main__":
    main()
