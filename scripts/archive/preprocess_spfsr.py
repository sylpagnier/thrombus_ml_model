"""Preprocess COMSOL spf.sr exports -> compact per-graph cache.

Source: data/processed/cfd_results_biochem/patientXXX_spfsr.txt (5 col/block: x,y,spf.sr,
d(spf.sr,x),d(spf.sr,y)) or patient007_sr.txt (3 col/block: x,y,spf.sr). 201 export times
(t=0..30000 step 150), 2 leading reference-coord cols.

Maps each export onto its biochem-anchor graph nodes (nearest neighbour in nd coords, auto-
detecting the length unit), then saves [Texp, N_graph, C] (C=spf.sr[,dsrx,dsry]) to
data/processed/spfsr_cache/patientXXX.pt. Run: python scripts/preprocess_spfsr.py [patientXXX ...]
"""
from __future__ import annotations
import re, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SRC = ROOT / "data" / "processed" / "cfd_results_biochem"
ANCHOR = ROOT / "data" / "processed" / "graphs_biochem_anchors"
OUT = ROOT / "data" / "processed" / "spfsr_cache"


def _src_file(stem: str) -> Path:
    for name in (f"{stem}_spfsr.txt", f"{stem}_sr.txt"):
        if (SRC / name).exists():
            return SRC / name
    raise FileNotFoundError(f"no spf.sr export for {stem}")


def _parse_header(path: Path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        hdr = ""
        first_data = ""
        for line in f:
            if line.startswith("%"):
                hdr = line
            else:
                first_data = line
                break
    times = [float(t) for t in re.findall(r"spf\.sr @ t=([0-9.eE+-]+)", hdr)]
    ncols = len(first_data.split())                # count actual data fields
    block = (ncols - 2) // len(times)              # 2 leading coord cols
    return np.asarray(times), block, ncols


def _auto_scale(coords_phys, graph_nd, d_bar):
    from scipy.spatial import cKDTree
    best = None
    for s in (0.01, 0.001, 1.0, 0.1):
        nd = coords_phys * s / d_bar
        dist, _ = cKDTree(nd).query(graph_nd)
        md = float(np.median(dist))
        if best is None or md < best[1]:
            best = (s, md)
    return best


def process(stem: str):
    path = _src_file(stem)
    times, block, ncols = _parse_header(path)
    has_deriv = block >= 5
    NT = len(times)
    sr_cols = [2 + block * k + 2 for k in range(NT)]
    use = [0, 1] + sr_cols
    if has_deriv:
        dx_cols = [2 + block * k + 3 for k in range(NT)]
        dy_cols = [2 + block * k + 4 for k in range(NT)]
        use += dx_cols + dy_cols
    t0 = time.time()
    print(f"[{stem}] reading {path.name} ({path.stat().st_size/1e9:.2f} GB, {NT} times, block={block})")
    df = pd.read_csv(path, sep=r"\s+", comment="%", header=None,
                     usecols=use, dtype=np.float64, engine="c")
    df = df[use]                                   # enforce column order
    a = df.to_numpy()
    coords = a[:, :2]
    sr = a[:, 2:2 + NT]
    print(f"[{stem}] parsed {a.shape[0]} mesh nodes in {time.time()-t0:.0f}s")

    d = torch.load(ANCHOR / f"{stem}.pt", map_location="cpu", weights_only=False)
    d_bar = float(d.d_bar.view(-1)[0])
    graph_nd = d.x[:, :2].cpu().numpy()
    s, md = _auto_scale(coords, graph_nd, d_bar)
    from scipy.spatial import cKDTree
    exp_nd = coords * s / d_bar
    dist, idx = cKDTree(exp_nd).query(graph_nd)
    print(f"[{stem}] unit_scale={s} median_nn={md:.4g} nd (graph N={len(idx)}, mean_nn={float(dist.mean()):.4g})")

    out = {"export_times": torch.tensor(times, dtype=torch.float32),
           "unit_scale": s, "nn_dist": torch.tensor(dist, dtype=torch.float32),
           "sr": torch.tensor(sr[idx].T, dtype=torch.float32)}      # [NT, N_graph]
    if has_deriv:
        dx = a[:, 2 + NT:2 + 2 * NT]; dy = a[:, 2 + 2 * NT:2 + 3 * NT]
        out["dsrx"] = torch.tensor(dx[idx].T, dtype=torch.float32)
        out["dsry"] = torch.tensor(dy[idx].T, dtype=torch.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(out, OUT / f"{stem}.pt")
    print(f"[{stem}] saved sr{tuple(out['sr'].shape)} deriv={has_deriv} -> {OUT / f'{stem}.pt'}\n")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        stems = args
    else:
        stems = sorted(p.stem for p in ANCHOR.glob("patient*.pt") if "_metadata" not in p.stem)
    for stem in stems:
        try:
            process(stem)
        except FileNotFoundError as e:
            print(f"[skip] {e}")


if __name__ == "__main__":
    main()
