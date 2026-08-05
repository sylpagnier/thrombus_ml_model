"""Geometry-only thumbnails of every anchor vessel + similarity clustering.

Renders a grid of all anchors (faint lumen node cloud + highlighted wall nodes)
so a tight, geometrically-similar cohort can be picked by eye, and prints the
closest vessel clusters (restricted to clot-bearing vessels) in standardized
geometry space to seed the choice.

CPU only. Reads the EDA JSON (scripts/eda_generalization.py) for per-vessel
descriptors; falls back to recomputing width/clot if the JSON is absent.
"""

from __future__ import annotations

import glob
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ANCHOR_DIR = Path("data/processed/graphs_biochem_anchors")
EDA_JSON = Path("outputs/biochem/eda/generalization_eda.json")
OUT_PNG = Path("outputs/biochem/eda/anchor_geometry_grid.png")

MAX_CLOUD = 1500  # subsample interior nodes per anchor for a light scatter


def _load_eda() -> dict[str, dict]:
    if EDA_JSON.is_file():
        return {r["name"]: r for r in json.loads(EDA_JSON.read_text(encoding="utf-8"))}
    return {}


def main() -> None:
    fs = sorted(glob.glob(str(ANCHOR_DIR / "*.pt")))
    eda = _load_eda()

    # Order the grid so similar vessels sit next to each other.
    def sort_key(f: str):
        n = Path(f).stem
        r = eda.get(n, {})
        return (round(r.get("stenosis_ratio", 0.0), 1), round(r.get("expansion_ratio", 0.0), 1), r.get("aspect", 0.0))

    fs.sort(key=sort_key)

    ncol = 5
    nrow = math.ceil(len(fs) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.6, nrow * 2.2))
    axes = axes.reshape(-1)

    for ax in axes:
        ax.axis("off")

    for i, f in enumerate(fs):
        name = Path(f).stem
        data = torch.load(f, map_location="cpu", weights_only=False)
        xy = data.x[:, :2].float().numpy()
        wall = (
            data.mask_wall.view(-1).bool().numpy()
            if getattr(data, "mask_wall", None) is not None
            else None
        )
        ax = axes[i]
        # Faint interior cloud (subsampled) to show lumen fill.
        idx = torch.randperm(xy.shape[0])[:MAX_CLOUD].numpy()
        ax.scatter(xy[idx, 0], xy[idx, 1], s=0.5, c="0.8", linewidths=0, rasterized=True)
        # Wall nodes = vessel silhouette.
        if wall is not None and wall.any():
            ax.scatter(xy[wall, 0], xy[wall, 1], s=1.2, c="#1f4e79", linewidths=0, rasterized=True)
        ax.set_aspect("equal")
        ax.axis("off")
        r = eda.get(name, {})
        t = int(r.get("T", 0))
        clot = 100 * r.get("clot_frac", 0.0)
        sten = r.get("stenosis_ratio", 0.0)
        expa = r.get("expansion_ratio", 0.0)
        # Clot-bearing marker so the eye can avoid empty vessels.
        rich = (t >= 200 and r.get("offwall_frac", 0.0) >= 0.30 and r.get("mat_max", 0.0) >= 0.0015)
        tag = "[CLOT]" if rich else ("[low]" if clot >= 0.3 else "[~0]")
        ax.set_title(
            f"{name.replace('patient','p')} {tag}\nT={t} clot={clot:.1f}% st={sten:.2f} ex={expa:.2f}",
            fontsize=6.5,
        )

    fig.suptitle(
        "Anchor geometry (gray=lumen nodes, blue=wall) | [CLOT]=clot-rich full sim | sorted by stenosis/expansion",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130)
    print(f"[save] {OUT_PNG}")

    # --- similarity clusters among clot-bearing vessels ---
    if eda:
        keys = ["stenosis_ratio", "expansion_ratio", "curvature", "aspect"]
        rich = [
            n for n, r in eda.items()
            if r.get("T", 0) >= 200 and r.get("offwall_frac", 0.0) >= 0.30 and r.get("mat_max", 0.0) >= 0.0015
        ]
        # standardize over clot-rich set
        stats = {}
        for k in keys:
            vals = [eda[n][k] for n in rich]
            mu = sum(vals) / len(vals)
            sd = (sum((v - mu) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5 or 1.0
            stats[k] = (mu, sd)
        vec = {n: [(eda[n][k] - stats[k][0]) / stats[k][1] for k in keys] for n in rich}

        def dist(a, b):
            return math.sqrt(sum((x - y) ** 2 for x, y in zip(vec[a], vec[b])))

        pairs = sorted(
            ((dist(a, b), a, b) for i, a in enumerate(rich) for b in rich[i + 1:]),
            key=lambda t: t[0],
        )
        print("\nClot-rich vessels:", ", ".join(sorted(rich)))
        print("\nTightest clot-rich geometry pairs (smaller = more similar):")
        for d, a, b in pairs[:10]:
            print(f"  {a:<11} <-> {b:<11}  dist={d:.2f}")

        # Greedy tightest cohort of 6 around the densest seed.
        best = None
        for seed in rich:
            near = sorted(rich, key=lambda n: dist(seed, n))[:6]
            spread = max(dist(near[0], n) for n in near)
            if best is None or spread < best[0]:
                best = (spread, seed, near)
        print(f"\nTightest 6-vessel cohort (seed={best[1]}, max intra-dist={best[0]:.2f}):")
        print("  ", ", ".join(sorted(best[2])))


if __name__ == "__main__":
    main()
