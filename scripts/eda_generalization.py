"""EDA for the generalization plan: geometry families, clot burden, and
in-distribution (family) similarity of every anchor vessel.

Reads the biochem anchor graphs (``data/processed/graphs_biochem_anchors/*.pt``)
and, for each patient, derives:

* geometry descriptors (width min/median/max, stenosis + expansion ratio,
  wall curvature index, aspect ratio, Reynolds number),
* ground-truth clot burden at the deploy horizon (Mat active fraction, and the
  off-wall / interior share that the lumen specialist must recover),
* a rule-based geometry family label, and
* the nearest training-set vessel in standardized geometry space (drives the
  ``family_validation`` design: a val anchor is "in-distribution" only if some
  training vessel sits close to it).

Pure analysis: loads on CPU, writes a JSON + prints an ASCII table. Re-run as
new vessels land so the split + feature priorities stay current.
"""

from __future__ import annotations

import glob
import json
import math
import os
from pathlib import Path

import torch

ANCHOR_DIR = Path("data/processed/graphs_biochem_anchors")
OUT_JSON = Path("outputs/biochem/eda/generalization_eda.json")

# Sealed 8h split (see docs/GENERALIZATION_PLAN.md). Used only to compute
# nearest-train-neighbor distance; not required for the raw descriptors.
TRAIN_16 = [
    "patient001", "patient002", "patient005", "patient006", "patient007",
    "patient008", "patient010", "patient011", "patient013", "patient014",
    "patient016", "patient020", "patient024", "patient025", "patient028",
    "patient029",
]

# y-channel index of mature clot (Mat_log1p_nd) confirmed from y_channel_names.
MAT_Y_IDX = 15
# NodeFeat (18-col) convention.
SDF_COL = 2
WNORM_COLS = (4, 6)
WIDTH_COL = 15


def _active_threshold() -> float:
    """Mat log1p-nd active threshold; mirror the training default (1e-4) when importable."""
    try:
        from src.core_physics.species_snapshot_gnn import snapshot_active_log_nd

        return float(snapshot_active_log_nd())
    except Exception:
        return 1e-4


def _wall_curvature_index(wn: torch.Tensor, edge_index: torch.Tensor) -> float:
    """Mean (1 - cos) between adjacent near-wall normals -- higher = more bend.

    Wall-normal vectors are defined on the near-wall band (unit norm); the solid
    boundary set carries a zero normal, so restrict the adjacency to nodes whose
    normal is non-degenerate.
    """
    row, col = edge_index
    nrm = wn.norm(dim=1)
    has = nrm > 1e-6
    both = has[row] & has[col]
    if both.sum() == 0:
        return 0.0
    a = wn[row[both]]
    b = wn[col[both]]
    a = a / a.norm(dim=1, keepdim=True).clamp(min=1e-6)
    b = b / b.norm(dim=1, keepdim=True).clamp(min=1e-6)
    cos = (a * b).sum(dim=1).clamp(-1, 1)
    return float((1.0 - cos).mean().item())


def _hop_from_wall(edge_index: torch.Tensor, wall: torch.Tensor, n: int, max_hop: int = 3) -> torch.Tensor:
    """BFS hop distance from the wall set (capped); -1 stays for unreached."""
    hop = torch.full((n,), -1, dtype=torch.long)
    hop[wall] = 0
    frontier = wall.clone()
    row, col = edge_index
    for h in range(1, max_hop + 1):
        # neighbours of the current frontier
        nbr = torch.zeros(n, dtype=torch.bool)
        mask = frontier[row]
        nbr[col[mask]] = True
        newly = nbr & (hop < 0)
        if newly.sum() == 0:
            break
        hop[newly] = h
        frontier = newly
    return hop


def profile(path: Path) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    name = path.stem
    n = int(data.num_nodes)
    x = data.x.float()
    ei = data.edge_index.long()

    # Robust width quantiles over real lumen nodes (drop the ~0.05 wall floor).
    width = x[:, WIDTH_COL]
    wpos = width[width > 0.06]
    if wpos.numel() < 16:
        wpos = width[width > 0]
    w_p5 = float(wpos.quantile(0.05))
    w_med = float(wpos.median())
    w_p95 = float(wpos.quantile(0.95))
    stenosis_ratio = w_med / max(w_p5, 1e-6)       # >1 : throat narrower than typical
    expansion_ratio = w_p95 / max(w_med, 1e-6)     # >1 : bulge wider than typical

    wall = data.mask_wall.view(-1).bool() if getattr(data, "mask_wall", None) is not None else torch.zeros(n, dtype=torch.bool)
    wn = x[:, WNORM_COLS[0]:WNORM_COLS[1]]
    curvature = _wall_curvature_index(wn, ei)

    xy = x[:, :2]
    span = float((xy[:, 0].max() - xy[:, 0].min()).item())
    aspect = span / max(w_med, 1e-6)

    re = float(data.re_actual) if getattr(data, "re_actual", None) is not None else float("nan")
    dbar = float(data.d_bar) if getattr(data, "d_bar", None) is not None else float("nan")

    # GT clot burden at final time.
    thr = _active_threshold()
    y = data.y  # [T, N, C]
    T = int(y.shape[0])
    mat_final = y[-1, :, MAT_Y_IDX]
    active = mat_final > thr
    n_active = int(active.sum())
    clot_frac = n_active / max(n, 1)
    mat_max = float(data.y[:, :, MAT_Y_IDX].max())

    hop = _hop_from_wall(ei, wall, n, max_hop=3)
    active_hop = hop[active]
    offwall = int((active_hop >= 1).sum()) if n_active else 0
    interior = int((active_hop >= 2).sum()) if n_active else 0
    offwall_frac = offwall / max(n_active, 1)
    interior_frac = interior / max(n_active, 1)

    # Onset: first time index whose active count clears a small floor.
    onset = -1
    for t in range(T):
        if int((y[t, :, MAT_Y_IDX] > thr).sum()) > 5:
            onset = t
            break
    onset_frac = onset / max(T - 1, 1) if onset >= 0 else float("nan")

    return {
        "name": name,
        "n_nodes": n,
        "re_actual": re,
        "d_bar": dbar,
        "w_p5": w_p5,
        "w_med": w_med,
        "w_p95": w_p95,
        "stenosis_ratio": stenosis_ratio,
        "expansion_ratio": expansion_ratio,
        "curvature": curvature,
        "aspect": aspect,
        "T": T,
        "mat_max": mat_max,
        "clot_frac": clot_frac,
        "n_active": n_active,
        "offwall_frac": offwall_frac,
        "interior_frac": interior_frac,
        "onset_frac": onset_frac,
    }


def family_label(p: dict) -> str:
    """Rule-based coarse geometry family for readability / coverage checks."""
    labels = []
    if p["stenosis_ratio"] >= 1.8:
        labels.append("stenosis")
    if p["expansion_ratio"] >= 1.8:
        labels.append("expansion")
    if p["curvature"] >= 0.20:
        labels.append("bend")
    if not labels:
        labels.append("mild/straight")
    return "+".join(labels)


def standardize(rows: list[dict], keys: list[str]) -> dict[str, list[float]]:
    stats = {}
    for k in keys:
        vals = [r[k] for r in rows if not math.isnan(r[k])]
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / max(len(vals) - 1, 1)
        stats[k] = (mu, math.sqrt(var) or 1.0)
    return stats


def main() -> None:
    fs = sorted(glob.glob(str(ANCHOR_DIR / "*.pt")))
    rows = [profile(Path(f)) for f in fs]

    geo_keys = ["stenosis_ratio", "expansion_ratio", "curvature", "aspect", "re_actual"]
    stats = standardize(rows, geo_keys)

    def vec(r: dict) -> list[float]:
        return [(r[k] - stats[k][0]) / stats[k][1] for k in geo_keys]

    by_name = {r["name"]: r for r in rows}
    train_vecs = {t: vec(by_name[t]) for t in TRAIN_16 if t in by_name}

    for r in rows:
        r["family"] = family_label(r)
        v = vec(r)
        best_d, best_t = float("inf"), None
        for t, tv in train_vecs.items():
            if t == r["name"]:
                continue
            d = math.sqrt(sum((a - b) ** 2 for a, b in zip(v, tv)))
            if d < best_d:
                best_d, best_t = d, t
        r["nn_train"] = best_t
        r["nn_train_dist"] = best_d

    rows.sort(key=lambda r: r["name"])

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    hdr = (
        f"{'patient':<12}{'T':>4}{'sten':>6}{'expa':>6}{'curv':>6}{'aspect':>7}"
        f"{'matmax':>8}{'clot%':>7}{'offwl%':>7}{'inter%':>7}{'onset':>6}  {'family':<22}{'nnTrain':>12}{'nnDist':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['name']:<12}{r['T']:>4d}{r['stenosis_ratio']:>6.2f}{r['expansion_ratio']:>6.2f}"
            f"{r['curvature']:>6.2f}{r['aspect']:>7.1f}{r['mat_max']:>8.4f}{100*r['clot_frac']:>7.2f}{100*r['offwall_frac']:>7.1f}"
            f"{100*r['interior_frac']:>7.1f}{r['onset_frac']:>6.2f}  {r['family']:<22}"
            f"{str(r['nn_train']):>12}{r['nn_train_dist']:>7.2f}"
        )

    print()
    fams: dict[str, int] = {}
    for r in rows:
        fams[r["family"]] = fams.get(r["family"], 0) + 1
    print("family counts:", dict(sorted(fams.items(), key=lambda kv: -kv[1])))
    res = [f"{v:.2f}" for v in sorted(r["re_actual"] for r in rows)]
    print("Re values:", sorted(set(res)))
    print(f"[save] {OUT_JSON}")


if __name__ == "__main__":
    main()
