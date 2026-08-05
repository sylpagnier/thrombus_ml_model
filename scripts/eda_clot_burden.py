"""Per-vessel clot burden + wall-hop structure under the **graded** clot label.

Existing burden numbers in ``eda_generalization.py`` use raw ``Mat_log1p_nd``
activity. The metric the wall model is actually scored on is not that: it is the
viscosity-rise label from ``gt_clot_phi_at_time`` --

    clot(t) = relu(mu_eff(t) - mu_eff(0)) >= CLOT_PHI_THRESH_SI   (default 0.055 Pa.s)

Those two labels disagree materially (raw Mat is ~2x looser and puts far more
mass at hops 2-3), so cohort-design decisions must be made on *this* one.

Reports, per anchor, restricted to the 3-hop wall band with the 4-hop label
ceiling (matching ``SPECIES_SNAPSHOT_WALL_HOPS`` / ``CLOT_PHI_CEILING_HOPS``):

* ``n_clot``  -- graded clot nodes at the deploy horizon,
* ``burden``  -- ``n_clot / n_band`` (how thick the clot is), and
* the hop-0..4 histogram + off-wall (hop>=2) share (what *shape* it is).

Cohort selection should match the holdout on **both** burden and off-wall share;
matching only "is clot-rich" is what produced the fat-cohort/thin-holdout
mismatch documented in docs/WALL_MODEL_PLAN.md.

Pure analysis: CPU only, no model, no rollout. Prints a table and writes JSON.

    python scripts/eda_clot_burden.py
    python scripts/eda_clot_burden.py --anchors patient005,patient020
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import PhysicsConfig  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

ANCHOR_DIR = Path("data/processed/graphs_biochem_anchors")
OUT_JSON = Path("outputs/biochem/eda/clot_burden_graded.json")

WALL_HOPS = 3   # SPECIES_SNAPSHOT_WALL_HOPS
CEIL_HOPS = 4   # CLOT_PHI_CEILING_HOPS

# Named cohorts referenced by docs/WALL_MODEL_PLAN.md.
COHORTS: dict[str, list[str]] = {
    "small_train": ["patient005", "patient006", "patient010"],
    "nplus_train": [
        "patient001", "patient005", "patient006", "patient007", "patient010",
        "patient013", "patient016", "patient021", "patient029", "patient032",
        "patient035", "patient037",
    ],
    "holdout": ["patient020"],
    "batch_1b": [
        "patient012", "patient040", "patient041", "patient042",
        "patient043", "patient044",
    ],
}


def hops_from_wall(edge_index: torch.Tensor, wall: torch.Tensor, n: int, max_hops: int) -> torch.Tensor:
    """Graph hop distance from the wall mask (unreached nodes keep a large sentinel)."""
    hops = torch.full((n,), 10_000, dtype=torch.long)
    hops[wall] = 0
    row, col = edge_index[0].long(), edge_index[1].long()
    frontier = wall.clone()
    for k in range(1, max_hops + 1):
        nxt = torch.zeros(n, dtype=torch.bool)
        nxt[col[frontier[row]]] = True
        nxt &= hops > k
        if not nxt.any():
            break
        hops[nxt] = k
        frontier = nxt
    return hops


def anchor_burden(path: Path, phys: PhysicsConfig) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    n = int(data.x.shape[0])
    wall = data.mask_wall.reshape(-1).bool()
    hops = hops_from_wall(data.edge_index, wall, n, CEIL_HOPS + 1)
    band = hops <= WALL_HOPS
    n_times = int(data.y.shape[0])

    phi = gt_clot_phi_at_time(data, n_times - 1, phys, device=torch.device("cpu"))
    clot = (phi.reshape(-1) > 0.5) & (hops <= CEIL_HOPS)

    hist = [int((clot & (hops == k)).sum()) for k in range(CEIL_HOPS + 1)]
    n_clot = int(clot.sum())
    n_band = max(int(band.sum()), 1)
    return {
        "n_times": n_times,
        "n_band": int(band.sum()),
        "n_clot": n_clot,
        "burden": n_clot / n_band,
        "hop_hist": hist,
        "offwall_share": (sum(hist[2:]) / n_clot) if n_clot else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Graded clot burden + hop structure per anchor")
    ap.add_argument("--anchors", default="", help="comma-separated; default = all cohort members")
    ap.add_argument("--out", default=str(OUT_JSON))
    args = ap.parse_args()

    if args.anchors.strip():
        names = [a.strip() for a in args.anchors.split(",") if a.strip()]
    else:
        names = sorted({a for members in COHORTS.values() for a in members})

    phys = PhysicsConfig(phase="biochem")
    rows: dict[str, dict] = {}
    for anchor in names:
        path = ANCHOR_DIR / f"{anchor}.pt"
        if not path.exists():
            print(f"[miss] {anchor}")
            continue
        rows[anchor] = anchor_burden(path, phys)

    def emit(title: str, members: list[str]) -> dict | None:
        sel = [(a, rows[a]) for a in members if a in rows]
        if not sel:
            return None
        print(f"\n=== {title} ===")
        print(f"{'anchor':12s} {'T':>4s} {'band':>6s} {'clot':>6s} {'burden':>8s} {'offwall':>8s}   hop0..4")
        for anchor, r in sel:
            print(f"{anchor:12s} {r['n_times']:4d} {r['n_band']:6d} {r['n_clot']:6d} "
                  f"{r['burden']*100:7.2f}% {r['offwall_share']*100:7.1f}%   {r['hop_hist']}")
        mean_burden = sum(r["burden"] for _, r in sel) / len(sel)
        tot = [0] * (CEIL_HOPS + 1)
        for _, r in sel:
            for i, v in enumerate(r["hop_hist"]):
                tot[i] += v
        share = sum(tot[2:]) / max(sum(tot), 1)
        print(f"{'MEAN':12s} {'':4s} {'':6s} {sum(r['n_clot'] for _, r in sel)/len(sel):6.1f} "
              f"{mean_burden*100:7.2f}% {share*100:7.1f}%")
        return {"mean_burden": mean_burden, "offwall_share": share, "n": len(sel)}

    summary = {}
    for title, members in COHORTS.items():
        got = emit(title, members)
        if got:
            summary[title] = got

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"label": "gt_clot_phi_at_time (viscosity-rise)", "wall_hops": WALL_HOPS,
         "ceiling_hops": CEIL_HOPS, "per_anchor": rows, "cohorts": summary},
        indent=2))
    print(f"\n[OK] -> {out_path}")


if __name__ == "__main__":
    main()
