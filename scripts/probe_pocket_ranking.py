"""Free cross-vessel validation: can component-level flow rank true clot pockets?

Finding this validates (docs/WALL_MODEL_PLAN.md s2, rewritten 2026-08-05): on patient020
the wall model's failure is POCKET SELECTION, not growth or perception. Its predicted
connected components are pure (53/53, 16/16, 13/13 GT), but it commits to 33 spurious
pockets alongside 6 real ones. Keeping only the true ones would give F1 0.887 vs the
0.500 baseline, and a single threshold on component-min hop-2 speed reaches 0.876.

That threshold was fitted ON patient020, the holdout -- a 1-parameter fit on the test
set, so it proves nothing by itself. This script tests the *physical* claim behind it
across every vessel, with no model and no rollout:

    Among candidate stagnation pockets on the wall, do the ones that actually clot have
    systematically lower flow than the ones that do not?

Candidates are connected components of low-flow wall nodes -- a superset of what the
model would ever predict -- so the negatives here are the same kind of plausible-but-
wrong pocket the model keeps committing to.

Also tests shear rate alongside speed: COMSOL_PHYSICS_VALIDATION.md s"deposition" reports
the dominant trigger is the low-shear stagnation gate (`spf.sr < lss`, on for 79.7% of
growing nodes), not speed, so shear may rank pockets better than speed does.

    python scripts/probe_pocket_ranking.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import PhysicsConfig  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402

ANCHOR_DIR = get_project_root() / "data/processed/graphs_biochem_anchors"


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    a = np.concatenate([pos, neg]).astype(float)
    o = a.argsort()
    r = np.empty_like(a)
    r[o] = np.arange(1, a.size + 1)
    uq, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    s = np.zeros(uq.size)
    np.add.at(s, inv, r)
    r = (s / cnt)[inv]
    n1 = pos.size
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * neg.size))


def _nz_hop(vals: torch.Tensor, row, col, n: int) -> torch.Tensor:
    v = vals.reshape(-1)[col].float()
    m = (v > 1e-9).float()
    s = torch.zeros(n)
    c = torch.zeros(n)
    s.index_add_(0, row, v * m)
    c.index_add_(0, row, m)
    return s / c.clamp(min=1.0)


def _shear_rate(d, u, v, n: int) -> torch.Tensor:
    """COMSOL-style shear rate magnitude from the graph gradient operators."""
    from src.utils.rheology import compute_shear_rate

    if getattr(d, "G_x", None) is None or getattr(d, "G_y", None) is None:
        return torch.zeros(n)
    gx, gy = d.G_x, d.G_y
    uu = u.reshape(-1, 1).float()
    vv = v.reshape(-1, 1).float()
    du_dx = torch.sparse.mm(gx, uu).squeeze(1) if gx.is_sparse else (gx @ uu).squeeze(1)
    du_dy = torch.sparse.mm(gy, uu).squeeze(1) if gy.is_sparse else (gy @ uu).squeeze(1)
    dv_dx = torch.sparse.mm(gx, vv).squeeze(1) if gx.is_sparse else (gx @ vv).squeeze(1)
    dv_dy = torch.sparse.mm(gy, vv).squeeze(1) if gy.is_sparse else (gy @ vv).squeeze(1)
    return compute_shear_rate(du_dx, du_dy, dv_dx, dv_dy, eps=1e-6).float()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/biochem/eda/probe_pocket_ranking.json")
    ap.add_argument("--min-clot", type=int, default=20)
    ap.add_argument(
        "--cand-pct",
        type=float,
        default=60.0,
        help="Candidate pockets = wall nodes below this percentile of hop-2 speed. "
             "Generous on purpose: a superset of what the model would predict.",
    )
    args = ap.parse_args()

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    phys = PhysicsConfig(phase="biochem")
    per: dict[str, dict] = {}
    agg: dict[str, list[float]] = {"h2min": [], "h2mean": [], "srmin": [], "size": []}

    print(f"{'vessel':>14} {'nT':>3} {'nF':>4} {'h2min':>7} {'h2mean':>7} {'srmin':>7} {'size':>7}")
    for p in sorted(ANCHOR_DIR.glob("patient*.pt")):
        name = p.stem
        if name.endswith("_mirror_y"):
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        if not hasattr(d, "mask_wall") or d.mask_wall is None:
            continue
        n = int(d.num_nodes)
        wall = d.mask_wall.reshape(-1).bool().numpy()
        T = int(d.y.shape[0])
        gt = (gt_clot_phi_at_time(d, T - 1, phys, device=torch.device("cpu")).reshape(-1) > 0.5).numpy()
        if int((gt & wall).sum()) < args.min_clot:
            continue
        row, col = d.edge_index
        u, v = d.y[0, :, 0].float(), d.y[0, :, 1].float()
        sp = torch.sqrt(u * u + v * v)
        h1 = _nz_hop(sp, row, col, n)
        h2 = _nz_hop(h1, row, col, n).numpy()
        try:
            sr = _nz_hop(_shear_rate(d, u, v, n), row, col, n).numpy()
        except Exception:
            sr = np.zeros(n)

        # Candidate pockets: low-flow wall nodes, generously thresholded.
        thr = np.percentile(h2[wall], args.cand_pct)
        cand = wall & (h2 <= thr)
        idx = np.where(cand)[0]
        if idx.size < 10:
            continue
        ei = d.edge_index.numpy()
        keep = np.isin(ei[0], idx) & np.isin(ei[1], idx)
        remap = {val: i for i, val in enumerate(idx)}
        rr = np.array([remap[x] for x in ei[0][keep]], dtype=int)
        cc = np.array([remap[x] for x in ei[1][keep]], dtype=int)
        if rr.size == 0:
            continue
        A = coo_matrix((np.ones(rr.size), (rr, cc)), shape=(idx.size, idx.size))
        ncomp, lab = connected_components(A, directed=False)

        feats = []
        for k in range(ncomp):
            nd = idx[lab == k]
            if nd.size < 3:
                continue
            feats.append({
                "istrue": bool(gt[nd].any()),
                "h2min": float(h2[nd].min()), "h2mean": float(h2[nd].mean()),
                "srmin": float(sr[nd].min()), "size": float(nd.size),
            })
        Tp = [f for f in feats if f["istrue"]]
        Fp = [f for f in feats if not f["istrue"]]
        if len(Tp) < 1 or len(Fp) < 3:
            continue
        row_out = {}
        for key in ("h2min", "h2mean", "srmin", "size"):
            a = _auc(np.array([f[key] for f in Tp]), np.array([f[key] for f in Fp]))
            row_out[key] = a
            if a == a:
                agg[key].append(a)
        per[name] = {"n_true": len(Tp), "n_false": len(Fp), **row_out}
        print(f"{name:>14} {len(Tp):3d} {len(Fp):4d} "
              f"{row_out['h2min']:7.3f} {row_out['h2mean']:7.3f} "
              f"{row_out['srmin']:7.3f} {row_out['size']:7.3f}")

    print(f"\n{'MEAN':>14} {'':3} {'':4} " + " ".join(
        f"{np.nanmean(agg[k]):7.3f}" for k in ("h2min", "h2mean", "srmin", "size")))
    print("(<0.5 = true pockets have LOWER value = physically expected for flow)")

    # Direction-agnostic strength, since flow features rank downward.
    print(f"\n{'|separation| from chance':>30}")
    strength = {}
    for k in ("h2min", "h2mean", "srmin", "size"):
        arr = np.array(agg[k])
        strength[k] = float(np.nanmean(np.abs(arr - 0.5) * 2))
        print(f"{k:>30}: {strength[k]:.3f}   (n={len(arr)} vessels)")

    best = max(strength, key=strength.get)
    verdict = (
        f"Component-level `{best}` separates true from false pockets by "
        f"{strength[best]:.3f} on average across {len(agg[best])} vessels. "
        + ("Pocket ranking is real and generalises." if strength[best] > 0.4
           else "Weak/inconsistent -- the patient020 result may not generalise.")
    )
    print(f"\n=> {verdict}")

    out = Path(args.out)
    if not out.is_absolute():
        out = get_project_root() / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"per_vessel": per, "mean_auc": {k: float(np.nanmean(agg[k])) for k in agg},
         "separation": strength, "cand_pct": args.cand_pct, "verdict": verdict},
        indent=2), encoding="utf-8")
    print(f"[save] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
