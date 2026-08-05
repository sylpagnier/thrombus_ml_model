"""Free probe: does hop-2/3 flow predict GT clot on wall nodes better than hop-1?

The multi-hop feature was derived from ONE model's error on ONE vessel (patient020's
distant FP pocket). Before spending GPU hours retraining on it, this asks the
model-independent question directly: across the cohort, does a classifier given hop-2/3
neighbourhood speed locate GT clot on wall nodes better than one given hop-1 alone?

Leave-one-VESSEL-out, so the reported AUC is genuinely held out -- an in-sample lift
here would be worth nothing. Mirrored vessels (`*_mirror_y`) are dropped: they are the
same vessel as their twin and would leak across the split.

Gate for the sweep: if `hop123` does not beat `hop1` on held-out AUC, the feature is not
the bottleneck and the retrain should not run.

CPU only, no model, no rollout.

    python scripts/probe_multihop_flow.py
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


def _nz_neighbor_mean(vals: torch.Tensor, row, col, n: int) -> torch.Tensor:
    v = vals.reshape(-1)[col]
    m = (v > 1e-9).to(dtype=v.dtype)
    s = torch.zeros(n, dtype=v.dtype)
    c = torch.zeros(n, dtype=v.dtype)
    s.index_add_(0, row, v * m)
    c.index_add_(0, row, m)
    return s / c.clamp(min=1.0)


def _auc(score: np.ndarray, label: np.ndarray) -> float:
    p, q = score[label], score[~label]
    if p.size == 0 or q.size == 0:
        return float("nan")
    a = np.concatenate([p, q]).astype(float)
    o = a.argsort()
    r = np.empty_like(a)
    r[o] = np.arange(1, a.size + 1)
    uq, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    s = np.zeros(uq.size)
    np.add.at(s, inv, r)
    r = (s / cnt)[inv]
    n1 = p.size
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * q.size))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/biochem/eda/probe_multihop.json")
    ap.add_argument("--min-clot", type=int, default=20)
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    phys = PhysicsConfig(phase="biochem")
    rows = []
    for p in sorted(ANCHOR_DIR.glob("patient*.pt")):
        name = p.stem
        if name.endswith("_mirror_y"):
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        if not hasattr(d, "mask_wall") or d.mask_wall is None:
            continue
        n = int(d.num_nodes)
        wall = d.mask_wall.reshape(-1).bool()
        T = int(d.y.shape[0])
        gt = (gt_clot_phi_at_time(d, T - 1, phys, device=torch.device("cpu")).reshape(-1) > 0.5)
        if int((gt & wall).sum()) < args.min_clot:
            continue
        row, col = d.edge_index
        sp = torch.sqrt(d.y[0, :, 0] ** 2 + d.y[0, :, 1] ** 2).float()
        h1 = _nz_neighbor_mean(sp, row, col, n)
        h2 = _nz_neighbor_mean(h1, row, col, n)
        h3 = _nz_neighbor_mean(h2, row, col, n)
        w = wall.numpy()
        X = np.stack([np.log1p(h.numpy()) for h in (h1, h2, h3)], axis=1)[w]
        y = gt.numpy()[w]
        rows.append({"name": name, "X": X, "y": y})
        print(f"[calc] {name:14s} wall={w.sum():5d} clot={int(y.sum()):4d}", flush=True)

    if len(rows) < 4:
        print("[ERROR] too few vessels")
        return 1

    sets = {"hop1": [0], "hop2": [1], "hop12": [0, 1], "hop123": [0, 1, 2]}
    results: dict[str, dict] = {}
    print(f"\nLeave-one-VESSEL-out held-out AUC (n={len(rows)} vessels)")
    print(f"{'vessel':>14} " + " ".join(f"{k:>8}" for k in sets))
    per_vessel: dict[str, dict[str, float]] = {}
    for i, r in enumerate(rows):
        tr = [rows[j] for j in range(len(rows)) if j != i]
        Xtr = np.concatenate([t["X"] for t in tr])
        ytr = np.concatenate([t["y"] for t in tr])
        line = f"{r['name']:>14} "
        per_vessel[r["name"]] = {}
        for k, cols in sets.items():
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced"),
            )
            clf.fit(Xtr[:, cols], ytr)
            s = clf.predict_proba(r["X"][:, cols])[:, 1]
            a = _auc(s, r["y"])
            per_vessel[r["name"]][k] = a
            line += f"{a:8.4f} "
        print(line, flush=True)

    print(f"\n{'MEAN':>14} " + " ".join(
        f"{np.nanmean([per_vessel[v][k] for v in per_vessel]):8.4f}" for k in sets))
    means = {k: float(np.nanmean([per_vessel[v][k] for v in per_vessel])) for k in sets}
    # Compare against the BEST multi-hop set, not a fixed one -- hop3 measurably hurts
    # (it over-smooths), so gating on hop123 would understate the feature.
    best = max((k for k in sets if k != "hop1"), key=lambda k: means[k])
    lift = means[best] - means["hop1"]
    wins = sum(1 for v in per_vessel if per_vessel[v][best] > per_vessel[v]["hop1"])

    # Sign-oracle ceiling: AUC far BELOW 0.5 is signal too, just inverted. Vessels split
    # into a stagnation regime (clot in slow flow) and an inverted one (clot in fast
    # flow). A global linear probe cannot serve both; a GNN conditioning on z_kin +
    # geometry can, so the plain mean understates what training can extract.
    arr = np.array([per_vessel[v][best] for v in per_vessel])
    oracle = float(np.maximum(arr, 1 - arr).mean())
    stagnation = [v for v in per_vessel if per_vessel[v][best] > 0.7]
    inverted = [v for v in per_vessel if per_vessel[v][best] < 0.4]

    print(f"\nbest multi-hop set = {best}")
    print(f"{best} - hop1 = {lift:+.4f}   ({best} wins on {wins}/{len(per_vessel)} vessels)")
    print(f"oracle-sign mean = {oracle:.4f}   stagnation={len(stagnation)} inverted={len(inverted)}")

    if lift > 0.03 or oracle > 0.70:
        verdict = (
            f"GO -- {best} lifts held-out AUC {lift:+.3f} over hop1 ({means['hop1']:.3f}, "
            f"i.e. chance), oracle-sign ceiling {oracle:.3f}. Retrain justified."
        )
    elif lift > 0.0:
        verdict = "MARGINAL -- small lift. Retrain is a reasonable bet but not a strong one."
    else:
        verdict = "STOP -- no held-out lift. Do not spend GPU on the multi-hop retrain."
    print(f"=> {verdict}")

    results = {"means": means, "per_vessel": per_vessel, "best_set": best,
               "lift_vs_hop1": lift, "oracle_sign_mean": oracle,
               "stagnation_regime": stagnation, "inverted_regime": inverted,
               "n_vessels": len(per_vessel), "wins": wins, "verdict": verdict}
    out = Path(args.out)
    if not out.is_absolute():
        out = get_project_root() / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[save] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
