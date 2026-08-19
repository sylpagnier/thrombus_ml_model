"""What IS the off-wall problem, in time and per vessel? -- measure before modelling again.

Four modelling attempts at off-wall timing have now failed the same way (PHASE9 12.2's
owner-threshold rule; the per-vessel physics clock; the two-stage owner-onset head; and
before them the curve head of PHASE9 12.4).  Each was a plausible mechanism fitted before
anyone measured the quantity it was supposed to predict.  This measures it.

Three questions, in the order that decides what to build:

1. **Does off-wall clot lag its owner, and by how much?**  If the lag is concentrated, the
   owner constraint plus good wall timing nearly solves off-wall and the remaining work is on
   the wall.  If it is broad, the lag itself is the target.
2. **Is the loss early or late?**  A mask that fires too early and one that fires too late
   need opposite corrections, and the mean-over-time score hides which it is.
3. **Which vessels carry the loss?**  Off-wall means are taken over 13 vessels and the
   burdens span 4 to 122 nodes.

    python scripts/diag_offwall_structure.py --tags v5a,v5b,v5c --cache v5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.clot_ml.data import attach_physics, load_cache  # noqa: E402
from src.clot_ml.severity_metric import DEFAULT, SeverityScorer  # noqa: E402
from src.config import PhysicsConfig  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="v5a,v5b,v5c")
    ap.add_argument("--cache", default="v5")
    ap.add_argument("--masks", default="outputs/v4_set_masks.npz")
    ap.add_argument("--n-times", type=int, default=11)
    args = ap.parse_args()

    cache = attach_physics(load_cache(args.cache))
    phys = PhysicsConfig(phase="biochem")
    sets = {}
    mp = REPO / args.masks
    if mp.exists():
        z = np.load(mp)
        sets = {a: z[a].astype(bool) for a in z.files}

    print("%-11s %5s | %6s %6s %6s | %5s %5s | %s"
          % ("vessel", "n_off", "lag<=0", "lag=1", "lag>1", "early", "late", "score@t"))
    lags_all, rows = [], []
    for a in sorted(cache):
        S = cache[a]
        d = torch.load(PACKS / f"{a}.pt", map_location="cpu", weights_only=False)
        T = int(d.y.shape[0])
        times = [int(round(x)) for x in np.linspace(0, T - 1, args.n_times)]
        wall, owner = S["wall"], S["owner"]
        gt = {ti: (gt_clot_phi_at_time(d, ti, phys, device=torch.device("cpu"))
                   .reshape(-1).numpy() > 0.5) for ti in times}
        go = np.full(len(wall), args.n_times, dtype=int)      # GT onset, as a GRID index
        for j, ti in reversed(list(enumerate(times))):
            go[gt[ti]] = j
        off = (~wall) & (go < args.n_times)
        n_off = int(off.sum())
        if n_off == 0:
            continue
        # 1. lag of an off-wall node behind its own owner, in grid steps
        lag = go[off] - go[owner][off]
        lags_all.append(lag)

        # 2/3. where the committed set's own error sits, per timestep
        early = late = 0
        per_t = []
        if a in sets:
            m = sets[a]
            for j, ti in enumerate(times):
                sc = SeverityScorer(S["edge_index"], gt[ti], len(wall), DEFAULT)
                v = sc.score(m & ~wall, ~wall)
                per_t.append(v)
                # the FROZEN set at this time: predicted-not-GT vs GT-not-predicted
                early += int((m & ~wall & ~gt[ti]).sum())
                late += int((~m & ~wall & gt[ti]).sum())
        rows.append((a, n_off, float(np.mean(np.asarray(per_t, float)[
            ~np.isnan(np.asarray(per_t, float))])) if per_t else float("nan")))
        print("%-11s %5d | %5.0f%% %5.0f%% %5.0f%% | %5d %5d | %s"
              % (a, n_off,
                 100 * float((lag <= 0).mean()), 100 * float((lag == 1).mean()),
                 100 * float((lag > 1).mean()), early, late,
                 " ".join(("%.2f" % x) if x == x else "  - " for x in per_t)))

    L = np.concatenate(lags_all)
    print("\nOFF-WALL LAG BEHIND OWNER, all vessels pooled (n=%d nodes, grid steps)" % len(L))
    for q in (0, 10, 25, 50, 75, 90, 100):
        print("   p%-3d %+.1f" % (q, np.percentile(L, q)))
    print("   lag <= 0 : %.1f%%    lag == 1 : %.1f%%    lag >= 2 : %.1f%%"
          % (100 * (L <= 0).mean(), 100 * (L == 1).mean(), 100 * (L >= 2).mean()))

    ok = [r for r in rows if r[2] == r[2]]
    ok.sort(key=lambda r: r[2])
    print("\nMEAN-OVER-TIME off-wall score of the FROZEN committed set, worst first")
    for a, n, v in ok:
        print("   %-11s n_off %4d   %.3f" % (a, n, v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
