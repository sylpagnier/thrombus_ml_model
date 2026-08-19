"""Entry point: predicted clot mask at ANY time.  Follows the locked pointer, so this
automatically runs whichever generation is currently shipped (v2 ODE-timing or v3
time-conditioned) without the caller needing to know which.

Current shipped generation is v3 (docs/PHASE9_ML.md 13.9), mean-over-time out-of-fold,
19 vessels:

    frozen mask (was shipped)   wall 0.7953   off 0.4209
    v2 + ODE wall timing        wall 0.8547   off 0.5369
    v3 time-conditioned         wall 0.8845   off 0.6110     <- this entry point
    perfect timing              wall 0.9705   off 0.8396

    python scripts/predict_clot_temporal.py --anchor patient020
    python scripts/predict_clot_temporal.py --anchor patient020 --times 0,50,100,150,200 --score
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

from src.clot_ml.locked import load_default, predict_default_series  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--times", default="", help="comma list of time INDICES; default 11 evenly")
    ap.add_argument("--flow", default="gt", choices=["gt", "pred"])
    ap.add_argument("--score", action="store_true", help="also score against GT")
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    path = PACKS / f"{args.anchor}.pt"
    if not path.exists():
        print("[ERR] no pack at %s" % path)
        return 1
    data = torch.load(path, map_location="cpu", weights_only=False)
    T = int(data.y.shape[0])
    times = ([int(x) for x in args.times.split(",") if x.strip()] if args.times
             else [int(round(v)) for v in np.linspace(0, T - 1, 11)])

    bundle, kind = load_default()
    out = predict_default_series(bundle, kind, data, times, flow=args.flow)
    wall = data.mask_wall.reshape(-1).bool().numpy()
    t_s = data.t.reshape(-1).numpy()

    print("vessel %s  model=%s (%s)  T=%d" % (args.anchor, bundle["manifest"]["name"], kind, T))
    print("%8s %10s | %8s %8s" % ("t_index", "t [s]", "wall", "off-wall"))
    for ti in times:
        m = out["series"][ti]
        print("%8d %10.0f | %8d %8d" % (ti, t_s[min(ti, T - 1)], int((m & wall).sum()),
                                        int((m & ~wall).sum())))

    if args.score:
        from src.clot_ml.severity_metric import DEFAULT, SeverityScorer
        from src.core_physics.t0_mu_physics import gt_clot_phi_at_time
        phys = PhysicsConfig(phase="biochem")
        ws, os_ = [], []
        print("\n%8s | %8s %8s" % ("t_index", "wall", "off"))
        for ti in times:
            gt = (gt_clot_phi_at_time(data, ti, phys, device=torch.device("cpu"))
                  .reshape(-1).numpy() > 0.5)
            sc = SeverityScorer(data.edge_index.numpy(), gt, len(wall), DEFAULT)
            w = sc.score(out["series"][ti], wall)
            o = sc.score(out["series"][ti], ~wall)
            ws.append(w)
            os_.append(o)
            f = lambda x: ("%8.4f" % x) if x == x else "     n/a"
            print("%8d | %s %s" % (ti, f(w), f(o)))
        print("\nmean-over-time  wall %.4f   off %.4f"
              % (np.nanmean(ws), np.nanmean(os_)))

    if args.save:
        np.savez_compressed(args.save, onset=out["onset"], mask=out["mask"],
                            score=out["score"], times=np.array(times), wall=wall)
        print("\nwrote %s" % args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
