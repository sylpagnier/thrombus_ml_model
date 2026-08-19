"""Cache per-node features + targets for the PHASE9 clot-ML stack.

    python scripts/build_clot_ml_cache.py --flow gt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.clot_ml.features import build_features, feature_matrix  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.wall_cohort_splits import DEV, FIT, MIN_T  # noqa: E402

DIR = REPO / "data/processed/graphs_biochem_anchors"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", default="gt", choices=["gt", "pred"])
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out = Path(args.out or f"outputs/clot_ml_cache_{args.flow}")
    out.mkdir(parents=True, exist_ok=True)
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")

    todo = list(FIT) + list(DEV)
    for a in todo:
        p = DIR / f"{a}.pt"
        dst = out / f"{a}.npz"
        if dst.exists():
            print("[skip] %s" % a, flush=True)
            continue
        if not p.exists():
            print("[miss] %s" % a, flush=True)
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        if int(d.y.shape[0]) < MIN_T:
            print("[drop] %s T=%d" % (a, int(d.y.shape[0])), flush=True)
            continue
        t0 = time.time()
        try:
            S = build_features(d, bio, phys, flow=args.flow)
        except Exception as e:  # noqa: BLE001
            print("[ERR ] %s %s" % (a, e), flush=True)
            continue
        if S["y"].sum() == 0:
            print("[drop] %s empty GT" % a, flush=True)
            continue
        X, cols = feature_matrix(S["F"])
        np.savez_compressed(
            dst, X=X, cols=np.array(cols), y=S["y"], mat_gt=S["mat_gt"],
            wall=S["wall"], shell=S["shell"], owner=S["owner"],
            edge_index=S["edge_index"], pos=S["pos"], mat_phys=S["mat_phys"],
            gate=S["gate"], sr=S["sr"], spd=S["spd"], u=S["u"], v=S["v"])
        print("[ok  ] %-12s n=%6d feats=%d clot=%5d (off %4d)  %.1fs"
              % (a, S["n"], X.shape[1], int(S["y"].sum()),
                 int((S["y"] > 0.5).sum() - ((S["y"] > 0.5) & S["wall"]).sum()),
                 time.time() - t0), flush=True)
    print("done -> %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
