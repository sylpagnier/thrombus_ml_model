"""Score the SHIPPED temporal entry point across the pool, and stamp it into the manifest.

Scores exactly what `src.clot_ml.locked.predict_clot_series` returns -- no re-derivation --
so the number in the manifest is the number the entry point produces.

    python scripts/verify_temporal_shipped.py --update-manifest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.clot_ml.data import attach_physics, load_cache  # noqa: E402
from src.clot_ml.geometry_splits import classes_for, eligible_pool, is_priority  # noqa: E402
from src.clot_ml.locked import load_ensemble, predict_clot_series  # noqa: E402
from src.clot_ml.severity_metric import DEFAULT, SeverityScorer  # noqa: E402
from src.config import PhysicsConfig  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-times", type=int, default=11)
    ap.add_argument("--update-manifest", action="store_true")
    args = ap.parse_args()

    cache = attach_physics(load_cache("gt"))
    pool = [a for a in eligible_pool() if a in cache]
    classes = classes_for(pool, PACKS)
    phys = PhysicsConfig(phase="biochem")
    ens = load_ensemble()

    W, O, Wf, Of, per = [], [], [], [], {}
    for a in pool:
        S = cache[a]
        d = torch.load(PACKS / f"{a}.pt", map_location="cpu", weights_only=False)
        T = int(d.y.shape[0])
        times = [int(round(x)) for x in np.linspace(0, T - 1, args.n_times)]
        out = predict_clot_series(ens, d, times, sample=S)
        wall = S["wall"]
        ws, os_, wfs, ofs = [], [], [], []
        for ti in times:
            gt = (gt_clot_phi_at_time(d, ti, phys, device=torch.device("cpu"))
                  .reshape(-1).numpy() > 0.5)
            sc = SeverityScorer(S["edge_index"], gt, len(wall), DEFAULT)
            ws.append(sc.score(out["series"][ti], wall))
            os_.append(sc.score(out["series"][ti], ~wall))
            wfs.append(sc.score(out["mask"], wall))          # frozen reference
            ofs.append(sc.score(out["mask"], ~wall))
        per[a] = dict(cls=classes.get(a, "?"), wall=float(np.nanmean(ws)),
                      off=float(np.nanmean(os_)), wall_frozen=float(np.nanmean(wfs)),
                      off_frozen=float(np.nanmean(ofs)))
        W.append(per[a]["wall"]); O.append(per[a]["off"])
        Wf.append(per[a]["wall_frozen"]); Of.append(per[a]["off_frozen"])
        print("   %-11s %-9s wall %.4f (frozen %.4f)  off %s"
              % (a, classes.get(a, "?")[:9], per[a]["wall"], per[a]["wall_frozen"],
                 ("%.4f" % per[a]["off"]) if per[a]["off"] == per[a]["off"] else "n/a"),
              flush=True)

    prio = [a for a in pool if is_priority(classes.get(a, ""))]
    summary = dict(
        n_vessels=len(pool), n_times=args.n_times,
        temporal=dict(wall=float(np.nanmean(W)), off=float(np.nanmean(O))),
        frozen=dict(wall=float(np.nanmean(Wf)), off=float(np.nanmean(Of))),
        priority=dict(wall=float(np.nanmean([per[a]["wall"] for a in prio])),
                      off=float(np.nanmean([per[a]["off"] for a in prio]))),
        note=("mean-over-time domain-restricted severity deploy score; GT-empty times "
              "skipped. IN-SAMPLE for the locked weights (they train on the whole pool); "
              "the timing half adds no learned parameters."))
    print("\nSHIPPED temporal entry point, mean-over-time (n=%d)" % len(pool))
    print("   frozen mask      wall %.4f   off %.4f"
          % (summary["frozen"]["wall"], summary["frozen"]["off"]))
    print("   + ODE timing     wall %.4f   off %.4f"
          % (summary["temporal"]["wall"], summary["temporal"]["off"]))
    print("   priority class   wall %.4f   off %.4f"
          % (summary["priority"]["wall"], summary["priority"]["off"]))

    Path("outputs/verify_temporal_shipped.json").write_text(
        json.dumps(dict(summary=summary, per_vessel=per), indent=2, default=float))
    if args.update_manifest:
        ptr_p = REPO / "data/reference/clot_gnn_locked.json"
        ptr = json.loads(ptr_p.read_text())
        man_p = REPO / ptr["manifest"]
        man = json.loads(man_p.read_text())
        man["temporal"] = dict(
            entry_point="src.clot_ml.locked.predict_clot_series",
            cli="scripts/predict_clot_temporal.py",
            wall_timing="zero-parameter surface ODE first crossing of viscosity_mat_crit",
            off_wall_timing="owner trajectory reaching crit/OFF_ATT, OFF_ATT=0.80",
            scores=summary)
        man_p.write_text(json.dumps(man, indent=2))
        ptr["temporal"] = summary
        ptr_p.write_text(json.dumps(ptr, indent=2))
        print("\nstamped into %s and %s" % (man_p.name, ptr_p.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
