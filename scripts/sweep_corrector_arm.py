"""Calibrate the flow-coupled arm's seeding ramp on TRAIN, then spend dev/SEALED once.

Scored with the CANONICAL wall-masked ``deploy_clot_score`` (``clot_score_from_deploy_dict``
over ``compute_clot_relaxed_metrics``), the same protocol as
``scripts/report_phase3_results.py`` -- NOT the ``growth_count_metrics`` wall F1 the
rollout diagnostics used.  The two are different numbers and only this one is comparable to
docs/PHASE3_RESULTS.md.

``seed_ramp`` is the one new scalar.  ``0`` = no seeding (the corrector waits for the ODE to
commit something, which is the configuration that produced the original clean negative);
``r`` seeds the whole t=0 predicted mask by ``1/r`` of the horizon, strongest gates first.

    python scripts/sweep_corrector_arm.py                     # TRAIN sweep only
    python scripts/sweep_corrector_arm.py --spend --ramp 2.0   # dev + SEALED, once
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.biochem_gnn.mat_growth_simple import (  # noqa: E402
    WALL_COHORT_V2_GENERALIZATION, WALL_COHORT_V2_TRAIN,
)
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.physics_wall_model import (  # noqa: E402
    CorrectorArm, predict_corrector, predicted_seed_mask, t0_flow_fields,
)
from src.core_physics.species_pushforward_continuous import (  # noqa: E402
    resolve_deploy_eval_time_index,
)
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.evaluation.clot_relaxed_metrics import (  # noqa: E402
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
    scoring_fingerprint,
)

DIR = Path("data/processed/graphs_biochem_anchors")
CKPT = Path("outputs/kinematics/local_corrector/local_kinematic_corrector_best.pth")
DEV_HOLDOUT = ("patient042", "patient043")
STENCIL = {"gt": 3, "pred": 4}


def score(data, mask, gt, wall):
    m = compute_clot_relaxed_metrics(
        torch.tensor(mask.astype(np.float32)), gt, data.edge_index,
        wall_mask=torch.tensor(wall))
    o = metrics_to_deploy_prefix(m)
    return (clot_score_from_deploy_dict(o), o["deploy_clot_f1"],
            o["deploy_clot_relaxed_prec"], o["deploy_clot_relaxed_rec"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ramps", default="0,0.5,1.0,1.5,2.0,3.0")
    ap.add_argument("--dmu", type=float, default=0.68)
    ap.add_argument("--flow", default="gt", choices=["gt", "pred"])
    ap.add_argument("--spend", action="store_true", help="score dev + SEALED (spend once)")
    ap.add_argument("--ramp", type=float, default=None, help="fixed ramp for --spend")
    ap.add_argument("--out", default="outputs/corrector_arm_sweep.json")
    args = ap.parse_args()

    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from src.core_physics.coupled_shear_gnn import load_local_corrector
    corr = load_local_corrector(CKPT, device)
    print("SCORING FINGERPRINT %s" % scoring_fingerprint())
    print("flow=%s  delta_mu=%.2f Pa.s  device=%s  stencil=%d\n"
          % (args.flow, args.dmu, device, STENCIL[args.flow]))

    if args.spend:
        groups = {"dev-holdout": DEV_HOLDOUT, "SEALED": WALL_COHORT_V2_GENERALIZATION}
        ramps = [float(args.ramp if args.ramp is not None else 2.0)]
        print("*** SPENDING dev + SEALED at seed_ramp=%.2f ***\n" % ramps[0])
    else:
        groups = {"train": WALL_COHORT_V2_TRAIN}
        ramps = [float(x) for x in args.ramps.split(",") if x.strip()]

    names = sorted({a for g in groups.values() for a in g})
    packs = {}
    for a in names:
        p = DIR / f"{a}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        try:
            f = t0_flow_fields(d, bio, hops=STENCIL[args.flow], flow_source=args.flow)
        except ValueError:
            continue
        wall = d.mask_wall.reshape(-1).bool().numpy()
        t_eval = resolve_deploy_eval_time_index(int(d.y.shape[0]))
        gt = gt_clot_phi_at_time(d, t_eval, phys, device=torch.device("cpu")).reshape(-1)
        gt = gt * torch.tensor(wall.astype(np.float32))
        seed, _, _ = predicted_seed_mask(d, bio, f)
        packs[a] = dict(d=d, f=f, wall=wall, gt=gt, seed=seed)
    print("loaded %d packs\n" % len(packs))

    # shipped static baseline: the t=0 gates + shear-admitted graph growth
    base = {a: score(c["d"], c["seed"], c["gt"], c["wall"]) for a, c in packs.items()}
    for g, mem in groups.items():
        v = [base[a][0] for a in mem if a in base]
        print("  BASELINE static (shipped)  %-12s n=%2d  score %.4f  (>=0.6 %d/%d)"
              % (g, len(v), float(np.mean(v)), sum(x >= 0.6 for x in v), len(v)))

    rows, t0 = [], time.time()
    print("\n%7s | %s" % ("ramp", "  ".join("%-30s" % g for g in groups)))
    for r in ramps:
        arm = CorrectorArm(corrector=corr, phys_cfg=phys, device=device,
                           delta_mu=args.dmu, seed_ramp=r)
        res = {}
        for a, c in packs.items():
            mask, _, _, calls = predict_corrector(
                c["d"], bio, arm, hops=STENCIL[args.flow], flow_source=args.flow)
            res[a] = score(c["d"], mask, c["gt"], c["wall"]) + (int(mask.sum()), calls)
        cells = []
        for g, mem in groups.items():
            v = [res[a] for a in mem if a in res]
            cells.append("%.4f (>=0.6 %2d/%2d) dF %+0.3f"
                         % (np.mean([x[0] for x in v]), sum(x[0] >= 0.6 for x in v), len(v),
                            np.mean([x[1] for x in v])
                            - np.mean([base[a][1] for a in mem if a in base])))
        rows.append(dict(ramp=r, per_vessel={a: list(x) for a, x in res.items()},
                         groups={g: float(np.mean([res[a][0] for a in mem if a in res]))
                                 for g, mem in groups.items()}))
        print("%7.2f | %s" % (r, "  ".join("%-30s" % c for c in cells)))

    print("\n(%.0f s)" % (time.time() - t0))
    if not args.spend:
        gkey = "train"
        best = max(rows, key=lambda z: z["groups"][gkey])
        print("\nBEST on TRAIN: seed_ramp=%.2f -> %.4f (static baseline %.4f)"
              % (best["ramp"], best["groups"][gkey],
                 float(np.mean([base[a][0] for a in WALL_COHORT_V2_TRAIN if a in base]))))
        print("Now: python scripts/sweep_corrector_arm.py --spend --ramp %.2f" % best["ramp"])
    else:
        print("\nper-vessel at seed_ramp=%.2f" % ramps[0])
        print("%12s %8s %9s %9s %8s" % ("vessel", "split", "static", "corrector", "delta"))
        for g, mem in groups.items():
            for a in mem:
                if a not in packs:
                    continue
                s, c = base[a][0], rows[0]["per_vessel"][a][0]
                print("%12s %8s %9.4f %9.4f %+8.4f" % (a, g, s, c, c - s))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        dict(flow=args.flow, dmu=args.dmu, spend=args.spend,
             baseline={a: list(v) for a, v in base.items()}, rows=rows),
        indent=2, default=float), encoding="utf-8")
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
