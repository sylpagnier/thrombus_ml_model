"""Compare 1) the ML compound model, 2) the Phase-6 physics wall+lumen model.

Both scored FULL-MESH (no wall mask), same eval convention, same cohort as Phase 6
(FIT+DEV, SEALED opened once for this new question and disclosed).

MODEL 1 -- the locked ML compound: WC_v7 wall species GNN + a second GNN
(``SpeciesDualHeadContinuousGNN``) that grows off-wall clot in the frontier of the wall
net's OWN predicted committed Mat, all inside one coupled continuous-time rollout
(``src/inference/customer_pipeline.py::CustomerDeployPipeline``,
``data/reference/mat_compound_deploy.json``).

MODEL 2 -- ``scripts/predict_wall_clot.py`` with ``lumen=True``: the Phase-6 physics wall
mask (two t=0 gates + graph growth) extended by the retuned algebraic lumen rule
(``grow_into_lumen``, hops=2, speed<0.2).  Zero learned parameters.

WHY "PHYSICS WALL + THE TRAINED SPECIALIST" IS NOT ATTEMPTED HERE.  The specialist's growth
zone (``_frontier_nucleation_mask``) and its input features are read from the SAME live
rollout state the wall net is simultaneously writing -- there is no seam where an external
mask can be substituted without either (a) modifying the coupled rollout's internal
per-step state update to inject a physics-driven Mat trajectory in place of the wall net's,
which is a real re-plumbing exercise carrying its own OOD risk (the specialist was trained
on the WC_v7 net's specific error distribution, not on a physics ODE's), or (b) retraining
the specialist against physics-driven wall trajectories from scratch.  Neither is attempted
in this script; see the accompanying report for the scoped follow-up.

    python scripts/eval_compound_vs_physics.py --smoke        # 2 vessels, sanity check
    python scripts/eval_compound_vs_physics.py                # full FIT+DEV, then SEALED
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.species_pushforward_continuous import (  # noqa: E402
    resolve_deploy_eval_time_index,
)
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.evaluation.clot_relaxed_metrics import (  # noqa: E402
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
)
from src.inference.customer_pipeline import CustomerDeployPipeline  # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))
from predict_wall_clot import predict_wall_clot  # noqa: E402

DIR = Path("data/processed/graphs_biochem_anchors")
OUT = Path("outputs/rollout_trackA")


def score_full_mesh(pred: np.ndarray, gt: torch.Tensor, edge_index) -> float:
    m = compute_clot_relaxed_metrics(torch.tensor(pred.astype(np.float32)), gt, edge_index)
    return clot_score_from_deploy_dict(metrics_to_deploy_prefix(m))


def run_one(pipeline, d, bio, phys) -> dict | None:
    nt = int(d.y.shape[0])
    te = resolve_deploy_eval_time_index(nt)
    gt = gt_clot_phi_at_time(d, te, phys, device=torch.device("cpu")).reshape(-1)
    if float((gt > 0.5).sum()) == 0:
        return None
    t_eval_s = float(d.t.reshape(-1)[te].item())

    # ---- model 2: Phase-6 physics wall+lumen (zero learned parameters)
    p_wall, _ = predict_wall_clot(d, bio, flow="gt", lumen=False)
    p_lumen, _ = predict_wall_clot(d, bio, flow="gt", lumen=True)

    # ---- model 1: the ML compound (WC_v7 wall + learned lumen specialist)
    t0 = time.time()
    traj = pipeline.run(d, t_final_s=float(d.t.reshape(-1)[-1].item()), include_velocity=False)
    t_keys = sorted(traj.phi.keys())
    t_secs = np.asarray(traj.t_sec)
    ti = t_keys[int(np.argmin(np.abs(t_secs[t_keys] - t_eval_s)))] if len(t_keys) else None
    phi = traj.phi[ti] if ti is not None else np.zeros(int(d.num_nodes))
    p_compound = phi > 0.5
    dt = time.time() - t0

    return dict(
        n_gt=int((gt > 0.5).sum()),
        n_gt_wall=int(((gt > 0.5) & d.mask_wall.reshape(-1).bool()).sum()),
        n_gt_off=int(((gt > 0.5) & ~d.mask_wall.reshape(-1).bool()).sum()),
        s_physics_wall=score_full_mesh(p_wall, gt, d.edge_index),
        s_physics_lumen=score_full_mesh(p_lumen, gt, d.edge_index),
        s_compound=score_full_mesh(p_compound.astype(bool), gt, d.edge_index),
        n_wall=int(p_wall.sum()), n_lumen=int(p_lumen.sum()), n_compound=int(p_compound.sum()),
        compound_wall_time=float(t_secs[ti]) if ti is not None else float("nan"),
        target_time=t_eval_s, elapsed_s=dt,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--sealed", action="store_true", help="also run on SEALED (spends it)")
    args = ap.parse_args()
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    prot = json.load(open("outputs/ap_closure/protocol_gt_meanovertime.json"))
    names = prot["fit"] + prot["dev"]
    if args.smoke:
        names = names[:2]
    OUT.mkdir(parents=True, exist_ok=True)

    print("[i] loading compound pipeline (WC_v7 wall + lumen specialist)...")
    t0 = time.time()
    pipeline = CustomerDeployPipeline()
    pipeline._ensure_loaded()
    print("[i] loaded in %.0fs" % (time.time() - t0))

    def run_cohort(cohort_names, tag):
        rows = {}
        for n in cohort_names:
            p = DIR / f"{n}.pt"
            if not p.exists():
                continue
            d = torch.load(p, map_location="cpu", weights_only=False)
            r = run_one(pipeline, d, bio, phys)
            if r is None:
                continue
            rows[n] = r
            print("%-12s gt=%3d (wall %3d off %3d) | physics_wall %.4f  +lumen %.4f  | "
                  "compound %.4f  (t=%.0fs of %.0fs target, %.1fs)"
                  % (n, r["n_gt"], r["n_gt_wall"], r["n_gt_off"], r["s_physics_wall"],
                     r["s_physics_lumen"], r["s_compound"], r["compound_wall_time"],
                     r["target_time"], r["elapsed_s"]))
        if rows:
            pw = np.mean([r["s_physics_wall"] for r in rows.values()])
            pl = np.mean([r["s_physics_lumen"] for r in rows.values()])
            sc = np.mean([r["s_compound"] for r in rows.values()])
            print("\n%s  n=%d   mean full-mesh score" % (tag, len(rows)))
            print("   physics wall-only        %.4f" % pw)
            print("   physics wall+lumen        %.4f  (%+.4f vs wall-only)" % (pl, pl - pw))
            print("   ML compound (WC_v7+spec)  %.4f  (%+.4f vs physics wall+lumen)"
                  % (sc, sc - pl))
        return rows

    r_train = run_cohort(names, "TRAIN (FIT+DEV)")
    r_sealed = run_cohort(prot["sealed"], "SEALED") if args.sealed else {}

    (OUT / "compound_vs_physics.json").write_text(json.dumps(
        dict(train=r_train, sealed=r_sealed), indent=2, default=float), encoding="utf-8")
    print("\nwrote %s" % (OUT / "compound_vs_physics.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
