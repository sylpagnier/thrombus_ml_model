"""PHASE 8: the flow oracle and the removal term, crossed, on the metric of record.

WHY THIS EXISTS.  ``docs/PHASE7_FINDINGS.md`` 7 concluded that ordering is NOT flow-limited:
a perfect evolving-flow oracle -- the gate recomputed from GT velocity at every step, the
ceiling on any flow model including RGP-DEQ and the corrector -- moved ``Mat`` ordering only
0.534 -> 0.632 and the deploy score by +0.0003.  That killed the flow arm.

``scripts/diag_mat_washout.py`` says that conclusion was measured in a model that structurally
cannot use evolving flow.  The surface ODE accumulates and never removes, so shear enters only
through which gates are open; there is no channel through which a *change* in shear can take
material away.  Crossing inputs against removal on oracle chemistry:

                          accumulate-only   with washout
    frozen t=0 inputs               0.219          0.097
    time-varying inputs             0.310          0.464

Evolving inputs alone buy +0.091.  Removal alone COSTS 0.122.  Together they buy +0.245 --
a strong positive interaction, so neither term can be evaluated with the other switched off,
and FINDINGS 7 evaluated exactly that.

This script runs the same cross on the real model path and the canonical deploy score, so the
question "does the flow arm pay once the equation can use it" gets an answer in the units the
project is graded in.

ORACLE, NOT DEPLOYABLE.  The time-varying arms read GT velocity (via the wall cache's
``sr_t``/``dsrx_t``) at every timestep.  Per AGENTS.md's generalization policy that is a
CEILING and must never be quoted as a generalization result -- it says how much a perfect
flow model could be worth, which is exactly what decides whether to invest in the corrector.

    python scripts/eval_flow_washout_2x2.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from predict_wall_clot import node_pos, predict_wall_clot  # noqa: E402
from src.biochem_gnn.mat_growth_simple import WALL_COHORT_V2_TRAIN  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.ap_closure import SHIPPED, SHIPPED_DA_SCALE, make_rollout_hook  # noqa: E402
from src.core_physics.physics_lumen_model import (  # noqa: E402
    fill_grown_wall_mat, grow_into_lumen_by_mat, midside_nodes,
)
from src.core_physics.physics_wall_model import (  # noqa: E402
    WASHOUT_LAMBDA, gate_from_shear, integrate_mat_trajectory, t0_flow_fields,
)
from src.core_physics.species_pushforward_continuous import (  # noqa: E402
    resolve_deploy_eval_time_index,
)
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.core_physics.temporal_metrics import spearman  # noqa: E402
from src.evaluation.clot_relaxed_metrics import (  # noqa: E402
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
)

DIR = REPO / "data/processed/graphs_biochem_anchors"
CACHE = REPO / "outputs/wall_species_cache"
MAT_S = 7e10
FILL_HOPS = 6
ARMS = ("frozen gate, no removal", "frozen gate, washout",
        "GT-flow gate, no removal", "GT-flow gate, washout")


def f1(pred, gt):
    if gt.sum() == 0 and pred.sum() == 0:
        return float("nan")
    tp = int((pred & gt).sum())
    p, r = tp / max(int(pred.sum()), 1), tp / max(int(gt.sum()), 1)
    return 2 * p * r / max(p + r, 1e-9)


def gate_series_from_cache(z, d, bio):
    """``[T, N]`` gate on the full mesh, GT shear at every step, wall nodes from the cache."""
    n = int(d.num_nodes)
    nt = z["sr_t"].shape[0]
    out = np.zeros((nt, n))
    out[:, z["wall_idx"]] = gate_from_shear(z["sr_t"], z["dsrx_t"], bio)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default="outputs/phase8_flow_washout_2x2.json")
    args = ap.parse_args()
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    crit = float(bio.viscosity_mat_crit)
    acc = {a: {"score": [], "rho": [], "off_f1": [], "wall_f1": []} for a in ARMS}
    per_vessel = {}

    for anchor in WALL_COHORT_V2_TRAIN:
        pk, cf = DIR / f"{anchor}.pt", CACHE / f"{anchor}.npz"
        if not (pk.exists() and cf.exists()):
            continue
        z = np.load(cf)
        if "sr_t" not in z.files:
            continue
        d = torch.load(pk, map_location="cpu", weights_only=False)
        if int(d.y.shape[0]) < 150:
            continue
        wall = d.mask_wall.reshape(-1).bool().numpy()
        ei_np = d.edge_index.detach().cpu().numpy()
        n = len(wall)
        A = sp.coo_matrix((np.ones(ei_np.shape[1]), (ei_np[0], ei_np[1])), shape=(n, n)).tocsr()
        A = ((A + A.T) > 0).astype(np.int8)
        t_eval = resolve_deploy_eval_time_index(int(d.y.shape[0]))
        gt = gt_clot_phi_at_time(d, t_eval, phys, device=torch.device("cpu")
                                 ).reshape(-1).numpy() > 0.5
        if gt.sum() == 0:
            continue
        names = d.y_channel_names.split(",")
        mat_gt = np.expm1(d.y[-1, :, names.index("Mat_log1p_nd")].double().numpy()) * MAT_S
        ms = midside_nodes(node_pos(d), ei_np)
        f = t0_flow_fields(d, bio, hops=3, flow_source="gt")
        base, _ = predict_wall_clot(d, bio, flow="gt", lumen=False)
        gser = gate_series_from_cache(z, d, bio)
        # The GT-flow arms also need the shear that drives removal to evolve, else the
        # oracle is only half applied.  sr on the full mesh, wall nodes from the cache.
        sr_series = np.zeros_like(gser)
        sr_series[:, z["wall_idx"]] = z["sr_t"]
        hook = make_rollout_hook(SHIPPED, bio, f.sr)
        per_vessel[anchor] = {}

        for arm in ARMS:
            tv = arm.startswith("GT-flow")
            lam = WASHOUT_LAMBDA if "washout" in arm else 0.0
            blockage = (lambda m, g0, i, _g=gser: _g[i] * wall) if tv else None
            wsr = sr_series.max(0) if tv else f.sr
            traj, _ = integrate_mat_trajectory(
                d, bio, f.gate * wall, da_scale=SHIPPED_DA_SCALE, ap_closure=hook,
                blockage=blockage, washout=lam, washout_sr=wsr)
            mat_m = traj[-1]
            corner = wall & ~ms & (mat_gt > 0) & (mat_m > 0)
            rho = spearman(mat_m[corner], mat_gt[corner]) if corner.sum() > 8 else np.nan
            mw = fill_grown_wall_mat(mat_m, base, wall, A, hops=FILL_HOPS)
            pred = base | grow_into_lumen_by_mat(mw, wall, node_pos(d), ei_np, crit)
            m = compute_clot_relaxed_metrics(torch.tensor(pred.astype(np.float32)),
                                             torch.tensor(gt.astype(np.float32)),
                                             d.edge_index)
            s = float(clot_score_from_deploy_dict(metrics_to_deploy_prefix(m)))
            acc[arm]["score"].append(s)
            acc[arm]["rho"].append(float(rho))
            acc[arm]["off_f1"].append(f1(pred & ~wall, gt & ~wall))
            acc[arm]["wall_f1"].append(f1(pred & wall, gt & wall))
            per_vessel[anchor][arm] = dict(score=s, rho=float(rho))
        print("[ok] %-12s %s" % (anchor, "  ".join(
            "%s=%.4f" % (a.split(", ")[1][:4], per_vessel[anchor][a]["score"]) for a in ARMS)))

    print("\n=== FLOW ORACLE x REMOVAL, %d train vessels (ORACLE -- not a deploy claim) ==="
          % len(per_vessel))
    print("   %-26s %11s %10s %10s %10s" % ("arm", "rho_corner", "off F1", "wall F1", "score"))
    for arm in ARMS:
        r = acc[arm]
        print("   %-26s %11.3f %10.4f %10.4f %10.4f"
              % (arm, np.nanmean(r["rho"]), np.nanmean(r["off_f1"]),
                 np.nanmean(r["wall_f1"]), np.nanmean(r["score"])))

    b = np.nanmean(acc[ARMS[0]]["score"])
    print("\n   score deltas against the frozen accumulate-only model (%.4f):" % b)
    for arm in ARMS[1:]:
        print("      %-26s %+.4f" % (arm, np.nanmean(acc[arm]["score"]) - b))
    r0 = np.nanmean(acc[ARMS[0]]["rho"])
    print("   rho_corner deltas (%.3f):" % r0)
    for arm in ARMS[1:]:
        print("      %-26s %+.3f" % (arm, np.nanmean(acc[arm]["rho"]) - r0))

    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        summary={a: {k: float(np.nanmean(v)) for k, v in acc[a].items()} for a in ARMS},
        per_vessel=per_vessel), indent=2))
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
