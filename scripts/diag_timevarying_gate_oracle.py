"""ORACLE CEILING for arm 3: what would a PERFECT evolving-flow model buy?

Arm 3 would condition the flow corrector on the evolving Mat/viscosity field and call it
inside the integration loop.  Before paying for that, bound what it could possibly win:
integrate the same ODE but re-evaluate the gates from the **GT velocity field at the
current timestep** -- i.e. a flow model with zero error, which no learned corrector can
beat.  Illegal as a model, decisive as a ceiling.

If the oracle's onset timing is no better than the frozen-t=0 gate's, arm 3 has nothing
to win and should not be built.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.biochem_gnn.mat_growth_simple import (  # noqa: E402
    WALL_COHORT_V2_GENERALIZATION, WALL_COHORT_V2_TRAIN,
)
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.mls_gradient import build_mls_gradient, node_positions, shear_rate_2d  # noqa: E402
from src.core_physics.physics_wall_model import (  # noqa: E402
    T0Fields, first_crossing, graded_gate, integrate_mat_trajectory, t0_flow_fields,
)
from src.core_physics.species_pushforward_continuous import resolve_deploy_eval_time_index  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.core_physics.temporal_metrics import curve_l1, gt_onset_index, onset_metrics  # noqa: E402
from src.evaluation.clot_relaxed_metrics import (  # noqa: E402
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
)

DIR = Path("data/processed/graphs_biochem_anchors")
M_TO_CM = 100.0


def gt_gate_series(d, bio, wall, mode, tau_low):
    """Gate recomputed from the GT velocity at EVERY timestep (oracle)."""
    pos = node_positions(d)
    Dx, Dy = build_mls_gradient(pos, d.edge_index.numpy(), hops=3)
    u_ref = float(d.u_ref.reshape(-1)[0])
    d_bar = float(d.d_bar.reshape(-1)[0])
    lss = float(bio.lss)
    sgt = float(bio.sgt) / M_TO_CM
    nt = int(d.y.shape[0])
    gates = np.zeros((nt, len(wall)))
    for ti in range(nt):
        u = d.y[ti, :, 0].numpy().astype(np.float64)
        v = d.y[ti, :, 1].numpy().astype(np.float64)
        sr = shear_rate_2d(Dx @ u, Dy @ u, Dx @ v, Dy @ v) * (u_ref / d_bar)
        dsrx = (Dx @ sr) / (d_bar * M_TO_CM)
        f = T0Fields(sr=sr, dsrx=dsrx,
                     gate_low=(sr < lss).astype(np.float64),
                     gate_sep=(dsrx < sgt).astype(np.float64), gate=None)
        gates[ti] = graded_gate(f, bio, mode=mode, tau_low=tau_low, tau_sep=0.0) * wall
    return gates


def main() -> int:
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    names = sorted(set(WALL_COHORT_V2_TRAIN) | set(WALL_COHORT_V2_GENERALIZATION))
    arms = {}
    print("%12s | %-26s | %-26s" % ("vessel", "FROZEN t=0 gate", "ORACLE time-varying gate"))
    print("%12s | %6s %7s %7s | %6s %7s %7s" % ("", "rho", "curveL1", "score",
                                                 "rho", "curveL1", "score"))
    for a in names:
        p = DIR / f"{a}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        if int(d.y.shape[0]) < 150:
            continue
        wall = d.mask_wall.reshape(-1).bool().numpy()
        f0 = t0_flow_fields(d, bio, hops=3, flow_source="gt")
        gt_idx = gt_onset_index(d, phys, wall)
        t_eval = resolve_deploy_eval_time_index(int(d.y.shape[0]))
        phi_gt = gt_clot_phi_at_time(d, t_eval, phys, device=torch.device("cpu")).reshape(-1)
        phi_gt = phi_gt * torch.tensor(wall.astype(np.float32))

        def score(idx, t):
            m = onset_metrics(idx, gt_idx, t, wall)
            m["curve_l1"] = curve_l1(idx, gt_idx, t, wall)
            pred = torch.tensor((((idx >= 0) & wall)).astype(np.float32))
            mm = compute_clot_relaxed_metrics(pred, phi_gt, d.edge_index,
                                              wall_mask=torch.tensor(wall))
            m["score"] = clot_score_from_deploy_dict(metrics_to_deploy_prefix(mm))
            return m

        g0 = graded_gate(f0, bio, mode="hard") * wall
        traj, t = integrate_mat_trajectory(d, bio, g0, da_scale=40.0)
        frozen = score(first_crossing(traj, float(bio.viscosity_mat_crit)), t)

        gs = gt_gate_series(d, bio, wall, "hard", 0.0)
        traj2, t2 = integrate_mat_trajectory(
            d, bio, gs[0], da_scale=40.0,
            blockage=lambda mat, g0_, i, _gs=gs: _gs[min(i, len(_gs) - 1)])
        oracle = score(first_crossing(traj2, float(bio.viscosity_mat_crit)), t2)
        arms[a] = (frozen, oracle)
        print("%12s | %6.3f %7.4f %7.4f | %6.3f %7.4f %7.4f"
              % (a, frozen["rho"], frozen["curve_l1"], frozen["score"],
                 oracle["rho"], oracle["curve_l1"], oracle["score"]))

    def m(sel, which, key):
        v = [arms[a][which][key] for a in sel if a in arms
             and arms[a][which][key] == arms[a][which][key]]
        return float(np.mean(v)) if v else float("nan")

    tr = [a for a in WALL_COHORT_V2_TRAIN if a in arms]
    sl = [a for a in WALL_COHORT_V2_GENERALIZATION if a in arms]
    print("\n%-8s %-28s %-28s" % ("", "FROZEN t=0", "ORACLE time-varying"))
    for lbl, sel in (("train", tr), ("sealed", sl)):
        print("%-8s rho %.3f curveL1 %.4f score %.4f | rho %.3f curveL1 %.4f score %.4f"
              % (lbl, m(sel, 0, "rho"), m(sel, 0, "curve_l1"), m(sel, 0, "score"),
                 m(sel, 1, "rho"), m(sel, 1, "curve_l1"), m(sel, 1, "score")))
    print("\n  The oracle is an upper bound on ANY evolving-flow model, learned or not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
