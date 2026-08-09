"""Does the t0-gated ODE integration ignite gradually, or does everything flash on at once?

The deployed clot-readout (gate + graph growth) has no time axis -- it emits one final
mask. The only component in this project with real temporal dynamics is the ODE
integration in ``integrate_mat`` (gates frozen at t=0, Mat/Mas integrated forward through
the actual COMSOL surface ODE). Before building a time-lapse visualization, check whether
that integration produces a believable SPREAD of ignition times or whether every gated
node crosses the threshold in the same step (which would make an animation pointless --
a single flash, not a growth movie).

Compares per-node ignition time (first t where Mat(t) >= viscosity_mat_crit) against GT's
own per-node onset time (first t where the GT growth label goes hot).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.physics_wall_model import t0_flow_fields, wall_platelet_constants  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

DIR = Path("data/processed/graphs_biochem_anchors")
M_TO_CM = 100.0
PER_M2_TO_PER_CM2 = 1.0e-4


def integrate_mat_trajectory(data, bio_cfg, fields, da_scale=100.0):
    """Like ``integrate_mat`` but returns Mat at EVERY timestep, not just the final one."""
    k_rs = float(bio_cfg.k_rs) * M_TO_CM
    k_as = float(bio_cfg.k_as) * M_TO_CM
    k_aa = float(bio_cfg.k_aa) * M_TO_CM
    minf = float(bio_cfg.Minf) * PER_M2_TO_PER_CM2
    da = float(bio_cfg.surface_damkohler) * float(da_scale)
    rp, ap = wall_platelet_constants(data, bio_cfg)
    t = data.t.reshape(-1).detach().cpu().numpy().astype(np.float64)
    wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
    gate = fields.gate.copy() * wall
    n = gate.shape[0]
    mas = np.zeros(n)
    mat = np.zeros(n)
    traj = np.zeros((len(t), n))
    gate_s, slope = float(bio_cfg.surface_time_gate_s), float(bio_cfg.surface_time_gate_slope)
    for i in range(len(t) - 1):
        h = t[i + 1] - t[i]
        step2t = 1.0 / (1.0 + np.exp(-np.clip((t[i] - gate_s) * slope, -50, 50)))
        sat = np.clip(1.0 - mas / minf, 0.0, 1.0)
        dep = sat * (k_rs * rp + k_as * ap)
        auto = (mas / minf) * k_aa * ap
        mas = mas + h * da * gate * dep * step2t
        mat = mat + h * da * gate * (dep + auto) * step2t
        traj[i + 1] = mat
    return traj, t


def first_crossing(series_over_t, thresh):
    """[T] per node -> index of first crossing, or -1 if never."""
    hot = series_over_t >= thresh
    idx = np.full(hot.shape[1], -1, dtype=int)
    for i in range(hot.shape[0]):
        newly = hot[i] & (idx == -1)
        idx[newly] = i
    return idx


def main() -> int:
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    for anchor in ("patient043", "patient014", "patient001", "patient007", "patient013"):
        p = DIR / f"{anchor}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        wall = d.mask_wall.reshape(-1).bool().numpy()
        f = t0_flow_fields(d, bio, hops=3, flow_source="gt")
        traj, t = integrate_mat_trajectory(d, bio, f, da_scale=100.0)
        crit = float(bio.viscosity_mat_crit)
        model_idx = first_crossing(traj, crit)
        model_committed = (model_idx >= 0) & wall
        model_t = t[model_idx[model_committed]]

        # GT onset time per node: first t where the growth label is hot
        gt_hot = np.zeros((len(t), len(wall)), dtype=bool)
        for i in range(len(t)):
            gt_hot[i] = gt_clot_phi_at_time(d, i, phys, device=torch.device("cpu")).numpy() > 0.5
        gt_idx = first_crossing(gt_hot.astype(np.float64), 0.5)
        gt_committed = (gt_idx >= 0) & wall
        gt_t = t[gt_idx[gt_committed]]

        print(f"\n=== {anchor}  (horizon {t[-1]:.0f}s, {len(t)} steps) ===")
        print(f"  MODEL ignition times [s]  n={model_committed.sum():4d}  "
              f"pct[0,25,50,75,100]={np.round(np.percentile(model_t, [0,25,50,75,100]),0) if len(model_t) else 'n/a'}")
        print(f"  GT    onset times [s]     n={gt_committed.sum():4d}  "
              f"pct[0,25,50,75,100]={np.round(np.percentile(gt_t, [0,25,50,75,100]),0) if len(gt_t) else 'n/a'}")
        if len(model_t):
            span_model = (model_t.max() - model_t.min()) / t[-1]
            print(f"  model onset spread: {span_model:.3f} of horizon")
        if len(gt_t):
            span_gt = (gt_t.max() - gt_t.min()) / t[-1]
            print(f"  GT    onset spread: {span_gt:.3f} of horizon")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
