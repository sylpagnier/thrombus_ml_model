"""The corrector's CEILING: hand it GT's occlusion at every step and integrate.

``diag_corrector_why_no_gain.py`` showed the corrector gates 72% of the GAIN nodes when
driven by GT's occlusion (46% when driven by its own).  Gating them is necessary but not
sufficient -- the law's bracket is not binary,

    gate = [dsrx < sgt] * (L/gamma_m) * |dsrx|  +  [sr < lss]

so the stagnation branch contributes exactly 1.0 while the separation branch contributes
``5e-4 * |dsrx|``, which can be far smaller.  A node can be "gated" and still integrate
too slowly to reach ``viscosity_mat_crit`` inside the horizon.

This closes the loop: run the ODE with the corrector's gate series computed from GT's own
occlusion -- the best this corrector can possibly do -- and compare the resulting mask
against the frozen-gate baseline and the GT-flow oracle.  Also reports the gate VALUE the
corrector assigns to GAIN nodes against the value GT assigns.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.ap_closure import SHIPPED, make_rollout_hook  # noqa: E402
from src.core_physics.mls_gradient import (  # noqa: E402
    build_mls_gradient, node_positions, shear_rate_2d,
)
from src.core_physics.physics_wall_model import (  # noqa: E402
    M_TO_CM, first_crossing, integrate_mat_trajectory, t0_flow_fields,
)

DIR = Path("data/processed/graphs_biochem_anchors")
CACHE = Path("outputs/wall_species_cache")
CKPT = Path("outputs/kinematics/local_corrector/local_kinematic_corrector_best.pth")
EVERY, HOPS, DMU = 10, 3, 0.68


def main() -> int:
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prot = json.load(open("outputs/ap_closure/protocol_gt_meanovertime.json"))
    names = prot["fit"] + prot["dev"]
    crit = float(bio.viscosity_mat_crit)
    lss, sgt = float(bio.lss), float(bio.sgt) / M_TO_CM
    coef = float(bio.L_char) * M_TO_CM / float(bio.gamma_m)
    from src.core_physics.coupled_shear_gnn import load_local_corrector
    from src.inference.corrector_coupling import couple_flow_with_corrector
    corr = load_local_corrector(CKPT, device)

    print("Masks (wall ignition) and the gate VALUE on GAIN nodes.\n")
    print("%-12s %8s %8s %8s %10s %10s %10s"
          % ("vessel", "frozen", "corr*", "oracle", "gtGateVal", "cGateVal", "cLowFrac"))
    A = {k: [] for k in ("f", "c", "o", "gv", "cv", "lf")}
    for n in names:
        p, c = DIR / f"{n}.pt", CACHE / f"{n}.npz"
        if not p.exists() or not c.exists():
            continue
        z = np.load(c)
        if "sr_t" not in z.files:
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        widx, nt = z["wall_idx"], len(z["t"])
        w = d.mask_wall.reshape(-1).bool().numpy()
        nn_ = len(w)
        gate_gt = np.zeros((nt, nn_))
        gate_gt[:, widx] = ((z["dsrx_t"] < sgt) * coef * np.abs(z["dsrx_t"])
                            + (z["sr_t"] < lss))
        mat_gt = np.zeros((nt, nn_))
        mat_gt[:, widx] = z["mat"]

        f0 = t0_flow_fields(d, bio, hops=HOPS, flow_source="gt")
        hook = make_rollout_hook(SHIPPED, bio, f0.sr)
        tr, _ = integrate_mat_trajectory(d, bio, f0.gate * w, da_scale=40.0, ap_closure=hook)
        S_static = (first_crossing(tr, crit) >= 0) & w
        tr, _ = integrate_mat_trajectory(d, bio, f0.gate * w, da_scale=40.0,
                                         blockage=lambda m, g, i: gate_gt[min(i, nt - 1)] * w,
                                         ap_closure=hook)
        S_or = (first_crossing(tr, crit) >= 0) & w
        GAIN = S_or & ~S_static & w
        if GAIN.sum() < 1:
            continue

        pos, ei = node_positions(d), d.edge_index.numpy()
        Dx, Dy = build_mls_gradient(pos, ei, hops=HOPS)
        u_ref = float(d.u_ref.reshape(-1)[0])
        d_bar = float(d.d_bar.reshape(-1)[0])
        u0 = d.y[0, :, 0].numpy().astype(np.float64)
        v0 = d.y[0, :, 1].numpy().astype(np.float64)
        u0_t = torch.tensor(u0, dtype=torch.float32, device=device)
        v0_t = torch.tensor(v0, dtype=torch.float32, device=device)

        class _V:
            def __init__(s, dd):
                s.x = dd.x.to(device)
                s.edge_index = dd.edge_index.to(device)
                s.num_nodes = int(dd.num_nodes)
        dv = _V(d)

        # precompute the corrector's gate series from GT occlusion (its ceiling)
        cache_g, cache_low = {}, {}
        for i in range(0, nt, EVERY):
            occ = mat_gt[i] >= crit
            if not occ.any():
                cache_g[i], cache_low[i] = f0.gate * w, (f0.sr < lss) & w
                continue
            delta = torch.tensor(occ.astype(np.float32) * DMU, device=device)
            with torch.no_grad():
                uu, vv, _ = couple_flow_with_corrector(
                    dv, u0_t, v0_t, delta, corrector=corr, phys_cfg=phys,
                    device=device, num_hops=5)
            un = uu.detach().cpu().numpy().astype(np.float64)
            vn = vv.detach().cpu().numpy().astype(np.float64)
            sr = shear_rate_2d(Dx @ un, Dy @ un, Dx @ vn, Dy @ vn) * (u_ref / d_bar)
            dsx = (Dx @ sr) / (d_bar * M_TO_CM)
            g = ((dsx < sgt) * coef * np.abs(dsx) + (sr < lss)) * w
            cache_g[i] = np.where(occ, np.maximum(g, f0.gate * w), g)
            cache_low[i] = (sr < lss) & w
        keys = sorted(cache_g)

        def blk_ceiling(mat, g0, i):
            k = max([q for q in keys if q <= i], default=keys[0])
            return cache_g[k]

        tr, _ = integrate_mat_trajectory(d, bio, f0.gate * w, da_scale=40.0,
                                         blockage=blk_ceiling, ap_closure=hook)
        S_c = (first_crossing(tr, crit) >= 0) & w

        gv = float(np.median(gate_gt[:, GAIN].max(axis=0)))
        cstack = np.stack([cache_g[k] for k in keys])
        cv = float(np.median(cstack[:, GAIN].max(axis=0)))
        lstack = np.stack([cache_low[k] for k in keys])
        lf = float(lstack[:, GAIN].any(axis=0).mean())
        for k, v in zip(("f", "c", "o", "gv", "cv", "lf"),
                        (S_static.sum(), S_c.sum(), S_or.sum(), gv, cv, lf)):
            A[k].append(float(v))
        print("%-12s %8d %8d %8d %10.3f %10.3f %9.0f%%"
              % (n, S_static.sum(), S_c.sum(), S_or.sum(), gv, cv, 100 * lf))

    print("\n%-12s %8.1f %8.1f %8.1f %10.3f %10.3f %9.0f%%"
          % ("MEAN", np.mean(A["f"]), np.mean(A["c"]), np.mean(A["o"]),
             np.mean(A["gv"]), np.mean(A["cv"]), 100 * np.mean(A["lf"])))
    print("\n  corr*     : corrector driven by GT's occlusion = this corrector's CEILING")
    print("  gtGateVal : peak gate value GT gives a GAIN node (1.0 = stagnation branch)")
    print("  cGateVal  : peak gate value the corrector gives it")
    print("  cLowFrac  : GAIN nodes the corrector ever puts BELOW lss (stagnation branch)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
