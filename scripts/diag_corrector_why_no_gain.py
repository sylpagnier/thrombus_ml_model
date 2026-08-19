"""WHY does right-sign, 1/3-magnitude near-field physics not move the mask?

The corrector lowers wake shear like GT does and opens low-shear gates at nearly GT's
rate (``diag_corrector_sign.py``), yet the rollout mask is 73.3 against a like-for-like
static baseline of 73.7 while the GT-flow oracle reaches 85.6
(``diag_corrector_mask_accounting.py``).

This decomposes the missing ~12 nodes.  Define

    GAIN = (ignites under the GT gate series) AND NOT (ignites under the frozen t=0 gate)

-- exactly the nodes the oracle wins -- and then ask, on those nodes only:

  1. WHEN does GT's gate open on them, and how much of the horizon is left afterwards?
  2. Does the corrector's gate ever open on them, driven by the model's OWN occlusion?
  3. Does it open when driven by GT's occlusion instead (``--oracle-occ``)?

(2) vs (3) separates the two candidate failures, which need opposite fixes:

    BOOTSTRAP  -- the model's own clot is too small/late to drive the corrector, so the
                  feedback loop never gets going.  Fix: seeding / earlier coupling.
    RESPONSE   -- even handed GT's exact clot, the corrector's flow change is too weak to
                  cross the threshold.  Fix: the corrector itself.

Uses ``outputs/wall_species_cache`` for the GT shear series.  SEALED is not opened.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
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
EVERY, HOPS = 10, 3


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
    DMU = 0.68          # the measured median GT delta-mu, not the 3.0 clamp

    print("GAIN = nodes the GT-flow oracle ignites that the frozen t=0 gate does not.\n")
    print("%-12s %6s %8s %10s %10s %10s %10s"
          % ("vessel", "GAIN", "gt_open%", "gt_open@", "own_open%", "orc_open%", "budget%"))
    acc = {k: [] for k in ("gain", "gto", "gtt", "own", "orc", "bud")}

    for n in names:
        p, c = DIR / f"{n}.pt", CACHE / f"{n}.npz"
        if not p.exists() or not c.exists():
            continue
        z = np.load(c)
        if "sr_t" not in z.files:
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        widx = z["wall_idx"]
        nt = len(z["t"])
        w = d.mask_wall.reshape(-1).bool().numpy()
        gate_gt_w = (z["dsrx_t"] < sgt) * coef * np.abs(z["dsrx_t"]) + (z["sr_t"] < lss)

        f0 = t0_flow_fields(d, bio, hops=HOPS, flow_source="gt")
        hook = make_rollout_hook(SHIPPED, bio, f0.sr)
        nn_ = len(w)
        gate_gt = np.zeros((nt, nn_))
        gate_gt[:, widx] = gate_gt_w

        # ignition under the frozen t=0 gate
        tr_s, _ = integrate_mat_trajectory(d, bio, f0.gate * w, da_scale=40.0, ap_closure=hook)
        S_static = (first_crossing(tr_s, crit) >= 0) & w
        # ignition under the GT gate series
        tr_o, _ = integrate_mat_trajectory(d, bio, f0.gate * w, da_scale=40.0,
                                           blockage=lambda m, g, i: gate_gt[min(i, nt - 1)] * w,
                                           ap_closure=hook)
        on_o = first_crossing(tr_o, crit)
        GAIN = (on_o >= 0) & ~S_static & w
        if GAIN.sum() < 1:
            continue

        # when does GT's gate open on the GAIN nodes?
        open_gt = gate_gt[:, GAIN] > 0
        ever_gt = open_gt.any(axis=0)
        first_gt = np.where(ever_gt, open_gt.argmax(axis=0), nt - 1)
        budget = 1.0 - first_gt / max(nt - 1, 1)

        pos = node_positions(d)
        ei = d.edge_index.numpy()
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

        def corr_gate(occ_mask):
            if not occ_mask.any():
                return f0.gate * w
            delta = torch.tensor(occ_mask.astype(np.float32) * DMU, device=device)
            with torch.no_grad():
                uu, vv, _ = couple_flow_with_corrector(
                    dv, u0_t, v0_t, delta, corrector=corr, phys_cfg=phys,
                    device=device, num_hops=5)
            un = uu.detach().cpu().numpy().astype(np.float64)
            vn = vv.detach().cpu().numpy().astype(np.float64)
            sr = shear_rate_2d(Dx @ un, Dy @ un, Dx @ vn, Dy @ vn) * (u_ref / d_bar)
            dsx = (Dx @ sr) / (d_bar * M_TO_CM)
            return ((dsx < sgt) * coef * np.abs(dsx) + (sr < lss)) * w

        # arm 2: corrector driven by the model's OWN occlusion
        st = {"g": f0.gate * w, "last": -10 ** 9, "seen": np.zeros(nn_, bool)}

        def blk_own(mat, g0, i, _s=st):
            if i - _s["last"] >= EVERY:
                occ = mat >= crit
                g = corr_gate(occ)
                g = np.where(occ, np.maximum(g, g0), g)
                _s["g"], _s["last"] = g, i
                _s["seen"] |= g > 0
            return _s["g"]

        integrate_mat_trajectory(d, bio, f0.gate * w, da_scale=40.0,
                                 blockage=blk_own, ap_closure=hook)
        own_open = st["seen"]

        # arm 3: corrector driven by GT's occlusion (perfect clot input)
        mat_gt = np.zeros((nt, nn_))
        mat_gt[:, widx] = z["mat"]
        seen_o = np.zeros(nn_, bool)
        for i in range(0, nt, EVERY):
            seen_o |= corr_gate(mat_gt[i] >= crit) > 0

        acc["gain"].append(int(GAIN.sum()))
        acc["gto"].append(float(ever_gt.mean()))
        acc["gtt"].append(float(np.median(first_gt) / max(nt - 1, 1)))
        acc["own"].append(float(own_open[GAIN].mean()))
        acc["orc"].append(float(seen_o[GAIN].mean()))
        acc["bud"].append(float(np.median(budget)))
        print("%-12s %6d %7.0f%% %9.2f %9.0f%% %9.0f%% %9.0f%%"
              % (n, GAIN.sum(), 100 * acc["gto"][-1], acc["gtt"][-1],
                 100 * acc["own"][-1], 100 * acc["orc"][-1], 100 * acc["bud"][-1]))

    print("\n%-12s %6.1f %7.0f%% %9.2f %9.0f%% %9.0f%% %9.0f%%"
          % ("MEAN", np.mean(acc["gain"]), 100 * np.mean(acc["gto"]), np.mean(acc["gtt"]),
             100 * np.mean(acc["own"]), 100 * np.mean(acc["orc"]), 100 * np.mean(acc["bud"])))
    print("\n  gt_open%%  : GAIN nodes whose GT gate ever opens")
    print("  gt_open@  : median fraction of the horizon elapsed when it opens")
    print("  own_open%%: GAIN nodes the corrector gates, driven by the MODEL's occlusion")
    print("  orc_open%%: ... driven by GT's occlusion (perfect clot handed to it)")
    print("  budget%%  : horizon remaining after GT opens the gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
