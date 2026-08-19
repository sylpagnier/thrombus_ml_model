"""The corrector's response is adequate; the FEEDBACK LOOP is what fails.  Test the fix.

``diag_corrector_ceiling.py``: handed GT's occlusion, the corrector reaches mask 83.9
against the GT-flow oracle's 88.1 and a frozen-gate 65.9 -- i.e. it recovers **81% of the
oracle's gain**.  Deployed on its own occlusion it recovers almost none.  So the deficit is
not the corrector's flow response, it is the bootstrap: the loop only starts once nodes have
already crossed ``viscosity_mat_crit``, and by then most of the horizon is spent, whereas
GT's clot has been growing under continuously-opening gates since t=0.

THE FIX UNDER TEST.  Seed the coupling with the model's OWN t=0 predicted mask -- the
shipped static prediction, which already scores wall F1 0.84 -- instead of waiting for ODE
commitment.  Fully deployable: it uses no GT.  Ramped in over the horizon so the occlusion
grows rather than appearing all at once.

    frozen        no coupling at all
    own-occ       the shipped loop: occlusion = nodes past crit          (what was tested)
    seeded        occlusion = t=0 predicted mask, ramped                 (the fix)
    ceiling       occlusion = GT's                                       (upper bound)
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
EVERY, HOPS, DMU, RELAX, GROW = 10, 3, 0.68, 2.0, 6


def f1(pred, gt):
    tp = float((pred & gt).sum())
    return float(2 * tp / (2 * tp + (pred & ~gt).sum() + (~pred & gt).sum())) if tp else 0.0


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

    A = {k: {"n": [], "f1": []} for k in ("frozen", "own-occ", "seeded", "ceiling")}
    print("%-12s %22s %22s %22s %22s"
          % ("vessel", "frozen", "own-occ", "seeded (FIX)", "ceiling (GT occ)"))
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
        gt_set = np.zeros(nn_, bool)
        gt_set[widx] = z["gt_onset"] >= 0
        gt_set &= w
        mat_gt = np.zeros((nt, nn_))
        mat_gt[:, widx] = z["mat"]

        ei = d.edge_index.numpy()
        Adj = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(nn_, nn_)).tocsr()
        Adj = ((Adj + Adj.T) > 0).astype(np.int8)
        f0 = t0_flow_fields(d, bio, hops=HOPS, flow_source="gt")
        hook = make_rollout_hook(SHIPPED, bio, f0.sr)
        Dx, Dy = build_mls_gradient(node_positions(d), ei, hops=HOPS)
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

        # the shipped t=0 predicted mask (gates + shear-admitted graph growth)
        pm = (f0.gate > 0) & w
        adm = (f0.sr < lss * RELAX) & w
        for _ in range(GROW):
            pm = pm | (((Adj @ pm.astype(np.int8)) > 0) & adm)
        order = np.argsort(-f0.gate * pm)          # strongest gates occlude first

        def cgate(occ, g0):
            if not occ.any():
                return g0
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
            return np.where(occ, np.maximum(g, g0), g)

        def run(occ_of):
            st = {"g": f0.gate * w, "last": -10 ** 9}

            def blk(mat, g0, i, _s=st):
                if i - _s["last"] >= EVERY:
                    _s["g"] = cgate(occ_of(mat, i), g0)
                    _s["last"] = i
                return _s["g"]
            tr, _ = integrate_mat_trajectory(d, bio, f0.gate * w, da_scale=40.0,
                                             blockage=blk, ap_closure=hook)
            return (first_crossing(tr, crit) >= 0) & w

        npm = int(pm.sum())
        arms = {}
        tr, _ = integrate_mat_trajectory(d, bio, f0.gate * w, da_scale=40.0, ap_closure=hook)
        arms["frozen"] = (first_crossing(tr, crit) >= 0) & w
        arms["own-occ"] = run(lambda mat, i: mat >= crit)
        # seeded: ramp the predicted mask in over the horizon, strongest gates first
        def seeded(mat, i, _o=order, _k=npm):
            k = int(np.clip(_k * (i / max(nt - 1, 1)) * 2.0, 0, _k))
            s = np.zeros(nn_, bool)
            s[_o[:k]] = True
            return (s & pm) | (mat >= crit)
        arms["seeded"] = run(seeded)
        arms["ceiling"] = run(lambda mat, i: mat_gt[min(i, nt - 1)] >= crit)

        cells = []
        for k in ("frozen", "own-occ", "seeded", "ceiling"):
            m = arms[k]
            A[k]["n"].append(int(m.sum()))
            A[k]["f1"].append(f1(m, gt_set))
            cells.append("%6d n / %.3f F1" % (int(m.sum()), f1(m, gt_set)))
        print("%-12s %22s %22s %22s %22s" % (n, *cells))

    print("\n%-12s %22s %22s %22s %22s"
          % ("MEAN", *["%6.1f n / %.3f F1" % (np.mean(A[k]["n"]), np.mean(A[k]["f1"]))
                       for k in ("frozen", "own-occ", "seeded", "ceiling")]))
    print("\n  GT-flow oracle reference (diag_rollout_trackA.py): F1 0.8953")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
