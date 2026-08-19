"""The deployable comparison: corrector at the PHYSICAL Delta-mu, with front admission.

``diag_corrector_rollout.py`` scored a clean negative, but at ``delta_mu = 3.0`` Pa.s -- the
stale clamp in ``corrector_max_delta_mu_si`` -- against a measured GT median of **0.68** at
committed wall nodes, and without the 6-hop front-admission term the static arm carries.
Both are fixed here, so every arm is the same estimator and the only difference is the flow.

    static Track A            frozen t=0 gate, ignition UNION 6-hop growth   (what ships)
    corrector + front         corrector on its own occlusion, + front admission
    + seeded                  ... occlusion seeded with the t=0 predicted mask, ramped
    FLOOR *                   the same mask scored with count-optimal onset

Delta-mu is 0.68 Pa.s throughout (``--dmu`` to override).  SEALED is not opened.
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

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.ap_closure import SHIPPED, make_rollout_hook  # noqa: E402
from src.core_physics.growth_count_metrics import (  # noqa: E402
    count_optimal_onset, growth_error,
)
from src.core_physics.mls_gradient import (  # noqa: E402
    build_mls_gradient, node_positions, shear_rate_2d,
)
from src.core_physics.physics_wall_model import (  # noqa: E402
    M_TO_CM, first_crossing, integrate_mat_trajectory, t0_flow_fields,
)
from src.core_physics.temporal_metrics import gt_onset_index  # noqa: E402

DIR = Path("data/processed/graphs_biochem_anchors")
CKPT = Path("outputs/kinematics/local_corrector/local_kinematic_corrector_best.pth")
EVERY, HOPS, RELAX, GROW = 10, 3, 2.0, 6
ARMS = ("static Track A", "corrector + front", "+ seeded",
        "FLOOR corr+front", "FLOOR seeded")


def f1(pred, gt):
    tp = float((pred & gt).sum())
    return float(2 * tp / (2 * tp + (pred & ~gt).sum() + (~pred & gt).sum())) if tp else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dmu", type=float, default=0.68)
    args = ap.parse_args()
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
    print("delta_mu = %.2f Pa.s   EVERY = %d steps   front admission ON\n" % (args.dmu, EVERY))

    R = {a: {} for a in ARMS}
    t0 = time.time()
    for n in names:
        p = DIR / f"{n}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        nt = int(d.y.shape[0])
        w = d.mask_wall.reshape(-1).bool().numpy()
        gt = gt_onset_index(d, phys, w)
        if not ((gt >= 0) & w).any():
            continue
        gt_set = (gt >= 0) & w
        nn_ = len(w)
        ei = d.edge_index.numpy()
        A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(nn_, nn_)).tocsr()
        A = ((A + A.T) > 0).astype(np.int8)
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

        # shipped static arm: ignition UNION 6-hop shear-admitted growth
        cur = (f0.gate > 0) & w
        adm = (f0.sr < lss * RELAX) & w
        for _ in range(GROW):
            cur = cur | (((A @ cur.astype(np.int8)) > 0) & adm)
        traj, _ = integrate_mat_trajectory(d, bio, f0.gate * w, da_scale=40.0, ap_closure=hook)
        idx = first_crossing(traj, crit)
        cr = idx >= 0
        med = int(np.median(idx[cr])) if cr.any() else 0
        on_static = np.where(cur, np.where(idx >= 0, idx, med), -1)

        def cgate(occ, g0):
            if not occ.any():
                return g0
            delta = torch.tensor(occ.astype(np.float32) * args.dmu, device=device)
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
                    g = cgate(occ_of(mat, i), g0).copy()
                    occ2 = (mat >= crit).astype(np.int8)
                    adj = (np.asarray(A @ occ2).reshape(-1) > 0) & adm   # front admission
                    g[adj] = np.maximum(g[adj], 1.0)
                    _s["g"], _s["last"] = g, i
                return _s["g"]
            tr, _ = integrate_mat_trajectory(d, bio, f0.gate * w, da_scale=40.0,
                                             blockage=blk, ap_closure=hook)
            return first_crossing(tr, crit)

        on_cf = run(lambda mat, i: mat >= crit)
        m_cf = (on_cf >= 0) & w
        order = np.argsort(-(f0.gate * cur))
        npm = int(cur.sum())

        def seeded(mat, i, _o=order, _k=npm):
            k = int(np.clip(_k * (i / max(nt - 1, 1)) * 2.0, 0, _k))
            s = np.zeros(nn_, bool)
            s[_o[:k]] = True
            return (s & cur) | (mat >= crit)
        on_sd = run(seeded)
        m_sd = (on_sd >= 0) & w

        for tag, on, mk in (("static Track A", on_static, cur),
                            ("corrector + front", np.where(w, on_cf, -1), m_cf),
                            ("+ seeded", np.where(w, on_sd, -1), m_sd)):
            e = growth_error(on, gt, nt, w)
            e.update(f1=f1(mk, gt_set), n_mask=int(mk.sum()))
            R[tag][n] = e
        for tag, mk in (("FLOOR corr+front", m_cf), ("FLOOR seeded", m_sd)):
            e = growth_error(count_optimal_onset(mk, gt, nt, w), gt, nt, w)
            e.update(f1=f1(mk, gt_set), n_mask=int(mk.sum()))
            R[tag][n] = e
        print("  %-12s mask %3d | corr+front %3d | seeded %3d  (GT %3d)"
              % (n, int(cur.sum()), int(m_cf.sum()), int(m_sd.sum()), int(gt_set.sum())))

    ok = sorted(R["static Track A"])
    print("\n" + "=" * 80)
    print("DELTA-MU %.2f Pa.s + FRONT ADMISSION, %d train vessels  (%.0f s)"
          % (args.dmu, len(ok), time.time() - t0))
    print("=" * 80)
    print("%-22s %10s %11s %10s %9s" % ("arm", "growth_l1", "final_err", "wall F1", "n_mask"))
    for a in ARMS:
        g = lambda k: float(np.nanmean([R[a][n][k] for n in ok]))          # noqa: E731
        print("%-22s %10.4f %+11.4f %10.4f %9.1f"
              % (a, g("growth_l1"), g("final_err"), g("f1"), g("n_mask")))
    print("\n   reference (GT-flow oracle, diag_rollout_trackA.py):")
    print("   %-22s %10s %11s %10s" % ("flow-oracle rollout", "0.1262", "-0.0211", "0.8953"))
    print("   %-22s %10s %11s %10s" % ("FLOOR oracle mask", "0.0254", "-0.0211", "0.8953"))
    print("   prior corrector run at delta_mu=3.0: F1 0.8243 / floor 0.0661 / mask 73.3")
    Path("outputs/rollout_trackA").mkdir(parents=True, exist_ok=True)
    Path("outputs/rollout_trackA/corrector_front_dmu.json").write_text(
        json.dumps(dict(dmu=args.dmu, per_vessel=R, names=ok), indent=2, default=float),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
