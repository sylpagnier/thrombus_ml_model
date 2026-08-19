"""Can the trained kinematic corrector stand in for GT flow in a rollout Track A?

CONTEXT.  ``scripts/diag_rollout_trackA.py`` showed that evolving the flow gives a
materially better committed set -- wall F1 0.8405 -> 0.9004, count floor 0.0424 -> 0.0254 --
while leaving end-to-end ``growth_l1`` unchanged (0.1224 -> 0.1262 / 0.1210).  That was a
GT-flow ORACLE.  This asks whether the repo's own ``LocalKinematicCorrector`` recovers the
same mask gain without GT, which is the only deployable route to it.

SET EXPECTATIONS FIRST.  The oracle bounds this arm from above, and the oracle's end-to-end
gain is ~0.  So the question here is explicitly **the mask**, not ``growth_l1``:

    does the corrector recover  F1 +0.06  and  floor -0.017 ?

THE LOOP.  Every ``EVERY`` steps: committed ``Mat`` -> a per-node viscosity bump (clamped to
the corrector's trained 1.5-3 Pa.s range) -> ``couple_flow_with_corrector`` -> MLS gradients
-> ``sr``/``dsrx`` -> the two gates -> keep integrating.  The base flow is GT at t=0, the same
Phase-3 bandaid the shipped arm A uses, so this is directly comparable to what ships.

CAVEAT WORTH CARRYING.  ``scripts/diag_rgp_deq_flow_audit.py`` found the DEQ's accuracy rests
on a leaked prior: deploy-legal ``analytic`` gives relL2_u 0.516 against ``stored``'s 0.134.
But clot AUC barely moves (0.792 vs 0.801), i.e. the discriminative structure survives even
when the field is numerically mediocre -- and the gate consumes structure, not field values.

SEALED is not opened.

    python scripts/diag_corrector_rollout.py
"""
from __future__ import annotations

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
OUT = Path("outputs/rollout_trackA")
CKPT = Path("outputs/kinematics/local_corrector/local_kinematic_corrector_best.pth")
EVERY = 10          # corrector calls per rollout; the operator has its own hysteresis too
RELAX, GROW, HOPS = 2.0, 6, 3


def f1(pred, gt):
    tp = float((pred & gt).sum())
    return float(2 * tp / (2 * tp + (pred & ~gt).sum() + (~pred & gt).sum())) if tp else 0.0


def main() -> int:
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    prot = json.load(open("outputs/ap_closure/protocol_gt_meanovertime.json"))
    names = prot["fit"] + prot["dev"]
    crit = float(bio.viscosity_mat_crit)
    lss, sgt = float(bio.lss), float(bio.sgt) / M_TO_CM
    coef = float(bio.L_char) * M_TO_CM / float(bio.gamma_m)

    from src.core_physics.coupled_shear_gnn import load_local_corrector
    from src.inference.corrector_coupling import (
        corrector_max_delta_mu_si, couple_flow_with_corrector,
    )
    if not CKPT.exists():
        print("[ABORT] no corrector checkpoint at %s" % CKPT)
        return 1
    corr = load_local_corrector(CKPT, device)
    dmu = float(corrector_max_delta_mu_si() or 3.0)
    print("device=%s  corrector=%s  delta_mu=%.2f Pa.s  EVERY=%d steps\n"
          % (device, CKPT.name, dmu, EVERY))

    R = {a: {} for a in ("static Track A", "corrector rollout", "corrector + front adm",
                         "FLOOR corrector mask", "FLOOR corr+front mask")}
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

        f0 = t0_flow_fields(d, bio, hops=HOPS, flow_source="gt")
        pos = node_positions(d)
        ei = d.edge_index.numpy()
        nn_ = len(w)
        A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(nn_, nn_)).tocsr()
        A = ((A + A.T) > 0).astype(np.int8)
        Dx, Dy = build_mls_gradient(pos, ei, hops=HOPS)
        u_ref = float(d.u_ref.reshape(-1)[0])
        d_bar = float(d.d_bar.reshape(-1)[0])
        u0 = d.y[0, :, 0].numpy().astype(np.float64)
        v0 = d.y[0, :, 1].numpy().astype(np.float64)
        u0_t = torch.tensor(u0, dtype=torch.float32, device=device)
        v0_t = torch.tensor(v0, dtype=torch.float32, device=device)

        class _DevView:
            """Only what the corrector reads, on GPU.

            ``Data.to(device)`` moves the pack IN PLACE, which then breaks the CPU-side ODE
            integrator (it calls ``.numpy()`` on ``data.y``).  Cloning a 300 MB pack to
            avoid that is wasteful when the corrector needs three small tensors.
            """

            def __init__(self, dd):
                self.x = dd.x.to(device)
                self.edge_index = dd.edge_index.to(device)
                self.num_nodes = int(dd.num_nodes)

        d_dev = _DevView(d)

        # ---- shipped static arm
        cur = (f0.gate > 0) & w
        adm = (f0.sr < lss * RELAX) & w
        for _ in range(GROW):
            cur = cur | (((A @ cur.astype(np.int8)) > 0) & adm)
        hook = make_rollout_hook(SHIPPED, bio, f0.sr)
        traj, _ = integrate_mat_trajectory(d, bio, f0.gate * w, da_scale=40.0, ap_closure=hook)
        idx = first_crossing(traj, crit)
        cr = idx >= 0
        med = int(np.median(idx[cr])) if cr.any() else 0
        on_static = np.where(cur, np.where(idx >= 0, idx, med), -1)

        # ---- corrector-driven rollout: the clot reroutes the flow, the flow moves the gate
        state = {"gate": f0.gate * w, "last": -10 ** 9, "n": 0}

        def blockage(mat, gate0, i, _s=state):
            if i - _s["last"] < EVERY:
                return _s["gate"]
            occ = mat >= crit
            if occ.any():
                delta = torch.tensor(occ.astype(np.float32) * dmu, device=device)
                with torch.no_grad():
                    uu, vv, _ = couple_flow_with_corrector(
                        d_dev, u0_t, v0_t, delta, corrector=corr, phys_cfg=phys,
                        device=device, num_hops=5)
                un = uu.detach().cpu().numpy().astype(np.float64)
                vn = vv.detach().cpu().numpy().astype(np.float64)
                _s["n"] += 1
            else:
                un, vn = u0, v0
            sr = shear_rate_2d(Dx @ un, Dy @ un, Dx @ vn, Dy @ vn) * (u_ref / d_bar)
            dsx = (Dx @ sr) / (d_bar * M_TO_CM)
            g = ((dsx < sgt) * coef * np.abs(dsx) + (sr < lss)) * w
            g = np.where(occ, np.maximum(g, gate0), g)
            _s["gate"], _s["last"] = g, i
            return g

        traj2, _ = integrate_mat_trajectory(d, bio, f0.gate * w, da_scale=40.0,
                                            blockage=blockage, ap_closure=hook)
        on_corr = first_crossing(traj2, crit)
        m_corr = (on_corr >= 0) & w

        # ---- corrector rollout PLUS the front-admission term the static arm carries.
        # Without this the two arms are different estimators: the static mask is ignition
        # UNION 6-hop growth, the corrector mask is ignition only.
        state2 = {"gate": f0.gate * w, "last": -10 ** 9}

        def blockage_front(mat, gate0, i, _s=state2):
            g = blockage(mat, gate0, i)
            occ2 = (mat >= crit).astype(np.int8)
            adj = (np.asarray(A @ occ2).reshape(-1) > 0) & adm
            g = g.copy()
            g[adj] = np.maximum(g[adj], 1.0)
            return g

        state["gate"], state["last"], state["n"] = f0.gate * w, -10 ** 9, 0
        traj3, _ = integrate_mat_trajectory(d, bio, f0.gate * w, da_scale=40.0,
                                            blockage=blockage_front, ap_closure=hook)
        on_cf = first_crossing(traj3, crit)
        m_cf = (on_cf >= 0) & w

        for tag, on, mk in (("static Track A", on_static, cur),
                            ("corrector rollout", np.where(w, on_corr, -1), m_corr),
                            ("corrector + front adm", np.where(w, on_cf, -1), m_cf)):
            e = growth_error(on, gt, nt, w)
            e.update(f1=f1(mk, gt_set), n_mask=int(mk.sum()))
            R[tag][n] = e
        for tag, mk in (("FLOOR corrector mask", m_corr), ("FLOOR corr+front mask", m_cf)):
            e = growth_error(count_optimal_onset(mk, gt, nt, w), gt, nt, w)
            e.update(f1=f1(mk, gt_set), n_mask=int(mk.sum()))
            R[tag][n] = e
        print("  %-12s corrector calls %2d | mask %3d -> %3d (GT %3d) | gl1 %.4f -> %.4f"
              % (n, state["n"], int(cur.sum()), int(m_corr.sum()), int(gt_set.sum()),
                 R["static Track A"][n]["growth_l1"], R["corrector rollout"][n]["growth_l1"]))

    ok = sorted(R["static Track A"])
    print("\n" + "=" * 86)
    print("CORRECTOR-DRIVEN ROLLOUT, %d train vessels   (%.0f s)" % (len(ok), time.time() - t0))
    print("=" * 86)
    print("%-26s %10s %11s %10s %9s" % ("arm", "growth_l1", "final_err", "wall F1", "n_mask"))
    for a in R:
        g = lambda k: float(np.nanmean([R[a][n][k] for n in ok]))          # noqa: E731
        print("%-26s %10.4f %+11.4f %10.4f %9.1f"
              % (a, g("growth_l1"), g("final_err"), g("f1"), g("n_mask")))
    print("\n   for reference, from scripts/diag_rollout_trackA.py (GT-flow oracle):")
    print("   %-26s %10s %11s %10s" % ("flow-oracle rollout", "0.1262", "-0.0211", "0.8953"))
    print("   %-26s %10s %11s %10s" % ("FLOOR oracle mask", "0.0254", "-0.0211", "0.8953"))

    (OUT / "corrector_rollout.json").write_text(json.dumps(
        dict(per_vessel={a: R[a] for a in R}, names=ok), indent=2, default=float),
        encoding="utf-8")
    print("\nwrote %s" % (OUT / "corrector_rollout.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
