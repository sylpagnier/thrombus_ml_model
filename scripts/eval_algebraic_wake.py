"""PHASE 8: algebraic wake feedback on the committed mask -- no flow model.

The local kinematic corrector opens ~15 new gates / vessel but they are the wrong nodes
(deploy score -0.019).  GT union-gate is +0.032 on TRAIN mean and *slightly worse* on
patient020 once hops=20 is already shipped.  So the remaining physical attack is the
one ``src/core_physics/shear_redistribution.py`` already wrote down:

    committed tissue is an 80x-viscosity no-slip obstacle and sheds a stagnation wake,
    ``sr_new = sr_t0 * (1 - wake * phi)``, with ``phi`` the occluded fraction of the
    local cross-section.

That arm previously saw ``phi ~ 0`` because it was driven by the ODE's Mat-crossing,
which almost never fired.  The shipped hops=20 mask IS a committed occlusion, so the
same algebra can be evaluated on a real phi.  Deploy-legal: positions, sdf, connectivity.

Also scored here: drop high-sr separation-only seeds (the FP over-ignition pool), and
wake-reduced lumen speed for the off-wall arm.

    python scripts/eval_algebraic_wake.py
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

from predict_wall_clot import GROW_HOPS, LUMEN_HOPS, LUMEN_SPEED, RELAX, STENCIL  # noqa: E402
from src.biochem_gnn.mat_growth_simple import WALL_COHORT_V2_TRAIN  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.physics_lumen_model import grow_into_lumen, speed_nd  # noqa: E402
from src.core_physics.physics_wall_model import gate_from_shear, t0_flow_fields  # noqa: E402
from src.core_physics.shear_redistribution import (  # noqa: E402
    build_crosssection_operator, sdf_nd,
)
from src.core_physics.species_pushforward_continuous import (  # noqa: E402
    resolve_deploy_eval_time_index,
)
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.evaluation.clot_relaxed_metrics import (  # noqa: E402
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
)

DIR = REPO / "data/processed/graphs_biochem_anchors"


def f1(pred, gt):
    if gt.sum() == 0 and pred.sum() == 0:
        return float("nan")
    tp = int((pred & gt).sum())
    p, r = tp / max(int(pred.sum()), 1), tp / max(int(gt.sum()), 1)
    return 2 * p * r / max(p + r, 1e-9)


def sc(pred, gt_t, ei):
    m = compute_clot_relaxed_metrics(torch.tensor(pred.astype(np.float32)), gt_t, ei)
    return float(clot_score_from_deploy_dict(metrics_to_deploy_prefix(m)))


def adj(ei, n):
    A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
    return ((A + A.T) > 0).astype(np.int8)


def grow(seed, wall, A, sr, bio, hops=GROW_HOPS, relax=RELAX):
    cur = seed.copy()
    adm = (sr < float(bio.lss) * relax) & wall
    for _ in range(int(hops)):
        cur = cur | (((A @ cur.astype(np.int8)) > 0) & adm)
    return cur


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default="outputs/phase8_algebraic_wake.json")
    args = ap.parse_args()
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    lss = float(bio.lss)

    packs = []
    for anchor in WALL_COHORT_V2_TRAIN:
        pth = DIR / f"{anchor}.pt"
        if not pth.exists():
            continue
        d = torch.load(pth, map_location="cpu", weights_only=False)
        if int(d.y.shape[0]) < 150:
            continue
        wall = d.mask_wall.reshape(-1).bool().numpy()
        ei = d.edge_index.detach().cpu().numpy()
        n = len(wall)
        A = adj(ei, n)
        t_eval = resolve_deploy_eval_time_index(int(d.y.shape[0]))
        gt_f = gt_clot_phi_at_time(d, t_eval, phys, device=torch.device("cpu")).reshape(-1)
        gt = gt_f.numpy() > 0.5
        if (gt & wall).sum() == 0:
            continue
        f = t0_flow_fields(d, bio, hops=STENCIL["gt"], flow_source="gt")
        seed = (f.gate > 0) & wall
        shipped = grow(seed, wall, A, f.sr, bio)
        pos = d.x[:, :2].detach().cpu().numpy().astype(np.float64)
        sdf = sdf_nd(d)
        B = build_crosssection_operator(pos, sdf, wall)
        phi = np.asarray(B @ shipped.astype(np.float64)).reshape(-1)
        packs.append(dict(anchor=anchor, d=d, wall=wall, A=A, gt=gt, gt_f=gt_f,
                          f=f, seed=seed, shipped=shipped, spd=speed_nd(d),
                          B=B, phi=phi))
    print("[i] %d vessels" % len(packs))
    from src.core_physics.wall_cohort_splits import format_split_means, split_of
    print("[i] protocol FIT n=%d DEV n=%d SEALED closed (patient020 is FIT)"
          % (sum(split_of(p["anchor"]) == "fit" for p in packs),
             sum(split_of(p["anchor"]) == "dev" for p in packs)))
    phis = np.concatenate([p["phi"][p["wall"]] for p in packs])
    print("[i] phi on wall at shipped mask: p50=%.4f p90=%.4f p99=%.4f max=%.4f"
          % (float(np.percentile(phis, 50)), float(np.percentile(phis, 90)),
             float(np.percentile(phis, 99)), float(phis.max())))

    acc = {}

    def with_lumen(p, msk, spd=None, sr=None):
        off = grow_into_lumen(msk, p["wall"], p["A"],
                              p["spd"] if spd is None else spd,
                              p["f"].sr if sr is None else sr,
                              lumen_hops=LUMEN_HOPS, speed_thresh=LUMEN_SPEED)
        return msk | off

    def report(name, mask_fn, lumen_fn=None):
        scores, wf, of, per = [], [], [], {}
        for p in packs:
            msk = mask_fn(p)
            pred = lumen_fn(p, msk) if lumen_fn else with_lumen(p, msk)
            s = sc(pred, p["gt_f"], p["d"].edge_index)
            scores.append(s)
            per[p["anchor"]] = s
            wf.append(f1(pred & p["wall"], p["gt"] & p["wall"]))
            of.append(f1(pred & ~p["wall"], p["gt"] & ~p["wall"]))
        row = dict(score=float(np.mean(scores)), wall_f1=float(np.nanmean(wf)),
                   off_f1=float(np.nanmean(of)), per=per)
        acc[name] = row
        print("   %-44s %s  wall %.4f  off %.4f"
              % (name, format_split_means(per), row["wall_f1"], row["off_f1"]))
        return row

    print("\n=== BASELINE / FP SUPPRESSION ===")
    shipped = report("shipped hops=%d" % GROW_HOPS, lambda p: p["shipped"])
    # Separation-only seeds at high sr: the over-ignition pool.  Keep low-shear seeds.
    for k in (1.0, 2.0, 4.0, 8.0):
        def _drop(p, k=k):
            sep_only = (p["f"].gate_sep > 0) & (p["f"].gate_low < 1) & p["wall"]
            seed = p["seed"] & ~(sep_only & (p["f"].sr > k * lss))
            return grow(seed, p["wall"], p["A"], p["f"].sr, bio)
        report("drop sep-only sr>%.1f*lss" % k, _drop)

    print("\n=== ALGEBRAIC WAKE (sr *= 1 - wake*phi, union onto shipped) ===")
    for wake in (0.5, 1.0, 2.0, 4.0, 8.0):
        def _wake(p, wake=wake):
            amp = np.clip(1.0 - wake * p["phi"], 0.02, 1.0)
            sr2 = p["f"].sr * amp
            dsx2 = p["f"].dsrx * amp
            g = gate_from_shear(sr2, dsx2, bio, wall=p["wall"])
            extra = (g > 0) & p["wall"]
            return p["shipped"] | extra
        report("wake=%.1f union extra gates" % wake, _wake)

        def _wake_grow(p, wake=wake):
            amp = np.clip(1.0 - wake * p["phi"], 0.02, 1.0)
            sr2 = p["f"].sr * amp
            dsx2 = p["f"].dsrx * amp
            g = gate_from_shear(sr2, dsx2, bio, wall=p["wall"])
            extra = (g > 0) & p["wall"]
            return grow(p["seed"] | extra, p["wall"], p["A"], sr2, bio)
        report("wake=%.1f re-grow hops=20 on new sr" % wake, _wake_grow)

    print("\n=== WAKE-REDUCED LUMEN SPEED (off-wall) ===")
    for wake in (1.0, 2.0, 4.0):
        def _lum(p, msk, wake=wake):
            amp = np.clip(1.0 - wake * p["phi"], 0.02, 1.0)
            # phi is only filled on wall rows; paint it onto nearby lumen via adjacency.
            phi_l = np.asarray(p["A"] @ p["phi"]).reshape(-1)
            amp_l = np.clip(1.0 - wake * phi_l, 0.02, 1.0)
            return with_lumen(p, msk, spd=p["spd"] * amp_l, sr=p["f"].sr * amp)
        report("shipped + lumen wake=%.1f" % wake,
               lambda p: p["shipped"], lumen_fn=_lum)

    print("\n=== LOCAL WALL OCCUPANCY (phi = wall-neighbour clot fraction) ===")
    for wake in (0.5, 1.0, 2.0):
        def _loc(p, wake=wake):
            w = p["wall"].astype(np.float64)
            Aw = p["A"].astype(np.float64)
            deg = np.asarray(Aw @ w).reshape(-1)
            deg[deg < 1] = 1.0
            phi = np.asarray(Aw @ p["shipped"].astype(np.float64)).reshape(-1) / deg
            phi = np.where(p["wall"], phi, 0.0)
            amp = np.clip(1.0 - wake * phi, 0.02, 1.0)
            sr2 = p["f"].sr * amp
            extra = (gate_from_shear(sr2, p["f"].dsrx * amp, bio, wall=p["wall"]) > 0)
            return grow(p["seed"] | extra, p["wall"], p["A"], sr2, bio)
        report("local-wall-phi wake=%.1f re-grow" % wake, _loc)

    print("\n=== PER-VESSEL  wake=0.5 / 1.0 re-grow  vs shipped ===")
    def wake_regrow(p, wake):
        amp = np.clip(1.0 - wake * p["phi"], 0.02, 1.0)
        sr2 = p["f"].sr * amp
        extra = (gate_from_shear(sr2, p["f"].dsrx * amp, bio, wall=p["wall"]) > 0)
        return grow(p["seed"] | extra, p["wall"], p["A"], sr2, bio)
    for wake in (0.5, 1.0):
        n_up = n_dn = 0
        deltas = []
        print("   -- wake=%.1f --" % wake)
        for p in packs:
            a = sc(with_lumen(p, p["shipped"]), p["gt_f"], p["d"].edge_index)
            b = sc(with_lumen(p, wake_regrow(p, wake)), p["gt_f"], p["d"].edge_index)
            dlt = b - a
            deltas.append(dlt)
            flag = "+" if dlt > 1e-6 else ("-" if dlt < -1e-6 else "=")
            if dlt > 1e-6:
                n_up += 1
            elif dlt < -1e-6:
                n_dn += 1
            print("   %-12s %s %+.4f" % (p["anchor"], flag, dlt))
        print("   mean %+.4f  median %+.4f  up %d  down %d / %d"
              % (float(np.mean(deltas)), float(np.median(deltas)), n_up, n_dn, len(packs)))

    # 2-iter: update phi from the wake-expanded mask, open again.
    print("\n=== 2-ITER WAKE FIXED POINT ===")
    for wake in (1.0, 2.0, 4.0):
        def _iter(p, wake=wake):
            msk = p["shipped"].copy()
            sr, dsx = p["f"].sr, p["f"].dsrx
            for _ in range(2):
                phi = np.clip(np.asarray(p["B"] @ msk.astype(np.float64)).reshape(-1), 0, 0.85)
                amp = np.clip(1.0 - wake * phi, 0.02, 1.0)
                g = gate_from_shear(sr * amp, dsx * amp, bio, wall=p["wall"])
                msk = msk | ((g > 0) & p["wall"])
            return msk
        report("2-iter wake=%.1f union" % wake, _iter)

    shipped_s = shipped["score"]
    print("\n=== vs shipped TRAIN-mean %.4f (select on DEV, not this mean) ===" % shipped_s)
    from src.core_physics.wall_cohort_splits import mean_by_split
    base_split = mean_by_split(shipped["per"])
    for k, v in acc.items():
        m = mean_by_split(v["per"])
        print("   %-44s FIT %+.4f  DEV %+.4f"
              % (k, m["fit"]["mean"] - base_split["fit"]["mean"],
                 m["dev"]["mean"] - base_split["dev"]["mean"]))

    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(acc=acc), indent=2))
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
