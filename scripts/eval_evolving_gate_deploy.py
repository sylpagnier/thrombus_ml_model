"""PHASE 8: collect the +0.05 evolving-gate mask prize, deployably.

``docs/PHASE7_FINDINGS.md`` 10.4: OR of the COMSOL gate over GT flow is +0.051 deploy
score.  A t=0 halo around ``lss`` (graded gate) loses.  Two remaining physical attacks:

    1. ALGEBRAIC extra seeds from t=0 flow/geometry -- local shear minima, a
       frame-invariant separation branch ``|grad sr|``, looser ``dsrx``.  No learned
       weights; if one of them recovers the ungated high-Mat FN without a FP flood, it
       ships.
    2. CORRECTOR FIXED POINT -- RGP-DEQ ``u0_pred`` (or GT t=0) as the base flow, the
       local kinematic corrector reroutes around the current mask, new gates open, grow,
       repeat.  That is the clot-occludes -> flow-slows -> more-clot map as an algebraic
       fixed point, using the flow model we actually have.

    python scripts/eval_evolving_gate_deploy.py
    python scripts/eval_evolving_gate_deploy.py --skip-corrector
    python scripts/eval_evolving_gate_deploy.py --flow pred
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
from src.core_physics.mls_gradient import build_mls_gradient, node_positions, shear_rate_2d  # noqa: E402
from src.core_physics.physics_lumen_model import grow_into_lumen, speed_nd, speed_nd_pred  # noqa: E402
from src.core_physics.physics_wall_model import (  # noqa: E402
    M_TO_CM, gate_from_shear, t0_flow_fields,
)
from src.core_physics.species_pushforward_continuous import (  # noqa: E402
    resolve_deploy_eval_time_index,
)
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.evaluation.clot_relaxed_metrics import (  # noqa: E402
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
)

DIR = REPO / "data/processed/graphs_biochem_anchors"
CACHE = REPO / "outputs/wall_species_cache"
CKPT = REPO / "outputs/kinematics/local_corrector/local_kinematic_corrector_best.pth"
MAT_S = 7e10
CRIT = 2.0e7


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


def local_sr_minima(sr, wall, A):
    """Wall nodes whose shear is strictly below both wall-neighbours' -- a stagnation seed."""
    n = len(sr)
    out = np.zeros(n, dtype=bool)
    widx = np.flatnonzero(wall)
    for i in widx:
        nbr = A.indices[A.indptr[i]:A.indptr[i + 1]]
        wn = nbr[wall[nbr]]
        if wn.size < 2:
            continue
        if sr[i] < sr[wn].min() - 1e-9:
            out[i] = True
    return out


def hop_from(seed, wall, A, maxh=40):
    hop = np.full(len(wall), 99, dtype=np.int32)
    hop[seed] = 0
    cur = seed.copy()
    for h in range(1, maxh):
        nxt = ((A @ cur.astype(np.int8)) > 0) & wall & ~cur
        hop[nxt] = np.minimum(hop[nxt], h)
        cur = cur | nxt
    return hop


def two_stage(seed, wall, A, sr, bio, extra_hops, extra_relax):
    """Shipped growth, then a short extra front into a looser shear band.

    Unlike raising RELAX globally, far-away high-sr wall never ignites -- only nodes the
    existing mask can walk into in ``extra_hops`` steps.  That is the t=0 surrogate for
    the gate migrating a short distance as neighbouring clot slows the flow.
    """
    m1 = grow(seed, wall, A, sr, bio, hops=GROW_HOPS, relax=RELAX)
    return grow(m1, wall, A, sr, bio, hops=int(extra_hops), relax=float(extra_relax))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", default="gt", choices=["gt", "pred"])
    ap.add_argument("--skip-corrector", action="store_true")
    ap.add_argument("--corr-iters", type=int, default=4)
    ap.add_argument("--save", default="outputs/phase8_evolving_gate_deploy.json")
    args = ap.parse_args()
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hops_mls = STENCIL[args.flow]
    lss = float(bio.lss)
    sgt_cgs = float(bio.sgt) / M_TO_CM

    corr = None
    if not args.skip_corrector and CKPT.exists():
        from src.core_physics.coupled_shear_gnn import load_local_corrector
        corr = load_local_corrector(CKPT, device)
        print("[i] corrector on %s  ckpt=%s" % (device, CKPT.name))
    else:
        print("[i] skipping corrector")

    packs = []
    for anchor in WALL_COHORT_V2_TRAIN:
        p = DIR / f"{anchor}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        if int(d.y.shape[0]) < 150:
            continue
        if args.flow == "pred" and getattr(d, "u0_pred", None) is None:
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
        f = t0_flow_fields(d, bio, hops=hops_mls, flow_source=args.flow)
        pos = node_positions(d)
        Dx, Dy = build_mls_gradient(pos, ei, hops=hops_mls)
        u_ref = float(d.u_ref.reshape(-1)[0])
        d_bar = float(d.d_bar.reshape(-1)[0])
        if args.flow == "pred":
            u0 = d.u0_pred.reshape(-1).detach().cpu().numpy().astype(np.float64)
            v0 = d.v0_pred.reshape(-1).detach().cpu().numpy().astype(np.float64)
            spd = speed_nd_pred(d)
        else:
            u0 = d.y[0, :, 0].detach().cpu().numpy().astype(np.float64)
            v0 = d.y[0, :, 1].detach().cpu().numpy().astype(np.float64)
            spd = speed_nd(d)
        dsry = (Dy @ f.sr) / (d_bar * M_TO_CM)
        grad = np.hypot(f.dsrx, dsry)
        names = d.y_channel_names.split(",")
        mat = np.expm1(d.y[-1, :, names.index("Mat_log1p_nd")].double().numpy()) * MAT_S
        packs.append(dict(anchor=anchor, d=d, wall=wall, A=A, ei=ei, gt=gt, gt_f=gt_f,
                          f=f, spd=spd, u0=u0, v0=v0, Dx=Dx, Dy=Dy, u_ref=u_ref,
                          d_bar=d_bar, grad=grad, dsry=dsry, mat=mat, pos=pos))
    print("[i] %d vessels  flow=%s  GROW_HOPS=%d" % (len(packs), args.flow, GROW_HOPS))
    from src.core_physics.wall_cohort_splits import format_split_means, split_of
    print("[i] protocol FIT n=%d DEV n=%d SEALED closed (patient020 is FIT)"
          % (sum(split_of(p["anchor"]) == "fit" for p in packs),
             sum(split_of(p["anchor"]) == "dev" for p in packs)))

    def with_lumen(p, msk):
        off = grow_into_lumen(msk, p["wall"], p["A"], p["spd"], p["f"].sr,
                              lumen_hops=LUMEN_HOPS, speed_thresh=LUMEN_SPEED)
        return msk | off

    def report(name, mask_fn):
        scores, wf, of, per = [], [], [], {}
        for p in packs:
            pred = with_lumen(p, mask_fn(p))
            s = sc(pred, p["gt_f"], p["d"].edge_index)
            scores.append(s)
            per[p["anchor"]] = s
            wf.append(f1(pred & p["wall"], p["gt"] & p["wall"]))
            of.append(f1(pred & ~p["wall"], p["gt"] & ~p["wall"]))
        row = dict(score=float(np.mean(scores)), wall_f1=float(np.nanmean(wf)),
                   off_f1=float(np.nanmean(of)), per=per)
        acc[name] = row
        print("   %-40s %s  wall %.4f  off %.4f"
              % (name, format_split_means(per), row["wall_f1"], row["off_f1"]))
        return row

    acc = {}
    print("\n=== ALGEBRAIC EXTRA SEEDS (t=0 flow only) ===")
    shipped = report(
        "shipped hops=%d" % GROW_HOPS,
        lambda p: grow((p["f"].gate > 0) & p["wall"], p["wall"], p["A"], p["f"].sr, bio))

    report(
        "sep |grad sr| > |sgt|",
        lambda p: grow(((p["f"].gate > 0) | (p["grad"] > abs(sgt_cgs))) & p["wall"],
                       p["wall"], p["A"], p["f"].sr, bio))
    report(
        "sep dsrx < 0.5*sgt (looser)",
        lambda p: grow(((p["f"].gate > 0) | (p["f"].dsrx < 0.5 * sgt_cgs)) & p["wall"],
                       p["wall"], p["A"], p["f"].sr, bio))
    report(
        "local sr minima as extra seeds",
        lambda p: grow(((p["f"].gate > 0) | local_sr_minima(p["f"].sr, p["wall"], p["A"]))
                       & p["wall"], p["wall"], p["A"], p["f"].sr, bio))
    report(
        "minima AND sr < 4*lss",
        lambda p: grow(((p["f"].gate > 0) | (
            local_sr_minima(p["f"].sr, p["wall"], p["A"]) & (p["f"].sr < 4.0 * lss)
        )) & p["wall"], p["wall"], p["A"], p["f"].sr, bio))

    print("\n=== HOP-BOUNDED EXTRA ADMISSION (gate migrates a short way) ===")
    for eh, er in ((4, 2.5), (4, 3.0), (8, 2.5), (8, 3.0), (8, 4.0), (4, 4.0)):
        report(
            "two-stage +%d hops @ %.1f*lss" % (eh, er),
            lambda p, eh=eh, er=er: two_stage(
                (p["f"].gate > 0) & p["wall"], p["wall"], p["A"], p["f"].sr, bio, eh, er))
    for k, alpha in ((2, 2.5), (3, 2.5), (4, 2.2), (4, 2.5), (4, 2.8),
                     (5, 2.5), (8, 2.5), (8, 3.0), (12, 3.0)):
        def _near(p, k=k, alpha=alpha):
            seed = (p["f"].gate > 0) & p["wall"]
            hop = hop_from(seed, p["wall"], p["A"])
            extra = p["wall"] & (hop <= k) & (p["f"].sr < alpha * lss)
            return grow(seed | extra, p["wall"], p["A"], p["f"].sr, bio)
        report("extra seed hop<=%d & sr<%.1f*lss" % (k, alpha), _near)

    print("\n=== PER-VESSEL  extra hop<=4 & sr<2.5*lss  vs shipped ===")
    def extra_k4(p):
        seed = (p["f"].gate > 0) & p["wall"]
        hop = hop_from(seed, p["wall"], p["A"])
        extra = p["wall"] & (hop <= 4) & (p["f"].sr < 2.5 * lss)
        return grow(seed | extra, p["wall"], p["A"], p["f"].sr, bio)
    n_up = n_dn = 0
    deltas = []
    for p in packs:
        a = sc(with_lumen(p, grow((p["f"].gate > 0) & p["wall"], p["wall"], p["A"],
                                  p["f"].sr, bio)), p["gt_f"], p["d"].edge_index)
        b = sc(with_lumen(p, extra_k4(p)), p["gt_f"], p["d"].edge_index)
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

    # FN geography -- what the extra seeds would have to hit.
    print("\n=== UNGATED FN vs TN, t=0 features (pooled) ===")
    fn_sr, tn_sr, fn_g, tn_g, fn_hop, tn_hop = [], [], [], [], [], []
    n_fn = n_fn_near = n_fn_mid = n_fn_far = 0
    for p in packs:
        seed = (p["f"].gate > 0) & p["wall"]
        msk = grow(seed, p["wall"], p["A"], p["f"].sr, bio)
        gt_w = p["gt"] & p["wall"]
        fn = gt_w & ~msk
        tn = p["wall"] & ~gt_w & ~msk
        hop = hop_from(seed, p["wall"], p["A"])
        fn_sr.append(p["f"].sr[fn]); tn_sr.append(p["f"].sr[tn])
        fn_g.append(p["grad"][fn]); tn_g.append(p["grad"][tn])
        fn_hop.append(hop[fn]); tn_hop.append(hop[tn])
        n_fn += int(fn.sum())
        n_fn_near += int((fn & (hop <= 8) & (p["f"].sr < 3.0 * lss)).sum())
        n_fn_mid += int((fn & (hop <= 12) & (p["f"].sr < 4.0 * lss)).sum())
        n_fn_far += int((fn & (p["f"].sr >= 4.0 * lss)).sum())
    def q(a, qs=(10, 50, 90)):
        v = np.concatenate(a) if a else np.array([np.nan])
        if v.size == 0:
            return (np.nan,) * 3
        return tuple(float(x) for x in np.percentile(v, qs))
    print("   sr     FN p10/50/90  %8.2f %8.2f %8.2f   TN %8.2f %8.2f %8.2f" % (q(fn_sr) + q(tn_sr)))
    print("   |gsr|  FN p10/50/90  %8.2f %8.2f %8.2f   TN %8.2f %8.2f %8.2f" % (q(fn_g) + q(tn_g)))
    print("   hop    FN p10/50/90  %8.1f %8.1f %8.1f   TN %8.1f %8.1f %8.1f" % (q(fn_hop) + q(tn_hop)))
    print("   lss=%.1f  2*lss=%.1f  |sgt|=%.1f" % (lss, 2 * lss, abs(sgt_cgs)))
    print("   remaining FN after hops=%d: %d" % (GROW_HOPS, n_fn))
    print("     hop<=8  & sr<3*lss : %d  (hop-bounded extra-seed ceiling)" % n_fn_near)
    print("     hop<=12 & sr<4*lss : %d" % n_fn_mid)
    print("     sr>=4*lss          : %d  (needs evolving flow / true gate open)" % n_fn_far)

    if corr is not None:
        from src.inference.corrector_coupling import couple_flow_with_corrector

        print("\n=== CORRECTOR FIXED POINT (base flow = %s t=0, delta_mu=0.68) ===" % args.flow)
        dmu = 0.68

        def corr_state(p, niter, evolve_adm=False):
            """Union of gates opened by iterating clot -> corrector -> new shear.

            Returns ``(seed_union, sr_last, spd_nd)``.  Occlusion for the corrector is the
            shipped hops-grown mask from the *current* seed union, so the loop can start
            from the model's own prediction.  The prize readout is hops=0 on that union
            -- stacking hops=20 on an evolving gate is how the +0.05 ceiling collapsed.
            """
            wall, A, f = p["wall"], p["A"], p["f"]
            print("   [corr] %-12s niter=%d evolve_adm=%s" % (p["anchor"], niter, evolve_adm),
                  flush=True)
            seed = (f.gate > 0) & wall
            sr_grow = f.sr
            spd = p["spd"]
            u0_t = torch.tensor(p["u0"], dtype=torch.float32, device=device)
            v0_t = torch.tensor(p["v0"], dtype=torch.float32, device=device)

            class _V:
                def __init__(self, dd):
                    self.x = dd.x.to(device)
                    self.edge_index = dd.edge_index.to(device)
                    self.num_nodes = int(dd.num_nodes)
            dv = _V(p["d"])
            for _it in range(niter):
                msk = grow(seed, wall, A, sr_grow, bio)
                if not msk.any():
                    break
                delta = torch.tensor(msk.astype(np.float32) * dmu, device=device)
                with torch.no_grad():
                    uu, vv, _ = couple_flow_with_corrector(
                        dv, u0_t, v0_t, delta, corrector=corr, phys_cfg=phys,
                        device=device, num_hops=5)
                un = uu.detach().cpu().numpy().astype(np.float64)
                vn = vv.detach().cpu().numpy().astype(np.float64)
                spd = np.hypot(un, vn)
                sr = shear_rate_2d(p["Dx"] @ un, p["Dy"] @ un, p["Dx"] @ vn, p["Dy"] @ vn
                                   ) * (p["u_ref"] / p["d_bar"])
                dsx = (p["Dx"] @ sr) / (p["d_bar"] * M_TO_CM)
                g = gate_from_shear(sr, dsx, bio, wall=wall)
                nxt = seed | (g > 0)
                if evolve_adm:
                    sr_grow = sr
                if np.array_equal(nxt, seed) and not evolve_adm:
                    break
                seed = nxt
            return seed, sr_grow, spd

        cache = {}
        def corr_cached(p, niter, evolve_adm=False):
            key = (p["anchor"], int(niter), bool(evolve_adm))
            if key not in cache:
                cache[key] = corr_state(p, niter, evolve_adm=evolve_adm)
            return cache[key]

        def _corr_hops(p, niter, hops, evolve_adm=False):
            seed, sr_grow, _ = corr_cached(p, niter, evolve_adm=evolve_adm)
            return grow(seed, p["wall"], p["A"], sr_grow, bio, hops=hops)

        print("   [i] hops=0 is the prize: GT union-gate with no growth is ~0.834")
        n_new1, n_newN = [], []
        for p in packs:
            t0 = (p["f"].gate > 0) & p["wall"]
            s1, _, _ = corr_cached(p, 1)
            sN, _, _ = corr_cached(p, args.corr_iters)
            n_new1.append(int((s1 & ~t0).sum()))
            n_newN.append(int((sN & ~t0).sum()))
        print("   [i] new gates vs t=0: 1-iter mean %.1f nodes  %d-iter mean %.1f"
              % (float(np.mean(n_new1)), args.corr_iters, float(np.mean(n_newN))))
        for hops in (0, 6, GROW_HOPS):
            report("corrector 1-iter hops=%d" % hops,
                   lambda p, h=hops: _corr_hops(p, 1, h))
        report("corrector %d-iter hops=0" % args.corr_iters,
               lambda p: _corr_hops(p, args.corr_iters, 0))
        report("corrector %d-iter hops=0 evolve-adm" % args.corr_iters,
               lambda p: _corr_hops(p, args.corr_iters, 0, evolve_adm=True))
        report("corrector %d-iter hops=%d" % (args.corr_iters, GROW_HOPS),
               lambda p: _corr_hops(p, args.corr_iters, GROW_HOPS))

        scores, wf, of = [], [], []
        for p in packs:
            seed, sr_grow, spd = corr_cached(p, args.corr_iters, evolve_adm=True)
            msk = grow(seed, p["wall"], p["A"], sr_grow, bio, hops=0)
            off = grow_into_lumen(msk, p["wall"], p["A"], spd, sr_grow,
                                  lumen_hops=LUMEN_HOPS, speed_thresh=LUMEN_SPEED)
            pred = msk | off
            scores.append(sc(pred, p["gt_f"], p["d"].edge_index))
            wf.append(f1(pred & p["wall"], p["gt"] & p["wall"]))
            of.append(f1(pred & ~p["wall"], p["gt"] & ~p["wall"]))
        row = dict(score=float(np.mean(scores)), wall_f1=float(np.nanmean(wf)),
                   off_f1=float(np.nanmean(of)))
        acc["corrector hops=0 + evolved lumen spd"] = row
        print("   %-40s score %.4f  wall F1 %.4f  off F1 %.4f"
              % ("corrector hops=0 + evolved lumen spd",
                 row["score"], row["wall_f1"], row["off_f1"]))

    print("\n=== GT-FLOW UNION CEILING (same lumen) ===")
    def union_seed(p):
        cf = CACHE / f"{p['anchor']}.npz"
        if not cf.exists() or "sr_t" not in np.load(cf).files:
            return (p["f"].gate > 0) & p["wall"]
        z = np.load(cf)
        gser = np.zeros(len(p["wall"]))
        gser[z["wall_idx"]] = gate_from_shear(z["sr_t"], z["dsrx_t"], bio).max(0)
        return (gser > 0) & p["wall"]
    report("GT union-gate, no growth", lambda p: union_seed(p))
    report("GT union-gate + hops=6",
           lambda p: grow(union_seed(p), p["wall"], p["A"], p["f"].sr, bio, hops=6))
    report("GT union-gate + hops=%d" % GROW_HOPS,
           lambda p: grow(union_seed(p), p["wall"], p["A"], p["f"].sr, bio))

    shipped_s = shipped["score"]
    print("\n=== vs shipped TRAIN-mean %.4f (select on DEV) ===" % shipped_s)
    from src.core_physics.wall_cohort_splits import mean_by_split
    if "per" in shipped:
        b = mean_by_split(shipped["per"])
        for k, v in acc.items():
            if "per" not in v:
                print("   %-40s TRAIN-mean %+.4f" % (k, v["score"] - shipped_s))
                continue
            m = mean_by_split(v["per"])
            print("   %-40s FIT %+.4f  DEV %+.4f"
                  % (k, m["fit"]["mean"] - b["fit"]["mean"],
                     m["dev"]["mean"] - b["dev"]["mean"]))
    else:
        for k, v in acc.items():
            print("   %-40s %+.4f" % (k, v["score"] - shipped_s))

    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(flow=args.flow, acc=acc), indent=2))
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
