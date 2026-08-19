"""PHASE 8: the corrector opens gates, but the wrong ones.

``eval_evolving_gate_deploy.py``: 1-iter corrector opens ~15 new wall gates / vessel and
the deploy score drops.  GT union-gate with no growth is still +0.032.  This script asks
the next physical question: are the *right* new gates in the wake of existing clot, and
can we keep only those -- either by filtering the corrector, or by an algebraic
downstream/hop rule that never calls it.

    python scripts/eval_local_wake_gate.py
    python scripts/eval_local_wake_gate.py --skip-corrector
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


def hop_from(seed, wall, A, maxh=40):
    hop = np.full(len(wall), 99, dtype=np.int32)
    hop[seed] = 0
    cur = seed.copy()
    for h in range(1, maxh):
        nxt = ((A @ cur.astype(np.int8)) > 0) & wall & ~cur
        hop[nxt] = np.minimum(hop[nxt], h)
        cur = cur | nxt
    return hop


def lumen_uv(wall, A, u, v):
    """Mean velocity of non-wall neighbours -- wall velocity itself is ~0."""
    n = len(wall)
    ux, uy = np.zeros(n), np.zeros(n)
    widx = np.flatnonzero(wall)
    for i in widx:
        nbr = A.indices[A.indptr[i]:A.indptr[i + 1]]
        ln = nbr[~wall[nbr]]
        if ln.size == 0:
            continue
        ux[i] = float(u[ln].mean())
        uy[i] = float(v[ln].mean())
    return ux, uy


def downstream_of(seed, wall, A, ux, uy, pos, maxh=12):
    """Wall walk from seeds along the local near-wall flow direction."""
    out = np.zeros(len(wall), dtype=bool)
    cur = seed.copy()
    for _ in range(int(maxh)):
        nxt = np.zeros(len(wall), dtype=bool)
        for i in np.flatnonzero(cur):
            nbr = A.indices[A.indptr[i]:A.indptr[i + 1]]
            wn = nbr[wall[nbr] & ~cur[nbr] & ~out[nbr]]
            if wn.size == 0:
                continue
            flow = np.array([ux[i], uy[i]])
            nrm = np.linalg.norm(flow)
            if nrm < 1e-12:
                continue
            dxy = pos[wn] - pos[i]
            proj = dxy @ flow
            nxt[wn[proj > 0]] = True
        if not nxt.any():
            break
        out = out | nxt
        cur = nxt
    return out


def union_gate(p, bio):
    cf = CACHE / f"{p['anchor']}.npz"
    if not cf.exists():
        return (p["f"].gate > 0) & p["wall"]
    z = np.load(cf)
    if "sr_t" not in z.files:
        return (p["f"].gate > 0) & p["wall"]
    gser = np.zeros(len(p["wall"]))
    gser[z["wall_idx"]] = gate_from_shear(z["sr_t"], z["dsrx_t"], bio).max(0)
    return (gser > 0) & p["wall"]


def q(v, qs=(10, 50, 90)):
    v = np.asarray(v)
    if v.size == 0:
        return (np.nan,) * 3
    return tuple(float(x) for x in np.percentile(v, qs))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", default="gt", choices=["gt", "pred"])
    ap.add_argument("--skip-corrector", action="store_true")
    ap.add_argument("--save", default="outputs/phase8_local_wake_gate.json")
    args = ap.parse_args()
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lss = float(bio.lss)

    corr = None
    if not args.skip_corrector and CKPT.exists():
        from src.core_physics.coupled_shear_gnn import load_local_corrector
        from src.inference.corrector_coupling import couple_flow_with_corrector
        corr = load_local_corrector(CKPT, device)
        print("[i] corrector on %s" % device)
    else:
        print("[i] skipping corrector")

    packs = []
    for anchor in WALL_COHORT_V2_TRAIN:
        pth = DIR / f"{anchor}.pt"
        if not pth.exists():
            continue
        d = torch.load(pth, map_location="cpu", weights_only=False)
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
        f = t0_flow_fields(d, bio, hops=STENCIL[args.flow], flow_source=args.flow)
        pos = node_positions(d)
        if args.flow == "pred":
            u0 = d.u0_pred.reshape(-1).detach().cpu().numpy().astype(np.float64)
            v0 = d.v0_pred.reshape(-1).detach().cpu().numpy().astype(np.float64)
            spd = speed_nd_pred(d)
        else:
            u0 = d.y[0, :, 0].detach().cpu().numpy().astype(np.float64)
            v0 = d.y[0, :, 1].detach().cpu().numpy().astype(np.float64)
            spd = speed_nd(d)
        packs.append(dict(anchor=anchor, d=d, wall=wall, A=A, ei=ei, gt=gt, gt_f=gt_f,
                          f=f, spd=spd, u0=u0, v0=v0, pos=pos))
    print("[i] %d vessels  flow=%s" % (len(packs), args.flow))
    from src.core_physics.wall_cohort_splits import format_split_means, split_of
    print("[i] protocol FIT n=%d DEV n=%d SEALED closed (patient020 is FIT)"
          % (sum(split_of(p["anchor"]) == "fit" for p in packs),
             sum(split_of(p["anchor"]) == "dev" for p in packs)))

    def with_lumen(p, msk):
        off = grow_into_lumen(msk, p["wall"], p["A"], p["spd"], p["f"].sr,
                              lumen_hops=LUMEN_HOPS, speed_thresh=LUMEN_SPEED)
        return msk | off

    acc = {}

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
        print("   %-42s %s  wall %.4f  off %.4f"
              % (name, format_split_means(per), row["wall_f1"], row["off_f1"]))
        return row

    for p in packs:
        seed = (p["f"].gate > 0) & p["wall"]
        p["seed"] = seed
        p["shipped"] = grow(seed, p["wall"], p["A"], p["f"].sr, bio)
        p["union"] = union_gate(p, bio)
        p["hop"] = hop_from(seed, p["wall"], p["A"])
        ux, uy = lumen_uv(p["wall"], p["A"], p["u0"], p["v0"])
        p["down"] = downstream_of(seed, p["wall"], p["A"], ux, uy, p["pos"], maxh=12)
        p["prize"] = p["union"] & ~seed
        p["tn"] = p["wall"] & ~p["gt"] & ~p["shipped"]

    print("\n=== SHIPPED / CEILING ===")
    shipped = report("shipped hops=%d" % GROW_HOPS, lambda p: p["shipped"])
    report("GT union-gate, no growth", lambda p: p["union"])
    report("GT union extras OR shipped", lambda p: p["shipped"] | p["prize"])

    # Prize vs TN vs (later) corrector-new.
    print("\n=== PRIZE NODES (union extras) vs TN, t=0 ===")
    pr_sr, tn_sr, pr_hop, tn_hop, pr_dn, tn_dn = [], [], [], [], [], []
    n_pr = n_tn = n_pr_dn = n_tn_dn = 0
    n_pr_near = n_pr_in_shipped = 0
    for p in packs:
        pr, tn = p["prize"], p["tn"]
        n_pr += int(pr.sum()); n_tn += int(tn.sum())
        n_pr_dn += int((pr & p["down"]).sum()); n_tn_dn += int((tn & p["down"]).sum())
        n_pr_near += int((pr & (p["hop"] <= 8) & (p["f"].sr < 3.0 * lss)).sum())
        n_pr_in_shipped += int((pr & p["shipped"]).sum())
        pr_sr.append(p["f"].sr[pr]); tn_sr.append(p["f"].sr[tn])
        pr_hop.append(p["hop"][pr]); tn_hop.append(p["hop"][tn])
        pr_dn.append(p["down"][pr].astype(np.float64)); tn_dn.append(p["down"][tn].astype(np.float64))
    print("   n prize=%d  of which already in hops=20 shipped=%d" % (n_pr, n_pr_in_shipped))
    print("   sr     prize p10/50/90 %8.2f %8.2f %8.2f   TN %8.2f %8.2f %8.2f"
          % (q(np.concatenate(pr_sr)) + q(np.concatenate(tn_sr))))
    print("   hop    prize p10/50/90 %8.1f %8.1f %8.1f   TN %8.1f %8.1f %8.1f"
          % (q(np.concatenate(pr_hop)) + q(np.concatenate(tn_hop))))
    print("   downstream-of-seed frac   prize %.3f   TN %.3f"
          % (n_pr_dn / max(n_pr, 1), n_tn_dn / max(n_tn, 1)))
    print("   prize hop<=8 & sr<3*lss: %d / %d" % (n_pr_near, n_pr))

    print("\n=== ALGEBRAIC WAKE (no corrector) ===")
    for k, alpha in ((4, 2.5), (8, 2.5), (8, 3.0), (12, 3.0)):
        report(
            "shipped OR (down & hop<=%d & sr<%.1f lss)" % (k, alpha),
            lambda p, k=k, alpha=alpha: p["shipped"] | (
                p["down"] & (p["hop"] <= k) & (p["f"].sr < alpha * lss) & p["wall"]))
        report(
            "t0|(down hop<=%d sr<%.1f lss) +hops20" % (k, alpha),
            lambda p, k=k, alpha=alpha: grow(
                p["seed"] | (p["down"] & (p["hop"] <= k) & (p["f"].sr < alpha * lss)),
                p["wall"], p["A"], p["f"].sr, bio))

    if corr is not None:
        from src.inference.corrector_coupling import couple_flow_with_corrector
        print("\n=== CORRECTOR NEW GATES, LOCALITY FILTER ===")
        dmu = 0.68
        DxDy = {}
        for p in packs:
            Dx, Dy = build_mls_gradient(p["pos"], p["ei"], hops=STENCIL[args.flow])
            u_ref = float(p["d"].u_ref.reshape(-1)[0])
            d_bar = float(p["d"].d_bar.reshape(-1)[0])
            class _V:
                def __init__(self, dd):
                    self.x = dd.x.to(device)
                    self.edge_index = dd.edge_index.to(device)
                    self.num_nodes = int(dd.num_nodes)
            msk = p["shipped"]
            delta = torch.tensor(msk.astype(np.float32) * dmu, device=device)
            u0_t = torch.tensor(p["u0"], dtype=torch.float32, device=device)
            v0_t = torch.tensor(p["v0"], dtype=torch.float32, device=device)
            with torch.no_grad():
                uu, vv, _ = couple_flow_with_corrector(
                    _V(p["d"]), u0_t, v0_t, delta, corrector=corr, phys_cfg=phys,
                    device=device, num_hops=5)
            un = uu.detach().cpu().numpy().astype(np.float64)
            vn = vv.detach().cpu().numpy().astype(np.float64)
            sr = shear_rate_2d(Dx @ un, Dy @ un, Dx @ vn, Dy @ vn) * (u_ref / d_bar)
            dsx = (Dx @ sr) / (d_bar * M_TO_CM)
            g = gate_from_shear(sr, dsx, bio, wall=p["wall"])
            p["corr_new"] = (g > 0) & p["wall"] & ~p["seed"]
            print("   [corr] %-12s new=%d  prize-hit=%d  prize=%d"
                  % (p["anchor"], int(p["corr_new"].sum()),
                     int((p["corr_new"] & p["prize"]).sum()), int(p["prize"].sum())),
                  flush=True)

        cn_sr, cn_hop, n_cn, n_hit, n_pr2 = [], [], 0, 0, 0
        for p in packs:
            cn = p["corr_new"]
            n_cn += int(cn.sum()); n_hit += int((cn & p["prize"]).sum()); n_pr2 += int(p["prize"].sum())
            cn_sr.append(p["f"].sr[cn]); cn_hop.append(p["hop"][cn])
        print("   pooled corr_new=%d  intersect prize=%d / prize=%d  prec=%.3f rec=%.3f"
              % (n_cn, n_hit, n_pr2, n_hit / max(n_cn, 1), n_hit / max(n_pr2, 1)))
        print("   sr     corr_new p10/50/90 %8.2f %8.2f %8.2f" % q(np.concatenate(cn_sr)))
        print("   hop    corr_new p10/50/90 %8.1f %8.1f %8.1f" % q(np.concatenate(cn_hop)))

        report("shipped OR all corr_new", lambda p: p["shipped"] | p["corr_new"])
        for k in (2, 4, 8, 12):
            report("shipped OR (corr_new hop<=%d)" % k,
                   lambda p, k=k: p["shipped"] | (p["corr_new"] & (p["hop"] <= k)))
        report("shipped OR (corr_new & downstream)",
               lambda p: p["shipped"] | (p["corr_new"] & p["down"]))
        report("shipped OR (corr_new & down & hop<=8)",
               lambda p: p["shipped"] | (p["corr_new"] & p["down"] & (p["hop"] <= 8)))

    shipped_s = shipped["score"]
    print("\n=== vs shipped TRAIN-mean %.4f (select on DEV) ===" % shipped_s)
    from src.core_physics.wall_cohort_splits import mean_by_split
    b = mean_by_split(shipped["per"])
    for k, v in acc.items():
        m = mean_by_split(v["per"])
        print("   %-42s FIT %+.4f  DEV %+.4f"
              % (k, m["fit"]["mean"] - b["fit"]["mean"],
                 m["dev"]["mean"] - b["dev"]["mean"]))

    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(flow=args.flow, acc=acc), indent=2))
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
