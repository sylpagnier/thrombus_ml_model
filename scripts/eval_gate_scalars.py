"""Sweep the COMSOL gate scalars for the 0.9 wall-domain target.

FIT wall deploy is 0.858, dragged by over-ignition (018/019/025: FN=0, all error is
t=0-gate FP) and by ungated FN (012/028).  Longer growth cannot fix both.  Tightening
the separation threshold ``sgt`` is a 1-D precision knob on the same law; loosening
``lss`` is the recall knob.  Select on DEV, do not open SEALED.

    python scripts/eval_gate_scalars.py
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from predict_wall_clot import GROW_HOPS, LUMEN_HOPS, LUMEN_SPEED, RELAX, STENCIL  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.physics_lumen_model import grow_into_lumen, speed_nd  # noqa: E402
from src.core_physics.physics_wall_model import gate_from_shear, t0_flow_fields  # noqa: E402
from src.core_physics.species_pushforward_continuous import (  # noqa: E402
    resolve_deploy_eval_time_index,
)
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.core_physics.wall_cohort_splits import (  # noqa: E402
    DEV, FIT, MIN_T, SEALED, format_split_means, mean_by_split, split_of,
)
from src.evaluation.clot_relaxed_metrics import (  # noqa: E402
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
)

DIR = REPO / "data/processed/graphs_biochem_anchors"


def domain_score(pred, gt, ei, domain, wall):
    if int((gt & domain).sum()) == 0:
        return float("nan")
    dom = torch.tensor(domain.astype(np.float32))
    m = compute_clot_relaxed_metrics(
        torch.tensor(pred.astype(np.float32)) * dom,
        torch.tensor(gt.astype(np.float32)) * dom,
        ei, wall_mask=torch.tensor(wall))
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
    ap.add_argument("--save", default="outputs/phase8_gate_scalars.json")
    args = ap.parse_args()
    bio0, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    print("[i] SEALED closed (%s)" % ", ".join(SEALED))
    print("[i] shipped sgt=%.3e  lss=%.1f" % (float(bio0.sgt), float(bio0.lss)))

    packs = []
    for anchor in list(FIT) + list(DEV):
        pth = DIR / f"{anchor}.pt"
        if not pth.exists():
            continue
        d = torch.load(pth, map_location="cpu", weights_only=False)
        if int(d.y.shape[0]) < MIN_T:
            continue
        wall = d.mask_wall.reshape(-1).bool().numpy()
        t_eval = resolve_deploy_eval_time_index(int(d.y.shape[0]))
        gt = gt_clot_phi_at_time(d, t_eval, phys, device=torch.device("cpu")).reshape(-1).numpy() > 0.5
        if (gt & wall).sum() == 0:
            continue
        ei = d.edge_index.detach().cpu().numpy()
        A = adj(ei, len(wall))
        f = t0_flow_fields(d, bio0, hops=STENCIL["gt"], flow_source="gt")
        packs.append(dict(anchor=anchor, split=split_of(anchor), d=d, wall=wall, A=A,
                          gt=gt, sr=f.sr, dsrx=f.dsrx, spd=speed_nd(d), ei=d.edge_index))

    def score_cfg(bio):
        wall_s, full_s = {}, {}
        for p in packs:
            g = gate_from_shear(p["sr"], p["dsrx"], bio, wall=p["wall"])
            msk = grow((g > 0) & p["wall"], p["wall"], p["A"], p["sr"], bio)
            off = grow_into_lumen(msk, p["wall"], p["A"], p["spd"], p["sr"],
                                  lumen_hops=LUMEN_HOPS, speed_thresh=LUMEN_SPEED)
            pred = msk | off
            wall_s[p["anchor"]] = domain_score(pred, p["gt"], p["ei"], p["wall"], p["wall"])
            full_s[p["anchor"]] = domain_score(
                pred, p["gt"], p["ei"], np.ones(len(p["wall"]), dtype=bool), p["wall"])
        return wall_s, full_s

    shipped_w, shipped_f = score_cfg(bio0)
    print("shipped wall %s" % format_split_means(shipped_w))
    print("        full %s" % format_split_means(shipped_f))

    print("\n=== sgt sweep (lss fixed) ===")
    acc = {}
    for sgt in (-5.0e4, -6.0e4, -7.5e4, -8.5e4, -9.0e4, -1.0e5, -1.2e5, -1.5e5):
        bio = replace(bio0, sgt=sgt)
        w, fsc = score_cfg(bio)
        acc["sgt=%.1e" % sgt] = dict(wall=w, full=fsc)
        dw = {a: w[a] - shipped_w[a] for a in w}
        print("   sgt=%+.1e  wall %s" % (sgt, format_split_means(w)))
        print("               dW   %s  full %s" % (format_split_means(dw), format_split_means(fsc)))

    print("\n=== lss sweep (sgt fixed) ===")
    for lss in (15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0):
        bio = replace(bio0, lss=lss)
        w, fsc = score_cfg(bio)
        acc["lss=%.0f" % lss] = dict(wall=w, full=fsc)
        dw = {a: w[a] - shipped_w[a] for a in w}
        print("   lss=%4.0f     wall %s" % (lss, format_split_means(w)))
        print("               dW   %s  full %s" % (format_split_means(dw), format_split_means(fsc)))

    print("\n=== DEV selection (SEALED closed) ===")
    freeze = []
    for k, v in acc.items():
        dw = {a: v["wall"][a] - shipped_w[a] for a in v["wall"]}
        df = {a: v["full"][a] - shipped_f[a] for a in v["full"]}
        mw, mf = mean_by_split(dw), mean_by_split(df)
        if ((mw["fit"]["mean"] or 0) > 1e-6 and (mw["dev"]["mean"] or 0) > 1e-6
                and (mf["fit"]["mean"] or 0) > -1e-4 and (mf["dev"]["mean"] or 0) > -1e-4):
            freeze.append((k, mw["fit"]["mean"], mw["dev"]["mean"],
                           mf["fit"]["mean"], mf["dev"]["mean"]))
    if freeze:
        print("   same-sign wall gain, full-mesh not down:")
        for row in freeze:
            print("      %s  wall FIT %+.4f DEV %+.4f  full FIT %+.4f DEV %+.4f" % row)
    else:
        print("   no scalar improves wall on FIT+DEV without costing full-mesh.")

    Path(args.save).write_text(json.dumps(dict(
        shipped_wall=shipped_w, shipped_full=shipped_f,
        sweeps={k: {kk: vv for kk, vv in v.items()} for k, v in acc.items()},
    ), indent=2, default=float))
    print("wrote %s" % args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
