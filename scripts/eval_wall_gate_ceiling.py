"""PHASE 8: can the wall MASK pick up the 339 ungated false negatives?

``scripts/diag_wall_error.py``: 19.3% of GT wall clot sits behind a CLOSED t=0 gate, and
0% of GT wall clot that IS gated is missed.  Graph-growth FP is 2 nodes.  So the wall
score is the t=0 gate, and the misses are nodes the frozen law never opens.

Two physics stories, scored as ceilings:

    union-gate     OR of the COMSOL gate over every timestep of GT flow
                   -- ceiling on any evolving-flow model for the MASK
    field          wall & (GT Mat >= crit) -- the viscosity law itself
    relax/hops     looser admission / longer growth, still from the t=0 gate
                   -- deployable, no new field

    python scripts/eval_wall_gate_ceiling.py
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

from predict_wall_clot import GROW_HOPS, LUMEN_HOPS, LUMEN_SPEED, RELAX, node_pos  # noqa: E402
from src.biochem_gnn.mat_growth_simple import WALL_COHORT_V2_TRAIN  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.physics_lumen_model import grow_into_lumen, speed_nd  # noqa: E402
from src.core_physics.physics_wall_model import gate_from_shear, t0_flow_fields  # noqa: E402
from src.core_physics.species_pushforward_continuous import (  # noqa: E402
    resolve_deploy_eval_time_index,
)
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.evaluation.clot_relaxed_metrics import (  # noqa: E402
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
)

DIR = REPO / "data/processed/graphs_biochem_anchors"
CACHE = REPO / "outputs/wall_species_cache"
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


def grow(seed, wall, A, bio, sr, hops, relax):
    cur = seed.copy()
    adm = (sr < float(bio.lss) * relax) & wall
    for _ in range(hops):
        cur = cur | (((A @ cur.astype(np.int8)) > 0) & adm)
    return cur


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default="outputs/phase8_wall_gate_ceiling.json")
    args = ap.parse_args()
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")

    packs = []
    for anchor in WALL_COHORT_V2_TRAIN:
        pk, cf = DIR / f"{anchor}.pt", CACHE / f"{anchor}.npz"
        if not (pk.exists() and cf.exists()):
            continue
        z = np.load(cf)
        if "sr_t" not in z.files:
            continue
        d = torch.load(pk, map_location="cpu", weights_only=False)
        if int(d.y.shape[0]) < 150:
            continue
        wall = d.mask_wall.reshape(-1).bool().numpy()
        ei = d.edge_index.detach().cpu().numpy()
        n = len(wall)
        A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
        A = ((A + A.T) > 0).astype(np.int8)
        t_eval = resolve_deploy_eval_time_index(int(d.y.shape[0]))
        gt_f = gt_clot_phi_at_time(d, t_eval, phys, device=torch.device("cpu")).reshape(-1)
        gt = gt_f.numpy() > 0.5
        if (gt & wall).sum() == 0:
            continue
        f = t0_flow_fields(d, bio, hops=3, flow_source="gt")
        names = d.y_channel_names.split(",")
        mat = np.expm1(d.y[-1, :, names.index("Mat_log1p_nd")].double().numpy()) * MAT_S
        gser = np.zeros((z["sr_t"].shape[0], n))
        gser[:, z["wall_idx"]] = gate_from_shear(z["sr_t"], z["dsrx_t"], bio)
        union = (gser.max(0) > 0) & wall
        packs.append(dict(anchor=anchor, d=d, wall=wall, A=A, gt=gt, gt_f=gt_f,
                          f=f, mat=mat, union=union, spd=speed_nd(d)))

    print("[i] %d vessels" % len(packs))

    def report(name, mask_fn, lumen=True):
        scores, wf, of = [], [], []
        for p in packs:
            msk = mask_fn(p)
            if lumen:
                off = grow_into_lumen(msk, p["wall"], p["A"], p["spd"], p["f"].sr,
                                      lumen_hops=LUMEN_HOPS, speed_thresh=LUMEN_SPEED)
                pred = msk | off
            else:
                pred = msk
            scores.append(sc(pred, p["gt_f"], p["d"].edge_index))
            wf.append(f1(pred & p["wall"], p["gt"] & p["wall"]))
            of.append(f1(pred & ~p["wall"], p["gt"] & ~p["wall"]))
        print("   %-36s score %.4f  wall F1 %.4f  off F1 %.4f"
              % (name, np.mean(scores), np.nanmean(wf), np.nanmean(of)))
        return dict(score=float(np.mean(scores)), wall_f1=float(np.nanmean(wf)),
                    off_f1=float(np.nanmean(of)))

    print("\n=== WALL MASK CEILINGS (with shipped speed lumen) ===")
    out = {}
    out["t0 gate+growth"] = report(
        "t0 gate + growth (shipped)",
        lambda p: grow((p["f"].gate > 0) & p["wall"], p["wall"], p["A"], bio,
                       p["f"].sr, GROW_HOPS, RELAX))
    out["union gate+growth"] = report(
        "UNION-over-time gate + growth",
        lambda p: grow(p["union"], p["wall"], p["A"], bio, p["f"].sr, GROW_HOPS, RELAX))
    out["union gate only"] = report(
        "UNION-over-time gate, no growth",
        lambda p: p["union"])
    out["field Mat>=crit"] = report(
        "wall & (GT Mat >= crit)",
        lambda p: p["wall"] & (p["mat"] >= CRIT))
    out["field no lumen"] = report(
        "wall & (GT Mat >= crit), no lumen",
        lambda p: p["wall"] & (p["mat"] >= CRIT), lumen=False)

    print("\n=== DEPLOYABLE: t=0 gate, sweep admission / hops ===")
    best = None
    for hops in (4, 6, 8, 12):
        for relax in (1.5, 2.0, 3.0, 4.0, 6.0):
            r = report(
                "hops=%d relax=%.1f" % (hops, relax),
                lambda p, h=hops, rx=relax: grow(
                    (p["f"].gate > 0) & p["wall"], p["wall"], p["A"], bio,
                    p["f"].sr, h, rx))
            if best is None or r["score"] > best["score"]:
                best = dict(hops=hops, relax=relax, **r)
    out["best_relax"] = best
    print("   best deployable: hops=%d relax=%.1f  score %.4f"
          % (best["hops"], best["relax"], best["score"]))

    shipped = out["t0 gate+growth"]["score"]
    print("\n=== vs shipped wall+speed %.4f ===" % shipped)
    for k, v in out.items():
        if isinstance(v, dict) and "score" in v:
            print("   %-36s %+.4f" % (k, v["score"] - shipped))

    path = Path(args.save)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print("\nwrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
