"""Re-score wall-model knobs under the FIT / DEV / SEALED protocol.

Phase 7/8 evals averaged WALL_COHORT_V2_TRAIN, which mixes FIT with DEV (039/040/041/044)
and treats patient020 as if it were a holdout -- it is FIT.  This script is the comparison
table going forward:

    FIT     propose / fit scalars (in-sample, not the decision)
    DEV     select  -- never fitted
    SEALED  not opened here

Truncated (T<150) and empty-GT vessels dropped everywhere.

    python scripts/eval_wall_protocol.py
    python scripts/eval_wall_protocol.py --open-sealed   # only after DEV has frozen a choice
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
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.physics_lumen_model import grow_into_lumen, speed_nd  # noqa: E402
from src.core_physics.physics_wall_model import gate_from_shear, t0_flow_fields  # noqa: E402
from src.core_physics.shear_redistribution import build_crosssection_operator, sdf_nd  # noqa: E402
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


def grow(seed, wall, A, sr, bio, hops, relax=RELAX):
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--open-sealed", action="store_true",
                    help="score SEALED (spend it). Default: FIT+DEV only.")
    ap.add_argument("--save", default="outputs/phase8_wall_protocol.json")
    args = ap.parse_args()
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    lss = float(bio.lss)

    want = list(FIT) + list(DEV)
    if args.open_sealed:
        want += list(SEALED)
        print("[WARN] SEALED is open -- do not use these numbers to choose a knob")
    else:
        print("[i] SEALED closed (%s)" % ", ".join(SEALED))

    packs = []
    skipped = []
    for anchor in want:
        pth = DIR / f"{anchor}.pt"
        if not pth.exists():
            skipped.append((anchor, split_of(anchor), "no pack"))
            continue
        d = torch.load(pth, map_location="cpu", weights_only=False)
        if int(d.y.shape[0]) < MIN_T:
            skipped.append((anchor, split_of(anchor), "T=%d" % int(d.y.shape[0])))
            continue
        wall = d.mask_wall.reshape(-1).bool().numpy()
        t_eval = resolve_deploy_eval_time_index(int(d.y.shape[0]))
        gt_f = gt_clot_phi_at_time(d, t_eval, phys, device=torch.device("cpu")).reshape(-1)
        gt = gt_f.numpy() > 0.5
        if (gt & wall).sum() == 0:
            skipped.append((anchor, split_of(anchor), "empty GT"))
            continue
        ei = d.edge_index.detach().cpu().numpy()
        A = adj(ei, len(wall))
        f = t0_flow_fields(d, bio, hops=STENCIL["gt"], flow_source="gt")
        seed = (f.gate > 0) & wall
        pos = d.x[:, :2].detach().cpu().numpy().astype(np.float64)
        B = build_crosssection_operator(pos, sdf_nd(d), wall)
        packs.append(dict(anchor=anchor, split=split_of(anchor), d=d, wall=wall, A=A,
                          gt=gt, gt_f=gt_f, f=f, seed=seed, spd=speed_nd(d), B=B))
    by = {}
    for p in packs:
        by.setdefault(p["split"], []).append(p["anchor"])
    print("[i] eligible  FIT n=%d  DEV n=%d  SEALED n=%d"
          % (len(by.get("fit", [])), len(by.get("dev", [])), len(by.get("sealed", []))))
    print("    FIT  %s" % " ".join(a[-3:] for a in by.get("fit", [])))
    print("    DEV  %s" % " ".join(a[-3:] for a in by.get("dev", [])))
    if skipped:
        print("[i] dropped (not a split, a different quantity):")
        for a, sp, why in skipped:
            print("    %-12s %-6s %s" % (a, sp, why))

    def with_lumen(p, msk):
        off = grow_into_lumen(msk, p["wall"], p["A"], p["spd"], p["f"].sr,
                              lumen_hops=LUMEN_HOPS, speed_thresh=LUMEN_SPEED)
        return msk | off

    def score_fn(p, msk):
        pred = with_lumen(p, msk)
        return dict(
            score=sc(pred, p["gt_f"], p["d"].edge_index),
            wall_f1=f1(pred & p["wall"], p["gt"] & p["wall"]),
            off_f1=f1(pred & ~p["wall"], p["gt"] & ~p["wall"]),
        )

    def report(name, mask_fn):
        per = {}
        for p in packs:
            per[p["anchor"]] = score_fn(p, mask_fn(p))["score"]
        means = mean_by_split(per)
        print("   %-40s %s" % (name, format_split_means(per)))
        return dict(per=per, split=means)

    acc = {}
    print("\n=== HOPS (select on DEV) ===")
    for hops in (6, 12, 20, 40):
        acc["hops=%d" % hops] = report(
            "hops=%d" % hops,
            lambda p, h=hops: grow(p["seed"], p["wall"], p["A"], p["f"].sr, bio, hops=h))
    shipped_h = GROW_HOPS
    print("   shipped GROW_HOPS=%d  (was chosen on TRAIN-mean, which includes DEV)"
          % shipped_h)

    print("\n=== ALGEBRAIC EXTRA SEED (select on DEV; do not refit) ===")
    acc["extra hop<=4 sr<2.5 lss"] = report(
        "extra hop<=4 & sr<2.5*lss",
        lambda p: grow(
            p["seed"] | (p["wall"] & (hop_from(p["seed"], p["wall"], p["A"]) <= 4)
                         & (p["f"].sr < 2.5 * lss)),
            p["wall"], p["A"], p["f"].sr, bio, hops=GROW_HOPS))

    print("\n=== ALGEBRAIC WAKE re-grow (select on DEV) ===")
    def wake_mask(p, wake):
        shipped = grow(p["seed"], p["wall"], p["A"], p["f"].sr, bio, hops=GROW_HOPS)
        phi = np.asarray(p["B"] @ shipped.astype(np.float64)).reshape(-1)
        amp = np.clip(1.0 - wake * phi, 0.02, 1.0)
        sr2 = p["f"].sr * amp
        extra = gate_from_shear(sr2, p["f"].dsrx * amp, bio, wall=p["wall"]) > 0
        return grow(p["seed"] | extra, p["wall"], p["A"], sr2, bio, hops=GROW_HOPS)
    for wake in (0.5, 1.0, 2.0):
        acc["wake=%.1f re-grow" % wake] = report(
            "wake=%.1f re-grow hops=%d" % (wake, GROW_HOPS),
            lambda p, w=wake: wake_mask(p, w))

    # DEV n=3 is too coarse to override FIT (PHASE6 4.1).  Report both; only a
    # same-sign FIT+DEV gain is a freeze candidate, and SEALED stays closed.
    print("\n=== DEV SELECTION (SEALED not consulted) ===")
    def dev_mean(row):
        v = row["split"]["dev"]["mean"]
        return v if v is not None else float("-inf")
    def fit_mean(row):
        v = row["split"]["fit"]["mean"]
        return v if v is not None else float("-inf")
    hops_cands = {k: v for k, v in acc.items() if k.startswith("hops=")}
    best_hops_dev = max(hops_cands, key=lambda k: dev_mean(hops_cands[k]))
    best_hops_fit = max(hops_cands, key=lambda k: fit_mean(hops_cands[k]))
    print("   hops FIT picks %s  FIT %.4f  DEV %.4f"
          % (best_hops_fit, fit_mean(hops_cands[best_hops_fit]),
             dev_mean(hops_cands[best_hops_fit])))
    print("   hops DEV picks %s  DEV %.4f  FIT %.4f  shipped was hops=%d"
          % (best_hops_dev, dev_mean(hops_cands[best_hops_dev]),
             fit_mean(hops_cands[best_hops_dev]), GROW_HOPS))
    if best_hops_dev != best_hops_fit:
        print("   [i] FIT/DEV disagree on hops -- DEV n=3 is too coarse to override FIT.")
        print("       keep %s (FIT argmax); do not move shipped to DEV's %s."
              % (best_hops_fit, best_hops_dev))
    base = acc["hops=%d" % GROW_HOPS]
    extras = {k: v for k, v in acc.items() if not k.startswith("hops=")}
    print("   vs shipped hops=%d:" % GROW_HOPS)
    freeze = []
    for k, v in extras.items():
        dd = dev_mean(v) - dev_mean(base)
        df = fit_mean(v) - fit_mean(base)
        print("   %-36s DEV %+.4f  FIT %+.4f" % (k, dd, df))
        if dd > 1e-6 and df > 1e-6:
            freeze.append((k, dd, df))
    if freeze:
        print("   same-sign FIT+DEV gains (still do not open SEALED):")
        for k, dd, df in freeze:
            print("      %s  DEV %+.4f  FIT %+.4f" % (k, dd, df))
    else:
        print("   no extra arm improves both FIT and DEV.")

    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    dump = {k: dict(split=v["split"], per=v["per"]) for k, v in acc.items()}
    dump["_meta"] = dict(open_sealed=bool(args.open_sealed), grow_hops_shipped=GROW_HOPS,
                         eligible={s: by.get(s, []) for s in ("fit", "dev", "sealed")},
                         skipped=[list(x) for x in skipped])
    out.write_text(json.dumps(dump, indent=2))
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
