"""PHASE 8: apply the quadratic-mesh shell to the SHIPPED speed lumen arm.

``scripts/diag_offwall_owner.py`` showed the model's own ``Mat`` cannot beat the speed
heuristic at any attenuation (0.766 vs 0.783).  The remaining off-wall prize is therefore
inside the speed arm itself.  That arm still dilates by graph hops, so it admits the
wall-normal mid-side family that carries no species (FINDINGS 8) -- the same bug the
Mat-magnitude arm already fixed with ``first_corner_shell``.

This script asks whether restricting the speed arm to the species-carrying shell raises
the full-mesh deploy score, which is the only number that decides whether it ships.

Arms, all seeded from the same wall mask:

    speed (shipped)     2 hops, speed_nd < 0.2
    speed & shell       same, intersected with the topological shell
    shell & owner clot  species row behind a committed wall node (no speed)
    shell & owner & spd  ... and speed_nd < thresh, thresh swept
    shell & spd         species row that is stagnant, no wall-clot seed

    python scripts/eval_speed_shell.py
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

from predict_wall_clot import LUMEN_HOPS, LUMEN_SPEED, node_pos, predict_wall_clot  # noqa: E402
from src.biochem_gnn.mat_growth_simple import WALL_COHORT_V2_TRAIN  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.physics_lumen_model import (  # noqa: E402
    grow_into_lumen, resolve_offwall_shell, speed_nd, wall_normal_projection,
)
from src.core_physics.physics_wall_model import t0_flow_fields  # noqa: E402
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default="outputs/phase8_speed_shell.json")
    args = ap.parse_args()
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")

    packs = []
    for anchor in WALL_COHORT_V2_TRAIN:
        p = DIR / f"{anchor}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        if int(d.y.shape[0]) < 150:
            continue
        wall = d.mask_wall.reshape(-1).bool().numpy()
        ei_np = d.edge_index.detach().cpu().numpy()
        n = len(wall)
        A = sp.coo_matrix((np.ones(ei_np.shape[1]), (ei_np[0], ei_np[1])), shape=(n, n)).tocsr()
        A = ((A + A.T) > 0).astype(np.int8)
        t_eval = resolve_deploy_eval_time_index(int(d.y.shape[0]))
        gt_f = gt_clot_phi_at_time(d, t_eval, phys, device=torch.device("cpu")).reshape(-1)
        gt = gt_f.numpy() > 0.5
        if gt.sum() == 0:
            continue
        pos = node_pos(d)
        shell = resolve_offwall_shell(pos, wall, ei_np)
        _, owner = wall_normal_projection(pos, wall)
        f = t0_flow_fields(d, bio, hops=3, flow_source="gt")
        spd = speed_nd(d)
        base, _ = predict_wall_clot(d, bio, flow="gt", lumen=False)
        packs.append(dict(anchor=anchor, d=d, wall=wall, A=A, gt=gt, gt_f=gt_f,
                          shell=shell, owner=owner, spd=spd, sr=f.sr, base=base))

    print("[i] %d vessels" % len(packs))

    def summarise(name, pred_fn):
        scores, off, wallf = [], [], []
        for p in packs:
            pred = pred_fn(p)
            scores.append(sc(pred, p["gt_f"], p["d"].edge_index))
            off.append(f1(pred & ~p["wall"], p["gt"] & ~p["wall"]))
            wallf.append(f1(pred & p["wall"], p["gt"] & p["wall"]))
        print("   %-28s score %.4f  wall F1 %.4f  off F1 %.4f"
              % (name, np.mean(scores), np.nanmean(wallf), np.nanmean(off)))
        return dict(score=float(np.mean(scores)), wall_f1=float(np.nanmean(wallf)),
                    off_f1=float(np.nanmean(off)))

    print("\n=== SPEED ARM x SPECIES SHELL ===")
    out = {}
    out["wall-only"] = summarise("wall-only", lambda p: p["base"])
    out["speed shipped"] = summarise(
        "speed (shipped)",
        lambda p: p["base"] | grow_into_lumen(
            p["base"], p["wall"], p["A"], p["spd"], p["sr"],
            lumen_hops=LUMEN_HOPS, speed_thresh=LUMEN_SPEED))
    out["speed & shell"] = summarise(
        "speed & shell",
        lambda p: p["base"] | (
            grow_into_lumen(p["base"], p["wall"], p["A"], p["spd"], p["sr"],
                            lumen_hops=LUMEN_HOPS, speed_thresh=LUMEN_SPEED)
            & p["shell"]))
    out["shell owner"] = summarise(
        "shell & owner clot",
        lambda p: p["base"] | (p["shell"] & p["base"][p["owner"]]))

    print("\n=== shell & owner clot & speed_nd < thresh ===")
    best = None
    for th in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0, np.inf):
        tag = "inf" if not np.isfinite(th) else "%.2f" % th
        r = summarise(
            "thresh " + tag,
            lambda p, t=th: p["base"] | (
                p["shell"] & p["base"][p["owner"]] & (p["spd"] < t)))
        if best is None or r["score"] > best["score"]:
            best = dict(th=float(th) if np.isfinite(th) else None, **r)
    out["best_shell_owner_spd"] = best

    print("\n=== shell & speed_nd < thresh  (no wall-clot seed) ===")
    best2 = None
    for th in (0.05, 0.10, 0.15, 0.20, 0.30):
        r = summarise(
            "spd-only " + "%.2f" % th,
            lambda p, t=th: p["base"] | (p["shell"] & (p["spd"] < t)))
        if best2 is None or r["score"] > best2["score"]:
            best2 = dict(th=th, **r)
    out["best_shell_spd"] = best2

    shipped = out["speed shipped"]["score"]
    print("\n=== vs shipped speed %.4f ===" % shipped)
    for k, v in out.items():
        if isinstance(v, dict) and "score" in v:
            print("   %-28s %+.4f" % (k, v["score"] - shipped))

    path = Path(args.save)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print("\nwrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
