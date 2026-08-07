#!/usr/bin/env python
"""Z4 (WALL_MODEL_PLAN.md s17) -- is shear decodable, and is it worth decoding?

Two separate questions that are easy to conflate:

  (1) IS SHEAR USEFUL?   Does the TRUE shear field predict where clot forms, beyond the
      geometric features we already have for free? If GT shear adds nothing over
      `-anaSpd`/`-sdf`/`wgrad`, then a shear decoder head cannot help no matter how good it is.

  (2) IS SHEAR DECODABLE? How well can shear be recovered from deploy-legal inputs? Reported
      here as the ceiling a decoder would have to hit, plus -- when a shear-head checkpoint is
      supplied via --kine-ckpt -- what the actual head achieves.

Question (1) gates question (2). Run this before investing in the decoder.

Context from s16.3: across 35 vessels the GT-CFD `shear` field has mean clot-AUC 0.343, i.e.
consistently INVERTED (low shear -> clot, AUC ~0.657 as `-shear`), while `mu_eff` -- a pure
function of shear -- averages 0.482 and is anti-predictive on 19/35. So shear-derived quantities
are not automatically useful, and that is exactly what (1) measures.

Usage:
    python scripts/diag_shear_decodability.py --all
    python scripts/diag_shear_decodability.py --all --kine-ckpt path/to/kine_with_shear_head.pth
"""
from __future__ import annotations

import argparse
import glob
import os
import statistics as st
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_gen.lib.legal_priors import apply_prior_source  # noqa: E402
from src.data_gen.lib.node_feature_assembly import (  # noqa: E402
    mass_conserving_umax_nd,
    width_nd_to_radius_nd,
)

ANCHOR_DIR = Path("data/processed/graphs_biochem_anchors")
MAT = 15


def auc(s: torch.Tensor, l: torch.Tensor) -> float:
    s = s.double().reshape(-1)
    l = l.reshape(-1).bool()
    if int(l.sum()) < 5 or int((~l).sum()) < 5:
        return float("nan")
    r = s.argsort().argsort().double() + 1
    n1, n0 = int(l.sum()), int((~l).sum())
    return (r[l].sum().item() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _hop(f, row, col, deg, n):
    a = torch.zeros(n, dtype=f.dtype)
    a.index_add_(0, row, f[col])
    return a / deg


def features(d):
    """Returns (band, label, dict of feature tensors, gt_shear)."""
    x, y, ei = d.x, d.y, d.edge_index
    n = x.shape[0]
    row, col = ei
    deg = torch.zeros(n)
    deg.index_add_(0, row, torch.ones(row.shape[0]))
    deg = deg.clamp(min=1.0)
    band = d.mask_wall.reshape(-1).bool().clone()
    for _ in range(3):
        band = band | (_hop(band.float(), row, col, deg, n) > 0)
    lab = (y[-1, :, MAT] > 1e-4)

    sdf = x[:, 2].clamp_min(0.0)
    w = x[:, 15]
    r = width_nd_to_radius_nd(w).reshape(-1)
    umax = mass_conserving_umax_nd(r).reshape(-1)
    rl = (r - torch.minimum(sdf, r)).clamp_min(0.0)
    aspd = torch.clamp(umax * (1.0 - (rl**2 / (r**2 + 1e-12))), min=0.0)
    wg = _hop(w, row, col, deg, n) - w

    # true shear rate from the converged CFD field
    u, v = x[:, 11], x[:, 12]
    spd = torch.sqrt(u * u + v * v)
    pos = x[:, :2]
    dist = (pos[row] - pos[col]).norm(dim=1).clamp(min=1e-6)
    sg = torch.zeros(n)
    sg.index_add_(0, row, (spd[row] - spd[col]).abs() / dist)
    sg = sg / deg

    geo = {"-anaSpd": -aspd, "-sdf": -sdf, "wgrad": wg}
    return band, lab, geo, sg


def fit_logreg(X, y, iters=400, lr=0.05):
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    pw = ((y == 0).sum() / y.sum().clamp(min=1)).item()
    for _ in range(iters):
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(
            X @ w + b, y, pos_weight=torch.tensor(pw)
        ).backward()
        opt.step()
    return w.detach(), b.detach()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anchors", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--kine-ckpt", default="", help="kine model WITH a shear decoder head")
    args = ap.parse_args()

    if args.all:
        pids = [os.path.basename(f)[:-3] for f in sorted(glob.glob(str(ANCHOR_DIR / "patient*.pt")))
                if "mirror" not in f]
    elif args.anchors.strip():
        pids = [s.strip() for s in args.anchors.split(",") if s.strip()]
    else:
        pids = ["patient039", "patient040", "patient041", "patient042", "patient043", "patient044"]

    D = {}
    for pid in pids:
        f = ANCHOR_DIR / f"{pid}.pt"
        if not f.exists():
            continue
        d = torch.load(f, map_location="cpu", weights_only=False)
        if int((d.y[-1, :, MAT] > 1e-4).sum()) < 20:
            continue
        band, lab, geo, sg = features(d)
        D[pid] = (band, lab, geo, sg)
    print(f"[i] vessels: {len(D)}")

    # ---- Q1: does TRUE shear add anything over free geometry? ----
    print("\nQ1. Does GT shear add predictive power over deploy-legal geometry?")
    print(f"{'vessel':<12}{'geom only':>11}{'geom+shear':>12}{'gain':>8}{'shear alone':>13}")
    gains, g_only, gs = [], [], []
    for pid, (band, lab, geo, sg) in D.items():
        Xg = torch.stack([geo[k][band] for k in geo], 1)
        Xs = torch.cat([Xg, sg[band].reshape(-1, 1)], 1)
        y = lab[band].float()
        Xg = (Xg - Xg.mean(0)) / Xg.std(0).clamp(min=1e-6)
        Xs = (Xs - Xs.mean(0)) / Xs.std(0).clamp(min=1e-6)
        wg_, bg_ = fit_logreg(Xg, y)
        ws_, bs_ = fit_logreg(Xs, y)
        a_g, a_s = auc(Xg @ wg_ + bg_, y), auc(Xs @ ws_ + bs_, y)
        a_sh = auc(-sg[band], lab[band])
        gains.append(a_s - a_g); g_only.append(a_g); gs.append(a_s)
        print(f"{pid:<12}{a_g:>11.3f}{a_s:>12.3f}{a_s - a_g:>+8.3f}{a_sh:>13.3f}")
    print(f"{'MEAN':<12}{st.mean(g_only):>11.3f}{st.mean(gs):>12.3f}{st.mean(gains):>+8.3f}")
    print("\n  VERDICT: if mean gain is < ~0.02, a shear decoder cannot help the clot model,")
    print("  however accurate it is -- geometry already carries that information.")

    # ---- Q2: decodability ceiling from legal inputs ----
    print("\nQ2. How well is shear recoverable from deploy-legal inputs?")
    print(f"{'vessel':<12}{'r(analytic,GT)':>16}{'relL2':>9}")
    rs = []
    for pid, (band, lab, geo, sg) in D.items():
        d = torch.load(ANCHOR_DIR / f"{pid}.pt", map_location="cpu", weights_only=False)
        a = apply_prior_source(d, "analytic")
        n = d.x.shape[0]
        row, col = d.edge_index
        deg = torch.zeros(n); deg.index_add_(0, row, torch.ones(row.shape[0])); deg = deg.clamp(min=1)
        au, av = a.x[:, 11], a.x[:, 12]
        asp = torch.sqrt(au * au + av * av)
        pos = d.x[:, :2]
        dist = (pos[row] - pos[col]).norm(dim=1).clamp(min=1e-6)
        ag = torch.zeros(n); ag.index_add_(0, row, (asp[row] - asp[col]).abs() / dist); ag = ag / deg
        m = band
        r = torch.corrcoef(torch.stack([ag[m], sg[m]]))[0, 1].item()
        rel = (torch.norm(ag[m] - sg[m]) / torch.norm(sg[m]).clamp(min=1e-12)).item()
        rs.append(r)
        print(f"{pid:<12}{r:>16.3f}{rel:>9.3f}")
    print(f"{'MEAN':<12}{st.mean(rs):>16.3f}")

    if args.kine_ckpt:
        print(f"\n[i] shear-head checkpoint supplied: {args.kine_ckpt}")
        print("[i] wire the head's shear output in here and re-run Q2 against it;")
        print("    the analytic column above is the baseline it must beat.")
    else:
        print("\n[i] no --kine-ckpt given: Q2 reports the ANALYTIC baseline only.")
        print("    Re-run with the new shear-head model to see whether it beats this.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
