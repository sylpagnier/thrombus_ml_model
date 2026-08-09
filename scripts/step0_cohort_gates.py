"""The COMSOL t=0 deposition gates, computed with CORRECT derivative operators, on the cohort.

Validated on patient007 against the raw COMSOL export: MLS(hops=3) reconstructs
``spf.sr`` at spearman 0.998 and ``d(spf.sr,x)`` at 0.990, and the two-gate union
classifies the final committed wall set at F1 0.848 (COMSOL's own gates: 0.854).
The repo's shipped ``G_x``/``G_y`` score 0.19 / 0.00 on the same comparison.

Everything here is deploy-legal under the Phase-3 bandaid: node positions, connectivity,
``u_ref``/``d_bar``, and the GT velocity field at t=0 only.

Usage:  python scripts/step0_cohort_gates.py [--out outputs/step0_cohort_gates.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig  # noqa: E402
from src.core_physics.clot_growth_masks import resolve_ceiling_mask  # noqa: E402
from src.core_physics.mls_gradient import build_mls_gradient, shear_rate_2d  # noqa: E402

CEILING_HOPS = 3
MAT_REL_FRAC = 0.10


def prf(pred, gt):
    tp = int((pred & gt).sum())
    p = tp / max(int(pred.sum()), 1)
    r = tp / max(int(gt.sum()), 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def auc(score, gt):
    pos, neg = score[gt], score[~gt]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    a = np.concatenate([pos, neg])
    o = a.argsort()
    r = np.empty(len(a))
    r[o] = np.arange(1, len(a) + 1)
    u, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    if cnt.max() > 1:
        s = np.zeros(len(u))
        np.add.at(s, inv, r)
        r = (s / cnt)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def gates_at_t0(data, bio, hops=3):
    """Return (sr_si [1/s], dsrx_si [1/(s*m)]) at every node from the t=0 GT flow."""
    pos = data.siren_pos.detach().numpy().astype(np.float64)
    ei = data.edge_index.numpy()
    u_ref = float(data.u_ref.reshape(-1)[0])
    d_bar = float(data.d_bar.reshape(-1)[0])
    Dx, Dy = build_mls_gradient(pos, ei, hops=hops)
    u = data.y[0, :, 0].numpy().astype(np.float64)
    v = data.y[0, :, 1].numpy().astype(np.float64)
    sr_nd = shear_rate_2d(Dx @ u, Dy @ u, Dx @ v, Dy @ v)
    sr = sr_nd * (u_ref / d_bar)                  # 1/s
    dsrx = (Dx @ sr) / d_bar                      # 1/(s*m)
    return sr, dsrx


def one(path, bio, hops):
    data = torch.load(path, map_location="cpu", weights_only=False)
    dev = torch.device("cpu")
    band = resolve_ceiling_mask(data, dev, bio, ceiling_hops=CEILING_HOPS).reshape(-1).bool().numpy()
    wall = data.mask_wall.reshape(-1).bool().numpy()
    names = data.y_channel_names.split(",")
    series = data.y[:, :, names.index("Mat_log1p_nd")]
    thr = float(series.max()) * MAT_REL_FRAC
    if thr <= 0 or not band.any():
        return None
    commit = ((series > thr).any(dim=0).numpy()) & band
    if commit.sum() == 0:
        return None
    sr, dsrx = gates_at_t0(data, bio, hops=hops)
    g_low = sr < float(bio.lss)
    g_sep = dsrx < float(bio.sgt)
    g_any = g_low | g_sep

    out = {"anchor": Path(path).stem, "n_band": int(band.sum()), "n_wall": int(wall.sum()),
           "n_commit": int(commit.sum()),
           "commit_on_wall": float(commit[wall].sum() / max(commit.sum(), 1)),
           "base_band": float(commit.sum() / band.sum()),
           "base_wall": float(commit[wall].sum() / max(wall.sum(), 1))}
    for nm, pred in (("low", g_low), ("sep", g_sep), ("any", g_any)):
        p, r, f1 = prf(pred[band] & wall[band], commit[band])
        out[f"band_{nm}_p"], out[f"band_{nm}_r"], out[f"band_{nm}_f1"] = p, r, f1
    p, r, f1 = prf(g_any[wall], commit[wall])
    out["wall_any_p"], out["wall_any_r"], out["wall_any_f1"] = p, r, f1
    out["auc_neg_sr_band"] = auc(-sr[band], commit[band])
    out["auc_neg_dsrx_band"] = auc(-dsrx[band], commit[band])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hops", type=int, default=3)
    ap.add_argument("--out", default="outputs/step0_cohort_gates.json")
    ap.add_argument("--anchors", default="")
    args = ap.parse_args()
    bio = BiochemConfig(phase="biochem")
    if args.anchors.strip():
        paths = [f"data/processed/graphs_biochem_anchors/{a.strip()}.pt"
                 for a in args.anchors.split(",") if a.strip()]
    else:
        paths = [p for p in sorted(glob.glob("data/processed/graphs_biochem_anchors/patient*.pt"))
                 if "mirror" not in p]
    rows = []
    print("%12s %6s %6s %7s %6s | %6s %6s %6s | %6s %6s %6s | %6s %6s"
          % ("vessel", "band", "wall", "commit", "onwall", "P", "R", "F1", "wP", "wR", "wF1",
             "aucSR", "aucDX"))
    for p in paths:
        try:
            r = one(p, bio, args.hops)
        except Exception as e:
            print("%12s  skip %s: %s" % (Path(p).stem, type(e).__name__, e))
            continue
        if r is None:
            continue
        rows.append(r)
        print("%12s %6d %6d %7d %6.2f | %6.3f %6.3f %6.3f | %6.3f %6.3f %6.3f | %6.3f %6.3f"
              % (r["anchor"], r["n_band"], r["n_wall"], r["n_commit"], r["commit_on_wall"],
                 r["band_any_p"], r["band_any_r"], r["band_any_f1"],
                 r["wall_any_p"], r["wall_any_r"], r["wall_any_f1"],
                 r["auc_neg_sr_band"], r["auc_neg_dsrx_band"]))
    if not rows:
        return 1

    def m(k):
        v = [r[k] for r in rows if r[k] == r[k]]
        return sum(v) / max(len(v), 1)

    print("\nn=%d vessels" % len(rows))
    print("  commit-on-wall fraction        : %.3f" % m("commit_on_wall"))
    print("  band base rate                 : %.3f" % m("base_band"))
    print("  BAND  two-gate union  P %.3f  R %.3f  F1 %.3f"
          % (m("band_any_p"), m("band_any_r"), m("band_any_f1")))
    print("  BAND  low-shear only  F1 %.3f | separation only F1 %.3f"
          % (m("band_low_f1"), m("band_sep_f1")))
    print("  WALL  two-gate union  P %.3f  R %.3f  F1 %.3f"
          % (m("wall_any_p"), m("wall_any_r"), m("wall_any_f1")))
    print("  mean AUC  -sr %.3f   -dsrx %.3f" % (m("auc_neg_sr_band"), m("auc_neg_dsrx_band")))
    f1s = sorted(r["band_any_f1"] for r in rows)
    print("  band F1 quantiles [min,25,50,75,max] = %s"
          % np.round(np.percentile(f1s, [0, 25, 50, 75, 100]), 3))
    print("  vessels with band F1 >= 0.6: %d/%d" % (sum(f > 0.6 for f in f1s), len(f1s)))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("  wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
