"""Which STATIC geometry contexts discriminate GT clot from non-clot in our data?

Motivation (docs/SPECIES_LEARNING_STRATEGY.md 6.10/6.11): the deploy bottleneck is
PRECISION -- the model over-paints eligible wall pockets because "eligible" (low shear)
looks like "committed" in every deployable observable EXCEPT geometry. Leg C showed a
minimal 3-channel geometry block (width / width_grad / wall_curvature) is the first real
non-flow precision lever. Before enriching the geometry block architecturally, MEASURE
which geometry contexts actually carry clot signal in the data, and whether that signal
GENERALIZES across patients (LOAO), and whether it is ORTHOGONAL to the analytic
shear/eligibility proxy (mu_prior / wss_prior).

All features are static, clot-blind, deployable (no kine solve, no GT flow):
  geometry block:
    sdf        x[:, SDF]          distance to wall (nd)
    width      x[:, WIDTH_ND]     local lumen width
    width_d1   x[:, WIDTH_D1]     flow-direction width gradient (expansion>0)
    width_d2   x[:, WIDTH_D2]     flow-direction width curvature
    expansion  nbr_mean(width)-width        1-hop expansion/contraction
    wall_curv1 mean_j(1-cos(n_i,n_j))       1-hop wall-normal bend
    wall_curv2 2-hop wall-normal bend        broader curvature
    downstream normalized BFS hop-distance from inlet (entrance length / position)
  shear/eligibility reference (NOT pure geometry):
    mu_prior   x[:, MU_PRIOR]     analytic Carreau viscosity ~ monotone low-shear proxy
    wss_prior  x[:, WSS_PRIOR]    analytic wall-shear-stress prior

Outputs (per anchor + pooled):
  (A) univariate AUC + standardized clot/non-clot mean-diff inside the wall+Khop band
  (B) LOAO logistic AUC for geometry-only / shear-only / geom+shear  (does geometry
      add over the shear proxy, and does it transfer across patients?)
  (C) pooled standardized logistic coefficients -> feature importance ranking
  (D) wall-only vs full-band univariate AUC (is geometry sharper at the wall?)

Run: python scripts/_diag_geom_context_importance.py [--hops 3] [--time last]

See also the comprehensive probe: scripts/_diag_clot_context_comprehensive.py
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.config import BiochemConfig, NodeFeat, PhysicsConfig  # noqa: E402
from src.core_physics.clot_growth_masks import resolve_ceiling_mask  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

ANCHOR_DIR = ROOT / "data" / "processed" / "graphs_biochem_anchors"
OUT = ROOT / "outputs" / "reports" / "comsol_validation" / "geom_context_importance.json"

GEOM_FEATS = ["sdf", "width", "width_d1", "width_d2", "expansion", "wall_curv1", "wall_curv2", "downstream"]
SHEAR_FEATS = ["mu_prior", "wss_prior"]
ALL_FEATS = GEOM_FEATS + SHEAR_FEATS


def _slice0(x: torch.Tensor, sl: slice) -> torch.Tensor:
    return x[:, sl].reshape(-1).to(torch.float64)


def _symmetric_edges(edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    row = torch.cat([edge_index[0], edge_index[1]])
    col = torch.cat([edge_index[1], edge_index[0]])
    return row, col


def _nbr_mean(values: torch.Tensor, row: torch.Tensor, col: torch.Tensor, n: int) -> torch.Tensor:
    deg = torch.zeros(n, dtype=torch.float64)
    deg.index_add_(0, row, torch.ones(row.numel(), dtype=torch.float64))
    deg = deg.clamp(min=1.0)
    acc = torch.zeros(n, dtype=torch.float64)
    acc.index_add_(0, row, values[col])
    return acc / deg


def _bfs_hops(n: int, row: torch.Tensor, col: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    INF = n + 5
    dist = torch.full((n,), INF, dtype=torch.long)
    dist[src] = 0
    frontier = src.clone()
    d = 0
    while bool(frontier.any()):
        d += 1
        nxt = torch.zeros(n, dtype=torch.bool)
        nxt[col[frontier[row]]] = True
        upd = nxt & (dist > d)
        if not bool(upd.any()):
            break
        dist[upd] = d
        frontier = upd
        if d > n:
            break
    finite = dist[dist < INF]
    dmax = float(finite.max().item()) if finite.numel() else 1.0
    dist = dist.clamp(max=int(dmax)).to(torch.float64)
    return dist / max(dmax, 1.0)


def _features(d, dev) -> dict[str, torch.Tensor]:
    x = d.x.to(dev)
    n = int(d.num_nodes)
    row, col = _symmetric_edges(d.edge_index.to(dev))
    sdf = _slice0(x, NodeFeat.SDF)
    width = _slice0(x, NodeFeat.WIDTH_ND)
    width_d1 = _slice0(x, NodeFeat.WIDTH_D1)
    width_d2 = _slice0(x, NodeFeat.WIDTH_D2)
    expansion = _nbr_mean(width, row, col, n) - width
    wn = x[:, NodeFeat.WALL_NORMAL].to(torch.float64)  # (n, 2)
    # 1-hop wall-normal bend: mean_j (1 - cos(n_i, n_j))
    dot = (wn[row] * wn[col]).sum(dim=1)
    cacc = torch.zeros(n, dtype=torch.float64)
    cacc.index_add_(0, row, (1.0 - dot))
    deg = torch.zeros(n, dtype=torch.float64)
    deg.index_add_(0, row, torch.ones(row.numel(), dtype=torch.float64))
    deg = deg.clamp(min=1.0)
    wall_curv1 = cacc / deg
    wall_curv2 = _nbr_mean(wall_curv1, row, col, n)  # 1 extra hop of smoothing = broader bend
    src = None
    for attr in ("mask_inlet", "inlet_mask"):
        m = getattr(d, attr, None)
        if m is not None:
            src = m.reshape(-1).bool().to(dev)
            break
    downstream = _bfs_hops(n, row, col, src) if src is not None and bool(src.any()) else torch.zeros(n, dtype=torch.float64)
    mu_prior = _slice0(x, NodeFeat.MU_PRIOR)
    wss_prior = _slice0(x, NodeFeat.WSS_PRIOR)
    return dict(sdf=sdf, width=width, width_d1=width_d1, width_d2=width_d2, expansion=expansion,
                wall_curv1=wall_curv1, wall_curv2=wall_curv2, downstream=downstream,
                mu_prior=mu_prior, wss_prior=wss_prior)


def _auc(score: np.ndarray, label: np.ndarray) -> float:
    pos = label > 0.5
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ties
    _, inv, counts = np.unique(score, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _logit_fit(X: np.ndarray, y: np.ndarray, steps: int = 400, lr: float = 0.05) -> np.ndarray:
    Xt = torch.tensor(X, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    pos = float(yt.sum()); neg = float(len(yt) - pos)
    pw = torch.tensor(max(neg / max(pos, 1.0), 1.0), dtype=torch.float64)
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        z = Xt @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, yt, pos_weight=pw)
        loss = loss + 1e-3 * (w * w).sum()
        loss.backward()
        opt.step()
    return np.concatenate([w.detach().numpy(), b.detach().numpy()])


def _standardize(train: np.ndarray, *mats: np.ndarray):
    mu = train.mean(axis=0, keepdims=True)
    sd = train.std(axis=0, keepdims=True)
    sd[sd < 1e-9] = 1.0
    return [(m - mu) / sd for m in (train, *mats)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Static geometry-context importance for clot prediction")
    ap.add_argument("--hops", type=int, default=3, help="wall band dilation hops (deploy band)")
    ap.add_argument("--time", default="last", help="'last' or integer frame index")
    args = ap.parse_args()

    dev = torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    anchors = sorted(p.stem for p in ANCHOR_DIR.glob("patient*.pt") if "_metadata" not in p.stem)
    if not anchors:
        print(f"[ERR] no anchors under {ANCHOR_DIR}")
        return

    per_anchor: dict[str, dict] = {}
    band_feat: dict[str, np.ndarray] = {}
    band_lab: dict[str, np.ndarray] = {}
    print(f"[i] anchors={anchors} hops={args.hops} time={args.time}")
    print(f"\n{'== (A) UNIVARIATE AUC inside wall+%dhop band (clot vs non-clot) ==' % args.hops}")
    hdr = f"{'anchor':12}{'nclot':>7}{'nband':>7}  " + "".join(f"{f:>11}" for f in ALL_FEATS)
    print(hdr)
    for a in anchors:
        d = torch.load(ANCHOR_DIR / f"{a}.pt", map_location=dev, weights_only=False)
        T = int(d.y.shape[0])
        t = T - 1 if args.time == "last" else max(0, min(int(args.time), T - 1))
        band = resolve_ceiling_mask(d, dev, bio, ceiling_hops=args.hops).reshape(-1).bool()
        gt = gt_clot_phi_at_time(d, t, phys, device=dev).reshape(-1).bool()
        feats = _features(d, dev)
        bidx = band.numpy()
        lab = gt.numpy()[bidx].astype(np.float64)
        if lab.sum() == 0 or (1 - lab).sum() == 0:
            print(f"{a:12}{int(lab.sum()):>7}{int(bidx.sum()):>7}  (degenerate label in band; skipped)")
            continue
        cols = {f: feats[f].numpy()[bidx] for f in ALL_FEATS}
        aucs = {f: _auc(cols[f], lab) for f in ALL_FEATS}
        # standardized clot/non-clot mean diff (cohen-ish)
        mdiff = {}
        for f in ALL_FEATS:
            v = cols[f]
            s = v.std() or 1.0
            mdiff[f] = float((v[lab > 0.5].mean() - v[lab < 0.5].mean()) / s)
        per_anchor[a] = dict(t=t, nclot=int(lab.sum()), nband=int(bidx.sum()), auc=aucs, mean_diff=mdiff)
        band_feat[a] = np.stack([cols[f] for f in ALL_FEATS], axis=1)
        band_lab[a] = lab
        row = f"{a:12}{int(lab.sum()):>7}{int(bidx.sum()):>7}  " + "".join(f"{aucs[f]:>11.3f}" for f in ALL_FEATS)
        print(row)

    if not band_feat:
        print("[ERR] no usable anchors")
        return

    # pooled univariate AUC
    Xall = np.concatenate([band_feat[a] for a in band_feat], axis=0)
    yall = np.concatenate([band_lab[a] for a in band_feat], axis=0)
    pooled_auc = {f: _auc(Xall[:, i], yall) for i, f in enumerate(ALL_FEATS)}
    print("-" * len(hdr))
    print(f"{'POOLED':12}{int(yall.sum()):>7}{int(len(yall)):>7}  " + "".join(f"{pooled_auc[f]:>11.3f}" for f in ALL_FEATS))
    print("[i] AUC>0.5 means higher feature -> more clot; <0.5 means lower -> more clot.")

    # (B) LOAO logistic for three feature sets
    sets = {"geom_only": GEOM_FEATS, "shear_only": SHEAR_FEATS, "geom+shear": ALL_FEATS}
    idx_of = {f: i for i, f in enumerate(ALL_FEATS)}
    print("\n== (B) LOAO logistic holdout AUC (does geometry add / transfer?) ==")
    loao = {}
    names = list(band_feat.keys())
    for sname, feats in sets.items():
        cols = [idx_of[f] for f in feats]
        per = {}
        for held in names:
            tr = [a for a in names if a != held]
            Xtr = np.concatenate([band_feat[a][:, cols] for a in tr], axis=0)
            ytr = np.concatenate([band_lab[a] for a in tr], axis=0)
            Xte = band_feat[held][:, cols]
            yte = band_lab[held]
            Xtr_s, Xte_s = _standardize(Xtr, Xte)
            wb = _logit_fit(Xtr_s, ytr)
            score = Xte_s @ wb[:-1] + wb[-1]
            per[held] = _auc(score, yte)
        loao[sname] = per
        vals = [v for v in per.values() if not np.isnan(v)]
        print(f"  {sname:11}: holdout mean AUC = {np.mean(vals):.3f}   per-anchor: " +
              " ".join(f"{a.replace('patient','p')}={per[a]:.2f}" for a in names))

    # (C) pooled standardized coefficients -> importance ranking (geom+shear)
    Xs = _standardize(Xall)[0]
    wb = _logit_fit(Xs, yall, steps=600)
    coefs = {f: float(wb[i]) for i, f in enumerate(ALL_FEATS)}
    rank = sorted(ALL_FEATS, key=lambda f: abs(coefs[f]), reverse=True)
    print("\n== (C) pooled standardized logistic coefficients (|coef| = importance) ==")
    for f in rank:
        print(f"  {f:12} coef={coefs[f]:+.3f}")

    # (D) wall-only vs band univariate AUC for the top geometry features
    print("\n== (D) wall-only vs full-band univariate AUC ==")
    wall_auc = {f: [] for f in ALL_FEATS}
    band_auc = {f: [] for f in ALL_FEATS}
    for a in names:
        d = torch.load(ANCHOR_DIR / f"{a}.pt", map_location=dev, weights_only=False)
        T = int(d.y.shape[0])
        t = T - 1 if args.time == "last" else max(0, min(int(args.time), T - 1))
        wall = d.mask_wall.reshape(-1).bool().numpy()
        band = resolve_ceiling_mask(d, dev, bio, ceiling_hops=args.hops).reshape(-1).bool().numpy()
        gt = gt_clot_phi_at_time(d, t, phys, device=dev).reshape(-1).bool().numpy().astype(np.float64)
        feats = _features(d, dev)
        for f in ALL_FEATS:
            v = feats[f].numpy()
            for mask, store in ((wall, wall_auc), (band, band_auc)):
                lab = gt[mask]
                if lab.sum() and (1 - lab).sum():
                    store[f].append(_auc(v[mask], lab))
    print(f"{'feature':12}{'wall_AUC':>10}{'band_AUC':>10}")
    for f in rank:
        wa = np.nanmean(wall_auc[f]) if wall_auc[f] else float("nan")
        ba = np.nanmean(band_auc[f]) if band_auc[f] else float("nan")
        print(f"{f:12}{wa:>10.3f}{ba:>10.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(
        hops=args.hops, time=args.time, anchors=names,
        per_anchor=per_anchor, pooled_univariate_auc=pooled_auc,
        loao_holdout_auc=loao, pooled_coefs=coefs, importance_rank=rank,
        wall_vs_band_auc={f: dict(wall=float(np.nanmean(wall_auc[f]) if wall_auc[f] else float("nan")),
                                  band=float(np.nanmean(band_auc[f]) if band_auc[f] else float("nan")))
                         for f in ALL_FEATS},
    ), indent=2))
    print(f"\n[save] {OUT}")


if __name__ == "__main__":
    main()
