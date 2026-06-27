"""Comprehensive clot-context probe: geometry, flow/stagnation, chemistry, oracle shear.

Measures which deployable and oracle features discriminate GT clot nodes inside the
wall+Khop deploy band, and whether signal GENERALIZES (LOAO logistic holdout).

Feature groups (all evaluated at the main eval time unless noted):
  geom          static topology: expansion, wall bend, sdf, width, downstream, ...
  static_prior  analytic Carreau / Poiseuille priors (no flow solve)
  kine_t0       deployable frozen-kine flow at t=0: speed, shear proxy, divergence, WLS/wallfunc sr
  gt_flow       oracle COMSOL velocity at eval time (upper bound for flow stagnation/accel)
  oracle        exact COMSOL spf.sr (+ separation derivative when cached)
  chem          species at eval time: Mat/Mas/FG/FI/AP + FG depletion since t=0
  neighbor      1-hop neighbour Mat/Mas (autocatalytic recruitment proxy)
  gate_phys     simplified deposition/autocat proxies from GT shear + species

Outputs:
  outputs/reports/comsol_validation/clot_context_comprehensive.json
  console: pooled univariate AUC by group, LOAO incremental sets, top features, commit-vs-eligible

Run:
  python scripts/_diag_clot_context_comprehensive.py --hops 3
  python scripts/_diag_clot_context_comprehensive.py --hops 3 --time last --no-kine
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.config import BiochemConfig, NodeFeat, PhysicsConfig  # noqa: E402
from src.core_physics.clot_growth_masks import resolve_ceiling_mask  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.utils import species_channels as sc  # noqa: E402
from src.utils.kinematics_inference import (  # noqa: E402
    load_kinematics_predictor,
    predict_kinematics,
    resolve_kinematics_checkpoint,
)
from src.utils.rheology import compute_shear_rate  # noqa: E402

import scripts.s2_kine_flow_test as kft  # noqa: E402
import scripts.spfsr_lib as spfsr  # noqa: E402

ANCHOR_DIR = ROOT / "data" / "processed" / "graphs_biochem_anchors"
OUT = ROOT / "outputs" / "reports" / "comsol_validation" / "clot_context_comprehensive.json"


@dataclass(frozen=True)
class FeatSpec:
    name: str
    group: str


def _slice0(x: torch.Tensor, sl: slice) -> torch.Tensor:
    return x[:, sl].reshape(-1).to(torch.float64)


def _sym_edges(edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


def _bfs_norm_dist(n: int, row: torch.Tensor, col: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    inf = n + 5
    dist = torch.full((n,), inf, dtype=torch.long)
    dist[src] = 0
    frontier = src.clone()
    d = 0
    while bool(frontier.any()) and d < n:
        d += 1
        nxt = torch.zeros(n, dtype=torch.bool)
        nxt[col[frontier[row]]] = True
        upd = nxt & (dist > d)
        if not bool(upd.any()):
            break
        dist[upd] = d
        frontier = upd
    finite = dist[dist < inf]
    dmax = float(finite.max().item()) if finite.numel() else 1.0
    return dist.clamp(max=int(dmax)).to(torch.float64) / max(dmax, 1.0)


def _flow_scalars(data, u: torch.Tensor, v: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    """Speed, neighbour shear proxy, graph divergence from explicit (u,v)."""
    u = u.reshape(-1).to(device=device, dtype=torch.float64)
    v = v.reshape(-1).to(device=device, dtype=torch.float64)
    speed = torch.sqrt(u * u + v * v)
    pos = data.x[:, :2].to(device=device, dtype=torch.float64)
    row, col = data.edge_index.to(device=device)
    diff = pos[row] - pos[col]
    dist = diff.norm(dim=1).clamp(min=1e-6)
    grad = (speed[row] - speed[col]).abs() / dist
    div_edge = ((u[row] - u[col]) * diff[:, 0] + (v[row] - v[col]) * diff[:, 1]) / (dist * dist)
    n = int(data.num_nodes)
    acc_g = torch.zeros(n, device=device, dtype=torch.float64)
    acc_d = torch.zeros(n, device=device, dtype=torch.float64)
    deg = torch.zeros(n, device=device, dtype=torch.float64)
    acc_g.index_add_(0, row, grad)
    acc_d.index_add_(0, row, div_edge)
    deg.index_add_(0, row, torch.ones_like(grad))
    shear_proxy = acc_g / deg.clamp(min=1.0)
    divergence = acc_d / deg.clamp(min=1.0)
    return dict(
        log_speed=torch.log1p(speed.clamp(min=0)),
        shear_proxy=torch.log1p(shear_proxy.clamp(min=0)),
        divergence=torch.tanh(divergence),
        speed=speed,
    )


def _wls_sr(data, u: torch.Tensor, v: torch.Tensor, device: torch.device) -> torch.Tensor:
    return kft.wls_shear_uv(data, u.to(device), v.to(device), device).to(torch.float64)


def _wallfunc_sr(data, u: torch.Tensor, v: torch.Tensor, device: torch.device) -> torch.Tensor:
    return kft.wallfunc_shear_uv(data, u.to(device), v.to(device), device).to(torch.float64)


def _geom_feats(d, dev) -> dict[str, torch.Tensor]:
    x = d.x.to(dev)
    n = int(d.num_nodes)
    row, col = _sym_edges(d.edge_index.to(dev))
    sdf = _slice0(x, NodeFeat.SDF)
    width = _slice0(x, NodeFeat.WIDTH_ND)
    width_d1 = _slice0(x, NodeFeat.WIDTH_D1)
    width_d2 = _slice0(x, NodeFeat.WIDTH_D2)
    expansion = _nbr_mean(width, row, col, n) - width
    expansion_2 = _nbr_mean(expansion, row, col, n)
    wn = x[:, NodeFeat.WALL_NORMAL].to(torch.float64)
    dot = (wn[row] * wn[col]).sum(dim=1)
    cacc = torch.zeros(n, dtype=torch.float64)
    cacc.index_add_(0, row, (1.0 - dot))
    deg = torch.zeros(n, dtype=torch.float64)
    deg.index_add_(0, row, torch.ones(row.numel(), dtype=torch.float64))
    deg = deg.clamp(min=1.0)
    wall_curv1 = cacc / deg
    wall_curv2 = _nbr_mean(wall_curv1, row, col, n)
    src = None
    for attr in ("mask_inlet", "inlet_mask"):
        m = getattr(d, attr, None)
        if m is not None:
            src = m.reshape(-1).bool().to(dev)
            break
    downstream = _bfs_norm_dist(n, row, col, src) if src is not None and bool(src.any()) else torch.zeros(n, dtype=torch.float64)
    shear_pot = _slice0(x, NodeFeat.SHEAR_POT)
    return dict(
        sdf=sdf, width=width, width_d1=width_d1, width_d2=width_d2,
        expansion=expansion, expansion_2hop=expansion_2,
        wall_curv1=wall_curv1, wall_curv2=wall_curv2,
        downstream=downstream, shear_potential=shear_pot,
    )


def _static_prior_feats(d, dev) -> dict[str, torch.Tensor]:
    x = d.x.to(dev)
    uv = x[:, NodeFeat.UV_PRIOR].to(torch.float64)
    speed_pr = torch.sqrt((uv * uv).sum(dim=1))
    width = _slice0(x, NodeFeat.WIDTH_ND).clamp(min=1e-6)
    return dict(
        mu_prior=_slice0(x, NodeFeat.MU_PRIOR),
        wss_prior=_slice0(x, NodeFeat.WSS_PRIOR),
        uv_prior_speed=speed_pr,
        speed_prior_over_width=speed_pr / width,
    )


def _species_log(y_t: torch.Tensor, name: str) -> torch.Tensor:
    idx = sc.y_index(name)
    return y_t[:, idx].clamp(-10, 8).to(torch.float64)


def _feat_np(feats: dict[str, torch.Tensor], name: str, bidx: np.ndarray) -> np.ndarray:
    """Safe band-masked numpy slice (kine outputs may carry grad)."""
    if name not in feats:
        return np.full(int(bidx.sum()), np.nan, dtype=np.float64)
    return feats[name].detach().cpu().numpy()[bidx]


@torch.no_grad()
def _build_features(
    d,
    dev,
    t: int,
    bio: BiochemConfig,
    *,
    kine_model=None,
    stem: str = "",
) -> tuple[dict[str, torch.Tensor], list[FeatSpec]]:
    specs: list[FeatSpec] = []
    out: dict[str, torch.Tensor] = {}

    def put(group: str, feats: dict[str, torch.Tensor]) -> None:
        for k, v in feats.items():
            key = k if k.startswith(group + "_") or "_" in k else f"{group}_{k}" if group else k
            # keep explicit prefixes from caller
            name = k
            out[name] = v
            specs.append(FeatSpec(name=name, group=group))

    put("geom", _geom_feats(d, dev))
    put("static", _static_prior_feats(d, dev))

    # kine t0 deployable flow
    if kine_model is not None:
        kine_model.eval()
        uv = predict_kinematics(kine_model, d.clone()).to(dev)
        u0, v0 = uv[:, 0], uv[:, 1]
        fs = _flow_scalars(d, u0, v0, dev)
        put("kine", {
            "kine_log_speed": fs["log_speed"],
            "kine_shear_proxy": fs["shear_proxy"],
            "kine_divergence": fs["divergence"],
            "kine_wls_sr": torch.log1p(_wls_sr(d, u0, v0, dev).clamp(min=0)),
            "kine_wallfunc_sr": torch.log1p(_wallfunc_sr(d, u0, v0, dev).clamp(min=0)),
            "kine_low_shear": -torch.log1p(_wls_sr(d, u0, v0, dev).clamp(min=0)),  # higher = more stagnant
        })

    # GT COMSOL flow at eval time (oracle stagnation/accel)
    y_t = d.y[int(t)].to(dev)
    u_gt, v_gt = y_t[:, 0], y_t[:, 1]
    fs_gt = _flow_scalars(d, u_gt, v_gt, dev)
    put("gt", {
        "gt_log_speed": fs_gt["log_speed"],
        "gt_shear_proxy": fs_gt["shear_proxy"],
        "gt_divergence": fs_gt["divergence"],
        "gt_wls_sr": torch.log1p(_wls_sr(d, u_gt, v_gt, dev).clamp(min=0)),
        "gt_low_shear": -torch.log1p(_wls_sr(d, u_gt, v_gt, dev).clamp(min=0)),
    })

    # exact COMSOL spf.sr cache
    if stem and spfsr.has_cache(stem):
        ali = spfsr.aligned(d, dev, stem)
        ti = min(int(t), int(ali["sr"].shape[0]) - 1)
        sr = ali["sr"][ti].to(torch.float64)
        put("oracle", {
            "spf_sr": sr,
            "spf_low_shear": -torch.log1p(sr.clamp(min=0)),
            "spf_gate_on": (sr < float(bio.lss)).to(torch.float64),
        })
        if ali.get("has_deriv"):
            dsrx = ali["dsrx"][ti].to(torch.float64)
            dsry = ali["dsry"][ti].to(torch.float64)
            put("oracle", {
                "spf_dsrx": dsrx,
                "spf_dsry": dsry,
                "spf_sep_grad": torch.sqrt(dsrx * dsrx + dsry * dsry),
            })

    # chemistry at t0 and eval
    y0 = d.y[0].to(dev)
    mat_t = _species_log(y_t, "Mat")
    mat_0 = _species_log(y0, "Mat")
    fg_t = _species_log(y_t, "FG")
    fg_0 = _species_log(y0, "FG")
    put("chem", {
        "mat_log_nd": mat_t,
        "mat_growth_log": mat_t - mat_0,
        "mas_log_nd": _species_log(y_t, "Mas"),
        "m_log_nd": _species_log(y_t, "M"),
        "fg_log_nd": fg_t,
        "fg_depletion": fg_0 - fg_t,
        "fi_log_nd": _species_log(y_t, "FI"),
        "ap_log_nd": _species_log(y_t, "AP"),
        "rp_log_nd": _species_log(y_t, "RP"),
    })

    # neighbour autocatalysis proxies (GT Mat/Mas field -- placement diagnostic)
    n = int(d.num_nodes)
    row, col = _sym_edges(d.edge_index.to(dev))
    mat_si = torch.expm1(mat_t) * float(bio.Minf)
    mas_si = torch.expm1(_species_log(y_t, "Mas")) * float(bio.Minf)
    put("neighbor", {
        "nbr_mat_log": torch.log1p(_nbr_mean(mat_si, row, col, n).clamp(min=0)),
        "nbr_mas_log": torch.log1p(_nbr_mean(mas_si, row, col, n).clamp(min=0)),
        "nbr_mat_gt_thresh": (_nbr_mean(mat_si, row, col, n) >= float(bio.viscosity_mat_crit)).to(torch.float64),
    })

    # simplified physics gate proxies (GT flow + species -- oracle deposition structure)
    lss = float(bio.lss)
    sgt = float(bio.sgt)
    sr_gt = _wls_sr(d, u_gt, v_gt, dev)
    dudx = torch.sparse.mm(d.G_x.to(dev), u_gt.reshape(-1, 1).float()).squeeze(1).double()
    dsrx = dudx / max(float(d.d_bar.view(-1)[0]), 1e-8)
    g_low = (sr_gt < lss).to(torch.float64)
    g_sep = (dsrx < sgt).to(torch.float64)
    gate_mix = g_sep * abs(dsrx) + g_low
    put("gate", {
        "gate_low_shear": g_low,
        "gate_sep": g_sep,
        "gate_mix": gate_mix,
        "depo_proxy": gate_mix * torch.expm1(_species_log(y_t, "AP").clamp(-10, 8)),
        "autocat_proxy": gate_mix * (mas_si / float(bio.Minf)) * torch.expm1(_species_log(y_t, "AP").clamp(-10, 8)),
    })

    return {k: v.detach() for k, v in out.items()}, specs


def _auc(score: np.ndarray, label: np.ndarray) -> float:
    mask = np.isfinite(score)
    if mask.sum() < 10:
        return float("nan")
    score = score[mask]
    label = label[mask]
    pos = label > 0.5
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    _, inv, counts = np.unique(score, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _logit_fit(X: np.ndarray, y: np.ndarray, steps: int = 500, lr: float = 0.05) -> np.ndarray:
    col_ok = np.isfinite(X).all(axis=0)
    if not col_ok.any():
        return np.zeros(X.shape[1] + 1)
    X = np.where(np.isfinite(X), X, 0.0)
    Xt = torch.tensor(X[:, col_ok], dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    w = torch.zeros(int(col_ok.sum()), dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    pos = float(yt.sum())
    pw = torch.tensor(max((len(yt) - pos) / max(pos, 1.0), 1.0), dtype=torch.float64)
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        z = Xt @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, yt, pos_weight=pw)
        (loss + 1e-3 * (w * w).sum()).backward()
        opt.step()
    w_full = np.zeros(X.shape[1], dtype=np.float64)
    w_full[col_ok] = w.detach().numpy()
    return np.concatenate([w_full, b.detach().numpy()])


def _standardize(train: np.ndarray, *mats: np.ndarray):
    mu = train.mean(axis=0, keepdims=True)
    sd = train.std(axis=0, keepdims=True)
    sd[sd < 1e-9] = 1.0
    return [(m - mu) / sd for m in (train, *mats)]


def _feat_names_by_group(specs: list[FeatSpec]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for s in specs:
        groups.setdefault(s.group, []).append(s.name)
    return groups


def _loao_auc(band_feat: dict[str, np.ndarray], band_lab: dict[str, np.ndarray], feat_names: list[str], idx_of: dict[str, int]) -> tuple[float, dict[str, float]]:
    cols = [idx_of[f] for f in feat_names if f in idx_of]
    if not cols:
        return float("nan"), {}
    names = list(band_feat.keys())
    per = {}
    for held in names:
        tr = [a for a in names if a != held]
        Xtr = np.concatenate([band_feat[a][:, cols] for a in tr], axis=0)
        ytr = np.concatenate([band_lab[a] for a in tr], axis=0)
        Xte = band_feat[held][:, cols]
        yte = band_lab[held]
        Xtr_s, Xte_s = _standardize(Xtr, Xte)
        wb = _logit_fit(Xtr_s, ytr)
        per[held] = _auc(Xte_s @ wb[:-1] + wb[-1], yte)
    vals = [v for v in per.values() if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan"), per


def main() -> None:
    ap = argparse.ArgumentParser(description="Comprehensive clot-context feature probe")
    ap.add_argument("--hops", type=int, default=3)
    ap.add_argument("--time", default="last")
    ap.add_argument("--no-kine", action="store_true", help="skip kine-model flow block")
    ap.add_argument("--logit-steps", type=int, default=500)
    args = ap.parse_args()

    dev = torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    anchors = sorted(p.stem for p in ANCHOR_DIR.glob("patient*.pt") if "_metadata" not in p.stem)
    if not anchors:
        print(f"[ERR] no anchors under {ANCHOR_DIR}")
        return

    kine = None
    if not args.no_kine:
        try:
            kine = load_kinematics_predictor(str(resolve_kinematics_checkpoint()), device=dev)
            kine.eval()
            print("[OK] kine model loaded")
        except Exception as exc:
            print(f"[WARN] kine model unavailable ({exc}); kine_* features skipped")

    # discover full feature list (union across anchors -- spfsr/oracle may be per-patient)
    feat_names_set: set[str] = set()
    group_of: dict[str, str] = {}
    for a in anchors:
        d_probe = torch.load(ANCHOR_DIR / f"{a}.pt", map_location=dev, weights_only=False)
        t_probe = int(d_probe.y.shape[0]) - 1 if args.time == "last" else max(0, min(int(args.time), int(d_probe.y.shape[0]) - 1))
        _, specs_probe = _build_features(d_probe, dev, t_probe, bio, kine_model=kine, stem=a)
        for s in specs_probe:
            feat_names_set.add(s.name)
            group_of[s.name] = s.group
    feat_names = sorted(feat_names_set)
    groups: dict[str, list[str]] = {}
    for f in feat_names:
        groups.setdefault(group_of[f], []).append(f)
    idx_of = {f: i for i, f in enumerate(feat_names)}

    band_feat: dict[str, np.ndarray] = {}
    band_lab: dict[str, np.ndarray] = {}
    eligible_lab: dict[str, np.ndarray] = {}
    per_anchor: dict[str, dict] = {}

    print(f"[i] anchors={len(anchors)} hops={args.hops} time={args.time} n_features={len(feat_names)}")
    print(f"[i] groups: " + ", ".join(f"{g}={len(v)}" for g, v in groups.items()))

    for a in anchors:
        d = torch.load(ANCHOR_DIR / f"{a}.pt", map_location=dev, weights_only=False)
        T = int(d.y.shape[0])
        t = T - 1 if args.time == "last" else max(0, min(int(args.time), T - 1))
        band = resolve_ceiling_mask(d, dev, bio, ceiling_hops=args.hops).reshape(-1).bool()
        gt = gt_clot_phi_at_time(d, t, phys, device=dev).reshape(-1).bool()
        feats, _ = _build_features(d, dev, t, bio, kine_model=kine, stem=a)
        bidx = band.numpy()
        lab = gt.numpy()[bidx].astype(np.float64)
        if lab.sum() == 0 or (1 - lab).sum() == 0:
            print(f"[WARN] {a}: degenerate clot label in band; skipped")
            continue
        cols = np.stack([_feat_np(feats, f, bidx) for f in feat_names], axis=1)
        band_feat[a] = cols
        band_lab[a] = lab
        aucs = {f: _auc(cols[:, i], lab) for i, f in enumerate(feat_names)}
        per_anchor[a] = dict(t=t, nclot=int(lab.sum()), nband=int(bidx.sum()), auc=aucs)

        # eligible = kine low-shear OR spf gate (deployable-ish eligibility)
        elig = np.zeros(int(band.sum()), dtype=bool)
        if "kine_low_shear" in feats:
            kls = _feat_np(feats, "kine_low_shear", bidx)
            thr = np.quantile(kls, 0.70)
            elig |= kls >= thr
        if "spf_gate_on" in feats:
            elig |= _feat_np(feats, "spf_gate_on", bidx) > 0.5
        if not elig.any() and "gate_low_shear" in feats:
            elig = _feat_np(feats, "gate_low_shear", bidx) > 0.5
        eligible_lab[a] = elig

    names = list(band_feat.keys())
    if not names:
        print("[ERR] no usable anchors")
        return

    Xall = np.concatenate([band_feat[a] for a in names], axis=0)
    yall = np.concatenate([band_lab[a] for a in names], axis=0)
    pooled_auc = {f: _auc(Xall[:, i], yall) for i, f in enumerate(feat_names)}

    # (A) top features by pooled AUC distance from 0.5
    ranked = sorted(feat_names, key=lambda f: abs(pooled_auc[f] - 0.5), reverse=True)
    print("\n== (A) TOP 25 features by pooled |AUC-0.5| (band, clot vs non-clot) ==")
    print(f"{'feature':28}{'group':10}{'AUC':>8}{'|d|':>8}")
    feat_group = group_of
    for f in ranked[:25]:
        auc = pooled_auc[f]
        print(f"{f:28}{feat_group.get(f,'?'):10}{auc:8.3f}{abs(auc-0.5):8.3f}")

    # (B) group mean |AUC-0.5|
    print("\n== (B) group signal strength (mean |AUC-0.5| over features) ==")
    group_strength = {}
    for g, flist in groups.items():
        vals = [abs(pooled_auc[f] - 0.5) for f in flist if f in pooled_auc and not np.isnan(pooled_auc[f])]
        group_strength[g] = float(np.mean(vals)) if vals else float("nan")
        print(f"  {g:10} mean_|d|= {group_strength[g]:.3f}  (n={len(flist)})")

    # (C) incremental LOAO sets.
    # CIRCULAR groups read GT species/flow at the EVAL time -> trivially reconstruct the label
    # (clot == Mat>crit, 6.11). They are an oracle sanity check, NOT deployable. DEPLOYABLE groups
    # are clot-blind / t0 only. ORACLE-flow (gt/oracle) uses eval-time COMSOL flow (partly circular,
    # unrecoverable at deploy). spf 'oracle' is p007-only -> excluded from cohort LOAO with a note.
    DEPLOYABLE_GROUPS = ("geom", "static", "kine")
    CIRCULAR_GROUPS = ("chem", "neighbor", "gate")
    ORACLE_FLOW_GROUPS = ("gt", "oracle")
    spf_full_cohort = all(spfsr.has_cache(a) for a in names)
    g = lambda *gs: [f for gg in gs for f in groups.get(gg, [])]
    incremental_sets = {
        "geom": g("geom"),
        "geom+static": g("geom", "static"),
        "deployable_t0 [geom+static+kine]": g("geom", "static", "kine"),
        "+gt_flow (ORACLE eval-time)": g("geom", "static", "kine", "gt"),
        "circular_check [+chem+neighbor]": g("geom", "static", "kine", "chem", "neighbor"),
    }
    if spf_full_cohort:
        incremental_sets["+oracle_spf"] = g("geom", "static", "kine", "oracle")
        incremental_sets["all"] = feat_names
    print("\n== (C) LOAO holdout AUC (incremental sets; deployable vs oracle/circular) ==")
    if not spf_full_cohort:
        print("  [note] spf.sr cached for subset only -> oracle/all LOAO skipped (p007 univariate only)")
    loao = {}
    for sname, flist in incremental_sets.items():
        flist = [f for f in flist if f in idx_of]
        mean_auc, per = _loao_auc(band_feat, band_lab, flist, idx_of)
        loao[sname] = dict(mean=mean_auc, per_anchor=per)
        tag = "  <- DEPLOYABLE" if sname.startswith("deployable_t0") else ""
        print(f"  {sname:34} holdout AUC = {mean_auc:.3f}{tag}")

    # (D) pooled logistic importance over DEPLOYABLE features only (no circular leakage)
    dep_feats = [f for f in feat_names if feat_group.get(f) in DEPLOYABLE_GROUPS]
    dep_cols = [idx_of[f] for f in dep_feats]
    Xs = _standardize(Xall[:, dep_cols])[0]
    wb = _logit_fit(Xs, yall, steps=args.logit_steps)
    dep_coefs = {dep_feats[i]: float(wb[i]) for i in range(len(dep_feats))}
    dep_rank = sorted(dep_feats, key=lambda f: abs(dep_coefs[f]), reverse=True)
    print("\n== (D) deployable pooled logistic |coef| top 15 ==")
    for f in dep_rank[:15]:
        print(f"  {f:28} coef={dep_coefs[f]:+.3f}")

    # (E) commit vs eligible (within band): clot vs eligible-non-clot only
    print("\n== (E) commit vs eligible-non-clot (bifurcation probe, univariate AUC) ==")
    commit_auc = {}
    for f in ranked[:15]:
        scores = []
        for a in names:
            elig = eligible_lab.get(a)
            if elig is None:
                continue
            lab_full = band_lab[a]
            # eligible non-clot OR clot
            mask = (lab_full > 0.5) | ((lab_full < 0.5) & elig)
            if mask.sum() < 10:
                continue
            y_bin = lab_full[mask]
            if y_bin.sum() == 0 or (1 - y_bin).sum() == 0:
                continue
            scores.append(_auc(band_feat[a][mask, idx_of[f]], y_bin))
        commit_auc[f] = float(np.nanmean(scores)) if scores else float("nan")
    for f in ranked[:15]:
        print(f"  {f:28} commit-AUC={commit_auc.get(f, float('nan')):.3f}  pooled-AUC={pooled_auc[f]:.3f}")

    # (F) wall vs band for top geom/flow
    print("\n== (F) wall-only vs band univariate AUC (top 10) ==")
    wall_band = {}
    for f in ranked[:10]:
        wa, ba = [], []
        for a in names:
            d = torch.load(ANCHOR_DIR / f"{a}.pt", map_location=dev, weights_only=False)
            T = int(d.y.shape[0])
            t = T - 1 if args.time == "last" else max(0, min(int(args.time), T - 1))
            wall = d.mask_wall.reshape(-1).bool().numpy()
            band = resolve_ceiling_mask(d, dev, bio, ceiling_hops=args.hops).reshape(-1).bool().numpy()
            gt_np = gt_clot_phi_at_time(d, t, phys, device=dev).reshape(-1).bool().numpy().astype(np.float64)
            feats, _ = _build_features(d, dev, t, bio, kine_model=kine, stem=a)
            v = feats[f].detach().cpu().numpy()
            for mask, store in ((wall, wa), (band, ba)):
                lab = gt_np[mask]
                if lab.sum() and (1 - lab).sum():
                    store.append(_auc(v[mask], lab))
        wall_band[f] = dict(wall=float(np.nanmean(wa)) if wa else float("nan"), band=float(np.nanmean(ba)) if ba else float("nan"))
        print(f"  {f:28} wall={wall_band[f]['wall']:.3f}  band={wall_band[f]['band']:.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        hops=args.hops,
        time=args.time,
        anchors=names,
        feature_groups=groups,
        feature_names=feat_names,
        per_anchor=per_anchor,
        pooled_univariate_auc=pooled_auc,
        top_features=ranked[:30],
        group_strength=group_strength,
        loao_incremental=loao,
        deployable_coef_rank=dep_rank,
        deployable_coefs=dep_coefs,
        commit_vs_eligible_auc=commit_auc,
        wall_vs_band=wall_band,
    )
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"\n[save] {OUT}")


if __name__ == "__main__":
    main()
