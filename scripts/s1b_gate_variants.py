"""S1b: which low-shear gate source best reproduces the GT clot, deployably?

The S0->S1 diagnostics showed the on-graph WLS velocity-gradient shear is rank-
decorrelated from COMSOL spf.sr at the no-slip wall (Pearson ~0.1), and that this --
not the k_aa closure or the u_ref/d_bar calibration -- caps wall recall (~0.62 vs
0.80 with exact shear). This harness compares deployable replacements for the gate,
all scored the same way: wall-surface law (oracle Mas/ap) -> nucleate at wall ->
dilate into the lumen band -> F1 vs the canonical GT clot at the final frame.

Variants:
  baseline : WLS velocity-gradient shear (current S1)
  (a) carreau   : analytic Carreau/Poiseuille wall shear inverted from mu_prior (static, geometry)
  (b) wallfunc  : near-wall wall-function shear |u_nearwall| / d_wall (time-varying, from flow)
  (c) learned   : tiny logistic gate on deployable features, leave-one-anchor-out
  ceiling  : exact COMSOL spf.sr (patient007 only, from the calibration export)

Run: python scripts/s1b_gate_variants.py
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.utils.rheology import compute_shear_rate  # noqa: E402
import scripts.s1_kaa_closure_generalization as s1  # noqa: E402

ANCHOR_DIR = ROOT / "data" / "processed" / "graphs_biochem_anchors"
OUT = ROOT / "outputs" / "reports" / "comsol_validation" / "s1b_gate_variants.json"
DILATE_HOPS = 1
SDF_CH, MU_PRIOR_CH, WIDTH_CH = 2, 13, 15


# ---------------------------------------------------------------- shear sources
def _wls_shear(d, dev):
    return s1._shear_series(d, dev)  # sr[T,N] 1/s, dsrx


def _carreau_shear(d, dev):
    """Static analytic wall shear inverted from mu_prior_nd (geometry+Carreau)."""
    kc = PhysicsConfig(phase="kinematics", rheology="carreau")
    u_ref = float(d.u_ref.view(-1)[0]); d_bar = float(d.d_bar.view(-1)[0])
    ref_mu = float(kc.mu_viscosity_nd_scale)
    mu_inf_nd = kc.mu_inf / ref_mu; mu0_nd = kc.mu_0 / ref_mu
    lam_nd = kc.lam * (u_ref / d_bar)
    a, n = float(kc.a), float(kc.n)
    mu = d.x[:, MU_PRIOR_CH].to(dev).double()
    frac = ((mu - mu_inf_nd) / (mu0_nd - mu_inf_nd)).clamp(1e-9, 1.0)
    base = frac ** (a / (n - 1.0))                  # invert (1+(lam g)^a)^((n-1)/a)
    val = (base - 1.0).clamp_min(0.0)
    g_nd = (val ** (1.0 / a)) / lam_nd
    sr = (g_nd * (u_ref / d_bar)).float()           # [N] 1/s
    return sr


def _ring_op(d, dev):
    """Return a function that averages a node field over interior (non-wall) 1-hop neighbours."""
    ei = d.edge_index.cpu().numpy(); src, dst = ei[0], ei[1]
    wall = d.mask_wall.reshape(-1).bool().cpu().numpy()
    interior = ~wall

    def ring_mean(vals):
        v = vals.detach().cpu().numpy()
        acc = np.zeros(len(v)); cnt = np.zeros(len(v))
        for s_, t_ in ((src, dst), (dst, src)):
            m = interior[s_]
            np.add.at(acc, t_[m], v[s_[m]]); np.add.at(cnt, t_[m], 1.0)
        out = np.where(cnt > 0, acc / np.maximum(cnt, 1), v)
        return torch.from_numpy(out).float().to(dev)
    return ring_mean


def _wallfunc_shear(d, dev):
    """Time-varying near-wall wall-function shear: |u_nearwall|_phys / d_wall_phys."""
    y = d.y.to(dev); T, N, _ = y.shape
    u_ref = float(d.u_ref.view(-1)[0]); d_bar = float(d.d_bar.view(-1)[0])
    sdf_phys = (d.x[:, SDF_CH].to(dev).clamp_min(0.0) * d_bar)
    ring_mean = _ring_op(d, dev)
    sdf_ring = ring_mean(sdf_phys).clamp_min(1e-5)
    sr = torch.empty(T, N, device=dev)
    for t in range(T):
        speed_phys = torch.sqrt(y[t, :, 0] ** 2 + y[t, :, 1] ** 2) * u_ref
        speed_ring = ring_mean(speed_phys)
        sr[t] = speed_ring / sdf_ring
    return sr


def _exact_shear_p007(d, dev):
    """Exact COMSOL spf.sr mapped onto graph nodes, interpolated to all graph times."""
    import scripts._diag_shear_resolution as dr
    from scipy.spatial import cKDTree
    d_bar = float(d.d_bar.view(-1)[0])
    x_cm, y_cm, sr_exp, _, _ = dr.load_export()
    exp_nd = np.stack([x_cm * dr.CM_TO_M / d_bar, y_cm * dr.CM_TO_M / d_bar], axis=1)
    _, idx = cKDTree(exp_nd).query(d.x[:, :2].cpu().numpy(), k=1)
    sr4 = sr_exp[idx]                                   # [N,4] at 0/6000/18000/30000
    t4 = np.array([0., 6000., 18000., 30000.])
    tg = d.t.reshape(-1).cpu().numpy()
    sr = np.stack([np.interp(tg, t4, sr4[i]) for i in range(sr4.shape[0])], axis=1)  # [T,N]
    return torch.from_numpy(sr).float().to(dev)


# ---------------------------------------------------------------- law + scoring
def _species(d, cfg, dev):
    y = d.y.to(dev); T = y.shape[0]
    t_s = d.t.to(dev).reshape(-1).float()
    Minf = cfg.Minf; sc = cfg.get_species_scales(device=dev)
    AP, M, MAS, MAT, RP = s1.AP_CH, s1.M_CH, s1.MAS_CH, s1.MAT_CH, s1.RP_CH
    mat = torch.expm1(y[:, :, MAT].clamp(-10, 8)) * Minf
    mas = torch.expm1(y[:, :, MAS].clamp(-10, 8)) * Minf
    Mv = torch.expm1(y[:, :, M].clamp(-10, 8)) * Minf
    ap = torch.expm1(y[:, :, AP].clamp(-10, 8)) * float(sc[1])
    rp = torch.expm1(y[:, :, RP].clamp(-10, 8)) * float(sc[0])
    sat = (1.0 - (Mv / Minf)).clamp(0, 1)
    step2t = torch.sigmoid((t_s - float(cfg.surface_time_gate_s)) *
                           float(cfg.surface_time_gate_slope)).reshape(-1, 1)
    dmat = torch.zeros_like(mat); dt = (t_s[1:] - t_s[:-1]).reshape(-1, 1)
    dmat[1:] = (mat[1:] - mat[:-1]) / dt.clamp(min=1e-8)
    return dict(t_s=t_s, mat=mat, mas=mas, ap=ap, rp=rp, sat=sat, step2t=step2t,
                dmat=dmat, Minf=Minf)


def _integrate(sp, g_low, cfg):
    """g_low: [T,N] in [0,1]. Fit (c_dep,k_aa) own-patient, return mat_rec[final]."""
    Da, krs, kas = cfg.surface_damkohler, cfg.k_rs, cfg.k_as
    step2t = sp["step2t"]
    deposition = Da * g_low * (sp["sat"] * (krs * sp["rp"] + kas * sp["ap"])) * step2t
    autocat = g_low * (sp["mas"] / sp["Minf"]) * sp["ap"] * step2t
    s2 = step2t.expand_as(sp["mat"]); msk = (s2 > 0.5) & torch.isfinite(sp["dmat"])
    A = torch.stack([deposition[msk], autocat[msk]], 1).cpu().numpy()
    b = sp["dmat"][msk].cpu().numpy(); ok = np.all(np.isfinite(A), 1) & np.isfinite(b)
    coef, *_ = np.linalg.lstsq(A[ok], b[ok], rcond=None)
    c_dep, k_aa = float(coef[0]), float(coef[1])
    rate = torch.nan_to_num(c_dep * deposition + k_aa * autocat)
    t = sp["t_s"]; dt = (t[1:] - t[:-1]).reshape(-1, 1)
    incr = 0.5 * (rate[1:] + rate[:-1]) * dt
    mat_rec = sp["mat"][:1] + torch.cat(
        [torch.zeros(1, rate.shape[1]), torch.cumsum(incr, 0)], 0)
    return mat_rec[-1], k_aa


def _dilate(mask, ei, hops):
    m = mask.clone(); src, dst = ei[0], ei[1]
    for _ in range(hops):
        nb = torch.zeros_like(m); nb[dst] |= m[src]; nb[src] |= m[dst]; m = m | nb
    return m


def _score(d, mat_final, cfg, phys, dev, hops=DILATE_HOPS):
    crit = float(cfg.viscosity_mat_crit)
    wall = d.mask_wall.reshape(-1).bool()
    gt = gt_clot_phi_at_time(d, d.y.shape[0] - 1, phys, device=dev).reshape(-1).bool()
    pred_wall = (mat_final >= crit) & wall
    pred = _dilate(pred_wall, d.edge_index, hops)
    tp = float((pred & gt).sum()); pp = float(pred.sum()); gg = float(gt.sum())
    p = tp / max(pp, 1); r = tp / max(gg, 1)
    return dict(f1=2 * p * r / max(p + r, 1e-9), precision=p, recall=r)


def _gate(sr, cfg):
    return (sr < float(cfg.lss)).float()


# ---------------------------------------------------------------- learned gate (c)
def _learned_features(d, sp, srs, dev):
    """Static per-node deployable features for the learned gate (final-frame flow)."""
    wall = d.mask_wall.reshape(-1).bool().cpu().numpy()
    x = d.x.cpu().numpy()
    feats = np.stack([
        srs["carreau"].cpu().numpy(),
        srs["wallfunc"][-1].cpu().numpy(),
        srs["wls"][-1].cpu().numpy(),
        x[:, SDF_CH], x[:, WIDTH_CH], x[:, MU_PRIOR_CH],
    ], axis=1)
    return feats, wall


def main():
    cfg = BiochemConfig(phase="biochem"); phys = PhysicsConfig(phase="biochem")
    dev = torch.device("cpu")
    anchors = sorted(p.stem for p in ANCHOR_DIR.glob("patient*.pt") if "_metadata" not in p.stem)
    print(f"[i] anchors: {anchors}  lss={cfg.lss}  dilate={DILATE_HOPS}")

    cache = {}
    for a in anchors:
        d = torch.load(ANCHOR_DIR / f"{a}.pt", map_location=dev, weights_only=False)
        sp = _species(d, cfg, dev)
        srs = {"wls": _wls_shear(d, dev)[0], "carreau": _carreau_shear(d, dev),
               "wallfunc": _wallfunc_shear(d, dev)}
        cache[a] = (d, sp, srs)
        print(f"[load] {a}")

    # ---- learned gate (c): leave-one-anchor-out logistic regression ----
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        have_sk = True
    except Exception:
        have_sk = False
        print("[warn] sklearn missing; skipping learned variant (c)")
    learned_prob = {}
    if have_sk:
        feat_all, tgt_all, wmask_all = {}, {}, {}
        for a, (d, sp, srs) in cache.items():
            feats, wall = _learned_features(d, sp, srs, dev)
            gt = gt_clot_phi_at_time(d, d.y.shape[0] - 1, phys, device=dev).reshape(-1).cpu().numpy().astype(bool)
            feat_all[a] = feats; tgt_all[a] = (gt & wall); wmask_all[a] = wall
        for a in anchors:
            Xtr = np.concatenate([feat_all[b][wmask_all[b]] for b in anchors if b != a])
            ytr = np.concatenate([tgt_all[b][wmask_all[b]] for b in anchors if b != a])
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
            prob = np.zeros(len(wmask_all[a]))
            prob[wmask_all[a]] = clf.predict_proba(sc.transform(feat_all[a][wmask_all[a]]))[:, 1]
            learned_prob[a] = torch.from_numpy(prob).float().to(dev)

    # ---- score every variant ----
    variants = ["baseline", "carreau", "wallfunc", "learned"]
    rows = {v: [] for v in variants}
    report = {"dilate_hops": DILATE_HOPS, "per_patient": {}}
    hdr = f"{'patient':<12}" + "".join(f"{v:>11}" for v in variants) + f"{'ceiling':>11}"
    print("\n" + hdr)
    for a in anchors:
        d, sp, srs = cache[a]
        T = sp["mat"].shape[0]
        res = {}
        # baseline WLS
        mf, _ = _integrate(sp, _gate(srs["wls"], cfg), cfg)
        res["baseline"] = _score(d, mf, cfg, phys, dev)
        # (a) carreau (static -> broadcast)
        g = _gate(srs["carreau"].reshape(1, -1).expand(T, -1), cfg)
        mf, _ = _integrate(sp, g, cfg); res["carreau"] = _score(d, mf, cfg, phys, dev)
        # (b) wallfunc
        mf, _ = _integrate(sp, _gate(srs["wallfunc"], cfg), cfg)
        res["wallfunc"] = _score(d, mf, cfg, phys, dev)
        # (c) learned (soft prob gate, static)
        if a in learned_prob:
            g = learned_prob[a].reshape(1, -1).expand(T, -1)
            mf, _ = _integrate(sp, g, cfg); res["learned"] = _score(d, mf, cfg, phys, dev)
        # ceiling (p007 only)
        cf = None
        if a == "patient007":
            sr_ex = _exact_shear_p007(d, dev)
            mf, _ = _integrate(sp, _gate(sr_ex, cfg), cfg); cf = _score(d, mf, cfg, phys, dev)
        report["per_patient"][a] = {k: res.get(k) for k in variants}
        if cf: report["per_patient"][a]["ceiling"] = cf
        line = f"{a:<12}"
        for v in variants:
            rows[v].append(res[v]["f1"]) if v in res else None
            line += f"{res[v]['f1']:>11.3f}" if v in res else f"{'-':>11}"
        line += f"{cf['f1']:>11.3f}" if cf else f"{'-':>11}"
        print(line)

    print("\n" + f"{'MEAN F1':<12}" + "".join(
        f"{np.mean(rows[v]):>11.3f}" if rows[v] else f"{'-':>11}" for v in variants))
    report["mean_f1"] = {v: float(np.mean(rows[v])) if rows[v] else None for v in variants}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"\n[save] {OUT}")
    best = max((v for v in variants if rows[v]), key=lambda v: np.mean(rows[v]))
    print(f"[i] best variant: {best} (mean F1={np.mean(rows[best]):.3f})")


if __name__ == "__main__":
    main()
