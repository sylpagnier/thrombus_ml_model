"""Diagnostic: decompose p007 wall-recall loss into calibration vs operator-resolution.

Maps COMSOL-exact spf.sr from the domain calibration export onto the graph nodes
(nearest neighbour in nondim coords), then compares to the on-graph WLS shear and
re-scores the low-shear gate / S1 Mat law on a common 4-time grid so the ONLY
difference is the shear operator.

Run: python scripts/_diag_shear_resolution.py
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
import scripts.s1_kaa_closure_generalization as s1  # noqa: E402

DOMAIN = ROOT / "data" / "reference" / "comsol_calibration" / "patient007_calibration_domain.txt"
GRAPH = ROOT / "data" / "processed" / "graphs_biochem_anchors" / "patient007.pt"
CM_TO_M = 0.01
BLOCK = 28          # cols per timestep block
NTB = 4             # export times: 0, 6000, 18000, 30000
GTIDX = [0, 40, 120, 200]   # graph time indices matching export times
SR_OFF, DSRX_OFF, LSS_OFF = 5, 6, 17   # offsets within a block


def load_export():
    rows = []
    with open(DOMAIN, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("%") or not line.strip():
                continue
            rows.append([float(v) for v in line.split()])
    a = np.asarray(rows, dtype=np.float64)
    x_cm, y_cm = a[:, 0], a[:, 1]
    sr = np.stack([a[:, 2 + BLOCK * b + SR_OFF] for b in range(NTB)], axis=1)     # [Nexp,4]
    dsrx = np.stack([a[:, 2 + BLOCK * b + DSRX_OFF] for b in range(NTB)], axis=1)
    gate = np.stack([a[:, 2 + BLOCK * b + LSS_OFF] for b in range(NTB)], axis=1)
    return x_cm, y_cm, sr, dsrx, gate


def main():
    cfg = BiochemConfig(phase="biochem"); phys = PhysicsConfig(phase="biochem")
    dev = torch.device("cpu")
    d = torch.load(GRAPH, map_location=dev, weights_only=False)
    d_bar = float(d.d_bar.view(-1)[0])
    lss = float(cfg.lss); crit = float(cfg.viscosity_mat_crit)
    print(f"[i] lss={lss} 1/s  mat_crit={crit:.3g}  d_bar={d_bar:.5g} m")

    # --- map exact spf.sr onto graph nodes ---
    x_cm, y_cm, sr_exp, dsrx_exp, gate_exp = load_export()
    exp_nd = np.stack([x_cm * CM_TO_M / d_bar, y_cm * CM_TO_M / d_bar], axis=1)
    g_nd = d.x[:, :2].cpu().numpy()
    tree = cKDTree(exp_nd)
    dist, idx = tree.query(g_nd, k=1)
    typical = float(np.median(np.sqrt(np.bincount(idx, minlength=len(exp_nd)).clip(1))))  # noqa
    print(f"[i] NN map dist (nd): median={np.median(dist):.4f} p95={np.percentile(dist,95):.4f} max={dist.max():.4f}")
    sr_exact = torch.from_numpy(sr_exp[idx]).float()        # [N,4]  1/s
    # exact d(spf.sr,x): export coords in cm -> convert to 1/(s*m)
    dsrx_exact = torch.from_numpy(dsrx_exp[idx]).float() / CM_TO_M

    # --- graph WLS shear at the 4 matching times ---
    sr_wls_full, dsrx_wls_full = s1._shear_series(d, dev)    # [201,N]
    sr_wls = sr_wls_full[GTIDX].T.contiguous()              # [N,4]
    dsrx_wls = dsrx_wls_full[GTIDX].T.contiguous()

    wall = d.mask_wall.reshape(-1).bool().cpu().numpy()
    # ---------- calibration vs resolution on the shear field (wall, active) ----------
    se = sr_exact.numpy(); sw = sr_wls.numpy()
    print("\n=== shear operator quality on WALL nodes (per export time) ===")
    print(f"{'t':>7}{'pearson':>9}{'spearman':>10}{'slope a':>9}{'med wls/ex':>12}")
    from scipy.stats import pearsonr, spearmanr
    cal_a = []
    for b, ts in enumerate([0, 6000, 18000, 30000]):
        m = wall & np.isfinite(se[:, b]) & np.isfinite(sw[:, b]) & (se[:, b] > 1e-3)
        pr = pearsonr(np.log(se[m, b] + 1e-3), np.log(sw[m, b] + 1e-3))[0]
        spm = spearmanr(se[m, b], sw[m, b]).correlation
        a = float(np.median(sw[m, b] / (se[m, b] + 1e-9)))
        cal_a.append(a)
        print(f"{ts:>7}{pr:>9.3f}{spm:>10.3f}{a:>9.3f}{a:>12.3f}")
    a_glob = float(np.median(np.concatenate([sw[wall, b] / (se[wall, b] + 1e-9) for b in range(NTB)])))
    print(f"[i] global calibration factor a (median wls/exact, wall) = {a_glob:.3f}")

    # ---------- gate flip accounting at final time (wall) ----------
    b = NTB - 1
    ge = se[:, b] < lss
    gw = sw[:, b] < lss
    gwc = (sw[:, b] / a_glob) < lss     # recalibrated WLS
    def frac(x): return float(x[wall].mean())
    print("\n=== low-shear gate at final time (wall) ===")
    print(f" exact gate active   : {frac(ge):.3f}")
    print(f" WLS gate active     : {frac(gw):.3f}   agree_with_exact={float((gw[wall]==ge[wall]).mean()):.3f}")
    print(f" WLS-recal active    : {frac(gwc):.3f}   agree_with_exact={float((gwc[wall]==ge[wall]).mean()):.3f}")

    # ---------- ceiling test: S1 Mat law on common 4-time grid, swap shear ----------
    print("\n=== S1 Mat-law wall/full F1 on common 4-time grid (only shear differs) ===")
    res = run_law(d, cfg, phys, dev, sr_wls, dsrx_wls, sr_wls, label="(A) WLS shear")
    res_c = run_law(d, cfg, phys, dev, sr_wls / a_glob, dsrx_wls / a_glob, sr_wls / a_glob,
                    label="(B) WLS recalibrated (sr/a)")
    res_e = run_law(d, cfg, phys, dev, sr_exact, dsrx_exact, sr_exact, label="(C) exact spf.sr")
    # (D) deployable mu_prior gate: high mu_prior == low analytic Carreau shear.
    mu_pr = d.x[:, 13].cpu().numpy()
    thr = float(np.quantile(mu_pr[wall], 1.0 - float((sr_exact[:, NTB - 1].numpy() < lss)[wall].mean())))
    gate_mu = torch.from_numpy((mu_pr > thr).astype(np.float32)).reshape(1, -1).expand(NTB, -1).contiguous()
    res_m = run_law(d, cfg, phys, dev, sr_wls * 0 + 1e9, dsrx_wls, sr_wls,
                    label="(D) mu_prior gate (deployable)", gate_low=gate_mu)
    print("\n--- decomposition of WALL recall ---")
    print(f" baseline WLS            : {res['wall_rec']:.3f}")
    print(f" + calibration (sr/a)    : {res_c['wall_rec']:.3f}   (+{res_c['wall_rec']-res['wall_rec']:.3f})")
    print(f" + exact operator        : {res_e['wall_rec']:.3f}   (+{res_e['wall_rec']-res_c['wall_rec']:.3f})")
    print(f" residual (non-shear)    : {1-res_e['wall_rec']:.3f}")


def run_law(d, cfg, phys, dev, sr, dsrx, sr_for_auto, label, gate_low=None):
    """Rebuild S1 terms on the 4-time grid with provided shear, fit kaa, score.

    gate_low: optional [N,4] float low-shear gate to override (srT<lss).
    """
    y = d.y.to(dev)[GTIDX]            # [4,N,*]
    t_s = torch.tensor([0., 6000., 18000., 30000.], device=dev)
    Minf = cfg.Minf; sc = cfg.get_species_scales(device=dev)
    AP, M, MAS, MAT, RP = s1.AP_CH, s1.M_CH, s1.MAS_CH, s1.MAT_CH, s1.RP_CH
    mat = torch.expm1(y[:, :, MAT].clamp(-10, 8)) * Minf
    mas = torch.expm1(y[:, :, MAS].clamp(-10, 8)) * Minf
    Mv = torch.expm1(y[:, :, M].clamp(-10, 8)) * Minf
    ap = torch.expm1(y[:, :, AP].clamp(-10, 8)) * float(sc[1])
    rp = torch.expm1(y[:, :, RP].clamp(-10, 8)) * float(sc[0])
    sat = (1.0 - (Mv / Minf)).clamp(0, 1)
    srT = sr.T.contiguous(); dsrxT = dsrx.T.contiguous(); srA = sr_for_auto.T.contiguous()
    step2t = torch.sigmoid((t_s - float(cfg.surface_time_gate_s)) * float(cfg.surface_time_gate_slope)).reshape(-1, 1)
    Da = cfg.surface_damkohler; krs, kas = cfg.k_rs, cfg.k_as
    L, gm, sgt = cfg.L_char, cfg.gamma_m, cfg.sgt
    g_low = (srT < cfg.lss).float() if gate_low is None else gate_low
    g_sep = (dsrxT < sgt).float()
    gate = g_sep * (L / gm) * dsrxT.abs() + g_low
    deposition = Da * gate * (sat * (krs * rp + kas * ap)) * step2t
    autocat = gate * (mas / Minf) * ap * step2t
    dmat = torch.zeros_like(mat); dt = (t_s[1:] - t_s[:-1]).reshape(-1, 1)
    dmat[1:] = (mat[1:] - mat[:-1]) / dt.clamp(min=1e-8)
    s2 = step2t.expand_as(mat); msk = (s2 > 0.5) & torch.isfinite(dmat)
    A = torch.stack([deposition[msk], autocat[msk]], 1).cpu().numpy()
    bb = dmat[msk].cpu().numpy(); ok = np.all(np.isfinite(A), 1) & np.isfinite(bb)
    coef, *_ = np.linalg.lstsq(A[ok], bb[ok], rcond=None)
    c_dep, k_aa = float(coef[0]), float(coef[1])
    incr = 0.5 * (deposition[1:] * c_dep + k_aa * autocat[1:] + deposition[:-1] * c_dep + k_aa * autocat[:-1]) * dt
    mat_rec = mat[:1] + torch.cat([torch.zeros(1, mat.shape[1]), torch.cumsum(torch.nan_to_num(incr), 0)], 0)
    pred = (mat_rec[-1] >= float(cfg.viscosity_mat_crit))
    gt = gt_clot_phi_at_time(d, d.y.shape[0] - 1, phys, device=dev).reshape(-1).bool()
    wall = d.mask_wall.reshape(-1).bool()
    def f1(p, g):
        tp = float((p & g).sum()); pp = float(p.sum()); gg = float(g.sum())
        pr = tp / max(pp, 1); rc = tp / max(gg, 1); return 2 * pr * rc / max(pr + rc, 1e-9), pr, rc
    full = f1(pred, gt); wf = f1(pred & wall, gt & wall)
    print(f" {label:<28} full F1={full[0]:.3f}(p{full[1]:.2f}/r{full[2]:.2f})  "
          f"wall F1={wf[0]:.3f}(p{wf[1]:.2f}/r{wf[2]:.2f})  k_aa={k_aa:.3g}")
    return {"full_f1": full[0], "wall_f1": wf[0], "wall_rec": wf[2], "wall_prec": wf[1]}


if __name__ == "__main__":
    main()
