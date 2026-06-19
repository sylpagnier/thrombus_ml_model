"""S1: does ONE autocatalytic closure coefficient k_aa_eff generalize across patients?

Builds on the S0 law-sufficiency gate. For each biochem anchor graph:
  - decode CGS surface state (Mat, Mas) + activated platelets (ap) from GT y,
  - compute shear sr and shear-gradient dsrx on the graph (deploy-faithful, not COMSOL
    spf.sr), form the low-shear/separation gate,
  - fit the closure dMat/dt ~ c_dep*deposition + k_aa_eff*autocat (per patient),
  - forward-integrate Mat with (a) the patient's own k_aa_eff and (b) the patient007
    k_aa_eff (transfer), gelate (Mat >= viscosity_mat_crit), and score clot F1 vs the
    canonical GT clot label (growth-only mu_eff) at the final frame.

Pass criterion: per-patient fitted k_aa_eff clusters tightly (low CV) AND the
patient007-transfer k_aa_eff yields F1 close to the own-fit F1. That means a single
learned closure scalar suffices (S1 ladder rung).

NOTE: oracle ap/Mas (from GT y); this isolates the closure. The deployable forward
(ap/Mas from IC+cascade+kine flow) is S2.

Run: python scripts/s1_kaa_closure_generalization.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.utils.rheology import compute_shear_rate  # noqa: E402

ANCHOR_DIR = ROOT / "data" / "processed" / "graphs_biochem_anchors"
OUT = ROOT / "outputs" / "reports" / "comsol_validation" / "s1_kaa_generalization.json"

AP_CH, M_CH, MAS_CH, MAT_CH, RP_CH = 4 + 1, 4 + 9, 4 + 10, 4 + 11, 4 + 0  # in y[:, :, .]
FIT_ANCHOR = "patient007"


def _shear_series(d, device):
    """Physical shear sr [1/s] and dsrx = d(sr)/dx [1/(s*m)] per time, from GT u,v."""
    y = d.y.to(device)
    T, N, _ = y.shape
    u_ref = float(d.u_ref.view(-1)[0]); d_bar = float(d.d_bar.view(-1)[0])
    Gx, Gy = d.G_x.to(device), d.G_y.to(device)
    sr = torch.empty(T, N, device=device)
    dsrx = torch.empty(T, N, device=device)
    for t in range(T):
        u = y[t, :, 0].reshape(-1, 1); v = y[t, :, 1].reshape(-1, 1)
        dudx = torch.sparse.mm(Gx, u).squeeze(1); dudy = torch.sparse.mm(Gy, u).squeeze(1)
        dvdx = torch.sparse.mm(Gx, v).squeeze(1); dvdy = torch.sparse.mm(Gy, v).squeeze(1)
        g_nd = compute_shear_rate(dudx, dudy, dvdx, dvdy, eps=1e-6)
        s = g_nd * (u_ref / max(d_bar, 1e-8))           # 1/s
        sr[t] = s
        dsrx[t] = torch.sparse.mm(Gx, s.reshape(-1, 1)).squeeze(1) / max(d_bar, 1e-8)
    return sr, dsrx


def _build_terms(d, cfg, device):
    """Return dMat_gt/dt, deposition, autocat-unit (T3), gel-threshold mask helpers (CGS)."""
    y = d.y.to(device)
    T = y.shape[0]
    t_s = d.t.to(device).reshape(-1).float() if hasattr(d, "t") and torch.is_tensor(d.t) else \
        torch.linspace(0, float(cfg.t_final), T, device=device)
    # SI decode (matches repo mat_si_for_gelation: expm1*Minf -> plt/m^2; crit=2e7 in these units).
    Minf = cfg.Minf
    sc = cfg.get_species_scales(device=device)
    mat = torch.expm1(y[:, :, MAT_CH].clamp(-10, 8)) * Minf
    mas = torch.expm1(y[:, :, MAS_CH].clamp(-10, 8)) * Minf
    M = torch.expm1(y[:, :, M_CH].clamp(-10, 8)) * Minf
    # ap/rp consistent working-unit decode; absolute scale is absorbed by the fitted
    # coefficients (same decode for every patient -> transfer is apples-to-apples).
    ap = torch.expm1(y[:, :, AP_CH].clamp(-10, 8)) * float(sc[1])
    rp = torch.expm1(y[:, :, RP_CH].clamp(-10, 8)) * float(sc[0])
    sat = (1.0 - (M / Minf)).clamp(0.0, 1.0)
    sr, dsrx = _shear_series(d, device)
    step2t = torch.sigmoid((t_s - float(cfg.surface_time_gate_s)) * float(cfg.surface_time_gate_slope))
    step2t = step2t.reshape(-1, 1)
    Da = cfg.surface_damkohler
    krs, kas = cfg.k_rs, cfg.k_as                 # SI m/s
    L, gm = cfg.L_char, cfg.gamma_m               # SI m, 1/s
    sgt_si = cfg.sgt                              # SI 1/(s*m)
    g_low = (sr < cfg.lss).float()
    g_sep = (dsrx < sgt_si).float()
    gate = g_sep * (L / gm) * dsrx.abs() + g_low
    deposition = Da * gate * (sat * (krs * rp + kas * ap)) * step2t
    autocat = gate * (mas / Minf) * ap * step2t              # multiply by k_aa_eff later
    dmat_gt = torch.zeros_like(mat)
    dt = (t_s[1:] - t_s[:-1]).reshape(-1, 1)
    dmat_gt[1:] = (mat[1:] - mat[:-1]) / dt.clamp(min=1e-8)
    return {
        "t_s": t_s, "mat_gt": mat, "deposition": deposition, "autocat": autocat,
        "dmat_gt": dmat_gt, "step2t": step2t,
    }


def _fit_kaa(terms):
    """Least squares dMat_gt ~ [deposition, autocat] on active points -> (c_dep, k_aa_eff)."""
    s2 = terms["step2t"].expand_as(terms["mat_gt"])
    mask = (s2 > 0.5) & torch.isfinite(terms["dmat_gt"])
    A = torch.stack([terms["deposition"][mask], terms["autocat"][mask]], dim=1).cpu().numpy()
    b = terms["dmat_gt"][mask].cpu().numpy()
    ok = np.all(np.isfinite(A), axis=1) & np.isfinite(b)
    coef, *_ = np.linalg.lstsq(A[ok], b[ok], rcond=None)
    return float(coef[0]), float(coef[1])


def _cum_trapz(y, t):
    dt = (t[1:] - t[:-1]).reshape(-1, 1)
    incr = 0.5 * (y[1:] + y[:-1]) * dt
    out = torch.zeros_like(y)
    out[1:] = torch.cumsum(incr, dim=0)
    return out


def _f1(pred, gt):
    pred = pred.bool(); gt = gt.bool()
    tp = float((pred & gt).sum()); fp = float((pred & ~gt).sum()); fn = float((~pred & gt).sum())
    p = tp / (tp + fp + 1e-12); r = tp / (tp + fn + 1e-12)
    return 2 * p * r / (p + r + 1e-12), p, r


def _score(d, terms, c_dep, k_aa, cfg, phys, device):
    dmatdt = c_dep * terms["deposition"] + k_aa * terms["autocat"]
    dmatdt = torch.nan_to_num(dmatdt)
    mat_rec = terms["mat_gt"][:1] + _cum_trapz(dmatdt, terms["t_s"])
    crit = float(cfg.viscosity_mat_crit)
    t_last = d.y.shape[0] - 1
    gt = gt_clot_phi_at_time(d, t_last, phys, device=device).reshape(-1)
    pred = (mat_rec[t_last] >= crit)
    f1, p, r = _f1(pred, gt)
    return {"f1": f1, "precision": p, "recall": r, "gt_pos": float(gt.sum())}


def main():
    cfg = BiochemConfig(phase="biochem")
    phys = PhysicsConfig(phase="biochem")
    device = torch.device("cpu")
    anchors = sorted(p.stem for p in ANCHOR_DIR.glob("patient*.pt") if "_metadata" not in p.stem)
    print(f"[i] anchors: {anchors}")

    # Fit on patient007 first.
    terms_cache = {}
    fits = {}
    for a in anchors:
        try:
            d = torch.load(ANCHOR_DIR / f"{a}.pt", map_location=device, weights_only=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {a}: {exc}")
            continue
        terms = _build_terms(d, cfg, device)
        c_dep, k_aa = _fit_kaa(terms)
        fits[a] = {"c_dep": c_dep, "k_aa_eff": k_aa}
        terms_cache[a] = (d, terms)
        print(f"[fit] {a}: k_aa_eff={k_aa:.4g}  c_dep={c_dep:.3g}")

    if FIT_ANCHOR not in fits:
        print(f"[ERR] {FIT_ANCHOR} not available; cannot run transfer test")
        return
    k_aa_p007 = fits[FIT_ANCHOR]["k_aa_eff"]
    c_dep_p007 = fits[FIT_ANCHOR]["c_dep"]

    kaa_vals = np.array([v["k_aa_eff"] for v in fits.values()])
    report = {
        "fit_anchor": FIT_ANCHOR,
        "k_aa_eff_p007": k_aa_p007,
        "k_aa_eff_mean": float(kaa_vals.mean()),
        "k_aa_eff_cv": float(kaa_vals.std() / (abs(kaa_vals.mean()) + 1e-30)),
        "per_patient": {},
    }
    print(f"\n[i] k_aa_eff across patients: mean={kaa_vals.mean():.4g} "
          f"CV={report['k_aa_eff_cv']:.3f}")
    print(f"\n{'patient':<12}{'own F1':>8}{'p007 F1':>9}{'recall(p007)':>13}{'k_aa_eff':>11}")
    for a, (d, terms) in terms_cache.items():
        own = _score(d, terms, fits[a]["c_dep"], fits[a]["k_aa_eff"], cfg, phys, device)
        xfer = _score(d, terms, c_dep_p007, k_aa_p007, cfg, phys, device)
        report["per_patient"][a] = {"own_fit": own, "p007_transfer": xfer,
                                    "k_aa_eff": fits[a]["k_aa_eff"]}
        print(f"{a:<12}{own['f1']:>8.3f}{xfer['f1']:>9.3f}{xfer['recall']:>13.3f}"
              f"{fits[a]['k_aa_eff']:>11.3g}")

    f1_xfer = np.array([v["p007_transfer"]["f1"] for v in report["per_patient"].values()])
    f1_own = np.array([v["own_fit"]["f1"] for v in report["per_patient"].values()])
    report["mean_f1_p007_transfer"] = float(f1_xfer.mean())
    report["mean_f1_own_fit"] = float(f1_own.mean())
    print(f"\n[i] mean F1: own-fit={f1_own.mean():.3f}  p007-transfer={f1_xfer.mean():.3f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"[save] {OUT}")


if __name__ == "__main__":
    main()
