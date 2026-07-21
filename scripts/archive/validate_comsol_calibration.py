"""Definitive validation of the COMSOL phase-2 physics against repo assumptions.

Uses the patient007 calibration exports (wall + domain, COMSOL wide-format
spreadsheets) to verify, end to end and unit-by-unit:

  1. File structure / time vectors / node counts.
  2. Unit + scale audit of every field vs BiochemConfig (species ICs, geometry
     length unit, surface species magnitude vs Minf).
  3. COMSOL self-consistency: exported d(Mat,t) vs exported J0_Mat (surface ODE).
  4. Deposition-law reconstruction: rebuild J0_Mat / J0_M / J0_rp / J0_th from the
     exported *inputs* + repo constants, using COMSOL's own exported boolean gates,
     and recover the effective Da / beta*phi_at constant (constancy == law verified).
  5. Gate selectivity statistics (where/when adhesion fires).
  6. Viscosity / clot definition: spf.mu vs mu1(Mat), mu2(fi) and mu_eff growth.

Outputs a human-readable report to stdout and a JSON summary under
outputs/reports/comsol_validation/patient007_validation.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import BiochemConfig  # noqa: E402

CALIB = ROOT / "data" / "reference" / "comsol_calibration"
WALL = CALIB / "patient007_calibration_wall.txt"
DOMAIN = CALIB / "patient007_calibration_domain.txt"
OUT_DIR = ROOT / "outputs" / "reports" / "comsol_validation"


# ---- per-block column layout (0-indexed within a single time block) ----
WALL_COLS = {
    "x": 0, "y": 1, "u": 2, "v": 3, "p": 4,
    "sr": 5, "dsrx": 6, "dsry": 7, "mu": 8, "mu1_Mat": 9, "mu2_fi": 10,
    "Sat_M": 11, "Omega": 12, "kpa_chem": 13, "kpa_mech": 14, "k_pa": 15,
    "Gamma": 16, "step2t": 17, "sr_lt_lss": 18, "dsrx_lt_sgt": 19,
    "M": 20, "Mas": 21, "Mat": 22, "rp": 23, "ap": 24, "apr": 25, "aps": 26,
    "PT": 27, "th": 28, "at": 29, "fg": 30, "fi": 31,
    "dMt": 32, "dMast": 33, "dMatt": 34,
    "J0_M": 35, "J0_Mas": 36, "J0_Mat": 37, "J0_rp": 38, "J0_ap": 39, "J0_th": 40,
}
WALL_BLOCK = 41

DOMAIN_COLS = {
    "x": 0, "y": 1, "u": 2, "v": 3, "p": 4,
    "sr": 5, "dsrx": 6, "dsry": 7, "mu": 8, "mu1_Mat": 9, "mu2_fi": 10,
    "Omega": 11, "kpa_chem": 12, "kpa_mech": 13, "k_pa": 14, "Gamma": 15,
    "step2t": 16, "sr_lt_lss": 17, "dsrx_lt_sgt": 18,
    "rp": 19, "ap": 20, "apr": 21, "aps": 22, "PT": 23, "th": 24, "at": 25,
    "fg": 26, "fi": 27,
}
DOMAIN_BLOCK = 28


def _parse_header_times(path: Path):
    """Read the last '%' header line and extract the sorted unique '@ t=' values."""
    last_header = None
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("%"):
                last_header = line
            else:
                break
    import re
    times = []
    seen = set()
    for m in re.finditer(r"@ t=([0-9.eE+\-]+)", last_header):
        v = float(m.group(1))
        if v not in seen:
            seen.add(v)
            times.append(v)
    return times


def _load_block_array(path: Path, block: int):
    """Load data rows -> array [n_nodes, n_times, block].

    Columns 0,1 are time-independent geometry x,y; the remaining columns are
    n_times blocks of `block` expressions each.
    """
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("%") or not line.strip():
                continue
            rows.append(np.fromstring(line, sep=" "))
    arr = np.array(rows, dtype=np.float64)  # [n_nodes, 2 + n_times*block]
    geom = arr[:, :2]
    body = arr[:, 2:]
    n_nodes = arr.shape[0]
    n_times = body.shape[1] // block
    body = body[:, : n_times * block].reshape(n_nodes, n_times, block)
    return geom, body, n_nodes, n_times


def _stats(a):
    a = np.asarray(a, dtype=np.float64)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return dict(min=None, max=None, mean=None, median=None, n=0)
    return dict(
        min=float(np.min(finite)), max=float(np.max(finite)),
        mean=float(np.mean(finite)), median=float(np.median(finite)),
        n=int(finite.size),
    )


def _pearson(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def main():
    cfg = BiochemConfig()
    report = {}
    print("=" * 78)
    print("COMSOL phase-2 calibration validation  (patient007)")
    print("=" * 78)

    # ----------------------------------------------------------------- WALL
    print("\n## 1. WALL export structure")
    wall_times = _parse_header_times(WALL)
    geom_w, W, nW, nTW = _load_block_array(WALL, WALL_BLOCK)
    print(f"   nodes={nW}  time-blocks={nTW}  header-times={len(wall_times)}")
    print(f"   t in [{wall_times[0]}, {wall_times[-1]}]  dt~{wall_times[1]-wall_times[0]:g}s"
          f"  (n={len(wall_times)})")
    report["wall"] = {"nodes": nW, "n_times": nTW, "t0": wall_times[0], "tend": wall_times[-1]}
    tW = np.array(wall_times[:nTW])

    def w(name, ti):
        return W[:, ti, WALL_COLS[name]]

    # ------------------------------------------------------- 2. UNIT AUDIT
    print("\n## 2. Unit / scale audit (wall, t=0 unless noted)")
    geom_stats = _stats(geom_w)
    print(f"   geometry coord range: [{geom_stats['min']:.3f}, {geom_stats['max']:.3f}]"
          f"  -> magnitude implies CM (vessel ~10cm) not M")
    # species at the wall over all times
    audit = {}
    for nm, key, cfgval, cfgname in [
        ("rp", "rp", cfg.c_RP0, "c_RP0 [plt/m^3]"),
        ("ap", "ap", cfg.c_AP0, "c_AP0 [plt/m^3]"),
        ("PT", "PT", cfg.c_pT0, "c_pT0 [mol/m^3]"),
        ("fg", "fg", cfg.c_Fg0, "c_Fg0 [mol/m^3]"),
        ("at", "at", cfg.cAT0, "cAT0 [mol/m^3]"),
        ("th", "th", cfg.Tcrit, "Tcrit [mol/m^3]"),
    ]:
        s0 = _stats(w(key, 0))
        sall = _stats(W[:, :, WALL_COLS[key]])
        audit[nm] = {"t0": s0, "all": sall, "cfg": cfgval, "cfg_name": cfgname}
        print(f"   {nm:>4}: t0 median={s0['median']:.4g}  all max={sall['max']:.4g}"
              f"   | cfg {cfgname}={cfgval:.4g}  ratio(t0/cfg)={s0['median']/cfgval if cfgval else float('nan'):.3g}")
    # surface species vs Minf
    Minf = cfg.Minf
    for nm in ["M", "Mas", "Mat"]:
        sall = _stats(W[:, :, WALL_COLS[nm]])
        audit[nm] = {"all": sall, "Minf": Minf, "max_over_Minf": sall["max"] / Minf}
        print(f"   {nm:>4}: all max={sall['max']:.4g}  Minf={Minf:.3g}  max/Minf={sall['max']/Minf:.3g}")
    report["unit_audit"] = audit

    # ---------------------------------------- 3. COMSOL SELF-CONSISTENCY
    print("\n## 3. COMSOL self-consistency: exported d(.,t) vs exported J0_.")
    self_cons = {}
    for sp, dcol, jcol in [("Mat", "dMatt", "J0_Mat"), ("M", "dMt", "J0_M"), ("Mas", "dMast", "J0_Mas")]:
        d = W[:, :, WALL_COLS[dcol]].ravel()
        j = W[:, :, WALL_COLS[jcol]].ravel()
        r = _pearson(d, j)
        m = np.isfinite(d) & np.isfinite(j) & (np.abs(j) > 0)
        ratio = np.median(d[m] / j[m]) if m.any() else None
        # relative L2
        denom = np.sqrt(np.sum(j[np.isfinite(j)] ** 2)) + 1e-30
        relL2 = float(np.sqrt(np.nansum((d - j) ** 2)) / denom)
        self_cons[sp] = {"pearson": r, "median_ratio_d_over_J": float(ratio) if ratio else None, "relL2": relL2}
        print(f"   {sp:>4}: pearson(d,J0)={r}  median(d/J0)={ratio}  relL2={relL2:.3e}")
    report["self_consistency"] = self_cons

    # 3b. WHY d(Mat,t) != J0_Mat: decompose the true derivative into the
    # deposition source (J0_M) vs the autocatalytic platelet-aggregation term
    # (Mas/Minf)*k_aa*AP, gated. Integration check + recovered coefficients.
    print("\n## 3b. d(Mat,t) decomposition (deposition vs autocatalytic aggregation)")
    tW_arr = tW
    J0_Mat = W[:, :, WALL_COLS["J0_Mat"]]
    J0_M = W[:, :, WALL_COLS["J0_M"]]
    dMat = W[:, :, WALL_COLS["dMatt"]]
    Matw = W[:, :, WALL_COLS["Mat"]]
    # integration: does J0_Mat account for the actual Mat growth?
    intJ = float(np.nansum(np.trapz(J0_Mat, tW_arr, axis=1)))
    intd = float(np.nansum(np.trapz(dMat, tW_arr, axis=1)))
    matgain = float(np.nansum(Matw[:, -1] - Matw[:, 0]))
    print(f"   integral(J0_Mat dt)={intJ:.4g}  integral(d(Mat,t) dt)={intd:.4g}"
          f"  actual dMat={matgain:.4g}")
    print(f"   -> d(Mat,t) integrates to actual growth (ratio {matgain/intd:.3f}),"
          f" but J0_Mat is only 1/{matgain/intJ:.0f} of it")
    # autocatalytic sub-term T3 = gate*(Mas/Minf)*ap*step2t  (CGS), and J0_M deposition
    g_sr = W[:, :, WALL_COLS["sr_lt_lss"]] > 0.5
    g_dx = W[:, :, WALL_COLS["dsrx_lt_sgt"]] > 0.5
    gate = np.where(g_dx, (cfg.L_char * 100.0 / cfg.gamma_m) * np.abs(W[:, :, WALL_COLS["dsrx"]]), 0.0) \
        + np.where(g_sr, 1.0, 0.0)
    Masw = W[:, :, WALL_COLS["Mas"]]
    apw = W[:, :, WALL_COLS["ap"]]
    Minf_cgs = cfg.Minf * 1e-4
    T3 = gate * (Masw / Minf_cgs) * apw * W[:, :, WALL_COLS["step2t"]]
    mm = np.isfinite(dMat) & (W[:, :, WALL_COLS["step2t"]] > 0.5)
    A = np.stack([J0_M[mm], T3[mm]], axis=1)
    ok = np.all(np.isfinite(A), axis=1)
    coef, *_ = np.linalg.lstsq(A[ok], dMat[mm][ok], rcond=None)
    pred = A[ok] @ coef
    r2 = 1.0 - np.sum((dMat[mm][ok] - pred) ** 2) / (np.sum((dMat[mm][ok] - dMat[mm][ok].mean()) ** 2) + 1e-30)
    da_kaa = cfg.surface_damkohler * cfg.k_aa * 100.0
    contrib_auto = float(np.nansum(np.trapz(coef[1] * T3, tW_arr, axis=1)))
    print(f"   d(Mat,t) ~ {coef[0]:.4g}*J0_M + {coef[1]:.4g}*T3   R2={r2:.4f}")
    print(f"   autocatalytic k_aa_eff={coef[1]:.4g} cm/s  vs J0-export Da*k_aa={da_kaa:.4g}"
          f"  -> boost x{coef[1]/da_kaa:.0f}")
    print(f"   autocatalytic term explains {contrib_auto/intd*100:.0f}% of total Mat growth")
    report["dMat_decomp"] = {
        "int_J0Mat": intJ, "int_dMat": intd, "actual_dMat": matgain,
        "J0Mat_fraction_of_growth": intJ / matgain if matgain else None,
        "coef_J0M": float(coef[0]), "k_aa_eff_cm_s": float(coef[1]),
        "Da_k_aa_export": float(da_kaa), "boost": float(coef[1] / da_kaa),
        "R2": float(r2), "autocat_fraction_of_growth": contrib_auto / intd if intd else None,
    }

    # --------------------------- 4. DEPOSITION-LAW RECONSTRUCTION (J0_Mat)
    print("\n## 4. Deposition-law reconstruction (recover effective Da)")
    # Reconstruct J0_Mat from exported INPUTS in fully self-consistent CGS units,
    # using COMSOL's own exported boolean gates. CGS unit system (from Section 2):
    #   length cm, bulk platelets plt/cm^3, surface plt/cm^2, dsrx 1/(s*cm).
    L_cgs = cfg.L_char * 100.0          # 7.5e-4 m -> 0.075 cm
    gm = cfg.gamma_m                    # 150 [1/s]
    krs_cgs = cfg.k_rs * 100.0          # m/s -> cm/s
    kas_cgs = cfg.k_as * 100.0
    kaa_cgs = cfg.k_aa * 100.0
    Minf_cgs = cfg.Minf * 1e-4          # plt/m^2 -> plt/cm^2

    gate_sr = W[:, :, WALL_COLS["sr_lt_lss"]] > 0.5
    gate_dx = W[:, :, WALL_COLS["dsrx_lt_sgt"]] > 0.5
    Sat = W[:, :, WALL_COLS["Sat_M"]]
    Mas = W[:, :, WALL_COLS["Mas"]]
    rp = W[:, :, WALL_COLS["rp"]]
    ap = W[:, :, WALL_COLS["ap"]]
    dsrx = np.abs(W[:, :, WALL_COLS["dsrx"]])
    step2t = W[:, :, WALL_COLS["step2t"]]
    Jmat = W[:, :, WALL_COLS["J0_Mat"]]

    common = Sat * krs_cgs * rp + Sat * kas_cgs * ap + (Mas / Minf_cgs) * kaa_cgs * ap
    low_term = common
    sep_term = (L_cgs / gm) * dsrx * common
    bracket = np.where(gate_dx, sep_term, 0.0) + np.where(gate_sr, low_term, 0.0)
    bracket_s = bracket * step2t

    def _recover(mask, label):
        m = mask & np.isfinite(Jmat) & np.isfinite(bracket_s) & (np.abs(bracket_s) > 0) & (np.abs(Jmat) > 0)
        if not m.any():
            print(f"   [{label}] no active points")
            return None
        Da = Jmat[m] / bracket_s[m]
        st = _stats(Da)
        cv = (np.std(Da) / np.mean(Da)) if np.mean(Da) != 0 else float("nan")
        print(f"   [{label}] pts={int(m.sum()):>6}  Da median={st['median']:.4g}  CV={cv:.3g}")
        return {"label": label, "n": int(m.sum()), "Da_median": st["median"], "Da_cv": float(cv)}

    print("   CGS-consistent reconstruction, per-gate (Da should be constant if law verified):")
    rec = {}
    rec["all"] = _recover(np.ones_like(gate_sr, dtype=bool), "all active")
    rec["low_only"] = _recover(gate_sr & ~gate_dx, "low-shear gate only")
    rec["sep_only"] = _recover(gate_dx & ~gate_sr, "separation gate only")
    rec["both"] = _recover(gate_sr & gate_dx, "both gates")
    print(f"   (cfg surface_damkohler = {cfg.surface_damkohler:.3g})")
    report["deposition_law_Da"] = rec

    # Direct correlation of reconstructed bracket vs exported J0_Mat (shape match)
    r_shape = _pearson(bracket_s, Jmat)
    print(f"   pearson(reconstructed bracket, exported J0_Mat) = {r_shape}")
    report["deposition_law_shape_r"] = r_shape

    # J0_th = beta * phi_at * Mat * PT * step2t  ->  recover beta*phi_at
    print("\n   thrombin source: J0_th = beta*phi_at*Mat*PT*step2t")
    Mat = W[:, :, WALL_COLS["Mat"]]
    PT = W[:, :, WALL_COLS["PT"]]
    step2t = W[:, :, WALL_COLS["step2t"]]
    Jth = W[:, :, WALL_COLS["J0_th"]]
    denom = Mat * PT * step2t
    m = np.isfinite(Jth) & np.isfinite(denom) & (np.abs(denom) > 0) & (np.abs(Jth) > 0)
    if m.any():
        const = Jth[m] / denom[m]
        st = _stats(const)
        cfg_const = cfg.beta * cfg.phi_at
        print(f"   recovered beta*phi_at median={st['median']:.4g}  CV="
              f"{np.std(const)/np.mean(const):.3g}  | cfg beta*phi_at={cfg_const:.4g}"
              f"  ratio={st['median']/cfg_const:.3g}")
        report["thrombin_source"] = {"recovered_median": st["median"], "cfg": cfg_const,
                                     "ratio": st["median"] / cfg_const}

    # ------------------------------------------- 5. GATE SELECTIVITY STATS
    print("\n## 5. Gate selectivity (fraction of wall-nodes x time active)")
    g_sr = W[:, :, WALL_COLS["sr_lt_lss"]]
    g_dx = W[:, :, WALL_COLS["dsrx_lt_sgt"]]
    frac_sr = float(np.mean(g_sr > 0.5))
    frac_dx = float(np.mean(g_dx > 0.5))
    frac_any = float(np.mean((g_sr > 0.5) | (g_dx > 0.5)))
    print(f"   (spf.sr<lss) active: {frac_sr*100:.2f}%   (d(spf.sr,x)<sgt) active: {frac_dx*100:.2f}%"
          f"   any: {frac_any*100:.2f}%")
    # where Mat actually grows, which gate was on?
    dMat = W[:, :, WALL_COLS["dMatt"]]
    grow = dMat > (np.nanmax(dMat) * 1e-3)
    if grow.any():
        print(f"   among Mat-growing pts: sr-gate on {np.mean(g_sr[grow]>0.5)*100:.1f}%"
              f"  dx-gate on {np.mean(g_dx[grow]>0.5)*100:.1f}%")
    report["gates"] = {"frac_sr": frac_sr, "frac_dx": frac_dx, "frac_any": frac_any}

    # final Mat localization
    Mat_final = W[:, -1, WALL_COLS["Mat"]]
    Mat_t0 = W[:, 0, WALL_COLS["Mat"]]
    growth = Mat_final - Mat_t0
    pos = growth > (np.nanmax(growth) * 0.05) if np.nanmax(growth) > 0 else growth > 0
    print(f"   final Mat growth: max={np.nanmax(growth):.4g}  nodes>5%max={int(pos.sum())}/{nW}"
          f" ({pos.sum()/nW*100:.1f}% of wall)")
    report["mat_localization"] = {"max_growth": float(np.nanmax(growth)),
                                  "frac_wall_active": float(pos.sum() / nW)}

    # ----------------------------------------------------------- 6. DOMAIN
    print("\n## 6. DOMAIN export: bulk ICs + viscosity/clot definition")
    dom_times = _parse_header_times(DOMAIN)
    geom_d, D, nD, nTD = _load_block_array(DOMAIN, DOMAIN_BLOCK)
    print(f"   nodes={nD}  time-blocks={nTD}  times={dom_times[:nTD]}")
    report["domain"] = {"nodes": nD, "n_times": nTD, "times": dom_times[:nTD]}

    def d(name, ti):
        return D[:, ti, DOMAIN_COLS[name]]

    print("   bulk species t=0 vs config IC (domain median):")
    dom_audit = {}
    for nm, cfgval, cfgname in [
        ("rp", cfg.c_RP0, "c_RP0"), ("ap", cfg.c_AP0, "c_AP0"),
        ("PT", cfg.c_pT0, "c_pT0"), ("fg", cfg.c_Fg0, "c_Fg0"),
        ("at", cfg.cAT0, "cAT0"),
    ]:
        s0 = _stats(d(nm, 0))
        dom_audit[nm] = {"t0_median": s0["median"], "cfg": cfgval}
        print(f"     {nm:>4}: t0 median={s0['median']:.4g}  cfg {cfgname}={cfgval:.4g}"
              f"  ratio={s0['median']/cfgval if cfgval else float('nan'):.3g}")
    report["domain_ic_audit"] = dom_audit

    # viscosity composition: spf.mu vs mu1(Mat), mu2(fi)
    print("\n   viscosity: spf.mu vs mu1(Mat), mu2(fi)  (last domain time)")
    mu = d("mu", nTD - 1)
    mu1 = d("mu1_Mat", nTD - 1)
    mu2 = d("mu2_fi", nTD - 1)
    print(f"     spf.mu  range [{_stats(mu)['min']:.4g}, {_stats(mu)['max']:.4g}] median {_stats(mu)['median']:.4g}")
    print(f"     mu1(Mat) range [{_stats(mu1)['min']:.4g}, {_stats(mu1)['max']:.4g}] (domain has no Mat: expect ~0/const)")
    print(f"     mu2(fi)  range [{_stats(mu2)['min']:.4g}, {_stats(mu2)['max']:.4g}]")
    # mu_eff growth over domain time
    mu_t0 = d("mu", 0)
    mu_growth = d("mu", nTD - 1) - mu_t0
    print(f"     mu_eff growth (tend-t0): max={np.nanmax(mu_growth):.4g}  median={np.nanmedian(mu_growth):.4g}")
    report["viscosity"] = {
        "spf_mu": _stats(mu), "mu1_Mat": _stats(mu1), "mu2_fi": _stats(mu2),
        "mu_growth_max": float(np.nanmax(mu_growth)),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "patient007_validation.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\n[save] {out}")


if __name__ == "__main__":
    main()
