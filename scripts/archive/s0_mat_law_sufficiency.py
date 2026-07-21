"""S0 (law sufficiency gate): can the validated Mat law reproduce the GT clot?

Precondition for the Mat-centric gray-box strategy
(docs/SPECIES_LEARNING_STRATEGY.md). Forward-integrates the surface platelet-matrix
field ``Mat(t)`` on the patient007 wall using ONLY the validated, unit-consistent
COMSOL law -- deposition (low-shear/separation gated) + autocatalytic aggregation --
driven by the exported activated-platelet ``ap`` and surface ``Mas`` state, then
checks whether the resulting gelation footprint matches the GT clot.

This isolates the *Mat law* (oracle ap/Mas/flow). It is NOT the deployable model:
the deployable variant replaces exported ap/Mas with the cascade forward-solve from
inlet/IC + kine flow. If S0 fails here it fails everywhere, so this is the gate.

Inputs: data/reference/comsol_calibration/patient007_calibration_wall.txt (CPU, no GT
clot labels used as model input; GT Mat used only for scoring).

Run: python scripts/s0_mat_law_sufficiency.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.validate_comsol_calibration import WALL, WALL_BLOCK, WALL_COLS, _load_block_array, _parse_header_times  # noqa: E402
from src.config import BiochemConfig  # noqa: E402

OUT = ROOT / "outputs" / "reports" / "comsol_validation" / "s0_mat_law_sufficiency.json"


def _cum_trapz(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Cumulative trapezoid along axis=1 (time), same shape as y, starting at 0."""
    dt = np.diff(t)
    incr = 0.5 * (y[:, 1:] + y[:, :-1]) * dt[None, :]
    out = np.zeros_like(y)
    out[:, 1:] = np.cumsum(incr, axis=1)
    return out


def _f1(pred: np.ndarray, gt: np.ndarray) -> dict:
    tp = float(np.sum(pred & gt))
    fp = float(np.sum(pred & ~gt))
    fn = float(np.sum(~pred & gt))
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return {"f1": f1, "precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn}


def main():
    cfg = BiochemConfig()
    times = np.asarray(_parse_header_times(WALL), dtype=float)
    _, W, nW, nTW = _load_block_array(WALL, WALL_BLOCK)
    t = times[:nTW]

    Minf_cgs = cfg.Minf * 1e-4
    sr_lt = W[:, :, WALL_COLS["sr_lt_lss"]] > 0.5
    dx_lt = W[:, :, WALL_COLS["dsrx_lt_sgt"]] > 0.5
    dsrx = W[:, :, WALL_COLS["dsrx"]]
    step2t = W[:, :, WALL_COLS["step2t"]]
    Sat = W[:, :, WALL_COLS["Sat_M"]]
    rp = W[:, :, WALL_COLS["rp"]]
    ap = W[:, :, WALL_COLS["ap"]]
    Mas = W[:, :, WALL_COLS["Mas"]]
    Mat_gt = W[:, :, WALL_COLS["Mat"]]
    mu1_gt = W[:, :, WALL_COLS["mu1_Mat"]]
    J0_M = W[:, :, WALL_COLS["J0_M"]]
    dMat_gt = W[:, :, WALL_COLS["dMatt"]]

    # CGS constants
    Da = cfg.surface_damkohler
    krs, kas, kaa = cfg.k_rs * 100.0, cfg.k_as * 100.0, cfg.k_aa * 100.0
    L, gm = cfg.L_char * 100.0, cfg.gamma_m

    # gate combining low-shear (primary) + separation legs
    gate = np.where(dx_lt, (L / gm) * np.abs(dsrx), 0.0) + np.where(sr_lt, 1.0, 0.0)

    # --- recover the autocatalytic closure coefficient (k_aa_eff), as validated ---
    T3 = gate * (Mas / Minf_cgs) * ap * step2t
    mm = np.isfinite(dMat_gt) & (step2t > 0.5)
    A = np.stack([J0_M[mm], T3[mm]], axis=1)
    ok = np.all(np.isfinite(A), axis=1)
    coef, *_ = np.linalg.lstsq(A[ok], dMat_gt[mm][ok], rcond=None)
    coef0, k_aa_eff = float(coef[0]), float(coef[1])

    # --- forward law for dMat/dt (CGS) from STATE (ap, Mas) + FLOW (gates) ---
    deposition = Da * gate * (Sat * (krs * rp + kas * ap)) * step2t       # fresh recruitment
    autocat = k_aa_eff * gate * (Mas / Minf_cgs) * ap * step2t            # learned closure boost
    autocat_bare = Da * gate * (Mas / Minf_cgs) * kaa * ap * step2t       # bare law (no closure)

    variants = {
        "bare_law_no_closure": deposition + autocat_bare,
        "with_autocat_closure": deposition + autocat,
    }

    mat_crit = float(cfg.viscosity_mat_crit)
    gt_gel = (mu1_gt > 1.0 + 1e-6) | (Mat_gt >= mat_crit)

    report = {
        "anchor": "patient007",
        "n_wall_nodes": int(nW),
        "n_times": int(nTW),
        "mat_crit_cgs": mat_crit,
        "k_aa_eff_cm_s": k_aa_eff,
        "coef_J0M": coef0,
        "gt_gelled_fraction": float(gt_gel.mean()),
        "variants": {},
    }
    print(f"[i] patient007 wall: {nW} nodes x {nTW} steps; GT gelled frac={gt_gel.mean():.4f}")
    print(f"[i] recovered k_aa_eff={k_aa_eff:.4g} cm/s (closure), coef_J0M={coef0:.3f}")

    for name, dMatdt in variants.items():
        dMatdt = np.nan_to_num(dMatdt, nan=0.0, posinf=0.0, neginf=0.0)
        mat_rec = Mat_gt[:, :1] + _cum_trapz(dMatdt, t)  # seed at GT Mat(0) (~0)
        # trajectory agreement
        fin_gt, fin_rec = Mat_gt[:, -1], mat_rec[:, -1]
        rel_l2 = float(np.linalg.norm(fin_rec - fin_gt) / (np.linalg.norm(fin_gt) + 1e-30))
        finite = np.isfinite(fin_gt) & np.isfinite(fin_rec)
        if finite.sum() > 2:
            corr = float(np.corrcoef(fin_rec[finite], fin_gt[finite])[0, 1])
        else:
            corr = float("nan")
        # gelation footprint F1 (node x time) vs GT
        pred_gel = mat_rec >= mat_crit
        f1_all = _f1(pred_gel.ravel(), gt_gel.ravel())
        f1_final = _f1(pred_gel[:, -1], gt_gel[:, -1])
        report["variants"][name] = {
            "final_mat_rel_l2": rel_l2,
            "final_mat_corr": corr,
            "gel_f1_node_time": f1_all["f1"],
            "gel_f1_final_frame": f1_final["f1"],
            "gel_precision_final": f1_final["precision"],
            "gel_recall_final": f1_final["recall"],
        }
        print(f"\n[{name}]")
        print(f"   final Mat: corr={corr:.3f}  rel_l2={rel_l2:.3f}")
        print(f"   gelation F1 (node x time)={f1_all['f1']:.3f}  (final frame)={f1_final['f1']:.3f}"
              f"  P={f1_final['precision']:.3f} R={f1_final['recall']:.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"\n[save] {OUT}")


if __name__ == "__main__":
    main()
