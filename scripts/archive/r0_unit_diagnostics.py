"""R0 unit/scaling audit: explain clot overlap via the exact COMSOL viscosity relation.

COMSOL defines effective viscosity  mu = Carreau(shear) * (mu1(Mat) + mu2(FI))  with hard
steps  mu1 @ viscosity_mat_crit (1->80)  and  mu2 @ viscosity_fi_crit (0->80).
GT clot label (canonical) = growth-only relu(mu_eff(t) - mu_eff(t0)) >= clot_phi_thresh_si.

This audits every scale on the species->viscosity->clot path:
  1. Does the deploy gelation readout (mat_si_for_gelation_from_log1p + mu1/mu2_comsol with
     the in-repo crits) reproduce the canonical GT clot? (internal consistency)
  2. Mat: empirical gelation threshold in stored gelation-decode units vs viscosity_mat_crit.
  3. FI: deploy compares fi_si in WORKING units (uM*1000) to viscosity_fi_crit=0.6.
     Quantify the unit gap vs the physical 0.6 uM threshold, and whether fibrin gelation is
     spurious (does dropping mu2 change F1 vs GT?).
All decodes use the production functions in clot_phi_simple (no re-implementation).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch

from src.config import BiochemConfig, PhysicsConfig, VesselConfig
from src.core_physics.clot_phi_simple import (
    mat_si_for_gelation_from_log1p,
    clot_phi_thresh_si,
)
from src.core_physics.clot_growth_masks import resolve_ceiling_mask
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time
from src.core_physics.species_gelation_readout import differentiable_clot_phi_from_species12

FI_IDX, MAT_IDX = 8, 11
WORK_PER_UM = 1000.0  # working units = uM * 1000 (scale_FG=7000 for c_Fg0=7uM)


def _f1(pred: torch.Tensor, gt: torch.Tensor) -> dict:
    pred = pred.bool(); gt = gt.bool()
    tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum()); fn = int((~pred & gt).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"f1": round(f1, 3), "prec": round(prec, 3), "rec": round(rec, 3), "tp": tp, "fp": fp, "fn": fn}


def analyze(anchor, device, phys, bio, scales) -> dict:
    gp = REPO / VesselConfig(phase="biochem_anchors").graph_output_dir / f"{anchor}.pt"
    data = torch.load(gp, map_location="cpu", weights_only=False)
    data.y = data.y.to(device)
    n_t = int(data.y.shape[0]); tf = n_t - 1
    ceil = resolve_ceiling_mask(data, device, bio).reshape(-1).bool()

    # canonical GT clot @ final frame
    gt = gt_clot_phi_at_time(data, tf, phys, bio, device).reshape(-1).bool()

    # deploy gelation decode @ final frame
    yT = data.y[tf]
    nd_mat = yT[:, 4 + MAT_IDX]
    nd_fi = yT[:, 4 + FI_IDX]
    mat_si = mat_si_for_gelation_from_log1p(nd_mat, bio).reshape(-1)      # gelation units
    fi_si = (torch.expm1(nd_fi.clamp(-10, 8)) * scales[FI_IDX]).reshape(-1)  # working (uM*1000)
    fi_uM = fi_si / WORK_PER_UM

    mat_crit = float(bio.viscosity_mat_crit)   # 2e7
    fi_crit = float(bio.viscosity_fi_crit)     # 0.6

    mu_ratio = float(bio.mu_ratio_max)
    mu1 = torch.where(mat_si >= mat_crit, torch.full_like(mat_si, mu_ratio), torch.ones_like(mat_si))
    mu2_deploy = torch.where(fi_si >= fi_crit, torch.full_like(fi_si, mu_ratio), torch.zeros_like(fi_si))   # WORKING vs 0.6
    mu2_phys = torch.where(fi_uM >= fi_crit, torch.full_like(fi_si, mu_ratio), torch.zeros_like(fi_si))     # uM vs 0.6

    gelled_matonly = mu1 > 1.5
    gelled_deploy = (mu1 + mu2_deploy) > 1.5     # Mat OR (deploy)FI
    gelled_fideploy = mu2_deploy > 1.5
    gelled_physfi = mu2_phys > 1.5

    # PRODUCTION readout (post-fix): exercises the real gelation code path end-to-end.
    phi_prod = differentiable_clot_phi_from_species12(yT[:, 4:16].to(device), bio).reshape(-1)
    clot_prod = phi_prod > 0.5

    m = ceil
    out = {
        "anchor": anchor,
        "n_ceiling": int(m.sum()), "n_gt_clot_ceiling": int((gt & m).sum()),
        # internal consistency: does gelation readout reproduce canonical GT clot?
        "f1_matOnly_vs_gt": _f1(gelled_matonly & m, gt & m),
        "f1_deploy(mat+FIworking)_vs_gt": _f1(gelled_deploy & m, gt & m),
        "f1_production_readout_vs_gt": _f1(clot_prod & m, gt & m),
        # FI threshold audit
        "fi_max_working": round(float(fi_si.max()), 4),
        "fi_max_uM": round(float(fi_uM.max()), 6),
        "n_FI_gel_deploy(working>=0.6)": int((gelled_fideploy & m).sum()),
        "n_FI_gel_physical(uM>=0.6)": int((gelled_physfi & m).sum()),
        "fibrin_contribution_spurious": bool(((gelled_deploy & m).sum() == (gelled_matonly & m).sum())) is False,
        # Mat threshold reverse-engineering against canonical GT clot
        "mat_si_decode_max": float(mat_si.max()),
        "mat_si_crit_used": mat_crit,
        "mat_si_p50_at_clot": round(float(mat_si[gt & m].median()) if int((gt & m).sum()) else float("nan"), 2),
        "mat_si_p95_at_nonclot": round(float(torch.quantile(mat_si[(~gt) & m], 0.95)) if int(((~gt) & m).sum()) else float("nan"), 2),
        "clot_phi_thresh_si": round(float(clot_phi_thresh_si(phys)), 5),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", default="patient001,patient002,patient003,patient004,patient006,patient007")
    ap.add_argument("--out", default="outputs/biochem/biochem_gnn/r0_adr_consistency/r0_unit_audit.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    scales = bio.get_species_scales(device=device)[:12].to(device)

    print("=== config scales ===")
    print(f"mu_ref(blood, Pa*s)={phys.mu_ref}  mu_inf={phys.mu_inf}  mu_ratio_max={bio.mu_ratio_max}")
    print(f"mu_viscosity_nd_scale={phys.mu_viscosity_nd_scale}  clot_phi_thresh_si={clot_phi_thresh_si(phys):.5f}")
    print(f"Minf={bio.Minf}  surface_scale={bio.surface_scale}  bulk_scale={bio.bulk_scale}")
    print(f"scales[FI]={float(scales[FI_IDX])}  scales[Mat]={float(scales[MAT_IDX])}")
    print(f"viscosity_mat_crit={bio.viscosity_mat_crit}  viscosity_fi_crit={bio.viscosity_fi_crit}")
    print(f"Mat gelation decode = expm1(nd)*Minf = expm1(nd)*{bio.Minf}")
    print()

    results = []
    for anchor in [a.strip() for a in args.anchors.split(",") if a.strip()]:
        try:
            r = analyze(anchor, device, phys, bio, scales)
            print(f"[{anchor}] gt_clot={r['n_gt_clot_ceiling']:5d}  "
                  f"F1(Mat-only)={r['f1_matOnly_vs_gt']['f1']:.3f}  "
                  f"F1(deploy-buggy)={r['f1_deploy(mat+FIworking)_vs_gt']['f1']:.3f}  "
                  f"F1(PRODUCTION-fixed)={r['f1_production_readout_vs_gt']['f1']:.3f}  "
                  f"| FIgel phys={r['n_FI_gel_physical(uM>=0.6)']:3d}  FImax={r['fi_max_uM']:.4f}uM")
        except Exception as e:
            r = {"anchor": anchor, "error": repr(e)}
            print(f"[{anchor}] ERROR {e!r}")
        results.append(r)

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": {
        "mu_ref": phys.mu_ref, "mu_inf": phys.mu_inf, "mu_ratio_max": bio.mu_ratio_max,
        "Minf": bio.Minf, "surface_scale": bio.surface_scale,
        "scale_FI": float(scales[FI_IDX]), "scale_Mat": float(scales[MAT_IDX]),
        "viscosity_mat_crit": bio.viscosity_mat_crit, "viscosity_fi_crit": bio.viscosity_fi_crit,
    }, "anchors": results}, indent=2), encoding="utf-8")
    print(f"\n[OK] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
