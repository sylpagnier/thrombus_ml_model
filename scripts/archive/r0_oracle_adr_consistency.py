"""R0 oracle check: does the known COMSOL reaction reproduce GT fibrin evolution?

Gray-box premise: with upstream species (thrombin T, fibrinogen FG) + flow + the known
(COMSOL-parity) rate law, the fibrin field FI is determined by advection-diffusion-reaction.
The fibrin source is COMSOL `reac1`:  R_FI = kfi * T * FG / (kmfi + FG)  (independent of FI;
verified against oracle_kinetics.csv in test_biochem_kinetics_parity).

We feed GROUND-TRUTH T, FG, flow into the operator and ask, in physical units (working/s),
how well the predicted local fibrin formation matches the observed dFI/dt on GT:

  reaction-only :  pred = R_FI
  full local ADR:  pred = R_FI - advection(FI) + diffusion(FI)

Reported (no training, no time-stepping):
  - corr(pred, dFI/dt_obs)        spatial/temporal PATTERN agreement (unit-robust)
  - ratio  dFI/dt_obs / pred       MAGNITUDE calibration (assumes t in seconds)
  - mass balance  sum(dFI) vs sum(R_FI*dt)
on the clot-relevant ceiling region. advection/diffusion mirror biochem_adr_residual.
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
from src.core_physics.biochem_physics_kernels import BiochemPhysicsKernels
from src.core_physics.physics_kernels import PhysicsKernels

IDX = {"T": 5, "FG": 7, "FI": 8}  # species-block-relative (y[:,:,4:16])


def _ceiling_mask(data, device, bio_cfg) -> torch.Tensor:
    try:
        from src.core_physics.clot_growth_masks import resolve_ceiling_mask
        return resolve_ceiling_mask(data, device, bio_cfg).reshape(-1).bool()
    except Exception:
        return torch.ones(int(data.num_nodes), dtype=torch.bool, device=device)


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean(); b = b - b.mean()
    denom = float(torch.sqrt((a ** 2).sum() * (b ** 2).sum()).item()) + 1e-30
    return float((a * b).sum().item()) / denom


def analyze_anchor(anchor, device, bio, kernels, scales, args) -> dict:
    gp = REPO / VesselConfig(phase="biochem_anchors").graph_output_dir / f"{anchor}.pt"
    data = torch.load(gp, map_location="cpu", weights_only=False)
    for a in ("G_x", "G_y", "Laplacian"):
        setattr(data, a, getattr(data, a).to(device).coalesce())
    data.y = data.y.to(device)
    n_times = int(data.y.shape[0])
    u_ref = float(data.u_ref.reshape(-1)[0]); d_bar = float(data.d_bar.reshape(-1)[0])
    t = data.t.reshape(-1).to(device).float()
    ceil = _ceiling_mask(data, device, bio)
    scale_FI = float(scales[IDX["FI"]])
    inv_pe = kernels.D_coeff["FI"] / (u_ref * d_bar)

    def fi_work(k):
        return torch.expm1(torch.clamp(data.y[k][:, 4 + IDX["FI"]], -10, 8)) * scale_FI

    # pooled across interior frames, restricted to ceiling
    pred_rx, pred_full, obs = [], [], []
    tot_dFI = 0.0; tot_prod = 0.0
    for k in range(n_times - 1):
        yk = data.y[k]
        state = torch.clamp(yk[:, 4:16], -10, 8)
        linear = torch.expm1(state) * scales[:12]
        T_si, FG_si, FI_si = linear[:, IDX["T"]], linear[:, IDX["FG"]], linear[:, IDX["FI"]]
        _, R_FI = kernels.kinetics.compute_fibrin_kinetics(T_si, FG_si, FI_si)  # working/s

        FIk = FI_si
        FIk1 = fi_work(k + 1)
        dt = max(float(t[k + 1] - t[k]), 1e-9)
        dFIdt = (FIk1 - FIk) / dt  # working/s

        # transport of FI in working units (mirror kernel ND terms -> physical /s)
        C = FIk.unsqueeze(1)
        dCdx = torch.sparse.mm(data.G_x, C).squeeze(1)
        dCdy = torch.sparse.mm(data.G_y, C).squeeze(1)
        lap = torch.sparse.mm(data.Laplacian, C).squeeze(1)
        u_nd, v_nd = yk[:, 0], yk[:, 1]
        # ND terms (same prefactor d_bar/u_ref as reaction_nd); convert to working/s by * u_ref/d_bar
        adv_nd = u_nd * dCdx + v_nd * dCdy
        dif_nd = inv_pe * lap
        adv_phys = adv_nd * (u_ref / d_bar)
        dif_phys = dif_nd * (u_ref / d_bar)
        pred_full_k = R_FI - adv_phys + dif_phys

        m = ceil
        pred_rx.append(R_FI[m]); pred_full.append(pred_full_k[m]); obs.append(dFIdt[m])

        tot_dFI += float((FIk1 - FIk)[m].sum().item())
        tot_prod += float((R_FI[m] * dt).sum().item())

    PR = torch.cat(pred_rx); PF = torch.cat(pred_full); OB = torch.cat(obs)

    # ---- clot-driver attribution at final frame (FI vs Mat) ----
    WORK_PER_UM = 1000.0  # working units = uM * 1000 (scale_FG=7000 for c_Fg0=7uM)
    FI_CRIT_UM = float(bio.viscosity_fi_crit)        # 0.6 uM (COMSOL mu2 step)
    MAT_CRIT = float(bio.viscosity_mat_crit)         # 2e7 (COMSOL mu1 step), Mat working units
    lin0 = torch.expm1(torch.clamp(data.y[0][:, 4:16], -10, 8)) * scales[:12]
    linT = torch.expm1(torch.clamp(data.y[-1][:, 4:16], -10, 8)) * scales[:12]
    FI_T = linT[:, IDX["FI"]]                          # working
    FG_0 = lin0[:, IDX["FG"]]; FG_T = linT[:, IDX["FG"]]
    Mat_T = linT[:, 11]
    fi_max_uM = float(FI_T.max()) / WORK_PER_UM
    fg_depletion_frac = float(((FG_0 - FG_T).clamp(min=0).sum() / (FG_0.sum() + 1e-30)).item())
    mu_eff_nd = data.y[-1][:, 3]
    clot = mu_eff_nd > (1.5 * float(mu_eff_nd.median()))
    fi_over_phys = (FI_T / WORK_PER_UM) > FI_CRIT_UM   # physical 0.6 uM threshold
    mat_over = Mat_T > MAT_CRIT
    n_clot = int(clot.sum().item())

    return {
        "anchor": anchor,
        "n_ceiling": int(ceil.sum().item()),
        "u_ref": u_ref, "d_bar": d_bar, "t_end": float(t[-1]),
        "corr_reaction_vs_dFIdt": _corr(PR, OB),
        "corr_fullADR_vs_dFIdt": _corr(PF, OB),
        "mass_balance_ratio_dFI_over_prod": (tot_dFI / tot_prod) if abs(tot_prod) > 0 else None,
        "fi_max_uM": fi_max_uM,
        "fi_gelation_threshold_uM": FI_CRIT_UM,
        "fi_reaches_gelation": bool(fi_over_phys.any().item()),
        "fg_depletion_frac": fg_depletion_frac,
        "n_clot_nodes": n_clot,
        "n_FI_over_crit_phys": int(fi_over_phys.sum().item()),
        "n_Mat_over_crit": int(mat_over.sum().item()),
        "clot_overlap_FI": int((clot & fi_over_phys).sum().item()),
        "clot_overlap_Mat": int((clot & mat_over).sum().item()),
        "clot_driver": "Mat" if int((clot & mat_over).sum()) >= int((clot & fi_over_phys).sum()) else "FI",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", default="patient001,patient002,patient003,patient004,patient006,patient007")
    ap.add_argument("--out", default="outputs/biochem/biochem_gnn/r0_adr_consistency/r0_adr_consistency.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bio = BiochemConfig(phase="biochem")
    kernels = BiochemPhysicsKernels(bio, PhysicsKernels(phys_cfg=PhysicsConfig(phase="biochem")))
    scales = kernels._get_species_scales(device, torch.float32)

    results = []
    for anchor in [a.strip() for a in args.anchors.split(",") if a.strip()]:
        try:
            r = analyze_anchor(anchor, device, bio, kernels, scales, args)
            print(f"[{anchor}] FI_max={r['fi_max_uM']:.4f}uM (gel@{r['fi_gelation_threshold_uM']}uM -> "
                  f"{'REACHES' if r['fi_reaches_gelation'] else 'never gels'})  "
                  f"FG_depl={r['fg_depletion_frac']*100:.2f}%  "
                  f"clot={r['n_clot_nodes']}  clot&Mat={r['clot_overlap_Mat']}  clot&FI={r['clot_overlap_FI']}  "
                  f"driver={r['clot_driver']}")
        except Exception as e:
            r = {"anchor": anchor, "error": repr(e)}
            print(f"[{anchor}] ERROR {e!r}")
        results.append(r)

    ok = [r for r in results if "error" not in r]
    def avg(key):
        vals = [r[key] for r in ok if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None
    summary = {
        "anchors": results,
        "mean_corr_reaction_vs_dFIdt": avg("corr_reaction_vs_dFIdt"),
        "mean_corr_fullADR_vs_dFIdt": avg("corr_fullADR_vs_dFIdt"),
        "mean_mass_balance_ratio": avg("mass_balance_ratio_dFI_over_prod"),
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out}")
    print(f"[MEAN] corr(reaction)={summary['mean_corr_reaction_vs_dFIdt']} "
          f"corr(fullADR)={summary['mean_corr_fullADR_vs_dFIdt']} "
          f"mass_ratio={summary['mean_mass_balance_ratio']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
