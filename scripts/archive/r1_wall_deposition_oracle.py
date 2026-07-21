"""R1 oracle check: does the COMSOL wall platelet-deposition law reproduce GT `Mat`?

R0 proved clots are driven by the wall platelet matrix `Mat` (COMSOL surface physics
`tds2`/`srf1`), NOT bulk fibrin. `Mat` has D=0 in COMSOL, so M/Mas/Mat are *purely local*
surface ODEs accumulating on wall nodes (no advection/diffusion):

    dMat/dt = Da * R_Mat,  active for t > step2t (=12 s)

    R_M  = [sep](L/gamma_m)|dgamma/ds| Sat * (k_rs*RP + k_as*AP) + [lss] Sat*(k_rs*RP + k_as*AP)
    R_Mat= R_M + [sep](L/gamma_m)|dgamma/ds|(Mas/Minf)k_aa*AP + [lss](Mas/Minf)k_aa*AP
    Sat  = 1 - M/Minf            (COMSOL Analytic 4; `--sat kernel` uses 1-(M+Mas+Mat)/Minf)
    sep  = dgamma/ds < sgt  ;  lss = gamma < lss_crit

We feed GROUND-TRUTH wall RP/AP + flow (-> shear gamma and dgamma/ds) through this law and ask:

  (1) Teacher-forced rate: corr / mass-balance of  Da*R_Mat  vs observed  dMat/dt  on wall nodes.
  (2) Free-run forward: integrate M/Mas/Mat from t0 using ONLY GT bulk species + GT velocity,
      then compare integrated Mat to GT Mat (rel-L2, corr) and the resulting clot (mu1 step at
      viscosity_mat_crit) to the canonical GT clot (F1).

All rate math mirrors `biochem_physics_kernels.biochem_wall_residual`; integration is in the
kernel's surface units (Minf*surface_scale) and converted to gelation units (expm1*Minf) only
for the mu1 clot threshold. No training, COMSOL constants only.
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

from src.config import BiochemConfig, PhysicsConfig, VesselConfig, BulkSpecies
from src.core_physics.biochem_physics_kernels import BiochemPhysicsKernels
from src.core_physics.physics_kernels import PhysicsKernels
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time

# species-block-relative indices (y[:, :, 4:16])
M_IDX, MAS_IDX, MAT_IDX = 9, 10, 11


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean(); b = b - b.mean()
    denom = float(torch.sqrt((a ** 2).sum() * (b ** 2).sum()).item()) + 1e-30
    return float((a * b).sum().item()) / denom


def _f1(pred: torch.Tensor, gt: torch.Tensor) -> dict:
    pred = pred.bool(); gt = gt.bool()
    tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum()); fn = int((~pred & gt).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"f1": round(f1, 4), "prec": round(prec, 4), "rec": round(rec, 4), "tp": tp, "fp": fp, "fn": fn}


def _ceiling_mask(data, device, bio_cfg) -> torch.Tensor:
    try:
        from src.core_physics.clot_growth_masks import resolve_ceiling_mask
        return resolve_ceiling_mask(data, device, bio_cfg).reshape(-1).bool()
    except Exception:
        return torch.ones(int(data.num_nodes), dtype=torch.bool, device=device)


def _node_vec(val, n, device) -> torch.Tensor:
    t = torch.as_tensor(val, device=device, dtype=torch.float32).reshape(-1)
    return t if t.numel() == n else t[:1].expand(n).contiguous()


class DepositionLaw:
    """COMSOL surface platelet-deposition rates (kernel-units = expm1(nd)*Minf*surface_scale)."""

    def __init__(self, data, device, bio, kernels, scales, *, sat_mode: str, grad_mode: str = "streamwise"):
        self.k = kernels
        self.cfg = bio
        self.sat_mode = sat_mode
        self.grad_mode = grad_mode
        self.device = device
        self.scales = scales
        self.n = int(data.num_nodes)
        self.wall = data.mask_wall.view(-1).bool().to(device)
        self.Minf_scaled = float(bio.Minf) * float(bio.surface_scale)
        self.Da = float(bio.surface_damkohler)
        self.k_rs = float(bio.k_rs); self.k_as = float(bio.k_as); self.k_aa = float(bio.k_aa)
        self.L_over_gm = float(bio.L_char) / float(bio.gamma_m)
        self.sgt = float(bio.sgt); self.lss = float(bio.lss)
        self.gate_t = float(bio.surface_time_gate_s)
        self.gate_slope = float(bio.surface_time_gate_slope)
        self.spatial = {
            "u_ref": _node_vec(data.u_ref, self.n, device),
            "d_bar": _node_vec(data.d_bar, self.n, device),
        }
        self.d_bar_safe = torch.clamp(self.spatial["d_bar"], min=1e-8)

    def shear_fields(self, yk: torch.Tensor):
        """Return (gamma[N], dgamma/ds[N]) in physical 1/s and 1/(m*s)."""
        u, v = yk[:, 0], yk[:, 1]
        gamma = self.k._compute_shear_rate(u, v, self.spatial, self.data)
        dgx = torch.sparse.mm(self.data.G_x, gamma.unsqueeze(1)).squeeze(1)
        dgy = torch.sparse.mm(self.data.G_y, gamma.unsqueeze(1)).squeeze(1)
        if self.grad_mode == "x":  # COMSOL d(spf.sr, x): Cartesian x-derivative
            dgamma_ds = dgx / self.d_bar_safe
        else:  # streamwise directional derivative (repo kernel convention)
            vmag = torch.sqrt(u ** 2 + v ** 2) + 1e-8
            dgamma_ds = ((u / vmag) * dgx + (v / vmag) * dgy) / self.d_bar_safe
        return gamma, dgamma_ds

    def rates(self, yk, M, Mas, Mat, *, gate: float):
        """R_M, R_Mas, R_Mat (kernel-units/s, pre-Da) on the FULL node set (0 off-wall)."""
        lin = torch.expm1(torch.clamp(yk[:, 4:16], -10, 8)) * self.scales[:12]
        RP = lin[:, BulkSpecies.RP.value]; AP = lin[:, BulkSpecies.AP.value]
        gamma, dgamma_ds = self.shear_fields(yk)
        kin = self.k.kinetics
        is_sep = kin._soft_step(dgamma_ds, self.sgt, kin.T_grad, reverse=True)
        is_lss = kin._soft_step(gamma, self.lss, kin.T_low_shear, reverse=True)
        dabs = torch.abs(dgamma_ds) + 1e-6
        if self.sat_mode == "kernel":
            sat = torch.clamp(1.0 - (M + Mas + Mat) / self.Minf_scaled, min=0.0, max=1.0)
        else:  # COMSOL Analytic 4: Sat(M) = 1 - M/Minf
            sat = torch.clamp(1.0 - M / self.Minf_scaled, min=0.0, max=1.0)
        sep_pref = is_sep * self.L_over_gm * dabs
        rp_ad = sep_pref * sat * self.k_rs * RP + is_lss * sat * self.k_rs * RP
        ap_ad = sep_pref * sat * self.k_as * AP + is_lss * sat * self.k_as * AP
        mas_ratio = Mas / self.Minf_scaled
        mas_ad = sep_pref * mas_ratio * self.k_aa * AP + is_lss * mas_ratio * self.k_aa * AP
        R_M = (rp_ad + ap_ad) * gate
        R_Mat = (rp_ad + ap_ad + mas_ad) * gate
        return R_M, R_M, R_Mat

    def gate(self, t_s: float) -> float:
        return float(torch.sigmoid(torch.tensor((t_s - self.gate_t) * self.gate_slope)))


def analyze_anchor(anchor, device, bio, kernels, scales, args) -> dict:
    gp = REPO / VesselConfig(phase="biochem_anchors").graph_output_dir / f"{anchor}.pt"
    data = torch.load(gp, map_location="cpu", weights_only=False)
    for a in ("G_x", "G_y", "Laplacian"):
        setattr(data, a, getattr(data, a).to(device).coalesce())
    data.y = data.y.to(device)
    n_times = int(data.y.shape[0])
    t = data.t.reshape(-1).to(device).float()
    phys = PhysicsConfig(phase="biochem")

    law = DepositionLaw(data, device, bio, kernels, scales, sat_mode=args.sat, grad_mode=args.grad)
    law.data = data
    wall = law.wall
    ceil = _ceiling_mask(data, device, bio)
    surf_scale = float(bio.surface_scale)
    mat_crit = float(bio.viscosity_mat_crit)
    K = law.Minf_scaled

    def mat_kernel(k):  # GT Mat in kernel units
        return torch.expm1(torch.clamp(data.y[k][:, 4 + MAT_IDX], -10, 8)) * K

    # ---- (1) teacher-forced rate check (GT M/Mas/Mat fed in) ----
    pred_rate, obs_rate = [], []
    tot_dMat = 0.0; tot_pred = 0.0
    for k in range(n_times - 1):
        yk = data.y[k]
        M = torch.expm1(torch.clamp(yk[:, 4 + M_IDX], -10, 8)) * K
        Mas = torch.expm1(torch.clamp(yk[:, 4 + MAS_IDX], -10, 8)) * K
        Mat = torch.expm1(torch.clamp(yk[:, 4 + MAT_IDX], -10, 8)) * K
        g = law.gate(float(t[k]))
        _, _, R_Mat = law.rates(yk, M, Mas, Mat, gate=g)
        dMatdt_pred = law.Da * R_Mat
        dt = max(float(t[k + 1] - t[k]), 1e-9)
        dMatdt_obs = (mat_kernel(k + 1) - mat_kernel(k)) / dt
        m = wall & ceil
        if int(m.sum()) == 0:
            m = wall
        pred_rate.append(dMatdt_pred[m]); obs_rate.append(dMatdt_obs[m])
        tot_dMat += float((mat_kernel(k + 1) - mat_kernel(k))[m].sum())
        tot_pred += float((dMatdt_pred * dt)[m].sum())
    PR = torch.cat(pred_rate); OB = torch.cat(obs_rate)

    # ---- (2) free-run forward integration (only GT bulk species + GT velocity) ----
    M = mat_kernel(0) * 0.0  # init M/Mas/Mat from GT t0 (~0); use zeros for purity
    Mas = M.clone(); Mat = M.clone()
    M = torch.where(wall, torch.expm1(torch.clamp(data.y[0][:, 4 + M_IDX], -10, 8)) * K, M)
    Mas = torch.where(wall, torch.expm1(torch.clamp(data.y[0][:, 4 + MAS_IDX], -10, 8)) * K, Mas)
    Mat = torch.where(wall, mat_kernel(0), Mat)
    for k in range(n_times - 1):
        yk = data.y[k]
        g = law.gate(float(t[k]))
        R_M, R_Mas, R_Mat = law.rates(yk, M, Mas, Mat, gate=g)
        dt = max(float(t[k + 1] - t[k]), 1e-9)
        M = (M + dt * law.Da * R_M) * wall
        Mas = (Mas + dt * law.Da * R_Mas) * wall
        Mat = (Mat + dt * law.Da * R_Mat) * wall
        M = torch.clamp(M, min=0.0, max=K); Mas = torch.clamp(Mas, min=0.0, max=K)
        Mat = torch.clamp(Mat, min=0.0)

    ti = n_times - 1
    Mat_gt_gel = mat_kernel(ti) / surf_scale
    Mat_pred_gel = Mat / surf_scale
    gt_clot = gt_clot_phi_at_time(data, ti, phys, device).reshape(-1).bool()
    pred_clot = (Mat_pred_gel >= mat_crit) & wall
    matgt_clot = (Mat_gt_gel >= mat_crit) & wall  # GT-Mat gelation (R0 upper bound)

    mw = wall & ceil
    if int(mw.sum()) == 0:
        mw = wall
    rel_l2 = float(((Mat_pred_gel - Mat_gt_gel)[mw]).norm() / (Mat_gt_gel[mw].norm() + 1e-30))

    # effective-Da least-squares fit (removes the unit-ambiguous global magnitude; tests PATTERN)
    pv = Mat_pred_gel[mw]; gv = Mat_gt_gel[mw]
    c_fit = float((pv * gv).sum() / (pv * pv).sum().clamp(min=1e-30))
    pred_clot_fit = (c_fit * Mat_pred_gel >= mat_crit) & wall

    return {
        "anchor": anchor,
        "n_wall": int(wall.sum()), "n_wall_ceiling": int(mw.sum()),
        "sat_mode": args.sat,
        "corr_rate_pred_vs_obs": round(_corr(PR, OB), 4),
        "mass_balance_dMat_over_pred": round((tot_dMat / tot_pred), 4) if abs(tot_pred) > 0 else None,
        "freerun_Mat_relL2": round(rel_l2, 4),
        "freerun_Mat_corr": round(_corr(Mat_pred_gel[mw], Mat_gt_gel[mw]), 4),
        "Mat_pred_max_gel": float(Mat_pred_gel.max()),
        "Mat_gt_max_gel": float(Mat_gt_gel.max()),
        "n_gt_clot_wall": int((gt_clot & wall).sum()),
        "grad_mode": args.grad,
        "eff_Da_fit_factor": round(c_fit, 4),
        "f1_freerun_clot_vs_gt": _f1(pred_clot & ceil, gt_clot & wall & ceil),
        "f1_freerun_fitDa_vs_gt": _f1(pred_clot_fit & ceil, gt_clot & wall & ceil),
        "f1_gtMat_clot_vs_gt": _f1(matgt_clot & ceil, gt_clot & wall & ceil),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", default="patient001,patient002,patient003,patient004,patient006,patient007")
    ap.add_argument("--sat", choices=["comsol", "kernel"], default="comsol")
    ap.add_argument("--grad", choices=["streamwise", "x"], default="streamwise")
    ap.add_argument("--out", default="outputs/biochem/biochem_gnn/r1_wall_deposition/r1_wall_deposition.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bio = BiochemConfig(phase="biochem")
    kernels = BiochemPhysicsKernels(bio, PhysicsKernels(phys_cfg=PhysicsConfig(phase="biochem")))
    scales = kernels._get_species_scales(device, torch.float32)

    results = []
    for anchor in [a.strip() for a in args.anchors.split(",") if a.strip()]:
        try:
            r = analyze_anchor(anchor, device, bio, kernels, scales, args)
            print(f"[{anchor}] rate_corr={r['corr_rate_pred_vs_obs']:.3f} "
                  f"| Mat corr={r['freerun_Mat_corr']:.3f}  "
                  f"F1(free)={r['f1_freerun_clot_vs_gt']['f1']:.3f} "
                  f"F1(fitDa x{r['eff_Da_fit_factor']:.1f})={r['f1_freerun_fitDa_vs_gt']['f1']:.3f} "
                  f"F1(GT-Mat)={r['f1_gtMat_clot_vs_gt']['f1']:.3f}")
        except Exception as e:
            import traceback; traceback.print_exc()
            r = {"anchor": anchor, "error": repr(e)}
            print(f"[{anchor}] ERROR {e!r}")
        results.append(r)

    ok = [r for r in results if "error" not in r]
    def avg(key, sub=None):
        vals = []
        for r in ok:
            v = r.get(key)
            if sub and isinstance(v, dict):
                v = v.get(sub)
            if v is not None:
                vals.append(v)
        return round(sum(vals) / len(vals), 4) if vals else None
    summary = {
        "sat_mode": args.sat,
        "anchors": results,
        "mean_rate_corr": avg("corr_rate_pred_vs_obs"),
        "mean_freerun_Mat_corr": avg("freerun_Mat_corr"),
        "mean_freerun_Mat_relL2": avg("freerun_Mat_relL2"),
        "mean_f1_freerun": avg("f1_freerun_clot_vs_gt", "f1"),
        "mean_f1_freerun_fitDa": avg("f1_freerun_fitDa_vs_gt", "f1"),
        "mean_f1_gtMat_upper": avg("f1_gtMat_clot_vs_gt", "f1"),
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out}")
    print(f"[MEAN grad={args.grad} sat={args.sat}] rate_corr={summary['mean_rate_corr']} "
          f"Mat_corr={summary['mean_freerun_Mat_corr']} "
          f"F1_freerun={summary['mean_f1_freerun']} F1_fitDa={summary['mean_f1_freerun_fitDa']} "
          f"F1_gtMat_upper={summary['mean_f1_gtMat_upper']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
